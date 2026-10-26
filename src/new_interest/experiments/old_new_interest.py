"""Run and independently verify the approved old/new-interest comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import sha256_directory
from new_interest.experiments import backbone_scaling as shared
from new_interest.training.semantic_id_tokens import write_json as _write_json
from new_interest.training.sft_state_audit import tensor_state_manifest


CONTRACT = Path("configs/experiments/mind_old_new_interest_v1.json")
DATA_MANIFEST = Path("manifests/mind_old_new_interest_data_v1.json")
RUNTIME_MANIFEST = Path("manifests/mind_old_new_runtime_v1.json")
DATA = Path("data/training/mind_old_new_interest_v1")
METHODS = ("two_stage", "mixed")
load_json = shared.load_json
sha256 = shared.sha256
require = shared.require


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


def identity(project: Path) -> dict[str, str]:
    return {
        "version": "mind-old-new-experiment-v1",
        "contract_sha256": sha256(project / CONTRACT),
        "data_manifest_sha256": sha256(project / DATA_MANIFEST),
        "runtime_manifest_sha256": sha256(project / RUNTIME_MANIFEST),
    }


def verify_runtime(project: Path) -> None:
    for relative, expected in load_json(project / RUNTIME_MANIFEST)["files"].items():
        require(sha256(project / relative) == expected, f"Frozen runtime changed: {relative}")


def audit_inputs(project: Path) -> dict[str, Any]:
    contract = load_json(project / CONTRACT)
    frozen = load_json(project / DATA_MANIFEST)
    require(frozen["protocol_sha256"] == sha256(project / CONTRACT), "Data belongs to another protocol")
    require(contract["shared_baseline"]["submit_baseline_here"] is False, "NI baseline must be reused")
    require(not contract["independent_sid_warmup"], "Independent SID warm-up was enabled")
    require(not contract["methods"]["mixed"]["mid_training_reset"], "Mixed acquired a mid-training reset")
    for entry in contract["new_interest"]["input_files"] + frozen["files"]:
        p = project / entry["path"]
        require(p.is_file() and p.stat().st_size == entry["bytes"] and sha256(p) == entry["sha256"], f"Frozen input differs: {entry['path']}")
    old = pq.read_table(project / DATA / "old_interest.parquet")
    new = pq.read_table(project / shared.TRAIN)
    mixed = pq.read_table(project / DATA / "mixed.parquet")
    mapping = pq.read_table(project / shared.MAPPING).to_pylist()
    labels = {row["target_subcategory"]: row["interest_token"] for row in mapping}
    require(len(labels) == 108, "SID mapping does not contain 108 classes")
    require((old.num_rows, new.num_rows, mixed.num_rows) == (frozen["old_rows"], 21600, frozen["mixed_rows"]), "Dataset dimensions differ")
    require(mixed.num_rows == old.num_rows + new.num_rows, "Mixed rows were oversampled or omitted")
    require(len(set(old["user_id"].to_pylist())) == old.num_rows, "OI contains repeated users")
    dev_users = set(pq.read_table(project / shared.DEV, columns=["user_id"])["user_id"].to_pylist())
    require(not set(old["user_id"].to_pylist()) & dev_users, "OI and Dev users overlap")
    require(not set(new["user_id"].to_pylist()) & dev_users, "NI and Dev users overlap")
    for task, table in (("old_interest", old), ("new_interest", new)):
        expected = table.drop(["training_task"]) if "training_task" in table.column_names else table
        actual = mixed.filter(pc.equal(mixed["training_task"], task)).drop(["training_task"])
        require(actual.sort_by("sample_id").equals(expected.sort_by("sample_id")), f"Mixed changed {task} rows/messages")
    require(len(set(mixed["sample_id"].to_pylist())) == mixed.num_rows, "Mixed sample IDs collide")
    for row in old.to_pylist():
        require(row["response"] == labels.get(row["target_subcategory"]), "OI target is outside frozen SID vocabulary")
        require(row["messages"] == [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}], "OI messages differ from prompt/target")
    counts = Counter(old["target_subcategory"].to_pylist())
    require(dict(counts) == frozen["per_class_old_counts"] and max(counts.values()) <= 200, "Actual OI class distribution changed")
    provenance = pq.read_table(project / DATA / "old_selection_manifest.parquet").to_pylist()
    require(len(provenance) == old.num_rows, "OI provenance coverage differs")
    for row in provenance:
        require(row["target_subcategory"] in row["visible_history_subcategories"], "OI target lacks visible history support")
        require(row["visible_history_news_ids"] == row["full_history_news_ids"][-50:], "OI History window changed")
        require(bool(row["target_clicked_news_ids"]), "OI target has no observed clicked article")
    token = load_json(project / "artifacts/mind-old-new-interest-v1/local_token_audit.json")
    require(token["status"] == "passed", "Full token-length audit is not accepted")
    for task, expected_rows in (("old_interest", old.num_rows), ("new_interest", new.num_rows)):
        require(token["tasks"][task]["audit"]["samples"] == expected_rows and token["tasks"][task]["audit"]["over_max_seq_length"] == 0, "Full token audit coverage/length mismatch")
    for stage in ("old", "new", "mixed"):
        config = load_json(project / f"configs/sft/experiments/mind_old_new_interest_{stage}_v1.json")
        for key, value in contract["recipe"].items():
            require(config.get(key) == value, f"Recipe differs for {stage}: {key}")
        require(config["model_name"] == contract["model"]["model_name"] and config["model_revision"] == contract["model"]["model_revision"], "Model identity changed")
        require(config["audit_training_exposure"] and config["record_initial_adapter_state"], "Required training audits are disabled")
        require((stage == "new") == ("adapter_path" in config), "Wrong stage warm start")
    return {"identity": identity(project), "old_rows": old.num_rows, "new_rows": new.num_rows,
            "mixed_rows": mixed.num_rows, "old_classes": len(counts), "old_class_counts": dict(counts),
            "old_class_min": min(counts.values()), "old_class_max": max(counts.values()),
            "actually_balanced": len(set(counts.values())) == 1, "dev_user_overlap": 0,
            "exposure_budget": frozen["exposure_budget"], "full_token_audit": "passed"}


def make_smoke_data(project: Path, root: Path, *, verify_only: bool = False) -> None:
    tables = []
    for task, source in (("old", DATA / "old_interest.parquet"), ("new", shared.TRAIN)):
        table = pq.read_table(project / source)
        ids = table["sample_id"].to_pylist()
        indices = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"42:old-new-smoke:{ids[i]}".encode()).digest())[:32]
        require(len(indices) == 32, "Smoke needs 32 distinct examples per task")
        table = table.take(pa.array(indices, type=pa.int64()))
        if "training_task" not in table.column_names:
            table = table.append_column("training_task", pa.array(["new_interest"] * 32))
        target = root / "smoke_data" / f"{task}.parquet"
        if not verify_only:
            target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(pq.read_table(target).equals(table), "Existing smoke inputs differ")
        else:
            require(not verify_only, "Downloaded smoke data is missing")
            pq.write_table(table, target, compression="zstd")
        tables.append(table)
    joined = pa.concat_tables(tables)
    # The Trainer then shuffles this whole mixed set without replacement.
    order = [value for i in range(32) for value in (i, i + 32)]
    joined = joined.take(pa.array(order, type=pa.int64()))
    target = root / "smoke_data/mixed.parquet"
    if target.exists():
        require(pq.read_table(target).equals(joined), "Existing mixed smoke inputs differ")
    else:
        require(not verify_only, "Downloaded mixed smoke data is missing")
        pq.write_table(joined, target, compression="zstd")


def stage_specs(method: str, phase: str, project: Path, root: Path) -> list[dict[str, Any]]:
    require(method in METHODS and phase in {"smoke", "full"}, "Unknown method/phase")
    frozen = load_json(project / DATA_MANIFEST)
    if phase == "full":
        train = {"old": project / DATA / "old_interest.parquet", "new": project / shared.TRAIN, "mixed": project / DATA / "mixed.parquet"}
        sizes = {"old": frozen["old_rows"], "new": 21600, "mixed": frozen["mixed_rows"]}
        entries = [("old_sft", "old", None), ("sft", "new", "old_sft")] if method == "two_stage" else [("sft", "mixed", None)]
        return [{"key": key, "task": task, "parent": parent, "train_file": train[task], "samples": sizes[task], "epochs": 3, "accumulation": 32} for key, task, parent in entries]
    train = {task: root / "smoke_data" / f"{task}.parquet" for task in ("old", "new", "mixed")}
    if method == "two_stage":
        entries = [("single_old", "old", None, 5, 8), ("single_new", "new", "single_old", 5, 8),
                   ("old_sft", "old", None, 2, 32), ("sft", "new", "old_sft", 2, 32)]
    else:
        entries = [("single", "mixed", None, 5, 8), ("sft", "mixed", None, 2, 32)]
    return [{"key": key, "task": task, "parent": parent, "train_file": train[task], "samples": 64 if task == "mixed" else 32, "epochs": epochs, "accumulation": accumulation} for key, task, parent, epochs, accumulation in entries]


def gradient_command(project: Path, root: Path) -> list[str]:
    return [sys.executable, "-m", "new_interest.training.audit_semantic_id_token_gradients",
            "--config", str(project / "configs/sft/experiments/mind_old_new_interest_mixed_v1.json"),
            "--output", str(root / "gradient/gradient_audit.json"), "--batch-size", "1"]


def audit_stage(project: Path, method_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    from safetensors.torch import load_file

    contract = load_json(project / CONTRACT)
    folder = method_root / spec["key"]
    summary = load_json(folder / "run_summary.json")
    config = load_json(folder / "resolved_config.json")
    global_batch = spec.get("global_batch_size", spec["accumulation"])
    world_size = spec.get("world_size", 1)
    microbatch = spec.get("per_device_train_batch_size", 1)
    steps = spec["epochs"] * math.ceil(spec["samples"] / global_batch)
    actual_steps = spec.get("stop_after_global_step", steps)
    is_partial = actual_steps < steps
    expected = {"model_name": contract["model"]["model_name"], "model_revision": contract["model"]["model_revision"],
                "train_samples": spec["samples"], "configured_epochs": spec["epochs"],
                "expected_optimizer_steps": steps, "actual_optimizer_steps": actual_steps, "world_size": world_size,
                "per_device_train_batch_size": microbatch, "gradient_accumulation_steps": global_batch // (world_size * microbatch),
                "effective_global_batch_size": global_batch, "seed": 42, "enable_thinking": False,
                "max_seq_length": 2048, "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary"}
    for key, value in expected.items():
        require(summary.get(key) == value, f"{spec['key']} training differs: {key}")
    require(config["per_device_train_batch_size"] == microbatch
            and config["gradient_accumulation_steps"] == global_batch // (world_size * microbatch),
            "Actual stage execution config differs")
    for key, value in contract["recipe"].items():
        if key not in {"epochs", "gradient_accumulation_steps", "per_device_train_batch_size"}:
            require(config.get(key) == value, f"{spec['key']} recipe differs: {key}")
    require(summary["train_file_sha256"] == sha256(spec["train_file"]), "Actual train file differs")
    require(summary["sample_weight_file"] is None, "Unexpected weighting")
    require(summary["token_audit"]["samples"] == spec["samples"] and summary["token_audit"]["over_max_seq_length"] == 0, "Training token audit failed")
    require(math.isfinite(summary["train_metrics"]["train_loss"]), "Nonfinite training loss")
    state = load_json(folder / "trainer_state.json")
    require(state["global_step"] == actual_steps, "Actual Trainer step differs")
    if not is_partial:
        require(state["epoch"] == spec["epochs"], "Actual Trainer completion differs")
    else:
        require(config.get("stop_after_global_step") == actual_steps and 0 < state["epoch"] < spec["epochs"],
                "Partial probe stop differs or falsely claims full completion")
    initial = load_json(folder / "initial_adapter_state.json")
    start = load_json(folder / "training_start_audit.json")
    require(start["fresh_optimization_state"] and start["optimizer_state_entries"] == 0 and start["global_step"] == 0, "Optimizer/scheduler did not start fresh")
    require(initial["state_sha256"] == start["source_adapter_state_sha256"] == start["actual_adapter_state_sha256"], "Parameters changed before the first update")
    if spec["parent"]:
        parent = method_root / spec["parent"] / "final_adapter/adapter_model.safetensors"
        require(summary["checkpoint_identity"]["input_adapter_sha256"] == sha256(parent), "NI loaded the wrong OI adapter")
        actual_parent = tensor_state_manifest(load_file(str(parent)))["state_sha256"]
        require(initial["state_sha256"] == initial["input_adapter_state_sha256"] == actual_parent, "Stage handoff did not preserve every trained parameter")
    else:
        require(summary["input_adapter"] is None, "Unexpected warm-start adapter")
    if spec["key"].startswith("single"):
        history = summary["loss_history"]
        require(len(history) >= 2 and all(math.isfinite(item["loss"]) for item in history), "Missing finite learning-probe losses")
        require(history[-1]["loss"] < history[0]["loss"], "Small-sample learning probe did not decrease")
    semantic = summary["model_audit"]["semantic_id_tokens"]
    require(semantic["tie_word_embeddings"] is True and semantic["count"] == len(pq.read_table(project / shared.MAPPING)) and semantic["save_embedding_layers"] is False, "Native SID structure changed")
    require(not summary["model_audit"]["forbidden_trainable_parameters"], "Forbidden model parameters were trained")
    require(summary["semantic_token_initialization"]["initialization_version"] == "label-token-mean-norm-matched-cpu-fp32-v2", "SID initialization changed")
    require(summary["semantic_token_initialization"]["mapping_sha256"] == sha256(project / shared.MAPPING), "SID initialization mapping changed")
    final_state = tensor_state_manifest(load_file(str(folder / "final_adapter/adapter_model.safetensors")))["state_sha256"]
    require(final_state != initial["state_sha256"], "No trainable parameter changed in the stage")
    if not spec["key"].startswith("single"):
        drift = summary["semantic_token_drift"]
        require(drift["all_input_rows_changed"] and drift["all_output_rows_changed"], "Some SID rows were never updated")
    exposure = load_json(folder / ("training_exposure_partial.json" if is_partial else "training_exposure.json"))
    schema = pq.ParquetFile(spec["train_file"]).schema_arrow.names
    columns = ["sample_id", "target_subcategory"]
    columns += [name for name in ("training_task", "expected_assistant_tokens") if name in schema]
    expected_rows = pq.read_table(spec["train_file"], columns=columns).to_pylist()
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    expected_status = "verified_observed_cursor" if is_partial else "verified_observed_training"
    require(exposure["status"] == expected_status and exposure["identity"]["train_sha256"] == summary["train_file_sha256"], "Exposure ledger has another input identity")
    require(len(exposure["samples"]) == len(expected_by_id), "Exposure ledger coverage differs")
    actual_ids = set()
    totals: Counter[str] = Counter()
    if is_partial:
        from new_interest.training.old_new_execution import expected_counts
        counts, _ = expected_counts(spec["samples"], actual_steps, global_batch=global_batch, epochs=spec["epochs"])
        expected_presentations = {row["sample_id"]: count for row, count in zip(expected_rows, counts)}
    else:
        expected_presentations = {row["sample_id"]: spec["epochs"] for row in expected_rows}
    for row in exposure["samples"]:
        source = expected_by_id.get(row["sample_id"])
        require(source is not None and row["sample_id"] not in actual_ids, "Missing or duplicated exposure key")
        actual_ids.add(row["sample_id"])
        task = source.get("training_task", "new_interest")
        require(row["training_task"] == task and row["target_subcategory"] == source["target_subcategory"], "Exposure task/class identity differs")
        require(row["presentations"] == expected_presentations[row["sample_id"]], "Per-sample exposure budget differs")
        require(row["assistant_tokens"] == source.get("expected_assistant_tokens", 3) * expected_presentations[row["sample_id"]], "Supervised-token exposure differs")
        totals[task] += row["presentations"]
    require(exposure["presentations"] == sum(totals.values()), "Exposure total differs from individual records")
    for task, count in totals.items():
        require(exposure["by_task"][task]["presentations"] == count, "Task exposure aggregate differs")
    return {"samples": spec["samples"], "epochs": spec["epochs"], "optimizer_steps": actual_steps,
            "exposure_by_task": dict(totals), "initial_state_sha256": initial["state_sha256"],
            "stage_parent": spec["parent"], "train_file_sha256": summary["train_file_sha256"]}


def run_stage(project: Path, method_root: Path, spec: dict[str, Any], *, cold_state: str | None) -> None:
    folder = method_root / spec["key"]
    checkpoint = Path("/scratch") / f"mind-old-new-{method_root.name}-{spec['key']}-checkpoints"
    if not (folder / "run_summary.json").is_file():
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                   "-m", "new_interest.training.train_sft", "--config", str(project / f"configs/sft/experiments/mind_old_new_interest_{spec['task']}_v1.json"),
                   "--train-file", str(spec["train_file"]), "--output-dir", str(folder),
                   "--checkpoint-dir", str(checkpoint), "--persistent-checkpoint-dir", str(method_root / f"checkpoints-{spec['key']}"),
                   "--resume-from-checkpoint", "auto", "--epochs", str(spec["epochs"]),
                   "--per-device-train-batch-size", "1", "--gradient-accumulation-steps", str(spec["accumulation"]),
                   "--run-name", f"mind-old-new-{method_root.name}-{spec['key']}-seed42", "--save-steps", "100", "--wandb-mode", "disabled"]
        if spec["parent"]:
            command += ["--adapter-path", str(method_root / spec["parent"] / "final_adapter")]
        elif cold_state is not None:
            command += ["--expected-initial-adapter-state-sha256", cold_state]
        shared.run_command(command, project, method_root / "logs" / f"train-{spec['key']}.log")
    if not (folder / "trainer_state.json").is_file():
        state = checkpoint / "trainer_state.json"
        if not state.is_file():
            from new_interest.training.checkpoint_persistence import select_latest_complete_checkpoint
            summary = load_json(folder / "run_summary.json")
            persisted, _ = select_latest_complete_checkpoint(method_root / f"checkpoints-{spec['key']}", summary["checkpoint_identity"])
            require(persisted is not None, "No complete checkpoint contains the final Trainer state")
            state = persisted / "trainer_state.json"
        shutil.copyfile(state, folder / "trainer_state.json")
    audit_stage(project, method_root, spec)
    write_json(folder / "semantic_adapter_state_audit.json", shared.audit_adapter(project, "2b", folder))
    reload_path = folder / "semantic_adapter_reload_audit.json"
    if not reload_path.is_file():
        model = load_json(project / CONTRACT)["model"]
        shared.run_command([sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
                            "--model-name", model["model_name"], "--model-revision", model["model_revision"],
                            "--adapter-path", str(folder / "final_adapter"), "--mapping-file", str(project / shared.MAPPING),
                            "--reference-file", str(folder / "semantic_reload_reference.json"), "--output", str(reload_path),
                            "--atol", "0", "--rtol", "0"], project, method_root / "logs" / f"reload-{spec['key']}.log")
    shared.audit_reload(project, "2b", folder)
    print("OLD_NEW_STAGE_VERIFIED", method_root.name, spec["key"], flush=True)


def evaluate(project: Path, method_root: Path, method: str, phase: str) -> None:
    config = project / f"configs/id_token/mind_old_new_interest_{method}_v1.json"
    merged = method_root / "real_eval/merged"
    shards = 1 if phase == "smoke" else 4
    started = time.monotonic()
    for index in range(shards):
        target = merged if phase == "smoke" else method_root / f"real_eval/shard-{index:02d}"
        command = [sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking", "--config", str(config),
                   "--adapter-path", str(method_root / "sft/final_adapter"), "--output-dir", str(target),
                   "--num-shards", str(shards), "--shard-index", str(index), "--batch-size", "8"]
        if phase == "smoke":
            command += ["--max-examples", "64"]
        shared.run_command(command, project, method_root / "logs" / f"eval-{index}.log")
    if phase == "full":
        from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
        metrics = merge_id_token_shards([method_root / f"real_eval/shard-{index:02d}" for index in range(4)],
                                        project / shared.DEV, project / shared.MAPPING, merged, expected_records=31392)
        metrics.pop("elapsed_seconds_parallel_wall", None)
        metrics.pop("examples_per_second_parallel", None)
        metrics["execution_mode"] = "sequential_one_gpu"
        metrics["evaluation_seconds_in_this_worker_call"] = time.monotonic() - started
        write_json(merged / "metrics.json", metrics)
        merge_manifest = load_json(merged / "merge_manifest.json")
        merge_manifest["metrics_sha256"] = sha256(merged / "metrics.json")
        write_json(merged / "merge_manifest.json", merge_manifest)
    report = shared.audit_evaluation(project, "2b", method_root, phase)
    metrics = load_json(merged / "metrics.json")
    require(metrics["adapter_sha256"] == sha256_directory(method_root / "sft/final_adapter"), "Evaluator used another adapter")
    write_json(method_root / "evaluation_readback.json", report)


def audit_method(project: Path, root: Path, method: str, phase: str) -> dict[str, Any]:
    folder = root / method
    stages = stage_specs(method, phase, project, root)
    reports = {}
    for spec in stages:
        reports[spec["key"]] = audit_stage(project, folder, spec)
        shared.audit_adapter(project, "2b", folder / spec["key"])
        shared.audit_reload(project, "2b", folder / spec["key"])
    cold = {value["initial_state_sha256"] for value in reports.values() if value["stage_parent"] is None}
    require(len(cold) == 1, "Cold stages began with different parameters")
    keys = ["old_sft", "sft"] if method == "two_stage" else ["sft"]
    observed: Counter[str] = Counter()
    for key in keys:
        observed.update(reports[key]["exposure_by_task"])
    budget = load_json(project / DATA_MANIFEST)["exposure_budget"]
    expected = {"old_interest": 64, "new_interest": 64} if phase == "smoke" else {"old_interest": budget["old_presentations"], "new_interest": budget["new_presentations"]}
    require(dict(observed) == expected, "Methods do not match each task's accepted exposure budget")
    evaluation = shared.audit_evaluation(project, "2b", folder, phase)
    require(load_json(folder / "real_eval/merged/metrics.json")["adapter_sha256"] == sha256_directory(folder / "sft/final_adapter"), "Evaluation adapter lineage differs")
    return {"method": method, "stages": reports, "cold_initial_state_sha256": next(iter(cold)),
            "observed_exposure": dict(observed), "evaluation": evaluation}


def _artifact_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and not any(part.startswith("checkpoints-") for part in p.relative_to(root).parts)
                  and p.name not in {"artifact_manifest.json", "success.json", "readback_artifacts.tar.gz", "readback_package.json"}
                  and not p.name.endswith(".partial"))


def publish(project: Path, root: Path, methods: list[str], phase: str) -> None:
    verify_runtime(project)
    for relative, expected in load_json(project / RUNTIME_MANIFEST)["files"].items():
        target = root / "runtime_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
        require(sha256(target) == expected, "Published runtime source differs")
    shutil.copyfile(project / RUNTIME_MANIFEST, root / "runtime_source_manifest.json")
    shutil.copyfile(project / CONTRACT, root / "contract.json")
    shutil.copyfile(project / DATA_MANIFEST, root / "frozen_data_manifest.json")
    files = {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in _artifact_paths(root)}
    manifest = {"identity": identity(project), "phase": phase, "methods": methods, "files": files}
    write_json(root / "artifact_manifest.json", manifest)
    write_json(root / "success.json", {"identity": identity(project), "phase": phase, "methods": methods, "artifact_manifest_sha256": sha256(root / "artifact_manifest.json")})
    partial = root / "readback_artifacts.tar.gz.partial"
    with tarfile.open(partial, "w:gz") as archive:
        for rel in sorted(files):
            archive.add(root / rel, arcname=rel, recursive=False)
        for name in ("artifact_manifest.json", "success.json"):
            archive.add(root / name, arcname=name, recursive=False)
    partial.replace(root / "readback_artifacts.tar.gz")
    package = root / "readback_artifacts.tar.gz"
    write_json(root / "readback_package.json", {"sha256": sha256(package), "bytes": package.stat().st_size})


def readback(project: Path, root: Path, output: Path) -> dict[str, Any]:
    verify_runtime(project)
    marker = load_json(root / "success.json")
    manifest = load_json(root / "artifact_manifest.json")
    require(marker["artifact_manifest_sha256"] == sha256(root / "artifact_manifest.json"), "Artifact manifest was changed")
    require(marker["identity"] == manifest["identity"] == identity(project), "Run identity differs")
    require(marker["phase"] == manifest["phase"] and marker["methods"] == manifest["methods"], "Run phase or method differs")
    for relative, item in manifest["files"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
        p = root / path
        require(p.is_file() and p.stat().st_size == item["bytes"] and sha256(p) == item["sha256"], f"Downloaded artifact differs: {relative}")
    contract = load_json(project / CONTRACT)
    require(sha256(root / "contract.json") == sha256(project / CONTRACT), "Published protocol differs")
    require(sha256(root / "frozen_data_manifest.json") == sha256(project / DATA_MANIFEST), "Published data contract differs")
    require(sha256(root / "runtime_source_manifest.json") == sha256(project / RUNTIME_MANIFEST), "Published runtime identity differs")
    for relative, expected in load_json(root / "runtime_source_manifest.json")["files"].items():
        require(sha256(root / "runtime_sources" / relative) == expected, "Downloaded source code differs")
    require(load_json(root / "input_audit.json") == audit_inputs(project), "Worker input audit differs from local readback")
    environment = load_json(root / "environment.json")
    require(environment["packages"] == contract["worker_packages"] and environment["cuda_devices"] == 1, "Worker stack or world size differs")
    snapshot = load_json(root / "snapshot_audit.json")
    require(snapshot["model_revision"] == contract["model"]["model_revision"] and snapshot["files"]["config.json"] == contract["model"]["model_config_sha256"], "Model snapshot differs")
    phase = marker["phase"]
    methods = marker["methods"]
    require(methods == list(METHODS) if phase == "smoke" else len(methods) == 1 and methods[0] in METHODS, "Unexpected method coverage")
    if phase == "smoke":
        shared.audit_gradient(load_json(root / "gradient/gradient_audit.json"), contract["model"])
        make_smoke_data(project, root, verify_only=True)
    reports = {method: audit_method(project, root, method, phase) for method in methods}
    cold = {value["cold_initial_state_sha256"] for value in reports.values()}
    require(len(cold) == 1, "Methods did not share the same initial parameters")
    if phase == "full":
        accepted = load_json(root / "smoke_acceptance.json")
        require(accepted["identity"] == identity(project) and accepted["status"] == "passed" and accepted["phase"] == "smoke", "Full lacks an accepted independent smoke")
        require(next(iter(cold)) == accepted["cold_initial_state_sha256"], "Full was initialized from a different state")
        require(snapshot["files"] == accepted["snapshot_files"], "Full model weights differ from accepted smoke")
    result = {"version": "mind-old-new-independent-readback-v1", "status": "passed", "phase": phase,
              "identity": identity(project), "methods": reports, "cold_initial_state_sha256": next(iter(cold)),
              "snapshot_files": snapshot["files"], "artifact_manifest_sha256": sha256(root / "artifact_manifest.json"),
              "artifact_files_verified": len(manifest["files"]), "artifact_bytes_verified": sum(item["bytes"] for item in manifest["files"].values())}
    write_json(output, result)
    return result


def worker(project: Path, root: Path, phase: str, method: str, acceptance: Path | None) -> None:
    verify_runtime(project)
    methods = list(METHODS) if phase == "smoke" else [method]
    require(phase in {"smoke", "full"} and all(value in METHODS for value in methods), "Invalid worker method/phase")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "success.json").is_file() and (root / "readback_artifacts.tar.gz").is_file():
        existing = readback(project, root, Path("/scratch") / f"old-new-{method}-{phase}-existing-readback.json")
        require(existing["phase"] == phase and set(existing["methods"]) == set(methods), "Existing output belongs to another phase/method")
        package = load_json(root / "readback_package.json")
        require(package["sha256"] == sha256(root / "readback_artifacts.tar.gz"), "Existing readback package differs")
        print("OLD_NEW_REUSED_VERIFIED_OUTPUT", phase, methods, flush=True)
        return
    accepted = None
    if phase == "full":
        require(acceptance is not None and acceptance.is_file(), "Full requires downloaded independent smoke readback")
        accepted = load_json(acceptance)
        require(accepted["version"] == "mind-old-new-independent-readback-v1" and accepted["status"] == "passed" and accepted["phase"] == "smoke" and accepted["identity"] == identity(project), "Smoke acceptance identity is invalid")
        shutil.copyfile(acceptance, root / "smoke_acceptance.json")
    write_json(root / "run_identity.json", identity(project))
    write_json(root / "input_audit.json", audit_inputs(project))
    shared.environment_audit(project, root / "environment.json")
    shared.prepare_snapshot(project, "2b", root / "snapshot_audit.json")
    if accepted is not None:
        require(load_json(root / "snapshot_audit.json")["files"] == accepted["snapshot_files"], "Full changed its model snapshot")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if phase == "smoke":
        make_smoke_data(project, root)
        if not (root / "gradient/gradient_audit.json").is_file():
            shared.run_command(gradient_command(project, root), project, root / "gradient.log")
        shared.audit_gradient(load_json(root / "gradient/gradient_audit.json"), load_json(project / CONTRACT)["model"])
    for value in methods:
        folder = root / value
        folder.mkdir(parents=True, exist_ok=True)
        for spec in stage_specs(value, phase, project, root):
            run_stage(project, folder, spec, cold_state=accepted["cold_initial_state_sha256"] if accepted is not None else None)
        evaluate(project, folder, value, phase)
        write_json(folder / "method_gate.json", audit_method(project, root, value, phase))
        print("OLD_NEW_METHOD_VERIFIED", value, phase, flush=True)
    reports = {value: audit_method(project, root, value, phase) for value in methods}
    require(len({value["cold_initial_state_sha256"] for value in reports.values()}) == 1, "Initial parameters differ across methods")
    publish(project, root, methods, phase)
    print("OLD_NEW_WORKER_GATE_PASSED", phase, methods, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "worker", "readback"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--smoke-acceptance", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = args.root.resolve()
    if args.action == "preflight":
        verify_runtime(project)
        write_json(root / "input_audit.json", audit_inputs(project))
        make_smoke_data(project, root)
        print("OLD_NEW_LOCAL_PREFLIGHT_PASSED")
    elif args.action == "worker":
        worker(project, root, args.phase, args.method, args.smoke_acceptance)
    else:
        require(args.output is not None, "Readback requires a receipt output path")
        result = readback(project, root, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
