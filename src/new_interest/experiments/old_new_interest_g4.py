"""Four-GPU continuation of the frozen OI/NI experiment, with separate receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from typing import Any

import pyarrow.parquet as pq

from new_interest.experiments import old_new_interest as original
from new_interest.experiments import backbone_scaling as shared
from new_interest.training.old_new_execution import (
    audit_checkpoint_cursor, epoch_permutation, make_cursor_contract,
)

OVERLAY = Path("configs/experiments/mind_old_new_interest_g4_execution_v1.json")
RUNTIME = Path("manifests/mind_old_new_runtime_g4_v1.json")
INPUTS = Path("manifests/mind_old_new_execution_inputs_v1.json")
LENGTHS = Path("migration_inputs/all_token_lengths.parquet")
load_json, sha256, require, write_json = shared.load_json, shared.sha256, shared.require, original.write_json


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execution_recipe(project: Path) -> dict[str, Any]:
    overlay = load_json(project / OVERLAY)
    require(overlay["source_protocol_sha256"] == sha256(project / original.CONTRACT), "Execution overlay changed the scientific protocol")
    micro = overlay["chosen_per_device_batch_size"]
    require(micro in (2, 4, 8), "Backbone has not selected an accepted 2B execution recipe")
    profile = overlay.get("kernel_profile")
    require(profile in {"torch-fallback", "fla-0.5.2-causal-conv1d-1.7.0"}, "Backbone kernel profile has not been selected")
    require(overlay["world_size"] == 4 and overlay["effective_global_batch_size"] == 32, "Execution batch/world changed")
    require(overlay["ignore_data_skip"] is False, "Cursor would be skipped twice")
    require(overlay["two_stage_optimizer_steps"] == [1605, 2025] and overlay["mixed_optimizer_steps"] == 3630,
            "Execution overlay changed optimizer budgets")
    require(overlay.get("kernel_recipe_file") and overlay.get("kernel_recipe_sha256"), "The common kernel recipe is not frozen")
    kernel = project / overlay["kernel_recipe_file"]
    require(sha256(kernel) == overlay["kernel_recipe_sha256"], "Common kernel recipe changed")
    require(overlay.get("recipe_selection_receipt_file"), "The shared 2B benchmark receipt is missing")
    selection = project / overlay["recipe_selection_receipt_file"]
    require(sha256(selection) == overlay["recipe_selection_receipt_sha256"], "Shared 2B benchmark receipt changed")
    accepted = load_json(selection)
    require(accepted["version"] == "mind-four-gpu-probe-readback-v1" and accepted["status"] == "passed"
            and accepted["model_size"] == "2b", "The shared recipe lacks independent 2B acceptance")
    selected = accepted["selected"]
    require(selected["status"] == "passed" and selected["eligible_memory_headroom"]
            and selected["actual_four_rank_restore"] and selected["actual_sample_cursor_exact"],
            "The shared recipe did not pass restore/cursor/memory gates")
    require(selected["per_device_batch_size"] == micro and selected["gradient_accumulation_steps"] == 32 // (4 * micro)
            and selected["gradient_checkpointing_enabled"] == bool(overlay.get("gradient_checkpointing_enabled", True))
            and accepted["kernel_profile"] == profile, "Execution differs from the selected shared 2B recipe")
    require(selected["reload"]["all_tensors_exact"] and selected["reload"]["logits_exact"]
            and accepted["evaluation"]["records"] == 64, "Selected recipe lacks exact reload and real Dev acceptance")
    return {"world_size": 4, "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": 32 // (4 * micro),
            "kernel_profile": profile,
            "gradient_checkpointing_enabled": bool(overlay.get("gradient_checkpointing_enabled", True)),
            "kernel_recipe_sha256": overlay["kernel_recipe_sha256"],
            "recipe_selection_receipt_sha256": overlay["recipe_selection_receipt_sha256"]}


def identity(project: Path) -> dict[str, Any]:
    return {"version": "mind-old-new-four-gpu-run-v1", "contract_sha256": sha256(project / original.CONTRACT),
            "data_manifest_sha256": sha256(project / original.DATA_MANIFEST),
            "execution_overlay_sha256": sha256(project / OVERLAY),
            "runtime_manifest_sha256": sha256(project / RUNTIME),
            "migration_inputs_manifest_sha256": sha256(project / INPUTS),
            "execution_recipe": execution_recipe(project)}


def verify_runtime(project: Path) -> None:
    import ast
    for name, digest in load_json(project / RUNTIME)["files"].items():
        require(sha256(project / name) == digest, f"Frozen G4 source differs: {name}")
    integration = load_json(project / "manifests/mind_old_new_shared_core_integration_v2.json")
    require(sha256(project / "manifests/qwen35_shared_execution_core_v2.json") == integration["source_manifest_sha256"],
            "Shared execution origin changed")
    for name, digest in integration["copied_files"].items():
        require(sha256(project / name) == digest, "Shared execution module changed after integration")
    trainer_source = (project / "src/new_interest/training/train_sft.py").read_text()
    for name, digest in integration["seams"].items():
        node = next(item for item in ast.parse(trainer_source).body if isinstance(item, ast.FunctionDef) and item.name == name)
        source = "".join(trainer_source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        require(hashlib.sha256(source.encode()).hexdigest() == digest, "Shared checkpoint I/O seam changed")
    for name, entry in load_json(project / INPUTS)["files"].items():
        path = project / name
        require(path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                f"Frozen migration input differs: {name}")


def source_info(project: Path, method: str) -> dict[str, Any]:
    info = load_json(project / INPUTS)["sources"][method]
    checkpoint = project / info["checkpoint"]
    marker = load_json(checkpoint / "checkpoint_complete.json")
    require(sha256(checkpoint / "checkpoint_complete.json") == info["checkpoint_marker_sha256"], "Source checkpoint marker differs")
    return {**info, "checkpoint_path": checkpoint, "stage_path": project / info["stage_artifacts"],
            "step": int(marker["global_step"]), "identity": marker["identity"]}


def specs(project: Path, root: Path, method: str, *, probe: bool) -> list[dict[str, Any]]:
    recipe = execution_recipe(project)
    result = original.stage_specs(method, "full", project, root)
    if probe:
        result = result[:1]
    for spec in result:
        spec.update({"world_size": 4, "per_device_train_batch_size": recipe["per_device_train_batch_size"],
                     "global_batch_size": 32})
        if spec["parent"] is None:
            spec["migration_source_method"] = method
            if probe:
                spec["stop_after_global_step"] = source_info(project, method)["step"] + 6
    return result


def tail_specs(project: Path, method: str) -> list[dict[str, Any]]:
    recipe = execution_recipe(project)
    entries = [("old_sft", "old", None, 42), ("sft", "new", "old_sft", 64)] if method == "two_stage" else [("sft", "mixed", None, 42)]
    return [{"key": key, "task": task, "parent": parent, "samples": size,
             "train_file": project / f"migration_inputs/tail_{task}.parquet", "epochs": 3,
             "accumulation": 32, "global_batch_size": 32, "world_size": 4,
             "per_device_train_batch_size": recipe["per_device_train_batch_size"]}
            for key, task, parent, size in entries]


def run_command(command: list[str], project: Path, log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if any(marker in line for marker in ("loss", "Error", "Traceback", "CHECKPOINT_", "EXECUTION_", "OLD_NEW_", "%|")):
                print(line.strip()[:1600], flush=True)
        code = process.wait()
    require(code == 0, f"Command failed with exit {code}; see {log}")


def _single_gpu_env(rank: int) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                 "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [value.strip() for value in visible.split(",")]
        require(len(devices) == 4 and 0 <= rank < 4, "Evaluation must use the allocated four GPUs")
        env["CUDA_VISIBLE_DEVICES"] = devices[rank]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
    return env


def training_config(project: Path, method_root: Path, spec: dict[str, Any], *, stop: int | None = None) -> dict[str, Any]:
    recipe = execution_recipe(project)
    folder = method_root / spec["key"]
    folder.mkdir(parents=True, exist_ok=True)
    source = source_info(project, spec["migration_source_method"]) if "migration_source_method" in spec else None
    stage = (spec["migration_source_method"] if source else ("two_stage" if spec["task"] != "mixed" else "mixed")) + "/" + spec["key"]
    cursor = make_cursor_contract(spec["train_file"], stage=stage, world_size=4,
        per_device_batch_size=recipe["per_device_train_batch_size"],
        source_checkpoint=source["checkpoint_path"] if source else None,
        token_lengths_file=project / LENGTHS)
    cursor_file = folder / "cursor_contract.json"
    if cursor_file.exists():
        require(load_json(cursor_file) == cursor, "Stage cursor was changed")
    else:
        write_json(cursor_file, cursor)
    config = load_json(project / f"configs/sft/experiments/mind_old_new_interest_{spec['task']}_v1.json")
    scratch_id = hashlib.sha256(str(method_root).encode()).hexdigest()[:12]
    config.update({"train_file": str(spec["train_file"]), "output_dir": str(folder),
        "checkpoint_dir": f"/scratch/old-new-g4-{scratch_id}-{spec['key']}",
        "persistent_checkpoint_dir": str(method_root / f"checkpoints-{spec['key']}"),
        "resume_from_checkpoint": "auto", "old_new_cursor_contract_file": str(cursor_file),
        "ignore_data_skip": False, "epochs": spec["epochs"], "wandb_mode": "disabled",
        "per_device_train_batch_size": recipe["per_device_train_batch_size"],
        "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
        "gradient_checkpointing_enabled": recipe["gradient_checkpointing_enabled"],
        "run_name": f"mind-old-new-{method_root.name}-{spec['key']}-seed42", "save_steps": 100})
    if spec["parent"]:
        config["adapter_path"] = str(method_root / spec["parent"] / "final_adapter")
    else:
        config.pop("adapter_path", None)
    if source:
        config["run_name"] = source["identity"]["run_name"]
        config["execution_migration"] = {"source_checkpoint": str(source["checkpoint_path"]),
            "source_stage_artifacts": str(source["stage_path"]), "cursor_contract_file": str(cursor_file)}
    if stop is not None:
        config["stop_after_global_step"] = stop
    return config


def train_to(project: Path, method_root: Path, spec: dict[str, Any], *, stop: int | None = None) -> None:
    folder = method_root / spec["key"]
    config = training_config(project, method_root, spec, stop=stop)
    expected = stop if stop is not None else 3 * math.ceil(spec["samples"] / 32)
    if (folder / "run_summary.json").exists():
        previous = load_json(folder / "run_summary.json")
        completed = previous["actual_optimizer_steps"]
        limit = spec.get("stop_after_global_step", 3 * math.ceil(spec["samples"] / 32))
        require(completed <= limit, "Existing stage exceeded its complete budget")
        if completed >= expected:
            return
        archive = folder / "probe_segments" / f"step-{completed}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("run_summary.json", "resolved_config.json", "trainer_state.json", "training_exposure_partial.json", "old_new_execution_cursor.json"):
            source = folder / name
            if source.exists():
                destination = archive / name
                if destination.exists():
                    require(sha256(destination) == sha256(source), "Probe segment evidence changed")
                else:
                    shutil.copyfile(source, destination)
    config_file = method_root / "launch_configs" / f"{spec['key']}-to-{expected}.json"
    write_json(config_file, config)
    run_command([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4",
                 "-m", "new_interest.training.train_sft", "--config", str(config_file)],
                project, method_root / "logs" / f"train-{spec['key']}-to-{expected}.log")
    shutil.copyfile(Path(config["checkpoint_dir"]) / "trainer_state.json", folder / "trainer_state.json")
    require(load_json(folder / "run_summary.json")["actual_optimizer_steps"] == expected, "Actual training endpoint differs")


def finish_stage(project: Path, method_root: Path, spec: dict[str, Any]) -> None:
    from new_interest.training.checkpoint_persistence import hydrate_checkpoint, select_latest_complete_checkpoint
    folder = method_root / spec["key"]
    original.audit_stage(project, method_root, spec)
    write_json(folder / "semantic_adapter_state_audit.json", shared.audit_adapter(project, "2b", folder))
    output = folder / "semantic_adapter_reload_audit.json"
    if not output.exists():
        model = load_json(project / original.CONTRACT)["model"]
        run_command([sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
            "--model-name", model["model_name"], "--model-revision", model["model_revision"],
            "--adapter-path", str(folder / "final_adapter"), "--mapping-file", str(project / shared.MAPPING),
            "--reference-file", str(folder / "semantic_reload_reference.json"), "--output", str(output),
            "--atol", "0", "--rtol", "0"], project, method_root / "logs" / f"reload-{spec['key']}.log", env=_single_gpu_env(0))
    shared.audit_reload(project, "2b", folder)
    summary = load_json(folder / "run_summary.json")
    require(load_json(folder / "resolved_config.json")["gradient_checkpointing_enabled"]
            == execution_recipe(project)["gradient_checkpointing_enabled"], "Stage checkpointing recipe differs")
    checkpoint, _ = select_latest_complete_checkpoint(method_root / f"checkpoints-{spec['key']}", summary["checkpoint_identity"])
    require(checkpoint is not None, "Final stage has no complete persisted optimizer checkpoint")
    require(load_json(checkpoint / "trainer_state.json")["global_step"] == summary["actual_optimizer_steps"], "Persisted final step differs")
    target_root = folder / "accepted_checkpoints"
    candidate = target_root / checkpoint.name
    if not candidate.exists():
        hydrate_checkpoint(checkpoint, target_root, summary["checkpoint_identity"])
    marker = candidate / "checkpoint_complete.json"
    if not marker.exists():
        shutil.copyfile(checkpoint / "checkpoint_complete.json", marker)
    require(sha256(marker) == sha256(checkpoint / "checkpoint_complete.json"), "Accepted checkpoint marker changed")
    audit_stage(project, method_root, spec)
    print("OLD_NEW_G4_STAGE_VERIFIED", method_root.name, spec["key"], summary["actual_optimizer_steps"], flush=True)


def audit_cursor_journals(folder: Path, cursor: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    saved = load_json(checkpoint / "old_new_execution_cursor.json")
    require(saved["last_update_audit"]["global_step"] == saved["global_step"], "Saved update journal endpoint differs")
    orders, endings, tails, records = {}, set(), 0, 0
    for start_path in sorted((folder / "cursor_attempts").glob("*/start.json")):
        start = load_json(start_path)
        require(start["contract"] == cursor and start["status"] == "passed", "Cursor attempt identity differs")
        previous, step = None, start["restored_global_step"]
        trace = start_path.parent / "updates.jsonl"
        if not trace.exists():
            continue
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            digest = row.pop("record_sha256")
            require(digest == canonical_sha(row), "Actual batch journal hash differs")
            require(row["previous_record_sha256"] == previous, "Actual batch journal chain is broken")
            previous = digest
            step += 1
            require(row["global_step"] == step, "Actual update journal skipped or repeated a step")
            epoch, epoch_step = divmod(step - 1, math.ceil(cursor["samples"] / 32))
            if epoch not in orders:
                orders[epoch] = epoch_permutation(cursor["samples"], epoch, cursor["seed"])
            expected = orders[epoch][epoch_step * 32:(epoch_step + 1) * 32]
            require(row["indices"] == expected and row["indices_sha256"] == canonical_sha(expected), "Actual input order differs")
            require(row["real_presentations"] == len(expected) and row["supervised_tokens"] == len(expected) * 3
                    and row["global_supervised_token_denominator"] == len(expected) * 3, "Actual batch denominator differs")
            width = cursor["world_size"] * cursor["per_device_batch_size"]
            require(row["masked_padding_slots"] == (-len(expected)) % width, "Masked tail padding differs")
            tails += int(row["epoch_tail"])
            records += 1
            endings.add(digest)
    require(saved["last_update_audit"]["record_sha256"] in endings, "Committed checkpoint lacks its actual batch journal")
    return {"attempts": len(list((folder / "cursor_attempts").glob("*/start.json"))),
            "verified_update_records": records, "verified_epoch_tails": tails, "status": "passed"}


def audit_stage(project: Path, method_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from new_interest.training.checkpoint_persistence import validate_persistent_checkpoint
    from new_interest.training.execution_migration import checkpoint_state_audit
    from new_interest.training.sft_state_audit import tensor_state_manifest
    report = original.audit_stage(project, method_root, spec)
    folder = method_root / spec["key"]
    summary = load_json(folder / "run_summary.json")
    checkpoint = folder / "accepted_checkpoints" / f"checkpoint-{report['optimizer_steps']}"
    require(load_json(folder / "resolved_config.json")["gradient_checkpointing_enabled"]
            == execution_recipe(project)["gradient_checkpointing_enabled"], "Stage checkpointing recipe differs")
    validate_persistent_checkpoint(checkpoint, summary["checkpoint_identity"])
    checkpoint_audit = checkpoint_state_audit(checkpoint, summary["checkpoint_identity"])
    tensors = load_file(str(folder / "final_adapter/adapter_model.safetensors"))
    require(all(bool(torch.isfinite(value).all()) for value in tensors.values()), "Final adapter contains nonfinite values")
    final_state = tensor_state_manifest(tensors)
    require(checkpoint_audit["adapter_state_sha256"] == final_state["state_sha256"], "Final adapter differs from the complete checkpoint")
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    for entry in optimizer["state"].values():
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), "Persisted optimizer contains nonfinite values")
                if key == "exp_avg_sq":
                    require(bool((value >= 0).all()), "Persisted second moments are negative")
    checkpoint_audit["all_optimizer_and_adapter_values_finite"] = True
    cursor = load_json(folder / "cursor_contract.json")
    saved_cursor = audit_checkpoint_cursor(checkpoint, cursor)
    recorded_cursor = load_json(checkpoint / "old_new_execution_cursor.json")
    require(all(recorded_cursor[key] == value for key, value in saved_cursor.items()), "Checkpoint cursor evidence differs")
    make_cursor_contract(spec["train_file"], stage=cursor["stage"], world_size=4,
        per_device_batch_size=cursor["per_device_batch_size"], source_checkpoint=checkpoint,
        token_lengths_file=project / LENGTHS)
    report["cursor_journal"] = audit_cursor_journals(folder, cursor, checkpoint)
    report["complete_checkpoint"] = checkpoint_audit
    report["reload"] = shared.audit_reload(project, "2b", folder)
    if "migration_source_method" in spec:
        source = source_info(project, spec["migration_source_method"])
        source_audit = checkpoint_state_audit(source["checkpoint_path"], source["identity"])
        start = load_json(folder / "execution_migration_start.json")
        require(start["status"] == "passed" and start["source_identity"] == source["identity"]
                and start["target_identity"] == summary["checkpoint_identity"], "Migration start identity differs")
        require(start["global_step"] == source["step"] and len(start["rank_restores"]) == 4, "Actual migration rank coverage differs")
        for rank, row in enumerate(start["rank_restores"]):
            require(row["rank"] == rank and row["status"] == "passed" and not row["differences"], "A rank did not restore exactly")
            require(all(row["actual"][key] == source_audit[key] for key in
                ("global_step", "epoch", "max_steps", "adapter_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256")),
                "Actual source model/optimizer/scheduler state differs")
            rng = load_json(folder / f"execution_rng_restore_rank_{rank}.json")
            require(rng["status"] == "passed" and rng["actual_rng_sha256"] == rng["expected_rng_sha256"], "RNG restore was not audited")
        require(sha256(folder / "training_start_audit.json") == sha256(source["stage_path"] / "training_start_audit.json"),
                "Original first-start evidence changed")
        report["migration_source_step"] = source["step"]
        report["execution_lineage_id"] = start["lineage_id"]
        overlay = load_json(project / OVERLAY)
        reference_path = project / overlay["actual_model_kernel_bindings_file"]
        require(sha256(reference_path) == overlay["actual_model_kernel_bindings_sha256"], "Model kernel reference changed")
        reference = load_json(reference_path)
        kernel_files = list((folder / "execution_attempts").glob("*/kernels_rank_*.json"))
        require(len(kernel_files) >= 4, "Missing actual model kernel bindings")
        for path in kernel_files:
            actual = load_json(path)
            require(actual["module_bindings"] == reference["module_bindings"]
                    and actual["is_fast_path_available"] == reference["is_fast_path_available"], "Actual model kernel bindings differ")
            require(actual["linear_attention_modules"] == reference["linear_attention_modules"],
                    "Actual gated-delta layer bindings differ from the shared 2B model")
        report["actual_model_kernel_binding_audits"] = len(kernel_files)
    return report


def evaluate(project: Path, method_root: Path, method: str, *, probe: bool, adapter_stage: str) -> dict[str, Any]:
    from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
    config = project / f"configs/id_token/mind_old_new_interest_{method}_v1.json"
    def one(index: int) -> None:
        command = [sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking", "--config", str(config),
            "--adapter-path", str(method_root / adapter_stage / "final_adapter"),
            "--output-dir", str(method_root / f"real_eval/shard-{index:02d}"),
            "--num-shards", "4", "--shard-index", str(index), "--batch-size", "8"]
        if probe:
            command += ["--max-examples", "64"]
        run_command(command, project, method_root / "logs" / f"eval-{index}.log", env=_single_gpu_env(index))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in [executor.submit(one, index) for index in range(4)]:
            future.result()
    merge_id_token_shards([method_root / f"real_eval/shard-{index:02d}" for index in range(4)],
        project / shared.DEV, project / shared.MAPPING, method_root / "real_eval/merged", expected_records=64 if probe else 31392)
    report = shared.audit_evaluation(project, "2b", method_root, "smoke" if probe else "full", logical_shards=4)
    metrics = load_json(method_root / "real_eval/merged/metrics.json")
    from new_interest.evaluation.evaluate_beam import sha256_directory
    require(metrics["adapter_sha256"] == sha256_directory(method_root / adapter_stage / "final_adapter"), "Evaluation adapter lineage differs")
    write_json(method_root / "evaluation_readback.json", report)
    return report


def environment_audit(project: Path, output: Path) -> None:
    import torch
    contract, overlay = load_json(project / original.CONTRACT), load_json(project / OVERLAY)
    fast = overlay["kernel_profile"] != "torch-fallback"
    expected = {**contract["worker_packages"], **(overlay.get("kernel_packages", {}) if fast else {})}
    versions = {name: importlib.metadata.version(name) for name in expected}
    require(versions == expected, "G4 worker package versions differ")
    require(torch.cuda.device_count() == 4, "Exactly four visible GPUs are required")
    devices = [{"name": torch.cuda.get_device_properties(rank).name,
                "memory_bytes": torch.cuda.get_device_properties(rank).total_memory} for rank in range(4)]
    require(all("A100" in gpu["name"] and gpu["memory_bytes"] >= 79 * 1024**3 for gpu in devices), "Expected four A100-80GB GPUs")
    write_json(output, {"packages": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
                        "cuda_devices": 4, "devices": devices})
    if not fast:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
        bindings = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn",
                    "causal_conv1d_update", "FusedRMSNormGated")
        active = [name for name in bindings if getattr(implementation, name, None) is not None]
        require(not active and not implementation.is_fast_path_available, "Selected reference profile has unexpected optional kernels")
        write_json(output.parent / "reference_kernel_environment.json", {"kernel_profile": "torch-fallback",
            "is_fast_path_available": False, "active_optional_functions": active, "packages": versions})


def audit_environment(project: Path, root: Path) -> dict[str, Any]:
    overlay = load_json(project / OVERLAY)
    contract = load_json(project / original.CONTRACT)
    environment = load_json(root / "environment.json")
    fast = overlay["kernel_profile"] != "torch-fallback"
    expected = {**contract["worker_packages"], **(overlay.get("kernel_packages", {}) if fast else {})}
    require(environment["packages"] == expected and environment["cuda_devices"] == 4, "Actual G4 environment differs")
    require(len(environment["devices"]) == 4 and all("A100" in item["name"] and item["memory_bytes"] >= 79 * 1024**3
        for item in environment["devices"]), "Actual accelerator identities differ")
    if not fast:
        reference = load_json(root / "reference_kernel_environment.json")
        require(reference["kernel_profile"] == "torch-fallback" and not reference["is_fast_path_available"]
                and not reference["active_optional_functions"] and reference["packages"] == expected,
                "Actual reference kernel profile differs")
        return {"packages": expected, "cuda_devices": 4, "reference_bindings_verified": True}
    installed = load_json(root / "fast_kernel_environment.json")
    accepted = load_json(project / "manifests/qwen35_fast_kernel_component_acceptance_v1.json")
    require(installed["base_runtime_before"] == installed["base_runtime_after"], "Kernel setup changed the base runtime")
    require(installed["manifest_sha256"] == overlay["kernel_recipe_sha256"], "Installed kernel wheel manifest differs")
    require(installed["installer_sha256"] == sha256(project / "scripts/amlt/common/install_qwen35_fast_kernels.py"),
            "Installed kernel source differs")
    require(installed["bindings"] == accepted["fast_bindings"], "Actual kernel function bindings differ from the shared accepted stack")
    wheels = load_json(project / overlay["kernel_recipe_file"])["install_wheels"]
    expected_wheels = [{"filename": entry["filename"], "sha256": entry["sha256"], "bytes": entry["size"]} for entry in wheels]
    require(installed["wheels"] == expected_wheels, "Installed wheel bytes differ")
    return {"packages": expected, "cuda_devices": 4, "fast_bindings_verified": True}


def validate_recovery_runtime(project: Path, receipt: dict[str, Any]) -> None:
    """Allow only the reviewed reader/orchestration repair around unchanged training."""
    import ast
    require(receipt["version"] == "old-new-g4-two-stage-recovery-v1"
            and receipt["status"] == "passed" and receipt["phase"] == "two_stage_migration_smoke_component"
            and receipt["methods"] == ["two_stage"]
            and receipt["diagnostic_optimizer_updates_preserved"] == 18
            and receipt["mixed_optimizer_updates_remaining"] == 12,
            "Unexpected diagnostic recovery acceptance")
    require(receipt["execution_recipe"] == execution_recipe(project), "Recovery execution recipe changed")
    for key in ("contract_sha256", "data_manifest_sha256", "execution_overlay_sha256"):
        require(receipt["source_identity"][key] == identity(project)[key], "Recovery scientific identity changed")
    source_root = project / "migration_inputs/recovery/source_runtime"
    source_manifest = project / "migration_inputs/recovery/source_runtime_manifest.json"
    require(sha256(source_manifest) == receipt["source_runtime_manifest_sha256"], "Recovery source manifest changed")
    previous = load_json(source_manifest)["files"]
    require(previous == receipt["source_runtime_files"], "Recovery source inventory changed")
    current = load_json(project / RUNTIME)["files"]
    require(set(previous) == set(current), "Recovery introduced or removed runtime modules")
    allowed = {
        "src/new_interest/experiments/backbone_scaling.py": {"audit_evaluation"},
        "src/new_interest/experiments/old_new_interest_g4.py": {
            "evaluate", "readback", "worker", "validate_recovery_runtime", "audit_recovery_seed", "seed_recovery",
        },
    }
    for name, digest in previous.items():
        if current[name] == digest:
            continue
        require(name in allowed, "Recovery changes frozen training or evaluation-generation code")
        old_path = source_root / name
        require(sha256(old_path) == digest, "Original recovery source bytes changed")
        def protected(path: Path) -> list[str]:
            nodes = ast.parse(path.read_text()).body
            return [ast.dump(node, include_attributes=False) for node in nodes
                    if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed[name])]
        require(protected(old_path) == protected(project / name), "Recovery changes protected implementation")


def audit_recovery_seed(project: Path, root: Path, *, probe: bool) -> bool:
    accepted_path = project / "migration_inputs/recovery/accepted_two_stage.json"
    if not accepted_path.exists():
        return False
    require(probe, "Diagnostic recovery artifacts cannot initialize formal Full")
    accepted = load_json(accepted_path)
    validate_recovery_runtime(project, accepted)
    actual = load_json(root / "recovery_seed_receipt.json")
    require(actual["status"] == "passed" and actual["recovery_acceptance_sha256"] == sha256(accepted_path)
            and actual["source_backend"] == accepted["source_backend"], "Recovery seed provenance differs")
    require(sha256(root / "recovery_acceptance.json") == sha256(accepted_path), "Published recovery acceptance changed")
    require(len(accepted["files"]) == accepted["file_count"]
            and sum(item["bytes"] for item in accepted["files"].values()) == accepted["bytes"], "Recovery file budget differs")
    for name, item in accepted["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] in {"two_stage", "tail_two_stage"}, "Recovery path is outside the accepted component")
        target = root / relative
        require(target.is_file() and not target.is_symlink()
                and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"],
                f"Recovered artifact changed: {name}")
    return True


def seed_recovery(project: Path, root: Path, *, probe: bool) -> bool:
    accepted_path = project / "migration_inputs/recovery/accepted_two_stage.json"
    if not accepted_path.exists():
        return False
    require(probe, "Diagnostic recovery artifacts cannot initialize formal Full")
    accepted = load_json(accepted_path)
    validate_recovery_runtime(project, accepted)
    if (root / "recovery_seed_receipt.json").exists():
        return audit_recovery_seed(project, root, probe=probe)
    prefix = Path(accepted["source_blob_prefix"])
    require(not prefix.is_absolute() and ".." not in prefix.parts
            and prefix.as_posix().startswith("projects/sft4safety/amlt-results/"), "Recovery source is outside task storage")
    candidates = [Path(os.environ.get("AMLT_DEFAULT", "/mnt/default")) / prefix,
                  root.parent / prefix.name]
    source = next((path for path in candidates if (path / "two_stage/old_sft/run_summary.json").is_file()), None)
    require(source is not None and source.resolve() != root.resolve(), "Accepted recovery source is not mounted")
    def copy(item: tuple[str, dict[str, Any]]) -> None:
        name, expected = item
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] in {"two_stage", "tail_two_stage"}, "Unsafe diagnostic recovery path")
        target = root / relative
        if target.exists():
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == expected["bytes"]
                    and sha256(target) == expected["sha256"], "Existing recovered artifact has another identity")
            return
        origin = source / relative
        require(origin.is_file() and not origin.is_symlink() and origin.stat().st_size == expected["bytes"]
                and sha256(origin) == expected["sha256"], "Source recovery artifact differs from independent acceptance")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".recovery.partial")
        shutil.copyfile(origin, temporary)
        require(temporary.stat().st_size == expected["bytes"] and sha256(temporary) == expected["sha256"], "Recovery copy differs")
        temporary.replace(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy, accepted["files"].items()))
    shutil.copyfile(accepted_path, root / "recovery_acceptance.json")
    write_json(root / "recovery_seed_receipt.json", {
        "status": "passed", "source_backend": accepted["source_backend"],
        "source_root": str(source), "recovery_acceptance_sha256": sha256(accepted_path),
        "files": accepted["file_count"], "bytes": accepted["bytes"],
        "diagnostic_optimizer_updates_preserved": 18,
    })
    return audit_recovery_seed(project, root, probe=probe)


def validate_epoch_repair_runtime(project: Path, receipt: dict[str, Any]) -> None:
    """Bind reused diagnostics to the accepted runtime and one epoch-forwarding fix."""
    import ast
    require(receipt["version"] == "old-new-g4-epoch-repair-components-v1"
            and receipt["status"] == "passed"
            and receipt["phase"] == "reused_components_only"
            and receipt["reused_prefixes"] == ["mixed", "tail_mixed", "tail_two_stage"]
            and receipt["diagnostic_optimizer_updates_preserved"] == 24
            and receipt["new_two_stage_optimizer_updates"] == 6,
            "Unexpected epoch-repair component acceptance")
    inputs = project / "migration_inputs/epoch_repair"
    previous_acceptance = load_json(inputs / "source_acceptance.json")
    require(sha256(inputs / "source_acceptance.json") == receipt["source_acceptance_sha256"]
            and previous_acceptance["status"] == "passed"
            and previous_acceptance["phase"] == "migration_smoke"
            and previous_acceptance["artifact_manifest_sha256"] == receipt["source_artifact_manifest_sha256"],
            "Prior complete smoke acceptance changed")
    source_manifest = inputs / "source_artifact_manifest.json"
    require(sha256(source_manifest) == previous_acceptance["artifact_manifest_sha256"],
            "Prior complete smoke artifact manifest changed")
    accepted_files = load_json(source_manifest)["files"]
    require(receipt["files"] == {name: item for name, item in accepted_files.items()
                                if Path(name).parts[0] in receipt["reused_prefixes"]},
            "Reused components differ from the accepted complete file manifest")
    for key in ("contract_sha256", "data_manifest_sha256", "execution_overlay_sha256", "execution_recipe"):
        require(previous_acceptance["identity"][key] == identity(project)[key], "Epoch repair changed scientific inputs or recipe")
    require(source_info(project, "two_stage")["step"] == 800
            and source_info(project, "mixed")["step"] == 300,
            "Epoch-repair smoke must test OI CP800 and retain the accepted Mixed CP300 diagnostic")
    old_manifest = inputs / "source_runtime_manifest.json"
    require(sha256(old_manifest) == previous_acceptance["identity"]["runtime_manifest_sha256"],
            "Epoch repair lacks the accepted original runtime")
    previous = load_json(old_manifest)["files"]
    current = load_json(project / RUNTIME)["files"]
    require(set(previous) == set(current), "Epoch repair changed runtime inventory")
    allowed = {
        "src/new_interest/training/old_new_execution.py",
        "src/new_interest/experiments/old_new_interest_g4.py",
    }
    for name, digest in previous.items():
        if current[name] == digest:
            continue
        require(name in allowed, "Epoch repair changed protected runtime code")
        old_path = inputs / "source_runtime" / name
        require(sha256(old_path) == digest, "Original epoch-repair source changed")

        def protected(path: Path) -> str:
            tree = ast.parse(path.read_text())
            if name.endswith("/old_new_execution.py"):
                wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "wrap_trainer")
                trainer = next(node for node in wrapper.body if isinstance(node, ast.ClassDef) and node.name == "StageCursorTrainer")
                methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_run_epoch"]
                require(len(methods) == 1, "Missing unique epoch-forwarding seam")
                trainer.body.remove(methods[0])
            else:
                permitted = {"validate_epoch_repair_runtime", "audit_epoch_repair_seed", "seed_epoch_repair", "worker", "readback"}
                tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name in permitted)]
            return ast.dump(tree, include_attributes=False)

        require(protected(old_path) == protected(project / name), "Epoch repair altered sampling, loss, cursor guards or evaluation")


def audit_epoch_repair_seed(project: Path, root: Path, *, probe: bool) -> bool:
    accepted_path = project / "migration_inputs/epoch_repair/accepted_components.json"
    if not accepted_path.exists():
        return False
    require(probe, "Epoch-repair diagnostic artifacts cannot initialize Full")
    accepted = load_json(accepted_path)
    validate_epoch_repair_runtime(project, accepted)
    actual = load_json(root / "epoch_repair_seed_receipt.json")
    require(actual["status"] == "passed" and actual["source_backend"] == accepted["source_backend"]
            and actual["component_acceptance_sha256"] == sha256(accepted_path), "Epoch-repair seed provenance changed")
    require(sha256(root / "epoch_repair_acceptance.json") == sha256(accepted_path), "Published epoch-repair acceptance changed")
    require(len(accepted["files"]) == accepted["file_count"]
            and sum(item["bytes"] for item in accepted["files"].values()) == accepted["bytes"], "Epoch-repair file budget changed")
    for name, item in accepted["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] in accepted["reused_prefixes"], "Unsafe epoch-repair component path")
        target = root / relative
        require(target.is_file() and not target.is_symlink() and target.stat().st_size == item["bytes"]
                and sha256(target) == item["sha256"], "Reused epoch-repair artifact changed")
    return True


def seed_epoch_repair(project: Path, root: Path, *, probe: bool) -> bool:
    accepted_path = project / "migration_inputs/epoch_repair/accepted_components.json"
    if not accepted_path.exists():
        return False
    require(probe, "Epoch-repair diagnostic artifacts cannot initialize Full")
    accepted = load_json(accepted_path)
    validate_epoch_repair_runtime(project, accepted)
    if (root / "epoch_repair_seed_receipt.json").exists():
        return audit_epoch_repair_seed(project, root, probe=probe)
    prefix = Path(accepted["source_blob_prefix"])
    require(not prefix.is_absolute() and ".." not in prefix.parts
            and prefix.as_posix().startswith("projects/sft4safety/amlt-results/"), "Unsafe epoch-repair source prefix")
    candidates = [Path(os.environ.get("AMLT_DEFAULT", "/mnt/default")) / prefix, root.parent / prefix.name]
    source = next((path for path in candidates if (path / "mixed/sft/run_summary.json").is_file()), None)
    require(source is not None and source.resolve() != root.resolve(), "Accepted epoch-repair components are not mounted")

    def copy(item: tuple[str, dict[str, Any]]) -> None:
        name, expected = item
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] in accepted["reused_prefixes"], "Unsafe epoch-repair copy path")
        origin, target = source / relative, root / relative
        require(origin.is_file() and not origin.is_symlink() and origin.stat().st_size == expected["bytes"]
                and sha256(origin) == expected["sha256"], "Epoch-repair source bytes changed")
        if target.exists():
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == expected["bytes"]
                    and sha256(target) == expected["sha256"], "Existing epoch-repair artifact changed")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".epoch-repair.partial")
        shutil.copyfile(origin, temporary)
        require(temporary.stat().st_size == expected["bytes"] and sha256(temporary) == expected["sha256"], "Epoch-repair copy differs")
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy, accepted["files"].items()))
    shutil.copyfile(accepted_path, root / "epoch_repair_acceptance.json")
    write_json(root / "epoch_repair_seed_receipt.json", {
        "status": "passed", "source_backend": accepted["source_backend"], "source_root": str(source),
        "component_acceptance_sha256": sha256(accepted_path), "files": accepted["file_count"],
        "bytes": accepted["bytes"], "diagnostic_optimizer_updates_preserved": 24,
    })
    return audit_epoch_repair_seed(project, root, probe=probe)


def publish(project: Path, root: Path, methods: list[str], probe: bool) -> None:
    verify_runtime(project)
    for relative, expected in load_json(project / RUNTIME)["files"].items():
        target = root / "runtime_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
        require(sha256(target) == expected, "Published G4 source differs")
    for source, target in ((RUNTIME, "runtime_source_manifest.json"), (OVERLAY, "execution_overlay.json"),
                           (INPUTS, "migration_inputs_manifest.json")):
        shutil.copyfile(project / source, root / target)
    files = {str(path.relative_to(root)): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in original._artifact_paths(root) if path.name != "worker_acceptance.json"}
    manifest = {"identity": identity(project), "phase": "migration_smoke" if probe else "full", "methods": methods, "files": files}
    write_json(root / "artifact_manifest.json", manifest)
    write_json(root / "success.json", {"identity": identity(project), "phase": manifest["phase"], "methods": methods,
               "artifact_manifest_sha256": sha256(root / "artifact_manifest.json")})
    archive_path = root / "readback_artifacts.tar.gz.partial"
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for name in [*sorted(files), "artifact_manifest.json", "success.json"]:
            archive.add(root / name, arcname=name, recursive=False)
    archive_path.replace(root / "readback_artifacts.tar.gz")
    archive_path = root / "readback_artifacts.tar.gz"
    write_json(root / "readback_package.json", {"sha256": sha256(archive_path), "bytes": archive_path.stat().st_size})


def readback(project: Path, root: Path, output: Path) -> dict[str, Any]:
    verify_runtime(project)
    marker, manifest = load_json(root / "success.json"), load_json(root / "artifact_manifest.json")
    require(marker["identity"] == manifest["identity"] == identity(project), "G4 result identity differs")
    require(marker["artifact_manifest_sha256"] == sha256(root / "artifact_manifest.json"), "G4 artifact manifest changed")
    for name, item in manifest["files"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe result path")
        target = root / path
        require(target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"], f"G4 artifact differs: {name}")
    require(marker["phase"] == manifest["phase"] and marker["methods"] == manifest["methods"], "G4 method/phase differs")
    probe = marker["phase"] == "migration_smoke"
    require(marker["phase"] in {"migration_smoke", "full"}, "Unknown G4 phase")
    audit_recovery_seed(project, root, probe=probe)
    audit_epoch_repair_seed(project, root, probe=probe)
    require(marker["methods"] == list(original.METHODS) if probe else
            len(marker["methods"]) == 1 and marker["methods"][0] in original.METHODS, "G4 method coverage differs")
    for source, target in ((RUNTIME, "runtime_source_manifest.json"), (OVERLAY, "execution_overlay.json"),
                           (INPUTS, "migration_inputs_manifest.json")):
        require(sha256(project / source) == sha256(root / target), "Published execution contract differs")
    for name, digest in load_json(project / RUNTIME)["files"].items():
        require(sha256(root / "runtime_sources" / name) == digest, "Published source bytes differ")
    accepted = load_json(root / "original_smoke_acceptance.json")
    require(accepted["status"] == "passed" and accepted["phase"] == "smoke"
            and accepted["identity"] == original.identity(project), "Original smoke acceptance changed")
    require(load_json(root / "snapshot_audit.json")["files"] == accepted["snapshot_files"], "Model snapshot changed")
    require(load_json(root / "input_audit.json") == original.audit_inputs(project), "Data audit differs")
    audit_environment(project, root)
    if not probe:
        migration_acceptance = load_json(root / "migration_smoke_acceptance.json")
        require(migration_acceptance["status"] == "passed" and migration_acceptance["phase"] == "migration_smoke"
                and migration_acceptance["identity"]["execution_recipe"] == execution_recipe(project)
                and migration_acceptance["identity"]["runtime_manifest_sha256"] == identity(project)["runtime_manifest_sha256"],
                "Full continuation lacks an accepted matching G4 smoke")
    reports = {}
    for method in marker["methods"]:
        stage_reports = {spec["key"]: audit_stage(project, root / method, spec) for spec in specs(project, root, method, probe=probe)}
        require(stage_reports[next(iter(stage_reports))]["initial_state_sha256"] == accepted["cold_initial_state_sha256"], "Original cold state changed")
        report = {"stages": stage_reports, "evaluation": shared.audit_evaluation(project, "2b", root / method, "smoke" if probe else "full", logical_shards=4)}
        if probe:
            report["tail_stages"] = {spec["key"]: audit_stage(project, root / ("tail_" + method), spec) for spec in tail_specs(project, method)}
            require(all(item["cursor_journal"]["verified_epoch_tails"] >= 3 for item in report["tail_stages"].values()), "Real Qwen epoch-boundary coverage is incomplete")
        else:
            totals = {task: sum(row["exposure_by_task"].get(task, 0) for row in stage_reports.values())
                      for task in ("old_interest", "new_interest")}
            require(totals == {"old_interest": 51294, "new_interest": 64800}, "Formal task exposures changed")
            report["observed_exposure"] = totals
        reports[method] = report
    result = {"version": "old-new-g4-independent-readback-v1", "status": "passed", "phase": marker["phase"],
              "identity": identity(project), "methods": reports, "artifact_manifest_sha256": sha256(root / "artifact_manifest.json"),
              "artifact_files_verified": len(manifest["files"]), "artifact_bytes_verified": sum(item["bytes"] for item in manifest["files"].values())}
    write_json(output, result)
    return result


def worker(project: Path, root: Path, method: str, *, probe: bool) -> None:
    verify_runtime(project)
    methods = list(original.METHODS) if probe else [method]
    require(all(value in original.METHODS for value in methods), "Unknown method")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "success.json").exists() and (root / "readback_package.json").exists():
        receipt = readback(project, root, Path("/scratch") / f"old-new-g4-{method}-existing-readback.json")
        require(receipt["phase"] == ("migration_smoke" if probe else "full") and set(receipt["methods"]) == set(methods),
                "Existing output belongs to another phase or method")
        require(sha256(root / "readback_artifacts.tar.gz") == load_json(root / "readback_package.json")["sha256"],
                "Existing readback package changed")
        print("OLD_NEW_G4_REUSED_VERIFIED_OUTPUT", method, flush=True)
        return
    accepted_path = project / "manifests/old_new_smoke_acceptance.json"
    accepted = load_json(accepted_path)
    require(accepted["status"] == "passed" and accepted["identity"] == original.identity(project), "Original smoke is not independently accepted")
    shutil.copyfile(accepted_path, root / "original_smoke_acceptance.json")
    if not probe:
        smoke = project / "manifests/old_new_g4_smoke_acceptance.json"
        receipt = load_json(smoke)
        require(receipt["status"] == "passed" and receipt["phase"] == "migration_smoke"
                and receipt["identity"]["execution_recipe"] == execution_recipe(project)
                and receipt["identity"]["runtime_manifest_sha256"] == identity(project)["runtime_manifest_sha256"],
                "Full continuation requires a downloaded matching four-GPU smoke")
        shutil.copyfile(smoke, root / "migration_smoke_acceptance.json")
    write_json(root / "input_audit.json", original.audit_inputs(project))
    environment_audit(project, root / "environment.json")
    audit_environment(project, root)
    shared.prepare_snapshot(project, "2b", root / "snapshot_audit.json")
    require(load_json(root / "snapshot_audit.json")["files"] == accepted["snapshot_files"], "Migration model snapshot changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    recovered_two_stage = seed_recovery(project, root, probe=probe)
    epoch_repair_components = seed_epoch_repair(project, root, probe=probe)
    for current in methods:
        if epoch_repair_components and current == "mixed":
            print("OLD_NEW_G4_REUSED_VERIFIED_METHOD", current, "diagnostic_updates", 12, flush=True)
            continue
        if recovered_two_stage and current == "two_stage":
            print("OLD_NEW_G4_REUSED_VERIFIED_METHOD", current, "diagnostic_updates", 18, flush=True)
            continue
        for spec in specs(project, root, current, probe=probe):
            if probe:
                initial = source_info(project, current)["step"]
                train_to(project, root / current, spec, stop=initial + 4)
                train_to(project, root / current, spec, stop=initial + 6)
            else:
                train_to(project, root / current, spec)
            finish_stage(project, root / current, spec)
        if probe and epoch_repair_components:
            print("OLD_NEW_G4_REUSED_VERIFIED_TAILS", current, "diagnostic_updates", 12, flush=True)
        elif probe:
            for spec in tail_specs(project, current):
                if spec["parent"] is None:
                    train_to(project, root / ("tail_" + current), spec, stop=2)
                train_to(project, root / ("tail_" + current), spec)
                finish_stage(project, root / ("tail_" + current), spec)
        evaluate(project, root / current, current, probe=probe, adapter_stage="old_sft" if probe and current == "two_stage" else "sft")
    publish(project, root, methods, probe)
    readback(project, root, root / "worker_acceptance.json")
    print("OLD_NEW_G4_WORKER_GATE_PASSED", "migration_smoke" if probe else "full", methods, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("worker", "readback", "preflight"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("two_stage", "mixed", "all"), default="all")
    parser.add_argument("--phase", choices=("migration_smoke", "full"), default="migration_smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project, root = args.project.resolve(), args.root.resolve()
    if args.action == "worker":
        worker(project, root, args.method, probe=args.phase == "migration_smoke")
    elif args.action == "readback":
        require(args.output is not None, "Readback requires --output")
        print(json.dumps(readback(project, root, args.output)))
    else:
        verify_runtime(project)
        print(json.dumps({"identity": identity(project), "data": original.audit_inputs(project)}))


if __name__ == "__main__":
    main()
