"""Four-A100 execution probes and acceptance for the frozen MIND Backbone task."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

from new_interest.experiments import backbone_scaling as legacy
from new_interest.training.execution_migration import (
    checkpoint_state_audit, file_sha256, validate_execution_transition,
)
from new_interest.training.global_sample_cursor import build_hf_cursor_contract, epoch_order, index_sha256

SOURCE_MANIFEST = Path("manifests/mind_four_gpu_runtime_sources_v1.json")
REQUEST = Path("manifests/four_gpu_migration_request.json")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_inputs(project: Path, size: str) -> dict:
    contract = legacy.audit_configs(project, size)
    for item in contract["input_files"]:
        if file_sha256(project / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    source = legacy.load_json(project / SOURCE_MANIFEST)
    for name, expected in source["files"].items():
        if file_sha256(project / name) != expected:
            raise ValueError(f"Migration runtime source changed: {name}")
    request = legacy.load_json(project / REQUEST)
    if request["model_size"] != size or request["target_world_size"] != 4:
        raise ValueError("Wrong model or execution request")
    checkpoint = project / request["source_checkpoint"]
    marker = legacy.load_json(checkpoint / "checkpoint_complete.json")
    if file_sha256(checkpoint / "checkpoint_complete.json") != request["source_checkpoint_marker_sha256"]:
        raise ValueError("Source checkpoint marker changed")
    audit = checkpoint_state_audit(checkpoint, marker["identity"])
    for relative, expected in request.get("stress_inputs", {}).items():
        if file_sha256(project / relative) != expected:
            raise ValueError("Longest real sample input changed")
    return {"contract": contract, "request": request, "source_marker": marker, "source_state": audit,
            "runtime_source_manifest_sha256": file_sha256(project / SOURCE_MANIFEST)}


def environment_audit(project: Path, output: Path) -> dict:
    import torch
    expected = legacy.load_json(project / legacy.CONTRACT)["worker_packages"]
    packages = {key: importlib.metadata.version(key) for key in expected}
    if packages != expected or torch.cuda.device_count() != 4:
        raise ValueError("Four-GPU worker package/device contract differs")
    devices = [{"index": index, "name": torch.cuda.get_device_properties(index).name,
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory} for index in range(4)]
    if any("A100" not in item["name"] or item["memory_bytes"] < 79 * 1024**3 for item in devices):
        raise ValueError("Expected four A100 80GB devices")
    result = {"packages": packages, "cuda": torch.version.cuda, "devices": devices,
              "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)}
    write_json(output, result)
    return result


def run(command: list[str], project: Path, log: Path, *, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("EXECUTION_COMMAND", json.dumps(command), flush=True)
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if any(word in line for word in ("EXECUTION_", "CHECKPOINT_", "loss", "%|", "Error", "Traceback")):
                print(line.strip()[:1600], flush=True)
        return process.wait()


def case_config(project: Path, size: str, case: Path, batch: int, checkpointing: bool, *, stop_steps: int = 4) -> dict:
    request = legacy.load_json(project / REQUEST)
    marker = legacy.load_json(project / request["source_checkpoint"] / "checkpoint_complete.json")
    config = legacy.load_json(project / legacy.config_paths(size)[0])
    scratch = Path(os.environ.get("TMPDIR", "/scratch")) / f"mind-g4-{size}-{case.parent.parent.name}-{case.name}"
    config.update({"train_file": str(project / legacy.TRAIN),
                   "semantic_id_token_mapping_file": str(project / legacy.MAPPING),
                   "output_dir": str(case / "sft"), "checkpoint_dir": str(scratch),
                   "persistent_checkpoint_dir": str(case / "checkpoints"), "resume_from_checkpoint": "auto",
                   "run_name": marker["identity"]["run_name"], "per_device_train_batch_size": batch,
                   "gradient_accumulation_steps": 8 // batch, "gradient_checkpointing_enabled": checkpointing,
                   "save_strategy": "steps", "save_steps": 100, "save_total_limit": 3,
                   "wandb_mode": "disabled", "logging_steps": 1,
                   "expected_initial_adapter_state_sha256": request["initial_adapter_state_sha256"],
                   "stop_after_global_step": int(marker["global_step"]) + stop_steps,
                   "long_sample_probe_files": [str(project / name) for name in request["stress_inputs"]],
                   "execution_migration": {"source_checkpoint": str(project / request["source_checkpoint"]),
                       "source_stage_artifacts": str(project / request["source_stage_artifacts"]),
                       "cursor_contract_file": str(project / request["cursor_contract_file"])}})
    if case.name == "torch-gc-b2" or case.name.startswith("fast-"):
        config["numerical_probe_file"] = str(project / "migration_stress/ni_longest32.parquet")
    return config


def compare_model_numerics(reference: Path, candidate: Path, thresholds: dict) -> dict:
    import torch
    from safetensors.torch import load_file
    records = []
    for rank in range(4):
        first = legacy.load_json(reference / f"numerical_probe_rank_{rank}.json")
        second = legacy.load_json(candidate / f"numerical_probe_rank_{rank}.json")
        if first["input_row"] != second["input_row"] or first["input_tokens"] != second["input_tokens"]:
            raise ValueError("Kernel numerical inputs differ")
        left, right = torch.tensor(first["logits"], dtype=torch.float64), torch.tensor(second["logits"], dtype=torch.float64)
        relative = float(torch.linalg.vector_norm(right-left) / torch.linalg.vector_norm(left).clamp_min(1e-12))
        loss_relative = abs(first["loss"]-second["loss"]) / max(abs(first["loss"]), 1e-12)
        records.append({"rank": rank, "logit_relative_rms": relative, "logit_max_abs": float((right-left).abs().max()),
                        "loss_relative_error": loss_relative})
    first = load_file(str(reference / "numerical_probe_gradients.safetensors"))
    second = load_file(str(candidate / "numerical_probe_gradients.safetensors"))
    if first.keys() != second.keys():
        raise ValueError("Kernel gradient parameter sets differ")
    squared, source_squared, target_squared, dot = 0.0, 0.0, 0.0, 0.0
    for key in first:
        left, right = first[key].double().flatten(), second[key].double().flatten()
        if left.shape != right.shape:
            raise ValueError("Kernel gradient shapes differ")
        squared += float((left-right).square().sum())
        source_squared += float(left.square().sum())
        target_squared += float(right.square().sum())
        dot += float((left*right).sum())
    relative_gradient = (squared / max(source_squared, 1e-24)) ** 0.5
    cosine = dot / max((source_squared*target_squared)**0.5, 1e-24)
    passed = (relative_gradient <= thresholds["relative_rms_max"] and cosine >= thresholds["gradient_cosine_min"]
              and all(row["logit_relative_rms"] <= thresholds["relative_rms_max"]
                      and row["loss_relative_error"] <= thresholds["relative_rms_max"] for row in records))
    return {"status": "passed" if passed else "failed", "ranks": records,
            "gradient_relative_rms": relative_gradient, "gradient_cosine": cosine,
            "thresholds": thresholds, "source_checkpoint_and_inputs_identical": True,
            "scope": "Real model source-state forward/backward with diagnostic dropout disabled; no optimizer updates."}


def audit_case(project: Path, size: str, case: Path, expected_stop: int) -> dict:
    summary = legacy.load_json(case / "sft/run_summary.json")
    config = legacy.load_json(case / "training_config.json")
    identity = summary["checkpoint_identity"]
    request = legacy.load_json(project / REQUEST)
    source = legacy.load_json(project / request["source_checkpoint"] / "checkpoint_complete.json")
    transition = validate_execution_transition(source["identity"], identity)
    state = legacy.load_json(case / "sft/trainer_state.json")
    if state["global_step"] != expected_stop or state["max_steps"] != 2025:
        raise ValueError("Probe changed the global step or full scheduler horizon")
    if summary["world_size"] != 4 or summary["effective_global_batch_size"] != 32:
        raise ValueError("Probe did not execute four GPUs at global batch 32")
    start = legacy.load_json(case / "sft/execution_migration_start.json")
    if start["status"] != "passed" or len(start["rank_restores"]) != 4 or start["global_step"] != source["global_step"]:
        raise ValueError("Four-rank actual checkpoint restore was not verified")
    attempts = sorted((case / "sft/execution_attempts").glob("attempt-*"))
    if not attempts:
        raise ValueError("Missing actual execution attempt evidence")
    for attempt in attempts:
        for rank in range(4):
            rng = legacy.load_json(attempt / f"rng_rank_{rank}.json")
            if rng["status"] != "passed" or rng["actual_rng_sha256"] != rng["expected_rng_sha256"]:
                raise ValueError("Actual RNG restore differs")
    checkpoint = case / "checkpoints" / f"checkpoint-{expected_stop}"
    current_audit = checkpoint_state_audit(checkpoint, identity)
    cursor = legacy.load_json(checkpoint / "training_sample_cursor.json")
    if cursor["global_step"] != expected_stop or cursor["consumed_global_samples"] != expected_stop * 32 or not cursor["all_rank_real_sample_order_exact"]:
        raise ValueError("Actual global sample cursor differs")
    stress = legacy.load_json(case / "sft/long_sample_stress.json")
    if stress["status"] != "passed" or stress["optimizer_updates"] != 0:
        raise ValueError("Longest real sample stress failed")
    adapter = legacy.audit_adapter(project, size, case / "sft")
    reload_audit = legacy.audit_reload(project, size, case / "sft")
    by_step = {}
    for attempt in attempts:
        for line in (attempt / "step_metrics.jsonl").read_text().splitlines():
            if line:
                item = json.loads(line)
                by_step[item["global_step"]] = item
    metrics = [by_step[step] for step in sorted(by_step)]
    timed = metrics[1:]
    if len(timed) < 2:
        raise ValueError("Too few measured optimizer steps")
    seconds = statistics.median(item["seconds"] for item in timed)
    peaks = [rank["peak_allocated_bytes"] for item in metrics for rank in item["ranks"]]
    peaks += [rank["peak_allocated_bytes"] for record in stress["records"] for rank in record["ranks"]]
    reserved = [rank["peak_reserved_bytes"] for item in metrics for rank in item["ranks"]]
    reserved += [rank["peak_reserved_bytes"] for record in stress["records"] for rank in record["ranks"]]
    eligible = max(peaks) <= 70 * 1024**3 and max(reserved) <= 73 * 1024**3
    return {"status": "passed", "eligible_memory_headroom": eligible, "model_size": size,
            "per_device_batch_size": config["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "gradient_checkpointing_enabled": config["gradient_checkpointing_enabled"],
            "median_optimizer_step_seconds": seconds, "samples_per_second": 32 / seconds,
            "peak_allocated_bytes": max(peaks), "peak_reserved_bytes": max(reserved),
            "source_global_step": source["global_step"], "completed_global_step": expected_stop,
            "actual_four_rank_restore": True, "actual_sample_cursor_exact": True,
            "adapter": adapter, "reload": reload_audit, "checkpoint_state": current_audit,
            "transition": transition, "case_name": case.name}


def run_case(project: Path, size: str, case: Path, batch: int, checkpointing: bool) -> dict:
    config = case_config(project, size, case, batch, checkpointing)
    if (case / "case_result.json").is_file():
        existing = legacy.load_json(case / "case_result.json")
        if existing["status"] == "passed":
            return audit_case(project, size, case, int(config["stop_after_global_step"]))
        if existing.get("failure_class") == "capacity_oom":
            return existing
    write_json(case / "training_config.json", config)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4",
               "-m", "new_interest.training.train_sft", "--config", str(case / "training_config.json")]
    code = run(command, project, case / "logs/train.log")
    if code != 0:
        log_text = (case / "logs/train.log").read_text()
        capacity = "CUDA out of memory" in log_text or "EXECUTION_PROBE_CAPACITY_FAILED" in log_text
        result = {"status": "failed", "exit_code": code, "case_name": case.name,
                  "failure_class": "capacity_oom" if capacity else "execution_error",
                  "per_device_batch_size": batch, "gradient_checkpointing_enabled": checkpointing}
        write_json(case / "case_result.json", result)
        if not capacity:
            raise ValueError(f"Non-capacity migration probe failure: {case.name}")
        return result
    scratch_state = Path(config["checkpoint_dir"]) / "trainer_state.json"
    if scratch_state.is_file():
        shutil.copyfile(scratch_state, case / "sft/trainer_state.json")
    model = legacy.load_json(project / legacy.CONTRACT)["models"][size]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    reload_command = [sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
        "--model-name", model["model_name"], "--model-revision", model["model_revision"],
        "--adapter-path", str(case / "sft/final_adapter"), "--mapping-file", str(project / legacy.MAPPING),
        "--reference-file", str(case / "sft/semantic_reload_reference.json"),
        "--output", str(case / "sft/semantic_adapter_reload_audit.json"), "--atol", "0", "--rtol", "0"]
    if run(reload_command, project, case / "logs/reload.log", env=env) != 0:
        raise ValueError("Migrated adapter failed exact fresh reload")
    result = audit_case(project, size, case, int(config["stop_after_global_step"]))
    write_json(case / "case_result.json", result)
    print("EXECUTION_PROBE_CASE_PASSED", json.dumps(result), flush=True)
    return result


def package_probe(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "probe_artifact_manifest.json" or path.name.endswith(".partial") or path.name.startswith("readback_"):
            continue
        files[path.relative_to(root).as_posix()] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    write_json(root / "probe_artifact_manifest.json", {"files": files})
    with tarfile.open(root / "readback_artifacts.tar.gz.partial", "w:gz", compresslevel=1) as archive:
        for relative in sorted([*files, "probe_artifact_manifest.json"]):
            archive.add(root / relative, arcname=relative, recursive=False)
    (root / "readback_artifacts.tar.gz.partial").replace(root / "readback_artifacts.tar.gz")
    path = root / "readback_artifacts.tar.gz"
    write_json(root / "readback_package.json", {"sha256": file_sha256(path), "bytes": path.stat().st_size})


def probe_worker(project: Path, size: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs(project, size)
    write_json(root / "migration_input_audit.json", inputs)
    environment_audit(project, root / "environment.json")
    legacy.prepare_snapshot(project, size, root / "snapshot_audit.json")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    results = []
    for batch in (2, 4, 8):
        result = run_case(project, size, root / "cases" / f"torch-gc-b{batch}", batch, True)
        results.append(result)
    eligible = [result for result in results if result["status"] == "passed" and result["eligible_memory_headroom"]]
    if not eligible:
        raise ValueError("No approved microbatch passed the full migration and memory gates")
    selected = min(eligible, key=lambda result: result["median_optimizer_step_seconds"])
    # A separate no-recompute measurement is attempted only with substantial
    # measured activation headroom; it cannot replace a failed candidate gate.
    if selected["peak_allocated_bytes"] < 35 * 1024**3:
        result = run_case(project, size, root / "cases" / f"torch-nogc-b{selected['per_device_batch_size']}", selected["per_device_batch_size"], False)
        results.append(result)
        if result["status"] == "passed" and result["eligible_memory_headroom"] and result["median_optimizer_step_seconds"] < selected["median_optimizer_step_seconds"]:
            selected = result
    kernel_profile = "torch-fallback"
    fast = inputs["request"].get("fast_kernels")
    if fast:
        acceptance = legacy.load_json(project / fast["component_acceptance"])
        if not acceptance["base_runtime_preserved"] or not acceptance["fast_bindings"]["is_fast_path_available"]:
            raise ValueError("Fast component acceptance was not established")
        command = [sys.executable, str(project / fast["installer"]), "--manifest", str(project / fast["install_manifest"]),
                   "--wheel-cache", str(Path("/scratch") / f"mind-fast-kernel-wheels-{size}"),
                   "--output", str(root / "fast_kernel_environment.json")]
        if run(command, project, root / "logs/install-fast-kernels.log") != 0:
            raise ValueError("Pinned fast-kernel installation failed")
        case = root / "cases" / f"fast-{'gc' if selected['gradient_checkpointing_enabled'] else 'nogc'}-b{selected['per_device_batch_size']}"
        result = run_case(project, size, case, selected["per_device_batch_size"], selected["gradient_checkpointing_enabled"])
        results.append(result)
        if result["status"] == "passed":
            comparison = compare_model_numerics(root / "cases/torch-gc-b2/sft", case / "sft", fast["numeric_thresholds"])
            write_json(root / "whole_model_kernel_comparison.json", comparison)
            if comparison["status"] == "passed" and result["eligible_memory_headroom"] and result["median_optimizer_step_seconds"] < selected["median_optimizer_step_seconds"]:
                selected, kernel_profile = result, "fla-0.5.2-causal-conv1d-1.7.0"
        if kernel_profile == "torch-fallback":
            uninstall = [sys.executable, "-m", "pip", "uninstall", "-y", "flash-linear-attention", "fla-core", "causal-conv1d"]
            if run(uninstall, project, root / "logs/remove-rejected-fast-profile.log") != 0:
                raise ValueError("Could not restore the selected reference kernel profile")
            check = [sys.executable, "-c", "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; assert not m.is_fast_path_available; print('EXECUTION_REFERENCE_KERNEL_RESTORED')"]
            if run(check, project, root / "logs/reference-kernel-restored.log") != 0:
                raise ValueError("Reference kernel restoration could not be verified")
    evaluation = root / "real_eval/merged"
    command = [sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking",
               "--config", str(project / legacy.config_paths(size)[1]),
               "--adapter-path", str(root / "cases" / selected["case_name"] / "sft/final_adapter"),
               "--output-dir", str(evaluation), "--num-shards", "1", "--shard-index", "0",
               "--max-examples", "64", "--batch-size", "8"]
    if run(command, project, root / "logs/eval64.log", env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}) != 0:
        raise ValueError("Selected migrated adapter failed real Dev smoke")
    evaluation_audit = legacy.audit_evaluation(project, size, root, "smoke")
    write_json(root / "evaluation_readback.json", evaluation_audit)
    write_json(root / "probe_summary.json", {"status": "worker_gates_passed_pending_independent_readback", "model_size": size,
               "cases": results, "selected": selected, "full_training_started": False,
               "kernel_profile": kernel_profile,
               "evaluation": evaluation_audit,
               "resource": "four A100 80GB", "source_state": inputs["source_state"]})
    package_probe(root)
    print("EXECUTION_PROBE_WORKER_PASSED", size, selected["case_name"], flush=True)


def verify_artifacts(root: Path, manifest_name: str) -> dict:
    manifest = legacy.load_json(root / manifest_name)
    if not manifest.get("files"):
        raise ValueError("Empty artifact manifest")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact path")
        path = root / relative
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Downloaded artifact differs: {name}")
    return {"files": len(manifest["files"]), "bytes": sum(item["bytes"] for item in manifest["files"].values()),
            "manifest_sha256": file_sha256(root / manifest_name)}


def readback_probe(project: Path, size: str, root: Path, output: Path) -> dict:
    inputs = verify_inputs(project, size)
    artifacts = verify_artifacts(root, "probe_artifact_manifest.json")
    reported = legacy.load_json(root / "probe_summary.json")
    results = []
    for item in reported["cases"]:
        if item["status"] != "passed":
            if item.get("failure_class") != "capacity_oom":
                raise ValueError("A technical failure was classified as a capacity result")
            continue
        results.append(audit_case(project, size, root / "cases" / item["case_name"], inputs["source_state"]["global_step"] + 4))
    reference = [item for item in results if item["case_name"].startswith("torch-") and item["eligible_memory_headroom"]]
    if not reference:
        raise ValueError("No admissible reference execution")
    selected = min(reference, key=lambda item: item["median_optimizer_step_seconds"])
    kernel_profile = "torch-fallback"
    for item in results:
        if not item["case_name"].startswith("fast-"):
            continue
        comparison = compare_model_numerics(root / "cases/torch-gc-b2/sft", root / "cases" / item["case_name"] / "sft",
                                            inputs["request"]["fast_kernels"]["numeric_thresholds"])
        if comparison["status"] == "passed" and item["eligible_memory_headroom"] and item["median_optimizer_step_seconds"] < selected["median_optimizer_step_seconds"]:
            selected, kernel_profile = item, "fla-0.5.2-causal-conv1d-1.7.0"
    if selected["case_name"] != reported["selected"]["case_name"] or kernel_profile != reported["kernel_profile"]:
        raise ValueError("Independent execution selection differs")
    evaluation = legacy.audit_evaluation(project, size, root, "smoke")
    result = {"version": "mind-four-gpu-probe-readback-v1", "status": "passed", "model_size": size,
              "artifacts": artifacts, "selected": selected, "kernel_profile": kernel_profile,
              "source_state": inputs["source_state"], "evaluation": evaluation,
              "frozen_bundle_runtime_sha256": inputs["runtime_source_manifest_sha256"],
              "full_accepted": False}
    write_json(output, result)
    return result


def audit_full_training(project: Path, size: str, root: Path, recipe: dict) -> dict:
    inputs = verify_inputs(project, size)
    folder = root / "sft"
    summary = legacy.load_json(folder / "run_summary.json")
    state = legacy.load_json(folder / "trainer_state.json")
    config = legacy.load_json(folder / "resolved_config.json")
    if (state["global_step"], state["max_steps"], state["num_train_epochs"], state["epoch"]) != (2025, 2025, 3, 3.0):
        raise ValueError("Full did not complete exactly 2025 updates / three epochs")
    target = summary["checkpoint_identity"]
    validate_execution_transition(inputs["source_marker"]["identity"], target)
    expected_summary = {"train_samples": 21600, "configured_epochs": 3,
                        "expected_optimizer_steps": 2025, "actual_optimizer_steps": 2025,
                        "seed": 42, "enable_thinking": False, "max_seq_length": 2048,
                        "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary"}
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"Full scientific summary differs: {key}")
    expected_recipe = {**inputs["contract"]["recipe"], "per_device_train_batch_size": recipe["per_device_batch_size"],
                       "gradient_accumulation_steps": recipe["gradient_accumulation_steps"]}
    legacy.audit_resolved_recipe(config, target, expected_recipe)
    model = inputs["contract"]["models"][size]
    for key in ("model_name", "model_revision"):
        if summary.get(key) != model[key] or config.get(key) != model[key]:
            raise ValueError("Full model/revision differs")
    token_audit = summary["token_audit"]
    if token_audit["samples"] != 21600 or token_audit["over_max_seq_length"] != 0:
        raise ValueError("Full token coverage differs")
    if not math.isfinite(summary["train_metrics"]["train_loss"]):
        raise ValueError("Full aggregate loss is nonfinite")
    semantic = summary["model_audit"]["semantic_id_tokens"]
    if semantic["count"] != 108 or semantic["save_embedding_layers"] is not False or semantic["tie_word_embeddings"] is not True:
        raise ValueError("Full native tied-SID structure changed")
    if summary["model_audit"]["forbidden_trainable_parameters"]:
        raise ValueError("Forbidden parameters were trained")
    if summary["semantic_token_initialization"]["initialization_version"] != "label-token-mean-norm-matched-cpu-fp32-v2":
        raise ValueError("SID initialization changed")
    if summary["world_size"] != 4 or summary["effective_global_batch_size"] != 32:
        raise ValueError("Full execution size differs")
    for field, key in (("per_device_train_batch_size", "per_device_batch_size"),
                       ("gradient_accumulation_steps", "gradient_accumulation_steps"),
                       ("gradient_checkpointing_enabled", "gradient_checkpointing_enabled")):
        if target[field] != recipe[key] or config[field] != recipe[key]:
            raise ValueError(f"Full did not use the independently selected execution: {field}")
    initial = legacy.load_json(folder / "initial_adapter_state.json")
    first = legacy.load_json(folder / "training_start_audit.json")
    if initial["state_sha256"] != inputs["request"]["initial_adapter_state_sha256"] or not first["fresh_optimization_state"]:
        raise ValueError("Historical first stage / initial parameters differ")
    if initial["checkpoint_identity"] != first["checkpoint_identity"] or first["checkpoint_identity"] != inputs["source_marker"]["identity"]:
        raise ValueError("Source stage evidence differs")
    migration = legacy.load_json(folder / "execution_migration_start.json")
    if migration["global_step"] != inputs["source_state"]["global_step"] or migration["status"] != "passed" or len(migration["rank_restores"]) != 4:
        raise ValueError("Original four-rank migration was not validated")
    successful_attempts, interrupted_attempts = [], []
    selected = legacy.load_json(root / "selected_execution_recipe.json")
    expect_fast = selected["kernel_profile"] != "torch-fallback"
    for attempt in (folder / "execution_attempts").glob("attempt-*"):
        metrics_path = attempt / "step_metrics.jsonl"
        if not metrics_path.is_file():
            interrupted_attempts.append(attempt.name)
            continue
        timings = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if not timings:
            interrupted_attempts.append(attempt.name)
            continue
        for rank in range(4):
            rng = legacy.load_json(attempt / f"rng_rank_{rank}.json")
            if rng["status"] != "passed" or rng["expected_rng_sha256"] != rng["actual_rng_sha256"]:
                raise ValueError("Full contains an unverified RNG restore")
            kernels = legacy.load_json(attempt / f"kernels_rank_{rank}.json")
            if kernels["is_fast_path_available"] != expect_fast:
                raise ValueError("Full kernel binding differs from selected execution")
        successful_attempts.append({"attempt": attempt.name, "last_step": max(item["global_step"] for item in timings)})
    if not successful_attempts or max(item["last_step"] for item in successful_attempts) != 2025:
        raise ValueError("No verified execution attempt completed step 2025")
    cursor = legacy.load_json(folder / "training_sample_cursor.json")
    if cursor["global_step"] != 2025 or cursor["consumed_global_samples"] != 64800 or not cursor["all_rank_real_sample_order_exact"]:
        raise ValueError("Full global sample cursor is incomplete")
    expected_order = [index for epoch in range(3) for index in epoch_order(21600, 42, epoch)]
    if cursor["consumed_prefix_sha256"] != index_sha256(expected_order):
        raise ValueError("Full consumed sample sequence differs from original seed/epochs")
    if summary["train_file_sha256"] != inputs["source_marker"]["identity"]["train_file_sha256"]:
        raise ValueError("Full training data differs")
    if not summary["semantic_token_drift"]["all_input_rows_changed"] or not summary["semantic_token_drift"]["all_output_rows_changed"]:
        raise ValueError("Some SID rows did not learn")
    adapter, reload_result = legacy.audit_adapter(project, size, folder), legacy.audit_reload(project, size, folder)
    return {"samples": 21600, "optimizer_steps": 2025, "epochs": 3, "global_batch": 32,
            "initial_adapter_state_sha256": initial["state_sha256"], "adapter": adapter, "reload": reload_result,
            "execution_migration": migration, "sample_cursor": cursor,
            "successful_attempts": successful_attempts, "interrupted_attempts_without_committed_step_metrics": interrupted_attempts}


def package_full(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or "checkpoints-full" in relative.parts or path.name in {"artifact_manifest.json"} or path.name.endswith(".partial") or path.name.startswith("readback_"):
            continue
        files[str(relative)] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    write_json(root / "artifact_manifest.json", {"files": files})
    with tarfile.open(root / "readback_artifacts.tar.gz.partial", "w:gz", compresslevel=1) as archive:
        for relative in sorted([*files, "artifact_manifest.json"]):
            archive.add(root / relative, arcname=relative, recursive=False)
    (root / "readback_artifacts.tar.gz.partial").replace(root / "readback_artifacts.tar.gz")
    archive = root / "readback_artifacts.tar.gz"
    write_json(root / "readback_package.json", {"sha256": file_sha256(archive), "bytes": archive.stat().st_size})


def full_worker(project: Path, size: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs(project, size)
    accepted = legacy.load_json(project / "manifests/selected_execution_recipe.json")
    if accepted["status"] != "passed" or accepted["model_size"] != size:
        raise ValueError("Full lacks independently accepted model-specific execution selection")
    recipe = accepted["selected"]
    write_json(root / "selected_execution_recipe.json", accepted)
    write_json(root / "migration_input_audit.json", inputs)
    environment_audit(project, root / "environment.json")
    legacy.prepare_snapshot(project, size, root / "snapshot_audit.json")
    if accepted["kernel_profile"] != "torch-fallback":
        fast = inputs["request"]["fast_kernels"]
        command = [sys.executable, str(project / fast["installer"]), "--manifest", str(project / fast["install_manifest"]),
                   "--wheel-cache", str(Path("/scratch") / f"mind-fast-kernel-wheels-{size}"), "--output", str(root / "fast_kernel_environment.json")]
        if run(command, project, root / "logs/install-fast.log"):
            raise ValueError("Full fast-kernel environment could not be verified")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    config = case_config(project, size, root, int(recipe["per_device_batch_size"]), recipe["gradient_checkpointing_enabled"])
    for key in ("stop_after_global_step", "long_sample_probe_files", "numerical_probe_file"):
        config.pop(key, None)
    config.update({"output_dir": str(root / "sft"), "checkpoint_dir": str(Path("/scratch") / f"mind-g4-full-{size}-{root.name}"),
                   "persistent_checkpoint_dir": str(root / "checkpoints-full"), "logging_steps": 5})
    write_json(root / "training_config.json", config)
    if not (root / "training_gate.json").exists():
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4", "-m", "new_interest.training.train_sft",
                   "--config", str(root / "training_config.json")]
        if run(command, project, root / "logs/train-full.log"):
            raise ValueError("Four-GPU Full training failed")
        shutil.copyfile(Path(config["checkpoint_dir"]) / "trainer_state.json", root / "sft/trainer_state.json")
        model = inputs["contract"]["models"][size]
        command = [sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
                   "--model-name", model["model_name"], "--model-revision", model["model_revision"],
                   "--adapter-path", str(root / "sft/final_adapter"), "--mapping-file", str(project / legacy.MAPPING),
                   "--reference-file", str(root / "sft/semantic_reload_reference.json"),
                   "--output", str(root / "sft/semantic_adapter_reload_audit.json"), "--atol", "0", "--rtol", "0"]
        if run(command, project, root / "logs/reload-full.log", env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}):
            raise ValueError("Full adapter failed exact fresh reload")
    training = audit_full_training(project, size, root, recipe)
    write_json(root / "training_gate.json", training)
    def evaluate(shard: int):
        command = [sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking",
                   "--config", str(project / legacy.config_paths(size)[1]), "--adapter-path", str(root / "sft/final_adapter"),
                   "--output-dir", str(root / f"real_eval/shard-{shard:02d}"), "--num-shards", "4", "--shard-index", str(shard), "--batch-size", "8"]
        return run(command, project, root / f"logs/eval-{shard}.log", env={**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)})
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(evaluate, range(4)))
    if any(outcomes):
        raise ValueError("Full Dev shard failed")
    from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
    merge_id_token_shards([root / f"real_eval/shard-{shard:02d}" for shard in range(4)], project / legacy.DEV, project / legacy.MAPPING,
                         root / "real_eval/merged", expected_records=31392)
    evaluation = legacy.audit_evaluation(project, size, root, "full")
    write_json(root / "evaluation_readback.json", evaluation)
    write_json(root / "full_success.json", {"status": "worker_gates_passed_pending_independent_readback", "model_size": size,
               "training": training, "evaluation": evaluation})
    package_full(root)
    print("BACKBONE_FOUR_GPU_FULL_WORKER_PASSED", size, flush=True)


def readback_full(project: Path, size: str, root: Path, output: Path) -> dict:
    artifacts = verify_artifacts(root, "artifact_manifest.json")
    accepted = legacy.load_json(project / "manifests/selected_execution_recipe.json")
    if accepted != legacy.load_json(root / "selected_execution_recipe.json"):
        raise ValueError("Full execution recipe differs from frozen acceptance")
    training = audit_full_training(project, size, root, accepted["selected"])
    evaluation = legacy.audit_evaluation(project, size, root, "full")
    result = {"version": "mind-four-gpu-full-readback-v1", "status": "passed", "model_size": size,
              "identity": {"source_contract_sha256": file_sha256(project / legacy.CONTRACT),
                           "runtime_sources_sha256": file_sha256(project / SOURCE_MANIFEST)},
              "training": training, "evaluation": evaluation, "artifacts": artifacts,
              "execution_migration": training["execution_migration"], "selected_execution_recipe": accepted}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "full", "readback-probe", "readback-full"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-size", choices=("2b", "4b"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "probe":
        probe_worker(args.project.resolve(), args.model_size, args.root.resolve())
    elif args.action == "full":
        full_worker(args.project.resolve(), args.model_size, args.root.resolve())
    else:
        if args.output is None:
            parser.error("Independent readback requires --output")
        reader = readback_probe if args.action == "readback-probe" else readback_full
        print(json.dumps(reader(args.project.resolve(), args.model_size, args.root.resolve(), args.output), indent=2))


if __name__ == "__main__":
    main()
