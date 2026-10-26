"""Frozen, single-GPU MIND Backbone runs and independent artifact acceptance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import record_input_sha256
from new_interest.training.semantic_id_tokens import write_json

CONTRACT = Path("configs/experiments/mind_backbone_qwen35_21600_v1.json")
SOURCE_MANIFEST = Path("manifests/mind_backbone_runtime_sources_v1.json")
DATA = Path("data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1")
TRAIN = DATA / "train/behavior_interest/21600.parquet"
DEV = DATA / "eval/behavior_interest/dev_full_grouped.parquet"
MAPPING = DATA / "interest_token_mapping.parquet"
CANDIDATE_MAPPING = MAPPING


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def config_paths(size: str) -> tuple[Path, Path]:
    require(size in {"2b", "4b"}, "Only the approved 2B and 4B models are supported")
    return (Path(f"configs/sft/experiments/mind_backbone_qwen35_{size}_21600_v1.json"),
            Path(f"configs/id_token/mind_backbone_qwen35_{size}_21600_v1.json"))


def identity(project: Path, size: str) -> dict[str, Any]:
    c = load_json(project / CONTRACT)
    sft, evaluation = config_paths(size)
    return {"contract_version": c["version"], "contract_sha256": sha256(project / CONTRACT),
            "model_size": size, "model_name": c["models"][size]["model_name"],
            "model_revision": c["models"][size]["model_revision"],
            "sft_config_sha256": sha256(project / sft),
            "eval_config_sha256": sha256(project / evaluation),
            "runtime_sources_sha256": sha256(project / SOURCE_MANIFEST)}


def audit_configs(project: Path, size: str) -> dict[str, Any]:
    c = load_json(project / CONTRACT)
    model = c["models"][size]
    sft_path, eval_path = config_paths(size)
    sft, evaluation = load_json(project / sft_path), load_json(project / eval_path)
    for key, expected in c["recipe"].items():
        require(sft.get(key) == expected, f"Frozen recipe mismatch: {key}")
    for config in (sft, evaluation):
        for key in ("model_name", "model_revision"):
            require(config.get(key) == model[key], f"Model identity mismatch: {key}")
        require(config.get("enable_thinking") is False, "Thinking must remain disabled")
    require(sft["train_file"] == str(TRAIN), "Wrong training view or size")
    require(sft["semantic_id_token_mapping_file"] == str(MAPPING), "Wrong SID mapping")
    require(evaluation["eval_file"] == str(DEV), "Wrong Dev input")
    require(evaluation["interest_token_mapping_file"] == str(MAPPING), "Wrong evaluation mapping")
    for key in ("sample_weight_file", "adapter_path", "input_adapter", "mera_relations_file"):
        require(sft.get(key) is None, f"Unapproved training input: {key}")
    for key in ("top_k", "cutoffs", "batch_size", "checkpoint_size"):
        require(evaluation.get(key) == c["evaluation"][key], f"Evaluation config mismatch: {key}")
    require(c["gpus_per_model"] == 1 and c["max_concurrent_gpus"] == 2, "GPU limit changed")
    require(1 * sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"] == 32,
            "Effective global batch must be 32")
    require(c["train_rows"] // 32 * sft["epochs"] == 2025, "Optimizer-step mismatch")
    return c


def audit_inputs(project: Path, size: str, output: Path) -> dict[str, Any]:
    c = audit_configs(project, size)
    for item in c["input_files"]:
        p = project / item["path"]
        require(p.is_file() and sha256(p) == item["sha256"], f"Frozen input differs: {item['path']}")
    source = load_json(project / SOURCE_MANIFEST)
    for relative, expected in source["files"].items():
        require(sha256(project / relative) == expected, f"Frozen source differs: {relative}")
    rows = pq.read_table(project / TRAIN).to_pylist()
    records = pq.read_table(project / DEV).to_pylist()
    mapping = pq.read_table(project / MAPPING).to_pylist()
    by_label = {str(r["target_subcategory"]): str(r["interest_token"]) for r in mapping}
    require(len(mapping) == len(by_label) == len(set(by_label.values())) == 108, "SID mapping is not bijective")
    counts = Counter(str(r["target_subcategory"]) for r in rows)
    require(len(rows) == 21600 and len(counts) == 108 and set(counts.values()) == {200}, "Class balance differs")
    require(len({str(r["source_sample_id"]) for r in rows}) == 21600, "Duplicate training IDs")
    users = {str(r["user_id"]) for r in rows}
    require(len(users) == 21600, "Training is not one row per user")
    require(not users.intersection(str(r["user_id"]) for r in records), "Train/Dev user leakage")
    require(len(records) == len({str(r["impression_id"]) for r in records}) == 31392, "Dev coverage differs")
    support = Counter(str(t) for r in records for t in r["acceptable_subcategories"])
    require(len(support) == 104, "Dev support differs")
    for r in rows:
        require(r["task_type"] == "behavior_interest", "Training input view differs")
        target = by_label[str(r["target_subcategory"])]
        require(r["response"] == target, "Training target differs from SID")
        require(len(r["messages"]) == 2 and r["messages"][0]["role"] == "user"
                and r["messages"][1] == {"role": "assistant", "content": target}, "Training message contract differs")
    manifest = pq.read_table(project / c["input_files"][-1]["path"], columns=["source_sample_id", "within_class_rank"]).to_pylist()
    selected = {str(r["source_sample_id"]) for r in manifest if int(r["within_class_rank"]) < 200}
    require(selected == {str(r["source_sample_id"]) for r in rows}, "Training sample selection differs")
    result = {"identity": identity(project, size), "train_rows": 21600, "train_classes": 108,
              "dev_impressions": 31392, "supported_dev_classes": 104, "target_occurrences": sum(support.values()),
              "train_dev_user_overlap": 0, "input_sha256": {i["path"]: i["sha256"] for i in c["input_files"]}}
    write_json(output, result)
    return result


def audit_gradient(audit: dict[str, Any], model: dict[str, Any]) -> None:
    require(audit["model_name"] == model["model_name"] and audit["model_revision"] == model["model_revision"], "Gradient model differs")
    require(audit["representative_samples"] == 108 and audit["batch_size"] == 1, "Gradient coverage differs")
    require(audit["all_token_rows_finite_and_nonzero"] is True, "Gradient gate failed")
    require(audit["semantic_initialization"]["tie_word_embeddings"] is True, "Native tied embeddings were changed")
    params = audit["gradient_parameters"]
    require(len(params) == model["expected_trainable_token_tensors"], "Wrong number of shared SID tensors")
    for p in params:
        require(p["shape"] == [108, model["hidden_size"]], "SID gradient shape differs")
        require(p["finite_rows"] == p["nonzero_rows"] == 108, "Missing finite nonzero SID gradient")


def audit_training(project: Path, size: str, root: Path, kind: str) -> dict[str, Any]:
    c = load_json(project / CONTRACT)
    model = c["models"][size]
    expected = {"single": (32, 5, 20, 8, 21600), "smoke": (108, 2, 8, 32, 108), "full": (21600, 3, 2025, 32, 21600)}[kind]
    n, epochs, steps, accumulation, token_count = expected
    summary = load_json(root / "run_summary.json")
    config = load_json(root / "resolved_config.json")
    for key in ("model_name", "model_revision"):
        require(summary.get(key) == model[key] and config.get(key) == model[key], f"Training {key} differs")
    for key, value in {"train_samples": n, "configured_epochs": epochs, "expected_optimizer_steps": steps,
                       "actual_optimizer_steps": steps, "world_size": 1, "per_device_train_batch_size": 1,
                       "gradient_accumulation_steps": accumulation, "effective_global_batch_size": accumulation,
                       "seed": 42, "enable_thinking": False, "max_seq_length": 2048,
                       "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary"}.items():
        require(summary.get(key) == value, f"Training summary differs: {key}")
    for key, value in c["recipe"].items():
        if key not in {"epochs", "gradient_accumulation_steps"}:
            require(config.get(key) == value, f"Resolved training recipe differs: {key}")
    for key in ("input_adapter", "sample_weight_file", "sample_weight_file_sha256"):
        require(summary.get(key) is None, f"Unapproved warm start or weighting: {key}")
    require(summary["gradient_checkpointing_kwargs"] == {"use_reentrant": False}, "Checkpointing mode differs")
    audit = summary["token_audit"]
    require(audit["samples"] == token_count and audit["over_max_seq_length"] == 0, "Training token audit failed")
    require(math.isfinite(summary["train_metrics"]["train_loss"]), "Nonfinite training loss")
    state = load_json(root / "trainer_state.json")
    require(state["global_step"] == steps and state["epoch"] == epochs, "Actual trainer completion differs")
    initial = load_json(root / "initial_adapter_state.json")
    start = load_json(root / "training_start_audit.json")
    require(start["fresh_optimization_state"] and start["global_step"] == 0 and start["optimizer_state_entries"] == 0,
            "First optimizer state was not fresh")
    require(initial["state_sha256"] == start["source_adapter_state_sha256"] == start["actual_adapter_state_sha256"],
            "Adapter changed before the first update")
    require(initial["checkpoint_identity"] == start["checkpoint_identity"] == summary["checkpoint_identity"], "Initial state belongs to another condition")
    semantic = summary["model_audit"]["semantic_id_tokens"]
    require(semantic["count"] == 108 and semantic["save_embedding_layers"] is False, "SID training policy differs")
    require(semantic["tie_word_embeddings"] is True, "Native embedding sharing was changed")
    require(not summary["model_audit"]["forbidden_trainable_parameters"], "Forbidden parameters trained")
    require(summary["semantic_token_initialization"]["initialization_version"] == "label-token-mean-norm-matched-cpu-fp32-v2", "SID initialization changed")
    require(summary["semantic_token_initialization"]["mapping_sha256"] == sha256(project / MAPPING), "SID mapping changed")
    if kind == "single":
        history = summary["loss_history"]
        require(len(history) >= 2 and history[-1]["loss"] < history[0]["loss"], "Small-sample loss did not decrease")
    else:
        drift = summary["semantic_token_drift"]
        require(drift["all_input_rows_changed"] and drift["all_output_rows_changed"], "Not all SID rows learned")
    if kind in {"single", "full"}:
        require(summary["train_file_sha256"] == sha256(project / TRAIN), "Trainer used wrong full input")
    else:
        require(summary["train_file_sha256"] == sha256(root.parent / "smoke_data/train.parquet"), "Trainer used wrong smoke input")
    return {"samples": n, "optimizer_steps": steps, "epochs": epochs, "global_batch": accumulation,
            "initial_adapter_state_sha256": initial["state_sha256"],
            "trainable_parameters": summary["model_audit"]["trainable_parameters"],
            "trainable_parameter_tensors": summary["model_audit"]["trainable_parameter_tensors"]}


def audit_adapter(project: Path, size: str, root: Path) -> dict[str, Any]:
    from safetensors import safe_open
    model = load_json(project / CONTRACT)["models"][size]
    summary = load_json(root / "run_summary.json")
    adapter = root / "final_adapter/adapter_model.safetensors"
    with safe_open(adapter, framework="np") as handle:
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}
    token_shapes = {k: s for k, s in shapes.items() if "trainable_tokens_delta" in k}
    require(len(token_shapes) == model["expected_trainable_token_tensors"], "Compact tied SID tensor count differs")
    require(all(s == [pq.ParquetFile(project / MAPPING).metadata.num_rows, model["hidden_size"]] for s in token_shapes.values()), "Compact SID tensor dimensions differ")
    require(all("lora_" in k or "trainable_tokens_delta" in k for k in shapes), "Unexpected full or extra trainable tensor saved")
    require(any("lora_" in k for k in shapes), "Missing LoRA weights")
    require(len(shapes) == summary["model_audit"]["trainable_parameter_tensors"], "Saved/live trainable tensor inventory differs")
    require(sha256(root / "final_adapter/interest_token_mapping.parquet") == sha256(project / MAPPING), "Saved mapping differs")
    return {"adapter_sha256": sha256(adapter), "tensor_count": len(shapes), "token_shapes": token_shapes,
            "adapter_bytes": adapter.stat().st_size}


def audit_reload(project: Path, size: str, root: Path) -> dict[str, Any]:
    audit = load_json(root / "semantic_adapter_reload_audit.json")
    model = load_json(project / CONTRACT)["models"][size]
    require(audit["model_name"] == model["model_name"] and audit["model_revision"] == model["model_revision"], "Reload model differs")
    require(audit["sample_count"] == 4 and audit["all_close"], "Reload logits failed")
    require(audit["atol"] == audit["rtol"] == 0.0, "Reload tolerance changed")
    state = audit["adapter_state_integrity"]
    require(state["all_tensors_exact"] and state["saved_state_sha256"] == state["loaded_state_sha256"], "Reloaded tensor bytes differ")
    require(state["tensor_count"] == audit_adapter(project, size, root)["tensor_count"], "Reload tensor count differs")
    require(audit["semantic_base_initialization"]["tie_word_embeddings"] is True, "Reload lost embedding sharing")
    return {"all_tensors_exact": True, "logits_exact": True, "tensor_count": state["tensor_count"]}


def audit_evaluation(project: Path, size: str, root: Path, phase: str, *, logical_shards: int | None = None) -> dict[str, Any]:
    c = load_json(project / CONTRACT)
    model = c["models"][size]
    merged = root / "real_eval/merged"
    metrics = load_json(merged / "metrics.json")
    records = pq.read_table(project / DEV).to_pylist()
    if phase == "smoke":
        records = records[:64]
    shard_count = (4 if phase == "full" else 1) if logical_shards is None else logical_shards
    require(shard_count in {1, 4}, "Unexpected logical shard count")
    require(phase != "full" or shard_count == 4, "Full logical shard count changed")
    predictions = pq.read_table(merged / "id_token_predictions.parquet").to_pylist()
    require(len(predictions) == len(records) == metrics["samples"], "Prediction coverage differs")
    for key in ("model_name", "model_revision"):
        require(metrics.get(key) == model[key], f"Evaluation {key} differs")
    require(metrics["top_k"] == 20 and metrics["cutoffs"] == [1, 5, 10, 20], "Ranking cutoffs differ")
    require(metrics["history_prediction_filtering"] is False, "History filtering changed")
    require(metrics["eval_sha256"] == sha256(project / DEV), "Evaluation data hash differs")
    require(metrics["mapping_sha256"] == sha256(project / CANDIDATE_MAPPING), "Evaluation mapping hash differs")
    require(metrics["label_vocabulary_size"] == 108, "Evaluation vocabulary differs")
    mapping = pq.read_table(project / CANDIDATE_MAPPING).to_pylist()
    token_by_label = {str(r["target_subcategory"]): str(r["interest_token"]) for r in mapping}
    support: Counter[str] = Counter()
    hits = {k: Counter() for k in (1, 5, 10, 20)}
    gains = {k: defaultdict(list) for k in hits}
    for i, (record, prediction) in enumerate(zip(records, predictions)):
        require(prediction["record_index"] == i, f"Missing, duplicate, or reordered prediction: {i}")
        require(str(prediction["impression_id"]) == str(record["impression_id"]), f"Impression mismatch: {i}")
        require(prediction["input_sha256"] == record_input_sha256(record), f"Prediction input hash mismatch: {i}")
        require(prediction["evaluation_identity"] == metrics["evaluation_identity"], "Mixed evaluation identities")
        ranked = prediction["ranked_labels"]
        require(len(ranked) == len(set(ranked)) == 20 and set(ranked).issubset(token_by_label), f"Illegal ranking: {i}")
        require(prediction["ranked_tokens"] == [token_by_label[label] for label in ranked], f"Token mapping differs: {i}")
        scores = prediction["ranked_scores"]
        require(len(scores) == 20 and all(math.isfinite(s) for s in scores), f"Invalid scores: {i}")
        require(all(a >= b for a, b in zip(scores, scores[1:])), f"Unsorted scores: {i}")
        require(math.isfinite(prediction["restricted_entropy"]), f"Invalid entropy: {i}")
        rank_by_label = {label: rank for rank, label in enumerate(ranked, 1)}
        for target in record["acceptable_subcategories"]:
            support[target] += 1
            rank = rank_by_label.get(target, 21)
            for k in hits:
                if rank <= k:
                    hits[k][target] += 1
                    gains[k][target].append(1.0 / math.log2(rank + 1))
    require(len(support) == metrics["supported_interest_count"], "Supported-category denominator differs")
    if phase == "full":
        require(len(records) == 31392 and len(support) == 104, "Full Dev coverage differs")
    recomputed = {}
    for k in hits:
        recall = {label: hits[k][label] / count for label, count in support.items()}
        ndcg = {label: math.fsum(gains[k][label]) / count for label, count in support.items()}
        for name, values in (("recall", recall), ("ndcg", ndcg)):
            key = f"interest_macro_{name}_at_{k}"
            recomputed[key] = math.fsum(values.values()) / len(support)
            require(abs(metrics[key] - recomputed[key]) <= 1e-12, f"Independent metric mismatch: {key}")
            remote = metrics[f"per_class_{name}"][str(k)]
            require(set(remote) == set(values), "Per-class metric coverage differs")
            require(all(abs(remote[label] - value) <= 1e-12 for label, value in values.items()), "Per-class metric mismatch")
    if shard_count > 1:
        manifest = load_json(merged / "merge_manifest.json")
        require(manifest["record_count"] == manifest["expected_record_count"] == len(records), "Merge coverage differs")
        require(manifest["num_shards"] == shard_count, "Logical shard count changed")
        for key, filename in (("merged_prediction_sha256", "id_token_predictions.parquet"),
                              ("metrics_sha256", "metrics.json"), ("per_interest_metrics_sha256", "per_interest_metrics.parquet")):
            require(manifest[key] == sha256(merged / filename), f"Merge hash differs: {filename}")
        for s in range(shard_count):
            shard = root / f"real_eval/shard-{s:02d}"
            shard_manifest = load_json(shard / "shard_manifest.json")
            digest = sha256(shard / "id_token_predictions.parquet")
            require(manifest["shard_prediction_sha256"][str(s)] == digest == shard_manifest["prediction_sha256"], "Shard hash differs")
            rows = pq.read_table(shard / "id_token_predictions.parquet", columns=["record_index", "impression_id", "input_sha256", "ranked_labels", "ranked_tokens", "ranked_scores"]).to_pylist()
            require(len(rows) == shard_manifest["shard_record_count"] == len(range(s, len(records), shard_count)), "Shard coverage differs")
            for row in rows:
                i = row["record_index"]
                require(i % shard_count == s, "Prediction belongs to wrong shard")
                require(all(row[k] == predictions[i][k] for k in row), "Merged prediction differs from shard")
    else:
        manifest = load_json(merged / "shard_manifest.json")
        require(manifest["shard_record_count"] == 64 and manifest["num_shards"] == 1, "Smoke shard differs")
        require(manifest["prediction_sha256"] == sha256(merged / "id_token_predictions.parquet"), "Smoke prediction hash differs")
    return {"records": len(records), "supported_classes": len(support), "target_occurrences": sum(support.values()),
            "metrics": recomputed, "prediction_sha256": sha256(merged / "id_token_predictions.parquet"),
            "metrics_sha256": sha256(merged / "metrics.json"), "evaluation_identity": metrics["evaluation_identity"]}


def verify_manifest(project: Path, size: str, root: Path, phase: str) -> dict[str, Any]:
    marker = load_json(root / f"{phase}_success.json")
    manifest_path = root / "artifact_manifest.json"
    require(marker["artifact_manifest_sha256"] == sha256(manifest_path), "Artifact manifest hash differs")
    manifest = load_json(manifest_path)
    expected_identity = identity(project, size)
    require(marker["identity"] == manifest["identity"] == expected_identity, "Artifact identity differs")
    require(marker["phase"] == manifest["phase"] == phase, "Artifact phase differs")
    require(set(required_artifact_paths(root, phase)).issubset(manifest["files"]), "Artifact manifest omits a required file")
    total = 0
    for relative, item in manifest["files"].items():
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts, "Unsafe artifact path")
        p = root / p
        require(p.is_file() and p.stat().st_size == item["bytes"] and sha256(p) == item["sha256"], f"Artifact bytes differ: {relative}")
        total += item["bytes"]
    return {"files": len(manifest["files"]), "bytes": total, "manifest_sha256": sha256(manifest_path)}


def readback(project: Path, size: str, root: Path, phase: str, output: Path) -> dict[str, Any]:
    files = verify_manifest(project, size, root, phase)
    contract = load_json(project / CONTRACT)
    require(sha256(root / "contract.json") == sha256(project / CONTRACT), "Archived contract differs")
    require(sha256(root / "runtime_source_manifest.json") == sha256(project / SOURCE_MANIFEST), "Archived source manifest differs")
    require(load_json(root / "environment.json")["packages"] == contract["worker_packages"], "Worker environment differs")
    inputs = load_json(root / "input_audit.json")
    require(inputs["identity"] == identity(project, size), "Worker input audit identity differs")
    require(inputs["input_sha256"] == {i["path"]: i["sha256"] for i in contract["input_files"]}, "Worker input hashes differ")
    snapshot = load_json(root / "snapshot_audit.json")
    require(snapshot["model_revision"] == contract["models"][size]["model_revision"], "Worker snapshot revision differs")
    require(snapshot["files"]["config.json"] == contract["models"][size]["model_config_sha256"], "Worker model config differs")
    training = audit_training(project, size, root / "sft", phase)
    reload = audit_reload(project, size, root / "sft")
    if phase == "smoke":
        audit_gradient(load_json(root / "gradient/gradient_audit.json"), load_json(project / CONTRACT)["models"][size])
        audit_training(project, size, root / "single", "single")
        audit_reload(project, size, root / "single")
    evaluation = audit_evaluation(project, size, root, phase)
    report = {"version": "mind-backbone-independent-readback-v1", "status": "passed", "phase": phase,
              "identity": identity(project, size), "files": files, "training": training, "reload": reload,
              "evaluation": evaluation, "artifact_root": str(root.resolve()),
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output, report)
    return report


def validate_smoke_acceptance(project: Path, size: str, receipt: Path | None) -> dict[str, Any]:
    require(receipt is not None and receipt.is_file(), "Full requires an independent downloaded smoke readback")
    accepted = load_json(receipt)
    require(accepted.get("version") == "mind-backbone-independent-readback-v1" and accepted.get("status") == "passed"
            and accepted.get("phase") == "smoke", "Invalid smoke acceptance receipt")
    require(accepted["identity"] == identity(project, size), "Smoke acceptance belongs to another frozen condition")
    require(accepted["training"]["optimizer_steps"] == 8 and accepted["evaluation"]["records"] == 64,
            "Smoke acceptance coverage differs")
    require(accepted["reload"]["logits_exact"] and accepted["reload"]["all_tensors_exact"], "Smoke reload acceptance failed")
    return accepted


def required_artifact_paths(root: Path, phase: str) -> list[str]:
    required = ["environment.json", "input_audit.json", "snapshot_audit.json", "run_identity.json",
                "contract.json", "runtime_source_manifest.json", "training_gate.json", "evaluation_readback.json"]
    stages = ["sft"] + (["single"] if phase == "smoke" else [])
    if phase == "smoke":
        required += ["gradient/gradient_audit.json", "smoke_data/train.parquet"]
    else:
        required += ["smoke_acceptance.json"]
    for stage in stages:
        required += [f"{stage}/{name}" for name in ("run_summary.json", "resolved_config.json", "trainer_state.json", "token_audit.json",
                     "initial_adapter_state.json", "training_start_audit.json",
                     "model_audit.json", "semantic_token_initialization.json", "semantic_token_drift.json", "semantic_reload_reference.json",
                     "semantic_adapter_state_audit.json", "semantic_adapter_reload_audit.json")]
        adapter = root / stage / "final_adapter"
        for name in ("adapter_model.safetensors", "adapter_config.json", "interest_token_mapping.parquet", "tokenizer.json"):
            require((adapter / name).is_file(), f"Missing adapter artifact: {name}")
        required += [str(p.relative_to(root)) for p in adapter.rglob("*") if p.is_file()]
    eval_dirs = [root / "real_eval/merged"] + ([root / f"real_eval/shard-{s:02d}" for s in range(4)] if phase == "full" else [])
    for directory in eval_dirs:
        required += [str(p.relative_to(root)) for p in directory.iterdir() if p.is_file() and p.suffix in {".json", ".parquet"}]
    return sorted(set(required))


def publish(project: Path, size: str, root: Path, phase: str) -> None:
    files = {}
    for relative in required_artifact_paths(root, phase):
        p = root / relative
        require(p.is_file() and p.stat().st_size > 0, f"Missing required artifact: {relative}")
        files[relative] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    manifest = {"version": "mind-backbone-artifact-v1", "phase": phase, "identity": identity(project, size), "files": files}
    write_json(root / "artifact_manifest.json", manifest)
    write_json(root / f"{phase}_success.json", {"phase": phase, "identity": identity(project, size),
               "artifact_manifest_sha256": sha256(root / "artifact_manifest.json")})
    package(root, phase)


def package(root: Path, phase: str) -> None:
    manifest = load_json(root / "artifact_manifest.json")
    partial = root / "readback_artifacts.tar.gz.partial"
    with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
        for relative in sorted([*manifest["files"], "artifact_manifest.json", f"{phase}_success.json"]):
            archive.add(root / relative, arcname=relative, recursive=False)
    partial.replace(root / "readback_artifacts.tar.gz")
    p = root / "readback_artifacts.tar.gz"
    write_json(root / "readback_package.json", {"sha256": sha256(p), "bytes": p.stat().st_size})


def environment_audit(project: Path, output: Path) -> None:
    import torch
    c = load_json(project / CONTRACT)
    versions = {key: importlib.metadata.version(key) for key in c["worker_packages"]}
    require(versions == c["worker_packages"], f"Worker package mismatch: {versions}")
    require(torch.cuda.device_count() == 1, "Exactly one GPU must be visible")
    gpu = torch.cuda.get_device_properties(0)
    require("A100" in gpu.name and gpu.total_memory >= 79 * 1024**3, "Expected A100 80GB")
    write_json(output, {"packages": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
               "cuda_devices": 1, "gpu_name": gpu.name, "gpu_memory_bytes": gpu.total_memory})


def prepare_snapshot(project: Path, size: str, output: Path, *, tokenizer_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    model = load_json(project / CONTRACT)["models"][size]
    patterns = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json"]
    if not tokenizer_only:
        patterns += ["model.safetensors.index.json", "*.safetensors"]

    def validate(value: str) -> tuple[Path, dict[str, Any]]:
        path = Path(value)
        require(path.name == model["model_revision"], "Snapshot revision differs")
        require(sha256(path / "config.json") == model["model_config_sha256"], "Base model config differs")
        files = {name: sha256(path / name) for name in patterns if "*" not in name}
        if not tokenizer_only:
            index = load_json(path / "model.safetensors.index.json")
            for filename in sorted(set(index["weight_map"].values())):
                p = path / filename
                require(p.is_file() and p.stat().st_size > 0, f"Missing model shard: {filename}")
                files[filename] = sha256(p)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        config = config.get_text_config()
        require(config.hidden_size == model["hidden_size"] and config.tie_word_embeddings is True, "Native model structure differs")
        AutoTokenizer.from_pretrained(path, local_files_only=True)
        return path, files

    try:
        path, files = validate(snapshot_download(model["model_name"], revision=model["model_revision"], allow_patterns=patterns, local_files_only=True))
        source = "persistent-cache"
    except (OSError, ValueError):
        path, files = validate(snapshot_download(model["model_name"], revision=model["model_revision"], allow_patterns=patterns, max_workers=4))
        source = "downloaded-or-repaired"
    write_json(output, {"model_name": model["model_name"], "model_revision": model["model_revision"],
               "snapshot_path": str(path), "files": files, "source": source, "tokenizer_only": tokenizer_only,
               "hidden_size": model["hidden_size"], "tie_word_embeddings": True})
    return path


def local_preflight(project: Path, size: str, root: Path) -> None:
    from transformers import AutoTokenizer
    from new_interest.training.sft_data import audit_token_lengths
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    root.mkdir(parents=True, exist_ok=True)
    audit_inputs(project, size, root / "input_audit.json")
    snapshot = prepare_snapshot(project, size, root / "tokenizer_snapshot.json", tokenizer_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    spec = register_semantic_interest_tokens(tokenizer, project / MAPPING)
    audit = audit_token_lengths(project / TRAIN, tokenizer, max_seq_length=2048, enable_thinking=False, output_path=root / "token_audit.json")
    require(audit["samples"] == 21600 and audit["over_max_seq_length"] == 0, "Local token audit failed")
    require(audit["min_assistant_tokens"] == audit["max_assistant_tokens"] == 3, "Completion target tokenization differs from 9B")
    write_json(root / "preflight_passed.json", {"identity": identity(project, size), "token_audit": audit,
               "token_ids": spec.token_ids, "tokenizer_snapshot_sha256": sha256(root / "tokenizer_snapshot.json")})


def run_command(command: list[str], project: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("BACKBONE_COMMAND", json.dumps(command), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            # Preserve full local logs, expose bounded progress in the AMLT log.
            stripped = line.strip()
            if ("loss" in stripped or "CHECKPOINT_" in stripped or "Traceback" in stripped
                    or "Error" in stripped or "%|" in stripped):
                print(stripped[:1200], flush=True)
        exit_code = process.wait()
    require(exit_code == 0, f"Command failed with exit {exit_code}; see {log}")


def run_training(project: Path, size: str, root: Path, kind: str) -> None:
    c = load_json(project / CONTRACT)
    model = c["models"][size]
    sft_config, _ = config_paths(size)
    folder = root / ("single" if kind == "single" else "sft")
    checkpoint = Path("/scratch") / f"mind-backbone-{size}-{root.name}-{kind}-checkpoints"
    n, epochs, accumulation = {"single": (32, 5, 8), "smoke": (108, 2, 32), "full": (21600, 3, 32)}[kind]
    train_file = root / "smoke_data/train.parquet" if kind == "smoke" else project / TRAIN
    if not (folder / "run_summary.json").is_file():
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                   "-m", "new_interest.training.train_sft", "--config", str(project / sft_config),
                   "--train-file", str(train_file), "--output-dir", str(folder), "--checkpoint-dir", str(checkpoint),
                   "--persistent-checkpoint-dir", str(root / f"checkpoints-{kind}"), "--resume-from-checkpoint", "auto",
                   "--run-name", f"qwen35-{size}-bi-sid-{kind}-21600-seed42-v1", "--epochs", str(epochs),
                   "--per-device-train-batch-size", "1", "--gradient-accumulation-steps", str(accumulation),
                   "--save-steps", "100", "--wandb-mode", "disabled"]
        if kind == "single":
            command += ["--max-samples", str(n)]
        if kind == "full":
            accepted = load_json(root / "smoke_acceptance.json")
            command += ["--expected-initial-adapter-state-sha256", accepted["training"]["initial_adapter_state_sha256"]]
        run_command(command, project, root / "logs" / f"train-{kind}.log")
    if not (folder / "trainer_state.json").is_file():
        state_file = checkpoint / "trainer_state.json"
        if not state_file.is_file():
            from new_interest.training.checkpoint_persistence import select_latest_complete_checkpoint
            summary = load_json(folder / "run_summary.json")
            persisted, diagnostics = select_latest_complete_checkpoint(root / f"checkpoints-{kind}", summary["checkpoint_identity"])
            require(persisted is not None and not diagnostics, "Missing committed final Trainer state")
            state_file = persisted / "trainer_state.json"
        require(state_file.is_file(), "Missing actual Trainer state")
        shutil.copyfile(state_file, folder / "trainer_state.json")
    audit_training(project, size, folder, kind)
    write_json(folder / "semantic_adapter_state_audit.json", audit_adapter(project, size, folder))
    reload_path = folder / "semantic_adapter_reload_audit.json"
    if not reload_path.is_file():
        run_command([sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
                     "--model-name", model["model_name"], "--model-revision", model["model_revision"],
                     "--adapter-path", str(folder / "final_adapter"), "--mapping-file", str(project / MAPPING),
                     "--reference-file", str(folder / "semantic_reload_reference.json"), "--output", str(reload_path),
                     "--atol", "0", "--rtol", "0"], project, root / "logs" / f"reload-{kind}.log")
    audit_reload(project, size, folder)
    print("BACKBONE_TRAIN_RELOAD_PASSED", size, kind, flush=True)


def worker(project: Path, size: str, root: Path, phase: str, acceptance: Path | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    expected_identity = identity(project, size)
    if (root / "run_identity.json").exists():
        require(load_json(root / "run_identity.json") == expected_identity, "Output directory belongs to another condition")
    if (root / f"{phase}_success.json").is_file():
        verify_manifest(project, size, root, phase)
        if not (root / "readback_artifacts.tar.gz").is_file():
            package(root, phase)
        print("BACKBONE_REUSED_COMPLETE_OUTPUT", size, phase, flush=True)
        return
    if phase == "full":
        accepted = validate_smoke_acceptance(project, size, acceptance)
        write_json(root / "smoke_acceptance.json", accepted)
    write_json(root / "run_identity.json", expected_identity)
    shutil.copyfile(project / CONTRACT, root / "contract.json")
    shutil.copyfile(project / SOURCE_MANIFEST, root / "runtime_source_manifest.json")
    environment_audit(project, root / "environment.json")
    audit_inputs(project, size, root / "input_audit.json")
    prepare_snapshot(project, size, root / "snapshot_audit.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("BACKBONE_SNAPSHOT_READY", size, flush=True)
    c = load_json(project / CONTRACT)
    if phase == "smoke":
        gradient = root / "gradient/gradient_audit.json"
        if not gradient.is_file():
            run_command([sys.executable, "-m", "new_interest.training.audit_semantic_id_token_gradients",
                         "--config", str(project / config_paths(size)[0]), "--output", str(gradient), "--batch-size", "1"],
                        project, root / "logs/gradient.log")
        audit_gradient(load_json(gradient), c["models"][size])
        print("BACKBONE_GRADIENT_PASSED", size, flush=True)
        run_training(project, size, root, "single")
        table = pq.read_table(project / TRAIN)
        selected = {}
        for row in table.to_pylist():
            selected.setdefault(str(row["target_subcategory"]), row)
        rows = [selected[label] for label in sorted(selected)]
        require(len(rows) == 108, "Smoke must cover every class")
        smoke_file = root / "smoke_data/train.parquet"
        smoke_file.parent.mkdir(parents=True, exist_ok=True)
        expected = pa.Table.from_pylist(rows, schema=table.schema)
        if smoke_file.exists():
            require(pq.read_table(smoke_file).equals(expected), "Existing smoke dataset differs")
        else:
            pq.write_table(expected, smoke_file, compression="zstd", use_dictionary=True)
    run_training(project, size, root, phase)
    if phase == "full":
        initial = load_json(root / "sft/initial_adapter_state.json")
        require(initial["state_sha256"] == accepted["training"]["initial_adapter_state_sha256"],
                "Full and smoke did not start from identical base-plus-adapter parameters")
    write_json(root / "training_gate.json", audit_training(project, size, root / "sft", phase))
    eval_config = project / config_paths(size)[1]
    merged = root / "real_eval/merged"
    count = 1 if phase == "smoke" else 4
    evaluation_started = time.monotonic()
    for shard in range(count):
        output = merged if phase == "smoke" else root / f"real_eval/shard-{shard:02d}"
        command = [sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking", "--config", str(eval_config),
                   "--adapter-path", str(root / "sft/final_adapter"), "--output-dir", str(output),
                   "--num-shards", str(count), "--shard-index", str(shard), "--batch-size", "8"]
        if phase == "smoke":
            command += ["--max-examples", "64"]
        run_command(command, project, root / "logs" / f"eval-{shard}.log")
        print("BACKBONE_EVAL_SHARD_PASSED", size, shard, flush=True)
    if phase == "full":
        from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
        merged_metrics = merge_id_token_shards([root / f"real_eval/shard-{s:02d}" for s in range(4)], project / DEV, project / MAPPING,
                                              merged, expected_records=31392)
        elapsed_wall = time.monotonic() - evaluation_started
        merged_metrics.pop("elapsed_seconds_parallel_wall", None)
        merged_metrics.pop("examples_per_second_parallel", None)
        merged_metrics.update({"execution_mode": "sequential_one_gpu", "elapsed_seconds_sequential_wall": elapsed_wall,
                               "examples_per_second_sequential": 31392 / elapsed_wall})
        write_json(merged / "metrics.json", merged_metrics)
        manifest = load_json(merged / "merge_manifest.json")
        manifest["metrics_sha256"] = sha256(merged / "metrics.json")
        write_json(merged / "merge_manifest.json", manifest)
    result = audit_evaluation(project, size, root, phase)
    result["elapsed_seconds_sequential_wall"] = time.monotonic() - evaluation_started
    write_json(root / "evaluation_readback.json", result)
    publish(project, size, root, phase)
    print("BACKBONE_WORKER_GATE_PASSED", size, phase, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "worker", "readback"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--model-size", choices=("2b", "4b"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-acceptance", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project, root = args.project.resolve(), args.root.resolve()
    if args.action == "preflight":
        local_preflight(project, args.model_size, root)
    elif args.action == "worker":
        worker(project, args.model_size, root, args.phase, args.smoke_acceptance)
    else:
        require(args.output is not None, "Readback requires --output")
        result = readback(project, args.model_size, root, args.phase, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
