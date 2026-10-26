#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_static_sud_semantic_id_llama_vanilla_scaling.sh <smoke|full> <train-size>}"
TRAIN_SIZE="${2:?Usage: run_static_sud_semantic_id_llama_vanilla_scaling.sh <smoke|full> <train-size>}"
case "$TRAIN_SIZE" in
  5400)
    EXPECTED_FULL_STEPS=507
    TRAIN_SHA256="cc1a5921ccaf83bfcef17bdecbcf756561e97448be54e28c3b8e3a673333f72c"
    ;;
  10800)
    EXPECTED_FULL_STEPS=1014
    TRAIN_SHA256="ebcb2f322fc994c19c630cdb16b910e92ad56ebfd012c148b2aa9680a492559a"
    ;;
  16200)
    EXPECTED_FULL_STEPS=1521
    TRAIN_SHA256="188ce84cca2f90afca88a6338bb461c5d40a85dcae0c2a5e6076865d9feb5479"
    ;;
  21600)
    EXPECTED_FULL_STEPS=2025
    TRAIN_SHA256="e76acd152ed6e0e92f8b8015dfbc599603267a480d9321b41b5acb95d10487f0"
    ;;
  *)
    echo "Unsupported Llama Vanilla scaling Train size: $TRAIN_SIZE" >&2
    exit 2
    ;;
esac
case "$MODE" in
  smoke)
    if [[ "$TRAIN_SIZE" != "10800" ]]; then
      echo "Llama scaling smoke is fixed to the 10,800-row subset" >&2
      exit 2
    fi
    EXPECTED_JOB_NAME="llama31-behavior-interest-semantic-id-vanilla-scaling-smoke"
    ;;
  full)
    EXPECTED_JOB_NAME="llama31-behavior-interest-semantic-id-vanilla-${TRAIN_SIZE}"
    ;;
  *)
    echo "Unsupported Llama Vanilla scaling mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected Llama Vanilla scaling job name: $AMLT_JOB_NAME" >&2
  exit 2
fi
NUM_GPUS="${SEMANTIC_ID_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Llama Vanilla scaling requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
export SEMANTIC_ID_TRAIN_SIZE="$TRAIN_SIZE"
export SEMANTIC_ID_GRADIENT_AUDIT_BATCH_SIZE=1
export SEMANTIC_ID_FULL_SAVE_STEPS=100
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
TRAIN_FILE="$DATA_ROOT/train/behavior_interest/${TRAIN_SIZE}.parquet"
EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_llama31_behavior_interest_${TRAIN_SIZE}.yaml"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_llama31_behavior_interest_full.json"
GENERIC_RUNNER="$AMLT_CODE_DIR/scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_semantic_id_token_sft.sh"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"
EVAL_SHA256="c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"

for required_file in \
  "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE" \
  "$CONFIG" "$EVAL_CONFIG" "$GENERIC_RUNNER"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing Llama Vanilla scaling input: $required_file" >&2
    exit 1
  fi
done

if [[ -f "$SUCCESS_MARKER" ]]; then
  python - "$AMLT_OUTPUT_DIR" "$SUCCESS_MARKER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
marker_file = Path(sys.argv[2])


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


marker = json.loads(marker_file.read_text(encoding="utf-8"))
manifest_file = root / "artifact_manifest.json"
if sha256(manifest_file) != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed Llama artifact manifest hash mismatch")
manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    artifact_file = root / relative
    if not artifact_file.is_file() or sha256(artifact_file) != expected:
        raise RuntimeError(f"Committed Llama artifact is corrupt: {relative}")
print("LLAMA_SEMANTIC_ID_VANILLA_SCALING_REUSE_OK", len(manifest["files"]))
PY
  exit 0
fi

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

expected = {
    "accelerate": "1.8.1",
    "huggingface_hub": "0.36.2",
    "peft": "0.18.0",
    "pyarrow": "20.0.0",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "transformers": "4.53.3",
    "wandb": "0.20.1",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Llama worker dependency mismatch: {actual}")
payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": actual,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_devices": torch.cuda.device_count(),
}
if payload["cuda_devices"] != 4:
    raise RuntimeError("Expected four GPUs for Llama Vanilla scaling")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("LLAMA_SEMANTIC_ID_VANILLA_SCALING_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - \
  "$CONFIG" "$EVAL_CONFIG" \
  "$TRAIN_FILE" "$TRAIN_SHA256" \
  "$EVAL_FILE" "$EVAL_SHA256" \
  "$MAPPING_FILE" "$MAPPING_SHA256" \
  "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS" \
  "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_file = Path(sys.argv[1])
eval_config_file = Path(sys.argv[2])
train_file = Path(sys.argv[3])
eval_file = Path(sys.argv[5])
mapping_file = Path(sys.argv[7])
train_size = int(sys.argv[9])
expected_steps = int(sys.argv[10])
output_file = Path(sys.argv[11])
expected_hashes = {
    train_file: sys.argv[4],
    eval_file: sys.argv[6],
    mapping_file: sys.argv[8],
}


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for input_file, expected in expected_hashes.items():
    if sha256(input_file) != expected:
        raise RuntimeError(f"Frozen Llama input hash mismatch: {input_file}")
config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
if config["model_name"] != "meta-llama/Llama-3.1-8B-Instruct":
    raise RuntimeError("Llama config model identity mismatch")
if "sample_weight_file" in config:
    raise RuntimeError("Llama Vanilla config unexpectedly has sample weights")
if config.get("enable_thinking") is not False:
    raise RuntimeError("Llama Vanilla config unexpectedly enables thinking")
if config.get("gradient_checkpointing_use_reentrant") is not False:
    raise RuntimeError("Llama Vanilla config must use non-reentrant checkpointing")
if int(config["epochs"]) != 3 or int(config["seed"]) != 42:
    raise RuntimeError("Llama epoch/seed identity mismatch")
if int(config["per_device_train_batch_size"]) != 1:
    raise RuntimeError("Llama per-device batch mismatch")
if int(config["gradient_accumulation_steps"]) != 8:
    raise RuntimeError("Llama accumulation mismatch")
if float(config["learning_rate"]) != 2e-4:
    raise RuntimeError("Llama learning-rate mismatch")

mapping_rows = pq.read_table(mapping_file).to_pylist()
token_by_label = {
    str(row["target_subcategory"]): str(row["interest_token"])
    for row in mapping_rows
}
if len(token_by_label) != 108 or len(set(token_by_label.values())) != 108:
    raise RuntimeError("Llama mapping is not one-to-one over 108 IDs")
rows = pq.read_table(train_file).to_pylist()
if len(rows) != train_size:
    raise RuntimeError("Llama Train row count mismatch")
if len({str(row["user_id"]) for row in rows}) != train_size:
    raise RuntimeError("Llama Train is not one row per user")
if {str(row["task_type"]) for row in rows} != {"behavior_interest"}:
    raise RuntimeError("Llama Train input is not Behavior+Interest")
counts = Counter(str(row["target_subcategory"]) for row in rows)
if len(counts) != 108 or set(counts.values()) != {train_size // 108}:
    raise RuntimeError("Llama Train is not class-balanced")
for row in rows:
    label = str(row["target_subcategory"])
    response = str(row["response"])
    if response != token_by_label[label] or not re.fullmatch(r"<interest_\d{3}>", response):
        raise RuntimeError("Llama Train response is not its Semantic ID")

eval_rows = pq.read_table(eval_file).to_pylist()
if len(eval_rows) != 31392:
    raise RuntimeError("Llama real Dev row count mismatch")
if len({str(row["impression_id"]) for row in eval_rows}) != 31392:
    raise RuntimeError("Llama real Dev impression IDs are not unique")
supported = {
    str(label)
    for row in eval_rows
    for label in row["acceptable_subcategories"]
}
if len(supported) != 104:
    raise RuntimeError("Llama real Dev supported-interest count mismatch")
eval_config = json.loads(eval_config_file.read_text(encoding="utf-8"))
if eval_config["model_name"] != config["model_name"]:
    raise RuntimeError("Llama evaluator model mismatch")
if eval_config["top_k"] != 20 or eval_config["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Llama evaluator Top-K mismatch")
audit = {
    "audit_version": "llama-semantic-id-vanilla-scaling-input-v1",
    "model_name": config["model_name"],
    "task_type": "behavior_interest",
    "output_type": "semantic_id",
    "sample_weight_file": None,
    "train_size": train_size,
    "samples_per_interest": train_size // 108,
    "expected_optimizer_steps": expected_steps,
    "seed": 42,
    "real_eval_rows": len(eval_rows),
    "real_eval_supported_interests": len(supported),
    "input_sha256": {file_path.name: value for file_path, value in expected_hashes.items()},
}
output_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("LLAMA_SEMANTIC_ID_VANILLA_SCALING_INPUT_OK", json.dumps(audit, sort_keys=True))
PY

bash "$GENERIC_RUNNER" llama31 "$MODE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ "$MODE" == "smoke" ]]; then
  SMOKE_SFT_DIR="$AMLT_OUTPUT_DIR/smoke/ddp"
  SMOKE_EVAL_DIR="$AMLT_OUTPUT_DIR/smoke/eval"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
    --config "$EVAL_CONFIG" \
    --adapter-path "$SMOKE_SFT_DIR/final_adapter" \
    --eval-file "$EVAL_FILE" \
    --interest-token-mapping-file "$MAPPING_FILE" \
    --output-dir "$SMOKE_EVAL_DIR" \
    --run-name "llama31-behavior-interest-semantic-id-vanilla-scaling-smoke" \
    --max-examples 64 \
    --num-shards 1 \
    --shard-index 0 \
    --batch-size 8
fi

python - "$AMLT_OUTPUT_DIR" "$MODE" "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
mode = sys.argv[2]
train_size = int(sys.argv[3])
expected_steps = int(sys.argv[4])
if mode == "smoke":
    gradient = json.loads((root / "smoke/gradient/gradient_audit.json").read_text())
    if gradient["batch_size"] != 1 or gradient["representative_samples"] != 108:
        raise RuntimeError("Llama gradient audit identity mismatch")
    if not gradient["all_token_rows_finite_and_nonzero"]:
        raise RuntimeError("Llama all-token gradient audit failed")
    for phase, expected_world, expected_train, expected_phase_steps in (
        ("single", 1, 32, 20),
        ("ddp", 4, 32, 2),
    ):
        summary = json.loads((root / f"smoke/{phase}/run_summary.json").read_text())
        if summary["world_size"] != expected_world:
            raise RuntimeError(f"Llama {phase} world-size mismatch")
        if summary["train_samples"] != expected_train:
            raise RuntimeError(f"Llama {phase} sample-count mismatch")
        if summary["expected_optimizer_steps"] != expected_phase_steps:
            raise RuntimeError(f"Llama {phase} step-count mismatch")
        if summary["token_audit"]["samples"] != train_size:
            raise RuntimeError(f"Llama {phase} token-audit scope mismatch")
        if summary["token_audit"]["over_max_seq_length"] != 0:
            raise RuntimeError(f"Llama {phase} has over-length samples")
        if summary["sample_weight_file"] is not None:
            raise RuntimeError(f"Llama {phase} consumed sample weights")
        reload = json.loads(
            (root / f"smoke/{phase}/semantic_adapter_reload_audit.json").read_text()
        )
        integrity = reload["adapter_state_integrity"]
        if not reload["all_close"] or not integrity["all_tensors_exact"]:
            raise RuntimeError(f"Llama {phase} adapter reload failed")
        if integrity["tensor_count"] != 450:
            raise RuntimeError(f"Llama {phase} adapter tensor count mismatch")
    eval_root = root / "smoke/eval"
    expected_eval = 64
else:
    summary = json.loads((root / "sft/run_summary.json").read_text())
    if summary["world_size"] != 4 or summary["train_samples"] != train_size:
        raise RuntimeError("Llama full DDP/data identity mismatch")
    if summary["expected_optimizer_steps"] != expected_steps:
        raise RuntimeError("Llama full step-count mismatch")
    if summary["configured_epochs"] != 3.0:
        raise RuntimeError("Llama full epoch-count mismatch")
    if summary["sample_weight_file"] is not None:
        raise RuntimeError("Llama full consumed sample weights")
    if summary["token_audit"]["samples"] != train_size:
        raise RuntimeError("Llama full token-audit scope mismatch")
    checkpoint = root / f"sft/checkpoints/checkpoint-{expected_steps}"
    for name in (
        "trainer_state.json", "optimizer.pt", "scheduler.pt",
        "adapter_model.safetensors", "adapter_config.json",
    ):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise RuntimeError(f"Llama final checkpoint is incomplete: {name}")
    reload = json.loads((root / "sft/semantic_adapter_reload_audit.json").read_text())
    integrity = reload["adapter_state_integrity"]
    if not reload["all_close"] or not integrity["all_tensors_exact"]:
        raise RuntimeError("Llama full adapter reload failed")
    if integrity["tensor_count"] != 450:
        raise RuntimeError("Llama full adapter tensor count mismatch")
    eval_root = root / "eval/merged"
    expected_eval = 31392
metrics = json.loads((eval_root / "metrics.json").read_text())
if metrics["samples"] != expected_eval:
    raise RuntimeError("Llama evaluation row-count mismatch")
if metrics["label_vocabulary_size"] != 108 or metrics["top_k"] != 20:
    raise RuntimeError("Llama evaluation vocabulary/Top-K mismatch")
if mode == "full" and metrics["supported_interest_count"] != 104:
    raise RuntimeError("Llama full Macro denominator mismatch")
for key in ("recall_at_10", "ndcg_at_10", "interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Llama evaluation metric is non-finite: {key}")
predictions = pq.read_table(eval_root / "id_token_predictions.parquet").to_pylist()
if len(predictions) != expected_eval:
    raise RuntimeError("Llama prediction row-count mismatch")
if len({int(row["record_index"]) for row in predictions}) != expected_eval:
    raise RuntimeError("Llama prediction record indices are not unique")
print(
    "LLAMA_SEMANTIC_ID_VANILLA_SCALING_OUTPUT_OK",
    mode,
    train_size,
    metrics["recall_at_10"],
    metrics["interest_macro_recall_at_10"],
)
PY

python - \
  "$AMLT_OUTPUT_DIR" "$MODE" "$TRAIN_SIZE" \
  "$TRAIN_SHA256" "$EVAL_SHA256" "$MAPPING_SHA256" <<'PY'
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
train_size = int(sys.argv[3])
identity = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "task_type": "behavior_interest",
    "output_type": "semantic_id",
    "training": "vanilla",
    "sample_weight_file": None,
    "train_size": train_size,
    "seed": 42,
    "train_sha256": sys.argv[4],
    "eval_sha256": sys.argv[5],
    "mapping_sha256": sys.argv[6],
}


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(file_path: Path, payload: object) -> None:
    temporary = file_path.with_name(f".{file_path.name}.partial-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file_path)


required = ["environment.json", "input_audit.json"]
if mode == "smoke":
    required.extend(
        [
            "smoke/gradient/gradient_audit.json",
            "smoke/single/run_summary.json",
            "smoke/single/semantic_adapter_state_audit.json",
            "smoke/single/semantic_adapter_reload_audit.json",
            "smoke/ddp/run_summary.json",
            "smoke/ddp/semantic_adapter_state_audit.json",
            "smoke/ddp/semantic_adapter_reload_audit.json",
            "smoke/ddp/final_adapter/adapter_model.safetensors",
            "smoke/ddp/final_adapter/adapter_config.json",
            "smoke/ddp/final_adapter/interest_token_mapping.parquet",
            "smoke/eval/metrics.json",
            "smoke/eval/id_token_predictions.parquet",
            "smoke/eval/shard_manifest.json",
        ]
    )
else:
    required.extend(
        [
            "sft/run_summary.json",
            "sft/semantic_adapter_state_audit.json",
            "sft/semantic_adapter_reload_audit.json",
            "sft/final_adapter/adapter_model.safetensors",
            "sft/final_adapter/adapter_config.json",
            "sft/final_adapter/interest_token_mapping.parquet",
            "eval/merged/metrics.json",
            "eval/merged/id_token_predictions.parquet",
            "eval/merged/per_interest_metrics.parquet",
            "eval/merged/merge_manifest.json",
        ]
    )
files = {}
for relative in required:
    artifact_file = root / relative
    if not artifact_file.is_file() or artifact_file.stat().st_size == 0:
        raise RuntimeError(f"Missing Llama artifact: {relative}")
    files[relative] = sha256(artifact_file)
manifest_file = root / "artifact_manifest.json"
atomic_json(
    manifest_file,
    {
        "manifest_version": "llama-semantic-id-vanilla-scaling-artifact-v1",
        "mode": mode,
        "identity": identity,
        "files": dict(sorted(files.items())),
    },
)
atomic_json(
    root / f"{mode}_success.json",
    {
        "mode": mode,
        "identity": identity,
        "artifact_manifest_sha256": sha256(manifest_file),
    },
)
print("LLAMA_SEMANTIC_ID_VANILLA_SCALING_GATE_OK", mode, train_size, len(files))
PY
