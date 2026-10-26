#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${1:?Usage: run_static_sud_text_label_vanilla_scaling.sh <qwen35|llama31> <smoke|full> <train-size>}"
MODE="${2:?Usage: run_static_sud_text_label_vanilla_scaling.sh <qwen35|llama31> <smoke|full> <train-size>}"
TRAIN_SIZE="${3:?Usage: run_static_sud_text_label_vanilla_scaling.sh <qwen35|llama31> <smoke|full> <train-size>}"

case "$TRAIN_SIZE" in
  5400)
    EXPECTED_FULL_STEPS=507
    TRAIN_SHA256="b44fc82823a337b807ad193541466ba76424d2afb97ecff1999be166911cabd0"
    ;;
  10800)
    EXPECTED_FULL_STEPS=1014
    TRAIN_SHA256="1e091c828d9a72ce1426631cf260f9ce11255827623efc7aa34f947fabc01d15"
    ;;
  16200)
    EXPECTED_FULL_STEPS=1521
    TRAIN_SHA256="e81c3704e1cc09ac1a52450d38a58409bd6597b60ade84833b3d8070650aebed"
    ;;
  21600)
    EXPECTED_FULL_STEPS=2025
    TRAIN_SHA256="3738aa67fc563c08a0e671c8f9c6deb0456c2156e6b0cfb4b3ae700c78227e87"
    ;;
  *)
    echo "Unsupported Text-label Train size: $TRAIN_SIZE" >&2
    exit 2
    ;;
esac

case "$MODEL_FAMILY" in
  qwen35)
    MODEL_NAME="Qwen/Qwen3.5-9B"
    EXPECTED_ADAPTER_TENSORS=496
    EXPECTED_LEARNING_RATE="0.00002"
    ;;
  llama31)
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    EXPECTED_ADAPTER_TENSORS=448
    EXPECTED_LEARNING_RATE="0.0002"
    ;;
  *)
    echo "Unsupported Text-label model family: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke)
    if [[ "$TRAIN_SIZE" != "10800" ]]; then
      echo "Text-label scaling smoke is fixed to 10,800 rows" >&2
      exit 2
    fi
    EXPECTED_JOB_NAME="${MODEL_FAMILY}-bi-text-label-vanilla-scaling-smoke"
    ;;
  full)
    EXPECTED_JOB_NAME="${MODEL_FAMILY}-bi-text-label-vanilla-${TRAIN_SIZE}"
    ;;
  *)
    echo "Unsupported Text-label mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected Text-label job name: $AMLT_JOB_NAME" >&2
  exit 2
fi
NUM_GPUS="${TEXT_LABEL_SCALING_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Text-label scaling requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"

DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_text_label_200_h50_behavior_interest_scaling_v1"
TRAIN_FILE="$DATA_ROOT/train/behavior_interest/${TRAIN_SIZE}.parquet"
EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
TRAIN_CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_text_label_${MODEL_FAMILY}_behavior_interest_vanilla_${TRAIN_SIZE}_v1.yaml"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_${MODEL_FAMILY}_behavior_interest_full.json"
EVAL_SHA256="59d4889a61c67fd54f7c413900373c70c11084c3b85c9137ec083ba9f6347237"
LABELS_SHA256="1ee3a12f90b4a3f14aab21518915ca30122a052cbcc53fc29c420af0e49c6f51"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"

for required in "$TRAIN_FILE" "$EVAL_FILE" "$LABELS_FILE" "$TRAIN_CONFIG" "$BEAM_CONFIG"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing Text-label scaling input: $required" >&2
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
marker = json.loads(Path(sys.argv[2]).read_text())
manifest_path = root / "artifact_manifest.json"
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed Text-label manifest hash mismatch")
manifest = json.loads(manifest_path.read_text())
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Committed Text-label artifact is corrupt: {relative}")
print("TEXT_LABEL_VANILLA_SCALING_REUSE_OK", len(manifest["files"]))
PY
  exit 0
fi

python - "$MODEL_NAME" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = sys.argv[1]
token = os.environ.get("HF_TOKEN")


def validate_snapshot(value: str) -> Path:
    path = Path(value)
    if not any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in path.glob("*.safetensors")
    ) and not any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in path.glob("*.bin")
    ):
        raise RuntimeError("Model snapshot contains no weight files")
    AutoConfig.from_pretrained(path, local_files_only=True)
    AutoTokenizer.from_pretrained(path, local_files_only=True)
    return path


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception as cached_error:
    if name.startswith("meta-llama/") and not token:
        raise RuntimeError("Llama snapshot is unavailable and HF_TOKEN was not injected") from cached_error
    snapshot = validate_snapshot(snapshot_download(name, token=token))
    source = "downloaded-or-repaired"
print("TEXT_LABEL_MODEL_SNAPSHOT_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - "$AMLT_OUTPUT_DIR/environment.json" "$MODEL_FAMILY" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

family = sys.argv[2]
expected = {
    "qwen35": {"transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0"},
    "llama31": {"transformers": "4.53.3", "peft": "0.18.0", "accelerate": "1.8.1"},
}[family]
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Text-label worker dependency mismatch: {actual}")
payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": actual,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_devices": torch.cuda.device_count(),
}
if payload["cuda_devices"] != 4:
    raise RuntimeError("Expected four GPUs")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
print("TEXT_LABEL_VANILLA_SCALING_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - \
  "$TRAIN_CONFIG" "$TRAIN_FILE" "$TRAIN_SHA256" \
  "$EVAL_FILE" "$EVAL_SHA256" "$LABELS_FILE" "$LABELS_SHA256" \
  "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS" "$MODEL_NAME" \
  "$EXPECTED_LEARNING_RATE" "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
eval_path = Path(sys.argv[4])
labels_path = Path(sys.argv[6])
train_size = int(sys.argv[8])
expected_steps = int(sys.argv[9])
model_name = sys.argv[10]
expected_lr = float(sys.argv[11])
output_path = Path(sys.argv[12])
expected_hashes = {train_path: sys.argv[3], eval_path: sys.argv[5], labels_path: sys.argv[7]}

for path, expected in expected_hashes.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Frozen Text-label input hash mismatch: {path}")
config = yaml.safe_load(config_path.read_text())
expected_config = {
    "model_name": model_name,
    "loss_reduction": "token_mean",
    "enable_thinking": False,
    "epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing_use_reentrant": False,
    "seed": 42,
}
for key, expected in expected_config.items():
    if config.get(key) != expected:
        raise RuntimeError(f"Text-label config mismatch for {key}: {config.get(key)}")
if float(config["learning_rate"]) != expected_lr:
    raise RuntimeError("Text-label learning-rate mismatch")
if "sample_weight_file" in config or "semantic_id_token_mapping_file" in config:
    raise RuntimeError("Text-label config contains a forbidden weight or Semantic-ID mapping")

rows = pq.read_table(train_path).to_pylist()
if len(rows) != train_size or len({str(row["user_id"]) for row in rows}) != train_size:
    raise RuntimeError("Text-label Train row/user count mismatch")
if {str(row["task_type"]) for row in rows} != {"behavior_interest"}:
    raise RuntimeError("Text-label Train is not Behavior+Interest")
counts = Counter(str(row["target_subcategory"]) for row in rows)
if len(counts) != 108 or set(counts.values()) != {train_size // 108}:
    raise RuntimeError("Text-label Train is not class-balanced")
for row in rows:
    expected = f"New topic: {row['target_subcategory']}"
    if str(row["response"]) != expected or str(row["messages"][1]["content"]) != expected:
        raise RuntimeError("Text-label response mismatch")
    if re.fullmatch(r"<interest_\d{3}>", str(row["response"])):
        raise RuntimeError("Semantic-ID leaked into Text-label output")

labels = [str(row["target_subcategory"]) for row in pq.read_table(labels_path).to_pylist()]
if len(labels) != 108 or len(set(labels)) != 108:
    raise RuntimeError("Text-label vocabulary is not 108 unique labels")
eval_rows = pq.read_table(eval_path).to_pylist()
if len(eval_rows) != 31392 or len({str(row["impression_id"]) for row in eval_rows}) != 31392:
    raise RuntimeError("Text-label Dev identity mismatch")
supported = {str(label) for row in eval_rows for label in row["acceptable_subcategories"]}
if len(supported) != 104 or not supported.issubset(set(labels)):
    raise RuntimeError("Text-label Dev support mismatch")
payload = {
    "audit_version": "text-label-vanilla-scaling-input-v1",
    "model_name": model_name,
    "task_type": "behavior_interest",
    "output_type": "text_label",
    "response_prefix": "New topic: ",
    "loss_reduction": "token_mean",
    "sample_weight_file": None,
    "train_size": train_size,
    "samples_per_interest": train_size // 108,
    "expected_optimizer_steps": expected_steps,
    "seed": 42,
    "real_eval_rows": len(eval_rows),
    "real_eval_supported_interests": len(supported),
    "input_sha256": {path.name: value for path, value in expected_hashes.items()},
}
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print("TEXT_LABEL_VANILLA_SCALING_INPUT_OK", json.dumps(payload, sort_keys=True))
PY

validate_training() {
  local summary_path="$1"
  local expected_world="$2"
  local expected_samples="$3"
  local expected_steps="$4"
  local require_decrease="$5"
  python - "$summary_path" "$expected_world" "$expected_samples" "$expected_steps" "$require_decrease" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
expected = {
    "world_size": int(sys.argv[2]),
    "train_samples": int(sys.argv[3]),
    "expected_optimizer_steps": int(sys.argv[4]),
    "effective_global_batch_size": int(sys.argv[2]) * 8,
    "loss_reduction": "token_mean",
}
actual = {key: summary[key] for key in expected}
if actual != expected:
    raise RuntimeError(f"Text-label training contract mismatch: {actual}")
if summary.get("sample_weight_file") is not None:
    raise RuntimeError("Text-label training consumed sample weights")
if summary.get("gradient_checkpointing_kwargs") != {"use_reentrant": False}:
    raise RuntimeError("Text-label training used reentrant checkpointing")
if int(summary["token_audit"]["over_max_seq_length"]) != 0:
    raise RuntimeError("Text-label training has over-length rows")
configured_epochs = float(summary["configured_epochs"])
completed_epochs = float(summary["train_metrics"].get("epoch", -1.0))
epoch_tolerance = configured_epochs / max(
    int(summary["expected_optimizer_steps"]),
    1,
)
if not (
    configured_epochs - epoch_tolerance
    <= completed_epochs
    <= configured_epochs + epoch_tolerance
):
    raise RuntimeError(
        "Text-label training did not complete all configured epochs: "
        f"completed={completed_epochs}, configured={configured_epochs}, "
        f"one_step_tolerance={epoch_tolerance}"
    )
loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(loss):
    raise RuntimeError("Text-label train loss is non-finite")
if sys.argv[5] == "true":
    history = [float(row["loss"]) for row in summary["loss_history"] if "loss" in row]
    if len(history) < 2 or not history[-1] < history[0]:
        raise RuntimeError(f"Text-label smoke loss did not decrease: {history}")
adapter = path.parent / "final_adapter" / "adapter_model.safetensors"
if not adapter.is_file() or adapter.stat().st_size == 0:
    raise RuntimeError("Text-label final adapter is missing")
print("TEXT_LABEL_TRAIN_GATE_OK", expected, loss)
PY
}

verify_adapter() {
  local sft_dir="$1"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_lora_adapter \
    --model-name "$MODEL_NAME" \
    --adapter-path "$sft_dir/final_adapter" \
    --expected-tensors "$EXPECTED_ADAPTER_TENSORS" \
    --output "$sft_dir/adapter_reload_audit.json"
}

run_smoke_eval() {
  local sft_dir="$1"
  local beam_dir="$2"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_beam \
    --config "$BEAM_CONFIG" \
    --adapter-path "$sft_dir/final_adapter" \
    --eval-file "$EVAL_FILE" \
    --output-dir "$beam_dir" \
    --run-name "${MODEL_FAMILY}-bi-text-label-vanilla-scaling-smoke" \
    --max-examples 16 \
    --num-shards 1 \
    --shard-index 0 \
    --wandb-mode disabled
  python - "$beam_dir" "$LABELS_FILE" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
labels = {str(row["target_subcategory"]) for row in pq.read_table(sys.argv[2]).to_pylist()}
metrics = json.loads((root / "metrics.json").read_text())
rows = pq.read_table(root / "beam_predictions.parquet").to_pylist()
if int(metrics["samples"]) != 16 or len(rows) != 16:
    raise RuntimeError("Text-label smoke evaluation row-count mismatch")
if len(labels) != 108 or int(metrics["top_k"]) != 20:
    raise RuntimeError("Text-label smoke vocabulary/Top-K mismatch")
for row in rows:
    ranked = [str(value) for value in row["ranked_labels"]]
    responses = [str(value) for value in row["ranked_responses"]]
    if len(ranked) != 20 or len(set(ranked)) != 20 or not set(ranked).issubset(labels):
        raise RuntimeError("Text-label smoke ranking is illegal")
    if responses != [f"New topic: {label}" for label in ranked]:
        raise RuntimeError("Text-label smoke response/label mismatch")
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Non-finite Text-label smoke metric: {key}")
print("TEXT_LABEL_BEAM_SMOKE_GATE_OK", len(rows))
PY
}

if [[ "$MODE" == "smoke" ]]; then
  SINGLE_DIR="$AMLT_OUTPUT_DIR/smoke/single"
  DDP_DIR="$AMLT_OUTPUT_DIR/smoke/ddp"
  BEAM_DIR="$AMLT_OUTPUT_DIR/smoke/beam"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.train_sft \
    --config "$TRAIN_CONFIG" --train-file "$TRAIN_FILE" \
    --output-dir "$SINGLE_DIR" --run-name "${MODEL_FAMILY}-text-label-single-smoke" \
    --max-samples 32 --epochs 5 --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 --wandb-mode disabled
  validate_training "$SINGLE_DIR/run_summary.json" 1 32 20 true
  verify_adapter "$SINGLE_DIR"
  python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    --config "$TRAIN_CONFIG" --train-file "$TRAIN_FILE" \
    --output-dir "$DDP_DIR" --run-name "${MODEL_FAMILY}-text-label-ddp-smoke" \
    --max-samples 32 --epochs 2 --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 --wandb-mode disabled
  validate_training "$DDP_DIR/run_summary.json" 4 32 2 false
  verify_adapter "$DDP_DIR"
  run_smoke_eval "$DDP_DIR" "$BEAM_DIR"
else
  SFT_DIR="$AMLT_OUTPUT_DIR/sft"
  LOCAL_CHECKPOINT_DIR="/scratch/${AMLT_JOB_ID:-text-label-scaling}/checkpoints"
  PERSISTENT_CHECKPOINT_DIR="$AMLT_OUTPUT_DIR/persistent_checkpoints"
  mkdir -p "$SFT_DIR" "$LOCAL_CHECKPOINT_DIR" "$PERSISTENT_CHECKPOINT_DIR"
  python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    --config "$TRAIN_CONFIG" --train-file "$TRAIN_FILE" \
    --output-dir "$SFT_DIR" --run-name "${MODEL_FAMILY}-bi-text-label-vanilla-${TRAIN_SIZE}" \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    --checkpoint-dir "$LOCAL_CHECKPOINT_DIR" \
    --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_DIR" \
    --resume-from-checkpoint auto --save-steps 100 --wandb-mode disabled
  validate_training "$SFT_DIR/run_summary.json" 4 "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS" false
  verify_adapter "$SFT_DIR"
  RECOVERY_CHECKPOINT="$(python - \
    "$PERSISTENT_CHECKPOINT_DIR" "$SFT_DIR/run_summary.json" \
    "$EXPECTED_FULL_STEPS" 100 "$AMLT_OUTPUT_DIR/recovery_checkpoint_audit.json" <<'PY'
import json
import sys
from pathlib import Path

from new_interest.training.checkpoint_persistence import (
    select_latest_complete_checkpoint,
    validate_persistent_checkpoint,
)

persistent_root = Path(sys.argv[1])
summary = json.loads(Path(sys.argv[2]).read_text())
expected_steps = int(sys.argv[3])
save_steps = int(sys.argv[4])
audit_path = Path(sys.argv[5])
selected, diagnostics = select_latest_complete_checkpoint(
    persistent_root,
    summary["checkpoint_identity"],
)
if selected is None:
    raise RuntimeError(
        "Text-label run has no complete same-identity recovery checkpoint: "
        + "; ".join(diagnostics)
    )
marker = validate_persistent_checkpoint(
    selected,
    summary["checkpoint_identity"],
)
selected_step = int(marker["global_step"])
if selected_step > expected_steps or expected_steps - selected_step >= save_steps:
    raise RuntimeError(
        "Latest Text-label recovery checkpoint is outside the final save interval: "
        f"step={selected_step}, expected={expected_steps}, save_steps={save_steps}"
    )
relative = selected.relative_to(audit_path.parent).as_posix()
audit = {
    "checkpoint_relative": relative,
    "global_step": selected_step,
    "expected_optimizer_steps": expected_steps,
    "tail_steps_after_checkpoint_upper_bound": expected_steps - selected_step,
    "save_steps": save_steps,
    "identity_sha256": marker["identity_sha256"],
    "diagnostics": diagnostics,
}
audit_path.write_text(json.dumps(audit, indent=2) + "\n")
print(selected)
PY
)"
  echo "TEXT_LABEL_RECOVERY_CHECKPOINT_GATE_OK $RECOVERY_CHECKPOINT"

  BEAM_ROOT="$AMLT_OUTPUT_DIR/beam"
  mkdir -p "$BEAM_ROOT"
  PIDS=()
  SHARD_DIRS=()
  for SHARD_INDEX in 0 1 2 3; do
    SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"
    SHARD_DIR="$BEAM_ROOT/$SHARD_NAME"
    SHARD_DIRS+=("$SHARD_DIR")
    (
      export CUDA_VISIBLE_DEVICES="$SHARD_INDEX"
      python -m new_interest.evaluation.evaluate_beam \
        --config "$BEAM_CONFIG" \
        --adapter-path "$SFT_DIR/final_adapter" \
        --eval-file "$EVAL_FILE" \
        --output-dir "$SHARD_DIR" \
        --run-name "${MODEL_FAMILY}-bi-text-label-vanilla-${TRAIN_SIZE}-${SHARD_NAME}" \
        --num-shards 4 --shard-index "$SHARD_INDEX" --wandb-mode disabled \
        >"$BEAM_ROOT/${SHARD_NAME}.log" 2>&1
    ) &
    PIDS+=("$!")
  done
  FAILED=0
  for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then FAILED=1; fi
  done
  if [[ "$FAILED" != "0" ]]; then
    for LOG in "$BEAM_ROOT"/shard-*.log; do tail -n 80 "$LOG" >&2 || true; done
    exit 1
  fi
  python -m new_interest.evaluation.merge_beam_shards \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --eval-file "$EVAL_FILE" \
    --retained-interests-file "$LABELS_FILE" \
    --output-dir "$BEAM_ROOT/merged" --expected-records 31392
  python - "$BEAM_ROOT/merged" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
metrics = json.loads((root / "metrics.json").read_text())
rows = pq.read_table(root / "beam_predictions.parquet").to_pylist()
if len(rows) != 31392 or int(metrics["samples"]) != 31392:
    raise RuntimeError("Text-label full evaluation row-count mismatch")
if int(metrics["supported_interest_count"]) != 104:
    raise RuntimeError("Text-label full Macro denominator mismatch")
if int(metrics["label_vocabulary_size"]) != 108 or int(metrics["top_k"]) != 20:
    raise RuntimeError("Text-label full vocabulary/Top-K mismatch")
if len({int(row["record_index"]) for row in rows}) != 31392:
    raise RuntimeError("Text-label full record indices are not unique")
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Non-finite Text-label metric: {key}")
print("TEXT_LABEL_BEAM_FULL_GATE_OK", metrics["interest_macro_recall_at_10"], metrics["interest_macro_ndcg_at_10"])
PY
fi

python - "$AMLT_OUTPUT_DIR" "$MODE" "$MODEL_FAMILY" "$TRAIN_SIZE" "$TRAIN_SHA256" "$EVAL_SHA256" "$LABELS_SHA256" <<'PY'
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

root = Path(sys.argv[1])
mode, family = sys.argv[2], sys.argv[3]
train_size = int(sys.argv[4])
identity = {
    "model_family": family,
    "task_type": "behavior_interest",
    "output_type": "text_label",
    "response_prefix": "New topic: ",
    "loss_reduction": "token_mean",
    "sample_weight_file": None,
    "train_size": train_size,
    "seed": 42,
    "train_sha256": sys.argv[5],
    "eval_sha256": sys.argv[6],
    "labels_sha256": sys.argv[7],
}
required = ["environment.json", "input_audit.json"]
if mode == "smoke":
    required += [
        "smoke/single/run_summary.json",
        "smoke/single/adapter_reload_audit.json",
        "smoke/ddp/run_summary.json",
        "smoke/ddp/adapter_reload_audit.json",
        "smoke/ddp/final_adapter/adapter_model.safetensors",
        "smoke/beam/beam_predictions.parquet",
        "smoke/beam/metrics.json",
        "smoke/beam/shard_manifest.json",
    ]
else:
    checkpoint_audit_path = root / "recovery_checkpoint_audit.json"
    checkpoint_audit = json.loads(checkpoint_audit_path.read_text())
    checkpoint_relative = Path(checkpoint_audit["checkpoint_relative"])
    if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
        raise RuntimeError("Unsafe Text-label recovery checkpoint path")
    required += [
        "sft/run_summary.json",
        "sft/adapter_reload_audit.json",
        "sft/final_adapter/adapter_model.safetensors",
        "sft/final_adapter/adapter_config.json",
        "recovery_checkpoint_audit.json",
        f"{checkpoint_relative.as_posix()}/checkpoint_complete.json",
        "beam/merged/beam_predictions.parquet",
        "beam/merged/metrics.json",
        "beam/merged/per_interest_metrics.parquet",
        "beam/merged/merge_manifest.json",
    ]


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


files = {}
for relative in required:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing Text-label artifact: {relative}")
    files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "manifest_version": "text-label-vanilla-scaling-artifact-v1",
    "mode": mode,
    "identity": identity,
    "files": files,
}
manifest_path = root / "artifact_manifest.json"
atomic_json(manifest_path, manifest)
marker = {
    "mode": mode,
    "identity": identity,
    "artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
}
atomic_json(root / f"{mode}_success.json", marker)
print("TEXT_LABEL_VANILLA_SCALING_GATE_OK", family, mode, train_size, len(files))
PY
