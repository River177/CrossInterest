#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${STATIC_SUD_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Behavior+Interest serial training requires exactly 4 GPUs" >&2
  exit 2
fi
CHECKPOINT_SAVE_STEPS="${BEHAVIOR_INTEREST_SAVE_STEPS:-100}"
if [[ ! "$CHECKPOINT_SAVE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "BEHAVIOR_INTEREST_SAVE_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for Llama-3.1 access" >&2
  exit 1
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true

TRAIN_FILE="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1/train/behavior_interest/21600.parquet"
DATASET_DIR="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1"
SERIAL_ROOT="$AMLT_OUTPUT_DIR/behavior_interest_serial"
mkdir -p "$SERIAL_ROOT"

bootstrap_mapped_checkpoint() {
  if [[ -z "${AMLT_MAP_INPUT_DIR:-}" ]]; then
    return
  fi

  local source_root="$AMLT_MAP_INPUT_DIR/behavior_interest_serial"
  if [[ ! -d "$source_root" ]]; then
    echo "Mapped recovery input is missing: $source_root" >&2
    exit 1
  fi

  # A retried map job can already have a newer checkpoint in its own output.
  # Only seed a fresh output directory from the failed predecessor once.
  local existing_state=""
  existing_state="$(
    (find "$SERIAL_ROOT" -type f \
      \( -path '*/checkpoints/checkpoint-*/trainer_state.json' \
      -o -path '*/final_adapter/adapter_model.safetensors' \) \
      -print -quit 2>/dev/null || true)
  )"
  if [[ -n "$existing_state" ]]; then
    echo "RECOVERY_BOOTSTRAP_SKIPPED existing_state=$existing_state"
    return
  fi

  cp -R "$source_root/." "$SERIAL_ROOT/"

  local latest_checkpoint=""
  latest_checkpoint="$(
    (find "$SERIAL_ROOT/qwen35/sft/checkpoints" -maxdepth 1 -type d \
      -name 'checkpoint-*' 2>/dev/null || true) |
      sort -V |
      tail -n 1
  )"
  if [[ -z "$latest_checkpoint" ]]; then
    echo "Mapped recovery input contains no Qwen checkpoint" >&2
    exit 1
  fi
  for required in \
    trainer_state.json \
    adapter_model.safetensors \
    optimizer.pt \
    scheduler.pt; do
    if [[ ! -s "$latest_checkpoint/$required" ]]; then
      echo "Mapped Qwen checkpoint is incomplete: $latest_checkpoint/$required" >&2
      exit 1
    fi
  done
  echo "RECOVERY_BOOTSTRAP_OK $latest_checkpoint"
}

bootstrap_mapped_checkpoint

for required in \
  "$TRAIN_FILE" \
  "$DATASET_DIR/statistics.json" \
  "$DATASET_DIR/metadata.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Required Behavior+Interest input not found: $required" >&2
    exit 1
  fi
done

# The worker base image does not include PyArrow. Install the Qwen worker
# environment before the dataset gate because that gate reads Parquet.
bash "$AMLT_CODE_DIR/scripts/amlt/evaluation/install_qwen35_worker_deps.sh"

python - "$DATASET_DIR" "$TRAIN_FILE" "$SERIAL_ROOT/dataset_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

dataset_dir = Path(sys.argv[1])
train_file = Path(sys.argv[2])
output = Path(sys.argv[3])
statistics = json.loads((dataset_dir / "statistics.json").read_text())
metadata = json.loads((dataset_dir / "metadata.json").read_text())

expected = {
    "rows": 21600,
    "users": 21600,
    "interests": 108,
    "min_per_interest": 200,
    "max_per_interest": 200,
}
actual = {key: int(statistics["train"][key]) for key in expected}
if actual != expected:
    raise RuntimeError(f"Behavior+Interest dataset mismatch: {actual}")
if int(statistics["pairing"]["train_dev_user_overlap"]) != 0:
    raise RuntimeError("Train/Dev users are not disjoint")
if pq.ParquetFile(train_file).metadata.num_rows != 21600:
    raise RuntimeError("Train Parquet does not contain 21,600 rows")

digest = hashlib.sha256(train_file.read_bytes()).hexdigest()
relative = "train/behavior_interest/21600.parquet"
if digest != metadata["artifact_sha256"][relative]:
    raise RuntimeError("Train Parquet SHA-256 does not match metadata")

audit = {
    "dataset_rule_version": metadata["rule_version"],
    "history_limit": int(metadata["history_limit"]),
    "history_item_format": metadata["history_item_format"],
    "train": actual,
    "train_sha256": digest,
    "train_dev_user_overlap": 0,
}
output.write_text(json.dumps(audit, indent=2) + "\n")
print("BEHAVIOR_INTEREST_DATASET_GATE_OK", digest)
PY

run_training() {
  local model_family="$1"
  local config="$2"
  local output_dir="$SERIAL_ROOT/$model_family/sft"
  local run_name="static-sud-${model_family}-behavior-interest-21600-seed42-full-g4"
  mkdir -p "$output_dir"

  local resume_args=()
  local latest_checkpoint=""
  if [[ ! -f "$output_dir/final_adapter/adapter_model.safetensors" ]]; then
    latest_checkpoint="$(
      (find "$output_dir/checkpoints" -maxdepth 1 -type d \
        -name 'checkpoint-*' 2>/dev/null || true) |
        sort -V |
        tail -n 1
    )"
    if [[ -n "$latest_checkpoint" ]]; then
      resume_args=(--resume-from-checkpoint "$latest_checkpoint")
    fi
    python -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NUM_GPUS" \
      -m new_interest.training.train_sft \
      --config "$config" \
      --train-file "$TRAIN_FILE" \
      --output-dir "$output_dir" \
      --run-name "$run_name" \
      --wandb-mode disabled \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --save-steps "$CHECKPOINT_SAVE_STEPS" \
      "${resume_args[@]}"
  fi

  python - "$output_dir/run_summary.json" "$model_family" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
model_family = sys.argv[2]
summary = json.loads(summary_path.read_text())
checks = {
    "train_samples": int(summary["train_samples"]),
    "world_size": int(summary["world_size"]),
    "effective_global_batch_size": int(summary["effective_global_batch_size"]),
    "expected_optimizer_steps": int(summary["expected_optimizer_steps"]),
    "token_audit_samples": int(summary["token_audit"]["samples"]),
    "over_max_seq_length": int(summary["token_audit"]["over_max_seq_length"]),
}
expected = {
    "train_samples": 21600,
    "world_size": 4,
    "effective_global_batch_size": 32,
    "expected_optimizer_steps": 2025,
    "token_audit_samples": 21600,
    "over_max_seq_length": 0,
}
if checks != expected:
    raise RuntimeError(f"{model_family} training contract mismatch: {checks}")
if float(summary["configured_epochs"]) != 3.0:
    raise RuntimeError(f"{model_family} did not train for three epochs")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError(f"{model_family} has forbidden trainable parameters")
loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(loss):
    raise RuntimeError(f"{model_family} has non-finite train loss: {loss}")
adapter = summary_path.parent / "final_adapter" / "adapter_model.safetensors"
if not adapter.is_file():
    raise RuntimeError(f"{model_family} final adapter is missing")
print("BEHAVIOR_INTEREST_TRAIN_GATE_OK", model_family, loss)
PY
}

echo "SERIAL_PHASE_START qwen35"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    "meta-llama/Llama-3.1-8B-Instruct",
    "config.json",
    token=os.environ["HF_TOKEN"],
)
print("LLAMA31_REMOTE_ACCESS_GATE_OK")
PY
export HF_HOME=/scratch/static-sud-behavior-interest-qwen35-hf-cache
mkdir -p "$HF_HOME"
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("Qwen/Qwen3.5-9B")
print("QWEN35_SNAPSHOT_READY", path)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
run_training \
  qwen35 \
  "$AMLT_CODE_DIR/configs/sft/experiments/static_sud_qwen35_behavior_interest_21600.yaml"
echo "SERIAL_PHASE_COMPLETE qwen35"

echo "SERIAL_PHASE_START llama31"
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
bash "$AMLT_CODE_DIR/scripts/amlt/evaluation/install_training_worker_deps.sh"
export HF_HOME=/scratch/static-sud-behavior-interest-llama31-hf-cache
mkdir -p "$HF_HOME"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
path = snapshot_download(name, token=os.environ["HF_TOKEN"])
AutoConfig.from_pretrained(path, local_files_only=True)
AutoTokenizer.from_pretrained(path, local_files_only=True)
print("LLAMA31_SNAPSHOT_READY", path)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
run_training \
  llama31 \
  "$AMLT_CODE_DIR/configs/sft/experiments/static_sud_llama31_behavior_interest_21600.yaml"
echo "SERIAL_PHASE_COMPLETE llama31"

python - "$SERIAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = {}
for family in ("qwen35", "llama31"):
    summary = json.loads((root / family / "sft/run_summary.json").read_text())
    models[family] = {
        "train_loss": float(summary["train_metrics"]["train_loss"]),
        "optimizer_steps": int(summary["expected_optimizer_steps"]),
        "final_adapter": str(root / family / "sft/final_adapter"),
    }
(root / "serial_summary.json").write_text(
    json.dumps({"status": "complete", "models": models}, indent=2) + "\n"
)
print("BEHAVIOR_INTEREST_SERIAL_TRAINING_COMPLETE")
PY
