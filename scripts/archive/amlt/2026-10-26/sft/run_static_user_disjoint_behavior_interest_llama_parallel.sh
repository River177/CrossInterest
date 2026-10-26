#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${STATIC_SUD_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Behavior+Interest Llama full training and evaluation requires exactly 4 GPUs" >&2
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

DATASET_DIR="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1"
TRAIN_FILE="$DATASET_DIR/train/behavior_interest/21600.parquet"
EVAL_FILE="$DATASET_DIR/eval/behavior_interest/dev_full_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
TRAIN_CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_llama31_behavior_interest_21600.yaml"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_llama31_behavior_interest_full.json"
PARALLEL_ROOT="$AMLT_OUTPUT_DIR/behavior_interest_parallel/llama31"
SFT_DIR="$PARALLEL_ROOT/sft"
BEAM_ROOT="$PARALLEL_ROOT/beam"
RUN_NAME="static-sud-llama31-behavior-interest-21600-seed42-parallel-full-g4"
mkdir -p "$SFT_DIR" "$BEAM_ROOT"

for required in \
  "$TRAIN_FILE" \
  "$EVAL_FILE" \
  "$LABELS_FILE" \
  "$TRAIN_CONFIG" \
  "$BEAM_CONFIG" \
  "$DATASET_DIR/statistics.json" \
  "$DATASET_DIR/metadata.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Required Behavior+Interest Llama input not found: $required" >&2
    exit 1
  fi
done

python - "$DATASET_DIR" "$TRAIN_FILE" "$EVAL_FILE" "$PARALLEL_ROOT/dataset_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

dataset_dir = Path(sys.argv[1])
train_file = Path(sys.argv[2])
eval_file = Path(sys.argv[3])
output = Path(sys.argv[4])
statistics = json.loads((dataset_dir / "statistics.json").read_text())
metadata = json.loads((dataset_dir / "metadata.json").read_text())

expected_train = {
    "rows": 21600,
    "users": 21600,
    "interests": 108,
    "min_per_interest": 200,
    "max_per_interest": 200,
}
actual_train = {key: int(statistics["train"][key]) for key in expected_train}
if actual_train != expected_train:
    raise RuntimeError(f"Behavior+Interest train mismatch: {actual_train}")
expected_dev = {
    "impressions": 31392,
    "users": 27082,
    "target_occurrences": 37728,
    "supported_interests": 104,
}
actual_dev = {key: int(statistics["dev"][key]) for key in expected_dev}
if actual_dev != expected_dev:
    raise RuntimeError(f"Behavior+Interest Dev mismatch: {actual_dev}")
if int(statistics["pairing"]["train_dev_user_overlap"]) != 0:
    raise RuntimeError("Train/Dev users are not disjoint")
if pq.ParquetFile(train_file).metadata.num_rows != 21600:
    raise RuntimeError("Train Parquet does not contain 21,600 rows")
if pq.ParquetFile(eval_file).metadata.num_rows != 31392:
    raise RuntimeError("Dev Parquet does not contain 31,392 rows")

train_digest = hashlib.sha256(train_file.read_bytes()).hexdigest()
eval_digest = hashlib.sha256(eval_file.read_bytes()).hexdigest()
if train_digest != metadata["artifact_sha256"]["train/behavior_interest/21600.parquet"]:
    raise RuntimeError("Train Parquet SHA-256 does not match metadata")
if eval_digest != metadata["artifact_sha256"]["eval/behavior_interest/dev_full_grouped.parquet"]:
    raise RuntimeError("Dev Parquet SHA-256 does not match metadata")

audit = {
    "dataset_rule_version": metadata["rule_version"],
    "history_limit": int(metadata["history_limit"]),
    "history_item_format": metadata["history_item_format"],
    "train": actual_train,
    "dev": actual_dev,
    "train_sha256": train_digest,
    "eval_sha256": eval_digest,
    "train_dev_user_overlap": 0,
}
output.write_text(json.dumps(audit, indent=2) + "\n")
print("BEHAVIOR_INTEREST_LLAMA_DATASET_GATE_OK", train_digest, eval_digest)
PY

: "${HF_HOME:=/mnt/default/huggingface-cache}"
export HF_HOME
mkdir -p "$HF_HOME"
python - <<'PY'
import os

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
snapshot = snapshot_download(name, token=os.environ["HF_TOKEN"])
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
print(
    "LLAMA31_PARALLEL_SNAPSHOT_READY",
    snapshot,
    type(config).__name__,
    type(tokenizer).__name__,
)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "$SFT_DIR/final_adapter/adapter_model.safetensors" ]]; then
  RESUME_ARGS=()
  LATEST_CHECKPOINT="$(
    (find "$SFT_DIR/checkpoints" -maxdepth 1 -type d \
      -name 'checkpoint-*' 2>/dev/null || true) |
      sort -V |
      tail -n 1
  )"
  if [[ -n "$LATEST_CHECKPOINT" ]]; then
    for required in trainer_state.json adapter_model.safetensors optimizer.pt scheduler.pt; do
      if [[ ! -s "$LATEST_CHECKPOINT/$required" ]]; then
        echo "Incomplete Llama resume checkpoint: $LATEST_CHECKPOINT/$required" >&2
        exit 1
      fi
    done
    RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
    echo "BEHAVIOR_INTEREST_LLAMA_RESUME_CHECKPOINT_READY $LATEST_CHECKPOINT"
  fi
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    --config "$TRAIN_CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$SFT_DIR" \
    --run-name "$RUN_NAME" \
    --wandb-mode disabled \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --save-steps "$CHECKPOINT_SAVE_STEPS" \
    "${RESUME_ARGS[@]}"
fi

python - "$SFT_DIR/run_summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
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
    raise RuntimeError(f"Llama training contract mismatch: {checks}")
if float(summary["configured_epochs"]) != 3.0:
    raise RuntimeError("Llama did not train for exactly three epochs")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("Llama did not use non-reentrant gradient checkpointing")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Llama has forbidden trainable parameters")
loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(loss):
    raise RuntimeError(f"Llama has non-finite train loss: {loss}")
adapter = summary_path.parent / "final_adapter" / "adapter_model.safetensors"
if not adapter.is_file() or adapter.stat().st_size <= 0:
    raise RuntimeError("Llama final adapter is missing or empty")
print("BEHAVIOR_INTEREST_LLAMA_PARALLEL_TRAIN_GATE_OK", loss)
PY

PIDS=()
SHARD_DIRS=()
for SHARD_INDEX in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"
  SHARD_DIR="$BEAM_ROOT/$SHARD_NAME"
  SHARD_DIRS+=("$SHARD_DIR")
  (
    export CUDA_VISIBLE_DEVICES="$SHARD_INDEX"
    python -m new_interest.evaluation.evaluate_beam \
      --config "$BEAM_CONFIG" \
      --adapter-path "$SFT_DIR/final_adapter" \
      --output-dir "$SHARD_DIR" \
      --run-name "${RUN_NAME}-beam-${SHARD_NAME}" \
      --num-shards "$NUM_GPUS" \
      --shard-index "$SHARD_INDEX" \
      --wandb-mode disabled \
      >"$BEAM_ROOT/${SHARD_NAME}.log" 2>&1
  ) &
  PIDS+=("$!")
done

EVAL_FAILURE=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    EVAL_FAILURE=1
  fi
done
if [[ "$EVAL_FAILURE" -ne 0 ]]; then
  echo "At least one Llama Behavior+Interest Beam shard failed" >&2
  exit 1
fi

python -m new_interest.evaluation.merge_beam_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "$EVAL_FILE" \
  --retained-interests-file "$LABELS_FILE" \
  --output-dir "$BEAM_ROOT/merged" \
  --expected-records 31392

python - "$PARALLEL_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_summary = json.loads((root / "sft/run_summary.json").read_text())
metrics = json.loads((root / "beam/merged/metrics.json").read_text())
if int(metrics["samples"]) != 31392:
    raise RuntimeError("Merged Beam prediction count is not 31,392")
if int(metrics["supported_interest_count"]) != 104:
    raise RuntimeError("Interest Macro denominator is not 104")
if int(metrics["label_vocabulary_size"]) != 108:
    raise RuntimeError("Constrained vocabulary size is not 108")
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Non-finite Beam metric: {key}={metrics[key]}")

summary = {
    "status": "complete",
    "train_loss": float(train_summary["train_metrics"]["train_loss"]),
    "optimizer_steps": int(train_summary["expected_optimizer_steps"]),
    "final_adapter": str(root / "sft/final_adapter"),
    "beam_metrics": str(root / "beam/merged/metrics.json"),
    "interest_macro_recall_at_10": float(metrics["interest_macro_recall_at_10"]),
    "interest_macro_ndcg_at_10": float(metrics["interest_macro_ndcg_at_10"]),
}
(root / "parallel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    "BEHAVIOR_INTEREST_LLAMA_FULL_EVAL_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
print("BEHAVIOR_INTEREST_LLAMA_PARALLEL_COMPLETE")
PY
