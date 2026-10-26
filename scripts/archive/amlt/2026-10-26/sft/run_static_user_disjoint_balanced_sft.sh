#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${1:?Usage: run_static_user_disjoint_balanced_sft.sh <qwen35|llama31> <smoke|ddp-smoke|full>}"
MODE="${2:?Usage: run_static_user_disjoint_balanced_sft.sh <qwen35|llama31> <smoke|ddp-smoke|full|eval-smoke|eval-only>}"

NUM_GPUS="${STATIC_SUD_NUM_GPUS:-8}"
case "$NUM_GPUS" in
  1|2|4|8|16|32) ;;
  *)
    echo "STATIC_SUD_NUM_GPUS must be a positive divisor of 32; got $NUM_GPUS" >&2
    exit 2
    ;;
esac
DISTRIBUTED_GRAD_ACCUM=$((32 / NUM_GPUS))

case "$AMLT_JOB_NAME" in
  *behavior*)
    TASK_TYPE="behavior_only"
    SHORT_TASK="behavior"
    ;;
  *interest*)
    TASK_TYPE="interest_only"
    SHORT_TASK="interest"
    ;;
  *)
    echo "Cannot infer task type from AMLT job name: $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

case "$MODEL_FAMILY" in
  qwen35)
    MODEL_NAME="Qwen/Qwen3.5-9B"
    SHORT_MODEL="qwen35"
    ;;
  llama31)
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    SHORT_MODEL="llama31"
    ;;
  *)
    echo "Unsupported model family: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke|ddp-smoke|full|eval-smoke|eval-only) ;;
  *)
    echo "Unsupported run mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true

prepare_qwen35_offline() {
  export HF_HOME=/scratch/static-sud-qwen35-hf-cache
  mkdir -p "$HF_HOME"
  python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("Qwen/Qwen3.5-9B")
print("QWEN35_SNAPSHOT_READY", path)
PY
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

prepare_llama31_offline() {
  export HF_HOME=/scratch/static-sud-llama31-hf-cache
  mkdir -p "$HF_HOME"
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for authenticated Llama snapshot download" >&2
    exit 1
  fi
  python - <<'PY'
import os

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
snapshot = snapshot_download(name, token=os.environ["HF_TOKEN"])
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
print(
    "LLAMA31_SNAPSHOT_READY",
    snapshot,
    type(config).__name__,
    type(tokenizer).__name__,
)
PY
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

if [[ "$MODEL_FAMILY" == "qwen35" ]]; then
  prepare_qwen35_offline
else
  prepare_llama31_offline
fi

CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_${SHORT_MODEL}_${SHORT_TASK}_21600.yaml"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_${SHORT_MODEL}_${SHORT_TASK}_full.json"
TRAIN_FILE="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_v1/train/$TASK_TYPE/21600.parquet"
EVAL_FILE="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_v1/eval/$TASK_TYPE/dev_full_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
SFT_DIR="${STATIC_SUD_SFT_DIR:-$AMLT_OUTPUT_DIR/sft}"
BEAM_ROOT="$AMLT_OUTPUT_DIR/beam"
RUN_NAME="static-sud-${SHORT_MODEL}-${SHORT_TASK}-21600-seed42-${MODE}-g${NUM_GPUS}"

for path in "$CONFIG" "$BEAM_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$LABELS_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "Required experiment input not found: $path" >&2
    exit 1
  fi
done

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_COMMAND=(python -m new_interest.training.train_sft)
  TRAIN_ARGS=(
    --max-samples 32
    --epochs 20
    --per-device-train-batch-size 4
    --gradient-accumulation-steps 8
  )
elif [[ "$MODE" == "ddp-smoke" ]]; then
  TRAIN_COMMAND=(
    python -m torch.distributed.run
    --standalone
    --nproc_per_node="$NUM_GPUS"
    -m new_interest.training.train_sft
  )
  TRAIN_ARGS=(
    --max-samples 32
    --epochs 2
    --per-device-train-batch-size 1
    --gradient-accumulation-steps "$DISTRIBUTED_GRAD_ACCUM"
  )
else
  TRAIN_COMMAND=(
    python -m torch.distributed.run
    --standalone
    --nproc_per_node="$NUM_GPUS"
    -m new_interest.training.train_sft
  )
  TRAIN_ARGS=(
    --per-device-train-batch-size 1
    --gradient-accumulation-steps "$DISTRIBUTED_GRAD_ACCUM"
  )
fi

if [[ "$MODE" != eval-* && ! -f "$SFT_DIR/final_adapter/adapter_model.safetensors" ]]; then
  RESUME_ARGS=()
  LATEST_CHECKPOINT="$(
    (find "$SFT_DIR/checkpoints" -maxdepth 1 -type d \
      -name 'checkpoint-*' 2>/dev/null || true) |
      sort -V |
      tail -n 1
  )"
  if [[ -n "$LATEST_CHECKPOINT" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
  fi
  "${TRAIN_COMMAND[@]}" \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$SFT_DIR" \
    --run-name "$RUN_NAME" \
    --wandb-mode disabled \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
fi

if [[ "$MODE" != eval-* ]]; then
  python - "$SFT_DIR/run_summary.json" "$MODE" "$NUM_GPUS" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
expected_world_size = int(sys.argv[3])
if int(summary["token_audit"]["samples"]) != 21600:
    raise RuntimeError("Token audit did not cover all 21,600 rows")
if int(summary["token_audit"]["over_max_seq_length"]) != 0:
    raise RuntimeError("Token audit found over-length examples")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Forbidden visual parameters are trainable")
train_loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(train_loss):
    raise RuntimeError(f"Non-finite train loss: {train_loss}")
if mode in {"smoke", "ddp-smoke"}:
    if int(summary["train_samples"]) != 32:
        raise RuntimeError("Smoke did not train exactly 32 samples")
    if mode == "smoke":
        history = summary["loss_history"]
        if len(history) < 2 or not float(history[-1]["loss"]) < float(history[0]["loss"]):
            raise RuntimeError(f"Smoke loss did not decrease: {history}")
    else:
        if int(summary["world_size"]) != expected_world_size:
            raise RuntimeError(
                "DDP Smoke world size mismatch: "
                f"expected {expected_world_size}, got {summary['world_size']}"
            )
        if int(summary["effective_global_batch_size"]) != 32:
            raise RuntimeError("DDP Smoke effective global batch is not 32")
        if int(summary["expected_optimizer_steps"]) != 2:
            raise RuntimeError("DDP Smoke did not run two optimizer steps")
        if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
            raise RuntimeError("DDP Smoke did not use non-reentrant checkpointing")
else:
    if int(summary["world_size"]) != expected_world_size:
        raise RuntimeError(
            "Full world size mismatch: "
            f"expected {expected_world_size}, got {summary['world_size']}"
        )
    if int(summary["train_samples"]) != 21600:
        raise RuntimeError("Full run did not train all 21,600 samples")
    if float(summary["configured_epochs"]) != 3.0:
        raise RuntimeError("Full run did not use exactly 3 epochs")
    if int(summary["effective_global_batch_size"]) != 32:
        raise RuntimeError("Full effective global batch is not 32")
    if int(summary["expected_optimizer_steps"]) != 2025:
        raise RuntimeError("Full expected optimizer steps are not 2,025")
print("STATIC_SUD_TRAIN_GATE_OK", mode, train_loss)
PY
else
  if [[ ! -f "$SFT_DIR/final_adapter/adapter_model.safetensors" ]]; then
    echo "Mapped final adapter not found: $SFT_DIR/final_adapter" >&2
    exit 1
  fi
  echo "STATIC_SUD_EVAL_ONLY_ADAPTER_OK $SFT_DIR/final_adapter"
fi

if [[ "$MODE" == "ddp-smoke" ]]; then
  exit 0
fi

if [[ "$MODE" == "smoke" || "$MODE" == "eval-smoke" ]]; then
  python -m new_interest.evaluation.evaluate_beam \
    --config "$BEAM_CONFIG" \
    --adapter-path "$SFT_DIR/final_adapter" \
    --output-dir "$BEAM_ROOT/smoke" \
    --run-name "${RUN_NAME}-beam32" \
    --max-examples 32 \
    --wandb-mode disabled
  python - "$BEAM_ROOT/smoke/metrics.json" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
if int(metrics["samples"]) != 32:
    raise RuntimeError("Smoke Beam did not produce exactly 32 predictions")
if int(metrics["top_k"]) != 20 or int(metrics["num_beams"]) != 32:
    raise RuntimeError("Smoke Beam settings do not match the protocol")
if mode == "eval-smoke" and int(metrics["dedup_retry_example_count"]) < 1:
    raise RuntimeError(
        "Beam v2 smoke did not exercise the unique-label retry path"
    )
print("STATIC_SUD_BEAM_SMOKE_GATE_OK", metrics["samples"])
PY
  exit 0
fi

mkdir -p "$BEAM_ROOT"
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
  echo "At least one Beam shard failed; inspect $BEAM_ROOT/shard-*.log" >&2
  exit 1
fi

python -m new_interest.evaluation.merge_beam_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "$EVAL_FILE" \
  --retained-interests-file "$LABELS_FILE" \
  --output-dir "$BEAM_ROOT/merged" \
  --expected-records 31392

python - "$BEAM_ROOT/merged/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(metrics["samples"]) != 31392:
    raise RuntimeError("Merged Beam prediction count is not 31,392")
if int(metrics["supported_interest_count"]) != 104:
    raise RuntimeError("Interest Macro denominator is not 104")
if int(metrics["label_vocabulary_size"]) != 108:
    raise RuntimeError("Constrained vocabulary size is not 108")
print(
    "STATIC_SUD_FULL_EVAL_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY
