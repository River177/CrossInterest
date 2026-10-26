#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: run_qwen_cot_sft_v2.sh <behavior|interest> <smoke|full>}"
MODE="${2:?Usage: run_qwen_cot_sft_v2.sh <behavior|interest> <smoke|full>}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/CrossInterest}"
SFT_ENV="${SFT_ENV:-$HOME/venvs/cross-interest-sft}"

case "$TASK" in
  behavior)
    TASK_TYPE="behavior_only"
    ADAPTER="$PROJECT_DIR/artifacts/sft/qwen3-behavior-22k/final_adapter"
    CONFIG="$PROJECT_DIR/configs/sft/experiments/qwen3_behavior_cot_v2_1k.yaml"
    ;;
  interest)
    TASK_TYPE="interest_only"
    ADAPTER="$PROJECT_DIR/artifacts/sft/qwen3-interest-22k/final_adapter"
    CONFIG="$PROJECT_DIR/configs/sft/experiments/qwen3_interest_cot_v2_1k.yaml"
    ;;
  *)
    echo "Unsupported task: $TASK" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke)
    MAX_SAMPLE_ARGS=(--max-samples 32 --epochs 10)
    WANDB_MODE=disabled
    RUN_SUFFIX=smoke32
    ;;
  full)
    MAX_SAMPLE_ARGS=()
    WANDB_MODE=offline
    RUN_SUFFIX=full
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

source "$SFT_ENV/bin/activate"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="$PROJECT_DIR/artifacts/hf-datasets-cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_FILE="$PROJECT_DIR/data/training/cot_sft_v2_experiments/$TASK_TYPE/1k.parquet"
OUTPUT_DIR="$PROJECT_DIR/artifacts/sft/qwen3-$TASK-cot-v2-$MODE"
RUN_NAME="qwen3-$TASK-sft22k-cot-v2-sft1k-$RUN_SUFFIX"

test -f "$CONFIG"
test -f "$TRAIN_FILE"
test -f "$ADAPTER/adapter_model.safetensors"
mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE"

python -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$TRAIN_FILE" \
  --adapter-path "$ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --wandb-mode "$WANDB_MODE" \
  "${MAX_SAMPLE_ARGS[@]}"
