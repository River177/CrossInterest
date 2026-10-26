#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-1k}"
MODE="${2:-full}"
VARIANT="${3:-cot}"
SOURCE_JOB="${AMLT_MAP_INPUT_JOB_NAME:?AMLT_MAP_INPUT_JOB_NAME is required}"

case "$SOURCE_JOB" in
  *interest*)
    TASK_TYPE="interest_only"
    SHORT_TASK="interest"
    ;;
  *behavior*)
    TASK_TYPE="behavior_only"
    SHORT_TASK="behavior"
    ;;
  *)
    echo "Cannot infer task type from $SOURCE_JOB" >&2
    exit 2
    ;;
esac

case "$SIZE" in
  1k|5k) ;;
  *)
    echo "Unsupported CoT-SFT size: $SIZE" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  cot)
    CONFIG_SUFFIX="cot_${SIZE}"
    TRAIN_ROOT="cot_sft_experiments"
    RUN_VARIANT="cot-sft${SIZE}"
    ;;
  cot_v2)
    CONFIG_SUFFIX="cot_v2_${SIZE}"
    TRAIN_ROOT="cot_sft_v2_experiments"
    RUN_VARIANT="cot-v2-sft${SIZE}"
    ;;
  rehearsal)
    CONFIG_SUFFIX="cot_rehearsal_${SIZE}"
    TRAIN_ROOT="cot_sft_rehearsal_experiments"
    RUN_VARIANT="cot-rehearsal${SIZE}"
    ;;
  *)
    echo "Unsupported CoT-SFT variant: $VARIANT" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke)
    MAX_SAMPLE_ARGS=(--max-samples 32)
    WANDB_MODE="disabled"
    RUN_SUFFIX="smoke32"
    ;;
  full)
    MAX_SAMPLE_ARGS=()
    WANDB_MODE="online"
    RUN_SUFFIX="full"
    ;;
  *)
    echo "Unsupported CoT-SFT mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/qwen3_${SHORT_TASK}_${CONFIG_SUFFIX}.yaml"
TRAIN_FILE="$AMLT_CODE_DIR/data/training/$TRAIN_ROOT/$TASK_TYPE/$SIZE.parquet"
ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/sft/final_adapter"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/cot_sft"
RUN_NAME="qwen3-${SHORT_TASK}-sft22k-${RUN_VARIANT}-${RUN_SUFFIX}"

for path in \
  "$CONFIG" \
  "$TRAIN_FILE" \
  "$ADAPTER_PATH/adapter_model.safetensors"
do
  if [[ ! -f "$path" ]]; then
    echo "Required CoT-SFT input not found: $path" >&2
    exit 1
  fi
done

python -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$TRAIN_FILE" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --wandb-mode "$WANDB_MODE" \
  "${MAX_SAMPLE_ARGS[@]}"
