#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-1k}"
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
    echo "Unsupported preflight size: $SIZE" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/cot_sft/final_adapter"
DATA_PATH="$AMLT_CODE_DIR/data/training/grpo_experiments/$TASK_TYPE/$SIZE.parquet"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/preflight"

for path in "$ADAPTER_PATH/adapter_model.safetensors" "$DATA_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "Required preflight input not found: $path" >&2
    exit 1
  fi
done

python -m new_interest.evaluation.preflight_grpo_rewards \
  --adapter-path "$ADAPTER_PATH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-prompts 128 \
  --num-generations 16 \
  --batch-size 2 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 768
