#!/usr/bin/env bash
set -euo pipefail

SOURCE_JOB="${AMLT_MAP_INPUT_JOB_NAME:?AMLT_MAP_INPUT_JOB_NAME is required}"

case "$SOURCE_JOB" in
  *interest*)
    SHORT_TASK="interest"
    ;;
  *behavior*)
    SHORT_TASK="behavior"
    ;;
  *)
    echo "Cannot infer task type from mapped job: $SOURCE_JOB" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="$AMLT_CODE_DIR/configs/beam/qwen35_${SHORT_TASK}_22k.json"
ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/sft/final_adapter"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/beam"
RUN_NAME="qwen35-9b-${SHORT_TASK}-22k-seed42-beam10k"

for path in "$CONFIG" "$ADAPTER_PATH/adapter_model.safetensors"; do
  if [[ ! -f "$path" ]]; then
    echo "Required mapped Beam input not found: $path" >&2
    exit 1
  fi
done

python -m new_interest.evaluation.evaluate_beam \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --wandb-mode online
