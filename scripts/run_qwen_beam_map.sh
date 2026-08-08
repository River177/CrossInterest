#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${AMLT_JOB_NAME#*qwen3-}"
SHORT_TASK="${JOB_NAME%-*}"
SIZE="${JOB_NAME##*-}"

case "$SHORT_TASK" in
  interest)
    ;;
  behavior)
    ;;
  *)
    echo "Cannot parse mapped job name: $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

BEAM_CONFIG="configs/beam/qwen3_${SHORT_TASK}_${SIZE}.json"
ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/sft/final_adapter"

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  echo "Adapter not found: $ADAPTER_PATH" >&2
  exit 1
fi

python -m new_interest.evaluate_beam \
  --config "$BEAM_CONFIG" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$AMLT_OUTPUT_DIR/beam" \
  --run-name "qwen3-8b-${SHORT_TASK}-${SIZE}-seed42-beam10k-recovery" \
  --wandb-mode online
