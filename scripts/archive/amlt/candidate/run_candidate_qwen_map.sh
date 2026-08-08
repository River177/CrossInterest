#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${AMLT_JOB_NAME#*qwen3-}"
SHORT_TASK="${JOB_NAME%%-*}"

case "$SHORT_TASK" in
  behavior)
    CONFIG="configs/candidate/qwen_sft_behavior.json"
    ;;
  interest)
    CONFIG="configs/candidate/qwen_sft_interest.json"
    ;;
  *)
    echo "Cannot parse mapped job name: $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

ADAPTER="$AMLT_MAP_INPUT_DIR/sft/final_adapter"
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Adapter not found: $ADAPTER" >&2
  exit 1
fi

python -m new_interest.evaluation.evaluate_candidate_qwen \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --output-dir "$AMLT_OUTPUT_DIR/evaluation" \
  --wandb-mode online
