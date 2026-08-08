#!/usr/bin/env bash
set -euo pipefail

SHORT_TASK="$1"
case "$SHORT_TASK" in
  behavior)
    CONFIG="configs/candidate/qwen_base_behavior.json"
    ;;
  interest)
    CONFIG="configs/candidate/qwen_base_interest.json"
    ;;
  *)
    echo "Unsupported task: $SHORT_TASK" >&2
    exit 2
    ;;
esac

python -m new_interest.evaluate_candidate_qwen \
  --config "$CONFIG" \
  --output-dir "$AMLT_OUTPUT_DIR/evaluation" \
  --wandb-mode online
