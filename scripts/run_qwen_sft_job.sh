#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="$1"
SIZE="$2"

case "$TASK_TYPE" in
  interest_only)
    SHORT_TASK="interest"
    ;;
  behavior_only)
    SHORT_TASK="behavior"
    ;;
  *)
    echo "Unsupported task type: $TASK_TYPE" >&2
    exit 2
    ;;
esac

RUN_NAME="qwen3-8b-${SHORT_TASK}-${SIZE}-seed42"
TRAIN_FILE="data/training/sft_experiments/train/${TASK_TYPE}/${SIZE}.parquet"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
BEAM_CONFIG="configs/beam/qwen3_${SHORT_TASK}_${SIZE}.json"

if [[ ! -f "$SFT_DIR/final_adapter/adapter_model.safetensors" ]]; then
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

  python -m new_interest.train_sft \
    --config configs/sft/qwen3_8b_lora.yaml \
    --train-file "$TRAIN_FILE" \
    --output-dir "$SFT_DIR" \
    --run-name "$RUN_NAME" \
    --wandb-mode online \
    "${RESUME_ARGS[@]}"
fi

python -m new_interest.evaluate_beam \
  --config "$BEAM_CONFIG" \
  --adapter-path "$SFT_DIR/final_adapter" \
  --output-dir "$AMLT_OUTPUT_DIR/beam" \
  --run-name "${RUN_NAME}-beam10k" \
  --wandb-mode online

rm -rf "$SFT_DIR/checkpoints"
