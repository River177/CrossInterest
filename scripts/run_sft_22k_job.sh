#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="$1"
TASK_TYPE="$2"

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

case "$MODEL_FAMILY" in
  qwen3)
    MODEL_NAME="Qwen/Qwen3-8B"
    CONFIG="configs/sft/experiments/qwen3_${SHORT_TASK}_22k.yaml"
    BEAM_CONFIG="configs/beam/qwen3_${SHORT_TASK}_22k.json"
    RUN_NAME="qwen3-8b-${SHORT_TASK}-22k-seed42"
    ;;
  llama31)
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    CONFIG="configs/sft/experiments/llama31_${SHORT_TASK}_22k.yaml"
    BEAM_CONFIG="configs/beam/llama31_${SHORT_TASK}_22k.json"
    RUN_NAME="llama31-8b-${SHORT_TASK}-22k-seed42"
    ;;
  *)
    echo "Unsupported model family: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac

TRAIN_FILE="data/training/sft_experiments/train/${TASK_TYPE}/22k.parquet"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
BEAM_DIR="$AMLT_OUTPUT_DIR/beam"

python - <<PY
from transformers import AutoConfig, AutoTokenizer
name = "$MODEL_NAME"
AutoConfig.from_pretrained(name)
AutoTokenizer.from_pretrained(name)
print("MODEL_ACCESS_OK", name)
PY

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
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$SFT_DIR" \
    --run-name "$RUN_NAME" \
    --wandb-mode online \
    "${RESUME_ARGS[@]}"
fi

python -m new_interest.evaluate_beam \
  --config "$BEAM_CONFIG" \
  --model-name "$MODEL_NAME" \
  --adapter-path "$SFT_DIR/final_adapter" \
  --output-dir "$BEAM_DIR" \
  --run-name "${RUN_NAME}-beam10k" \
  --wandb-mode online

rm -rf "$SFT_DIR/checkpoints"
