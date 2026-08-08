#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="$1"
SIZE="$2"
MODE="${3:-full}"

if [[ "$TASK_TYPE" == "auto" ]]; then
  case "${AMLT_MAP_INPUT_JOB_NAME:?AMLT_MAP_INPUT_JOB_NAME is required}" in
    *interest*)
      TASK_TYPE="interest_only"
      ;;
    *behavior*)
      TASK_TYPE="behavior_only"
      ;;
    *)
      echo "Cannot infer task type from $AMLT_MAP_INPUT_JOB_NAME" >&2
      exit 2
      ;;
  esac
fi

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

case "$SIZE" in
  1k|5k) ;;
  *) echo "Unsupported GRPO size: $SIZE" >&2; exit 2 ;;
esac

export PROJECT_DIR="$AMLT_CODE_DIR"
export VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-$HOME/verl-runtime}"
export PYTHONPATH="$VERL_RUNTIME_DIR:$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="Qwen/Qwen3-8B"
export LORA_ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/sft/final_adapter"
export TRAIN_FILE="$AMLT_CODE_DIR/data/training/grpo_experiments/$TASK_TYPE/$SIZE.parquet"
export OUTPUT_DIR="$AMLT_OUTPUT_DIR/grpo"
export EXPERIMENT_NAME="qwen3-${SHORT_TASK}-sft22k-grpo${SIZE}-cot"
export CHECKPOINT_DIR="/scratch/crossinterest-grpo/$EXPERIMENT_NAME/checkpoints"
export N_GPUS=4
export TRAIN_BATCH_SIZE=32
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=768
export TOTAL_EPOCHS=1
export LR=1e-5
export LORA_RANK=16
export LORA_ALPHA=32
export ROLLOUT_N=16
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY=0.4
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE=1
export SAVE_FREQ=25
export TEST_FREQ=-1
export AGENT_LOOP_WORKERS=8

if [[ "$MODE" == "smoke" ]]; then
  export TRAIN_MAX_SAMPLES=128
  export TOTAL_STEPS=2
  export SAVE_FREQ=2
else
  export TRAIN_MAX_SAMPLES=-1
  export TOTAL_STEPS=null
fi

test -f "$LORA_ADAPTER_PATH/adapter_model.safetensors"
test -f "$TRAIN_FILE"

mkdir -p "$CHECKPOINT_DIR"
bash "$PROJECT_DIR/scripts/amlt/grpo/run_verl_grpo_new_interest.sh"

LATEST_STEP="$(tr -d '[:space:]' < "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
LATEST_ADAPTER_DIR="$CHECKPOINT_DIR/global_step_$LATEST_STEP/actor/lora_adapter"
REMOTE_ADAPTER_DIR="$OUTPUT_DIR/final_adapter"
test -f "$LATEST_ADAPTER_DIR/adapter_model.safetensors"
mkdir -p "$REMOTE_ADAPTER_DIR"
cp -R "$LATEST_ADAPTER_DIR/." "$REMOTE_ADAPTER_DIR/"
printf '%s\n' "$LATEST_STEP" > "$OUTPUT_DIR/final_global_step.txt"
