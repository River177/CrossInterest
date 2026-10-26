#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: run_qwen_format_bootstrap.sh <behavior|interest> <smoke|20step>}"
MODE="${2:?Usage: run_qwen_format_bootstrap.sh <behavior|interest> <smoke|20step>}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/CrossInterest}"
VERL_ENV="${VERL_ENV:-$HOME/venvs/cross-interest-verl}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-$HOME/verl-runtime-format-bootstrap}"

case "$TASK" in
  behavior)
    TASK_TYPE="behavior_only"
    ADAPTER="$PROJECT_DIR/artifacts/sft/qwen3-behavior-22k/final_adapter"
    ;;
  interest)
    TASK_TYPE="interest_only"
    ADAPTER="$PROJECT_DIR/artifacts/sft/qwen3-interest-22k/final_adapter"
    ;;
  *)
    echo "Unsupported task: $TASK" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke)
    TOTAL_STEPS=2
    SAVE_FREQ=2
    ;;
  20step)
    TOTAL_STEPS=20
    TOTAL_EPOCHS=2
    SAVE_FREQ=5
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

source "$VERL_ENV/bin/activate"
cd "$PROJECT_DIR"

export PROJECT_DIR
export VERL_RUNTIME_DIR
export PYTHONPATH="$VERL_RUNTIME_DIR:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="$PROJECT_DIR/artifacts/hf-datasets-cache"
export TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAINER_LOGGER=console

source "$PROJECT_DIR/scripts/amlt/grpo/prepare_writable_verl.sh"

RUN_NAME="qwen3-${TASK}-format-bootstrap-strict-${MODE}"
OUTPUT_DIR="$PROJECT_DIR/artifacts/grpo-format-bootstrap/$RUN_NAME"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

export MODEL_PATH="Qwen/Qwen3-8B"
export LORA_ADAPTER_PATH="$ADAPTER"
export TRAIN_FILE="$PROJECT_DIR/data/training/grpo_format_bootstrap_mixed_experiments/$TASK_TYPE/preflight_mixed.parquet"
export REWARD_FUNCTION_PATH="$PROJECT_DIR/src/new_interest/training/grpo_format_bootstrap_reward.py"
export OUTPUT_DIR
export CHECKPOINT_DIR
export EXPERIMENT_NAME="$RUN_NAME"
export N_GPUS=1
export TRAIN_MAX_SAMPLES=128
export TRAIN_BATCH_SIZE=2
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=768
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_STEPS
export LR=2e-6
export LORA_RANK=16
export LORA_ALPHA=32
export ROLLOUT_N=16
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEMORY=0.35
export PPO_MINI_BATCH_SIZE=2
export PPO_MICRO_BATCH_SIZE=1
export SAVE_FREQ
export TEST_FREQ=-1
export AGENT_LOOP_WORKERS=2

test -f "$LORA_ADAPTER_PATH/adapter_model.safetensors"
test -f "$TRAIN_FILE"
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$HF_DATASETS_CACHE"

bash "$PROJECT_DIR/scripts/amlt/grpo/run_verl_grpo_new_interest.sh"

LATEST_STEP="$(
  tr -d '[:space:]' < "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
)"
LATEST_ADAPTER_DIR="$CHECKPOINT_DIR/global_step_$LATEST_STEP/actor/lora_adapter"
FINAL_ADAPTER_DIR="$OUTPUT_DIR/final_adapter"
test -f "$LATEST_ADAPTER_DIR/adapter_model.safetensors"
mkdir -p "$FINAL_ADAPTER_DIR"
cp -R "$LATEST_ADAPTER_DIR/." "$FINAL_ADAPTER_DIR/"
printf '%s\n' "$LATEST_STEP" > "$OUTPUT_DIR/final_global_step.txt"
