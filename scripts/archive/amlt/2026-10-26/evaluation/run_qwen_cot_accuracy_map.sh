#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Evaluation mode is required}"
SHARD_INDEX="${2:-0}"
SHARD_COUNT="${3:-1}"
RUN_KIND="${4:-full}"
SOURCE_JOB="${AMLT_MAP_INPUT_JOB_NAME:?AMLT_MAP_INPUT_JOB_NAME is required}"

case "$MODE" in
  cot|no_cot) ;;
  *)
    echo "Unsupported accuracy mode: $MODE" >&2
    exit 2
    ;;
esac

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

if [[ -f "$AMLT_MAP_INPUT_DIR/cot_sft/final_adapter/adapter_model.safetensors" ]]; then
  ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/cot_sft/final_adapter"
  case "$SOURCE_JOB" in
    *rehearsal*5k*) CONDITION="cot-rehearsal5k" ;;
    *rehearsal*) CONDITION="cot-rehearsal1k" ;;
    *5k*) CONDITION="cot-sft5k" ;;
    *) CONDITION="cot-sft1k" ;;
  esac
elif [[ -f "$AMLT_MAP_INPUT_DIR/grpo/final_adapter/adapter_model.safetensors" ]]; then
  ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/grpo/final_adapter"
  case "$SOURCE_JOB" in
    *correctness*5k*) CONDITION="new-grpo5k" ;;
    *correctness*) CONDITION="new-grpo1k" ;;
    *) CONDITION="old-grpo1k" ;;
  esac
elif [[ -f "$AMLT_MAP_INPUT_DIR/sft/final_adapter/adapter_model.safetensors" ]]; then
  ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/sft/final_adapter"
  CONDITION="sft22k"
else
  echo "No supported adapter found in $AMLT_MAP_INPUT_DIR" >&2
  exit 1
fi

if [[ -n "${COT_INSTRUCTION_VERSION_OVERRIDE:-}" ]]; then
  COT_INSTRUCTION_VERSION="$COT_INSTRUCTION_VERSION_OVERRIDE"
elif [[ "$CONDITION" == "old-grpo1k" && "$MODE" == "cot" ]]; then
  COT_INSTRUCTION_VERSION="legacy"
else
  COT_INSTRUCTION_VERSION="current"
fi

case "$COT_INSTRUCTION_VERSION" in
  current|legacy|native) ;;
  *)
    echo "Unsupported CoT instruction version: $COT_INSTRUCTION_VERSION" >&2
    exit 2
    ;;
esac

case "$RUN_KIND" in
  smoke)
    RANGE_ARGS=(--start-index 0 --max-examples 128)
    WANDB_MODE="disabled"
    RUN_SUFFIX="smoke128"
    ;;
  full)
    if (( SHARD_COUNT <= 0 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
      echo "Invalid shard specification: $SHARD_INDEX/$SHARD_COUNT" >&2
      exit 2
    fi
    if (( 10000 % SHARD_COUNT != 0 )); then
      echo "Dev10K is not divisible by shard count $SHARD_COUNT" >&2
      exit 2
    fi
    SHARD_SIZE=$((10000 / SHARD_COUNT))
    START_INDEX=$((SHARD_INDEX * SHARD_SIZE))
    RANGE_ARGS=(
      --start-index "$START_INDEX"
      --max-examples "$SHARD_SIZE"
    )
    WANDB_MODE="online"
    RUN_SUFFIX="dev10k-s${SHARD_INDEX}of${SHARD_COUNT}"
    ;;
  *)
    echo "Unsupported run kind: $RUN_KIND" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_FILE="$AMLT_CODE_DIR/data/training/sft_experiments/eval/$TASK_TYPE/dev_10k_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_balanced_200_h50/retained_interests.parquet"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/accuracy/$MODE"
RUN_NAME="qwen3-${CONDITION}-${SHORT_TASK}-${MODE}-${RUN_SUFFIX}"

for path in \
  "$ADAPTER_PATH/adapter_model.safetensors" \
  "$EVAL_FILE" \
  "$LABELS_FILE"
do
  if [[ ! -f "$path" ]]; then
    echo "Required accuracy input not found: $path" >&2
    exit 1
  fi
done

python -m new_interest.evaluation.evaluate_cot_accuracy \
  --adapter-path "$ADAPTER_PATH" \
  --eval-file "$EVAL_FILE" \
  --retained-interests-file "$LABELS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --cot-instruction-version "$COT_INSTRUCTION_VERSION" \
  --run-name "$RUN_NAME" \
  --batch-size 4 \
  --checkpoint-size 64 \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project cross-interest-cot-accuracy \
  "${RANGE_ARGS[@]}"
