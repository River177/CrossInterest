#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SHARD_INDEX="${2:-0}"
SHARD_COUNT="${3:-1}"
SOURCE_JOB="${AMLT_MAP_INPUT_JOB_NAME:?AMLT_MAP_INPUT_JOB_NAME is required}"
: "${CANDIDATE_BATCH_SIZE:=22}"

export PROJECT_DIR="$AMLT_CODE_DIR"
export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "CoT label map: source=$SOURCE_JOB mode=$MODE code=$AMLT_CODE_DIR"
python -c "import new_interest.evaluation.evaluate_cot_label_ranking; print('CoT label map: module import verified', flush=True)"

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

if [[ -f "$AMLT_MAP_INPUT_DIR/grpo/final_adapter/adapter_model.safetensors" ]]; then
  CONDITION="grpo1k"
  ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/grpo/final_adapter"
elif [[ -f "$AMLT_MAP_INPUT_DIR/sft/final_adapter/adapter_model.safetensors" ]]; then
  CONDITION="sft22k"
  ADAPTER_PATH="$AMLT_MAP_INPUT_DIR/sft/final_adapter"
else
  echo "No SFT or GRPO adapter found in $AMLT_MAP_INPUT_DIR" >&2
  exit 1
fi

case "$MODE" in
  smoke)
    MAX_EXAMPLES_ARGS=(--max-examples 128)
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
    MAX_EXAMPLES_ARGS=(
      --start-index "$START_INDEX"
      --max-examples "$SHARD_SIZE"
    )
    WANDB_MODE="online"
    RUN_SUFFIX="dev10k-s${SHARD_INDEX}of${SHARD_COUNT}"
    ;;
  *)
    echo "Unsupported evaluation mode: $MODE" >&2
    exit 2
    ;;
esac

EVAL_FILE="$AMLT_CODE_DIR/data/training/sft_experiments/eval/$TASK_TYPE/dev_10k_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_balanced_200_h50/retained_interests.parquet"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/cot_label_eval"
RUN_NAME="qwen3-${CONDITION}-${SHORT_TASK}-${RUN_SUFFIX}-cot-label-ranking"

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "Evaluation file not found: $EVAL_FILE" >&2
  exit 1
fi
if [[ ! -f "$LABELS_FILE" ]]; then
  echo "Label vocabulary not found: $LABELS_FILE" >&2
  exit 1
fi

echo "CoT label map: adapter, Dev10K, and labels verified"

python -m new_interest.evaluation.evaluate_cot_label_ranking \
  --model-name Qwen/Qwen3-8B \
  --adapter-path "$ADAPTER_PATH" \
  --eval-file "$EVAL_FILE" \
  --retained-interests-file "$LABELS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --condition "$CONDITION" \
  --task-type "$TASK_TYPE" \
  --batch-size 4 \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --max-new-tokens 768 \
  --checkpoint-size 16 \
  --cutoffs 10 20 50 \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project cross-interest-generative-grpo-eval \
  "${MAX_EXAMPLES_ARGS[@]}"
