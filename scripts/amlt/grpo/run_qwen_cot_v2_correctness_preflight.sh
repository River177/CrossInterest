#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: run_qwen_cot_v2_correctness_preflight.sh <behavior|interest>}"

case "$TASK" in
  behavior)
    TASK_TYPE="behavior_only"
    ADAPTER_PATH="/mnt/default/projects/sft4safety/manual-artifacts/cot-v2-20260811/qwen3-behavior/final_adapter"
    ;;
  interest)
    TASK_TYPE="interest_only"
    ADAPTER_PATH="/mnt/default/projects/sft4safety/manual-artifacts/cot-v2-20260811/qwen3-interest/final_adapter"
    ;;
  *)
    echo "Unsupported task: $TASK" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="$AMLT_CODE_DIR/data/training/cot_grpo_v2_experiments/$TASK_TYPE/1k.parquet"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/preflight"

for path in "$ADAPTER_PATH/adapter_model.safetensors" "$DATA_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "Required correctness preflight input not found: $path" >&2
    exit 1
  fi
done

python -m new_interest.evaluation.preflight_grpo_rewards \
  --adapter-path "$ADAPTER_PATH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-prompts 128 \
  --num-generations 16 \
  --batch-size 2 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0 \
  --seed 20260811 \
  --max-new-tokens 256 \
  --reward-mode correctness
