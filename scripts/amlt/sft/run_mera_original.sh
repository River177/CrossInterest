#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_mera_original.sh <smoke|full> <arts|grocery> <code-faithful|paper-spec>}"
DATASET="${2:?dataset is required}"
CONTRACT="${3:?contract is required}"

case "$MODE" in smoke|full) ;; *) echo "Unsupported mode: $MODE" >&2; exit 2;; esac
case "$DATASET" in arts|grocery) ;; *) echo "Unsupported dataset: $DATASET" >&2; exit 2;; esac
case "$CONTRACT" in code-faithful|paper-spec) ;; *) echo "Unsupported contract: $CONTRACT" >&2; exit 2;; esac

EXPECTED_JOB="${DATASET}-${CONTRACT%-faithful}-${MODE}"
if [[ "$CONTRACT" == "code-faithful" ]]; then
  EXPECTED_JOB="${DATASET}-code-${MODE}"
else
  EXPECTED_JOB="${DATASET}-paper-${MODE}"
fi
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB" ]]; then
  echo "Unexpected job name: $AMLT_JOB_NAME (expected $EXPECTED_JOB)" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"

CONFIG="$AMLT_CODE_DIR/configs/mera_original/experiment_matrix_v1.yaml"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"
if [[ -f "$SUCCESS_MARKER" ]]; then
  python -m new_interest.experiments.audit_mera_original verify-published \
    --config "$CONFIG" --dataset "$DATASET" --contract "$CONTRACT" \
    --mode "$MODE" --output-dir "$AMLT_OUTPUT_DIR"
  exit 0
fi

LOCAL_ROOT="/scratch/mera-original-${AMLT_JOB_NAME}"
LOCAL_RUN="$LOCAL_ROOT/run"
PERSISTENT_CHECKPOINTS="$AMLT_OUTPUT_DIR/checkpoints"
mkdir -p "$LOCAL_RUN" "$PERSISTENT_CHECKPOINTS"

TRAIN_ARGS=(
  --config "$CONFIG"
  --dataset "$DATASET"
  --contract "$CONTRACT"
  --output-dir "$LOCAL_RUN"
  --resume-from-checkpoint auto
  --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINTS"
)
EVAL_ARGS=(
  --config "$CONFIG"
  --dataset "$DATASET"
  --contract "$CONTRACT"
  --adapter-dir "$LOCAL_RUN/final_adapter"
  --training-dir "$LOCAL_RUN"
  --output-dir "$LOCAL_RUN/evaluation"
  --batch-size 32
)
if [[ "$MODE" == "smoke" ]]; then
  TRAIN_ARGS+=(--max-steps 8 --max-eval-examples 64)
  EVAL_ARGS+=(--max-eval-examples 64)
fi

python -m new_interest.training.train_mera_original "${TRAIN_ARGS[@]}"
python -m new_interest.evaluation.evaluate_mera_original "${EVAL_ARGS[@]}"
python -m new_interest.experiments.audit_mera_original publish \
  --config "$CONFIG" --dataset "$DATASET" --contract "$CONTRACT" \
  --mode "$MODE" --source-dir "$LOCAL_RUN" --output-dir "$AMLT_OUTPUT_DIR"
python -m new_interest.experiments.audit_mera_original verify-published \
  --config "$CONFIG" --dataset "$DATASET" --contract "$CONTRACT" \
  --mode "$MODE" --output-dir "$AMLT_OUTPUT_DIR"
