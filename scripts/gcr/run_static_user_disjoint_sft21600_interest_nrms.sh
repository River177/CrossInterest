#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_static_user_disjoint_sft21600_interest_nrms.sh <smoke|fixed|full>}"
PROJECT_DIR="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-interest-nrms-sft21600-v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/v-wuhao1/envs/lgclean/bin/python}"

case "$MODE" in
  smoke)
    CONFIG="configs/future_interest/static_user_disjoint_sft21600_h50_interest_nrms_smoke_v1.json"
    ;;
  fixed)
    CONFIG="configs/future_interest/static_user_disjoint_sft21600_h50_interest_nrms_fixed_v1.json"
    ;;
  full)
    CONFIG="configs/future_interest/static_user_disjoint_sft21600_h50_interest_nrms_v1.json"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

test -x "$PYTHON_BIN"
test -f "$CONFIG"
test -f data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/train.parquet
test -f data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/dev.parquet
test -f artifacts/future-novel-interest/static-user-disjoint-nrms-interest-embeddings-v1/interest_embeddings.parquet
mkdir -p logs

"$PYTHON_BIN" -m new_interest.experiments.tune_semantic_history_nrms \
  "$CONFIG" 2>&1 | tee "logs/interest-nrms-sft21600-$MODE.log"
