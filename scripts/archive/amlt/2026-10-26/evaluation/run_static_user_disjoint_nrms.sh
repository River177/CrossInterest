#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT_ROOT="$AMLT_CODE_DIR/amlt_inputs/static_user_disjoint_nrms_v1"

install -D "$INPUT_ROOT/dataset/train.parquet" \
  "$AMLT_CODE_DIR/data/processed/new_interest_static_user_disjoint_v1/train.parquet"
install -D "$INPUT_ROOT/dataset/interest_vocabulary.parquet" \
  "$AMLT_CODE_DIR/data/processed/new_interest_static_user_disjoint_v1/interest_vocabulary.parquet"

for split in fit validation dev; do
  install -D "$INPUT_ROOT/semantic/$split.parquet" \
    "$AMLT_CODE_DIR/data/processed/new_interest_static_user_disjoint_semantic_users_v1/$split.parquet"
done

for file in title_embeddings.parquet metadata.json manifest.json; do
  install -D "$INPUT_ROOT/title_embeddings/$file" \
    "$AMLT_CODE_DIR/artifacts/future-novel-interest/static-user-disjoint-nrms-title-embeddings-v1/$file"
done

install -D "$INPUT_ROOT/linear_ranker/summary.json" \
  "$AMLT_CODE_DIR/artifacts/future-novel-interest/static-user-disjoint-balanced-semantic-linear-ranker-v1/summary.json"
install -D "$INPUT_ROOT/linear_ranker/tuning.json" \
  "$AMLT_CODE_DIR/artifacts/future-novel-interest/static-user-disjoint-balanced-semantic-linear-ranker-v1/tuning.json"
install -D "$INPUT_ROOT/balanced_ridge/summary.json" \
  "$AMLT_CODE_DIR/artifacts/future-novel-interest/static-user-disjoint-balanced-semantic-history-ridge-v1/summary.json"

python -m new_interest.experiments.tune_semantic_history_nrms \
  configs/future_interest/static_user_disjoint_nrms_v1.json

cp -a \
  "$AMLT_CODE_DIR/artifacts/future-novel-interest/static-user-disjoint-nrms-v1" \
  "$AMLT_OUTPUT_DIR/"
