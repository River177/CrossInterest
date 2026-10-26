#!/usr/bin/env bash
set -euo pipefail
cd "${AMLT_CODE_DIR:?}"
export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true HF_HUB_DISABLE_XET=1
export HF_HOME=/mnt/default/huggingface-cache-mind-backbone-0911/2b
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR=/mnt/default/pip-cache-history-order-0923
mkdir -p "$PIP_CACHE_DIR" "${AMLT_OUTPUT_DIR:?}"
python -m pip install --user --no-warn-script-location --retries 10 --timeout 120 \
  "torch==2.8.0+cu126" "torchvision==0.23.0+cu126" "torchaudio==2.8.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --user --no-warn-script-location --retries 10 --timeout 120 \
  -r requirements-backbone-qwen35-worker.txt
python -m new_interest.experiments.history_order_ablation worker \
  --project "$AMLT_CODE_DIR" --root "$AMLT_OUTPUT_DIR"
