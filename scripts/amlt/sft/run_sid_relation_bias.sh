#!/usr/bin/env bash
set -euo pipefail
cd "${AMLT_CODE_DIR:?}"
export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/default/huggingface-cache-mind-old-new-0911
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export PIP_CACHE_DIR=/mnt/default/pip-cache-mind-old-new-0911
mkdir -p "$AMLT_OUTPUT_DIR" "$PIP_CACHE_DIR"
python -m pip install --user --no-warn-script-location --retries 4 --timeout 90 \
  'torch==2.8.0+cu126' 'torchvision==0.23.0+cu126' 'torchaudio==2.8.0+cu126' \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --user --no-warn-script-location --retries 4 --timeout 90 -r requirements-worker.txt
bash scripts/amlt/common/install_qwen35_fast_kernels.sh
python -m new_interest.experiments.sid_relation_bias worker \
  --project "$AMLT_CODE_DIR" --root "$AMLT_OUTPUT_DIR" --phase "${1:?}" --condition "${2:?}"
