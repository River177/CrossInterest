#!/usr/bin/env bash
set -euo pipefail
alpha_index="${1:?Expected original calibration alpha index 1 or 2}"
case "$alpha_index" in 1|2) ;; *) exit 2 ;; esac
cd "${AMLT_CODE_DIR:?}"
export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HUB_DISABLE_XET=1
export HF_HOME="/mnt/default/huggingface-cache-mind-old-new-0911"
export PIP_CACHE_DIR="/mnt/default/pip-cache-mind-old-new-0911"
export META_LOCAL_SCRATCH="/scratch/oi-ni-meta-checkpoints"
mkdir -p "$META_LOCAL_SCRATCH" "${AMLT_OUTPUT_DIR:?}"
cp source_bundle_manifest.json dispatch_manifest.json recovery_parent.json "$AMLT_OUTPUT_DIR/"
cp scripts/amlt/sft/run_oi_ni_meta_calibration_arm.sh "$AMLT_OUTPUT_DIR/operational_launcher.sh"
python -m pip install --user --no-warn-script-location --retries 5 --timeout 120 \
  "torch==2.8.0+cu126" "torchvision==0.23.0+cu126" "torchaudio==2.8.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --user --no-warn-script-location --retries 5 --timeout 120 \
  -r requirements-oi-ni-meta-worker.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout --signal=TERM --kill-after=120 43200 \
  python -m new_interest.experiments.oi_ni_meta_calibration_arm \
  --config configs/experiments/mind_oi_ni_meta_pilot_v1.json \
  --root "$AMLT_OUTPUT_DIR" --alpha-index "$alpha_index" \
  --dispatch-manifest dispatch_manifest.json --parent-recovery recovery_parent.json \
  2>&1 | tee "$AMLT_OUTPUT_DIR/worker.log"
