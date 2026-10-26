#!/usr/bin/env bash
set -euo pipefail
mode="${1:?Expected mixed or meta}"
case "$mode" in mixed|meta) ;; *) exit 2 ;; esac
cd "${AMLT_CODE_DIR:?}"
export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true HF_HUB_DISABLE_XET=1
export HF_HOME=/mnt/default/huggingface-cache-mind-old-new-0911
export PIP_CACHE_DIR=/mnt/default/pip-cache-mind-old-new-0911
export META_LOCAL_SCRATCH=/scratch/oi-ni-full-sft
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$META_LOCAL_SCRATCH" "${AMLT_OUTPUT_DIR:?}"
python - <<'PY'
import hashlib,json,pathlib
m=json.loads(pathlib.Path('full_sft_bundle_manifest.json').read_text())
for name,digest in m['files'].items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Frozen source mismatch: '+name)
print('FULL_SFT_SOURCE_VERIFIED',flush=True)
PY
python -m pip install --user --no-warn-script-location --retries 5 --timeout 120 \
  "torch==2.8.0+cu126" "torchvision==0.23.0+cu126" "torchaudio==2.8.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --user --no-warn-script-location --retries 5 --timeout 120 \
  -r requirements-oi-ni-meta-worker.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_oi_ni_full_sft.py \
  tests/test_oi_ni_meta_backend.py tests/test_oi_ni_meta_distributed_objective.py \
  > "$AMLT_OUTPUT_DIR/startup_cpu_tests.log" 2>&1
timeout --signal=TERM --kill-after=120 1209600 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m new_interest.experiments.oi_ni_full_sft \
  --config configs/experiments/mind_oi_ni_full_sft_v1.json \
  --root "$AMLT_OUTPUT_DIR" --mode "$mode"
