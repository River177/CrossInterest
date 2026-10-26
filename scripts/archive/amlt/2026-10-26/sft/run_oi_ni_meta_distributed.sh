#!/usr/bin/env bash
set -euo pipefail
phase="${1:?Expected gate, pilot, or compute_matched}"
mode="${2:-meta}"
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
export META_LOCAL_SCRATCH="/scratch/oi-ni-meta-four-rank-checkpoints"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$META_LOCAL_SCRATCH" "${AMLT_OUTPUT_DIR:?}"
python -m pip install --user --no-warn-script-location --retries 5 --timeout 120 \
  "torch==2.8.0+cu126" "torchvision==0.23.0+cu126" "torchaudio==2.8.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --user --no-warn-script-location --retries 5 --timeout 120 \
  -r requirements-oi-ni-meta-worker.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
arguments=(--config configs/experiments/mind_oi_ni_meta_pilot_v1.json --root "$AMLT_OUTPUT_DIR" --phase "$phase" --mode "$mode")
case "$phase" in
  gate)
    python -m pytest -q tests/test_oi_ni_meta_rng.py tests/test_oi_ni_meta_distributed_objective.py \
      tests/test_oi_ni_meta_distributed.py > "$AMLT_OUTPUT_DIR/worker_cpu_tests.log" 2>&1
    limit=7200
    ;;
  pilot|compute_matched)
    arguments+=(--selection-file manifests/oi_ni_meta_alpha_selection.json \
      --selection-lineage-file manifests/oi_ni_meta_alpha_selection_lineage.json \
      --compatibility-gate-file manifests/oi_ni_meta_four_rank_compatibility.json)
    if [[ "$phase" == compute_matched ]]; then
      arguments+=(--budget-file manifests/oi_ni_meta_four_rank_compute_budget.json)
    fi
    limit=259200
    ;;
  *) exit 2 ;;
esac
timeout --signal=TERM --kill-after=120 "$limit" \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m new_interest.experiments.oi_ni_meta_distributed "${arguments[@]}" \
  2>&1 | tee "$AMLT_OUTPUT_DIR/worker.log"
