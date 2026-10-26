#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"
manifest="configs/candidate/static_user_disjoint_compact_single_view_gpu_v4.json"
pilot_gate="configs/candidate/static_user_disjoint_compact_single_view_gpu_v4_gate.json"
conditions=(
    qwen35_behavior_uniform
    qwen35_interest_uniform
    llama31_behavior_uniform
    llama31_interest_uniform
)

cd "$project_dir"
log_dir="logs/static-candidate-gpu-pilot-v4"
mkdir -p "$log_dir"

echo "QUEUE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for condition_id in "${conditions[@]}"; do
    echo "RUNNING_CONDITION=$condition_id"
    bash scripts/gcr/run_static_candidate_gpu_condition.sh pilot "$condition_id" \
        2>&1 | tee "$log_dir/${condition_id}.log"
done

PYTHONPATH="$project_dir/src" /home/v-wuhao1/venvs/cross-interest-sft/bin/python \
    -m new_interest.experiments.finalize_static_candidate_pilots \
    --manifest "$manifest" \
    --output "$pilot_gate"
sha256sum "$pilot_gate"
echo "QUEUE_COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
