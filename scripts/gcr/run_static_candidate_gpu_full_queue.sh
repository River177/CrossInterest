#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"
conditions=(
    qwen35_behavior_uniform
    qwen35_interest_uniform
    llama31_behavior_uniform
    llama31_interest_uniform
)

cd "$project_dir"
log_dir="logs/static-candidate-gpu-full-v4"
mkdir -p "$log_dir"

echo "QUEUE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for condition_id in "${conditions[@]}"; do
    echo "RUNNING_CONDITION=$condition_id"
    bash scripts/gcr/run_static_candidate_gpu_condition.sh full "$condition_id" \
        2>&1 | tee "$log_dir/${condition_id}.log"
done
echo "QUEUE_COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
