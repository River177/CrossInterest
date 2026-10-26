#!/usr/bin/env bash
set -euo pipefail

reasoning="${1:?Usage: run_static_candidate_gpt5_full_queue.sh REASONING}"
project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"

case "$reasoning" in
    none)
        conditions=(
            gpt5_none_behavior_uniform
            gpt5_none_interest_uniform
            gpt5_none_behavior_interest_uniform
        )
        ;;
    medium)
        conditions=(
            gpt5_medium_behavior_uniform
            gpt5_medium_interest_uniform
            gpt5_medium_behavior_interest_uniform
        )
        ;;
    *)
        echo "Reasoning must be none or medium: $reasoning" >&2
        exit 2
        ;;
esac

cd "$project_dir"
log_dir="logs/static-candidate-gpt5-full-v1"
mkdir -p "$log_dir"
echo "QUEUE_REASONING=$reasoning"
echo "QUEUE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for condition_id in "${conditions[@]}"; do
    echo "RUNNING_CONDITION=$condition_id"
    bash scripts/gcr/run_static_candidate_gpt5_full_condition.sh "$condition_id" \
        2>&1 | tee "$log_dir/${condition_id}.log"
done
echo "QUEUE_COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
