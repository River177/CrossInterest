#!/usr/bin/env bash
set -euo pipefail

queue_id="${1:?Usage: run_behavior_interest_candidate_trapi_full_queue.sh QUEUE_ID}"
project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"

case "$queue_id" in
    gpt55-bi)
        conditions=(
            gpt55_none_behavior_interest_uniform
            gpt55_medium_behavior_interest_uniform
        )
        ;;
    grok43-bi)
        conditions=(grok43_behavior_interest_uniform)
        ;;
    *)
        echo "Unknown B+I queue: $queue_id" >&2
        exit 2
        ;;
esac

cd "$project_dir"
log_dir="logs/static-candidate-bi-full-v3"
mkdir -p "$log_dir"

echo "QUEUE_ID=$queue_id"
echo "QUEUE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for condition_id in "${conditions[@]}"; do
    log_file="$log_dir/${condition_id}.log"
    echo "RUNNING_CONDITION=$condition_id"
    bash scripts/gcr/run_behavior_interest_candidate_trapi_full.sh "$condition_id" \
        2>&1 | tee "$log_file"
done
echo "QUEUE_COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
