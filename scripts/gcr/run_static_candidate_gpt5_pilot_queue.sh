#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"
single_manifest="configs/candidate/static_user_disjoint_compact_single_view_gpt5_v1.json"
single_gate="configs/candidate/static_user_disjoint_compact_single_view_gpt5_v1_gate.json"
bi_manifest="configs/candidate/static_user_disjoint_behavior_interest_gpt5_v1.json"
bi_gate="configs/candidate/static_user_disjoint_behavior_interest_gpt5_v1_gate.json"
conditions=(
    gpt5_none_behavior_uniform
    gpt5_none_interest_uniform
    gpt5_none_behavior_interest_uniform
    gpt5_medium_behavior_uniform
    gpt5_medium_interest_uniform
    gpt5_medium_behavior_interest_uniform
)

cd "$project_dir"
log_dir="logs/static-candidate-gpt5-pilot-v1"
mkdir -p "$log_dir"

echo "QUEUE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
failed_conditions=()
for condition_id in "${conditions[@]}"; do
    case "$condition_id" in
        gpt5_none_behavior_uniform|gpt5_none_interest_uniform|\
        gpt5_medium_behavior_uniform|gpt5_medium_interest_uniform)
            condition_root="artifacts/candidate-selection-static-user-disjoint-k10-compact-v4/trapi/gpt5/models/pilot/$condition_id"
            ;;
        gpt5_none_behavior_interest_uniform|gpt5_medium_behavior_interest_uniform)
            condition_root="artifacts/candidate-selection-static-user-disjoint-k10-behavior-interest-v3/gpt5/models/pilot/$condition_id"
            ;;
    esac
    if [[ -f "$condition_root/acceptance.json" ]]; then
        if .venv/bin/python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$condition_root/acceptance.json"; then
            echo "SKIP_EXISTING_PASS=$condition_id"
        else
            echo "SKIP_EXISTING_FAILURE=$condition_id"
            failed_conditions+=("$condition_id")
        fi
        continue
    fi
    echo "RUNNING_CONDITION=$condition_id"
    if ! bash scripts/gcr/run_static_candidate_gpt5_pilot_condition.sh "$condition_id" \
        2>&1 | tee "$log_dir/${condition_id}.log"; then
        failed_conditions+=("$condition_id")
    fi
done

if ! PYTHONPATH="$project_dir/src" .venv/bin/python \
    -m new_interest.experiments.finalize_static_candidate_pilots \
    --manifest "$single_manifest" \
    --output "$single_gate"; then
    echo "SINGLE_VIEW_GATE_NOT_ISSUED"
fi
if ! PYTHONPATH="$project_dir/src" .venv/bin/python \
    -m new_interest.experiments.finalize_static_candidate_pilots \
    --manifest "$bi_manifest" \
    --output "$bi_gate"; then
    echo "BEHAVIOR_INTEREST_GATE_NOT_ISSUED"
fi
[[ ! -f "$single_gate" ]] || sha256sum "$single_gate"
[[ ! -f "$bi_gate" ]] || sha256sum "$bi_gate"
echo "QUEUE_COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if (( ${#failed_conditions[@]} > 0 )); then
    printf 'FAILED_CONDITION=%s\n' "${failed_conditions[@]}"
    exit 1
fi
