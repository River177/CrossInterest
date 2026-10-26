#!/usr/bin/env bash
set -euo pipefail

condition_id="${1:?Usage: run_static_candidate_gpt5_full_condition.sh CONDITION_ID}"
project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"

case "$condition_id" in
    gpt5_none_behavior_uniform|gpt5_none_interest_uniform)
        manifest="configs/candidate/static_user_disjoint_compact_single_view_gpt5_v1.json"
        pilot_gate="configs/candidate/static_user_disjoint_compact_single_view_gpt5_none_v1_gate.json"
        ;;
    gpt5_medium_behavior_uniform|gpt5_medium_interest_uniform)
        manifest="configs/candidate/static_user_disjoint_compact_single_view_gpt5_v1.json"
        pilot_gate="configs/candidate/static_user_disjoint_compact_single_view_gpt5_medium_v1_gate.json"
        ;;
    gpt5_none_behavior_interest_uniform)
        manifest="configs/candidate/static_user_disjoint_behavior_interest_gpt5_v1.json"
        pilot_gate="configs/candidate/static_user_disjoint_behavior_interest_gpt5_none_v1_gate.json"
        ;;
    gpt5_medium_behavior_interest_uniform)
        manifest="configs/candidate/static_user_disjoint_behavior_interest_gpt5_v1.json"
        pilot_gate="configs/candidate/static_user_disjoint_behavior_interest_gpt5_medium_v1_gate.json"
        ;;
    *)
        echo "Condition is outside the six-condition GPT-5 full allowlist: $condition_id" >&2
        exit 2
        ;;
esac

cd "$project_dir"
echo "STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=$(hostname)"
echo "PROJECT_DIR=$project_dir"
echo "CONDITION_ID=$condition_id"
.venv/bin/python --version
sha256sum "$manifest" "$pilot_gate"

export PYTHONPATH="$project_dir/src"
exec .venv/bin/python -m new_interest.experiments.run_static_candidate_condition \
    --manifest "$manifest" \
    --condition "$condition_id" \
    --phase full \
    --pilot-gate "$pilot_gate"
