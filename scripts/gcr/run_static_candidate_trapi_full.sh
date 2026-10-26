#!/usr/bin/env bash
set -euo pipefail

condition_id="${1:?Usage: run_static_candidate_trapi_full.sh CONDITION_ID}"
project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"
manifest="configs/candidate/static_user_disjoint_compact_single_view_pilots_v4.json"
pilot_gate="configs/candidate/static_user_disjoint_compact_single_view_pilots_v4_gate.json"

case "$condition_id" in
    gpt55_none_behavior_uniform|gpt55_none_interest_uniform|\
    gpt55_medium_behavior_uniform|gpt55_medium_interest_uniform|\
    grok43_behavior_uniform|grok43_interest_uniform)
        ;;
    *)
        echo "Condition is outside the frozen six-condition allowlist: $condition_id" >&2
        exit 2
        ;;
esac

cd "$project_dir"
mkdir -p logs/static-candidate-full-v4

echo "STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=$(hostname)"
echo "PROJECT_DIR=$project_dir"
echo "CONDITION_ID=$condition_id"
.venv/bin/python --version
sha256sum "$manifest" "$pilot_gate"

exec .venv/bin/python -m new_interest.experiments.run_static_candidate_condition \
    --manifest "$manifest" \
    --condition "$condition_id" \
    --phase full \
    --pilot-gate "$pilot_gate"
