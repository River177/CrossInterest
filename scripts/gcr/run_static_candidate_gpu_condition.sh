#!/usr/bin/env bash
set -euo pipefail

phase="${1:?Usage: run_static_candidate_gpu_condition.sh PHASE CONDITION_ID}"
condition_id="${2:?Usage: run_static_candidate_gpu_condition.sh PHASE CONDITION_ID}"
project_dir="${PROJECT_DIR:-/home/v-wuhao1/crossinterest-k10-full-20260830}"
manifest="configs/candidate/static_user_disjoint_compact_single_view_gpu_v4.json"
pilot_gate="configs/candidate/static_user_disjoint_compact_single_view_gpu_v4_gate.json"

case "$phase" in
    pilot|full) ;;
    *)
        echo "Phase must be pilot or full: $phase" >&2
        exit 2
        ;;
esac

case "$condition_id" in
    qwen35_behavior_uniform|qwen35_interest_uniform)
        python_bin="$project_dir/.venv-qwen35/bin/python"
        ;;
    llama31_behavior_uniform|llama31_interest_uniform)
        python_bin="/home/v-wuhao1/venvs/cross-interest-sft/bin/python"
        ;;
    *)
        echo "Condition is outside the compact-v4 GPU allowlist: $condition_id" >&2
        exit 2
        ;;
esac

cd "$project_dir"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing Python environment: $python_bin" >&2
    exit 3
fi

echo "STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=$(hostname)"
echo "PROJECT_DIR=$project_dir"
echo "PHASE=$phase"
echo "CONDITION_ID=$condition_id"
echo "PYTHON_BIN=$python_bin"
"$python_bin" --version
"$python_bin" -c 'import pyarrow, torch, transformers; print("pyarrow=" + pyarrow.__version__); print("torch=" + torch.__version__); print("transformers=" + transformers.__version__)'
sha256sum "$manifest"

args=(
    --manifest "$manifest"
    --condition "$condition_id"
    --phase "$phase"
)
if [[ "$phase" == "full" ]]; then
    sha256sum "$pilot_gate"
    args+=(--pilot-gate "$pilot_gate")
fi

export PYTHONPATH="$project_dir/src"
export TOKENIZERS_PARALLELISM=false
exec "$python_bin" -m new_interest.experiments.run_static_candidate_condition "${args[@]}"
