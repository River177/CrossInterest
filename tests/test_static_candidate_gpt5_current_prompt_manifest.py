from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "configs/candidate/static_user_disjoint_compact_single_view_gpt5_v1.json"
BI = ROOT / "configs/candidate/static_user_disjoint_behavior_interest_gpt5_v1.json"


def _conditions(path: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row for row in manifest["conditions"]}


def test_gpt5_manifests_cover_exact_current_prompt_pilots() -> None:
    single_manifest = json.loads(SINGLE.read_text(encoding="utf-8"))
    bi_manifest = json.loads(BI.read_text(encoding="utf-8"))
    assert single_manifest["data"]["pilot_root"].endswith(
        "candidate_selection_static_user_disjoint_k10_compact_v4_pilot1000"
    )
    assert bi_manifest["data"]["pilot_root"].endswith(
        "candidate_selection_static_user_disjoint_k10_behavior_interest_v3_pilot1000"
    )
    assert single_manifest["data"]["pilot_samples"] == 1000
    assert bi_manifest["data"]["pilot_samples"] == 1000

    single = _conditions(SINGLE)
    bi = _conditions(BI)
    assert set(single) == {
        "gpt5_none_behavior_uniform",
        "gpt5_none_interest_uniform",
        "gpt5_medium_behavior_uniform",
        "gpt5_medium_interest_uniform",
    }
    assert set(bi) == {
        "gpt5_none_behavior_interest_uniform",
        "gpt5_medium_behavior_interest_uniform",
    }
    for condition in [*single.values(), *bi.values()]:
        parameters = condition["parameters"]
        assert condition["evaluator"] == "trapi"
        assert condition["strategy"] == "uniform"
        assert parameters["instance"] == "redmond/interactive"
        assert parameters["deployment"] == "gpt-5_2025-08-07"
        assert parameters["api_version"] == "2024-10-21"
        assert parameters["concurrency"] == 4
        assert parameters["checkpoint_size"] == 64
        assert parameters["minimum_success_rate"] == 0.99


def test_gpt5_pilot_scripts_are_fail_closed_and_never_launch_full() -> None:
    condition_script = (
        ROOT / "scripts/gcr/run_static_candidate_gpt5_pilot_condition.sh"
    ).read_text(encoding="utf-8")
    queue_script = (
        ROOT / "scripts/gcr/run_static_candidate_gpt5_pilot_queue.sh"
    ).read_text(encoding="utf-8")
    for condition_id in (
        "gpt5_none_behavior_uniform",
        "gpt5_none_interest_uniform",
        "gpt5_none_behavior_interest_uniform",
        "gpt5_medium_behavior_uniform",
        "gpt5_medium_interest_uniform",
        "gpt5_medium_behavior_interest_uniform",
    ):
        assert condition_id in condition_script
        assert condition_id in queue_script
    assert "--phase pilot" in condition_script
    assert "--phase full" not in condition_script
    assert "finalize_static_candidate_pilots" in queue_script
    assert "SKIP_EXISTING_PASS" in queue_script
    assert "SKIP_EXISTING_FAILURE" in queue_script
    assert "failed_conditions" in queue_script


def test_gpt5_full_scripts_require_setting_specific_gates() -> None:
    condition_script = (
        ROOT / "scripts/gcr/run_static_candidate_gpt5_full_condition.sh"
    ).read_text(encoding="utf-8")
    queue_script = (
        ROOT / "scripts/gcr/run_static_candidate_gpt5_full_queue.sh"
    ).read_text(encoding="utf-8")
    assert "--phase full" in condition_script
    assert "--pilot-gate" in condition_script
    assert "gpt5_none_v1_gate.json" in condition_script
    assert "gpt5_medium_v1_gate.json" in condition_script
    assert "gpt5_none_behavior_uniform" in queue_script
    assert "gpt5_none_behavior_interest_uniform" in queue_script
    assert "gpt5_medium_behavior_uniform" in queue_script
    assert "gpt5_medium_behavior_interest_uniform" in queue_script


def test_gpt5_none_interest_waiver_is_narrow_and_preserves_failure() -> None:
    waiver_script = (
        ROOT / "scripts/gcr/issue_gpt5_none_interest_waiver_gate.py"
    ).read_text(encoding="utf-8")
    assert 'WAIVED_CONDITION = "gpt5_none_interest_uniform"' in waiver_script
    assert 'PASSING_CONDITION = "gpt5_none_behavior_uniform"' in waiver_script
    assert 'float(metrics["candidate_valid_rate"]) != 0.97' in waiver_script
    assert '"candidate_invalid_count": 30' in waiver_script
    assert '"failed_acceptance_preserved": True' in waiver_script
