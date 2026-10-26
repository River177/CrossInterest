from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path(
    "configs/candidate/static_user_disjoint_behavior_interest_v3.json"
)
GPT55_MANIFEST = Path(
    "configs/candidate/static_user_disjoint_behavior_interest_gpt55_v1.json"
)
GROK43_MANIFEST = Path(
    "configs/candidate/static_user_disjoint_behavior_interest_grok43_v1.json"
)


def test_behavior_interest_candidate_manifest_has_exact_approved_conditions() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    conditions = manifest["conditions"]

    assert len(conditions) == 7
    assert {condition["id"] for condition in conditions} == {
        "qwen35_behavior_interest_uniform",
        "llama31_behavior_interest_uniform",
        "gpt56_medium_behavior_interest_uniform",
        "gpt56_none_behavior_interest_uniform",
        "gpt5_medium_behavior_interest_uniform",
        "gpt5_none_behavior_interest_uniform",
        "grok46_behavior_interest_uniform",
    }
    assert {condition["strategy"] for condition in conditions} == {"uniform"}
    assert {condition["task_type"] for condition in conditions} == {
        "behavior_interest"
    }

    by_id = {condition["id"]: condition for condition in conditions}
    assert by_id["qwen35_behavior_interest_uniform"]["parameters"][
        "model_name"
    ] == "Qwen/Qwen3.5-9B"
    assert by_id["llama31_behavior_interest_uniform"]["parameters"][
        "model_name"
    ] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert by_id["gpt56_medium_behavior_interest_uniform"]["parameters"][
        "deployment"
    ] == "gpt-5.6-sol_2026-07-09"
    assert by_id["gpt5_medium_behavior_interest_uniform"]["parameters"][
        "deployment"
    ] == "gpt-5_2025-08-07"
    assert by_id["grok46_behavior_interest_uniform"]["parameters"][
        "deployment"
    ] == "grok-4.6_1"


def test_behavior_interest_candidate_manifest_keeps_fixed_pilot_gate() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["data"]["pilot_samples"] == 1000
    assert manifest["data"]["full_samples"] == 37817
    assert manifest["pilot_acceptance"] == {
        "minimum_request_success_rate": 0.99,
        "minimum_candidate_valid_rate": 0.99,
        "minimum_prefix_compliance_rate": 0.99,
        "require_no_truncation": True,
        "require_zero_reasoning_tokens_when_expected": True,
    }


def test_gpt55_replacement_manifest_has_medium_and_none_only() -> None:
    manifest = json.loads(GPT55_MANIFEST.read_text(encoding="utf-8"))
    conditions = manifest["conditions"]
    assert [condition["id"] for condition in conditions] == [
        "gpt55_medium_behavior_interest_uniform",
        "gpt55_none_behavior_interest_uniform",
    ]
    assert {condition["task_type"] for condition in conditions} == {
        "behavior_interest"
    }
    assert {condition["strategy"] for condition in conditions} == {"uniform"}
    assert {
        condition["parameters"]["deployment"] for condition in conditions
    } == {"gpt-5.5_2026-04-24"}
    assert {
        condition["parameters"]["reasoning_effort"]
        for condition in conditions
    } == {"medium", "none"}


def test_grok43_fallback_manifest_is_isolated_from_grok46() -> None:
    manifest = json.loads(GROK43_MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["conditions"]) == 1
    condition = manifest["conditions"][0]
    assert condition["id"] == "grok43_behavior_interest_uniform"
    assert condition["task_type"] == "behavior_interest"
    assert condition["strategy"] == "uniform"
    assert condition["parameters"]["deployment"] == "grok-4.3_1"
    assert condition["parameters"]["reasoning_effort"] is None
    assert "grok43" in manifest["artifact_root"]
