from __future__ import annotations

from types import SimpleNamespace

import pytest

from new_interest.data.generate_solar_gnip_teacher import (
    PROFILE_FIELDS,
    _call_json,
    _icl_examples,
    _profile_prompt,
    validate_profile,
    validate_profile_calibration,
    validate_ranking,
    validate_rerank_calibration,
)


def _candidate(example_id: str, user_id: str, target: str) -> dict[str, object]:
    ids = [f"C{index:03d}" for index in range(20)]
    return {
        "example_id": example_id,
        "user_id": user_id,
        "split": "train",
        "prompt": "History:\n\n1. [Old] headline\n\nBased on both views, predict.",
        "history_interests": ["Old"],
        "positive_interest_ids": ["C019"],
        "positive_interests": [target],
        "candidate_interest_ids": ids,
        "candidate_interests": [f"interest-{index}" for index in range(20)],
        "candidate_scores": [float(20 - index) for index in range(20)],
        "target_in_candidates": True,
        "candidate_recall_at_20": 1.0,
    }


def test_profile_validation_requires_all_six_short_fields() -> None:
    profile = {field: f"short {field}" for field in PROFILE_FIELDS}

    assert validate_profile(profile) == profile
    with pytest.raises(ValueError, match="six frozen fields"):
        validate_profile({"implicit_preference": "only one"})


def test_ranking_validation_requires_exact_permutation() -> None:
    ids = [f"C{index:03d}" for index in range(20)]

    assert validate_ranking({"ranking": ids[::-1]}, ids) == ids[::-1]
    with pytest.raises(ValueError, match="duplicates"):
        validate_ranking({"ranking": [ids[0]] * 20}, ids)
    with pytest.raises(ValueError, match="candidate set"):
        validate_ranking({"ranking": ids[:-1] + ["C999"]}, ids)


def test_icl_examples_exclude_current_source_and_user() -> None:
    current = _candidate("current", "same-user", "new-current")
    pool = [current, _candidate("same-user-other", "same-user", "new-same")]
    pool.extend(_candidate(f"other-{index}", f"user-{index}", f"new-{index}") for index in range(6))

    first = _icl_examples(current, pool, count=4, seed=42)
    second = _icl_examples(current, pool, count=4, seed=42)

    assert first == second
    assert len(first) == 4
    assert {row["example_id"] for row in first}.isdisjoint({"current", "same-user-other"})
    assert all(row["historical_negative"] == "Old" for row in first)


def test_profile_prompt_contains_no_current_positive_or_candidates() -> None:
    record = _candidate("current", "user", "SECRET CURRENT TARGET")
    calibration = {
        "field_guidance": {field: f"guidance for {field}" for field in PROFILE_FIELDS}
    }

    prompt = _profile_prompt(record, calibration)

    assert "SECRET CURRENT TARGET" not in prompt
    assert "C019" not in prompt
    assert "1. [Old] headline" in prompt


def test_calibration_validators_are_strict() -> None:
    profile = {
        "field_guidance": {field: f"guidance {field}" for field in PROFILE_FIELDS}
    }
    rerank = {"ranking_principles": ["one", "two", "three", "four"]}

    assert validate_profile_calibration(profile) == profile
    assert validate_rerank_calibration(rerank) == rerank
    with pytest.raises(ValueError, match="four principles"):
        validate_rerank_calibration({"ranking_principles": ["one"]})


def test_call_json_uses_gpt55_json_mode_without_temperature() -> None:
    profile = {field: f"value {field}" for field in PROFILE_FIELDS}

    class Completions:
        def __init__(self) -> None:
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content=__import__("json").dumps(profile))
            choice = SimpleNamespace(message=message, finish_reason="stop")
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            return SimpleNamespace(
                choices=[choice],
                usage=usage,
                model_dump_json=lambda: "{}",
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = _call_json(
        client,
        system_prompt="system",
        user_prompt="user",
        validator=validate_profile,
        sleep_fn=lambda _: None,
    )

    assert result["status"] == "success"
    assert result["parsed"] == profile
    assert completions.kwargs["reasoning_effort"] == "none"
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert "temperature" not in completions.kwargs
