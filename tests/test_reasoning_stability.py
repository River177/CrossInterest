from __future__ import annotations

import math

import pytest

from new_interest.experiments.evaluate_reasoning_stability import (
    DIRECT_SUFFIX,
    reasoning_prompt,
    summarize_reasoning_group,
)


def test_reasoning_prompt_replaces_only_the_canonical_suffix() -> None:
    prompt = "History. " + DIRECT_SUFFIX
    transformed = reasoning_prompt(prompt)
    assert transformed.startswith("History. ")
    assert DIRECT_SUFFIX not in transformed
    assert "final line" in transformed
    with pytest.raises(ValueError, match="canonical"):
        reasoning_prompt("noncanonical prompt")


def test_reasoning_group_reports_mixed_accuracy_and_entropy() -> None:
    summary = summarize_reasoning_group(
        ["<interest_000>", "<interest_001>", "<interest_001>", None],
        acceptable_tokens={"<interest_000>"},
        vocabulary={"<interest_000>", "<interest_001>"},
    )
    assert summary["rollouts"] == 4
    assert summary["valid_rollouts"] == 3
    assert summary["correct_rollouts"] == 1
    assert summary["valid_rate"] == pytest.approx(0.75)
    assert summary["correct_rate"] == pytest.approx(0.25)
    assert summary["unique_valid_predictions"] == 2
    assert summary["prediction_entropy"] == pytest.approx(
        -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
    )
    assert summary["mixed_correctness"] is True


def test_reasoning_group_rejects_empty_rollouts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_reasoning_group(
            [], acceptable_tokens=set(), vocabulary={"<interest_000>"}
        )
