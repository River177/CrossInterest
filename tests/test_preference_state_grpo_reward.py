from __future__ import annotations

import pytest

from new_interest.training.preference_state_grpo_reward import (
    EXACT_REWARD,
    INVALID_REWARD,
    NOVEL_REWARD,
    SAME_PARENT_REWARD,
    score_hierarchical_reward,
)


def _extra() -> dict[str, object]:
    return {
        "target_token": "<interest_001>",
        "target_category": "Sports",
        "history_subcategories": ["Football"],
        "interest_tokens": [
            "<interest_000>",
            "<interest_001>",
            "<interest_002>",
            "<interest_003>",
        ],
        "interest_subcategories": [
            "Football",
            "Basketball",
            "Baseball",
            "Markets",
        ],
        "interest_categories": ["Sports", "Sports", "Sports", "Business"],
    }


def _completion(token: str) -> str:
    return f"<think>nonempty reasoning</think><answer>{token}</answer>"


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("bad", INVALID_REWARD),
        (_completion("<interest_000>"), INVALID_REWARD),
        (_completion("<interest_003>"), NOVEL_REWARD),
        (_completion("<interest_002>"), SAME_PARENT_REWARD),
        (_completion("<interest_001>"), EXACT_REWARD),
    ],
)
def test_hierarchical_reward_ladder(completion: str, expected: float) -> None:
    result = score_hierarchical_reward(completion, _extra())
    assert result.reward == expected


def test_reward_requires_one_strict_final_answer() -> None:
    result = score_hierarchical_reward(
        _completion("<interest_001>") + " extra",
        _extra(),
    )
    assert result.reward == 0.0
    assert result.valid_format is False
