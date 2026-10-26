from __future__ import annotations

import pytest

from new_interest.training.grpo_format_bootstrap_reward import (
    MALFORMED_VALID_REWARD,
    compute_score,
    reasoning_length_score,
    score_format_bootstrap,
)

EXTRA_INFO = {
    "retained_subcategories": [
        "Cars & Driving",
        "Music",
        "Wine, Beer & Spirits",
    ],
    "history_subcategories": ["Music"],
}


def _score(output: str) -> float:
    return compute_score(
        "cross_interest",
        output,
        "New topic: Cars & Driving",
        EXTRA_INFO,
    )


@pytest.mark.parametrize(
    ("word_count", "expected"),
    [
        (8, 0.0),
        (12, 0.5),
        (16, 1.0),
        (64, 1.0),
        (80, 0.8),
        (96, 2 / 3),
    ],
)
def test_reasoning_length_score(
    word_count: int,
    expected: float,
) -> None:
    assert reasoning_length_score(word_count) == expected


def test_valid_novel_bare_label_gets_zero() -> None:
    assert _score("Cars & Driving") == MALFORMED_VALID_REWARD


def test_invalid_or_non_novel_label_gets_zero() -> None:
    assert _score("Unknown") == 0.0
    assert _score("Music") == 0.0


def test_ideal_structured_output_gets_full_reward() -> None:
    reasoning = " ".join(["reason"] * 32)
    details = score_format_bootstrap(
        f"<think>{reasoning}</think>\nNew topic: Cars & Driving",
        EXTRA_INFO,
    )
    assert details.reward == 1.0
    assert details.structured is True
    assert details.reasoning_words == 32
    assert details.length_score == 1.0


def test_long_structured_output_gets_continuous_shortness_reward() -> None:
    reasoning = " ".join(["reason"] * 100)
    details = score_format_bootstrap(
        f"<think>{reasoning}</think>\n"
        "New topic: Wine, Beer & Spirits",
        EXTRA_INFO,
    )
    assert details.reward == pytest.approx(0.73)
    assert details.length_score == pytest.approx(0.64)


def test_target_correctness_is_intentionally_ignored() -> None:
    reasoning = " ".join(["reason"] * 24)
    assert _score(
        f"<think>{reasoning}</think>\n"
        "New topic: Wine, Beer & Spirits"
    ) == 1.0


def test_trailing_text_and_empty_think_get_zero() -> None:
    assert _score(
        "<think>reason</think>\nNew topic: Cars & Driving\nextra"
    ) == 0.0
    assert _score(
        "<think></think>\nNew topic: Cars & Driving"
    ) == MALFORMED_VALID_REWARD
