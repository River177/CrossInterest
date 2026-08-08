from __future__ import annotations

import pytest

from new_interest.grpo_reward import (
    compute_score,
    extract_final_answer,
    is_structured_exact,
    parse_structured_output,
)

TARGET = "New topic: Cars & Driving"


@pytest.mark.parametrize(
    "output",
    [
        "<think>History suggests vehicle interest.</think>\n"
        "New topic: Cars & Driving",
        "  <think>\nReasoning over several lines.\nSecond line.\n</think>\n"
        "New topic: Cars & Driving  ",
    ],
)
def test_structured_exact_positive(output: str) -> None:
    assert is_structured_exact(output, TARGET)
    assert compute_score("cross_interest", output, TARGET) == 1.0


@pytest.mark.parametrize(
    "output",
    [
        "New topic: Cars & Driving",
        "<think></think>\nNew topic: Cars & Driving",
        "<think>Correct label is Cars & Driving.</think>\n"
        "New topic: Music",
        "<think>Reasoning</think>\nNew topic: Cars & Driving\nExtra",
        "<think>Reasoning\nNew topic: Cars & Driving",
        "<think>Reasoning</think> Cars & Driving",
        "<think>Reasoning</think>\nnew topic: Cars & Driving",
    ],
)
def test_structured_exact_negative(output: str) -> None:
    assert not is_structured_exact(output, TARGET)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("New topic: Cars & Driving", 0.8),
        ("<think></think>\nNew topic: Cars & Driving", 0.8),
        ("<think>Reasoning</think>\nNew topic: Music", 0.2),
        (
            "<think>Reasoning</think>\n"
            "New topic: Cars & Driving\nExtra",
            0.0,
        ),
        ("<think>Reasoning\nNew topic: Cars & Driving", 0.8),
        ("<think>Reasoning</think> Cars & Driving", 0.0),
        ("<think>Reasoning</think>\nnew topic: Cars & Driving", 0.0),
    ],
)
def test_partial_reward_components(
    output: str,
    expected: float,
) -> None:
    assert compute_score("cross_interest", output, TARGET) == expected


def test_regex_special_characters_are_exact_strings() -> None:
    ground_truth = "New topic: Wine, Beer & Spirits"
    assert compute_score(
        "cross_interest",
        "<think>Reasoning</think>\nNew topic: Wine, Beer & Spirits",
        ground_truth,
    ) == 1.0


def test_parse_structured_output_returns_parts() -> None:
    parsed = parse_structured_output(
        "<think>Reasoning here.</think>\nNew topic: Music"
    )
    assert parsed is not None
    assert parsed.think == "Reasoning here."
    assert parsed.answer == "New topic: Music"


def test_extract_final_answer_from_malformed_cot() -> None:
    assert extract_final_answer(
        "<think>Unclosed reasoning\nNew topic: Cars & Driving"
    ) == TARGET
