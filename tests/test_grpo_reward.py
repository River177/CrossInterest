from __future__ import annotations

import pytest

from new_interest.training.grpo_reward import (
    compute_score,
    extract_final_answer,
    is_structured_exact,
    parse_structured_output,
)

TARGET = "New topic: Cars & Driving"
EXTRA_INFO = {
    "retained_subcategories": [
        "Cars & Driving",
        "Music",
        "Wine, Beer & Spirits",
    ],
    "retained_label_count": 3,
    "acceptable_subcategories": [
        "Cars & Driving",
        "Wine, Beer & Spirits",
    ],
    "history_subcategories": ["Music"],
}


def _score(output: str, extra_info=EXTRA_INFO) -> float:
    return compute_score(
        "cross_interest",
        output,
        TARGET,
        extra_info,
    )


def test_non_selected_acceptable_target_is_correct() -> None:
    output = (
        "<think>This is a new retained interest.</think>\n"
        "New topic: Wine, Beer & Spirits"
    )
    assert _score(output) == 1.0


def test_history_leakage_is_zero_even_when_acceptable() -> None:
    extra_info = {
        **EXTRA_INFO,
        "acceptable_subcategories": ["Music"],
    }
    assert _score(
        "<think>The history names it.</think>\nNew topic: Music",
        extra_info,
    ) == 0.0


def test_invalid_label_is_zero() -> None:
    extra_info = {
        **EXTRA_INFO,
        "acceptable_subcategories": ["Unknown label"],
    }
    assert _score(
        "<think>It looks plausible.</think>\nNew topic: Unknown label",
        extra_info,
    ) == 0.0


@pytest.mark.parametrize(
    "output",
    [
        "New topic: Cars & Driving",
        "<think></think>\nNew topic: Cars & Driving",
        "<think>Unclosed reasoning\nNew topic: Cars & Driving",
        "<think>Reasoning</think>\nNew topic: Cars & Driving\n",
    ],
)
def test_correct_malformed_output_gets_point_eight(output: str) -> None:
    assert _score(output) == 0.8


def test_strict_correct_output_gets_one() -> None:
    output = (
        "<think>History suggests vehicle interest.</think>\n"
        "New topic: Cars & Driving"
    )
    assert is_structured_exact(output, TARGET)
    assert _score(output) == 1.0


def test_strict_correct_output_allows_blank_line_before_answer() -> None:
    output = (
        "<think>\nHistory suggests vehicle interest.\n</think>\n\n"
        "New topic: Cars & Driving"
    )
    assert parse_structured_output(output) is not None
    assert _score(output) == 1.0


def test_strict_reward_has_no_word_count_component() -> None:
    think = " ".join(["reason"] * 31)
    assert _score(f"<think>{think}</think>\n{TARGET}") == 1.0


def test_strict_wrong_output_gets_zero() -> None:
    output = "<think>This is novel.</think>\nNew topic: Music"
    assert not is_structured_exact(output, TARGET)
    assert _score(output) == 0.0


def test_parse_structured_output_requires_nonempty_think_and_no_trailing() -> None:
    parsed = parse_structured_output(
        "<think>Reasoning here.</think>\nNew topic: Cars & Driving"
    )
    assert parsed is not None
    assert parsed.think == "Reasoning here."
    assert parsed.answer == TARGET
    assert parse_structured_output(f"<think>Reasoning.</think>\n{TARGET}\n") is None
    assert parse_structured_output(f"<think>Reasoning.</think>\n{TARGET} ") is None
    assert parse_structured_output(f"<think>   </think>\n{TARGET}") is None


def test_extract_final_answer_from_malformed_cot() -> None:
    assert extract_final_answer(
        "<think>Unclosed reasoning\nNew topic: Cars & Driving"
    ) == TARGET
