from __future__ import annotations

from new_interest.evaluation.evaluate_preference_state_format import (
    parse_preference_state_completion,
)


TOKENS = {"<interest_000>", "<interest_001>"}
LABELS = {
    "<interest_000>": "American Football",
    "<interest_001>": "Markets & Investing",
}


def test_parser_accepts_generation_text_after_template_think_prefix() -> None:
    parsed = parse_preference_state_completion(
        "History summary: sports coverage.\n"
        "Preference state: competitive stories.\n"
        "Potential expansion: another team-based area.\n"
        "</think>\n<answer><interest_000></answer>",
        legal_tokens=TOKENS,
        label_by_token=LABELS,
    )
    assert parsed["valid_format"] is True
    assert parsed["legal_token"] is True
    assert parsed["reasoning_hides_prediction"] is True


def test_parser_rejects_prediction_disclosure_and_trailing_text() -> None:
    disclosed = parse_preference_state_completion(
        "<think>American Football is likely.</think>"
        "<answer><interest_000></answer>",
        legal_tokens=TOKENS,
        label_by_token=LABELS,
    )
    assert disclosed["valid_format"] is True
    assert disclosed["reasoning_hides_prediction"] is False
    trailing = parse_preference_state_completion(
        "<think>reason</think><answer><interest_000></answer> extra",
        legal_tokens=TOKENS,
        label_by_token=LABELS,
    )
    assert trailing["valid_format"] is False
