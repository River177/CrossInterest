from __future__ import annotations

from new_interest.evaluation.candidate_metrics import (
    evaluate_candidate_outputs,
    parse_candidate_output,
)


def _record() -> dict[str, object]:
    return {
        "selection_id": "s1",
        "impression_id": "1",
        "answer": "New topic: Music",
        "answer_category": "Music",
        "candidates": ["Music", "Golf", "Film"],
        "correct_position": 0,
        "history_length": 3,
        "all_acceptable_subcategories": ["Music"],
    }


def test_candidate_parser_distinguishes_strict_and_category() -> None:
    strict = parse_candidate_output(
        "New topic: Music",
        ["Music", "Golf"],
        "New topic: Music",
        "Music",
    )
    assert strict["strict_correct"]
    assert strict["category_correct"]
    assert strict["candidate_valid"]
    assert strict["prefix_compliant"]

    missing_prefix = parse_candidate_output(
        "Music",
        ["Music", "Golf"],
        "New topic: Music",
        "Music",
    )
    assert not missing_prefix["strict_correct"]
    assert missing_prefix["category_correct"]
    assert missing_prefix["candidate_valid"]
    assert not missing_prefix["prefix_compliant"]

    numbered = parse_candidate_output(
        "New topic: 2. Golf",
        ["Music", "Golf"],
        "New topic: Golf",
        "Golf",
    )
    assert not numbered["strict_correct"]
    assert numbered["category_correct"]
    assert numbered["candidate_valid"]

    option_letter = parse_candidate_output(
        'Option B: "Golf".',
        ["Music", "Golf"],
        "New topic: Golf",
        "Golf",
    )
    assert option_letter["category_correct"]
    assert option_letter["candidate_valid"]


def test_candidate_parser_rejects_explanations_and_multiple() -> None:
    explanation = parse_candidate_output(
        "New topic: Music because it fits.",
        ["Music", "Golf"],
        "New topic: Music",
        "Music",
    )
    assert not explanation["category_correct"]
    assert not explanation["candidate_valid"]

    multiple = parse_candidate_output(
        "New topic: Music or Golf",
        ["Music", "Golf"],
        "New topic: Music",
        "Music",
    )
    assert multiple["multiple_candidates_mentioned"]


def test_candidate_metrics_group_results() -> None:
    second = {
        **_record(),
        "selection_id": "s2",
        "answer": "New topic: Golf",
        "answer_category": "Golf",
        "correct_position": 1,
        "history_length": 12,
    }
    metrics, details = evaluate_candidate_outputs(
        [_record(), second],
        ["New topic: Music", "Golf"],
    )
    assert metrics["strict_accuracy"] == 0.5
    assert metrics["category_accuracy"] == 1.0
    assert metrics["candidate_valid_rate"] == 1.0
    assert metrics["prefix_compliance_rate"] == 0.5
    assert metrics["by_group"]["position:1"]["strict_correct"] == 1.0
    assert metrics["by_group"]["position:2"]["strict_correct"] == 0.0
    assert details[1]["normalized_category"] == "Golf"
