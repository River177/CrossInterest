from __future__ import annotations

from new_interest.metrics import (
    evaluate_outputs,
    evaluate_ranked_outputs,
    parse_prediction,
)


def _record(
    impression_id: str,
    targets: list[str],
    history: list[str],
) -> dict[str, object]:
    return {
        "impression_id": impression_id,
        "prompt": "prompt",
        "acceptable_subcategories": targets,
        "acceptable_responses": [
            f"New topic: {target}" for target in targets
        ],
        "history_subcategories": history,
    }


def test_parse_prediction_requires_prefix_and_vocabulary() -> None:
    labels = {"Music", "Golf"}
    assert parse_prediction(
        "New topic: Music",
        labels,
    ).valid_label
    assert not parse_prediction("Music", labels).has_prefix
    assert not parse_prediction(
        "New topic: Unknown",
        labels,
    ).valid_label


def test_multi_answer_metrics() -> None:
    records = [
        _record("1", ["Music", "Golf"], ["Film"]),
        _record("2", ["Film"], ["Music"]),
        _record("3", ["Golf"], ["Golf"]),
    ]
    outputs = [
        "New topic: Golf",
        "Film",
        "New topic: Golf",
    ]
    metrics, details = evaluate_outputs(
        records,
        outputs,
        {"Music", "Golf", "Film"},
    )
    assert metrics["set_hit_at_1"] == 2 / 3
    assert metrics["strict_response_hit_at_1"] == 2 / 3
    assert metrics["prefix_compliance_rate"] == 2 / 3
    assert metrics["valid_label_rate"] == 2 / 3
    assert metrics["novelty_compliance_rate"] == 1 / 3
    assert metrics["per_class_recall"] == {
        "Film": 0.0,
        "Golf": 1.0,
        "Music": 0.0,
    }
    assert details[0]["set_hit_at_1"] is True
    assert details[2]["novelty_compliant"] is False


def test_ranked_recall_and_ndcg() -> None:
    records = [
        _record("1", ["A", "B"], []),
        _record("2", ["C"], []),
    ]
    ranked = [
        [
            {"label": "A", "score": 0.9},
            {"label": "X", "score": 0.8},
            {"label": "B", "score": 0.7},
        ],
        [
            {"label": "X", "score": 0.9},
            {"label": "C", "score": 0.8},
        ],
    ]
    metrics, details = evaluate_ranked_outputs(
        records,
        ranked,
        cutoffs=(1, 3),
    )
    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_at_3"] == 1.0
    assert metrics["hit_at_1"] == 0.5
    assert metrics["hit_at_3"] == 1.0
    assert metrics["mrr_at_3"] == 0.75
    assert metrics["ndcg_at_1"] == 0.5
    assert 0.0 < metrics["ndcg_at_3"] < 1.0
    assert metrics["per_class_recall"]["1"] == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
    }
    assert details[0]["ranked_labels"] == ["A", "X", "B"]
