from __future__ import annotations

import pytest

from new_interest.evaluation.ranking_metrics import (
    RankingExample,
    RankingPrediction,
    evaluate_rankings,
    paired_bootstrap_metric_delta,
)


def _examples() -> list[RankingExample]:
    return [
        RankingExample(
            example_id="u1",
            positive_labels=("b", "c"),
            candidate_labels=("a", "b", "c", "d"),
            attributes={"target_count": "2", "history_length": "short"},
        ),
        RankingExample(
            example_id="u2",
            positive_labels=("a",),
            candidate_labels=("a", "c", "d"),
            attributes={"target_count": "1", "history_length": "long"},
        ),
    ]


def test_evaluate_rankings_computes_user_and_macro_metrics() -> None:
    result = evaluate_rankings(
        _examples(),
        [
            RankingPrediction(
                example_id="u1",
                ranked_labels=("b", "a", "c", "d"),
                ranked_scores=(4.0, 3.0, 2.0, 1.0),
            ),
            RankingPrediction(
                example_id="u2",
                ranked_labels=("c", "a", "d"),
                ranked_scores=(3.0, 2.0, 1.0),
            ),
        ],
        cutoffs=(1, 2, 3),
    )

    assert result.metrics["samples"] == 2
    assert result.metrics["recall_at_1"] == 0.25
    assert result.metrics["recall_at_2"] == 0.75
    assert result.metrics["hit_at_1"] == 0.5
    assert result.metrics["macro_recall_at_k"]["1"] == pytest.approx(
        1 / 3
    )
    assert result.metrics["macro_recall_at_k"]["2"] == pytest.approx(
        2 / 3
    )
    assert result.metrics["by_attribute"]["target_count"]["1"][
        "samples"
    ] == 1
    assert result.details[0]["example_id"] == "u1"


def test_evaluate_rankings_requires_exact_candidate_coverage() -> None:
    with pytest.raises(ValueError, match="exactly cover candidates"):
        evaluate_rankings(
            _examples()[:1],
            [
                RankingPrediction(
                    example_id="u1",
                    ranked_labels=("b", "a", "c"),
                    ranked_scores=(3.0, 2.0, 1.0),
                )
            ],
        )


def test_evaluate_rankings_rejects_positive_outside_candidates() -> None:
    with pytest.raises(ValueError, match="positives missing"):
        evaluate_rankings(
            [
                RankingExample(
                    example_id="u1",
                    positive_labels=("missing",),
                    candidate_labels=("a",),
                )
            ],
            [
                RankingPrediction(
                    example_id="u1",
                    ranked_labels=("a",),
                    ranked_scores=(1.0,),
                )
            ],
        )


def test_paired_bootstrap_returns_metric_intervals() -> None:
    baseline = [
        {"example_id": "u1", "recall_at_1": 0.0},
        {"example_id": "u2", "recall_at_1": 1.0},
    ]
    comparison = [
        {"example_id": "u1", "recall_at_1": 1.0},
        {"example_id": "u2", "recall_at_1": 1.0},
    ]
    result = paired_bootstrap_metric_delta(
        baseline,
        comparison,
        metric_names=("recall_at_1",),
        replicates=2_000,
        seed=7,
    )
    assert result["metrics"]["recall_at_1"]["observed_delta"] == 0.5
