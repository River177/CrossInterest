from __future__ import annotations

import numpy as np

from new_interest.experiments.tune_semantic_history_ridge import (
    _normalised_class_weights,
    _pareto_front,
    _select_configuration,
)


def test_class_balance_weights_are_mean_one_and_beta_one_equalises_classes() -> None:
    counts = np.asarray([100, 10, 1], dtype=np.float64)

    beta_zero = _normalised_class_weights(counts, 0.0)
    beta_one = _normalised_class_weights(counts, 1.0)

    np.testing.assert_allclose(beta_zero, np.ones(3))
    np.testing.assert_allclose(np.dot(counts, beta_one) / counts.sum(), 1.0)
    np.testing.assert_allclose(counts * beta_one, np.repeat(counts.sum() / 3, 3))


def test_pareto_front_removes_dominated_configurations() -> None:
    rows = [
        {
            "ridge_alpha": 1.0,
            "class_balance_beta": 0.0,
            "interest_macro_recall_at_10": 0.20,
            "interest_macro_ndcg_at_10": 0.10,
        },
        {
            "ridge_alpha": 3.0,
            "class_balance_beta": 0.5,
            "interest_macro_recall_at_10": 0.21,
            "interest_macro_ndcg_at_10": 0.11,
        },
        {
            "ridge_alpha": 10.0,
            "class_balance_beta": 1.0,
            "interest_macro_recall_at_10": 0.22,
            "interest_macro_ndcg_at_10": 0.09,
        },
    ]

    frontier = _pareto_front(rows)

    assert [(row["ridge_alpha"], row["class_balance_beta"]) for row in frontier] == [
        (3.0, 0.5),
        (10.0, 1.0),
    ]


def test_selection_maximises_ndcg_within_recall_tolerance() -> None:
    rows = [
        {
            "ridge_alpha": 1.0,
            "class_balance_beta": 0.25,
            "interest_macro_recall_at_10": 0.220,
            "interest_macro_ndcg_at_10": 0.100,
        },
        {
            "ridge_alpha": 3.0,
            "class_balance_beta": 0.5,
            "interest_macro_recall_at_10": 0.219,
            "interest_macro_ndcg_at_10": 0.120,
        },
        {
            "ridge_alpha": 10.0,
            "class_balance_beta": 0.75,
            "interest_macro_recall_at_10": 0.215,
            "interest_macro_ndcg_at_10": 0.130,
        },
    ]

    selected, frontier = _select_configuration(
        rows, global_recall=0.10, recall_tolerance=0.002
    )

    assert len(frontier) == 3
    assert selected["ridge_alpha"] == 3.0
    assert selected["class_balance_beta"] == 0.5
