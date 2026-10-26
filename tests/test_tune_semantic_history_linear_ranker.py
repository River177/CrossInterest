from __future__ import annotations

import numpy as np
import pytest

from new_interest.experiments.tune_semantic_history_linear_ranker import (
    _masked_weighted_cross_entropy,
    _select_configuration,
    _single_target_metrics,
)
from new_interest.experiments.tune_semantic_history_ridge import SingleTargetSplit


def test_masked_weighted_cross_entropy_removes_history_from_denominator() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 5.0, 1.0], [2.0, 0.0, 1.0]])
    targets = torch.tensor([2, 0])
    history = torch.tensor([[False, True, False], [False, False, True]])
    weights = torch.tensor([1.0, 3.0])

    actual = _masked_weighted_cross_entropy(logits, targets, history, weights)
    first = -torch.log_softmax(torch.tensor([0.0, 1.0]), dim=0)[1]
    second = -torch.log_softmax(torch.tensor([2.0, 0.0]), dim=0)[0]
    expected = (first + 3.0 * second) / 4.0

    torch.testing.assert_close(actual, expected)


def test_masked_weighted_cross_entropy_rejects_history_target() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="target is excluded"):
        _masked_weighted_cross_entropy(
            torch.zeros((1, 3)),
            torch.tensor([1]),
            torch.tensor([[False, True, False]]),
            torch.ones(1),
        )


def test_linear_selection_applies_interest_and_user_recall_constraints() -> None:
    rows = [
        {
            "class_balance_beta": 0.75,
            "l2": 0.0,
            "temperature": 1.0,
            "learning_rate": 0.01,
            "epoch": 4,
            "interest_macro_recall_at_10": 0.260,
            "interest_macro_ndcg_at_10": 0.130,
            "user_macro_recall_at_10": 0.300,
        },
        {
            "class_balance_beta": 0.5,
            "l2": 0.001,
            "temperature": 1.0,
            "learning_rate": 0.01,
            "epoch": 6,
            "interest_macro_recall_at_10": 0.259,
            "interest_macro_ndcg_at_10": 0.125,
            "user_macro_recall_at_10": 0.350,
        },
        {
            "class_balance_beta": 0.25,
            "l2": 0.001,
            "temperature": 0.5,
            "learning_rate": 0.03,
            "epoch": 8,
            "interest_macro_recall_at_10": 0.250,
            "interest_macro_ndcg_at_10": 0.140,
            "user_macro_recall_at_10": 0.390,
        },
    ]

    selected, frontier = _select_configuration(
        rows,
        global_recall=0.12,
        balanced_ridge_user_recall=0.34,
        recall_tolerance=0.002,
        user_recall_tolerance=0.0,
    )

    assert len(frontier) == 2
    assert selected["class_balance_beta"] == 0.5
    assert selected["epoch"] == 6


def test_linear_selection_fails_when_user_floor_excludes_all_pareto_points() -> None:
    row = {
        "class_balance_beta": 0.5,
        "l2": 0.0,
        "temperature": 1.0,
        "learning_rate": 0.01,
        "epoch": 4,
        "interest_macro_recall_at_10": 0.2,
        "interest_macro_ndcg_at_10": 0.1,
        "user_macro_recall_at_10": 0.3,
    }
    with pytest.raises(ValueError, match="User-Recall constraints"):
        _select_configuration(
            [row],
            global_recall=0.1,
            balanced_ridge_user_recall=0.4,
            recall_tolerance=0.002,
            user_recall_tolerance=0.0,
        )


def test_fast_top_k_metrics_preserve_stable_label_ties() -> None:
    split = SingleTargetSplit(
        ("u1", "u2"),
        np.asarray([[1.0], [1.0]], dtype=np.float32),
        np.asarray([0, 2], dtype=np.int64),
        np.zeros((2, 4), dtype=np.bool_),
    )
    weights = np.zeros((1, 4), dtype=np.float32)

    metrics = _single_target_metrics(split, weights, k=2)

    assert metrics["interest_macro_recall_at_2"] == 0.5
    assert metrics["user_macro_recall_at_2"] == 0.5
