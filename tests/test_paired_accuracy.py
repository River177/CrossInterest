from __future__ import annotations

import pytest

from new_interest.evaluation.compare_paired_accuracy import (
    compare_prediction_rows,
    exact_mcnemar_p_value,
)


def _row(impression_id: str, correct: bool) -> dict[str, object]:
    return {
        "impression_id": impression_id,
        "set_hit_at_1": correct,
    }


def test_paired_comparison_is_deterministic_and_reports_mcnemar() -> None:
    baseline = [
        _row("1", True),
        _row("2", False),
        _row("3", False),
        _row("4", False),
    ]
    comparison = [
        _row("3", True),
        _row("1", False),
        _row("4", True),
        _row("2", True),
    ]

    first = compare_prediction_rows(
        baseline,
        comparison,
        bootstrap_replicates=2000,
        seed=17,
    )
    second = compare_prediction_rows(
        list(reversed(baseline)),
        list(reversed(comparison)),
        bootstrap_replicates=2000,
        seed=17,
    )

    assert first == second
    assert first["baseline_correct_count"] == 1
    assert first["comparison_correct_count"] == 3
    assert first["accuracy_delta"] == 0.5
    assert first["baseline_only_correct"] == 1
    assert first["comparison_only_correct"] == 3
    assert first["mcnemar_exact_p_value"] == pytest.approx(0.625)
    assert first["bootstrap"]["replicates"] == 2000


@pytest.mark.parametrize(
    ("baseline", "comparison", "message"),
    [
        (
            [_row("1", True), _row("1", False)],
            [_row("1", True)],
            "duplicate",
        ),
        (
            [_row("1", True)],
            [_row("2", True)],
            "same impression IDs",
        ),
    ],
)
def test_alignment_failures_are_rejected(
    baseline: list[dict[str, object]],
    comparison: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_prediction_rows(
            baseline,
            comparison,
            bootstrap_replicates=2000,
            seed=1,
        )


def test_exact_mcnemar_handles_no_discordant_pairs() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
