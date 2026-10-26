from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.evaluation.evaluate_future_interest import (
    evaluate_future_interest_predictions,
)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    benchmark = root / "benchmark.parquet"
    vocabulary = root / "vocabulary.parquet"
    predictions = root / "predictions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_id": "u1",
                    "history_subcategories": ["a"],
                    "future_novel_subcategories": ["c"],
                    "future_exposed_subcategories": ["a", "b", "c"],
                    "history_length": 1,
                    "profile_group_size": 2,
                },
                {
                    "user_id": "u2",
                    "history_subcategories": ["b"],
                    "future_novel_subcategories": [],
                    "future_exposed_subcategories": ["a", "b"],
                    "history_length": 60,
                    "profile_group_size": 1,
                },
            ]
        ),
        benchmark,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"interest_id": 0, "category": "A", "subcategory": "a"},
                {"interest_id": 1, "category": "B", "subcategory": "b"},
                {"interest_id": 2, "category": "C", "subcategory": "c"},
                {"interest_id": 3, "category": "D", "subcategory": "d"},
            ]
        ),
        vocabulary,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_id": "u1",
                    "ranked_labels": ["d", "b", "c", "a"],
                    "ranked_scores": [4.0, 3.0, 2.0, 1.0],
                },
                {
                    "user_id": "u2",
                    "ranked_labels": ["a", "c", "d", "b"],
                    "ranked_scores": [4.0, 3.0, 2.0, 1.0],
                },
            ]
        ),
        predictions,
    )
    return benchmark, vocabulary, predictions


def test_evaluates_both_tracks_from_same_predictions(tmp_path: Path) -> None:
    benchmark, vocabulary, predictions = _write_fixture(tmp_path)
    exposure = evaluate_future_interest_predictions(
        benchmark,
        vocabulary,
        predictions,
        tmp_path / "exposure",
        track="exposure_restricted",
        cutoffs=(1, 2, 3),
    )
    full = evaluate_future_interest_predictions(
        benchmark,
        vocabulary,
        predictions,
        tmp_path / "full",
        track="full_taxonomy",
        cutoffs=(1, 2, 3),
    )

    assert exposure["total_users"] == 2
    assert exposure["positive_target_users"] == 1
    assert exposure["zero_target_users"] == 1
    assert exposure["recall_at_2"] == 1.0
    assert full["recall_at_2"] == 0.0
    assert full["recall_at_3"] == 1.0
    assert (tmp_path / "exposure/details.parquet").is_file()
    assert (tmp_path / "full/metrics.json").is_file()


def test_rejects_incomplete_full_vocabulary_prediction(
    tmp_path: Path,
) -> None:
    benchmark, vocabulary, predictions = _write_fixture(tmp_path)
    rows = pq.read_table(predictions).to_pylist()
    rows[0]["ranked_labels"] = rows[0]["ranked_labels"][:-1]
    rows[0]["ranked_scores"] = rows[0]["ranked_scores"][:-1]
    pq.write_table(pa.Table.from_pylist(rows), predictions)

    with pytest.raises(ValueError, match="each vocabulary label exactly once"):
        evaluate_future_interest_predictions(
            benchmark,
            vocabulary,
            predictions,
            tmp_path / "output",
            track="full_taxonomy",
        )


def test_overwrite_replaces_complete_evaluation_directory(
    tmp_path: Path,
) -> None:
    benchmark, vocabulary, predictions = _write_fixture(tmp_path)
    output = tmp_path / "output"
    evaluate_future_interest_predictions(
        benchmark,
        vocabulary,
        predictions,
        output,
        track="full_taxonomy",
    )
    (output / "stale.txt").write_text("old generation", encoding="utf-8")

    evaluate_future_interest_predictions(
        benchmark,
        vocabulary,
        predictions,
        output,
        track="full_taxonomy",
        overwrite=True,
    )

    assert {path.name for path in output.iterdir()} == {
        "details.parquet",
        "metrics.json",
    }
