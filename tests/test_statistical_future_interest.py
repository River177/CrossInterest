from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.statistical_future_interest import (
    fit_statistical_future_interest,
    predict_statistical_future_interest,
)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    vocabulary = root / "vocabulary.parquet"
    train = root / "train.parquet"
    test = root / "test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"interest_id": 0, "category": "A", "subcategory": "a"},
                {"interest_id": 1, "category": "B", "subcategory": "b"},
                {"interest_id": 2, "category": "C", "subcategory": "c"},
            ]
        ),
        vocabulary,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_id": "u1",
                    "history_unique_subcategories": ["a"],
                    "future_novel_subcategories": ["b"],
                },
                {
                    "user_id": "u2",
                    "history_unique_subcategories": ["a"],
                    "future_novel_subcategories": ["b"],
                },
                {
                    "user_id": "u3",
                    "history_unique_subcategories": ["c"],
                    "future_novel_subcategories": ["a"],
                },
            ]
        ),
        train,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "user_id": "u4",
                    "history_subcategories": ["a", "a"],
                }
            ]
        ),
        test,
    )
    return train, test, vocabulary


def test_statistical_baselines_fit_train_only_and_rank_all_labels(
    tmp_path: Path,
) -> None:
    train, test, vocabulary = _write_fixture(tmp_path)
    model = fit_statistical_future_interest(train, vocabulary, alpha=1.0)
    result = predict_statistical_future_interest(
        model,
        test,
        tmp_path / "output",
    )

    assert model.train_users == 3
    assert model.target_user_counts["b"] == 2
    assert result["users"] == 1
    global_row = pq.read_table(
        tmp_path / "output/global_popularity/predictions.parquet"
    ).to_pylist()[0]
    conditional_row = pq.read_table(
        tmp_path / "output/conditional_frequency/predictions.parquet"
    ).to_pylist()[0]
    assert global_row["ranked_labels"][0] == "b"
    assert conditional_row["ranked_labels"][0] == "b"
    assert set(global_row["ranked_labels"]) == {"a", "b", "c"}
