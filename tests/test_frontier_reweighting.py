from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.semantic_id_crossfit import (
    build_semantic_id_frontier_weights,
    compute_frontier_weights,
)


def _score(source_id: str, label: str, difficulty: float) -> dict[str, object]:
    return {
        "source_sample_id": source_id,
        "user_id": f"u-{source_id}",
        "target_subcategory": label,
        "fold": int(difficulty) % 2,
        "prior_adjusted_nll": difficulty,
        "prior_adjusted_margin": difficulty - 0.5,
    }


def test_frontier_weights_are_symmetric_and_class_normalized() -> None:
    rows = [
        _score(f"{label}{index}", label, offset + float(index))
        for label, offset in (("A", 0.0), ("B", 10.0))
        for index in range(5)
    ]
    weights = compute_frontier_weights(rows)
    for label in ("A", "B"):
        selected = sorted(
            (row for row in weights if row["target_subcategory"] == label),
            key=lambda row: int(row["within_class_rank"]),
        )
        values = [float(row["sample_weight"]) for row in selected]
        assert sum(values) / len(values) == pytest.approx(1.0)
        assert values[0] == pytest.approx(values[-1])
        assert values[1] == pytest.approx(values[-2])
        assert values[2] > values[1] > values[0]
        assert max(values) / min(values) == pytest.approx(2.0)


def test_frontier_builder_filters_full_oof_to_exact_train_subset(
    tmp_path: Path,
) -> None:
    score_file = tmp_path / "scores.parquet"
    train_file = tmp_path / "train.parquet"
    rows = [
        _score(f"{label}{index}", label, offset + float(index))
        for label, offset in (("A", 0.0), ("B", 10.0))
        for index in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(rows), score_file)
    selected = [row for row in rows if int(row["source_sample_id"][1:]) < 3]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_sample_id": row["source_sample_id"],
                    "user_id": row["user_id"],
                    "target_subcategory": row["target_subcategory"],
                }
                for row in selected
            ]
        ),
        train_file,
    )
    output = tmp_path / "output"
    summary = build_semantic_id_frontier_weights(
        [score_file], train_file, output
    )
    assert summary["rows"] == 6
    assert summary["classes"] == 2
    assert summary["class_counts"] == {"A": 3, "B": 3}
    weights = pq.read_table(output / "sample_weights.parquet").to_pylist()
    assert {row["source_sample_id"] for row in weights} == {
        row["source_sample_id"] for row in selected
    }
    assert summary["formula"]["shape"] == (
        "symmetric triangular learnability frontier"
    )
