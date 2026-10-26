from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_static_user_disjoint_dataset import DATASET_SCHEMA
from new_interest.data.build_static_user_semantic_benchmark import (
    build_static_user_semantic_benchmark,
)


def _history(title: str, subcategory: str) -> list[dict[str, str]]:
    return [
        {
            "news_id": f"N-{title}",
            "category": "History",
            "subcategory": subcategory,
            "title": title,
        }
    ]


def _source_row(
    user_id: str,
    sample_id: str,
    target: str,
    *,
    history_title: str,
    history_label: str,
    split: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "impression_id": sample_id.split(":")[1],
        "user_id": user_id,
        "timestamp": datetime(2023, 11, 14, 22, 13, 20),
        "history": _history(history_title, history_label),
        "target_interest": {"category": "Target", "subcategory": target},
        "target_news": [{"news_id": f"N-{sample_id}", "title": sample_id}],
    }


def _write_fixture(root: Path, *, reverse: bool = False) -> tuple[Path, Path]:
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    vocabulary = [
        {"interest_id": index, "category": "C", "subcategory": label}
        for index, label in enumerate(("a", "b", "c", "d"))
    ]
    pq.write_table(
        pa.Table.from_pylist(vocabulary),
        dataset / "interest_vocabulary.parquet",
    )
    train = [
        _source_row(
            "u1", "train:1:0", "b", history_title="one", history_label="a", split="train"
        ),
        _source_row(
            "u2", "train:2:0", "c", history_title="two", history_label="a", split="train"
        ),
        _source_row(
            "u3", "train:3:0", "d", history_title="three", history_label="b", split="train"
        ),
    ]
    dev = [
        _source_row(
            "u4", "dev:4:0", "b", history_title="four", history_label="a", split="dev"
        ),
        _source_row(
            "u4", "dev:5:0", "c", history_title="four", history_label="a", split="dev"
        ),
        _source_row(
            "u5", "dev:6:0", "d", history_title="five", history_label="b", split="dev"
        ),
    ]
    if reverse:
        train.reverse()
        dev.reverse()
    pq.write_table(
        pa.Table.from_pylist(train, schema=DATASET_SCHEMA),
        dataset / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev, schema=DATASET_SCHEMA),
        dataset / "dev.parquet",
    )
    candidates = root / "candidates.parquet"
    candidate_rows = [
        train[0] if not reverse else train[2],
        _source_row(
            "u1", "train:7:0", "d", history_title="one", history_label="a", split="train"
        ),
        train[1],
        train[2] if not reverse else train[0],
    ]
    if reverse:
        candidate_rows.reverse()
    pq.write_table(
        pa.Table.from_pylist(candidate_rows, schema=DATASET_SCHEMA), candidates
    )
    return dataset, candidates


def test_builds_user_level_multi_positive_data_deterministically(
    tmp_path: Path,
) -> None:
    dataset, candidates = _write_fixture(tmp_path / "first")
    first = tmp_path / "first-output"
    result = build_static_user_semantic_benchmark(
        dataset,
        candidates,
        first,
        validation_fraction=0.5,
        batch_size=2,
    )
    second_dataset, second_candidates = _write_fixture(
        tmp_path / "second", reverse=True
    )
    second = tmp_path / "second-output"
    build_static_user_semantic_benchmark(
        second_dataset,
        second_candidates,
        second,
        validation_fraction=0.5,
        batch_size=1,
    )

    def indexed(output: Path) -> dict[str, dict[str, object]]:
        rows = []
        for split in ("fit", "validation"):
            rows.extend(pq.read_table(output / f"{split}.parquet").to_pylist())
        return {str(row["user_id"]): row for row in rows}

    assert indexed(first) == indexed(second)
    train = indexed(first)
    assert train["u1"]["future_novel_subcategories"] == ["b", "d"]
    assert train["u1"]["history_titles"] == ["one"]
    assert result["statistics"]["train_recovery"][
        "distinct_user_interest_pairs"
    ] == 4

    dev_rows = pq.read_table(first / "dev.parquet").to_pylist()
    assert len(dev_rows) == 2
    assert dev_rows[0]["user_id"] == "u4"
    assert dev_rows[0]["future_novel_subcategories"] == ["b", "c"]
    statistics = json.loads((first / "statistics.json").read_text())
    assert statistics["dev_source_rows"] == 3
    assert statistics["dev_users"] == 2


def test_rejects_overwrite_and_inconsistent_dev_history(tmp_path: Path) -> None:
    dataset, candidates = _write_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    build_static_user_semantic_benchmark(dataset, candidates, output)
    with pytest.raises(FileExistsError):
        build_static_user_semantic_benchmark(dataset, candidates, output)

    rows = pq.read_table(dataset / "dev.parquet").to_pylist()
    rows[1]["history"][0]["title"] = "changed"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=DATASET_SCHEMA),
        dataset / "dev.parquet",
    )
    with pytest.raises(ValueError, match="inconsistent raw History"):
        build_static_user_semantic_benchmark(
            dataset, candidates, tmp_path / "invalid"
        )
