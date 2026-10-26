from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from new_interest.data.build_future_interest_benchmark import (
    build_future_interest_benchmark,
)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def _news(news_id: str, subcategory: str, title: str) -> list[str]:
    return [news_id, "News", subcategory, title, "", "", "[]", "[]"]


def _behavior(
    impression_id: str,
    user_id: str,
    timestamp: str,
    history: str,
    candidates: str,
) -> list[str]:
    return [impression_id, user_id, timestamp, history, candidates]


def _write_fixture(root: Path) -> None:
    news = [
        _news("N0", "History", "History title"),
        _news("N1", "Sports", "Sports title"),
        _news("N2", "Politics", "Politics title"),
        _news("N3", "Travel", "Travel title"),
        _news("N4", "Science", "Science title"),
    ]
    for split in ("train", "dev"):
        _write_tsv(root / split / "news.tsv", news)
    _write_tsv(
        root / "train" / "behaviors.tsv",
        [
            _behavior(
                "t1",
                "U1",
                "11/09/2019 12:00:00 PM",
                "N0",
                "N1-1 N1-1 N2-0",
            ),
            _behavior(
                "t2",
                "U2",
                "11/09/2019 12:00:00 PM",
                "N0",
                "N2-0",
            ),
        ],
    )
    _write_tsv(
        root / "dev" / "behaviors.tsv",
        [
            _behavior(
                "d1",
                "U1",
                "11/09/2019 12:00:00 PM",
                "N0",
                "N3-1 N2-0 N2-1",
            ),
            _behavior(
                "d2",
                "U3",
                "11/10/2019 12:00:00 PM",
                "N1",
                "N2-1",
            ),
        ],
    )


def _build(root: Path, output: Path, *, overwrite: bool = False):
    return build_future_interest_benchmark(
        {
            "input_dir": str(root),
            "output_dir": str(output),
            "expected_interest_count": 5,
            "split_seed": 20260812,
            "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "overwrite": overwrite,
        }
    )


def test_build_preserves_repeats_masks_and_user_granularity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "benchmark"
    _write_fixture(raw)
    result = _build(raw, output)

    assert result.users == 3
    rows = {
        row["user_id"]: row
        for split in ("train", "validation", "test")
        for row in pq.read_table(output / f"{split}.parquet").to_pylist()
    }
    assert set(rows) == {"U1", "U2", "U3"}
    assert rows["U1"]["future_clicked_news_ids"] == ["N1", "N1", "N3", "N2"]
    assert rows["U1"]["future_clicked_subcategories"] == [
        "Sports",
        "Sports",
        "Travel",
        "Politics",
    ]
    assert rows["U1"]["future_novel_subcategories"] == [
        "Politics",
        "Sports",
        "Travel",
    ]
    assert rows["U1"]["future_exposed_subcategories"] == [
        "Politics",
        "Sports",
        "Travel",
    ]
    assert rows["U1"]["future_exposed_novel_subcategories"] == [
        "Politics",
        "Sports",
        "Travel",
    ]
    assert rows["U1"]["future_exposed_unclicked_subcategories"] == []
    assert "Politics" not in rows["U1"]["future_exposed_unclicked_subcategories"]
    assert rows["U1"]["has_future_novel_interest"] is True
    assert rows["U2"]["has_future_novel_interest"] is False
    assert rows["U2"]["future_clicked_news_ids"] == []
    assert rows["U1"]["future_impression_count"] == 2


def test_profile_groups_are_atomic_and_users_are_not_duplicated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "benchmark"
    _write_fixture(raw)
    _build(raw, output)

    manifest = pq.read_table(output / "split_manifest.parquet").to_pylist()
    assert all(len(row["user_ids"]) == row["profile_group_size"] for row in manifest)
    grouped = {user_id: row["split"] for row in manifest for user_id in row["user_ids"]}
    assert grouped["U1"] == grouped["U2"]
    rows = [
        row
        for split in ("train", "validation", "test")
        for row in pq.read_table(output / f"{split}.parquet").to_pylist()
    ]
    assert len(rows) == len({row["user_id"] for row in rows}) == 3
    assert all(row["profile_group_size"] == 2 for row in rows if row["user_id"] in {"U1", "U2"})


def test_output_is_deterministic_and_protected_without_overwrite(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "benchmark"
    _write_fixture(raw)
    _build(raw, output)
    first = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in (
            "train.parquet",
            "validation.parquet",
            "test.parquet",
            "interest_vocabulary.parquet",
            "split_manifest.parquet",
            "statistics.json",
            "metadata.json",
        )
    }
    with pytest.raises(FileExistsError):
        _build(raw, output)
    _build(raw, output, overwrite=True)
    second = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in first
    }
    assert first == second

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_sha256"]["statistics.json"] == (
        hashlib.sha256((output / "statistics.json").read_bytes()).hexdigest()
    )
    statistics = json.loads(
        (output / "statistics.json").read_text(encoding="utf-8")
    )
    assert set(statistics["per_split"]) == {"train", "validation", "test"}
    assert statistics["train_only_audit"] == statistics["per_split"]["train"]
    for split_stats in statistics["per_split"].values():
        assert "per_label_distinct_user_positive_support" in split_stats
        assert "per_label_distinct_user_novel_exposure" in split_stats


def test_inconsistent_history_is_explicit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_fixture(raw)
    _write_tsv(
        raw / "dev" / "behaviors.tsv",
        [
            _behavior(
                "d1",
                "U1",
                "11/09/2019 12:00:00 PM",
                "N4",
                "N3-1",
            )
        ],
    )
    with pytest.raises(ValueError, match="inconsistent static history"):
        _build(raw, tmp_path / "benchmark")
