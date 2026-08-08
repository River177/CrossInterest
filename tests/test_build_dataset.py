from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from new_interest.data.build_dataset import build_dataset


def _news_row(
    news_id: str,
    category: str,
    subcategory: str,
    title: str,
) -> list[str]:
    return [
        news_id,
        category,
        subcategory,
        title,
        "",
        f"https://example.test/{news_id}",
        "[]",
        "[]",
    ]


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def _behavior(
    impression_id: str,
    user_id: str,
    history: list[str],
    impressions: list[tuple[str, int]],
) -> list[str]:
    return [
        impression_id,
        user_id,
        "11/09/2019 12:00:00 PM",
        " ".join(history),
        " ".join(f"{news_id}-{label}" for news_id, label in impressions),
    ]


def _write_fixture(root: Path) -> None:
    news_rows = [
        _news_row("N0", "Category A", "Subcategory A", "History A"),
        _news_row("N1", "Category B", "Subcategory B", "New B one"),
        _news_row("N2", "Category C", "Subcategory C", "New C"),
        _news_row("N3", "Category B", "Subcategory B", "New B two"),
        _news_row("N4", "Category D", "Subcategory D", '"Quoted title'),
    ]
    for split in ("train", "dev"):
        _write_tsv(root / split / "news.tsv", news_rows)

    train_rows = [
        _behavior("1", "U1", ["N0"], [("N1", 1), ("N2", 0)]),
        _behavior("2", "U2", ["N0", "N1"], [("N3", 1)]),
        _behavior("3", "U3", ["N0"], [("N1", 1), ("N3", 1)]),
        _behavior("4", "U4", ["N0"], [("N1", 1), ("N2", 1)]),
    ]
    dev_rows = [
        _behavior("5", "U5", ["N0"], [("N4", 1)]),
    ]
    _write_tsv(root / "train" / "behaviors.tsv", train_rows)
    _write_tsv(root / "dev" / "behaviors.tsv", dev_rows)
    (root / "MANIFEST.json").write_text(
        json.dumps({"metadata": {"test": True}}),
        encoding="utf-8",
    )
    (root / "FILTERED_NEWS_IDS.json").write_text(
        json.dumps({"removed_news_ids": []}),
        encoding="utf-8",
    )


def test_builds_one_sample_per_distinct_new_interest(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_fixture(source)

    result = build_dataset(source, output, batch_size=2)

    assert result.train_samples == 4
    assert result.dev_samples == 1
    train = pq.read_table(output / "train.parquet").to_pylist()
    dev = pq.read_table(output / "dev.parquet").to_pylist()

    assert {row["impression_id"] for row in train} == {"1", "3", "4"}
    impression_three = [
        row for row in train if row["impression_id"] == "3"
    ]
    assert len(impression_three) == 1
    assert [item["news_id"] for item in impression_three[0]["target_news"]] == [
        "N1",
        "N3",
    ]
    assert len([row for row in train if row["impression_id"] == "4"]) == 2

    for row in [*train, *dev]:
        history_interests = {
            (item["category"], item["subcategory"]) for item in row["history"]
        }
        target = (
            row["target_interest"]["category"],
            row["target_interest"]["subcategory"],
        )
        assert target not in history_interests
        assert row["target_news"]
        assert set(row["history"][0]) == {
            "news_id",
            "category",
            "subcategory",
            "title",
        }

    assert dev[0]["target_news"][0]["title"] == '"Quoted title'
    assert pq.ParquetFile(output / "interest_vocabulary.parquet").metadata.num_rows == 4


def test_negative_impressions_do_not_affect_samples(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_fixture(source)
    build_dataset(source, output)
    rows = pq.read_table(output / "train.parquet").to_pylist()
    first = next(row for row in rows if row["impression_id"] == "1")
    assert first["target_interest"]["subcategory"] == "Subcategory B"


def test_unknown_news_id_is_an_explicit_error(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_fixture(source)
    _write_tsv(
        source / "dev" / "behaviors.tsv",
        [_behavior("9", "U9", ["UNKNOWN"], [("N1", 1)])],
    )
    with pytest.raises(ValueError, match="Unknown news ID"):
        build_dataset(source, output)
