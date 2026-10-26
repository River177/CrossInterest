from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from new_interest.data.build_dynamic_dataset import build_dynamic_dataset


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
    timestamp: str,
    history: list[str],
    impressions: list[tuple[str, int]],
) -> list[str]:
    return [
        impression_id,
        user_id,
        timestamp,
        " ".join(history),
        " ".join(f"{news_id}-{label}" for news_id, label in impressions),
    ]


def _write_fixture(root: Path) -> None:
    news_rows = [
        _news_row("N0", "Parent A", "Topic A", "Initial A"),
        _news_row("N1", "Parent B", "Topic B", "Clicked B one"),
        _news_row("N2", "Parent B", "Topic B", "Clicked B two"),
        _news_row("N3", "Parent C", "Topic C", "Clicked C"),
        _news_row("N4", "Parent D", "Topic D", "Clicked D"),
        _news_row("N5", "Parent E", "Topic E", "Unclicked E"),
    ]
    for split in ("train", "dev"):
        _write_tsv(root / split / "news.tsv", news_rows)

    static_history = ["N0"]
    train_rows = [
        _behavior(
            "1",
            "U1",
            "11/09/2019 12:00:00 PM",
            static_history,
            [("N1", 1), ("N5", 0)],
        ),
        _behavior(
            "2",
            "U1",
            "11/10/2019 12:00:00 PM",
            static_history,
            [("N2", 1), ("N3", 1)],
        ),
        _behavior(
            "3",
            "U2",
            "11/10/2019 01:00:00 PM",
            static_history,
            [("N5", 0)],
        ),
    ]
    dev_rows = [
        _behavior(
            "4",
            "U1",
            "11/15/2019 12:00:00 PM",
            static_history,
            [("N3", 1), ("N4", 1)],
        ),
        _behavior(
            "5",
            "U3",
            "11/15/2019 01:00:00 PM",
            static_history,
            [("N4", 1)],
        ),
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


def _subcategories(row: dict[str, object], field: str) -> list[str]:
    return [str(item["subcategory"]) for item in row[field]]


def test_replays_train_into_online_dynamic_dev(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "dynamic"
    _write_fixture(source)

    result = build_dynamic_dataset(
        source,
        output,
        batch_size=2,
        staging_batch_size=2,
    )

    assert result.train_impressions == 3
    assert result.dev_impressions == 2
    assert result.train_new_interest_impressions == 2
    assert result.dev_new_interest_impressions == 2

    train = {
        row["impression_id"]: row
        for row in pq.read_table(output / "train.parquet").to_pylist()
    }
    dev = {
        row["impression_id"]: row
        for row in pq.read_table(output / "dev.parquet").to_pylist()
    }

    assert train["1"]["history_length"] == 1
    assert _subcategories(train["1"], "novel_interests") == ["Topic B"]
    assert not train["1"]["repeated_interests"]

    assert train["2"]["history_length"] == 2
    assert _subcategories(train["2"], "novel_interests") == ["Topic C"]
    assert _subcategories(train["2"], "repeated_interests") == ["Topic B"]

    assert train["3"]["clicked_news"] == []
    assert train["3"]["has_new_interest"] is False

    assert dev["4"]["history_length"] == 4
    assert _subcategories(dev["4"], "novel_interests") == ["Topic D"]
    assert _subcategories(dev["4"], "repeated_interests") == ["Topic C"]
    assert [item["news_id"] for item in dev["4"]["history"]] == [
        "N0",
        "N1",
        "N2",
        "N3",
    ]

    assert dev["5"]["history_length"] == 1
    assert _subcategories(dev["5"], "novel_interests") == ["Topic D"]

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["dev_policy"] == "online_dynamic"
    assert metadata["row_granularity"] == "one row per impression"


def test_same_timestamp_events_do_not_leak_into_each_other(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "dynamic"
    _write_fixture(source)
    same_time = "11/09/2019 12:00:00 PM"
    _write_tsv(
        source / "train" / "behaviors.tsv",
        [
            _behavior("1", "U1", same_time, ["N0"], [("N1", 1)]),
            _behavior("2", "U1", same_time, ["N0"], [("N2", 1)]),
            _behavior(
                "3",
                "U1",
                "11/10/2019 12:00:00 PM",
                ["N0"],
                [("N2", 1)],
            ),
        ],
    )

    build_dynamic_dataset(source, output)
    train = {
        row["impression_id"]: row
        for row in pq.read_table(output / "train.parquet").to_pylist()
    }

    assert _subcategories(train["1"], "novel_interests") == ["Topic B"]
    assert _subcategories(train["2"], "novel_interests") == ["Topic B"]
    assert train["1"]["history_length"] == 1
    assert train["2"]["history_length"] == 1
    assert _subcategories(train["3"], "repeated_interests") == ["Topic B"]
    assert train["3"]["history_length"] == 3


def test_inconsistent_static_history_is_an_explicit_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "dynamic"
    _write_fixture(source)
    _write_tsv(
        source / "train" / "behaviors.tsv",
        [
            _behavior(
                "1",
                "U1",
                "11/09/2019 12:00:00 PM",
                ["N0"],
                [("N1", 1)],
            ),
            _behavior(
                "2",
                "U1",
                "11/10/2019 12:00:00 PM",
                ["N0", "N1"],
                [("N3", 1)],
            ),
        ],
    )

    with pytest.raises(ValueError, match="inconsistent static history"):
        build_dynamic_dataset(source, output)


def test_dev_must_follow_train_for_online_replay(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "dynamic"
    _write_fixture(source)
    _write_tsv(
        source / "dev" / "behaviors.tsv",
        [
            _behavior(
                "4",
                "U1",
                "11/08/2019 12:00:00 PM",
                ["N0"],
                [("N4", 1)],
            )
        ],
    )

    with pytest.raises(ValueError, match="every Train timestamp"):
        build_dynamic_dataset(source, output)


def test_overwrite_replaces_the_complete_output_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "dynamic"
    _write_fixture(source)
    build_dynamic_dataset(source, output)
    (output / "stale.txt").write_text("old generation", encoding="utf-8")

    build_dynamic_dataset(source, output, overwrite=True)

    assert not (output / "stale.txt").exists()
    assert not (output / "events.sqlite3").exists()
    assert {path.name for path in output.iterdir()} == {
        "train.parquet",
        "dev.parquet",
        "interest_vocabulary.parquet",
        "statistics.json",
        "metadata.json",
    }
