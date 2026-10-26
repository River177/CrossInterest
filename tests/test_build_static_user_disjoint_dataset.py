from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from new_interest.data.build_static_user_disjoint_dataset import (
    DEFAULT_SELECTION_SEED,
    _selection_priority,
    build_static_user_disjoint_dataset,
)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def _news(news_id: str, subcategory: str) -> list[str]:
    return [
        news_id,
        "News",
        subcategory,
        f"{subcategory} title",
        "",
        f"https://example.test/{news_id}",
        "[]",
        "[]",
    ]


def _behavior(
    impression_id: str,
    user_id: str,
    history: list[str],
    impressions: list[tuple[str, int]],
    *,
    minute: int = 0,
) -> list[str]:
    return [
        impression_id,
        user_id,
        f"11/09/2019 12:{minute:02d}:00 PM",
        " ".join(history),
        " ".join(f"{news_id}-{label}" for news_id, label in impressions),
    ]


def _fixture_rows() -> tuple[list[list[str]], list[list[str]]]:
    train = [
        _behavior("t1", "U1", ["N0"], [("N1", 1)], minute=1),
        _behavior(
            "t2",
            "U1",
            ["N0"],
            [("N2", 1), ("N3", 1)],
            minute=2,
        ),
        # U2 has no eligible Train target, but it must still be excluded from Dev.
        _behavior("t3", "U2", ["N1"], [("N1", 1)], minute=3),
        _behavior("t4", "U3", ["N0"], [("N4", 1)], minute=4),
        _behavior("t5", "U5", ["N0"], [("N1", 1)], minute=5),
        _behavior("t6", "U5", ["N0"], [("N1", 1)], minute=6),
    ]
    dev = [
        _behavior("d1", "U1", ["N0"], [("N4", 1)], minute=7),
        _behavior("d2", "U2", ["N1"], [("N3", 1)], minute=8),
        _behavior(
            "d3",
            "U4",
            ["N0"],
            [("N1", 1), ("N2", 1)],
            minute=9,
        ),
        # Earlier Dev clicks are deliberately not appended to this raw history.
        _behavior("d4", "U4", ["N0"], [("N1", 1)], minute=10),
    ]
    return train, dev


def _write_fixture(
    root: Path,
    *,
    reverse_train: bool = False,
) -> None:
    news = [
        _news("N0", "History"),
        _news("N1", "Topic B"),
        _news("N2", "Topic C"),
        _news("N3", "Topic D"),
        _news("N4", "Topic E"),
    ]
    for split in ("train", "dev"):
        _write_tsv(root / split / "news.tsv", news)
    train, dev = _fixture_rows()
    if reverse_train:
        train.reverse()
    _write_tsv(root / "train" / "behaviors.tsv", train)
    _write_tsv(root / "dev" / "behaviors.tsv", dev)
    (root / "MANIFEST.json").write_text(
        json.dumps({"fixture": True}),
        encoding="utf-8",
    )
    (root / "FILTERED_NEWS_IDS.json").write_text(
        json.dumps({"removed_news_ids": []}),
        encoding="utf-8",
    )


def test_builds_one_train_target_per_user_and_user_disjoint_dev(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "output"
    _write_fixture(source)

    result = build_static_user_disjoint_dataset(
        source,
        output,
        staging_batch_size=2,
        output_batch_size=1,
    )

    assert result.train_samples == 3
    assert result.train_users_with_multiple_candidates == 2
    assert result.dev_samples == 3
    assert result.dev_users == 1

    train = pq.read_table(output / "train.parquet").to_pylist()
    dev = pq.read_table(output / "dev.parquet").to_pylist()
    manifest = pq.read_table(
        output / "train_selection_manifest.parquet"
    ).to_pylist()

    assert len(train) == len({row["user_id"] for row in train}) == 3
    assert {row["user_id"] for row in train} == {"U1", "U3", "U5"}
    assert {row["user_id"] for row in dev} == {"U4"}
    assert [row["impression_id"] for row in dev] == ["d3", "d3", "d4"]
    assert [row["target_interest"]["subcategory"] for row in dev] == [
        "Topic B",
        "Topic C",
        "Topic B",
    ]
    assert all([item["news_id"] for item in row["history"]] == ["N0"] for row in dev)

    manifest_by_user = {row["user_id"]: row for row in manifest}
    assert manifest_by_user["U1"]["eligible_candidate_count"] == 3
    assert manifest_by_user["U1"]["distinct_interest_count"] == 3
    assert manifest_by_user["U1"]["eligible_impression_count"] == 2
    assert manifest_by_user["U5"]["eligible_candidate_count"] == 2
    assert manifest_by_user["U5"]["distinct_interest_count"] == 1
    assert manifest_by_user["U5"]["eligible_impression_count"] == 2
    assert all(row["selection_seed"] == DEFAULT_SELECTION_SEED for row in manifest)

    u1_candidates = ["train:t1:1", "train:t2:2", "train:t2:3"]
    expected = min(
        u1_candidates,
        key=lambda sample_id: (
            _selection_priority(DEFAULT_SELECTION_SEED, sample_id),
            sample_id,
        ),
    )
    assert manifest_by_user["U1"]["selected_sample_id"] == expected

    statistics = json.loads(
        (output / "statistics.json").read_text(encoding="utf-8")
    )
    selection = statistics["train"]
    assert selection["raw_users"] == 4
    assert selection["eligible_candidate_rows_before_selection"] == 6
    assert selection["eligible_users"] == 3
    assert selection["selected_samples"] == 3
    assert selection["discarded_candidate_rows"] == 3
    assert selection["users_with_multiple_candidate_rows"] == 2
    assert selection["users_with_multiple_distinct_interests"] == 1
    assert selection["users_with_multiple_eligible_impressions"] == 2
    assert selection["users_with_repeated_same_interest_candidates_only"] == 1
    assert selection["candidate_rows_per_eligible_user"][
        "count_distribution"
    ] == {"1": 1, "2": 1, "3": 1}

    dev_stats = statistics["dev"]
    assert dev_stats["raw_users_overlapping_train"] == 2
    assert dev_stats["excluded_candidate_rows"] == 2
    assert dev_stats["output_users"] == 1
    assert dev_stats["output_samples"] == 3

    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["history_state"] == "raw_behavior_history_only"
    assert metadata["impression_clicks_appended_to_history"] is False
    assert metadata["dev_exclusion_scope"] == (
        "all user IDs in raw Train behaviors"
    )
    for name, expected_hash in metadata["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash


def test_selection_is_independent_of_input_order_and_batch_size(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first-raw"
    second_source = tmp_path / "second-raw"
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    _write_fixture(first_source)
    _write_fixture(second_source, reverse_train=True)

    build_static_user_disjoint_dataset(
        first_source,
        first_output,
        staging_batch_size=1,
        output_batch_size=1,
    )
    build_static_user_disjoint_dataset(
        second_source,
        second_output,
        staging_batch_size=5,
        output_batch_size=3,
    )

    def selection(path: Path) -> dict[str, str]:
        return {
            row["user_id"]: row["selected_sample_id"]
            for row in pq.read_table(
                path / "train_selection_manifest.parquet"
            ).to_pylist()
        }

    assert selection(first_output) == selection(second_output)
    assert {
        row["sample_id"]
        for row in pq.read_table(first_output / "train.parquet").to_pylist()
    } == {
        row["sample_id"]
        for row in pq.read_table(second_output / "train.parquet").to_pylist()
    }


def test_inconsistent_raw_history_and_unknown_news_are_explicit(
    tmp_path: Path,
) -> None:
    inconsistent = tmp_path / "inconsistent"
    _write_fixture(inconsistent)
    train, _ = _fixture_rows()
    train[1][3] = "N4"
    _write_tsv(inconsistent / "train" / "behaviors.tsv", train)
    with pytest.raises(ValueError, match="inconsistent static history"):
        build_static_user_disjoint_dataset(
            inconsistent,
            tmp_path / "bad-history-output",
        )

    unknown = tmp_path / "unknown"
    _write_fixture(unknown)
    _, dev = _fixture_rows()
    dev.append(_behavior("d5", "U6", ["UNKNOWN"], [("N1", 1)]))
    _write_tsv(unknown / "dev" / "behaviors.tsv", dev)
    with pytest.raises(ValueError, match="Unknown news ID"):
        build_static_user_disjoint_dataset(
            unknown,
            tmp_path / "unknown-output",
        )


def test_existing_output_is_protected(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "output"
    _write_fixture(source)
    build_static_user_disjoint_dataset(source, output)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_static_user_disjoint_dataset(source, output)
