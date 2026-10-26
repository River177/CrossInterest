from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_dataset import DATASET_SCHEMA, VOCABULARY_SCHEMA
from new_interest.data.build_static_user_disjoint_sft_matched_dataset import (
    build_static_user_disjoint_sft_matched_dataset,
)


def _history(prefix: str, length: int) -> list[dict[str, str]]:
    return [
        {
            "news_id": f"N-{prefix}-{index}",
            "category": "History",
            "subcategory": "history-topic",
            "title": f"{prefix} title {index}",
        }
        for index in range(length)
    ]


def _row(
    sample_id: str,
    user_id: str,
    target: str,
    *,
    split: str,
    history_prefix: str,
    history_length: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "impression_id": sample_id.split(":")[1],
        "user_id": user_id,
        "timestamp": datetime(2026, 8, 17, 0, 0, 0),
        "history": _history(history_prefix, history_length),
        "target_interest": {"category": "Target", "subcategory": target},
        "target_news": [{"news_id": f"N-{sample_id}", "title": sample_id}],
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    sft = root / "sft"
    source.mkdir(parents=True)
    (sft / "train" / "behavior_only").mkdir(parents=True)
    (sft / "train" / "interest_only").mkdir(parents=True)

    train = [
        _row(
            "train:1:0",
            "u1",
            "a",
            split="train",
            history_prefix="u1",
            history_length=3,
        ),
        _row(
            "train:2:0",
            "u2",
            "b",
            split="train",
            history_prefix="u2",
            history_length=1,
        ),
        _row(
            "train:3:0",
            "u3",
            "a",
            split="train",
            history_prefix="u3",
            history_length=4,
        ),
        _row(
            "train:4:0",
            "u4",
            "b",
            split="train",
            history_prefix="u4",
            history_length=2,
        ),
        _row(
            "train:5:0",
            "unselected",
            "c",
            split="train",
            history_prefix="unselected",
            history_length=1,
        ),
    ]
    dev = [
        _row(
            "dev:6:0",
            "u5",
            "a",
            split="dev",
            history_prefix="u5",
            history_length=3,
        ),
        _row(
            "dev:7:0",
            "u5",
            "c",
            split="dev",
            history_prefix="u5",
            history_length=3,
        ),
        _row(
            "dev:8:0",
            "u6",
            "c",
            split="dev",
            history_prefix="u6",
            history_length=1,
        ),
        _row(
            "dev:9:0",
            "u7",
            "b",
            split="dev",
            history_prefix="u7",
            history_length=4,
        ),
    ]
    pq.write_table(
        pa.Table.from_pylist(train, schema=DATASET_SCHEMA),
        source / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev, schema=DATASET_SCHEMA),
        source / "dev.parquet",
    )
    vocabulary = [
        {"interest_id": index, "category": "Target", "subcategory": label}
        for index, label in enumerate(("a", "b", "c"))
    ]
    pq.write_table(
        pa.Table.from_pylist(vocabulary, schema=VOCABULARY_SCHEMA),
        source / "interest_vocabulary.parquet",
    )

    selection_order = ("train:4:0", "train:1:0", "train:3:0", "train:2:0")
    by_id = {str(row["sample_id"]): row for row in train}
    manifest = []
    task_rows = []
    for rank, sample_id in enumerate(selection_order):
        row = by_id[sample_id]
        target = str(row["target_interest"]["subcategory"])
        manifest.append(
            {
                "source_sample_id": sample_id,
                "target_subcategory": target,
                "class_rank": 0 if target == "a" else 1,
                "within_class_rank": rank // 2,
                "global_rank": rank,
                "included_sizes": [4],
            }
        )
        task_rows.append(
            {
                "source_sample_id": sample_id,
                "user_id": row["user_id"],
                "impression_id": row["impression_id"],
                "history_length": min(len(row["history"]), 2),
                "target_subcategory": target,
            }
        )
    pq.write_table(pa.Table.from_pylist(manifest), sft / "subset_manifest.parquet")
    for task_type in ("behavior_only", "interest_only"):
        pq.write_table(
            pa.Table.from_pylist(list(reversed(task_rows))),
            sft / "train" / task_type / "4.parquet",
        )
    return source, sft


def test_builds_sft_matched_h50_dataset_and_filters_dev(tmp_path: Path) -> None:
    source, sft = _write_fixture(tmp_path)
    output = tmp_path / "output"

    result = build_static_user_disjoint_sft_matched_dataset(
        source,
        sft,
        output,
        train_size=4,
        history_limit=2,
        batch_size=2,
    )

    assert result.train_samples == 4
    assert result.target_interests == 2
    assert result.dev_samples == 2
    assert result.dev_users == 2
    assert result.dev_target_interests == 2

    train = pq.read_table(output / "train.parquet").to_pylist()
    assert [row["sample_id"] for row in train] == [
        "train:4:0",
        "train:1:0",
        "train:3:0",
        "train:2:0",
    ]
    assert all(len(row["history"]) <= 2 for row in train)
    assert [item["news_id"] for item in train[1]["history"]] == [
        "N-u1-1",
        "N-u1-2",
    ]

    dev = pq.read_table(output / "dev.parquet").to_pylist()
    assert [row["sample_id"] for row in dev] == ["dev:6:0", "dev:9:0"]
    assert all(len(row["history"]) <= 2 for row in dev)
    exclusions = pq.read_table(
        output / "dev_exclusion_manifest.parquet"
    ).to_pylist()
    assert [row["sample_id"] for row in exclusions] == ["dev:7:0", "dev:8:0"]

    full_vocabulary = pq.read_table(
        output / "interest_vocabulary.parquet"
    )["subcategory"].to_pylist()
    target_vocabulary = pq.read_table(
        output / "target_interest_vocabulary.parquet"
    )["subcategory"].to_pylist()
    assert full_vocabulary == ["a", "b", "c"]
    assert target_vocabulary == ["a", "b"]

    statistics = json.loads((output / "statistics.json").read_text())
    assert statistics["train"]["source_rows"] == 5
    assert statistics["train"]["users_truncated_to_history_limit"] == 2
    assert statistics["dev"]["source_rows"] == 4
    assert statistics["dev"]["excluded_rows"] == 2
    assert statistics["dev"]["users_removed_entirely"] == 1
    metadata = json.loads((output / "metadata.json").read_text())
    assert set(metadata["artifact_sha256"]) == {
        "train.parquet",
        "dev.parquet",
        "interest_vocabulary.parquet",
        "target_interest_vocabulary.parquet",
        "train_selection_manifest.parquet",
        "dev_exclusion_manifest.parquet",
        "statistics.json",
    }


def test_refuses_overwrite_and_detects_sft_identity_mismatch(
    tmp_path: Path,
) -> None:
    source, sft = _write_fixture(tmp_path)
    output = tmp_path / "output"
    build_static_user_disjoint_sft_matched_dataset(
        source,
        sft,
        output,
        train_size=4,
        history_limit=2,
    )
    with pytest.raises(FileExistsError):
        build_static_user_disjoint_sft_matched_dataset(
            source,
            sft,
            output,
            train_size=4,
            history_limit=2,
        )

    interest_path = sft / "train" / "interest_only" / "4.parquet"
    rows = pq.read_table(interest_path).to_pylist()
    rows[0]["user_id"] = "wrong-user"
    pq.write_table(pa.Table.from_pylist(rows), interest_path)
    with pytest.raises(ValueError, match="Behavior/Interest mismatch"):
        build_static_user_disjoint_sft_matched_dataset(
            source,
            sft,
            tmp_path / "invalid",
            train_size=4,
            history_limit=2,
        )
