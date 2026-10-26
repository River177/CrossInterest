from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_behavior_interest_sft_dataset import (
    build_behavior_interest_sft_dataset,
)
from new_interest.data.build_dataset import DATASET_SCHEMA
from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.build_sft_subsets import GROUPED_EVAL_SCHEMA


def _history(index: int) -> list[dict[str, str]]:
    return [
        {
            "news_id": f"N{index}a",
            "category": "Parent One",
            "subcategory": "History One",
            "title": f"Clicked headline {index}a",
        },
        {
            "news_id": f"N{index}b",
            "category": "Parent Two",
            "subcategory": "History Two",
            "title": f"Clicked headline {index}b",
        },
    ]


def _source(
    split: str,
    impression_id: str,
    user_id: str,
    target: str,
    history: list[dict[str, str]],
    index: int,
) -> dict[str, object]:
    return {
        "sample_id": f"{split}:{impression_id}:{index}",
        "split": split,
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": datetime(2019, 11, 13, 12, 34, index),
        "history": history,
        "target_interest": {
            "category": "Target Parent",
            "subcategory": target,
        },
        "target_news": [{"news_id": f"T{index}", "title": "Target"}],
    }


def _paired_train_row(
    source: dict[str, object],
    task_type: str,
) -> dict[str, object]:
    target = str(source["target_interest"]["subcategory"])
    response = f"New topic: {target}"
    prompt = f"{task_type} prompt"
    return {
        "sample_id": f"{source['sample_id']}:{task_type}",
        "source_sample_id": str(source["sample_id"]),
        "split": "train",
        "task_type": task_type,
        "impression_id": str(source["impression_id"]),
        "user_id": str(source["user_id"]),
        "timestamp": source["timestamp"],
        "history_length": len(source["history"]),
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target,
    }


def _paired_dev_row(
    source: dict[str, object],
    targets: list[str],
    task_type: str,
) -> dict[str, object]:
    return {
        "impression_id": str(source["impression_id"]),
        "user_id": str(source["user_id"]),
        "timestamp": source["timestamp"],
        "history_length": len(source["history"]),
        "prompt": f"{task_type} prompt",
        "acceptable_responses": [f"New topic: {target}" for target in targets],
        "acceptable_subcategories": targets,
        "history_subcategories": [
            str(item["subcategory"]) for item in source["history"]
        ],
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    matched = root / "matched"
    paired = root / "paired"
    matched.mkdir(parents=True)
    train = [
        _source("train", str(index), f"U{index}", target, _history(index), index)
        for index, target in enumerate(
            ["Target A", "Target A", "Target B", "Target B"],
            start=1,
        )
    ]
    dev_history_one = _history(10)
    dev_history_two = _history(11)
    dev = [
        _source("dev", "10", "UD10", "Target A", dev_history_one, 10),
        _source("dev", "10", "UD10", "Target B", dev_history_one, 10),
        _source("dev", "11", "UD11", "Target A", dev_history_two, 12),
    ]
    pq.write_table(
        pa.Table.from_pylist(train, schema=DATASET_SCHEMA),
        matched / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev, schema=DATASET_SCHEMA),
        matched / "dev.parquet",
    )

    for task_type in ("behavior_only", "interest_only"):
        train_path = paired / "train" / task_type / "21600.parquet"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [_paired_train_row(row, task_type) for row in train],
                schema=GENERATIVE_SCHEMA,
            ),
            train_path,
        )
        dev_path = paired / "eval" / task_type / "dev_full_grouped.parquet"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _paired_dev_row(dev[0], ["Target A", "Target B"], task_type),
                    _paired_dev_row(dev[2], ["Target A"], task_type),
                ],
                schema=GROUPED_EVAL_SCHEMA,
            ),
            dev_path,
        )
    return matched, paired


def test_builds_paired_behavior_interest_sft_data(tmp_path: Path) -> None:
    matched, paired = _write_fixture(tmp_path)
    output = tmp_path / "output"

    result = build_behavior_interest_sft_dataset(
        matched,
        paired,
        output,
        history_limit=2,
        eval_subset_size=1,
        batch_size=1,
    )

    assert result.train_rows == 4
    assert result.train_users == 4
    assert result.train_interests == 2
    assert result.dev_impressions == 2
    assert result.dev_users == 2
    assert result.dev_target_occurrences == 3
    assert result.dev_supported_interests == 2

    train = pq.read_table(
        output / "train/behavior_interest/21600.parquet"
    ).to_pylist()
    assert [row["source_sample_id"] for row in train] == [
        f"train:{index}:{index}" for index in range(1, 5)
    ]
    assert train[0]["task_type"] == "behavior_interest"
    assert "1. [History One] Clicked headline 1a" in train[0]["prompt"]
    assert "2. [History Two] Clicked headline 1b" in train[0]["prompt"]
    assert train[0]["response"] == "New topic: Target A"
    assert train[0]["messages"][1]["content"] == train[0]["response"]

    full_dev = pq.read_table(
        output / "eval/behavior_interest/dev_full_grouped.parquet"
    ).to_pylist()
    assert len(full_dev) == 2
    assert full_dev[0]["acceptable_subcategories"] == [
        "Target A",
        "Target B",
    ]
    assert pq.ParquetFile(
        output / "eval/behavior_interest/dev_1k_grouped.parquet"
    ).metadata.num_rows == 1

    statistics = json.loads((output / "statistics.json").read_text())
    assert statistics["train"]["min_per_interest"] == 2
    assert statistics["train"]["max_per_interest"] == 2
    assert statistics["pairing"]["train_dev_user_overlap"] == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["history_item_format"] == "[Topic category] Headline"
    assert len(metadata["artifact_sha256"]) == 4

    with pytest.raises(FileExistsError):
        build_behavior_interest_sft_dataset(
            matched,
            paired,
            output,
            history_limit=2,
            eval_subset_size=1,
        )
