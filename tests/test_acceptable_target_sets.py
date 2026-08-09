from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_acceptable_target_sets import (
    build_acceptable_target_sets,
)
from new_interest.data.build_dataset import DATASET_SCHEMA
from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA


def _source_row(
    impression_id: str,
    target: str,
    interest_id: int,
) -> dict[str, object]:
    return {
        "sample_id": f"train:{impression_id}:{interest_id}",
        "split": "train",
        "impression_id": impression_id,
        "user_id": f"U{impression_id}",
        "timestamp": datetime(2019, 11, 13),
        "history": [
            {
                "news_id": f"N{impression_id}",
                "category": "History",
                "subcategory": "History label",
                "title": "History title",
            }
        ],
        "target_interest": {
            "category": "Target",
            "subcategory": target,
        },
        "target_news": [
            {"news_id": f"T{impression_id}", "title": "Target title"}
        ],
    }


def _sft_row(
    index: int,
    task_type: str,
    target: str,
) -> dict[str, object]:
    prompt = f"Prompt {index}"
    response = f"New topic: {target}"
    return {
        "sample_id": f"train:{index}:{index}:{task_type}",
        "source_sample_id": f"train:{index}:{index}",
        "split": "train",
        "task_type": task_type,
        "impression_id": str(index),
        "user_id": f"U{index}",
        "timestamp": datetime(2019, 11, 13),
        "history_length": 1,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target,
    }


def _write_fixture(root: Path, *, mismatch_pair: bool = False) -> tuple[Path, Path]:
    source_path = root / "processed" / "train.parquet"
    source_path.parent.mkdir(parents=True)
    source_rows = [
        _source_row("1", "Target A", 1),
        _source_row("1", "Target B", 2),
        _source_row("1", "Dropped label", 3),
        _source_row("2", "Target B", 2),
        _source_row("3", "Target C", 3),
        _source_row("4", "Target A", 1),
        _source_row("4", "Target C", 3),
    ]
    pq.write_table(
        pa.Table.from_pylist(source_rows, schema=DATASET_SCHEMA),
        source_path,
    )

    sft_dir = root / "sft"
    selected = ["Target A", "Target B", "Target C", "Target A"]
    for task_type in ("behavior_only", "interest_only"):
        directory = sft_dir / "train" / task_type
        directory.mkdir(parents=True)
        rows = [
            _sft_row(index, task_type, target)
            for index, target in enumerate(selected, start=1)
        ]
        if mismatch_pair and task_type == "interest_only":
            rows[0]["target_subcategory"] = "Target B"
        for name, count in (("2", 2), ("4", 4)):
            pq.write_table(
                pa.Table.from_pylist(
                    rows[:count],
                    schema=GENERATIVE_SCHEMA,
                ),
                directory / f"{name}.parquet",
            )
    return source_path, sft_dir


def test_builds_nested_paired_acceptable_target_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, sft_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "acceptable"
    monkeypatch.chdir(tmp_path)

    manifest_path = build_acceptable_target_sets(
        source_path,
        sft_dir,
        output_dir,
        sizes=(2, 4),
        expected_retained_label_count=3,
    )

    rows = pq.read_table(manifest_path).to_pylist()
    assert [row["source_sample_id"] for row in rows] == [
        "train:1:1",
        "train:2:2",
        "train:3:3",
        "train:4:4",
    ]
    assert rows[0]["acceptable_subcategories"] == [
        "Target A",
        "Target B",
    ]
    assert "Dropped label" not in rows[0]["acceptable_subcategories"]
    assert rows[0]["included_sizes"] == [2, 4]
    assert rows[2]["included_sizes"] == [4]

    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["retained_label_count"] == 3
    assert metadata["retained_subcategories"] == [
        "Target A",
        "Target B",
        "Target C",
    ]
    assert metadata["source_train_path"] == "processed/train.parquet"
    assert metadata["source_sft_dir"] == "sft"


def test_rejects_unpaired_behavior_interest_rows(tmp_path: Path) -> None:
    source_path, sft_dir = _write_fixture(
        tmp_path,
        mismatch_pair=True,
    )
    with pytest.raises(ValueError, match="Behavior/Interest"):
        build_acceptable_target_sets(
            source_path,
            sft_dir,
            tmp_path / "acceptable",
            sizes=(2, 4),
            expected_retained_label_count=3,
        )
