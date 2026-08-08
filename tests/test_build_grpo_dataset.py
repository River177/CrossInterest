from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.build_grpo_dataset import (
    COT_INSTRUCTION,
    build_grpo_datasets,
)


def _row(index: int, task_type: str) -> dict[str, object]:
    prompt = (
        "The topic categories associated with the news items I recently "
        "viewed, in chronological order, are listed below:\n\n"
        f"1. History {index}\n\n"
        "Based on this sequence, predict a topic."
        if task_type == "interest_only"
        else f"History prompt {index}"
    )
    response = f"New topic: Target {index}"
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
        "target_subcategory": f"Target {index}",
    }


def _write_source(root: Path) -> None:
    for task_type in ("behavior_only", "interest_only"):
        directory = root / "train" / task_type
        directory.mkdir(parents=True)
        rows = [_row(index, task_type) for index in range(1, 6)]
        for name, count in (("2", 2), ("5", 5)):
            pq.write_table(
                pa.Table.from_pylist(
                    rows[:count],
                    schema=GENERATIVE_SCHEMA,
                ),
                directory / f"{name}.parquet",
            )


def test_builds_nested_verl_records(tmp_path: Path) -> None:
    source = tmp_path / "sft"
    output = tmp_path / "grpo"
    _write_source(source)

    build_grpo_datasets(source, output, sizes=(2, 5))

    for task_type in ("behavior_only", "interest_only"):
        small = pq.read_table(
            output / task_type / "2.parquet"
        ).to_pylist()
        large = pq.read_table(
            output / task_type / "5.parquet"
        ).to_pylist()
        assert len(small) == 2
        assert len(large) == 5
        assert [
            row["extra_info"]["source_sample_id"] for row in large[:2]
        ] == [
            row["extra_info"]["source_sample_id"] for row in small
        ]
        for row in large:
            prompt = row["prompt"]
            assert len(prompt) == 1
            assert prompt[0]["role"] == "user"
            assert COT_INSTRUCTION in prompt[0]["content"]
            assert row["reward_model"]["style"] == "rule"
            assert row["reward_model"]["ground_truth"].startswith(
                "New topic: "
            )
            assert row["reward_model"]["ground_truth"] not in (
                prompt[0]["content"]
            )
            assert row["extra_info"]["task_type"] == task_type
            assert row["extra_info"]["history_subcategories"] == [
                f"History {row['extra_info']['impression_id']}"
            ]
