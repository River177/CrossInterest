from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cot_rehearsal_dataset import (
    build_cot_rehearsal_datasets,
)
from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA


def _row(index: int, task: str, response: str) -> dict[str, object]:
    prompt = f"Prompt {index}"
    return {
        "sample_id": f"train:{index}:{task}",
        "source_sample_id": f"train:{index}",
        "split": "train",
        "task_type": task,
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


def test_builds_three_to_one_rehearsal_mix(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    cot = tmp_path / "cot"
    for task in ("behavior_only", "interest_only"):
        direct_path = direct / "train" / task / "2.parquet"
        cot_path = cot / task / "2.parquet"
        direct_path.parent.mkdir(parents=True)
        cot_path.parent.mkdir(parents=True)
        direct_rows = [
            _row(index, task, f"New topic: Target {index}")
            for index in (1, 2)
        ]
        cot_rows = [
            _row(
                index,
                task,
                f"<think>Reason {index}</think>\n"
                f"New topic: Target {index}",
            )
            for index in (1, 2)
        ]
        pq.write_table(
            pa.Table.from_pylist(direct_rows, schema=GENERATIVE_SCHEMA),
            direct_path,
        )
        pq.write_table(
            pa.Table.from_pylist(cot_rows, schema=GENERATIVE_SCHEMA),
            cot_path,
        )

    output = tmp_path / "mixed"
    build_cot_rehearsal_datasets(
        direct,
        cot,
        output,
        size=2,
        cot_repeats=3,
        direct_repeats=1,
    )
    rows = pq.read_table(
        output / "behavior_only" / "2.parquet"
    ).to_pylist()
    assert len(rows) == 8
    assert [row["response"].startswith("<think>") for row in rows[:4]] == [
        True,
        True,
        True,
        False,
    ]
    assert len({row["sample_id"] for row in rows}) == 8
