from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import DATASET_SCHEMA
from new_interest.data.build_generative_dataset import (
    GENERATIVE_SCHEMA,
    RETAINED_INTEREST_SCHEMA,
    SELECTED_TRAIN_SCHEMA,
)
from new_interest.data.build_sft_subsets import build_sft_subsets


def _history() -> list[dict[str, str]]:
    return [
        {
            "news_id": f"N{index}",
            "category": f"Parent {index}",
            "subcategory": f"History {index}",
            "title": f"Title {index}",
        }
        for index in range(1, 4)
    ]


def _generative_row(
    sample_index: int,
    target: str,
    task_type: str,
) -> dict[str, object]:
    source_sample_id = f"train:{sample_index}:{sample_index}"
    prompt = f"{task_type} prompt {sample_index}"
    response = f"New topic: {target}"
    return {
        "sample_id": f"{source_sample_id}:{task_type}",
        "source_sample_id": source_sample_id,
        "split": "train",
        "task_type": task_type,
        "impression_id": str(sample_index),
        "user_id": f"U{sample_index}",
        "timestamp": datetime(2019, 11, 13, 12, 0, 0),
        "history_length": 3,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target,
    }


def _dev_source_row(
    impression_id: str,
    interest_id: int,
    target: str,
) -> dict[str, object]:
    return {
        "sample_id": f"dev:{impression_id}:{interest_id}",
        "split": "dev",
        "impression_id": impression_id,
        "user_id": f"U{impression_id}",
        "timestamp": datetime(2019, 11, 13, 12, 0, 0),
        "history": _history(),
        "target_interest": {
            "category": "Target Parent",
            "subcategory": target,
        },
        "target_news": [
            {"news_id": f"NT{interest_id}", "title": "Target title"}
        ],
    }


def _write_fixture(balanced_dir: Path, source_dir: Path) -> None:
    targets = ("Target A", "Target B", "Target C")
    selected_rows = []
    task_rows = {
        "behavior_only": [],
        "interest_only": [],
    }
    sample_index = 0
    for target_index, target in enumerate(targets):
        for within_class in range(4):
            sample_index += 1
            source_sample_id = f"train:{sample_index}:{sample_index}"
            selected_rows.append(
                {
                    "source_sample_id": source_sample_id,
                    "impression_id": str(sample_index),
                    "target_subcategory": target,
                    "target_selection_priority": (
                        f"target-{target_index}-{within_class}"
                    ),
                    "balance_sampling_priority": (
                        f"{within_class:02d}-{target_index:02d}"
                    ),
                    "sampling_seed": 17,
                }
            )
            for task_type in task_rows:
                task_rows[task_type].append(
                    _generative_row(sample_index, target, task_type)
                )

    balanced_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            selected_rows,
            schema=SELECTED_TRAIN_SCHEMA,
        ),
        balanced_dir / "selected_train_samples.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "target_subcategory": target,
                    "source_train_count": 4,
                    "selected_train_count": 4,
                    "filtered_dev_count": 1,
                }
                for target in targets
            ],
            schema=RETAINED_INTEREST_SCHEMA,
        ),
        balanced_dir / "retained_interests.parquet",
    )
    for task_type, rows in task_rows.items():
        path = balanced_dir / task_type / "train.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=GENERATIVE_SCHEMA),
            path,
        )

    source_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                _dev_source_row("1", 1, "Target A"),
                _dev_source_row("1", 2, "Target B"),
                _dev_source_row("2", 3, "Target C"),
                _dev_source_row("3", 4, "Filtered Target"),
            ],
            schema=DATASET_SCHEMA,
        ),
        source_dir / "dev.parquet",
    )


def test_builds_nested_balanced_train_and_grouped_dev(
    tmp_path: Path,
) -> None:
    balanced = tmp_path / "balanced"
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_fixture(balanced, source)

    result = build_sft_subsets(
        balanced,
        source,
        output,
        sizes=(6, 12),
        eval_subset_size=1,
        history_limit=2,
        seed=17,
        batch_size=2,
    )

    assert result.train_sizes == (6, 12)
    assert result.grouped_dev_rows == 2
    assert result.grouped_dev_subset_rows == 1

    task_ids = {}
    for task_type in ("behavior_only", "interest_only"):
        six = pq.read_table(
            output / "train" / task_type / "6.parquet"
        ).to_pylist()
        twelve = pq.read_table(
            output / "train" / task_type / "12.parquet"
        ).to_pylist()
        six_ids = [row["source_sample_id"] for row in six]
        twelve_ids = [row["source_sample_id"] for row in twelve]
        assert len(six) == 6
        assert len(twelve) == 12
        assert twelve_ids[:6] == six_ids
        assert Counter(row["target_subcategory"] for row in six) == {
            "Target A": 2,
            "Target B": 2,
            "Target C": 2,
        }
        task_ids[task_type] = (six_ids, twelve_ids)
    assert task_ids["behavior_only"] == task_ids["interest_only"]

    full_eval = pq.read_table(
        output
        / "eval"
        / "interest_only"
        / "dev_full_grouped.parquet"
    ).to_pylist()
    assert len(full_eval) == 2
    multi = next(row for row in full_eval if row["impression_id"] == "1")
    assert multi["acceptable_subcategories"] == ["Target A", "Target B"]
    assert multi["acceptable_responses"] == [
        "New topic: Target A",
        "New topic: Target B",
    ]
    assert multi["history_length"] == 2
    assert multi["history_subcategories"] == ["History 2", "History 3"]

    eval_subset = pq.read_table(
        output / "eval" / "interest_only" / "dev_1_grouped.parquet"
    )
    assert eval_subset.num_rows == 1

    manifest = pq.read_table(output / "subset_manifest.parquet").to_pylist()
    assert len(manifest) == 12
    assert manifest[0]["included_sizes"] == [6, 12]
    assert manifest[5]["included_sizes"] == [6, 12]
    assert manifest[6]["included_sizes"] == [12]


def test_subset_build_is_deterministic(tmp_path: Path) -> None:
    balanced = tmp_path / "balanced"
    source = tmp_path / "source"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fixture(balanced, source)

    options = {
        "sizes": (6, 12),
        "eval_subset_size": 1,
        "history_limit": 2,
        "seed": 17,
    }
    build_sft_subsets(balanced, source, first, **options)
    build_sft_subsets(balanced, source, second, **options)

    relative_paths = [
        Path("train/behavior_only/6.parquet"),
        Path("train/interest_only/12.parquet"),
        Path("eval/behavior_only/dev_full_grouped.parquet"),
        Path("eval/interest_only/dev_1_grouped.parquet"),
        Path("subset_manifest.parquet"),
    ]
    for relative in relative_paths:
        assert pq.read_table(first / relative).equals(
            pq.read_table(second / relative)
        )


def test_full_size_contains_every_balanced_sample(tmp_path: Path) -> None:
    balanced = tmp_path / "balanced"
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_fixture(balanced, source)

    build_sft_subsets(
        balanced,
        source,
        output,
        sizes=(6, 12),
        eval_subset_size=1,
        history_limit=2,
        seed=17,
    )

    manifest = pq.read_table(output / "subset_manifest.parquet").to_pylist()
    behavior = pq.read_table(
        output / "train/behavior_only/12.parquet",
        columns=["source_sample_id"],
    ).to_pylist()
    interest = pq.read_table(
        output / "train/interest_only/12.parquet",
        columns=["source_sample_id"],
    ).to_pylist()
    expected_ids = [row["source_sample_id"] for row in manifest]
    assert [row["source_sample_id"] for row in behavior] == expected_ids
    assert [row["source_sample_id"] for row in interest] == expected_ids
