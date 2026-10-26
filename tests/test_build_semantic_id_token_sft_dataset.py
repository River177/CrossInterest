from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_semantic_id_token_sft_dataset import (
    ID_TOKEN_PROMPT_SUFFIX,
    PROMPT_SUFFIX,
    build_semantic_id_token_sft_dataset,
)


def _prompt(prefix: str) -> str:
    return f"{prefix}\n\n{PROMPT_SUFFIX}"


def _write_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    interests = root / "retained_interests.parquet"
    labels = ["Zulu Topic", "Alpha Topic"]
    pq.write_table(
        pa.Table.from_pylist(
            [{"target_subcategory": label} for label in labels]
        ),
        interests,
    )
    train_common = []
    for index, label in enumerate(
        ["Alpha Topic", "Zulu Topic", "Alpha Topic", "Zulu Topic"]
    ):
        train_common.append(
            {
                "sample_id": f"train:{index}",
                "source_sample_id": f"train:{index}",
                "split": "train",
                "impression_id": str(index),
                "user_id": f"U{index}",
                "timestamp": datetime(2019, 1, 1),
                "history_length": 2,
                "target_subcategory": label,
            }
        )
    for task in ("behavior_only", "interest_only"):
        rows = []
        for common in train_common:
            prompt = _prompt(f"{task} history")
            response = f"New topic: {common['target_subcategory']}"
            rows.append(
                {
                    **common,
                    "task_type": task,
                    "prompt": prompt,
                    "response": response,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                }
            )
        path = source / "train" / task / "4.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

        dev_rows = [
            {
                "impression_id": "d1",
                "user_id": "D1",
                "timestamp": datetime(2019, 1, 2),
                "history_length": 2,
                "prompt": _prompt(f"{task} dev one"),
                "acceptable_responses": [
                    "New topic: Alpha Topic",
                    "New topic: Zulu Topic",
                ],
                "acceptable_subcategories": ["Alpha Topic", "Zulu Topic"],
                "history_subcategories": ["Old A", "Old B"],
            },
            {
                "impression_id": "d2",
                "user_id": "D2",
                "timestamp": datetime(2019, 1, 3),
                "history_length": 1,
                "prompt": _prompt(f"{task} dev two"),
                "acceptable_responses": ["New topic: Alpha Topic"],
                "acceptable_subcategories": ["Alpha Topic"],
                "history_subcategories": ["Old A"],
            },
        ]
        dev_path = source / "eval" / task / "dev_full_grouped.parquet"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(dev_rows), dev_path)

    pq.write_table(
        pa.Table.from_pylist([{"source_sample_id": f"train:{i}"} for i in range(4)]),
        source / "subset_manifest.parquet",
    )
    (source / "statistics.json").write_text("{}\n", encoding="utf-8")
    (source / "metadata.json").write_text("{}\n", encoding="utf-8")
    return source, interests


def _build(source: Path, interests: Path, output: Path):
    return build_semantic_id_token_sft_dataset(
        source,
        interests,
        output,
        batch_size=2,
        train_size=4,
        expected_interests=2,
        samples_per_interest=2,
    )


def test_builds_paired_semantic_id_token_data(tmp_path: Path) -> None:
    source, interests = _write_fixture(tmp_path)
    output = tmp_path / "output"
    result = _build(source, interests, output)

    assert result.train_rows == 4
    assert result.train_users == 4
    assert result.train_interests == 2
    assert result.dev_impressions == 2
    assert result.dev_users == 2
    assert result.dev_target_occurrences == 3
    assert result.dev_supported_interests == 2

    mapping = pq.read_table(
        output / "interest_token_mapping.parquet"
    ).to_pylist()
    assert mapping == [
        {
            "interest_index": 0,
            "interest_token": "<interest_000>",
            "target_subcategory": "Alpha Topic",
        },
        {
            "interest_index": 1,
            "interest_token": "<interest_001>",
            "target_subcategory": "Zulu Topic",
        },
    ]

    behavior = pq.read_table(
        output / "train/behavior_only/4.parquet"
    ).to_pylist()
    interest = pq.read_table(
        output / "train/interest_only/4.parquet"
    ).to_pylist()
    assert [row["source_sample_id"] for row in behavior] == [
        row["source_sample_id"] for row in interest
    ]
    assert behavior[0]["response"] == "<interest_000>"
    assert behavior[0]["messages"][1]["content"] == "<interest_000>"
    assert behavior[0]["prompt"].endswith(ID_TOKEN_PROMPT_SUFFIX)
    assert "New topic:" not in behavior[0]["prompt"]

    dev = pq.read_table(
        output / "eval/behavior_only/dev_full_grouped.parquet"
    ).to_pylist()
    assert dev[0]["acceptable_responses"] == [
        "<interest_000>",
        "<interest_001>",
    ]
    assert dev[0]["acceptable_subcategories"] == [
        "Alpha Topic",
        "Zulu Topic",
    ]
    statistics = json.loads(
        (output / "statistics.json").read_text(encoding="utf-8")
    )
    assert statistics["train_dev_user_overlap"] == 0
    assert statistics["train"]["samples_per_interest"] == 2
    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["random_initialization_condition"] is False
    assert metadata["history_filtering_at_evaluation"] is False


def test_build_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    source, interests = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(source, interests, first)
    _build(source, interests, second)
    for relative in (
        "interest_token_mapping.parquet",
        "train/behavior_only/4.parquet",
        "train/interest_only/4.parquet",
        "eval/behavior_only/dev_full_grouped.parquet",
        "eval/interest_only/dev_full_grouped.parquet",
        "statistics.json",
    ):
        left = first / relative
        right = second / relative
        if left.suffix == ".parquet":
            assert pq.read_table(left).equals(pq.read_table(right))
        else:
            assert left.read_bytes() == right.read_bytes()
    with pytest.raises(FileExistsError, match="Use --overwrite"):
        _build(source, interests, first)


def test_build_rejects_unexpected_prompt_contract(tmp_path: Path) -> None:
    source, interests = _write_fixture(tmp_path)
    path = source / "train/behavior_only/4.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["prompt"] = "changed prompt"
    rows[0]["messages"][0]["content"] = "changed prompt"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="expected suffix"):
        _build(source, interests, tmp_path / "output")
