from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_behavior_interest_semantic_id_token_sft_dataset import (
    build_behavior_interest_semantic_id_token_sft_dataset,
)
from new_interest.data.build_semantic_id_token_sft_dataset import (
    ID_TOKEN_PROMPT_SUFFIX,
    PROMPT_SUFFIX,
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

    train_rows = []
    for index, label in enumerate(
        ["Alpha Topic", "Zulu Topic", "Alpha Topic", "Zulu Topic"]
    ):
        prompt = _prompt(f"[{index}] Behavior plus Interest")
        response = f"New topic: {label}"
        train_rows.append(
            {
                "sample_id": f"train:{index}:behavior_interest",
                "source_sample_id": f"train:{index}",
                "split": "train",
                "task_type": "behavior_interest",
                "impression_id": str(index),
                "user_id": f"U{index}",
                "timestamp": datetime(2019, 1, 1),
                "history_length": 2,
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "target_subcategory": label,
            }
        )
    train_path = source / "train/behavior_interest/4.parquet"
    train_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(train_rows), train_path)

    dev_rows = [
        {
            "impression_id": "d1",
            "user_id": "D1",
            "timestamp": datetime(2019, 1, 2),
            "history_length": 2,
            "prompt": _prompt("Behavior plus Interest Dev one"),
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
            "prompt": _prompt("Behavior plus Interest Dev two"),
            "acceptable_responses": ["New topic: Alpha Topic"],
            "acceptable_subcategories": ["Alpha Topic"],
            "history_subcategories": ["Old A"],
        },
    ]
    dev_path = source / "eval/behavior_interest/dev_full_grouped.parquet"
    dev_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(dev_rows), dev_path)
    (source / "statistics.json").write_text("{}\n", encoding="utf-8")
    (source / "metadata.json").write_text("{}\n", encoding="utf-8")
    return source, interests


def _build(source: Path, interests: Path, output: Path):
    return build_behavior_interest_semantic_id_token_sft_dataset(
        source,
        interests,
        output,
        batch_size=2,
        train_size=4,
        expected_interests=2,
        samples_per_interest=2,
    )


def test_builds_behavior_interest_semantic_id_data(tmp_path: Path) -> None:
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

    train = pq.read_table(
        output / "train/behavior_interest/4.parquet"
    ).to_pylist()
    assert {row["task_type"] for row in train} == {"behavior_interest"}
    assert train[0]["response"] == "<interest_000>"
    assert train[0]["messages"][1]["content"] == "<interest_000>"
    assert train[0]["prompt"].endswith(ID_TOKEN_PROMPT_SUFFIX)
    assert "Behavior plus Interest" in train[0]["prompt"]

    dev = pq.read_table(
        output / "eval/behavior_interest/dev_full_grouped.parquet"
    ).to_pylist()
    assert dev[0]["acceptable_responses"] == [
        "<interest_000>",
        "<interest_001>",
    ]
    statistics = json.loads(
        (output / "statistics.json").read_text(encoding="utf-8")
    )
    assert statistics["task_type"] == "behavior_interest"
    assert statistics["train_dev_user_overlap"] == 0
    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["task_type"] == "behavior_interest"
    assert metadata["history_filtering_at_evaluation"] is False


def test_build_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    source, interests = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(source, interests, first)
    _build(source, interests, second)
    for relative in (
        "interest_token_mapping.parquet",
        "train/behavior_interest/4.parquet",
        "eval/behavior_interest/dev_full_grouped.parquet",
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
