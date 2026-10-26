from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_mera_semantic_id_dataset import (
    build_mera_semantic_id_dataset,
)
from new_interest.data.build_sft_subsets import GROUPED_EVAL_SCHEMA


def _history(label: str, parent: str = "Parent") -> dict[str, str]:
    return {
        "news_id": f"news-{label}",
        "category": parent,
        "subcategory": label,
        "title": f"Headline for {label}",
    }


def _source_row(
    index: int,
    history_labels: list[str],
    target: str,
) -> dict[str, object]:
    return {
        "sample_id": f"train:{index}",
        "split": "train",
        "impression_id": str(index),
        "user_id": f"train-user-{index}",
        "timestamp": datetime(2026, 1, 1, 0, 0, index),
        "history": [_history(label) for label in history_labels],
        "target_interest": {"category": "Parent", "subcategory": target},
        "target_news": [{"news_id": f"target-{index}", "title": target}],
    }


def _write_sources(root: Path, grouped_dev: Path, mapping: Path) -> None:
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "interest_index": 0,
                    "interest_token": "<interest_000>",
                    "target_subcategory": "Alpha",
                },
                {
                    "interest_index": 1,
                    "interest_token": "<interest_001>",
                    "target_subcategory": "Beta",
                },
            ]
        ),
        mapping,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _source_row(1, ["Beta", "Gamma"], "Alpha"),
                _source_row(2, ["Gamma"], "Beta"),
                _source_row(3, ["Beta"], "Alpha"),
                _source_row(4, ["Alpha", "Gamma"], "Beta"),
            ]
        ),
        root / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "impression_id": "dev-1",
                    "user_id": "dev-user-1",
                    "timestamp": datetime(2026, 1, 2, 0, 0, 0),
                    "history_length": 2,
                    "prompt": "old prompt",
                    "acceptable_responses": ["old"],
                    "acceptable_subcategories": ["Alpha"],
                    "history_subcategories": ["Beta", "Gamma"],
                }
            ],
            schema=GROUPED_EVAL_SCHEMA,
        ),
        grouped_dev,
    )


def test_build_mera_semantic_id_dataset_uses_sid_only_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    grouped_dev = tmp_path / "dev.parquet"
    mapping_file = tmp_path / "mapping.parquet"
    output = tmp_path / "output"
    _write_sources(source, grouped_dev, mapping_file)

    result = build_mera_semantic_id_dataset(
        source,
        grouped_dev,
        mapping_file,
        output,
        train_size=4,
        samples_per_target=2,
        batch_size=2,
    )

    assert result.category_tokens == 2
    mapping = pq.read_table(output / "interest_token_mapping.parquet").to_pylist()
    assert [row["target_subcategory"] for row in mapping] == ["Alpha", "Beta"]
    train = pq.read_table(
        output / "train/interest_sid/4.parquet"
    ).to_pylist()
    assert train[0]["response"] == "<interest_000>"
    assert "<interest_001>" in train[0]["prompt"]
    assert "Beta" not in train[0]["prompt"]
    assert "Headline" not in train[0]["prompt"]
    assert train[0]["messages"][1] == {
        "role": "assistant",
        "content": "<interest_000>",
    }
    dev = pq.read_table(
        output / "eval/interest_sid/dev_full_grouped.parquet"
    ).to_pylist()
    assert dev[0]["acceptable_responses"] == ["<interest_000>"]
    assert "<interest_001>" in dev[0]["prompt"]
    stats = json.loads((output / "statistics.json").read_text())
    assert stats["category_tokens"] == 2
    assert stats["train"]["empty_histories_after_filter"] == 1
    assert stats["train"]["dropped_out_of_108_history_events"] == 3
    assert stats["prompt_contract"]["headlines_present"] is False

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        build_mera_semantic_id_dataset(
            source,
            grouped_dev,
            mapping_file,
            output,
        )
