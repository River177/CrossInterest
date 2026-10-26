from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.data.build_amazon_kitchen_dataset import (
    METADATA_RELATIVE,
    RATINGS_RELATIVE,
    build_amazon_kitchen_dataset,
    user_split,
)


def _user_for(split: str, start: int) -> str:
    for index in range(start, start + 100_000):
        value = f"user-{index}"
        if user_split(value, 20260806) == split:
            return value
    raise AssertionError(split)


def test_builds_balanced_user_disjoint_dataset_and_filters_eval_targets(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    ratings = raw / RATINGS_RELATIVE
    metadata = raw / METADATA_RELATIVE
    ratings.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    users = {
        "train": _user_for("train", 0),
        "dev": _user_for("dev", 100_000),
    }
    with ratings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "parent_asin", "rating", "timestamp"],
        )
        writer.writeheader()
        for split, user_id in users.items():
            for index in range(1, 6):
                writer.writerow(
                    {"user_id": user_id, "parent_asin": f"A{index}", "rating": 5, "timestamp": index}
                )
            writer.writerow(
                {"user_id": user_id, "parent_asin": "B", "rating": 5, "timestamp": 6}
            )
            if split == "dev":
                writer.writerow(
                    {"user_id": user_id, "parent_asin": "C", "rating": 5, "timestamp": 7}
                )
    rows = [
        *[
            {"parent_asin": f"A{index}", "title": f"a{index}", "categories": ["Home & Kitchen", "Kitchen & Dining", "A"]}
            for index in range(1, 6)
        ],
        {"parent_asin": "B", "title": "b", "categories": ["Home & Kitchen", "Kitchen & Dining", "B"]},
        {"parent_asin": "C", "title": "c", "categories": ["Home & Kitchen", "Kitchen & Dining", "C"]},
    ]
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))

    output = tmp_path / "output"
    result = build_amazon_kitchen_dataset(
        raw,
        output,
        samples_per_category=1,
        target_category_count=1,
    )

    assert result.retained_categories == 1
    assert result.train_rows == result.dev_rows == 1
    assert pq.read_table(output / "train.parquet")["target_categories"].to_pylist() == [["Kitchen & Dining > B"]]
    assert pq.read_table(output / "dev.parquet")["target_categories"].to_pylist() == [["Kitchen & Dining > B"]]
    removed = pq.read_table(output / "evaluation_filter_manifest.parquet").to_pylist()
    assert removed == [
        {
            "split": "dev",
            "user_id": users["dev"],
            "category": "Kitchen & Dining > C",
            "reason": "target-category-absent-from-balanced-train",
        }
    ]
