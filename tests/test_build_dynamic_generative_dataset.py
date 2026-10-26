from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import VOCABULARY_SCHEMA
from new_interest.data.build_dynamic_dataset import DYNAMIC_DATASET_SCHEMA
from new_interest.data.build_dynamic_generative_dataset import (
    build_dynamic_generative_dataset,
)


def _history(news_id: str, topic: str) -> dict[str, str]:
    return {
        "news_id": news_id,
        "category": f"{topic} Parent",
        "subcategory": topic,
        "title": f"{topic} history title",
    }


def _interest(topic: str) -> dict[str, str]:
    return {
        "category": f"{topic} Parent",
        "subcategory": topic,
    }


def _row(
    split: str,
    impression_id: str,
    topics: list[str],
    *,
    history_size: int = 3,
) -> dict[str, object]:
    history = [
        _history(f"H{index}", f"History {index}")
        for index in range(history_size)
    ]
    clicked_news = [
        {
            "news_id": f"N{impression_id}{index}",
            "category": f"{topic} Parent",
            "subcategory": topic,
            "title": f"{topic} clicked title",
        }
        for index, topic in enumerate(topics)
    ]
    novel_interests = [_interest(topic) for topic in topics]
    return {
        "sample_id": f"{split}:{impression_id}",
        "split": split,
        "impression_id": impression_id,
        "user_id": f"U{impression_id}",
        "timestamp": datetime(2019, 11, 10 if split == "train" else 15),
        "initial_history_length": history_size,
        "history_length": history_size,
        "history": history,
        "clicked_news": clicked_news,
        "clicked_interests": novel_interests,
        "novel_interests": novel_interests,
        "repeated_interests": [],
        "has_new_interest": bool(topics),
    }


def _write_source(source: Path) -> None:
    source.mkdir(parents=True)
    train = [
        _row("train", "1", ["Target A", "Target B"]),
        _row("train", "2", ["Target A"]),
        _row("train", "3", ["Target A"]),
        _row("train", "4", ["Target B"]),
        _row("train", "5", ["Target B"]),
        _row("train", "6", ["Target C"]),
        _row("train", "7", []),
    ]
    dev = [
        _row("dev", "8", ["Target A", "Target B"], history_size=4),
        _row("dev", "9", ["Target C"], history_size=4),
        _row("dev", "10", [], history_size=4),
    ]
    pq.write_table(
        pa.Table.from_pylist(train, schema=DYNAMIC_DATASET_SCHEMA),
        source / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev, schema=DYNAMIC_DATASET_SCHEMA),
        source / "dev.parquet",
    )
    vocabulary = [
        {
            "interest_id": index,
            "category": f"{topic} Parent",
            "subcategory": topic,
        }
        for index, topic in enumerate(
            [
                "History 0",
                "History 1",
                "History 2",
                "History 3",
                "Target A",
                "Target B",
                "Target C",
            ]
        )
    ]
    pq.write_table(
        pa.Table.from_pylist(vocabulary, schema=VOCABULARY_SCHEMA),
        source / "interest_vocabulary.parquet",
    )
    (source / "metadata.json").write_text(
        json.dumps({"rule_version": "dynamic-click-history-v1"}),
        encoding="utf-8",
    )


def test_builds_balanced_dynamic_generative_data(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    result = build_dynamic_generative_dataset(
        source,
        output,
        samples_per_class=2,
        sampling_seed=17,
        history_limit=2,
        batch_size=2,
    )

    assert result.behavior_train_samples == 4
    assert result.interest_train_samples == 4
    assert result.behavior_dev_samples == 2
    assert result.interest_dev_samples == 2

    behavior_train = pq.read_table(
        output / "behavior_only" / "train.parquet"
    ).to_pylist()
    interest_train = pq.read_table(
        output / "interest_only" / "train.parquet"
    ).to_pylist()
    behavior_dev = pq.read_table(
        output / "behavior_only" / "dev.parquet"
    ).to_pylist()

    assert Counter(
        row["target_subcategory"] for row in behavior_train
    ) == {"Target A": 2, "Target B": 2}
    assert Counter(
        row["target_subcategory"] for row in behavior_dev
    ) == {"Target A": 1, "Target B": 1}
    assert len({row["impression_id"] for row in behavior_train}) == 4
    assert [row["source_sample_id"] for row in behavior_train] == [
        row["source_sample_id"] for row in interest_train
    ]
    assert all(row["history_length"] == 2 for row in behavior_train)

    retained = pq.read_table(
        output / "retained_interests.parquet"
    ).to_pylist()
    assert {row["target_subcategory"] for row in retained} == {
        "Target A",
        "Target B",
    }

    statistics = json.loads(
        (output / "statistics.json").read_text(encoding="utf-8")
    )
    natural_train = statistics["dynamic_source"]["train"]
    assert natural_train["all_impressions"]["count_distribution"] == {
        "0": 1,
        "1": 5,
        "2": 1,
    }
    assert natural_train["positive_impressions_only"][
        "mean_novel_targets_per_impression"
    ] == 7 / 6

    final_train = statistics["novel_targets_per_impression"][
        "balanced_train_selected_impressions"
    ]
    assert final_train["impressions"] == 4
    assert sum(final_train["count_distribution"].values()) == 4

    final_dev = statistics["novel_targets_per_impression"][
        "online_dev_after_retained_class_filter"
    ]
    assert final_dev["all_impressions"]["count_distribution"] == {
        "0": 2,
        "2": 1,
    }
    assert final_dev["positive_impressions_only"][
        "mean_novel_targets_per_impression"
    ] == 2.0

    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["dev_policy"] == "online_dynamic"
    assert metadata["source_dir"] == str(source.resolve())
