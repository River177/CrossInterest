from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_dataset import DATASET_SCHEMA
from new_interest.data.build_generative_dataset import (
    GENERATIVE_SCHEMA,
    _sampling_priority,
    _target_selection_priority,
    build_generative_dataset,
)


def _source_row(
    split: str,
    impression_id: str,
    *,
    history: list[dict[str, str]],
    target_category: str = "Target Parent",
    target_subcategory: str = "New Topic",
    interest_id: int = 7,
) -> dict[str, object]:
    return {
        "sample_id": f"{split}:{impression_id}:{interest_id}",
        "split": split,
        "impression_id": impression_id,
        "user_id": f"U{impression_id}",
        "timestamp": datetime(2019, 11, 13, 12, 34, 8),
        "history": history,
        "target_interest": {
            "category": target_category,
            "subcategory": target_subcategory,
        },
        "target_news": [{"news_id": "NT", "title": "Target title"}],
    }


def _history_item(
    index: int,
    category: str,
    subcategory: str,
) -> dict[str, str]:
    return {
        "news_id": f"N{index}",
        "category": category,
        "subcategory": subcategory,
        "title": f"Clicked title {index}",
    }


def _write_source(source_dir: Path) -> None:
    train_history = [
        _history_item(1, "Parent Alpha", "History One"),
        _history_item(2, "Parent Beta", "Repeated Topic"),
        _history_item(3, "Parent Beta", "Repeated Topic"),
    ]
    dev_history = [
        _history_item(index, f"Parent {index}", f"History {index}")
        for index in range(1, 61)
    ]
    source_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_source_row("train", "1", history=train_history)],
            schema=DATASET_SCHEMA,
        ),
        source_dir / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_source_row("dev", "2", history=dev_history)],
            schema=DATASET_SCHEMA,
        ),
        source_dir / "dev.parquet",
    )


def _write_balancing_source(
    source_dir: Path,
    *,
    reverse_train: bool = False,
) -> None:
    history = [
        _history_item(index, f"History Parent {index}", f"History {index}")
        for index in range(1, 5)
    ]
    train_rows = [
        *[
            _source_row(
                "train",
                f"1{index}",
                history=history,
                target_subcategory="Target A",
            )
            for index in range(5)
        ],
        *[
            _source_row(
                "train",
                f"2{index}",
                history=history,
                target_subcategory="Target B",
            )
            for index in range(3)
        ],
        _source_row(
            "train",
            "30",
            history=history,
            target_subcategory="Target C",
        ),
    ]
    if reverse_train:
        train_rows.reverse()
    dev_rows = [
        _source_row(
            "dev",
            "40",
            history=history,
            target_subcategory="Target A",
        ),
        _source_row(
            "dev",
            "41",
            history=history,
            target_subcategory="Target A",
        ),
        _source_row(
            "dev",
            "50",
            history=history,
            target_subcategory="Target B",
        ),
        _source_row(
            "dev",
            "60",
            history=history,
            target_subcategory="Target C",
        ),
    ]
    source_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(train_rows, schema=DATASET_SCHEMA),
        source_dir / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev_rows, schema=DATASET_SCHEMA),
        source_dir / "dev.parquet",
    )


def _write_multi_target_source(
    source_dir: Path,
    *,
    reverse_candidates: bool = False,
    non_contiguous: bool = False,
) -> None:
    history = [
        _history_item(index, f"Parent {index}", f"History {index}")
        for index in range(1, 4)
    ]
    first_group = [
        _source_row(
            "train",
            "1",
            history=history,
            target_subcategory="Target A",
            interest_id=1,
        ),
        _source_row(
            "train",
            "1",
            history=history,
            target_subcategory="Target B",
            interest_id=2,
        ),
    ]
    if reverse_candidates:
        first_group.reverse()
    second = _source_row(
        "train",
        "2",
        history=history,
        target_subcategory="Target A",
        interest_id=1,
    )
    train_rows = (
        [first_group[0], second, first_group[1]]
        if non_contiguous
        else [*first_group, second]
    )
    dev_rows = [
        _source_row(
            "dev",
            "3",
            history=history,
            target_subcategory="Target A",
            interest_id=1,
        ),
        _source_row(
            "dev",
            "3",
            history=history,
            target_subcategory="Target B",
            interest_id=2,
        ),
    ]
    source_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(train_rows, schema=DATASET_SCHEMA),
        source_dir / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev_rows, schema=DATASET_SCHEMA),
        source_dir / "dev.parquet",
    )


def test_builds_strict_behavior_and_interest_only_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    result = build_generative_dataset(source, output, batch_size=1)

    assert result.behavior_train_samples == 1
    assert result.behavior_dev_samples == 1
    assert result.interest_train_samples == 1
    assert result.interest_dev_samples == 1

    behavior = pq.read_table(
        output / "behavior_only" / "train.parquet"
    ).to_pylist()[0]
    interest = pq.read_table(
        output / "interest_only" / "train.parquet"
    ).to_pylist()[0]

    assert behavior["response"] == "New topic: New Topic"
    assert behavior["target_subcategory"] == "New Topic"
    assert behavior["history_length"] == 3
    assert "1. Clicked title 1" in behavior["prompt"]
    assert "2. Clicked title 2" in behavior["prompt"]
    assert "3. Clicked title 3" in behavior["prompt"]
    assert behavior["prompt"].startswith(
        "The headlines of the news items I recently viewed are listed below:"
    )
    assert "Based on these headlines" in behavior["prompt"]
    assert "new topic category" in behavior["prompt"]
    assert "topics reflected in my viewing history" in behavior["prompt"]
    assert 'Begin your response with "New topic: "' in behavior["prompt"]
    assert "History One" not in behavior["prompt"]
    assert "Parent Alpha" not in behavior["prompt"]

    assert interest["response"] == "New topic: New Topic"
    assert interest["target_subcategory"] == "New Topic"
    assert interest["history_length"] == 3
    assert interest["prompt"].count("Repeated Topic") == 2
    assert interest["prompt"].startswith(
        "The topic categories associated with the news items I recently "
        "viewed, in chronological order, are listed below:"
    )
    assert "Based on this sequence of topic categories" in interest["prompt"]
    assert "should not appear in the sequence above" in interest["prompt"]
    assert 'Begin your response with "New topic: "' in interest["prompt"]
    assert interest["prompt"].index("History One") < interest["prompt"].index(
        "Repeated Topic"
    )
    assert "Clicked title" not in interest["prompt"]
    assert "Parent Alpha" not in interest["prompt"]
    assert "Target Parent" not in interest["prompt"]
    assert "target_category" not in interest
    assert "category" not in GENERATIVE_SCHEMA.names

    for row in (behavior, interest):
        assert row["messages"] == [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        assert "Choose" not in row["prompt"]
        assert "Options" not in row["prompt"]
        assert row["response"].startswith("New topic: ")
        assert "<think>" not in row["response"]
        assert row["response"] == row["response"].strip()


def test_preserves_the_complete_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    build_generative_dataset(source, output)

    for task_type, final_item in (
        ("behavior_only", "60. Clicked title 60"),
        ("interest_only", "60. History 60"),
    ):
        row = pq.read_table(
            output / task_type / "dev.parquet"
        ).to_pylist()[0]
        assert row["history_length"] == 60
        assert final_item in row["prompt"]


def test_output_is_deterministic_and_protected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_source(source)

    build_generative_dataset(source, first_output)
    build_generative_dataset(source, second_output)

    for task_type in ("behavior_only", "interest_only"):
        for split in ("train", "dev"):
            relative = Path(task_type) / f"{split}.parquet"
            assert pq.read_table(first_output / relative).equals(
                pq.read_table(second_output / relative)
            )

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        build_generative_dataset(source, first_output)

    build_generative_dataset(source, first_output, overwrite=True)


def test_rejects_target_already_present_in_interest_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    invalid_history = [_history_item(1, "Parent", "New Topic")]
    for split in ("train", "dev"):
        pq.write_table(
            pa.Table.from_pylist(
                [_source_row(split, "1", history=invalid_history)],
                schema=DATASET_SCHEMA,
            ),
            source / f"{split}.parquet",
        )

    with pytest.raises(ValueError, match="appears in the interest-only history"):
        build_generative_dataset(source, output)


def test_builds_balanced_train_filtered_dev_and_recent_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_balancing_source(source)

    result = build_generative_dataset(
        source,
        output,
        history_limit=2,
        train_samples_per_class=2,
        train_min_class_count=2,
        filter_dev_to_train_classes=True,
        sampling_seed=17,
    )

    assert result.behavior_train_samples == 4
    assert result.interest_train_samples == 4
    assert result.behavior_dev_samples == 3
    assert result.interest_dev_samples == 3

    behavior_train = pq.read_table(
        output / "behavior_only" / "train.parquet"
    ).to_pylist()
    interest_train = pq.read_table(
        output / "interest_only" / "train.parquet"
    ).to_pylist()
    behavior_dev = pq.read_table(
        output / "behavior_only" / "dev.parquet"
    ).to_pylist()
    interest_dev = pq.read_table(
        output / "interest_only" / "dev.parquet"
    ).to_pylist()

    assert Counter(row["target_subcategory"] for row in behavior_train) == {
        "Target A": 2,
        "Target B": 2,
    }
    assert Counter(row["target_subcategory"] for row in behavior_dev) == {
        "Target A": 2,
        "Target B": 1,
    }
    assert [row["source_sample_id"] for row in behavior_train] == [
        row["source_sample_id"] for row in interest_train
    ]
    assert [row["source_sample_id"] for row in behavior_dev] == [
        row["source_sample_id"] for row in interest_dev
    ]

    for row in [*behavior_train, *behavior_dev]:
        assert row["history_length"] == 2
        assert "1. Clicked title 3" in row["prompt"]
        assert "2. Clicked title 4" in row["prompt"]
        assert "Clicked title 1" not in row["prompt"]
        assert "Clicked title 2" not in row["prompt"]
    for row in [*interest_train, *interest_dev]:
        assert row["history_length"] == 2
        assert "1. History 3" in row["prompt"]
        assert "2. History 4" in row["prompt"]
        assert "History 1" not in row["prompt"]
        assert "History 2" not in row["prompt"]

    selected = pq.read_table(
        output / "selected_train_samples.parquet"
    ).to_pylist()
    assert len(selected) == 4
    assert len({row["source_sample_id"] for row in selected}) == 4
    assert {row["sampling_seed"] for row in selected} == {17}

    retained = pq.read_table(
        output / "retained_interests.parquet"
    ).to_pylist()
    assert retained == [
        {
            "target_subcategory": "Target A",
            "source_train_count": 5,
            "selected_train_count": 2,
            "filtered_dev_count": 2,
        },
        {
            "target_subcategory": "Target B",
            "source_train_count": 3,
            "selected_train_count": 2,
            "filtered_dev_count": 1,
        },
    ]


def test_balanced_selection_is_stable_across_input_order(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first-source"
    second_source = tmp_path / "second-source"
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    _write_balancing_source(first_source)
    _write_balancing_source(second_source, reverse_train=True)

    options = {
        "train_samples_per_class": 2,
        "train_min_class_count": 2,
        "sampling_seed": 17,
    }
    build_generative_dataset(first_source, first_output, **options)
    build_generative_dataset(second_source, second_output, **options)

    assert pq.read_table(
        first_output / "selected_train_samples.parquet"
    ).equals(
        pq.read_table(second_output / "selected_train_samples.parquet")
    )
    assert _sampling_priority(17, "train:10:7") == _sampling_priority(
        17,
        "train:10:7",
    )
    assert _sampling_priority(17, "train:10:7") != _sampling_priority(
        18,
        "train:10:7",
    )


def test_balancing_options_are_validated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)

    with pytest.raises(
        ValueError,
        match="requires train_samples_per_class",
    ):
        build_generative_dataset(
            source,
            tmp_path / "output",
            filter_dev_to_train_classes=True,
        )

    with pytest.raises(ValueError, match="must be at least"):
        build_generative_dataset(
            source,
            tmp_path / "output",
            train_samples_per_class=3,
            train_min_class_count=2,
        )


def test_one_target_applies_only_to_train(tmp_path: Path) -> None:
    source = tmp_path / "source"
    default_output = tmp_path / "default"
    one_target_output = tmp_path / "one-target"
    _write_multi_target_source(source)

    build_generative_dataset(source, default_output)
    build_generative_dataset(
        source,
        one_target_output,
        one_target_per_train_impression=True,
        sampling_seed=17,
    )

    default_train = pq.read_table(
        default_output / "behavior_only" / "train.parquet"
    ).to_pylist()
    one_target_train = pq.read_table(
        one_target_output / "behavior_only" / "train.parquet"
    ).to_pylist()
    one_target_dev = pq.read_table(
        one_target_output / "behavior_only" / "dev.parquet"
    ).to_pylist()

    assert len(default_train) == 3
    assert len(one_target_train) == 2
    assert len({row["impression_id"] for row in one_target_train}) == 2
    assert len(one_target_dev) == 2
    assert [row["impression_id"] for row in one_target_dev] == ["3", "3"]

    candidates = ["train:1:1", "train:1:2"]
    expected = min(
        candidates,
        key=lambda sample_id: (
            _target_selection_priority(17, sample_id),
            sample_id,
        ),
    )
    chosen = next(
        row["source_sample_id"]
        for row in one_target_train
        if row["impression_id"] == "1"
    )
    assert chosen == expected


def test_target_selection_is_independent_of_candidate_order(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first-source"
    second_source = tmp_path / "second-source"
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    _write_multi_target_source(first_source)
    _write_multi_target_source(second_source, reverse_candidates=True)

    build_generative_dataset(
        first_source,
        first_output,
        one_target_per_train_impression=True,
        sampling_seed=17,
    )
    build_generative_dataset(
        second_source,
        second_output,
        one_target_per_train_impression=True,
        sampling_seed=17,
    )

    first = pq.read_table(
        first_output / "interest_only" / "train.parquet",
        columns=["source_sample_id"],
    ).to_pylist()
    second = pq.read_table(
        second_output / "interest_only" / "train.parquet",
        columns=["source_sample_id"],
    ).to_pylist()
    assert first == second
    assert _target_selection_priority(
        17,
        "train:1:1",
    ) != _target_selection_priority(18, "train:1:1")


def test_rejects_non_contiguous_train_impressions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_multi_target_source(source, non_contiguous=True)

    with pytest.raises(ValueError, match="non-contiguous groups"):
        build_generative_dataset(
            source,
            tmp_path / "output",
            one_target_per_train_impression=True,
        )
