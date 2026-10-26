from __future__ import annotations

import pytest

from new_interest.evaluation.evaluate_candidate_qwen import _shard_records


def _records(prefix: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "selection_id": f"{prefix}-{index}",
            "prompt": f"prompt {index}",
        }
        for index in range(count)
    ]


def test_shards_cover_records_once_and_preserve_source_order() -> None:
    records = _records("record", 11)

    shards = [
        _shard_records(
            records,
            records,
            num_shards=3,
            shard_index=shard_index,
        )
        for shard_index in range(3)
    ]

    for shard_index, shard in enumerate(shards):
        assert [row["selection_id"] for row in shard] == [
            row["selection_id"]
            for source_index, row in enumerate(records)
            if source_index % 3 == shard_index
        ]
    assert sorted(
        row["selection_id"] for shard in shards for row in shard
    ) == sorted(row["selection_id"] for row in records)


@pytest.mark.parametrize(
    ("num_shards", "shard_index"),
    [(0, 0), (-1, 0), (2, -1), (2, 2)],
)
def test_invalid_shard_settings_fail(
    num_shards: int,
    shard_index: int,
) -> None:
    records = _records("record", 3)

    with pytest.raises(ValueError):
        _shard_records(
            records,
            records,
            num_shards=num_shards,
            shard_index=shard_index,
        )


def test_paired_inputs_with_same_ordering_have_same_shard_ids() -> None:
    first = _records("paired", 8)
    second = [
        {
            "selection_id": row["selection_id"],
            "prompt": f"different {index}",
        }
        for index, row in enumerate(first)
    ]

    for shard_index in range(3):
        first_ids = [
            row["selection_id"]
            for row in _shard_records(
                first,
                first,
                num_shards=3,
                shard_index=shard_index,
            )
        ]
        second_ids = [
            row["selection_id"]
            for row in _shard_records(
                second,
                second,
                num_shards=3,
                shard_index=shard_index,
            )
        ]

        assert first_ids == second_ids


def test_selected_subset_is_partitioned_by_global_source_index() -> None:
    source_records = _records("record", 8)
    selected_records = [
        source_records[index] for index in (1, 2, 5, 7)
    ]

    shard = _shard_records(
        source_records,
        selected_records,
        num_shards=2,
        shard_index=1,
    )

    assert [row["selection_id"] for row in shard] == [
        "record-1",
        "record-5",
        "record-7",
    ]
