from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.select_cot_preview import select_cot_preview
from new_interest.data.select_cot_teacher import select_cot_teacher


def _interest_prompt(history: list[str]) -> str:
    numbered = "\n".join(
        f"{index}. {value}"
        for index, value in enumerate(history, start=1)
    )
    return (
        "The topic categories associated with the news items I recently "
        "viewed, in chronological order, are listed below:\n\n"
        f"{numbered}\n\n"
        "Based on this sequence of topic categories, predict a new topic "
        "category that I am likely to be interested in next."
    )


def _history(index: int) -> list[str]:
    length = 4 + index
    mode = index % 4
    if mode == 0:
        return [f"Unique {index}-{position}" for position in range(length)]
    if mode == 1:
        unique_count = max(2, (length * 4 + 4) // 5)
    elif mode == 2:
        unique_count = max(2, (length * 3 + 4) // 5)
    else:
        unique_count = max(2, (length * 2) // 5)
    values = [f"Repeated {index}-{position}" for position in range(unique_count)]
    return [values[position % unique_count] for position in range(length)]


def _source_row(
    index: int,
    task_type: str,
    history: list[str],
) -> dict[str, object]:
    source_id = f"train:{index}:{index}"
    target = f"Target {index:02d}"
    if task_type == "interest_only":
        prompt = _interest_prompt(history)
    else:
        padding = "x" * ((24 - index) * 13)
        headlines = "\n".join(
            f"{position}. Headline {index}-{position} {padding}"
            for position in range(1, len(history) + 1)
        )
        prompt = (
            "The headlines of the news items I recently viewed are listed "
            f"below:\n\n{headlines}\n\nBased on these headlines, predict."
        )
    return {
        "sample_id": f"{source_id}:{task_type}",
        "source_sample_id": source_id,
        "split": "train",
        "task_type": task_type,
        "impression_id": str(index),
        "user_id": f"U{index}",
        "timestamp": datetime(2019, 11, 13, 12, 0, 0),
        "history_length": len(history),
        "prompt": prompt,
        "response": f"New topic: {target}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"New topic: {target}"},
        ],
        "target_subcategory": target,
    }


def _write_fixture(source_dir: Path) -> dict[str, dict[str, dict[str, object]]]:
    expected: dict[str, dict[str, dict[str, object]]] = {}
    for task_type in ("behavior_only", "interest_only"):
        rows = []
        for index in range(24):
            row = _source_row(index, task_type, _history(index))
            rows.append(row)
            expected.setdefault(str(row["source_sample_id"]), {})[
                task_type
            ] = row
        path = source_dir / "train" / task_type / "1k.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=GENERATIVE_SCHEMA),
            path,
        )
    return expected


def test_selection_is_deterministic_diverse_and_paired(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    expected = _write_fixture(source_dir)
    first = tmp_path / "first" / "selection.parquet"
    second = tmp_path / "second" / "selection.parquet"

    select_cot_preview(source_dir, first, root=tmp_path)
    select_cot_preview(source_dir, second, root=tmp_path)

    first_table = pq.read_table(first)
    second_table = pq.read_table(second)
    assert first_table.equals(second_table)
    rows = first_table.to_pylist()
    assert len(rows) == 20
    assert len({row["source_sample_id"] for row in rows}) == 20
    assert len({row["target_subcategory"] for row in rows}) == 20
    assert [row["selection_rank"] for row in rows] == list(range(20))

    for row in rows:
        paired = expected[row["source_sample_id"]]
        behavior = paired["behavior_only"]
        interest = paired["interest_only"]
        assert row["impression_id"] == behavior["impression_id"]
        assert row["impression_id"] == interest["impression_id"]
        assert row["target_subcategory"] == behavior["target_subcategory"]
        assert row["target_subcategory"] == interest["target_subcategory"]
        assert row["behavior_original_prompt"] == behavior["prompt"]
        assert row["interest_original_prompt"] == interest["prompt"]
        assert len(row["history_subcategories"]) == row["history_length"]
        assert len(row["sha256_priority"]) == 64

    assert {row["history_length_bin"] for row in rows} == {
        "q1",
        "q2",
        "q3",
        "q4",
    }
    assert {
        row["behavior_prompt_character_bin"] for row in rows
    } == {"q1", "q2", "q3", "q4"}
    assert {
        row["interest_prompt_character_bin"] for row in rows
    } == {"q1", "q2", "q3", "q4"}
    assert {row["interest_history_stratum"] for row in rows} == {
        "all_unique",
        "light_repetition",
        "moderate_repetition",
        "concentrated_repetition",
    }


def test_metadata_uses_portable_paths_and_records_hashes(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    _write_fixture(source_dir)
    output = tmp_path / "artifacts" / "selection.parquet"

    result = select_cot_preview(source_dir, output, root=tmp_path)

    metadata_text = result.metadata_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert str(tmp_path.resolve()) not in metadata_text
    assert metadata["source_dir"] == "sft"
    assert metadata["output_path"] == "artifacts/selection.parquet"
    assert metadata["seed"] == 20260810
    assert metadata["selection_size"] == 20
    assert metadata["selected_distinct_targets"] == 20
    assert set(metadata["source_sha256"]) == {
        "behavior_only",
        "interest_only",
    }
    assert all(
        len(value["sha256"]) == 64
        and not Path(value["path"]).is_absolute()
        for value in metadata["source_sha256"].values()
    )
    assert len(next(iter(metadata["artifact_sha256"].values()))) == 64


def test_full_selection_preserves_source_order_and_pairing(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    expected = _write_fixture(source_dir)
    output = tmp_path / "full" / "selection.parquet"

    result = select_cot_teacher(
        source_dir,
        output,
        selection_size=24,
        root=tmp_path,
    )

    rows = pq.read_table(result.output_path).to_pylist()
    assert [row["selection_rank"] for row in rows] == list(range(24))
    assert [row["source_sample_id"] for row in rows] == [
        f"train:{index}:{index}" for index in range(24)
    ]
    for row in rows:
        paired = expected[row["source_sample_id"]]
        assert row["behavior_original_prompt"] == paired[
            "behavior_only"
        ]["prompt"]
        assert row["interest_original_prompt"] == paired[
            "interest_only"
        ]["prompt"]
    metadata = json.loads(
        result.metadata_path.read_text(encoding="utf-8")
    )
    assert metadata["selection_size"] == 24
    assert metadata["policy"]["selection"] == "source-order prefix"
