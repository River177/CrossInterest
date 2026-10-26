from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_cot_sft_dataset import (
    build_cot_sft_datasets,
    format_cot_completion,
)
from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.generate_cot_teacher import (
    MAX_COMPLETION_TOKENS,
    TEACHER_RECORD_SCHEMA,
    _call_teacher,
    build_teacher_prompt,
    generate_teacher_reasoning,
    parse_teacher_response,
)


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
        "category that I am likely to be interested in next. The predicted "
        "category should not appear in the sequence above. Begin your "
        'response with "New topic: " and provide only the category name '
        "after it."
    )


def _source_row(
    index: int,
    task_type: str,
    target: str,
    history: list[str],
) -> dict[str, object]:
    source_id = f"train:{index}:{index}"
    if task_type == "interest_only":
        prompt = _interest_prompt(history)
    else:
        prompt = (
            "The headlines of the news items I recently viewed are listed "
            f"below:\n\n1. Headline {index}A\n2. Headline {index}B\n\n"
            "Based on these headlines, predict a new topic category that I "
            "am likely to be interested in next."
        )
    response = f"New topic: {target}"
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
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target,
    }


def _write_source_fixture(
    source_dir: Path,
    *,
    targets: tuple[str, ...] = ("Music", "Travel"),
    histories: tuple[tuple[str, ...], ...] = (
        ("Sports", "Finance"),
        ("Autos", "Weather"),
    ),
) -> dict[str, list[dict[str, object]]]:
    rows_by_task: dict[str, list[dict[str, object]]] = {}
    for task_type in ("behavior_only", "interest_only"):
        rows = [
            _source_row(
                index,
                task_type,
                targets[index - 1],
                list(histories[index - 1]),
            )
            for index in range(1, len(targets) + 1)
        ]
        rows_by_task[task_type] = rows
        task_dir = source_dir / "train" / task_type
        task_dir.mkdir(parents=True, exist_ok=True)
        for size in range(1, len(rows) + 1):
            pq.write_table(
                pa.Table.from_pylist(
                    rows[:size],
                    schema=GENERATIVE_SCHEMA,
                ),
                task_dir / f"{size}.parquet",
            )
    return rows_by_task


class FakeCompletions:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            raw = (
                self.responses.pop(0)
                if self.responses
                else '{"reasoning":"The history supports a distinct shift."}'
            )
        usage = SimpleNamespace(
            prompt_tokens=40,
            completion_tokens=12,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=0
            ),
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=raw)
                )
            ],
            usage=usage,
        )


def _fake_client(
    responses: list[str] | None = None,
) -> tuple[SimpleNamespace, FakeCompletions]:
    completions = FakeCompletions(responses)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def _teacher_row(
    source: dict[str, object],
    reasoning: str,
    *,
    selected_target: str | None = None,
) -> dict[str, object]:
    target = selected_target or str(source["target_subcategory"])
    raw = json.dumps({"reasoning": reasoning})
    return {
        "request_hash": f"hash-{source['sample_id']}",
        "sample_id": source["sample_id"],
        "source_sample_id": source["source_sample_id"],
        "impression_id": source["impression_id"],
        "task_type": source["task_type"],
        "original_prompt": source["prompt"],
        "selected_target": target,
        "history_subcategories": ["placeholder"],
        "status": "success",
        "filter_reason": None,
        "reasoning": reasoning,
        "raw_response": raw,
        "attempts": 1,
        "error_type": None,
        "error_message": None,
        "status_code": None,
        "prompt_tokens": 40,
        "completion_tokens": 12,
        "reasoning_tokens": 0,
        "latency_seconds": 0.1,
    }


def test_teacher_call_uses_json_and_no_temperature() -> None:
    client, completions = _fake_client(
        ['{"reasoning":"A concise transition is plausible."}']
    )
    record = {
        "request_hash": "abc",
        "sample_id": "train:1:1:behavior_only",
        "source_sample_id": "train:1:1",
        "impression_id": "1",
        "task_type": "behavior_only",
        "original_prompt": "Original prompt without the answer.",
        "selected_target": "Music",
        "history_subcategories": ["Sports"],
        "teacher_prompt": build_teacher_prompt(
            "Original prompt without the answer.",
            "Music",
        ),
    }

    result = _call_teacher(
        client,
        record,
        deployment="gpt-5.6-sol_2026-07-09",
        sleep_fn=lambda _: None,
    )

    assert result["status"] == "success"
    assert result["reasoning"] == "A concise transition is plausible."
    call = completions.calls[0]
    assert call["max_completion_tokens"] == MAX_COMPLETION_TOKENS
    assert call["response_format"] == {"type": "json_object"}
    assert "temperature" not in call
    teacher_input = call["messages"][1]["content"]
    assert "Original prompt without the answer." in teacher_input
    assert '"Music"' in teacher_input


def test_invalid_teacher_reasoning_retries_at_most_three() -> None:
    client, completions = _fake_client(
        [
            '{"reasoning":"<think>bad</think>"}',
            '{"reasoning":"New topic: Music"}',
            '{"reasoning":""}',
        ]
    )
    record = {
        "request_hash": "abc",
        "sample_id": "s",
        "source_sample_id": "source",
        "impression_id": "1",
        "task_type": "interest_only",
        "original_prompt": "prompt",
        "selected_target": "Music",
        "history_subcategories": ["Sports"],
        "teacher_prompt": build_teacher_prompt("prompt", "Music"),
    }
    result = _call_teacher(
        client,
        record,
        deployment="gpt-5.6-sol_2026-07-09",
        max_attempts=3,
        sleep_fn=lambda _: None,
    )

    assert len(completions.calls) == 3
    assert result["status"] == "filtered"
    assert result["filter_reason"] == "empty_reasoning"
    with pytest.raises(ValueError):
        parse_teacher_response('{"reasoning":"ok","answer":"Music"}')


def test_generation_is_resumable_with_atomic_shards(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    output_dir = tmp_path / "teacher"
    _write_source_fixture(source_dir)
    client, completions = _fake_client()
    config = {
        "source_dir": source_dir,
        "source_size": 2,
        "output_dir": output_dir,
        "instance": "redmond/interactive",
        "deployment": "gpt-5.6-sol_2026-07-09",
        "api_version": "2024-10-21",
        "scope": "api://trapi/.default",
        "max_completion_tokens": 256,
        "max_attempts": 3,
        "concurrency": 2,
        "shard_size": 1,
        "pilot_size": 2,
        "pilot_seed": 9,
    }

    first = generate_teacher_reasoning(
        config,
        client=client,
        sleep_fn=lambda _: None,
    )
    second = generate_teacher_reasoning(
        config,
        client=client,
        sleep_fn=lambda _: None,
    )

    assert first.requested == first.successes == 2
    assert first.generated_this_run == 2
    assert second.generated_this_run == 0
    assert second.resumed == 2
    assert len(completions.calls) == 2
    pilot_records = pq.read_table(
        output_dir / "teacher_records.parquet"
    ).to_pylist()
    assert len(pilot_records) == 2
    assert all(row["request_hash"] for row in pilot_records)
    assert all(row["status"] == "success" for row in pilot_records)
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["temperature"] is None
    assert metadata["max_attempts"] == 3
    assert metadata["status_counts"] == {"success": 2}
    assert len(list((output_dir / "shards").glob("*.parquet"))) == 2

    full_config = {**config, "pilot_size": None}
    full = generate_teacher_reasoning(
        full_config,
        client=client,
        sleep_fn=lambda _: None,
    )
    full_records = pq.read_table(
        output_dir / "teacher_records.parquet"
    ).to_pylist()
    assert full.requested == full.successes == 4
    assert full.generated_this_run == 2
    assert full.resumed == 2
    assert len(completions.calls) == 4
    assert full_records[:2] == pilot_records


def test_generation_expands_from_nested_1k_to_5k_order(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    output_dir = tmp_path / "teacher"
    _write_source_fixture(
        source_dir,
        targets=("Music", "Travel"),
        histories=(
            ("Sports", "Finance"),
            ("Autos", "Weather"),
        ),
    )
    client, completions = _fake_client()
    base_config = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "instance": "redmond/interactive",
        "deployment": "gpt-5.6-sol_2026-07-09",
        "api_version": "2024-10-21",
        "scope": "api://trapi/.default",
        "max_completion_tokens": 256,
        "max_attempts": 3,
        "concurrency": 2,
        "shard_size": 1,
        "pilot_size": None,
        "pilot_seed": 9,
    }
    first = generate_teacher_reasoning(
        {**base_config, "source_size": 1},
        client=client,
        sleep_fn=lambda _: None,
    )
    first_rows = pq.read_table(
        output_dir / "teacher_records.parquet"
    ).to_pylist()
    expanded = generate_teacher_reasoning(
        {**base_config, "source_size": 2},
        client=client,
        sleep_fn=lambda _: None,
    )
    expanded_rows = pq.read_table(
        output_dir / "teacher_records.parquet"
    ).to_pylist()

    assert first.requested == 2
    assert expanded.requested == 4
    assert expanded.resumed == 2
    assert expanded.generated_this_run == 2
    assert expanded_rows[:2] == first_rows
    assert [
        (row["source_sample_id"], row["task_type"])
        for row in expanded_rows
    ] == [
        ("train:1:1", "behavior_only"),
        ("train:1:1", "interest_only"),
        ("train:2:2", "behavior_only"),
        ("train:2:2", "interest_only"),
    ]


def test_builds_separate_nested_behavior_and_interest_datasets(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    output_dir = tmp_path / "cot"
    source_rows = _write_source_fixture(source_dir)
    teacher_path = tmp_path / "teacher_records.parquet"
    teacher_rows = [
        _teacher_row(
            row,
            f"Reasoning for {row['task_type']} sample "
            f"{row['source_sample_id']}.",
        )
        for task_type in ("behavior_only", "interest_only")
        for row in source_rows[task_type]
    ]
    pq.write_table(
        pa.Table.from_pylist(
            teacher_rows,
            schema=TEACHER_RECORD_SCHEMA,
        ),
        teacher_path,
    )

    build_cot_sft_datasets(
        source_dir,
        teacher_path,
        output_dir,
        sizes=(1, 2),
    )

    for task_type in ("behavior_only", "interest_only"):
        one = pq.read_table(
            output_dir / task_type / "1.parquet"
        ).to_pylist()
        two = pq.read_table(
            output_dir / task_type / "2.parquet"
        ).to_pylist()
        assert two[:1] == one
        for source, output in zip(source_rows[task_type], two):
            target = str(source["target_subcategory"])
            expected_reasoning = (
                f"Reasoning for {task_type} sample "
                f"{source['source_sample_id']}."
            )
            assert output["prompt"] == source["prompt"]
            assert output["messages"][0] == source["messages"][0]
            assert target not in output["messages"][0]["content"]
            assert output["response"] == format_cot_completion(
                expected_reasoning,
                target,
            )
            assert (
                output["messages"][1]["content"] == output["response"]
            )


def test_rejects_target_in_history_and_teacher_identity_mismatch(
    tmp_path: Path,
) -> None:
    invalid_source = tmp_path / "invalid-sft"
    _write_source_fixture(
        invalid_source,
        targets=("Music",),
        histories=(("Sports", "Music"),),
    )
    client, _ = _fake_client()
    with pytest.raises(ValueError, match="target appears in history"):
        generate_teacher_reasoning(
            {
                "source_dir": invalid_source,
                "source_size": 1,
                "output_dir": tmp_path / "invalid-teacher",
            },
            client=client,
        )

    source_dir = tmp_path / "valid-sft"
    rows = _write_source_fixture(
        source_dir,
        targets=("Music",),
        histories=(("Sports", "Finance"),),
    )
    teacher_path = tmp_path / "mismatch.parquet"
    teacher_rows = [
        _teacher_row(
            row,
            "A valid rationale.",
            selected_target=(
                "Travel"
                if row["task_type"] == "behavior_only"
                else None
            ),
        )
        for task_type in ("behavior_only", "interest_only")
        for row in rows[task_type]
    ]
    pq.write_table(
        pa.Table.from_pylist(
            teacher_rows,
            schema=TEACHER_RECORD_SCHEMA,
        ),
        teacher_path,
    )
    with pytest.raises(ValueError, match="selected targets do not match"):
        build_cot_sft_datasets(
            source_dir,
            teacher_path,
            tmp_path / "mismatch-output",
            sizes=(1,),
        )


def test_rejects_changed_v2_student_prompt(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sft"
    rows = _write_source_fixture(
        source_dir,
        targets=("Music",),
        histories=(("Sports", "Finance"),),
    )
    teacher_path = tmp_path / "teacher.parquet"
    teacher_rows = []
    for task_type in ("behavior_only", "interest_only"):
        row = _teacher_row(
            rows[task_type][0],
            "A valid history-grounded rationale.",
        )
        row["new_student_prompt"] = "Changed Prompt"
        row["strict_valid"] = True
        teacher_rows.append(row)
    pq.write_table(pa.Table.from_pylist(teacher_rows), teacher_path)

    with pytest.raises(ValueError, match="Student Prompt"):
        build_cot_sft_datasets(
            source_dir,
            teacher_path,
            tmp_path / "output",
            sizes=(1,),
        )
