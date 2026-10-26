from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.generate_cot_teacher_v2 import (
    MAX_COMPLETION_TOKENS,
    TRAPI_DEPLOYMENT,
    _call_teacher,
    build_teacher_prompt,
    build_student_prompt,
    format_completion,
    generate_teacher_reasoning,
    parse_teacher_response,
)
from new_interest.data.select_cot_preview import SELECTION_SCHEMA


def _behavior_prompt() -> str:
    return (
        "The headlines of the news items I recently viewed are listed "
        "below:\n\n"
        "1. Team wins championship\n"
        "2. Markets close higher\n\n"
        "Based on these headlines, predict a new topic category that I am "
        "likely to be interested in next. The predicted category should be "
        "different from the topics reflected in my viewing history. Begin "
        'your response with "New topic: " and provide only the category '
        "name after it."
    )


def test_student_prompt_matches_sft22k_prompt_exactly() -> None:
    original = _behavior_prompt()

    prompt = build_student_prompt(original)

    assert prompt == original
    assert "provide only the category name after it" in prompt


def test_teacher_json_is_target_hidden_and_completion_is_strict() -> None:
    reasoning = parse_teacher_response(
        json.dumps(
            {
                "reasoning": (
                    "The recent mix suggests a plausible shift into a "
                    "different leisure-oriented area."
                )
            }
        ),
        selected_target="Music",
    )

    assert format_completion(reasoning, "Music") == (
        "<think>\n"
        "The recent mix suggests a plausible shift into a different "
        "leisure-oriented area.\n"
        "</think>\n"
        "New topic: Music"
    )

    invalid = (
        '{"reasoning":"A move toward   mUsIc is plausible."}'
    )
    with pytest.raises(ValueError, match="canonical target"):
        parse_teacher_response(invalid, selected_target=" Music ")
    with pytest.raises(ValueError, match="only the reasoning field"):
        parse_teacher_response(
            '{"reasoning":"Valid rationale.","answer":"Music"}',
            selected_target="Music",
        )
    for raw, message in (
        ('{"reasoning":""}', "nonempty"),
        ('{"reasoning":"<think>hidden</think>"}', "think tag"),
        ('{"reasoning":"New topic: something"}', "New topic"),
        (
            '{"reasoning":"The private target is plausible."}',
            "private/reference target",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            parse_teacher_response(raw, selected_target="Music")


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=50,
                completion_tokens=14,
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=0
                ),
            ),
        )


class _RateLimitError(Exception):
    status_code = 429


def test_teacher_call_uses_fixed_request_and_429_backoff() -> None:
    completions = _FakeCompletions(
        [
            _RateLimitError("slow down"),
            _RateLimitError("still slow"),
            '{"reasoning":"The sequence supports a shift toward a '
            'different leisure area."}',
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    sleeps: list[float] = []
    record = {
        "request_hash": "hash",
        "selection_rank": 0,
        "sample_id": "train:1:behavior_only",
        "source_sample_id": "train:1",
        "impression_id": "1",
        "task_type": "behavior_only",
        "original_history": "1. Team wins\n2. Markets rise",
        "original_prompt": _behavior_prompt(),
        "new_student_prompt": build_student_prompt(_behavior_prompt()),
        "selected_target": "Music",
        "history_length": 2,
        "history_subcategories": ["Sports", "Finance"],
        "original_prompt_length": len(_behavior_prompt()),
        "student_prompt_length": len(build_student_prompt(_behavior_prompt())),
        "teacher_prompt": build_teacher_prompt(
            "1. Team wins\n2. Markets rise",
            "Music",
            task_type="behavior_only",
        ),
    }

    result = _call_teacher(
        client,
        record,
        deployment=TRAPI_DEPLOYMENT,
        sleep_fn=sleeps.append,
    )

    assert sleeps == [5.0, 10.0]
    assert result["status"] == "success"
    assert result["completion"].endswith("New topic: Music")
    call = completions.calls[-1]
    assert call["model"] == TRAPI_DEPLOYMENT
    assert call["max_completion_tokens"] == MAX_COMPLETION_TOKENS
    assert call["response_format"] == {"type": "json_object"}
    assert "temperature" not in call
    assert "Music" in call["messages"][1]["content"]
    assert record["new_student_prompt"] not in call["messages"][1]["content"]
    assert "concise" not in call["messages"][1]["content"].casefold()
    assert "short rationale" not in call["messages"][1]["content"].casefold()


def _interest_prompt() -> str:
    return (
        "The topic categories associated with the news items I recently "
        "viewed, in chronological order, are listed below:\n\n"
        "1. Sports\n"
        "2. Finance\n\n"
        "Based on this sequence of topic categories, predict a new topic "
        "category that I am likely to be interested in next. The predicted "
        "category should not appear in the sequence above. Begin your "
        'response with "New topic: " and provide only the category name '
        "after it."
    )


def _selection_row(rank: int, target: str) -> dict[str, object]:
    source_id = f"train:{rank}:{rank}"
    return {
        "selection_rank": rank,
        "source_sample_id": source_id,
        "impression_id": str(rank),
        "target_subcategory": target,
        "behavior_sample_id": f"{source_id}:behavior_only",
        "interest_sample_id": f"{source_id}:interest_only",
        "behavior_original_prompt": _behavior_prompt(),
        "interest_original_prompt": _interest_prompt(),
        "history_length": 2,
        "history_subcategories": ["Sports", "Finance"],
        "history_unique_subcategories": 2,
        "history_repeated_items": 0,
        "history_unique_ratio": 1.0,
        "history_max_frequency": 1,
        "history_length_bin": "q1",
        "behavior_prompt_characters": len(_behavior_prompt()),
        "behavior_prompt_character_bin": "q1",
        "interest_prompt_characters": len(_interest_prompt()),
        "interest_prompt_character_bin": "q1",
        "interest_history_stratum": "all_unique",
        "strata": ["history_length:q1"],
        "sha256_priority": f"{rank:064x}",
    }


def test_generation_expands_pairs_and_resumes_atomic_shards(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.parquet"
    output_dir = tmp_path / "teacher"
    pq.write_table(
        pa.Table.from_pylist(
            [_selection_row(1, "Travel"), _selection_row(0, "Music")],
            schema=SELECTION_SCHEMA,
        ),
        selection_path,
    )
    completions = _FakeCompletions(
        [
            '{"reasoning":"The viewed items support a plausible shift into '
            'another leisure area."}',
            '{"reasoning":"The category sequence points toward a different '
            'recreational direction."}',
            '{"reasoning":"The mixed history makes a fresh lifestyle area '
            'plausible."}',
            '{"reasoning":"The observed sequence leaves room for an '
            'unseen leisure theme."}',
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    config = {
        "selection_path": selection_path,
        "output_dir": output_dir,
        "instance": "redmond/interactive",
        "deployment": "gpt-5.6-sol_2026-07-09",
        "api_version": "2024-10-21",
        "scope": "api://trapi/.default",
        "max_completion_tokens": 256,
        "max_attempts": 3,
        "concurrency": 4,
        "shard_size": 1,
        "retry_failed_on_resume": True,
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

    assert first.requested == first.successes == 4
    assert first.generated_this_run == 4
    assert second.generated_this_run == 0
    assert second.resumed == 4
    assert len(completions.calls) == 4
    rows = pq.read_table(
        output_dir / "teacher_records.parquet"
    ).to_pylist()
    assert [
        (row["selection_rank"], row["task_type"])
        for row in rows
    ] == [
        (0, "behavior_only"),
        (0, "interest_only"),
        (1, "behavior_only"),
        (1, "interest_only"),
    ]
    assert all(row["strict_valid"] for row in rows)
    assert all(
        row["new_student_prompt"] == row["original_prompt"]
        for row in rows
    )
    assert all(row["reasoning_word_count"] > 0 for row in rows)
    assert all(
        row["completion"].endswith(
            f"New topic: {row['selected_target']}"
        )
        for row in rows
    )
    assert len(list((output_dir / "shards").glob("*.parquet"))) == 4
    assert not list(output_dir.rglob("*.pending"))
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["concurrency"] == 4
    assert metadata["status_counts"] == {"success": 4}


def test_generation_imports_matching_seed_reasoning(
    tmp_path: Path,
) -> None:
    seed_selection = tmp_path / "seed-selection.parquet"
    full_selection = tmp_path / "full-selection.parquet"
    seed_output = tmp_path / "seed-teacher"
    full_output = tmp_path / "full-teacher"
    pq.write_table(
        pa.Table.from_pylist(
            [_selection_row(0, "Music")],
            schema=SELECTION_SCHEMA,
        ),
        seed_selection,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _selection_row(0, "Music"),
                _selection_row(1, "Travel"),
            ],
            schema=SELECTION_SCHEMA,
        ),
        full_selection,
    )
    seed_completions = _FakeCompletions(
        [
            '{"reasoning":"The viewed items support another leisure area."}',
            '{"reasoning":"The sequence leaves room for another theme."}',
        ]
    )
    seed_client = SimpleNamespace(
        chat=SimpleNamespace(completions=seed_completions)
    )
    base_config = {
        "instance": "redmond/interactive",
        "deployment": "gpt-5.6-sol_2026-07-09",
        "api_version": "2024-10-21",
        "scope": "api://trapi/.default",
        "max_completion_tokens": 256,
        "max_attempts": 3,
        "concurrency": 4,
        "shard_size": 2,
        "retry_failed_on_resume": True,
    }
    generate_teacher_reasoning(
        {
            **base_config,
            "selection_path": seed_selection,
            "output_dir": seed_output,
        },
        client=seed_client,
        sleep_fn=lambda _: None,
    )

    full_completions = _FakeCompletions(
        [
            '{"reasoning":"The history supports a fresh lifestyle area."}',
            '{"reasoning":"The observed pattern permits an unseen theme."}',
        ]
    )
    full_client = SimpleNamespace(
        chat=SimpleNamespace(completions=full_completions)
    )
    result = generate_teacher_reasoning(
        {
            **base_config,
            "selection_path": full_selection,
            "output_dir": full_output,
            "seed_records_path": (
                seed_output / "teacher_records.parquet"
            ),
        },
        client=full_client,
        sleep_fn=lambda _: None,
    )

    assert result.requested == result.successes == 4
    assert result.generated_this_run == 2
    assert len(full_completions.calls) == 2
    rows = pq.read_table(
        full_output / "teacher_records.parquet"
    ).to_pylist()
    assert [row["attempts"] for row in rows] == [0, 0, 1, 1]
    assert all(
        row["new_student_prompt"] == row["original_prompt"]
        for row in rows
    )
    metadata = json.loads(
        (full_output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["imported_seed_records"] == 2
