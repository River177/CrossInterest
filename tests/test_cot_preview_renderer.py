from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.render_cot_preview import (
    main,
    render_cot_preview,
    render_preview_markdown,
)


def _row(index: int, task_type: str) -> dict[str, object]:
    source_id = f"train:{index:02d}"
    target = f"Target {index:02d}"
    label = "Behavior" if task_type == "behavior_only" else "Interest"
    reasoning = (
        f"The {label.lower()} history suggests a plausible adjacent theme "
        "without repeating the observed categories."
    )
    completion = (
        f"<think>\n{reasoning}\n</think>\n"
        f"New topic: {target}"
    )
    return {
        "request_hash": f"hash-{index:02d}-{task_type}",
        "selection_rank": index - 1,
        "sample_id": f"{source_id}:{task_type}",
        "source_sample_id": source_id,
        "impression_id": f"I{index:02d}",
        "task_type": task_type,
        "original_history": [f"History {index}A", f"History {index}B"],
        "original_prompt": f"Original {label} prompt {index}",
        "new_student_prompt": f"New student {label} prompt {index}",
        "selected_target": target,
        "status": "success",
        "filter_reason": None,
        "reasoning": reasoning,
        "completion": completion,
        "reasoning_length": len(reasoning),
        "reasoning_word_count": len(reasoning.split()),
        "reasoning_tokens": 17,
        "strict_valid": True,
        "target_name_violation": False,
        "think_tag_violation": False,
        "new_topic_violation": False,
        "empty_reasoning": False,
    }


def _approved_rows() -> list[dict[str, object]]:
    return [
        _row(index, task_type)
        for index in range(1, 21)
        for task_type in ("behavior_only", "interest_only")
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_renders_approved_paired_preview_deterministically(
    tmp_path: Path,
) -> None:
    rows = list(reversed(_approved_rows()))
    for row in rows:
        source_number = int(str(row["source_sample_id"]).rsplit(":", 1)[1])
        row["selection_rank"] = 20 - source_number
    markdown, validation = render_preview_markdown(rows)

    assert validation.approved
    assert validation.total_rows == 40
    assert validation.success_rows == 40
    assert validation.strict_rows == 40
    assert validation.paired_source_ids == 20
    assert validation.pairing_violations == 0
    assert "**Validation status: APPROVED**" in markdown
    assert "Original Behavior prompt 1" in markdown
    assert "New student Interest prompt 1" in markdown
    assert "Reasoning words:" in markdown
    assert "Reasoning tokens:" in markdown
    assert "**Selection rank:** 0" in markdown
    assert "New topic: Target 01" in markdown
    first_pair = markdown.index("## Pair 1: `train:20`")
    second_pair = markdown.index("## Pair 2: `train:19`")
    behavior = markdown.index("### Behavior", first_pair)
    interest = markdown.index("### Interest", first_pair)
    assert first_pair < behavior < interest < second_pair
    assert markdown == render_preview_markdown(rows)[0]

    records = tmp_path / "teacher_records.parquet"
    output = tmp_path / "preview.md"
    _write_rows(records, rows)
    result = render_cot_preview(records, output)
    assert result.validation.approved
    assert output.read_text(encoding="utf-8") == markdown
    assert (
        main(
            [
                "--teacher-records",
                str(records),
                "--output",
                str(output),
            ]
        )
        == 0
    )


def test_failed_preview_is_written_and_cli_exits_nonzero(
    tmp_path: Path,
) -> None:
    rows = _approved_rows()
    rows[0]["status"] = "filtered"
    rows[0]["filter_reason"] = "target_name_in_reasoning"
    rows[0]["reasoning"] = "Target 01 is explicitly named."
    rows[0]["target_name_violation"] = True
    rows[1]["reasoning"] = "<think>bad</think> New topic: Target 01"
    rows[1]["think_tag_violation"] = True
    rows[1]["new_topic_violation"] = True
    rows.pop()

    records = tmp_path / "teacher_records.parquet"
    output = tmp_path / "preview.md"
    _write_rows(records, rows)

    result = render_cot_preview(records, output)
    markdown = output.read_text(encoding="utf-8")
    assert not result.validation.approved
    assert result.validation.total_rows == 39
    assert result.validation.success_rows == 38
    assert result.validation.strict_rows == 37
    assert result.validation.target_name_violations == 2
    assert result.validation.think_tag_violations == 1
    assert result.validation.new_topic_violations == 1
    assert result.validation.paired_source_ids == 19
    assert result.validation.pairing_violations == 1
    assert "**Validation status: FAILED**" in markdown
    assert "`target_name_in_reasoning`" in markdown
    assert main([str(records), "--output", str(output)]) == 1


def test_explicit_failed_validation_flag_prevents_approval() -> None:
    rows = _approved_rows()
    rows[0]["strict_valid"] = False

    _, validation = render_preview_markdown(rows)

    assert not validation.approved
    assert validation.strict_rows == 39


def test_renders_all_bilingual_fields() -> None:
    rows = _approved_rows()
    for row in rows:
        row.update(
            {
                "selected_target_zh": "中文目标",
                "original_history_zh": "中文历史",
                "original_prompt_zh": "中文原始 Prompt",
                "new_student_prompt_zh": "中文 Student Prompt",
                "reasoning_zh": "中文推理",
                "completion_zh": (
                    "<think>\n中文推理\n</think>\n"
                    "新主题：中文目标"
                ),
            }
        )
    markdown, validation = render_preview_markdown(rows)
    assert validation.approved
    assert "**Target 中文：** `中文目标`" in markdown
    assert markdown.count("#### 原始历史（中文翻译）") == 40
    assert markdown.count("#### 原始 Prompt（中文翻译）") == 40
    assert markdown.count("#### 新 Student Prompt（中文翻译）") == 40
    assert markdown.count("#### GPT reasoning（中文翻译）") == 40
    assert markdown.count("#### 最终 Completion（中文翻译）") == 40


def test_duplicate_selection_rank_is_a_pairing_failure() -> None:
    rows = _approved_rows()
    for row in rows:
        if row["source_sample_id"] == "train:02":
            row["selection_rank"] = 0

    _, validation = render_preview_markdown(rows)

    assert not validation.approved
    assert validation.paired_source_ids == 18
    assert validation.pairing_violations == 2
