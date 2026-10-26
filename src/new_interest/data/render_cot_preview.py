from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow.parquet as pq


EXPECTED_ROWS = 40
EXPECTED_SOURCE_PAIRS = 20
TASK_ORDER = {"behavior_only": 0, "interest_only": 1}

_THINK_TAG = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
_NEW_TOPIC = re.compile(r"\bnew\s+topic\b", re.IGNORECASE)


@dataclass(frozen=True)
class PreviewValidation:
    total_rows: int
    success_rows: int
    strict_rows: int
    target_name_violations: int
    think_tag_violations: int
    new_topic_violations: int
    empty_reasoning_rows: int
    paired_source_ids: int
    pairing_violations: int
    approved: bool


@dataclass(frozen=True)
class PreviewResult:
    output_path: Path
    validation: PreviewValidation


def _value(
    row: Mapping[str, object],
    *names: str,
    default: object | None = None,
) -> object | None:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _flag(
    row: Mapping[str, object],
    names: tuple[str, ...],
) -> bool | None:
    value = _value(row, *names)
    return None if value is None else bool(value)


def _rank_value(row: Mapping[str, object]) -> int | None:
    value = _value(row, "selection_rank")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _target_name_present(reasoning: str, target: str) -> bool:
    clean_target = target.strip()
    if not clean_target:
        return True
    pattern = re.compile(
        rf"(?<!\w){re.escape(clean_target)}(?!\w)",
        re.IGNORECASE,
    )
    return pattern.search(reasoning) is not None


def _row_validation(
    row: Mapping[str, object],
) -> tuple[bool, bool, bool, bool, bool]:
    reasoning = str(_value(row, "reasoning", "gpt_reasoning", default=""))
    target = str(
        _value(
            row,
            "selected_target",
            "target_subcategory",
            "target",
            default="",
        )
    )

    empty = not reasoning.strip()
    think_tag = _THINK_TAG.search(reasoning) is not None
    new_topic = _NEW_TOPIC.search(reasoning) is not None
    target_name = _target_name_present(reasoning, target)

    explicit_empty = _flag(
        row,
        ("empty_reasoning", "reasoning_empty", "is_empty"),
    )
    explicit_think = _flag(
        row,
        (
            "think_tag_violation",
            "contains_think_tag",
            "think_tag_present",
        ),
    )
    explicit_new_topic = _flag(
        row,
        (
            "new_topic_violation",
            "contains_new_topic",
            "new_topic_present",
        ),
    )
    explicit_target = _flag(
        row,
        (
            "target_name_violation",
            "contains_target_name",
            "target_name_present",
        ),
    )
    if explicit_empty is not None:
        empty = empty or explicit_empty
    if explicit_think is not None:
        think_tag = think_tag or explicit_think
    if explicit_new_topic is not None:
        new_topic = new_topic or explicit_new_topic
    if explicit_target is not None:
        target_name = target_name or explicit_target

    passing_flags = (
        "strict_valid",
        "strict_pass",
        "strict_validation",
        "strict_validation_passed",
        "validation_passed",
        "is_valid",
        "valid",
    )
    explicit_passes = [
        bool(row[name])
        for name in passing_flags
        if name in row and row[name] is not None
    ]
    explicit_strict = all(explicit_passes) if explicit_passes else True
    strict = (
        str(_value(row, "status", default="")) == "success"
        and explicit_strict
        and not empty
        and not think_tag
        and not new_topic
        and not target_name
    )
    return strict, target_name, think_tag, new_topic, empty


def validate_preview_rows(
    rows: Iterable[Mapping[str, object]],
) -> PreviewValidation:
    materialized = list(rows)
    success_rows = sum(
        str(_value(row, "status", default="")) == "success"
        for row in materialized
    )
    validations = [_row_validation(row) for row in materialized]

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    rank_counts: Counter[int] = Counter()
    for row in materialized:
        source_id = str(
            _value(row, "source_sample_id", "source_id", default="")
        )
        grouped[source_id].append(row)
        rank = _rank_value(row)
        if rank is not None:
            rank_counts[rank] += 1

    paired_source_ids = 0
    pairing_violations = 0
    for source_id, group in grouped.items():
        tasks = Counter(
            str(_value(row, "task_type", default="")) for row in group
        )
        identities = {
            (
                str(_value(row, "impression_id", default="")),
                str(
                    _value(
                        row,
                        "selected_target",
                        "target_subcategory",
                        "target",
                        default="",
                    )
                ),
            )
            for row in group
        }
        ranks = {_rank_value(row) for row in group}
        paired_rank = next(iter(ranks)) if len(ranks) == 1 else None
        paired = (
            bool(source_id)
            and tasks == Counter(
                {"behavior_only": 1, "interest_only": 1}
            )
            and len(identities) == 1
            and all(
                impression_id and target
                for impression_id, target in identities
            )
            and paired_rank is not None
            and 0 <= paired_rank < EXPECTED_SOURCE_PAIRS
            and rank_counts[paired_rank] == 2
        )
        if paired:
            paired_source_ids += 1
        else:
            pairing_violations += 1

    strict_rows = sum(result[0] for result in validations)
    target_name_violations = sum(result[1] for result in validations)
    think_tag_violations = sum(result[2] for result in validations)
    new_topic_violations = sum(result[3] for result in validations)
    empty_reasoning_rows = sum(result[4] for result in validations)
    total_rows = len(materialized)
    approved = (
        total_rows == EXPECTED_ROWS
        and success_rows == EXPECTED_ROWS
        and strict_rows == EXPECTED_ROWS
        and target_name_violations == 0
        and think_tag_violations == 0
        and new_topic_violations == 0
        and empty_reasoning_rows == 0
        and paired_source_ids == EXPECTED_SOURCE_PAIRS
        and pairing_violations == 0
    )
    return PreviewValidation(
        total_rows=total_rows,
        success_rows=success_rows,
        strict_rows=strict_rows,
        target_name_violations=target_name_violations,
        think_tag_violations=think_tag_violations,
        new_topic_violations=new_topic_violations,
        empty_reasoning_rows=empty_reasoning_rows,
        paired_source_ids=paired_source_ids,
        pairing_violations=pairing_violations,
        approved=approved,
    )


def _fenced(value: object | None, language: str = "text") -> str:
    text = "" if value is None else str(value)
    fence = "````" if "```" in text else "```"
    return f"{fence}{language}\n{text}\n{fence}"


def _history_text(row: Mapping[str, object]) -> str:
    value = _value(
        row,
        "original_history",
        "history",
        "history_subcategories",
    )
    if isinstance(value, list):
        return "\n".join(
            f"{index}. {item}" for index, item in enumerate(value, start=1)
        )
    return "" if value is None else str(value)


def _final_completion(row: Mapping[str, object]) -> str:
    existing = _value(row, "final_completion", "completion")
    if existing is not None:
        return str(existing)
    reasoning = str(_value(row, "reasoning", "gpt_reasoning", default=""))
    target = str(
        _value(
            row,
            "selected_target",
            "target_subcategory",
            "target",
            default="",
        )
    )
    return f"<think>\n{reasoning.strip()}\n</think>\nNew topic: {target}"


def _selection_rank(row: Mapping[str, object]) -> int:
    value = _rank_value(row)
    return value if value is not None else 2**31 - 1


def _summary_lines(validation: PreviewValidation) -> list[str]:
    state = "APPROVED" if validation.approved else "FAILED"
    return [
        "# CoT v2 Teacher Preview",
        "",
        f"> **Validation status: {state}**",
        "",
        "| Validation | Result | Required |",
        "|---|---:|---:|",
        f"| Total rows | {validation.total_rows} | {EXPECTED_ROWS} |",
        f"| Successful rows | {validation.success_rows} | {EXPECTED_ROWS} |",
        f"| Strict-valid rows | {validation.strict_rows} | {EXPECTED_ROWS} |",
        (
            "| Target-name violations | "
            f"{validation.target_name_violations} | 0 |"
        ),
        f"| Think-tag violations | {validation.think_tag_violations} | 0 |",
        (
            "| `New topic` violations | "
            f"{validation.new_topic_violations} | 0 |"
        ),
        (
            "| Empty reasoning rows | "
            f"{validation.empty_reasoning_rows} | 0 |"
        ),
        (
            "| Paired source IDs | "
            f"{validation.paired_source_ids} | {EXPECTED_SOURCE_PAIRS} |"
        ),
        f"| Pairing violations | {validation.pairing_violations} | 0 |",
        "",
        (
            "Every requirement in the table must be met before the preview "
            "can pass review."
        ),
        "",
    ]


def _row_lines(row: Mapping[str, object]) -> list[str]:
    task_type = str(_value(row, "task_type", default=""))
    label = {
        "behavior_only": "Behavior",
        "interest_only": "Interest",
    }.get(task_type, task_type or "Unknown")
    target = _value(
        row,
        "selected_target",
        "target_subcategory",
        "target",
        default="",
    )
    filter_reason = _value(row, "filter_reason", default="—")
    if filter_reason in (None, ""):
        filter_reason = "—"
    lines = [
        f"### {label}",
        "",
        f"- **Task type:** `{task_type}`",
        f"- **Target:** `{target}`",
        f"- **Request hash:** `{_value(row, 'request_hash', default='')}`",
        f"- **Status:** `{_value(row, 'status', default='')}`",
        f"- **Filter:** `{filter_reason}`",
    ]
    target_zh = _value(row, "selected_target_zh", "target_zh")
    if target_zh is not None:
        lines.insert(4, f"- **Target 中文：** `{target_zh}`")
    word_count = _value(
        row,
        "reasoning_word_count",
        "reasoning_words",
        "word_count",
    )
    token_count = _value(
        row,
        "reasoning_token_count",
        "reasoning_tokens",
        "token_count",
    )
    if word_count is not None:
        lines.append(f"- **Reasoning words:** {word_count}")
    if token_count is not None:
        lines.append(f"- **Reasoning tokens:** {token_count}")
    lines.extend(
        [
            "",
            "#### Original history",
            "",
            _fenced(_history_text(row)),
        ]
    )
    original_history_zh = _value(row, "original_history_zh")
    if original_history_zh is not None:
        lines.extend(
            [
                "",
                "#### 原始历史（中文翻译）",
                "",
                _fenced(original_history_zh),
            ]
        )
    lines.extend(
        [
            "",
            "#### Original prompt",
            "",
            _fenced(_value(row, "original_prompt", "prompt")),
        ]
    )
    original_prompt_zh = _value(row, "original_prompt_zh")
    if original_prompt_zh is not None:
        lines.extend(
            [
                "",
                "#### 原始 Prompt（中文翻译）",
                "",
                _fenced(original_prompt_zh),
            ]
        )
    lines.extend(
        [
            "",
            "#### New student prompt",
            "",
            _fenced(
                _value(
                    row,
                    "new_student_prompt",
                    "student_prompt",
                    default="",
                )
            ),
        ]
    )
    student_prompt_zh = _value(
        row,
        "new_student_prompt_zh",
        "student_prompt_zh",
    )
    if student_prompt_zh is not None:
        lines.extend(
            [
                "",
                "#### 新 Student Prompt（中文翻译）",
                "",
                _fenced(student_prompt_zh),
            ]
        )
    lines.extend(
        [
            "",
            "#### GPT reasoning",
            "",
            _fenced(_value(row, "reasoning", "gpt_reasoning")),
        ]
    )
    reasoning_zh = _value(row, "reasoning_zh", "gpt_reasoning_zh")
    if reasoning_zh is not None:
        lines.extend(
            [
                "",
                "#### GPT reasoning（中文翻译）",
                "",
                _fenced(reasoning_zh),
            ]
        )
    lines.extend(
        [
            "",
            "#### Final completion",
            "",
            _fenced(_final_completion(row)),
        ]
    )
    completion_zh = _value(row, "completion_zh", "final_completion_zh")
    if completion_zh is not None:
        lines.extend(
            [
                "",
                "#### 最终 Completion（中文翻译）",
                "",
                _fenced(completion_zh),
            ]
        )
    lines.append("")
    return lines


def render_preview_markdown(
    rows: Iterable[Mapping[str, object]],
) -> tuple[str, PreviewValidation]:
    materialized = list(rows)
    validation = validate_preview_rows(materialized)
    ordered = sorted(
        materialized,
        key=lambda row: (
            _selection_rank(row),
            str(_value(row, "source_sample_id", "source_id", default="")),
            TASK_ORDER.get(str(_value(row, "task_type", default="")), 99),
            str(_value(row, "sample_id", default="")),
        ),
    )
    lines = _summary_lines(validation)
    current_source: str | None = None
    pair_number = 0
    for row in ordered:
        source_id = str(
            _value(row, "source_sample_id", "source_id", default="")
        )
        if source_id != current_source:
            pair_number += 1
            current_source = source_id
            impression_id = _value(row, "impression_id", default="")
            selection_rank = _value(row, "selection_rank")
            rank_line = (
                f"- **Selection rank:** {selection_rank}"
                if selection_rank is not None
                else None
            )
            lines.extend(
                [
                    "---",
                    "",
                    f"## Pair {pair_number}: `{source_id}`",
                    "",
                    f"- **Source ID:** `{source_id}`",
                    f"- **Impression ID:** `{impression_id}`",
                ]
            )
            if rank_line is not None:
                lines.append(rank_line)
            lines.append("")
        lines.extend(_row_lines(row))
    return "\n".join(lines).rstrip() + "\n", validation


def render_cot_preview(
    teacher_records: Path | str,
    output_path: Path | str,
) -> PreviewResult:
    teacher_records = Path(teacher_records)
    output_path = Path(output_path)
    if not teacher_records.is_file():
        raise FileNotFoundError(teacher_records)
    rows = pq.read_table(teacher_records).to_pylist()
    markdown, validation = render_preview_markdown(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return PreviewResult(output_path=output_path, validation=validation)


def render_preview(
    teacher_records: Path | str,
    output_path: Path | str,
) -> PreviewResult:
    return render_cot_preview(teacher_records, output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a deterministic paired Markdown preview from local CoT "
            "v2 teacher records. This command makes no model or network calls."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
    )
    parser.add_argument(
        "--teacher-records",
        type=Path,
        dest="teacher_records",
    )
    parser.set_defaults(
        default_teacher_records=Path(
            "artifacts/cot_teacher/gpt56_sol_v2_preview/"
            "teacher_records.parquet"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/cot_teacher/gpt56_sol_v2_preview/preview.md"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    teacher_records = (
        args.teacher_records
        or args.input_path
        or args.default_teacher_records
    )
    result = render_cot_preview(teacher_records, args.output)
    state = "APPROVED" if result.validation.approved else "FAILED"
    print(f"{state}: {result.output_path}")
    return 0 if result.validation.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
