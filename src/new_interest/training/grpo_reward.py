from __future__ import annotations

import re
from dataclasses import dataclass

_STRICT_STRUCTURED_OUTPUT = re.compile(
    r"\A<think>(?P<think>[\s\S]*?)</think>\s*"
    r"(?P<answer>New topic: [^\r\n]*\S)\Z"
)

ANSWER_WEIGHT = 0.8
FORMAT_WEIGHT = 0.2


@dataclass(frozen=True)
class ParsedStructuredOutput:
    think: str
    answer: str


def parse_structured_output(
    solution_str: str,
) -> ParsedStructuredOutput | None:
    match = _STRICT_STRUCTURED_OUTPUT.fullmatch(solution_str)
    if match is None or not match.group("think").strip():
        return None
    return ParsedStructuredOutput(
        think=match.group("think").strip(),
        answer=match.group("answer").strip(),
    )


def is_structured_exact(
    solution_str: str,
    ground_truth: str,
) -> bool:
    parsed = parse_structured_output(solution_str)
    if parsed is None:
        return False
    return parsed.answer == ground_truth


def extract_final_answer(
    solution_str: str,
    *,
    allow_bare_label: bool = False,
) -> str | None:
    parsed = parse_structured_output(solution_str)
    if parsed is not None:
        return parsed.answer
    lines = [
        line.strip()
        for line in solution_str.strip().splitlines()
        if line.strip()
    ]
    if (
        allow_bare_label
        and len(lines) == 1
        and not lines[0].startswith("New topic: ")
    ):
        return f"New topic: {lines[0]}"
    if not lines or not lines[-1].startswith("New topic: "):
        return None
    return lines[-1]


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
) -> float:
    del data_source, ground_truth
    parsed = parse_structured_output(solution_str)
    answer = (
        parsed.answer
        if parsed is not None
        else extract_final_answer(solution_str, allow_bare_label=True)
    )
    if answer is None or extra_info is None:
        return 0.0
    label = answer.removeprefix("New topic: ")
    retained = {
        str(value)
        for value in extra_info.get("retained_subcategories", [])
    }
    acceptable = {
        str(value)
        for value in extra_info.get("acceptable_subcategories", [])
    }
    history = {
        str(value)
        for value in extra_info.get("history_subcategories", [])
    }
    valid = label in retained
    novel = label not in history
    correct = label in acceptable
    if not (valid and novel and correct):
        return 0.0
    return ANSWER_WEIGHT + (FORMAT_WEIGHT if parsed is not None else 0.0)
