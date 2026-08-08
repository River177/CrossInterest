from __future__ import annotations

import re
from dataclasses import dataclass

_STRUCTURED_OUTPUT = re.compile(
    r"\A\s*<think>\s*(?P<think>\S[\s\S]*?)\s*</think>\s*"
    r"(?P<answer>New topic: [^\r\n]+)\s*\Z"
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
    match = _STRUCTURED_OUTPUT.fullmatch(solution_str)
    if match is None:
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


def extract_final_answer(solution_str: str) -> str | None:
    parsed = parse_structured_output(solution_str)
    if parsed is not None:
        return parsed.answer
    lines = [
        line.strip()
        for line in solution_str.strip().splitlines()
        if line.strip()
    ]
    if not lines or not lines[-1].startswith("New topic: "):
        return None
    return lines[-1]


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
) -> float:
    del data_source, extra_info
    parsed = parse_structured_output(solution_str)
    answer = parsed.answer if parsed is not None else extract_final_answer(
        solution_str
    )
    score = FORMAT_WEIGHT if parsed is not None else 0.0
    if answer == ground_truth:
        score += ANSWER_WEIGHT
    return score
