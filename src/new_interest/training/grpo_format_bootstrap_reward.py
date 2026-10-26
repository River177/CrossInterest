from __future__ import annotations

from dataclasses import dataclass

from new_interest.training.grpo_reward import (
    extract_final_answer,
    parse_structured_output,
)

MALFORMED_VALID_REWARD = 0.0
STRUCTURE_REWARD = 0.25
LENGTH_REWARD = 0.75
SOFT_MIN_WORDS = 8
IDEAL_MIN_WORDS = 16
IDEAL_MAX_WORDS = 64


@dataclass(frozen=True)
class FormatBootstrapScore:
    reward: float
    structured: bool
    valid: bool
    novel: bool
    label: str | None
    reasoning_words: int | None
    length_score: float


def reasoning_length_score(word_count: int) -> float:
    if word_count <= SOFT_MIN_WORDS:
        return 0.0
    if word_count < IDEAL_MIN_WORDS:
        return (
            (word_count - SOFT_MIN_WORDS)
            / (IDEAL_MIN_WORDS - SOFT_MIN_WORDS)
        )
    if word_count <= IDEAL_MAX_WORDS:
        return 1.0
    return IDEAL_MAX_WORDS / word_count


def score_format_bootstrap(
    solution_str: str,
    extra_info: dict[str, object] | None,
) -> FormatBootstrapScore:
    parsed = parse_structured_output(solution_str)
    answer = (
        parsed.answer
        if parsed is not None
        else extract_final_answer(solution_str, allow_bare_label=True)
    )
    if answer is None or extra_info is None:
        return FormatBootstrapScore(
            reward=0.0,
            structured=False,
            valid=False,
            novel=False,
            label=None,
            reasoning_words=None,
            length_score=0.0,
        )

    label = answer.removeprefix("New topic: ")
    retained = {
        str(value)
        for value in extra_info.get("retained_subcategories", [])
    }
    history = {
        str(value)
        for value in extra_info.get("history_subcategories", [])
    }
    valid = label in retained
    novel = label not in history
    if not (valid and novel):
        return FormatBootstrapScore(
            reward=0.0,
            structured=parsed is not None,
            valid=valid,
            novel=novel,
            label=label,
            reasoning_words=(
                len(parsed.think.split()) if parsed is not None else None
            ),
            length_score=0.0,
        )
    if parsed is None:
        return FormatBootstrapScore(
            reward=MALFORMED_VALID_REWARD,
            structured=False,
            valid=True,
            novel=True,
            label=label,
            reasoning_words=None,
            length_score=0.0,
        )

    reasoning_words = len(parsed.think.split())
    length_score = reasoning_length_score(reasoning_words)
    return FormatBootstrapScore(
        reward=STRUCTURE_REWARD + LENGTH_REWARD * length_score,
        structured=True,
        valid=True,
        novel=True,
        label=label,
        reasoning_words=reasoning_words,
        length_score=length_score,
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
) -> float:
    del data_source, ground_truth
    return score_format_bootstrap(solution_str, extra_info).reward
