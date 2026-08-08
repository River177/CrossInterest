from __future__ import annotations

from collections import Counter, defaultdict
import re
from statistics import mean

PREFIX = "New topic: "


def normalize_candidate_category(output: str) -> str:
    normalized = output.strip()
    if normalized.startswith(PREFIX):
        normalized = normalized[len(PREFIX) :].strip()
    normalized = normalized.strip("`\"' ")
    normalized = re.sub(
        r"^(?:option\s*)?(?:\d+|[A-J])\s*[.)\-:]\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    normalized = normalized.strip("`\"' ")
    if normalized.endswith("."):
        normalized = normalized[:-1].rstrip()
    return normalized.strip("`\"' ")


def parse_candidate_output(
    output: str,
    candidates: list[str],
    answer: str,
    answer_category: str,
) -> dict[str, object]:
    stripped = output.strip()
    prefix_compliant = stripped.startswith(PREFIX)
    normalized = normalize_candidate_category(stripped)
    matching_candidates = [
        candidate
        for candidate in candidates
        if candidate in stripped
    ]
    return {
        "output": output,
        "stripped_output": stripped,
        "normalized_category": normalized,
        "strict_correct": stripped == answer,
        "category_correct": normalized == answer_category,
        "candidate_valid": normalized in candidates,
        "prefix_compliant": prefix_compliant,
        "empty_output": not stripped,
        "multiple_candidates_mentioned": len(
            set(matching_candidates)
        )
        > 1,
        "extra_text": bool(matching_candidates) and normalized not in candidates,
    }


def _history_bucket(length: int) -> str:
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    if length <= 35:
        return "21-35"
    return "36-50"


def evaluate_candidate_outputs(
    records: list[dict[str, object]],
    outputs: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(records) != len(outputs):
        raise ValueError(
            f"Got {len(records)} records and {len(outputs)} outputs"
        )
    if not records:
        raise ValueError("Candidate evaluation records must be non-empty")

    details: list[dict[str, object]] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    positive_support: Counter[str] = Counter()
    positive_correct: Counter[str] = Counter()

    for record, output in zip(records, outputs):
        candidates = [str(value) for value in record["candidates"]]
        answer = str(record["answer"])
        answer_category = str(record["answer_category"])
        parsed = parse_candidate_output(
            output,
            candidates,
            answer,
            answer_category,
        )
        correct_position = int(record["correct_position"])
        history_length = int(record["history_length"])
        acceptable_count = len(record["all_acceptable_subcategories"])
        positive_support[answer_category] += 1
        positive_correct[answer_category] += int(
            parsed["category_correct"]
        )

        detail = {
            "selection_id": str(record["selection_id"]),
            "impression_id": str(record["impression_id"]),
            "answer": answer,
            "answer_category": answer_category,
            "candidates": candidates,
            "correct_position": correct_position,
            "history_length": history_length,
            "acceptable_count": acceptable_count,
            **parsed,
        }
        details.append(detail)

        groups = {
            "overall": "all",
            "position": str(correct_position + 1),
            "history_length": _history_bucket(history_length),
            "acceptable_count": str(acceptable_count),
        }
        for group_type, group_value in groups.items():
            key = f"{group_type}:{group_value}"
            for metric in (
                "strict_correct",
                "category_correct",
                "candidate_valid",
                "prefix_compliant",
                "empty_output",
                "multiple_candidates_mentioned",
                "extra_text",
            ):
                grouped[key][metric].append(float(parsed[metric]))

    def summarize(values: dict[str, list[float]]) -> dict[str, float]:
        return {
            metric: mean(metric_values)
            for metric, metric_values in sorted(values.items())
        }

    overall = summarize(grouped.pop("overall:all"))
    metrics: dict[str, object] = {
        "samples": len(records),
        "strict_accuracy": overall["strict_correct"],
        "category_accuracy": overall["category_correct"],
        "candidate_valid_rate": overall["candidate_valid"],
        "prefix_compliance_rate": overall["prefix_compliant"],
        "empty_output_rate": overall["empty_output"],
        "multiple_candidates_rate": overall[
            "multiple_candidates_mentioned"
        ],
        "extra_text_rate": overall["extra_text"],
        "by_group": {
            key: summarize(values)
            for key, values in sorted(grouped.items())
        },
        "per_category_accuracy": {
            category: positive_correct[category] / support
            for category, support in sorted(positive_support.items())
        },
        "per_category_support": dict(sorted(positive_support.items())),
    }
    return metrics, details
