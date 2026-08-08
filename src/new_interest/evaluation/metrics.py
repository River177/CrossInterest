from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from statistics import mean

RESPONSE_PREFIX = "New topic: "


@dataclass(frozen=True)
class ParsedPrediction:
    raw: str
    stripped: str
    has_prefix: bool
    label: str | None
    valid_label: bool


def parse_prediction(
    output: str,
    valid_labels: set[str],
) -> ParsedPrediction:
    stripped = output.strip()
    has_prefix = stripped.startswith(RESPONSE_PREFIX)
    label = (
        stripped[len(RESPONSE_PREFIX) :].strip()
        if has_prefix
        else None
    )
    return ParsedPrediction(
        raw=output,
        stripped=stripped,
        has_prefix=has_prefix,
        label=label,
        valid_label=label in valid_labels if label is not None else False,
    )


def evaluate_outputs(
    records: list[dict[str, object]],
    outputs: list[str],
    valid_labels: set[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(records) != len(outputs):
        raise ValueError(
            f"Got {len(records)} records and {len(outputs)} outputs"
        )
    if not records:
        raise ValueError("Evaluation records must be non-empty")

    set_hits = 0
    strict_hits = 0
    prefix_hits = 0
    valid_hits = 0
    novelty_hits = 0
    target_support: Counter[str] = Counter()
    target_hits: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    details: list[dict[str, object]] = []

    for record, output in zip(records, outputs):
        acceptable_subcategories = [
            str(value)
            for value in record["acceptable_subcategories"]
        ]
        acceptable_responses = {
            str(value) for value in record["acceptable_responses"]
        }
        history_subcategories = {
            str(value) for value in record["history_subcategories"]
        }
        parsed = parse_prediction(output, valid_labels)
        set_hit = (
            parsed.label in acceptable_subcategories
            if parsed.label is not None
            else False
        )
        strict_hit = parsed.stripped in acceptable_responses
        novelty_compliant = (
            parsed.valid_label
            and parsed.label not in history_subcategories
        )

        set_hits += int(set_hit)
        strict_hits += int(strict_hit)
        prefix_hits += int(parsed.has_prefix)
        valid_hits += int(parsed.valid_label)
        novelty_hits += int(novelty_compliant)
        if parsed.label is not None:
            prediction_counts[parsed.label] += 1
        for target in acceptable_subcategories:
            target_support[target] += 1
            if parsed.label == target:
                target_hits[target] += 1

        details.append(
            {
                "impression_id": str(record["impression_id"]),
                "prompt": str(record["prompt"]),
                "output": output,
                "parsed_label": parsed.label,
                "has_prefix": parsed.has_prefix,
                "valid_label": parsed.valid_label,
                "set_hit_at_1": set_hit,
                "strict_response_hit_at_1": strict_hit,
                "novelty_compliant": novelty_compliant,
                "acceptable_subcategories": acceptable_subcategories,
                "acceptable_responses": sorted(acceptable_responses),
            }
        )

    per_class_recall = {
        target: target_hits[target] / support
        for target, support in sorted(target_support.items())
    }
    total = len(records)
    metrics: dict[str, object] = {
        "samples": total,
        "set_hit_at_1": set_hits / total,
        "strict_response_hit_at_1": strict_hits / total,
        "prefix_compliance_rate": prefix_hits / total,
        "valid_label_rate": valid_hits / total,
        "novelty_compliance_rate": novelty_hits / total,
        "macro_target_recall": mean(per_class_recall.values()),
        "per_class_recall": per_class_recall,
        "per_class_support": dict(sorted(target_support.items())),
        "prediction_counts": dict(
            sorted(
                prediction_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }
    return metrics, details


def _dcg_at_k(
    ranked_labels: list[str],
    acceptable: set[str],
    k: int,
) -> float:
    return sum(
        1.0 / log2(rank + 1)
        for rank, label in enumerate(ranked_labels[:k], start=1)
        if label in acceptable
    )


def evaluate_ranked_outputs(
    records: list[dict[str, object]],
    ranked_outputs: list[list[dict[str, object]]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(records) != len(ranked_outputs):
        raise ValueError(
            f"Got {len(records)} records and "
            f"{len(ranked_outputs)} ranked outputs"
        )
    if not records:
        raise ValueError("Evaluation records must be non-empty")
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("cutoffs must be positive")
    cutoffs = tuple(sorted(set(cutoffs)))

    metric_values: dict[str, list[float]] = defaultdict(list)
    answer_count_metrics: dict[
        int,
        dict[str, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    class_support: Counter[str] = Counter()
    class_retrieved: dict[int, Counter[str]] = {
        k: Counter() for k in cutoffs
    }
    details: list[dict[str, object]] = []

    for record, predictions in zip(records, ranked_outputs):
        acceptable_list = [
            str(value)
            for value in record["acceptable_subcategories"]
        ]
        acceptable = set(acceptable_list)
        labels = [str(item["label"]) for item in predictions]
        scores = [float(item["score"]) for item in predictions]
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"{record['impression_id']}: duplicate ranked labels"
            )
        answer_count = len(acceptable)
        row_metrics: dict[str, float] = {}
        first_relevant_rank: int | None = None
        for rank, label in enumerate(labels, start=1):
            if label in acceptable:
                first_relevant_rank = rank
                break

        for k in cutoffs:
            top_k = labels[:k]
            retrieved = acceptable.intersection(top_k)
            recall = len(retrieved) / answer_count
            hit = float(bool(retrieved))
            dcg = _dcg_at_k(labels, acceptable, k)
            ideal_hits = min(k, answer_count)
            idcg = sum(
                1.0 / log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )
            ndcg = dcg / idcg if idcg else 0.0
            row_metrics[f"recall_at_{k}"] = recall
            row_metrics[f"ndcg_at_{k}"] = ndcg
            row_metrics[f"hit_at_{k}"] = hit
            for target in acceptable:
                class_support[target] += int(k == cutoffs[0])
                if target in top_k:
                    class_retrieved[k][target] += 1
        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            and first_relevant_rank <= cutoffs[-1]
            else 0.0
        )
        row_metrics[f"mrr_at_{cutoffs[-1]}"] = reciprocal_rank
        row_metrics["ranking_length"] = float(len(labels))
        for key, value in row_metrics.items():
            metric_values[key].append(value)
            answer_count_metrics[answer_count][key].append(value)
        details.append(
            {
                "impression_id": str(record["impression_id"]),
                "ranked_labels": labels,
                "ranked_scores": scores,
                "acceptable_subcategories": acceptable_list,
                **row_metrics,
            }
        )

    metrics: dict[str, object] = {
        "samples": len(records),
        **{
            key: mean(values)
            for key, values in sorted(metric_values.items())
        },
        "by_answer_count": {
            str(answer_count): {
                key: mean(values)
                for key, values in sorted(group_metrics.items())
            }
            for answer_count, group_metrics in sorted(
                answer_count_metrics.items()
            )
        },
        "per_class_recall": {
            str(k): {
                target: (
                    class_retrieved[k][target] / support
                    if support
                    else 0.0
                )
                for target, support in sorted(class_support.items())
            }
            for k in cutoffs
        },
        "per_class_support": dict(sorted(class_support.items())),
    }
    return metrics, details
