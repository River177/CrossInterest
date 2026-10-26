from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RankingExample:
    example_id: str
    positive_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingPrediction:
    example_id: str
    ranked_labels: tuple[str, ...]
    ranked_scores: tuple[float, ...]


@dataclass(frozen=True)
class RankingEvaluation:
    metrics: dict[str, object]
    details: list[dict[str, object]]


def _validate_cutoffs(cutoffs: tuple[int, ...]) -> tuple[int, ...]:
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    return tuple(sorted(set(cutoffs)))


def _index_examples(
    examples: Sequence[RankingExample],
) -> dict[str, RankingExample]:
    indexed: dict[str, RankingExample] = {}
    for example in examples:
        if not example.example_id:
            raise ValueError("example_id must be non-empty")
        if example.example_id in indexed:
            raise ValueError(f"Duplicate example_id {example.example_id!r}")
        positives = set(example.positive_labels)
        candidates = set(example.candidate_labels)
        if not positives:
            raise ValueError(
                f"{example.example_id}: positive_labels must be non-empty"
            )
        if len(positives) != len(example.positive_labels):
            raise ValueError(
                f"{example.example_id}: duplicate positive labels"
            )
        if len(candidates) != len(example.candidate_labels):
            raise ValueError(
                f"{example.example_id}: duplicate candidate labels"
            )
        if not positives.issubset(candidates):
            missing = sorted(positives - candidates)
            raise ValueError(
                f"{example.example_id}: positives missing from candidates "
                f"{missing}"
            )
        indexed[example.example_id] = example
    if not indexed:
        raise ValueError("examples must be non-empty")
    return indexed


def _index_predictions(
    predictions: Sequence[RankingPrediction],
) -> dict[str, RankingPrediction]:
    indexed: dict[str, RankingPrediction] = {}
    for prediction in predictions:
        if prediction.example_id in indexed:
            raise ValueError(
                f"Duplicate prediction for {prediction.example_id!r}"
            )
        if len(prediction.ranked_labels) != len(
            prediction.ranked_scores
        ):
            raise ValueError(
                f"{prediction.example_id}: labels and scores differ in length"
            )
        if len(set(prediction.ranked_labels)) != len(
            prediction.ranked_labels
        ):
            raise ValueError(
                f"{prediction.example_id}: duplicate ranked labels"
            )
        if any(not math.isfinite(score) for score in prediction.ranked_scores):
            raise ValueError(
                f"{prediction.example_id}: ranked scores must be finite"
            )
        indexed[prediction.example_id] = prediction
    return indexed


def _row_metrics(
    ranked_labels: tuple[str, ...],
    positives: set[str],
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    first_relevant_rank = next(
        (
            rank
            for rank, label in enumerate(ranked_labels, start=1)
            if label in positives
        ),
        None,
    )
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        retrieved = positives.intersection(ranked_labels[:cutoff])
        recall = len(retrieved) / len(positives)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, label in enumerate(
                ranked_labels[:cutoff],
                start=1,
            )
            if label in positives
        )
        ideal_hits = min(cutoff, len(positives))
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        metrics[f"recall_at_{cutoff}"] = recall
        metrics[f"ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0
        metrics[f"hit_at_{cutoff}"] = float(bool(retrieved))
    largest_cutoff = cutoffs[-1]
    metrics[f"mrr_at_{largest_cutoff}"] = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        and first_relevant_rank <= largest_cutoff
        else 0.0
    )
    metrics["ranking_length"] = float(len(ranked_labels))
    return metrics


def _mean_metrics(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not rows:
        return {}
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            values[name].append(float(value))
    return {
        name: mean(metric_values)
        for name, metric_values in sorted(values.items())
    }


def evaluate_rankings(
    examples: Sequence[RankingExample],
    predictions: Sequence[RankingPrediction],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> RankingEvaluation:
    cutoffs = _validate_cutoffs(cutoffs)
    examples_by_id = _index_examples(examples)
    predictions_by_id = _index_predictions(predictions)
    if examples_by_id.keys() != predictions_by_id.keys():
        missing_predictions = sorted(
            examples_by_id.keys() - predictions_by_id.keys()
        )
        unexpected_predictions = sorted(
            predictions_by_id.keys() - examples_by_id.keys()
        )
        raise ValueError(
            "Prediction IDs do not match examples; "
            f"missing={missing_predictions[:5]}, "
            f"unexpected={unexpected_predictions[:5]}"
        )

    per_example_metrics: list[dict[str, float]] = []
    details: list[dict[str, object]] = []
    per_attribute_rows: dict[
        str,
        dict[str, list[dict[str, float]]],
    ] = defaultdict(lambda: defaultdict(list))
    label_support: Counter[str] = Counter()
    label_retrieved: dict[int, Counter[str]] = {
        cutoff: Counter() for cutoff in cutoffs
    }
    top1_counts: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()

    for example_id in sorted(examples_by_id):
        example = examples_by_id[example_id]
        prediction = predictions_by_id[example_id]
        candidates = set(example.candidate_labels)
        ranked = set(prediction.ranked_labels)
        if ranked != candidates:
            missing = sorted(candidates - ranked)
            unexpected = sorted(ranked - candidates)
            raise ValueError(
                f"{example_id}: ranking does not exactly cover candidates; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        positives = set(example.positive_labels)
        row_metrics = _row_metrics(
            prediction.ranked_labels,
            positives,
            cutoffs,
        )
        per_example_metrics.append(row_metrics)
        candidate_counts[len(example.candidate_labels)] += 1
        target_counts[len(example.positive_labels)] += 1
        if prediction.ranked_labels:
            top1_counts[prediction.ranked_labels[0]] += 1
        for label in positives:
            label_support[label] += 1
            for cutoff in cutoffs:
                if label in prediction.ranked_labels[:cutoff]:
                    label_retrieved[cutoff][label] += 1
        for attribute_name, attribute_value in example.attributes.items():
            per_attribute_rows[attribute_name][str(attribute_value)].append(
                row_metrics
            )
        details.append(
            {
                "example_id": example_id,
                "positive_labels": list(example.positive_labels),
                "candidate_labels": list(example.candidate_labels),
                "ranked_labels": list(prediction.ranked_labels),
                "ranked_scores": list(prediction.ranked_scores),
                "attributes": dict(example.attributes),
                **row_metrics,
            }
        )

    macro_recall = {
        str(cutoff): mean(
            label_retrieved[cutoff][label] / support
            for label, support in label_support.items()
        )
        for cutoff in cutoffs
    }
    per_label_recall = {
        str(cutoff): {
            label: label_retrieved[cutoff][label] / support
            for label, support in sorted(label_support.items())
        }
        for cutoff in cutoffs
    }
    top1_total = sum(top1_counts.values())
    top1_entropy = -sum(
        (count / top1_total) * math.log(count / top1_total)
        for count in top1_counts.values()
    )
    label_space_size = len(
        {label for example in examples for label in example.candidate_labels}
    )
    metrics: dict[str, object] = {
        "samples": len(examples),
        "cutoffs": list(cutoffs),
        **_mean_metrics(per_example_metrics),
        "macro_recall_at_k": macro_recall,
        "per_label_recall": per_label_recall,
        "per_label_support": dict(sorted(label_support.items())),
        "by_attribute": {
            attribute_name: {
                attribute_value: {
                    "samples": len(rows),
                    **_mean_metrics(rows),
                }
                for attribute_value, rows in sorted(values.items())
            }
            for attribute_name, values in sorted(
                per_attribute_rows.items()
            )
        },
        "candidate_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(candidate_counts.items())
        },
        "target_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(target_counts.items())
        },
        "top1_label_counts": dict(
            sorted(
                top1_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "unique_top1_labels": len(top1_counts),
        "max_top1_label_share": (
            max(top1_counts.values(), default=0) / top1_total
            if top1_total
            else 0.0
        ),
        "top1_label_entropy": top1_entropy,
        "normalized_top1_label_entropy": (
            top1_entropy / math.log(label_space_size)
            if label_space_size > 1
            else 0.0
        ),
    }
    return RankingEvaluation(metrics=metrics, details=details)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def paired_bootstrap_metric_delta(
    baseline_details: Sequence[Mapping[str, object]],
    comparison_details: Sequence[Mapping[str, object]],
    *,
    metric_names: tuple[str, ...],
    replicates: int = 2_000,
    seed: int = 20260812,
    confidence: float = 0.95,
) -> dict[str, object]:
    if replicates < 2_000:
        raise ValueError("bootstrap replicates must be at least 2000")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    baseline = {
        str(row["example_id"]): row for row in baseline_details
    }
    comparison = {
        str(row["example_id"]): row for row in comparison_details
    }
    if baseline.keys() != comparison.keys() or not baseline:
        raise ValueError(
            "Paired bootstrap requires identical non-empty example IDs"
        )
    example_ids = sorted(baseline)
    deltas: dict[str, list[float]] = {}
    for metric_name in metric_names:
        deltas[metric_name] = [
            float(comparison[example_id][metric_name])
            - float(baseline[example_id][metric_name])
            for example_id in example_ids
        ]

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "paired bootstrap requires NumPy"
        ) from error
    rng = np.random.default_rng(seed)
    sample_count = len(example_ids)
    samples = {
        metric_name: [] for metric_name in metric_names
    }
    chunk_size = min(32, replicates)
    completed = 0
    delta_arrays = {
        metric_name: np.asarray(
            deltas[metric_name],
            dtype=np.float64,
        )
        for metric_name in metric_names
    }
    while completed < replicates:
        current_chunk = min(chunk_size, replicates - completed)
        indices = rng.integers(
            0,
            sample_count,
            size=(current_chunk, sample_count),
            dtype=np.int32,
        )
        for metric_name in metric_names:
            samples[metric_name].extend(
                delta_arrays[metric_name][indices].mean(axis=1).tolist()
            )
        completed += current_chunk
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "metrics": {
            metric_name: {
                "observed_delta": mean(deltas[metric_name]),
                "bootstrap_mean": mean(samples[metric_name]),
                "lower": _percentile(samples[metric_name], alpha),
                "upper": _percentile(
                    samples[metric_name],
                    1.0 - alpha,
                ),
            }
            for metric_name in metric_names
        },
    }
