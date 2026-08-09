from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq


def exact_mcnemar_p_value(
    baseline_only_correct: int,
    comparison_only_correct: int,
) -> float:
    if baseline_only_correct < 0 or comparison_only_correct < 0:
        raise ValueError("Discordant counts must be non-negative")
    discordant = baseline_only_correct + comparison_only_correct
    if discordant == 0:
        return 1.0
    tail_end = min(baseline_only_correct, comparison_only_correct)
    log_probabilities = [
        (
            math.lgamma(discordant + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(discordant - successes + 1)
            - discordant * math.log(2.0)
        )
        for successes in range(tail_end + 1)
    ]
    largest = max(log_probabilities)
    one_sided = math.exp(largest) * sum(
        math.exp(value - largest) for value in log_probabilities
    )
    return min(1.0, 2.0 * one_sided)


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


def paired_bootstrap_delta(
    baseline_correct: list[bool],
    comparison_correct: list[bool],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    if len(baseline_correct) != len(comparison_correct):
        raise ValueError("Paired correctness vectors must have equal length")
    if not baseline_correct:
        raise ValueError("Paired correctness vectors must be non-empty")
    if replicates < 2000:
        raise ValueError("bootstrap replicates must be at least 2000")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    differences = [
        int(comparison) - int(baseline)
        for baseline, comparison in zip(
            baseline_correct,
            comparison_correct,
        )
    ]
    rng = random.Random(seed)
    sample_count = len(differences)
    deltas = [
        sum(
            differences[rng.randrange(sample_count)]
            for _ in range(sample_count)
        )
        / sample_count
        for _ in range(replicates)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "lower": _percentile(deltas, alpha),
        "upper": _percentile(deltas, 1.0 - alpha),
        "bootstrap_mean": mean(deltas),
    }


def _index_rows(
    rows: list[dict[str, object]],
    *,
    name: str,
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        if "impression_id" not in row:
            raise ValueError(f"{name} row is missing impression_id")
        impression_id = str(row["impression_id"])
        if impression_id in indexed:
            raise ValueError(
                f"{name} has duplicate impression_id {impression_id!r}"
            )
        indexed[impression_id] = row
    if not indexed:
        raise ValueError(f"{name} predictions must be non-empty")
    return indexed


def _correctness(
    row: dict[str, object],
    *,
    column: str,
    name: str,
) -> bool:
    if column not in row:
        raise ValueError(f"{name} row is missing {column!r}")
    value = row[column]
    if not isinstance(value, bool):
        raise ValueError(f"{name} {column!r} values must be booleans")
    return value


def compare_prediction_rows(
    baseline_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    *,
    accuracy_column: str = "set_hit_at_1",
    bootstrap_replicates: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
) -> dict[str, object]:
    baseline_by_id = _index_rows(baseline_rows, name="baseline")
    comparison_by_id = _index_rows(
        comparison_rows,
        name="comparison",
    )
    if baseline_by_id.keys() != comparison_by_id.keys():
        missing_from_comparison = sorted(
            baseline_by_id.keys() - comparison_by_id.keys()
        )
        missing_from_baseline = sorted(
            comparison_by_id.keys() - baseline_by_id.keys()
        )
        raise ValueError(
            "Prediction files must contain the same impression IDs; "
            f"missing from comparison={missing_from_comparison[:5]}, "
            f"missing from baseline={missing_from_baseline[:5]}"
        )

    impression_ids = sorted(baseline_by_id)
    baseline_correct = [
        _correctness(
            baseline_by_id[impression_id],
            column=accuracy_column,
            name="baseline",
        )
        for impression_id in impression_ids
    ]
    comparison_correct = [
        _correctness(
            comparison_by_id[impression_id],
            column=accuracy_column,
            name="comparison",
        )
        for impression_id in impression_ids
    ]
    samples = len(impression_ids)
    baseline_count = sum(baseline_correct)
    comparison_count = sum(comparison_correct)
    baseline_only = sum(
        baseline and not comparison
        for baseline, comparison in zip(
            baseline_correct,
            comparison_correct,
        )
    )
    comparison_only = sum(
        comparison and not baseline
        for baseline, comparison in zip(
            baseline_correct,
            comparison_correct,
        )
    )
    both_correct = sum(
        baseline and comparison
        for baseline, comparison in zip(
            baseline_correct,
            comparison_correct,
        )
    )
    both_wrong = samples - both_correct - baseline_only - comparison_only
    baseline_rate = baseline_count / samples
    comparison_rate = comparison_count / samples
    return {
        "samples": samples,
        "accuracy_column": accuracy_column,
        "baseline_correct_count": baseline_count,
        "baseline_incorrect_count": samples - baseline_count,
        "baseline_accuracy": baseline_rate,
        "comparison_correct_count": comparison_count,
        "comparison_incorrect_count": samples - comparison_count,
        "comparison_accuracy": comparison_rate,
        "accuracy_delta": comparison_rate - baseline_rate,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only_correct": baseline_only,
        "comparison_only_correct": comparison_only,
        "bootstrap": paired_bootstrap_delta(
            baseline_correct,
            comparison_correct,
            replicates=bootstrap_replicates,
            seed=seed,
            confidence=confidence,
        ),
        "mcnemar_exact_p_value": exact_mcnemar_p_value(
            baseline_only,
            comparison_only,
        ),
    }


def compare_paired_accuracy(
    baseline_path: Path | str,
    comparison_path: Path | str,
    *,
    accuracy_column: str = "set_hit_at_1",
    bootstrap_replicates: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
) -> dict[str, object]:
    baseline_path = Path(baseline_path)
    comparison_path = Path(comparison_path)
    baseline_rows = pq.read_table(
        baseline_path,
        columns=["impression_id", accuracy_column],
    ).to_pylist()
    comparison_rows = pq.read_table(
        comparison_path,
        columns=["impression_id", accuracy_column],
    ).to_pylist()
    metrics = compare_prediction_rows(
        baseline_rows,
        comparison_rows,
        accuracy_column=accuracy_column,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        confidence=confidence,
    )
    metrics.update(
        {
            "baseline_predictions": str(baseline_path),
            "comparison_predictions": str(comparison_path),
        }
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare paired accuracy predictions by impression ID."
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--comparison-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--accuracy-column", default="set_hit_at_1")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = compare_paired_accuracy(
        args.baseline_predictions,
        args.comparison_predictions,
        accuracy_column=args.accuracy_column,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        confidence=args.confidence,
    )
    rendered = json.dumps(metrics, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
