from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.evaluate_cot_label_ranking import (
    _write_rows_atomically,
    compute_all_metrics,
)


def aggregate_prediction_shards(
    *,
    prediction_files: list[Path],
    eval_file: Path,
    output_dir: Path,
    condition: str,
    task_type: str,
    cutoffs: tuple[int, ...] = (10, 20, 50),
) -> dict[str, object]:
    if not prediction_files:
        raise ValueError("At least one prediction file is required")
    rows = []
    for path in prediction_files:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["record_index"]))

    records = pq.read_table(eval_file).to_pylist()
    expected_indices = list(range(len(records)))
    actual_indices = [int(row["record_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(
            "Prediction shards do not cover each evaluation index exactly once"
        )
    expected_impressions = [
        str(record["impression_id"]) for record in records
    ]
    actual_impressions = [str(row["impression_id"]) for row in rows]
    if actual_impressions != expected_impressions:
        raise ValueError(
            "Prediction impressions do not match the evaluation data"
        )

    metric_groups, prediction_rows = compute_all_metrics(
        records,
        rows,
        cutoffs=cutoffs,
    )
    metrics = {
        "samples": len(records),
        "condition": condition,
        "task_type": task_type,
        "cutoffs": list(cutoffs),
        "source_prediction_files": [
            str(path) for path in prediction_files
        ],
        **metric_groups,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_atomically(
        output_dir / "predictions.parquet",
        prediction_rows,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate sharded CoT label-ranking predictions."
    )
    parser.add_argument(
        "--prediction-files",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("sft22k", "grpo1k"),
        required=True,
    )
    parser.add_argument(
        "--task-type",
        choices=("behavior_only", "interest_only"),
        required=True,
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[10, 20, 50],
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = aggregate_prediction_shards(
        prediction_files=args.prediction_files,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        condition=args.condition,
        task_type=args.task_type,
        cutoffs=tuple(args.cutoffs),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
