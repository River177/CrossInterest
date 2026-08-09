from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_cot_accuracy import (
    _write_rows_atomically,
    summarize_accuracy_rows,
)


def aggregate_accuracy_predictions(
    prediction_files: list[Path | str],
    eval_file: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    if not prediction_files:
        raise ValueError("At least one prediction file is required")
    paths = [Path(path) for path in prediction_files]
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["record_index"]))

    records = pq.read_table(eval_file).to_pylist()
    expected_indices = list(range(len(records)))
    actual_indices = [int(row["record_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(
            "Prediction shards do not cover each evaluation index exactly once"
        )
    expected_ids = [str(row["impression_id"]) for row in records]
    actual_ids = [str(row["impression_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            "Prediction impressions do not match evaluation data"
        )
    modes = {str(row["mode"]) for row in rows}
    if len(modes) != 1:
        raise ValueError("Prediction shards contain multiple modes")

    metrics = {
        **summarize_accuracy_rows(rows),
        "mode": next(iter(modes)),
        "source_prediction_files": [str(path) for path in paths],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_atomically(output_dir / "predictions.parquet", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate sharded CoT/no-CoT accuracy predictions."
    )
    parser.add_argument(
        "--prediction-files",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = aggregate_accuracy_predictions(
        args.prediction_files,
        args.eval_file,
        args.output_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
