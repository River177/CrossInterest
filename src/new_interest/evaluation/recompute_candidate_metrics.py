from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.candidate_metrics import evaluate_candidate_outputs


def recompute(data_file: Path, result_dir: Path) -> dict[str, object]:
    predictions_path = result_dir / "predictions.parquet"
    metrics_path = result_dir / "metrics.json"
    records = pq.read_table(data_file).to_pylist()
    prediction_rows = pq.read_table(predictions_path).to_pylist()
    if len(records) != len(prediction_rows):
        raise ValueError(
            f"{data_file}: {len(records)} rows but "
            f"{predictions_path}: {len(prediction_rows)} rows"
        )
    if [row["selection_id"] for row in records] != [
        row["selection_id"] for row in prediction_rows
    ]:
        raise ValueError("Selection IDs do not match")

    outputs = [str(row["output"]) for row in prediction_rows]
    new_metrics, new_details = evaluate_candidate_outputs(records, outputs)
    old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    merged_metrics = {**old_metrics, **new_metrics}
    merged_predictions = [
        {**old_row, **new_detail}
        for old_row, new_detail in zip(prediction_rows, new_details)
    ]

    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=".predictions.",
        dir=result_dir,
        delete=False,
    ) as handle:
        temporary_predictions = Path(handle.name)
    with tempfile.NamedTemporaryFile(
        suffix=".json",
        prefix=".metrics.",
        dir=result_dir,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        json.dump(merged_metrics, handle, indent=2)
        handle.write("\n")
        temporary_metrics = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(merged_predictions),
            temporary_predictions,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary_predictions, predictions_path)
        os.replace(temporary_metrics, metrics_path)
    finally:
        temporary_predictions.unlink(missing_ok=True)
        temporary_metrics.unlink(missing_ok=True)
    return merged_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute candidate metrics from saved raw outputs."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        json.dumps(
            recompute(args.data_file, args.result_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
