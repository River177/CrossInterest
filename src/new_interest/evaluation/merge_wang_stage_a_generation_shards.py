from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    _atomic_write_json,
    _atomic_write_parquet,
    canonical_sha256,
    record_input_sha256,
    sha256_file,
)
from new_interest.evaluation.evaluate_wang_stage_a_generation import (
    evaluate_generations,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def merge_generation_shards(
    shard_dirs: list[Path | str],
    eval_file: Path | str,
    labels_file: Path | str,
    output_dir: Path | str,
    *,
    expected_records: int | None = None,
) -> dict[str, object]:
    shard_dirs = [Path(path) for path in shard_dirs]
    eval_file = Path(eval_file)
    labels_file = Path(labels_file)
    output_dir = Path(output_dir)
    if not shard_dirs:
        raise ValueError("At least one generation shard is required")
    manifests = [_load_json(path / "shard_manifest.json") for path in shard_dirs]
    identities = {str(value["evaluation_identity"]) for value in manifests}
    if len(identities) != 1:
        raise ValueError("Generation shard identity mismatch")
    identity = next(iter(identities))
    num_shards_values = {int(value["num_shards"]) for value in manifests}
    if len(num_shards_values) != 1:
        raise ValueError("Generation shard num_shards mismatch")
    num_shards = next(iter(num_shards_values))
    if len(shard_dirs) != num_shards:
        raise ValueError("Generation shard directory count mismatch")
    indices = [int(value["shard_index"]) for value in manifests]
    if sorted(indices) != list(range(num_shards)):
        raise ValueError("Generation shard indices are incomplete")
    by_index = {
        int(manifest["shard_index"]): (directory, manifest)
        for directory, manifest in zip(shard_dirs, manifests)
    }
    reference = manifests[0]
    if sha256_file(eval_file) != str(reference["eval_sha256"]):
        raise ValueError("Generation eval hash mismatch")
    if sha256_file(labels_file) != str(reference["labels_sha256"]):
        raise ValueError("Generation labels hash mismatch")
    labels = sorted(
        {
            str(value)
            for value in pq.read_table(
                labels_file,
                columns=["target_subcategory"],
            )["target_subcategory"].to_pylist()
        }
    )
    if labels != [str(value) for value in reference["labels"]]:
        raise ValueError("Generation label order mismatch")
    records = pq.read_table(eval_file).to_pylist()
    max_examples = reference.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]
    if len(records) != int(reference["full_record_count"]):
        raise ValueError("Generation full record count mismatch")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError("Generation expected record count mismatch")

    rows_by_index: dict[int, dict[str, object]] = {}
    shard_hashes: dict[str, str] = {}
    parallel_elapsed = 0.0
    generated_samples = 0
    resumed_samples = 0
    for shard_index in range(num_shards):
        directory, manifest = by_index[shard_index]
        prediction_path = directory / "generation_predictions.parquet"
        actual_hash = sha256_file(prediction_path)
        if actual_hash != str(manifest["prediction_sha256"]):
            raise ValueError(f"Generation shard {shard_index} hash mismatch")
        shard_hashes[str(shard_index)] = actual_hash
        shard_metrics = _load_json(directory / "metrics.json")
        parallel_elapsed = max(parallel_elapsed, float(shard_metrics["elapsed_seconds"]))
        generated_samples += int(shard_metrics["generated_samples_this_run"])
        resumed_samples += int(shard_metrics["resumed_samples"])
        rows = pq.read_table(prediction_path).to_pylist()
        if len(rows) != int(manifest["logical_shard_record_count"]):
            raise ValueError(f"Generation shard {shard_index} row count mismatch")
        for row in rows:
            record_index = int(row["record_index"])
            if record_index % num_shards != shard_index:
                raise ValueError("Generation row belongs to another shard")
            if record_index in rows_by_index:
                raise ValueError("Duplicate generation record index")
            record = records[record_index]
            if str(row["impression_id"]) != str(record["impression_id"]):
                raise ValueError("Generation impression mismatch")
            if str(row["input_sha256"]) != record_input_sha256(record):
                raise ValueError("Generation input hash mismatch")
            if str(row["evaluation_identity"]) != identity:
                raise ValueError("Generation row identity mismatch")
            rows_by_index[record_index] = row
    missing = sorted(set(range(len(records))).difference(rows_by_index))
    if missing:
        raise ValueError(f"Missing {len(missing)} generation records")
    ordered_rows = [rows_by_index[index] for index in range(len(records))]
    outputs = [str(row["output"]) for row in ordered_rows]
    metrics, details = evaluate_generations(
        records,
        outputs,
        labels=set(labels),
        response_prefix=str(reference["response_prefix"]),
    )
    metrics.update(
        {
            "evaluation_identity": identity,
            "logical_num_shards": num_shards,
            "merged_prediction_count": len(records),
            "full_record_count": len(records),
            "model_name": reference["model_name"],
            "model_revision": reference.get("model_revision"),
            "adapter_sha256": reference.get("adapter_sha256"),
            "eval_sha256": reference["eval_sha256"],
            "labels_sha256": reference["labels_sha256"],
            "response_prefix": reference["response_prefix"],
            "decode": reference["decode"],
            "elapsed_seconds_parallel_wall": parallel_elapsed,
            "examples_per_second_parallel": (
                len(records) / parallel_elapsed if parallel_elapsed else None
            ),
            "generated_samples_this_run": generated_samples,
            "resumed_samples": resumed_samples,
        }
    )
    merged_rows = [
        {
            "record_index": int(source["record_index"]),
            "input_sha256": str(source["input_sha256"]),
            "evaluation_identity": identity,
            **detail,
        }
        for source, detail in zip(ordered_rows, details)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "generation_predictions.parquet"
    _atomic_write_parquet(prediction_path, merged_rows)
    _atomic_write_json(output_dir / "metrics.json", metrics)
    _atomic_write_json(
        output_dir / "merge_manifest.json",
        {
            "evaluation_identity": identity,
            "num_shards": num_shards,
            "record_count": len(records),
            "expected_record_count": expected_records,
            "shard_prediction_sha256": shard_hashes,
            "merged_prediction_sha256": sha256_file(prediction_path),
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
            "merge_identity": canonical_sha256(
                {
                    "evaluation_identity": identity,
                    "shard_prediction_sha256": shard_hashes,
                }
            ),
        },
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge Wang Stage-A generation shards."
    )
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()
    metrics = merge_generation_shards(
        args.shard_dirs,
        args.eval_file,
        args.labels_file,
        args.output_dir,
        expected_records=args.expected_records,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
