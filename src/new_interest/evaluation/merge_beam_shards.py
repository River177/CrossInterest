from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    PARQUET_PUBLICATION_VERSION,
    _atomic_write_json,
    _atomic_write_parquet,
    _load_labels,
    canonical_sha256,
    record_input_sha256,
    sha256_file,
)
from new_interest.evaluation.metrics import evaluate_ranked_outputs


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _per_interest_rows(
    metrics: dict[str, object],
    cutoffs: list[int],
) -> list[dict[str, object]]:
    supports = metrics["per_class_support"]
    recalls = metrics["per_class_recall"]
    ndcgs = metrics["per_class_ndcg"]
    if not isinstance(supports, dict):
        raise TypeError("per_class_support must be a mapping")
    if not isinstance(recalls, dict) or not isinstance(ndcgs, dict):
        raise TypeError("per-class metrics must be mappings")
    rows = []
    for interest, support in sorted(supports.items()):
        row: dict[str, object] = {
            "interest": str(interest),
            "support": int(support),
        }
        for cutoff in cutoffs:
            recall_values = recalls[str(cutoff)]
            ndcg_values = ndcgs[str(cutoff)]
            if not isinstance(recall_values, dict) or not isinstance(
                ndcg_values, dict
            ):
                raise TypeError("cutoff metrics must be mappings")
            row[f"recall_at_{cutoff}"] = float(recall_values[str(interest)])
            row[f"ndcg_at_{cutoff}"] = float(ndcg_values[str(interest)])
        rows.append(row)
    return rows


def merge_beam_shards(
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
        raise ValueError("At least one shard directory is required")

    manifests = [
        _load_json(shard_dir / "shard_manifest.json")
        for shard_dir in shard_dirs
    ]
    identities = {
        str(manifest["evaluation_identity"]) for manifest in manifests
    }
    if len(identities) != 1:
        raise ValueError(
            f"Shard evaluation identity mismatch: {sorted(identities)}"
        )
    evaluation_identity = next(iter(identities))
    num_shards_values = {int(item["num_shards"]) for item in manifests}
    if len(num_shards_values) != 1:
        raise ValueError("Shard num_shards values do not match")
    num_shards = next(iter(num_shards_values))
    if len(manifests) != num_shards:
        raise ValueError(
            f"Expected {num_shards} shard directories, got {len(manifests)}"
        )
    shard_indices = [int(item["shard_index"]) for item in manifests]
    if sorted(shard_indices) != list(range(num_shards)):
        raise ValueError(
            f"Shard indices must be 0..{num_shards - 1}, got "
            f"{sorted(shard_indices)}"
        )

    manifest_by_index = {
        int(manifest["shard_index"]): (shard_dir, manifest)
        for shard_dir, manifest in zip(shard_dirs, manifests)
    }
    labels = _load_labels(labels_file)
    label_set = set(labels)
    reference = manifests[0]
    if sha256_file(eval_file) != str(reference["eval_sha256"]):
        raise ValueError("Eval file SHA-256 does not match shard identity")
    if sha256_file(labels_file) != str(reference["labels_sha256"]):
        raise ValueError("Labels file SHA-256 does not match shard identity")
    if labels != [str(value) for value in reference["labels"]]:
        raise ValueError("Label vocabulary does not match shard identity")

    records = pq.read_table(eval_file).to_pylist()
    max_examples = reference.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]
    full_record_count = int(reference["full_record_count"])
    if len(records) != full_record_count:
        raise ValueError(
            f"Expected {full_record_count} eval rows, got {len(records)}"
        )
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(
            f"Expected exactly {expected_records} records, got {len(records)}"
        )

    top_k = int(reference["top_k"])
    cutoffs = [int(value) for value in reference["cutoffs"]]
    rows_by_index: dict[int, dict[str, object]] = {}
    shard_prediction_hashes: dict[str, str] = {}
    for shard_index in range(num_shards):
        shard_dir, manifest = manifest_by_index[shard_index]
        prediction_path = shard_dir / "beam_predictions.parquet"
        actual_prediction_hash = sha256_file(prediction_path)
        if actual_prediction_hash != str(manifest["prediction_sha256"]):
            raise ValueError(
                f"Shard {shard_index}: prediction SHA-256 mismatch"
            )
        shard_prediction_hashes[str(shard_index)] = actual_prediction_hash
        shard_rows = pq.read_table(prediction_path).to_pylist()
        if len(shard_rows) != int(manifest["shard_record_count"]):
            raise ValueError(
                f"Shard {shard_index}: manifest says "
                f"{manifest['shard_record_count']} rows, got "
                f"{len(shard_rows)}"
            )
        for row in shard_rows:
            record_index = int(row["record_index"])
            if record_index < 0 or record_index >= len(records):
                raise ValueError(
                    f"Shard {shard_index}: out-of-range record index "
                    f"{record_index}"
                )
            if record_index % num_shards != shard_index:
                raise ValueError(
                    f"Record {record_index} belongs to a different shard"
                )
            if record_index in rows_by_index:
                raise ValueError(f"Duplicate record index {record_index}")
            if str(row["evaluation_identity"]) != evaluation_identity:
                raise ValueError(
                    f"Record {record_index}: evaluation identity mismatch"
                )
            record = records[record_index]
            if str(row["impression_id"]) != str(record["impression_id"]):
                raise ValueError(
                    f"Record {record_index}: impression ID mismatch"
                )
            expected_input_hash = record_input_sha256(record)
            if str(row["input_sha256"]) != expected_input_hash:
                raise ValueError(
                    f"Record {record_index}: input SHA-256 mismatch"
                )
            ranked_labels = [str(value) for value in row["ranked_labels"]]
            ranked_responses = [
                str(value) for value in row["ranked_responses"]
            ]
            ranked_scores = [float(value) for value in row["ranked_scores"]]
            if not (
                len(ranked_labels)
                == len(ranked_responses)
                == len(ranked_scores)
                == top_k
            ):
                raise ValueError(
                    f"Record {record_index}: expected Top-{top_k} output"
                )
            if len(set(ranked_labels)) != top_k:
                raise ValueError(
                    f"Record {record_index}: ranked labels are not unique"
                )
            unknown = sorted(set(ranked_labels).difference(label_set))
            if unknown:
                raise ValueError(
                    f"Record {record_index}: unknown labels {unknown}"
                )
            expected_responses = [
                f"New topic: {label}" for label in ranked_labels
            ]
            if ranked_responses != expected_responses:
                raise ValueError(
                    f"Record {record_index}: response text does not match "
                    "ranked labels"
                )
            rows_by_index[record_index] = row

    missing = sorted(set(range(len(records))).difference(rows_by_index))
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"Missing {len(missing)} record indices; first missing: {preview}"
        )
    ordered_rows = [rows_by_index[index] for index in range(len(records))]
    ranked_outputs = [
        [
            {
                "label": label,
                "response": response,
                "score": score,
            }
            for label, response, score in zip(
                row["ranked_labels"],
                row["ranked_responses"],
                row["ranked_scores"],
            )
        ]
        for row in ordered_rows
    ]
    metrics, details = evaluate_ranked_outputs(
        records,
        ranked_outputs,
        cutoffs=tuple(cutoffs),
    )
    metrics.update(
        {
            "evaluation_identity": evaluation_identity,
            "logical_num_shards": num_shards,
            "merged_prediction_count": len(ordered_rows),
            "full_record_count": len(records),
            "num_beams": int(reference["num_beams"]),
            "top_k": top_k,
            "cutoffs": cutoffs,
            "model_name": str(reference["model_name"]),
            "adapter_sha256": str(reference["adapter_sha256"]),
            "adapter_hash_version": str(
                reference.get("adapter_hash_version", "legacy-directory-hash")
            ),
            "evaluation_identity_version": str(
                reference["identity_version"]
            ),
            "parquet_publication_version": PARQUET_PUBLICATION_VERSION,
            "eval_file": str(eval_file.resolve()),
            "eval_sha256": str(reference["eval_sha256"]),
            "labels_file": str(labels_file.resolve()),
            "labels_sha256": str(reference["labels_sha256"]),
            "label_vocabulary_size": len(labels),
            "dedup_retry_example_count": sum(
                int(int(row["dedup_retry_rounds"]) > 0)
                for row in ordered_rows
            ),
            "dedup_retry_round_count": sum(
                int(row["dedup_retry_rounds"])
                for row in ordered_rows
            ),
            "minimum_initial_unique_label_count": min(
                int(row["initial_unique_label_count"])
                for row in ordered_rows
            ),
        }
    )

    merged_rows = [
        {
            "record_index": int(source["record_index"]),
            "input_sha256": str(source["input_sha256"]),
            "evaluation_identity": evaluation_identity,
            **detail,
            "dedup_retry_rounds": int(source["dedup_retry_rounds"]),
            "initial_unique_label_count": int(
                source["initial_unique_label_count"]
            ),
            "ranked_responses": list(source["ranked_responses"]),
        }
        for source, detail in zip(ordered_rows, details)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "beam_predictions.parquet"
    _atomic_write_parquet(prediction_path, merged_rows)
    _atomic_write_json(output_dir / "metrics.json", metrics)
    per_interest_rows = _per_interest_rows(metrics, cutoffs)
    per_interest_path = output_dir / "per_interest_metrics.parquet"
    _atomic_write_parquet(per_interest_path, per_interest_rows)
    merge_manifest = {
        "evaluation_identity": evaluation_identity,
        "num_shards": num_shards,
        "record_count": len(merged_rows),
        "expected_record_count": expected_records,
        "shard_prediction_sha256": shard_prediction_hashes,
        "merged_prediction_sha256": sha256_file(prediction_path),
        "per_interest_metrics_sha256": sha256_file(per_interest_path),
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "merge_identity": canonical_sha256(
            {
                "evaluation_identity": evaluation_identity,
                "shard_prediction_sha256": shard_prediction_hashes,
            }
        ),
    }
    _atomic_write_json(output_dir / "merge_manifest.json", merge_manifest)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge constrained Beam logical shards."
    )
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument(
        "--retained-interests-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = merge_beam_shards(
        args.shard_dirs,
        args.eval_file,
        args.retained_interests_file,
        args.output_dir,
        expected_records=args.expected_records,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
