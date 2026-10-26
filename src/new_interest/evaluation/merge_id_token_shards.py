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
from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.training.semantic_id_tokens import load_interest_token_mapping


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
    rows = []
    for interest, support in sorted(supports.items()):
        row: dict[str, object] = {
            "interest": str(interest),
            "support": int(support),
        }
        for cutoff in cutoffs:
            row[f"recall_at_{cutoff}"] = float(
                recalls[str(cutoff)][interest]
            )
            row[f"ndcg_at_{cutoff}"] = float(
                ndcgs[str(cutoff)][interest]
            )
        rows.append(row)
    return rows


def merge_id_token_shards(
    shard_dirs: list[Path | str],
    eval_file: Path | str,
    mapping_file: Path | str,
    output_dir: Path | str,
    *,
    expected_records: int | None = None,
) -> dict[str, object]:
    shard_dirs = [Path(path) for path in shard_dirs]
    eval_file = Path(eval_file)
    mapping_file = Path(mapping_file)
    output_dir = Path(output_dir)
    if not shard_dirs:
        raise ValueError("At least one shard directory is required")
    manifests = [
        _load_json(path / "shard_manifest.json") for path in shard_dirs
    ]
    identities = {
        str(manifest["evaluation_identity"]) for manifest in manifests
    }
    if len(identities) != 1:
        raise ValueError("Shard evaluation identity mismatch")
    evaluation_identity = next(iter(identities))
    num_shards_values = {
        int(manifest["num_shards"]) for manifest in manifests
    }
    if len(num_shards_values) != 1:
        raise ValueError("Shard num_shards mismatch")
    num_shards = next(iter(num_shards_values))
    if len(manifests) != num_shards:
        raise ValueError(
            f"Expected {num_shards} shard directories, got {len(manifests)}"
        )
    indices = [int(manifest["shard_index"]) for manifest in manifests]
    if sorted(indices) != list(range(num_shards)):
        raise ValueError("Shard indices do not cover 0..num_shards-1")
    manifest_by_index = {
        int(manifest["shard_index"]): (path, manifest)
        for path, manifest in zip(shard_dirs, manifests)
    }

    reference = manifests[0]
    if sha256_file(eval_file) != str(reference["eval_sha256"]):
        raise ValueError("Eval SHA-256 does not match shard identity")
    if sha256_file(mapping_file) != str(reference["mapping_sha256"]):
        raise ValueError("Mapping SHA-256 does not match shard identity")
    mapping_rows = load_interest_token_mapping(mapping_file)
    labels = [str(row["target_subcategory"]) for row in mapping_rows]
    tokens = [str(row["interest_token"]) for row in mapping_rows]
    token_by_label = dict(zip(labels, tokens))
    if labels != [str(value) for value in reference["labels"]]:
        raise ValueError("Label order does not match shard identity")
    if tokens != [str(value) for value in reference["tokens"]]:
        raise ValueError("Token order does not match shard identity")

    records = pq.read_table(eval_file).to_pylist()
    max_examples = reference.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]
    if len(records) != int(reference["full_record_count"]):
        raise ValueError("Eval row count does not match shard identity")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(
            f"Expected exactly {expected_records} records, got {len(records)}"
        )
    top_k = int(reference["top_k"])
    cutoffs = [int(value) for value in reference["cutoffs"]]
    rows_by_index: dict[int, dict[str, object]] = {}
    shard_hashes: dict[str, str] = {}
    elapsed_seconds = 0.0
    generated_samples = 0
    resumed_samples = 0
    for shard_index in range(num_shards):
        shard_dir, manifest = manifest_by_index[shard_index]
        prediction_path = shard_dir / "id_token_predictions.parquet"
        actual_hash = sha256_file(prediction_path)
        if actual_hash != str(manifest["prediction_sha256"]):
            raise ValueError(f"Shard {shard_index}: prediction hash mismatch")
        shard_hashes[str(shard_index)] = actual_hash
        elapsed_seconds = max(
            elapsed_seconds,
            float(manifest.get("elapsed_seconds", 0.0)),
        )
        generated_samples += int(manifest.get("generated_samples", 0))
        resumed_samples += int(manifest.get("resumed_samples", 0))
        rows = pq.read_table(prediction_path).to_pylist()
        if len(rows) != int(manifest["shard_record_count"]):
            raise ValueError(f"Shard {shard_index}: row count mismatch")
        for row in rows:
            record_index = int(row["record_index"])
            if record_index % num_shards != shard_index:
                raise ValueError(
                    f"Record {record_index} belongs to another shard"
                )
            if record_index in rows_by_index:
                raise ValueError(f"Duplicate record index {record_index}")
            if not 0 <= record_index < len(records):
                raise ValueError(f"Out-of-range record index {record_index}")
            record = records[record_index]
            if str(row["evaluation_identity"]) != evaluation_identity:
                raise ValueError(
                    f"Record {record_index}: evaluation identity mismatch"
                )
            if str(row["impression_id"]) != str(record["impression_id"]):
                raise ValueError(
                    f"Record {record_index}: impression ID mismatch"
                )
            if str(row["input_sha256"]) != record_input_sha256(record):
                raise ValueError(f"Record {record_index}: input hash mismatch")
            ranked_labels = [str(value) for value in row["ranked_labels"]]
            ranked_tokens = [str(value) for value in row["ranked_tokens"]]
            ranked_scores = [float(value) for value in row["ranked_scores"]]
            if not (
                len(ranked_labels)
                == len(ranked_tokens)
                == len(ranked_scores)
                == top_k
            ):
                raise ValueError(
                    f"Record {record_index}: expected Top-{top_k}"
                )
            if len(set(ranked_labels)) != top_k:
                raise ValueError(
                    f"Record {record_index}: duplicate ranked labels"
                )
            if ranked_tokens != [
                token_by_label.get(label) for label in ranked_labels
            ]:
                raise ValueError(
                    f"Record {record_index}: token/label mapping mismatch"
                )
            rows_by_index[record_index] = row

    missing = sorted(set(range(len(records))).difference(rows_by_index))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} record indices; first: {missing[:20]}"
        )
    ordered_rows = [rows_by_index[index] for index in range(len(records))]
    ranked_outputs = [
        [
            {
                "label": label,
                "response": token,
                "score": score,
            }
            for label, token, score in zip(
                row["ranked_labels"],
                row["ranked_tokens"],
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
    entropies = [float(row["restricted_entropy"]) for row in ordered_rows]
    metrics.update(
        {
            "evaluation_identity": evaluation_identity,
            "logical_num_shards": num_shards,
            "merged_prediction_count": len(ordered_rows),
            "full_record_count": len(records),
            "top_k": top_k,
            "cutoffs": cutoffs,
            "model_name": str(reference["model_name"]),
            "adapter_sha256": str(reference["adapter_sha256"]),
            "eval_file": str(eval_file.resolve()),
            "eval_sha256": str(reference["eval_sha256"]),
            "mapping_file": str(mapping_file.resolve()),
            "mapping_sha256": str(reference["mapping_sha256"]),
            "label_vocabulary_size": len(labels),
            "mean_restricted_entropy": sum(entropies) / len(entropies),
            "history_prediction_filtering": False,
            "elapsed_seconds_parallel_wall": elapsed_seconds,
            "examples_per_second_parallel": (
                len(records) / elapsed_seconds if elapsed_seconds else None
            ),
            "generated_samples": generated_samples,
            "resumed_samples": resumed_samples,
        }
    )
    if reference.get("model_revision") is not None:
        metrics["model_revision"] = str(reference["model_revision"])
    merged_rows = [
        {
            "record_index": int(source["record_index"]),
            "input_sha256": str(source["input_sha256"]),
            "evaluation_identity": evaluation_identity,
            **detail,
            "ranked_tokens": list(source["ranked_tokens"]),
            "restricted_entropy": float(source["restricted_entropy"]),
        }
        for source, detail in zip(ordered_rows, details)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "id_token_predictions.parquet"
    _atomic_write_parquet(prediction_path, merged_rows)
    _atomic_write_json(output_dir / "metrics.json", metrics)
    per_interest_path = output_dir / "per_interest_metrics.parquet"
    _atomic_write_parquet(
        per_interest_path,
        _per_interest_rows(metrics, cutoffs),
    )
    merge_manifest = {
        "evaluation_identity": evaluation_identity,
        "num_shards": num_shards,
        "record_count": len(merged_rows),
        "expected_record_count": expected_records,
        "shard_prediction_sha256": shard_hashes,
        "merged_prediction_sha256": sha256_file(prediction_path),
        "per_interest_metrics_sha256": sha256_file(per_interest_path),
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "merge_identity": canonical_sha256(
            {
                "evaluation_identity": evaluation_identity,
                "shard_prediction_sha256": shard_hashes,
            }
        ),
    }
    _atomic_write_json(output_dir / "merge_manifest.json", merge_manifest)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge Semantic ID Token logical shards."
    )
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument(
        "--interest-token-mapping-file", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = merge_id_token_shards(
        args.shard_dirs,
        args.eval_file,
        args.interest_token_mapping_file,
        args.output_dir,
        expected_records=args.expected_records,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
