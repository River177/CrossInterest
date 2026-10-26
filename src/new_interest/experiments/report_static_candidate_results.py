from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.run_static_candidate_condition import (
    DEFAULT_MANIFEST,
    _load_manifest,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-candidate-report-v1"
DEFAULT_DATASET_DIR = Path("data/processed/new_interest_static_user_disjoint_v1")
DEFAULT_BASELINE_DIR = Path(
    "artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/candidate-selection-static-user-disjoint-k10-v1/analysis"
)
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_SEED = 20260815
BASELINES = ("global_popularity", "last_interest_actual", "semantic_actual")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_bucket(length: int) -> str:
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    if length <= 35:
        return "21-35"
    return "36+"


def _frequency_tiers(train_counts: Counter[str]) -> dict[str, str]:
    labels = sorted(train_counts, key=lambda label: (-train_counts[label], label))
    head_end = max(1, math.ceil(len(labels) * 0.2))
    torso_end = max(head_end + 1, math.ceil(len(labels) * 0.6))
    return {
        label: (
            "head"
            if index < head_end
            else "torso"
            if index < torso_end
            else "tail"
        )
        for index, label in enumerate(labels)
    }


def _macro_accuracy(
    correct: np.ndarray,
    categories: np.ndarray,
) -> float:
    labels, inverse = np.unique(categories, return_inverse=True)
    support = np.bincount(inverse, minlength=len(labels))
    hits = np.bincount(
        inverse, weights=correct.astype(np.int8), minlength=len(labels)
    )
    return float(np.mean(hits / support))


def _metric_rows(
    condition_key: str,
    metadata: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    result: Mapping[str, np.ndarray],
    frequency_tier: Mapping[str, str],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    answers = np.asarray([str(row["answer_category"]) for row in records])
    correct = np.asarray(result["correct"], dtype=bool)
    strict = np.asarray(result["strict_correct"], dtype=bool)
    valid = np.asarray(result["candidate_valid"], dtype=bool)
    prefix = np.asarray(result["prefix_compliant"], dtype=bool)
    if not (len(records) == len(correct) == len(strict) == len(valid) == len(prefix)):
        raise ValueError(f"{condition_key}: result length mismatch")

    per_interest: list[dict[str, object]] = []
    for label in sorted(set(answers)):
        mask = answers == label
        per_interest.append(
            {
                "condition": condition_key,
                **metadata,
                "interest": label,
                "frequency_tier": frequency_tier[label],
                "support": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
            }
        )

    slices: list[dict[str, object]] = []
    slice_values = {
        "answer_position": [
            str(int(row["correct_position"]) + 1) for row in records
        ],
        "history_length": [
            _history_bucket(int(row["history_length"])) for row in records
        ],
        "interest_frequency": [frequency_tier[value] for value in answers],
    }
    for slice_name, values in slice_values.items():
        values_array = np.asarray(values)
        for value in sorted(set(values)):
            mask = values_array == value
            slices.append(
                {
                    "condition": condition_key,
                    **metadata,
                    "slice": slice_name,
                    "value": value,
                    "samples": int(mask.sum()),
                    "supported_interests": int(len(set(answers[mask]))),
                    "micro_accuracy": float(correct[mask].mean()),
                    "interest_macro_accuracy": _macro_accuracy(
                        correct[mask], answers[mask]
                    ),
                }
            )

    summary = {
        "condition": condition_key,
        **metadata,
        "samples": len(records),
        "supported_interests": len(set(answers)),
        "interest_macro_accuracy": _macro_accuracy(correct, answers),
        "micro_accuracy": float(correct.mean()),
        "strict_format_accuracy": float(strict.mean()),
        "candidate_valid_rate": float(valid.mean()),
        "prefix_compliance_rate": float(prefix.mean()),
    }
    return summary, per_interest, slices


def _aligned_rows(
    path: Path,
    expected_ids: Sequence[str],
) -> list[dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    by_id = {str(row["selection_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{path}: duplicate selection_id")
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))[:5]
        extra = sorted(set(by_id) - set(expected_ids))[:5]
        raise ValueError(f"{path}: identity mismatch; missing={missing}, extra={extra}")
    return [by_id[value] for value in expected_ids]


def _load_model_result(
    prediction_path: Path,
    records: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    expected_ids = [str(row["selection_id"]) for row in records]
    rows = _aligned_rows(prediction_path, expected_ids)
    for expected, row in zip(records, rows, strict=True):
        if str(row["answer_category"]) != str(expected["answer_category"]):
            raise ValueError(f"{prediction_path}: answer mismatch")
    return {
        "correct": np.asarray([bool(row["category_correct"]) for row in rows]),
        "strict_correct": np.asarray([bool(row["strict_correct"]) for row in rows]),
        "candidate_valid": np.asarray([bool(row["candidate_valid"]) for row in rows]),
        "prefix_compliant": np.asarray(
            [bool(row["prefix_compliant"]) for row in rows]
        ),
    }


def _load_baseline_result(
    prediction_path: Path,
    records: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    expected_ids = [str(row["selection_id"]) for row in records]
    rows = _aligned_rows(prediction_path, expected_ids)
    correct = np.asarray(
        [
            str(row["predicted_category"]) == str(expected["answer_category"])
            for expected, row in zip(records, rows, strict=True)
        ]
    )
    ones = np.ones(len(rows), dtype=bool)
    return {
        "correct": correct,
        "strict_correct": correct.copy(),
        "candidate_valid": ones,
        "prefix_compliant": ones.copy(),
    }


def _comparison_specs(
    model_metadata: Mapping[str, Mapping[str, object]],
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for condition_id, metadata in sorted(model_metadata.items()):
        strategy = str(metadata["strategy"])
        model_key = f"{strategy}/{condition_id}"
        for baseline in BASELINES:
            specs.append(
                {
                    "family": "model_vs_baseline",
                    "name": f"{model_key}_minus_{strategy}/{baseline}",
                    "left": model_key,
                    "right": f"{strategy}/{baseline}",
                }
            )

    identifiers = set(model_metadata)
    for condition_id, metadata in sorted(model_metadata.items()):
        if "_interest_" in condition_id:
            behavior = condition_id.replace("_interest_", "_behavior_", 1)
            if behavior in identifiers:
                strategy = str(metadata["strategy"])
                specs.append(
                    {
                        "family": "interest_vs_behavior",
                        "name": f"{condition_id}_minus_{behavior}",
                        "left": f"{strategy}/{condition_id}",
                        "right": f"{strategy}/{behavior}",
                    }
                )
        if "_medium_" in condition_id:
            none = condition_id.replace("_medium_", "_none_", 1)
            if none in identifiers:
                strategy = str(metadata["strategy"])
                specs.append(
                    {
                        "family": "medium_vs_none",
                        "name": f"{condition_id}_minus_{none}",
                        "left": f"{strategy}/{condition_id}",
                        "right": f"{strategy}/{none}",
                    }
                )
        if condition_id.endswith("_frequency_matched"):
            uniform = condition_id.removesuffix("_frequency_matched") + "_uniform"
            if uniform in identifiers:
                specs.append(
                    {
                        "family": "frequency_matched_vs_uniform",
                        "name": f"{condition_id}_minus_{uniform}",
                        "left": f"train_frequency_matched/{condition_id}",
                        "right": f"uniform/{uniform}",
                    }
                )
    names = [row["name"] for row in specs]
    if len(names) != len(set(names)):
        raise AssertionError("Duplicate comparison names")
    return specs


def bootstrap_interest_macro(
    records: Sequence[Mapping[str, object]],
    correctness: Mapping[str, np.ndarray],
    comparisons: Sequence[Mapping[str, str]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("Bootstrap requires scipy") from error
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    users = sorted({str(row["user_id"]) for row in records})
    categories = sorted({str(row["answer_category"]) for row in records})
    user_to_id = {value: index for index, value in enumerate(users)}
    category_to_id = {value: index for index, value in enumerate(categories)}
    row_users = np.asarray(
        [user_to_id[str(row["user_id"])] for row in records], dtype=np.int32
    )
    row_categories = np.asarray(
        [category_to_id[str(row["answer_category"])] for row in records],
        dtype=np.int16,
    )
    support = sparse.csr_matrix(
        (np.ones(len(records), dtype=np.int16), (row_users, row_categories)),
        shape=(len(users), len(categories)),
    )
    matrices = {}
    for condition, values in correctness.items():
        array = np.asarray(values, dtype=bool)
        if len(array) != len(records):
            raise ValueError(f"{condition}: correctness length mismatch")
        matrices[condition] = sparse.csr_matrix(
            (array.astype(np.int8), (row_users, row_categories)),
            shape=(len(users), len(categories)),
        )
    needed = {str(row[key]) for row in comparisons for key in ("left", "right")}
    missing = sorted(needed - set(matrices))
    if missing:
        raise ValueError(f"Comparison conditions are missing: {missing}")

    samples = {
        condition: np.empty(replicates, dtype=np.float64)
        for condition in matrices
    }
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(users), 1.0 / len(users))
    offset = 0
    while offset < replicates:
        size = min(50, replicates - offset)
        weights = rng.multinomial(len(users), probabilities, size=size).T
        sampled_support = np.asarray(support.T @ weights, dtype=np.float64)
        present = sampled_support > 0
        present_count = present.sum(axis=0)
        for condition, matrix in matrices.items():
            hits = np.asarray(matrix.T @ weights, dtype=np.float64)
            accuracy = np.divide(
                hits,
                sampled_support,
                out=np.zeros_like(hits),
                where=present,
            )
            samples[condition][offset : offset + size] = (
                accuracy.sum(axis=0) / present_count
            )
        offset += size

    answers = np.asarray([str(row["answer_category"]) for row in records])
    intervals = {
        "rule_version": RULE_VERSION,
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "resampling_unit": "user",
        "aggregation": "interest macro accuracy recomputed per replicate",
        "conditions": {},
    }
    for condition, values in samples.items():
        intervals["conditions"][condition] = {
            "point_estimate": _macro_accuracy(
                np.asarray(correctness[condition], dtype=bool), answers
            ),
            "bootstrap_mean": float(values.mean()),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }

    paired = {
        "rule_version": RULE_VERSION,
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "resampling_unit": "user",
        "comparisons": [],
    }
    for spec in comparisons:
        left = str(spec["left"])
        right = str(spec["right"])
        delta = samples[left] - samples[right]
        paired["comparisons"].append(
            {
                **dict(spec),
                "point_delta": (
                    intervals["conditions"][left]["point_estimate"]
                    - intervals["conditions"][right]["point_estimate"]
                ),
                "bootstrap_mean": float(delta.mean()),
                "lower": float(np.quantile(delta, 0.025)),
                "upper": float(np.quantile(delta, 0.975)),
            }
        )
    return intervals, paired


def _publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")
    backup = staging.parent / "previous-output"
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def report_static_candidate_results(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    baseline_dir: Path | str = DEFAULT_BASELINE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Path:
    manifest_path = Path(manifest_path)
    dataset_dir = Path(dataset_dir)
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)
    manifest = _load_manifest(manifest_path)
    report_scope = dict(manifest.get("report_scope", {}))
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")

    train_path = dataset_dir / "train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    train_counts = Counter(
        str(row["target_interest"]["subcategory"])
        for batch in pq.ParquetFile(train_path).iter_batches(
            columns=["target_interest"], batch_size=16_384
        )
        for row in batch.to_pylist()
    )
    frequency_tier = _frequency_tiers(train_counts)
    full_root = Path(str(manifest["data"]["full_root"]))
    data_paths = {
        strategy: full_root / strategy / "behavior_only/dev_expanded.parquet"
        for strategy in ("uniform", "train_frequency_matched")
    }
    records_by_strategy = {
        strategy: pq.read_table(path).to_pylist()
        for strategy, path in data_paths.items()
    }
    uniform_records = records_by_strategy["uniform"]
    identity_fields = ("selection_id", "user_id", "answer_category", "correct_position")
    uniform_identity = [
        tuple(row[field] for field in identity_fields) for row in uniform_records
    ]
    other_identity = [
        tuple(row[field] for field in identity_fields)
        for row in records_by_strategy["train_frequency_matched"]
    ]
    if other_identity != uniform_identity:
        raise ValueError("Uniform and frequency-matched sample identities differ")

    all_results: dict[str, dict[str, np.ndarray]] = {}
    all_metadata: dict[str, dict[str, object]] = {}
    source_paths = [manifest_path, train_path, *data_paths.values()]
    for strategy, records in records_by_strategy.items():
        for baseline in BASELINES:
            path = baseline_dir / strategy / baseline / "predictions.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            key = f"{strategy}/{baseline}"
            all_results[key] = _load_baseline_result(path, records)
            all_metadata[key] = {
                "kind": "baseline",
                "condition_id": baseline,
                "evaluator": "statistical",
                "strategy": strategy,
                "task_type": "history_based" if baseline != "global_popularity" else "none",
            }
            source_paths.append(path)

    model_metadata: dict[str, dict[str, object]] = {}
    model_root = Path(str(manifest["artifact_root"])) / "full"
    for condition in manifest["conditions"]:
        condition_id = str(condition["id"])
        strategy = str(condition["strategy"])
        records = records_by_strategy[strategy]
        prediction_path = model_root / condition_id / "predictions.parquet"
        metrics_path = model_root / condition_id / "metrics.json"
        identity_path = model_root / condition_id / "experiment_identity.json"
        for path in (prediction_path, metrics_path, identity_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics = json.loads(metrics_path.read_text())
        identity = json.loads(identity_path.read_text())
        if int(metrics.get("samples", -1)) != int(manifest["data"]["full_samples"]):
            raise ValueError(f"{condition_id}: metrics sample count mismatch")
        if identity.get("condition_id") != condition_id or identity.get("phase") != "full":
            raise ValueError(f"{condition_id}: full experiment identity mismatch")
        key = f"{strategy}/{condition_id}"
        all_results[key] = _load_model_result(prediction_path, records)
        metadata = {
            "kind": "model",
            "condition_id": condition_id,
            "evaluator": str(condition["evaluator"]),
            "strategy": strategy,
            "task_type": str(condition["task_type"]),
        }
        all_metadata[key] = metadata
        model_metadata[condition_id] = metadata
        source_paths.extend((prediction_path, metrics_path, identity_path))

    summaries: list[dict[str, object]] = []
    per_interest_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    for key in sorted(all_results):
        strategy = str(all_metadata[key]["strategy"])
        summary, per_interest, slices = _metric_rows(
            key,
            all_metadata[key],
            records_by_strategy[strategy],
            all_results[key],
            frequency_tier,
        )
        summaries.append(summary)
        per_interest_rows.extend(per_interest)
        slice_rows.extend(slices)

    comparisons = _comparison_specs(model_metadata)
    correctness = {key: value["correct"] for key, value in all_results.items()}
    intervals, paired = bootstrap_interest_macro(
        uniform_records,
        correctness,
        comparisons,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-candidate-report-", dir=output_dir.parent
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        pq.write_table(
            pa.Table.from_pylist(summaries),
            staging / "condition_metrics.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(per_interest_rows),
            staging / "per_interest_metrics.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(slice_rows),
            staging / "slice_metrics.parquet",
            compression="zstd",
        )
        (staging / "bootstrap_intervals.json").write_text(
            json.dumps(intervals, ensure_ascii=False, indent=2) + "\n"
        )
        (staging / "paired_comparisons.json").write_text(
            json.dumps(paired, ensure_ascii=False, indent=2) + "\n"
        )
        summary = {
            "rule_version": RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_status": str(
                report_scope.get("status", "exploratory_dev")
            ),
            "primary_metric": "Interest Macro Accuracy over 111 Dev interests",
            "invalid_output_policy": "count_as_incorrect_without_denominator_removal",
            "samples": len(uniform_records),
            "users": len({str(row["user_id"]) for row in uniform_records}),
            "interests": len({str(row["answer_category"]) for row in uniform_records}),
            "model_conditions": len(model_metadata),
            "baseline_conditions": len(BASELINES) * 2,
            "bootstrap_replicates": bootstrap_replicates,
            "report_scope": report_scope,
            "artifacts": {
                "condition_metrics": "condition_metrics.parquet",
                "per_interest_metrics": "per_interest_metrics.parquet",
                "slice_metrics": "slice_metrics.parquet",
                "bootstrap_intervals": "bootstrap_intervals.json",
                "paired_comparisons": "paired_comparisons.json",
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        artifacts = sorted(path for path in staging.iterdir() if path.is_file())
        artifact_manifest = {
            "rule_version": RULE_VERSION,
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "artifact_sha256": {
                path.name: _sha256(path) for path in artifacts
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n"
        )
        _publish(staging, output_dir, overwrite=overwrite)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate the static user-disjoint candidate experiment."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = report_static_candidate_results(
        args.manifest,
        args.dataset_dir,
        args.baseline_dir,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
