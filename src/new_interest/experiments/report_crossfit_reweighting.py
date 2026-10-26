from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import sha256_file
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-sud-crossfit-reweighting-report-v2"
DEFAULT_DATA_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_crossfit_reweighting_qwen35_v1"
)
DEFAULT_STATIC_TRAIN = Path(
    "data/processed/new_interest_static_user_disjoint_v1/train.parquet"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "artifacts/static-sud-crossfit-reweighting/qwen35-v1"
)
CONDITIONS = {
    "behavior_vanilla": ("behavior_only", "behavior-vanilla"),
    "behavior_weighted": ("behavior_only", "behavior-weighted"),
    "interest_vanilla": ("interest_only", "interest-vanilla"),
    "interest_weighted": ("interest_only", "interest-weighted"),
}
EVAL_NAMES = (
    "real",
    "shuffled_20260818",
    "shuffled_20260819",
    "shuffled_20260820",
)
COMPARISONS = (
    ("behavior_weighted_minus_vanilla", "behavior_weighted", "behavior_vanilla"),
    ("interest_weighted_minus_vanilla", "interest_weighted", "interest_vanilla"),
    ("vanilla_interest_minus_behavior", "interest_vanilla", "behavior_vanilla"),
    ("weighted_interest_minus_behavior", "interest_weighted", "behavior_weighted"),
)


def _history_bucket(length: int) -> str:
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    if length <= 35:
        return "21-35"
    return "36-50"


def _frequency_tiers(counts: Mapping[str, int], labels: set[str]) -> dict[str, str]:
    ordered = sorted(labels, key=lambda label: (-counts.get(label, 0), label))
    head_end = max(1, math.ceil(len(ordered) * 0.2))
    torso_end = max(head_end + 1, math.ceil(len(ordered) * 0.6))
    return {
        label: (
            "head"
            if index < head_end
            else "torso"
            if index < torso_end
            else "tail"
        )
        for index, label in enumerate(ordered)
    }


def _load_predictions(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    by_index = {int(row["record_index"]): row for row in rows}
    if len(by_index) != len(rows) or set(by_index) != set(range(len(records))):
        raise ValueError(f"{path}: prediction indices do not cover eval data")
    ordered = [by_index[index] for index in range(len(records))]
    for record, row in zip(records, ordered):
        if str(record["impression_id"]) != str(row["impression_id"]):
            raise ValueError(f"{path}: impression identity mismatch")
        labels = [str(value) for value in row["ranked_labels"]]
        if len(labels) != 20 or len(set(labels)) != 20:
            raise ValueError(f"{path}: prediction is not a unique Top-20")
    return ordered


def _expanded_occurrences(
    records: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> list[dict[str, object]]:
    output = []
    for record, prediction in zip(records, predictions):
        ranked = [str(value) for value in prediction["ranked_labels"]]
        for target in record["acceptable_subcategories"]:
            label = str(target)
            rank = ranked.index(label) + 1 if label in ranked else None
            row: dict[str, object] = {
                "impression_id": str(record["impression_id"]),
                "user_id": str(record["user_id"]),
                "target_subcategory": label,
                "history_length": int(record["history_length"]),
            }
            for cutoff in cutoffs:
                retrieved = rank is not None and rank <= cutoff
                row[f"recall_at_{cutoff}"] = float(retrieved)
                row[f"ndcg_at_{cutoff}"] = (
                    1.0 / math.log2(rank + 1) if retrieved else 0.0
                )
            output.append(row)
    return output


def _macro(values: np.ndarray, labels: np.ndarray) -> float:
    unique, inverse = np.unique(labels, return_inverse=True)
    support = np.bincount(inverse, minlength=len(unique))
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    return float(np.mean(sums / support))


def bootstrap_interest_macro(
    identity_rows: Sequence[Mapping[str, object]],
    values: Mapping[str, Mapping[str, np.ndarray]],
    comparisons: Sequence[tuple[str, str, str]],
    *,
    replicates: int = 2_000,
    seed: int = 20260818,
) -> dict[str, object]:
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("Bootstrap requires scipy") from error
    if replicates < 2_000:
        raise ValueError("Bootstrap requires at least 2,000 replicates")
    users = sorted({str(row["user_id"]) for row in identity_rows})
    labels = sorted(
        {str(row["target_subcategory"]) for row in identity_rows}
    )
    user_index = {value: index for index, value in enumerate(users)}
    label_index = {value: index for index, value in enumerate(labels)}
    row_users = np.asarray(
        [user_index[str(row["user_id"])] for row in identity_rows],
        dtype=np.int32,
    )
    row_labels = np.asarray(
        [label_index[str(row["target_subcategory"])] for row in identity_rows],
        dtype=np.int16,
    )
    support_matrix = sparse.csr_matrix(
        (
            np.ones(len(identity_rows), dtype=np.int16),
            (row_users, row_labels),
        ),
        shape=(len(users), len(labels)),
    )
    matrices = {
        condition: {
            metric: sparse.csr_matrix(
                (np.asarray(array, dtype=np.float64), (row_users, row_labels)),
                shape=(len(users), len(labels)),
            )
            for metric, array in metrics.items()
        }
        for condition, metrics in values.items()
    }
    samples = {
        condition: {
            metric: np.empty(replicates, dtype=np.float64)
            for metric in metrics
        }
        for condition, metrics in values.items()
    }
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(users), 1.0 / len(users))
    offset = 0
    while offset < replicates:
        size = min(25, replicates - offset)
        user_weights = rng.multinomial(
            len(users), probabilities, size=size
        ).T
        support = np.asarray(
            support_matrix.T @ user_weights, dtype=np.float64
        )
        present = support > 0
        present_count = present.sum(axis=0)
        for condition, metric_matrices in matrices.items():
            for metric, matrix in metric_matrices.items():
                sums = np.asarray(matrix.T @ user_weights, dtype=np.float64)
                per_class = np.divide(
                    sums,
                    support,
                    out=np.zeros_like(sums),
                    where=present,
                )
                samples[condition][metric][offset : offset + size] = (
                    per_class.sum(axis=0) / present_count
                )
        offset += size

    target_labels = np.asarray(
        [str(row["target_subcategory"]) for row in identity_rows]
    )
    conditions = {}
    for condition, metrics in values.items():
        conditions[condition] = {}
        for metric, array in metrics.items():
            sampled = samples[condition][metric]
            conditions[condition][metric] = {
                "point_estimate": _macro(np.asarray(array), target_labels),
                "bootstrap_mean": float(sampled.mean()),
                "lower": float(np.quantile(sampled, 0.025)),
                "upper": float(np.quantile(sampled, 0.975)),
            }
    paired = []
    for name, left, right in comparisons:
        row: dict[str, object] = {
            "name": name,
            "left": left,
            "right": right,
            "metrics": {},
        }
        for metric in values[left]:
            delta = samples[left][metric] - samples[right][metric]
            row["metrics"][metric] = {
                "point_delta": (
                    conditions[left][metric]["point_estimate"]
                    - conditions[right][metric]["point_estimate"]
                ),
                "bootstrap_mean": float(delta.mean()),
                "lower": float(np.quantile(delta, 0.025)),
                "upper": float(np.quantile(delta, 0.975)),
            }
        paired.append(row)
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "resampling_unit": "user",
        "aggregation": "per-interest metric recomputed in every replicate",
        "conditions": conditions,
        "paired_comparisons": paired,
    }


def _weight_agreement(data_dir: Path) -> dict[str, object]:
    paths = {
        view: data_dir / "sample_weights" / f"{view}.parquet"
        for view in ("behavior", "interest")
    }
    if not all(path.is_file() for path in paths.values()):
        return {"available": False}
    rows = {
        view: {
            str(row["source_sample_id"]): row
            for row in pq.read_table(path).to_pylist()
        }
        for view, path in paths.items()
    }
    if rows["behavior"].keys() != rows["interest"].keys():
        raise ValueError("Behavior and Interest weight identities differ")
    ids = sorted(rows["behavior"])
    behavior_nll = np.asarray(
        [float(rows["behavior"][key]["category_token_nll"]) for key in ids]
    )
    interest_nll = np.asarray(
        [float(rows["interest"][key]["category_token_nll"]) for key in ids]
    )
    behavior_weight = np.asarray(
        [float(rows["behavior"][key]["sample_weight"]) for key in ids]
    )
    interest_weight = np.asarray(
        [float(rows["interest"][key]["sample_weight"]) for key in ids]
    )
    cutoff = max(1, len(ids) // 10)
    behavior_low = set(np.asarray(ids)[np.argsort(behavior_weight)[:cutoff]])
    interest_low = set(np.asarray(ids)[np.argsort(interest_weight)[:cutoff]])
    return {
        "available": True,
        "rows": len(ids),
        "nll_pearson": float(np.corrcoef(behavior_nll, interest_nll)[0, 1]),
        "weight_pearson": float(
            np.corrcoef(behavior_weight, interest_weight)[0, 1]
        ),
        "lowest_weight_decile_overlap": len(behavior_low & interest_low),
        "lowest_weight_decile_jaccard": (
            len(behavior_low & interest_low) / len(behavior_low | interest_low)
        ),
    }


def report_crossfit_reweighting(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    static_train_file: Path | str = DEFAULT_STATIC_TRAIN,
    *,
    weight_dir: Path | str | None = None,
    model_name: str | None = None,
    bootstrap_replicates: int = 2_000,
    seed: int = 20260818,
    real_only: bool = False,
    overwrite: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    artifact_root = Path(artifact_root)
    static_train_file = Path(static_train_file)
    weight_dir = Path(weight_dir) if weight_dir is not None else data_dir
    output_dir = artifact_root / "analysis"
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")

    static_counts = Counter(
        str(row["target_interest"]["subcategory"])
        for batch in pq.ParquetFile(static_train_file).iter_batches(
            columns=["target_interest"], batch_size=16_384
        )
        for row in batch.to_pylist()
    )
    records_by_view = {
        task: pq.read_table(
            data_dir / "eval" / task / "real.parquet"
        ).to_pylist()
        for task in ("behavior_only", "interest_only")
    }
    identity_columns = (
        "impression_id",
        "user_id",
        "acceptable_subcategories",
    )
    left_identity = [
        tuple(row[column] for column in identity_columns)
        for row in records_by_view["behavior_only"]
    ]
    right_identity = [
        tuple(row[column] for column in identity_columns)
        for row in records_by_view["interest_only"]
    ]
    if left_identity != right_identity:
        raise ValueError("Behavior and Interest real Dev identities differ")

    source_paths = [static_train_file]
    condition_rows = []
    per_interest_rows = []
    slice_rows = []
    real_expanded: dict[str, list[dict[str, object]]] = {}
    bootstrap_values: dict[str, dict[str, np.ndarray]] = {}
    eval_names = ("real",) if real_only else EVAL_NAMES
    real_minus_shuffled = []
    for condition, (task_type, directory) in CONDITIONS.items():
        records = records_by_view[task_type]
        metrics_by_eval = {}
        for eval_name in eval_names:
            root = artifact_root / "final" / directory / "eval" / eval_name / "merged"
            metrics_path = root / "metrics.json"
            prediction_path = root / "beam_predictions.parquet"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            predictions = _load_predictions(prediction_path, records)
            metrics_by_eval[eval_name] = metrics
            source_paths.extend((metrics_path, prediction_path))
            condition_rows.append(
                {
                    "condition": condition,
                    "eval_name": eval_name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float, str, bool))
                    },
                }
            )
            if eval_name == "real":
                expanded = _expanded_occurrences(records, predictions)
                real_expanded[condition] = expanded
                bootstrap_values[condition] = {
                    "interest_macro_recall_at_10": np.asarray(
                        [float(row["recall_at_10"]) for row in expanded]
                    ),
                    "interest_macro_ndcg_at_10": np.asarray(
                        [float(row["ndcg_at_10"]) for row in expanded]
                    ),
                }

        if not real_only:
            real_metrics = metrics_by_eval["real"]
            shuffled_metrics = [
                metrics_by_eval[name] for name in EVAL_NAMES if name != "real"
            ]
            real_minus_shuffled.append(
                {
                    "condition": condition,
                    "recall_at_10_delta": float(
                        real_metrics["interest_macro_recall_at_10"]
                    )
                    - float(
                        np.mean(
                            [
                                row["interest_macro_recall_at_10"]
                                for row in shuffled_metrics
                            ]
                        )
                    ),
                    "ndcg_at_10_delta": float(
                        real_metrics["interest_macro_ndcg_at_10"]
                    )
                    - float(
                        np.mean(
                            [
                                row["interest_macro_ndcg_at_10"]
                                for row in shuffled_metrics
                            ]
                        )
                    ),
                }
            )

    reference = real_expanded["behavior_vanilla"]
    reference_identity = [
        (row["impression_id"], row["user_id"], row["target_subcategory"])
        for row in reference
    ]
    for condition, rows in real_expanded.items():
        identity = [
            (row["impression_id"], row["user_id"], row["target_subcategory"])
            for row in rows
        ]
        if identity != reference_identity:
            raise ValueError(f"{condition}: target-occurrence identity differs")

    labels = {str(row["target_subcategory"]) for row in reference}
    tiers = _frequency_tiers(static_counts, labels)
    for condition, rows in real_expanded.items():
        for label in sorted(labels):
            selected = [row for row in rows if row["target_subcategory"] == label]
            per_interest_rows.append(
                {
                    "condition": condition,
                    "interest": label,
                    "frequency_tier": tiers[label],
                    "support": len(selected),
                    **{
                        metric: float(np.mean([row[metric] for row in selected]))
                        for metric in (
                            "recall_at_1",
                            "ndcg_at_1",
                            "recall_at_5",
                            "ndcg_at_5",
                            "recall_at_10",
                            "ndcg_at_10",
                            "recall_at_20",
                            "ndcg_at_20",
                        )
                    },
                }
            )
        for slice_name, value_fn in (
            ("history_length", lambda row: _history_bucket(int(row["history_length"]))),
            ("interest_frequency", lambda row: tiers[str(row["target_subcategory"])]),
        ):
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in rows:
                grouped[str(value_fn(row))].append(row)
            for value, selected in sorted(grouped.items()):
                selected_labels = np.asarray(
                    [str(row["target_subcategory"]) for row in selected]
                )
                slice_rows.append(
                    {
                        "condition": condition,
                        "slice": slice_name,
                        "value": value,
                        "target_occurrences": len(selected),
                        "supported_interests": len(set(selected_labels)),
                        "interest_macro_recall_at_10": _macro(
                            np.asarray([row["recall_at_10"] for row in selected]),
                            selected_labels,
                        ),
                        "interest_macro_ndcg_at_10": _macro(
                            np.asarray([row["ndcg_at_10"] for row in selected]),
                            selected_labels,
                        ),
                    }
                )

    bootstrap = bootstrap_interest_macro(
        reference,
        bootstrap_values,
        COMPARISONS,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    agreement = _weight_agreement(weight_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".crossfit-report-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        pq.write_table(
            pa.Table.from_pylist(condition_rows),
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
        json_outputs = [
            ("bootstrap_intervals.json", bootstrap),
            ("oof_weight_agreement.json", agreement),
        ]
        if not real_only:
            json_outputs.append(
                ("real_minus_shuffled.json", real_minus_shuffled)
            )
        for name, value in json_outputs:
            (staging / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "evaluation_status": "exploratory_dev_not_fresh_final_test",
            "impressions": len(records_by_view["behavior_only"]),
            "users": len({str(row["user_id"]) for row in reference}),
            "target_occurrences": len(reference),
            "supported_interests": len(labels),
            "primary_metrics": [
                "Interest-Macro Recall@10",
                "Interest-Macro NDCG@10",
            ],
            "evaluation_names": list(eval_names),
            "shuffled_controls_included": not real_only,
            "bootstrap_replicates": bootstrap_replicates,
            "model_name": model_name,
            "weight_dir": portable_path(weight_dir),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated = sorted(path for path in staging.iterdir() if path.is_file())
        manifest = {
            "source_sha256": {
                portable_path(path): sha256_file(path) for path in source_paths
            },
            "artifact_sha256": {
                path.name: sha256_file(path) for path in generated
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            backup = Path(temporary) / "previous"
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--static-train-file", type=Path, default=DEFAULT_STATIC_TRAIN)
    parser.add_argument("--weight-dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Read and report only eval/real; do not require shuffled controls.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = report_crossfit_reweighting(
        args.data_dir,
        args.artifact_root,
        args.static_train_file,
        weight_dir=args.weight_dir,
        model_name=args.model_name,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        real_only=args.real_only,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
