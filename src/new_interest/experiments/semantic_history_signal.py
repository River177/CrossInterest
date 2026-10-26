from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.ranking_metrics import paired_bootstrap_metric_delta
from new_interest.experiments.embedding_future_interest import (
    SentenceTransformerEncoder,
    TextEncoder,
    write_history_embeddings,
)
from new_interest.experiments.future_interest_prediction import load_vocabulary
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-semantic-history-signal-v1"
DEFAULT_CONFIG = Path(
    "configs/future_interest/static_user_disjoint_semantic_v1.json"
)


@dataclass
class SplitData:
    user_ids: tuple[str, ...]
    embeddings: np.ndarray
    positives: Any
    history_mask: np.ndarray | None
    history_length: np.ndarray
    history_topic_count: np.ndarray

    @property
    def target_count(self) -> np.ndarray:
        return np.diff(self.positives.indptr)


@dataclass
class EvaluationResult:
    metrics: dict[str, object]
    rankings: np.ndarray
    ranked_scores: np.ndarray
    per_user: dict[str, np.ndarray]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(
    config_path_or_config: Path | str | Mapping[str, object],
) -> tuple[dict[str, object], Path | None]:
    if isinstance(config_path_or_config, Mapping):
        return dict(config_path_or_config), None
    path = Path(config_path_or_config)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    return config, path


def _normalise_rows(values: Sequence[Sequence[float]], *, context: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{context}: expected a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{context}: embeddings contain non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError(f"{context}: embeddings contain zero-norm rows")
    return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)


def _topic_texts(vocabulary_path: Path) -> tuple[tuple[str, ...], list[str]]:
    labels = load_vocabulary(vocabulary_path)
    table = pq.read_table(vocabulary_path, columns=["category", "subcategory"])
    by_label: dict[str, str] = {}
    for row in table.to_pylist():
        label = str(row["subcategory"])
        if label in by_label:
            raise ValueError(f"{vocabulary_path}: duplicate label {label!r}")
        by_label[label] = (
            f"category: {str(row['category'])}; topic: {label}"
        )
    return labels, [by_label[label] for label in labels]


def _list_array_matrix(array: pa.Array, *, context: str) -> np.ndarray:
    if array.null_count:
        raise ValueError(f"{context}: null embedding")
    lengths = np.asarray(array.value_lengths().to_numpy(), dtype=np.int64)
    if not lengths.size or np.any(lengths != lengths[0]) or lengths[0] <= 0:
        raise ValueError(f"{context}: inconsistent embedding dimensions")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return np.ascontiguousarray(values.reshape(len(array), int(lengths[0])))


def _load_embedding_matrix(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    parquet = pq.ParquetFile(path)
    required = {"user_id", "embedding"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    total = parquet.metadata.num_rows
    user_ids: list[str] = []
    matrix: np.ndarray | None = None
    offset = 0
    for batch in parquet.iter_batches(
        columns=["user_id", "embedding"], batch_size=8_192
    ):
        user_column = batch.schema.get_field_index("user_id")
        embedding_column = batch.schema.get_field_index("embedding")
        values = _list_array_matrix(
            batch.column(embedding_column), context=str(path)
        )
        if matrix is None:
            matrix = np.empty((total, values.shape[1]), dtype=np.float32)
        elif matrix.shape[1] != values.shape[1]:
            raise ValueError(f"{path}: embedding dimension changed")
        matrix[offset : offset + len(values)] = values
        user_ids.extend(
            str(value) for value in batch.column(user_column).to_pylist()
        )
        offset += len(values)
    if matrix is None or offset != total:
        raise ValueError(f"{path}: empty or incomplete embedding artifact")
    if len(user_ids) != len(set(user_ids)):
        raise ValueError(f"{path}: duplicate user IDs")
    return tuple(user_ids), matrix


def _load_split(
    benchmark_path: Path,
    embedding_path: Path,
    labels: tuple[str, ...],
    *,
    include_history_mask: bool,
) -> SplitData:
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("semantic experiment requires scipy") from error
    embedding_users, embeddings = _load_embedding_matrix(embedding_path)
    parquet = pq.ParquetFile(benchmark_path)
    required = {
        "user_id",
        "history_length",
        "history_unique_subcategory_count",
        "history_unique_subcategories",
        "future_novel_subcategories",
    }
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{benchmark_path}: missing columns {sorted(missing)}")
    row_count = parquet.metadata.num_rows
    if row_count != len(embedding_users):
        raise ValueError("benchmark and embedding row counts differ")
    label_to_index = {label: index for index, label in enumerate(labels)}
    history_mask = (
        np.zeros((row_count, len(labels)), dtype=np.bool_)
        if include_history_mask
        else None
    )
    history_length = np.empty(row_count, dtype=np.int32)
    history_topic_count = np.empty(row_count, dtype=np.int16)
    row_indices: list[int] = []
    label_indices: list[int] = []
    benchmark_users: list[str] = []
    offset = 0
    columns = sorted(required)
    for batch in parquet.iter_batches(columns=columns, batch_size=8_192):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            positives = {str(value) for value in row["future_novel_subcategories"]}
            history = {str(value) for value in row["history_unique_subcategories"]}
            if not positives:
                raise ValueError(f"{benchmark_path}: {user_id!r} has no targets")
            invalid = (positives | history) - set(label_to_index)
            if invalid:
                raise ValueError(
                    f"{benchmark_path}: {user_id!r} has invalid labels {sorted(invalid)}"
                )
            if positives & history:
                raise ValueError(
                    f"{benchmark_path}: {user_id!r} targets occur in History"
                )
            benchmark_users.append(user_id)
            history_length[offset] = int(row["history_length"])
            history_topic_count[offset] = int(
                row["history_unique_subcategory_count"]
            )
            for label in sorted(positives):
                row_indices.append(offset)
                label_indices.append(label_to_index[label])
            if history_mask is not None:
                history_mask[
                    offset,
                    [label_to_index[label] for label in history],
                ] = True
            offset += 1
    if tuple(benchmark_users) != embedding_users:
        raise ValueError(
            f"{embedding_path}: user order does not match {benchmark_path}"
        )
    positives = sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.bool_),
            (np.asarray(row_indices), np.asarray(label_indices)),
        ),
        shape=(row_count, len(labels)),
        dtype=np.bool_,
    )
    return SplitData(
        user_ids=embedding_users,
        embeddings=embeddings,
        positives=positives,
        history_mask=history_mask,
        history_length=history_length,
        history_topic_count=history_topic_count,
    )


def _history_length_bucket(length: int) -> str:
    if length <= 10:
        return "1-10"
    if length <= 25:
        return "11-25"
    if length <= 50:
        return "26-50"
    if length <= 100:
        return "51-100"
    return "101+"


def _topic_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _target_count_bucket(count: int) -> str:
    if count <= 2:
        return str(count)
    if count <= 5:
        return "3-5"
    return "6+"


def _label_slices(support: np.ndarray, labels: tuple[str, ...]) -> dict[str, list[int]]:
    order = sorted(
        range(len(labels)),
        key=lambda index: (-float(support[index]), labels[index]),
    )
    head_end = max(1, math.ceil(len(labels) * 0.2))
    torso_end = max(head_end + 1, math.ceil(len(labels) * 0.6))
    return {
        "head": order[:head_end],
        "torso": order[head_end:torso_end],
        "tail": order[torso_end:],
    }


def _mean_by_mask(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if np.any(mask) else None


def _evaluate_scores(
    scores: np.ndarray,
    data: SplitData,
    labels: tuple[str, ...],
    *,
    cutoffs: tuple[int, ...],
    label_slices: Mapping[str, Sequence[int]],
) -> EvaluationResult:
    if data.history_mask is None:
        raise ValueError("evaluation requires the actual History mask")
    if scores.ndim == 1:
        scores = np.broadcast_to(scores, (len(data.user_ids), len(labels)))
    if scores.shape != (len(data.user_ids), len(labels)):
        raise ValueError("score matrix shape does not match evaluation data")
    masked_scores = np.array(scores, dtype=np.float32, copy=True)
    masked_scores[data.history_mask] = -np.inf
    if np.any(np.isneginf(masked_scores).all(axis=1)):
        raise ValueError("History mask removed the entire candidate space")
    max_cutoff = max(cutoffs)
    full_order = np.argsort(-masked_scores, axis=1, kind="stable")
    rankings = full_order[:, :max_cutoff]
    ranked_scores = np.take_along_axis(masked_scores, rankings, axis=1)
    positive_dense = data.positives.toarray().astype(np.bool_, copy=False)
    hits = np.take_along_axis(positive_dense, rankings, axis=1)
    target_count = data.target_count.astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(max_cutoff, dtype=np.float64) + 2.0)
    per_user: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {"users": len(data.user_ids)}
    per_label_support = np.asarray(data.positives.sum(axis=0)).ravel().astype(np.int64)
    label_metrics: dict[str, object] = {}
    for cutoff in cutoffs:
        effective_cutoff = min(cutoff, hits.shape[1])
        prefix_hits = hits[:, :effective_cutoff]
        hit_count = prefix_hits.sum(axis=1)
        recall = hit_count / target_count
        dcg = (prefix_hits * discounts[:effective_cutoff]).sum(axis=1)
        ideal_count = np.minimum(data.target_count, effective_cutoff)
        idcg = np.asarray(
            [discounts[: int(count)].sum() for count in ideal_count],
            dtype=np.float64,
        )
        ndcg = dcg / idcg
        per_user[f"recall_at_{cutoff}"] = recall
        per_user[f"ndcg_at_{cutoff}"] = ndcg
        metrics[f"recall_at_{cutoff}"] = float(recall.mean())
        metrics[f"ndcg_at_{cutoff}"] = float(ndcg.mean())
        retrieved_labels = rankings[:, :effective_cutoff][prefix_hits]
        retrieved = np.bincount(retrieved_labels, minlength=len(labels))
        supported = per_label_support > 0
        per_label_recall = np.zeros(len(labels), dtype=np.float64)
        per_label_recall[supported] = (
            retrieved[supported] / per_label_support[supported]
        )
        metrics[f"macro_recall_at_{cutoff}"] = float(
            per_label_recall[supported].mean()
        )
        for slice_name, indices_value in label_slices.items():
            indices = np.asarray(indices_value, dtype=np.int64)
            slice_supported = indices[per_label_support[indices] > 0]
            macro = (
                float(per_label_recall[slice_supported].mean())
                if slice_supported.size
                else None
            )
            total_support = int(per_label_support[indices].sum())
            micro = (
                float(retrieved[indices].sum() / total_support)
                if total_support
                else None
            )
            label_metrics.setdefault(slice_name, {})[str(cutoff)] = {
                "labels": int(len(indices)),
                "positive_support": total_support,
                "macro_recall": macro,
                "micro_recall": micro,
            }
    metrics["label_slices"] = label_metrics
    attributes: dict[str, dict[str, object]] = {}
    bucket_values = {
        "target_count": np.asarray(
            [_target_count_bucket(int(value)) for value in data.target_count]
        ),
        "history_length": np.asarray(
            [_history_length_bucket(int(value)) for value in data.history_length]
        ),
    }
    for attribute, values in bucket_values.items():
        attributes[attribute] = {}
        for bucket in sorted(set(values.tolist())):
            mask = values == bucket
            attributes[attribute][bucket] = {
                "users": int(mask.sum()),
                **{
                    metric: _mean_by_mask(array, mask)
                    for metric, array in per_user.items()
                },
            }
    metrics["by_attribute"] = attributes
    metrics["target_count_distribution"] = {
        str(value): int((data.target_count == value).sum())
        for value in sorted(set(data.target_count.tolist()))
    }
    return EvaluationResult(
        metrics=metrics,
        rankings=rankings,
        ranked_scores=ranked_scores,
        per_user=per_user,
    )


def _fit_linear_residual(
    fit: SplitData,
    *,
    label_count: int,
    prior_alpha: float,
    ridge_alphas: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], np.ndarray]:
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("semantic experiment requires scipy") from error
    if prior_alpha <= 0.0 or any(alpha <= 0.0 for alpha in ridge_alphas):
        raise ValueError("prior and ridge alphas must be positive")
    target_count = fit.target_count.astype(np.float32)
    q = fit.positives.astype(np.float32)
    q = sparse.diags(1.0 / target_count) @ q
    weighted_counts = np.asarray(q.sum(axis=0)).ravel().astype(np.float64)
    probabilities = (weighted_counts + prior_alpha) / (
        len(fit.user_ids) + prior_alpha * label_count
    )
    global_scores = np.log(probabilities).astype(np.float32)
    x = fit.embeddings
    sum_x = x.sum(axis=0, dtype=np.float64)
    mean_x = (sum_x / len(x)).astype(np.float32)
    xtx = np.asarray(x.T @ x, dtype=np.float64)
    covariance = xtx - np.outer(sum_x, sum_x) / len(x)
    covariance = (covariance + covariance.T) * 0.5
    q_mean = weighted_counts / len(x)
    qtx = np.asarray(q.T @ x, dtype=np.float64)
    cross = qtx.T - np.outer(sum_x, q_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross
    weights = {
        alpha: np.asarray(
            eigenvectors @ (projected / (eigenvalues[:, None] + alpha)),
            dtype=np.float32,
        )
        for alpha in ridge_alphas
    }
    return global_scores, mean_x, weights, weighted_counts


def _tune_lambda(
    global_scores: np.ndarray,
    residual: np.ndarray,
    validation: SplitData,
    labels: tuple[str, ...],
    *,
    lambdas: tuple[float, ...],
    cutoffs: tuple[int, ...],
    label_slices: Mapping[str, Sequence[int]],
) -> tuple[float, dict[str, object]]:
    records: list[dict[str, object]] = []
    for value in lambdas:
        evaluation = _evaluate_scores(
            global_scores + value * residual,
            validation,
            labels,
            cutoffs=cutoffs,
            label_slices=label_slices,
        )
        records.append({"lambda": value, "metrics": evaluation.metrics})
    selected = max(
        records,
        key=lambda row: (
            float(row["metrics"]["ndcg_at_10"]),
            float(row["metrics"]["recall_at_10"]),
            -float(row["lambda"]),
        ),
    )
    return float(selected["lambda"]), {
        "selected_lambda": float(selected["lambda"]),
        "selection_metric": "validation ndcg_at_10, recall_at_10 tie-break",
        "grid": records,
    }


def _shuffle_donors(data: SplitData, *, seed: int) -> tuple[np.ndarray, list[str]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (length, topics) in enumerate(
        zip(data.history_length, data.history_topic_count, strict=True)
    ):
        groups[
            (
                _history_length_bucket(int(length)),
                _topic_count_bucket(int(topics)),
            )
        ].append(index)
    donors = np.arange(len(data.user_ids), dtype=np.int64)
    bucket_names = [""] * len(data.user_ids)
    for key, indices_value in sorted(groups.items()):
        indices = np.asarray(
            sorted(indices_value, key=lambda value: data.user_ids[value]),
            dtype=np.int64,
        )
        bucket = f"history={key[0]}|topics={key[1]}"
        for index in indices:
            bucket_names[int(index)] = bucket
        if len(indices) <= 1:
            continue
        digest = hashlib.sha256(f"{seed}:{bucket}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        order = rng.permutation(indices)
        shift = int(rng.integers(1, len(order)))
        donors[order] = np.roll(order, shift)
    return donors, bucket_names


def _write_condition(
    root: Path,
    condition_id: str,
    result: EvaluationResult,
    data: SplitData,
    labels: tuple[str, ...],
) -> dict[str, str]:
    output = root / "conditions" / condition_id
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    detail_rows = []
    for index, user_id in enumerate(data.user_ids):
        prediction_rows.append(
            {
                "user_id": user_id,
                "ranked_labels": [
                    labels[int(value)] for value in result.rankings[index]
                ],
                "ranked_scores": result.ranked_scores[index].astype(float).tolist(),
            }
        )
        detail_rows.append(
            {
                "example_id": user_id,
                "target_count": int(data.target_count[index]),
                "history_length": int(data.history_length[index]),
                **{
                    name: float(values[index])
                    for name, values in result.per_user.items()
                },
            }
        )
    pq.write_table(
        pa.Table.from_pylist(prediction_rows),
        output / "top20_predictions.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )
    pq.write_table(
        pa.Table.from_pylist(detail_rows),
        output / "details.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": f"conditions/{condition_id}/metrics.json",
        "details": f"conditions/{condition_id}/details.parquet",
        "predictions": f"conditions/{condition_id}/top20_predictions.parquet",
    }


def _bootstrap(
    baseline: EvaluationResult,
    comparison: EvaluationResult,
    user_ids: tuple[str, ...],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    metric_names = tuple(
        name
        for cutoff in (1, 5, 10, 20)
        for name in (f"recall_at_{cutoff}", f"ndcg_at_{cutoff}")
    )
    baseline_details = [
        {
            "example_id": user_id,
            **{
                metric: float(baseline.per_user[metric][index])
                for metric in metric_names
            },
        }
        for index, user_id in enumerate(user_ids)
    ]
    comparison_details = [
        {
            "example_id": user_id,
            **{
                metric: float(comparison.per_user[metric][index])
                for metric in metric_names
            },
        }
        for index, user_id in enumerate(user_ids)
    ]
    return paired_bootstrap_metric_delta(
        baseline_details,
        comparison_details,
        metric_names=metric_names,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )


def _delta(
    comparison: EvaluationResult,
    baseline: EvaluationResult,
    metric: str,
) -> float:
    return float(comparison.metrics[metric]) - float(baseline.metrics[metric])


def _gate(
    actual: EvaluationResult,
    global_result: EvaluationResult,
    shuffled: Mapping[int, EvaluationResult],
    comparisons: Mapping[str, object],
    *,
    minimum_delta: float,
) -> dict[str, object]:
    global_bootstrap = comparisons["vs_global"]
    global_metrics = global_bootstrap["metrics"]
    actual_beats_global = all(
        _delta(actual, global_result, metric) >= minimum_delta
        and float(global_metrics[metric]["lower"]) > 0.0
        for metric in ("recall_at_10", "ndcg_at_10")
    )
    shuffle_checks = {}
    for seed, result in shuffled.items():
        bootstrap = comparisons[f"vs_shuffled_{seed}"]
        shuffle_checks[str(seed)] = all(
            _delta(actual, result, metric) > 0.0
            and float(bootstrap["metrics"][metric]["lower"]) > 0.0
            for metric in ("recall_at_10", "ndcg_at_10")
        )
    macro_ok = _delta(actual, global_result, "macro_recall_at_10") >= 0.0
    actual_tail = actual.metrics["label_slices"]["tail"]["10"]["micro_recall"]
    global_tail = global_result.metrics["label_slices"]["tail"]["10"]["micro_recall"]
    tail_ok = (
        actual_tail is not None
        and global_tail is not None
        and float(actual_tail) >= float(global_tail)
    )
    passed = actual_beats_global and all(shuffle_checks.values()) and macro_ok and tail_ok
    return {
        "passed": passed,
        "minimum_absolute_delta": minimum_delta,
        "actual_beats_global_by_required_margin_with_ci": actual_beats_global,
        "actual_beats_each_shuffle_with_ci": shuffle_checks,
        "macro_recall_does_not_collapse": macro_ok,
        "tail_micro_recall_does_not_collapse": tail_ok,
    }


def run_semantic_history_signal_experiment(
    config_path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    encoder: TextEncoder | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    config, config_path = _load_config(config_path_or_config)
    if str(config.get("rule_version")) != RULE_VERSION:
        raise ValueError(
            "config rule_version does not match the experiment implementation"
        )
    benchmark_dir = Path(str(config["benchmark_dir"]))
    artifact_dir = Path(str(config["artifact_dir"]))
    if artifact_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Artifact directory already exists: {artifact_dir}. Use --overwrite."
        )
    backup = artifact_dir.with_name(artifact_dir.name + ".old")
    if artifact_dir.exists() and overwrite and backup.exists():
        raise FileExistsError(f"Cannot overwrite while backup exists: {backup}")
    split_paths = {
        split: benchmark_dir / f"{split}.parquet"
        for split in ("fit", "validation", "dev")
    }
    vocabulary_path = benchmark_dir / "interest_vocabulary.parquet"
    for path in (*split_paths.values(), vocabulary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cutoffs = tuple(int(value) for value in config["cutoffs"])
    if 10 not in cutoffs or max(cutoffs) != 20:
        raise ValueError("cutoffs must include 10 and have maximum 20")
    embedding_config = dict(config["embedding"])
    model_name = str(embedding_config["model_name"])
    revision = str(embedding_config["revision"])
    device_value = embedding_config.get("device")
    device = str(device_value) if device_value is not None else None
    batch_size = int(embedding_config["batch_size"])
    model_config = dict(config["model"])
    prior_alpha = float(model_config["prior_alpha"])
    ridge_alphas = tuple(float(value) for value in model_config["ridge_alphas"])
    lambdas = tuple(float(value) for value in model_config["lambda_grid"])
    if 0.0 not in lambdas:
        raise ValueError("lambda_grid must include 0.0 as the no-History control")
    shuffle_seeds = tuple(int(value) for value in config["shuffle"]["seeds"])
    if len(shuffle_seeds) < 3:
        raise ValueError("at least three matched-shuffle seeds are required")
    bootstrap_config = dict(config["bootstrap"])
    bootstrap_replicates = int(bootstrap_config["replicates"])
    bootstrap_seed = int(bootstrap_config["seed"])
    confidence = float(bootstrap_config["confidence"])
    minimum_delta = float(config["acceptance"]["minimum_absolute_delta"])

    labels, topic_texts = _topic_texts(vocabulary_path)
    shared_encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".semantic-history-signal-",
        dir=str(artifact_dir.parent),
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        embedding_paths: dict[str, Path] = {}
        embedding_summaries: dict[str, object] = {}
        for split, benchmark_path in split_paths.items():
            output = staging / "embeddings" / f"{split}.parquet"
            embedding_summaries[split] = write_history_embeddings(
                benchmark_path,
                output,
                encoder=shared_encoder,
                model_name=model_name,
                revision=revision,
                device=device,
                batch_size=batch_size,
            )
            embedding_paths[split] = output
        topic_vectors = _normalise_rows(
            shared_encoder.encode(topic_texts, batch_size=batch_size),
            context="taxonomy topics",
        )

        fit = _load_split(
            split_paths["fit"],
            embedding_paths["fit"],
            labels,
            include_history_mask=False,
        )
        validation = _load_split(
            split_paths["validation"],
            embedding_paths["validation"],
            labels,
            include_history_mask=True,
        )
        dev = _load_split(
            split_paths["dev"],
            embedding_paths["dev"],
            labels,
            include_history_mask=True,
        )
        global_scores, mean_x, ridge_weights, weighted_support = (
            _fit_linear_residual(
                fit,
                label_count=len(labels),
                prior_alpha=prior_alpha,
                ridge_alphas=ridge_alphas,
            )
        )
        slices = _label_slices(
            np.asarray(fit.positives.sum(axis=0)).ravel(), labels
        )

        validation_zero = validation.embeddings @ topic_vectors.T
        zero_lambda, zero_tuning = _tune_lambda(
            global_scores,
            validation_zero,
            validation,
            labels,
            lambdas=lambdas,
            cutoffs=cutoffs,
            label_slices=slices,
        )
        ridge_tuning_records = []
        selected_ridge: tuple[float, float, np.ndarray, dict[str, object]] | None = None
        centered_validation = validation.embeddings - mean_x
        for alpha in ridge_alphas:
            residual = centered_validation @ ridge_weights[alpha]
            value, tuning = _tune_lambda(
                global_scores,
                residual,
                validation,
                labels,
                lambdas=lambdas,
                cutoffs=cutoffs,
                label_slices=slices,
            )
            selected_metrics = next(
                row["metrics"]
                for row in tuning["grid"]
                if float(row["lambda"]) == value
            )
            record = {
                "ridge_alpha": alpha,
                **tuning,
                "selected_metrics": selected_metrics,
            }
            ridge_tuning_records.append(record)
            candidate = (alpha, value, residual, record)
            if selected_ridge is None or (
                float(record["selected_metrics"]["ndcg_at_10"]),
                float(record["selected_metrics"]["recall_at_10"]),
                -value,
                -alpha,
            ) > (
                float(selected_ridge[3]["selected_metrics"]["ndcg_at_10"]),
                float(selected_ridge[3]["selected_metrics"]["recall_at_10"]),
                -selected_ridge[1],
                -selected_ridge[0],
            ):
                selected_ridge = candidate
        assert selected_ridge is not None
        ridge_alpha, ridge_lambda, _, ridge_record = selected_ridge

        global_result = _evaluate_scores(
            global_scores,
            dev,
            labels,
            cutoffs=cutoffs,
            label_slices=slices,
        )
        zero_actual = _evaluate_scores(
            global_scores + zero_lambda * (dev.embeddings @ topic_vectors.T),
            dev,
            labels,
            cutoffs=cutoffs,
            label_slices=slices,
        )
        selected_weights = ridge_weights[ridge_alpha]
        ridge_actual = _evaluate_scores(
            global_scores
            + ridge_lambda * ((dev.embeddings - mean_x) @ selected_weights),
            dev,
            labels,
            cutoffs=cutoffs,
            label_slices=slices,
        )
        actual_results = {
            "zero_shot": zero_actual,
            "linear_ridge": ridge_actual,
        }
        condition_results: dict[str, EvaluationResult] = {
            "global": global_result,
            "zero_shot_actual": zero_actual,
            "linear_ridge_actual": ridge_actual,
        }
        shuffled_results: dict[str, dict[int, EvaluationResult]] = {
            "zero_shot": {},
            "linear_ridge": {},
        }
        shuffle_rows: list[dict[str, object]] = []
        for seed in shuffle_seeds:
            donors, buckets = _shuffle_donors(dev, seed=seed)
            shuffled_embeddings = dev.embeddings[donors]
            zero_result = _evaluate_scores(
                global_scores
                + zero_lambda * (shuffled_embeddings @ topic_vectors.T),
                dev,
                labels,
                cutoffs=cutoffs,
                label_slices=slices,
            )
            ridge_result = _evaluate_scores(
                global_scores
                + ridge_lambda * (
                    (shuffled_embeddings - mean_x) @ selected_weights
                ),
                dev,
                labels,
                cutoffs=cutoffs,
                label_slices=slices,
            )
            shuffled_results["zero_shot"][seed] = zero_result
            shuffled_results["linear_ridge"][seed] = ridge_result
            condition_results[f"zero_shot_shuffled_{seed}"] = zero_result
            condition_results[f"linear_ridge_shuffled_{seed}"] = ridge_result
            for index, donor in enumerate(donors):
                shuffle_rows.append(
                    {
                        "seed": seed,
                        "user_id": dev.user_ids[index],
                        "donor_user_id": dev.user_ids[int(donor)],
                        "bucket": buckets[index],
                        "is_fixed_point": bool(index == int(donor)),
                    }
                )

        comparison_outputs: dict[str, dict[str, object]] = {}
        gates: dict[str, object] = {}
        for model_id, actual in actual_results.items():
            model_comparisons: dict[str, object] = {
                "vs_global": _bootstrap(
                    global_result,
                    actual,
                    dev.user_ids,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
            }
            for seed, shuffled in shuffled_results[model_id].items():
                model_comparisons[f"vs_shuffled_{seed}"] = _bootstrap(
                    shuffled,
                    actual,
                    dev.user_ids,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + seed,
                    confidence=confidence,
                )
            comparison_outputs[model_id] = model_comparisons
            gates[model_id] = _gate(
                actual,
                global_result,
                shuffled_results[model_id],
                model_comparisons,
                minimum_delta=minimum_delta,
            )

        zero_validation_metrics = next(
            row["metrics"]
            for row in zero_tuning["grid"]
            if float(row["lambda"]) == zero_lambda
        )
        primary_model = max(
            (
                ("zero_shot", zero_validation_metrics),
                ("linear_ridge", ridge_record["selected_metrics"]),
            ),
            key=lambda item: (
                float(item[1]["ndcg_at_10"]),
                float(item[1]["recall_at_10"]),
                item[0],
            ),
        )[0]

        condition_paths = {
            condition_id: _write_condition(
                staging, condition_id, result, dev, labels
            )
            for condition_id, result in condition_results.items()
        }
        pq.write_table(
            pa.Table.from_pylist(shuffle_rows),
            staging / "shuffle_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
            row_group_size=8_192,
        )
        np.savez_compressed(
            staging / "linear_model.npz",
            labels=np.asarray(labels),
            global_scores=global_scores,
            fit_embedding_mean=mean_x,
            ridge_weights=selected_weights,
            ridge_alpha=np.asarray([ridge_alpha]),
            ridge_lambda=np.asarray([ridge_lambda]),
            zero_shot_lambda=np.asarray([zero_lambda]),
            topic_vectors=topic_vectors,
            weighted_positive_support=weighted_support,
        )
        tuning = {
            "zero_shot": zero_tuning,
            "linear_ridge": {
                "selected_ridge_alpha": ridge_alpha,
                "selected_lambda": ridge_lambda,
                "grid": ridge_tuning_records,
            },
            "primary_personalized_model": primary_model,
        }
        (staging / "tuning.json").write_text(
            json.dumps(tuning, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "comparisons.json").write_text(
            json.dumps(comparison_outputs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        acceptance = {
            "primary_personalized_model": primary_model,
            "primary_passed": bool(gates[primary_model]["passed"]),
            "models": gates,
        }
        (staging / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        resolved_config = {
            **config,
            "overwrite": overwrite,
            "config_path": portable_path(config_path) if config_path else None,
        }
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        source_paths = [*split_paths.values(), vocabulary_path]
        for name in ("metadata.json", "statistics.json"):
            path = benchmark_dir / name
            if path.is_file():
                source_paths.append(path)
        implementation_paths = {
            Path(__file__).resolve(),
            Path(write_history_embeddings.__code__.co_filename).resolve(),
            Path(paired_bootstrap_metric_delta.__code__.co_filename).resolve(),
        }
        artifact_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "rule_version": RULE_VERSION,
            "benchmark_dir": portable_path(benchmark_dir),
            "artifact_dir": portable_path(artifact_dir),
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "implementation_sha256": {
                portable_path(path): _sha256(path)
                for path in sorted(implementation_paths)
            },
            "artifact_sha256": artifact_hashes,
            "model": {
                "name": model_name,
                "revision": revision,
                "device": device,
                "embedding_dimensions": int(fit.embeddings.shape[1]),
            },
            "users": {
                "fit": len(fit.user_ids),
                "validation": len(validation.user_ids),
                "dev": len(dev.user_ids),
            },
            "conditions": condition_paths,
            "embedding_summaries": {
                split: {
                    **summary,
                    "path": f"embeddings/{split}.parquet",
                }
                for split, summary in embedding_summaries.items()
            },
            "tuning_path": "tuning.json",
            "comparisons_path": "comparisons.json",
            "acceptance_path": "acceptance.json",
            "shuffle_manifest_path": "shuffle_manifest.parquet",
            "linear_model_path": "linear_model.npz",
            "acceptance": acceptance,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "scipy": importlib.metadata.version("scipy"),
                "sentence_transformers": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "torch": importlib.metadata.version("torch"),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if artifact_dir.exists():
            os.replace(artifact_dir, backup)
            try:
                os.replace(staging, artifact_dir)
            except BaseException:
                os.replace(backup, artifact_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, artifact_dir)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_semantic_history_signal_experiment(
        args.config, overwrite=args.overwrite
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
