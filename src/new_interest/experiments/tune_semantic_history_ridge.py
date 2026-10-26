from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.evaluate_static_candidate_baselines import (
    _embedding_batches,
    _internal_role,
    _read_train,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-balanced-semantic-history-ridge-v1"
DEFAULT_CONFIG = Path(
    "configs/future_interest/"
    "static_user_disjoint_balanced_semantic_history_ridge_v1.json"
)


@dataclass
class WeightedStatistics:
    weight_sum: float
    sum_x: np.ndarray
    xtx: np.ndarray
    label_counts: np.ndarray
    label_sum_x: np.ndarray


@dataclass(frozen=True)
class SingleTargetSplit:
    user_ids: tuple[str, ...]
    embeddings: np.ndarray
    targets: np.ndarray
    history_mask: np.ndarray


@dataclass(frozen=True)
class MultiTargetSplit:
    user_ids: tuple[str, ...]
    embeddings: np.ndarray
    positives: np.ndarray
    history_mask: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, object]
    per_interest: tuple[dict[str, object], ...]
    top_k: np.ndarray
    top_scores: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(
    path_or_config: Path | str | Mapping[str, object],
) -> tuple[dict[str, object], Path | None]:
    if isinstance(path_or_config, Mapping):
        return dict(path_or_config), None
    path = Path(path_or_config)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment config must be a JSON object")
    return value, path


def _empty_statistics(dimensions: int, label_count: int) -> WeightedStatistics:
    return WeightedStatistics(
        weight_sum=0.0,
        sum_x=np.zeros(dimensions, dtype=np.float64),
        xtx=np.zeros((dimensions, dimensions), dtype=np.float64),
        label_counts=np.zeros(label_count, dtype=np.float64),
        label_sum_x=np.zeros((label_count, dimensions), dtype=np.float64),
    )


def _normalised_class_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    """Return n_y^-beta weights, normalised to mean sample weight one."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not counts.size or np.any(counts < 0):
        raise ValueError("class counts must be a non-empty non-negative vector")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("class-balance beta must be finite and in [0, 1]")
    present = counts > 0
    if not np.any(present):
        raise ValueError("class counts contain no observations")
    raw = np.zeros_like(counts)
    raw[present] = counts[present] ** (-beta)
    scale = counts.sum() / np.dot(counts, raw)
    return raw * scale


def _update_statistics(
    stats: WeightedStatistics,
    matrix: np.ndarray,
    targets: np.ndarray,
    class_weights: np.ndarray,
) -> None:
    if not len(matrix):
        return
    weights = class_weights[targets]
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("observed samples must have finite positive weights")
    stats.weight_sum += float(weights.sum())
    stats.sum_x += matrix.T @ weights
    scaled = np.asarray(
        matrix * np.sqrt(weights).astype(np.float32)[:, None],
        dtype=np.float32,
    )
    stats.xtx += np.asarray(scaled.T @ scaled, dtype=np.float64)
    stats.label_counts += np.bincount(
        targets,
        weights=weights,
        minlength=len(stats.label_counts),
    )
    np.add.at(
        stats.label_sum_x,
        targets,
        matrix.astype(np.float64) * weights[:, None],
    )


def _fit_ridge(
    stats: WeightedStatistics,
    alphas: Sequence[float],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    if stats.weight_sum <= 0.0:
        raise ValueError("cannot fit empty weighted statistics")
    if not alphas or any(not math.isfinite(value) or value <= 0.0 for value in alphas):
        raise ValueError("ridge alphas must be finite and positive")
    mean_x = (stats.sum_x / stats.weight_sum).astype(np.float32)
    covariance = stats.xtx - np.outer(stats.sum_x, stats.sum_x) / stats.weight_sum
    covariance = (covariance + covariance.T) * 0.5
    q_mean = stats.label_counts / stats.weight_sum
    cross = stats.label_sum_x.T - np.outer(stats.sum_x, q_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross
    weights = {
        float(alpha): np.asarray(
            eigenvectors
            @ (projected / (eigenvalues[:, None] + float(alpha))),
            dtype=np.float32,
        )
        for alpha in alphas
    }
    return mean_x, weights


def _counts_by_role(
    target_by_user: Mapping[str, int],
    label_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    all_counts = np.zeros(label_count, dtype=np.int64)
    fit_counts = np.zeros(label_count, dtype=np.int64)
    role_counts = {"model_fit": 0, "tune": 0, "holdout": 0}
    for user_id, target in target_by_user.items():
        all_counts[target] += 1
        role = _internal_role(user_id, seed)
        role_counts[role] += 1
        if role == "model_fit":
            fit_counts[target] += 1
    return all_counts, fit_counts, role_counts


def _load_statistics_and_internal_splits(
    embedding_paths: Sequence[Path],
    target_by_user: Mapping[str, int],
    contexts: Mapping[str, Mapping[str, object]],
    *,
    label_count: int,
    seed: int,
    betas: Sequence[float],
) -> tuple[
    dict[float, WeightedStatistics],
    dict[float, WeightedStatistics],
    dict[str, SingleTargetSplit],
    dict[str, object],
]:
    all_counts, fit_counts, role_counts = _counts_by_role(
        target_by_user, label_count, seed
    )
    all_class_weights = {
        float(beta): _normalised_class_weights(all_counts, float(beta))
        for beta in betas
    }
    fit_class_weights = {
        float(beta): _normalised_class_weights(fit_counts, float(beta))
        for beta in betas
    }
    all_stats: dict[float, WeightedStatistics] = {}
    fit_stats: dict[float, WeightedStatistics] = {}
    retained_users: dict[str, list[str]] = {"tune": [], "holdout": []}
    retained_x: dict[str, list[np.ndarray]] = {"tune": [], "holdout": []}
    retained_y: dict[str, list[np.ndarray]] = {"tune": [], "holdout": []}
    seen: set[str] = set()
    dimensions: int | None = None
    for path in embedding_paths:
        for users, matrix in _embedding_batches(path):
            if dimensions is None:
                dimensions = matrix.shape[1]
                all_stats = {
                    float(beta): _empty_statistics(dimensions, label_count)
                    for beta in betas
                }
                fit_stats = {
                    float(beta): _empty_statistics(dimensions, label_count)
                    for beta in betas
                }
            elif matrix.shape[1] != dimensions:
                raise ValueError("embedding dimensions changed")
            targets = np.asarray(
                [target_by_user[user_id] for user_id in users], dtype=np.int64
            )
            roles = np.asarray([_internal_role(user_id, seed) for user_id in users])
            for user_id in users:
                if user_id in seen:
                    raise ValueError(f"duplicate Train embedding user {user_id}")
                seen.add(user_id)
            fit_mask = roles == "model_fit"
            for beta in betas:
                value = float(beta)
                _update_statistics(
                    all_stats[value], matrix, targets, all_class_weights[value]
                )
                _update_statistics(
                    fit_stats[value],
                    matrix[fit_mask],
                    targets[fit_mask],
                    fit_class_weights[value],
                )
            for role in ("tune", "holdout"):
                mask = roles == role
                if np.any(mask):
                    retained_users[role].extend(
                        user for user, keep in zip(users, mask, strict=True) if keep
                    )
                    retained_x[role].append(matrix[mask].copy())
                    retained_y[role].append(targets[mask].copy())
    if seen != set(target_by_user):
        missing = sorted(set(target_by_user) - seen)[:5]
        extra = sorted(seen - set(target_by_user))[:5]
        raise ValueError(f"Train embedding users differ; missing={missing}, extra={extra}")
    splits: dict[str, SingleTargetSplit] = {}
    for role in ("tune", "holdout"):
        users = tuple(retained_users[role])
        matrix = np.ascontiguousarray(np.concatenate(retained_x[role], axis=0))
        targets = np.concatenate(retained_y[role], axis=0)
        history_mask = np.zeros((len(users), label_count), dtype=np.bool_)
        for row_index, user_id in enumerate(users):
            context = contexts[user_id]
            if context["role"] != role:
                raise ValueError(f"{user_id}: internal role mismatch")
            history_mask[row_index, list(context["history_ids"])] = True
            if history_mask[row_index, targets[row_index]]:
                raise ValueError(f"{user_id}: target occurs in History")
        splits[role] = SingleTargetSplit(users, matrix, targets, history_mask)
    metadata = {
        "role_counts": role_counts,
        "all_class_counts": all_counts.tolist(),
        "model_fit_class_counts": fit_counts.tolist(),
        "class_weight_normalisation": "mean observed sample weight equals 1",
    }
    return fit_stats, all_stats, splits, metadata


def _top_k(
    scores: np.ndarray,
    history_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape != history_mask.shape:
        raise ValueError("score and History-mask shapes differ")
    if not 0 < k <= scores.shape[1]:
        raise ValueError("k is outside the label space")
    masked = np.asarray(scores, dtype=np.float32).copy()
    masked[history_mask] = -np.inf
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    values = np.take_along_axis(masked, order, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("not enough non-History candidates to produce Top-K")
    return order, values


def _evaluate_top_k(
    scores: np.ndarray,
    positives: np.ndarray,
    history_mask: np.ndarray,
    labels: Sequence[str],
    *,
    k: int,
) -> Evaluation:
    top_k, top_scores = _top_k(scores, history_mask, k)
    selected_hits = np.take_along_axis(positives, top_k, axis=1)
    support = positives.sum(axis=0, dtype=np.int64)
    retrieved_labels = top_k[selected_hits]
    retrieved = np.bincount(retrieved_labels, minlength=len(labels))
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    selected_gains = np.broadcast_to(discounts, selected_hits.shape)[selected_hits]
    gains = np.bincount(
        retrieved_labels,
        weights=selected_gains,
        minlength=len(labels),
    )
    recall = np.zeros(len(labels), dtype=np.float64)
    ndcg = np.zeros(len(labels), dtype=np.float64)
    present = support > 0
    recall[present] = retrieved[present] / support[present]
    ndcg[present] = gains[present] / support[present]
    target_count = positives.sum(axis=1, dtype=np.int64)
    hit_count = selected_hits.sum(axis=1, dtype=np.int64)
    user_recall = hit_count / target_count
    user_dcg = (selected_hits * discounts[None, :]).sum(axis=1)
    idcg_prefix = np.cumsum(discounts)
    user_idcg = idcg_prefix[np.minimum(target_count, k) - 1]
    user_ndcg = user_dcg / user_idcg
    per_interest = tuple(
        {
            "interest_id": index,
            "interest": str(label),
            "support": int(support[index]),
            f"recall_at_{k}": float(recall[index]) if support[index] else None,
            f"ndcg_at_{k}": float(ndcg[index]) if support[index] else None,
        }
        for index, label in enumerate(labels)
    )
    return Evaluation(
        metrics={
            "users": int(len(positives)),
            "distinct_positive_pairs": int(positives.sum()),
            "supported_interests": int(present.sum()),
            "k": int(k),
            f"interest_macro_recall_at_{k}": float(recall[present].mean()),
            f"interest_macro_ndcg_at_{k}": float(ndcg[present].mean()),
            f"user_macro_recall_at_{k}": float(user_recall.mean()),
            f"user_macro_ndcg_at_{k}": float(user_ndcg.mean()),
        },
        per_interest=per_interest,
        top_k=top_k,
        top_scores=top_scores,
    )


def _single_target_positives(split: SingleTargetSplit, label_count: int) -> np.ndarray:
    positives = np.zeros((len(split.user_ids), label_count), dtype=np.bool_)
    positives[np.arange(len(split.user_ids)), split.targets] = True
    return positives


def _primary_metrics(
    split: SingleTargetSplit,
    mean_x: np.ndarray,
    weights: np.ndarray,
    labels: Sequence[str],
    k: int,
) -> tuple[float, float]:
    scores = (split.embeddings - mean_x) @ weights
    result = _evaluate_top_k(
        scores,
        _single_target_positives(split, len(labels)),
        split.history_mask,
        labels,
        k=k,
    )
    return (
        float(result.metrics[f"interest_macro_recall_at_{k}"]),
        float(result.metrics[f"interest_macro_ndcg_at_{k}"]),
    )


def _global_scores(counts: np.ndarray, prior_alpha: float = 1.0) -> np.ndarray:
    probabilities = (counts + prior_alpha) / (
        counts.sum() + prior_alpha * len(counts)
    )
    return np.log(probabilities).astype(np.float32)


def _pareto_front(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in records:
        recall = float(row["interest_macro_recall_at_10"])
        ndcg = float(row["interest_macro_ndcg_at_10"])
        dominated = any(
            (
                float(other["interest_macro_recall_at_10"]) >= recall
                and float(other["interest_macro_ndcg_at_10"]) >= ndcg
                and (
                    float(other["interest_macro_recall_at_10"]) > recall
                    or float(other["interest_macro_ndcg_at_10"]) > ndcg
                )
            )
            for other in records
        )
        if not dominated:
            output.append(dict(row))
    return sorted(
        output,
        key=lambda row: (
            float(row["interest_macro_recall_at_10"]),
            float(row["interest_macro_ndcg_at_10"]),
        ),
    )


def _select_configuration(
    records: Sequence[Mapping[str, object]],
    *,
    global_recall: float,
    recall_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not records:
        raise ValueError("cannot select from an empty grid")
    if recall_tolerance < 0.0:
        raise ValueError("recall tolerance must be non-negative")
    frontier = _pareto_front(records)
    best_recall = max(float(row["interest_macro_recall_at_10"]) for row in records)
    floor = max(float(global_recall), best_recall - recall_tolerance)
    eligible = [
        row
        for row in frontier
        if float(row["interest_macro_recall_at_10"]) >= floor
    ]
    if not eligible:
        raise ValueError("no Pareto configuration satisfies the Recall constraint")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["interest_macro_ndcg_at_10"]),
            float(row["interest_macro_recall_at_10"]),
            -float(row["class_balance_beta"]),
            -float(row["ridge_alpha"]),
        ),
    )
    return dict(selected), frontier


def _load_dev_split(
    benchmark_path: Path,
    embedding_path: Path,
    labels: Sequence[str],
) -> MultiTargetSplit:
    label_to_id = {label: index for index, label in enumerate(labels)}
    embedding_users: list[str] = []
    embedding_parts: list[np.ndarray] = []
    for users, matrix in _embedding_batches(embedding_path):
        embedding_users.extend(users)
        embedding_parts.append(matrix)
    embeddings = np.ascontiguousarray(np.concatenate(embedding_parts, axis=0))
    embedding_index = {user: index for index, user in enumerate(embedding_users)}
    if len(embedding_index) != len(embedding_users):
        raise ValueError("duplicate Dev embedding user")
    users: list[str] = []
    matrices: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    history_masks: list[np.ndarray] = []
    for batch in pq.ParquetFile(benchmark_path).iter_batches(
        columns=[
            "user_id",
            "history_unique_subcategories",
            "future_novel_subcategories",
        ],
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id not in embedding_index:
                raise ValueError(f"missing Dev embedding for {user_id}")
            positive = np.zeros(len(labels), dtype=np.bool_)
            for label in row["future_novel_subcategories"]:
                value = str(label)
                if value not in label_to_id:
                    raise ValueError(f"Dev target lacks Train support: {value!r}")
                positive[label_to_id[value]] = True
            history = np.zeros(len(labels), dtype=np.bool_)
            for label in row["history_unique_subcategories"]:
                value = str(label)
                if value in label_to_id:
                    history[label_to_id[value]] = True
            if not positive.any() or np.any(positive & history):
                raise ValueError(f"{user_id}: invalid Dev positives or History overlap")
            users.append(user_id)
            matrices.append(embeddings[embedding_index[user_id]])
            positives.append(positive)
            history_masks.append(history)
    if set(users) != set(embedding_users):
        raise ValueError("Dev benchmark and embedding users differ")
    return MultiTargetSplit(
        tuple(users),
        np.ascontiguousarray(np.stack(matrices)),
        np.stack(positives),
        np.stack(history_masks),
    )


def _evaluate_model(
    split: SingleTargetSplit | MultiTargetSplit,
    mean_x: np.ndarray,
    weights: np.ndarray,
    labels: Sequence[str],
    *,
    k: int,
) -> Evaluation:
    positives = (
        _single_target_positives(split, len(labels))
        if isinstance(split, SingleTargetSplit)
        else split.positives
    )
    scores = (split.embeddings - mean_x) @ weights
    return _evaluate_top_k(scores, positives, split.history_mask, labels, k=k)


def _evaluate_global(
    split: SingleTargetSplit | MultiTargetSplit,
    scores: np.ndarray,
    labels: Sequence[str],
    *,
    k: int,
) -> Evaluation:
    positives = (
        _single_target_positives(split, len(labels))
        if isinstance(split, SingleTargetSplit)
        else split.positives
    )
    matrix = np.broadcast_to(scores, (len(split.user_ids), len(scores)))
    return _evaluate_top_k(matrix, positives, split.history_mask, labels, k=k)


def _write_condition(
    staging: Path,
    split_name: str,
    condition_name: str,
    split: SingleTargetSplit | MultiTargetSplit,
    result: Evaluation,
    labels: Sequence[str],
) -> dict[str, str]:
    output = staging / "conditions" / split_name / condition_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(list(result.per_interest)),
        output / "per_interest_metrics.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    prediction_rows = [
        {
            "user_id": user_id,
            "ranked_interests": [labels[int(value)] for value in result.top_k[index]],
            "ranked_scores": result.top_scores[index].astype(float).tolist(),
        }
        for index, user_id in enumerate(split.user_ids)
    ]
    pq.write_table(
        pa.Table.from_pylist(prediction_rows),
        output / "top10_predictions.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=4_096,
    )
    prefix = f"conditions/{split_name}/{condition_name}"
    return {
        "metrics": f"{prefix}/metrics.json",
        "per_interest_metrics": f"{prefix}/per_interest_metrics.parquet",
        "predictions": f"{prefix}/top10_predictions.parquet",
    }


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    backup = output.with_name(output.name + ".old")
    if backup.exists():
        raise FileExistsError(f"stale backup blocks publication: {backup}")
    if output.exists():
        os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output)


def run_balanced_semantic_history_ridge(
    path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    overwrite: bool = False,
) -> Path:
    config, config_path = _load_config(path_or_config)
    dataset_dir = Path(str(config["dataset_dir"]))
    benchmark_dir = Path(str(config["semantic_benchmark_dir"]))
    embedding_dir = Path(str(config["embedding_dir"]))
    output_dir = Path(str(config["output_dir"]))
    seed = int(config["seed"])
    k = int(config["k"])
    alphas = tuple(float(value) for value in config["ridge_alphas"])
    betas = tuple(float(value) for value in config["class_balance_betas"])
    selection = dict(config["selection"])
    recall_tolerance = float(selection["best_recall_tolerance_absolute"])
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")
    if not alphas or any(not math.isfinite(value) or value <= 0 for value in alphas):
        raise ValueError("ridge_alphas must contain finite positive values")
    if not betas or any(not math.isfinite(value) or not 0 <= value <= 1 for value in betas):
        raise ValueError("class_balance_betas must be in [0, 1]")
    train_path = dataset_dir / "train.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    dev_benchmark_path = benchmark_dir / "dev.parquet"
    embedding_paths = (embedding_dir / "fit.parquet", embedding_dir / "validation.parquet")
    dev_embedding_path = embedding_dir / "dev.parquet"
    source_paths = [
        train_path,
        vocabulary_path,
        dev_benchmark_path,
        *embedding_paths,
        dev_embedding_path,
    ]
    if config_path is not None:
        source_paths.append(config_path)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    target_counts_raw: dict[str, int] = {}
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["target_interest"], batch_size=16_384
    ):
        for row in batch.to_pylist():
            label = str(row["target_interest"]["subcategory"])
            target_counts_raw[label] = target_counts_raw.get(label, 0) + 1
    labels = tuple(sorted(target_counts_raw))
    target_by_user, _, _, contexts = _read_train(train_path, labels, seed=seed)
    fit_stats, all_stats, internal, split_metadata = (
        _load_statistics_and_internal_splits(
            embedding_paths,
            target_by_user,
            contexts,
            label_count=len(labels),
            seed=seed,
            betas=betas,
        )
    )
    tune = internal["tune"]
    holdout = internal["holdout"]
    fit_counts = np.asarray(split_metadata["model_fit_class_counts"], dtype=np.float64)
    tune_global_scores = _global_scores(fit_counts)
    tune_global = _evaluate_global(tune, tune_global_scores, labels, k=k)
    grid: list[dict[str, object]] = []
    fitted_internal: dict[float, tuple[np.ndarray, dict[float, np.ndarray]]] = {}
    for beta in betas:
        value = float(beta)
        mean_x, weights_by_alpha = _fit_ridge(fit_stats[value], alphas)
        fitted_internal[value] = (mean_x, weights_by_alpha)
        for alpha in alphas:
            recall, ndcg = _primary_metrics(
                tune, mean_x, weights_by_alpha[float(alpha)], labels, k
            )
            grid.append(
                {
                    "ridge_alpha": float(alpha),
                    "class_balance_beta": value,
                    f"interest_macro_recall_at_{k}": recall,
                    f"interest_macro_ndcg_at_{k}": ndcg,
                    "supported_interests": int(
                        np.unique(tune.targets).size
                    ),
                    "users": len(tune.user_ids),
                }
            )
    if k != 10:
        raise ValueError("the v1 Pareto selector is defined for k=10")
    selected, frontier = _select_configuration(
        grid,
        global_recall=float(tune_global.metrics[f"interest_macro_recall_at_{k}"]),
        recall_tolerance=recall_tolerance,
    )
    selected_alpha = float(selected["ridge_alpha"])
    selected_beta = float(selected["class_balance_beta"])
    internal_mean, internal_weights_by_alpha = fitted_internal[selected_beta]
    selected_holdout = _evaluate_model(
        holdout,
        internal_mean,
        internal_weights_by_alpha[selected_alpha],
        labels,
        k=k,
    )
    current_mean, current_weights_by_alpha = fitted_internal[0.0]
    if 10.0 not in current_weights_by_alpha:
        _, extra = _fit_ridge(fit_stats[0.0], (10.0,))
        current_weights = extra[10.0]
    else:
        current_weights = current_weights_by_alpha[10.0]
    current_holdout = _evaluate_model(
        holdout, current_mean, current_weights, labels, k=k
    )
    holdout_global = _evaluate_global(holdout, tune_global_scores, labels, k=k)
    dev = _load_dev_split(dev_benchmark_path, dev_embedding_path, labels)
    final_mean, final_weights_by_alpha = _fit_ridge(
        all_stats[selected_beta], (selected_alpha,)
    )
    selected_dev = _evaluate_model(
        dev, final_mean, final_weights_by_alpha[selected_alpha], labels, k=k
    )
    current_final_mean, current_final_weights = _fit_ridge(
        all_stats[0.0], (10.0,)
    )
    current_dev = _evaluate_model(
        dev, current_final_mean, current_final_weights[10.0], labels, k=k
    )
    all_counts = np.asarray(split_metadata["all_class_counts"], dtype=np.float64)
    dev_global = _evaluate_global(dev, _global_scores(all_counts), labels, k=k)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".balanced-semantic-history-ridge-",
        dir=str(output_dir.parent),
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        np.savez_compressed(
            staging / "selected_model.npz",
            labels=np.asarray(labels),
            mean_embedding=final_mean,
            ridge_weights=final_weights_by_alpha[selected_alpha],
            ridge_alpha=np.asarray([selected_alpha]),
            class_balance_beta=np.asarray([selected_beta]),
        )
        tuning = {
            "selection_scope": "Static Train internal tune only",
            "selection_metric": (
                "Pareto frontier; require Recall@10 >= max(Global, best grid "
                f"Recall@10 - {recall_tolerance}); maximize NDCG@10"
            ),
            "model_fit_users": split_metadata["role_counts"]["model_fit"],
            "tune_users": split_metadata["role_counts"]["tune"],
            "holdout_users": split_metadata["role_counts"]["holdout"],
            "tune_global": tune_global.metrics,
            "selected": selected,
            "pareto_frontier": frontier,
            "grid": grid,
        }
        (staging / "tuning.json").write_text(
            json.dumps(tuning, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        conditions = {
            "internal_holdout": {
                "global_popularity": _write_condition(
                    staging, "internal_holdout", "global_popularity", holdout,
                    holdout_global, labels
                ),
                "current_ridge_alpha10_beta0": _write_condition(
                    staging, "internal_holdout", "current_ridge_alpha10_beta0",
                    holdout, current_holdout, labels
                ),
                "selected_balanced_ridge": _write_condition(
                    staging, "internal_holdout", "selected_balanced_ridge",
                    holdout, selected_holdout, labels
                ),
            },
            "dev_exploratory": {
                "global_popularity": _write_condition(
                    staging, "dev_exploratory", "global_popularity", dev,
                    dev_global, labels
                ),
                "current_ridge_alpha10_beta0": _write_condition(
                    staging, "dev_exploratory", "current_ridge_alpha10_beta0",
                    dev, current_dev, labels
                ),
                "selected_balanced_ridge": _write_condition(
                    staging, "dev_exploratory", "selected_balanced_ridge", dev,
                    selected_dev, labels
                ),
            },
        }
        resolved = {
            **config,
            "rule_version": RULE_VERSION,
            "dataset_dir": portable_path(dataset_dir),
            "semantic_benchmark_dir": portable_path(benchmark_dir),
            "embedding_dir": portable_path(embedding_dir),
            "output_dir": portable_path(output_dir),
            "supervision": "Static Train selected target only",
            "embedding_status": "frozen; no re-encoding",
            "class_weight_formula": (
                "w_y = n_y^(-beta), normalised so observed sample weights mean 1"
            ),
            "candidate_space": (
                "124 Static-Train-supported interests minus each user's raw "
                "History interests"
            ),
        }
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "selected_ridge_alpha": selected_alpha,
            "selected_class_balance_beta": selected_beta,
            "conditions": conditions,
            "internal_holdout": {
                "global_popularity": holdout_global.metrics,
                "current_ridge_alpha10_beta0": current_holdout.metrics,
                "selected_balanced_ridge": selected_holdout.metrics,
            },
            "dev_exploratory": {
                "global_popularity": dev_global.metrics,
                "current_ridge_alpha10_beta0": current_dev.metrics,
                "selected_balanced_ridge": selected_dev.metrics,
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "rule_version": RULE_VERSION,
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "output_sha256": output_hashes,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune fixed-embedding Semantic History Ridge over alpha x class "
            "balance beta using Interest-Macro Recall/NDCG Pareto selection."
        )
    )
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = run_balanced_semantic_history_ridge(
        args.config, overwrite=args.overwrite
    )
    print(output)


if __name__ == "__main__":
    main()
