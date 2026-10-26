from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.candidate_metrics import evaluate_candidate_outputs
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-candidate-baselines-v1"
DEFAULT_DATASET_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_v1"
)
DEFAULT_CANDIDATE_DIR = Path(
    "data/training/candidate_selection_static_user_disjoint_k10_v1"
)
DEFAULT_EMBEDDING_DIR = Path(
    "artifacts/future-novel-interest/"
    "static-user-disjoint-semantic-v1/embeddings"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines"
)
DEFAULT_SEED = 20260815
DEFAULT_SHUFFLE_SEEDS = (20260815, 20260816, 20260817)
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_RIDGE_ALPHAS = (1.0, 10.0, 100.0)
DEFAULT_SEMANTIC_WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
STRATEGIES = ("uniform", "train_frequency_matched")


PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("selection_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("answer_category", pa.string(), nullable=False),
        pa.field("predicted_category", pa.string(), nullable=False),
        pa.field("output", pa.string(), nullable=False),
        pa.field("correct", pa.bool_(), nullable=False),
    ]
)


@dataclass
class SufficientStatistics:
    count: int
    sum_x: np.ndarray
    xtx: np.ndarray
    label_counts: np.ndarray
    label_sum_x: np.ndarray


@dataclass(frozen=True)
class SemanticModel:
    labels: tuple[str, ...]
    global_scores: np.ndarray
    mean_embedding: np.ndarray
    ridge_weights: np.ndarray
    ridge_alpha: float
    semantic_weight: float


def _float_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of numbers"
        ) from error
    if not values or not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated list of finite numbers"
        )
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(seed: int, namespace: str, *parts: str) -> str:
    material = ":".join((str(seed), namespace, *parts))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _internal_role(user_id: str, seed: int) -> str:
    bucket = int(_priority(seed, "semantic-internal-split", user_id)[:16], 16) % 10_000
    if bucket < 8_000:
        return "model_fit"
    if bucket < 9_000:
        return "tune"
    return "holdout"


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


def _head_torso_tail(
    train_counts: Counter[str],
) -> dict[str, set[str]]:
    labels = sorted(train_counts, key=lambda label: (-train_counts[label], label))
    head_end = max(1, math.ceil(len(labels) * 0.2))
    torso_end = max(head_end + 1, math.ceil(len(labels) * 0.6))
    return {
        "head": set(labels[:head_end]),
        "torso": set(labels[head_end:torso_end]),
        "tail": set(labels[torso_end:]),
    }


def _read_train(
    train_path: Path,
    labels: tuple[str, ...],
    *,
    seed: int,
) -> tuple[
    dict[str, int],
    Counter[str],
    dict[str, Counter[str]],
    dict[str, dict[str, object]],
]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    target_by_user: dict[str, int] = {}
    target_counts: Counter[str] = Counter()
    last_conditional: dict[str, Counter[str]] = defaultdict(Counter)
    heldout_context: dict[str, dict[str, object]] = {}
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["user_id", "history", "target_interest"],
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in target_by_user:
                raise ValueError(f"Duplicate Static Train user {user_id}")
            target = str(row["target_interest"]["subcategory"])
            history = row["history"]
            history_labels = tuple(str(item["subcategory"]) for item in history)
            if not history_labels:
                raise ValueError(f"{user_id}: empty Train History")
            if target not in label_to_id:
                raise ValueError(f"{user_id}: unknown Train target {target!r}")
            target_by_user[user_id] = label_to_id[target]
            target_counts[target] += 1
            last_conditional[history_labels[-1]][target] += 1
            role = _internal_role(user_id, seed)
            if role in {"tune", "holdout"}:
                heldout_context[user_id] = {
                    "role": role,
                    "target_id": label_to_id[target],
                    "history_ids": tuple(
                        label_to_id[value]
                        for value in history_labels
                        if value in label_to_id
                    ),
                }
    if not target_by_user:
        raise ValueError(f"No Train rows in {train_path}")
    return (
        target_by_user,
        target_counts,
        dict(last_conditional),
        heldout_context,
    )


def _embedding_batches(path: Path) -> Iterable[tuple[list[str], np.ndarray]]:
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["user_id", "embedding"], batch_size=4_096
    ):
        users = [str(value) for value in batch["user_id"].to_pylist()]
        matrix = np.asarray(batch["embedding"].to_pylist(), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(users):
            raise ValueError(f"Malformed embedding batch in {path}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite embedding in {path}")
        yield users, matrix


def _empty_stats(dimensions: int, labels: int) -> SufficientStatistics:
    return SufficientStatistics(
        count=0,
        sum_x=np.zeros(dimensions, dtype=np.float64),
        xtx=np.zeros((dimensions, dimensions), dtype=np.float64),
        label_counts=np.zeros(labels, dtype=np.float64),
        label_sum_x=np.zeros((labels, dimensions), dtype=np.float64),
    )


def _update_stats(
    stats: SufficientStatistics,
    matrix: np.ndarray,
    targets: np.ndarray,
) -> None:
    stats.count += len(matrix)
    stats.sum_x += matrix.sum(axis=0, dtype=np.float64)
    stats.xtx += np.asarray(matrix.T @ matrix, dtype=np.float64)
    stats.label_counts += np.bincount(
        targets, minlength=len(stats.label_counts)
    )
    np.add.at(stats.label_sum_x, targets, matrix)


def _load_train_embedding_statistics(
    embedding_paths: Sequence[Path],
    target_by_user: dict[str, int],
    heldout_context: dict[str, dict[str, object]],
    *,
    label_count: int,
    seed: int,
) -> tuple[
    SufficientStatistics,
    SufficientStatistics,
    dict[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]],
]:
    seen: set[str] = set()
    fit_stats: SufficientStatistics | None = None
    all_stats: SufficientStatistics | None = None
    retained_users: dict[str, list[str]] = {"tune": [], "holdout": []}
    retained_x: dict[str, list[np.ndarray]] = {"tune": [], "holdout": []}
    retained_y: dict[str, list[np.ndarray]] = {"tune": [], "holdout": []}
    for path in embedding_paths:
        for users, matrix in _embedding_batches(path):
            if fit_stats is None:
                fit_stats = _empty_stats(matrix.shape[1], label_count)
                all_stats = _empty_stats(matrix.shape[1], label_count)
            elif matrix.shape[1] != len(fit_stats.sum_x):
                raise ValueError("Embedding dimensions changed")
            targets = np.asarray(
                [target_by_user[user_id] for user_id in users], dtype=np.int64
            )
            for user_id in users:
                if user_id in seen:
                    raise ValueError(f"Duplicate Train embedding user {user_id}")
                seen.add(user_id)
            assert all_stats is not None
            _update_stats(all_stats, matrix, targets)
            roles = np.asarray(
                [_internal_role(user_id, seed) for user_id in users], dtype=object
            )
            fit_mask = roles == "model_fit"
            if np.any(fit_mask):
                _update_stats(fit_stats, matrix[fit_mask], targets[fit_mask])
            for role in ("tune", "holdout"):
                mask = roles == role
                if np.any(mask):
                    retained_users[role].extend(
                        user_id
                        for user_id, keep in zip(users, mask, strict=True)
                        if keep
                    )
                    retained_x[role].append(matrix[mask].copy())
                    retained_y[role].append(targets[mask].copy())
    if fit_stats is None or all_stats is None:
        raise ValueError("No Train embeddings")
    if seen != set(target_by_user):
        missing = sorted(set(target_by_user) - seen)[:5]
        extra = sorted(seen - set(target_by_user))[:5]
        raise ValueError(
            f"Train embedding users differ; missing={missing}, extra={extra}"
        )
    retained = {
        role: (
            tuple(retained_users[role]),
            np.concatenate(retained_x[role], axis=0),
            np.concatenate(retained_y[role], axis=0),
        )
        for role in ("tune", "holdout")
    }
    for role, (users, matrix, targets) in retained.items():
        if not (len(users) == len(matrix) == len(targets)):
            raise ValueError(f"{role}: retained embedding alignment failed")
        if any(heldout_context[user_id]["role"] != role for user_id in users):
            raise ValueError(f"{role}: internal role mismatch")
    return fit_stats, all_stats, retained


def _fit_weights(
    stats: SufficientStatistics,
    *,
    prior_alpha: float,
    ridge_alphas: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    if stats.count <= 0:
        raise ValueError("Cannot fit an empty semantic model")
    label_count = len(stats.label_counts)
    probabilities = (stats.label_counts + prior_alpha) / (
        stats.count + prior_alpha * label_count
    )
    global_scores = np.log(probabilities).astype(np.float32)
    mean_x = (stats.sum_x / stats.count).astype(np.float32)
    covariance = stats.xtx - np.outer(stats.sum_x, stats.sum_x) / stats.count
    covariance = (covariance + covariance.T) * 0.5
    q_mean = stats.label_counts / stats.count
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
        for alpha in ridge_alphas
    }
    return global_scores, mean_x, weights


def _internal_candidates(
    users: Sequence[str],
    contexts: Mapping[str, Mapping[str, object]],
    *,
    label_count: int,
    k: int,
    seed: int,
) -> np.ndarray:
    output = np.empty((len(users), k), dtype=np.int16)
    for row_index, user_id in enumerate(users):
        context = contexts[user_id]
        target = int(context["target_id"])
        excluded = set(int(value) for value in context["history_ids"])
        excluded.add(target)
        pool = [value for value in range(label_count) if value not in excluded]
        negatives = sorted(
            pool,
            key=lambda label_id: (
                _priority(
                    seed,
                    "semantic-internal-negative",
                    user_id,
                    str(label_id),
                ),
                label_id,
            ),
        )[: k - 1]
        candidates = negatives + [target]
        candidates.sort(
            key=lambda label_id: (
                _priority(
                    seed,
                    "semantic-internal-order",
                    user_id,
                    str(label_id),
                ),
                label_id,
            )
        )
        output[row_index] = candidates
    return output


def _macro_accuracy_ids(targets: np.ndarray, predictions: np.ndarray) -> float:
    support = np.bincount(targets)
    correct = np.bincount(targets, weights=targets == predictions, minlength=len(support))
    present = support > 0
    return float(np.mean(correct[present] / support[present]))


def _semantic_candidate_predictions(
    matrix: np.ndarray,
    candidates: np.ndarray,
    global_scores: np.ndarray,
    mean_x: np.ndarray,
    weights: np.ndarray,
    semantic_weight: float,
) -> np.ndarray:
    residual = (matrix - mean_x) @ weights
    scores = np.take(global_scores, candidates) + semantic_weight * np.take_along_axis(
        residual, candidates, axis=1
    )
    return candidates[np.arange(len(candidates)), np.argmax(scores, axis=1)]


def _tune_semantic_model(
    fit_stats: SufficientStatistics,
    retained: Mapping[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]],
    contexts: Mapping[str, Mapping[str, object]],
    labels: tuple[str, ...],
    *,
    k: int,
    seed: int,
    ridge_alphas: Sequence[float],
    semantic_weights: Sequence[float],
) -> tuple[float, float, dict[str, object]]:
    global_scores, mean_x, weight_models = _fit_weights(
        fit_stats, prior_alpha=1.0, ridge_alphas=ridge_alphas
    )
    tune_users, tune_x, tune_y = retained["tune"]
    tune_candidates = _internal_candidates(
        tune_users,
        contexts,
        label_count=len(labels),
        k=k,
        seed=seed,
    )
    grid: list[dict[str, object]] = []
    selected: tuple[tuple[float, float, float, float], float, float] | None = None
    for alpha in ridge_alphas:
        weights = weight_models[float(alpha)]
        for semantic_weight in semantic_weights:
            predictions = _semantic_candidate_predictions(
                tune_x,
                tune_candidates,
                global_scores,
                mean_x,
                weights,
                float(semantic_weight),
            )
            macro = _macro_accuracy_ids(tune_y, predictions)
            micro = float(np.mean(tune_y == predictions))
            row = {
                "ridge_alpha": float(alpha),
                "semantic_weight": float(semantic_weight),
                "macro_category_accuracy": macro,
                "category_accuracy": micro,
                "samples": len(tune_y),
                "supported_categories": int(len(np.unique(tune_y))),
            }
            grid.append(row)
            key = (macro, micro, -float(semantic_weight), -float(alpha))
            if selected is None or key > selected[0]:
                selected = (key, float(alpha), float(semantic_weight))
    assert selected is not None
    alpha, semantic_weight = selected[1], selected[2]
    holdout_users, holdout_x, holdout_y = retained["holdout"]
    holdout_candidates = _internal_candidates(
        holdout_users,
        contexts,
        label_count=len(labels),
        k=k,
        seed=seed,
    )
    holdout_predictions = _semantic_candidate_predictions(
        holdout_x,
        holdout_candidates,
        global_scores,
        mean_x,
        weight_models[alpha],
        semantic_weight,
    )
    return alpha, semantic_weight, {
        "selection_scope": "Static Train internal tune only",
        "selection_metric": (
            "interest macro candidate accuracy; micro accuracy tie-break; "
            "then smaller semantic weight and ridge alpha"
        ),
        "split_seed": seed,
        "model_fit_users": fit_stats.count,
        "tune_users": len(tune_users),
        "holdout_users": len(holdout_users),
        "selected_ridge_alpha": alpha,
        "selected_semantic_weight": semantic_weight,
        "grid": grid,
        "internal_holdout": {
            "samples": len(holdout_y),
            "supported_categories": int(len(np.unique(holdout_y))),
            "macro_category_accuracy": _macro_accuracy_ids(
                holdout_y, holdout_predictions
            ),
            "category_accuracy": float(np.mean(holdout_y == holdout_predictions)),
        },
    }


def _load_dev_embeddings(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    users: list[str] = []
    matrices: list[np.ndarray] = []
    for batch_users, matrix in _embedding_batches(path):
        users.extend(batch_users)
        matrices.append(matrix)
    if len(users) != len(set(users)):
        raise ValueError("Duplicate Dev embedding user")
    return tuple(users), np.concatenate(matrices, axis=0)


def _load_candidate_rows(path: Path) -> list[dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    selection_ids = [str(row["selection_id"]) for row in rows]
    if len(selection_ids) != len(set(selection_ids)):
        raise ValueError(f"Duplicate selection_id in {path}")
    return rows


def _user_contexts(
    records: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    contexts: dict[str, dict[str, object]] = {}
    for row in records:
        user_id = str(row["user_id"])
        history = tuple(str(value) for value in row["history_subcategories"])
        value = {
            "history": history,
            "history_length": int(row["history_length"]),
            "topic_count": len(set(history)),
        }
        previous = contexts.setdefault(user_id, value)
        if previous != value:
            raise ValueError(f"{user_id}: inconsistent candidate History")
    return tuple(sorted(contexts)), contexts


def _shuffle_donors(
    users: Sequence[str],
    contexts: Mapping[str, Mapping[str, object]],
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, user_id in enumerate(users):
        context = contexts[user_id]
        groups[
            (
                _history_length_bucket(int(context["history_length"])),
                _topic_count_bucket(int(context["topic_count"])),
            )
        ].append(index)
    donors = np.arange(len(users), dtype=np.int64)
    levels = ["exact_bucket"] * len(users)
    buckets = [""] * len(users)
    for key, values in sorted(groups.items()):
        indices = np.asarray(sorted(values, key=lambda index: users[index]))
        bucket = f"history={key[0]}|topics={key[1]}"
        for index in indices:
            buckets[int(index)] = bucket
        if len(indices) <= 1:
            continue
        digest = hashlib.sha256(f"{seed}:{bucket}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        order = rng.permutation(indices)
        shift = int(rng.integers(1, len(order)))
        donors[order] = np.roll(order, shift)
    fixed = np.flatnonzero(donors == np.arange(len(users)))
    if len(fixed) > 1:
        ordered = np.asarray(
            sorted(fixed.tolist(), key=lambda index: (buckets[index], users[index]))
        )
        digest = hashlib.sha256(
            f"{seed}:singleton-relaxed-derangement".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        donors[ordered] = np.roll(
            ordered, int(rng.integers(1, len(ordered)))
        )
        for index in ordered:
            levels[int(index)] = "relaxed_singleton_derangement"
    elif len(fixed) == 1:
        index = int(fixed[0])
        candidates = np.asarray([value for value in range(len(users)) if value != index])
        source = contexts[users[index]]
        distance = np.asarray(
            [
                abs(
                    int(contexts[users[value]]["history_length"])
                    - int(source["history_length"])
                )
                + abs(
                    int(contexts[users[value]]["topic_count"])
                    - int(source["topic_count"])
                )
                for value in candidates
            ]
        )
        nearest = candidates[distance == distance.min()]
        donors[index] = min(
            nearest.tolist(),
            key=lambda value: _priority(seed, "nearest-donor", users[index], users[value]),
        )
        levels[index] = "relaxed_singleton_nearest_donor"
    if np.any(donors == np.arange(len(users))):
        raise AssertionError("Shuffle contains a fixed point")
    manifest = [
        {
            "seed": seed,
            "user_id": user_id,
            "donor_user_id": users[int(donors[index])],
            "bucket": buckets[index],
            "match_level": levels[index],
        }
        for index, user_id in enumerate(users)
    ]
    return donors, manifest


def _select_by_counts(
    candidates: Sequence[str],
    primary: Mapping[str, int],
    global_counts: Mapping[str, int],
) -> str:
    return min(
        candidates,
        key=lambda label: (
            -int(primary.get(label, 0)),
            -int(global_counts.get(label, 0)),
            label,
        ),
    )


def _augment_metrics(
    metrics: dict[str, object],
    train_counts: Counter[str],
) -> dict[str, object]:
    slices = _head_torso_tail(train_counts)
    per_category = dict(metrics["per_category_accuracy"])
    metrics["interest_frequency_slices"] = {
        name: {
            "supported_categories": len(
                [label for label in labels if label in per_category]
            ),
            "macro_category_accuracy": (
                float(
                    np.mean(
                        [
                            float(per_category[label])
                            for label in labels
                            if label in per_category
                        ]
                    )
                )
                if any(label in per_category for label in labels)
                else None
            ),
        }
        for name, labels in slices.items()
    }
    return metrics


def _evaluate_predictions(
    records: Sequence[dict[str, object]],
    predictions: Sequence[str],
    train_counts: Counter[str],
) -> dict[str, object]:
    outputs = [f"New topic: {prediction}" for prediction in predictions]
    metrics, _ = evaluate_candidate_outputs(list(records), outputs)
    return _augment_metrics(metrics, train_counts)


def _write_condition(
    root: Path,
    strategy: str,
    condition: str,
    records: Sequence[dict[str, object]],
    predictions: Sequence[str],
    metrics: Mapping[str, object],
) -> dict[str, str]:
    output = root / strategy / condition
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "selection_id": str(record["selection_id"]),
            "user_id": str(record["user_id"]),
            "answer_category": str(record["answer_category"]),
            "predicted_category": prediction,
            "output": f"New topic: {prediction}",
            "correct": prediction == str(record["answer_category"]),
        }
        for record, prediction in zip(records, predictions, strict=True)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA),
        output / "predictions.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "predictions": f"{strategy}/{condition}/predictions.parquet",
        "metrics": f"{strategy}/{condition}/metrics.json",
    }


def _bootstrap_comparisons(
    records: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Sequence[str]],
    pairs: Sequence[tuple[str, str]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("Bootstrap requires scipy") from error
    users = sorted({str(row["user_id"]) for row in records})
    categories = sorted({str(row["answer_category"]) for row in records})
    user_to_id = {user: index for index, user in enumerate(users)}
    category_to_id = {label: index for index, label in enumerate(categories)}
    row_users = np.asarray(
        [user_to_id[str(row["user_id"])] for row in records], dtype=np.int32
    )
    row_categories = np.asarray(
        [category_to_id[str(row["answer_category"])] for row in records],
        dtype=np.int16,
    )
    answers = np.asarray([str(row["answer_category"]) for row in records])
    support = sparse.csr_matrix(
        (
            np.ones(len(records), dtype=np.int16),
            (row_users, row_categories),
        ),
        shape=(len(users), len(categories)),
    )
    correct_matrices = {
        condition: sparse.csr_matrix(
            (
                (np.asarray(values) == answers).astype(np.int8),
                (row_users, row_categories),
            ),
            shape=(len(users), len(categories)),
        )
        for condition, values in predictions.items()
    }
    needed = sorted({condition for pair in pairs for condition in pair})
    rng = np.random.default_rng(seed)
    samples = {
        f"{left}_minus_{right}": np.empty(replicates, dtype=np.float64)
        for left, right in pairs
    }
    probabilities = np.full(len(users), 1.0 / len(users))
    offset = 0
    batch_size = 50
    while offset < replicates:
        size = min(batch_size, replicates - offset)
        weights = rng.multinomial(len(users), probabilities, size=size).T
        sampled_support = np.asarray(support.T @ weights, dtype=np.float64)
        present = sampled_support > 0
        condition_macro: dict[str, np.ndarray] = {}
        for condition in needed:
            sampled_correct = np.asarray(
                correct_matrices[condition].T @ weights, dtype=np.float64
            )
            accuracy = np.divide(
                sampled_correct,
                sampled_support,
                out=np.zeros_like(sampled_correct),
                where=present,
            )
            condition_macro[condition] = np.sum(accuracy, axis=0) / np.sum(
                present, axis=0
            )
        for left, right in pairs:
            samples[f"{left}_minus_{right}"][offset : offset + size] = (
                condition_macro[left] - condition_macro[right]
            )
        offset += size
    lower_q = 0.025
    upper_q = 0.975
    result: dict[str, object] = {
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "resampling_unit": "user",
        "aggregation": (
            "recompute supported-category accuracy in every user-cluster "
            "replicate, then average categories equally"
        ),
        "comparisons": {},
    }
    for left, right in pairs:
        name = f"{left}_minus_{right}"
        left_metrics = _evaluate_predictions(
            list(records), predictions[left], Counter()
        )
        right_metrics = _evaluate_predictions(
            list(records), predictions[right], Counter()
        )
        result["comparisons"][name] = {
            "point_delta": float(left_metrics["macro_category_accuracy"])
            - float(right_metrics["macro_category_accuracy"]),
            "bootstrap_mean": float(samples[name].mean()),
            "lower": float(np.quantile(samples[name], lower_q)),
            "upper": float(np.quantile(samples[name], upper_q)),
        }
    return result


def _publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    backup = staging.parent / "previous-output"
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def run_static_candidate_baselines(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    candidate_dir: Path | str = DEFAULT_CANDIDATE_DIR,
    embedding_dir: Path | str = DEFAULT_EMBEDDING_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = DEFAULT_SEED,
    shuffle_seeds: Sequence[int] = DEFAULT_SHUFFLE_SEEDS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    semantic_weights: Sequence[float] = DEFAULT_SEMANTIC_WEIGHTS,
    overwrite: bool = False,
) -> Path:
    dataset_dir = Path(dataset_dir)
    candidate_dir = Path(candidate_dir)
    embedding_dir = Path(embedding_dir)
    output_dir = Path(output_dir)
    ridge_alphas = tuple(float(value) for value in ridge_alphas)
    semantic_weights = tuple(float(value) for value in semantic_weights)
    if not ridge_alphas or any(
        not math.isfinite(value) or value <= 0 for value in ridge_alphas
    ):
        raise ValueError("ridge_alphas must contain finite positive values")
    if not semantic_weights or any(
        not math.isfinite(value) or value < 0 for value in semantic_weights
    ):
        raise ValueError("semantic_weights must contain finite non-negative values")
    if len(shuffle_seeds) < 3:
        raise ValueError("At least three shuffle seeds are required")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    train_path = dataset_dir / "train.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    train_embedding_paths = (
        embedding_dir / "fit.parquet",
        embedding_dir / "validation.parquet",
    )
    dev_embedding_path = embedding_dir / "dev.parquet"
    candidate_paths = {
        strategy: candidate_dir / strategy / "behavior_only/dev_expanded.parquet"
        for strategy in STRATEGIES
    }
    source_paths = [
        train_path,
        vocabulary_path,
        *train_embedding_paths,
        dev_embedding_path,
        *candidate_paths.values(),
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    vocabulary = [
        str(value)
        for value in pq.read_table(
            vocabulary_path, columns=["subcategory"]
        )["subcategory"].to_pylist()
    ]
    target_by_user_raw = Counter(
        str(row["target_interest"]["subcategory"])
        for batch in pq.ParquetFile(train_path).iter_batches(
            columns=["target_interest"], batch_size=16_384
        )
        for row in batch.to_pylist()
    )
    labels = tuple(sorted(target_by_user_raw))
    if not set(labels).issubset(vocabulary):
        raise ValueError("Train-supported labels differ from vocabulary")
    (
        target_by_user,
        train_counts,
        last_conditional,
        heldout_context,
    ) = _read_train(train_path, labels, seed=seed)
    fit_stats, all_stats, retained = _load_train_embedding_statistics(
        train_embedding_paths,
        target_by_user,
        heldout_context,
        label_count=len(labels),
        seed=seed,
    )
    selected_alpha, selected_weight, tuning = _tune_semantic_model(
        fit_stats,
        retained,
        heldout_context,
        labels,
        k=10,
        seed=seed,
        ridge_alphas=ridge_alphas,
        semantic_weights=semantic_weights,
    )
    final_global, final_mean, final_weights = _fit_weights(
        all_stats,
        prior_alpha=1.0,
        ridge_alphas=(selected_alpha,),
    )
    semantic_model = SemanticModel(
        labels=labels,
        global_scores=final_global,
        mean_embedding=final_mean,
        ridge_weights=final_weights[selected_alpha],
        ridge_alpha=selected_alpha,
        semantic_weight=selected_weight,
    )
    dev_embedding_users, dev_embeddings = _load_dev_embeddings(dev_embedding_path)
    dev_user_to_embedding = {
        user_id: index for index, user_id in enumerate(dev_embedding_users)
    }
    label_to_id = {label: index for index, label in enumerate(labels)}
    dev_residual = (
        dev_embeddings - semantic_model.mean_embedding
    ) @ semantic_model.ridge_weights

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-candidate-baselines-", dir=str(output_dir.parent)
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        (staging / "semantic_tuning.json").write_text(
            json.dumps(tuning, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            staging / "semantic_model.npz",
            labels=np.asarray(labels),
            global_scores=semantic_model.global_scores,
            mean_embedding=semantic_model.mean_embedding,
            ridge_weights=semantic_model.ridge_weights,
            ridge_alpha=np.asarray([semantic_model.ridge_alpha]),
            semantic_weight=np.asarray([semantic_model.semantic_weight]),
        )

        all_outputs: dict[str, object] = {}
        all_bootstrap: dict[str, object] = {}
        shuffle_manifest_rows: list[dict[str, object]] = []
        for strategy, candidate_path in candidate_paths.items():
            records = _load_candidate_rows(candidate_path)
            users, contexts = _user_contexts(records)
            if set(users) != set(dev_embedding_users):
                raise ValueError(
                    f"{strategy}: candidate and Dev embedding users differ"
                )
            user_to_index = {user_id: index for index, user_id in enumerate(users)}
            donor_maps: dict[int, np.ndarray] = {}
            for shuffle_seed in shuffle_seeds:
                donors, manifest = _shuffle_donors(
                    users, contexts, seed=int(shuffle_seed)
                )
                donor_maps[int(shuffle_seed)] = donors
                shuffle_manifest_rows.extend(
                    {"strategy": strategy, **row} for row in manifest
                )

            predictions: dict[str, list[str]] = {}
            for random_seed in shuffle_seeds:
                condition = f"uniform_random_{int(random_seed)}"
                predictions[condition] = [
                    list(row["candidates"])[
                        int(
                            _priority(
                                int(random_seed),
                                "random-candidate",
                                str(row["selection_id"]),
                            ),
                            16,
                        )
                        % len(row["candidates"])
                    ]
                    for row in records
                ]
            predictions["global_popularity"] = [
                _select_by_counts(
                    list(row["candidates"]), train_counts, train_counts
                )
                for row in records
            ]
            predictions["last_interest_actual"] = [
                _select_by_counts(
                    list(row["candidates"]),
                    last_conditional.get(
                        str(row["history_subcategories"][-1]), Counter()
                    ),
                    train_counts,
                )
                for row in records
            ]
            for shuffle_seed, donors in donor_maps.items():
                condition = f"last_interest_shuffled_{shuffle_seed}"
                predictions[condition] = []
                for row in records:
                    user_index = user_to_index[str(row["user_id"])]
                    donor_user = users[int(donors[user_index])]
                    donor_last = str(contexts[donor_user]["history"][-1])
                    predictions[condition].append(
                        _select_by_counts(
                            list(row["candidates"]),
                            last_conditional.get(donor_last, Counter()),
                            train_counts,
                        )
                    )

            semantic_actual: list[str] = []
            semantic_shuffled: dict[int, list[str]] = {
                int(value): [] for value in shuffle_seeds
            }
            for row in records:
                candidates = [label_to_id[str(value)] for value in row["candidates"]]
                embedding_index = dev_user_to_embedding[str(row["user_id"])]
                scores = semantic_model.global_scores[candidates] + (
                    semantic_model.semantic_weight
                    * dev_residual[embedding_index, candidates]
                )
                semantic_actual.append(labels[candidates[int(np.argmax(scores))]])
                user_index = user_to_index[str(row["user_id"])]
                for shuffle_seed, donors in donor_maps.items():
                    donor_user = users[int(donors[user_index])]
                    donor_embedding_index = dev_user_to_embedding[donor_user]
                    donor_scores = semantic_model.global_scores[candidates] + (
                        semantic_model.semantic_weight
                        * dev_residual[donor_embedding_index, candidates]
                    )
                    semantic_shuffled[shuffle_seed].append(
                        labels[candidates[int(np.argmax(donor_scores))]]
                    )
            predictions["semantic_actual"] = semantic_actual
            for shuffle_seed, values in semantic_shuffled.items():
                predictions[f"semantic_shuffled_{shuffle_seed}"] = values

            strategy_outputs: dict[str, object] = {}
            for condition, values in predictions.items():
                metrics = _evaluate_predictions(records, values, train_counts)
                strategy_outputs[condition] = {
                    "metrics_summary": {
                        key: metrics[key]
                        for key in (
                            "samples",
                            "category_accuracy",
                            "macro_category_accuracy",
                            "macro_category_count",
                            "candidate_valid_rate",
                            "prefix_compliance_rate",
                        )
                    },
                    **_write_condition(
                        staging,
                        strategy,
                        condition,
                        records,
                        values,
                        metrics,
                    ),
                }
            pairs = [
                ("last_interest_actual", "global_popularity"),
                ("semantic_actual", "global_popularity"),
                ("semantic_actual", "last_interest_actual"),
                *[
                    (
                        "last_interest_actual",
                        f"last_interest_shuffled_{int(value)}",
                    )
                    for value in shuffle_seeds
                ],
                *[
                    ("semantic_actual", f"semantic_shuffled_{int(value)}")
                    for value in shuffle_seeds
                ],
            ]
            bootstrap = _bootstrap_comparisons(
                records,
                predictions,
                pairs,
                replicates=bootstrap_replicates,
                seed=seed + (0 if strategy == "uniform" else 1_000_000),
            )
            (staging / strategy / "bootstrap_comparisons.json").write_text(
                json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            all_outputs[strategy] = strategy_outputs
            all_bootstrap[strategy] = f"{strategy}/bootstrap_comparisons.json"

        pq.write_table(
            pa.Table.from_pylist(shuffle_manifest_rows),
            staging / "shuffle_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
            row_group_size=8_192,
        )
        resolved = {
            "rule_version": RULE_VERSION,
            "dataset_dir": portable_path(dataset_dir),
            "candidate_dir": portable_path(candidate_dir),
            "embedding_dir": portable_path(embedding_dir),
            "output_dir": portable_path(output_dir),
            "seed": seed,
            "shuffle_seeds": [int(value) for value in shuffle_seeds],
            "bootstrap_replicates": bootstrap_replicates,
            "ridge_alphas": [float(value) for value in ridge_alphas],
            "semantic_weights": [float(value) for value in semantic_weights],
            "semantic_supervision": "Static Train selected target only",
            "random_expected_macro_accuracy": 0.1,
        }
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "rule_version": RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_users": all_stats.count,
            "train_supported_categories": len(labels),
            "semantic_tuning": "semantic_tuning.json",
            "semantic_model": "semantic_model.npz",
            "shuffle_manifest": "shuffle_manifest.parquet",
            "strategies": all_outputs,
            "bootstrap": all_bootstrap,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts = sorted(
            path.relative_to(staging)
            for path in staging.rglob("*")
            if path.is_file()
        )
        manifest = {
            "rule_version": RULE_VERSION,
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "artifact_sha256": {
                str(path): _sha256(staging / path) for path in artifacts
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Train-only baselines on static K=10 candidates."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--ridge-alphas",
        type=_float_grid,
        default=DEFAULT_RIDGE_ALPHAS,
        help="Comma-separated positive Ridge penalties.",
    )
    parser.add_argument(
        "--semantic-weights",
        type=_float_grid,
        default=DEFAULT_SEMANTIC_WEIGHTS,
        help="Comma-separated non-negative semantic score weights.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = run_static_candidate_baselines(
        args.dataset_dir,
        args.candidate_dir,
        args.embedding_dir,
        args.output_dir,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        ridge_alphas=args.ridge_alphas,
        semantic_weights=args.semantic_weights,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
