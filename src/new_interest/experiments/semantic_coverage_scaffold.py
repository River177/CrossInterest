from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.semantic_history_signal import (
    EvaluationResult,
    SplitData,
    _evaluate_scores,
    _fit_linear_residual,
    _label_slices,
    _load_split,
    _sha256,
    _shuffle_donors,
    _topic_texts,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-semantic-coverage-interest-macro-v1"
DEFAULT_CONFIG = Path(
    "configs/future_interest/"
    "static_user_disjoint_semantic_coverage_interest_macro_v1.json"
)


@dataclass(frozen=True)
class ScaffoldParameters:
    global_pool_size: int
    semantic_expansion: int
    semantic_weight: float


@dataclass(frozen=True)
class RankingContext:
    global_order: np.ndarray
    global_rank: np.ndarray
    semantic_order: np.ndarray
    residual: np.ndarray


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


def _split_priority(user_id: str, *, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:semantic-coverage-internal-split:{user_id}".encode()
    ).hexdigest()


def _internal_partition(
    data: SplitData,
    *,
    seed: int,
    modulus: int,
    model_fit_end: int,
    tune_end: int,
) -> tuple[dict[str, SplitData], list[dict[str, object]]]:
    if not 0 < model_fit_end < tune_end < modulus:
        raise ValueError("internal split thresholds must satisfy 0 < fit < tune < modulus")
    roles: dict[str, list[int]] = {
        "model_fit": [],
        "scaffold_tune": [],
        "internal_holdout": [],
    }
    rows: list[dict[str, object]] = []
    for index, user_id in enumerate(data.user_ids):
        priority = _split_priority(user_id, seed=seed)
        bucket = int(priority[:16], 16) % modulus
        if bucket < model_fit_end:
            role = "model_fit"
        elif bucket < tune_end:
            role = "scaffold_tune"
        else:
            role = "internal_holdout"
        roles[role].append(index)
        rows.append(
            {
                "user_id": user_id,
                "role": role,
                "bucket": bucket,
                "priority": priority,
            }
        )
    partitions = {
        role: _subset_split(data, np.asarray(indices, dtype=np.int64))
        for role, indices in roles.items()
    }
    if any(not split.user_ids for split in partitions.values()):
        raise ValueError("internal hash split produced an empty partition")
    return partitions, rows


def _subset_split(data: SplitData, indices: np.ndarray) -> SplitData:
    history_mask = (
        None if data.history_mask is None else data.history_mask[indices].copy()
    )
    return SplitData(
        user_ids=tuple(data.user_ids[int(index)] for index in indices),
        embeddings=np.ascontiguousarray(data.embeddings[indices]),
        positives=data.positives[indices].tocsr(),
        history_mask=history_mask,
        history_length=data.history_length[indices].copy(),
        history_topic_count=data.history_topic_count[indices].copy(),
    )


def _concatenate_splits(parts: Sequence[SplitData]) -> SplitData:
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("semantic coverage experiment requires scipy") from error
    if not parts:
        raise ValueError("cannot concatenate an empty split sequence")
    masks = [part.history_mask for part in parts]
    if any(mask is None for mask in masks) and not all(mask is None for mask in masks):
        raise ValueError("cannot concatenate mixed History-mask availability")
    return SplitData(
        user_ids=tuple(user_id for part in parts for user_id in part.user_ids),
        embeddings=np.ascontiguousarray(
            np.concatenate([part.embeddings for part in parts], axis=0)
        ),
        positives=sparse.vstack([part.positives for part in parts], format="csr"),
        history_mask=(
            None
            if all(mask is None for mask in masks)
            else np.concatenate(masks, axis=0)
        ),
        history_length=np.concatenate(
            [part.history_length for part in parts], axis=0
        ),
        history_topic_count=np.concatenate(
            [part.history_topic_count for part in parts], axis=0
        ),
    )


def _explicit_metric_names(
    result: EvaluationResult,
    data: SplitData,
    labels: tuple[str, ...],
    *,
    cutoffs: Sequence[int],
) -> EvaluationResult:
    metrics = dict(result.metrics)
    per_interest = [
        {
            "interest_id": index,
            "interest": label,
            "support": int(data.positives[:, index].sum()),
        }
        for index, label in enumerate(labels)
    ]
    for cutoff in cutoffs:
        metrics[f"user_macro_recall_at_{cutoff}"] = float(
            metrics[f"recall_at_{cutoff}"]
        )
        metrics[f"user_macro_ndcg_at_{cutoff}"] = float(
            metrics[f"ndcg_at_{cutoff}"]
        )
        support, recall, ndcg = _interest_metrics_at_k(
            result.rankings,
            data,
            cutoff=cutoff,
        )
        supported = support > 0
        metrics[f"interest_macro_recall_at_{cutoff}"] = float(
            recall[supported].mean()
        )
        metrics[f"interest_macro_ndcg_at_{cutoff}"] = float(
            ndcg[supported].mean()
        )
        for index, row in enumerate(per_interest):
            row[f"recall_at_{cutoff}"] = (
                float(recall[index]) if supported[index] else None
            )
            row[f"ndcg_at_{cutoff}"] = (
                float(ndcg[index]) if supported[index] else None
            )
    metrics["supported_interest_count"] = sum(
        int(row["support"]) > 0 for row in per_interest
    )
    metrics["per_interest"] = per_interest
    metrics["metric_definitions"] = {
        "interest_macro_recall_at_k": (
            "For each supported interest, average its Top-K hit indicator over "
            "users for whom it is a positive; then average interests equally."
        ),
        "interest_macro_ndcg_at_k": (
            "Treat each supported interest as a separate single-relevant-item "
            "task. Average 1/log2(rank+1), or zero beyond K, over its positive "
            "users; then average interests equally."
        ),
        "user_macro_recall_at_k_secondary_diagnostic": (
            "For each user, retrieved distinct positives divided by that user's "
            "distinct positives; then arithmetic mean over users."
        ),
        "user_macro_ndcg_at_k_secondary_diagnostic": (
            "Binary NDCG over each user's distinct positives with user-specific "
            "IDCG; then arithmetic mean over users."
        ),
        "legacy_aliases": {
            "recall_at_k": "user_macro_recall_at_k",
            "ndcg_at_k": "user_macro_ndcg_at_k",
            "macro_recall_at_k": "interest_macro_recall_at_k",
        },
    }
    return EvaluationResult(
        metrics=metrics,
        rankings=result.rankings,
        ranked_scores=result.ranked_scores,
        per_user=result.per_user,
    )


def _ranking_context(
    global_scores: np.ndarray,
    residual: np.ndarray,
    data: SplitData,
) -> RankingContext:
    if data.history_mask is None:
        raise ValueError("ranking requires the actual History mask")
    expected = (len(data.user_ids), len(global_scores))
    if residual.shape != expected:
        raise ValueError("semantic residual shape does not match evaluation split")
    global_matrix = np.broadcast_to(global_scores, expected).copy()
    global_matrix[data.history_mask] = -np.inf
    semantic_matrix = np.array(residual, dtype=np.float32, copy=True)
    semantic_matrix[data.history_mask] = -np.inf
    global_order = np.argsort(-global_matrix, axis=1, kind="stable")
    semantic_order = np.argsort(-semantic_matrix, axis=1, kind="stable")
    global_rank = np.empty_like(global_order)
    row_indices = np.arange(len(data.user_ids), dtype=np.int64)[:, None]
    global_rank[row_indices, global_order] = np.arange(
        len(global_scores), dtype=global_order.dtype
    )
    return RankingContext(
        global_order=global_order,
        global_rank=global_rank,
        semantic_order=semantic_order,
        residual=residual,
    )


def _scaffold_priority(
    global_scores: np.ndarray,
    context: RankingContext,
    data: SplitData,
    parameters: ScaffoldParameters,
) -> np.ndarray:
    if data.history_mask is None:
        raise ValueError("ranking requires the actual History mask")
    users, labels = context.global_order.shape
    if not 1 <= parameters.global_pool_size <= labels:
        raise ValueError("global_pool_size is outside the label space")
    if not 0 <= parameters.semantic_expansion <= labels:
        raise ValueError("semantic_expansion is outside the label space")
    rows = np.arange(users, dtype=np.int64)[:, None]
    candidate = np.zeros((users, labels), dtype=np.bool_)
    candidate[
        rows,
        context.global_order[:, : parameters.global_pool_size],
    ] = True
    if parameters.semantic_expansion:
        candidate[
            rows,
            context.semantic_order[:, : parameters.semantic_expansion],
        ] = True
    fused = global_scores + parameters.semantic_weight * context.residual
    fused = np.array(fused, dtype=np.float32, copy=True)
    fused[data.history_mask] = -np.inf
    fused_order = np.argsort(-fused, axis=1, kind="stable")
    fused_rank = np.empty_like(fused_order)
    fused_rank[rows, fused_order] = np.arange(labels, dtype=fused_order.dtype)
    # All scaffold candidates precede the fallback. Candidates are ordered by
    # fused Global+History score; non-candidates retain their Global order.
    return np.where(candidate, fused_rank, labels + context.global_rank)


def _top_k_from_priority(priority: np.ndarray, cutoff: int) -> np.ndarray:
    if not 0 < cutoff <= priority.shape[1]:
        raise ValueError("cutoff is outside the ranking width")
    partial = np.argpartition(priority, cutoff - 1, axis=1)[:, :cutoff]
    partial_priority = np.take_along_axis(priority, partial, axis=1)
    order = np.argsort(partial_priority, axis=1, kind="stable")
    return np.take_along_axis(partial, order, axis=1)


def _primary_metrics(
    rankings: np.ndarray,
    data: SplitData,
    *,
    cutoff: int,
) -> tuple[float, float]:
    support, recall, ndcg = _interest_metrics_at_k(
        rankings,
        data,
        cutoff=cutoff,
    )
    supported = support > 0
    return float(recall[supported].mean()), float(ndcg[supported].mean())


def _interest_metrics_at_k(
    rankings: np.ndarray,
    data: SplitData,
    *,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return support, Recall@K and single-item NDCG@K per interest."""
    effective_cutoff = min(cutoff, rankings.shape[1])
    positive = data.positives.toarray().astype(np.bool_, copy=False)
    selected = rankings[:, :effective_cutoff]
    hits = np.take_along_axis(positive, selected, axis=1)
    support = np.asarray(data.positives.sum(axis=0)).ravel().astype(np.int64)
    retrieved_labels = selected[hits]
    retrieved = np.bincount(retrieved_labels, minlength=positive.shape[1])
    discounts = 1.0 / np.log2(
        np.arange(effective_cutoff, dtype=np.float64) + 2.0
    )
    discounted = np.broadcast_to(discounts, hits.shape)[hits]
    gains = np.bincount(
        retrieved_labels,
        weights=discounted,
        minlength=positive.shape[1],
    )
    recall = np.zeros(positive.shape[1], dtype=np.float64)
    ndcg = np.zeros(positive.shape[1], dtype=np.float64)
    supported = support > 0
    recall[supported] = retrieved[supported] / support[supported]
    # Each interest is evaluated as a separate single-relevant-item task, so
    # IDCG is one and its NDCG contribution is exactly the rank discount.
    ndcg[supported] = gains[supported] / support[supported]
    return support, recall, ndcg


def _scaffold_result(
    global_scores: np.ndarray,
    context: RankingContext,
    data: SplitData,
    labels: tuple[str, ...],
    parameters: ScaffoldParameters,
    *,
    cutoffs: tuple[int, ...],
    label_slices: Mapping[str, Sequence[int]],
) -> EvaluationResult:
    priority = _scaffold_priority(global_scores, context, data, parameters)
    result = _evaluate_scores(
        -priority.astype(np.float32),
        data,
        labels,
        cutoffs=cutoffs,
        label_slices=label_slices,
    )
    return _explicit_metric_names(
        result,
        data,
        labels,
        cutoffs=cutoffs,
    )


def _global_result(
    global_scores: np.ndarray,
    data: SplitData,
    labels: tuple[str, ...],
    *,
    cutoffs: tuple[int, ...],
    label_slices: Mapping[str, Sequence[int]],
) -> EvaluationResult:
    return _explicit_metric_names(
        _evaluate_scores(
            global_scores,
            data,
            labels,
            cutoffs=cutoffs,
            label_slices=label_slices,
        ),
        data,
        labels,
        cutoffs=cutoffs,
    )


def _tune_scaffold(
    global_scores: np.ndarray,
    residual: np.ndarray,
    data: SplitData,
    *,
    primary_cutoff: int,
    global_pool_sizes: tuple[int, ...],
    semantic_expansions: tuple[int, ...],
    semantic_weights: tuple[float, ...],
    minimum_recall_delta: float,
) -> tuple[ScaffoldParameters, dict[str, object]]:
    context = _ranking_context(global_scores, residual, data)
    baseline_ranking = context.global_order[:, :primary_cutoff]
    baseline_recall, baseline_ndcg = _primary_metrics(
        baseline_ranking, data, cutoff=primary_cutoff
    )
    records: list[dict[str, object]] = []
    eligible: list[tuple[ScaffoldParameters, float, float]] = []
    for semantic_weight in semantic_weights:
        for global_pool_size in global_pool_sizes:
            for semantic_expansion in semantic_expansions:
                parameters = ScaffoldParameters(
                    global_pool_size=global_pool_size,
                    semantic_expansion=semantic_expansion,
                    semantic_weight=semantic_weight,
                )
                priority = _scaffold_priority(
                    global_scores, context, data, parameters
                )
                rankings = _top_k_from_priority(priority, primary_cutoff)
                recall, ndcg = _primary_metrics(
                    rankings, data, cutoff=primary_cutoff
                )
                passes = recall >= baseline_recall + minimum_recall_delta - 1e-12
                records.append(
                    {
                        "global_pool_size": global_pool_size,
                        "semantic_expansion": semantic_expansion,
                        "semantic_weight": semantic_weight,
                        f"interest_macro_recall_at_{primary_cutoff}": recall,
                        f"interest_macro_ndcg_at_{primary_cutoff}": ndcg,
                        "passes_recall_constraint": passes,
                    }
                )
                if passes:
                    eligible.append((parameters, recall, ndcg))
    if not eligible:
        raise RuntimeError("no scaffold configuration passed the tune recall constraint")
    selected, selected_recall, selected_ndcg = max(
        eligible,
        key=lambda item: (
            item[2],
            item[1],
            -item[0].semantic_expansion,
            -item[0].global_pool_size,
            -item[0].semantic_weight,
        ),
    )
    return selected, {
        "selection_unit": "interest",
        "selection_rule": (
            f"maximize interest_macro_ndcg_at_{primary_cutoff} subject to "
            f"interest_macro_recall_at_{primary_cutoff} >= Global + "
            f"{minimum_recall_delta}"
        ),
        "global_baseline": {
            f"interest_macro_recall_at_{primary_cutoff}": baseline_recall,
            f"interest_macro_ndcg_at_{primary_cutoff}": baseline_ndcg,
        },
        "selected": {
            "global_pool_size": selected.global_pool_size,
            "semantic_expansion": selected.semantic_expansion,
            "semantic_weight": selected.semantic_weight,
            f"interest_macro_recall_at_{primary_cutoff}": selected_recall,
            f"interest_macro_ndcg_at_{primary_cutoff}": selected_ndcg,
        },
        "grid": records,
    }


def _paired_user_cluster_bootstrap(
    baseline: EvaluationResult,
    comparison: EvaluationResult,
    data: SplitData,
    *,
    cutoffs: tuple[int, ...],
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    """Resample users, then re-aggregate every interest inside each replicate."""
    if replicates < 2_000:
        raise ValueError("bootstrap replicates must be at least 2000")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("interest-macro bootstrap requires scipy") from error

    user_metric_names = tuple(
        name
        for cutoff in cutoffs
        for name in (
            f"user_macro_recall_at_{cutoff}",
            f"user_macro_ndcg_at_{cutoff}",
        )
    )
    interest_metric_names = tuple(
        name
        for cutoff in cutoffs
        for name in (
            f"interest_macro_recall_at_{cutoff}",
            f"interest_macro_ndcg_at_{cutoff}",
        )
    )
    all_metric_names = user_metric_names + interest_metric_names
    samples: dict[str, list[float]] = {
        name: [] for name in all_metric_names
    }
    user_deltas = {
        f"user_macro_recall_at_{cutoff}": (
            comparison.per_user[f"recall_at_{cutoff}"]
            - baseline.per_user[f"recall_at_{cutoff}"]
        )
        for cutoff in cutoffs
    }
    user_deltas.update(
        {
            f"user_macro_ndcg_at_{cutoff}": (
                comparison.per_user[f"ndcg_at_{cutoff}"]
                - baseline.per_user[f"ndcg_at_{cutoff}"]
            )
            for cutoff in cutoffs
        }
    )

    positive = data.positives.astype(np.float64).tocsr()
    positive.sum_duplicates()
    positive.data[:] = 1.0
    positive_coo = positive.tocoo()
    positive_rows = positive_coo.row.astype(np.int64, copy=False)
    positive_columns = positive_coo.col.astype(np.int64, copy=False)
    max_rank = baseline.rankings.shape[1]
    if comparison.rankings.shape != baseline.rankings.shape:
        raise ValueError("paired bootstrap rankings must have identical shapes")

    def positive_ranks(result: EvaluationResult) -> np.ndarray:
        lookup = np.full(
            (len(data.user_ids), positive.shape[1]),
            max_rank + 1,
            dtype=np.int16,
        )
        rows = np.arange(len(data.user_ids), dtype=np.int64)[:, None]
        lookup[rows, result.rankings] = np.arange(
            1,
            max_rank + 1,
            dtype=np.int16,
        )
        return lookup[positive_rows, positive_columns]

    baseline_ranks = positive_ranks(baseline)
    comparison_ranks = positive_ranks(comparison)
    interest_delta_matrices: dict[str, object] = {}
    for cutoff in cutoffs:
        recall_delta = (
            (comparison_ranks <= cutoff).astype(np.float64)
            - (baseline_ranks <= cutoff).astype(np.float64)
        )
        baseline_gain = np.where(
            baseline_ranks <= cutoff,
            1.0 / np.log2(baseline_ranks.astype(np.float64) + 1.0),
            0.0,
        )
        comparison_gain = np.where(
            comparison_ranks <= cutoff,
            1.0 / np.log2(comparison_ranks.astype(np.float64) + 1.0),
            0.0,
        )
        for name, values in (
            (f"interest_macro_recall_at_{cutoff}", recall_delta),
            (f"interest_macro_ndcg_at_{cutoff}", comparison_gain - baseline_gain),
        ):
            matrix = sparse.csr_matrix(
                (values, (positive_rows, positive_columns)),
                shape=positive.shape,
            )
            matrix.eliminate_zeros()
            interest_delta_matrices[name] = matrix

    rng = np.random.default_rng(seed)
    sample_count = len(data.user_ids)
    completed = 0
    chunk_size = min(32, replicates)
    while completed < replicates:
        current_chunk = min(chunk_size, replicates - completed)
        indices = rng.integers(
            0,
            sample_count,
            size=(current_chunk, sample_count),
            dtype=np.int32,
        )
        for name, values in user_deltas.items():
            samples[name].extend(values[indices].mean(axis=1).tolist())
        counts = np.stack(
            [
                np.bincount(row, minlength=sample_count)
                for row in indices
            ]
        ).astype(np.float64, copy=False)
        support = np.asarray((positive.T @ counts.T).T)
        supported = support > 0.0
        supported_count = supported.sum(axis=1)
        for name, matrix in interest_delta_matrices.items():
            numerator = np.asarray((matrix.T @ counts.T).T)
            per_interest_delta = np.divide(
                numerator,
                support,
                out=np.zeros_like(numerator),
                where=supported,
            )
            macro_delta = per_interest_delta.sum(axis=1) / supported_count
            samples[name].extend(macro_delta.tolist())
        completed += current_chunk

    alpha = (1.0 - confidence) / 2.0
    observed = {
        name: float(comparison.metrics[name]) - float(baseline.metrics[name])
        for name in all_metric_names
    }
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "resampling_unit": "user",
        "aggregation": (
            "Within each user bootstrap replicate, recompute per-interest "
            "support and per-interest metrics, then average supported interests."
        ),
        "metrics": {
            name: {
                "observed_delta": observed[name],
                "bootstrap_mean": float(np.mean(samples[name])),
                "lower": float(np.quantile(samples[name], alpha)),
                "upper": float(np.quantile(samples[name], 1.0 - alpha)),
            }
            for name in all_metric_names
        },
    }


def _comparisons(
    global_result: EvaluationResult,
    actual_result: EvaluationResult,
    shuffled_results: Mapping[int, EvaluationResult],
    data: SplitData,
    *,
    cutoffs: tuple[int, ...],
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    output = {
        "actual_vs_global": _paired_user_cluster_bootstrap(
            global_result,
            actual_result,
            data,
            cutoffs=cutoffs,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
        )
    }
    for shuffle_seed, shuffled in shuffled_results.items():
        output[f"actual_vs_shuffled_{shuffle_seed}"] = (
            _paired_user_cluster_bootstrap(
                shuffled,
                actual_result,
                data,
                cutoffs=cutoffs,
                replicates=replicates,
                seed=seed + shuffle_seed,
                confidence=confidence,
            )
        )
    return output


def _gate(
    actual: EvaluationResult,
    global_result: EvaluationResult,
    shuffled: Mapping[int, EvaluationResult],
    comparisons: Mapping[str, object],
    *,
    primary_cutoff: int,
) -> dict[str, object]:
    required = (
        f"interest_macro_recall_at_{primary_cutoff}",
        f"interest_macro_ndcg_at_{primary_cutoff}",
    )
    global_bootstrap = comparisons["actual_vs_global"]["metrics"]
    global_checks = {
        metric: (
            float(actual.metrics[metric]) > float(global_result.metrics[metric])
            and float(global_bootstrap[metric]["lower"]) > 0.0
        )
        for metric in required
    }
    shuffle_checks: dict[str, dict[str, bool]] = {}
    for shuffle_seed, result in shuffled.items():
        bootstrap = comparisons[f"actual_vs_shuffled_{shuffle_seed}"]["metrics"]
        shuffle_checks[str(shuffle_seed)] = {
            metric: (
                float(actual.metrics[metric]) > float(result.metrics[metric])
                and float(bootstrap[metric]["lower"]) > 0.0
            )
            for metric in required
        }
    actual_tail = actual.metrics["label_slices"]["tail"][str(primary_cutoff)][
        "micro_recall"
    ]
    global_tail = global_result.metrics["label_slices"]["tail"][
        str(primary_cutoff)
    ]["micro_recall"]
    tail_ok = (
        actual_tail is not None
        and global_tail is not None
        and float(actual_tail) >= float(global_tail)
    )
    passed = (
        all(global_checks.values())
        and all(all(seed_checks.values()) for seed_checks in shuffle_checks.values())
    )
    return {
        "passed": passed,
        "primary_metrics": list(required),
        "actual_beats_global_with_positive_paired_ci": global_checks,
        "actual_beats_each_shuffle_with_positive_paired_ci": shuffle_checks,
        "secondary_tail_micro_recall_does_not_collapse": tail_ok,
    }


def _recall_safe_gate(
    actual: EvaluationResult,
    global_result: EvaluationResult,
    shuffled: Mapping[int, EvaluationResult],
    comparisons: Mapping[str, object],
    *,
    primary_cutoff: int,
) -> dict[str, object]:
    recall_metric = f"interest_macro_recall_at_{primary_cutoff}"
    ndcg_metric = f"interest_macro_ndcg_at_{primary_cutoff}"
    user_recall_is_invariant = bool(
        np.array_equal(
            actual.per_user[f"recall_at_{primary_cutoff}"],
            global_result.per_user[f"recall_at_{primary_cutoff}"],
        )
    )
    interest_recall_is_invariant = (
        float(actual.metrics[recall_metric])
        == float(global_result.metrics[recall_metric])
    )
    global_bootstrap = comparisons["actual_vs_global"]["metrics"]
    ndcg_beats_global = (
        float(actual.metrics[ndcg_metric]) > float(global_result.metrics[ndcg_metric])
        and float(global_bootstrap[ndcg_metric]["lower"]) > 0.0
    )
    shuffle_checks = {}
    for shuffle_seed, result in shuffled.items():
        bootstrap = comparisons[f"actual_vs_shuffled_{shuffle_seed}"]["metrics"]
        shuffle_checks[str(shuffle_seed)] = (
            float(actual.metrics[ndcg_metric]) > float(result.metrics[ndcg_metric])
            and float(bootstrap[ndcg_metric]["lower"]) > 0.0
        )
    passed = (
        user_recall_is_invariant
        and interest_recall_is_invariant
        and ndcg_beats_global
        and all(shuffle_checks.values())
    )
    return {
        "passed": passed,
        "guaranteed_metric": recall_metric,
        "interest_macro_recall_is_exactly_invariant": (
            interest_recall_is_invariant
        ),
        "secondary_user_recall_is_exactly_invariant_per_user": (
            user_recall_is_invariant
        ),
        "ndcg_beats_global_with_positive_paired_ci": ndcg_beats_global,
        "ndcg_beats_each_shuffle_with_positive_paired_ci": shuffle_checks,
    }


def _write_condition(
    root: Path,
    split_name: str,
    condition_name: str,
    result: EvaluationResult,
    data: SplitData,
    labels: tuple[str, ...],
) -> dict[str, str]:
    output = root / "conditions" / split_name / condition_name
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    details = []
    for index, user_id in enumerate(data.user_ids):
        predictions.append(
            {
                "user_id": user_id,
                "ranked_labels": [
                    labels[int(value)] for value in result.rankings[index]
                ],
                "ranking_priority_score": result.ranked_scores[index]
                .astype(float)
                .tolist(),
            }
        )
        row: dict[str, object] = {
            "user_id": user_id,
            "target_count": int(data.target_count[index]),
            "history_length": int(data.history_length[index]),
        }
        for metric, values in result.per_user.items():
            row[metric] = float(values[index])
            if metric.startswith("recall_at_"):
                row[metric.replace("recall_at_", "user_macro_recall_at_")] = float(
                    values[index]
                )
            elif metric.startswith("ndcg_at_"):
                row[metric.replace("ndcg_at_", "user_macro_ndcg_at_")] = float(
                    values[index]
                )
        details.append(row)
    pq.write_table(
        pa.Table.from_pylist(predictions),
        output / "top20_predictions.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )
    pq.write_table(
        pa.Table.from_pylist(details),
        output / "details.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )
    pq.write_table(
        pa.Table.from_pylist(list(result.metrics["per_interest"])),
        output / "per_interest_metrics.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prefix = f"conditions/{split_name}/{condition_name}"
    return {
        "metrics": f"{prefix}/metrics.json",
        "per_interest_metrics": f"{prefix}/per_interest_metrics.parquet",
        "details": f"{prefix}/details.parquet",
        "predictions": f"{prefix}/top20_predictions.parquet",
    }


def _evaluate_split(
    data: SplitData,
    labels: tuple[str, ...],
    global_scores: np.ndarray,
    mean_embedding: np.ndarray,
    ridge_weights: np.ndarray,
    parameters: ScaffoldParameters,
    *,
    cutoffs: tuple[int, ...],
    label_slices: Mapping[str, Sequence[int]],
    shuffle_seeds: tuple[int, ...],
) -> tuple[
    EvaluationResult,
    EvaluationResult,
    dict[int, EvaluationResult],
    list[dict[str, object]],
]:
    actual_residual = (data.embeddings - mean_embedding) @ ridge_weights
    actual_context = _ranking_context(global_scores, actual_residual, data)
    global_result = _global_result(
        global_scores,
        data,
        labels,
        cutoffs=cutoffs,
        label_slices=label_slices,
    )
    actual_result = _scaffold_result(
        global_scores,
        actual_context,
        data,
        labels,
        parameters,
        cutoffs=cutoffs,
        label_slices=label_slices,
    )
    shuffled_results: dict[int, EvaluationResult] = {}
    shuffle_rows: list[dict[str, object]] = []
    for shuffle_seed in shuffle_seeds:
        donors, buckets, match_levels = _shuffle_donors_without_fixed_points(
            data, seed=shuffle_seed
        )
        residual = (data.embeddings[donors] - mean_embedding) @ ridge_weights
        context = _ranking_context(global_scores, residual, data)
        shuffled_results[shuffle_seed] = _scaffold_result(
            global_scores,
            context,
            data,
            labels,
            parameters,
            cutoffs=cutoffs,
            label_slices=label_slices,
        )
        for index, donor in enumerate(donors):
            shuffle_rows.append(
                {
                    "seed": shuffle_seed,
                    "user_id": data.user_ids[index],
                    "donor_user_id": data.user_ids[int(donor)],
                    "bucket": buckets[index],
                    "donor_bucket": buckets[int(donor)],
                    "match_level": match_levels[index],
                    "is_fixed_point": bool(index == int(donor)),
                }
            )
    return global_result, actual_result, shuffled_results, shuffle_rows


def _shuffle_donors_without_fixed_points(
    data: SplitData, *, seed: int
) -> tuple[np.ndarray, list[str], list[str]]:
    donors, buckets = _shuffle_donors(data, seed=seed)
    match_levels = ["exact_bucket"] * len(data.user_ids)
    fixed = np.flatnonzero(donors == np.arange(len(donors), dtype=np.int64))
    if fixed.size > 1:
        ordered = np.asarray(
            sorted(
                fixed.tolist(),
                key=lambda index: (buckets[index], data.user_ids[index]),
            ),
            dtype=np.int64,
        )
        digest = hashlib.sha256(
            f"{seed}:singleton-relaxed-derangement".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        shift = int(rng.integers(1, len(ordered)))
        donors[ordered] = np.roll(ordered, shift)
        for index in ordered:
            match_levels[int(index)] = "relaxed_singleton_derangement"
    elif fixed.size == 1:
        index = int(fixed[0])
        candidates = np.asarray(
            [value for value in range(len(donors)) if value != index],
            dtype=np.int64,
        )
        if not candidates.size:
            raise ValueError("cannot shuffle a one-user evaluation split")
        length_distance = np.abs(
            data.history_length[candidates].astype(np.int64)
            - int(data.history_length[index])
        )
        topic_distance = np.abs(
            data.history_topic_count[candidates].astype(np.int64)
            - int(data.history_topic_count[index])
        )
        distance = length_distance + topic_distance
        nearest = candidates[distance == distance.min()]
        chosen = min(
            nearest.tolist(),
            key=lambda value: hashlib.sha256(
                f"{seed}:{data.user_ids[index]}:{data.user_ids[value]}".encode()
            ).hexdigest(),
        )
        donors[index] = int(chosen)
        match_levels[index] = "relaxed_singleton_nearest_donor"
    if np.any(donors == np.arange(len(donors), dtype=np.int64)):
        raise AssertionError("shuffle donor mapping contains fixed points")
    return donors, buckets, match_levels


def _save_model(
    path: Path,
    labels: tuple[str, ...],
    global_scores: np.ndarray,
    mean_embedding: np.ndarray,
    ridge_weights: np.ndarray,
    *,
    ridge_alpha: float,
) -> None:
    np.savez_compressed(
        path,
        labels=np.asarray(labels),
        global_scores=global_scores,
        fit_embedding_mean=mean_embedding,
        ridge_weights=ridge_weights,
        ridge_alpha=np.asarray([ridge_alpha]),
    )


def run_semantic_coverage_scaffold_experiment(
    config_path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    config, config_path = _load_config(config_path_or_config)
    if str(config.get("rule_version")) != RULE_VERSION:
        raise ValueError("config rule_version does not match implementation")
    benchmark_dir = Path(str(config["benchmark_dir"]))
    semantic_artifact_dir = Path(str(config["semantic_artifact_dir"]))
    artifact_dir = Path(str(config["artifact_dir"]))
    if artifact_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Artifact directory already exists: {artifact_dir}. Use --overwrite."
        )
    backup = artifact_dir.with_name(artifact_dir.name + ".old")
    if artifact_dir.exists() and overwrite and backup.exists():
        raise FileExistsError(f"Cannot overwrite while backup exists: {backup}")

    vocabulary_path = benchmark_dir / "interest_vocabulary.parquet"
    benchmark_paths = {
        split: benchmark_dir / f"{split}.parquet"
        for split in ("fit", "validation", "dev")
    }
    embedding_paths = {
        split: semantic_artifact_dir / "embeddings" / f"{split}.parquet"
        for split in benchmark_paths
    }
    source_paths = [vocabulary_path, *benchmark_paths.values(), *embedding_paths.values()]
    for name in ("metadata.json", "statistics.json"):
        path = benchmark_dir / name
        if path.is_file():
            source_paths.append(path)
    previous_manifest = semantic_artifact_dir / "manifest.json"
    if previous_manifest.is_file():
        source_paths.append(previous_manifest)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    cutoffs = tuple(sorted({int(value) for value in config["cutoffs"]}))
    scaffold_config = dict(config["scaffold"])
    primary_cutoff = int(scaffold_config["primary_cutoff"])
    if primary_cutoff not in cutoffs:
        raise ValueError("primary_cutoff must be present in cutoffs")
    global_pool_sizes = tuple(
        int(value) for value in scaffold_config["global_pool_size_grid"]
    )
    semantic_expansions = tuple(
        int(value) for value in scaffold_config["semantic_expansion_grid"]
    )
    semantic_weights = tuple(
        float(value) for value in scaffold_config["lambda_grid"]
    )
    if any(value < primary_cutoff for value in global_pool_sizes):
        raise ValueError("every global pool size must cover the primary cutoff")
    minimum_recall_delta = float(
        scaffold_config["minimum_tune_recall_delta"]
    )
    model_config = dict(config["model"])
    prior_alpha = float(model_config["prior_alpha"])
    ridge_alphas = tuple(
        float(value) for value in model_config["ridge_alpha_grid"]
    )
    if not ridge_alphas or any(value <= 0.0 for value in ridge_alphas):
        raise ValueError("ridge_alpha_grid must contain positive values")
    split_config = dict(config["internal_split"])
    shuffle_seeds = tuple(int(value) for value in config["shuffle"]["seeds"])
    if len(shuffle_seeds) < 3:
        raise ValueError("at least three matched-shuffle seeds are required")
    bootstrap_config = dict(config["bootstrap"])
    replicates = int(bootstrap_config["replicates"])
    bootstrap_seed = int(bootstrap_config["seed"])
    confidence = float(bootstrap_config["confidence"])

    labels, _ = _topic_texts(vocabulary_path)
    fit_source = _load_split(
        benchmark_paths["fit"],
        embedding_paths["fit"],
        labels,
        include_history_mask=True,
    )
    partitions, split_rows = _internal_partition(
        fit_source,
        seed=int(split_config["seed"]),
        modulus=int(split_config["modulus"]),
        model_fit_end=int(split_config["model_fit_end"]),
        tune_end=int(split_config["tune_end"]),
    )
    model_fit = partitions["model_fit"]
    tune = partitions["scaffold_tune"]
    holdout = partitions["internal_holdout"]
    internal_global, internal_mean, internal_weights_by_alpha, _ = _fit_linear_residual(
        model_fit,
        label_count=len(labels),
        prior_alpha=prior_alpha,
        ridge_alphas=ridge_alphas,
    )
    ridge_tuning: list[dict[str, object]] = []
    selected_candidate: tuple[
        float, ScaffoldParameters, dict[str, object]
    ] | None = None
    for candidate_alpha in ridge_alphas:
        candidate_residual = (
            tune.embeddings - internal_mean
        ) @ internal_weights_by_alpha[candidate_alpha]
        candidate_parameters, candidate_tuning = _tune_scaffold(
            internal_global,
            candidate_residual,
            tune,
            primary_cutoff=primary_cutoff,
            global_pool_sizes=global_pool_sizes,
            semantic_expansions=semantic_expansions,
            semantic_weights=semantic_weights,
            minimum_recall_delta=minimum_recall_delta,
        )
        candidate_record = {
            "ridge_alpha": candidate_alpha,
            **candidate_tuning,
        }
        ridge_tuning.append(candidate_record)
        candidate = (
            candidate_alpha,
            candidate_parameters,
            candidate_record,
        )
        if selected_candidate is None:
            selected_candidate = candidate
            continue
        current_metrics = candidate_record["selected"]
        selected_metrics = selected_candidate[2]["selected"]
        current_key = (
            float(current_metrics[f"interest_macro_ndcg_at_{primary_cutoff}"]),
            float(current_metrics[f"interest_macro_recall_at_{primary_cutoff}"]),
            -candidate_parameters.semantic_expansion,
            -candidate_parameters.global_pool_size,
            -candidate_parameters.semantic_weight,
            -candidate_alpha,
        )
        selected_parameters = selected_candidate[1]
        selected_key = (
            float(selected_metrics[f"interest_macro_ndcg_at_{primary_cutoff}"]),
            float(selected_metrics[f"interest_macro_recall_at_{primary_cutoff}"]),
            -selected_parameters.semantic_expansion,
            -selected_parameters.global_pool_size,
            -selected_parameters.semantic_weight,
            -selected_candidate[0],
        )
        if current_key > selected_key:
            selected_candidate = candidate
    assert selected_candidate is not None
    ridge_alpha, selected, selected_tuning = selected_candidate
    internal_weights = internal_weights_by_alpha[ridge_alpha]
    safe_row = max(
        (
            row
            for row in selected_tuning["grid"]
            if int(row["global_pool_size"]) == primary_cutoff
            and int(row["semantic_expansion"]) == 0
        ),
        key=lambda row: (
            float(row[f"interest_macro_ndcg_at_{primary_cutoff}"]),
            -float(row["semantic_weight"]),
        ),
    )
    recall_safe = ScaffoldParameters(
        global_pool_size=primary_cutoff,
        semantic_expansion=0,
        semantic_weight=float(safe_row["semantic_weight"]),
    )
    tuning = {
        "selection_unit": "interest",
        "selection_scope": "ridge alpha and scaffold parameters jointly",
        "ridge_grid": ridge_tuning,
        "selected_ridge_alpha": ridge_alpha,
        "selected": selected_tuning["selected"],
        "global_baseline": selected_tuning["global_baseline"],
        "selection_rule": selected_tuning["selection_rule"],
        "recall_safe_ablation": {
            "constraint": (
                f"global_pool_size={primary_cutoff}, semantic_expansion=0; "
                f"therefore every user's Top-{primary_cutoff} set equals Global"
            ),
            **safe_row,
        },
    }
    internal_slices = _label_slices(
        np.asarray(model_fit.positives.sum(axis=0)).ravel(), labels
    )
    (
        holdout_global,
        holdout_actual,
        holdout_shuffled,
        holdout_shuffle_rows,
    ) = _evaluate_split(
        holdout,
        labels,
        internal_global,
        internal_mean,
        internal_weights,
        selected,
        cutoffs=cutoffs,
        label_slices=internal_slices,
        shuffle_seeds=shuffle_seeds,
    )
    holdout_comparisons = _comparisons(
        holdout_global,
        holdout_actual,
        holdout_shuffled,
        holdout,
        cutoffs=cutoffs,
        replicates=replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    holdout_gate = _gate(
        holdout_actual,
        holdout_global,
        holdout_shuffled,
        holdout_comparisons,
        primary_cutoff=primary_cutoff,
    )
    (
        _,
        holdout_safe,
        holdout_safe_shuffled,
        _,
    ) = _evaluate_split(
        holdout,
        labels,
        internal_global,
        internal_mean,
        internal_weights,
        recall_safe,
        cutoffs=cutoffs,
        label_slices=internal_slices,
        shuffle_seeds=shuffle_seeds,
    )
    holdout_safe_comparisons = _comparisons(
        holdout_global,
        holdout_safe,
        holdout_safe_shuffled,
        holdout,
        cutoffs=cutoffs,
        replicates=replicates,
        seed=bootstrap_seed + 2_000_000,
        confidence=confidence,
    )
    holdout_safe_gate = _recall_safe_gate(
        holdout_safe,
        holdout_global,
        holdout_safe_shuffled,
        holdout_safe_comparisons,
        primary_cutoff=primary_cutoff,
    )

    # Hyperparameters are now frozen. Refit only the linear semantic component
    # on all Train users before the exploratory user-disjoint Dev evaluation.
    validation_fit = _load_split(
        benchmark_paths["validation"],
        embedding_paths["validation"],
        labels,
        include_history_mask=True,
    )
    full_train = _concatenate_splits((fit_source, validation_fit))
    deploy_global, deploy_mean, deploy_weights_by_alpha, _ = _fit_linear_residual(
        full_train,
        label_count=len(labels),
        prior_alpha=prior_alpha,
        ridge_alphas=(ridge_alpha,),
    )
    deploy_weights = deploy_weights_by_alpha[ridge_alpha]
    dev = _load_split(
        benchmark_paths["dev"],
        embedding_paths["dev"],
        labels,
        include_history_mask=True,
    )
    dev_slices = _label_slices(
        np.asarray(full_train.positives.sum(axis=0)).ravel(), labels
    )
    dev_global, dev_actual, dev_shuffled, dev_shuffle_rows = _evaluate_split(
        dev,
        labels,
        deploy_global,
        deploy_mean,
        deploy_weights,
        selected,
        cutoffs=cutoffs,
        label_slices=dev_slices,
        shuffle_seeds=shuffle_seeds,
    )
    dev_comparisons = _comparisons(
        dev_global,
        dev_actual,
        dev_shuffled,
        dev,
        cutoffs=cutoffs,
        replicates=replicates,
        seed=bootstrap_seed + 1_000_000,
        confidence=confidence,
    )
    dev_gate = _gate(
        dev_actual,
        dev_global,
        dev_shuffled,
        dev_comparisons,
        primary_cutoff=primary_cutoff,
    )
    _, dev_safe, dev_safe_shuffled, _ = _evaluate_split(
        dev,
        labels,
        deploy_global,
        deploy_mean,
        deploy_weights,
        recall_safe,
        cutoffs=cutoffs,
        label_slices=dev_slices,
        shuffle_seeds=shuffle_seeds,
    )
    dev_safe_comparisons = _comparisons(
        dev_global,
        dev_safe,
        dev_safe_shuffled,
        dev,
        cutoffs=cutoffs,
        replicates=replicates,
        seed=bootstrap_seed + 3_000_000,
        confidence=confidence,
    )
    dev_safe_gate = _recall_safe_gate(
        dev_safe,
        dev_global,
        dev_safe_shuffled,
        dev_safe_comparisons,
        primary_cutoff=primary_cutoff,
    )

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".semantic-coverage-scaffold-",
        dir=str(artifact_dir.parent),
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        condition_paths: dict[str, dict[str, str]] = {}
        for split_name, data, global_result, actual, shuffled, safe, safe_shuffled in (
            (
                "internal_holdout",
                holdout,
                holdout_global,
                holdout_actual,
                holdout_shuffled,
                holdout_safe,
                holdout_safe_shuffled,
            ),
            (
                "dev_exploratory",
                dev,
                dev_global,
                dev_actual,
                dev_shuffled,
                dev_safe,
                dev_safe_shuffled,
            ),
        ):
            paths = {
                "global": _write_condition(
                    staging, split_name, "global", global_result, data, labels
                ),
                "semantic_coverage_actual": _write_condition(
                    staging,
                    split_name,
                    "semantic_coverage_actual",
                    actual,
                    data,
                    labels,
                ),
                "recall_safe_rerank_actual": _write_condition(
                    staging,
                    split_name,
                    "recall_safe_rerank_actual",
                    safe,
                    data,
                    labels,
                ),
            }
            for shuffle_seed, result in shuffled.items():
                paths[f"semantic_coverage_shuffled_{shuffle_seed}"] = (
                    _write_condition(
                        staging,
                        split_name,
                        f"semantic_coverage_shuffled_{shuffle_seed}",
                        result,
                        data,
                        labels,
                    )
                )
            for shuffle_seed, result in safe_shuffled.items():
                paths[f"recall_safe_rerank_shuffled_{shuffle_seed}"] = (
                    _write_condition(
                        staging,
                        split_name,
                        f"recall_safe_rerank_shuffled_{shuffle_seed}",
                        result,
                        data,
                        labels,
                    )
                )
            condition_paths[split_name] = paths
        pq.write_table(
            pa.Table.from_pylist(split_rows),
            staging / "internal_split_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
            row_group_size=8_192,
        )
        shuffle_rows = [
            {"evaluation_split": "internal_holdout", **row}
            for row in holdout_shuffle_rows
        ] + [
            {"evaluation_split": "dev_exploratory", **row}
            for row in dev_shuffle_rows
        ]
        pq.write_table(
            pa.Table.from_pylist(shuffle_rows),
            staging / "shuffle_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
            row_group_size=8_192,
        )
        _save_model(
            staging / "internal_model.npz",
            labels,
            internal_global,
            internal_mean,
            internal_weights,
            ridge_alpha=ridge_alpha,
        )
        _save_model(
            staging / "deployment_model.npz",
            labels,
            deploy_global,
            deploy_mean,
            deploy_weights,
            ridge_alpha=ridge_alpha,
        )
        tuning["selected_parameters"] = {
            "ridge_alpha": ridge_alpha,
            "global_pool_size": selected.global_pool_size,
            "semantic_expansion": selected.semantic_expansion,
            "semantic_weight": selected.semantic_weight,
        }
        tuning["recall_safe_parameters"] = {
            "ridge_alpha": ridge_alpha,
            "global_pool_size": recall_safe.global_pool_size,
            "semantic_expansion": recall_safe.semantic_expansion,
            "semantic_weight": recall_safe.semantic_weight,
        }
        (staging / "tuning.json").write_text(
            json.dumps(tuning, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        comparisons = {
            "internal_holdout": {
                "semantic_coverage": holdout_comparisons,
                "recall_safe_rerank": holdout_safe_comparisons,
            },
            "dev_exploratory": {
                "semantic_coverage": dev_comparisons,
                "recall_safe_rerank": dev_safe_comparisons,
            },
        }
        (staging / "comparisons.json").write_text(
            json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        acceptance = {
            "primary_evidence_split": "internal_holdout",
            "internal_holdout": holdout_gate,
            "dev_exploratory": dev_gate,
            "recall_safe_rerank": {
                "internal_holdout": holdout_safe_gate,
                "dev_exploratory": dev_safe_gate,
            },
            "passed": bool(holdout_gate["passed"]),
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
        implementation_paths = {
            Path(__file__).resolve(),
            Path(_evaluate_scores.__code__.co_filename).resolve(),
            Path(_paired_user_cluster_bootstrap.__code__.co_filename).resolve(),
        }
        artifact_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "rule_version": RULE_VERSION,
            "benchmark_dir": portable_path(benchmark_dir),
            "semantic_artifact_dir": portable_path(semantic_artifact_dir),
            "artifact_dir": portable_path(artifact_dir),
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "implementation_sha256": {
                portable_path(path): _sha256(path)
                for path in sorted(implementation_paths)
            },
            "artifact_sha256": artifact_hashes,
            "users": {
                "model_fit": len(model_fit.user_ids),
                "scaffold_tune": len(tune.user_ids),
                "internal_holdout": len(holdout.user_ids),
                "deployment_train": len(full_train.user_ids),
                "dev_exploratory": len(dev.user_ids),
            },
            "selected_parameters": tuning["selected_parameters"],
            "recall_safe_parameters": tuning["recall_safe_parameters"],
            "primary_metrics": [
                f"interest_macro_recall_at_{primary_cutoff}",
                f"interest_macro_ndcg_at_{primary_cutoff}",
            ],
            "conditions": condition_paths,
            "tuning_path": "tuning.json",
            "comparisons_path": "comparisons.json",
            "acceptance_path": "acceptance.json",
            "internal_split_manifest_path": "internal_split_manifest.parquet",
            "shuffle_manifest_path": "shuffle_manifest.parquet",
            "models": {
                "internal": "internal_model.npz",
                "deployment": "deployment_model.npz",
            },
            "acceptance": acceptance,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
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
    result = run_semantic_coverage_scaffold_experiment(
        args.config, overwrite=args.overwrite
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
