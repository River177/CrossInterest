from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from new_interest.experiments.evaluate_static_candidate_baselines import (
    _embedding_batches,
    _internal_role,
)
from new_interest.experiments.tune_semantic_history_ridge import (
    Evaluation,
    MultiTargetSplit,
    SingleTargetSplit,
    WeightedStatistics,
    _empty_statistics,
    _evaluate_global,
    _evaluate_model,
    _fit_ridge,
    _global_scores,
    _load_config,
    _load_dev_split,
    _normalised_class_weights,
    _pareto_front,
    _publish,
    _sha256,
    _update_statistics,
    _write_condition,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-balanced-semantic-linear-ranker-v1"
DEFAULT_CONFIG = Path(
    "configs/future_interest/"
    "static_user_disjoint_balanced_semantic_linear_ranker_v1.json"
)


@dataclass(frozen=True)
class LinearTrial:
    class_balance_beta: float
    l2: float
    temperature: float
    learning_rate: float


@dataclass(frozen=True)
class LoadedTrain:
    labels: tuple[str, ...]
    splits: Mapping[str, SingleTargetSplit]
    target_counts: np.ndarray
    role_counts: Mapping[str, int]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Semantic History Linear Ranker requires PyTorch at runtime"
        ) from error
    return torch


def _resolve_device(requested: str) -> str:
    torch = _torch()
    if requested != "auto":
        return str(torch.device(requested))
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_grid(name: str, values: Sequence[float], *, positive: bool) -> tuple[float, ...]:
    output = tuple(float(value) for value in values)
    if not output or not all(math.isfinite(value) for value in output):
        raise ValueError(f"{name} must be a non-empty finite grid")
    if positive and any(value <= 0.0 for value in output):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and any(value < 0.0 for value in output):
        raise ValueError(f"{name} must contain only non-negative values")
    if len(set(output)) != len(output):
        raise ValueError(f"{name} contains duplicates")
    return output


def _load_labels_and_contexts(
    train_path: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[int, tuple[int, ...]]], np.ndarray]:
    raw_rows: list[tuple[str, str, tuple[str, ...]]] = []
    label_counts: dict[str, int] = {}
    seen_users: set[str] = set()
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["user_id", "history", "target_interest"],
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in seen_users:
                raise ValueError(f"duplicate Static Train user {user_id}")
            seen_users.add(user_id)
            target = str(row["target_interest"]["subcategory"])
            history = tuple(str(item["subcategory"]) for item in row["history"])
            if not history:
                raise ValueError(f"{user_id}: empty History")
            if target in history:
                raise ValueError(f"{user_id}: target occurs in History")
            raw_rows.append((user_id, target, history))
            label_counts[target] = label_counts.get(target, 0) + 1
    if not raw_rows:
        raise ValueError(f"no Train rows in {train_path}")
    labels = tuple(sorted(label_counts))
    label_to_id = {label: index for index, label in enumerate(labels)}
    contexts: dict[str, tuple[int, tuple[int, ...]]] = {}
    for user_id, target, history in raw_rows:
        contexts[user_id] = (
            label_to_id[target],
            tuple(sorted({label_to_id[value] for value in history if value in label_to_id})),
        )
    counts = np.asarray([label_counts[label] for label in labels], dtype=np.int64)
    return labels, contexts, counts


def _load_train(
    train_path: Path,
    embedding_paths: Sequence[Path],
    *,
    seed: int,
) -> LoadedTrain:
    labels, contexts, counts = _load_labels_and_contexts(train_path)
    retained_users: dict[str, list[str]] = {
        "model_fit": [],
        "tune": [],
        "holdout": [],
    }
    retained_x: dict[str, list[np.ndarray]] = {key: [] for key in retained_users}
    retained_y: dict[str, list[np.ndarray]] = {key: [] for key in retained_users}
    retained_history: dict[str, list[np.ndarray]] = {
        key: [] for key in retained_users
    }
    seen: set[str] = set()
    dimensions: int | None = None
    for path in embedding_paths:
        for users, matrix in _embedding_batches(path):
            if dimensions is None:
                dimensions = int(matrix.shape[1])
            elif matrix.shape[1] != dimensions:
                raise ValueError("embedding dimensions changed")
            targets = np.empty(len(users), dtype=np.int64)
            history_mask = np.zeros((len(users), len(labels)), dtype=np.bool_)
            roles: list[str] = []
            for row_index, user_id in enumerate(users):
                if user_id in seen:
                    raise ValueError(f"duplicate Train embedding user {user_id}")
                seen.add(user_id)
                if user_id not in contexts:
                    raise ValueError(f"unknown Train embedding user {user_id}")
                target, history_ids = contexts[user_id]
                targets[row_index] = target
                history_mask[row_index, list(history_ids)] = True
                if history_mask[row_index, target]:
                    raise ValueError(f"{user_id}: target is masked as History")
                roles.append(_internal_role(user_id, seed))
            for role in retained_users:
                role_mask = np.asarray([value == role for value in roles])
                if np.any(role_mask):
                    retained_users[role].extend(
                        user_id
                        for user_id, keep in zip(users, role_mask, strict=True)
                        if keep
                    )
                    retained_x[role].append(matrix[role_mask].copy())
                    retained_y[role].append(targets[role_mask].copy())
                    retained_history[role].append(history_mask[role_mask].copy())
    if seen != set(contexts):
        missing = sorted(set(contexts) - seen)[:5]
        extra = sorted(seen - set(contexts))[:5]
        raise ValueError(f"Train embedding users differ; missing={missing}, extra={extra}")
    splits: dict[str, SingleTargetSplit] = {}
    for role in retained_users:
        splits[role] = SingleTargetSplit(
            tuple(retained_users[role]),
            np.ascontiguousarray(np.concatenate(retained_x[role], axis=0)),
            np.concatenate(retained_y[role], axis=0),
            np.concatenate(retained_history[role], axis=0),
        )
    return LoadedTrain(
        labels=labels,
        splits=splits,
        target_counts=counts,
        role_counts={role: len(split.user_ids) for role, split in splits.items()},
    )


def _merge_splits(splits: Sequence[SingleTargetSplit]) -> SingleTargetSplit:
    return SingleTargetSplit(
        tuple(user for split in splits for user in split.user_ids),
        np.ascontiguousarray(
            np.concatenate([split.embeddings for split in splits], axis=0)
        ),
        np.concatenate([split.targets for split in splits], axis=0),
        np.concatenate([split.history_mask for split in splits], axis=0),
    )


def _single_target_metrics(
    split: SingleTargetSplit,
    weights: np.ndarray,
    *,
    k: int,
) -> dict[str, float]:
    scores = np.asarray(split.embeddings @ weights, dtype=np.float32)
    scores[split.history_mask] = -np.inf
    candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    local_order = np.lexsort((candidates, -candidate_scores), axis=1)
    order = np.take_along_axis(candidates, local_order, axis=1)
    # np.argpartition does not define which labels survive an exact tie at the
    # K boundary.  Fall back to the canonical stable full ordering only for
    # rows where that ambiguity actually occurs.
    threshold = np.take_along_axis(scores, order[:, -1:], axis=1)[:, 0]
    greater = (scores > threshold[:, None]).sum(axis=1)
    tied = (scores == threshold[:, None]).sum(axis=1)
    ambiguous = tied > (k - greater)
    if np.any(ambiguous):
        order[ambiguous] = np.argsort(
            -scores[ambiguous], axis=1, kind="stable"
        )[:, :k]
    matches = order == split.targets[:, None]
    hit = matches.any(axis=1)
    ranks = np.argmax(matches, axis=1)
    discounts = np.zeros(len(split.user_ids), dtype=np.float64)
    discounts[hit] = 1.0 / np.log2(ranks[hit].astype(np.float64) + 2.0)
    support = np.bincount(split.targets, minlength=weights.shape[1])
    retrieved = np.bincount(split.targets[hit], minlength=weights.shape[1])
    gains = np.bincount(
        split.targets[hit], weights=discounts[hit], minlength=weights.shape[1]
    )
    present = support > 0
    recall = retrieved[present] / support[present]
    ndcg = gains[present] / support[present]
    return {
        f"interest_macro_recall_at_{k}": float(recall.mean()),
        f"interest_macro_ndcg_at_{k}": float(ndcg.mean()),
        f"user_macro_recall_at_{k}": float(hit.mean()),
        f"user_macro_ndcg_at_{k}": float(discounts.mean()),
    }


def _masked_weighted_cross_entropy(
    logits: Any,
    targets: Any,
    history_mask: Any,
    sample_weights: Any,
    *,
    validate: bool = True,
) -> Any:
    torch = _torch()
    if logits.ndim != 2 or history_mask.shape != logits.shape:
        raise ValueError("logits and History mask must be aligned matrices")
    if targets.ndim != 1 or sample_weights.ndim != 1:
        raise ValueError("targets and sample weights must be vectors")
    if len(targets) != len(logits) or len(sample_weights) != len(logits):
        raise ValueError("batch dimensions differ")
    if validate and bool(
        history_mask.gather(1, targets[:, None]).any().detach().cpu()
    ):
        raise ValueError("a target is excluded by its History mask")
    masked_logits = logits.masked_fill(history_mask, -torch.inf)
    losses = torch.nn.functional.cross_entropy(
        masked_logits, targets, reduction="none"
    )
    denominator = sample_weights.sum()
    if validate and float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("sample weights must have positive sum")
    return (losses * sample_weights).sum() / denominator


def _fit_linear_trial(
    fit: SingleTargetSplit,
    tune: SingleTargetSplit | None,
    trial: LinearTrial,
    class_weights: np.ndarray,
    *,
    epochs: int,
    evaluation_epochs: Sequence[int],
    batch_size: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
    device_tensors: tuple[Any, Any, Any] | None = None,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray], list[dict[str, float]]]:
    torch = _torch()
    _seed_everything(seed)
    current_device = torch.device(device)
    owns_device_tensors = device_tensors is None
    if device_tensors is None:
        features = torch.as_tensor(
            fit.embeddings, dtype=torch.float32, device=current_device
        )
        targets = torch.as_tensor(
            fit.targets, dtype=torch.long, device=current_device
        )
        history = torch.as_tensor(
            fit.history_mask, dtype=torch.bool, device=current_device
        )
    else:
        features, targets, history = device_tensors
    per_class = torch.as_tensor(class_weights, dtype=torch.float32, device=current_device)
    model = torch.nn.Linear(fit.embeddings.shape[1], len(class_weights), bias=False)
    model.to(current_device)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial.learning_rate)
    checkpoints: dict[int, np.ndarray] = {}
    records: list[dict[str, object]] = []
    history_rows: list[dict[str, float]] = []
    evaluation_set = set(int(value) for value in evaluation_epochs)
    for epoch in range(1, epochs + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        permutation = torch.randperm(
            len(fit.user_ids), generator=generator
        ).to(current_device)
        ce_total = torch.zeros((), dtype=torch.float32, device=current_device)
        sample_weight_total = torch.zeros(
            (), dtype=torch.float32, device=current_device
        )
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_targets = targets[indices]
            batch_weights = per_class[batch_targets]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indices]) / trial.temperature
            ce = _masked_weighted_cross_entropy(
                logits,
                batch_targets,
                history[indices],
                batch_weights,
                validate=False,
            )
            l2_penalty = 0.5 * trial.l2 * model.weight.square().sum()
            loss = ce + l2_penalty
            loss.backward()
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_weight_sum = batch_weights.sum().detach()
            ce_total += ce.detach() * batch_weight_sum
            sample_weight_total += batch_weight_sum
        history_rows.append(
            {
                "epoch": float(epoch),
                "weighted_cross_entropy": float(
                    (ce_total / sample_weight_total).detach().cpu()
                ),
                "weight_l2_norm": float(model.weight.norm().detach().cpu()),
            }
        )
        if epoch not in evaluation_set:
            continue
        raw_weights = np.asarray(model.weight.detach().cpu(), dtype=np.float32).T.copy()
        checkpoints[epoch] = raw_weights
        if tune is not None:
            metrics = _single_target_metrics(
                tune, raw_weights / trial.temperature, k=10
            )
            records.append(
                {
                    "class_balance_beta": trial.class_balance_beta,
                    "l2": trial.l2,
                    "temperature": trial.temperature,
                    "learning_rate": trial.learning_rate,
                    "epoch": epoch,
                    **metrics,
                }
            )
    del per_class, model, optimizer
    if owns_device_tensors:
        del features, targets, history
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
    return records, checkpoints, history_rows


def _select_configuration(
    records: Sequence[Mapping[str, object]],
    *,
    global_recall: float,
    balanced_ridge_user_recall: float,
    recall_tolerance: float,
    user_recall_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not records:
        raise ValueError("cannot select from an empty Linear Ranker grid")
    if recall_tolerance < 0.0 or user_recall_tolerance < 0.0:
        raise ValueError("selection tolerances must be non-negative")
    user_floor = balanced_ridge_user_recall - user_recall_tolerance
    user_eligible = [
        row
        for row in records
        if float(row["user_macro_recall_at_10"]) >= user_floor
    ]
    if not user_eligible:
        raise ValueError(
            "no Pareto Linear Ranker satisfies the Interest-Recall and "
            "User-Recall constraints"
        )
    frontier = _pareto_front(user_eligible)
    best_recall = max(
        float(row["interest_macro_recall_at_10"]) for row in user_eligible
    )
    recall_floor = max(global_recall, best_recall - recall_tolerance)
    eligible = [
        row
        for row in frontier
        if float(row["interest_macro_recall_at_10"]) >= recall_floor
    ]
    if not eligible:
        raise ValueError(
            "no Pareto Linear Ranker satisfies the Interest-Recall and "
            "User-Recall constraints"
        )
    selected = max(
        eligible,
        key=lambda row: (
            float(row["interest_macro_ndcg_at_10"]),
            float(row["interest_macro_recall_at_10"]),
            float(row["user_macro_recall_at_10"]),
            -float(row["class_balance_beta"]),
            -float(row["l2"]),
            -abs(math.log2(float(row["temperature"]))),
            -float(row["learning_rate"]),
            -int(row["epoch"]),
        ),
    )
    return dict(selected), frontier


def _statistics(split: SingleTargetSplit, beta: float, label_count: int) -> WeightedStatistics:
    counts = np.bincount(split.targets, minlength=label_count)
    class_weights = _normalised_class_weights(counts, beta)
    stats = _empty_statistics(split.embeddings.shape[1], label_count)
    for start in range(0, len(split.user_ids), 8_192):
        end = start + 8_192
        _update_statistics(
            stats,
            split.embeddings[start:end],
            split.targets[start:end],
            class_weights,
        )
    return stats


def _ridge_evaluation(
    train: SingleTargetSplit,
    evaluation: SingleTargetSplit | MultiTargetSplit,
    labels: Sequence[str],
    *,
    beta: float,
    alpha: float,
    k: int,
) -> Evaluation:
    mean_x, weights = _fit_ridge(_statistics(train, beta, len(labels)), (alpha,))
    return _evaluate_model(evaluation, mean_x, weights[alpha], labels, k=k)


def _trial_key(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["class_balance_beta"]),
        float(row["l2"]),
        float(row["temperature"]),
        float(row["learning_rate"]),
    )


def _input_sha256(paths: Sequence[Path]) -> dict[str, str]:
    return {portable_path(path): _sha256(path) for path in paths}


def run_balanced_semantic_history_linear_ranker(
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
    if k != 10:
        raise ValueError("the v1 selector is defined for k=10")
    training = dict(config["training"])
    betas = _validate_grid(
        "class_balance_betas", training["class_balance_betas"], positive=False
    )
    if any(value > 1.0 for value in betas):
        raise ValueError("class_balance_betas must be in [0, 1]")
    l2_values = _validate_grid("l2_values", training["l2_values"], positive=False)
    temperatures = _validate_grid(
        "temperatures", training["temperatures"], positive=True
    )
    learning_rates = _validate_grid(
        "learning_rates", training["learning_rates"], positive=True
    )
    epochs = int(training["epochs"])
    evaluation_epochs = tuple(int(value) for value in training["evaluation_epochs"])
    batch_size = int(training["batch_size"])
    gradient_clip_norm = float(training["gradient_clip_norm"])
    if epochs <= 0 or batch_size <= 0 or gradient_clip_norm < 0.0:
        raise ValueError("epochs/batch_size must be positive and gradient clipping non-negative")
    if not evaluation_epochs or min(evaluation_epochs) <= 0 or max(evaluation_epochs) > epochs:
        raise ValueError("evaluation_epochs must fall within the training epoch range")
    selection = dict(config["selection"])
    recall_tolerance = float(selection["best_recall_tolerance_absolute"])
    user_recall_tolerance = float(selection["user_recall_tolerance_absolute"])
    requested_device = str(training.get("device", "auto"))
    device = _resolve_device(requested_device)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    train_path = dataset_dir / "train.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    dev_benchmark_path = benchmark_dir / "dev.parquet"
    embedding_paths = (
        embedding_dir / "fit.parquet",
        embedding_dir / "validation.parquet",
    )
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

    loaded = _load_train(train_path, embedding_paths, seed=seed)
    labels = loaded.labels
    vocabulary_labels = {
        str(value)
        for batch in pq.ParquetFile(vocabulary_path).iter_batches(
            columns=["subcategory"], batch_size=256
        )
        for value in batch["subcategory"].to_pylist()
    }
    if not set(labels).issubset(vocabulary_labels):
        raise ValueError("Static Train contains interests outside the vocabulary")
    unsupported_vocabulary_labels = sorted(vocabulary_labels - set(labels))
    fit = loaded.splits["model_fit"]
    tune = loaded.splits["tune"]
    holdout = loaded.splits["holdout"]
    fit_counts = np.bincount(fit.targets, minlength=len(labels))
    tune_global = _evaluate_global(tune, _global_scores(fit_counts), labels, k=k)
    balanced_ridge_tune = _ridge_evaluation(
        fit, tune, labels, beta=0.5, alpha=10.0, k=k
    )

    grid: list[dict[str, object]] = []
    checkpoints: dict[tuple[tuple[float, float, float, float], int], np.ndarray] = {}
    training_histories: dict[str, list[dict[str, float]]] = {}
    trials = [
        LinearTrial(beta, l2, temperature, learning_rate)
        for beta in betas
        for l2 in l2_values
        for temperature in temperatures
        for learning_rate in learning_rates
    ]
    torch = _torch()
    current_device = torch.device(device)
    fit_device_tensors = (
        torch.as_tensor(fit.embeddings, dtype=torch.float32, device=current_device),
        torch.as_tensor(fit.targets, dtype=torch.long, device=current_device),
        torch.as_tensor(fit.history_mask, dtype=torch.bool, device=current_device),
    )
    for trial_index, trial in enumerate(trials):
        class_weights = _normalised_class_weights(fit_counts, trial.class_balance_beta)
        records, trial_checkpoints, history_rows = _fit_linear_trial(
            fit,
            tune,
            trial,
            class_weights,
            epochs=epochs,
            evaluation_epochs=evaluation_epochs,
            batch_size=batch_size,
            gradient_clip_norm=gradient_clip_norm,
            seed=seed,
            device=device,
            device_tensors=fit_device_tensors,
        )
        grid.extend(records)
        key = _trial_key(records[0])
        for epoch, weights in trial_checkpoints.items():
            checkpoints[(key, epoch)] = weights
        training_histories[f"trial_{trial_index:04d}"] = history_rows
        if (trial_index + 1) % 10 == 0 or trial_index + 1 == len(trials):
            print(
                f"linear-ranker grid: {trial_index + 1}/{len(trials)} trials",
                flush=True,
            )

    del fit_device_tensors
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()

    selected, frontier = _select_configuration(
        grid,
        global_recall=float(tune_global.metrics[f"interest_macro_recall_at_{k}"]),
        balanced_ridge_user_recall=float(
            balanced_ridge_tune.metrics[f"user_macro_recall_at_{k}"]
        ),
        recall_tolerance=recall_tolerance,
        user_recall_tolerance=user_recall_tolerance,
    )
    selected_trial = LinearTrial(*_trial_key(selected))
    selected_epoch = int(selected["epoch"])
    internal_raw_weights = checkpoints[(_trial_key(selected), selected_epoch)]
    internal_weights = internal_raw_weights / selected_trial.temperature
    zero_mean = np.zeros(fit.embeddings.shape[1], dtype=np.float32)
    selected_holdout = _evaluate_model(
        holdout, zero_mean, internal_weights, labels, k=k
    )
    holdout_global = _evaluate_global(
        holdout, _global_scores(fit_counts), labels, k=k
    )
    current_ridge_holdout = _ridge_evaluation(
        fit, holdout, labels, beta=0.0, alpha=10.0, k=k
    )
    balanced_ridge_holdout = _ridge_evaluation(
        fit, holdout, labels, beta=0.5, alpha=10.0, k=k
    )

    all_train = _merge_splits((fit, tune, holdout))
    all_counts = np.bincount(all_train.targets, minlength=len(labels))
    all_device_tensors = (
        torch.as_tensor(
            all_train.embeddings, dtype=torch.float32, device=current_device
        ),
        torch.as_tensor(all_train.targets, dtype=torch.long, device=current_device),
        torch.as_tensor(
            all_train.history_mask, dtype=torch.bool, device=current_device
        ),
    )
    final_records, final_checkpoints, final_history = _fit_linear_trial(
        all_train,
        None,
        selected_trial,
        _normalised_class_weights(all_counts, selected_trial.class_balance_beta),
        epochs=selected_epoch,
        evaluation_epochs=(selected_epoch,),
        batch_size=batch_size,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
        device_tensors=all_device_tensors,
    )
    del all_device_tensors
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()
    if final_records:
        raise AssertionError("final training must not evaluate on a tune split")
    final_raw_weights = final_checkpoints[selected_epoch]
    final_weights = final_raw_weights / selected_trial.temperature
    dev = _load_dev_split(dev_benchmark_path, dev_embedding_path, labels)
    selected_dev = _evaluate_model(dev, zero_mean, final_weights, labels, k=k)
    dev_global = _evaluate_global(dev, _global_scores(all_counts), labels, k=k)
    current_ridge_dev = _ridge_evaluation(
        all_train, dev, labels, beta=0.0, alpha=10.0, k=k
    )
    balanced_ridge_dev = _ridge_evaluation(
        all_train, dev, labels, beta=0.5, alpha=10.0, k=k
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".balanced-semantic-linear-ranker-",
        dir=str(output_dir.parent),
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        np.savez_compressed(
            staging / "selected_model.npz",
            labels=np.asarray(labels),
            raw_linear_weights=final_raw_weights,
            inference_linear_weights=final_weights,
            class_balance_beta=np.asarray([selected_trial.class_balance_beta]),
            l2=np.asarray([selected_trial.l2]),
            temperature=np.asarray([selected_trial.temperature]),
            learning_rate=np.asarray([selected_trial.learning_rate]),
            epochs=np.asarray([selected_epoch]),
        )
        tuning = {
            "selection_scope": "Static Train internal tune only",
            "model_fit_users": loaded.role_counts["model_fit"],
            "tune_users": loaded.role_counts["tune"],
            "holdout_users": loaded.role_counts["holdout"],
            "trial_count": len(trials),
            "evaluated_checkpoint_count": len(grid),
            "tune_global": tune_global.metrics,
            "tune_balanced_ridge_alpha10_beta05": balanced_ridge_tune.metrics,
            "selected": selected,
            "pareto_frontier": frontier,
            "grid": grid,
            "training_histories": training_histories,
            "final_training_history": final_history,
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
                    holdout, current_ridge_holdout, labels
                ),
                "balanced_ridge_alpha10_beta05": _write_condition(
                    staging, "internal_holdout", "balanced_ridge_alpha10_beta05",
                    holdout, balanced_ridge_holdout, labels
                ),
                "selected_linear_ranker": _write_condition(
                    staging, "internal_holdout", "selected_linear_ranker",
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
                    dev, current_ridge_dev, labels
                ),
                "balanced_ridge_alpha10_beta05": _write_condition(
                    staging, "dev_exploratory", "balanced_ridge_alpha10_beta05",
                    dev, balanced_ridge_dev, labels
                ),
                "selected_linear_ranker": _write_condition(
                    staging, "dev_exploratory", "selected_linear_ranker", dev,
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
            "resolved_device": device,
            "supervision": "Static Train selected target only",
            "embedding_status": "frozen; no re-encoding",
            "score": "s(u,y) = x_u^T W_y / temperature; no intercept or Global prior",
            "loss": (
                "class-weighted full-vocabulary softmax cross entropy plus "
                "0.5 * l2 * ||W||_F^2"
            ),
            "training_candidate_mask": "exclude all interests in raw user History",
            "candidate_space": (
                "124 Static-Train-supported interests minus each user's raw "
                "History interests"
            ),
            "vocabulary_interests_without_train_support": unsupported_vocabulary_labels,
        }
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "selected": selected,
            "conditions": conditions,
            "internal_holdout": {
                "global_popularity": holdout_global.metrics,
                "current_ridge_alpha10_beta0": current_ridge_holdout.metrics,
                "balanced_ridge_alpha10_beta05": balanced_ridge_holdout.metrics,
                "selected_linear_ranker": selected_holdout.metrics,
            },
            "dev_exploratory": {
                "global_popularity": dev_global.metrics,
                "current_ridge_alpha10_beta0": current_ridge_dev.metrics,
                "balanced_ridge_alpha10_beta05": balanced_ridge_dev.metrics,
                "selected_linear_ranker": selected_dev.metrics,
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
            "source_sha256": _input_sha256(source_paths),
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
            "Tune a class-balanced full-vocabulary Semantic History Linear "
            "Ranker with Train-only Pareto selection."
        )
    )
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = run_balanced_semantic_history_linear_ranker(
        args.config, overwrite=args.overwrite
    )
    print(output)


if __name__ == "__main__":
    main()
