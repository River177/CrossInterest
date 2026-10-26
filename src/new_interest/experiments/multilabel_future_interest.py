from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.experiments.future_interest_prediction import (
    PredictionWriter,
    load_vocabulary,
    rank_label_scores,
)


TRAIN_COLUMNS = (
    "user_id",
    "history_subcategories",
    "future_novel_subcategories",
    "future_exposed_unclicked_subcategories",
)
PREDICTION_COLUMNS = ("user_id", "history_subcategories")
EmbeddingPaths = Path | str | Mapping[str, Path | str]


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The multilabel baseline requires PyTorch at runtime"
        ) from exc
    return torch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(device: str) -> Any:
    torch = _torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _validate_schema(path: Path | str, columns: Sequence[str]) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    missing = set(columns) - set(schema.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _iter_rows(
    path: Path | str,
    columns: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    _validate_schema(path, columns)
    for batch in pq.ParquetFile(path).iter_batches(
        columns=list(columns), batch_size=batch_size
    ):
        yield from batch.to_pylist()


def _embedding_path(
    embedding_paths: EmbeddingPaths | None, split: str
) -> Path | None:
    if embedding_paths is None:
        return None
    if isinstance(embedding_paths, Mapping):
        if split not in embedding_paths:
            raise ValueError(
                f"embedding paths must contain a {split!r} entry"
            )
        return Path(embedding_paths[split])
    return Path(embedding_paths)


def _resolve_fit_embedding_paths(
    embedding_path: EmbeddingPaths | None,
    train_embedding_path: Path | str | None,
    validation_embedding_path: Path | str | None,
) -> EmbeddingPaths | None:
    if train_embedding_path is not None or validation_embedding_path is not None:
        if train_embedding_path is None or validation_embedding_path is None:
            raise ValueError(
                "train_embedding_path and validation_embedding_path "
                "must be supplied together"
            )
        return {
            "train": Path(train_embedding_path),
            "validation": Path(validation_embedding_path),
        }
    if isinstance(embedding_path, Mapping):
        if "train" not in embedding_path or "validation" not in embedding_path:
            raise ValueError(
                "embedding_path mapping must contain train and validation"
            )
        return embedding_path
    return embedding_path


def _resolve_test_embedding_path(
    embedding_path: EmbeddingPaths | None,
    test_embedding_path: Path | str | None,
) -> EmbeddingPaths | None:
    if test_embedding_path is not None:
        return {"benchmark": Path(test_embedding_path)}
    if isinstance(embedding_path, Mapping):
        if "benchmark" not in embedding_path:
            raise ValueError(
                "embedding_path mapping must contain benchmark for prediction"
            )
        return embedding_path
    return embedding_path


def _iter_aligned_rows(
    data_path: Path | str,
    embedding_path: Path | str,
    data_columns: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], tuple[float, ...]]]:
    """Yield data and embeddings in order without indexing the whole file."""
    data_rows = _iter_rows(data_path, data_columns, batch_size=batch_size)
    embedding_rows = _iter_rows(
        embedding_path, ("user_id", "embedding"), batch_size=batch_size
    )
    data_exhausted = embedding_exhausted = False
    while not data_exhausted or not embedding_exhausted:
        try:
            data_row = next(data_rows)
        except StopIteration:
            data_exhausted = True
            data_row = None
        try:
            embedding_row = next(embedding_rows)
        except StopIteration:
            embedding_exhausted = True
            embedding_row = None
        if data_row is None and embedding_row is None:
            break
        if data_row is None or embedding_row is None:
            raise ValueError(
                "embedding coverage differs from benchmark rows"
            )
        data_user_id = str(data_row["user_id"])
        embedding_user_id = str(embedding_row["user_id"])
        if data_user_id != embedding_user_id:
            raise ValueError(
                "embedding user order differs: "
                f"{data_user_id!r} != {embedding_user_id!r}"
            )
        values = tuple(float(value) for value in embedding_row["embedding"])
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(
                f"invalid embedding for user {data_user_id!r}"
            )
        yield data_row, values


def _iter_split_rows(
    path: Path | str,
    columns: Sequence[str],
    *,
    split: str,
    feature_mode: str,
    embedding_paths: EmbeddingPaths | None,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], tuple[float, ...] | None]]:
    embedding_path = _embedding_path(embedding_paths, split)
    if feature_mode == "history_embedding":
        if embedding_path is None:
            raise ValueError(
                f"embedding path is required for {feature_mode}"
            )
        yield from _iter_aligned_rows(
            path, embedding_path, columns, batch_size=batch_size
        )
        return
    yield from ((row, None) for row in _iter_rows(
        path, columns, batch_size=batch_size
    ))


def build_feature_vector(
    row: Mapping[str, Any],
    labels: Sequence[str],
    *,
    feature_mode: str,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    embedding: Sequence[float] | None = None,
) -> tuple[float, ...]:
    if feature_mode == "topic_frequency":
        history = [str(value) for value in row["history_subcategories"]]
        invalid = set(history) - set(labels)
        if invalid:
            raise ValueError(
                f"row {str(row['user_id'])!r}: invalid history labels "
                f"{sorted(invalid)}"
            )
        counts = Counter(history)
        total = float(len(history))
        if total == 0:
            return tuple(0.0 for _ in labels)
        return tuple(counts.get(label, 0) / total for label in labels)
    if feature_mode == "history_embedding":
        if embedding is None:
            if embeddings is None or str(row["user_id"]) not in embeddings:
                raise ValueError(
                    f"missing embedding for user {str(row['user_id'])!r}"
                )
            embedding = embeddings[str(row["user_id"])]
        values = tuple(float(value) for value in embedding)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(
                f"invalid embedding for user {str(row['user_id'])!r}"
            )
        return values
    raise ValueError(f"Unknown feature mode {feature_mode!r}")


def build_masked_targets(
    row: Mapping[str, Any], labels: Sequence[str]
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    label_set = set(labels)
    positives = {str(value) for value in row["future_novel_subcategories"]}
    negatives = {
        str(value) for value in row["future_exposed_unclicked_subcategories"]
    }
    invalid = (positives | negatives) - label_set
    if invalid:
        raise ValueError(
            f"row {row['user_id']!r}: invalid labels {sorted(invalid)}"
        )
    overlap = positives & negatives
    if overlap:
        raise ValueError(
            f"row {row['user_id']!r}: labels are both positive and negative "
            f"{sorted(overlap)}"
        )
    return (
        tuple(1.0 if label in positives else 0.0 for label in labels),
        tuple(label in positives or label in negatives for label in labels),
    )


def masked_bce_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    positive_weight: Any | None = None,
) -> Any:
    torch = _torch()
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have the same shape")
    weights = torch.ones_like(logits)
    if positive_weight is not None:
        weights = torch.where(targets > 0.5, positive_weight, weights)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    denominator = weights.sum()
    if float(denominator.detach().cpu()) == 0.0:
        raise ValueError("masked BCE has no observed labels")
    return (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction="sum"
        )
        / denominator
    )


def _masked_bce_components(
    logits: Any,
    targets: Any,
    mask: Any,
    positive_weight: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    weights = torch.where(targets > 0.5, positive_weight, torch.ones_like(targets))
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction="sum"
        ),
        weights.sum(),
    )


class MultilabelLinearModel:
    """A one-layer model whose torch dependency is loaded only at runtime."""

    def __init__(
        self,
        input_dim: int,
        label_count: int,
        *,
        unsupported_priors: Sequence[float] = (),
        unsupported_indices: Sequence[int] = (),
        device: str = "cpu",
    ) -> None:
        torch = _torch()
        self.input_dim = int(input_dim)
        self.label_count = int(label_count)
        self.module = torch.nn.Linear(self.input_dim, self.label_count)
        self.linear = self.module
        self.unsupported_indices = tuple(int(i) for i in unsupported_indices)
        self.unsupported_priors = tuple(float(v) for v in unsupported_priors)
        self.device = torch.device(device)
        self.module.to(self.device)
        self.reset_unsupported()

    def reset_unsupported(self) -> None:
        with _torch().no_grad():
            for index, prior in zip(
                self.unsupported_indices, self.unsupported_priors
            ):
                self.module.weight[index].zero_()
                self.module.bias[index] = prior

    def __call__(self, features: Any) -> Any:
        logits = self.module(features)
        if not self.unsupported_indices:
            return logits
        torch = _torch()
        indices = torch.tensor(
            self.unsupported_indices, device=logits.device, dtype=torch.long
        )
        priors = torch.tensor(
            self.unsupported_priors, device=logits.device, dtype=logits.dtype
        )
        replacement = torch.zeros_like(logits)
        replacement[:, indices] = priors
        selector = torch.zeros(
            self.label_count, device=logits.device, dtype=torch.bool
        )
        selector[indices] = True
        return torch.where(selector.unsqueeze(0), replacement, logits)

    def parameters(self) -> Iterable[Any]:
        return self.module.parameters()

    def state_dict(self) -> Mapping[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state)
        self.reset_unsupported()


def _feature_records(
    path: Path | str,
    labels: tuple[str, ...],
    *,
    split: str,
    feature_mode: str,
    embedding_paths: EmbeddingPaths | None,
    batch_size: int,
) -> Iterator[tuple[tuple[float, ...], tuple[float, ...], tuple[bool, ...]]]:
    for row, embedding in _iter_split_rows(
        path,
        TRAIN_COLUMNS,
        split=split,
        feature_mode=feature_mode,
        embedding_paths=embedding_paths,
        batch_size=batch_size,
    ):
        features = build_feature_vector(
            row,
            labels,
            feature_mode=feature_mode,
            embedding=embedding,
        )
        targets, mask = build_masked_targets(row, labels)
        yield features, targets, mask


def _feature_batches(
    path: Path | str,
    labels: tuple[str, ...],
    *,
    split: str,
    feature_mode: str,
    embedding_paths: EmbeddingPaths | None,
    batch_size: int,
    device: Any,
    expected_dim: int,
) -> Iterator[tuple[Any, Any, Any]]:
    torch = _torch()
    features: list[tuple[float, ...]] = []
    targets: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    for feature, target, mask in _feature_records(
        path,
        labels,
        split=split,
        feature_mode=feature_mode,
        embedding_paths=embedding_paths,
        batch_size=batch_size,
    ):
        if len(feature) != expected_dim:
            raise ValueError(
                f"{split} feature dimension differs from the trained model"
            )
        features.append(feature)
        targets.append(target)
        masks.append(mask)
        if len(features) == batch_size:
            yield (
                torch.tensor(features, dtype=torch.float32, device=device),
                torch.tensor(targets, dtype=torch.float32, device=device),
                torch.tensor(masks, dtype=torch.bool, device=device),
            )
            features.clear()
            targets.clear()
            masks.clear()
    if features:
        yield (
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(targets, dtype=torch.float32, device=device),
            torch.tensor(masks, dtype=torch.bool, device=device),
        )


def _prediction_feature_batches(
    path: Path | str,
    labels: tuple[str, ...],
    *,
    feature_mode: str,
    embedding_paths: EmbeddingPaths | None,
    batch_size: int,
    device: Any,
    expected_dim: int,
) -> Iterator[tuple[list[str], Any]]:
    torch = _torch()
    user_ids: list[str] = []
    features: list[tuple[float, ...]] = []
    for row, embedding in _iter_split_rows(
        path,
        PREDICTION_COLUMNS,
        split="benchmark",
        feature_mode=feature_mode,
        embedding_paths=embedding_paths,
        batch_size=batch_size,
    ):
        feature = build_feature_vector(
            row,
            labels,
            feature_mode=feature_mode,
            embedding=embedding,
        )
        if len(feature) != expected_dim:
            raise ValueError(
                "benchmark feature dimension differs from the trained model"
            )
        user_ids.append(str(row["user_id"]))
        features.append(feature)
        if len(features) == batch_size:
            yield user_ids, torch.tensor(
                features, dtype=torch.float32, device=device
            )
            user_ids.clear()
            features.clear()
    if features:
        yield user_ids, torch.tensor(
            features, dtype=torch.float32, device=device
        )


def _count_train_labels(
    train_path: Path | str,
    labels: tuple[str, ...],
    *,
    feature_mode: str,
    embedding_paths: EmbeddingPaths | None,
    batch_size: int,
) -> tuple[list[int], list[int], int, int]:
    positives = [0] * len(labels)
    negatives = [0] * len(labels)
    users = 0
    feature_dim: int | None = None
    for row, embedding in _iter_split_rows(
        train_path,
        TRAIN_COLUMNS,
        split="train",
        feature_mode=feature_mode,
        embedding_paths=embedding_paths,
        batch_size=batch_size,
    ):
        users += 1
        if feature_mode == "topic_frequency":
            row_history = [str(value) for value in row["history_subcategories"]]
            invalid = set(row_history) - set(labels)
            if invalid:
                raise ValueError(
                    f"row {row['user_id']!r}: invalid history labels "
                    f"{sorted(invalid)}"
                )
            row_feature_dim = len(labels)
        else:
            if embedding is None:
                raise ValueError("missing aligned embedding")
            row_feature_dim = len(embedding)
        feature_dim = row_feature_dim if feature_dim is None else feature_dim
        if row_feature_dim != feature_dim:
            raise ValueError("train embeddings have inconsistent dimensions")
        targets, mask = build_masked_targets(row, labels)
        for index, observed in enumerate(mask):
            if observed:
                if targets[index] > 0.5:
                    positives[index] += 1
                else:
                    negatives[index] += 1
    if users == 0 or feature_dim is None:
        raise ValueError(f"{train_path}: no users")
    return positives, negatives, users, feature_dim


def _stream_loss(
    model: MultilabelLinearModel,
    path: Path | str,
    labels: tuple[str, ...],
    *,
    split: str,
    feature_mode: str,
    embedding_paths: EmbeddingPaths | None,
    batch_size: int,
    device: Any,
    expected_dim: int,
    positive_weights: Any,
) -> float:
    total = 0.0
    denominator = 0.0
    torch = _torch()
    with torch.no_grad():
        for features, targets, mask in _feature_batches(
            path,
            labels,
            split=split,
            feature_mode=feature_mode,
            embedding_paths=embedding_paths,
            batch_size=batch_size,
            device=device,
            expected_dim=expected_dim,
        ):
            loss_sum, weight_sum = _masked_bce_components(
                model(features), targets, mask, positive_weights
            )
            total += float(loss_sum.cpu())
            denominator += float(weight_sum.cpu())
    if denominator == 0.0:
        raise ValueError(f"{split} data has no observed labels")
    return total / denominator


def fit_multilabel_baseline(
    train_path: Path | str,
    vocabulary_path: Path | str,
    *,
    validation_path: Path | str | None = None,
    embedding_path: EmbeddingPaths | None = None,
    train_embedding_path: Path | str | None = None,
    validation_embedding_path: Path | str | None = None,
    feature_mode: str = "topic_frequency",
    epochs: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    patience: int = 5,
    batch_size: int = 64,
    seed: int = 20260812,
    device: str = "auto",
    prior_alpha: float = 1.0,
) -> tuple[MultilabelLinearModel, dict[str, Any]]:
    """Fit from streaming Parquet passes.

    For embedding features, pass separate ``train_embedding_path`` and
    ``validation_embedding_path`` artifacts.  The legacy ``embedding_path``
    argument remains accepted for callers already passing a split mapping.
    """
    torch = _torch()
    if feature_mode not in {"topic_frequency", "history_embedding"}:
        raise ValueError(f"Unknown feature mode {feature_mode!r}")
    if validation_path is None:
        raise ValueError("validation_path is required for streaming training")
    if epochs <= 0 or lr <= 0 or weight_decay < 0 or patience < 0 or batch_size <= 0:
        raise ValueError("epochs, lr, and batch_size must be positive")
    if prior_alpha <= 0:
        raise ValueError("prior_alpha must be positive")
    embedding_paths = _resolve_fit_embedding_paths(
        embedding_path, train_embedding_path, validation_embedding_path
    )
    if feature_mode == "history_embedding" and (
        train_embedding_path is None
        and validation_embedding_path is None
        and not isinstance(embedding_path, Mapping)
    ):
        raise ValueError(
            "history_embedding requires separate train_embedding_path and "
            "validation_embedding_path"
        )

    _seed_everything(seed)
    labels = load_vocabulary(vocabulary_path)
    positives, negatives, train_users, input_dim = _count_train_labels(
        train_path,
        labels,
        feature_mode=feature_mode,
        embedding_paths=embedding_paths,
        batch_size=batch_size,
    )
    unsupported: dict[str, str] = {}
    priors: list[float] = []
    for index, label in enumerate(labels):
        probability = (positives[index] + prior_alpha) / (
            positives[index] + negatives[index] + 2 * prior_alpha
        )
        priors.append(math.log(probability / (1.0 - probability)))
        if positives[index] == 0 and negatives[index] == 0:
            unsupported[label] = "no_train_positives_or_negatives"
        elif positives[index] == 0:
            unsupported[label] = "no_train_positives"
        elif negatives[index] == 0:
            unsupported[label] = "no_train_negatives"
    unsupported_indices = tuple(
        index for index, label in enumerate(labels) if label in unsupported
    )
    current_device = _device(device)
    model = MultilabelLinearModel(
        input_dim,
        len(labels),
        unsupported_priors=[priors[index] for index in unsupported_indices],
        unsupported_indices=unsupported_indices,
        device=str(current_device),
    )
    positive_weights = torch.tensor(
        [
            negatives[index] / positives[index] if positives[index] else 1.0
            for index in range(len(labels))
        ],
        dtype=torch.float32,
        device=current_device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(dict(model.state_dict()))
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.module.train()
        total = 0.0
        denominator = 0.0
        for features, targets, mask in _feature_batches(
            train_path,
            labels,
            split="train",
            feature_mode=feature_mode,
            embedding_paths=embedding_paths,
            batch_size=batch_size,
            device=current_device,
            expected_dim=input_dim,
        ):
            optimizer.zero_grad()
            loss_sum, weight_sum = _masked_bce_components(
                model(features), targets, mask, positive_weights
            )
            (loss_sum / weight_sum).backward()
            optimizer.step()
            model.reset_unsupported()
            total += float(loss_sum.detach().cpu())
            denominator += float(weight_sum.detach().cpu())
        if denominator == 0.0:
            raise ValueError("train data has no observed labels")
        train_loss = total / denominator
        model.module.eval()
        validation_loss = _stream_loss(
            model,
            validation_path,
            labels,
            split="validation",
            feature_mode=feature_mode,
            embedding_paths=embedding_paths,
            batch_size=batch_size,
            device=current_device,
            expected_dim=input_dim,
            positive_weights=positive_weights,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-12:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(dict(model.state_dict()))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                break
    model.load_state_dict(best_state)
    summary: dict[str, Any] = {
        "labels": list(labels),
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "train_users": train_users,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "history": history,
        "positive_counts": dict(zip(labels, positives)),
        "negative_counts": dict(zip(labels, negatives)),
        "label_status": {
            label: unsupported.get(label, "supported") for label in labels
        },
        "unsupported_labels": unsupported,
        "prior_alpha": prior_alpha,
        "seed": seed,
        "device": str(current_device),
    }
    model.labels = labels
    model.feature_mode = feature_mode
    model.summary = summary
    model.config = {
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "label_count": len(labels),
        "labels": list(labels),
        "prior_alpha": prior_alpha,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "batch_size": batch_size,
        "device": str(current_device),
    }
    return model, summary


def predict_multilabel_baseline(
    model: MultilabelLinearModel,
    benchmark_path: Path | str,
    output_dir: Path | str,
    *,
    embedding_path: EmbeddingPaths | None = None,
    test_embedding_path: Path | str | None = None,
    overwrite: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Stream test features and write complete ranked predictions.

    ``test_embedding_path`` is the embedding artifact aligned to
    ``benchmark_path``.  It is intentionally independent of training
    artifacts.
    """
    labels = tuple(model.labels)
    output_dir = Path(output_dir)
    prediction_path = output_dir / "predictions.parquet"
    model_path = output_dir / "model.pt"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "training_summary.json"
    existing = [
        path
        for path in (prediction_path, model_path, config_path, summary_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Outputs already exist: {existing}. Use overwrite=True."
        )
    torch = _torch()
    model.module.eval()
    embedding_paths = _resolve_test_embedding_path(
        embedding_path, test_embedding_path
    )
    users = 0
    with PredictionWriter(
        prediction_path, overwrite=overwrite, batch_size=batch_size
    ) as writer:
        for user_ids, features in _prediction_feature_batches(
            benchmark_path,
            labels,
            feature_mode=model.feature_mode,
            embedding_paths=embedding_paths,
            batch_size=batch_size,
            device=model.device,
            expected_dim=model.input_dim,
        ):
            with torch.no_grad():
                scores = torch.sigmoid(model(features)).cpu().tolist()
            for user_id, user_scores in zip(user_ids, scores):
                ranked = rank_label_scores(
                    labels, dict(zip(labels, user_scores))
                )
                writer.append(
                    user_id=user_id,
                    ranked_labels=ranked["ranked_labels"],
                    ranked_scores=ranked["ranked_scores"],
                )
                users += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": dict(model.state_dict()),
            "config": model.config,
            "summary": model.summary,
            "unsupported_indices": list(model.unsupported_indices),
            "unsupported_priors": list(model.unsupported_priors),
        },
        model_path,
    )
    config_path.write_text(
        json.dumps(model.config, indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(model.summary, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "users": users,
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "training_summary_path": str(summary_path),
    }


def run_multilabel_baseline(
    train_path: Path | str,
    validation_path: Path | str,
    benchmark_path: Path | str,
    vocabulary_path: Path | str,
    output_dir: Path | str,
    **kwargs: Any,
) -> dict[str, Any]:
    fit_kwargs = dict(kwargs)
    overwrite = bool(fit_kwargs.pop("overwrite", False))
    prediction_batch_size = int(fit_kwargs.pop("prediction_batch_size", 64))
    test_embedding_path = fit_kwargs.pop("test_embedding_path", None)
    embedding_path = fit_kwargs.get("embedding_path")
    model, summary = fit_multilabel_baseline(
        train_path,
        vocabulary_path,
        validation_path=validation_path,
        **fit_kwargs,
    )
    result = predict_multilabel_baseline(
        model,
        benchmark_path,
        output_dir,
        embedding_path=embedding_path,
        test_embedding_path=test_embedding_path,
        overwrite=overwrite,
        batch_size=prediction_batch_size,
    )
    result["training_summary"] = summary
    return result


fit_multilabel_linear_baseline = fit_multilabel_baseline
predict_multilabel_linear_baseline = predict_multilabel_baseline
fit_multilabel_future_interest = fit_multilabel_baseline
predict_multilabel_future_interest = predict_multilabel_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--train-embedding-path", type=Path)
    parser.add_argument("--validation-embedding-path", type=Path)
    parser.add_argument("--test-embedding-path", type=Path)
    parser.add_argument(
        "--feature-mode",
        choices=("topic_frequency", "history_embedding"),
        default="topic_frequency",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_multilabel_baseline(
        args.train,
        args.validation,
        args.benchmark,
        args.vocabulary,
        args.output_dir,
        embedding_path=args.embedding_path,
        train_embedding_path=args.train_embedding_path,
        validation_embedding_path=args.validation_embedding_path,
        test_embedding_path=args.test_embedding_path,
        feature_mode=args.feature_mode,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
