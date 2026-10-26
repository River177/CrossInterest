from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import tempfile
from array import array
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.build_nrms_title_embeddings import (
    load_nrms_title_embeddings,
)
from new_interest.experiments.build_nrms_interest_embeddings import (
    load_nrms_interest_embeddings,
)
from new_interest.experiments.evaluate_static_candidate_baselines import (
    _internal_role,
)
from new_interest.experiments.tune_semantic_history_linear_ranker import (
    _masked_weighted_cross_entropy,
    _resolve_device,
    _seed_everything,
)
from new_interest.experiments.tune_semantic_history_ridge import (
    Evaluation,
    MultiTargetSplit,
    SingleTargetSplit,
    _evaluate_global,
    _evaluate_top_k,
    _global_scores,
    _load_config,
    _normalised_class_weights,
    _pareto_front,
    _publish,
    _sha256,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-frozen-minilm-nrms-v1"
DEFAULT_CONFIG = Path(
    "configs/future_interest/static_user_disjoint_nrms_v1.json"
)


@dataclass(frozen=True)
class Architecture:
    hidden_dimension: int
    attention_heads: int
    dropout: float


@dataclass(frozen=True)
class Trial:
    architecture: Architecture
    class_balance_beta: float
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class HistoryBucket:
    boundary: int
    user_ids: tuple[str, ...]
    title_ids: np.ndarray
    targets: np.ndarray | None
    positives: np.ndarray | None
    history_mask: np.ndarray
    history_lengths: np.ndarray
    unit_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HistorySplit:
    name: str
    buckets: tuple[HistoryBucket, ...]
    label_count: int
    unit: str = "user"

    @property
    def user_count(self) -> int:
        return sum(len(bucket.user_ids) for bucket in self.buckets)

    @property
    def history_item_count(self) -> int:
        return sum(
            int(bucket.history_lengths.sum(dtype=np.int64))
            for bucket in self.buckets
        )

    @property
    def maximum_history_length(self) -> int:
        return max(
            (int(bucket.history_lengths.max()) for bucket in self.buckets),
            default=0,
        )


@dataclass(frozen=True)
class DeviceBucket:
    boundary: int
    title_ids: Any
    targets: Any | None
    history_mask: Any


@dataclass(frozen=True)
class DeviceSplit:
    buckets: tuple[DeviceBucket, ...]


class _BucketAccumulator:
    def __init__(self, boundary: int, *, multi_target: bool) -> None:
        self.boundary = int(boundary)
        self.multi_target = bool(multi_target)
        self.user_ids: list[str] = []
        self.flat_title_ids = array("i")
        self.offsets = array("q", [0])
        self.targets = array("i")
        self.positive_ids: list[tuple[int, ...]] = []
        self.history_ids: list[tuple[int, ...]] = []
        self.lengths = array("i")
        self.unit_ids: list[str | None] = []

    def add(
        self,
        *,
        user_id: str,
        title_ids: Sequence[int],
        history_ids: Sequence[int],
        target: int | None,
        positive_ids: Sequence[int] | None,
        unit_id: str | None = None,
    ) -> None:
        if not title_ids or len(title_ids) > self.boundary:
            raise ValueError(f"{user_id}: invalid bucketed History length")
        if self.multi_target:
            if target is not None or not positive_ids:
                raise ValueError(f"{user_id}: invalid multi-target row")
            self.positive_ids.append(tuple(sorted(set(positive_ids))))
        else:
            if target is None or positive_ids is not None:
                raise ValueError(f"{user_id}: invalid single-target row")
            self.targets.append(int(target))
        self.user_ids.append(user_id)
        self.flat_title_ids.extend(int(value) for value in title_ids)
        self.offsets.append(len(self.flat_title_ids))
        self.history_ids.append(tuple(sorted(set(history_ids))))
        self.lengths.append(len(title_ids))
        self.unit_ids.append(unit_id)

    def finalize(self, *, label_count: int, padding_id: int) -> HistoryBucket:
        count = len(self.user_ids)
        flat = np.frombuffer(self.flat_title_ids, dtype=np.int32)
        offsets = np.frombuffer(self.offsets, dtype=np.int64)
        lengths = np.frombuffer(self.lengths, dtype=np.int32).copy()
        title_ids = np.full(
            (count, self.boundary), padding_id, dtype=np.int32
        )
        for index in range(count):
            start = int(offsets[index])
            end = int(offsets[index + 1])
            title_ids[index, : end - start] = flat[start:end]
        history_mask = np.zeros((count, label_count), dtype=np.bool_)
        for index, values in enumerate(self.history_ids):
            if values:
                history_mask[index, list(values)] = True
        if self.multi_target:
            positives = np.zeros((count, label_count), dtype=np.bool_)
            for index, values in enumerate(self.positive_ids):
                positives[index, list(values)] = True
            targets = None
            if np.any(positives & history_mask):
                raise ValueError("Dev positives overlap History")
        else:
            targets = np.frombuffer(self.targets, dtype=np.int32).astype(
                np.int64, copy=True
            )
            positives = None
            if count and np.any(history_mask[np.arange(count), targets]):
                raise ValueError("Train target overlaps History")
        has_unit_ids = [value is not None for value in self.unit_ids]
        if any(has_unit_ids) and not all(has_unit_ids):
            raise ValueError("bucket mixes explicit and implicit evaluation IDs")
        return HistoryBucket(
            boundary=self.boundary,
            user_ids=tuple(self.user_ids),
            title_ids=np.ascontiguousarray(title_ids),
            targets=targets,
            positives=positives,
            history_mask=np.ascontiguousarray(history_mask),
            history_lengths=lengths,
            unit_ids=(
                tuple(str(value) for value in self.unit_ids)
                if all(has_unit_ids)
                else None
            ),
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime environment
        raise RuntimeError("NRMS experiment requires PyTorch") from error
    return torch


def _architecture(value: Mapping[str, object]) -> Architecture:
    architecture = Architecture(
        hidden_dimension=int(value["hidden_dimension"]),
        attention_heads=int(value["attention_heads"]),
        dropout=float(value["dropout"]),
    )
    if architecture.hidden_dimension <= 0:
        raise ValueError("hidden_dimension must be positive")
    if architecture.attention_heads <= 0:
        raise ValueError("attention_heads must be positive")
    if architecture.hidden_dimension % architecture.attention_heads:
        raise ValueError("hidden_dimension must be divisible by attention_heads")
    if not 0.0 <= architecture.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return architecture


def _validate_float_grid(
    name: str,
    values: Sequence[object],
    *,
    minimum: float,
    maximum: float | None = None,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be a non-empty unique grid")
    if any(not math.isfinite(value) or value < minimum for value in result):
        raise ValueError(f"{name} contains an invalid value")
    if maximum is not None and any(value > maximum for value in result):
        raise ValueError(f"{name} contains an invalid value")
    return result


def _bucket_boundary(length: int, boundaries: Sequence[int]) -> int:
    for boundary in boundaries:
        if length <= boundary:
            return int(boundary)
    raise ValueError(
        f"History length {length} exceeds largest bucket {boundaries[-1]}"
    )


def _train_targets(
    train_path: Path,
) -> tuple[tuple[str, ...], dict[str, int], np.ndarray]:
    raw: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["user_id", "target_interest"], batch_size=16_384
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in seen:
                raise ValueError(f"duplicate Static Train user {user_id}")
            seen.add(user_id)
            target = str(row["target_interest"]["subcategory"])
            raw.append((user_id, target))
            counts[target] = counts.get(target, 0) + 1
    if not raw:
        raise ValueError(f"{train_path}: no rows")
    labels = tuple(sorted(counts))
    label_to_id = {label: index for index, label in enumerate(labels)}
    targets = {user: label_to_id[target] for user, target in raw}
    count_vector = np.asarray([counts[label] for label in labels], dtype=np.int64)
    return labels, targets, count_vector


def _new_accumulators(
    roles: Iterable[str], boundaries: Sequence[int], *, multi_target: bool
) -> dict[str, dict[int, _BucketAccumulator]]:
    return {
        role: {
            int(boundary): _BucketAccumulator(
                int(boundary), multi_target=multi_target
            )
            for boundary in boundaries
        }
        for role in roles
    }


def _load_history_splits(
    *,
    benchmark_dir: Path,
    train_targets: Mapping[str, int],
    labels: Sequence[str],
    title_to_id: Mapping[str, int],
    boundaries: Sequence[int],
    seed: int,
    padding_id: int,
) -> tuple[dict[str, HistorySplit], HistorySplit, dict[str, object]]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_roles = ("model_fit", "tune", "holdout")
    train_accumulators = _new_accumulators(
        train_roles, boundaries, multi_target=False
    )
    seen_train: set[str] = set()
    source_split_counts: dict[str, int] = {}
    columns = [
        "user_id",
        "history_length",
        "history_titles",
        "history_subcategories",
    ]
    for source_split in ("fit", "validation"):
        path = benchmark_dir / f"{source_split}.parquet"
        source_count = 0
        for batch in pq.ParquetFile(path).iter_batches(
            columns=columns, batch_size=4_096
        ):
            for row in batch.to_pylist():
                user_id = str(row["user_id"])
                if user_id in seen_train:
                    raise ValueError(f"duplicate Train benchmark user {user_id}")
                if user_id not in train_targets:
                    raise ValueError(f"unknown Train benchmark user {user_id}")
                seen_train.add(user_id)
                titles = tuple(str(value) for value in row["history_titles"])
                subcategories = tuple(
                    str(value) for value in row["history_subcategories"]
                )
                if len(titles) != int(row["history_length"]):
                    raise ValueError(f"{user_id}: History length metadata mismatch")
                if len(titles) != len(subcategories) or not titles:
                    raise ValueError(f"{user_id}: malformed History")
                try:
                    title_ids = tuple(title_to_id[title] for title in titles)
                except KeyError as error:
                    raise ValueError(
                        f"{user_id}: title missing from frozen cache"
                    ) from error
                history_ids = tuple(
                    sorted(
                        {
                            label_to_id[label]
                            for label in subcategories
                            if label in label_to_id
                        }
                    )
                )
                role = _internal_role(user_id, seed)
                boundary = _bucket_boundary(len(title_ids), boundaries)
                train_accumulators[role][boundary].add(
                    user_id=user_id,
                    title_ids=title_ids,
                    history_ids=history_ids,
                    target=int(train_targets[user_id]),
                    positive_ids=None,
                )
                source_count += 1
        source_split_counts[source_split] = source_count
    if seen_train != set(train_targets):
        missing = sorted(set(train_targets) - seen_train)[:5]
        extra = sorted(seen_train - set(train_targets))[:5]
        raise ValueError(
            f"Train benchmark differs from Static Train; missing={missing}, extra={extra}"
        )
    train_splits = {
        role: HistorySplit(
            name=role,
            buckets=tuple(
                train_accumulators[role][int(boundary)].finalize(
                    label_count=len(labels), padding_id=padding_id
                )
                for boundary in boundaries
                if train_accumulators[role][int(boundary)].user_ids
            ),
            label_count=len(labels),
        )
        for role in train_roles
    }

    dev_accumulators = _new_accumulators(
        ("dev",), boundaries, multi_target=True
    )["dev"]
    seen_dev: set[str] = set()
    dev_path = benchmark_dir / "dev.parquet"
    dev_columns = columns + ["future_novel_subcategories"]
    for batch in pq.ParquetFile(dev_path).iter_batches(
        columns=dev_columns, batch_size=4_096
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in seen_dev:
                raise ValueError(f"duplicate Dev benchmark user {user_id}")
            seen_dev.add(user_id)
            titles = tuple(str(value) for value in row["history_titles"])
            subcategories = tuple(
                str(value) for value in row["history_subcategories"]
            )
            if len(titles) != int(row["history_length"]):
                raise ValueError(f"{user_id}: Dev History length metadata mismatch")
            if len(titles) != len(subcategories) or not titles:
                raise ValueError(f"{user_id}: malformed Dev History")
            try:
                title_ids = tuple(title_to_id[title] for title in titles)
            except KeyError as error:
                raise ValueError(
                    f"{user_id}: Dev title missing from frozen cache"
                ) from error
            history_ids = tuple(
                sorted(
                    {
                        label_to_id[label]
                        for label in subcategories
                        if label in label_to_id
                    }
                )
            )
            positives: list[int] = []
            for value in row["future_novel_subcategories"]:
                label = str(value)
                if label not in label_to_id:
                    raise ValueError(f"Dev target lacks Train support: {label!r}")
                positives.append(label_to_id[label])
            boundary = _bucket_boundary(len(title_ids), boundaries)
            dev_accumulators[boundary].add(
                user_id=user_id,
                title_ids=title_ids,
                history_ids=history_ids,
                target=None,
                positive_ids=positives,
            )
    dev = HistorySplit(
        name="dev",
        buckets=tuple(
            dev_accumulators[int(boundary)].finalize(
                label_count=len(labels), padding_id=padding_id
            )
            for boundary in boundaries
            if dev_accumulators[int(boundary)].user_ids
        ),
        label_count=len(labels),
    )
    metadata = {
        "semantic_source_split_counts": source_split_counts,
        "role_counts": {
            role: split.user_count for role, split in train_splits.items()
        },
        "dev_users": dev.user_count,
        "history_item_counts": {
            **{
                role: split.history_item_count
                for role, split in train_splits.items()
            },
            "dev": dev.history_item_count,
        },
        "maximum_history_lengths": {
            **{
                role: split.maximum_history_length
                for role, split in train_splits.items()
            },
            "dev": dev.maximum_history_length,
        },
        "bucket_counts": {
            **{
                role: {
                    str(bucket.boundary): len(bucket.user_ids)
                    for bucket in split.buckets
                }
                for role, split in train_splits.items()
            },
            "dev": {
                str(bucket.boundary): len(bucket.user_ids)
                for bucket in dev.buckets
            },
        },
    }
    return train_splits, dev, metadata


def _encoded_history(
    history: object,
    *,
    owner_id: str,
    label_to_id: Mapping[str, int],
    input_to_id: Mapping[str, int],
    history_input: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[str, str], ...]]:
    if not isinstance(history, list) or not history:
        raise ValueError(f"{owner_id}: malformed History")
    signature: list[tuple[str, str]] = []
    input_ids: list[int] = []
    history_ids: set[int] = set()
    for item in history:
        if not isinstance(item, dict):
            raise ValueError(f"{owner_id}: malformed History item")
        title = str(item["title"])
        subcategory = str(item["subcategory"])
        input_value = title if history_input == "behavior_titles" else subcategory
        try:
            input_ids.append(int(input_to_id[input_value]))
        except KeyError as error:
            raise ValueError(
                f"{owner_id}: {history_input} value missing from frozen cache: "
                f"{input_value!r}"
            ) from error
        if subcategory in label_to_id:
            history_ids.add(int(label_to_id[subcategory]))
        signature.append((title, subcategory))
    return tuple(input_ids), tuple(sorted(history_ids)), tuple(signature)


def _load_dataset_row_splits(
    *,
    train_path: Path,
    dev_path: Path,
    train_targets: Mapping[str, int],
    labels: Sequence[str],
    title_to_id: Mapping[str, int] | None = None,
    input_to_id: Mapping[str, int] | None = None,
    history_input: str = "behavior_titles",
    boundaries: Sequence[int],
    seed: int,
    padding_id: int,
) -> tuple[dict[str, HistorySplit], HistorySplit, dict[str, object]]:
    """Load selected Train rows and group retained Dev targets by impression."""

    if history_input not in {"behavior_titles", "interest_names"}:
        raise ValueError(
            "history_input must be behavior_titles or interest_names"
        )
    if input_to_id is None:
        if title_to_id is None:
            raise ValueError("input_to_id is required")
        input_to_id = title_to_id
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_roles = ("model_fit", "tune", "holdout")
    train_accumulators = _new_accumulators(
        train_roles, boundaries, multi_target=False
    )
    seen_train: set[str] = set()
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["user_id", "history", "target_interest"],
        batch_size=4_096,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in seen_train:
                raise ValueError(f"duplicate Train user {user_id}")
            if user_id not in train_targets:
                raise ValueError(f"unknown Train user {user_id}")
            seen_train.add(user_id)
            target_label = str(row["target_interest"]["subcategory"])
            target = label_to_id.get(target_label)
            if target is None or target != int(train_targets[user_id]):
                raise ValueError(f"{user_id}: Train target identity mismatch")
            title_ids, history_ids, _ = _encoded_history(
                row["history"],
                owner_id=user_id,
                label_to_id=label_to_id,
                input_to_id=input_to_id,
                history_input=history_input,
            )
            role = _internal_role(user_id, seed)
            boundary = _bucket_boundary(len(title_ids), boundaries)
            train_accumulators[role][boundary].add(
                user_id=user_id,
                title_ids=title_ids,
                history_ids=history_ids,
                target=target,
                positive_ids=None,
            )
    if seen_train != set(train_targets):
        missing = sorted(set(train_targets) - seen_train)[:5]
        raise ValueError(f"Train rows are missing selected users: {missing}")
    train_splits = {
        role: HistorySplit(
            name=role,
            buckets=tuple(
                train_accumulators[role][int(boundary)].finalize(
                    label_count=len(labels), padding_id=padding_id
                )
                for boundary in boundaries
                if train_accumulators[role][int(boundary)].user_ids
            ),
            label_count=len(labels),
        )
        for role in train_roles
    }

    impression_groups: dict[str, dict[str, object]] = {}
    dev_rows = 0
    for batch in pq.ParquetFile(dev_path).iter_batches(
        columns=[
            "impression_id",
            "user_id",
            "history",
            "target_interest",
        ],
        batch_size=4_096,
    ):
        for row in batch.to_pylist():
            dev_rows += 1
            impression_id = str(row["impression_id"])
            user_id = str(row["user_id"])
            title_ids, history_ids, history_signature = _encoded_history(
                row["history"],
                owner_id=impression_id,
                label_to_id=label_to_id,
                input_to_id=input_to_id,
                history_input=history_input,
            )
            target_label = str(row["target_interest"]["subcategory"])
            if target_label not in label_to_id:
                raise ValueError(
                    f"{impression_id}: Dev target lacks Train support: "
                    f"{target_label!r}"
                )
            target = int(label_to_id[target_label])
            group = impression_groups.get(impression_id)
            if group is None:
                impression_groups[impression_id] = {
                    "user_id": user_id,
                    "title_ids": title_ids,
                    "history_ids": history_ids,
                    "history_signature": history_signature,
                    "positive_ids": {target},
                }
                continue
            if str(group["user_id"]) != user_id:
                raise ValueError(
                    f"{impression_id}: multiple users in one impression"
                )
            if group["history_signature"] != history_signature:
                raise ValueError(
                    f"{impression_id}: inconsistent History across targets"
                )
            positive_ids = group["positive_ids"]
            if not isinstance(positive_ids, set):
                raise AssertionError("invalid impression accumulator")
            if target in positive_ids:
                raise ValueError(
                    f"{impression_id}: duplicate target {target_label!r}"
                )
            positive_ids.add(target)

    dev_accumulators = _new_accumulators(
        ("dev",), boundaries, multi_target=True
    )["dev"]
    dev_user_ids: set[str] = set()
    for impression_id in sorted(impression_groups):
        group = impression_groups[impression_id]
        user_id = str(group["user_id"])
        dev_user_ids.add(user_id)
        title_ids = tuple(int(value) for value in group["title_ids"])
        history_ids = tuple(int(value) for value in group["history_ids"])
        positive_ids = tuple(sorted(int(value) for value in group["positive_ids"]))
        boundary = _bucket_boundary(len(title_ids), boundaries)
        dev_accumulators[boundary].add(
            user_id=user_id,
            unit_id=impression_id,
            title_ids=title_ids,
            history_ids=history_ids,
            target=None,
            positive_ids=positive_ids,
        )
    dev = HistorySplit(
        name="dev",
        buckets=tuple(
            dev_accumulators[int(boundary)].finalize(
                label_count=len(labels), padding_id=padding_id
            )
            for boundary in boundaries
            if dev_accumulators[int(boundary)].user_ids
        ),
        label_count=len(labels),
        unit="impression",
    )
    metadata = {
        "loader": "dataset_rows",
        "history_input": history_input,
        "role_counts": {
            role: split.user_count for role, split in train_splits.items()
        },
        "dev_rows": dev_rows,
        "dev_impressions": dev.user_count,
        "dev_users": len(dev_user_ids),
        "history_item_counts": {
            **{
                role: split.history_item_count
                for role, split in train_splits.items()
            },
            "dev": dev.history_item_count,
        },
        "maximum_history_lengths": {
            **{
                role: split.maximum_history_length
                for role, split in train_splits.items()
            },
            "dev": dev.maximum_history_length,
        },
        "bucket_counts": {
            **{
                role: {
                    str(bucket.boundary): len(bucket.user_ids)
                    for bucket in split.buckets
                }
                for role, split in train_splits.items()
            },
            "dev": {
                str(bucket.boundary): len(bucket.user_ids)
                for bucket in dev.buckets
            },
        },
    }
    return train_splits, dev, metadata


def _merge_splits(name: str, splits: Sequence[HistorySplit]) -> HistorySplit:
    if not splits:
        raise ValueError("cannot merge an empty split list")
    label_count = splits[0].label_count
    if any(split.label_count != label_count for split in splits):
        raise ValueError("split label counts differ")
    if any(split.unit != splits[0].unit for split in splits):
        raise ValueError("split evaluation units differ")
    boundaries = sorted(
        {bucket.boundary for split in splits for bucket in split.buckets}
    )
    output: list[HistoryBucket] = []
    for boundary in boundaries:
        values = [
            bucket
            for split in splits
            for bucket in split.buckets
            if bucket.boundary == boundary
        ]
        targets = None
        positives = None
        if all(bucket.targets is not None for bucket in values):
            targets = np.concatenate(
                [bucket.targets for bucket in values if bucket.targets is not None]
            )
        elif all(bucket.positives is not None for bucket in values):
            positives = np.concatenate(
                [
                    bucket.positives
                    for bucket in values
                    if bucket.positives is not None
                ],
                axis=0,
            )
        else:
            raise ValueError("cannot merge single- and multi-target splits")
        explicit_unit_ids = [
            bucket.unit_ids is not None for bucket in values
        ]
        if any(explicit_unit_ids) and not all(explicit_unit_ids):
            raise ValueError("cannot merge explicit and implicit evaluation IDs")
        output.append(
            HistoryBucket(
                boundary=boundary,
                user_ids=tuple(
                    user_id for bucket in values for user_id in bucket.user_ids
                ),
                title_ids=np.ascontiguousarray(
                    np.concatenate([bucket.title_ids for bucket in values], axis=0)
                ),
                targets=targets,
                positives=positives,
                history_mask=np.ascontiguousarray(
                    np.concatenate(
                        [bucket.history_mask for bucket in values], axis=0
                    )
                ),
                history_lengths=np.concatenate(
                    [bucket.history_lengths for bucket in values]
                ),
                unit_ids=(
                    tuple(
                        unit_id
                        for bucket in values
                        for unit_id in (bucket.unit_ids or ())
                    )
                    if all(explicit_unit_ids)
                    else None
                ),
            )
        )
    return HistorySplit(
        name=name,
        buckets=tuple(output),
        label_count=label_count,
        unit=splits[0].unit,
    )


def _unit_ids(split: HistorySplit) -> tuple[str, ...]:
    return tuple(
        unit_id
        for bucket in split.buckets
        for unit_id in (
            bucket.unit_ids
            if bucket.unit_ids is not None
            else bucket.user_ids
        )
    )


def _evaluation_view(
    split: HistorySplit,
) -> SingleTargetSplit | MultiTargetSplit:
    users = tuple(user for bucket in split.buckets for user in bucket.user_ids)
    history = np.concatenate(
        [bucket.history_mask for bucket in split.buckets], axis=0
    )
    dummy = np.zeros((len(users), 1), dtype=np.float32)
    if all(bucket.targets is not None for bucket in split.buckets):
        targets = np.concatenate(
            [bucket.targets for bucket in split.buckets if bucket.targets is not None]
        )
        return SingleTargetSplit(users, dummy, targets, history)
    if all(bucket.positives is not None for bucket in split.buckets):
        positives = np.concatenate(
            [
                bucket.positives
                for bucket in split.buckets
                if bucket.positives is not None
            ],
            axis=0,
        )
        return MultiTargetSplit(users, dummy, positives, history)
    raise ValueError("split mixes single- and multi-target buckets")


def _device_split(split: HistorySplit, device: str) -> DeviceSplit:
    torch = _torch()
    current_device = torch.device(device)
    return DeviceSplit(
        tuple(
            DeviceBucket(
                boundary=bucket.boundary,
                title_ids=torch.as_tensor(
                    bucket.title_ids,
                    dtype=torch.long,
                    device=current_device,
                ),
                targets=(
                    torch.as_tensor(
                        bucket.targets,
                        dtype=torch.long,
                        device=current_device,
                    )
                    if bucket.targets is not None
                    else None
                ),
                history_mask=torch.as_tensor(
                    bucket.history_mask,
                    dtype=torch.bool,
                    device=current_device,
                ),
            )
            for bucket in split.buckets
        )
    )


def _precision_context(device: str, precision: str) -> Any:
    torch = _torch()
    if precision == "float32":
        return nullcontext()
    if precision != "bfloat16" or device != "cuda":
        raise ValueError("bfloat16 is supported only on CUDA in this experiment")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class NRMSInterestRanker:  # wrapper makes importing tests cheap without Torch
    @staticmethod
    def build(
        title_embeddings: Any,
        *,
        padding_id: int,
        label_count: int,
        architecture: Architecture,
    ) -> Any:
        torch = _torch()

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Deliberately not a buffer/parameter: the frozen title matrix is
                # separately hash-pinned and excluded from every checkpoint.
                self.title_embeddings = title_embeddings
                self.padding_id = int(padding_id)
                self.projection = torch.nn.Linear(
                    int(title_embeddings.shape[1]),
                    architecture.hidden_dimension,
                )
                self.self_attention = torch.nn.MultiheadAttention(
                    architecture.hidden_dimension,
                    architecture.attention_heads,
                    dropout=architecture.dropout,
                    batch_first=True,
                )
                self.attention_norm = torch.nn.LayerNorm(
                    architecture.hidden_dimension
                )
                self.additive_projection = torch.nn.Linear(
                    architecture.hidden_dimension,
                    architecture.hidden_dimension,
                )
                self.additive_score = torch.nn.Linear(
                    architecture.hidden_dimension, 1, bias=False
                )
                self.dropout = torch.nn.Dropout(architecture.dropout)
                self.classifier = torch.nn.Linear(
                    architecture.hidden_dimension, label_count, bias=False
                )

            def forward(self, title_ids: Any) -> Any:
                padding = title_ids.eq(self.padding_id)
                if bool(padding.all(dim=1).any().detach().cpu()):
                    raise ValueError("NRMS received an empty History")
                values = self.title_embeddings[title_ids]
                projected = torch.nn.functional.gelu(self.projection(values))
                attended, _ = self.self_attention(
                    projected,
                    projected,
                    projected,
                    key_padding_mask=padding,
                    need_weights=False,
                )
                encoded = self.attention_norm(
                    projected + self.dropout(attended)
                )
                attention_logits = self.additive_score(
                    torch.tanh(self.additive_projection(encoded))
                ).squeeze(-1)
                attention_logits = attention_logits.masked_fill(
                    padding, -torch.inf
                )
                weights = torch.softmax(attention_logits, dim=1)
                user = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
                return self.classifier(self.dropout(user))

        return _Model()


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().float().clone()
        for name, value in model.state_dict().items()
    }


def _trial_key(trial: Trial) -> tuple[float | int, ...]:
    return (
        trial.architecture.hidden_dimension,
        trial.architecture.attention_heads,
        trial.architecture.dropout,
        trial.class_balance_beta,
        trial.learning_rate,
        trial.weight_decay,
    )


def _trial_fields(trial: Trial) -> dict[str, float | int]:
    return {
        "hidden_dimension": trial.architecture.hidden_dimension,
        "attention_heads": trial.architecture.attention_heads,
        "dropout": trial.architecture.dropout,
        "class_balance_beta": trial.class_balance_beta,
        "learning_rate": trial.learning_rate,
        "weight_decay": trial.weight_decay,
    }


def _score_split(
    model: Any,
    split: HistorySplit,
    device_split: DeviceSplit,
    *,
    batch_sizes: Mapping[int, int],
    precision: str,
    device: str,
    labels: Sequence[str],
    k: int,
    history_prediction_filtering: bool = True,
) -> Evaluation:
    torch = _torch()
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for host_bucket, device_bucket in zip(
            split.buckets, device_split.buckets, strict=True
        ):
            batch_size = batch_sizes[host_bucket.boundary]
            for start in range(0, len(host_bucket.user_ids), batch_size):
                end = start + batch_size
                with _precision_context(device, precision):
                    logits = model(device_bucket.title_ids[start:end])
                scores.append(
                    logits.detach().cpu().float().numpy().astype(
                        np.float32, copy=False
                    )
                )
    score_matrix = np.ascontiguousarray(np.concatenate(scores, axis=0))
    view = _evaluation_view(split)
    positives = (
        np.eye(len(labels), dtype=np.bool_)[view.targets]
        if isinstance(view, SingleTargetSplit)
        else view.positives
    )
    history_mask = (
        view.history_mask
        if history_prediction_filtering
        else np.zeros_like(view.history_mask, dtype=np.bool_)
    )
    annotated = _annotate_evaluation(
        _evaluate_top_k(
            score_matrix,
            positives,
            history_mask,
            labels,
            k=k,
        ),
        split,
    )
    metrics = dict(annotated.metrics)
    metrics["history_prediction_filtering"] = bool(
        history_prediction_filtering
    )
    return Evaluation(
        metrics=metrics,
        per_interest=annotated.per_interest,
        top_k=annotated.top_k,
        top_scores=annotated.top_scores,
    )


def _annotate_evaluation(
    result: Evaluation,
    split: HistorySplit,
) -> Evaluation:
    metrics = dict(result.metrics)
    metrics["evaluation_unit"] = split.unit
    metrics["evaluation_units"] = split.user_count
    if split.unit == "impression":
        metrics["impressions"] = split.user_count
        metrics["distinct_users"] = len(
            {user_id for bucket in split.buckets for user_id in bucket.user_ids}
        )
        metrics["target_occurrences"] = int(
            metrics["distinct_positive_pairs"]
        )
        metrics.pop("users", None)
        for metric in ("recall", "ndcg"):
            source = f"user_macro_{metric}_at_10"
            if source in metrics:
                metrics[f"impression_macro_{metric}_at_10"] = metrics.pop(
                    source
                )
    return Evaluation(
        metrics=metrics,
        per_interest=result.per_interest,
        top_k=result.top_k,
        top_scores=result.top_scores,
    )


def _write_nrms_condition(
    staging: Path,
    split_name: str,
    condition_name: str,
    split: HistorySplit,
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
    view = _evaluation_view(split)
    positives = (
        np.eye(len(labels), dtype=np.bool_)[view.targets]
        if isinstance(view, SingleTargetSplit)
        else view.positives
    )
    users = tuple(
        user_id for bucket in split.buckets for user_id in bucket.user_ids
    )
    units = _unit_ids(split)
    prediction_rows: list[dict[str, object]] = []
    for index, (unit_id, user_id) in enumerate(zip(units, users, strict=True)):
        row: dict[str, object] = {
            "user_id": user_id,
            "target_interests": [
                labels[value]
                for value in np.flatnonzero(positives[index]).tolist()
            ],
            "ranked_interests": [
                labels[int(value)] for value in result.top_k[index]
            ],
            "ranked_scores": result.top_scores[index].astype(float).tolist(),
        }
        if split.unit == "impression":
            row["impression_id"] = unit_id
        prediction_rows.append(row)
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


def _environment_metadata(
    torch: Any,
    *,
    device: str,
    precision: str,
) -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "resolved_device": device,
        "resolved_precision": precision,
    }


def _fit_trial(
    *,
    trial: Trial,
    train: HistorySplit,
    train_device: DeviceSplit,
    tune: HistorySplit | None,
    tune_device: DeviceSplit | None,
    title_embeddings: Any,
    padding_id: int,
    class_counts: np.ndarray,
    epochs: int,
    evaluation_epochs: Sequence[int],
    batch_sizes: Mapping[int, int],
    gradient_clip_norm: float,
    precision: str,
    seed: int,
    device: str,
    labels: Sequence[str],
    k: int,
) -> tuple[
    list[dict[str, object]],
    dict[int, dict[str, Any]],
    list[dict[str, float]],
]:
    torch = _torch()
    _seed_everything(seed)
    current_device = torch.device(device)
    model = NRMSInterestRanker.build(
        title_embeddings,
        padding_id=padding_id,
        label_count=len(labels),
        architecture=trial.architecture,
    ).to(current_device)
    # title_embeddings is deliberately unregistered, so restore its device after
    # Module.to() and assert that checkpoints cannot silently duplicate it.
    model.title_embeddings = title_embeddings
    if "title_embeddings" in model.state_dict():
        raise AssertionError("frozen title embeddings leaked into state_dict")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial.learning_rate,
        weight_decay=trial.weight_decay,
    )
    class_weights = torch.as_tensor(
        _normalised_class_weights(class_counts, trial.class_balance_beta),
        dtype=torch.float32,
        device=current_device,
    )
    evaluation_set = {int(value) for value in evaluation_epochs}
    records: list[dict[str, object]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    history_rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = 0.0
        weight_total = 0.0
        batch_count = 0
        for bucket_index, bucket in enumerate(train_device.buckets):
            if bucket.targets is None:
                raise ValueError("training split is missing targets")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1009 * epoch + 97 * bucket_index)
            permutation = torch.randperm(
                len(bucket.title_ids), generator=generator
            ).to(current_device)
            batch_size = batch_sizes[bucket.boundary]
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start : start + batch_size]
                targets = bucket.targets[indices]
                sample_weights = class_weights[targets]
                optimizer.zero_grad(set_to_none=True)
                with _precision_context(device, precision):
                    logits = model(bucket.title_ids[indices])
                    loss = _masked_weighted_cross_entropy(
                        logits,
                        targets,
                        bucket.history_mask[indices],
                        sample_weights,
                        validate=False,
                    )
                loss.backward()
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()
                current_weight = float(sample_weights.sum().detach().cpu())
                weighted_loss += float(loss.detach().cpu()) * current_weight
                weight_total += current_weight
                batch_count += 1
        history_rows.append(
            {
                "epoch": float(epoch),
                "weighted_cross_entropy": weighted_loss / weight_total,
                "batches": float(batch_count),
            }
        )
        if epoch not in evaluation_set:
            continue
        checkpoints[epoch] = _state_dict_cpu(model)
        if tune is not None and tune_device is not None:
            result = _score_split(
                model,
                tune,
                tune_device,
                batch_sizes=batch_sizes,
                precision=precision,
                device=device,
                labels=labels,
                k=k,
            )
            records.append(
                {
                    **_trial_fields(trial),
                    "epoch": epoch,
                    **result.metrics,
                }
            )
    del optimizer, class_weights, model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return records, checkpoints, history_rows


def _select_configuration(
    records: Sequence[Mapping[str, object]],
    *,
    global_recall: float,
    user_recall_floor: float | None,
    recall_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not records:
        raise ValueError("cannot select from an empty NRMS grid")
    eligible_user = list(records)
    if user_recall_floor is not None:
        eligible_user = [
            row
            for row in records
            if float(row["user_macro_recall_at_10"]) >= user_recall_floor
        ]
    if not eligible_user:
        raise ValueError("no NRMS configuration satisfies the User-Recall floor")
    frontier = _pareto_front(eligible_user)
    best_recall = max(
        float(row["interest_macro_recall_at_10"]) for row in eligible_user
    )
    recall_floor = max(global_recall, best_recall - recall_tolerance)
    eligible = [
        row
        for row in frontier
        if float(row["interest_macro_recall_at_10"]) >= recall_floor
    ]
    if not eligible:
        raise ValueError("no NRMS configuration satisfies the Interest-Recall floor")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["interest_macro_ndcg_at_10"]),
            float(row["interest_macro_recall_at_10"]),
            float(row["user_macro_recall_at_10"]),
            -int(row["hidden_dimension"]),
            -float(row["dropout"]),
            -float(row["class_balance_beta"]),
            -float(row["weight_decay"]),
            -float(row["learning_rate"]),
            -int(row["epoch"]),
        ),
    )
    return dict(selected), frontier


def _row_to_trial(row: Mapping[str, object]) -> Trial:
    return Trial(
        architecture=Architecture(
            hidden_dimension=int(row["hidden_dimension"]),
            attention_heads=int(row["attention_heads"]),
            dropout=float(row["dropout"]),
        ),
        class_balance_beta=float(row["class_balance_beta"]),
        learning_rate=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _metrics_from_summary(
    path: Path, split: str, condition: str
) -> dict[str, object]:
    summary = _load_json(path)
    return dict(summary[split][condition])


def _save_model(
    path: Path,
    state: Mapping[str, Any],
    *,
    labels: Sequence[str],
    selected: Mapping[str, object],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(labels),
        "selected_json": np.asarray(
            [json.dumps(dict(selected), ensure_ascii=False, sort_keys=True)]
        ),
    }
    for name, value in state.items():
        arrays[f"parameter::{name}"] = value.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def run_static_user_disjoint_nrms(
    path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    overwrite: bool = False,
) -> Path:
    torch = _torch()
    config, config_path = _load_config(path_or_config)
    dataset_dir = Path(str(config["dataset_dir"]))
    benchmark_value = config.get("semantic_benchmark_dir")
    benchmark_dir = (
        Path(str(benchmark_value)) if benchmark_value is not None else None
    )
    title_embedding_value = config.get("title_embedding_dir")
    title_embedding_dir = (
        Path(str(title_embedding_value))
        if title_embedding_value is not None
        else None
    )
    output_dir = Path(str(config["output_dir"]))
    linear_value = config.get("linear_ranker_output_dir")
    ridge_value = config.get("balanced_ridge_output_dir")
    linear_output_dir = (
        Path(str(linear_value)) if linear_value is not None else None
    )
    ridge_output_dir = (
        Path(str(ridge_value)) if ridge_value is not None else None
    )
    seed = int(config["seed"])
    k = int(config["k"])
    if k != 10:
        raise ValueError("the v1 NRMS selector is defined for k=10")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    data_config = dict(config["data"])
    data_loader = str(data_config.get("loader", "semantic_users"))
    if data_loader not in {"semantic_users", "dataset_rows"}:
        raise ValueError("data.loader must be semantic_users or dataset_rows")
    history_input = str(
        data_config.get("history_input", "behavior_titles")
    )
    if history_input not in {"behavior_titles", "interest_names"}:
        raise ValueError(
            "data.history_input must be behavior_titles or interest_names"
        )
    if data_loader == "semantic_users" and history_input != "behavior_titles":
        raise ValueError(
            "semantic_users loader supports only behavior_titles input"
        )
    boundaries = tuple(int(value) for value in data_config["bucket_boundaries"])
    if (
        not boundaries
        or tuple(sorted(set(boundaries))) != boundaries
    ):
        raise ValueError("bucket boundaries must be positive, sorted, and unique")
    if boundaries[0] <= 0:
        raise ValueError("bucket boundaries must be positive, sorted, and unique")
    batch_sizes = {
        int(key): int(value)
        for key, value in dict(data_config["batch_size_by_boundary"]).items()
    }
    if set(batch_sizes) != set(boundaries) or any(
        value <= 0 for value in batch_sizes.values()
    ):
        raise ValueError("batch sizes must cover every History bucket")

    training = dict(config["training"])
    requested_device = str(training.get("device", "auto"))
    device = _resolve_device(requested_device)
    requested_precision = str(training.get("precision", "auto"))
    precision = (
        "bfloat16"
        if requested_precision == "auto" and device == "cuda"
        else "float32"
        if requested_precision == "auto"
        else requested_precision
    )
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be auto, float32, or bfloat16")
    if precision == "bfloat16" and device != "cuda":
        raise ValueError("bfloat16 is supported only on CUDA")
    gradient_clip_norm = float(training["gradient_clip_norm"])
    if gradient_clip_norm < 0.0:
        raise ValueError("gradient_clip_norm must be non-negative")

    architecture_stage = dict(config["architecture_stage"])
    architectures = tuple(
        _architecture(dict(value))
        for value in architecture_stage["architectures"]
    )
    if len(set(architectures)) != len(architectures):
        raise ValueError("architecture grid contains duplicates")
    architecture_epochs = int(architecture_stage["epochs"])
    architecture_evaluation_epochs = tuple(
        int(value) for value in architecture_stage["evaluation_epochs"]
    )
    architecture_beta = float(architecture_stage["class_balance_beta"])
    architecture_lr = float(architecture_stage["learning_rate"])
    architecture_wd = float(architecture_stage["weight_decay"])

    optimization_stage = dict(config["optimization_stage"])
    betas = _validate_float_grid(
        "class_balance_betas",
        optimization_stage["class_balance_betas"],
        minimum=0.0,
        maximum=1.0,
    )
    learning_rates = _validate_float_grid(
        "learning_rates",
        optimization_stage["learning_rates"],
        minimum=np.finfo(np.float64).tiny,
    )
    weight_decays = _validate_float_grid(
        "weight_decays",
        optimization_stage["weight_decays"],
        minimum=0.0,
    )
    optimization_epochs = int(optimization_stage["epochs"])
    optimization_evaluation_epochs = tuple(
        int(value) for value in optimization_stage["evaluation_epochs"]
    )
    if (
        architecture_epochs <= 0
        or optimization_epochs <= 0
        or not architecture_evaluation_epochs
        or not optimization_evaluation_epochs
        or max(architecture_evaluation_epochs) > architecture_epochs
        or max(optimization_evaluation_epochs) > optimization_epochs
    ):
        raise ValueError("invalid epoch schedule")

    selection = dict(config["selection"])
    recall_tolerance = float(selection["best_recall_tolerance_absolute"])
    if recall_tolerance < 0.0:
        raise ValueError("recall tolerance must be non-negative")

    train_path = dataset_dir / "train.parquet"
    dev_path = dataset_dir / "dev.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    if data_loader == "semantic_users":
        if benchmark_dir is None:
            raise ValueError(
                "semantic_benchmark_dir is required for semantic_users"
            )
        data_paths = tuple(
            benchmark_dir / f"{split}.parquet"
            for split in ("fit", "validation", "dev")
        )
    else:
        data_paths = (dev_path,)
    if history_input == "behavior_titles":
        if title_embedding_dir is None:
            raise ValueError(
                "title_embedding_dir is required for behavior_titles input"
            )
        history_embedding_dir = title_embedding_dir
        embedding_paths = (
            history_embedding_dir / "title_embeddings.parquet",
            history_embedding_dir / "metadata.json",
            history_embedding_dir / "manifest.json",
        )
    else:
        if "interest_embedding_dir" not in config:
            raise ValueError(
                "interest_embedding_dir is required for interest_names input"
            )
        history_embedding_dir = Path(str(config["interest_embedding_dir"]))
        embedding_paths = (
            history_embedding_dir / "interest_embeddings.parquet",
            history_embedding_dir / "metadata.json",
            history_embedding_dir / "manifest.json",
        )
    linear_summary_path = (
        linear_output_dir / "summary.json"
        if linear_output_dir is not None
        else None
    )
    linear_tuning_path = (
        linear_output_dir / "tuning.json"
        if linear_output_dir is not None
        else None
    )
    ridge_summary_path = (
        ridge_output_dir / "summary.json"
        if ridge_output_dir is not None
        else None
    )
    source_paths = [
        train_path,
        vocabulary_path,
        *data_paths,
        *embedding_paths,
    ]
    for optional_path in (
        linear_summary_path,
        linear_tuning_path,
        ridge_summary_path,
    ):
        if optional_path is not None:
            source_paths.append(optional_path)
    if config_path is not None:
        source_paths.append(config_path)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels, train_targets, all_counts = _train_targets(train_path)
    vocabulary = {
        str(value)
        for value in pq.read_table(vocabulary_path, columns=["subcategory"])[
            "subcategory"
        ].to_pylist()
    }
    if not set(labels).issubset(vocabulary):
        raise ValueError("Static Train labels fall outside the vocabulary")
    if history_input == "behavior_titles":
        history_inputs, history_matrix, history_embedding_metadata = (
            load_nrms_title_embeddings(history_embedding_dir)
        )
    else:
        history_inputs, history_matrix, history_embedding_metadata = (
            load_nrms_interest_embeddings(history_embedding_dir)
        )
    input_to_id = {
        value: index for index, value in enumerate(history_inputs)
    }
    if len(input_to_id) != len(history_inputs):
        raise ValueError("frozen History input cache contains duplicates")
    padding_id = len(history_inputs)
    if data_loader == "semantic_users":
        if benchmark_dir is None:
            raise AssertionError("semantic benchmark was not resolved")
        train_splits, dev, data_audit = _load_history_splits(
            benchmark_dir=benchmark_dir,
            train_targets=train_targets,
            labels=labels,
            title_to_id=input_to_id,
            boundaries=boundaries,
            seed=seed,
            padding_id=padding_id,
        )
    else:
        train_splits, dev, data_audit = _load_dataset_row_splits(
            train_path=train_path,
            dev_path=dev_path,
            train_targets=train_targets,
            labels=labels,
            input_to_id=input_to_id,
            history_input=history_input,
            boundaries=boundaries,
            seed=seed,
            padding_id=padding_id,
        )
    fit = train_splits["model_fit"]
    tune = train_splits["tune"]
    holdout = train_splits["holdout"]
    if fit.user_count + tune.user_count + holdout.user_count != len(train_targets):
        raise ValueError("internal Train split counts do not sum to Static Train")
    expected = dict(config.get("expected_counts", {}))
    actual_counts = {
        "train_users": len(train_targets),
        "model_fit_users": fit.user_count,
        "tune_users": tune.user_count,
        "holdout_users": holdout.user_count,
        "labels": len(labels),
        "dev_evaluation_units": dev.user_count,
        "dev_users": len(
            {user_id for bucket in dev.buckets for user_id in bucket.user_ids}
        ),
        "dev_target_occurrences": int(
            sum(
                int(bucket.positives.sum())
                for bucket in dev.buckets
                if bucket.positives is not None
            )
        ),
        "dev_supported_interests": int(
            np.concatenate(
                [
                    bucket.positives
                    for bucket in dev.buckets
                    if bucket.positives is not None
                ],
                axis=0,
            ).any(axis=0).sum()
        ),
    }
    for name, expected_value in expected.items():
        if name not in actual_counts:
            raise ValueError(f"unknown expected count: {name}")
        if actual_counts[name] != int(expected_value):
            raise ValueError(
                f"unexpected {name}: {actual_counts[name]} != "
                f"{int(expected_value)}"
            )
    fit_view = _evaluation_view(fit)
    tune_view = _evaluation_view(tune)
    holdout_view = _evaluation_view(holdout)
    dev_view = _evaluation_view(dev)
    if not isinstance(fit_view, SingleTargetSplit):
        raise AssertionError("fit must be single-target")
    fit_counts = np.bincount(fit_view.targets, minlength=len(labels))
    tune_global = _annotate_evaluation(
        _evaluate_global(
            tune_view, _global_scores(fit_counts), labels, k=k
        ),
        tune,
    )

    floor_spec = selection.get("user_recall_floor")
    user_recall_floor: float | None
    user_recall_floor_source: str | None = None
    if floor_spec is None:
        user_recall_floor = None
    elif isinstance(floor_spec, (int, float)):
        user_recall_floor = float(floor_spec)
        user_recall_floor_source = "fixed config value"
    elif floor_spec == "selected Linear Ranker Tune User-Macro Recall@10":
        if linear_tuning_path is None:
            raise ValueError(
                "Linear Ranker tuning output is required by user_recall_floor"
            )
        linear_tuning = _load_json(linear_tuning_path)
        linear_selected_tune = dict(linear_tuning["selected"])
        user_recall_floor = float(
            linear_selected_tune["user_macro_recall_at_10"]
        )
        user_recall_floor_source = portable_path(linear_tuning_path)
    else:
        raise ValueError(f"unsupported user_recall_floor: {floor_spec!r}")

    current_device = torch.device(device)
    frozen_matrix = np.concatenate(
        [
            history_matrix,
            np.zeros((1, history_matrix.shape[1]), dtype=np.float32),
        ],
        axis=0,
    )
    title_tensor = torch.as_tensor(
        frozen_matrix, dtype=torch.float32, device=current_device
    )
    fit_device = _device_split(fit, device)
    tune_device = _device_split(tune, device)

    architecture_records: list[dict[str, object]] = []
    architecture_histories: dict[str, list[dict[str, float]]] = {}
    for trial_index, architecture in enumerate(architectures):
        trial = Trial(
            architecture,
            architecture_beta,
            architecture_lr,
            architecture_wd,
        )
        records, _, history_rows = _fit_trial(
            trial=trial,
            train=fit,
            train_device=fit_device,
            tune=tune,
            tune_device=tune_device,
            title_embeddings=title_tensor,
            padding_id=padding_id,
            class_counts=fit_counts,
            epochs=architecture_epochs,
            evaluation_epochs=architecture_evaluation_epochs,
            batch_sizes=batch_sizes,
            gradient_clip_norm=gradient_clip_norm,
            precision=precision,
            seed=seed,
            device=device,
            labels=labels,
            k=k,
        )
        architecture_records.extend(records)
        architecture_histories[f"architecture_{trial_index:02d}"] = history_rows
        print(
            "NRMS architecture metrics: "
            + json.dumps(records, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print(
            f"NRMS architecture search: {trial_index + 1}/{len(architectures)}",
            flush=True,
        )
    selected_architecture_row, architecture_frontier = _select_configuration(
        architecture_records,
        global_recall=float(tune_global.metrics["interest_macro_recall_at_10"]),
        # The architecture stage holds beta fixed. Apply any configured
        # User-Recall guard only after the optimization-stage search.
        user_recall_floor=None,
        recall_tolerance=recall_tolerance,
    )
    selected_architecture = _row_to_trial(
        selected_architecture_row
    ).architecture

    optimization_trials = [
        Trial(selected_architecture, beta, learning_rate, weight_decay)
        for beta in betas
        for learning_rate in learning_rates
        for weight_decay in weight_decays
    ]
    optimization_records: list[dict[str, object]] = []
    optimization_histories: dict[str, list[dict[str, float]]] = {}
    optimization_checkpoints: dict[
        tuple[tuple[float | int, ...], int], dict[str, Any]
    ] = {}
    for trial_index, trial in enumerate(optimization_trials):
        records, checkpoints, history_rows = _fit_trial(
            trial=trial,
            train=fit,
            train_device=fit_device,
            tune=tune,
            tune_device=tune_device,
            title_embeddings=title_tensor,
            padding_id=padding_id,
            class_counts=fit_counts,
            epochs=optimization_epochs,
            evaluation_epochs=optimization_evaluation_epochs,
            batch_sizes=batch_sizes,
            gradient_clip_norm=gradient_clip_norm,
            precision=precision,
            seed=seed,
            device=device,
            labels=labels,
            k=k,
        )
        optimization_records.extend(records)
        optimization_histories[f"optimization_{trial_index:02d}"] = history_rows
        for epoch, state in checkpoints.items():
            optimization_checkpoints[(_trial_key(trial), epoch)] = state
        print(
            "NRMS optimization metrics: "
            + json.dumps(records, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print(
            f"NRMS optimization search: {trial_index + 1}/{len(optimization_trials)}",
            flush=True,
        )
    selected, optimization_frontier = _select_configuration(
        optimization_records,
        global_recall=float(tune_global.metrics["interest_macro_recall_at_10"]),
        user_recall_floor=user_recall_floor,
        recall_tolerance=recall_tolerance,
    )
    selected_trial = _row_to_trial(selected)
    selected_epoch = int(selected["epoch"])
    selected_internal_state = optimization_checkpoints[
        (_trial_key(selected_trial), selected_epoch)
    ]

    internal_model = NRMSInterestRanker.build(
        title_tensor,
        padding_id=padding_id,
        label_count=len(labels),
        architecture=selected_trial.architecture,
    ).to(current_device)
    internal_model.title_embeddings = title_tensor
    internal_model.load_state_dict(selected_internal_state)
    holdout_device = _device_split(holdout, device)
    selected_holdout = _score_split(
        internal_model,
        holdout,
        holdout_device,
        batch_sizes=batch_sizes,
        precision=precision,
        device=device,
        labels=labels,
        k=k,
    )
    holdout_global = _annotate_evaluation(
        _evaluate_global(
            holdout_view, _global_scores(fit_counts), labels, k=k
        ),
        holdout,
    )
    del internal_model, holdout_device, fit_device, tune_device
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    all_train = _merge_splits("all_train", (fit, tune, holdout))
    all_device = _device_split(all_train, device)
    final_records, final_checkpoints, final_history = _fit_trial(
        trial=selected_trial,
        train=all_train,
        train_device=all_device,
        tune=None,
        tune_device=None,
        title_embeddings=title_tensor,
        padding_id=padding_id,
        class_counts=all_counts,
        epochs=selected_epoch,
        evaluation_epochs=(selected_epoch,),
        batch_sizes=batch_sizes,
        gradient_clip_norm=gradient_clip_norm,
        precision=precision,
        seed=seed,
        device=device,
        labels=labels,
        k=k,
    )
    if final_records:
        raise AssertionError("final training evaluated a tune split")
    final_state = final_checkpoints[selected_epoch]
    final_model = NRMSInterestRanker.build(
        title_tensor,
        padding_id=padding_id,
        label_count=len(labels),
        architecture=selected_trial.architecture,
    ).to(current_device)
    final_model.title_embeddings = title_tensor
    final_model.load_state_dict(final_state)
    dev_device = _device_split(dev, device)
    selected_dev = _score_split(
        final_model,
        dev,
        dev_device,
        batch_sizes=batch_sizes,
        precision=precision,
        device=device,
        labels=labels,
        k=k,
    )
    dev_global = _annotate_evaluation(
        _evaluate_global(
            dev_view, _global_scores(all_counts), labels, k=k
        ),
        dev,
    )

    comparison_metrics: dict[str, dict[str, dict[str, object]]] = {
        "internal_holdout": {},
        "dev_exploratory": {},
    }
    if linear_summary_path is not None:
        comparison_metrics["internal_holdout"]["linear_ranker"] = (
            _metrics_from_summary(
                linear_summary_path,
                "internal_holdout",
                "selected_linear_ranker",
            )
        )
        comparison_metrics["dev_exploratory"]["linear_ranker"] = (
            _metrics_from_summary(
                linear_summary_path,
                "dev_exploratory",
                "selected_linear_ranker",
            )
        )
    if ridge_summary_path is not None:
        comparison_metrics["internal_holdout"]["balanced_ridge"] = (
            _metrics_from_summary(
                ridge_summary_path,
                "internal_holdout",
                "selected_balanced_ridge",
            )
        )
        comparison_metrics["dev_exploratory"]["balanced_ridge"] = (
            _metrics_from_summary(
                ridge_summary_path,
                "dev_exploratory",
                "selected_balanced_ridge",
            )
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-user-disjoint-nrms-", dir=str(output_dir.parent)
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        _save_model(
            staging / "selected_model.npz",
            final_state,
            labels=labels,
            selected=selected,
        )
        tuning = {
            "selection_scope": "Static Train internal tune only",
            "user_recall_floor_source": user_recall_floor_source,
            "user_recall_floor": user_recall_floor,
            "tune_global": tune_global.metrics,
            "architecture_stage": {
                "trial_count": len(architectures),
                "evaluated_checkpoint_count": len(architecture_records),
                "selected": selected_architecture_row,
                "pareto_frontier": architecture_frontier,
                "grid": architecture_records,
                "training_histories": architecture_histories,
            },
            "optimization_stage": {
                "trial_count": len(optimization_trials),
                "evaluated_checkpoint_count": len(optimization_records),
                "selected": selected,
                "pareto_frontier": optimization_frontier,
                "grid": optimization_records,
                "training_histories": optimization_histories,
            },
            "final_training_history": final_history,
        }
        (staging / "tuning.json").write_text(
            json.dumps(tuning, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        conditions = {
            "internal_holdout": {
                "global_popularity": _write_nrms_condition(
                    staging,
                    "internal_holdout",
                    "global_popularity",
                    holdout,
                    holdout_global,
                    labels,
                ),
                "selected_nrms": _write_nrms_condition(
                    staging,
                    "internal_holdout",
                    "selected_nrms",
                    holdout,
                    selected_holdout,
                    labels,
                ),
            },
            "dev_exploratory": {
                "global_popularity": _write_nrms_condition(
                    staging,
                    "dev_exploratory",
                    "global_popularity",
                    dev,
                    dev_global,
                    labels,
                ),
                "selected_nrms": _write_nrms_condition(
                    staging,
                    "dev_exploratory",
                    "selected_nrms",
                    dev,
                    selected_dev,
                    labels,
                ),
            },
        }
        audit = {
            **data_audit,
            "train_users": all_train.user_count,
            "dev_distinct_positive_pairs": int(
                dev_view.positives.sum()
                if isinstance(dev_view, MultiTargetSplit)
                else 0
            ),
            "supported_train_interests": len(labels),
            "supported_dev_interests": int(
                dev_view.positives.any(axis=0).sum()
                if isinstance(dev_view, MultiTargetSplit)
                else 0
            ),
            "evaluation_unit": dev.unit,
            "expected_counts": expected,
            "actual_counts": actual_counts,
            "history_policy": str(
                data_config.get(
                    "history_policy",
                    "complete raw static History; no replay and no truncation",
                )
            ),
            "runner_history_truncation": False,
            "padding_id": padding_id,
            "history_input": history_input,
            "unique_frozen_history_inputs": len(history_inputs),
            "history_embedding_dimensions": int(history_matrix.shape[1]),
            **(
                {
                    "unique_frozen_titles": len(history_inputs),
                    "title_dimensions": int(history_matrix.shape[1]),
                }
                if history_input == "behavior_titles"
                else {}
            ),
            "top10_unique": bool(
                all(
                    len(set(row.tolist())) == k
                    for result in (selected_holdout, selected_dev)
                    for row in result.top_k
                )
            ),
            "history_leakage_in_top10": bool(
                any(
                    view.history_mask[index, result.top_k[index]].any()
                    for view, result in (
                        (holdout_view, selected_holdout),
                        (dev_view, selected_dev),
                    )
                    for index in range(len(view.user_ids))
                )
            ),
        }
        if not audit["top10_unique"] or audit["history_leakage_in_top10"]:
            raise ValueError("NRMS Top-10 audit failed")
        (staging / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        resolved_rule_version = str(config.get("rule_version", RULE_VERSION))
        resolved = {
            **config,
            "rule_version": resolved_rule_version,
            "dataset_dir": portable_path(dataset_dir),
            "history_input": history_input,
            "history_embedding_dir": portable_path(history_embedding_dir),
            "output_dir": portable_path(output_dir),
            "resolved_device": device,
            "resolved_precision": precision,
            "history_encoder": history_embedding_metadata["model"],
            "history_encoder_status": (
                "frozen; one cached vector per History title"
                if history_input == "behavior_titles"
                else "frozen; one cached vector per interest name"
            ),
            **(
                {
                    "title_embedding_dir": portable_path(
                        history_embedding_dir
                    ),
                    "title_encoder": history_embedding_metadata["model"],
                    "title_encoder_status": (
                        "frozen; one cached vector per title"
                    ),
                }
                if history_input == "behavior_titles"
                else {}
            ),
            "history": audit["history_policy"],
            "score": (
                "NRMS user encoder followed by a bias-free "
                f"{len(labels)}-way classifier"
            ),
            "explicit_global_popularity_feature": False,
            "training_candidate_mask": "exclude all Train-supported History interests",
            "supervision": "one selected Static Train target per user",
        }
        if benchmark_dir is not None:
            resolved["semantic_benchmark_dir"] = portable_path(benchmark_dir)
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_splits = {
            "internal_holdout": {
                "global_popularity": holdout_global.metrics,
                **comparison_metrics["internal_holdout"],
                "nrms": selected_holdout.metrics,
            },
            "dev_exploratory": {
                "global_popularity": dev_global.metrics,
                **comparison_metrics["dev_exploratory"],
                "nrms": selected_dev.metrics,
            },
        }
        deltas_vs_global = {
            split_name: {
                metric: float(values["nrms"][metric])
                - float(values["global_popularity"][metric])
                for metric in (
                    "interest_macro_recall_at_10",
                    "interest_macro_ndcg_at_10",
                )
            }
            for split_name, values in summary_splits.items()
        }
        summary = {
            "selected": selected,
            "conditions": conditions,
            **summary_splits,
            "nrms_minus_global": deltas_vs_global,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "environment.json").write_text(
            json.dumps(
                _environment_metadata(
                    torch, device=device, precision=precision
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "rule_version": resolved_rule_version,
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
            "Tune and evaluate a frozen-MiniLM NRMS-style full-taxonomy "
            "new-interest ranker with Train-only selection."
        )
    )
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = run_static_user_disjoint_nrms(
        args.config, overwrite=args.overwrite
    )
    print(output)


if __name__ == "__main__":
    main()
