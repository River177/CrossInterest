from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

TASK_TYPES = ("behavior_only", "interest_only")
SPLITS = ("train", "dev")
RULE_VERSION = "generative-new-interest-v1"

RESPONSE_PREFIX = "New topic: "

BEHAVIOR_PROMPT_TEMPLATE = """The headlines of the news items I recently viewed are listed below:

{history}

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it."""

INTEREST_PROMPT_TEMPLATE = """The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

{history}

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it."""

BASE_OUTPUT_RELATIVE_PATHS = (
    Path("behavior_only/train.parquet"),
    Path("behavior_only/dev.parquet"),
    Path("interest_only/train.parquet"),
    Path("interest_only/dev.parquet"),
    Path("statistics.json"),
    Path("metadata.json"),
)

SAMPLING_OUTPUT_RELATIVE_PATHS = (
    Path("selected_train_samples.parquet"),
    Path("retained_interests.parquet"),
)

MESSAGE_TYPE = pa.struct(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
    ]
)

GENERATIVE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("response", pa.string(), nullable=False),
        pa.field("messages", pa.list_(MESSAGE_TYPE), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
    ]
)

SELECTED_TRAIN_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("target_selection_priority", pa.string(), nullable=False),
        pa.field("balance_sampling_priority", pa.string(), nullable=False),
        pa.field("sampling_seed", pa.int64(), nullable=False),
    ]
)

RETAINED_INTEREST_SCHEMA = pa.schema(
    [
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("source_train_count", pa.int64(), nullable=False),
        pa.field("selected_train_count", pa.int32(), nullable=False),
        pa.field("filtered_dev_count", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    behavior_train_samples: int
    behavior_dev_samples: int
    interest_train_samples: int
    interest_dev_samples: int


@dataclass(frozen=True)
class SamplingSelection:
    selected_source_ids: frozenset[str]
    retained_interests: frozenset[str]
    source_train_counts: dict[str, int]
    manifest_rows: list[dict[str, object]]


class BatchedParquetWriter:
    def __init__(self, path: Path, batch_size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(
            path,
            GENERATIVE_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        self.row_count = 0

    def append(self, row: dict[str, object]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=GENERATIVE_SCHEMA)
        self._writer.write_table(table, row_group_size=self._batch_size)
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


class SplitStatistics:
    def __init__(self) -> None:
        self.samples = 0
        self.source_history_lengths: Counter[int] = Counter()
        self.history_lengths: Counter[int] = Counter()
        self.prompt_lengths: Counter[int] = Counter()
        self.response_counts: Counter[str] = Counter()
        self.target_subcategory_counts: Counter[str] = Counter()
        self.truncated_samples = 0
        self.empty_titles = 0
        self.empty_interests = 0
        self.empty_responses = 0
        self.max_prompt_characters = 0
        self.max_prompt_source_sample_id: str | None = None

    def add(
        self,
        *,
        source_sample_id: str,
        source_history_length: int,
        history_length: int,
        prompt: str,
        response: str,
        target_subcategory: str,
        empty_items: int,
    ) -> None:
        self.samples += 1
        self.source_history_lengths[source_history_length] += 1
        self.history_lengths[history_length] += 1
        self.prompt_lengths[len(prompt)] += 1
        self.response_counts[response] += 1
        self.target_subcategory_counts[target_subcategory] += 1
        self.truncated_samples += int(history_length < source_history_length)
        self.empty_responses += int(not response)
        if len(prompt) > self.max_prompt_characters:
            self.max_prompt_characters = len(prompt)
            self.max_prompt_source_sample_id = source_sample_id
        if empty_items:
            self.empty_titles += empty_items

    def as_dict(self, task_type: str) -> dict[str, object]:
        if task_type == "interest_only":
            self.empty_interests = self.empty_titles
            self.empty_titles = 0
        return {
            "samples": self.samples,
            "source_history_length": _distribution_summary(
                self.source_history_lengths
            ),
            "history_length": _distribution_summary(self.history_lengths),
            "truncated_samples": self.truncated_samples,
            "prompt_characters": _distribution_summary(self.prompt_lengths),
            "response_counts": dict(
                sorted(
                    self.response_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "target_subcategory_counts": dict(
                sorted(
                    self.target_subcategory_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "max_prompt_characters": self.max_prompt_characters,
            "max_prompt_source_sample_id": self.max_prompt_source_sample_id,
            "empty_titles": self.empty_titles,
            "empty_interests": self.empty_interests,
            "empty_responses": self.empty_responses,
        }


def _distribution_summary(
    counts: Counter[int],
) -> dict[str, float | int | None]:
    total = sum(counts.values())
    if total == 0:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }

    def percentile(fraction: float) -> int:
        rank = max(1, int(total * fraction + 0.999999))
        cumulative = 0
        for value in sorted(counts):
            cumulative += counts[value]
            if cumulative >= rank:
                return value
        raise AssertionError("Unreachable percentile calculation")

    weighted_sum = sum(value * count for value, count in counts.items())
    return {
        "count": total,
        "min": min(counts),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(counts),
        "mean": weighted_sum / total,
    }


def _format_numbered_history(items: list[str]) -> str:
    return "\n".join(
        f"{index}. {item}" for index, item in enumerate(items, start=1)
    )


def _format_prompt(task_type: str, items: list[str]) -> str:
    template = (
        BEHAVIOR_PROMPT_TEMPLATE
        if task_type == "behavior_only"
        else INTEREST_PROMPT_TEMPLATE
    )
    return template.format(history=_format_numbered_history(items))


def _make_row(
    source: dict[str, object],
    *,
    task_type: str,
    history_items: list[str],
) -> dict[str, object]:
    source_sample_id = str(source["sample_id"])
    target_interest = source["target_interest"]
    if not isinstance(target_interest, dict):
        raise ValueError(f"{source_sample_id}: invalid target_interest")
    target_subcategory = str(target_interest["subcategory"])
    response = f"{RESPONSE_PREFIX}{target_subcategory}"
    prompt = _format_prompt(task_type, history_items)
    return {
        "sample_id": f"{source_sample_id}:{task_type}",
        "source_sample_id": source_sample_id,
        "split": str(source["split"]),
        "task_type": task_type,
        "impression_id": str(source["impression_id"]),
        "user_id": str(source["user_id"]),
        "timestamp": source["timestamp"],
        "history_length": len(history_items),
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target_subcategory,
    }


def _validate_source_schema(path: Path) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    required = {
        "sample_id",
        "split",
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "target_interest",
    }
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")


def _stable_priority(
    seed: int,
    namespace: str,
    source_sample_id: str,
) -> str:
    value = f"{seed}:{namespace}:{source_sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _sampling_priority(seed: int, source_sample_id: str) -> str:
    return _stable_priority(seed, "balance", source_sample_id)


def _target_selection_priority(seed: int, source_sample_id: str) -> str:
    return _stable_priority(seed, "target", source_sample_id)


def _iter_source_rows(
    source_path: Path,
    *,
    columns: list[str],
    batch_size: int,
    one_target_per_impression: bool,
    sampling_seed: int,
):
    parquet = pq.ParquetFile(source_path)
    if not one_target_per_impression:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            yield from batch.to_pylist()
        return

    if "impression_id" not in columns or "sample_id" not in columns:
        raise ValueError(
            "one-target iteration requires impression_id and sample_id"
        )

    current_impression: str | None = None
    candidates: list[dict[str, object]] = []
    closed_impressions: set[str] = set()

    def choose_target() -> dict[str, object]:
        return min(
            candidates,
            key=lambda row: (
                _target_selection_priority(
                    sampling_seed,
                    str(row["sample_id"]),
                ),
                str(row["sample_id"]),
            ),
        )

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        for row in batch.to_pylist():
            impression_id = str(row["impression_id"])
            if current_impression is None:
                current_impression = impression_id
            elif impression_id != current_impression:
                closed_impressions.add(current_impression)
                if impression_id in closed_impressions:
                    raise ValueError(
                        f"{source_path}: impression {impression_id!r} "
                        "appears in non-contiguous groups"
                    )
                yield choose_target()
                candidates = []
                current_impression = impression_id
            candidates.append(row)

    if candidates:
        yield choose_target()


def _select_train_samples(
    source_path: Path,
    *,
    samples_per_class: int,
    min_class_count: int,
    sampling_seed: int,
    batch_size: int,
    one_target_per_impression: bool,
) -> SamplingSelection:
    counts: Counter[str] = Counter()
    heaps: dict[
        str,
        list[tuple[int, str, str, str, str]],
    ] = {}

    rows = _iter_source_rows(
        source_path,
        columns=[
            "sample_id",
            "impression_id",
            "target_interest",
        ],
        batch_size=batch_size,
        one_target_per_impression=one_target_per_impression,
        sampling_seed=sampling_seed,
    )
    for row in rows:
        source_sample_id = str(row["sample_id"])
        impression_id = str(row["impression_id"])
        target_interest = row["target_interest"]
        if not isinstance(target_interest, dict):
            raise ValueError(
                f"{source_sample_id}: invalid target_interest"
            )
        target_subcategory = str(target_interest["subcategory"])
        counts[target_subcategory] += 1
        balance_priority = _sampling_priority(
            sampling_seed,
            source_sample_id,
        )
        target_priority = _target_selection_priority(
            sampling_seed,
            source_sample_id,
        )
        priority_value = int(balance_priority, 16)
        heap = heaps.setdefault(target_subcategory, [])
        item = (
            -priority_value,
            source_sample_id,
            impression_id,
            target_priority,
            balance_priority,
        )
        if len(heap) < samples_per_class:
            heapq.heappush(heap, item)
        elif priority_value < -heap[0][0]:
            heapq.heapreplace(heap, item)

    retained_interests = frozenset(
        target
        for target, count in counts.items()
        if count >= min_class_count
    )
    manifest_rows: list[dict[str, object]] = []
    for target_subcategory in sorted(retained_interests):
        selected = heaps[target_subcategory]
        if len(selected) != samples_per_class:
            raise ValueError(
                f"{target_subcategory}: selected {len(selected)} samples, "
                f"expected {samples_per_class}"
            )
        for (
            _,
            source_sample_id,
            impression_id,
            target_priority,
            balance_priority,
        ) in selected:
            manifest_rows.append(
                {
                    "source_sample_id": source_sample_id,
                    "impression_id": impression_id,
                    "target_subcategory": target_subcategory,
                    "target_selection_priority": target_priority,
                    "balance_sampling_priority": balance_priority,
                    "sampling_seed": sampling_seed,
                }
            )
    manifest_rows.sort(
        key=lambda row: (
            str(row["target_subcategory"]),
            str(row["balance_sampling_priority"]),
            str(row["source_sample_id"]),
        )
    )
    selected_source_ids = frozenset(
        str(row["source_sample_id"]) for row in manifest_rows
    )
    expected = len(retained_interests) * samples_per_class
    if len(selected_source_ids) != expected:
        raise ValueError(
            f"Selected {len(selected_source_ids)} unique Train samples, "
            f"expected {expected}"
        )
    return SamplingSelection(
        selected_source_ids=selected_source_ids,
        retained_interests=retained_interests,
        source_train_counts=dict(counts),
        manifest_rows=manifest_rows,
    )


def _build_split(
    split: str,
    source_path: Path,
    staging: Path,
    batch_size: int,
    *,
    selected_source_ids: frozenset[str] | None = None,
    allowed_subcategories: frozenset[str] | None = None,
    history_limit: int | None = None,
    one_target_per_impression: bool = False,
    sampling_seed: int = 20260806,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, int],
    Counter[str],
]:
    _validate_source_schema(source_path)
    writers = {
        task_type: BatchedParquetWriter(
            staging / task_type / f"{split}.parquet",
            batch_size,
        )
        for task_type in TASK_TYPES
    }
    stats = {task_type: SplitStatistics() for task_type in TASK_TYPES}
    raw_source_rows = pq.ParquetFile(source_path).metadata.num_rows
    candidate_source_rows = 0
    output_rows = 0
    output_target_counts: Counter[str] = Counter()

    try:
        rows = _iter_source_rows(
            source_path,
            columns=[
                "sample_id",
                "split",
                "impression_id",
                "user_id",
                "timestamp",
                "history",
                "target_interest",
            ],
            batch_size=batch_size,
            one_target_per_impression=one_target_per_impression,
            sampling_seed=sampling_seed,
        )
        for source in rows:
            candidate_source_rows += 1
            source_sample_id = str(source["sample_id"])
            if source["split"] != split:
                raise ValueError(
                    f"{source_sample_id}: expected split {split!r}, "
                    f"got {source['split']!r}"
                )
            target_interest = source["target_interest"]
            if not isinstance(target_interest, dict):
                raise ValueError(
                    f"{source_sample_id}: invalid target_interest"
                )
            target_subcategory = str(target_interest["subcategory"])
            if not target_subcategory:
                raise ValueError(f"{source_sample_id}: empty response")
            if (
                selected_source_ids is not None
                and source_sample_id not in selected_source_ids
            ):
                continue
            if (
                allowed_subcategories is not None
                and target_subcategory not in allowed_subcategories
            ):
                continue

            source_history = source["history"]
            if not isinstance(source_history, list) or not source_history:
                raise ValueError(
                    f"{source_sample_id}: history must be a non-empty list"
                )
            history = (
                source_history[-history_limit:]
                if history_limit is not None
                else source_history
            )
            history_items = {
                "behavior_only": [str(item["title"]) for item in history],
                "interest_only": [
                    str(item["subcategory"]) for item in history
                ],
            }
            if target_subcategory in history_items["interest_only"]:
                raise ValueError(
                    f"{source_sample_id}: target subcategory appears in "
                    "the interest-only history"
                )

            for task_type in TASK_TYPES:
                items = history_items[task_type]
                row = _make_row(
                    source,
                    task_type=task_type,
                    history_items=items,
                )
                writers[task_type].append(row)
                stats[task_type].add(
                    source_sample_id=source_sample_id,
                    source_history_length=len(source_history),
                    history_length=len(items),
                    prompt=str(row["prompt"]),
                    response=str(row["response"]),
                    target_subcategory=target_subcategory,
                    empty_items=sum(not item for item in items),
                )
            output_target_counts[target_subcategory] += 1
            output_rows += 1

            if candidate_source_rows % 250_000 == 0:
                LOGGER.info(
                    "Converted %s %s source samples",
                    f"{candidate_source_rows:,}",
                    split,
                )
    finally:
        for writer in writers.values():
            writer.close()

    for task_type, writer in writers.items():
        if writer.row_count != output_rows:
            raise ValueError(
                f"{split}/{task_type}: wrote {writer.row_count} rows for "
                f"{output_rows} selected source rows"
            )

    return (
        {
            task_type: stats[task_type].as_dict(task_type)
            for task_type in TASK_TYPES
        },
        {
            "input_source_rows": raw_source_rows,
            "one_target_rows": candidate_source_rows,
            "target_selection_removed_samples": (
                raw_source_rows - candidate_source_rows
            ),
            "output_samples": output_rows,
            "class_filter_removed_samples": (
                candidate_source_rows - output_rows
            ),
            "filtered_samples": raw_source_rows - output_rows,
        },
        output_target_counts,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_output_available(
    output_dir: Path,
    output_relative_paths: tuple[Path, ...],
    overwrite: bool,
) -> None:
    existing = [
        output_dir / relative_path
        for relative_path in output_relative_paths
        if (output_dir / relative_path).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}: "
            f"{[str(path.relative_to(output_dir)) for path in existing]}. "
            "Use --overwrite."
        )


def build_generative_dataset(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    batch_size: int = 5_000,
    overwrite: bool = False,
    history_limit: int | None = None,
    train_samples_per_class: int | None = None,
    train_min_class_count: int | None = None,
    filter_dev_to_train_classes: bool = False,
    sampling_seed: int = 20260806,
    one_target_per_train_impression: bool = False,
) -> BuildResult:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if history_limit is not None and history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if train_samples_per_class is not None and train_samples_per_class <= 0:
        raise ValueError("train_samples_per_class must be positive")
    if train_min_class_count is not None and train_samples_per_class is None:
        raise ValueError(
            "train_min_class_count requires train_samples_per_class"
        )
    if filter_dev_to_train_classes and train_samples_per_class is None:
        raise ValueError(
            "filter_dev_to_train_classes requires train_samples_per_class"
        )
    if train_samples_per_class is not None:
        if train_min_class_count is None:
            train_min_class_count = train_samples_per_class
        if train_min_class_count < train_samples_per_class:
            raise ValueError(
                "train_min_class_count must be at least "
                "train_samples_per_class"
            )

    source_paths = {
        split: source_dir / f"{split}.parquet" for split in SPLITS
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    balancing_enabled = train_samples_per_class is not None
    output_relative_paths = BASE_OUTPUT_RELATIVE_PATHS + (
        SAMPLING_OUTPUT_RELATIVE_PATHS if balancing_enabled else ()
    )
    _ensure_output_available(output_dir, output_relative_paths, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    selection: SamplingSelection | None = None
    if balancing_enabled:
        LOGGER.info("Selecting deterministic balanced Train samples")
        selection = _select_train_samples(
            source_paths["train"],
            samples_per_class=train_samples_per_class,
            min_class_count=train_min_class_count,
            sampling_seed=sampling_seed,
            batch_size=batch_size,
            one_target_per_impression=(
                one_target_per_train_impression
            ),
        )
        LOGGER.info(
            "Retained %s classes and %s Train samples",
            f"{len(selection.retained_interests):,}",
            f"{len(selection.selected_source_ids):,}",
        )

    with tempfile.TemporaryDirectory(
        prefix=".generative-new-interest-output-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        statistics: dict[str, object] = {
            "rule_version": RULE_VERSION,
            "tasks": {task_type: {} for task_type in TASK_TYPES},
            "source_filtering": {},
        }
        output_target_counts: dict[str, Counter[str]] = {}

        for split in SPLITS:
            LOGGER.info("Converting %s source data", split)
            split_stats, filtering_stats, target_counts = _build_split(
                split,
                source_paths[split],
                staging,
                batch_size,
                selected_source_ids=(
                    selection.selected_source_ids
                    if split == "train" and selection is not None
                    else None
                ),
                allowed_subcategories=(
                    selection.retained_interests
                    if (
                        split == "dev"
                        and filter_dev_to_train_classes
                        and selection is not None
                    )
                    else None
                ),
                history_limit=history_limit,
                one_target_per_impression=(
                    split == "train"
                    and one_target_per_train_impression
                ),
                sampling_seed=sampling_seed,
            )
            for task_type in TASK_TYPES:
                statistics["tasks"][task_type][split] = split_stats[task_type]
            statistics["source_filtering"][split] = filtering_stats
            output_target_counts[split] = target_counts

        if selection is not None:
            pq.write_table(
                pa.Table.from_pylist(
                    selection.manifest_rows,
                    schema=SELECTED_TRAIN_SCHEMA,
                ),
                staging / "selected_train_samples.parquet",
                compression="zstd",
                use_dictionary=True,
            )
            retained_rows = [
                {
                    "target_subcategory": target_subcategory,
                    "source_train_count": selection.source_train_counts[
                        target_subcategory
                    ],
                    "selected_train_count": output_target_counts["train"][
                        target_subcategory
                    ],
                    "filtered_dev_count": output_target_counts["dev"][
                        target_subcategory
                    ],
                }
                for target_subcategory in sorted(
                    selection.retained_interests
                )
            ]
            pq.write_table(
                pa.Table.from_pylist(
                    retained_rows,
                    schema=RETAINED_INTEREST_SCHEMA,
                ),
                staging / "retained_interests.parquet",
                compression="zstd",
                use_dictionary=True,
            )
            statistics["sampling"] = {
                "algorithm": "sha256-priority",
                "sampling_seed": sampling_seed,
                "train_samples_per_class": train_samples_per_class,
                "train_min_class_count": train_min_class_count,
                "retained_interest_count": len(
                    selection.retained_interests
                ),
                "selected_train_samples": len(
                    selection.selected_source_ids
                ),
                "filter_dev_to_train_classes": (
                    filter_dev_to_train_classes
                ),
                "one_target_per_train_impression": (
                    one_target_per_train_impression
                ),
            }

        statistics_path = staging / "statistics.json"
        statistics_path.write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_paths = [
            Path(task_type) / f"{split}.parquet"
            for task_type in TASK_TYPES
            for split in SPLITS
        ]
        artifact_paths.append(Path("statistics.json"))
        if balancing_enabled:
            artifact_paths.extend(SAMPLING_OUTPUT_RELATIVE_PATHS)
        artifact_hashes = {
            str(path): _sha256(staging / path) for path in artifact_paths
        }
        source_hashes = {
            f"{split}.parquet": _sha256(path)
            for split, path in source_paths.items()
        }

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "definition": (
                "Generate one exact second-level new-interest name from a "
                "user's available history without candidates."
            ),
            "source_dir": str(source_dir.resolve()),
            "full_history": history_limit is None,
            "history_truncation": (
                None
                if history_limit is None
                else {
                    "limit": history_limit,
                    "direction": "most_recent",
                    "operation": "history[-limit:]",
                }
            ),
            "interest_history_preserves_order": True,
            "interest_history_preserves_duplicates": True,
            "first_level_interests_in_training_data": False,
            "candidate_options_used": False,
            "cot_used": False,
            "response_prefix": RESPONSE_PREFIX,
            "prompt_templates": {
                "behavior_only": BEHAVIOR_PROMPT_TEMPLATE,
                "interest_only": INTEREST_PROMPT_TEMPLATE,
            },
            "response_format": "New topic: {target_interest.subcategory}",
            "message_roles": ["user", "assistant"],
            "schema": str(GENERATIVE_SCHEMA),
            "one_target_per_train_impression": (
                one_target_per_train_impression
            ),
            "target_selection": (
                None
                if not one_target_per_train_impression
                else {
                    "algorithm": "sha256-priority",
                    "namespace": "target",
                    "sampling_seed": sampling_seed,
                    "scope": "train",
                }
            ),
            "sampling": (
                None
                if selection is None
                else {
                    "algorithm": "sha256-priority",
                    "namespace": "balance",
                    "sampling_seed": sampling_seed,
                    "train_samples_per_class": train_samples_per_class,
                    "train_min_class_count": train_min_class_count,
                    "filter_dev_to_train_classes": (
                        filter_dev_to_train_classes
                    ),
                    "retained_interests": sorted(
                        selection.retained_interests
                    ),
                }
            ),
            "source_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
        }
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for task_type in TASK_TYPES:
            (output_dir / task_type).mkdir(parents=True, exist_ok=True)
        for relative_path in output_relative_paths:
            os.replace(
                staging / relative_path,
                output_dir / relative_path,
            )

    task_stats = statistics["tasks"]
    return BuildResult(
        output_dir=output_dir,
        behavior_train_samples=int(
            task_stats["behavior_only"]["train"]["samples"]
        ),
        behavior_dev_samples=int(
            task_stats["behavior_only"]["dev"]["samples"]
        ),
        interest_train_samples=int(
            task_stats["interest_only"]["train"]["samples"]
        ),
        interest_dev_samples=int(
            task_stats["interest_only"]["dev"]["samples"]
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build generative Behavior-only and Interest-only data."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/new_interest"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/generative_new_interest"),
    )
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--train-samples-per-class", type=int)
    parser.add_argument("--train-min-class-count", type=int)
    parser.add_argument(
        "--filter-dev-to-train-classes",
        action="store_true",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260806)
    parser.add_argument(
        "--one-target-per-train-impression",
        action="store_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = build_generative_dataset(
        args.source_dir,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        history_limit=args.history_limit,
        train_samples_per_class=args.train_samples_per_class,
        train_min_class_count=args.train_min_class_count,
        filter_dev_to_train_classes=args.filter_dev_to_train_classes,
        sampling_seed=args.sampling_seed,
        one_target_per_train_impression=(
            args.one_target_per_train_impression
        ),
    )
    LOGGER.info(
        "Wrote Behavior-only Train/Dev %s/%s and Interest-only Train/Dev "
        "%s/%s samples to %s",
        f"{result.behavior_train_samples:,}",
        f"{result.behavior_dev_samples:,}",
        f"{result.interest_train_samples:,}",
        f"{result.interest_dev_samples:,}",
        result.output_dir,
    )


if __name__ == "__main__":
    main()
