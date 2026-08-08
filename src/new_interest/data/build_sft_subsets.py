from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path

from new_interest.data.build_generative_dataset import (
    GENERATIVE_SCHEMA,
    RESPONSE_PREFIX,
    TASK_TYPES,
    _format_prompt,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SIZES = (1_000, 5_000, 10_000, 22_000)
RULE_VERSION = "sft-nested-subsets-v1"

GROUPED_EVAL_SCHEMA = pa.schema(
    [
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field(
            "acceptable_responses",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "acceptable_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
    ]
)

SUBSET_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("class_rank", pa.int16(), nullable=False),
        pa.field("within_class_rank", pa.int16(), nullable=False),
        pa.field("global_rank", pa.int32(), nullable=False),
        pa.field(
            "included_sizes",
            pa.list_(pa.int32()),
            nullable=False,
        ),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_sizes: tuple[int, ...]
    grouped_dev_rows: int
    grouped_dev_subset_rows: int


class ParquetBatchWriter:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        batch_size: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(
            path,
            schema,
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
        table = pa.Table.from_pylist(self._rows, schema=self._schema)
        self._writer.write_table(table, row_group_size=self._batch_size)
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(namespace: str, value: str, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{namespace}:{value}".encode("utf-8")
    ).hexdigest()


def _size_name(size: int) -> str:
    return f"{size // 1_000}k" if size % 1_000 == 0 else str(size)


def _output_relative_paths(
    sizes: tuple[int, ...],
    eval_subset_size: int,
) -> tuple[Path, ...]:
    eval_name = _size_name(eval_subset_size)
    return (
        *(
            Path("train") / task_type / f"{_size_name(size)}.parquet"
            for task_type in TASK_TYPES
            for size in sizes
        ),
        *(
            path
            for task_type in TASK_TYPES
            for path in (
                Path("eval")
                / task_type
                / f"dev_{eval_name}_grouped.parquet",
                Path("eval") / task_type / "dev_full_grouped.parquet",
            )
        ),
        Path("subset_manifest.parquet"),
        Path("statistics.json"),
        Path("metadata.json"),
    )


def _load_train_sequence(
    balanced_dir: Path,
    *,
    seed: int,
) -> tuple[list[str], list[dict[str, object]]]:
    selected_rows = pq.read_table(
        balanced_dir / "selected_train_samples.parquet"
    ).to_pylist()
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected_rows:
        by_class[str(row["target_subcategory"])].append(row)

    class_order = sorted(
        by_class,
        key=lambda target: (_priority("subset-class", target, seed), target),
    )
    for target in class_order:
        by_class[target].sort(
            key=lambda row: (
                str(row["balance_sampling_priority"]),
                str(row["source_sample_id"]),
            )
        )

    class_lengths = {len(rows) for rows in by_class.values()}
    if len(class_lengths) != 1:
        raise ValueError(
            f"Expected equal class sizes, got {sorted(class_lengths)}"
        )

    sequence: list[dict[str, object]] = []
    per_class = next(iter(class_lengths))
    for within_class_rank in range(per_class):
        for class_rank, target in enumerate(class_order):
            source = by_class[target][within_class_rank]
            sequence.append(
                {
                    "source_sample_id": str(source["source_sample_id"]),
                    "target_subcategory": target,
                    "class_rank": class_rank,
                    "within_class_rank": within_class_rank,
                }
            )
    return class_order, sequence


def _load_train_rows(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            source_sample_id = str(row["source_sample_id"])
            if source_sample_id in rows:
                raise ValueError(
                    f"{path}: duplicate source_sample_id {source_sample_id}"
                )
            rows[source_sample_id] = row
    return rows


def _write_train_subsets(
    balanced_dir: Path,
    staging: Path,
    *,
    sizes: tuple[int, ...],
    seed: int,
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    class_order, sequence = _load_train_sequence(
        balanced_dir,
        seed=seed,
    )
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sizes must be unique and strictly increasing")
    if sizes[-1] > len(sequence):
        raise ValueError(
            f"Largest size {sizes[-1]} exceeds {len(sequence)} samples"
        )

    task_rows = {
        task_type: _load_train_rows(
            balanced_dir / task_type / "train.parquet"
        )
        for task_type in TASK_TYPES
    }
    expected_ids = set(task_rows[TASK_TYPES[0]])
    for task_type in TASK_TYPES[1:]:
        if set(task_rows[task_type]) != expected_ids:
            raise ValueError(
                f"{task_type}: Train source IDs do not match"
            )

    subset_counts: dict[str, object] = {}
    for size in sizes:
        subset_name = _size_name(size)
        selected = sequence[:size]
        selected_ids = [str(row["source_sample_id"]) for row in selected]
        selected_targets = Counter(
            str(row["target_subcategory"]) for row in selected
        )
        for task_type in TASK_TYPES:
            table = pa.Table.from_pylist(
                [task_rows[task_type][sample_id] for sample_id in selected_ids],
                schema=GENERATIVE_SCHEMA,
            )
            path = staging / "train" / task_type / f"{subset_name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                path,
                compression="zstd",
                use_dictionary=True,
                row_group_size=2_000,
            )
        subset_counts[subset_name] = {
            "samples": size,
            "classes": len(selected_targets),
            "min_per_class": min(selected_targets.values()),
            "max_per_class": max(selected_targets.values()),
            "target_counts": dict(sorted(selected_targets.items())),
        }

    manifest_rows = []
    for global_index, row in enumerate(sequence, start=1):
        manifest_rows.append(
            {
                **row,
                "global_rank": global_index - 1,
                "included_sizes": [
                    size for size in sizes if global_index <= size
                ],
            }
        )
    manifest_table = pa.Table.from_pylist(
        manifest_rows,
        schema=SUBSET_MANIFEST_SCHEMA,
    )
    pq.write_table(
        manifest_table,
        staging / "subset_manifest.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    return class_order, sequence, subset_counts


def _iter_filtered_dev_groups(
    source_path: Path,
    retained_interests: set[str],
    *,
    history_limit: int,
):
    columns = [
        "sample_id",
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "target_interest",
    ]
    current_impression: str | None = None
    group: list[dict[str, object]] = []

    def emit_group() -> tuple[dict[str, object], dict[str, object]]:
        first = group[0]
        history = first["history"]
        if not isinstance(history, list) or not history:
            raise ValueError(
                f"{first['sample_id']}: invalid history"
            )
        recent_history = history[-history_limit:]
        subcategories = [
            str(item["subcategory"]) for item in recent_history
        ]
        targets: list[str] = []
        seen_targets: set[str] = set()
        for row in group:
            target_interest = row["target_interest"]
            if not isinstance(target_interest, dict):
                raise ValueError(
                    f"{row['sample_id']}: invalid target_interest"
                )
            target = str(target_interest["subcategory"])
            if target not in seen_targets:
                targets.append(target)
                seen_targets.add(target)
        responses = [f"{RESPONSE_PREFIX}{target}" for target in targets]
        common = {
            "impression_id": str(first["impression_id"]),
            "user_id": str(first["user_id"]),
            "timestamp": first["timestamp"],
            "history_length": len(recent_history),
            "acceptable_responses": responses,
            "acceptable_subcategories": targets,
            "history_subcategories": subcategories,
        }
        behavior = {
            **common,
            "prompt": _format_prompt(
                "behavior_only",
                [str(item["title"]) for item in recent_history],
            ),
        }
        interest = {
            **common,
            "prompt": _format_prompt("interest_only", subcategories),
        }
        return behavior, interest

    for batch in pq.ParquetFile(source_path).iter_batches(
        batch_size=8192,
        columns=columns,
    ):
        for row in batch.to_pylist():
            target_interest = row["target_interest"]
            if not isinstance(target_interest, dict):
                raise ValueError(
                    f"{row['sample_id']}: invalid target_interest"
                )
            target = str(target_interest["subcategory"])
            if target not in retained_interests:
                continue
            impression_id = str(row["impression_id"])
            if current_impression is None:
                current_impression = impression_id
            elif impression_id != current_impression:
                yield emit_group()
                group = []
                current_impression = impression_id
            group.append(row)
    if group:
        yield emit_group()


def _write_grouped_dev(
    source_dir: Path,
    balanced_dir: Path,
    staging: Path,
    *,
    eval_subset_size: int,
    history_limit: int,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    retained_rows = pq.read_table(
        balanced_dir / "retained_interests.parquet",
        columns=["target_subcategory"],
    ).to_pylist()
    retained_interests = {
        str(row["target_subcategory"]) for row in retained_rows
    }
    full_writers = {
        task_type: ParquetBatchWriter(
            staging
            / "eval"
            / task_type
            / "dev_full_grouped.parquet",
            GROUPED_EVAL_SCHEMA,
            batch_size,
        )
        for task_type in TASK_TYPES
    }
    priorities: list[tuple[str, str]] = []
    answer_count_distribution: Counter[int] = Counter()

    try:
        for behavior, interest in _iter_filtered_dev_groups(
            source_dir / "dev.parquet",
            retained_interests,
            history_limit=history_limit,
        ):
            impression_id = str(behavior["impression_id"])
            full_writers["behavior_only"].append(behavior)
            full_writers["interest_only"].append(interest)
            priorities.append(
                (
                    _priority("eval-impression", impression_id, seed),
                    impression_id,
                )
            )
            answer_count_distribution[
                len(behavior["acceptable_subcategories"])
            ] += 1
    finally:
        for writer in full_writers.values():
            writer.close()

    full_count = len(priorities)
    if eval_subset_size > full_count:
        raise ValueError(
            f"Eval subset {eval_subset_size} exceeds {full_count}"
        )
    selected_impressions = {
        impression_id
        for _, impression_id in sorted(priorities)[:eval_subset_size]
    }

    for task_type in TASK_TYPES:
        input_path = (
            staging
            / "eval"
            / task_type
            / "dev_full_grouped.parquet"
        )
        writer = ParquetBatchWriter(
            staging
            / "eval"
            / task_type
            / f"dev_{_size_name(eval_subset_size)}_grouped.parquet",
            GROUPED_EVAL_SCHEMA,
            batch_size,
        )
        try:
            for batch in pq.ParquetFile(input_path).iter_batches(
                batch_size=batch_size
            ):
                for row in batch.to_pylist():
                    if row["impression_id"] in selected_impressions:
                        writer.append(row)
        finally:
            writer.close()
        if writer.row_count != eval_subset_size:
            raise ValueError(
                f"{task_type}: wrote {writer.row_count} eval rows, "
                f"expected {eval_subset_size}"
            )

    return {
        "full_grouped_impressions": full_count,
        "subset_grouped_impressions": eval_subset_size,
        "multi_answer_impressions": sum(
            count
            for answer_count, count in answer_count_distribution.items()
            if answer_count > 1
        ),
        "max_answers": max(answer_count_distribution),
        "answer_count_distribution": {
            str(answer_count): count
            for answer_count, count in sorted(
                answer_count_distribution.items()
            )
        },
    }


def _ensure_output_available(
    output_dir: Path,
    output_relative_paths: tuple[Path, ...],
    overwrite: bool,
) -> None:
    existing = [
        output_dir / relative
        for relative in output_relative_paths
        if (output_dir / relative).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}. Use --overwrite."
        )


def build_sft_subsets(
    balanced_dir: Path | str,
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    eval_subset_size: int = 10_000,
    history_limit: int = 50,
    seed: int = 20260806,
    batch_size: int = 2_000,
    overwrite: bool = False,
) -> BuildResult:
    balanced_dir = Path(balanced_dir)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if eval_subset_size <= 0:
        raise ValueError("eval_subset_size must be positive")
    if history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    required = [
        balanced_dir / "selected_train_samples.parquet",
        balanced_dir / "retained_interests.parquet",
        balanced_dir / "behavior_only" / "train.parquet",
        balanced_dir / "interest_only" / "train.parquet",
        source_dir / "dev.parquet",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_relative_paths = _output_relative_paths(
        sizes,
        eval_subset_size,
    )
    _ensure_output_available(
        output_dir,
        output_relative_paths,
        overwrite,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".sft-subsets-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        class_order, sequence, subset_stats = _write_train_subsets(
            balanced_dir,
            staging,
            sizes=sizes,
            seed=seed,
        )
        eval_stats = _write_grouped_dev(
            source_dir,
            balanced_dir,
            staging,
            eval_subset_size=eval_subset_size,
            history_limit=history_limit,
            seed=seed,
            batch_size=batch_size,
        )

        statistics = {
            "rule_version": RULE_VERSION,
            "train": subset_stats,
            "eval": eval_stats,
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_paths = [
            relative
            for relative in output_relative_paths
            if relative.name != "metadata.json"
        ]
        artifact_hashes = {
            str(relative): _sha256(staging / relative)
            for relative in artifact_paths
        }
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "balanced_source_dir": portable_path(balanced_dir),
            "new_interest_source_dir": portable_path(source_dir),
            "seed": seed,
            "train_sizes": list(sizes),
            "train_selection": "stable-class-round-robin-prefix",
            "class_order": class_order,
            "total_balanced_train_samples": len(sequence),
            "eval_subset_size": eval_subset_size,
            "eval_selection": "sha256-priority-over-impression-id",
            "history_limit": history_limit,
            "artifact_sha256": artifact_hashes,
            "source_sha256": {
                "selected_train_samples.parquet": _sha256(
                    balanced_dir / "selected_train_samples.parquet"
                ),
                "retained_interests.parquet": _sha256(
                    balanced_dir / "retained_interests.parquet"
                ),
                "new_interest_dev.parquet": _sha256(
                    source_dir / "dev.parquet"
                ),
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for relative in output_relative_paths:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)

    return BuildResult(
        output_dir=output_dir,
        train_sizes=sizes,
        grouped_dev_rows=eval_stats["full_grouped_impressions"],
        grouped_dev_subset_rows=eval_stats[
            "subset_grouped_impressions"
        ],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build nested SFT Train and grouped Dev datasets."
    )
    parser.add_argument(
        "--balanced-dir",
        type=Path,
        default=Path(
            "data/training/"
            "generative_new_interest_balanced_200_h50"
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/new_interest"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )
    parser.add_argument("--eval-subset-size", type=int, default=10_000)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--batch-size", type=int, default=2_000)
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
    result = build_sft_subsets(
        args.balanced_dir,
        args.source_dir,
        args.output_dir,
        sizes=tuple(args.sizes),
        eval_subset_size=args.eval_subset_size,
        history_limit=args.history_limit,
        seed=args.seed,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Wrote Train sizes %s and grouped Dev %s/%s to %s",
        result.train_sizes,
        result.grouped_dev_subset_rows,
        result.grouped_dev_rows,
        result.output_dir,
    )


if __name__ == "__main__":
    main()
