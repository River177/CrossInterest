from __future__ import annotations

import argparse
import hashlib
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

from new_interest.data.build_generative_dataset import (
    BEHAVIOR_INTEREST_PROMPT_TEMPLATE,
    GENERATIVE_SCHEMA,
    RESPONSE_PREFIX,
    _format_behavior_interest_prompt,
)
from new_interest.data.build_sft_subsets import (
    GROUPED_EVAL_SCHEMA,
    ParquetBatchWriter,
    _priority,
)
from new_interest.infrastructure.paths import portable_path

LOGGER = logging.getLogger(__name__)

TASK_TYPE = "behavior_interest"
RULE_VERSION = "static-user-disjoint-behavior-interest-sft-v1"
DEFAULT_MATCHED_SOURCE_DIR = Path(
    "data/processed/"
    "new_interest_static_user_disjoint_sft21600_h50_v1"
)
DEFAULT_PAIRED_SFT_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_balanced_200_h50_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1"
)

OUTPUT_RELATIVE_PATHS = (
    Path("train/behavior_interest/21600.parquet"),
    Path("eval/behavior_interest/dev_1k_grouped.parquet"),
    Path("eval/behavior_interest/dev_full_grouped.parquet"),
    Path("statistics.json"),
    Path("metadata.json"),
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_rows: int
    train_users: int
    train_interests: int
    dev_impressions: int
    dev_users: int
    dev_target_occurrences: int
    dev_supported_interests: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(counts: Counter[int]) -> dict[str, int | float | None]:
    total = sum(counts.values())
    if not total:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": total,
        "min": min(counts),
        "max": max(counts),
        "mean": sum(value * count for value, count in counts.items()) / total,
    }


def _paired_rows(
    path: Path,
    *,
    columns: list[str],
) -> list[dict[str, object]]:
    return pq.read_table(path, columns=columns).to_pylist()


def _validate_paired_train(
    paired_sft_dir: Path,
) -> list[dict[str, object]]:
    columns = [
        "source_sample_id",
        "split",
        "impression_id",
        "user_id",
        "history_length",
        "response",
        "target_subcategory",
    ]
    behavior = _paired_rows(
        paired_sft_dir / "train/behavior_only/21600.parquet",
        columns=columns,
    )
    interest = _paired_rows(
        paired_sft_dir / "train/interest_only/21600.parquet",
        columns=columns,
    )
    if behavior != interest:
        raise ValueError("Behavior and Interest Train identities are not paired")
    return behavior


def _paired_dev_by_impression(
    paired_sft_dir: Path,
) -> dict[str, dict[str, object]]:
    columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history_length",
        "acceptable_responses",
        "acceptable_subcategories",
        "history_subcategories",
    ]
    behavior = _paired_rows(
        paired_sft_dir / "eval/behavior_only/dev_full_grouped.parquet",
        columns=columns,
    )
    interest = _paired_rows(
        paired_sft_dir / "eval/interest_only/dev_full_grouped.parquet",
        columns=columns,
    )
    if behavior != interest:
        raise ValueError("Behavior and Interest Dev identities are not paired")
    indexed: dict[str, dict[str, object]] = {}
    for row in behavior:
        impression_id = str(row["impression_id"])
        if impression_id in indexed:
            raise ValueError(f"duplicate paired Dev impression {impression_id}")
        indexed[impression_id] = row
    return indexed


def _history(source: dict[str, object], *, history_limit: int) -> list[dict]:
    source_id = str(source["sample_id"])
    history = source["history"]
    if not isinstance(history, list) or not history:
        raise ValueError(f"{source_id}: History must be a non-empty list")
    recent = history[-history_limit:]
    if len(recent) > history_limit:
        raise AssertionError("History truncation failed")
    for index, item in enumerate(recent, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{source_id}: History item {index} is invalid")
        if not str(item.get("title", "")):
            raise ValueError(f"{source_id}: History item {index} has no title")
        if not str(item.get("subcategory", "")):
            raise ValueError(
                f"{source_id}: History item {index} has no subcategory"
            )
    return recent


def _target(source: dict[str, object]) -> str:
    target_interest = source["target_interest"]
    if not isinstance(target_interest, dict):
        raise ValueError(f"{source['sample_id']}: invalid target interest")
    target = str(target_interest.get("subcategory", ""))
    if not target:
        raise ValueError(f"{source['sample_id']}: empty target interest")
    return target


def _train_row(
    source: dict[str, object],
    *,
    history_limit: int,
) -> dict[str, object]:
    history = _history(source, history_limit=history_limit)
    target = _target(source)
    history_interests = {str(item["subcategory"]) for item in history}
    if target in history_interests:
        raise ValueError(
            f"{source['sample_id']}: target appears in History"
        )
    prompt = _format_behavior_interest_prompt(history)
    response = f"{RESPONSE_PREFIX}{target}"
    return {
        "sample_id": f"{source['sample_id']}:{TASK_TYPE}",
        "source_sample_id": str(source["sample_id"]),
        "split": "train",
        "task_type": TASK_TYPE,
        "impression_id": str(source["impression_id"]),
        "user_id": str(source["user_id"]),
        "timestamp": source["timestamp"],
        "history_length": len(history),
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target,
    }


def _grouped_dev_rows(
    source_path: Path,
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
    closed: set[str] = set()

    def emit() -> dict[str, object]:
        first = group[0]
        history = _history(first, history_limit=history_limit)
        user_id = str(first["user_id"])
        timestamp = first["timestamp"]
        history_signature = [
            (
                str(item["news_id"]),
                str(item["subcategory"]),
                str(item["title"]),
            )
            for item in history
        ]
        targets: list[str] = []
        seen_targets: set[str] = set()
        for source in group:
            if str(source["user_id"]) != user_id:
                raise ValueError(
                    f"{current_impression}: multiple users in one impression"
                )
            if source["timestamp"] != timestamp:
                raise ValueError(
                    f"{current_impression}: inconsistent timestamps"
                )
            source_history = _history(source, history_limit=history_limit)
            signature = [
                (
                    str(item["news_id"]),
                    str(item["subcategory"]),
                    str(item["title"]),
                )
                for item in source_history
            ]
            if signature != history_signature:
                raise ValueError(
                    f"{current_impression}: inconsistent History"
                )
            target = _target(source)
            if target not in seen_targets:
                targets.append(target)
                seen_targets.add(target)
        history_subcategories = [
            str(item["subcategory"]) for item in history
        ]
        if set(targets).intersection(history_subcategories):
            raise ValueError(
                f"{current_impression}: Dev target appears in History"
            )
        return {
            "impression_id": str(first["impression_id"]),
            "user_id": user_id,
            "timestamp": timestamp,
            "history_length": len(history),
            "prompt": _format_behavior_interest_prompt(history),
            "acceptable_responses": [
                f"{RESPONSE_PREFIX}{target}" for target in targets
            ],
            "acceptable_subcategories": targets,
            "history_subcategories": history_subcategories,
        }

    for batch in pq.ParquetFile(source_path).iter_batches(
        batch_size=4_096,
        columns=columns,
    ):
        for source in batch.to_pylist():
            impression_id = str(source["impression_id"])
            if current_impression is None:
                current_impression = impression_id
            elif impression_id != current_impression:
                closed.add(current_impression)
                if impression_id in closed:
                    raise ValueError(
                        f"Dev impression {impression_id} is non-contiguous"
                    )
                yield emit()
                group = []
                current_impression = impression_id
            group.append(source)
    if group:
        yield emit()


def _ensure_available(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / relative
        for relative in OUTPUT_RELATIVE_PATHS
        if (output_dir / relative).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}. Use --overwrite."
        )


def build_behavior_interest_sft_dataset(
    matched_source_dir: Path | str = DEFAULT_MATCHED_SOURCE_DIR,
    paired_sft_dir: Path | str = DEFAULT_PAIRED_SFT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    history_limit: int = 50,
    eval_subset_size: int = 1_000,
    seed: int = 20260806,
    batch_size: int = 2_000,
    overwrite: bool = False,
) -> BuildResult:
    matched_source_dir = Path(matched_source_dir)
    paired_sft_dir = Path(paired_sft_dir)
    output_dir = Path(output_dir)
    if history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if eval_subset_size <= 0:
        raise ValueError("eval_subset_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    train_source = matched_source_dir / "train.parquet"
    dev_source = matched_source_dir / "dev.parquet"
    required = [
        train_source,
        dev_source,
        paired_sft_dir / "train/behavior_only/21600.parquet",
        paired_sft_dir / "train/interest_only/21600.parquet",
        paired_sft_dir / "eval/behavior_only/dev_full_grouped.parquet",
        paired_sft_dir / "eval/interest_only/dev_full_grouped.parquet",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    _ensure_available(output_dir, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    paired_train = _validate_paired_train(paired_sft_dir)
    paired_dev = _paired_dev_by_impression(paired_sft_dir)

    with tempfile.TemporaryDirectory(
        prefix=".behavior-interest-sft-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        train_writer = ParquetBatchWriter(
            staging / "train/behavior_interest/21600.parquet",
            GENERATIVE_SCHEMA,
            batch_size,
        )
        train_users: set[str] = set()
        train_targets: Counter[str] = Counter()
        train_history_lengths: Counter[int] = Counter()
        train_prompt_lengths: Counter[int] = Counter()
        train_index = 0
        try:
            for batch in pq.ParquetFile(train_source).iter_batches(
                batch_size=batch_size
            ):
                for source in batch.to_pylist():
                    if train_index >= len(paired_train):
                        raise ValueError("matched Train has extra rows")
                    row = _train_row(source, history_limit=history_limit)
                    paired = paired_train[train_index]
                    expected_identity = {
                        "source_sample_id": row["source_sample_id"],
                        "split": row["split"],
                        "impression_id": row["impression_id"],
                        "user_id": row["user_id"],
                        "history_length": row["history_length"],
                        "response": row["response"],
                        "target_subcategory": row["target_subcategory"],
                    }
                    if paired != expected_identity:
                        raise ValueError(
                            f"Train row {train_index} does not match paired SFT"
                        )
                    user_id = str(row["user_id"])
                    if user_id in train_users:
                        raise ValueError(f"duplicate Train user {user_id}")
                    train_users.add(user_id)
                    target = str(row["target_subcategory"])
                    train_targets[target] += 1
                    train_history_lengths[int(row["history_length"])] += 1
                    train_prompt_lengths[len(str(row["prompt"]))] += 1
                    train_writer.append(row)
                    train_index += 1
        finally:
            train_writer.close()
        if train_index != len(paired_train):
            raise ValueError(
                f"matched Train has {train_index} rows; paired SFT has "
                f"{len(paired_train)}"
            )

        full_writer = ParquetBatchWriter(
            staging / "eval/behavior_interest/dev_full_grouped.parquet",
            GROUPED_EVAL_SCHEMA,
            batch_size,
        )
        dev_users: set[str] = set()
        dev_targets: Counter[str] = Counter()
        dev_history_lengths: Counter[int] = Counter()
        dev_prompt_lengths: Counter[int] = Counter()
        answer_counts: Counter[int] = Counter()
        priorities: list[tuple[str, str]] = []
        seen_dev: set[str] = set()
        try:
            for row in _grouped_dev_rows(
                dev_source,
                history_limit=history_limit,
            ):
                impression_id = str(row["impression_id"])
                if impression_id in seen_dev:
                    raise ValueError(f"duplicate Dev impression {impression_id}")
                seen_dev.add(impression_id)
                paired = paired_dev.get(impression_id)
                if paired is None:
                    raise ValueError(
                        f"Dev impression {impression_id} missing from paired SFT"
                    )
                identity = {key: row[key] for key in paired}
                if identity != paired:
                    raise ValueError(
                        f"Dev impression {impression_id} identity mismatch"
                    )
                full_writer.append(row)
                dev_users.add(str(row["user_id"]))
                targets = [
                    str(value) for value in row["acceptable_subcategories"]
                ]
                dev_targets.update(targets)
                answer_counts[len(targets)] += 1
                dev_history_lengths[int(row["history_length"])] += 1
                dev_prompt_lengths[len(str(row["prompt"]))] += 1
                priorities.append(
                    (
                        _priority("eval-impression", impression_id, seed),
                        impression_id,
                    )
                )
        finally:
            full_writer.close()
        if set(paired_dev) != seen_dev:
            missing = sorted(set(paired_dev) - seen_dev)[:5]
            raise ValueError(f"paired Dev impressions missing: {missing}")
        if eval_subset_size > len(seen_dev):
            raise ValueError("eval subset exceeds full Dev")

        selected_impressions = {
            impression_id
            for _, impression_id in sorted(priorities)[:eval_subset_size]
        }
        subset_writer = ParquetBatchWriter(
            staging / "eval/behavior_interest/dev_1k_grouped.parquet",
            GROUPED_EVAL_SCHEMA,
            batch_size,
        )
        try:
            full_path = (
                staging / "eval/behavior_interest/dev_full_grouped.parquet"
            )
            for batch in pq.ParquetFile(full_path).iter_batches(
                batch_size=batch_size
            ):
                for row in batch.to_pylist():
                    if str(row["impression_id"]) in selected_impressions:
                        subset_writer.append(row)
        finally:
            subset_writer.close()
        if subset_writer.row_count != eval_subset_size:
            raise ValueError(
                f"wrote {subset_writer.row_count} subset rows; expected "
                f"{eval_subset_size}"
            )

        overlap = train_users.intersection(dev_users)
        if overlap:
            raise ValueError(
                f"Train/Dev user overlap is non-zero: {sorted(overlap)[:5]}"
            )
        if set(dev_targets) - set(train_targets):
            raise ValueError("Dev contains targets outside the Train vocabulary")

        statistics = {
            "rule_version": RULE_VERSION,
            "task_type": TASK_TYPE,
            "train": {
                "rows": train_writer.row_count,
                "users": len(train_users),
                "interests": len(train_targets),
                "min_per_interest": min(train_targets.values()),
                "max_per_interest": max(train_targets.values()),
                "target_counts": dict(sorted(train_targets.items())),
                "history_length": _distribution(train_history_lengths),
                "prompt_characters": _distribution(train_prompt_lengths),
            },
            "dev": {
                "impressions": full_writer.row_count,
                "users": len(dev_users),
                "target_occurrences": sum(dev_targets.values()),
                "supported_interests": len(dev_targets),
                "target_counts": dict(sorted(dev_targets.items())),
                "history_length": _distribution(dev_history_lengths),
                "prompt_characters": _distribution(dev_prompt_lengths),
                "answer_count_distribution": {
                    str(count): frequency
                    for count, frequency in sorted(answer_counts.items())
                },
                "subset_impressions": subset_writer.row_count,
            },
            "pairing": {
                "train_rows_matched": train_index,
                "dev_impressions_matched": len(seen_dev),
                "train_dev_user_overlap": 0,
            },
        }
        statistics_path = staging / "statistics.json"
        statistics_path.write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_paths = [
            relative
            for relative in OUTPUT_RELATIVE_PATHS
            if relative.name != "metadata.json"
        ]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "task_type": TASK_TYPE,
            "definition": (
                "Predict one new-interest category from paired Behavior "
                "headlines and Interest categories."
            ),
            "matched_source_dir": portable_path(matched_source_dir),
            "paired_sft_dir": portable_path(paired_sft_dir),
            "history_limit": history_limit,
            "history_selection": f"history[-{history_limit}:]",
            "history_item_format": "[Topic category] Headline",
            "prompt_template": BEHAVIOR_INTEREST_PROMPT_TEMPLATE,
            "response_format": "New topic: {target_interest.subcategory}",
            "train_rows": train_writer.row_count,
            "dev_impressions": full_writer.row_count,
            "source_sha256": {
                portable_path(path): _sha256(path) for path in required
            },
            "artifact_sha256": {
                str(relative): _sha256(staging / relative)
                for relative in artifact_paths
            },
        }
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for relative in OUTPUT_RELATIVE_PATHS:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)

    return BuildResult(
        output_dir=output_dir,
        train_rows=train_writer.row_count,
        train_users=len(train_users),
        train_interests=len(train_targets),
        dev_impressions=full_writer.row_count,
        dev_users=len(dev_users),
        dev_target_occurrences=sum(dev_targets.values()),
        dev_supported_interests=len(dev_targets),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired Behavior + Interest SFT data."
    )
    parser.add_argument(
        "--matched-source-dir",
        type=Path,
        default=DEFAULT_MATCHED_SOURCE_DIR,
    )
    parser.add_argument(
        "--paired-sft-dir",
        type=Path,
        default=DEFAULT_PAIRED_SFT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--eval-subset-size", type=int, default=1_000)
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
    result = build_behavior_interest_sft_dataset(
        args.matched_source_dir,
        args.paired_sft_dir,
        args.output_dir,
        history_limit=args.history_limit,
        eval_subset_size=args.eval_subset_size,
        seed=args.seed,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Wrote %s Train rows and %s grouped Dev impressions to %s",
        result.train_rows,
        result.dev_impressions,
        result.output_dir,
    )


if __name__ == "__main__":
    main()
