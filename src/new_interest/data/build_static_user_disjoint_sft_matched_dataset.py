from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import DATASET_SCHEMA, VOCABULARY_SCHEMA
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-sft-matched-h50-v1"
DEFAULT_SOURCE_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_v1"
)
DEFAULT_SFT_DIR = Path(
    "data/training/sft_static_user_disjoint_balanced_200_h50_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_sft21600_h50_v1"
)
DEFAULT_TRAIN_SIZE = 21_600
DEFAULT_HISTORY_LIMIT = 50


TRAIN_SELECTION_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("class_rank", pa.int16(), nullable=False),
        pa.field("within_class_rank", pa.int16(), nullable=False),
        pa.field("global_rank", pa.int32(), nullable=False),
    ]
)

DEV_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)

OUTPUT_FILES = (
    "train.parquet",
    "dev.parquet",
    "interest_vocabulary.parquet",
    "target_interest_vocabulary.parquet",
    "train_selection_manifest.parquet",
    "dev_exclusion_manifest.parquet",
    "statistics.json",
    "metadata.json",
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_samples: int
    target_interests: int
    dev_samples: int
    dev_users: int
    dev_target_interests: int


class _ParquetWriter:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        row_group_size: int,
    ) -> None:
        self._schema = schema
        self._row_group_size = row_group_size
        self._pending: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=True,
        )
        self.row_count = 0

    def append(self, row: dict[str, object]) -> None:
        self._pending.append(row)
        if len(self._pending) >= self._row_group_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self._writer.write_table(
            pa.Table.from_pylist(self._pending, schema=self._schema),
            row_group_size=self._row_group_size,
        )
        self.row_count += len(self._pending)
        self._pending.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(path: Path, columns: Iterable[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    missing = set(columns) - set(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _history_limited_row(
    row: dict[str, object],
    *,
    history_limit: int,
) -> tuple[dict[str, object], int, int]:
    history = row.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{row.get('sample_id')}: invalid History")
    limited = history[-history_limit:]
    output = dict(row)
    output["history"] = limited
    return output, len(history), len(limited)


def _load_sft_identity(
    sft_dir: Path,
    *,
    train_size: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    set[str],
]:
    manifest_path = sft_dir / "subset_manifest.parquet"
    _require_columns(
        manifest_path,
        (
            "source_sample_id",
            "target_subcategory",
            "class_rank",
            "within_class_rank",
            "global_rank",
            "included_sizes",
        ),
    )
    selected = [
        row
        for row in pq.read_table(manifest_path).to_pylist()
        if train_size in {int(value) for value in row["included_sizes"]}
    ]
    selected.sort(key=lambda row: int(row["global_rank"]))
    if len(selected) != train_size:
        raise ValueError(
            f"{manifest_path}: selected {len(selected)} rows for {train_size}"
        )
    selected_ids = [str(row["source_sample_id"]) for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"{manifest_path}: duplicate selected source_sample_id")

    task_rows: dict[str, dict[str, dict[str, object]]] = {}
    columns = (
        "source_sample_id",
        "user_id",
        "impression_id",
        "history_length",
        "target_subcategory",
    )
    for task_type in ("behavior_only", "interest_only"):
        path = sft_dir / "train" / task_type / f"{train_size}.parquet"
        _require_columns(path, columns)
        indexed: dict[str, dict[str, object]] = {}
        for row in pq.read_table(path, columns=list(columns)).to_pylist():
            sample_id = str(row["source_sample_id"])
            if sample_id in indexed:
                raise ValueError(f"{path}: duplicate source_sample_id {sample_id}")
            indexed[sample_id] = row
        if set(indexed) != set(selected_ids):
            raise ValueError(
                f"{path}: sample identities differ from subset manifest"
            )
        task_rows[task_type] = indexed

    behavior = task_rows["behavior_only"]
    interest = task_rows["interest_only"]
    for sample_id in selected_ids:
        left = behavior[sample_id]
        right = interest[sample_id]
        for column in columns:
            if left[column] != right[column]:
                raise ValueError(
                    f"{sample_id}: Behavior/Interest mismatch in {column}"
                )
    targets = {str(row["target_subcategory"]) for row in selected}
    return selected, behavior, targets


def _write_train(
    source_path: Path,
    staging: Path,
    *,
    selected: list[dict[str, object]],
    sft_rows: dict[str, dict[str, object]],
    history_limit: int,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected_ids = {str(row["source_sample_id"]) for row in selected}
    source_rows: dict[str, dict[str, object]] = {}
    source_count = 0
    for batch in pq.ParquetFile(source_path).iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            source_count += 1
            sample_id = str(row["sample_id"])
            if sample_id not in selected_ids:
                continue
            if sample_id in source_rows:
                raise ValueError(f"{source_path}: duplicate sample_id {sample_id}")
            source_rows[sample_id] = row
    if set(source_rows) != selected_ids:
        missing = sorted(selected_ids - set(source_rows))[:5]
        raise ValueError(f"{source_path}: selected rows are missing: {missing}")

    writer = _ParquetWriter(
        staging / "train.parquet",
        DATASET_SCHEMA,
        row_group_size=batch_size,
    )
    manifest_rows: list[dict[str, object]] = []
    users: set[str] = set()
    target_counts: Counter[str] = Counter()
    source_history_lengths: Counter[int] = Counter()
    retained_history_lengths: Counter[int] = Counter()
    truncated_users = 0
    try:
        for manifest in selected:
            sample_id = str(manifest["source_sample_id"])
            raw = source_rows[sample_id]
            limited, source_length, retained_length = _history_limited_row(
                raw,
                history_limit=history_limit,
            )
            target = str(raw["target_interest"]["subcategory"])
            sft = sft_rows[sample_id]
            expected_target = str(manifest["target_subcategory"])
            if target != expected_target or target != str(
                sft["target_subcategory"]
            ):
                raise ValueError(f"{sample_id}: target differs from SFT identity")
            user_id = str(raw["user_id"])
            if user_id != str(sft["user_id"]):
                raise ValueError(f"{sample_id}: user differs from SFT identity")
            if retained_length != int(sft["history_length"]):
                raise ValueError(
                    f"{sample_id}: H{history_limit} length differs from SFT"
                )
            if user_id in users:
                raise ValueError(f"duplicate selected Train user {user_id}")
            users.add(user_id)
            target_counts[target] += 1
            source_history_lengths[source_length] += 1
            retained_history_lengths[retained_length] += 1
            truncated_users += int(source_length > retained_length)
            writer.append(limited)
            manifest_rows.append(
                {
                    "source_sample_id": sample_id,
                    "user_id": user_id,
                    "target_subcategory": target,
                    "class_rank": int(manifest["class_rank"]),
                    "within_class_rank": int(manifest["within_class_rank"]),
                    "global_rank": int(manifest["global_rank"]),
                }
            )
    finally:
        writer.close()
    if writer.row_count != len(selected):
        raise ValueError("Train writer count differs from SFT selection")
    return (
        {
            "source_rows": source_count,
            "selected_rows": writer.row_count,
            "selected_users": len(users),
            "target_interests": len(target_counts),
            "min_samples_per_interest": min(target_counts.values()),
            "max_samples_per_interest": max(target_counts.values()),
            "target_counts": dict(sorted(target_counts.items())),
            "source_history_max": max(source_history_lengths),
            "retained_history_max": max(retained_history_lengths),
            "users_truncated_to_history_limit": truncated_users,
        },
        manifest_rows,
    )


def _write_dev(
    source_path: Path,
    staging: Path,
    *,
    target_interests: set[str],
    history_limit: int,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    writer = _ParquetWriter(
        staging / "dev.parquet",
        DATASET_SCHEMA,
        row_group_size=batch_size,
    )
    exclusions: list[dict[str, object]] = []
    source_rows = 0
    source_users: set[str] = set()
    source_impressions: set[str] = set()
    source_targets: Counter[str] = Counter()
    retained_users: set[str] = set()
    retained_impressions: set[str] = set()
    retained_targets: Counter[str] = Counter()
    truncated_users: set[str] = set()
    history_by_user: dict[str, tuple[str, ...]] = {}
    try:
        for batch in pq.ParquetFile(source_path).iter_batches(
            batch_size=batch_size
        ):
            for row in batch.to_pylist():
                source_rows += 1
                user_id = str(row["user_id"])
                impression_id = str(row["impression_id"])
                target = str(row["target_interest"]["subcategory"])
                source_users.add(user_id)
                source_impressions.add(impression_id)
                source_targets[target] += 1
                if target not in target_interests:
                    exclusions.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "user_id": user_id,
                            "impression_id": impression_id,
                            "target_subcategory": target,
                            "reason": "target_not_in_sft_training_vocabulary",
                        }
                    )
                    continue
                limited, source_length, _ = _history_limited_row(
                    row,
                    history_limit=history_limit,
                )
                signature = tuple(
                    str(item["news_id"]) for item in limited["history"]
                )
                previous = history_by_user.setdefault(user_id, signature)
                if previous != signature:
                    raise ValueError(
                        f"{source_path}: inconsistent History for {user_id}"
                    )
                if source_length > history_limit:
                    truncated_users.add(user_id)
                retained_users.add(user_id)
                retained_impressions.add(impression_id)
                retained_targets[target] += 1
                writer.append(limited)
    finally:
        writer.close()
    return {
        "source_rows": source_rows,
        "source_users": len(source_users),
        "source_impressions": len(source_impressions),
        "source_target_interests": len(source_targets),
        "retained_rows": writer.row_count,
        "retained_users": len(retained_users),
        "retained_impressions": len(retained_impressions),
        "retained_target_interests": len(retained_targets),
        "excluded_rows": len(exclusions),
        "users_removed_entirely": len(source_users - retained_users),
        "impressions_removed_entirely": len(
            source_impressions - retained_impressions
        ),
        "users_truncated_to_history_limit": len(truncated_users),
        "excluded_target_counts": dict(
            sorted(
                Counter(
                    str(row["target_subcategory"]) for row in exclusions
                ).items()
            )
        ),
        "retained_target_counts": dict(sorted(retained_targets.items())),
        "training_target_interests_absent_from_dev": sorted(
            target_interests - set(retained_targets)
        ),
    }, exclusions


def _publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    backup = output_dir.with_name(output_dir.name + ".old")
    if overwrite and output_dir.exists() and backup.exists():
        raise FileExistsError(f"Cannot overwrite while backup exists: {backup}")
    if output_dir.exists():
        os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            os.replace(backup, output_dir)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output_dir)


def build_static_user_disjoint_sft_matched_dataset(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    sft_dir: Path | str = DEFAULT_SFT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    batch_size: int = 2_000,
    overwrite: bool = False,
) -> BuildResult:
    """Build the raw-schema H50 dataset matched to the frozen SFT sample IDs."""
    source_dir = Path(source_dir)
    sft_dir = Path(sft_dir)
    output_dir = Path(output_dir)
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )

    source_paths = (
        source_dir / "train.parquet",
        source_dir / "dev.parquet",
        source_dir / "interest_vocabulary.parquet",
        sft_dir / "subset_manifest.parquet",
        sft_dir / "train" / "behavior_only" / f"{train_size}.parquet",
        sft_dir / "train" / "interest_only" / f"{train_size}.parquet",
    )
    _require_columns(source_paths[0], DATASET_SCHEMA.names)
    _require_columns(source_paths[1], DATASET_SCHEMA.names)
    _require_columns(source_paths[2], VOCABULARY_SCHEMA.names)

    selected, sft_rows, target_interests = _load_sft_identity(
        sft_dir,
        train_size=train_size,
    )
    vocabulary = pq.read_table(source_paths[2], schema=VOCABULARY_SCHEMA)
    vocabulary_labels = {
        str(value) for value in vocabulary["subcategory"].to_pylist()
    }
    if not target_interests.issubset(vocabulary_labels):
        raise ValueError("SFT targets fall outside the source vocabulary")
    target_vocabulary = vocabulary.filter(
        pa.array(
            [
                str(value) in target_interests
                for value in vocabulary["subcategory"].to_pylist()
            ]
        )
    )
    if target_vocabulary.num_rows != len(target_interests):
        raise ValueError("target vocabulary is not one-to-one")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-user-disjoint-sft-matched-",
        dir=str(output_dir.parent),
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        train_stats, manifest_rows = _write_train(
            source_paths[0],
            staging,
            selected=selected,
            sft_rows=sft_rows,
            history_limit=history_limit,
            batch_size=batch_size,
        )
        dev_stats, exclusion_rows = _write_dev(
            source_paths[1],
            staging,
            target_interests=target_interests,
            history_limit=history_limit,
            batch_size=batch_size,
        )
        pq.write_table(
            vocabulary,
            staging / "interest_vocabulary.parquet",
            compression="zstd",
            use_dictionary=True,
        )
        pq.write_table(
            target_vocabulary,
            staging / "target_interest_vocabulary.parquet",
            compression="zstd",
            use_dictionary=True,
        )
        pq.write_table(
            pa.Table.from_pylist(
                manifest_rows,
                schema=TRAIN_SELECTION_SCHEMA,
            ),
            staging / "train_selection_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
        )
        pq.write_table(
            pa.Table.from_pylist(
                exclusion_rows,
                schema=DEV_EXCLUSION_SCHEMA,
            ),
            staging / "dev_exclusion_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
        )

        statistics = {
            "rule_version": RULE_VERSION,
            "train_size": train_size,
            "history_limit": history_limit,
            "history_operation": "history[-history_limit:]",
            "train": train_stats,
            "dev": dev_stats,
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts_without_metadata = [
            name for name in OUTPUT_FILES if name != "metadata.json"
        ]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_dataset_dir": portable_path(source_dir),
            "sft_dataset_dir": portable_path(sft_dir),
            "output_dir": portable_path(output_dir),
            "train_size": train_size,
            "history_limit": history_limit,
            "selection": (
                "exact source_sample_id set and order from the frozen SFT "
                "subset manifest"
            ),
            "dev_filter": (
                "target subcategory belongs to the "
                f"{len(target_interests)} SFT targets"
            ),
            "label_space": (
                f"{len(target_interests)} Train target interests; "
                "Dev metric support may be smaller"
            ),
            "history": (
                "last 50 raw static History items; no impression replay"
            ),
            "implementation_sha256": {
                portable_path(Path(__file__)): _sha256(Path(__file__))
            },
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "artifact_sha256": {
                name: _sha256(staging / name)
                for name in artifacts_without_metadata
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)

    return BuildResult(
        output_dir=output_dir,
        train_samples=int(train_stats["selected_rows"]),
        target_interests=int(train_stats["target_interests"]),
        dev_samples=int(dev_stats["retained_rows"]),
        dev_users=int(dev_stats["retained_users"]),
        dev_target_interests=int(dev_stats["retained_target_interests"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument(
        "--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT
    )
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_static_user_disjoint_sft_matched_dataset(
        args.source_dir,
        args.sft_dir,
        args.output_dir,
        train_size=args.train_size,
        history_limit=args.history_limit,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
