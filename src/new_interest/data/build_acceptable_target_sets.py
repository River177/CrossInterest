from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path

RULE_VERSION = "acceptable-target-sets-v1"
DEFAULT_SIZES = (1_000, 5_000)
DEFAULT_RETAINED_LABEL_COUNT = 110
MANIFEST_FILENAME = "acceptable_target_sets.parquet"

ACCEPTABLE_TARGET_SET_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("selected_target_subcategory", pa.string(), nullable=False),
        pa.field(
            "acceptable_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("included_sizes", pa.list_(pa.int32()), nullable=False),
    ]
)


def _size_name(size: int) -> str:
    return f"{size // 1_000}k" if size % 1_000 == 0 else str(size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_paired_sft_rows(
    sft_dir: Path,
    sizes: tuple[int, ...],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    largest_rows: list[dict[str, object]] = []
    previous_ids: list[str] | None = None
    source_hashes: dict[str, str] = {}
    columns = [
        "source_sample_id",
        "impression_id",
        "target_subcategory",
    ]
    for size in sizes:
        name = _size_name(size)
        paired: dict[str, list[dict[str, object]]] = {}
        for task_type in ("behavior_only", "interest_only"):
            path = sft_dir / "train" / task_type / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = str(path.relative_to(sft_dir))
            source_hashes[relative] = _sha256(path)
            rows = pq.read_table(path, columns=columns).to_pylist()
            if len(rows) != size:
                raise ValueError(f"{path}: expected {size}, got {len(rows)}")
            paired[task_type] = rows

        behavior_keys = [
            (
                str(row["source_sample_id"]),
                str(row["impression_id"]),
                str(row["target_subcategory"]),
            )
            for row in paired["behavior_only"]
        ]
        interest_keys = [
            (
                str(row["source_sample_id"]),
                str(row["impression_id"]),
                str(row["target_subcategory"]),
            )
            for row in paired["interest_only"]
        ]
        if behavior_keys != interest_keys:
            raise ValueError(
                f"Behavior/Interest rows do not match for {name}"
            )
        source_ids = [key[0] for key in behavior_keys]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{name}: duplicate source_sample_id")
        if previous_ids is not None and source_ids[: len(previous_ids)] != previous_ids:
            raise ValueError(f"{name} is not a nested prefix")
        previous_ids = source_ids
        largest_rows = paired["behavior_only"]
    return largest_rows, source_hashes


def _collect_acceptable_targets(
    source_path: Path,
    impression_ids: set[str],
    retained_subcategories: set[str],
) -> dict[str, list[str]]:
    targets = {impression_id: [] for impression_id in impression_ids}
    seen = {impression_id: set() for impression_id in impression_ids}
    parquet = pq.ParquetFile(source_path)
    for batch in parquet.iter_batches(
        batch_size=8192,
        columns=["impression_id", "target_interest"],
    ):
        for row in batch.to_pylist():
            impression_id = str(row["impression_id"])
            if impression_id not in targets:
                continue
            target_interest = row["target_interest"]
            if not isinstance(target_interest, dict):
                raise ValueError("Invalid target_interest in source dataset")
            subcategory = str(target_interest["subcategory"])
            if (
                subcategory in retained_subcategories
                and subcategory not in seen[impression_id]
            ):
                targets[impression_id].append(subcategory)
                seen[impression_id].add(subcategory)
    return targets


def build_acceptable_target_sets(
    source_path: Path | str,
    sft_dir: Path | str,
    output_dir: Path | str,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    expected_retained_label_count: int | None = DEFAULT_RETAINED_LABEL_COUNT,
    overwrite: bool = False,
) -> Path:
    source_path = Path(source_path)
    sft_dir = Path(sft_dir)
    output_dir = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sizes must be unique and increasing")

    output_paths = [
        output_dir / MANIFEST_FILENAME,
        output_dir / "statistics.json",
        output_dir / "metadata.json",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Acceptable-target outputs already exist in {output_dir}. "
            "Use --overwrite."
        )

    selected_rows, sft_hashes = _load_paired_sft_rows(sft_dir, sizes)
    retained_subcategories = sorted(
        {str(row["target_subcategory"]) for row in selected_rows}
    )
    if (
        expected_retained_label_count is not None
        and len(retained_subcategories) != expected_retained_label_count
    ):
        raise ValueError(
            "Expected "
            f"{expected_retained_label_count} retained labels, got "
            f"{len(retained_subcategories)}"
        )

    impression_ids = {
        str(row["impression_id"]) for row in selected_rows
    }
    if len(impression_ids) != len(selected_rows):
        raise ValueError("Selected SFT rows must have unique impression_id")
    acceptable_by_impression = _collect_acceptable_targets(
        source_path,
        impression_ids,
        set(retained_subcategories),
    )

    manifest_rows = []
    for index, row in enumerate(selected_rows, start=1):
        source_sample_id = str(row["source_sample_id"])
        impression_id = str(row["impression_id"])
        selected_target = str(row["target_subcategory"])
        acceptable = acceptable_by_impression[impression_id]
        if selected_target not in acceptable:
            raise ValueError(
                f"{source_sample_id}: selected target is not acceptable"
            )
        manifest_rows.append(
            {
                "source_sample_id": source_sample_id,
                "impression_id": impression_id,
                "selected_target_subcategory": selected_target,
                "acceptable_subcategories": acceptable,
                "included_sizes": [
                    size for size in sizes if index <= size
                ],
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".acceptable-target-sets-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        manifest_path = staging / MANIFEST_FILENAME
        pq.write_table(
            pa.Table.from_pylist(
                manifest_rows,
                schema=ACCEPTABLE_TARGET_SET_SCHEMA,
            ),
            manifest_path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=2_000,
        )
        target_count_distribution = Counter(
            len(row["acceptable_subcategories"]) for row in manifest_rows
        )
        statistics = {
            "rule_version": RULE_VERSION,
            "selected_rows": len(manifest_rows),
            "retained_label_count": len(retained_subcategories),
            "acceptable_target_count_distribution": {
                str(count): frequency
                for count, frequency in sorted(
                    target_count_distribution.items()
                )
            },
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_train_path": portable_path(source_path),
            "source_sft_dir": portable_path(sft_dir),
            "sizes": list(sizes),
            "retained_label_count": len(retained_subcategories),
            "retained_subcategories": retained_subcategories,
            "acceptable_target_policy": (
                "all-retained-targets-for-selected-impression"
            ),
            "source_sha256": {
                "new_interest_train.parquet": _sha256(source_path),
                **sft_hashes,
            },
            "artifact_sha256": {
                MANIFEST_FILENAME: _sha256(manifest_path),
                "statistics.json": _sha256(staging / "statistics.json"),
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_paths:
            os.replace(staging / path.name, path)
    return output_dir / MANIFEST_FILENAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build acceptable target sets for nested GRPO rows."
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("data/processed/new_interest/train.parquet"),
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/grpo_acceptable_target_sets"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )
    parser.add_argument(
        "--expected-retained-label-count",
        type=int,
        default=DEFAULT_RETAINED_LABEL_COUNT,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = build_acceptable_target_sets(
        args.source_path,
        args.sft_dir,
        args.output_dir,
        sizes=tuple(args.sizes),
        expected_retained_label_count=args.expected_retained_label_count,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
