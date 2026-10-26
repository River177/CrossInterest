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

from new_interest.data.build_generative_dataset import RESPONSE_PREFIX
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "behavior-interest-text-label-scaling-v1"
TASK_TYPE = "behavior_interest"
DEFAULT_TRAIN_SIZES = (5_400, 10_800, 16_200, 21_600)
DEFAULT_SOURCE_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1"
)
DEFAULT_SUBSET_MANIFEST = Path(
    "data/training/sft_static_user_disjoint_balanced_200_h50_v1/"
    "subset_manifest.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_text_label_200_h50_"
    "behavior_interest_scaling_v1"
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_sizes: tuple[int, ...]
    train_hashes: dict[int, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _validate_sizes(
    train_sizes: Iterable[int],
    *,
    expected_interests: int,
    source_rows: int,
) -> tuple[int, ...]:
    sizes = tuple(sorted({int(value) for value in train_sizes}))
    if not sizes:
        raise ValueError("train_sizes must not be empty")
    if expected_interests <= 0:
        raise ValueError("expected_interests must be positive")
    for size in sizes:
        if size <= 0 or size > source_rows:
            raise ValueError(f"Invalid train size: {size}")
        if size % expected_interests:
            raise ValueError(
                f"Train size {size} is not divisible by {expected_interests}"
            )
    return sizes


def _publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite."
        )
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.with_name(
            f".{output_dir.name}.backup-{os.getpid()}"
        )
        if backup.exists():
            raise FileExistsError(backup)
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_behavior_interest_text_label_scaling_dataset(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    subset_manifest: Path | str = DEFAULT_SUBSET_MANIFEST,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    train_sizes: Iterable[int] = DEFAULT_TRAIN_SIZES,
    expected_interests: int = 108,
    overwrite: bool = False,
) -> BuildResult:
    source_dir = Path(source_dir)
    subset_manifest = Path(subset_manifest)
    output_dir = Path(output_dir)
    source_train = source_dir / "train" / TASK_TYPE / "21600.parquet"
    source_eval = source_dir / "eval" / TASK_TYPE / "dev_full_grouped.parquet"
    for path in (source_train, source_eval, subset_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite."
        )

    train_table = pq.read_table(source_train)
    train_rows = train_table.to_pylist()
    sizes = _validate_sizes(
        train_sizes,
        expected_interests=expected_interests,
        source_rows=len(train_rows),
    )
    train_by_source = {
        str(row["source_sample_id"]): row for row in train_rows
    }
    if len(train_by_source) != len(train_rows):
        raise ValueError("Source Train contains duplicate source_sample_id")
    if {str(row["task_type"]) for row in train_rows} != {TASK_TYPE}:
        raise ValueError("Source Train is not Behavior+Interest")
    if len({str(row["user_id"]) for row in train_rows}) != len(train_rows):
        raise ValueError("Source Train is not one row per user")
    for row in train_rows:
        target = str(row["target_subcategory"])
        response = str(row["response"])
        if response != f"{RESPONSE_PREFIX}{target}":
            raise ValueError("Source Train response is not its Text-label")
        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or str(messages[0]["content"]) != str(row["prompt"])
            or str(messages[1]["content"]) != response
        ):
            raise ValueError("Source Train messages are not prompt/response aligned")

    manifest_rows = pq.read_table(
        subset_manifest,
        columns=[
            "source_sample_id",
            "target_subcategory",
            "within_class_rank",
            "global_rank",
        ],
    ).to_pylist()
    manifest_by_source = {
        str(row["source_sample_id"]): row for row in manifest_rows
    }
    if len(manifest_by_source) != len(manifest_rows):
        raise ValueError("Subset manifest contains duplicate source_sample_id")
    if set(manifest_by_source) != set(train_by_source):
        raise ValueError("Subset manifest identity differs from source Train")
    for source_id, manifest in manifest_by_source.items():
        if str(manifest["target_subcategory"]) != str(
            train_by_source[source_id]["target_subcategory"]
        ):
            raise ValueError("Subset manifest target differs from source Train")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        train_root = staging / "train" / TASK_TYPE
        eval_root = staging / "eval" / TASK_TYPE
        train_root.mkdir(parents=True)
        eval_root.mkdir(parents=True)
        shutil.copyfile(source_eval, eval_root / "dev_full_grouped.parquet")

        statistics: dict[str, object] = {
            "rule_version": RULE_VERSION,
            "task_type": TASK_TYPE,
            "output_type": "text_label",
            "response_prefix": RESPONSE_PREFIX,
            "expected_interests": expected_interests,
            "nested_selection": True,
            "selection_rule": "within_class_rank < samples_per_interest",
            "train": {},
        }
        previous_ids: set[str] = set()
        train_hashes: dict[int, str] = {}
        for size in sizes:
            per_class = size // expected_interests
            selected_manifest = sorted(
                (
                    row
                    for row in manifest_rows
                    if int(row["within_class_rank"]) < per_class
                ),
                key=lambda row: int(row["global_rank"]),
            )
            selected_ids = [
                str(row["source_sample_id"]) for row in selected_manifest
            ]
            if len(selected_ids) != size or len(set(selected_ids)) != size:
                raise ValueError(f"Subset {size} identity/count mismatch")
            selected_set = set(selected_ids)
            if not previous_ids.issubset(selected_set):
                raise ValueError(f"Subset {size} is not nested")
            previous_ids = selected_set
            selected_rows = [train_by_source[value] for value in selected_ids]
            target_counts = Counter(
                str(row["target_subcategory"]) for row in selected_rows
            )
            if (
                len(target_counts) != expected_interests
                or set(target_counts.values()) != {per_class}
            ):
                raise ValueError(f"Subset {size} is not class-balanced")
            if len({str(row["user_id"]) for row in selected_rows}) != size:
                raise ValueError(f"Subset {size} repeats a Train user")

            output_path = train_root / f"{size}.parquet"
            pq.write_table(
                pa.Table.from_pylist(selected_rows, schema=train_table.schema),
                output_path,
                compression="zstd",
                use_dictionary=True,
            )
            train_hashes[size] = _sha256(output_path)
            statistics["train"][str(size)] = {
                "samples": size,
                "users": size,
                "interests": expected_interests,
                "samples_per_interest": per_class,
                "min_per_interest": min(target_counts.values()),
                "max_per_interest": max(target_counts.values()),
            }

        _write_json(staging / "statistics.json", statistics)
        artifact_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_dir": portable_path(source_dir),
            "subset_manifest": portable_path(subset_manifest),
            "output_dir": portable_path(output_dir),
            "task_type": TASK_TYPE,
            "output_type": "text_label",
            "response_prefix": RESPONSE_PREFIX,
            "train_sizes": list(sizes),
            "samples_per_interest": [
                size // expected_interests for size in sizes
            ],
            "selection_rule": "within_class_rank < samples_per_interest",
            "source_sha256": {
                portable_path(source_train): _sha256(source_train),
                portable_path(source_eval): _sha256(source_eval),
                portable_path(subset_manifest): _sha256(subset_manifest),
            },
            "artifact_sha256": artifact_hashes,
        }
        _write_json(staging / "metadata.json", metadata)
        _publish(staging, output_dir, overwrite=overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return BuildResult(
        output_dir=output_dir,
        train_sizes=sizes,
        train_hashes=train_hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build nested class-balanced Behavior+Interest to Text-label "
            "training subsets for data-saturation experiments."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SIZES),
    )
    parser.add_argument("--expected-interests", type=int, default=108)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_behavior_interest_text_label_scaling_dataset(
        source_dir=args.source_dir,
        subset_manifest=args.subset_manifest,
        output_dir=args.output_dir,
        train_sizes=args.train_sizes,
        expected_interests=args.expected_interests,
        overwrite=args.overwrite,
    )
    print(
        "BEHAVIOR_INTEREST_TEXT_LABEL_SCALING_BUILD_OK",
        json.dumps(
            {
                "output_dir": portable_path(result.output_dir),
                "train_sizes": list(result.train_sizes),
                "train_sha256": result.train_hashes,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
