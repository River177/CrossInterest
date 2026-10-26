from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_semantic_id_token_sft_dataset import (
    INTEREST_TOKEN_MAPPING_SCHEMA,
    _mapping_rows,
    _transform_dev_row,
    _transform_train_row,
    _write_transformed_parquet,
    sha256_file,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "behavior-interest-semantic-id-token-sft-v1"
TASK_TYPE = "behavior_interest"
TRAIN_SIZE = 21_600
DEFAULT_SOURCE_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1"
)
DEFAULT_INTERESTS_FILE = Path(
    "data/training/"
    "generative_new_interest_static_user_disjoint_balanced_200_h50_v1/"
    "retained_interests.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1"
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _commit_staging(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
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


def _validate_and_summarize(
    train_path: Path,
    eval_path: Path,
    mapping_rows: list[dict[str, object]],
    *,
    train_size: int,
    samples_per_interest: int,
) -> dict[str, object]:
    train_rows = pq.read_table(train_path).to_pylist()
    eval_rows = pq.read_table(eval_path).to_pylist()
    token_by_label = {
        str(row["target_subcategory"]): str(row["interest_token"])
        for row in mapping_rows
    }

    source_ids = [str(row["source_sample_id"]) for row in train_rows]
    train_users = [str(row["user_id"]) for row in train_rows]
    target_counts = Counter(
        str(row["target_subcategory"]) for row in train_rows
    )
    if len(train_rows) != train_size:
        raise ValueError(f"Train must contain exactly {train_size:,} rows")
    if len(set(source_ids)) != train_size:
        raise ValueError("Train source_sample_id values are not unique")
    if len(set(train_users)) != train_size:
        raise ValueError("Train must contain one row per user")
    if set(target_counts) != set(token_by_label):
        raise ValueError("Train target vocabulary differs from token mapping")
    if set(target_counts.values()) != {samples_per_interest}:
        raise ValueError(
            "Train is not exactly "
            f"{samples_per_interest} samples per interest"
        )
    if {str(row["task_type"]) for row in train_rows} != {TASK_TYPE}:
        raise ValueError("Train is not Behavior+Interest")
    for row in train_rows:
        target = str(row["target_subcategory"])
        response = str(row["response"])
        messages = row["messages"]
        if response != token_by_label[target]:
            raise ValueError(
                f"{row['source_sample_id']}: response is not its Semantic ID"
            )
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0] != {"role": "user", "content": row["prompt"]}
            or messages[1] != {"role": "assistant", "content": response}
        ):
            raise ValueError(
                f"{row['source_sample_id']}: message contract mismatch"
            )

    impression_ids = [str(row["impression_id"]) for row in eval_rows]
    if len(impression_ids) != len(set(impression_ids)):
        raise ValueError("Dev impression_id values are not unique")
    dev_users = {str(row["user_id"]) for row in eval_rows}
    overlap = set(train_users).intersection(dev_users)
    if overlap:
        raise ValueError(f"Train/Dev user overlap: {len(overlap)}")
    dev_targets: list[str] = []
    for row in eval_rows:
        labels = [str(value) for value in row["acceptable_subcategories"]]
        responses = [str(value) for value in row["acceptable_responses"]]
        if responses != [token_by_label[label] for label in labels]:
            raise ValueError(
                f"{row['impression_id']}: Dev Semantic-ID responses mismatch"
            )
        if set(labels).intersection(
            str(value) for value in row["history_subcategories"]
        ):
            raise ValueError(
                f"{row['impression_id']}: Dev target appears in History"
            )
        dev_targets.extend(labels)

    train_history_lengths = [int(row["history_length"]) for row in train_rows]
    dev_history_lengths = [int(row["history_length"]) for row in eval_rows]
    if max(train_history_lengths + dev_history_lengths) > 50:
        raise ValueError("History length exceeds 50")
    return {
        "rule_version": RULE_VERSION,
        "task_type": TASK_TYPE,
        "mapping": {
            "interests": len(mapping_rows),
            "first_token": mapping_rows[0]["interest_token"],
            "last_token": mapping_rows[-1]["interest_token"],
        },
        "train": {
            "samples": len(train_rows),
            "users": len(set(train_users)),
            "interests": len(target_counts),
            "samples_per_interest": samples_per_interest,
            "max_history_length": max(train_history_lengths),
            "target_counts": dict(sorted(target_counts.items())),
        },
        "dev": {
            "impressions": len(eval_rows),
            "users": len(dev_users),
            "target_occurrences": len(dev_targets),
            "supported_interests": len(set(dev_targets)),
            "max_history_length": max(dev_history_lengths),
            "target_counts": dict(sorted(Counter(dev_targets).items())),
        },
        "train_dev_user_overlap": 0,
        "prompt_contract": {
            "input": "behavior-headline-plus-interest-category",
            "id_token_mapping_in_prompt": False,
            "response_is_single_id_token_string": True,
        },
    }


def build_behavior_interest_semantic_id_token_sft_dataset(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    interests_file: Path | str = DEFAULT_INTERESTS_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = 2_000,
    overwrite: bool = False,
    train_size: int = TRAIN_SIZE,
    expected_interests: int = 108,
    samples_per_interest: int = 200,
) -> BuildResult:
    source_dir = Path(source_dir)
    interests_file = Path(interests_file)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source_train = (
        source_dir / "train" / TASK_TYPE / f"{train_size}.parquet"
    )
    source_eval = (
        source_dir / "eval" / TASK_TYPE / "dev_full_grouped.parquet"
    )
    required = (
        source_train,
        source_eval,
        source_dir / "statistics.json",
        source_dir / "metadata.json",
        interests_file,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite."
        )

    mapping_rows = _mapping_rows(
        interests_file,
        expected_interests=expected_interests,
    )
    token_by_label = {
        str(row["target_subcategory"]): str(row["interest_token"])
        for row in mapping_rows
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        pq.write_table(
            pa.Table.from_pylist(
                mapping_rows,
                schema=INTEREST_TOKEN_MAPPING_SCHEMA,
            ),
            staging / "interest_token_mapping.parquet",
            compression="zstd",
            use_dictionary=True,
        )
        train_path = staging / "train" / TASK_TYPE / f"{train_size}.parquet"
        eval_path = (
            staging / "eval" / TASK_TYPE / "dev_full_grouped.parquet"
        )
        _write_transformed_parquet(
            source_train,
            train_path,
            lambda row: _transform_train_row(row, token_by_label),
            batch_size=batch_size,
        )
        _write_transformed_parquet(
            source_eval,
            eval_path,
            lambda row: _transform_dev_row(row, token_by_label),
            batch_size=batch_size,
        )
        statistics = _validate_and_summarize(
            train_path,
            eval_path,
            mapping_rows,
            train_size=train_size,
            samples_per_interest=samples_per_interest,
        )
        _write_json(staging / "statistics.json", statistics)
        artifact_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "metadata.json"
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_dir": portable_path(source_dir),
            "interests_file": portable_path(interests_file),
            "output_dir": portable_path(output_dir),
            "task_type": TASK_TYPE,
            "mapping_order": "target_subcategory-ascending",
            "token_template": "<interest_NNN>",
            "train_size": train_size,
            "expected_interests": expected_interests,
            "samples_per_interest": samples_per_interest,
            "random_initialization_condition": False,
            "history_filtering_at_evaluation": False,
            "source_sha256": {
                portable_path(path): sha256_file(path) for path in required
            },
            "artifact_sha256": {
                path.relative_to(staging).as_posix(): sha256_file(path)
                for path in artifact_paths
            },
        }
        _write_json(staging / "metadata.json", metadata)
        _commit_staging(staging, output_dir, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return BuildResult(
        output_dir=output_dir,
        train_rows=int(statistics["train"]["samples"]),
        train_users=int(statistics["train"]["users"]),
        train_interests=int(statistics["train"]["interests"]),
        dev_impressions=int(statistics["dev"]["impressions"]),
        dev_users=int(statistics["dev"]["users"]),
        dev_target_occurrences=int(statistics["dev"]["target_occurrences"]),
        dev_supported_interests=int(statistics["dev"]["supported_interests"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Behavior+Interest inputs with single-token Semantic-ID "
            "targets for static user-disjoint SFT."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--interests-file", type=Path, default=DEFAULT_INTERESTS_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_behavior_interest_semantic_id_token_sft_dataset(
        args.source_dir,
        args.interests_file,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
