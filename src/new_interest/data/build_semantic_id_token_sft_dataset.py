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

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "semantic-id-token-sft-v1"
DEFAULT_SOURCE_DIR = Path(
    "data/training/sft_static_user_disjoint_balanced_200_h50_v1"
)
DEFAULT_INTERESTS_FILE = Path(
    "data/training/"
    "generative_new_interest_static_user_disjoint_balanced_200_h50_v1/"
    "retained_interests.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_v1"
)
TASK_TYPES = ("behavior_only", "interest_only")
TRAIN_SIZE = 21_600
PROMPT_SUFFIX = (
    'Begin your response with "New topic: " and provide only the category '
    "name after it."
)
ID_TOKEN_PROMPT_SUFFIX = (
    "Respond with only the learned interest token assigned to the predicted "
    "category, in the form <interest_NNN>."
)


INTEREST_TOKEN_MAPPING_SCHEMA = pa.schema(
    [
        pa.field("interest_index", pa.int16(), nullable=False),
        pa.field("interest_token", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
    ]
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _required_source_paths(
    source_dir: Path,
    *,
    train_size: int,
) -> tuple[Path, ...]:
    return (
        *(
            source_dir / "train" / task_type / f"{train_size}.parquet"
            for task_type in TASK_TYPES
        ),
        *(
            source_dir / "eval" / task_type / "dev_full_grouped.parquet"
            for task_type in TASK_TYPES
        ),
        source_dir / "subset_manifest.parquet",
        source_dir / "statistics.json",
        source_dir / "metadata.json",
    )


def _mapping_rows(
    interests_file: Path,
    *,
    expected_interests: int,
) -> list[dict[str, object]]:
    schema_names = set(pq.ParquetFile(interests_file).schema_arrow.names)
    if "target_subcategory" not in schema_names:
        raise ValueError(
            f"{interests_file}: missing target_subcategory column"
        )
    labels = sorted(
        {
            str(row["target_subcategory"])
            for row in pq.read_table(
                interests_file,
                columns=["target_subcategory"],
            ).to_pylist()
        }
    )
    if len(labels) != expected_interests:
        raise ValueError(
            f"Expected exactly {expected_interests} retained interests, "
            f"got {len(labels)}"
        )
    return [
        {
            "interest_index": index,
            "interest_token": f"<interest_{index:03d}>",
            "target_subcategory": label,
        }
        for index, label in enumerate(labels)
    ]


def _replace_prompt_suffix(prompt: object, *, context: str) -> str:
    text = str(prompt)
    if not text.endswith(PROMPT_SUFFIX):
        raise ValueError(f"{context}: prompt does not use the expected suffix")
    return text[: -len(PROMPT_SUFFIX)] + ID_TOKEN_PROMPT_SUFFIX


def _transform_train_row(
    row: dict[str, object],
    token_by_label: dict[str, str],
) -> dict[str, object]:
    target = str(row["target_subcategory"])
    if target not in token_by_label:
        raise ValueError(f"Unknown Train target: {target}")
    token = token_by_label[target]
    prompt = _replace_prompt_suffix(
        row["prompt"],
        context=str(row.get("sample_id", row.get("source_sample_id"))),
    )
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{row.get('sample_id')}: invalid messages")
    if str(messages[0].get("role")) != "user":
        raise ValueError(f"{row.get('sample_id')}: first message is not user")
    if str(messages[1].get("role")) != "assistant":
        raise ValueError(
            f"{row.get('sample_id')}: second message is not assistant"
        )
    output = dict(row)
    output["prompt"] = prompt
    output["response"] = token
    output["messages"] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": token},
    ]
    return output


def _transform_dev_row(
    row: dict[str, object],
    token_by_label: dict[str, str],
) -> dict[str, object]:
    labels = [str(value) for value in row["acceptable_subcategories"]]
    missing = sorted(set(labels).difference(token_by_label))
    if missing:
        raise ValueError(
            f"{row.get('impression_id')}: unknown Dev targets {missing}"
        )
    output = dict(row)
    output["prompt"] = _replace_prompt_suffix(
        row["prompt"],
        context=str(row.get("impression_id")),
    )
    output["acceptable_responses"] = [
        token_by_label[label] for label in labels
    ]
    return output


def _write_transformed_parquet(
    source_path: Path,
    output_path: Path,
    transform,
    *,
    batch_size: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(source_path)
    writer = pq.ParquetWriter(
        output_path,
        source.schema_arrow,
        compression="zstd",
        use_dictionary=True,
    )
    count = 0
    try:
        for batch in source.iter_batches(batch_size=batch_size):
            rows = [transform(row) for row in batch.to_pylist()]
            writer.write_table(
                pa.Table.from_pylist(rows, schema=source.schema_arrow),
                row_group_size=batch_size,
            )
            count += len(rows)
    finally:
        writer.close()
    return count


def _identity_rows(path: Path, columns: Iterable[str]) -> list[tuple[object, ...]]:
    names = list(columns)
    return [
        tuple(row[name] for name in names)
        for row in pq.read_table(path, columns=names).to_pylist()
    ]


def _validate_and_summarize(
    staging: Path,
    mapping_rows: list[dict[str, object]],
    *,
    train_size: int,
    samples_per_interest: int,
) -> dict[str, object]:
    train_paths = {
        task: staging / "train" / task / f"{train_size}.parquet"
        for task in TASK_TYPES
    }
    eval_paths = {
        task: staging / "eval" / task / "dev_full_grouped.parquet"
        for task in TASK_TYPES
    }
    train_identity_columns = (
        "source_sample_id",
        "user_id",
        "impression_id",
        "history_length",
        "target_subcategory",
    )
    eval_identity_columns = (
        "impression_id",
        "user_id",
        "timestamp",
        "history_length",
        "acceptable_subcategories",
        "history_subcategories",
    )
    if _identity_rows(
        train_paths["behavior_only"], train_identity_columns
    ) != _identity_rows(train_paths["interest_only"], train_identity_columns):
        raise ValueError("Behavior/Interest Train identities differ")
    if _identity_rows(
        eval_paths["behavior_only"], eval_identity_columns
    ) != _identity_rows(eval_paths["interest_only"], eval_identity_columns):
        raise ValueError("Behavior/Interest Dev identities differ")

    train_rows = pq.read_table(train_paths["behavior_only"]).to_pylist()
    dev_rows = pq.read_table(eval_paths["behavior_only"]).to_pylist()
    train_users = {str(row["user_id"]) for row in train_rows}
    dev_users = {str(row["user_id"]) for row in dev_rows}
    overlap = train_users.intersection(dev_users)
    if overlap:
        raise ValueError(f"Train/Dev user overlap: {len(overlap)}")
    target_counts = Counter(
        str(row["target_subcategory"]) for row in train_rows
    )
    if set(target_counts) != {
        str(row["target_subcategory"]) for row in mapping_rows
    }:
        raise ValueError("Train target vocabulary differs from token mapping")
    if set(target_counts.values()) != {samples_per_interest}:
        raise ValueError(
            "Train is not exactly "
            f"{samples_per_interest} samples per interest"
        )
    if len(train_rows) != train_size or len(train_users) != train_size:
        raise ValueError(
            f"Train must contain {train_size:,} rows from "
            f"{train_size:,} distinct users"
        )
    history_lengths = [int(row["history_length"]) for row in train_rows]
    dev_history_lengths = [int(row["history_length"]) for row in dev_rows]
    if max(history_lengths + dev_history_lengths) > 50:
        raise ValueError("History length exceeds 50")
    dev_targets = [
        str(target)
        for row in dev_rows
        for target in row["acceptable_subcategories"]
    ]
    return {
        "rule_version": RULE_VERSION,
        "mapping": {
            "interests": len(mapping_rows),
            "first_token": mapping_rows[0]["interest_token"],
            "last_token": mapping_rows[-1]["interest_token"],
        },
        "train": {
            "samples": len(train_rows),
            "users": len(train_users),
            "interests": len(target_counts),
            "samples_per_interest": samples_per_interest,
            "max_history_length": max(history_lengths),
            "target_counts": dict(sorted(target_counts.items())),
        },
        "dev": {
            "impressions": len(dev_rows),
            "users": len(dev_users),
            "target_occurrences": len(dev_targets),
            "supported_interests": len(set(dev_targets)),
            "max_history_length": max(dev_history_lengths),
            "target_counts": dict(sorted(Counter(dev_targets).items())),
        },
        "train_dev_user_overlap": 0,
        "prompt_contract": {
            "old_suffix_present": False,
            "id_token_mapping_in_prompt": False,
            "response_is_single_id_token_string": True,
        },
    }


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


def build_semantic_id_token_sft_dataset(
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
    required = (
        *_required_source_paths(source_dir, train_size=train_size),
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
        for task_type in TASK_TYPES:
            _write_transformed_parquet(
                source_dir / "train" / task_type / f"{train_size}.parquet",
                staging / "train" / task_type / f"{train_size}.parquet",
                lambda row, mapping=token_by_label: _transform_train_row(
                    row, mapping
                ),
                batch_size=batch_size,
            )
            _write_transformed_parquet(
                source_dir
                / "eval"
                / task_type
                / "dev_full_grouped.parquet",
                staging
                / "eval"
                / task_type
                / "dev_full_grouped.parquet",
                lambda row, mapping=token_by_label: _transform_dev_row(
                    row, mapping
                ),
                batch_size=batch_size,
            )
        shutil.copy2(
            source_dir / "subset_manifest.parquet",
            staging / "subset_manifest.parquet",
        )
        statistics = _validate_and_summarize(
            staging,
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
        dev_target_occurrences=int(
            statistics["dev"]["target_occurrences"]
        ),
        dev_supported_interests=int(
            statistics["dev"]["supported_interests"]
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the static user-disjoint Semantic ID Token SFT data."
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
    result = build_semantic_id_token_sft_dataset(
        args.source_dir,
        args.interests_file,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
