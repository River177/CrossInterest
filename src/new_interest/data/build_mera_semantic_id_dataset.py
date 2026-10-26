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

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.build_sft_subsets import GROUPED_EVAL_SCHEMA
from new_interest.infrastructure.paths import portable_path
from new_interest.training.semantic_id_tokens import load_interest_token_mapping


RULE_VERSION = "mera-108-category-semantic-id-v1"
TASK_TYPE = "interest_sid"
TRAIN_SIZE = 21_600
DEFAULT_SOURCE_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_sft21600_h50_v1"
)
DEFAULT_GROUPED_DEV_FILE = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1/"
    "eval/behavior_interest/dev_full_grouped.parquet"
)
DEFAULT_MAPPING_FILE = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1/"
    "interest_token_mapping.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1"
)
PROMPT_TEMPLATE = """Given a user's chronological sequence of historical interest category tokens, predict one new interest category token that the user may interact with in the future. The predicted category must not occur in the historical sequence. Respond with exactly one category token and no other text.

Historical interest category tokens:
{history_tokens}"""


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_rows: int
    train_users: int
    category_tokens: int
    dev_impressions: int
    train_empty_histories: int
    dev_empty_histories: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _format_prompt(history_tokens: list[str]) -> str:
    # Preserve the aligned cohort when filtering removes every History event:
    # one Train row and six grouped Dev impressions have an empty SID sequence.
    return PROMPT_TEMPLATE.format(history_tokens=" ".join(history_tokens))


def _history_tokens(
    history_labels: list[str],
    token_by_label: dict[str, str],
) -> tuple[list[str], int]:
    tokens = [
        token_by_label[label]
        for label in history_labels
        if label in token_by_label
    ]
    return tokens, len(history_labels) - len(tokens)


def _transform_train_row(
    row: dict[str, object],
    token_by_label: dict[str, str],
) -> tuple[dict[str, object], int]:
    sample_id = str(row["sample_id"])
    history = row["history"]
    if not isinstance(history, list) or not history:
        raise ValueError(f"{sample_id}: invalid History")
    history_labels = [str(event["subcategory"]) for event in history]
    history_tokens, dropped = _history_tokens(history_labels, token_by_label)
    target = str(row["target_interest"]["subcategory"])
    if target not in token_by_label:
        raise ValueError(f"{sample_id}: target is outside the 108 categories")
    if target in history_labels:
        raise ValueError(f"{sample_id}: target appears in History")
    prompt = _format_prompt(history_tokens)
    response = token_by_label[target]
    return (
        {
            "sample_id": f"{sample_id}:{TASK_TYPE}",
            "source_sample_id": sample_id,
            "split": "train",
            "task_type": TASK_TYPE,
            "impression_id": str(row["impression_id"]),
            "user_id": str(row["user_id"]),
            "timestamp": row["timestamp"],
            "history_length": len(history_tokens),
            "prompt": prompt,
            "response": response,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "target_subcategory": target,
        },
        dropped,
    )


def _transform_dev_row(
    row: dict[str, object],
    token_by_label: dict[str, str],
) -> tuple[dict[str, object], int]:
    impression_id = str(row["impression_id"])
    history_labels = [str(value) for value in row["history_subcategories"]]
    targets = [str(value) for value in row["acceptable_subcategories"]]
    missing_targets = sorted(set(targets).difference(token_by_label))
    if missing_targets:
        raise ValueError(
            f"{impression_id}: targets outside 108 categories {missing_targets}"
        )
    if set(targets).intersection(history_labels):
        raise ValueError(f"{impression_id}: target appears in History")
    history_tokens, dropped = _history_tokens(history_labels, token_by_label)
    output = dict(row)
    output["history_length"] = len(history_tokens)
    output["history_subcategories"] = [
        label for label in history_labels if label in token_by_label
    ]
    output["prompt"] = _format_prompt(history_tokens)
    output["acceptable_responses"] = [
        token_by_label[target] for target in targets
    ]
    return output, dropped


def build_mera_semantic_id_dataset(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    grouped_dev_file: Path | str = DEFAULT_GROUPED_DEV_FILE,
    mapping_file: Path | str = DEFAULT_MAPPING_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = 2_000,
    overwrite: bool = False,
    train_size: int = TRAIN_SIZE,
    samples_per_target: int = 200,
) -> BuildResult:
    source_dir = Path(source_dir)
    grouped_dev_file = Path(grouped_dev_file)
    mapping_file = Path(mapping_file)
    output_dir = Path(output_dir)
    if batch_size <= 0 or train_size <= 0 or samples_per_target <= 0:
        raise ValueError(
            "batch_size, train_size, and samples_per_target must be positive"
        )
    train_source = source_dir / "train.parquet"
    required = (train_source, grouped_dev_file, mapping_file)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite."
        )

    mapping_rows = load_interest_token_mapping(mapping_file)
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
    train_path = staging / "train" / TASK_TYPE / f"{train_size}.parquet"
    dev_path = staging / "eval" / TASK_TYPE / "dev_full_grouped.parquet"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(mapping_file, staging / "interest_token_mapping.parquet")
        train_writer = pq.ParquetWriter(
            train_path,
            GENERATIVE_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        train_count = 0
        train_users: set[str] = set()
        target_counts: Counter[str] = Counter()
        train_dropped_events = 0
        train_empty_histories = 0
        try:
            for batch in pq.ParquetFile(train_source).iter_batches(
                batch_size=batch_size
            ):
                transformed = [
                    _transform_train_row(row, token_by_label)
                    for row in batch.to_pylist()
                ]
                rows = [row for row, _ in transformed]
                train_writer.write_table(
                    pa.Table.from_pylist(rows, schema=GENERATIVE_SCHEMA)
                )
                train_count += len(rows)
                train_users.update(str(row["user_id"]) for row in rows)
                target_counts.update(
                    str(row["target_subcategory"]) for row in rows
                )
                train_dropped_events += sum(value for _, value in transformed)
                train_empty_histories += sum(
                    int(row["history_length"]) == 0 for row in rows
                )
        finally:
            train_writer.close()

        dev_writer = pq.ParquetWriter(
            dev_path,
            GROUPED_EVAL_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        dev_count = 0
        dev_users: set[str] = set()
        dev_target_occurrences = 0
        dev_supported: set[str] = set()
        dev_dropped_events = 0
        dev_empty_histories = 0
        try:
            for batch in pq.ParquetFile(grouped_dev_file).iter_batches(
                batch_size=batch_size
            ):
                transformed = [
                    _transform_dev_row(row, token_by_label)
                    for row in batch.to_pylist()
                ]
                rows = [row for row, _ in transformed]
                dev_writer.write_table(
                    pa.Table.from_pylist(rows, schema=GROUPED_EVAL_SCHEMA)
                )
                dev_count += len(rows)
                dev_users.update(str(row["user_id"]) for row in rows)
                dev_dropped_events += sum(value for _, value in transformed)
                dev_empty_histories += sum(
                    int(row["history_length"]) == 0 for row in rows
                )
                for row in rows:
                    values = [
                        str(value)
                        for value in row["acceptable_subcategories"]
                    ]
                    dev_target_occurrences += len(values)
                    dev_supported.update(values)
        finally:
            dev_writer.close()

        if train_count != train_size or len(train_users) != train_size:
            raise ValueError(f"MeRA Train is not {train_size:,} rows/users")
        if set(target_counts) != set(token_by_label) or set(target_counts.values()) != {
            samples_per_target
        }:
            raise ValueError(
                f"MeRA Train is not {samples_per_target} rows per category"
            )
        train_dev_overlap = train_users.intersection(dev_users)
        if train_dev_overlap:
            raise ValueError(
                f"MeRA Train/Dev user overlap: {len(train_dev_overlap)}"
            )

        statistics = {
            "rule_version": RULE_VERSION,
            "task_type": TASK_TYPE,
            "category_tokens": len(mapping_rows),
            "train": {
                "samples": train_count,
                "users": len(train_users),
                "target_counts": dict(sorted(target_counts.items())),
                "dropped_out_of_108_history_events": train_dropped_events,
                "empty_histories_after_filter": train_empty_histories,
            },
            "dev": {
                "impressions": dev_count,
                "users": len(dev_users),
                "target_occurrences": dev_target_occurrences,
                "supported_interests": len(dev_supported),
                "dropped_out_of_108_history_events": dev_dropped_events,
                "empty_histories_after_filter": dev_empty_histories,
            },
            "train_dev_user_overlap": 0,
            "prompt_contract": {
                "history": "chronological Semantic-ID tokens from 108 categories",
                "out_of_vocabulary_history_categories": "dropped",
                "headlines_present": False,
                "category_names_present": False,
                "response": "one Semantic-ID token from the same 108 categories",
            },
        }
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
            "grouped_dev_file": portable_path(grouped_dev_file),
            "mapping_file": portable_path(mapping_file),
            "output_dir": portable_path(output_dir),
            "category_token_count": len(mapping_rows),
            "token_template": "<interest_NNN>",
            "train_size": train_size,
            "samples_per_target": samples_per_target,
            "source_sha256": {
                portable_path(path): _sha256(path) for path in required
            },
            "artifact_sha256": {
                path.relative_to(staging).as_posix(): _sha256(path)
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
        train_rows=train_count,
        train_users=len(train_users),
        category_tokens=len(mapping_rows),
        dev_impressions=dev_count,
        train_empty_histories=train_empty_histories,
        dev_empty_histories=dev_empty_histories,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build faithful MeRA 108-category SID History to SID target data."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--grouped-dev-file", type=Path, default=DEFAULT_GROUPED_DEV_FILE
    )
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_mera_semantic_id_dataset(
        args.source_dir,
        args.grouped_dev_file,
        args.mapping_file,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, default=str, indent=2))


if __name__ == "__main__":
    main()
