from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.generate_preference_state_teacher import (
    history_subcategories,
    student_prompt,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "preference-state-hierarchical-grpo-data-v1"
EXPECTED_ROWS = 21_600
EXPECTED_CLASSES = 108


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_grpo_dataset(
    *,
    train_file: Path | str,
    mapping_file: Path | str,
    vocabulary_file: Path | str,
    output_dir: Path | str,
) -> Path:
    train_file = Path(train_file)
    mapping_file = Path(mapping_file)
    vocabulary_file = Path(vocabulary_file)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    train = pq.read_table(train_file).to_pylist()
    if len(train) != EXPECTED_ROWS:
        raise ValueError("GRPO Train must contain 21,600 rows")
    mapping = sorted(
        pq.read_table(mapping_file).to_pylist(),
        key=lambda row: int(row["interest_index"]),
    )
    if len(mapping) != EXPECTED_CLASSES:
        raise ValueError("GRPO Semantic-ID mapping must contain 108 rows")
    parent_rows = pq.read_table(
        vocabulary_file, columns=["category", "subcategory"]
    ).to_pylist()
    parent_by_subcategory = {
        str(row["subcategory"]): str(row["category"])
        for row in parent_rows
    }
    tokens = [str(row["interest_token"]) for row in mapping]
    subcategories = [str(row["target_subcategory"]) for row in mapping]
    categories = [parent_by_subcategory[value] for value in subcategories]
    token_by_subcategory = dict(zip(subcategories, tokens))
    category_by_subcategory = dict(zip(subcategories, categories))

    output_rows = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in train:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"Duplicate GRPO source ID: {source_id}")
        seen.add(source_id)
        target = str(row["target_subcategory"])
        target_token = token_by_subcategory[target]
        if str(row["response"]) != target_token:
            raise ValueError(f"GRPO source response mismatch: {source_id}")
        history = history_subcategories(str(row["prompt"]))
        if target.casefold() in {value.casefold() for value in history}:
            raise ValueError(f"GRPO target appears in History: {source_id}")
        counts[target] += 1
        output_rows.append(
            {
                "data_source": "cross_interest_preference_state",
                "prompt": [
                    {
                        "role": "user",
                        "content": student_prompt(str(row["prompt"])),
                    }
                ],
                "ability": "new_interest_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": target_token,
                },
                "extra_info": {
                    "source_sample_id": source_id,
                    "impression_id": str(row["impression_id"]),
                    "user_id": str(row["user_id"]),
                    "target_token": target_token,
                    "target_subcategory": target,
                    "target_category": category_by_subcategory[target],
                    "history_subcategories": history,
                    "interest_tokens": tokens,
                    "interest_subcategories": subcategories,
                    "interest_categories": categories,
                },
            }
        )
    if len(counts) != EXPECTED_CLASSES or set(counts.values()) != {200}:
        raise ValueError("GRPO Train must contain 200 rows per class")

    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        train_output = staging / "train.parquet"
        pq.write_table(
            pa.Table.from_pylist(output_rows),
            train_output,
            compression="zstd",
            use_dictionary=True,
            row_group_size=1_000,
        )
        statistics = {
            "rows": len(output_rows),
            "classes": len(counts),
            "per_class": 200,
            "target_counts": dict(sorted(counts.items())),
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "rule_version": RULE_VERSION,
            "source_train": portable_path(train_file),
            "mapping_file": portable_path(mapping_file),
            "vocabulary_file": portable_path(vocabulary_file),
            "reward_levels": {
                "invalid_or_seen": 0.0,
                "novel_other_parent": 0.10,
                "novel_same_parent": 0.35,
                "exact": 1.0,
            },
            "source_sha256": {
                "train": _sha256(train_file),
                "mapping": _sha256(mapping_file),
                "vocabulary": _sha256(vocabulary_file),
            },
            "artifact_sha256": {
                "train.parquet": _sha256(train_output),
                "statistics.json": _sha256(staging / "statistics.json"),
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--vocabulary-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        build_grpo_dataset(
            train_file=args.train_file,
            mapping_file=args.mapping_file,
            vocabulary_file=args.vocabulary_file,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
