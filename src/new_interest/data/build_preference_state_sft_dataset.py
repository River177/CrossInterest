from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "preference-state-reasoning-sft-v1"
EXPECTED_ROWS = 1_080
EXPECTED_CLASSES = 108
EXPECTED_PER_CLASS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reasoning_sft_dataset(
    *,
    train_file: Path | str,
    teacher_records: Path | str,
    teacher_metadata: Path | str,
    output_dir: Path | str,
    expected_rows: int = EXPECTED_ROWS,
    expected_classes: int = EXPECTED_CLASSES,
    expected_per_class: int = EXPECTED_PER_CLASS,
) -> Path:
    train_file = Path(train_file)
    teacher_records = Path(teacher_records)
    teacher_metadata = Path(teacher_metadata)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    metadata = json.loads(teacher_metadata.read_text(encoding="utf-8"))
    if metadata["requested_rows"] != expected_rows:
        raise ValueError("Teacher requested-row count mismatch")
    if metadata["status_counts"] != {"success": expected_rows}:
        raise ValueError("Teacher generation is not completely successful")
    if (
        metadata["artifact_sha256"]["teacher_records.parquet"]
        != _sha256(teacher_records)
    ):
        raise ValueError("Teacher record hash mismatch")
    if metadata["source"]["train_sha256"] != _sha256(train_file):
        raise ValueError("Teacher and SFT source Train hashes differ")

    source_table = pq.read_table(train_file)
    source_rows = source_table.to_pylist()
    source_by_id = {
        str(row["source_sample_id"]): row for row in source_rows
    }
    if len(source_by_id) != len(source_rows):
        raise ValueError("Source Train has duplicate source_sample_id")
    teachers = sorted(
        pq.read_table(teacher_records).to_pylist(),
        key=lambda row: int(row["selection_rank"]),
    )
    if len(teachers) != expected_rows:
        raise ValueError("Teacher record row count mismatch")
    if [int(row["selection_rank"]) for row in teachers] != list(
        range(expected_rows)
    ):
        raise ValueError("Teacher selection ranks are not contiguous")

    output_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for teacher in teachers:
        source_id = str(teacher["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"Duplicate Teacher source ID: {source_id}")
        seen.add(source_id)
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Teacher row is absent from Train: {source_id}")
        target = str(source["target_subcategory"])
        if teacher["status"] != "success":
            raise ValueError(f"Teacher row is not successful: {source_id}")
        if str(teacher["target_subcategory"]) != target:
            raise ValueError(f"Teacher target mismatch: {source_id}")
        if str(teacher["target_token"]) != str(source["response"]):
            raise ValueError(f"Teacher Semantic ID mismatch: {source_id}")
        if str(teacher["original_prompt"]) != str(source["prompt"]):
            raise ValueError(f"Teacher original prompt mismatch: {source_id}")
        student = str(teacher["student_prompt"])
        completion = str(teacher["completion"])
        if not completion.endswith(
            f"<answer>{source['response']}</answer>"
        ):
            raise ValueError(f"Teacher completion mismatch: {source_id}")
        reasoning = completion.split("</think>", 1)[0]
        if target.casefold() in reasoning.casefold():
            raise ValueError(f"Teacher reasoning reveals target: {source_id}")
        output_rows.append(
            {
                **source,
                "prompt": student,
                "response": completion,
                "messages": [
                    {"role": "user", "content": student},
                    {"role": "assistant", "content": completion},
                ],
            }
        )

    counts = Counter(str(row["target_subcategory"]) for row in output_rows)
    if len(counts) != expected_classes or set(counts.values()) != {
        expected_per_class
    }:
        raise ValueError("Reasoning-SFT output is not class balanced")

    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        dataset_path = staging / "train.parquet"
        pq.write_table(
            type(source_table).from_pylist(output_rows, schema=source_table.schema),
            dataset_path,
            compression="zstd",
            use_dictionary=True,
        )
        statistics = {
            "rows": len(output_rows),
            "classes": len(counts),
            "per_class": expected_per_class,
            "target_counts": dict(sorted(counts.items())),
            "history_length_min": min(
                int(row["history_length"]) for row in output_rows
            ),
            "history_length_max": max(
                int(row["history_length"]) for row in output_rows
            ),
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_metadata = {
            "rule_version": RULE_VERSION,
            "source_train": portable_path(train_file),
            "teacher_records": portable_path(teacher_records),
            "teacher_metadata": portable_path(teacher_metadata),
            "assistant_format": (
                "<think>{history summary; preference state; potential expansion}"
                "</think><answer><interest_NNN></answer>"
            ),
            "target_disclosure": "final-answer-only",
            "student_input": "behavior_plus_interest_history_only",
            "source_sha256": {
                "train": _sha256(train_file),
                "teacher_records": _sha256(teacher_records),
                "teacher_metadata": _sha256(teacher_metadata),
            },
            "artifact_sha256": {
                "train.parquet": _sha256(dataset_path),
                "statistics.json": _sha256(staging / "statistics.json"),
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--teacher-records", type=Path, required=True)
    parser.add_argument("--teacher-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = build_reasoning_sft_dataset(
        train_file=args.train_file,
        teacher_records=args.teacher_records,
        teacher_metadata=args.teacher_metadata,
        output_dir=args.output_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
