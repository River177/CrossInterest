from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_generative_dataset import (
    GENERATIVE_SCHEMA,
    TASK_TYPES,
)
from new_interest.data.generate_cot_teacher import (
    extract_interest_history,
    validate_reasoning,
)
from new_interest.infrastructure.paths import portable_path

DEFAULT_SIZES = (1_000, 5_000)
RULE_VERSION = "cot-sft-teacher-join-v2-native-thinking"


def _size_name(size: int) -> str:
    return f"{size // 1_000}k" if size % 1_000 == 0 else str(size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_cot_completion(reasoning: str, target: str) -> str:
    clean_reasoning = validate_reasoning(reasoning)
    clean_target = target.strip()
    if not clean_target:
        raise ValueError("Target must be nonempty")
    if "\n" in clean_target or "\r" in clean_target:
        raise ValueError("Target must be a single line")
    return (
        f"<think>\n{clean_reasoning}\n</think>\n"
        f"New topic: {clean_target}"
    )


def _load_teacher_records(
    path: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    records: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["source_sample_id"]), str(row["task_type"]))
        if key in records:
            raise ValueError(f"Duplicate teacher record: {key}")
        records[key] = row
    return records


def _load_source_rows(
    source_dir: Path,
    sizes: tuple[int, ...],
) -> tuple[
    dict[tuple[str, int], list[dict[str, object]]],
    dict[tuple[str, int], Path],
]:
    rows_by_task_size: dict[
        tuple[str, int], list[dict[str, object]]
    ] = {}
    paths: dict[tuple[str, int], Path] = {}
    for task_type in TASK_TYPES:
        previous_ids: list[str] | None = None
        for size in sizes:
            path = (
                source_dir
                / "train"
                / task_type
                / f"{_size_name(size)}.parquet"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = pq.read_table(path).to_pylist()
            if len(rows) != size:
                raise ValueError(
                    f"{path}: expected {size} rows, got {len(rows)}"
                )
            ids = [str(row["source_sample_id"]) for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{path}: duplicate source IDs")
            if previous_ids is not None and ids[: len(previous_ids)] != previous_ids:
                raise ValueError(f"{path}: subsets are not nested")
            previous_ids = ids
            for row in rows:
                if row["task_type"] != task_type:
                    raise ValueError(
                        f"{row['source_sample_id']}: task_type mismatch"
                    )
            rows_by_task_size[(task_type, size)] = rows
            paths[(task_type, size)] = path

    for size in sizes:
        behavior = rows_by_task_size[("behavior_only", size)]
        interest = rows_by_task_size[("interest_only", size)]
        behavior_identity = [
            (
                str(row["source_sample_id"]),
                str(row["impression_id"]),
                str(row["target_subcategory"]),
            )
            for row in behavior
        ]
        interest_identity = [
            (
                str(row["source_sample_id"]),
                str(row["impression_id"]),
                str(row["target_subcategory"]),
            )
            for row in interest
        ]
        if behavior_identity != interest_identity:
            raise ValueError(
                f"{_size_name(size)}: Behavior and Interest IDs differ"
            )
    return rows_by_task_size, paths


def _history_maps(
    rows_by_task_size: dict[
        tuple[str, int], list[dict[str, object]]
    ],
    sizes: tuple[int, ...],
) -> dict[int, dict[str, list[str]]]:
    result: dict[int, dict[str, list[str]]] = {}
    for size in sizes:
        histories: dict[str, list[str]] = {}
        for row in rows_by_task_size[("interest_only", size)]:
            source_id = str(row["source_sample_id"])
            history = extract_interest_history(str(row["prompt"]))
            target = str(row["target_subcategory"])
            if target.casefold() in {item.casefold() for item in history}:
                raise ValueError(f"{source_id}: target appears in history")
            histories[source_id] = history
        result[size] = histories
    return result


def _convert_row(
    source: dict[str, object],
    teacher: dict[str, object],
    history: list[str],
) -> dict[str, object]:
    source_id = str(source["source_sample_id"])
    task_type = str(source["task_type"])
    target = str(source["target_subcategory"])
    if teacher["status"] != "success":
        raise ValueError(
            f"{source_id}/{task_type}: teacher status is "
            f"{teacher['status']}"
        )
    identity_pairs = (
        ("sample_id", "sample_id"),
        ("source_sample_id", "source_sample_id"),
        ("impression_id", "impression_id"),
        ("task_type", "task_type"),
    )
    for source_key, teacher_key in identity_pairs:
        if str(source[source_key]) != str(teacher[teacher_key]):
            raise ValueError(
                f"{source_id}/{task_type}: {source_key} does not match"
            )
    if str(teacher["selected_target"]) != target:
        raise ValueError(
            f"{source_id}/{task_type}: selected targets do not match"
        )
    prompt = str(source["prompt"])
    if str(teacher["original_prompt"]) != prompt:
        raise ValueError(
            f"{source_id}/{task_type}: original prompts do not match"
        )
    if (
        "new_student_prompt" in teacher
        and str(teacher["new_student_prompt"]) != prompt
    ):
        raise ValueError(
            f"{source_id}/{task_type}: Student Prompt does not match "
            "the original SFT22K Prompt"
        )
    if "strict_valid" in teacher and teacher["strict_valid"] is not True:
        raise ValueError(
            f"{source_id}/{task_type}: teacher record is not strict-valid"
        )
    if source["response"] != f"New topic: {target}":
        raise ValueError(
            f"{source_id}/{task_type}: response and target differ"
        )
    if target.casefold() in {item.casefold() for item in history}:
        raise ValueError(f"{source_id}: target appears in history")

    messages = source["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{source_id}/{task_type}: invalid messages")
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError(f"{source_id}/{task_type}: invalid message roles")
    if messages[0]["content"] != prompt:
        raise ValueError(
            f"{source_id}/{task_type}: user message and prompt differ"
        )
    if messages[1]["content"] != source["response"]:
        raise ValueError(
            f"{source_id}/{task_type}: assistant message and response differ"
        )

    completion = format_cot_completion(
        str(teacher["reasoning"]),
        target,
    )
    return {
        **source,
        "response": completion,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": completion,
            },
        ],
    }


def build_cot_sft_datasets(
    source_dir: Path | str,
    teacher_records: Path | str,
    output_dir: Path | str,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    overwrite: bool = False,
) -> Path:
    source_dir = Path(source_dir)
    teacher_records = Path(teacher_records)
    output_dir = Path(output_dir)
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sizes must be unique and strictly increasing")

    output_paths = [
        output_dir / task_type / f"{_size_name(size)}.parquet"
        for task_type in TASK_TYPES
        for size in sizes
    ]
    output_paths.extend(
        [output_dir / "statistics.json", output_dir / "metadata.json"]
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"CoT-SFT outputs exist in {output_dir}. Use --overwrite."
        )

    rows_by_task_size, source_paths = _load_source_rows(
        source_dir,
        sizes,
    )
    histories = _history_maps(rows_by_task_size, sizes)
    teachers = _load_teacher_records(teacher_records)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        output_dir.parent
        / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    statistics: dict[str, object] = {"tasks": {}}
    artifact_paths: list[Path] = []
    try:
        for task_type in TASK_TYPES:
            task_statistics: dict[str, object] = {}
            previous_output_ids: list[str] | None = None
            for size in sizes:
                source_rows = rows_by_task_size[(task_type, size)]
                output_rows: list[dict[str, object]] = []
                for source in source_rows:
                    source_id = str(source["source_sample_id"])
                    key = (source_id, task_type)
                    teacher = teachers.get(key)
                    if teacher is None:
                        raise ValueError(
                            f"Missing teacher record for {key}"
                        )
                    output_rows.append(
                        _convert_row(
                            source,
                            teacher,
                            histories[size][source_id],
                        )
                    )

                output_ids = [
                    str(row["source_sample_id"]) for row in output_rows
                ]
                if (
                    previous_output_ids is not None
                    and output_ids[: len(previous_output_ids)]
                    != previous_output_ids
                ):
                    raise ValueError(
                        f"{task_type}/{_size_name(size)} is not nested"
                    )
                previous_output_ids = output_ids
                relative = (
                    Path(task_type) / f"{_size_name(size)}.parquet"
                )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    pa.Table.from_pylist(
                        output_rows,
                        schema=GENERATIVE_SCHEMA,
                    ),
                    destination,
                    compression="zstd",
                    use_dictionary=True,
                    row_group_size=2_000,
                )
                artifact_paths.append(relative)
                reasoning_lengths = [
                    len(str(teachers[(source_id, task_type)]["reasoning"]))
                    for source_id in output_ids
                ]
                task_statistics[_size_name(size)] = {
                    "samples": len(output_rows),
                    "unique_targets": len(
                        {
                            str(row["target_subcategory"])
                            for row in output_rows
                        }
                    ),
                    "mean_reasoning_characters": (
                        sum(reasoning_lengths) / len(reasoning_lengths)
                    ),
                    "max_reasoning_characters": max(reasoning_lengths),
                    "target_counts": dict(
                        sorted(
                            Counter(
                                str(row["target_subcategory"])
                                for row in output_rows
                            ).items()
                        )
                    ),
                }
            statistics["tasks"][task_type] = task_statistics

        statistics_path = staging / "statistics.json"
        statistics_path.write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_paths.append(Path("statistics.json"))

        source_hashes = {
            str(path.relative_to(source_dir)): _sha256(path)
            for path in source_paths.values()
        }
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_dir": portable_path(source_dir),
            "teacher_records": portable_path(teacher_records),
            "sizes": list(sizes),
            "selection": "source-order-nested-prefix",
            "assistant_format": (
                "<think>\\n{reasoning}\\n</think>\\n"
                "New topic: {selected target}"
            ),
            "student_target_exposure": "assistant-only",
            "student_prompt_policy": "unchanged_sft22k_prompt",
            "qwen_enable_thinking": True,
            "source_sha256": source_hashes,
            "teacher_sha256": _sha256(teacher_records),
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
        artifact_paths.append(Path("metadata.json"))

        output_dir.mkdir(parents=True, exist_ok=True)
        for relative in artifact_paths:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join teacher reasoning to nested Behavior/Interest SFT."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--teacher-records",
        type=Path,
        default=Path(
            "artifacts/cot_teacher/gpt56_sol/teacher_records.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/cot_sft_experiments"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = build_cot_sft_datasets(
        args.source_dir,
        args.teacher_records,
        args.output_dir,
        sizes=tuple(args.sizes),
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
