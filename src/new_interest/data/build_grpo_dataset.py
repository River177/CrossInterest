from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path

RULE_VERSION = "grpo-cot-acceptable-targets-v5"
DEFAULT_SIZES = (1_000, 5_000)

COT_INSTRUCTION = """Reason about the user's history inside <think> and </think>. The reasoning must be non-empty, and you must close </think> before answering. Then, on a new line, provide the final answer in exactly this format:
New topic: <category name>

Output only the think block and the final answer. Do not include any other text."""

MESSAGE_TYPE = pa.struct(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
    ]
)

REWARD_MODEL_TYPE = pa.struct(
    [
        pa.field("style", pa.string(), nullable=False),
        pa.field("ground_truth", pa.string(), nullable=False),
    ]
)

EXTRA_INFO_TYPE = pa.struct(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "acceptable_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "retained_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("retained_label_count", pa.int16(), nullable=False),
    ]
)

GRPO_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("prompt", pa.list_(MESSAGE_TYPE), nullable=False),
        pa.field("ability", pa.string(), nullable=False),
        pa.field("reward_model", REWARD_MODEL_TYPE, nullable=False),
        pa.field("extra_info", EXTRA_INFO_TYPE, nullable=False),
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


def append_cot_instruction(prompt: str) -> str:
    prompt = prompt.rstrip()
    if "<think>" in prompt or "</think>" in prompt:
        raise ValueError("Source prompt already contains think tags")
    return f"{prompt}\n\n{COT_INSTRUCTION}"


def _extract_interest_history(prompt: str) -> list[str]:
    try:
        history_block = prompt.split("\n\nBased on ", 1)[0]
        numbered_history = history_block.split("\n\n", 1)[1]
    except (IndexError, ValueError) as error:
        raise ValueError("Cannot parse Interest-only history") from error
    history = []
    for line in numbered_history.splitlines():
        _, separator, value = line.partition(". ")
        if not separator or not value:
            raise ValueError(f"Invalid numbered history line: {line!r}")
        history.append(value)
    if not history:
        raise ValueError("Interest-only history is empty")
    return history


def _convert_row(
    row: dict[str, object],
    history_subcategories: list[str],
    acceptable_subcategories: list[str],
    retained_subcategories: list[str],
) -> dict[str, object]:
    prompt = str(row["prompt"])
    response = str(row["response"])
    target_subcategory = str(row["target_subcategory"])
    if response != f"New topic: {target_subcategory}":
        raise ValueError(
            f"{row['source_sample_id']}: response and target differ"
        )
    return {
        "data_source": "cross_interest",
        "prompt": [
            {
                "role": "user",
                "content": append_cot_instruction(prompt),
            }
        ],
        "ability": "new_topic_prediction",
        "reward_model": {
            "style": "rule",
            "ground_truth": response,
        },
        "extra_info": {
            "source_sample_id": str(row["source_sample_id"]),
            "impression_id": str(row["impression_id"]),
            "task_type": str(row["task_type"]),
            "target_subcategory": target_subcategory,
            "history_subcategories": list(history_subcategories),
            "acceptable_subcategories": list(acceptable_subcategories),
            "retained_subcategories": list(retained_subcategories),
            "retained_label_count": len(retained_subcategories),
        },
    }


def build_grpo_datasets(
    sft_dir: Path | str,
    output_dir: Path | str,
    *,
    acceptable_target_manifest: Path | str,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    overwrite: bool = False,
) -> Path:
    sft_dir = Path(sft_dir)
    output_dir = Path(output_dir)
    acceptable_target_manifest = Path(acceptable_target_manifest)
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sizes must be unique and increasing")
    if not acceptable_target_manifest.is_file():
        raise FileNotFoundError(acceptable_target_manifest)
    acceptable_metadata_path = (
        acceptable_target_manifest.parent / "metadata.json"
    )
    if not acceptable_metadata_path.is_file():
        raise FileNotFoundError(acceptable_metadata_path)
    acceptable_metadata = json.loads(
        acceptable_metadata_path.read_text(encoding="utf-8")
    )
    retained_subcategories = [
        str(value)
        for value in acceptable_metadata["retained_subcategories"]
    ]
    retained_label_count = int(
        acceptable_metadata["retained_label_count"]
    )
    if (
        retained_label_count != len(retained_subcategories)
        or len(set(retained_subcategories)) != retained_label_count
    ):
        raise ValueError("Invalid retained label metadata")
    expected_manifest_hash = acceptable_metadata.get(
        "artifact_sha256", {}
    ).get(acceptable_target_manifest.name)
    if (
        expected_manifest_hash is not None
        and expected_manifest_hash != _sha256(acceptable_target_manifest)
    ):
        raise ValueError("Acceptable-target manifest hash mismatch")
    manifest_rows = pq.read_table(acceptable_target_manifest).to_pylist()
    acceptable_by_source: dict[str, dict[str, object]] = {}
    for row in manifest_rows:
        source_sample_id = str(row["source_sample_id"])
        if source_sample_id in acceptable_by_source:
            raise ValueError(
                f"Duplicate manifest source_sample_id {source_sample_id}"
            )
        acceptable_by_source[source_sample_id] = row

    output_paths = [
        output_dir / task_type / f"{_size_name(size)}.parquet"
        for task_type in ("behavior_only", "interest_only")
        for size in sizes
    ]
    output_paths.extend(
        [output_dir / "statistics.json", output_dir / "metadata.json"]
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"GRPO outputs already exist in {output_dir}. Use --overwrite."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".grpo-datasets-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        statistics: dict[str, object] = {"tasks": {}}
        source_hashes: dict[str, str] = {}
        history_maps: dict[str, dict[str, list[str]]] = {}
        paired_source_ids: dict[str, list[str]] = {}
        for size in sizes:
            name = _size_name(size)
            interest_path = (
                sft_dir / "train" / "interest_only" / f"{name}.parquet"
            )
            interest_rows = pq.read_table(
                interest_path,
                columns=["source_sample_id", "prompt"],
            ).to_pylist()
            history_maps[name] = {
                str(row["source_sample_id"]): _extract_interest_history(
                    str(row["prompt"])
                )
                for row in interest_rows
            }
        for task_type in ("behavior_only", "interest_only"):
            statistics["tasks"][task_type] = {}
            previous_ids: list[str] | None = None
            for size in sizes:
                name = _size_name(size)
                source_path = (
                    sft_dir
                    / "train"
                    / task_type
                    / f"{name}.parquet"
                )
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                source_hashes[
                    str(source_path.relative_to(sft_dir))
                ] = _sha256(source_path)
                source_rows = pq.read_table(
                    source_path,
                    columns=[
                        "source_sample_id",
                        "impression_id",
                        "task_type",
                        "prompt",
                        "response",
                        "target_subcategory",
                    ],
                ).to_pylist()
                if len(source_rows) != size:
                    raise ValueError(
                        f"{source_path}: expected {size}, got "
                        f"{len(source_rows)}"
                    )
                source_ids = [
                    str(row["source_sample_id"]) for row in source_rows
                ]
                if task_type == "behavior_only":
                    paired_source_ids[name] = source_ids
                elif source_ids != paired_source_ids[name]:
                    raise ValueError(
                        f"Behavior/Interest source IDs differ for {name}"
                    )
                if previous_ids is not None and source_ids[: len(previous_ids)] != previous_ids:
                    raise ValueError(
                        f"{task_type}/{name} is not nested"
                    )
                previous_ids = source_ids
                for row in source_rows:
                    source_sample_id = str(row["source_sample_id"])
                    manifest = acceptable_by_source.get(source_sample_id)
                    if manifest is None:
                        raise ValueError(
                            f"Missing acceptable targets for "
                            f"{source_sample_id}"
                        )
                    if (
                        str(manifest["impression_id"])
                        != str(row["impression_id"])
                        or str(manifest["selected_target_subcategory"])
                        != str(row["target_subcategory"])
                        or size not in {
                            int(value)
                            for value in manifest["included_sizes"]
                        }
                    ):
                        raise ValueError(
                            f"Manifest mismatch for {source_sample_id}"
                        )
                    acceptable = {
                        str(value)
                        for value in manifest["acceptable_subcategories"]
                    }
                    if (
                        not acceptable
                        or str(row["target_subcategory"]) not in acceptable
                        or not acceptable.issubset(retained_subcategories)
                    ):
                        raise ValueError(
                            f"Invalid acceptable targets for "
                            f"{source_sample_id}"
                        )
                output_rows = [
                    _convert_row(
                        row,
                        history_maps[name][str(row["source_sample_id"])],
                        [
                            str(value)
                            for value in acceptable_by_source[
                                str(row["source_sample_id"])
                            ]["acceptable_subcategories"]
                        ],
                        retained_subcategories,
                    )
                    for row in source_rows
                ]
                destination = staging / task_type / f"{name}.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    pa.Table.from_pylist(output_rows, schema=GRPO_SCHEMA),
                    destination,
                    compression="zstd",
                    use_dictionary=True,
                    row_group_size=2_000,
                )
                prompt_lengths = [
                    len(row["prompt"][0]["content"])
                    for row in output_rows
                ]
                statistics["tasks"][task_type][name] = {
                    "samples": len(output_rows),
                    "unique_source_ids": len(set(source_ids)),
                    "min_prompt_characters": min(prompt_lengths),
                    "max_prompt_characters": max(prompt_lengths),
                    "mean_prompt_characters": (
                        sum(prompt_lengths) / len(prompt_lengths)
                    ),
                }

        artifact_paths = [
            Path(task_type) / f"{_size_name(size)}.parquet"
            for task_type in ("behavior_only", "interest_only")
            for size in sizes
        ]
        (staging / "statistics.json").write_text(
            json.dumps(statistics, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_paths.append(Path("statistics.json"))
        artifact_hashes = {
            str(path): _sha256(staging / path)
            for path in artifact_paths
        }
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "sizes": list(sizes),
            "branching": "independent_from_sft22k",
            "target_policy": "all_retained_targets_for_impression",
            "reasoning_mode": "cot",
            "qwen_enable_thinking": True,
            "cot_instruction": COT_INSTRUCTION,
            "reward": {
                "type": "correctness_gated_answer_and_format",
                "correct_malformed": 0.8,
                "correct_strict": 1.0,
                "invalid_novel_or_incorrect": 0.0,
            },
            "source_sft_dir": portable_path(sft_dir),
            "acceptable_target_manifest": portable_path(
                acceptable_target_manifest
            ),
            "retained_label_count": retained_label_count,
            "source_sha256": source_hashes,
            "acceptable_target_manifest_sha256": _sha256(
                acceptable_target_manifest
            ),
            "acceptable_target_metadata_sha256": _sha256(
                acceptable_metadata_path
            ),
            "artifact_sha256": artifact_hashes,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_paths.append(Path("metadata.json"))

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in artifact_paths:
            destination = output_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / path, destination)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build veRL GRPO CoT datasets."
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/grpo_experiments"),
    )
    parser.add_argument(
        "--acceptable-target-manifest",
        type=Path,
        default=Path(
            "data/training/grpo_acceptable_target_sets/"
            "acceptable_target_sets.parquet"
        ),
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
    output = build_grpo_datasets(
        args.sft_dir,
        args.output_dir,
        acceptable_target_manifest=args.acceptable_target_manifest,
        sizes=tuple(args.sizes),
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
