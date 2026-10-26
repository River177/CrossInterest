from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


SEED = 20260810
SELECTION_SIZE = 20
RULE_VERSION = "cot-teacher-gpt56-sol-v2-preview-selection-v1"
TASK_TYPES = ("behavior_only", "interest_only")
QUANTILE_BINS = 4

SELECTION_SCHEMA = pa.schema(
    [
        pa.field("selection_rank", pa.int16(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("behavior_sample_id", pa.string(), nullable=False),
        pa.field("interest_sample_id", pa.string(), nullable=False),
        pa.field("behavior_original_prompt", pa.string(), nullable=False),
        pa.field("interest_original_prompt", pa.string(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("history_unique_subcategories", pa.int32(), nullable=False),
        pa.field("history_repeated_items", pa.int32(), nullable=False),
        pa.field("history_unique_ratio", pa.float64(), nullable=False),
        pa.field("history_max_frequency", pa.int32(), nullable=False),
        pa.field("history_length_bin", pa.string(), nullable=False),
        pa.field("behavior_prompt_characters", pa.int32(), nullable=False),
        pa.field("behavior_prompt_character_bin", pa.string(), nullable=False),
        pa.field("interest_prompt_characters", pa.int32(), nullable=False),
        pa.field("interest_prompt_character_bin", pa.string(), nullable=False),
        pa.field("interest_history_stratum", pa.string(), nullable=False),
        pa.field("strata", pa.list_(pa.string()), nullable=False),
        pa.field("sha256_priority", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class SelectionResult:
    output_path: Path
    metadata_path: Path
    selected_source_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(source_sample_id: str, seed: int) -> str:
    material = f"{seed}:cot-v2-preview:{source_sample_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _extract_interest_history(prompt: str) -> list[str]:
    try:
        history_block = prompt.split("\n\nBased on ", 1)[0]
        numbered_history = history_block.split("\n\n", 1)[1]
    except (IndexError, ValueError) as error:
        raise ValueError("Cannot parse Interest-only history") from error

    history: list[str] = []
    for line in numbered_history.splitlines():
        _, separator, value = line.partition(". ")
        if not separator or not value.strip():
            raise ValueError(f"Invalid numbered history line: {line!r}")
        history.append(value.strip())
    if not history:
        raise ValueError("Interest-only history is empty")
    return history


def _quantile_labels(
    values: list[int],
    source_ids: list[str],
    *,
    bin_count: int = QUANTILE_BINS,
) -> list[str]:
    if len(values) != len(source_ids) or not values:
        raise ValueError("Quantile inputs must be nonempty and aligned")
    order = sorted(
        range(len(values)),
        key=lambda index: (values[index], source_ids[index]),
    )
    labels = [""] * len(values)
    for rank, index in enumerate(order):
        bin_index = min(bin_count - 1, rank * bin_count // len(order))
        labels[index] = f"q{bin_index + 1}"
    return labels


def _interest_history_stratum(
    history_length: int,
    unique_count: int,
) -> str:
    if unique_count == history_length:
        return "all_unique"
    unique_ratio = unique_count / history_length
    if unique_ratio >= 0.75:
        return "light_repetition"
    if unique_ratio >= 0.5:
        return "moderate_repetition"
    return "concentrated_repetition"


def _load_candidates(
    source_dir: Path,
    *,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    source_paths = {
        task_type: source_dir / "train" / task_type / "1k.parquet"
        for task_type in TASK_TYPES
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    columns = [
        "sample_id",
        "source_sample_id",
        "impression_id",
        "task_type",
        "history_length",
        "prompt",
        "target_subcategory",
    ]
    rows_by_task = {
        task_type: pq.read_table(path, columns=columns).to_pylist()
        for task_type, path in source_paths.items()
    }
    behavior_rows = rows_by_task["behavior_only"]
    interest_rows = rows_by_task["interest_only"]
    if len(behavior_rows) != len(interest_rows):
        raise ValueError("Behavior and Interest row counts do not match")
    if len(behavior_rows) < SELECTION_SIZE:
        raise ValueError(
            f"Need at least {SELECTION_SIZE} paired rows, "
            f"found {len(behavior_rows)}"
        )

    behavior_ids = [str(row["source_sample_id"]) for row in behavior_rows]
    interest_ids = [str(row["source_sample_id"]) for row in interest_rows]
    if behavior_ids != interest_ids:
        raise ValueError(
            "Behavior and Interest source IDs/order do not match"
        )
    if len(set(behavior_ids)) != len(behavior_ids):
        raise ValueError("Source files contain duplicate source IDs")

    histories: list[list[str]] = []
    for behavior, interest in zip(behavior_rows, interest_rows):
        source_id = str(behavior["source_sample_id"])
        if behavior["task_type"] != "behavior_only":
            raise ValueError(f"{source_id}: Behavior task_type mismatch")
        if interest["task_type"] != "interest_only":
            raise ValueError(f"{source_id}: Interest task_type mismatch")
        for field in ("impression_id", "target_subcategory"):
            if str(behavior[field]) != str(interest[field]):
                raise ValueError(f"{source_id}: paired {field} mismatch")
        if int(behavior["history_length"]) != int(
            interest["history_length"]
        ):
            raise ValueError(f"{source_id}: paired history lengths differ")
        history = _extract_interest_history(str(interest["prompt"]))
        if len(history) != int(interest["history_length"]):
            raise ValueError(
                f"{source_id}: extracted history length does not match"
            )
        histories.append(history)

    history_lengths = [len(history) for history in histories]
    behavior_prompt_lengths = [
        len(str(row["prompt"])) for row in behavior_rows
    ]
    interest_prompt_lengths = [
        len(str(row["prompt"])) for row in interest_rows
    ]
    history_bins = _quantile_labels(history_lengths, behavior_ids)
    behavior_prompt_bins = _quantile_labels(
        behavior_prompt_lengths,
        behavior_ids,
    )
    interest_prompt_bins = _quantile_labels(
        interest_prompt_lengths,
        behavior_ids,
    )

    candidates: list[dict[str, object]] = []
    for index, (behavior, interest, history) in enumerate(
        zip(behavior_rows, interest_rows, histories)
    ):
        frequencies = Counter(history)
        unique_count = len(frequencies)
        history_length = len(history)
        history_stratum = _interest_history_stratum(
            history_length,
            unique_count,
        )
        strata = [
            f"history_length:{history_bins[index]}",
            f"behavior_prompt_characters:{behavior_prompt_bins[index]}",
            f"interest_prompt_characters:{interest_prompt_bins[index]}",
            f"interest_history:{history_stratum}",
        ]
        candidates.append(
            {
                "source_sample_id": behavior_ids[index],
                "impression_id": str(behavior["impression_id"]),
                "target_subcategory": str(
                    behavior["target_subcategory"]
                ),
                "behavior_sample_id": str(behavior["sample_id"]),
                "interest_sample_id": str(interest["sample_id"]),
                "behavior_original_prompt": str(behavior["prompt"]),
                "interest_original_prompt": str(interest["prompt"]),
                "history_length": history_length,
                "history_subcategories": history,
                "history_unique_subcategories": unique_count,
                "history_repeated_items": history_length - unique_count,
                "history_unique_ratio": unique_count / history_length,
                "history_max_frequency": max(frequencies.values()),
                "history_length_bin": history_bins[index],
                "behavior_prompt_characters": behavior_prompt_lengths[index],
                "behavior_prompt_character_bin": (
                    behavior_prompt_bins[index]
                ),
                "interest_prompt_characters": interest_prompt_lengths[index],
                "interest_prompt_character_bin": (
                    interest_prompt_bins[index]
                ),
                "interest_history_stratum": history_stratum,
                "strata": strata,
                "sha256_priority": _priority(behavior_ids[index], seed),
            }
        )
    return candidates, source_paths


def _select_candidates(
    candidates: list[dict[str, object]],
    *,
    selection_size: int,
) -> list[dict[str, object]]:
    target_goal = min(
        selection_size,
        len({str(row["target_subcategory"]) for row in candidates}),
    )
    stratum_counts = Counter(
        str(stratum)
        for row in candidates
        for stratum in row["strata"]
    )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_targets: set[str] = set()
    covered_strata: set[str] = set()

    while len(selected) < selection_size:
        require_new_target = len(selected_targets) < target_goal
        eligible = [
            row
            for row in candidates
            if str(row["source_sample_id"]) not in selected_ids
            and (
                not require_new_target
                or str(row["target_subcategory"])
                not in selected_targets
            )
        ]
        if not eligible:
            raise ValueError(
                f"Cannot select {selection_size} rows under pairing policy"
            )

        def selection_key(
            row: dict[str, object],
        ) -> tuple[float, int, str, str]:
            new_strata = {
                str(value) for value in row["strata"]
            } - covered_strata
            rarity = sum(
                1.0 / stratum_counts[stratum]
                for stratum in new_strata
            )
            return (
                -rarity,
                -len(new_strata),
                str(row["sha256_priority"]),
                str(row["source_sample_id"]),
            )

        chosen = min(eligible, key=selection_key)
        selected.append(chosen)
        selected_ids.add(str(chosen["source_sample_id"]))
        selected_targets.add(str(chosen["target_subcategory"]))
        covered_strata.update(str(value) for value in chosen["strata"])

    return [
        {"selection_rank": rank, **row}
        for rank, row in enumerate(selected)
    ]


def _atomic_write(
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_path: Path,
    metadata_path: Path,
    *,
    root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    pending_output = output_path.parent / f".{output_path.name}.{token}.pending"
    pending_metadata = (
        metadata_path.parent / f".{metadata_path.name}.{token}.pending"
    )
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SELECTION_SCHEMA),
            pending_output,
            compression="zstd",
            use_dictionary=True,
        )
        metadata["artifact_sha256"] = {
            portable_path(output_path, root=root): _sha256(pending_output)
        }
        pending_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(pending_output, output_path)
        os.replace(pending_metadata, metadata_path)
    finally:
        pending_output.unlink(missing_ok=True)
        pending_metadata.unlink(missing_ok=True)


def select_cot_preview(
    source_dir: Path | str = Path("data/training/sft_experiments"),
    output_path: Path | str = Path(
        "artifacts/cot_teacher/gpt56_sol_v2_preview/selection.parquet"
    ),
    *,
    metadata_path: Path | str | None = None,
    seed: int = SEED,
    overwrite: bool = False,
    root: Path | str | None = None,
) -> SelectionResult:
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else output_path.with_name("metadata.json")
    )
    repository_root = Path(root) if root is not None else Path.cwd()
    existing = [
        path for path in (output_path, metadata_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Preview output already exists. Use --overwrite to replace it."
        )

    candidates, source_paths = _load_candidates(source_dir, seed=seed)
    selected = _select_candidates(
        candidates,
        selection_size=SELECTION_SIZE,
    )
    target_count = len(
        {str(row["target_subcategory"]) for row in selected}
    )
    available_target_count = len(
        {str(row["target_subcategory"]) for row in candidates}
    )
    metadata: dict[str, object] = {
        "rule_version": RULE_VERSION,
        "seed": seed,
        "selection_size": SELECTION_SIZE,
        "source_dir": portable_path(source_dir, root=repository_root),
        "output_path": portable_path(output_path, root=repository_root),
        "source_sha256": {
            task_type: {
                "path": portable_path(path, root=repository_root),
                "sha256": _sha256(path),
            }
            for task_type, path in source_paths.items()
        },
        "policy": {
            "source_subset": "balanced SFT train 1k",
            "pairing": (
                "Behavior-only and Interest-only must have identical "
                "source IDs, targets, impression IDs, and row order"
            ),
            "selection": (
                "greedy deterministic stratum coverage with SHA-256 "
                "priority tie-breaking"
            ),
            "target_objective": (
                "require a new target at every rank until min(20, "
                "available distinct targets) is reached"
            ),
            "quantile_bins": {
                "count": QUANTILE_BINS,
                "assignment": (
                    "stable rank bins ordered by value then source ID"
                ),
                "features": [
                    "history_length",
                    "behavior_prompt_characters",
                    "interest_prompt_characters",
                ],
            },
            "interest_history_strata": {
                "all_unique": "unique_ratio == 1",
                "light_repetition": "0.75 <= unique_ratio < 1",
                "moderate_repetition": "0.5 <= unique_ratio < 0.75",
                "concentrated_repetition": "unique_ratio < 0.5",
            },
            "priority": {
                "algorithm": "sha256",
                "material": (
                    "{seed}:cot-v2-preview:{source_sample_id}"
                ),
                "lower_hex_digest_wins_ties": True,
            },
        },
        "available_pairs": len(candidates),
        "available_distinct_targets": available_target_count,
        "selected_distinct_targets": target_count,
        "selected_strata": sorted(
            {
                str(stratum)
                for row in selected
                for stratum in row["strata"]
            }
        ),
        "schema": str(SELECTION_SCHEMA),
    }
    _atomic_write(
        selected,
        metadata,
        output_path,
        metadata_path,
        root=repository_root,
    )
    return SelectionResult(
        output_path=output_path,
        metadata_path=metadata_path,
        selected_source_ids=tuple(
            str(row["source_sample_id"]) for row in selected
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the deterministic paired 20-example CoT teacher preview."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/cot_teacher/gpt56_sol_v2_preview/"
            "selection.parquet"
        ),
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = select_cot_preview(
        args.source_dir,
        args.output,
        metadata_path=args.metadata,
        overwrite=args.overwrite,
    )
    print(result.output_path)
    print(result.metadata_path)


if __name__ == "__main__":
    main()
