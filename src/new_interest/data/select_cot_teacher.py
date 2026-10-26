from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_interest.data.select_cot_preview import (
    RULE_VERSION as PREVIEW_RULE_VERSION,
    SEED,
    SELECTION_SCHEMA,
    SelectionResult,
    _atomic_write,
    _load_candidates,
    _sha256,
)
from new_interest.infrastructure.paths import portable_path

SELECTION_SIZE = 1_000
RULE_VERSION = "cot-teacher-gpt56-sol-v2-full-selection-v1"


def select_cot_teacher(
    source_dir: Path | str = Path("data/training/sft_experiments"),
    output_path: Path | str = Path(
        "artifacts/cot_teacher/gpt56_sol_v2/selection.parquet"
    ),
    *,
    metadata_path: Path | str | None = None,
    selection_size: int = SELECTION_SIZE,
    seed: int = SEED,
    overwrite: bool = False,
    root: Path | str | None = None,
) -> SelectionResult:
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else output_path.with_name("selection_metadata.json")
    )
    repository_root = Path(root) if root is not None else Path.cwd()
    if selection_size <= 0:
        raise ValueError("selection_size must be positive")
    existing = [
        path for path in (output_path, metadata_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Teacher selection output exists. Use --overwrite to replace it."
        )

    candidates, source_paths = _load_candidates(source_dir, seed=seed)
    if len(candidates) < selection_size:
        raise ValueError(
            f"Need {selection_size} paired rows, found {len(candidates)}"
        )
    selected = [
        {"selection_rank": rank, **row}
        for rank, row in enumerate(candidates[:selection_size])
    ]
    metadata: dict[str, object] = {
        "rule_version": RULE_VERSION,
        "preview_rule_version": PREVIEW_RULE_VERSION,
        "seed": seed,
        "selection_size": selection_size,
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
                "Behavior-only and Interest-only have identical source "
                "IDs, targets, impression IDs, history lengths, and order"
            ),
            "selection": "source-order prefix",
            "teacher_calls_per_source": 2,
        },
        "available_pairs": len(candidates),
        "selected_distinct_targets": len(
            {str(row["target_subcategory"]) for row in selected}
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
        description="Build the source-order paired CoT teacher selection."
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
            "artifacts/cot_teacher/gpt56_sol_v2/selection.parquet"
        ),
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--selection-size",
        type=int,
        default=SELECTION_SIZE,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = select_cot_teacher(
        args.source_dir,
        args.output,
        metadata_path=args.metadata,
        selection_size=args.selection_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "metadata": str(result.metadata_path),
                "selected": len(result.selected_source_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
