from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-candidate-pilot-v1"
DEFAULT_SOURCE_DIR = Path(
    "data/training/candidate_selection_static_user_disjoint_k10_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/candidate_selection_static_user_disjoint_k10_v1_pilot1000"
)
DEFAULT_SIZE = 1_000
DEFAULT_SEED = 20260815
STRATEGIES = ("uniform", "train_frequency_matched")
KNOWN_TASK_TYPES = ("behavior_only", "interest_only", "behavior_interest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(seed: int, selection_id: str) -> tuple[str, str]:
    material = f"{seed}:shared-category-covered-pilot:{selection_id}"
    return hashlib.sha256(material.encode()).hexdigest(), selection_id


def _selected_ids(
    rows: list[dict[str, object]], *, size: int, seed: int
) -> set[str]:
    if size <= 0 or size > len(rows):
        raise ValueError("Pilot size must be within the source dataset")
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_category.setdefault(str(row["answer_category"]), []).append(row)
    if len(by_category) > size:
        raise ValueError("Pilot size is smaller than the category count")
    selected = {
        str(min(values, key=lambda row: _priority(seed, str(row["selection_id"])))["selection_id"])
        for values in by_category.values()
    }
    remaining = sorted(
        (
            row
            for row in rows
            if str(row["selection_id"]) not in selected
        ),
        key=lambda row: _priority(seed, str(row["selection_id"])),
    )
    selected.update(
        str(row["selection_id"])
        for row in remaining[: size - len(selected)]
    )
    return selected


def _publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    backup = staging.parent / "previous-output"
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def build_static_candidate_pilot(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Path:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    task_types = tuple(
        task_type
        for task_type in KNOWN_TASK_TYPES
        if all(
            (
                source_dir
                / strategy
                / task_type
                / "dev_expanded.parquet"
            ).is_file()
            for strategy in STRATEGIES
        )
    )
    if not task_types:
        raise FileNotFoundError(
            f"No complete candidate task tracks found under {source_dir}"
        )
    source_paths = {
        (strategy, task_type): (
            source_dir / strategy / task_type / "dev_expanded.parquet"
        )
        for strategy in STRATEGIES
        for task_type in task_types
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_task_type = (
        "behavior_interest"
        if "behavior_interest" in task_types
        else task_types[0]
    )
    reference = pq.read_table(
        source_paths[("uniform", reference_task_type)]
    ).to_pylist()
    selected = _selected_ids(reference, size=size, seed=seed)
    reference_categories = {
        str(row["answer_category"])
        for row in reference
        if str(row["selection_id"]) in selected
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-candidate-pilot-", dir=str(output_dir.parent)
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        artifact_paths: list[Path] = []
        identity: list[str] | None = None
        for (strategy, task_type), source_path in source_paths.items():
            table = pq.read_table(source_path)
            mask = [
                str(value) in selected
                for value in table["selection_id"].to_pylist()
            ]
            filtered = table.filter(mask)
            ids = [str(value) for value in filtered["selection_id"].to_pylist()]
            if len(ids) != size or set(ids) != selected:
                raise ValueError(f"{source_path}: pilot identity mismatch")
            if identity is None:
                identity = ids
            elif ids != identity:
                raise ValueError("Pilot source row order differs across tracks")
            relative = Path(strategy) / task_type / "dev_pilot_1000.parquet"
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                filtered,
                path,
                compression="zstd",
                use_dictionary=True,
                row_group_size=1_000,
            )
            artifact_paths.append(relative)

        category_counts = Counter(
            str(row["answer_category"])
            for row in reference
            if str(row["selection_id"]) in selected
        )
        statistics = {
            "rule_version": RULE_VERSION,
            "samples": size,
            "supported_categories": len(reference_categories),
            "category_counts": dict(
                sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_paths.append(Path("statistics.json"))
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_dir": portable_path(source_dir),
            "output_dir": portable_path(output_dir),
            "size": size,
            "seed": seed,
            "task_types": list(task_types),
            "selection_policy": (
                "one SHA-256-priority row per positive category, then global "
                "SHA-256-priority fill"
            ),
            "source_sha256": {
                portable_path(path): _sha256(path)
                for path in source_paths.values()
            },
            "artifact_sha256": {
                str(path): _sha256(staging / path) for path in artifact_paths
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one shared category-covered candidate pilot."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        build_static_candidate_pilot(
            args.source_dir,
            args.output_dir,
            size=args.size,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
