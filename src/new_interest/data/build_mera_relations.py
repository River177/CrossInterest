from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path
from new_interest.training.semantic_id_tokens import (
    load_interest_token_mapping,
)


RULE_VERSION = "mera-gnip-relations-v1"
DEFAULT_SOURCE_TRAIN_FILE = Path(
    "data/processed/"
    "new_interest_static_user_disjoint_sft21600_h50_v1/train.parquet"
)
DEFAULT_MAPPING_FILE = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1/"
    "interest_token_mapping.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/mera_static_user_disjoint_behavior_interest_200_h50_v1"
)
COVERAGE_THRESHOLDS = (0.0, 0.01, 0.02, 0.05)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    rows: int
    interests: int
    parent_groups: int


def _sha256_file(path: Path) -> str:
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


def _coverage_at_threshold(
    counts: list[list[int]],
    threshold: float,
) -> dict[str, object]:
    positive_counts: list[int] = []
    for anchor, row in enumerate(counts):
        denominator = sum(row)
        positives = sum(
            value > 0
            and candidate != anchor
            and value / denominator >= threshold
            for candidate, value in enumerate(row)
        )
        positive_counts.append(positives)
    sorted_counts = sorted(positive_counts)
    return {
        "threshold": threshold,
        "eligible_anchors": sum(value > 0 for value in positive_counts),
        "eligible_anchor_fraction": (
            sum(value > 0 for value in positive_counts)
            / len(positive_counts)
        ),
        "positive_edges": sum(positive_counts),
        "min_positives_per_anchor": min(positive_counts),
        "median_positives_per_anchor": sorted_counts[
            len(sorted_counts) // 2
        ],
        "max_positives_per_anchor": max(positive_counts),
    }


def build_mera_relations(
    source_train_file: Path | str = DEFAULT_SOURCE_TRAIN_FILE,
    mapping_file: Path | str = DEFAULT_MAPPING_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = 2_000,
    overwrite: bool = False,
) -> BuildResult:
    """Build MeRA structure and train-only collaborative relations.

    The co-occurrence contract follows the released MeRA notebook: each
    training row contributes the set of output-vocabulary interests appearing
    in History plus its supervised target, and every ordered pair in that set
    is counted once. Repeated History interests therefore do not add weight.
    """

    source_train_file = Path(source_train_file)
    mapping_file = Path(mapping_file)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for path in (source_train_file, mapping_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite."
        )

    mapping_rows = load_interest_token_mapping(mapping_file)
    labels = [str(row["target_subcategory"]) for row in mapping_rows]
    interest_tokens = [str(row["interest_token"]) for row in mapping_rows]
    target_labels = {
        str(row["target_subcategory"])
        for row in mapping_rows
        if bool(row.get("is_target", True))
    }
    index_by_label = {label: index for index, label in enumerate(labels)}
    interest_count = len(labels)
    counts = [[0] * interest_count for _ in range(interest_count)]
    parent_values: dict[str, set[str]] = defaultdict(set)
    target_counts: Counter[str] = Counter()
    history_output_interest_occurrences = 0
    history_only_interest_occurrences = 0
    users: set[str] = set()
    sample_ids: set[str] = set()
    rows = 0

    parquet = pq.ParquetFile(source_train_file)
    required = {"sample_id", "user_id", "history", "target_interest"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{source_train_file}: missing columns {sorted(missing)}"
        )
    for batch in parquet.iter_batches(
        columns=sorted(required),
        batch_size=batch_size,
    ):
        for row in batch.to_pylist():
            rows += 1
            sample_id = str(row["sample_id"])
            user_id = str(row["user_id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            users.add(user_id)

            target = row["target_interest"]
            target_label = str(target["subcategory"])
            if target_label not in index_by_label:
                raise ValueError(
                    f"{sample_id}: target not in Semantic-ID mapping: "
                    f"{target_label}"
                )
            parent_values[target_label].add(str(target["category"]))
            target_counts[target_label] += 1

            row_labels = {target_label}
            for event in row["history"]:
                label = str(event["subcategory"])
                if label in index_by_label:
                    history_output_interest_occurrences += 1
                    parent_values[label].add(str(event["category"]))
                    row_labels.add(label)
                else:
                    history_only_interest_occurrences += 1
            row_indices = sorted(index_by_label[label] for label in row_labels)
            for anchor in row_indices:
                for candidate in row_indices:
                    counts[anchor][candidate] += 1

    if not rows:
        raise ValueError("Source train file is empty")
    if set(target_counts) != target_labels:
        missing_targets = sorted(target_labels.difference(target_counts))
        raise ValueError(
            "Every target-enabled Semantic-ID interest must appear as a "
            "target; missing "
            f"{missing_targets}"
        )
    missing_parents = sorted(set(labels).difference(parent_values))
    if missing_parents:
        raise ValueError(
            "Every Semantic-ID interest must appear in Train History or as "
            f"a target; missing parents for {missing_parents}"
        )
    ambiguous_parents = {
        label: sorted(values)
        for label, values in parent_values.items()
        if len(values) != 1
    }
    if ambiguous_parents:
        raise ValueError(
            "Target subcategories do not have a unique parent: "
            f"{ambiguous_parents}"
        )
    parents = [next(iter(parent_values[label])) for label in labels]
    parent_counts = Counter(parents)
    structure_eligible = sum(parent_counts[parent] > 1 for parent in parents)

    relation_payload = {
        "format_version": RULE_VERSION,
        "labels": labels,
        "interest_tokens": interest_tokens,
        "parents": parents,
        "cooccurrence_counts": counts,
        "cooccurrence_contract": {
            "source_split": "train-only",
            "row_categories": (
                "set(history subcategories intersect output vocabulary) "
                "union supervised target"
            ),
            "repeated_history_categories": "deduplicated within row",
            "pair_direction": "ordered",
            "diagonal": "included in stored counts; excluded when sampling",
            "probability": "row-normalized stored counts",
            "threshold": "applied at training time",
        },
    }

    coverage = [
        _coverage_at_threshold(counts, threshold)
        for threshold in COVERAGE_THRESHOLDS
    ]
    statistics = {
        "rule_version": RULE_VERSION,
        "rows": rows,
        "users": len(users),
        "unique_sample_ids": len(sample_ids),
        "interests": interest_count,
        "target_interests": len(target_labels),
        "history_only_interests": interest_count - len(target_labels),
        "target_counts": dict(sorted(target_counts.items())),
        "parent_groups": len(parent_counts),
        "parent_group_sizes": dict(sorted(parent_counts.items())),
        "structure_eligible_anchors": structure_eligible,
        "structure_eligible_anchor_fraction": (
            structure_eligible / interest_count
        ),
        "history_output_interest_occurrences": (
            history_output_interest_occurrences
        ),
        "history_only_interest_occurrences_ignored": (
            history_only_interest_occurrences
        ),
        "cooccurrence_nonzero_cells": sum(
            value > 0 for row in counts for value in row
        ),
        "cooccurrence_threshold_coverage": coverage,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        relation_path = staging / "relations.json"
        _write_json(relation_path, relation_payload)
        _write_json(staging / "statistics.json", statistics)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "source_train_file": portable_path(source_train_file),
            "mapping_file": portable_path(mapping_file),
            "output_dir": portable_path(output_dir),
            "source_sha256": {
                portable_path(source_train_file): _sha256_file(
                    source_train_file
                ),
                portable_path(mapping_file): _sha256_file(mapping_file),
            },
            "artifact_sha256": {
                "relations.json": _sha256_file(relation_path),
                "statistics.json": _sha256_file(staging / "statistics.json"),
            },
        }
        _write_json(staging / "metadata.json", metadata)
        _commit_staging(staging, output_dir, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return BuildResult(
        output_dir=output_dir,
        rows=rows,
        interests=interest_count,
        parent_groups=len(parent_counts),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only structural and collaborative relations for "
            "the MeRA-GNIP adaptation."
        )
    )
    parser.add_argument(
        "--source-train-file",
        type=Path,
        default=DEFAULT_SOURCE_TRAIN_FILE,
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_mera_relations(
        args.source_train_file,
        args.mapping_file,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "rows": result.rows,
                "interests": result.interests,
                "parent_groups": result.parent_groups,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
