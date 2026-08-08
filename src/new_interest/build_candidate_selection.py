from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

RULE_VERSION = "candidate-selection-k10-v1"
DEFAULT_K = 10
DEFAULT_SEED = 20260806

CANDIDATE_DATA_SCHEMA = pa.schema(
    [
        pa.field("selection_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field(
            "candidates",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("answer_category", pa.string(), nullable=False),
        pa.field("answer", pa.string(), nullable=False),
        pa.field("correct_position", pa.int8(), nullable=False),
        pa.field(
            "all_acceptable_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("selection_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("positive", pa.string(), nullable=False),
        pa.field(
            "negative_candidates",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "candidates",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("correct_position", pa.int8(), nullable=False),
        pa.field(
            "all_acceptable_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
    ]
)

OUTPUT_FILES = (
    Path("behavior_only/dev_10k_expanded.parquet"),
    Path("interest_only/dev_10k_expanded.parquet"),
    Path("candidate_manifest.parquet"),
    Path("statistics.json"),
    Path("metadata.json"),
)

BEHAVIOR_INSTRUCTION = """Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should represent a new topic that is not reflected in my viewing history."""

INTEREST_INSTRUCTION = """Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should not appear in the history sequence above."""

ANSWER_INSTRUCTION = (
    'Respond with exactly "New topic: <category>", replacing <category> '
    "with one exact category name from the candidate list. Do not include "
    "any other text."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(
    seed: int,
    namespace: str,
    *values: str,
) -> str:
    material = ":".join((str(seed), namespace, *values))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _distribution_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = min(
            len(ordered) - 1,
            max(0, int(len(ordered) * fraction + 0.999999) - 1),
        )
        return ordered[index]

    return {
        "count": len(values),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": sum(values) / len(values),
    }


def _history_block(prompt: str) -> str:
    marker = "\n\nBased on "
    if marker not in prompt:
        raise ValueError("Source prompt does not contain a Based on marker")
    block = prompt.split(marker, 1)[0]
    if not block.strip():
        raise ValueError("Source prompt has an empty history block")
    return block


def _render_prompt(
    history_block: str,
    candidates: list[str],
    *,
    task_type: str,
) -> str:
    instruction = (
        BEHAVIOR_INSTRUCTION
        if task_type == "behavior_only"
        else INTEREST_INSTRUCTION
    )
    candidate_block = "\n".join(
        f"{index}. {candidate}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return (
        f"{history_block}\n\n{instruction}\n\n"
        f"Candidates:\n{candidate_block}\n\n{ANSWER_INSTRUCTION}"
    )


def _load_paired_eval_rows(
    sft_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    behavior = pq.read_table(
        sft_dir / "eval/behavior_only/dev_10k_grouped.parquet"
    ).to_pylist()
    interest = pq.read_table(
        sft_dir / "eval/interest_only/dev_10k_grouped.parquet"
    ).to_pylist()
    if len(behavior) != len(interest):
        raise ValueError("Behavior and Interest Dev row counts differ")
    for behavior_row, interest_row in zip(behavior, interest):
        for field in (
            "impression_id",
            "user_id",
            "timestamp",
            "history_length",
            "acceptable_subcategories",
            "history_subcategories",
        ):
            if behavior_row[field] != interest_row[field]:
                raise ValueError(
                    f"{behavior_row['impression_id']}: paired field "
                    f"{field!r} differs"
                )
    return behavior, interest


def _build_manifest_rows(
    interest_rows: list[dict[str, object]],
    labels: list[str],
    *,
    k: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[int]]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    draft_rows: list[dict[str, object]] = []
    negative_pool_sizes: list[int] = []

    for row in interest_rows:
        impression_id = str(row["impression_id"])
        acceptable = [
            str(value) for value in row["acceptable_subcategories"]
        ]
        history = [str(value) for value in row["history_subcategories"]]
        excluded = set(acceptable).union(history)
        negative_pool = [label for label in labels if label not in excluded]
        negative_pool_sizes.append(len(negative_pool))
        if len(negative_pool) < k - 1:
            raise ValueError(
                f"{impression_id}: only {len(negative_pool)} legal "
                f"negative candidates for K={k}"
            )

        for positive in acceptable:
            selection_id = (
                f"dev:{impression_id}:{label_to_id[positive]}"
            )
            negatives = sorted(
                negative_pool,
                key=lambda label: (
                    _priority(
                        seed,
                        "candidate-negative",
                        selection_id,
                        label,
                    ),
                    label,
                ),
            )[: k - 1]
            negatives.sort(
                key=lambda label: (
                    _priority(
                        seed,
                        "candidate-order",
                        selection_id,
                        label,
                    ),
                    label,
                )
            )
            draft_rows.append(
                {
                    "selection_id": selection_id,
                    "impression_id": impression_id,
                    "positive": positive,
                    "negative_candidates": negatives,
                    "all_acceptable_subcategories": acceptable,
                    "history_subcategories": history,
                }
            )

    position_order = sorted(
        range(len(draft_rows)),
        key=lambda index: (
            _priority(
                seed,
                "candidate-position",
                str(draft_rows[index]["selection_id"]),
            ),
            str(draft_rows[index]["selection_id"]),
        ),
    )
    correct_positions = [0] * len(draft_rows)
    for rank, row_index in enumerate(position_order):
        correct_positions[row_index] = rank % k

    manifest_rows: list[dict[str, object]] = []
    for row, correct_position in zip(draft_rows, correct_positions):
        negatives = list(row["negative_candidates"])
        candidates = negatives.copy()
        candidates.insert(correct_position, str(row["positive"]))
        manifest_rows.append(
            {
                **row,
                "candidates": candidates,
                "correct_position": correct_position,
            }
        )
    return manifest_rows, negative_pool_sizes


def _write_task_data(
    path: Path,
    source_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    *,
    task_type: str,
) -> None:
    source_by_impression = {
        str(row["impression_id"]): row for row in source_rows
    }
    output_rows: list[dict[str, object]] = []
    for manifest in manifest_rows:
        source = source_by_impression[str(manifest["impression_id"])]
        output_rows.append(
            {
                "selection_id": manifest["selection_id"],
                "impression_id": manifest["impression_id"],
                "user_id": source["user_id"],
                "timestamp": source["timestamp"],
                "task_type": task_type,
                "history_length": source["history_length"],
                "prompt": _render_prompt(
                    _history_block(str(source["prompt"])),
                    list(manifest["candidates"]),
                    task_type=task_type,
                ),
                "candidates": manifest["candidates"],
                "answer_category": manifest["positive"],
                "answer": f"New topic: {manifest['positive']}",
                "correct_position": manifest["correct_position"],
                "all_acceptable_subcategories": (
                    manifest["all_acceptable_subcategories"]
                ),
                "history_subcategories": manifest["history_subcategories"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_rows, schema=CANDIDATE_DATA_SCHEMA),
        path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / relative
        for relative in OUTPUT_FILES
        if (output_dir / relative).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}. Use --overwrite."
        )


def build_candidate_selection(
    sft_dir: Path | str,
    retained_interests_path: Path | str,
    output_dir: Path | str,
    *,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Path:
    sft_dir = Path(sft_dir)
    retained_interests_path = Path(retained_interests_path)
    output_dir = Path(output_dir)
    if k <= 1:
        raise ValueError("k must be greater than 1")
    required = [
        sft_dir / "eval/behavior_only/dev_10k_grouped.parquet",
        sft_dir / "eval/interest_only/dev_10k_grouped.parquet",
        retained_interests_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    _ensure_output_available(output_dir, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    behavior_rows, interest_rows = _load_paired_eval_rows(sft_dir)
    labels = sorted(
        {
            str(row["target_subcategory"])
            for row in pq.read_table(
                retained_interests_path,
                columns=["target_subcategory"],
            ).to_pylist()
        }
    )
    manifest_rows, negative_pool_sizes = _build_manifest_rows(
        interest_rows,
        labels,
        k=k,
        seed=seed,
    )

    with tempfile.TemporaryDirectory(
        prefix=".candidate-selection-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        _write_task_data(
            staging / "behavior_only/dev_10k_expanded.parquet",
            behavior_rows,
            manifest_rows,
            task_type="behavior_only",
        )
        _write_task_data(
            staging / "interest_only/dev_10k_expanded.parquet",
            interest_rows,
            manifest_rows,
            task_type="interest_only",
        )
        pq.write_table(
            pa.Table.from_pylist(
                manifest_rows,
                schema=MANIFEST_SCHEMA,
            ),
            staging / "candidate_manifest.parquet",
            compression="zstd",
            use_dictionary=True,
            row_group_size=2_000,
        )

        position_counts = Counter(
            int(row["correct_position"]) for row in manifest_rows
        )
        positive_counts = Counter(
            str(row["positive"]) for row in manifest_rows
        )
        negative_counts = Counter(
            negative
            for row in manifest_rows
            for negative in row["negative_candidates"]
        )
        history_lengths = [
            int(row["history_length"]) for row in interest_rows
        ]
        statistics = {
            "rule_version": RULE_VERSION,
            "k": k,
            "grouped_impressions": len(interest_rows),
            "expanded_samples": len(manifest_rows),
            "position_counts": {
                str(position): position_counts[position]
                for position in range(k)
            },
            "positive_counts": dict(
                sorted(
                    positive_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "negative_counts": dict(
                sorted(
                    negative_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "negative_pool_size": _distribution_summary(
                negative_pool_sizes
            ),
            "history_length": _distribution_summary(history_lengths),
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_paths = [
            relative
            for relative in OUTPUT_FILES
            if relative.name != "metadata.json"
        ]
        artifact_hashes = {
            str(relative): _sha256(staging / relative)
            for relative in artifact_paths
        }
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "k": k,
            "seed": seed,
            "source_sft_dir": str(sft_dir.resolve()),
            "retained_interests_path": str(
                retained_interests_path.resolve()
            ),
            "positive_policy": "expand_each_acceptable_subcategory",
            "negative_policy": (
                "exclude_all_acceptable_and_history_subcategories"
            ),
            "position_policy": (
                "sha256_order_then_global_rank_modulo_k"
            ),
            "answer_format": "New topic: <exact_category_name>",
            "model_evaluation_performed": False,
            "prompt_templates": {
                "behavior_instruction": BEHAVIOR_INSTRUCTION,
                "interest_instruction": INTEREST_INSTRUCTION,
                "answer_instruction": ANSWER_INSTRUCTION,
            },
            "source_sha256": {
                "behavior_dev_10k_grouped.parquet": _sha256(required[0]),
                "interest_dev_10k_grouped.parquet": _sha256(required[1]),
                "retained_interests.parquet": _sha256(required[2]),
            },
            "artifact_sha256": artifact_hashes,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for relative in OUTPUT_FILES:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)

    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired K=10 candidate-selection Dev data."
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--retained-interests",
        type=Path,
        default=Path(
            "data/training/"
            "generative_new_interest_balanced_200_h50/"
            "retained_interests.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/candidate_selection_k10"),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir = build_candidate_selection(
        args.sft_dir,
        args.retained_interests,
        args.output_dir,
        k=args.k,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    LOGGER.info("Wrote candidate-selection data to %s", output_dir)


if __name__ == "__main__":
    main()
