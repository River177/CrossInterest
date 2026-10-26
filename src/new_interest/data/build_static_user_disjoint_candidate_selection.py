from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


LOGGER = logging.getLogger(__name__)

RULE_VERSION = "static-user-disjoint-candidate-selection-k10-v1.6"
DEFAULT_DATASET_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/candidate_selection_static_user_disjoint_k10_v1"
)
DEFAULT_K = 10
DEFAULT_SEED = 20260806
STRATEGIES = ("uniform", "train_frequency_matched")
TASK_TYPES = ("behavior_only", "interest_only", "behavior_interest")


CANDIDATE_DATA_SCHEMA = pa.schema(
    [
        pa.field("selection_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("negative_sampling", pa.string(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("candidates", pa.list_(pa.string()), nullable=False),
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
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("negative_sampling", pa.string(), nullable=False),
        pa.field("positive", pa.string(), nullable=False),
        pa.field(
            "negative_candidates", pa.list_(pa.string()), nullable=False
        ),
        pa.field("candidates", pa.list_(pa.string()), nullable=False),
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
        pa.field("negative_pool_size", pa.int16(), nullable=False),
    ]
)


BEHAVIOR_HISTORY_HEADER = (
    "The headlines of the news items I recently viewed are listed below:"
)
INTEREST_HISTORY_HEADER = (
    "The topic categories associated with the news items I recently viewed, "
    "in chronological order, are listed below:"
)
BEHAVIOR_INTEREST_HISTORY_HEADER = (
    "The headlines and corresponding topic categories of the news items I "
    "recently viewed, in chronological order, are listed below. Each item is "
    'formatted as "[Topic category] Headline":'
)
BEHAVIOR_INSTRUCTION = (
    "Select the one topic category from the candidates below that I am most "
    "likely to be interested in next. The selected category should represent "
    "a new topic that is not reflected in my viewing history."
)
INTEREST_INSTRUCTION = (
    "Select the one topic category from the candidates below that I am most "
    "likely to be interested in next. The selected category should not appear "
    "in the history sequence above."
)
BEHAVIOR_INTEREST_INSTRUCTION = (
    "Based on the headlines and corresponding topic categories above, "
    "predict a new topic category that I am likely to be interested in next. "
    "The predicted category should not appear among the topic categories in "
    "the sequence above. Select the single most likely new topic from these "
    "candidates. The selected topic must not appear in the user's history."
)
ANSWER_INSTRUCTION = (
    "Copy the complete candidate name exactly. Return only "
    '"New topic: <complete candidate name>".'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(seed: int, namespace: str, *values: str) -> str:
    material = ":".join((str(seed), namespace, *values))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _unit_interval(seed: int, namespace: str, *values: str) -> float:
    value = int(_priority(seed, namespace, *values), 16)
    return (value + 1) / (2**256 + 1)


def _distribution_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        rank = max(1, math.ceil(len(ordered) * fraction))
        return ordered[rank - 1]

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


def _read_train_counts(train_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    users: set[str] = set()
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["user_id", "target_interest"],
        batch_size=16_384,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in users:
                raise ValueError(
                    f"Static Train must contain one row per user: {user_id}"
                )
            users.add(user_id)
            target = row["target_interest"]
            if not isinstance(target, dict) or not target.get("subcategory"):
                raise ValueError(f"Malformed Train target for {user_id}")
            counts[str(target["subcategory"])] += 1
    if not counts:
        raise ValueError(f"No Train targets found in {train_path}")
    return counts


def _history_signature(history: list[dict[str, object]]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            str(item.get("news_id", "")),
            str(item.get("category", "")),
            str(item.get("subcategory", "")),
            str(item.get("title", "")),
        )
        for item in history
    )


def _load_dev_rows(
    dev_path: Path,
    train_labels: set[str],
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    rows = pq.read_table(
        dev_path,
        columns=[
            "sample_id",
            "impression_id",
            "user_id",
            "timestamp",
            "history",
            "target_interest",
        ],
    ).to_pylist()
    if not rows:
        raise ValueError(f"No Dev rows found in {dev_path}")

    sample_ids: set[str] = set()
    impression_fields: dict[
        str, tuple[str, datetime, tuple[tuple[str, ...], ...]]
    ] = {}
    user_histories: dict[str, tuple[tuple[str, ...], ...]] = {}
    acceptable: dict[str, set[str]] = defaultdict(set)
    normalized: list[dict[str, object]] = []
    for source in rows:
        sample_id = str(source["sample_id"])
        impression_id = str(source["impression_id"])
        user_id = str(source["user_id"])
        timestamp = source["timestamp"]
        history = source["history"]
        target = source["target_interest"]
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate Dev sample_id {sample_id}")
        sample_ids.add(sample_id)
        if not isinstance(timestamp, datetime):
            raise ValueError(f"{sample_id}: malformed timestamp")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{sample_id}: History must be non-empty")
        if not isinstance(target, dict) or not target.get("subcategory"):
            raise ValueError(f"{sample_id}: malformed target")
        positive = str(target["subcategory"])
        if positive not in train_labels:
            raise ValueError(
                f"{sample_id}: Dev target {positive!r} is absent from Train"
            )
        signature = _history_signature(history)
        if any(not all(item) for item in signature):
            raise ValueError(f"{sample_id}: malformed History item")
        fields = (user_id, timestamp, signature)
        previous = impression_fields.setdefault(impression_id, fields)
        if previous != fields:
            raise ValueError(
                f"{impression_id}: inconsistent user, timestamp, or History"
            )
        previous_history = user_histories.setdefault(user_id, signature)
        if previous_history != signature:
            raise ValueError(f"{user_id}: inconsistent static Dev History")
        acceptable[impression_id].add(positive)
        normalized.append(
            {
                **source,
                "sample_id": sample_id,
                "impression_id": impression_id,
                "user_id": user_id,
                "positive": positive,
            }
        )

    normalized.sort(key=lambda row: str(row["sample_id"]))
    return normalized, {
        impression_id: sorted(values)
        for impression_id, values in acceptable.items()
    }


def _correct_positions(
    rows: list[dict[str, object]], *, k: int, seed: int
) -> dict[str, int]:
    ordered = sorted(
        (str(row["selection_id"]) for row in rows),
        key=lambda selection_id: (
            _priority(seed, "candidate-position", selection_id),
            selection_id,
        ),
    )
    return {
        selection_id: rank % k
        for rank, selection_id in enumerate(ordered)
    }


def _negative_order(
    pool: list[str],
    *,
    selection_id: str,
    strategy: str,
    train_counts: Counter[str],
    seed: int,
) -> list[str]:
    if strategy == "uniform":
        return sorted(
            pool,
            key=lambda label: (
                _priority(seed, "uniform-negative", selection_id, label),
                label,
            ),
        )
    if strategy == "train_frequency_matched":
        return sorted(
            pool,
            key=lambda label: (
                -math.log(
                    _unit_interval(
                        seed,
                        "frequency-negative",
                        selection_id,
                        label,
                    )
                )
                / train_counts[label],
                label,
            ),
        )
    raise ValueError(f"Unknown negative strategy: {strategy}")


def _build_manifests(
    dev_rows: list[dict[str, object]],
    acceptable_by_impression: dict[str, list[str]],
    train_counts: Counter[str],
    *,
    k: int,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    labels = sorted(train_counts)
    drafts: list[dict[str, object]] = []
    for row in dev_rows:
        sample_id = str(row["sample_id"])
        history = [
            str(item["subcategory"])
            for item in row["history"]
        ]
        impression_id = str(row["impression_id"])
        acceptable = acceptable_by_impression[impression_id]
        excluded = set(history).union(acceptable)
        pool = [label for label in labels if label not in excluded]
        if len(pool) < k - 1:
            raise ValueError(
                f"{sample_id}: only {len(pool)} legal negatives for K={k}"
            )
        drafts.append(
            {
                "selection_id": f"static-dev:{sample_id}",
                "sample_id": sample_id,
                "impression_id": impression_id,
                "user_id": str(row["user_id"]),
                "positive": str(row["positive"]),
                "all_acceptable_subcategories": acceptable,
                "history_subcategories": history,
                "negative_pool": pool,
                "negative_pool_size": len(pool),
            }
        )
    positions = _correct_positions(drafts, k=k, seed=seed)

    outputs: dict[str, list[dict[str, object]]] = {}
    for strategy in STRATEGIES:
        strategy_rows: list[dict[str, object]] = []
        for draft in drafts:
            selection_id = str(draft["selection_id"])
            negatives = _negative_order(
                list(draft["negative_pool"]),
                selection_id=selection_id,
                strategy=strategy,
                train_counts=train_counts,
                seed=seed,
            )[: k - 1]
            negatives.sort(
                key=lambda label: (
                    _priority(
                        seed,
                        f"{strategy}-candidate-order",
                        selection_id,
                        label,
                    ),
                    label,
                )
            )
            correct_position = positions[selection_id]
            candidates = negatives.copy()
            candidates.insert(correct_position, str(draft["positive"]))
            strategy_rows.append(
                {
                    key: value
                    for key, value in draft.items()
                    if key != "negative_pool"
                }
                | {
                    "negative_sampling": strategy,
                    "negative_candidates": negatives,
                    "candidates": candidates,
                    "correct_position": correct_position,
                }
            )
        outputs[strategy] = strategy_rows
    return outputs


def _render_prompt(
    history: list[dict[str, object]],
    candidates: list[str],
    *,
    task_type: str,
) -> str:
    if task_type == "behavior_only":
        header = BEHAVIOR_HISTORY_HEADER
        values = [str(item["title"]) for item in history]
        instruction = BEHAVIOR_INSTRUCTION
    elif task_type == "interest_only":
        header = INTEREST_HISTORY_HEADER
        values = [str(item["subcategory"]) for item in history]
        instruction = INTEREST_INSTRUCTION
    elif task_type == "behavior_interest":
        header = BEHAVIOR_INTEREST_HISTORY_HEADER
        values = [
            f"[{item['subcategory']}] {item['title']}" for item in history
        ]
        instruction = BEHAVIOR_INTEREST_INSTRUCTION
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    history_block = "\n".join(
        f"{index}. {value}" for index, value in enumerate(values, start=1)
    )
    candidate_block = "\n".join(
        f"{index}. {candidate}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return (
        f"{header}\n\n{history_block}\n\n{instruction}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        f"{ANSWER_INSTRUCTION}"
    )


def _write_strategy(
    staging: Path,
    strategy: str,
    dev_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> dict[str, object]:
    source_by_sample = {str(row["sample_id"]): row for row in dev_rows}
    task_rows: dict[str, list[dict[str, object]]] = {
        task_type: [] for task_type in TASK_TYPES
    }
    for manifest in manifest_rows:
        source = source_by_sample[str(manifest["sample_id"])]
        common = {
            "selection_id": manifest["selection_id"],
            "sample_id": manifest["sample_id"],
            "impression_id": manifest["impression_id"],
            "user_id": manifest["user_id"],
            "timestamp": source["timestamp"],
            "negative_sampling": strategy,
            "history_length": len(source["history"]),
            "candidates": manifest["candidates"],
            "answer_category": manifest["positive"],
            "answer": f"New topic: {manifest['positive']}",
            "correct_position": manifest["correct_position"],
            "all_acceptable_subcategories": manifest[
                "all_acceptable_subcategories"
            ],
            "history_subcategories": manifest["history_subcategories"],
        }
        for task_type in task_rows:
            task_rows[task_type].append(
                common
                | {
                    "task_type": task_type,
                    "prompt": _render_prompt(
                        source["history"],
                        list(manifest["candidates"]),
                        task_type=task_type,
                    ),
                }
            )

    relative_paths: list[Path] = []
    for task_type, rows in task_rows.items():
        relative = Path(strategy) / task_type / "dev_expanded.parquet"
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=CANDIDATE_DATA_SCHEMA),
            path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=2_000,
        )
        relative_paths.append(relative)
    manifest_relative = Path(strategy) / "candidate_manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(manifest_rows, schema=MANIFEST_SCHEMA),
        staging / manifest_relative,
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )
    relative_paths.append(manifest_relative)

    position_counts = Counter(
        int(row["correct_position"]) for row in manifest_rows
    )
    positive_counts = Counter(str(row["positive"]) for row in manifest_rows)
    negative_counts = Counter(
        negative
        for row in manifest_rows
        for negative in row["negative_candidates"]
    )
    return {
        "relative_paths": [str(path) for path in relative_paths],
        "statistics": {
            "samples": len(manifest_rows),
            "position_counts": {
                str(position): position_counts[position]
                for position in range(len(manifest_rows[0]["candidates"]))
            },
            "positive_counts": dict(
                sorted(positive_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "negative_counts": dict(
                sorted(negative_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "negative_pool_size": _distribution_summary(
                [int(row["negative_pool_size"]) for row in manifest_rows]
            ),
        },
    }


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


def build_static_user_disjoint_candidate_selection(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Path:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if k <= 1:
        raise ValueError("k must be greater than 1")
    train_path = dataset_dir / "train.parquet"
    dev_path = dataset_dir / "dev.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    for path in (train_path, dev_path, vocabulary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )

    train_counts = _read_train_counts(train_path)
    vocabulary = {
        str(value)
        for value in pq.read_table(
            vocabulary_path, columns=["subcategory"]
        )["subcategory"].to_pylist()
    }
    if not set(train_counts).issubset(vocabulary):
        raise ValueError("Train labels are not a subset of the vocabulary")
    dev_rows, acceptable_by_impression = _load_dev_rows(
        dev_path, set(train_counts)
    )
    manifests = _build_manifests(
        dev_rows,
        acceptable_by_impression,
        train_counts,
        k=k,
        seed=seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-candidate-selection-",
        dir=str(output_dir.parent),
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        strategy_outputs = {
            strategy: _write_strategy(
                staging,
                strategy,
                dev_rows,
                manifests[strategy],
            )
            for strategy in STRATEGIES
        }
        users = {str(row["user_id"]) for row in dev_rows}
        impressions = {str(row["impression_id"]) for row in dev_rows}
        positives = {str(row["positive"]) for row in dev_rows}
        history_lengths = [len(row["history"]) for row in dev_rows]
        statistics = {
            "rule_version": RULE_VERSION,
            "k": k,
            "samples": len(dev_rows),
            "users": len(users),
            "impressions": len(impressions),
            "train_supported_interest_count": len(train_counts),
            "dev_supported_interest_count": len(positives),
            "history_length": _distribution_summary(history_lengths),
            "strategies": {
                strategy: value["statistics"]
                for strategy, value in strategy_outputs.items()
            },
        }
        statistics_path = staging / "statistics.json"
        statistics_path.write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_relatives = [
            Path(relative)
            for value in strategy_outputs.values()
            for relative in value["relative_paths"]
        ] + [Path("statistics.json")]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "dataset_dir": portable_path(dataset_dir),
            "output_dir": portable_path(output_dir),
            "k": k,
            "seed": seed,
            "sample_unit": "one_static_dev_target_row",
            "candidate_universe": "static_train_supported_subcategories",
            "positive_policy": "one_sample_per_dev_target_row",
            "negative_exclusions": [
                "all_targets_for_same_impression",
                "all_raw_history_subcategories",
            ],
            "uniform_policy": "sha256_priority_without_replacement",
            "frequency_policy": (
                "deterministic_exponential_race_weighted_by_static_train_counts"
            ),
            "position_policy": "sha256_order_then_global_rank_modulo_k",
            "history_policy": "complete_raw_history_no_append_no_truncation",
            "answer_format": "New topic: <exact_category_name>",
            "source_sha256": {
                "train.parquet": _sha256(train_path),
                "dev.parquet": _sha256(dev_path),
                "interest_vocabulary.parquet": _sha256(vocabulary_path),
            },
            "artifact_sha256": {
                str(relative): _sha256(staging / relative)
                for relative in artifact_relatives
            },
            "prompt_templates": {
                "behavior_history_header": BEHAVIOR_HISTORY_HEADER,
                "interest_history_header": INTEREST_HISTORY_HEADER,
                "behavior_interest_history_header": (
                    BEHAVIOR_INTEREST_HISTORY_HEADER
                ),
                "behavior_instruction": BEHAVIOR_INSTRUCTION,
                "interest_instruction": INTEREST_INSTRUCTION,
                "behavior_interest_instruction": (
                    BEHAVIOR_INTEREST_INSTRUCTION
                ),
                "answer_instruction": ANSWER_INSTRUCTION,
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)

    LOGGER.info(
        "Wrote %s candidate rows for %s users to %s",
        f"{len(dev_rows):,}",
        f"{len(users):,}",
        output_dir,
    )
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired K=10 candidate-selection data from the static "
            "user-disjoint dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    build_static_user_disjoint_candidate_selection(
        args.dataset_dir,
        args.output_dir,
        k=args.k,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
