from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import DATASET_SCHEMA
from new_interest.data.build_dynamic_dataset import DYNAMIC_DATASET_SCHEMA
from new_interest.data.build_generative_dataset import (
    TASK_TYPES,
    BuildResult,
    build_generative_dataset,
)
from new_interest.infrastructure.paths import portable_path

LOGGER = logging.getLogger(__name__)

RULE_VERSION = "dynamic-generative-balanced-v1"
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_SAMPLES_PER_CLASS = 200
DEFAULT_SAMPLING_SEED = 20260806


class TargetRowWriter:
    def __init__(self, path: Path, batch_size: int) -> None:
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(
            path,
            DATASET_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        self.row_count = 0

    def append(self, row: dict[str, object]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        self._writer.write_table(
            pa.Table.from_pylist(self._rows, schema=DATASET_SCHEMA),
            row_group_size=self._batch_size,
        )
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dynamic_schema(path: Path) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    missing = set(DYNAMIC_DATASET_SCHEMA.names) - set(schema.names)
    if missing:
        raise ValueError(
            f"{path}: missing dynamic source columns {sorted(missing)}"
        )


def _distribution(counts: Counter[int]) -> dict[str, object]:
    total = sum(counts.values())
    memberships = sum(value * count for value, count in counts.items())
    return {
        "impressions": total,
        "novel_target_memberships": memberships,
        "mean_novel_targets_per_impression": (
            memberships / total if total else None
        ),
        "count_distribution": {
            str(value): count for value, count in sorted(counts.items())
        },
        "rate_distribution": {
            str(value): count / total
            for value, count in sorted(counts.items())
        }
        if total
        else {},
    }


def _load_interest_ids(source_dir: Path) -> dict[tuple[str, str], int]:
    rows = pq.read_table(source_dir / "interest_vocabulary.parquet").to_pylist()
    return {
        (str(row["category"]), str(row["subcategory"])): int(
            row["interest_id"]
        )
        for row in rows
    }


def _expand_dynamic_split(
    source_path: Path,
    output_path: Path,
    interest_ids: dict[tuple[str, str], int],
    *,
    batch_size: int,
) -> dict[str, object]:
    _validate_dynamic_schema(source_path)
    writer = TargetRowWriter(output_path, batch_size)
    all_counts: Counter[int] = Counter()
    positive_counts: Counter[int] = Counter()
    target_counts: Counter[str] = Counter()

    columns = [
        "split",
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "clicked_news",
        "novel_interests",
    ]
    try:
        for batch in pq.ParquetFile(source_path).iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            for row in batch.to_pylist():
                split = str(row["split"])
                impression_id = str(row["impression_id"])
                novel_interests = list(row["novel_interests"])
                target_count = len(novel_interests)
                all_counts[target_count] += 1
                if target_count:
                    positive_counts[target_count] += 1

                clicked_by_interest: dict[
                    tuple[str, str],
                    list[dict[str, str]],
                ] = {}
                for clicked in row["clicked_news"]:
                    key = (
                        str(clicked["category"]),
                        str(clicked["subcategory"]),
                    )
                    clicked_by_interest.setdefault(key, []).append(
                        {
                            "news_id": str(clicked["news_id"]),
                            "title": str(clicked["title"]),
                        }
                    )

                for target in novel_interests:
                    category = str(target["category"])
                    subcategory = str(target["subcategory"])
                    key = (category, subcategory)
                    try:
                        interest_id = interest_ids[key]
                    except KeyError as error:
                        raise ValueError(
                            f"{split}:{impression_id}: unknown interest {key}"
                        ) from error
                    target_news = clicked_by_interest.get(key, [])
                    if not target_news:
                        raise ValueError(
                            f"{split}:{impression_id}: no clicked news for "
                            f"novel interest {subcategory!r}"
                        )
                    writer.append(
                        {
                            "sample_id": (
                                f"{split}:{impression_id}:{interest_id}"
                            ),
                            "split": split,
                            "impression_id": impression_id,
                            "user_id": str(row["user_id"]),
                            "timestamp": row["timestamp"],
                            "history": row["history"],
                            "target_interest": {
                                "category": category,
                                "subcategory": subcategory,
                            },
                            "target_news": target_news,
                        }
                    )
                    target_counts[subcategory] += 1
    finally:
        writer.close()

    return {
        "source_impressions": sum(all_counts.values()),
        "positive_impressions": sum(positive_counts.values()),
        "expanded_target_rows": writer.row_count,
        "all_impressions": _distribution(all_counts),
        "positive_impressions_only": _distribution(positive_counts),
        "target_counts": dict(
            sorted(
                target_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


def _selected_train_distribution(
    source_path: Path,
    selected_impression_ids: set[str],
    *,
    batch_size: int,
) -> dict[str, object]:
    counts: Counter[int] = Counter()
    found: set[str] = set()
    for batch in pq.ParquetFile(source_path).iter_batches(
        batch_size=batch_size,
        columns=["impression_id", "novel_interests"],
    ):
        for row in batch.to_pylist():
            impression_id = str(row["impression_id"])
            if impression_id not in selected_impression_ids:
                continue
            if impression_id in found:
                raise ValueError(
                    f"Duplicate dynamic Train impression {impression_id}"
                )
            found.add(impression_id)
            counts[len(row["novel_interests"])] += 1
    missing = selected_impression_ids - found
    if missing:
        raise ValueError(
            f"Missing {len(missing)} selected Train impressions in source"
        )
    return _distribution(counts)


def _filtered_dev_distribution(
    source_path: Path,
    retained_interests: set[str],
    *,
    batch_size: int,
) -> dict[str, object]:
    all_counts: Counter[int] = Counter()
    positive_counts: Counter[int] = Counter()
    for batch in pq.ParquetFile(source_path).iter_batches(
        batch_size=batch_size,
        columns=["novel_interests"],
    ):
        for row in batch.to_pylist():
            retained_count = sum(
                str(target["subcategory"]) in retained_interests
                for target in row["novel_interests"]
            )
            all_counts[retained_count] += 1
            if retained_count:
                positive_counts[retained_count] += 1
    return {
        "all_impressions": _distribution(all_counts),
        "positive_impressions_only": _distribution(positive_counts),
    }


def _rewrite_provenance(
    output_dir: Path,
    source_dir: Path,
    source_statistics: dict[str, object],
    *,
    history_limit: int,
    samples_per_class: int,
    sampling_seed: int,
    batch_size: int,
) -> None:
    selected_rows = pq.read_table(
        output_dir / "selected_train_samples.parquet"
    ).to_pylist()
    selected_impression_ids = {
        str(row["impression_id"]) for row in selected_rows
    }
    if len(selected_impression_ids) != len(selected_rows):
        raise ValueError(
            "Balanced Train selection contains duplicate impressions"
        )

    retained_rows = pq.read_table(
        output_dir / "retained_interests.parquet"
    ).to_pylist()
    retained_interests = {
        str(row["target_subcategory"]) for row in retained_rows
    }
    final_distributions = {
        "balanced_train_selected_impressions": (
            _selected_train_distribution(
                source_dir / "train.parquet",
                selected_impression_ids,
                batch_size=batch_size,
            )
        ),
        "online_dev_after_retained_class_filter": (
            _filtered_dev_distribution(
                source_dir / "dev.parquet",
                retained_interests,
                batch_size=batch_size,
            )
        ),
    }

    statistics_path = output_dir / "statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["rule_version"] = RULE_VERSION
    statistics["dynamic_source"] = source_statistics
    statistics["novel_targets_per_impression"] = final_distributions
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "definition": (
                "Generate one exact novel-interest label from online dynamic "
                "click history. Train selects one deterministic novel target "
                "per impression before class-balanced sampling; Dev retains "
                "all novel targets in the retained Train label set."
            ),
            "source_dir": portable_path(source_dir),
            "dynamic_source_rule_version": "dynamic-click-history-v1",
            "dev_policy": "online_dynamic",
            "history_limit": history_limit,
            "train_samples_per_class": samples_per_class,
            "sampling_seed": sampling_seed,
            "source_sha256": {
                f"{split}.parquet": _sha256(
                    source_dir / f"{split}.parquet"
                )
                for split in ("train", "dev")
            },
        }
    )
    metadata["artifact_sha256"]["statistics.json"] = _sha256(statistics_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_dynamic_generative_dataset(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    batch_size: int = 5_000,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    overwrite: bool = False,
) -> BuildResult:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    for name in (
        "train.parquet",
        "dev.parquet",
        "interest_vocabulary.parquet",
        "metadata.json",
    ):
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)

    interest_ids = _load_interest_ids(source_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".dynamic-target-rows-",
        dir=str(output_dir.parent),
    ) as temporary_name:
        target_source_dir = Path(temporary_name)
        source_statistics: dict[str, object] = {}
        for split in ("train", "dev"):
            LOGGER.info("Expanding dynamic %s novel targets", split)
            source_statistics[split] = _expand_dynamic_split(
                source_dir / f"{split}.parquet",
                target_source_dir / f"{split}.parquet",
                interest_ids,
                batch_size=batch_size,
            )

        result = build_generative_dataset(
            target_source_dir,
            output_dir,
            batch_size=batch_size,
            overwrite=overwrite,
            history_limit=history_limit,
            train_samples_per_class=samples_per_class,
            train_min_class_count=samples_per_class,
            filter_dev_to_train_classes=True,
            sampling_seed=sampling_seed,
            one_target_per_train_impression=True,
        )

    _rewrite_provenance(
        output_dir,
        source_dir,
        source_statistics,
        history_limit=history_limit,
        samples_per_class=samples_per_class,
        sampling_seed=sampling_seed,
        batch_size=batch_size,
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired Behavior/Interest generative data from online "
            "dynamic histories with one Train target per impression and "
            "class-balanced sampling."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/new_interest_dynamic_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/training/"
            "generative_new_interest_dynamic_balanced_200_h50"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument(
        "--history-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=DEFAULT_SAMPLING_SEED,
    )
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
    result = build_dynamic_generative_dataset(
        args.source_dir,
        args.output_dir,
        batch_size=args.batch_size,
        history_limit=args.history_limit,
        samples_per_class=args.samples_per_class,
        sampling_seed=args.sampling_seed,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Wrote paired Behavior/Interest Train %s and online Dev %s to %s",
        f"{result.behavior_train_samples:,}",
        f"{result.behavior_dev_samples:,}",
        result.output_dir,
    )


if __name__ == "__main__":
    main()
