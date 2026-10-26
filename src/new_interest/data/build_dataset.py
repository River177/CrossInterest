from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.mind_io import (
    NewsCatalog,
    iter_behavior_events,
    load_news_catalog,
)
from new_interest.infrastructure.paths import portable_path

LOGGER = logging.getLogger(__name__)

RULE_VERSION = "clicked-interest-absent-from-history-v1"
OUTPUT_FILES = (
    "train.parquet",
    "dev.parquet",
    "interest_vocabulary.parquet",
    "statistics.json",
    "metadata.json",
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_samples: int
    dev_samples: int


HISTORY_ITEM_TYPE = pa.struct(
    [
        pa.field("news_id", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
    ]
)

TARGET_INTEREST_TYPE = pa.struct(
    [
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
    ]
)

TARGET_NEWS_ITEM_TYPE = pa.struct(
    [
        pa.field("news_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
    ]
)

DATASET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("s"), nullable=False),
        pa.field("history", pa.list_(HISTORY_ITEM_TYPE), nullable=False),
        pa.field("target_interest", TARGET_INTEREST_TYPE, nullable=False),
        pa.field("target_news", pa.list_(TARGET_NEWS_ITEM_TYPE), nullable=False),
    ]
)

VOCABULARY_SCHEMA = pa.schema(
    [
        pa.field("interest_id", pa.int16(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
    ]
)


class BatchedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int) -> None:
        self._schema = schema
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(
            path,
            schema,
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
        table = pa.Table.from_pylist(self._rows, schema=self._schema)
        self._writer.write_table(table, row_group_size=self._batch_size)
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _build_split(
    split: str,
    input_dir: Path,
    output_path: Path,
    catalog: NewsCatalog,
    batch_size: int,
) -> tuple[dict[str, object], Counter[int], Counter[int]]:
    writer = BatchedParquetWriter(output_path, DATASET_SCHEMA, batch_size)
    counters: Counter[str] = Counter()
    target_counts: Counter[int] = Counter()
    history_lengths: Counter[int] = Counter()
    target_news_lengths: Counter[int] = Counter()
    try:
        for event in iter_behavior_events(input_dir, (split,), catalog):
            impression_id = event.impression_id
            user_id = event.user_id
            counters["behavior_rows"] += 1

            history: list[dict[str, str]] = []
            history_interests: set[int] = set()
            for news_id in event.history_news_ids:
                record = catalog.news[news_id]
                history_interests.add(record.interest_id)
                history.append(
                    {
                        "news_id": record.news_id,
                        "category": record.category,
                        "subcategory": record.subcategory,
                        "title": record.title,
                    }
                )

            positive_news: dict[int, list[str]] = {}
            seen_positive_news: dict[int, set[str]] = {}
            for news_id in event.clicked_news_ids:
                record = catalog.news[news_id]
                target_list = positive_news.setdefault(record.interest_id, [])
                target_seen = seen_positive_news.setdefault(
                    record.interest_id,
                    set(),
                )
                if news_id not in target_seen:
                    target_list.append(news_id)
                    target_seen.add(news_id)

            counters["distinct_positive_interests"] += len(positive_news)
            new_targets = [
                (interest_id, target_news_ids)
                for interest_id, target_news_ids in positive_news.items()
                if interest_id not in history_interests
            ]
            if new_targets:
                counters["eligible_behavior_rows"] += 1

            for interest_id, target_news_ids in new_targets:
                category, subcategory = catalog.interests[interest_id]
                writer.append(
                    {
                        "sample_id": f"{split}:{impression_id}:{interest_id}",
                        "split": split,
                        "impression_id": impression_id,
                        "user_id": user_id,
                        "timestamp": event.timestamp,
                        "history": history,
                        "target_interest": {
                            "category": category,
                            "subcategory": subcategory,
                        },
                        "target_news": [
                            {
                                "news_id": news_id,
                                "title": catalog.news[news_id].title,
                            }
                            for news_id in target_news_ids
                        ],
                    }
                )
                target_counts[interest_id] += 1
                history_lengths[len(history)] += 1
                target_news_lengths[len(target_news_ids)] += 1
                counters["samples"] += 1

            if counters["behavior_rows"] % 250_000 == 0:
                LOGGER.info(
                    "Processed %s %s behavior rows",
                    f"{counters['behavior_rows']:,}",
                    split,
                )
    finally:
        writer.close()

    if writer.row_count != counters["samples"]:
        raise ValueError(
            f"{split}: wrote {writer.row_count} rows but counted "
            f"{counters['samples']} samples"
        )

    stats = {
        **dict(sorted(counters.items())),
        "target_interest_counts": {
            f"{catalog.interests[interest_id][0]} / "
            f"{catalog.interests[interest_id][1]}": count
            for interest_id, count in sorted(
                target_counts.items(),
                key=lambda item: (-item[1], catalog.interests[item[0]]),
            )
        },
        "history_length": _distribution_summary(history_lengths),
        "target_news_per_sample": {
            str(length): count
            for length, count in sorted(target_news_lengths.items())
        },
    }
    return stats, history_lengths, target_news_lengths


def _distribution_summary(counts: Counter[int]) -> dict[str, float | int | None]:
    total = sum(counts.values())
    if total == 0:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }

    def percentile(fraction: float) -> int:
        rank = max(1, int(total * fraction + 0.999999))
        cumulative = 0
        for value in sorted(counts):
            cumulative += counts[value]
            if cumulative >= rank:
                return value
        raise AssertionError("Unreachable percentile calculation")

    weighted_sum = sum(value * count for value, count in counts.items())
    return {
        "count": total,
        "min": min(counts),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(counts),
        "mean": weighted_sum / total,
    }


def _write_vocabulary(path: Path, interests: list[tuple[str, str]]) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "interest_id": interest_id,
                "category": category,
                "subcategory": subcategory,
            }
            for interest_id, (category, subcategory) in enumerate(interests)
        ],
        schema=VOCABULARY_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(input_dir: Path) -> dict[str, object] | None:
    path = input_dir / "MANIFEST.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}: "
            f"{[path.name for path in existing]}. Use --overwrite."
        )


def build_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    batch_size: int = 5_000,
    overwrite: bool = False,
) -> BuildResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for split in ("train", "dev"):
        for filename in ("news.tsv", "behaviors.tsv"):
            path = input_dir / split / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_output_available(output_dir, overwrite)

    LOGGER.info("Loading news metadata and interest vocabulary")
    catalog = load_news_catalog(input_dir)

    with tempfile.TemporaryDirectory(
        prefix=".new-interest-output-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        split_stats = {}
        sample_counts = {}
        for split in ("train", "dev"):
            LOGGER.info("Building %s dataset", split)
            stats, _, _ = _build_split(
                split,
                input_dir,
                staging / f"{split}.parquet",
                catalog,
                batch_size,
            )
            split_stats[split] = stats
            sample_counts[split] = int(stats["samples"])

        _write_vocabulary(
            staging / "interest_vocabulary.parquet",
            catalog.interests,
        )

        statistics = {
            "rule_version": RULE_VERSION,
            "splits": split_stats,
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_hashes = {
            name: _sha256(staging / name)
            for name in (
                "train.parquet",
                "dev.parquet",
                "interest_vocabulary.parquet",
                "statistics.json",
            )
        }
        source_manifest_path = input_dir / "MANIFEST.json"
        source_filtered_path = input_dir / "FILTERED_NEWS_IDS.json"
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "definition": (
                "A current clicked category-subcategory pair absent from the "
                "behavior record's clicked-history interest set."
            ),
            "prior_exposure_required": False,
            "negative_impressions_used": False,
            "source_dir": portable_path(input_dir),
            "source_manifest": _load_manifest(input_dir),
            "source_file_sha256": {
                "MANIFEST.json": (
                    _sha256(source_manifest_path)
                    if source_manifest_path.is_file()
                    else None
                ),
                "FILTERED_NEWS_IDS.json": (
                    _sha256(source_filtered_path)
                    if source_filtered_path.is_file()
                    else None
                ),
            },
            "news_rows": catalog.news_rows,
            "unique_news": len(catalog.news),
            "interest_count": len(catalog.interests),
            "supervised_splits": ["train", "dev"],
            "test_exclusion_reason": (
                "MIND test impressions do not contain -1/-0 click labels."
            ),
            "artifact_sha256": artifact_hashes,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_FILES:
            os.replace(staging / name, output_dir / name)

    return BuildResult(
        output_dir=output_dir,
        train_samples=sample_counts["train"],
        dev_samples=sample_counts["dev"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the simplified MIND new-interest dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/MINDlarge_reclassified"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/new_interest"),
    )
    parser.add_argument("--batch-size", type=int, default=5_000)
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
    result = build_dataset(
        args.input_dir,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Wrote %s Train and %s Dev samples to %s",
        f"{result.train_samples:,}",
        f"{result.dev_samples:,}",
        result.output_dir,
    )


if __name__ == "__main__":
    main()
