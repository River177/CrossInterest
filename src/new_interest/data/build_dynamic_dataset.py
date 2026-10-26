from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import (
    HISTORY_ITEM_TYPE,
    TARGET_INTEREST_TYPE,
)
from new_interest.data.mind_io import (
    NewsCatalog,
    NewsRecord,
    iter_behavior_events,
    load_news_catalog,
)
from new_interest.infrastructure.paths import portable_path

LOGGER = logging.getLogger(__name__)

RULE_VERSION = "dynamic-click-history-v1"
SPLITS = ("train", "dev")
OUTPUT_FILES = (
    "train.parquet",
    "dev.parquet",
    "interest_vocabulary.parquet",
    "statistics.json",
    "metadata.json",
)

DYNAMIC_DATASET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("initial_history_length", pa.int32(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("history", pa.list_(HISTORY_ITEM_TYPE), nullable=False),
        pa.field("clicked_news", pa.list_(HISTORY_ITEM_TYPE), nullable=False),
        pa.field(
            "clicked_interests",
            pa.list_(TARGET_INTEREST_TYPE),
            nullable=False,
        ),
        pa.field(
            "novel_interests",
            pa.list_(TARGET_INTEREST_TYPE),
            nullable=False,
        ),
        pa.field(
            "repeated_interests",
            pa.list_(TARGET_INTEREST_TYPE),
            nullable=False,
        ),
        pa.field("has_new_interest", pa.bool_(), nullable=False),
    ]
)

VOCABULARY_SCHEMA = pa.schema(
    [
        pa.field("interest_id", pa.int16(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class DynamicBuildResult:
    output_dir: Path
    train_impressions: int
    dev_impressions: int
    train_new_interest_impressions: int
    dev_new_interest_impressions: int


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


class ParquetBatchWriter:
    def __init__(self, path: Path, batch_size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(
            path,
            DYNAMIC_DATASET_SCHEMA,
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
        table = pa.Table.from_pylist(
            self._rows,
            schema=DYNAMIC_DATASET_SCHEMA,
        )
        self._writer.write_table(table, row_group_size=self._batch_size)
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


class SplitStatistics:
    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.initial_history_lengths: Counter[int] = Counter()
        self.history_lengths: Counter[int] = Counter()
        self.clicked_news_counts: Counter[int] = Counter()
        self.clicked_interest_counts: Counter[int] = Counter()
        self.novel_interest_counts: Counter[int] = Counter()
        self.repeated_interest_counts: Counter[int] = Counter()
        self.novel_target_counts: Counter[str] = Counter()
        self.repeated_target_counts: Counter[str] = Counter()

    def add(
        self,
        *,
        initial_history_length: int,
        history_length: int,
        clicked_news: list[dict[str, str]],
        clicked_interests: list[dict[str, str]],
        novel_interests: list[dict[str, str]],
        repeated_interests: list[dict[str, str]],
    ) -> None:
        self.counters["impressions"] += 1
        self.counters["impressions_with_clicks"] += int(bool(clicked_news))
        self.counters["impressions_without_clicks"] += int(not clicked_news)
        self.counters["new_interest_impressions"] += int(bool(novel_interests))
        self.counters["no_new_interest_impressions"] += int(
            not novel_interests
        )
        self.counters["clicked_news"] += len(clicked_news)
        self.counters["clicked_interest_memberships"] += len(
            clicked_interests
        )
        self.counters["novel_interest_memberships"] += len(novel_interests)
        self.counters["repeated_interest_memberships"] += len(
            repeated_interests
        )
        self.initial_history_lengths[initial_history_length] += 1
        self.history_lengths[history_length] += 1
        self.clicked_news_counts[len(clicked_news)] += 1
        self.clicked_interest_counts[len(clicked_interests)] += 1
        self.novel_interest_counts[len(novel_interests)] += 1
        self.repeated_interest_counts[len(repeated_interests)] += 1
        self.novel_target_counts.update(
            str(target["subcategory"]) for target in novel_interests
        )
        self.repeated_target_counts.update(
            str(target["subcategory"]) for target in repeated_interests
        )

    def as_dict(self) -> dict[str, object]:
        return {
            **dict(sorted(self.counters.items())),
            "initial_history_length": _distribution_summary(
                self.initial_history_lengths
            ),
            "dynamic_history_length": _distribution_summary(
                self.history_lengths
            ),
            "clicked_news_per_impression": _distribution_summary(
                self.clicked_news_counts
            ),
            "clicked_interests_per_impression": _distribution_summary(
                self.clicked_interest_counts
            ),
            "novel_interests_per_impression": _distribution_summary(
                self.novel_interest_counts
            ),
            "repeated_interests_per_impression": _distribution_summary(
                self.repeated_interest_counts
            ),
            "novel_target_counts": dict(
                sorted(
                    self.novel_target_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "repeated_target_counts": dict(
                sorted(
                    self.repeated_target_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }


def _configure_staging_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        PRAGMA locking_mode = EXCLUSIVE;

        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            history_text TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE events (
            split TEXT NOT NULL,
            split_rank INTEGER NOT NULL,
            impression_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            timestamp_text TEXT NOT NULL,
            source_order INTEGER NOT NULL,
            positive_news_text TEXT NOT NULL,
            PRIMARY KEY (split, impression_id)
        ) WITHOUT ROWID;
        """
    )


def _flush_staging_rows(
    connection: sqlite3.Connection,
    user_rows: list[tuple[str, str]],
    event_rows: list[tuple[object, ...]],
) -> None:
    if user_rows:
        connection.executemany(
            "INSERT INTO users (user_id, history_text) VALUES (?, ?)",
            user_rows,
        )
        user_rows.clear()
    if event_rows:
        connection.executemany(
            """
            INSERT INTO events (
                split,
                split_rank,
                impression_id,
                user_id,
                timestamp_text,
                source_order,
                positive_news_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )
        event_rows.clear()
    connection.commit()


def _stage_events(
    input_dir: Path,
    database_path: Path,
    catalog: NewsCatalog,
    *,
    staging_batch_size: int,
) -> dict[str, object]:
    connection = sqlite3.connect(database_path)
    _configure_staging_database(connection)
    history_digests: dict[str, bytes] = {}
    user_rows: list[tuple[str, str]] = []
    event_rows: list[tuple[object, ...]] = []
    split_counts: Counter[str] = Counter()
    split_min_time: dict[str, datetime] = {}
    split_max_time: dict[str, datetime] = {}

    try:
        for split_rank, split in enumerate(SPLITS):
            for source_order, event in enumerate(
                iter_behavior_events(input_dir, (split,), catalog)
            ):
                history_digest = hashlib.sha256(
                    event.history_text.encode("utf-8")
                ).digest()
                previous_digest = history_digests.get(event.user_id)
                if previous_digest is None:
                    history_digests[event.user_id] = history_digest
                    user_rows.append((event.user_id, event.history_text))
                elif previous_digest != history_digest:
                    raise ValueError(
                        f"User {event.user_id} has inconsistent static history "
                        f"at {split} impression {event.impression_id}"
                    )

                event_rows.append(
                    (
                        split,
                        split_rank,
                        event.impression_id,
                        event.user_id,
                        event.timestamp.isoformat(sep=" "),
                        source_order,
                        " ".join(event.clicked_news_ids),
                    )
                )
                split_counts[split] += 1
                split_min_time[split] = min(
                    event.timestamp,
                    split_min_time.get(split, event.timestamp),
                )
                split_max_time[split] = max(
                    event.timestamp,
                    split_max_time.get(split, event.timestamp),
                )

                if len(event_rows) >= staging_batch_size:
                    _flush_staging_rows(
                        connection,
                        user_rows,
                        event_rows,
                    )
                if split_counts[split] % 250_000 == 0:
                    LOGGER.info(
                        "Staged %s %s behavior rows",
                        f"{split_counts[split]:,}",
                        split,
                    )
        _flush_staging_rows(connection, user_rows, event_rows)
        connection.execute(
            """
            CREATE INDEX events_dynamic_order
            ON events (
                user_id,
                timestamp_text,
                split_rank,
                source_order
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    if split_max_time["train"] >= split_min_time["dev"]:
        raise ValueError(
            "Online Dev construction requires every Train timestamp to "
            "precede every Dev timestamp"
        )

    return {
        "users": len(history_digests),
        "split_behavior_rows": dict(split_counts),
        "split_time_range": {
            split: {
                "min": split_min_time[split].isoformat(),
                "max": split_max_time[split].isoformat(),
            }
            for split in SPLITS
        },
    }


def _history_item(record: NewsRecord) -> dict[str, str]:
    return {
        "news_id": record.news_id,
        "category": record.category,
        "subcategory": record.subcategory,
        "title": record.title,
    }


def _interest_item(record: NewsRecord) -> dict[str, str]:
    return {
        "category": record.category,
        "subcategory": record.subcategory,
    }


def _distinct_interests(
    clicked_records: list[NewsRecord],
) -> list[NewsRecord]:
    distinct: list[NewsRecord] = []
    seen: set[int] = set()
    for record in clicked_records:
        if record.interest_id in seen:
            continue
        seen.add(record.interest_id)
        distinct.append(record)
    return distinct


def _write_dynamic_rows(
    database_path: Path,
    staging: Path,
    catalog: NewsCatalog,
    *,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, int]]:
    writers = {
        split: ParquetBatchWriter(staging / f"{split}.parquet", batch_size)
        for split in SPLITS
    }
    statistics = {split: SplitStatistics() for split in SPLITS}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    current_user: str | None = None
    initial_history_length = 0
    history: list[dict[str, str]] = []
    seen_interests: set[int] = set()

    query = """
        SELECT
            split,
            impression_id,
            user_id,
            timestamp_text,
            source_order,
            positive_news_text
        FROM events
        ORDER BY
            user_id,
            timestamp_text,
            split_rank,
            source_order
    """

    try:
        rows = connection.execute(query)
        for (user_id, timestamp_text), timestamp_group in groupby(
            rows,
            key=lambda row: (str(row["user_id"]), str(row["timestamp_text"])),
        ):
            if user_id != current_user:
                history_row = connection.execute(
                    "SELECT history_text FROM users WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if history_row is None:
                    raise ValueError(f"Missing initial history for {user_id}")
                history = [
                    _history_item(catalog.news[news_id])
                    for news_id in str(history_row["history_text"]).split()
                ]
                initial_history_length = len(history)
                seen_interests = {
                    catalog.news[str(item["news_id"])].interest_id
                    for item in history
                }
                current_user = user_id

            pending_clicks: list[list[NewsRecord]] = []
            group_rows = list(timestamp_group)
            for row in group_rows:
                split = str(row["split"])
                impression_id = str(row["impression_id"])
                positive_news_ids = str(row["positive_news_text"]).split()
                clicked_records = [
                    catalog.news[news_id]
                    for news_id in positive_news_ids
                ]
                clicked_interests = _distinct_interests(clicked_records)
                novel_records = [
                    record
                    for record in clicked_interests
                    if record.interest_id not in seen_interests
                ]
                repeated_records = [
                    record
                    for record in clicked_interests
                    if record.interest_id in seen_interests
                ]
                clicked_news = [
                    _history_item(record) for record in clicked_records
                ]
                clicked_interest_rows = [
                    _interest_item(record) for record in clicked_interests
                ]
                novel_interest_rows = [
                    _interest_item(record) for record in novel_records
                ]
                repeated_interest_rows = [
                    _interest_item(record) for record in repeated_records
                ]
                writers[split].append(
                    {
                        "sample_id": f"{split}:{impression_id}",
                        "split": split,
                        "impression_id": impression_id,
                        "user_id": user_id,
                        "timestamp": datetime.fromisoformat(timestamp_text),
                        "initial_history_length": initial_history_length,
                        "history_length": len(history),
                        "history": list(history),
                        "clicked_news": clicked_news,
                        "clicked_interests": clicked_interest_rows,
                        "novel_interests": novel_interest_rows,
                        "repeated_interests": repeated_interest_rows,
                        "has_new_interest": bool(novel_interest_rows),
                    }
                )
                statistics[split].add(
                    initial_history_length=initial_history_length,
                    history_length=len(history),
                    clicked_news=clicked_news,
                    clicked_interests=clicked_interest_rows,
                    novel_interests=novel_interest_rows,
                    repeated_interests=repeated_interest_rows,
                )
                pending_clicks.append(clicked_records)

            for clicked_records in pending_clicks:
                for record in clicked_records:
                    history.append(_history_item(record))
                    seen_interests.add(record.interest_id)

            total_rows = sum(
                split_stats.counters["impressions"]
                for split_stats in statistics.values()
            )
            if total_rows and total_rows % 250_000 == 0:
                LOGGER.info(
                    "Wrote %s dynamic impression rows",
                    f"{total_rows:,}",
                )
    finally:
        connection.close()
        for writer in writers.values():
            writer.close()

    counts = {
        split: int(statistics[split].counters["impressions"])
        for split in SPLITS
    }
    for split in SPLITS:
        if writers[split].row_count != counts[split]:
            raise ValueError(
                f"{split}: wrote {writers[split].row_count} rows but "
                f"counted {counts[split]}"
            )
    return (
        {split: statistics[split].as_dict() for split in SPLITS},
        counts,
    )


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite."
        )
    backup = output_dir.with_name(output_dir.name + ".old")
    if output_dir.exists() and overwrite and backup.exists():
        raise FileExistsError(
            f"Cannot replace {output_dir}: backup exists at {backup}"
        )


def build_dynamic_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    batch_size: int = 2_000,
    staging_batch_size: int = 20_000,
    overwrite: bool = False,
) -> DynamicBuildResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if staging_batch_size <= 0:
        raise ValueError("staging_batch_size must be positive")

    for split in SPLITS:
        for filename in ("news.tsv", "behaviors.tsv"):
            path = input_dir / split / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_output_available(output_dir, overwrite)

    LOGGER.info("Loading news metadata and interest vocabulary")
    catalog = load_news_catalog(input_dir)

    with tempfile.TemporaryDirectory(
        prefix=".dynamic-new-interest-output-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        database_path = staging / "events.sqlite3"
        LOGGER.info("Staging Train and Dev events")
        staging_stats = _stage_events(
            input_dir,
            database_path,
            catalog,
            staging_batch_size=staging_batch_size,
        )
        LOGGER.info("Replaying per-user click timelines")
        split_stats, counts = _write_dynamic_rows(
            database_path,
            staging,
            catalog,
            batch_size=batch_size,
        )
        _write_vocabulary(
            staging / "interest_vocabulary.parquet",
            catalog.interests,
        )

        statistics = {
            "rule_version": RULE_VERSION,
            "timeline_policy": (
                "Per-user Train then online Dev replay; all events at the "
                "same timestamp are labeled before any clicks from that "
                "timestamp update history."
            ),
            "staging": staging_stats,
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
                "A clicked category-subcategory pair is novel when it has "
                "not appeared in the user's initial MIND history or any "
                "strictly earlier positive click. Train state carries into "
                "online dynamic Dev."
            ),
            "row_granularity": "one row per impression",
            "history_state": "all prior positive click events",
            "same_timestamp_policy": "label_all_then_update_all",
            "dev_policy": "online_dynamic",
            "all_behavior_impressions_included": True,
            "unclicked_news_used": False,
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
            "supervised_splits": list(SPLITS),
            "test_exclusion_reason": (
                "MIND test impressions do not contain -1/-0 click labels."
            ),
            "artifact_sha256": artifact_hashes,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        database_path.unlink()
        if output_dir.exists() and overwrite:
            backup = output_dir.with_name(output_dir.name + ".old")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        elif output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists: {output_dir}"
            )
        else:
            os.replace(staging, output_dir)

    return DynamicBuildResult(
        output_dir=output_dir,
        train_impressions=counts["train"],
        dev_impressions=counts["dev"],
        train_new_interest_impressions=int(
            split_stats["train"]["new_interest_impressions"]
        ),
        dev_new_interest_impressions=int(
            split_stats["dev"]["new_interest_impressions"]
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build impression-level natural-distribution new-interest data "
            "with per-user dynamic Train and online Dev click history."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/MINDlarge_reclassified"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/new_interest_dynamic_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--staging-batch-size", type=int, default=20_000)
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
    result = build_dynamic_dataset(
        args.input_dir,
        args.output_dir,
        batch_size=args.batch_size,
        staging_batch_size=args.staging_batch_size,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Wrote %s Train and %s online Dev impressions to %s "
        "(new-interest impressions: %s Train, %s Dev)",
        f"{result.train_impressions:,}",
        f"{result.dev_impressions:,}",
        result.output_dir,
        f"{result.train_new_interest_impressions:,}",
        f"{result.dev_new_interest_impressions:,}",
    )


if __name__ == "__main__":
    main()
