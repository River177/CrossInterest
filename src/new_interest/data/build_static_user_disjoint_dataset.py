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
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import DATASET_SCHEMA, VOCABULARY_SCHEMA
from new_interest.data.mind_io import (
    BehaviorEvent,
    NewsCatalog,
    iter_behavior_events,
    load_news_catalog,
)
from new_interest.infrastructure.paths import portable_path


LOGGER = logging.getLogger(__name__)

RULE_VERSION = "static-user-disjoint-single-train-target-v1"
DEFAULT_INPUT_DIR = Path("data/raw/MINDlarge_reclassified")
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_v1"
)
DEFAULT_SELECTION_SEED = 20260806
OUTPUT_FILES = (
    "train.parquet",
    "dev.parquet",
    "interest_vocabulary.parquet",
    "train_selection_manifest.parquet",
    "statistics.json",
    "metadata.json",
)


TRAIN_SELECTION_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("selected_sample_id", pa.string(), nullable=False),
        pa.field("selected_impression_id", pa.string(), nullable=False),
        pa.field("selected_target_category", pa.string(), nullable=False),
        pa.field("selected_target_subcategory", pa.string(), nullable=False),
        pa.field("target_selection_priority", pa.string(), nullable=False),
        pa.field("eligible_candidate_count", pa.int32(), nullable=False),
        pa.field("distinct_interest_count", pa.int16(), nullable=False),
        pa.field("eligible_impression_count", pa.int32(), nullable=False),
        pa.field("selection_seed", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_samples: int
    train_users_with_multiple_candidates: int
    dev_samples: int
    dev_users: int


class _BatchedParquetWriter:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_priority(seed: int, sample_id: str) -> str:
    payload = f"{seed}:train-user-target:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _distribution_summary(
    counts: Counter[int],
) -> dict[str, float | int | None | dict[str, int]]:
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
            "count_distribution": {},
        }

    def percentile(fraction: float) -> int:
        rank = max(1, int(total * fraction + 0.999999))
        cumulative = 0
        for value in sorted(counts):
            cumulative += counts[value]
            if cumulative >= rank:
                return value
        raise AssertionError("unreachable percentile")

    return {
        "count": total,
        "min": min(counts),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(counts),
        "mean": (
            sum(value * count for value, count in counts.items()) / total
        ),
        "count_distribution": {
            str(value): count for value, count in sorted(counts.items())
        },
    }


def _history_rows(
    history_news_ids: tuple[str, ...] | list[str],
    catalog: NewsCatalog,
) -> tuple[list[dict[str, str]], set[int]]:
    history: list[dict[str, str]] = []
    interests: set[int] = set()
    for news_id in history_news_ids:
        record = catalog.news[news_id]
        interests.add(record.interest_id)
        history.append(
            {
                "news_id": record.news_id,
                "category": record.category,
                "subcategory": record.subcategory,
                "title": record.title,
            }
        )
    return history, interests


def _eligible_targets(
    event: BehaviorEvent,
    catalog: NewsCatalog,
) -> tuple[list[tuple[int, list[str]]], int]:
    history_interests = {
        catalog.news[news_id].interest_id
        for news_id in event.history_news_ids
    }
    positive_news: dict[int, list[str]] = {}
    for news_id in event.clicked_news_ids:
        record = catalog.news[news_id]
        positive_news.setdefault(record.interest_id, []).append(news_id)
    targets = [
        (interest_id, news_ids)
        for interest_id, news_ids in positive_news.items()
        if interest_id not in history_interests
    ]
    return targets, len(positive_news)


def _dataset_row(
    *,
    split: str,
    impression_id: str,
    user_id: str,
    timestamp: datetime,
    history: list[dict[str, str]],
    interest_id: int,
    target_news_ids: list[str],
    catalog: NewsCatalog,
) -> dict[str, object]:
    category, subcategory = catalog.interests[interest_id]
    return {
        "sample_id": f"{split}:{impression_id}:{interest_id}",
        "split": split,
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": timestamp,
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


def _configure_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        PRAGMA locking_mode = EXCLUSIVE;

        CREATE TABLE train_users (
            user_id TEXT PRIMARY KEY,
            history_ids_json TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE train_selections (
            user_id TEXT PRIMARY KEY,
            selection_key TEXT NOT NULL,
            priority TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            impression_id TEXT NOT NULL,
            timestamp_text TEXT NOT NULL,
            interest_id INTEGER NOT NULL,
            target_news_ids_json TEXT NOT NULL,
            eligible_candidate_count INTEGER NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE train_candidate_interests (
            user_id TEXT NOT NULL,
            interest_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, interest_id)
        ) WITHOUT ROWID;

        CREATE TABLE train_candidate_impressions (
            user_id TEXT NOT NULL,
            impression_id TEXT NOT NULL,
            PRIMARY KEY (user_id, impression_id)
        ) WITHOUT ROWID;
        """
    )


_SELECTION_UPSERT = """
    INSERT INTO train_selections (
        user_id, selection_key, priority, sample_id, impression_id,
        timestamp_text, interest_id, target_news_ids_json,
        eligible_candidate_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
    ON CONFLICT(user_id) DO UPDATE SET
        eligible_candidate_count =
            train_selections.eligible_candidate_count + 1,
        selection_key = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.selection_key
            ELSE train_selections.selection_key
        END,
        priority = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.priority
            ELSE train_selections.priority
        END,
        sample_id = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.sample_id
            ELSE train_selections.sample_id
        END,
        impression_id = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.impression_id
            ELSE train_selections.impression_id
        END,
        timestamp_text = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.timestamp_text
            ELSE train_selections.timestamp_text
        END,
        interest_id = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.interest_id
            ELSE train_selections.interest_id
        END,
        target_news_ids_json = CASE
            WHEN excluded.selection_key < train_selections.selection_key
            THEN excluded.target_news_ids_json
            ELSE train_selections.target_news_ids_json
        END
"""


def _stage_train(
    input_dir: Path,
    database_path: Path,
    catalog: NewsCatalog,
    *,
    selection_seed: int,
    staging_batch_size: int,
) -> tuple[dict[str, bytes], dict[str, int]]:
    connection = sqlite3.connect(database_path)
    _configure_database(connection)
    history_digests: dict[str, bytes] = {}
    user_rows: list[tuple[str, str]] = []
    selection_rows: list[tuple[object, ...]] = []
    interest_rows: list[tuple[str, int]] = []
    impression_rows: list[tuple[str, str]] = []
    counters: Counter[str] = Counter()

    def flush() -> None:
        if user_rows:
            connection.executemany(
                "INSERT INTO train_users VALUES (?, ?)",
                user_rows,
            )
            user_rows.clear()
        if selection_rows:
            connection.executemany(_SELECTION_UPSERT, selection_rows)
            selection_rows.clear()
        if interest_rows:
            connection.executemany(
                "INSERT OR IGNORE INTO train_candidate_interests VALUES (?, ?)",
                interest_rows,
            )
            interest_rows.clear()
        if impression_rows:
            connection.executemany(
                "INSERT OR IGNORE INTO train_candidate_impressions VALUES (?, ?)",
                impression_rows,
            )
            impression_rows.clear()
        connection.commit()

    try:
        for event in iter_behavior_events(input_dir, ("train",), catalog):
            counters["behavior_rows"] += 1
            digest = hashlib.sha256(event.history_text.encode("utf-8")).digest()
            previous = history_digests.get(event.user_id)
            if previous is None:
                history_digests[event.user_id] = digest
                user_rows.append(
                    (
                        event.user_id,
                        json.dumps(
                            event.history_news_ids,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                )
            elif previous != digest:
                raise ValueError(
                    f"User {event.user_id} has inconsistent static history "
                    f"at train impression {event.impression_id}"
                )

            targets, positive_interest_count = _eligible_targets(
                event,
                catalog,
            )
            counters["distinct_positive_interests"] += positive_interest_count
            if targets:
                counters["eligible_behavior_rows"] += 1
            for interest_id, target_news_ids in targets:
                sample_id = f"train:{event.impression_id}:{interest_id}"
                priority = _selection_priority(selection_seed, sample_id)
                selection_rows.append(
                    (
                        event.user_id,
                        f"{priority}:{sample_id}",
                        priority,
                        sample_id,
                        event.impression_id,
                        event.timestamp.isoformat(sep=" "),
                        interest_id,
                        json.dumps(
                            target_news_ids,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                )
                interest_rows.append((event.user_id, interest_id))
                impression_rows.append((event.user_id, event.impression_id))
                counters["eligible_candidate_rows"] += 1

            if max(len(user_rows), len(selection_rows)) >= staging_batch_size:
                flush()
            if counters["behavior_rows"] % 250_000 == 0:
                LOGGER.info(
                    "Staged %s Train behavior rows",
                    f"{counters['behavior_rows']:,}",
                )
        flush()
        connection.executescript(
            """
            CREATE TABLE train_interest_counts AS
            SELECT user_id, COUNT(*) AS distinct_interest_count
            FROM train_candidate_interests
            GROUP BY user_id;
            CREATE UNIQUE INDEX train_interest_counts_by_user
            ON train_interest_counts (user_id);

            CREATE TABLE train_impression_counts AS
            SELECT user_id, COUNT(*) AS eligible_impression_count
            FROM train_candidate_impressions
            GROUP BY user_id;
            CREATE UNIQUE INDEX train_impression_counts_by_user
            ON train_impression_counts (user_id);
            """
        )
        connection.commit()
    finally:
        connection.close()
    counters["raw_users"] = len(history_digests)
    return history_digests, dict(counters)


def _train_selection_statistics(
    connection: sqlite3.Connection,
    staged: dict[str, int],
) -> tuple[dict[str, object], Counter[int]]:
    candidate_distribution = Counter(
        {
            int(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT eligible_candidate_count, COUNT(*)
                FROM train_selections
                GROUP BY eligible_candidate_count
                ORDER BY eligible_candidate_count
                """
            )
        }
    )
    selected_users = sum(candidate_distribution.values())
    multiple_candidates = sum(
        count
        for candidate_count, count in candidate_distribution.items()
        if candidate_count > 1
    )
    multiple_interests = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM train_interest_counts
            WHERE distinct_interest_count > 1
            """
        ).fetchone()[0]
    )
    multiple_impressions = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM train_impression_counts
            WHERE eligible_impression_count > 1
            """
        ).fetchone()[0]
    )
    repeated_same_interest_only = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM train_selections AS s
            JOIN train_interest_counts AS i USING (user_id)
            WHERE s.eligible_candidate_count > 1
              AND i.distinct_interest_count = 1
            """
        ).fetchone()[0]
    )
    candidates = int(staged.get("eligible_candidate_rows", 0))
    return (
        {
            "raw_behavior_rows": int(staged.get("behavior_rows", 0)),
            "raw_users": int(staged.get("raw_users", 0)),
            "eligible_behavior_rows": int(
                staged.get("eligible_behavior_rows", 0)
            ),
            "eligible_candidate_rows_before_selection": candidates,
            "eligible_users": selected_users,
            "selected_users": selected_users,
            "selected_samples": selected_users,
            "discarded_candidate_rows": candidates - selected_users,
            "users_with_multiple_candidate_rows": multiple_candidates,
            "users_with_multiple_distinct_interests": multiple_interests,
            "users_with_multiple_eligible_impressions": multiple_impressions,
            "users_with_repeated_same_interest_candidates_only": (
                repeated_same_interest_only
            ),
            "candidate_rows_per_eligible_user": _distribution_summary(
                candidate_distribution
            ),
        },
        candidate_distribution,
    )


def _write_train_outputs(
    database_path: Path,
    staging: Path,
    catalog: NewsCatalog,
    *,
    selection_seed: int,
    output_batch_size: int,
    staged_statistics: dict[str, int],
) -> dict[str, object]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    writer = _BatchedParquetWriter(
        staging / "train.parquet",
        DATASET_SCHEMA,
        output_batch_size,
    )
    manifest_writer = _BatchedParquetWriter(
        staging / "train_selection_manifest.parquet",
        TRAIN_SELECTION_SCHEMA,
        output_batch_size,
    )
    target_counts: Counter[str] = Counter()
    history_lengths: Counter[int] = Counter()
    target_news_lengths: Counter[int] = Counter()
    selected_sample_ids: set[str] = set()
    statistics, _ = _train_selection_statistics(
        connection,
        staged_statistics,
    )
    try:
        rows = connection.execute(
            """
            SELECT s.user_id, s.priority, s.sample_id, s.impression_id,
                   s.timestamp_text, s.interest_id,
                   s.target_news_ids_json, s.eligible_candidate_count,
                   u.history_ids_json, i.distinct_interest_count,
                   m.eligible_impression_count
            FROM train_selections AS s
            JOIN train_users AS u USING (user_id)
            JOIN train_interest_counts AS i USING (user_id)
            JOIN train_impression_counts AS m USING (user_id)
            ORDER BY s.user_id
            """
        )
        for row in rows:
            history_ids = list(json.loads(str(row["history_ids_json"])))
            history, history_interests = _history_rows(history_ids, catalog)
            interest_id = int(row["interest_id"])
            if interest_id in history_interests:
                raise ValueError(
                    f"Selected target appears in raw history for "
                    f"{row['sample_id']}"
                )
            target_news_ids = list(
                json.loads(str(row["target_news_ids_json"]))
            )
            output_row = _dataset_row(
                split="train",
                impression_id=str(row["impression_id"]),
                user_id=str(row["user_id"]),
                timestamp=datetime.fromisoformat(str(row["timestamp_text"])),
                history=history,
                interest_id=interest_id,
                target_news_ids=target_news_ids,
                catalog=catalog,
            )
            if output_row["sample_id"] != row["sample_id"]:
                raise ValueError(
                    f"Selected sample identity mismatch for {row['user_id']}"
                )
            sample_id = str(output_row["sample_id"])
            if sample_id in selected_sample_ids:
                raise ValueError(f"Duplicate selected sample ID {sample_id}")
            selected_sample_ids.add(sample_id)
            writer.append(output_row)
            category, subcategory = catalog.interests[interest_id]
            manifest_writer.append(
                {
                    "user_id": str(row["user_id"]),
                    "selected_sample_id": sample_id,
                    "selected_impression_id": str(row["impression_id"]),
                    "selected_target_category": category,
                    "selected_target_subcategory": subcategory,
                    "target_selection_priority": str(row["priority"]),
                    "eligible_candidate_count": int(
                        row["eligible_candidate_count"]
                    ),
                    "distinct_interest_count": int(
                        row["distinct_interest_count"]
                    ),
                    "eligible_impression_count": int(
                        row["eligible_impression_count"]
                    ),
                    "selection_seed": selection_seed,
                }
            )
            target_counts[subcategory] += 1
            history_lengths[len(history)] += 1
            target_news_lengths[len(target_news_ids)] += 1
    finally:
        writer.close()
        manifest_writer.close()
        connection.close()

    if writer.row_count != int(statistics["selected_users"]):
        raise ValueError(
            f"Wrote {writer.row_count} Train rows, expected "
            f"{statistics['selected_users']}"
        )
    if manifest_writer.row_count != writer.row_count:
        raise ValueError("Train selection manifest row count mismatch")
    statistics.update(
        {
            "target_interest_counts": dict(
                sorted(
                    target_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "history_length": _distribution_summary(history_lengths),
            "target_news_per_sample": _distribution_summary(
                target_news_lengths
            ),
        }
    )
    return statistics


def _write_dev_output(
    input_dir: Path,
    staging: Path,
    catalog: NewsCatalog,
    *,
    train_history_digests: dict[str, bytes],
    output_batch_size: int,
) -> dict[str, object]:
    writer = _BatchedParquetWriter(
        staging / "dev.parquet",
        DATASET_SCHEMA,
        output_batch_size,
    )
    dev_history_digests: dict[str, bytes] = {}
    raw_users: set[str] = set()
    overlapping_users: set[str] = set()
    eligible_users_before_exclusion: set[str] = set()
    output_users: set[str] = set()
    counters: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    history_lengths: Counter[int] = Counter()
    target_news_lengths: Counter[int] = Counter()
    try:
        for event in iter_behavior_events(input_dir, ("dev",), catalog):
            counters["raw_behavior_rows"] += 1
            raw_users.add(event.user_id)
            digest = hashlib.sha256(event.history_text.encode("utf-8")).digest()
            previous = train_history_digests.get(event.user_id)
            if previous is None:
                previous = dev_history_digests.get(event.user_id)
                if previous is None:
                    dev_history_digests[event.user_id] = digest
                elif previous != digest:
                    raise ValueError(
                        f"User {event.user_id} has inconsistent static history "
                        f"at dev impression {event.impression_id}"
                    )
            elif previous != digest:
                raise ValueError(
                    f"User {event.user_id} has inconsistent static history "
                    f"at dev impression {event.impression_id}"
                )

            targets, positive_interest_count = _eligible_targets(
                event,
                catalog,
            )
            counters["distinct_positive_interests"] += positive_interest_count
            counters["eligible_candidate_rows_before_user_exclusion"] += len(
                targets
            )
            if targets:
                counters["eligible_behavior_rows_before_user_exclusion"] += 1
                eligible_users_before_exclusion.add(event.user_id)

            if event.user_id in train_history_digests:
                overlapping_users.add(event.user_id)
                counters["excluded_behavior_rows"] += 1
                counters["excluded_candidate_rows"] += len(targets)
                continue

            counters["retained_behavior_rows"] += 1
            if targets:
                counters["output_eligible_behavior_rows"] += 1
                output_users.add(event.user_id)
                history, _ = _history_rows(event.history_news_ids, catalog)
            for interest_id, target_news_ids in targets:
                output_row = _dataset_row(
                    split="dev",
                    impression_id=event.impression_id,
                    user_id=event.user_id,
                    timestamp=event.timestamp,
                    history=history,
                    interest_id=interest_id,
                    target_news_ids=target_news_ids,
                    catalog=catalog,
                )
                writer.append(output_row)
                subcategory = catalog.interests[interest_id][1]
                target_counts[subcategory] += 1
                history_lengths[len(history)] += 1
                target_news_lengths[len(target_news_ids)] += 1
                counters["output_samples"] += 1

            if counters["raw_behavior_rows"] % 100_000 == 0:
                LOGGER.info(
                    "Processed %s Dev behavior rows",
                    f"{counters['raw_behavior_rows']:,}",
                )
    finally:
        writer.close()

    if writer.row_count != counters["output_samples"]:
        raise ValueError(
            f"Wrote {writer.row_count} Dev rows, counted "
            f"{counters['output_samples']}"
        )
    return {
        "raw_behavior_rows": int(counters["raw_behavior_rows"]),
        "raw_users": len(raw_users),
        "raw_users_overlapping_train": len(overlapping_users),
        "excluded_behavior_rows": int(counters["excluded_behavior_rows"]),
        "retained_behavior_rows": int(counters["retained_behavior_rows"]),
        "eligible_behavior_rows_before_user_exclusion": int(
            counters["eligible_behavior_rows_before_user_exclusion"]
        ),
        "eligible_users_before_user_exclusion": len(
            eligible_users_before_exclusion
        ),
        "eligible_candidate_rows_before_user_exclusion": int(
            counters["eligible_candidate_rows_before_user_exclusion"]
        ),
        "excluded_candidate_rows": int(counters["excluded_candidate_rows"]),
        "output_eligible_behavior_rows": int(
            counters["output_eligible_behavior_rows"]
        ),
        "output_users": len(output_users),
        "output_samples": int(counters["output_samples"]),
        "target_interest_counts": dict(
            sorted(
                target_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "history_length": _distribution_summary(history_lengths),
        "target_news_per_sample": _distribution_summary(
            target_news_lengths
        ),
    }


def _write_vocabulary(path: Path, catalog: NewsCatalog) -> None:
    rows = [
        {
            "interest_id": interest_id,
            "category": category,
            "subcategory": subcategory,
        }
        for interest_id, (category, subcategory) in enumerate(
            catalog.interests
        )
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=VOCABULARY_SCHEMA),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def _source_paths(input_dir: Path) -> list[Path]:
    required = [
        input_dir / split / filename
        for split in ("train", "dev")
        for filename in ("news.tsv", "behaviors.tsv")
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    optional = [input_dir / "MANIFEST.json", input_dir / "FILTERED_NEWS_IDS.json"]
    return [*required, *(path for path in optional if path.is_file())]


def _install_output(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    temporary_root: Path,
) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"Outputs already exist in {output_dir}; use --overwrite"
        )
    backup = temporary_root / "previous-output"
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def build_static_user_disjoint_dataset(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    staging_batch_size: int = 50_000,
    output_batch_size: int = 5_000,
    overwrite: bool = False,
) -> BuildResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if staging_batch_size <= 0:
        raise ValueError("staging_batch_size must be positive")
    if output_batch_size <= 0:
        raise ValueError("output_batch_size must be positive")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Outputs already exist in {output_dir}; use --overwrite"
        )
    sources = _source_paths(input_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading news catalog")
    catalog = load_news_catalog(input_dir)
    source_hashes = {portable_path(path): _sha256(path) for path in sources}

    with tempfile.TemporaryDirectory(
        prefix=".static-user-disjoint-",
        dir=str(output_dir.parent),
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        staging = temporary_root / "output"
        staging.mkdir()
        database_path = temporary_root / "staging.sqlite"

        LOGGER.info("Selecting one static-history target per Train user")
        train_history_digests, staged_train = _stage_train(
            input_dir,
            database_path,
            catalog,
            selection_seed=selection_seed,
            staging_batch_size=staging_batch_size,
        )
        train_statistics = _write_train_outputs(
            database_path,
            staging,
            catalog,
            selection_seed=selection_seed,
            output_batch_size=output_batch_size,
            staged_statistics=staged_train,
        )

        LOGGER.info("Filtering Dev against all raw Train users")
        dev_statistics = _write_dev_output(
            input_dir,
            staging,
            catalog,
            train_history_digests=train_history_digests,
            output_batch_size=output_batch_size,
        )
        _write_vocabulary(staging / "interest_vocabulary.parquet", catalog)

        statistics = {
            "rule_version": RULE_VERSION,
            "selection_seed": selection_seed,
            "train": train_statistics,
            "dev": dev_statistics,
            "invariants": {
                "train_duplicate_users": 0,
                "dev_users_overlapping_raw_train": 0,
                "history_source": "raw behavior History field",
                "impression_clicks_appended_to_history": False,
            },
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifact_hashes = {
            name: _sha256(staging / name)
            for name in OUTPUT_FILES
            if name != "metadata.json"
        }
        metadata = {
            "rule_version": RULE_VERSION,
            "definition": (
                "Train keeps one deterministic static-history novel-interest "
                "target row per eligible user. Dev removes every user that "
                "appears in raw Train and keeps all target rows for remaining "
                "users. History is the complete row-provided raw MIND History."
            ),
            "source_dir": portable_path(input_dir),
            "output_dir": portable_path(output_dir),
            "source_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
            "row_granularity": {
                "train": "one selected target row per eligible user",
                "dev": "one row per eligible impression-target membership",
            },
            "history_state": "raw_behavior_history_only",
            "full_history": True,
            "history_truncation": None,
            "impression_clicks_appended_to_history": False,
            "dev_exclusion_scope": "all user IDs in raw Train behaviors",
            "target_selection": {
                "scope": "train user",
                "candidate_unit": "impression-interest target row",
                "algorithm": "minimum sha256 priority, sample_id tie-break",
                "priority_payload": (
                    "{selection_seed}:train-user-target:{sample_id}"
                ),
                "selection_seed": selection_seed,
            },
            "dataset_schema": str(DATASET_SCHEMA),
            "train_selection_schema": str(TRAIN_SELECTION_SCHEMA),
            "news_rows": catalog.news_rows,
            "unique_news": len(catalog.news),
            "interest_count": len(catalog.interests),
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if set(path.name for path in staging.iterdir()) != set(OUTPUT_FILES):
            raise ValueError("Unexpected staged output file set")
        _install_output(
            staging,
            output_dir,
            overwrite=overwrite,
            temporary_root=temporary_root,
        )

    return BuildResult(
        output_dir=output_dir,
        train_samples=int(train_statistics["selected_samples"]),
        train_users_with_multiple_candidates=int(
            train_statistics["users_with_multiple_candidate_rows"]
        ),
        dev_samples=int(dev_statistics["output_samples"]),
        dev_users=int(dev_statistics["output_users"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a raw-history new-interest dataset with one Train target "
            "per user and no raw-Train users in Dev."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
    )
    parser.add_argument("--staging-batch-size", type=int, default=50_000)
    parser.add_argument("--output-batch-size", type=int, default=5_000)
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
    result = build_static_user_disjoint_dataset(
        args.input_dir,
        args.output_dir,
        selection_seed=args.selection_seed,
        staging_batch_size=args.staging_batch_size,
        output_batch_size=args.output_batch_size,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Wrote %s one-user Train samples (%s users had multiple candidates) "
        "and %s Dev samples from %s unseen users to %s",
        f"{result.train_samples:,}",
        f"{result.train_users_with_multiple_candidates:,}",
        f"{result.dev_samples:,}",
        f"{result.dev_users:,}",
        result.output_dir,
    )


if __name__ == "__main__":
    main()
