from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from itertools import groupby
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.mind_io import NewsCatalog, iter_behavior_events, load_news_catalog
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "future-interest-user-benchmark-v1"
DEFAULT_INPUT_DIR = Path("data/raw/MINDlarge_reclassified")
DEFAULT_OUTPUT_DIR = Path("data/processed/future_novel_interest_users_v1")
DEFAULT_SEED = 20260812
DEFAULT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
OUTPUT_FILES = (
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "interest_vocabulary.parquet",
    "split_manifest.parquet",
    "statistics.json",
    "metadata.json",
)


BENCHMARK_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("profile_hash", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("profile_group_size", pa.int32(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("history_news_ids", pa.list_(pa.string()), nullable=False),
        pa.field("history_titles", pa.list_(pa.string()), nullable=False),
        pa.field("history_subcategories", pa.list_(pa.string()), nullable=False),
        pa.field(
            "history_unique_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("future_impression_count", pa.int32(), nullable=False),
        pa.field(
            "future_clicked_news_ids",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_clicked_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_novel_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_exposed_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_exposed_novel_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_exposed_unclicked_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("has_future_novel_interest", pa.bool_(), nullable=False),
    ]
)

VOCABULARY_SCHEMA = pa.schema(
    [
        pa.field("interest_id", pa.int16(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("profile_hash", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("profile_group_size", pa.int32(), nullable=False),
        pa.field("user_ids", pa.list_(pa.string()), nullable=False),
    ]
)


@dataclass(frozen=True)
class BenchmarkBuildResult:
    output_dir: Path
    users: int
    profile_groups: int
    split_counts: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Distribution:
    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.total = 0
        self.total_value = 0

    def add(self, value: int) -> None:
        self.counts[value] += 1
        self.total += 1
        self.total_value += value

    def as_dict(self) -> dict[str, object]:
        if not self.total:
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
        ordered = sorted(self.counts)

        def percentile(fraction: float) -> int:
            rank = max(1, int(self.total * fraction + 0.999999))
            cumulative = 0
            for value in ordered:
                cumulative += self.counts[value]
                if cumulative >= rank:
                    return value
            raise AssertionError("unreachable percentile")

        return {
            "count": self.total,
            "min": ordered[0],
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": ordered[-1],
            "mean": self.total_value / self.total,
            "count_distribution": {
                str(value): self.counts[value] for value in ordered
            },
        }


class _SplitStatistics:
    def __init__(self, vocabulary: tuple[str, ...]) -> None:
        self.vocabulary = vocabulary
        self.users = 0
        self.history_lengths = _Distribution()
        self.clicked_counts = _Distribution()
        self.novel_counts = _Distribution()
        self.exposure_counts = _Distribution()
        self.positive_support: Counter[str] = Counter()
        self.exposure_support: Counter[str] = Counter()
        self.novel_exposure_support: Counter[str] = Counter()
        self.unclicked_support: Counter[str] = Counter()

    def add(
        self,
        *,
        history_length: int,
        clicked_count: int,
        novel: list[str],
        exposed: list[str],
        exposed_novel: list[str],
        exposed_unclicked: list[str],
    ) -> None:
        self.users += 1
        self.history_lengths.add(history_length)
        self.clicked_counts.add(clicked_count)
        self.novel_counts.add(len(novel))
        self.exposure_counts.add(len(exposed))
        self.positive_support.update(novel)
        self.exposure_support.update(exposed)
        self.novel_exposure_support.update(exposed_novel)
        self.unclicked_support.update(exposed_unclicked)

    def as_dict(self) -> dict[str, object]:
        users = self.users
        return {
            "users": users,
            "history_length": self.history_lengths.as_dict(),
            "future_clicked_target_count": self.clicked_counts.as_dict(),
            "future_novel_target_count": self.novel_counts.as_dict(),
            "future_exposed_subcategory_count": self.exposure_counts.as_dict(),
            "per_label_distinct_user_positive_support": {
                label: int(self.positive_support[label])
                for label in self.vocabulary
            },
            "per_label_distinct_user_exposure": {
                label: int(self.exposure_support[label])
                for label in self.vocabulary
            },
            "per_label_distinct_user_novel_exposure": {
                label: int(self.novel_exposure_support[label])
                for label in self.vocabulary
            },
            "per_label_distinct_user_unclicked_exposure": {
                label: int(self.unclicked_support[label])
                for label in self.vocabulary
            },
            "per_label_exposure_rates": {
                label: (
                    self.novel_exposure_support[label] / users
                    if users
                    else 0.0
                )
                for label in self.vocabulary
            },
            "per_label_positive_rates": {
                label: (
                    self.positive_support[label] / users
                    if users
                    else 0.0
                )
                for label in self.vocabulary
            },
            "per_label_all_exposure_rates": {
                label: (
                    self.exposure_support[label] / users
                    if users
                    else 0.0
                )
                for label in self.vocabulary
            },
        }


def _load_config(config_path_or_config: Path | str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(config_path_or_config, Mapping):
        config = dict(config_path_or_config)
    else:
        path = Path(config_path_or_config)
        config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a JSON object or mapping")
    return config


def _path_option(
    config: Mapping[str, object],
    names: tuple[str, ...],
    default: Path,
) -> Path:
    for name in names:
        value = config.get(name)
        if value is not None:
            return Path(str(value))
    return default


def _split_for_profile(
    profile_hash: str,
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> str:
    digest = hashlib.sha256(f"{seed}:{profile_hash}".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    train_end = float(ratios["train"])
    validation_end = train_end + float(ratios["validation"])
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def _profile_hash(history_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        history_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _configure_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        PRAGMA locking_mode = EXCLUSIVE;
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            history_digest BLOB NOT NULL,
            profile_hash TEXT NOT NULL,
            history_ids_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE events (
            event_row INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            split_rank INTEGER NOT NULL,
            source_order INTEGER NOT NULL,
            impression_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            timestamp_text TEXT NOT NULL,
            candidates_json TEXT NOT NULL,
            UNIQUE (split, impression_id)
        );
        CREATE INDEX events_by_user ON events (
            user_id, timestamp_text, split_rank, source_order
        );
        CREATE INDEX users_by_profile ON users (profile_hash, user_id);
        CREATE TABLE profile_groups (
            profile_hash TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            profile_group_size INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _stage(
    input_dir: Path,
    database_path: Path,
    catalog: NewsCatalog,
    *,
    batch_size: int,
) -> dict[str, object]:
    connection = sqlite3.connect(database_path)
    _configure_database(connection)
    history_digests: dict[str, bytes] = {}
    user_rows: list[tuple[str, bytes, str, str]] = []
    event_rows: list[tuple[object, ...]] = []
    split_counts: Counter[str] = Counter()
    invariant_checks = 0

    def flush_users() -> None:
        if not user_rows:
            return
        connection.executemany(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            user_rows,
        )
        user_rows.clear()
        connection.commit()

    def flush_events() -> None:
        if not event_rows:
            return
        connection.executemany(
            """
            INSERT INTO events (
                split, split_rank, source_order, impression_id,
                user_id, timestamp_text, candidates_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )
        event_rows.clear()
        connection.commit()

    try:
        for split_rank, split in enumerate(("train", "dev")):
            for source_order, event in enumerate(
                iter_behavior_events(input_dir, (split,), catalog)
            ):
                invariant_checks += 1
                history_ids = tuple(event.history_news_ids)
                history_digest = hashlib.sha256(
                    event.history_text.encode("utf-8")
                ).digest()
                previous = history_digests.get(event.user_id)
                if previous is None:
                    history_digests[event.user_id] = history_digest
                    user_rows.append(
                        (
                            event.user_id,
                            history_digest,
                            _profile_hash(history_ids),
                            json.dumps(history_ids, ensure_ascii=False),
                        )
                    )
                    if len(user_rows) >= batch_size:
                        flush_users()
                elif previous != history_digest:
                    raise ValueError(
                        f"User {event.user_id} has inconsistent static history "
                        f"at {split} impression {event.impression_id}"
                    )
                candidates = [
                    {"news_id": item.news_id, "clicked": item.clicked}
                    for item in event.impression_candidates
                ]
                event_rows.append(
                    (
                        split,
                        split_rank,
                        source_order,
                        event.impression_id,
                        event.user_id,
                        event.timestamp.isoformat(sep=" "),
                        json.dumps(candidates, separators=(",", ":")),
                    )
                )
                split_counts[split] += 1
                if len(event_rows) >= batch_size:
                    flush_events()
        flush_users()
        flush_events()
    except sqlite3.IntegrityError as error:
        raise ValueError(f"Duplicate raw impression ID: {error}") from error
    finally:
        connection.close()
    return {
        "users": len(history_digests),
        "split_behavior_rows": {
            split: int(split_counts[split]) for split in ("train", "dev")
        },
        "history_invariant_checks": invariant_checks,
        "history_invariant_violations": 0,
    }


def _unique_in_vocab(values: set[str], vocabulary: tuple[str, ...]) -> list[str]:
    return [label for label in vocabulary if label in values]


def _assign_profile_groups(
    database_path: Path,
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> int:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DELETE FROM profile_groups")
        count = 0
        group_rows: list[tuple[str, str, int]] = []
        for row in connection.execute(
            """
            SELECT profile_hash, COUNT(*) AS profile_group_size
            FROM users
            GROUP BY profile_hash
            ORDER BY profile_hash
            """
        ):
            profile_hash = str(row[0])
            group_rows.append(
                (
                    profile_hash,
                    _split_for_profile(
                        profile_hash,
                        seed=seed,
                        ratios=ratios,
                    ),
                    int(row[1]),
                )
            )
            count += 1
            if len(group_rows) >= 10_000:
                connection.executemany(
                    "INSERT INTO profile_groups VALUES (?, ?, ?)",
                    group_rows,
                )
                group_rows.clear()
                connection.commit()
        if group_rows:
            connection.executemany(
                "INSERT INTO profile_groups VALUES (?, ?, ?)",
                group_rows,
            )
        connection.commit()
        return count
    finally:
        connection.close()


class _BatchedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int) -> None:
        self._rows: list[dict[str, object]] = []
        self._batch_size = batch_size
        self._schema = schema
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
        self._writer.write_table(
            pa.Table.from_pylist(self._rows, schema=self._schema),
            row_group_size=self._batch_size,
        )
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _build_user_row(
    user_row: sqlite3.Row,
    event_rows: Iterable[sqlite3.Row],
    catalog: NewsCatalog,
    vocabulary: tuple[str, ...],
) -> dict[str, object]:
    history_ids = tuple(json.loads(str(user_row["history_ids_json"])))
    history_records = [catalog.news[news_id] for news_id in history_ids]
    history_subcategories = [record.subcategory for record in history_records]
    history_set = set(history_subcategories)
    clicked_news_ids: list[str] = []
    clicked_subcategories: list[str] = []
    exposed: set[str] = set()
    clicked: set[str] = set()
    impression_count = 0
    for event in event_rows:
        impression_count += 1
        for candidate in json.loads(str(event["candidates_json"])):
            record = catalog.news[str(candidate["news_id"])]
            exposed.add(record.subcategory)
            if bool(candidate["clicked"]):
                clicked_news_ids.append(record.news_id)
                clicked_subcategories.append(record.subcategory)
                clicked.add(record.subcategory)

    novel = _unique_in_vocab(clicked - history_set, vocabulary)
    exposed_labels = _unique_in_vocab(exposed, vocabulary)
    exposed_novel = _unique_in_vocab(exposed - history_set, vocabulary)
    # A topic with any click in the future window is never a weak negative.
    exposed_unclicked = _unique_in_vocab(
        (exposed - history_set) - clicked,
        vocabulary,
    )
    return {
        "user_id": str(user_row["user_id"]),
        "profile_hash": str(user_row["profile_hash"]),
        "split": str(user_row["split"]),
        "profile_group_size": int(user_row["profile_group_size"]),
        "history_length": len(history_ids),
        "history_news_ids": list(history_ids),
        "history_titles": [record.title for record in history_records],
        "history_subcategories": history_subcategories,
        "history_unique_subcategories": _unique_in_vocab(
            history_set,
            vocabulary,
        ),
        "future_impression_count": impression_count,
        "future_clicked_news_ids": clicked_news_ids,
        "future_clicked_subcategories": clicked_subcategories,
        "future_novel_subcategories": novel,
        "future_exposed_subcategories": exposed_labels,
        "future_exposed_novel_subcategories": exposed_novel,
        "future_exposed_unclicked_subcategories": exposed_unclicked,
        "has_future_novel_interest": bool(novel),
    }


def _write_streaming_outputs(
    database_path: Path,
    staging: Path,
    catalog: NewsCatalog,
    *,
    batch_size: int,
    invariant_checks: int,
) -> tuple[dict[str, object], dict[str, int], int]:
    vocabulary = tuple(subcategory for _, subcategory in catalog.interests)
    writers = {
        split: _BatchedParquetWriter(
            staging / f"{split}.parquet",
            BENCHMARK_SCHEMA,
            batch_size,
        )
        for split in ("train", "validation", "test")
    }
    manifest_writer = _BatchedParquetWriter(
        staging / "split_manifest.parquet",
        MANIFEST_SCHEMA,
        batch_size,
    )
    split_stats = {
        split: _SplitStatistics(vocabulary)
        for split in ("train", "validation", "test")
    }
    global_stats = _SplitStatistics(vocabulary)
    group_size_distribution = _Distribution()
    split_group_sizes = {
        split: _Distribution()
        for split in ("train", "validation", "test")
    }
    users_connection = sqlite3.connect(database_path)
    users_connection.row_factory = sqlite3.Row
    events_connection = sqlite3.connect(database_path)
    events_connection.row_factory = sqlite3.Row
    profile_groups = 0
    try:
        for group in users_connection.execute(
            """
            SELECT profile_hash, split, profile_group_size
            FROM profile_groups
            ORDER BY profile_hash
            """
        ):
            profile_hash = str(group["profile_hash"])
            split = str(group["split"])
            group_size = int(group["profile_group_size"])
            user_ids = [
                str(row[0])
                for row in users_connection.execute(
                    """
                    SELECT user_id FROM users
                    WHERE profile_hash = ?
                    ORDER BY user_id
                    """,
                    (profile_hash,),
                )
            ]
            if len(user_ids) != group_size:
                raise ValueError(
                    f"Profile group {profile_hash} has incorrect user count"
                )
            manifest_writer.append(
                {
                    "profile_hash": profile_hash,
                    "split": split,
                    "profile_group_size": group_size,
                    "user_ids": user_ids,
                }
            )
            group_size_distribution.add(group_size)
            split_group_sizes[split].add(group_size)
            profile_groups += 1
    finally:
        manifest_writer.close()

    try:
        users_cursor = users_connection.execute(
            """
            SELECT u.user_id, u.profile_hash, u.history_ids_json,
                   pg.split, pg.profile_group_size
            FROM users AS u
            JOIN profile_groups AS pg
              ON pg.profile_hash = u.profile_hash
            ORDER BY u.user_id
            """
        )
        event_groups = iter(
            groupby(
                events_connection.execute(
                    """
                    SELECT user_id, candidates_json
                    FROM events
                    ORDER BY user_id, timestamp_text, split_rank,
                             source_order, event_row
                    """
                ),
                key=lambda row: str(row["user_id"]),
            )
        )
        current_event_group = next(event_groups, None)
        for user in users_cursor:
            user_id = str(user["user_id"])
            if current_event_group is None:
                raise ValueError(f"Missing events for user {user_id}")
            event_user_id, event_rows = current_event_group
            if str(event_user_id) != user_id:
                raise ValueError(
                    f"Event/user stream mismatch: {event_user_id} vs {user_id}"
                )
            split = str(user["split"])
            output_row = _build_user_row(
                user,
                event_rows,
                catalog,
                vocabulary,
            )
            current_event_group = next(event_groups, None)
            writers[split].append(output_row)
            split_stats[split].add(
                history_length=int(output_row["history_length"]),
                clicked_count=len(output_row["future_clicked_news_ids"]),
                novel=output_row["future_novel_subcategories"],
                exposed=output_row["future_exposed_subcategories"],
                exposed_novel=output_row[
                    "future_exposed_novel_subcategories"
                ],
                exposed_unclicked=output_row[
                    "future_exposed_unclicked_subcategories"
                ],
            )
            global_stats.add(
                history_length=int(output_row["history_length"]),
                clicked_count=len(output_row["future_clicked_news_ids"]),
                novel=output_row["future_novel_subcategories"],
                exposed=output_row["future_exposed_subcategories"],
                exposed_novel=output_row[
                    "future_exposed_novel_subcategories"
                ],
                exposed_unclicked=output_row[
                    "future_exposed_unclicked_subcategories"
                ],
            )
        if current_event_group is not None:
            raise ValueError(
                f"Events exist for unknown user {current_event_group[0]}"
            )
    finally:
        for writer in writers.values():
            writer.close()
        users_connection.close()
        events_connection.close()

    split_counts = {
        split: split_stats[split].users
        for split in ("train", "validation", "test")
    }
    global_dict = global_stats.as_dict()
    statistics = {
        "rule_version": RULE_VERSION,
        "users": global_stats.users,
        "profile_groups": profile_groups,
        "split_counts": split_counts,
        "profile_group_size": group_size_distribution.as_dict(),
        "profile_group_sizes": group_size_distribution.as_dict(),
        "per_split_profile_group_sizes": {
            split: split_group_sizes[split].as_dict()
            for split in ("train", "validation", "test")
        },
        "history_length": global_dict["history_length"],
        "future_clicked_target_count": global_dict[
            "future_clicked_target_count"
        ],
        "future_novel_target_count": global_dict[
            "future_novel_target_count"
        ],
        "target_count": global_dict["future_novel_target_count"],
        "future_exposed_subcategory_count": global_dict[
            "future_exposed_subcategory_count"
        ],
        "per_label_distinct_user_positive_support": global_dict[
            "per_label_distinct_user_positive_support"
        ],
        "per_label_distinct_user_exposure": global_dict[
            "per_label_distinct_user_exposure"
        ],
        "per_label_distinct_user_novel_exposure": global_dict[
            "per_label_distinct_user_novel_exposure"
        ],
        "per_label_distinct_user_unclicked_exposure": global_dict[
            "per_label_distinct_user_unclicked_exposure"
        ],
        "per_label_exposure_rates": global_dict["per_label_exposure_rates"],
        "per_label_positive_rates": global_dict["per_label_positive_rates"],
        "exposure_rates": global_dict["per_label_exposure_rates"],
        "per_split": {
            split: split_stats[split].as_dict()
            for split in ("train", "validation", "test")
        },
        "train_only_audit": split_stats["train"].as_dict(),
        "invariant_counts": {
            "history_invariant_checks": invariant_checks,
            "history_invariant_violations": 0,
            "duplicate_user_ids": 0,
            "rows_written": global_stats.users,
        },
    }
    statistics["invariants"] = statistics["invariant_counts"]
    return statistics, split_counts, profile_groups


def _write_table(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    if not overwrite and output_dir.exists():
        raise FileExistsError(
            f"Benchmark outputs already exist in {output_dir}; use overwrite=True"
        )


def build_future_interest_benchmark(
    config_path_or_config: Path | str | Mapping[str, object],
) -> BenchmarkBuildResult:
    config = _load_config(config_path_or_config)
    input_dir = _path_option(
        config,
        ("input_dir", "source_dir", "raw_dir", "raw_mind_dir"),
        DEFAULT_INPUT_DIR,
    )
    output_dir = _path_option(config, ("output_dir",), DEFAULT_OUTPUT_DIR)
    seed = int(config.get("seed", config.get("split_seed", DEFAULT_SEED)))
    raw_ratios = config.get("split_ratios", DEFAULT_RATIOS)
    if not isinstance(raw_ratios, Mapping):
        raise ValueError("split_ratios must be an object")
    ratios = {split: float(raw_ratios.get(split, 0.0)) for split in DEFAULT_RATIOS}
    if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split_ratios must be non-negative and sum to 1")
    staging_batch_size = int(config.get("staging_batch_size", 50_000))
    if staging_batch_size <= 0:
        raise ValueError("staging_batch_size must be positive")
    output_batch_size = int(config.get("output_batch_size", 5_000))
    if output_batch_size <= 0:
        raise ValueError("output_batch_size must be positive")
    overwrite = bool(config.get("overwrite", False))
    _ensure_output_available(output_dir, overwrite)
    catalog = load_news_catalog(input_dir)
    expected_count = config.get("expected_interest_count")
    if expected_count is not None and len(catalog.interests) != int(expected_count):
        raise ValueError(
            f"Expected {int(expected_count)} catalog interests, found "
            f"{len(catalog.interests)}"
        )

    source_paths = [
        input_dir / split / filename
        for split in ("train", "dev")
        for filename in ("news.tsv", "behaviors.tsv")
    ]
    source_sha256 = {portable_path(path): _sha256(path) for path in source_paths}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".future-interest-benchmark-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        database_path = staging / "staging.sqlite"
        staging_stats = _stage(
            input_dir,
            database_path,
            catalog,
            batch_size=staging_batch_size,
        )
        _assign_profile_groups(
            database_path,
            seed=seed,
            ratios=ratios,
        )
        statistics, split_counts, profile_groups = _write_streaming_outputs(
            database_path,
            staging,
            catalog,
            batch_size=output_batch_size,
            invariant_checks=int(staging_stats["history_invariant_checks"]),
        )
        statistics["staging"] = staging_stats
        statistics["catalog"] = {
            "interest_count": len(catalog.interests),
            "news_count": len(catalog.news),
            "news_rows": catalog.news_rows,
        }
        _write_table(
            staging / "interest_vocabulary.parquet",
            [
                {
                    "interest_id": interest_id,
                    "category": category,
                    "subcategory": subcategory,
                }
                for interest_id, (category, subcategory) in enumerate(
                    catalog.interests
                )
            ],
            VOCABULARY_SCHEMA,
        )
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        artifact_sha256 = {
            name: _sha256(staging / name)
            for name in OUTPUT_FILES
            if name != "metadata.json"
        }
        metadata = {
            "rule_version": RULE_VERSION,
            "definition": (
                "One row per raw-MIND user. Train and Dev exposures are "
                "combined into one temporally ordered future labeled window; "
                "users are split by exact ordered-history profile."
            ),
            "input_dir": portable_path(input_dir),
            "output_dir": portable_path(output_dir),
            "source_sha256": source_sha256,
            "artifact_sha256": artifact_sha256,
            "split_seed": seed,
            "split_ratios": ratios,
            "expected_interest_count": len(catalog.interests),
            "row_granularity": "one row per user",
            "future_window": ["train", "dev"],
            "profile_hash": "sha256(canonical ordered history news ID JSON)",
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        database_path.unlink()

        if output_dir.exists() and overwrite:
            backup = output_dir.with_name(output_dir.name + ".old")
            if backup.exists():
                raise FileExistsError(f"Cannot replace existing output {output_dir}")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            # The backup is deliberately removed only after the new directory
            # has been installed successfully.
            import shutil

            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)

    return BenchmarkBuildResult(
        output_dir=output_dir,
        users=sum(split_counts.values()),
        profile_groups=profile_groups,
        split_counts={
            split: int(split_counts[split])
            for split in ("train", "validation", "test")
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the future novel-interest user benchmark."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/future_interest/benchmark_v1.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    if args.overwrite:
        config["overwrite"] = True
    result = build_future_interest_benchmark(config)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "users": result.users,
                "profile_groups": result.profile_groups,
                "split_counts": result.split_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
