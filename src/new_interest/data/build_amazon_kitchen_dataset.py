from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "amazon-kitchen-contract-q1000-v1"
DEFAULT_RAW_ROOT = Path("data/raw/amazon_reviews_2023_home_and_kitchen_contract")
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/amazon_home_and_kitchen_contract_q1000_v1"
)
RATINGS_RELATIVE = Path(
    "benchmark/0core/rating_only/Home_and_Kitchen.csv"
)
METADATA_RELATIVE = Path(
    "raw/meta_categories/meta_Home_and_Kitchen.jsonl"
)
SPLITS = ("train", "dev")
CANONICAL_LEVEL1 = frozenset(
    {
        "Bath",
        "Bedding",
        "Event & Party Supplies",
        "Furniture",
        "Heating, Cooling & Air Quality",
        "Home Décor Products",
        "Irons & Steamers",
        "Kitchen & Dining",
        "Seasonal Décor",
        "Storage & Organization",
        "Vacuums & Floor Care",
        "Wall Art",
    }
)
NON_TAXONOMY_TERMS = (
    "deal",
    "markdown",
    "featured",
    "seller storefront",
    "special offer",
    "prime day",
)

HISTORY_ITEM = pa.struct(
    [
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("parent_category", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("rating", pa.float32(), nullable=False),
        pa.field("timestamp", pa.int64(), nullable=False),
    ]
)
TARGET_ITEM = pa.struct(
    [
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("parent_category", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("rating", pa.float32(), nullable=False),
        pa.field("timestamp", pa.int64(), nullable=False),
    ]
)
DATASET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_timestamp", pa.int64(), nullable=False),
        pa.field("full_history_length", pa.int32(), nullable=False),
        pa.field("full_seen_interests", pa.list_(pa.string()), nullable=False),
        pa.field("history", pa.list_(HISTORY_ITEM), nullable=False),
        pa.field("targets", pa.list_(TARGET_ITEM), nullable=False),
        pa.field("target_categories", pa.list_(pa.string()), nullable=False),
    ]
)
TOKEN_SCHEMA = pa.schema(
    [
        pa.field("category_id", pa.int32(), nullable=False),
        pa.field("category_key", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("semantic_id", pa.string(), nullable=False),
        pa.field("parent_category", pa.string(), nullable=False),
    ]
)
SELECTION_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_category", pa.string(), nullable=False),
        pa.field("sampling_priority", pa.string(), nullable=False),
        pa.field("replication_index", pa.int32(), nullable=False),
        pa.field("sampling_seed", pa.int64(), nullable=False),
    ]
)
FILTER_SCHEMA = pa.schema(
    [
        pa.field("split", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    retained_categories: int
    train_rows: int
    dev_rows: int


class ParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 2048):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self._writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.count = 0

    def append(self, row: dict[str, object]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        self._writer.write_table(pa.Table.from_pylist(self._rows, self._schema))
        self.count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hex(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def user_split(user_id: str, seed: int) -> str:
    bucket = int(stable_hex(seed, "user-split", user_id)[:16], 16) % 10_000
    return "dev" if bucket < 1_000 else "train"


def category_key(row: dict[str, object]) -> str:
    return f"{row['parent_category']} > {row['category']}"


def _prepare_database(
    ratings_path: Path,
    metadata_path: Path,
    database_path: Path,
    *,
    min_rating: float,
) -> dict[str, int]:
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE interactions ("
        "user_id TEXT NOT NULL, parent_asin TEXT NOT NULL, "
        "rating REAL NOT NULL, timestamp INTEGER NOT NULL)"
    )
    connection.execute("CREATE TABLE needed_items (parent_asin TEXT PRIMARY KEY)")
    counters: Counter[str] = Counter()
    interaction_batch: list[tuple[str, str, float, int]] = []
    item_batch: list[tuple[str]] = []
    with ratings_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"user_id", "parent_asin", "rating", "timestamp"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected ratings columns: {reader.fieldnames}")
        for row in reader:
            counters["source_rating_rows"] += 1
            if counters["source_rating_rows"] % 2_000_000 == 0:
                print(
                    f"ratings rows: {counters['source_rating_rows']:,}",
                    file=sys.stderr,
                    flush=True,
                )
            rating = float(row["rating"])
            if rating < min_rating:
                counters["low_rating_rows"] += 1
                continue
            record = (
                row["user_id"],
                row["parent_asin"],
                rating,
                int(row["timestamp"]),
            )
            interaction_batch.append(record)
            item_batch.append((row["parent_asin"],))
            if len(interaction_batch) >= 50_000:
                connection.executemany(
                    "INSERT INTO interactions VALUES (?,?,?,?)", interaction_batch
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO needed_items VALUES (?)", item_batch
                )
                connection.commit()
                interaction_batch.clear()
                item_batch.clear()
        if interaction_batch:
            connection.executemany(
                "INSERT INTO interactions VALUES (?,?,?,?)", interaction_batch
            )
            connection.executemany(
                "INSERT OR IGNORE INTO needed_items VALUES (?)", item_batch
            )
            connection.commit()
    counters["high_rating_rows"] = connection.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]
    counters["needed_items"] = connection.execute(
        "SELECT COUNT(*) FROM needed_items"
    ).fetchone()[0]

    connection.execute(
        "CREATE TABLE items (parent_asin TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "parent_category TEXT NOT NULL, category TEXT NOT NULL)"
    )
    item_records: list[tuple[str, str, str, str]] = []
    needed_items = {
        str(row[0]) for row in connection.execute("SELECT parent_asin FROM needed_items")
    }
    with metadata_path.open("rb") as handle:
        for line in handle:
            counters["metadata_rows"] += 1
            if counters["metadata_rows"] % 250_000 == 0:
                print(
                    f"metadata rows: {counters['metadata_rows']:,}",
                    file=sys.stderr,
                    flush=True,
                )
            row = json.loads(line)
            parent_asin = str(row.get("parent_asin", ""))
            if parent_asin not in needed_items:
                continue
            categories = [str(value).strip() for value in row.get("categories", [])]
            title = str(row.get("title", "")).strip()
            if len(categories) < 3 or not title:
                counters["needed_items_missing_taxonomy_or_title"] += 1
                continue
            if (
                categories[0] != "Home & Kitchen"
                or categories[1] not in CANONICAL_LEVEL1
            ):
                counters["needed_items_special_taxonomy_node"] += 1
                continue
            combined = f"{categories[1]} {categories[2]}".lower()
            if (
                categories[2] in {"Home & Kitchen", categories[1]}
                or any(term in combined for term in NON_TAXONOMY_TERMS)
            ):
                counters["needed_items_non_taxonomy_level3"] += 1
                continue
            item_records.append(
                (parent_asin, title, categories[1], categories[2])
            )
            if len(item_records) >= 20_000:
                connection.executemany(
                    "INSERT OR IGNORE INTO items VALUES (?,?,?,?)", item_records
                )
                connection.commit()
                item_records.clear()
        if item_records:
            connection.executemany(
                "INSERT OR IGNORE INTO items VALUES (?,?,?,?)", item_records
            )
            connection.commit()
    counters["resolved_items"] = connection.execute(
        "SELECT COUNT(*) FROM items"
    ).fetchone()[0]
    connection.execute("CREATE INDEX interactions_user_time ON interactions(user_id,timestamp,parent_asin)")
    connection.commit()
    connection.close()
    return dict(counters)


def _iter_user_sequences(database_path: Path) -> Iterator[tuple[str, list[dict[str, object]]]]:
    connection = sqlite3.connect(database_path)
    query = """
        SELECT r.user_id,r.parent_asin,r.rating,r.timestamp,
               i.title,i.parent_category,i.category
        FROM interactions r JOIN items i USING(parent_asin)
        ORDER BY r.user_id,r.timestamp,r.parent_asin
    """
    current_user: str | None = None
    records: list[dict[str, object]] = []
    for values in connection.execute(query):
        user_id, parent_asin, rating, timestamp, title, parent, category = values
        if current_user is not None and user_id != current_user:
            yield current_user, records
            records = []
        current_user = str(user_id)
        records.append(
            {
                "parent_asin": str(parent_asin),
                "title": str(title),
                "parent_category": str(parent),
                "category": str(category),
                "rating": float(rating),
                "timestamp": int(timestamp),
            }
        )
    if current_user is not None:
        yield current_user, records
    connection.close()


def _make_candidates(
    user_id: str,
    records: list[dict[str, object]],
    *,
    split: str,
    seed: int,
) -> list[dict[str, object]]:
    if len(records) < 2:
        return []
    output: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    seen_categories: set[str] = set()
    seen_parent_asins: set[str] = set()
    position = 0
    while position < len(records):
        timestamp = int(records[position]["timestamp"])
        end = position + 1
        while end < len(records) and int(records[end]["timestamp"]) == timestamp:
            end += 1
        timestamp_group: list[dict[str, object]] = []
        group_asins: set[str] = set()
        for row in records[position:end]:
            parent_asin = str(row["parent_asin"])
            if parent_asin in seen_parent_asins or parent_asin in group_asins:
                continue
            group_asins.add(parent_asin)
            timestamp_group.append(row)
        new_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in timestamp_group:
            key = category_key(row)
            if key not in seen_categories:
                new_by_category[key].append(row)
        if len(history) >= 5 and new_by_category:
            categories = sorted(new_by_category)
            label_index = int(
                stable_hex(seed, "target", f"{user_id}:{timestamp}"), 16
            ) % len(categories)
            category = categories[label_index]
            target_options = sorted(
                new_by_category[category], key=lambda row: str(row["parent_asin"])
            )
            item_index = int(
                stable_hex(seed, "target-item", f"{user_id}:{timestamp}:{category}"),
                16,
            ) % len(target_options)
            target = target_options[item_index]
            output.append(
                {
                    "sample_id": f"home-kitchen:{split}:{user_id}:{timestamp}",
                    "split": split,
                    "user_id": user_id,
                    "target_timestamp": timestamp,
                    "full_history_length": len(history),
                    "full_seen_interests": sorted(seen_categories),
                    "history": list(history[-50:]),
                    "targets": [target],
                    "target_categories": [category],
                }
            )
        history.extend(timestamp_group)
        seen_categories.update(category_key(row) for row in timestamp_group)
        seen_parent_asins.update(group_asins)
        position = end
    return output


def _select_balanced_train(
    candidates_path: Path,
    *,
    samples_per_category: int,
    target_category_count: int,
    seed: int,
) -> tuple[dict[str, list[str]], set[str], Counter[str], list[dict[str, object]]]:
    counts: Counter[str] = Counter()
    heaps: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    for batch in pq.ParquetFile(candidates_path).iter_batches(
        columns=["sample_id", "user_id", "target_categories"]
    ):
        for row in batch.to_pylist():
            category = str(row["target_categories"][0])
            sample_id = str(row["sample_id"])
            user_id = str(row["user_id"])
            counts[category] += 1
            priority = stable_hex(seed, "balanced-train", sample_id)
            value = int(priority, 16)
            item = (-value, sample_id, user_id, priority)
            heap = heaps[category]
            if len(heap) < samples_per_category:
                heapq.heappush(heap, item)
            elif value < -heap[0][0]:
                heapq.heapreplace(heap, item)
    ranked_categories = sorted(
        (
            (category, count)
            for category, count in counts.items()
            if count >= samples_per_category
        ),
        key=lambda item: item[0],
    )
    if len(ranked_categories) != target_category_count:
        raise ValueError(
            f"Found {len(ranked_categories)} categories with at least "
            f"{samples_per_category} real Train samples; expected "
            f"{target_category_count}. No Top-K truncation or upsampling is allowed."
        )
    retained = {key for key, _ in ranked_categories}
    selection: list[dict[str, object]] = []
    for category in sorted(retained):
        source_rows = sorted(heaps[category], key=lambda item: (item[3], item[1]))
        for _, source_sample_id, user_id, priority in source_rows:
            selection.append(
                {
                    "sample_id": source_sample_id,
                    "source_sample_id": source_sample_id,
                    "user_id": user_id,
                    "target_category": category,
                    "sampling_priority": priority,
                    "replication_index": 0,
                    "sampling_seed": seed,
                }
            )
    selection.sort(key=lambda row: (str(row["target_category"]), str(row["sampling_priority"])))
    selected: dict[str, list[str]] = defaultdict(list)
    for row in selection:
        selected[str(row["source_sample_id"])].append(str(row["sample_id"]))
    return dict(selected), retained, counts, selection


def _filter_eval_row(
    row: dict[str, object], retained: set[str]
) -> tuple[dict[str, object] | None, list[str]]:
    removed = [value for value in row["target_categories"] if value not in retained]
    categories = [value for value in row["target_categories"] if value in retained]
    if not categories:
        return None, removed
    allowed = set(categories)
    output = dict(row)
    output["targets"] = [
        target for target in row["targets"] if category_key(target) in allowed
    ]
    output["target_categories"] = categories
    return output, removed


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def build_amazon_kitchen_dataset(
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    min_rating: float = 4.0,
    samples_per_category: int = 1000,
    target_category_count: int = 112,
    seed: int = 20260806,
    overwrite: bool = False,
) -> BuildResult:
    raw_root, output_dir = Path(raw_root), Path(output_dir)
    ratings_path, metadata_path = raw_root / RATINGS_RELATIVE, raw_root / METADATA_RELATIVE
    for path in (ratings_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        database_path = staging / "working.sqlite3"
        source_stats = _prepare_database(
            ratings_path, metadata_path, database_path, min_rating=min_rating
        )
        candidate_writers = {
            split: ParquetWriter(staging / "candidates" / f"{split}.parquet", DATASET_SCHEMA)
            for split in SPLITS
        }
        construction: Counter[str] = Counter()
        try:
            for user_id, records in _iter_user_sequences(database_path):
                construction["resolved_users"] += 1
                if construction["resolved_users"] % 250_000 == 0:
                    print(
                        f"resolved users: {construction['resolved_users']:,}",
                        file=sys.stderr,
                        flush=True,
                    )
                split = user_split(user_id, seed)
                construction[f"users_assigned_{split}"] += 1
                candidates = _make_candidates(
                    user_id,
                    records,
                    split=split,
                    seed=seed,
                )
                if not candidates:
                    construction[f"users_without_novel_target_{split}"] += 1
                    continue
                for candidate in candidates:
                    candidate_writers[split].append(candidate)
                    construction[f"candidate_rows_{split}"] += 1
        finally:
            for writer in candidate_writers.values():
                writer.close()

        selected, retained, train_counts, selection = _select_balanced_train(
            staging / "candidates/train.parquet",
            samples_per_category=samples_per_category,
            target_category_count=target_category_count,
            seed=seed,
        )
        pq.write_table(pa.Table.from_pylist(selection, SELECTION_SCHEMA), staging / "train_selection_manifest.parquet", compression="zstd")

        category_attributes: dict[str, tuple[str, str]] = {}
        for batch in pq.ParquetFile(staging / "candidates/train.parquet").iter_batches(columns=["targets"]):
            for row in batch.to_pylist():
                for target in row["targets"]:
                    key = category_key(target)
                    if key in retained:
                        category_attributes[key] = (
                            str(target["parent_category"]),
                            str(target["category"]),
                        )
        token_rows = []
        for category_id, key in enumerate(sorted(retained)):
            parent, leaf = category_attributes[key]
            token_rows.append({"category_id": category_id, "category_key": key, "category": leaf, "semantic_id": f"<interest_{category_id:03d}>", "parent_category": parent})
        pq.write_table(pa.Table.from_pylist(token_rows, TOKEN_SCHEMA), staging / "category_vocabulary.parquet", compression="zstd")

        final_counts: dict[str, int] = {}
        target_counts: dict[str, Counter[str]] = {}
        removed_writer = ParquetWriter(staging / "evaluation_filter_manifest.parquet", FILTER_SCHEMA)
        oov_writer = ParquetWriter(staging / "dev_oov.parquet", DATASET_SCHEMA)
        try:
            for split in SPLITS:
                writer = ParquetWriter(staging / f"{split}.parquet", DATASET_SCHEMA)
                counts: Counter[str] = Counter()
                try:
                    for batch in pq.ParquetFile(staging / "candidates" / f"{split}.parquet").iter_batches():
                        for row in batch.to_pylist():
                            construction[f"final_scan_rows_{split}"] += 1
                            if construction[f"final_scan_rows_{split}"] % 250_000 == 0:
                                print(
                                    f"final {split} rows: "
                                    f"{construction[f'final_scan_rows_{split}']:,}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            if split == "train":
                                output_ids = selected.get(str(row["sample_id"]))
                                if not output_ids:
                                    continue
                                filtered = row
                                removed: list[str] = []
                            else:
                                filtered, removed = _filter_eval_row(row, retained)
                                for category in removed:
                                    removed_writer.append({"split": split, "user_id": row["user_id"], "category": category, "reason": "target-category-absent-from-balanced-train"})
                                if filtered is None:
                                    oov_writer.append(row)
                                    construction[f"rows_removed_empty_{split}"] += 1
                                    continue
                            if split == "train":
                                for output_id in output_ids:
                                    output_row = dict(filtered)
                                    output_row["sample_id"] = output_id
                                    writer.append(output_row)
                                    for category in output_row["target_categories"]:
                                        counts[str(category)] += 1
                            else:
                                writer.append(filtered)
                                for category in filtered["target_categories"]:
                                    counts[str(category)] += 1
                finally:
                    writer.close()
                final_counts[split] = writer.count
                target_counts[split] = counts
        finally:
            removed_writer.close()
            oov_writer.close()

        os.replace(staging / "candidates", staging / "unbalanced")
        database_path.unlink()
        statistics = {
            "rule_version": RULE_VERSION,
            "source": source_stats,
            "construction": dict(sorted(construction.items())),
            "split_policy": {"unit": "user", "ratios": {"train": 0.9, "dev": 0.1}, "seed": seed},
            "positive_interaction": {"minimum_rating": min_rating},
            "temporal_sample_rule": "each first occurrence of a level-2 category after at least one strictly earlier timestamp is a novel-interest target",
            "balance": {"target_category_count": target_category_count, "samples_per_category": samples_per_category, "candidate_category_counts": dict(sorted(train_counts.items())), "retained_categories": len(retained), "sampling_with_replacement": False, "replicated_rows": 0},
            "final_rows": final_counts,
            "final_target_category_counts": {split: dict(sorted(counts.items())) for split, counts in target_counts.items()},
        }
        _write_json(staging / "statistics.json", statistics)
        artifacts = [path for path in staging.rglob("*") if path.is_file() and path.name != "metadata.json"]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "dataset": "McAuley-Lab/Amazon-Reviews-2023@2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e Home_and_Kitchen 0-core rating_only",
            "raw_root": portable_path(raw_root),
            "raw_sha256": {portable_path(path): sha256_file(path) for path in (ratings_path, metadata_path)},
            "artifact_sha256": {path.relative_to(staging).as_posix(): sha256_file(path) for path in artifacts},
            "history": "all mapped high-rating interactions at strictly earlier timestamps; order and duplicates preserved",
            "target": "first occurrence of an experimental level-2 category absent from the earlier history category set",
            "train_target": "one candidate per user-target category before class balancing",
            "train_balance": "all classes meeting samples_per_category support; deterministic sampling without replacement; no Top-K truncation",
            "evaluation_target_filter": "remove unsupported target categories; remove row only when no targets remain",
            "category_identity": "experimental_level1 > experimental_level2",
            "taxonomy_projection": {
                "source_root": "categories[0] = Home & Kitchen",
                "experimental_level1": "categories[1]",
                "experimental_level2_target": "categories[2]",
                "allowed_level1": sorted(CANONICAL_LEVEL1),
            },
            "semantic_id_order": "category_key ascending",
        }
        _write_json(staging / "metadata.json", metadata)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return BuildResult(output_dir, len(retained), final_counts["train"], final_counts["dev"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--samples-per-category", type=int, default=1000)
    parser.add_argument("--target-category-count", type=int, default=112)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_amazon_kitchen_dataset(**vars(args))
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
