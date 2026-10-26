"""Recover task-relevant static MIND fields from a complete, verified cache.

The cache contains dynamic histories, but its producer preserves the initial
History as an unmodified prefix. Never use its novel/repeated-interest labels.
Acceptance also requires an exact full static-NI candidate comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from new_interest.data.mind_io import BehaviorEvent, NewsCatalog, NewsRecord
from new_interest.infrastructure.paths import portable_path


LOGGER = logging.getLogger(__name__)
VERSION = "verified-static-mind-projection-v1"
CACHE = Path("data/processed/new_interest_dynamic_v1")
STATIC = Path("data/processed/new_interest")
USER_DISJOINT = Path("data/processed/new_interest_static_user_disjoint_v1")
NEWS_FIELDS = ["news_id", "category", "subcategory", "title"]
PROJECTION_SCHEMA = pa.schema([
    pa.field("split", pa.string(), nullable=False),
    pa.field("impression_id", pa.string(), nullable=False),
    pa.field("user_id", pa.string(), nullable=False),
    pa.field("timestamp", pa.timestamp("ms"), nullable=False),
    pa.field("history_news_ids", pa.list_(pa.string()), nullable=False),
    pa.field("clicked_news_ids", pa.list_(pa.string()), nullable=False),
])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def history_signature(ids: list[str]) -> bytes:
    return hashlib.sha256(_json_bytes(ids)).digest()


def candidate_signature(user: str, timestamp: datetime, history_hash: bytes, news_ids: list[str]) -> bytes:
    return hashlib.sha256(_json_bytes([user, timestamp.isoformat(), history_hash.hex(), news_ids])).digest()


def _list_field(values: pa.ListArray, name: str) -> pa.ListArray:
    return pa.ListArray.from_arrays(values.offsets, values.values.field(name))


def initial_history(history: pa.ListArray, initial_lengths: pa.Array) -> pa.ListArray:
    lengths = pc.list_value_length(history)
    if pc.any(pc.or_(pc.less(initial_lengths, 1), pc.greater(initial_lengths, lengths))).as_py():
        raise ValueError("Initial History length is outside the stored history")
    flat = pc.list_flatten(history)
    parents = pc.list_parent_indices(history)
    offsets = pc.subtract(history.offsets.slice(0, len(history)), history.offsets[0])
    positions = pc.subtract(pa.array(range(len(flat)), type=pa.int64()), offsets.take(parents))
    keep = pc.less(positions, initial_lengths.take(parents))
    selected = flat.filter(keep)
    new_offsets = [0]
    for length in initial_lengths.to_pylist():
        new_offsets.append(new_offsets[-1] + int(length))
    return pa.ListArray.from_arrays(pa.array(new_offsets, type=pa.int32()), selected)


def _absorb_news(known: dict[str, tuple[str, str, str]], records: pa.StructArray) -> None:
    if not len(records):
        return
    table = pa.Table.from_arrays([records.field(name) for name in NEWS_FIELDS], names=NEWS_FIELDS)
    unique = table.group_by(NEWS_FIELDS).aggregate([]).to_pydict()
    for news_id, category, subcategory, title in zip(*(unique[name] for name in NEWS_FIELDS)):
        if not all(isinstance(value, str) and value for value in (news_id, category, subcategory, title)):
            raise ValueError("Cached classified news has empty identity/label/title")
        value = (category, subcategory, title)
        previous = known.setdefault(news_id, value)
        if previous != value:
            raise ValueError(f"Conflicting classified news identity: {news_id}")


def _verified_metadata(reference_root: Path, contract: dict[str, Any]) -> tuple[dict, dict, dict]:
    cache_meta = json.loads((reference_root / CACHE / "metadata.json").read_text())
    static_meta = json.loads((reference_root / STATIC / "metadata.json").read_text())
    disjoint_meta = json.loads((reference_root / USER_DISJOINT / "metadata.json").read_text())
    if cache_meta.get("rule_version") != "dynamic-click-history-v1" or cache_meta.get("all_behavior_impressions_included") is not True:
        raise ValueError("Cache does not declare the complete verified impression contract")
    restored_manifest = json.dumps(cache_meta["source_manifest"], ensure_ascii=False, indent=2).encode() + b"\n"
    expected_manifest = contract["raw_source_sha256"]["data/raw/MINDlarge_reclassified/MANIFEST.json"]
    if hashlib.sha256(restored_manifest).hexdigest() != expected_manifest:
        raise ValueError("Embedded raw MANIFEST cannot be reconstructed to the frozen SHA-256")
    for meta in (cache_meta, static_meta):
        if meta["source_file_sha256"]["MANIFEST.json"] != expected_manifest:
            raise ValueError("Cached datasets refer to different raw manifests")
        if meta["source_file_sha256"]["FILTERED_NEWS_IDS.json"] != contract["raw_source_sha256"]["data/raw/MINDlarge_reclassified/FILTERED_NEWS_IDS.json"]:
            raise ValueError("Cached datasets refer to different raw filtering")
    for split in ("train", "dev"):
        for filename, field in (("news.tsv", "sha256"), ("behaviors.tsv", "behaviors_sha256")):
            key = f"data/raw/MINDlarge_reclassified/{split}/{filename}"
            if cache_meta["source_manifest"]["splits"][split][field] != contract["raw_source_sha256"][key]:
                raise ValueError("Embedded raw input identity differs from the frozen NI source")
    return cache_meta, static_meta, disjoint_meta


def restore_projection(
    *, reference_root: Path, contract_file: Path, output_dir: Path,
    expected_train_rows: int = 2_186_664, expected_train_users: int = 698_357,
    expected_ni_candidates: int = 1_489_348, batch_size: int = 4096,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Projection output already exists: {output_dir}")
    contract = json.loads(contract_file.read_text())
    cache_meta, static_meta, disjoint_meta = _verified_metadata(reference_root, contract)
    checked_inputs: dict[str, str] = {}
    for relative, expected in [
        (CACHE / "train.parquet", cache_meta["artifact_sha256"]["train.parquet"]),
        (CACHE / "interest_vocabulary.parquet", cache_meta["artifact_sha256"]["interest_vocabulary.parquet"]),
        (STATIC / "train.parquet", static_meta["artifact_sha256"]["train.parquet"]),
        (USER_DISJOINT / "statistics.json", disjoint_meta["artifact_sha256"]["statistics.json"]),
    ]:
        path = reference_root / relative
        if sha256_file(path) != expected:
            raise ValueError(f"Historical artifact SHA-256 mismatch: {relative}")
        checked_inputs[portable_path(path, root=reference_root)] = expected
    for relative in (CACHE / "metadata.json", STATIC / "metadata.json", USER_DISJOINT / "metadata.json"):
        checked_inputs[portable_path(reference_root / relative, root=reference_root)] = sha256_file(reference_root / relative)
    statistics = json.loads((reference_root / USER_DISJOINT / "statistics.json").read_text())["train"]
    if (
        int(statistics["raw_behavior_rows"]) != expected_train_rows
        or int(statistics["raw_users"]) != expected_train_users
        or int(statistics["eligible_candidate_rows_before_selection"]) != expected_ni_candidates
        or int(cache_meta["source_manifest"]["splits"]["train"]["behavior_rows"]) != expected_train_rows
    ):
        raise ValueError("Historical complete Train dimensions differ from the frozen projection gate")
    vocabulary = pq.read_table(reference_root / CACHE / "interest_vocabulary.parquet").to_pylist()
    interest_to_id = {(row["category"], row["subcategory"]): int(row["interest_id"]) for row in vocabulary}
    if sorted(interest_to_id.values()) != list(range(len(vocabulary))):
        raise ValueError("Raw vocabulary IDs are not one-to-one and contiguous")

    known: dict[str, tuple[str, str, str]] = {}
    golden: dict[str, bytes] = {}
    golden_histories: dict[str, bytes] = {}
    columns = ["sample_id", "user_id", "timestamp", "history", "target_interest", "target_news"]
    for batch in pq.ParquetFile(reference_root / STATIC / "train.parquet").iter_batches(batch_size=batch_size, columns=columns):
        histories = batch.column("history")
        _absorb_news(known, pc.list_flatten(histories))
        target_news = batch.column("target_news")
        flat_targets = pc.list_flatten(target_news)
        target_labels = batch.column("target_interest").take(pc.list_parent_indices(target_news))
        _absorb_news(known, pa.StructArray.from_arrays([
            flat_targets.field("news_id"), target_labels.field("category"),
            target_labels.field("subcategory"), flat_targets.field("title"),
        ], names=NEWS_FIELDS))
        light = pa.table({
            "sample_id": batch.column("sample_id"), "user_id": batch.column("user_id"),
            "timestamp": batch.column("timestamp"), "history_ids": _list_field(histories, "news_id"),
            "target_ids": _list_field(target_news, "news_id"),
        }).to_pydict()
        for source_id, user, timestamp, history_ids, target_ids in zip(*(light[name] for name in ["sample_id", "user_id", "timestamp", "history_ids", "target_ids"])):
            history_hash = history_signature(history_ids)
            if golden_histories.setdefault(user, history_hash) != history_hash:
                raise ValueError("Golden static Train changes History within a user")
            if source_id in golden:
                raise ValueError("Golden NI candidate IDs are duplicated")
            golden[source_id] = candidate_signature(user, timestamp, history_hash, target_ids)
        if len(golden) % 262_144 == 0:
            LOGGER.info("Indexed %s golden static NI candidates", len(golden))
    if len(golden) != expected_ni_candidates:
        raise ValueError("Golden static candidate count mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    profiles: dict[str, bytes] = {}
    impressions: set[str] = set()
    checked_candidates = 0
    observed_clicks = 0
    try:
        writer = pq.ParquetWriter(staging / "train.parquet", PROJECTION_SCHEMA, compression="zstd")
        try:
            columns = ["split", "impression_id", "user_id", "timestamp", "initial_history_length", "history", "clicked_news"]
            for batch in pq.ParquetFile(reference_root / CACHE / "train.parquet").iter_batches(batch_size=batch_size, columns=columns):
                # Discard every appended impression click before classifying new/old.
                histories = initial_history(batch.column("history"), batch.column("initial_history_length"))
                clicked = batch.column("clicked_news")
                _absorb_news(known, pc.list_flatten(histories))
                _absorb_news(known, pc.list_flatten(clicked))
                projected = pa.Table.from_arrays([
                    batch.column("split"), batch.column("impression_id"), batch.column("user_id"),
                    batch.column("timestamp"), _list_field(histories, "news_id"), _list_field(clicked, "news_id"),
                ], schema=PROJECTION_SCHEMA)
                light = projected.to_pydict()
                for split, impression, user, timestamp, history_ids, clicked_ids in zip(*(light[name] for name in PROJECTION_SCHEMA.names)):
                    if split != "train" or impression in impressions:
                        raise ValueError("Projection contains a wrong split or duplicate impression")
                    impressions.add(impression)
                    history_hash = history_signature(history_ids)
                    if profiles.setdefault(user, history_hash) != history_hash:
                        raise ValueError("Recovered raw History changes within a Train user")
                    if user in golden_histories and golden_histories[user] != history_hash:
                        raise ValueError("Recovered raw History differs from the independent static artifact")
                    history_interests = {interest_to_id[known[n][:2]] for n in history_ids}
                    positive_by_interest: dict[int, list[str]] = defaultdict(list)
                    if len(clicked_ids) != len(set(clicked_ids)):
                        raise ValueError("Cached clicked-news IDs contain duplicates")
                    for news_id in clicked_ids:
                        positive_by_interest[interest_to_id[known[news_id][:2]]].append(news_id)
                    observed_clicks += len(clicked_ids)
                    for interest_id, target_ids in positive_by_interest.items():
                        if interest_id in history_interests:
                            continue
                        source_id = f"train:{impression}:{interest_id}"
                        actual = candidate_signature(user, timestamp, history_hash, target_ids)
                        if golden.pop(source_id, None) != actual:
                            raise ValueError(f"Recovered static candidate differs from golden Train: {source_id}")
                        checked_candidates += 1
                writer.write_table(projected)
                if len(impressions) % 262_144 == 0:
                    LOGGER.info("Verified %s raw impressions / %s static NI candidates", len(impressions), checked_candidates)
        finally:
            writer.close()
        if len(impressions) != expected_train_rows or len(profiles) != expected_train_users or golden:
            raise ValueError(f"Incomplete raw projection: rows={len(impressions)}, users={len(profiles)}, missing_NI={len(golden)}")
        if checked_candidates != expected_ni_candidates:
            raise ValueError("Static NI candidate equivalence check is incomplete")
        rows = [{"news_id": news_id, "category": value[0], "subcategory": value[1], "title": value[2], "interest_id": interest_to_id[value[:2]]} for news_id, value in sorted(known.items())]
        pq.write_table(pa.Table.from_pylist(rows), staging / "task_news_catalog.parquet", compression="zstd")
        shutil.copyfile(reference_root / CACHE / "interest_vocabulary.parquet", staging / "interest_vocabulary.parquet")
        shutil.copyfile(reference_root / USER_DISJOINT / "statistics.json", staging / "static_statistics.json")
        restored_manifest = json.dumps(cache_meta["source_manifest"], ensure_ascii=False, indent=2) + "\n"
        (staging / "MANIFEST.json").write_text(restored_manifest)
        for relative, expected in checked_inputs.items():
            if sha256_file(reference_root / relative) != expected:
                raise ValueError("Historical source changed during the projection audit")
        receipt = {
            "version": VERSION, "status": "verified_task_field_equivalence",
            "contract_sha256": sha256_file(contract_file),
            "projection_code_sha256": sha256_file(Path(__file__)),
            "actual_source_sha256": checked_inputs,
            "historical_raw_source_sha256": contract["raw_source_sha256"],
            "restored_raw_manifest_sha256": sha256_file(staging / "MANIFEST.json"),
            "original_raw_tsv_bytes_recovered": False,
            "all_original_train_impressions": len(impressions),
            "all_original_train_users": len(profiles),
            "independent_static_ni_candidates_exactly_matched": checked_candidates,
            "candidate_comparison_fields": ["sample_id", "user_id", "timestamp", "ordered_original_history_news_ids", "ordered_target_clicked_news_ids"],
            "news_record_comparison_fields": NEWS_FIELDS,
            "clicked_news_count": observed_clicks,
            "task_news_catalog_size": len(known),
            "history_recovery": "history[:initial_history_length]; discard every later appended click",
            "cached_dynamic_novel_and_repeated_labels_used": False,
            "unclicked_candidates_included": False,
            "unclicked_candidates_required_by_approved_task": False,
            "artifact_sha256": {path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()},
        }
        (staging / "projection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staging.rename(output_dir)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_projection(path: Path, *, contract_file: Path) -> tuple[NewsCatalog, dict[str, Any]]:
    receipt = json.loads((path / "projection_receipt.json").read_text())
    if receipt.get("version") != VERSION or receipt.get("status") != "verified_task_field_equivalence":
        raise ValueError("Static projection does not have a successful equivalence audit")
    if receipt["contract_sha256"] != sha256_file(contract_file):
        raise ValueError("Static projection belongs to a different frozen protocol")
    for filename, expected in receipt["artifact_sha256"].items():
        if Path(filename).name != filename or sha256_file(path / filename) != expected:
            raise ValueError(f"Static projection artifact mismatch: {filename}")
    contract = json.loads(contract_file.read_text())
    if sha256_file(path / "MANIFEST.json") != contract["raw_source_sha256"]["data/raw/MINDlarge_reclassified/MANIFEST.json"]:
        raise ValueError("Recovered raw manifest does not match the approved source")
    manifest = json.loads((path / "MANIFEST.json").read_text())
    statistics = json.loads((path / "static_statistics.json").read_text())["train"]
    if (
        receipt["all_original_train_impressions"] != manifest["splits"]["train"]["behavior_rows"]
        or receipt["all_original_train_impressions"] != statistics["raw_behavior_rows"]
        or receipt["all_original_train_users"] != statistics["raw_users"]
        or receipt["independent_static_ni_candidates_exactly_matched"] != statistics["eligible_candidate_rows_before_selection"]
        or receipt["cached_dynamic_novel_and_repeated_labels_used"] is not False
    ):
        raise ValueError("Static projection equivalence dimensions or history policy differ")
    vocabulary = pq.read_table(path / "interest_vocabulary.parquet").to_pylist()
    interests = [None] * len(vocabulary)
    for row in vocabulary:
        interests[int(row["interest_id"])] = (str(row["category"]), str(row["subcategory"]))
    if any(value is None for value in interests):
        raise ValueError("Projection vocabulary has missing indices")
    news = {
        str(row["news_id"]): NewsRecord(
            news_id=str(row["news_id"]), category=str(row["category"]),
            subcategory=str(row["subcategory"]), title=str(row["title"]),
            interest_id=int(row["interest_id"]),
        ) for row in pq.read_table(path / "task_news_catalog.parquet").to_pylist()
    }
    return NewsCatalog(news=news, interests=interests, news_rows={"task_projection": len(news)}), receipt


def iter_projection_events(path: Path, *, batch_size: int = 8192) -> Iterator[BehaviorEvent]:
    for batch in pq.ParquetFile(path / "train.parquet").iter_batches(batch_size=batch_size):
        columns = batch.to_pydict()
        for split, impression, user, timestamp, history, clicked in zip(*(columns[name] for name in PROJECTION_SCHEMA.names)):
            yield BehaviorEvent(
                split=split, impression_id=impression, user_id=user, timestamp=timestamp,
                history_text=" ".join(history), history_news_ids=tuple(history),
                clicked_news_ids=tuple(clicked), impression_candidates=(),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--contract-file", type=Path, default=Path("configs/experiments/mind_old_new_interest_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recovered/mind_static_task_projection_v1"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    receipt = restore_projection(reference_root=args.reference_root, contract_file=args.contract_file, output_dir=args.output_dir)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
