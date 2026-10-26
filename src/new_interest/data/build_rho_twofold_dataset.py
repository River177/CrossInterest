"""Freeze user-disjoint, class-balanced halves for out-of-fold RHO teachers.

This builder changes membership only: each output has the exact source SFT
schema, values, and relative row order. It never builds prompts or loads models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "rho-reference-user-disjoint-twofold-v1"
REQUIRED_COLUMNS = {
    "sample_id", "source_sample_id", "user_id", "impression_id", "split",
    "task_type", "history_length", "prompt", "response", "messages",
    "target_subcategory",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _identity(path: Path) -> dict:
    return {"path": portable_path(path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def _priority(seed: int, row: dict) -> str:
    # JSON avoids collisions caused by separators inside user or sample IDs.
    payload = json.dumps([seed, RULE_VERSION, row["user_id"], row["source_sample_id"]],
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_twofold(rows: list[dict], *, seed: int, rows_per_class: int) -> list[dict]:
    """Return membership in source order; class hash ranking defines membership."""
    if rows_per_class <= 0 or rows_per_class % 2:
        raise ValueError("rows_per_class must be a positive even integer")
    by_target = defaultdict(list)
    seen = {key: set() for key in ("sample_id", "source_sample_id", "user_id")}
    for index, row in enumerate(rows):
        for key, values in seen.items():
            value = row[key]
            if not isinstance(value, str) or not value or value in values:
                raise ValueError(f"Missing or duplicate {key}: {value}")
            values.add(value)
        by_target[row["target_subcategory"]].append(index)
    assignments = {}
    for target, indices in sorted(by_target.items()):
        if len(indices) != rows_per_class:
            raise ValueError(f"Class {target}: expected {rows_per_class}, got {len(indices)}")
        ordered = sorted(indices, key=lambda i: (_priority(seed, rows[i]),
                                                rows[i]["source_sample_id"]))
        for rank, index in enumerate(ordered):
            assignments[index] = {"fold": rank // (rows_per_class // 2),
                                  "within_class_hash_rank": rank,
                                  "fold_priority": _priority(seed, rows[index])}
    return [assignments[index] for index in range(len(rows))]


def _audit_full_history(rows: list[dict], raw_file: Path, raw_metadata: Path,
                        raw_identity: dict, *, history_limit: int) -> dict:
    metadata = json.loads(raw_metadata.read_text())
    if metadata.get("full_history") is not True or metadata.get("history_truncation") is not None:
        raise ValueError("Raw history metadata does not certify complete untruncated history")
    if metadata.get("impression_clicks_appended_to_history") is not False:
        raise ValueError("Raw history must not include impression clicks")
    expected_hash = metadata.get("artifact_sha256", {}).get(raw_file.name)
    if expected_hash != raw_identity["sha256"]:
        raise ValueError("Raw history file SHA-256 differs from its metadata")
    requested = {row["source_sample_id"]: row for row in rows}
    ids = pa.array(list(requested), type=pa.string())
    found = set()
    max_full_history = 0
    columns = ["sample_id", "user_id", "impression_id", "target_interest", "history"]
    for batch in pq.ParquetFile(raw_file).iter_batches(batch_size=8192, columns=columns):
        selected = batch.filter(pc.is_in(batch.column("sample_id"), value_set=ids))
        for raw in selected.to_pylist():
            source_id = raw["sample_id"]
            if source_id in found:
                raise ValueError(f"Duplicate raw history identity: {source_id}")
            row = requested[source_id]
            for key in ("user_id", "impression_id"):
                if row[key] != raw[key]:
                    raise ValueError(f"Raw/source {key} mismatch: {source_id}")
            target = row["target_subcategory"]
            if raw["target_interest"]["subcategory"] != target:
                raise ValueError(f"Raw/source target mismatch: {source_id}")
            history = raw["history"]
            if target in {event["subcategory"] for event in history}:
                raise ValueError(f"Target appears in full raw history: {source_id}")
            if not history or row["history_length"] != min(len(history), history_limit):
                raise ValueError(f"Raw/source history length mismatch: {source_id}")
            # Verify that the preserved text belongs to this same raw history.
            expected_history = "\n".join(
                f'{i + 1}. [{event["subcategory"]}] {event["title"]}'
                for i, event in enumerate(history[-history_limit:])
            )
            if f"\n\n{expected_history}\n\n" not in row["prompt"]:
                raise ValueError(f"Preserved prompt differs from raw recent history: {source_id}")
            max_full_history = max(max_full_history, len(history))
            found.add(source_id)
    if found != set(requested):
        raise ValueError(f"Missing {len(set(requested) - found)} source IDs in raw history file")
    return {"rows_checked": len(found), "target_in_full_history": 0,
            "raw_source_identity_mismatches": 0, "prompt_recent_history_mismatches": 0,
            "max_full_history": max_full_history, "prompt_history_limit": history_limit,
            "full_history": True, "history_truncation": None}


def build_twofold_dataset(
    *, source_file: Path | str, mapping_file: Path | str, validation_file: Path | str,
    dev_file: Path | str, raw_train_file: Path | str, raw_metadata_file: Path | str,
    output_dir: Path | str, seed: int = 42, expected_class_count: int = 108,
    rows_per_class: int = 200, history_limit: int = 50, production: bool = False,
) -> dict:
    """Validate identities, audit full histories, then atomically publish two halves."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if expected_class_count <= 0 or history_limit <= 0:
        raise ValueError("expected_class_count and history_limit must be positive")
    if production and (expected_class_count, rows_per_class, history_limit, seed) != (108, 200, 50, 42):
        raise ValueError("Production profile requires 108 classes, 200 rows/class, h50, seed42")
    paths = {"training_source": Path(source_file), "mapping": Path(mapping_file),
             "validation": Path(validation_file), "dev": Path(dev_file),
             "full_history_source": Path(raw_train_file),
             "full_history_metadata": Path(raw_metadata_file)}
    identities = {key: _identity(path) for key, path in paths.items()}
    source = pq.read_table(paths["training_source"])
    missing = REQUIRED_COLUMNS.difference(source.column_names)
    if missing:
        raise ValueError(f"Source missing columns: {sorted(missing)}")
    if "training_task" in source.column_names:
        source_indices = [i for i, task in enumerate(source["training_task"].to_pylist())
                          if task == "new_interest"]
    else:
        source_indices = list(range(len(source)))
    train = source.take(pa.array(source_indices, type=pa.int64()))
    rows = train.to_pylist()
    if len(rows) != expected_class_count * rows_per_class:
        raise ValueError(f"Expected {expected_class_count * rows_per_class} NI rows, got {len(rows)}")
    mapping_rows = pq.read_table(paths["mapping"]).to_pylist()
    mapping = {row["target_subcategory"]: row for row in mapping_rows}
    indices = [row["interest_index"] for row in mapping_rows]
    if (len(mapping_rows) != expected_class_count or len(mapping) != expected_class_count
            or sorted(indices) != list(range(expected_class_count))):
        raise ValueError("Mapping must contain each target and contiguous interest index once")
    if any(row["interest_token"] != f'<interest_{row["interest_index"]:03d}>' for row in mapping_rows):
        raise ValueError("Mapping token/index mismatch")
    for row in rows:
        target = row["target_subcategory"]
        if target not in mapping or row["response"] != mapping[target]["interest_token"]:
            raise ValueError(f"Target/SID mismatch: {row['source_sample_id']}")
        if row["split"] != "train" or row["task_type"] != "behavior_interest":
            raise ValueError("Only training Behavior+Interest examples are accepted")
        expected_messages = [{"role": "user", "content": row["prompt"]},
                             {"role": "assistant", "content": row["response"]}]
        if row["messages"] != expected_messages:
            raise ValueError(f"Prompt/response/messages mismatch: {row['source_sample_id']}")
    if set(row["target_subcategory"] for row in rows) != set(mapping):
        raise ValueError("Training targets differ from mapping classes")
    assignments = assign_twofold(rows, seed=seed, rows_per_class=rows_per_class)
    users = {row["user_id"] for row in rows}
    heldout = {}
    for split in ("validation", "dev"):
        heldout_users = pq.read_table(paths[split], columns=["user_id"])["user_id"].to_pylist()
        if any(not isinstance(user, str) or not user for user in heldout_users):
            raise ValueError(f"Missing {split} user identity")
        overlap = users.intersection(heldout_users)
        if overlap:
            raise ValueError(f"Train/{split} user overlap: {len(overlap)}")
        heldout[split] = {"rows": len(heldout_users), "users": len(set(heldout_users)),
                          "training_user_overlap": 0}
    if production and (heldout["validation"]["rows"], heldout["dev"]["rows"]) != (5354, 31392):
        raise ValueError("Production profile requires frozen 5354 validation and 31392 Dev rows")
    novelty = _audit_full_history(rows, paths["full_history_source"],
                                 paths["full_history_metadata"], identities["full_history_source"],
                                 history_limit=history_limit)
    manifest = []
    for ni_index, (row, assignment) in enumerate(zip(rows, assignments)):
        manifest.append({"source_row_index": source_indices[ni_index], "ni_row_index": ni_index,
                         "sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
                         "user_id": row["user_id"], "impression_id": row["impression_id"],
                         "target_subcategory": row["target_subcategory"],
                         "interest_index": mapping[row["target_subcategory"]]["interest_index"],
                         "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                         **assignment})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        fold_stats = {}
        fold_users = []
        for fold in range(2):
            selected_indices = [i for i, assignment in enumerate(assignments) if assignment["fold"] == fold]
            selected = train.take(pa.array(selected_indices, type=pa.int64()))
            filename = f"fold{fold}.parquet"
            pq.write_table(selected, stage / filename, compression="zstd")
            if not pq.read_table(stage / filename).equals(selected, check_metadata=True):
                raise ValueError(f"Readback changes original SFT values/schema: {filename}")
            counts = Counter(selected["target_subcategory"].to_pylist())
            if set(counts.values()) != {rows_per_class // 2} or len(counts) != expected_class_count:
                raise ValueError(f"Fold {fold} class balance failed")
            these_users = set(selected["user_id"].to_pylist())
            fold_users.append(these_users)
            fold_stats[str(fold)] = {"rows": len(selected), "users": len(these_users),
                                     "classes": len(counts), "rows_per_class": rows_per_class // 2,
                                     "class_counts": dict(sorted(counts.items()))}
        if fold_users[0] & fold_users[1] or fold_users[0] | fold_users[1] != users:
            raise ValueError("Fold user intersection/union audit failed")
        shutil.copyfile(paths["mapping"], stage / "interest_token_mapping.parquet")
        pq.write_table(pa.Table.from_pylist(manifest), stage / "fold_manifest.parquet", compression="zstd")
        files = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                 for path in sorted(stage.iterdir())}
        metadata = {
            "rule_version": RULE_VERSION, "profile": "production" if production else "custom",
            "seed": seed, "sources": identities, "files": files, "source_rows": len(source),
            "training_rows": len(train), "training_users": len(users), "classes": expected_class_count,
            "folds": fold_stats, "heldout_audit": heldout, "novelty_audit": novelty,
            "assignment": "Within target, sort SHA256(JSON[seed, rule_version, user_id, source_sample_id]); first half fold0, second half fold1",
            "output_order": "Original source relative order within each half",
            "fold_user_overlap": 0, "exact_source_ni_union": True,
            "original_schema_and_values_preserved": True, "mapping_byte_copy": True,
            "teacher_directions": [{"teacher": "fold0", "train": "fold0.parquet", "score": "fold1.parquet"},
                                   {"teacher": "fold1", "train": "fold1.parquet", "score": "fold0.parquet"}],
            "teacher_initialization_requirement": "Original pretrained Qwen; no adapter trained on either held-out user half",
        }
        _json_write(stage / "metadata.json", metadata)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        stage.rename(output_dir)
        return metadata
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--raw-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-class-count", type=int, default=108)
    parser.add_argument("--rows-per-class", type=int, default=200)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--production", action="store_true", help="Enforce frozen 21600/5354/31392 production sizes")
    args = parser.parse_args()
    result = build_twofold_dataset(source_file=args.source, mapping_file=args.mapping,
        validation_file=args.validation, dev_file=args.dev, raw_train_file=args.raw_train,
        raw_metadata_file=args.raw_metadata, output_dir=args.output, seed=args.seed,
        expected_class_count=args.expected_class_count, rows_per_class=args.rows_per_class,
        history_limit=args.history_limit, production=args.production)
    print(json.dumps({"output_dir": portable_path(args.output), "training_rows": result["training_rows"],
                      "fold_rows": {key: value["rows"] for key, value in result["folds"].items()},
                      "files": result["files"]}, indent=2))


if __name__ == "__main__":
    main()
