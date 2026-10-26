from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from new_interest.data.mind_io import BehaviorEvent, iter_behavior_events, load_news_catalog
from new_interest.data.restore_static_mind_projection import iter_projection_events, load_projection
from new_interest.infrastructure.paths import portable_path


LOGGER = logging.getLogger(__name__)
RULE_VERSION = "mind-old-new-interest-v1"
NI_ROOT = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
)
NI_TRAIN = NI_ROOT / "train/behavior_interest/21600.parquet"
NI_DEV = NI_ROOT / "eval/behavior_interest/dev_full_grouped.parquet"
MAPPING = NI_ROOT / "interest_token_mapping.parquet"
OLD_PROMPT_TEMPLATE = '''The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

{history}

Based on both the headlines and their topic categories, predict a previously observed topic category that I am likely to click again. The predicted category must appear among the topic categories in the history above. Respond with only the learned interest token assigned to the predicted category, in the form <interest_NNN>.'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(seed: int, purpose: str, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{source_id}".encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _verify_inputs(
    raw_root: Path | None, reference_root: Path, contract: dict[str, Any]
) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {"raw": {}, "reference": {}}
    for relative in (("train/news.tsv", "train/behaviors.tsv", "dev/news.tsv") if raw_root is not None else ()):
        source = raw_root / relative
        expected = contract["raw_source_sha256"][
            f"data/raw/MINDlarge_reclassified/{relative}"
        ]
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(f"Raw source SHA-256 mismatch: {relative}")
        hashes["raw"][portable_path(source, root=raw_root)] = actual
    for entry in contract["new_interest"]["input_files"]:
        source = reference_root / entry["path"]
        actual = sha256_file(source)
        if actual != entry["sha256"]:
            raise ValueError(f"Frozen NI input SHA-256 mismatch: {entry['path']}")
        hashes["reference"][portable_path(source, root=reference_root)] = actual
    return hashes


def _load_reference(
    reference_root: Path, contract: dict[str, Any]
) -> tuple[pa.Table, dict[str, str], set[str], set[str]]:
    train = pq.read_table(reference_root / NI_TRAIN)
    mapping = pq.read_table(reference_root / MAPPING).to_pylist()
    labels = [str(row["target_subcategory"]) for row in mapping]
    tokens = [str(row["interest_token"]) for row in mapping]
    indices = [int(row["interest_index"]) for row in mapping]
    expected_count = int(contract["old_interest"]["allowed_label_count"])
    if (
        len(labels) != expected_count
        or labels != sorted(set(labels))
        or indices != list(range(expected_count))
        or tokens != [f"<interest_{index:03d}>" for index in indices]
    ):
        raise ValueError("NI mapping is not the frozen canonical SID vocabulary")
    token_by_label = dict(zip(labels, tokens))
    train_rows = train.to_pylist()
    if train.num_rows != int(contract["new_interest"]["unchanged_rows"]):
        raise ValueError("Frozen NI row count mismatch")
    for key in ("sample_id", "source_sample_id", "user_id"):
        values = train[key].to_pylist()
        if len(values) != len(set(values)):
            raise ValueError(f"NI contains duplicate {key}")
    for row in train_rows:
        if row["response"] != token_by_label.get(row["target_subcategory"]):
            raise ValueError("NI response does not match the frozen SID mapping")
        if row["messages"] != [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]:
            raise ValueError("NI messages differ from the frozen prompt/response")
    ni_users = set(train["user_id"].to_pylist())
    dev_users = set(
        pq.read_table(reference_root / NI_DEV, columns=["user_id"])[
            "user_id"
        ].to_pylist()
    )
    if ni_users & dev_users:
        raise ValueError("Frozen NI Train/Dev users overlap")
    return train, token_by_label, ni_users, dev_users


def _scan_train(
    connection: sqlite3.Connection,
    *,
    raw_root: Path,
    catalog: Any,
    labels: set[str],
    dev_users: set[str],
    history_limit: int,
    seed: int,
    events: Iterable[BehaviorEvent] | None = None,
) -> tuple[Counter[str], Counter[str]]:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY, history_sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE candidates (
            user_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            impression_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            label TEXT NOT NULL,
            target_news_ids TEXT NOT NULL,
            history_news_ids TEXT NOT NULL,
            user_priority TEXT NOT NULL,
            class_priority TEXT NOT NULL,
            eligible_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )
    counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    source_events = events if events is not None else iter_behavior_events(raw_root, ("train",), catalog)
    for event in source_events:
        counts["raw_train_impressions"] += 1
        if event.user_id in dev_users:
            raise ValueError(
                f"Raw Train/Dev user overlap, including users without OI: {event.user_id}"
            )
        history_hash = hashlib.sha256(" ".join(event.history_news_ids).encode()).hexdigest()
        previous_history = connection.execute(
            "SELECT history_sha256 FROM users WHERE user_id = ?", (event.user_id,)
        ).fetchone()
        if previous_history is None:
            connection.execute(
                "INSERT INTO users VALUES (?, ?)", (event.user_id, history_hash)
            )
        elif previous_history[0] != history_hash:
            raise ValueError(f"Inconsistent static History for user {event.user_id}")

        full_labels = {
            catalog.news[news_id].subcategory for news_id in event.history_news_ids
        }
        visible_labels = {
            catalog.news[news_id].subcategory
            for news_id in event.history_news_ids[-history_limit:]
        }
        positive_by_label: dict[str, list[str]] = {}
        for news_id in event.clicked_news_ids:
            label = catalog.news[news_id].subcategory
            positive_by_label.setdefault(label, []).append(news_id)
        eligible: list[tuple[str, str, str, list[str]]] = []
        for label, target_news_ids in sorted(positive_by_label.items()):
            counts["positive_impression_category_pairs"] += 1
            if label not in labels:
                counts["excluded_positive_outside_frozen_vocabulary"] += 1
                continue
            if label not in full_labels:
                counts["excluded_positive_new_to_full_history"] += 1
                continue
            if label not in visible_labels:
                counts["excluded_old_positive_not_visible_in_h50"] += 1
                continue
            interest_id = catalog.news[target_news_ids[0]].interest_id
            source_id = f"train:{event.impression_id}:{interest_id}:old_interest"
            priority = _priority(seed, "old-interest-user-target", source_id)
            eligible.append((priority, source_id, label, sorted(target_news_ids)))
            eligible_counts[label] += 1
        if eligible:
            counts["eligible_impression_category_pairs"] += len(eligible)
            priority, source_id, label, target_news_ids = min(eligible)
            previous = connection.execute(
                "SELECT user_priority, source_id, eligible_count FROM candidates WHERE user_id = ?",
                (event.user_id,),
            ).fetchone()
            total = len(eligible) + (int(previous[2]) if previous is not None else 0)
            if previous is None or (priority, source_id) < (previous[0], previous[1]):
                connection.execute(
                    "INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.user_id,
                        source_id,
                        event.impression_id,
                        event.timestamp.isoformat(),
                        label,
                        json.dumps(target_news_ids),
                        json.dumps(event.history_news_ids),
                        priority,
                        _priority(seed, "old-interest-class-cap", source_id),
                        total,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE candidates SET eligible_count = ? WHERE user_id = ?",
                    (total, event.user_id),
                )
        if counts["raw_train_impressions"] % 100_000 == 0:
            connection.commit()
            LOGGER.info(
                "Scanned %s Train impressions; %s eligible impression-category pairs",
                counts["raw_train_impressions"],
                counts["eligible_impression_category_pairs"],
            )
    connection.commit()
    connection.execute("CREATE INDEX candidate_class ON candidates(label, class_priority, source_id)")
    counts["raw_train_users"] = int(connection.execute("SELECT count(*) FROM users").fetchone()[0])
    counts["eligible_old_interest_users"] = int(
        connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
    )
    return counts, eligible_counts


def _select_rows(
    connection: sqlite3.Connection,
    *,
    catalog: Any,
    token_by_label: dict[str, str],
    cap: int,
    history_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    available = dict(connection.execute("SELECT label, count(*) FROM candidates GROUP BY label"))
    connection.row_factory = sqlite3.Row
    for label, token in token_by_label.items():
        for selected in connection.execute(
            "SELECT * FROM candidates WHERE label = ? ORDER BY class_priority, source_id LIMIT ?",
            (label, cap),
        ):
            history_ids = json.loads(selected["history_news_ids"])
            visible_ids = history_ids[-history_limit:]
            visible = [catalog.news[news_id] for news_id in visible_ids]
            if label not in {item.subcategory for item in visible}:
                raise AssertionError("Selected old target lacks visible History support")
            prompt = OLD_PROMPT_TEMPLATE.format(
                history="\n".join(
                    f"{index}. [{item.subcategory}] {item.title}"
                    for index, item in enumerate(visible, start=1)
                )
            )
            rows.append(
                {
                    "sample_id": f"{selected['source_id']}:behavior_interest",
                    "source_sample_id": selected["source_id"],
                    "split": "train",
                    "task_type": "behavior_interest",
                    "impression_id": selected["impression_id"],
                    "user_id": selected["user_id"],
                    "timestamp": datetime.fromisoformat(selected["timestamp"]),
                    "history_length": len(visible),
                    "prompt": prompt,
                    "response": token,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": token},
                    ],
                    "target_subcategory": label,
                }
            )
            provenance.append(
                {
                    "source_sample_id": selected["source_id"],
                    "user_id": selected["user_id"],
                    "impression_id": selected["impression_id"],
                    "target_subcategory": label,
                    "interest_token": token,
                    "target_clicked_news_ids": json.loads(selected["target_news_ids"]),
                    "full_history_news_ids": history_ids,
                    "visible_history_news_ids": visible_ids,
                    "full_history_subcategories": [catalog.news[n].subcategory for n in history_ids],
                    "visible_history_subcategories": [item.subcategory for item in visible],
                    "eligible_impression_category_pairs_for_user": selected["eligible_count"],
                    "user_selection_priority": selected["user_priority"],
                    "class_selection_priority": selected["class_priority"],
                }
            )
    return rows, provenance, {label: int(available.get(label, 0)) for label in token_by_label}


def exposure_budget(old_rows: int, new_rows: int, *, epochs: int, batch: int) -> dict[str, Any]:
    if min(old_rows, new_rows, epochs, batch) <= 0:
        raise ValueError("Data sizes, epochs, and effective batch must be positive")
    old_steps = epochs * math.ceil(old_rows / batch)
    new_steps = epochs * math.ceil(new_rows / batch)
    mixed_steps = epochs * math.ceil((old_rows + new_rows) / batch)
    return {
        "status": "planned_not_observed_training",
        "epochs_per_task": epochs,
        "effective_batch": batch,
        "old_presentations": epochs * old_rows,
        "new_presentations": epochs * new_rows,
        "mixed_task_row_ratio": [old_rows, new_rows],
        "mixed_old_fraction": old_rows / (old_rows + new_rows),
        "two_stage_optimizer_steps": [old_steps, new_steps],
        "mixed_optimizer_steps": mixed_steps,
        "matched_total_optimizer_steps": old_steps + new_steps == mixed_steps,
        "sampler_contract": "single GPU; shuffle without replacement; drop_last=false",
    }


def _commit(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    backup = output_dir.with_name(f".{output_dir.name}.backup-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(backup)
    had_output = output_dir.exists()
    if had_output:
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if had_output and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if had_output:
        shutil.rmtree(backup)


def build_dataset(
    *,
    raw_input_dir: Path | None,
    reference_root: Path,
    output_dir: Path,
    contract_file: Path,
    overwrite: bool = False,
    projected_input_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}; --overwrite must be explicit")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    old_config = contract["old_interest"]
    cap = int(old_config["max_per_label"])
    limit = int(old_config["require_visible_in_last_history_items"])
    if cap <= 0 or cap > 200 or limit <= 0 or limit > 50:
        raise ValueError("OI class cap must be in 1..200 and visible history in 1..50")
    if old_config["short_class_policy"] != "take_all_eligible" or old_config["global_equalization"]:
        raise ValueError("Approved OI cap retains all short classes without global equalization")
    if old_config["max_samples_per_user"] != 1:
        raise ValueError("Approved OI selection keeps one target per Train user")
    epochs = int(contract["methods"]["mixed"]["epochs"])
    if (
        int(contract["methods"]["two_stage"]["old_epochs"]) != epochs
        or int(contract["methods"]["two_stage"]["new_epochs"]) != epochs
    ):
        raise ValueError("Both methods must match each task's cumulative epochs")
    if (raw_input_dir is None) == (projected_input_dir is None):
        raise ValueError("Choose exactly one verified raw or static-projection source")
    hashes = _verify_inputs(raw_input_dir, reference_root, contract)
    train, tokens, ni_users, dev_users = _load_reference(reference_root, contract)
    projection_receipt = None
    if projected_input_dir is not None:
        catalog, projection_receipt = load_projection(projected_input_dir, contract_file=contract_file)
        hashes["verified_projection"] = {
            **projection_receipt["artifact_sha256"],
            "projection_receipt.json": sha256_file(projected_input_dir / "projection_receipt.json"),
        }
    else:
        catalog = load_news_catalog(raw_input_dir)
    missing = set(tokens) - {record.subcategory for record in catalog.news.values()}
    if missing:
        raise ValueError(f"Frozen labels absent from raw catalog: {sorted(missing)}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        database = staging / "selection.sqlite"
        with closing(sqlite3.connect(database)) as connection:
            scan_counts, candidate_counts = _scan_train(
                connection,
                raw_root=raw_input_dir or projected_input_dir,
                catalog=catalog,
                labels=set(tokens),
                dev_users=dev_users,
                history_limit=limit,
                seed=int(old_config["selection_seed"]),
                events=(iter_projection_events(projected_input_dir) if projected_input_dir is not None else None),
            )
            rows, provenance, available = _select_rows(
                connection,
                catalog=catalog,
                token_by_label=tokens,
                cap=cap,
                history_limit=limit,
            )
            missing_ni_users = [
                user for user in ni_users
                if connection.execute("SELECT 1 FROM users WHERE user_id = ?", (user,)).fetchone() is None
            ]
            if missing_ni_users:
                raise ValueError(f"Frozen NI users absent from raw Train: {len(missing_ni_users)}")
        if not rows:
            raise ValueError("No eligible old-interest training rows")
        old_users = {row["user_id"] for row in rows}
        if len(old_users) != len(rows) or old_users & dev_users:
            raise AssertionError("OI user uniqueness/isolation failed")
        selected_counts = Counter(row["target_subcategory"] for row in rows)
        for label in tokens:
            if selected_counts[label] != min(cap, available[label]):
                raise AssertionError("OI class cap or short-class retention failed")
        old_table = pa.Table.from_pylist(rows, schema=train.schema)
        old_table = old_table.append_column("training_task", pa.array(["old_interest"] * len(rows)))
        new_table = train.append_column("training_task", pa.array(["new_interest"] * train.num_rows))
        mixed = pa.concat_tables([old_table, new_table])
        order = list(range(mixed.num_rows))
        random.Random(int(contract["recipe"]["seed"])).shuffle(order)
        mixed = mixed.take(pa.array(order, type=pa.int64()))
        restored_new = mixed.filter(pc.equal(mixed["training_task"], "new_interest")).drop(["training_task"])
        if not restored_new.sort_by("sample_id").equals(train.sort_by("sample_id")):
            raise AssertionError("NI rows/messages changed in the mixed dataset")
        if len(set(mixed["sample_id"].to_pylist())) != mixed.num_rows:
            raise AssertionError("Mixed dataset sample IDs collide")
        pq.write_table(old_table, staging / "old_interest.parquet", compression="zstd")
        pq.write_table(mixed, staging / "mixed.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(provenance), staging / "old_selection_manifest.parquet", compression="zstd")
        database.unlink()
        per_class = {
            label: {
                "eligible_impression_category_pairs": int(candidate_counts[label]),
                "available_after_one_target_per_user": available[label],
                "selected": int(selected_counts[label]),
            }
            for label in tokens
        }
        budget = exposure_budget(
            len(rows), train.num_rows,
            epochs=epochs,
            batch=int(contract["recipe"]["per_device_train_batch_size"])
            * int(contract["recipe"]["gradient_accumulation_steps"]),
        )
        statistics = {
            "rule_version": RULE_VERSION,
            "raw_scan": dict(scan_counts),
            "old_rows": len(rows),
            "old_users": len(old_users),
            "old_supported_classes": len(selected_counts),
            "allowed_classes": len(tokens),
            "old_selected_class_min": min(selected_counts[label] for label in tokens),
            "old_selected_class_max": max(selected_counts[label] for label in tokens),
            "old_selected_classes_actually_balanced": len({selected_counts[label] for label in tokens}) == 1,
            "old_classes_without_support": [label for label in tokens if not selected_counts[label]],
            "new_rows": train.num_rows,
            "mixed_rows": mixed.num_rows,
            "old_new_overlapping_users": len(old_users & ni_users),
            "old_dev_overlapping_users": 0,
            "raw_train_dev_overlapping_users": 0,
            "new_rows_unchanged_in_mixed": True,
            "per_class": per_class,
            "exposure_budget": budget,
        }
        _write_json(staging / "statistics.json", statistics)
        _write_json(staging / "prompt_templates.json", {
            "old_interest": OLD_PROMPT_TEMPLATE,
            "new_interest": "unchanged frozen per-row NI prompt and messages",
            "future_target_news_in_prompt": False,
            "history_replay": False,
        })
        _write_json(staging / "metadata.json", {
            "rule_version": RULE_VERSION,
            "contract_sha256": sha256_file(contract_file),
            "builder_sha256": sha256_file(Path(__file__)),
            "input_sha256": hashes,
            "source_representation": "verified_static_task_projection" if projection_receipt is not None else "original_raw_tsv",
            "static_projection_equivalence": projection_receipt,
            "raw_source_identity": "MINDlarge_reclassified",
            "ni_stage2_train_file": portable_path(reference_root / NI_TRAIN, root=reference_root),
            "ni_stage2_train_sha256": sha256_file(reference_root / NI_TRAIN),
            "old_definition": "positive current-impression category in original static History and visible H50; target in frozen NI taxonomy",
            "sample_selection": old_config,
            "artifact_sha256": {
                path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
            },
        })
        # Independent Parquet readback before publishing the staging directory.
        if pq.read_table(staging / "old_interest.parquet").num_rows != len(rows):
            raise AssertionError("OI Parquet readback row mismatch")
        if pq.read_table(staging / "mixed.parquet").num_rows != len(rows) + train.num_rows:
            raise AssertionError("Mixed Parquet readback row mismatch")
        final_hashes = _verify_inputs(raw_input_dir, reference_root, contract)
        if projected_input_dir is not None:
            _, checked_receipt = load_projection(projected_input_dir, contract_file=contract_file)
            final_hashes["verified_projection"] = {
                **checked_receipt["artifact_sha256"],
                "projection_receipt.json": sha256_file(projected_input_dir / "projection_receipt.json"),
            }
        if final_hashes != hashes:
            raise ValueError("Frozen input files changed during the build")
        _commit(staging, output_dir, overwrite=overwrite)
        return statistics
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched old/new-interest SFT inputs without changing NI")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-input-dir", type=Path)
    source.add_argument("--projected-input-dir", type=Path)
    parser.add_argument("--reference-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/mind_old_new_interest_v1"))
    parser.add_argument("--contract-file", type=Path, default=Path("configs/experiments/mind_old_new_interest_v1.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = build_dataset(
        raw_input_dir=args.raw_input_dir,
        reference_root=args.reference_root,
        output_dir=args.output_dir,
        contract_file=args.contract_file,
        overwrite=args.overwrite,
        projected_input_dir=args.projected_input_dir,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
