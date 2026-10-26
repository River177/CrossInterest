"""Freeze history-only per-user CIR support and complete NI epoch schedules."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import PROMPT
from new_interest.infrastructure.paths import portable_path

VERSION = "personal-cir-support-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_history(row):
    lines = re.findall(r"^(\d+)\. \[([^\]]+)\] (.+)$", row["prompt"], re.M)
    if len(lines) != row["history_length"] or [int(x[0]) for x in lines] != list(range(1, len(lines) + 1)):
        raise ValueError("History serialization/length differs")
    if not 1 <= len(lines) <= 50:
        raise ValueError("Expected frozen H50")
    return [(category, title) for _, category, title in lines]


def support_views(row, mapping, seed=42):
    """No query label, source sample id, timestamp or impression id is read."""
    history = parse_history(row)
    counts = Counter(c for c, _ in history)
    eligible = [c for c in counts if c in mapping and len(history) - counts[c] >= 5]
    h = digest(history)
    eligible.sort(key=lambda c: digest([VERSION, seed, row["user_id"], h, c]))
    views = []
    for c in eligible[:2]:
        retained = [(category, title) for category, title in history if category != c]
        body = "\n".join(f"{i}. [{category}] {title}" for i, (category, title) in enumerate(retained, 1))
        prompt = PROMPT.format(history=body)
        views.append({"user_id": row["user_id"], "history_length": len(retained),
                      "original_history_sha256": h, "removed_category": c,
                      "removed_records": counts[c], "prompt": prompt, "response": mapping[c],
                      "messages": [{"role": "user", "content": prompt},
                                   {"role": "assistant", "content": mapping[c]}]})
    return views, eligible


def epoch_plan(count, seed=42, epochs=3, batch=32):
    rows = []
    for epoch in range(epochs):
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, count, batch):
            rows.append({"step": len(rows) + 1, "epoch": epoch,
                         "row_indices": order[start:start + batch]})
    return rows


def build(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rows = pq.read_table(source / "ni.parquet").to_pylist()
    mapping = {r["target_subcategory"]: r["interest_token"] for r in
               pq.read_table(source / "interest_token_mapping.parquet").to_pylist()}
    if len(rows) != 21600 or len({r["user_id"] for r in rows}) != 21600 or len(mapping) != 108:
        raise ValueError("Frozen training identity/count differs")
    dev_users = set(pq.read_table(source / "dev_full_grouped.parquet", columns=["user_id"])["user_id"].to_pylist())
    if dev_users.intersection(r["user_id"] for r in rows):
        raise ValueError("Train/Dev user overlap")
    supports, users = [], []
    for index, row in enumerate(rows):
        views, eligible = support_views(row, mapping)
        # NI query truth is checked AFTER selection and never affects selection.
        if row["target_subcategory"] in {c for c, _ in parse_history(row)}:
            raise ValueError("NI target occurs in visible history")
        indices = []
        for view in views:
            sid = len(supports)
            supports.append({**view, "support_index": sid, "query_index": index,
                             "sample_id": f"personal-cir:{index}:{sid}"})
            indices.append(sid)
        users.append({"query_index": index, "user_id": row["user_id"],
                      "support_indices": indices, "eligible_categories": eligible})
    counts = dict(sorted(Counter(len(u["support_indices"]) for u in users).items()))
    if counts != {0: 4329, 1: 71, 2: 17200} or len(supports) != 34471:
        raise ValueError(f"Support eligibility differs: {counts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".personal-cir-", dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        for name in ["ni.parquet", "dev_full_grouped.parquet", "interest_token_mapping.parquet"]:
            shutil.copyfile(source / name, stage / name)
        pq.write_table(pa.Table.from_pylist(supports), stage / "supports.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(users), stage / "users.parquet", compression="zstd")
        plan = epoch_plan(len(rows))
        pq.write_table(pa.Table.from_pylist(plan), stage / "episodes.parquet", compression="zstd")
        metadata = {"version": VERSION, "seed": 42, "source": portable_path(source),
                    "query_users": len(rows), "support_rows": len(supports),
                    "support_count_histogram": counts, "epochs": 3, "steps": len(plan),
                    "query_presentations": 64800, "support_presentations": 3 * len(supports),
                    "selection_uses_future_label": False, "test_time_support": False,
                    "minimum_retained_behaviors": 5, "backfill": False,
                    "files": {p.name: file_hash(p) for p in sorted(stage.iterdir())}}
        (stage / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        stage.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
