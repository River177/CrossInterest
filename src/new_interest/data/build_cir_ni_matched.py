"""Build paired temporal NI prefixes and length-matched CIR histories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import (
    history_from_record, history_key, render_history, sha256, write_json,
)
from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-ni-matched-20260928-v1"
ARMS = ("ni_prefix", "cir_length_matched")
PROMPT = (
    "The headlines and corresponding topic categories of the news items I viewed, "
    'in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":\n\n'
    "{history}\n\n"
    "Based on both the headlines and their topic categories, infer an additional topic category "
    "that may be of interest to me and does not appear among the topic categories in the history above. "
    "Respond with only the learned interest token assigned to the inferred category, "
    "in the form <interest_NNN>."
)


def select_pair(full_history: list[dict], history: list[dict], category: str,
                user_id: str, *, seed: int = 42) -> tuple[dict | None, str]:
    if full_history[-50:] != history:
        raise ValueError("Full history does not reproduce H50")
    first = next((i for i, row in enumerate(full_history) if row["subcategory"] == category), None)
    if first is None:
        raise ValueError("CIR category absent from full source")
    offset = len(full_history) - len(history)
    cut = first - offset
    if cut < 0:
        return None, "target_seen_before_H50"
    if cut < 5:
        return None, "fewer_than_five_pre_target_events_in_H50"
    retained = [i for i, row in enumerate(history) if row["subcategory"] != category]
    if len(retained) < cut:
        raise ValueError("Erased history cannot supply matched length")
    # Original future NI labels and source_sample_id never enter this selection.
    def key(i):
        value = [VERSION, seed, user_id, category, i]
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).digest()
    selected = sorted(sorted(retained, key=key)[:cut])
    return {"first_target_position_full": first, "h50_offset": offset,
            "ni_positions": list(range(cut)), "cir_positions": selected,
            "history_length": cut, "cir_full_length": len(retained)}, "kept"


def training_row(original: dict, history: list[dict], positions: list[int], arm: str) -> dict:
    view = [history[i] for i in positions]
    target = original["target_subcategory"]
    if target in {row["subcategory"] for row in view}:
        raise ValueError("Auxiliary target remains in input")
    prompt = PROMPT.format(history=render_history(view))
    pair_key = hashlib.sha256(json.dumps(
        [VERSION, original["user_id"], original["impression_id"], target],
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {**original, "sample_id": "cir-ni-pair:" + pair_key,
            "history_length": len(view), "prompt": prompt,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": original["response"]}],
            "training_task": "new_interest" if arm == "ni_prefix" else "cross_interest_reconstruction"}


def build(project: Path, output: Path, *, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = project / "data/training/mind_cir_108_v1"
    h50_path = project / "data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/train.parquet"
    full_path = project / "data/processed/new_interest_static_user_disjoint_v1/train.parquet"
    original = pq.read_table(source / "cir.parquet")
    rows = original.to_pylist()
    if len(rows) != 17271 or len({r["user_id"] for r in rows}) != 17271:
        raise ValueError("Original CIR source identity/count changed")
    h50 = {r["user_id"]: r for r in pq.read_table(h50_path).to_pylist()}
    by_user = {r["user_id"]: r for r in rows}
    keys = pa.array(list(by_user))
    full = {}
    for batch in pq.ParquetFile(full_path).iter_batches(batch_size=8192, columns=["user_id", "sample_id", "history"]):
        table = pa.Table.from_batches([batch])
        for row in table.filter(pc.is_in(table["user_id"], value_set=keys)).to_pylist():
            if row["user_id"] in full:
                raise ValueError("Duplicate full-history user")
            full[row["user_id"]] = row
    if set(full) != set(by_user):
        raise ValueError("Incomplete full-history source coverage")
    mapping = {r["target_subcategory"]: r["interest_token"] for r in
               pq.read_table(source / "interest_token_mapping.parquet").to_pylist()}
    heldout_files = [source / "dev_full_grouped.parquet",
                     project / "artifacts/sid-relation-bias-20260915/inputs-v1/validation.parquet"]
    heldout_users, heldout_histories = set(), set()
    for path in heldout_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                heldout_users.add(row["user_id"])
                heldout_histories.add(history_key(history_from_record(row)))
    if set(by_user) & heldout_users:
        raise ValueError("Training user overlaps held-out users")
    arms = {arm: [] for arm in ARMS}
    pairs, exclusions = [], Counter()
    for row in rows:
        user, category = row["user_id"], row["target_subcategory"]
        if full[user]["sample_id"] != row["source_sample_id"] or h50[user]["sample_id"] != row["source_sample_id"]:
            raise ValueError("Source sample identity differs")
        if row["response"] != mapping[category]:
            raise ValueError("CIR label/SID identity differs")
        history = h50[user]["history"]
        expected_cir = render_history([e for e in history if e["subcategory"] != category])
        if expected_cir != history_from_record(row):
            raise ValueError("Original CIR is not the expected whole-category erasure")
        pair, reason = select_pair(full[user]["history"], history, category, user, seed=seed)
        exclusions[reason] += 1
        if pair is None:
            continue
        record = {"user_id": user, "source_sample_id": row["source_sample_id"],
                  "target_subcategory": category, "response": row["response"], **pair,
                  "full_history_sha256": history_key(render_history(full[user]["history"])),
                  "h50_sha256": history_key(render_history(history))}
        for arm, position_key in (("ni_prefix", "ni_positions"), ("cir_length_matched", "cir_positions")):
            new = training_row(row, history, pair[position_key], arm)
            hist_hash = history_key(history_from_record(new))
            if hist_hash in heldout_histories:
                raise ValueError("New auxiliary history exactly duplicates held-out input")
            record[arm + "_history_sha256"] = hist_hash
            record[arm + "_news_ids"] = [history[i]["news_id"] for i in pair[position_key]]
            arms[arm].append(new)
        record["pair_id"] = arms["ni_prefix"][-1]["sample_id"]
        record["identical_inputs"] = record["ni_prefix_history_sha256"] == record["cir_length_matched_history_sha256"]
        pairs.append(record)
    if len(pairs) != 9431 or len({r["target_subcategory"] for r in pairs}) != 108:
        raise ValueError("Previously audited 9431-pair/108-class scope changed")
    for left, right in zip(*[arms[a] for a in ARMS]):
        if any(left[k] != right[k] for k in ("sample_id", "user_id", "source_sample_id", "response", "target_subcategory", "history_length")):
            raise ValueError("Paired identities/labels/lengths differ")
    counts = Counter(r["target_subcategory"] for r in pairs)
    report = {"version": VERSION, "seed": seed, "source_rows": 17271,
              "paired_rows": len(pairs), "paired_users": len(pairs), "classes": len(counts),
              "class_counts": dict(sorted(counts.items())), "exclusion_counts": dict(exclusions),
              "identical_pair_inputs": sum(r["identical_inputs"] for r in pairs),
              "future_ni_target_used_for_selection": False, "heldout_user_overlap": 0,
              "heldout_exact_input_overlap": 0, "backfill": False,
              "common_prompt": PROMPT, "first_stage_epochs": 5, "second_stage_epochs": 3,
              "first_stage_presentations": len(pairs) * 5,
              "first_stage_optimizer_steps": ((len(pairs)+31)//32)*5,
              "second_stage_presentations": 21600*3, "second_stage_optimizer_steps": 2025}
    input_paths = [source / n for n in ("cir.parquet", "ni.parquet", "interest_token_mapping.parquet")]
    input_paths += [h50_path, full_path, *heldout_files]
    source_hashes = [{"path": portable_path(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in input_paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cir-ni-matched-", dir=output.parent) as tmp:
        staging = Path(tmp) / "data"
        staging.mkdir()
        pq.write_table(pa.Table.from_pylist(pairs), staging / "pair_manifest.parquet", compression="zstd")
        write_json(staging / "statistics.json", report)
        for arm in ARMS:
            dest = staging / arm
            dest.mkdir()
            # The accepted runner calls its auxiliary input cir.parquet in both arms;
            # training_task and the arm contract preserve actual scientific meaning.
            pq.write_table(pa.Table.from_pylist(arms[arm], schema=original.schema), dest / "cir.parquet", compression="zstd")
            for name in ("ni.parquet", "dev_full_grouped.parquet", "interest_token_mapping.parquet"):
                shutil.copyfile(source / name, dest / name)
            shutil.copyfile(staging / "pair_manifest.parquet", dest / "selection_manifest.parquet")
            stats = {**report, "arm": arm, "cir_rows": len(pairs), "cir_users": len(pairs),
                     "label_vocabulary": 108, "supported_cir_classes": 108,
                     "cir_epochs": 5, "ni_epochs": 3, "cir_optimizer_steps": report["first_stage_optimizer_steps"],
                     "ni_optimizer_steps": 2025, "retained_length_counts": dict(sorted(Counter(r["history_length"] for r in arms[arm]).items()))}
            write_json(dest / "statistics.json", stats)
            metadata = {"version": VERSION, "arm": arm, "prompt": PROMPT, "source_files": source_hashes,
                        "files": {p.name: sha256(p) for p in dest.iterdir() if p.is_file()}}
            write_json(dest / "metadata.json", metadata)
            if pq.read_table(dest / "cir.parquet").to_pylist() != arms[arm]:
                raise ValueError("Serialized auxiliary rows differ")
        write_json(staging / "metadata.json", {"version": VERSION, "source_files": source_hashes,
                   "files": {str(p.relative_to(staging)): sha256(p) for p in staging.rglob("*") if p.is_file()}})
        staging.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
