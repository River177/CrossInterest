"""Build one deterministic category-erased reconstruction view per NI source."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-108-v1"
SOURCE = Path("data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/train.parquet")
NI_DIR = Path("data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1")
NI = Path("train/behavior_interest/21600.parquet")
DEV = Path("eval/behavior_interest/dev_full_grouped.parquet")
MAPPING = Path("interest_token_mapping.parquet")
MANIFEST_SCHEMA = pa.schema([
    ("source_sample_id", pa.string()), ("user_id", pa.string()), ("selection_unit_id", pa.string()),
    ("original_history_sha256", pa.string()), ("original_history_length", pa.int32()),
    ("eligible_categories", pa.list_(pa.string())), ("selected_category", pa.string()), ("reason", pa.string()),
    ("deleted_positions_zero_based", pa.list_(pa.int32())), ("retained_news_ids", pa.list_(pa.string())),
    ("history_sha256", pa.string()), ("history_length", pa.int32()), ("matched_protected_files", pa.list_(pa.string())),
])
PROMPT = (
    "The following news browsing history has all entries from one topic category removed. "
    'The remaining entries are in chronological order, formatted as "[Topic category] Headline":\n\n'
    "{history}\n\n"
    "Infer the removed topic category from the remaining observed interests and news items. "
    "This is historical interest reconstruction, not a prediction of the next click. "
    "Respond with only its single-token Interest ID in the exact format <interest_NNN>. "
    "Do not output the category name or any explanation."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def render_history(history: list[dict]) -> str:
    if not history or any(not x.get("subcategory") or not x.get("title") for x in history):
        raise ValueError("History requires nonempty categories and headlines")
    return "\n".join(f"{i}. [{x['subcategory']}] {x['title']}" for i, x in enumerate(history, 1))


def history_from_record(row: dict) -> str:
    """Hash actual visible B+I content; ignore the task prompt and target label."""
    if "prompt" not in row:
        if "history" in row:
            return render_history(row["history"][-50:])
        titles, categories = row["history_titles"], row["history_subcategories"]
        if len(titles) != len(categories):
            raise ValueError("Structured held-out title/category counts differ")
        return render_history([{"title": title, "subcategory": category}
                               for title, category in zip(titles, categories)][-50:])
    _, body = row["prompt"].split("\n\n", 1)
    body, _ = body.rsplit("\n\n", 1)
    lines = body.splitlines()
    if not 1 <= len(lines) <= 50 or len(lines) != row.get("history_length", len(lines)) or any(
        re.fullmatch(rf"{i}\. \[[^\]]+\] .+", line) is None
        for i, line in enumerate(lines, 1)
    ):
        raise ValueError("Unexpected B+I history serialization")
    return body


def history_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_category(history: list[dict], labels: set[str], source_id: str, seed: int) -> tuple[str | None, list[str]]:
    counts = Counter(x["subcategory"] for x in history)
    candidates = sorted(c for c in counts if c in labels and len(history) - counts[c] >= 5
                        and any(other != c for other in counts))
    if not candidates:
        return None, candidates
    # Target-independent and invariant to source-file and candidate iteration order.
    selected = min(candidates, key=lambda c: hashlib.sha256(
        json.dumps([VERSION, seed, source_id, c], ensure_ascii=False, separators=(",", ":")).encode()
    ).digest())
    return selected, candidates


def make_view(source: dict, training: dict, mapping: dict[str, str], seed: int = 42):
    history = source["history"][-50:]
    if render_history(history) != history_from_record(training):
        raise ValueError(f"Structured source does not match frozen NI input: {source['sample_id']}")
    for field in ("user_id", "impression_id", "timestamp"):
        if source[field] != training[field]:
            raise ValueError(f"Source identity mismatch: {field}")
    # Source sample IDs contain the original future-target index. Use only the
    # user/impression identity so even that indirect label dependency is absent.
    unit = json.dumps([source["user_id"], source["impression_id"]], separators=(",", ":"))
    selected, candidates = select_category(history, set(mapping), unit, seed)
    manifest = {"source_sample_id": source["sample_id"], "user_id": source["user_id"],
                "selection_unit_id": unit, "original_history_sha256": history_key(render_history(history)),
                "original_history_length": len(history), "eligible_categories": candidates,
                "selected_category": selected, "reason": "no_eligible_category"}
    if selected is None:
        return None, manifest
    deleted = [i for i, x in enumerate(history) if x["subcategory"] == selected]
    retained = [x for x in history if x["subcategory"] != selected]
    body = render_history(retained)
    prompt, response = PROMPT.format(history=body), mapping[selected]
    manifest.update(reason="kept", deleted_positions_zero_based=deleted,
                    retained_news_ids=[x["news_id"] for x in retained],
                    history_sha256=history_key(body), history_length=len(retained))
    row = {key: training[key] for key in ("split", "task_type", "impression_id", "user_id", "timestamp")}
    row.update(sample_id=f"cir:{source['sample_id']}", source_sample_id=source["sample_id"],
               history_length=len(retained), prompt=prompt, response=response,
               messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
               target_subcategory=selected, training_task="cross_interest_reconstruction")
    return row, manifest


def build(source: Path, ni_dir: Path, output: Path, *, protected: list[Path],
          heldout: list[Path], seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(output)
    ni_file, dev_file, mapping_file = (ni_dir / x for x in (NI, DEV, MAPPING))
    ni_table = pq.read_table(ni_file)
    training = ni_table.to_pylist()
    source_rows = pq.read_table(source).to_pylist()
    mapping_rows = pq.read_table(mapping_file).to_pylist()
    mapping = {x["target_subcategory"]: x["interest_token"] for x in mapping_rows}
    if len(mapping) != 108 or len(set(mapping.values())) != 108:
        raise ValueError("Expected frozen 108-category SID mapping")
    if len(training) != 21600 or len({x["user_id"] for x in training}) != 21600:
        raise ValueError("Expected 21,600 unique NI training users")
    source_by_id = {x["sample_id"]: x for x in source_rows}
    if len(source_by_id) != len(source_rows) or set(source_by_id) != {x["source_sample_id"] for x in training}:
        raise ValueError("Structured input must exactly cover the frozen NI sources")
    for row in training:
        if row["response"] != mapping[row["target_subcategory"]]:
            raise ValueError("NI target/SID mismatch")
        if row["target_subcategory"] in {x["subcategory"] for x in source_by_id[row["source_sample_id"]]["history"]}:
            raise ValueError("NI target appears in complete original history")
    # Original NI/OI/Dev histories are read only for exact input exclusion.
    protection_files = list(dict.fromkeys([ni_file, dev_file, *protected, *heldout]))
    blocked: dict[str, set[str]] = {}
    for path in protection_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
            for row in batch.to_pylist():
                blocked.setdefault(history_key(history_from_record(row)), set()).add(portable_path(path))
    heldout_users = set()
    for path in [dev_file, *heldout]:
        heldout_users.update(pq.read_table(path, columns=["user_id"])["user_id"].to_pylist())
    if heldout_users.intersection(row["user_id"] for row in training):
        raise ValueError("Frozen NI train overlaps supplied held-out users")
    rows, manifests = [], []
    for training_row in training:
        source_row = source_by_id[training_row["source_sample_id"]]
        row, manifest = make_view(source_row, training_row, mapping, seed)
        if row is not None and manifest["history_sha256"] in blocked:
            manifest.update(reason="duplicates_original_input",
                            matched_protected_files=sorted(blocked[manifest["history_sha256"]]))
        elif row is not None:
            rows.append(row)
        manifests.append(manifest)
    if not rows:
        raise ValueError("No eligible CIR samples remain")
    counts = Counter(x["target_subcategory"] for x in rows)
    reasons = Counter(x["reason"] for x in manifests)
    keys = Counter(history_key(history_from_record(x)) for x in rows)
    report = {"version": VERSION, "seed": seed, "source_rows": len(training), "cir_rows": len(rows),
              "cir_users": len({x["user_id"] for x in rows}), "label_vocabulary": 108,
              "supported_cir_classes": len(counts), "class_counts": dict(sorted(counts.items())),
              "missing_cir_classes": sorted(set(mapping) - set(counts)), "exclusion_counts": dict(reasons),
              "class_balancing": False, "per_class_cap": None, "views_per_source_max": 1,
              "future_target_used_for_selection": False, "minimum_retained_behaviors": 5,
              "visible_history_limit": 50, "backfill": False, "heldout_user_overlap": 0,
              "protected_input_overlap": 0, "heldout_users": len(heldout_users),
              "distinct_cir_inputs": len(keys), "repeated_cir_input_rows": sum(n - 1 for n in keys.values()),
              "internal_duplicate_policy": "retain one view per source; do not add cross-source filtering",
              "retained_length_counts": dict(sorted(Counter(x["history_length"] for x in rows).items())),
              "cir_epochs": 1, "ni_epochs": 3, "global_batch_size": 32,
              "cir_optimizer_steps": (len(rows) + 31) // 32, "ni_optimizer_steps": 2025}
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = list(dict.fromkeys([source, mapping_file, *protection_files]))
    with tempfile.TemporaryDirectory(prefix=".cir-build-", dir=output.parent) as tmp:
        staging = Path(tmp) / "data"
        staging.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), staging / "cir.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(manifests, schema=MANIFEST_SCHEMA), staging / "selection_manifest.parquet", compression="zstd")
        for src, dest in ((ni_file, "ni.parquet"), (dev_file, "dev_full_grouped.parquet"), (mapping_file, str(MAPPING))):
            shutil.copyfile(src, staging / dest)
        write_json(staging / "statistics.json", report)
        metadata = {"version": VERSION, "prompt": PROMPT,
                    "source_files": [{"path": portable_path(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in inputs],
                    "protected_files": [portable_path(p) for p in protection_files],
                    "heldout_files": [portable_path(p) for p in heldout],
                    "files": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()}}
        write_json(staging / "metadata.json", metadata)
        # Independent disk readback checks the actual serialized output.
        actual = pq.read_table(staging / "cir.parquet").to_pylist()
        if actual != rows or not pq.read_table(staging / "ni.parquet").equals(ni_table):
            raise ValueError("Serialized training data differs")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--ni-dir", type=Path, default=NI_DIR)
    parser.add_argument("--output", type=Path, default=Path("data/training/mind_cir_108_v1"))
    parser.add_argument("--protected", type=Path, action="append", default=[])
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.ni_dir, args.output, protected=args.protected,
                           heldout=args.heldout, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
