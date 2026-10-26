"""Pair frozen CIR-as-NI with original future NI on identical source records."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import (
    SOURCE, history_from_record, history_key, make_view, sha256, write_json,
)
from new_interest.data.build_cir_ni_prompt_dataset import replace_prompt, template_parts
from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-ni-direct-20260929-v1"


def make_pair(cir, ni, source, mapping):
    """Reconstruct the frozen view before changing its task wording or metadata."""
    if cir["source_sample_id"] != ni["source_sample_id"] or source["sample_id"] != ni["source_sample_id"]:
        raise ValueError("Source pairing mismatch")
    for key in ("user_id", "impression_id", "timestamp"):
        if cir[key] != ni[key] or ni[key] != source[key]:
            raise ValueError("Paired identity differs: " + key)
    if ni["response"] != mapping[ni["target_subcategory"]]:
        raise ValueError("Original NI target/SID mismatch")
    if ni["target_subcategory"] in {h["subcategory"] for h in source["history"]}:
        raise ValueError("Original NI target appears in full history")
    if ni["messages"] != [{"role": "user", "content": ni["prompt"]},
                           {"role": "assistant", "content": ni["response"]}]:
        raise ValueError("Original NI messages differ")
    reconstructed, selection = make_view(source, ni, mapping, seed=42)
    if reconstructed != cir:
        raise ValueError("Frozen CIR differs from reconstructed category erasure")
    changed = replace_prompt(cir, template_parts(ni))
    # Both files expose exactly the same SFT schema and task metadata. Label
    # origins stay explicit in the sidecar, never in the model's prompt.
    cir_as_ni = {key: changed[key] for key in ni}
    cir_as_ni["training_task"] = "new_interest"
    original_ni = {**ni, "training_task": "new_interest"}
    pair = {
        "source_sample_id": ni["source_sample_id"], "user_id": ni["user_id"],
        "impression_id": ni["impression_id"], "timestamp": ni["timestamp"],
        "cir_sample_id": cir["sample_id"], "ni_sample_id": ni["sample_id"],
        "cir_target": cir["target_subcategory"], "ni_target": ni["target_subcategory"],
        "cir_label_origin": "erased_observed_history_category",
        "ni_label_origin": "original_future_clicked_new_interest",
        "cir_history_length": cir["history_length"], "ni_history_length": ni["history_length"],
        "cir_history_sha256": history_key(history_from_record(cir_as_ni)),
        "ni_history_sha256": history_key(history_from_record(ni)),
        "erased_target_clicks_h50": len(selection["deleted_positions_zero_based"]),
        "deleted_positions_zero_based": selection["deleted_positions_zero_based"],
        "retained_news_ids": selection["retained_news_ids"],
    }
    return cir_as_ni, original_ni, pair


def unique_index(rows, key):
    index = {row[key]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate identity: " + key)
    return index


def build(source_dir: Path, structured: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((source_dir / "metadata.json").read_text())
    for name, digest in metadata["files"].items():
        if sha256(source_dir / name) != digest:
            raise ValueError("Frozen source hash differs: " + name)
    source_identity = next(x for x in metadata["source_files"] if x["path"] == portable_path(structured))
    if sha256(structured) != source_identity["sha256"]:
        raise ValueError("Structured source hash differs")
    ni_table = pq.read_table(source_dir / "ni.parquet")
    ni_rows = ni_table.to_pylist()
    cir_rows = pq.read_table(source_dir / "cir.parquet").to_pylist()
    sources = unique_index(pq.read_table(structured).to_pylist(), "sample_id")
    ni_index = unique_index(ni_rows, "source_sample_id")
    unique_index(cir_rows, "source_sample_id")
    unique_index(cir_rows, "user_id")
    if len(ni_rows) != 21600 or len(cir_rows) != 17271 or set(sources) != set(ni_index):
        raise ValueError("Frozen cohort identity changed")
    mapping = {r["target_subcategory"]: r["interest_token"] for r in
               pq.read_table(source_dir / "interest_token_mapping.parquet").to_pylist()}
    if len(mapping) != 108 or len(set(mapping.values())) != 108:
        raise ValueError("SID mapping changed")
    selection = unique_index(pq.read_table(source_dir / "selection_manifest.parquet").to_pylist(), "source_sample_id")
    if {k for k,v in selection.items() if v["reason"] == "kept"} != {r["source_sample_id"] for r in cir_rows}:
        raise ValueError("CIR selection cohort differs")
    cir_out, ni_out, pairs = [], [], []
    for cir in cir_rows:
        key = cir["source_sample_id"]
        c, n, pair = make_pair(cir, ni_index[key], sources[key], mapping)
        old = selection[key]
        if (old["selected_category"] != pair["cir_target"] or
                old["deleted_positions_zero_based"] != pair["deleted_positions_zero_based"] or
                old["retained_news_ids"] != pair["retained_news_ids"]):
            raise ValueError("Frozen selection provenance differs")
        pair["pair_index"] = len(pairs)
        cir_out.append(c); ni_out.append(n); pairs.append(pair)
    templates = {template_parts(r) for r in [*cir_out, *ni_out]}
    if len(templates) != 1:
        raise ValueError("Auxiliary prompts do not share the frozen NI template")
    heldout_paths = [source_dir / "dev_full_grouped.parquet", *map(Path, metadata["heldout_files"])]
    train_users = {r["user_id"] for r in ni_out}
    heldout_users = set()
    for path in heldout_paths:
        heldout_users.update(pq.read_table(path, columns=["user_id"])["user_id"].to_pylist())
    if train_users & heldout_users:
        raise ValueError("Paired users overlap heldout users")
    stats = {"version": VERSION, "status": "constructed_and_read_back", "paired_rows": len(pairs),
             "paired_users": len(train_users), "original_ni_rows": len(ni_rows),
             "excluded_original_ni_rows": len(ni_rows)-len(pairs),
             "selection": "all_existing_frozen_CIR_sources_in_existing_CIR_order",
             "original_selection_exclusions": dict(Counter(x["reason"] for x in selection.values())),
             "same_source_identity_and_order": True, "same_NI_prompt_template": True,
             "class_or_length_rebalancing": False, "history_backfill": False,
             "future_NI_label_used_for_CIR_selection": False, "heldout_user_overlap": 0,
             "equal_target_pairs": sum(r["cir_target"] == r["ni_target"] for r in pairs),
             "erased_target_clicks_h50": dict(sorted(Counter(r["erased_target_clicks_h50"] for r in pairs).items())),
             "standalone_training_only": True, "training_submitted": False, "arms": {}}
    for name, rows in (("cir_as_ni", cir_out), ("original_ni_paired", ni_out)):
        counts = Counter(r["target_subcategory"] for r in rows)
        lengths = [r["history_length"] for r in rows]
        stats["arms"][name] = {"rows": len(rows), "classes": len(counts),
            "class_counts": dict(sorted(counts.items())), "min_class_count": min(counts.values()),
            "max_class_count": max(counts.values()), "mean_history_length": sum(lengths)/len(lengths),
            "min_history_length": min(lengths), "max_history_length": max(lengths)}
    stats["label_distribution_total_variation"] = sum(abs(
        stats["arms"]["cir_as_ni"]["class_counts"].get(c, 0)-
        stats["arms"]["original_ni_paired"]["class_counts"].get(c, 0)) for c in mapping)/(2*len(pairs))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cir-ni-direct-", dir=output.parent) as tmp:
        staging = Path(tmp)/"data"; staging.mkdir()
        schema = ni_table.schema.append(pa.field("training_task", pa.string()))
        for name, rows in (("cir_as_ni", cir_out), ("original_ni_paired", ni_out)):
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), staging/f"{name}.parquet", compression="zstd")
            if pq.read_table(staging/f"{name}.parquet").to_pylist() != rows:
                raise ValueError("Training data readback differs")
        pq.write_table(pa.Table.from_pylist(pairs), staging/"pairs.parquet", compression="zstd")
        if pq.read_table(staging/"pairs.parquet").to_pylist() != pairs:
            raise ValueError("Pair provenance readback differs")
        for name in ("dev_full_grouped.parquet", "interest_token_mapping.parquet"):
            shutil.copyfile(source_dir/name, staging/name)
            if sha256(source_dir/name) != sha256(staging/name):
                raise ValueError("Unchanged evaluation companion differs")
        write_json(staging/"statistics.json", stats)
        inputs = [structured, *[source_dir/n for n in metadata["files"]], source_dir/"metadata.json", *heldout_paths]
        prefix, suffix = next(iter(templates))
        write_json(staging/"metadata.json", {"version": VERSION,
            "prompt_template": prefix+"\n\n{history}\n\n"+suffix,
            "source_files": [{"path": portable_path(p), "sha256": sha256(p)} for p in dict.fromkeys(inputs)],
            "builder_sha256": sha256(Path(__file__)),
            "files": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()},
            "label_origin_in_pair_sidecar_only": True,
            "synthetic_CIR_label_is_not_observed_future_new_interest": True,
            "scope": "data_only_no_training_authorization_inferred"})
        staging.rename(output)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=Path("data/training/mind_cir_108_v1"))
    p.add_argument("--structured", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=Path("data/training/mind_cir_ni_direct_20260929_v1"))
    a = p.parse_args()
    print(json.dumps(build(a.source_dir, a.structured, a.output), ensure_ascii=False, indent=2))
