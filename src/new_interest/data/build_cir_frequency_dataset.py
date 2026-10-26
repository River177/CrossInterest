"""Build paired CIR views differing only in eligible target frequency in H50."""
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
    MANIFEST_SCHEMA, PROMPT, history_from_record, history_key, make_view,
    render_history, sha256, write_json,
)
from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-frequency-paired-v1"
ARMS = ("singleton", "repeated")


def frequency_matches(count: int, arm: str) -> bool:
    if arm not in ARMS:
        raise ValueError(f"Unknown frequency arm: {arm}")
    return count == 1 if arm == "singleton" else count >= 4


def paired_views(source: dict, training: dict, mapping: dict, seed: int = 42):
    counts = Counter(x["subcategory"] for x in source["history"][-50:])
    # Keep the original seed/user/impression/category hash ranking. Only the
    # eligible label pool changes; neither arm uses future labels to choose it.
    return {arm: make_view(source, training, {
        c: token for c, token in mapping.items() if frequency_matches(counts[c], arm)
    }, seed) for arm in ARMS}


def check_view(source, row, manifest, mapping, arm):
    """Readback from structured source and serialized view, without make_view."""
    history = source["history"][-50:]
    category = row["target_subcategory"]
    positions = [i for i, item in enumerate(history) if item["subcategory"] == category]
    retained = [item for item in history if item["subcategory"] != category]
    if not frequency_matches(len(positions), arm) or len(retained) < 5:
        raise ValueError("Frequency or retained-history rule differs")
    if (row["user_id"] != source["user_id"] or row["impression_id"] != source["impression_id"]
            or row["timestamp"] != source["timestamp"] or row["source_sample_id"] != source["sample_id"]):
        raise ValueError("Source identity differs")
    body = render_history(retained)
    if (history_from_record(row) != body or row["prompt"] != PROMPT.format(history=body)
            or row["response"] != mapping[category] or row["history_length"] != len(retained)
            or row["messages"] != [{"role": "user", "content": row["prompt"]},
                                   {"role": "assistant", "content": mapping[category]}]):
        raise ValueError("Serialized prompt, target, or whole-category deletion differs")
    if (manifest["deleted_positions_zero_based"] != positions
            or manifest["selected_category"] != category or manifest["reason"] != "kept"
            or manifest["retained_news_ids"] != [x["news_id"] for x in retained]
            or manifest["history_sha256"] != history_key(body)
            or manifest["original_history_sha256"] != history_key(render_history(history))):
        raise ValueError("Selection provenance differs")


def build(project: Path, original: Path, output: Path, *, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((original / "metadata.json").read_text())
    for item in metadata["source_files"]:
        if sha256(project / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen source changed: {item['path']}")
    for name, digest in metadata["files"].items():
        if sha256(original / name) != digest:
            raise ValueError(f"Original CIR dataset changed: {name}")
    source_file = project / metadata["source_files"][0]["path"]
    sources = pq.read_table(source_file).to_pylist()
    source_by_id = {r["sample_id"]: r for r in sources}
    training = pq.read_table(original / "ni.parquet").to_pylist()
    mapping = {r["target_subcategory"]: r["interest_token"]
               for r in pq.read_table(original / "interest_token_mapping.parquet").to_pylist()}
    if (len(training) != 21600 or len({r["user_id"] for r in training}) != 21600
            or len(mapping) != 108 or len(source_by_id) != len(sources)
            or set(source_by_id) != {r["source_sample_id"] for r in training}):
        raise ValueError("Frozen source/NI/mapping scope differs")
    protected = set()
    for name in metadata["protected_files"]:
        for batch in pq.ParquetFile(project / name).iter_batches(batch_size=1000):
            protected.update(history_key(history_from_record(r)) for r in batch.to_pylist())
    heldout_users = set()
    heldout_files = [original / "dev_full_grouped.parquet"] + [project / p for p in metadata["heldout_files"]]
    for path in heldout_files:
        heldout_users.update(pq.read_table(path, columns=["user_id"])["user_id"].to_pylist())
    if heldout_users & {r["user_id"] for r in training}:
        raise ValueError("Source users overlap held-out users")
    rows, manifests = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    eligibility, reasons = Counter(), Counter()
    for ni in training:
        source = source_by_id[ni["source_sample_id"]]
        if ni["response"] != mapping[ni["target_subcategory"]] or ni["target_subcategory"] in {
            x["subcategory"] for x in source["history"]
        }:
            raise ValueError("Original NI target identity differs")
        views = paired_views(source, ni, mapping, seed)
        for arm in ARMS:
            eligibility[arm] += views[arm][0] is not None
        if any(views[arm][0] is None for arm in ARMS):
            reason = "missing_frequency_pool"
        elif any(views[arm][1]["history_sha256"] in protected for arm in ARMS):
            reason = "paired_protected_input_exclusion"
        else:
            reason = "kept"
        reasons[reason] += 1
        for arm, (row, manifest) in views.items():
            manifest["reason"] = reason
            manifests[arm].append(manifest)
            if reason == "kept":
                rows[arm].append(row)
    n = len(rows[ARMS[0]])
    if not n or any(len(rows[a]) != n for a in ARMS):
        raise ValueError("No equal-size paired data")
    paired_users = [r["user_id"] for r in rows[ARMS[0]]]
    if any([r["user_id"] for r in rows[a]] != paired_users for a in ARMS):
        raise ValueError("Paired user order differs")
    classes = {a: Counter(r["target_subcategory"] for r in rows[a]) for a in ARMS}
    tv = sum(abs(classes[ARMS[0]][c] - classes[ARMS[1]][c]) for c in mapping) / (2 * n)
    report = {"version": VERSION, "seed": seed, "source_users": len(training),
              "eligible_users_before_pairing": dict(eligibility), "pair_exclusion_counts": dict(reasons),
              "paired_users": n, "class_balancing": False,
              "selection": "original SHA256 minimum within each frequency pool; fixed view per user",
              "target_class_total_variation_distance": tv,
              "paired_users_sha256": history_key(json.dumps(paired_users, separators=(",", ":"))),
              "training_schedule": {"cir_epochs": 1, "ni_epochs": 3,
                                    "cir_optimizer_steps": (n + 31) // 32, "ni_optimizer_steps": 2025},
              "frequency_window": "last 50 behaviors before deletion; occurrences retain repeated news IDs",
              "comparison_limit": "Frequency policy also changes target-class distribution and deletion length",
              "original_metadata_sha256": sha256(original / "metadata.json"), "arms": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cir-frequency-", dir=output.parent) as tmp:
        staging = Path(tmp) / "paired"
        staging.mkdir()
        for arm in ARMS:
            folder = staging / arm
            folder.mkdir()
            kept = [m for m in manifests[arm] if m["reason"] == "kept"]
            deleted_counts = Counter(len(m["deleted_positions_zero_based"]) for m in kept)
            lengths = Counter(r["history_length"] for r in rows[arm])
            stats = {"version": VERSION, "arm": arm, "seed": seed, "source_rows": len(training),
                     "cir_rows": n, "cir_users": n, "label_vocabulary": 108,
                     "supported_cir_classes": len(classes[arm]), "class_counts": dict(sorted(classes[arm].items())),
                     "missing_cir_classes": sorted(set(mapping) - set(classes[arm])),
                     "class_balancing": False, "per_class_cap": None, "views_per_source_max": 1,
                     "future_target_used_for_selection": False, "minimum_retained_behaviors": 5,
                     "visible_history_limit": 50, "backfill": False, "heldout_user_overlap": 0,
                     "protected_input_overlap": 0, "frequency_min": 1 if arm == "singleton" else 4,
                     "frequency_max": 1 if arm == "singleton" else None,
                     "retained_length_counts": dict(sorted(lengths.items())),
                     "deleted_count_distribution": dict(sorted(deleted_counts.items())),
                     "mean_deleted_behaviors": sum(k * v for k, v in deleted_counts.items()) / n,
                     "mean_retained_behaviors": sum(k * v for k, v in lengths.items()) / n,
                     **report["training_schedule"], "global_batch_size": 32,
                     "pair_exclusion_counts": dict(reasons)}
            pq.write_table(pa.Table.from_pylist(rows[arm]), folder / "cir.parquet", compression="zstd")
            pq.write_table(pa.Table.from_pylist(manifests[arm], schema=MANIFEST_SCHEMA),
                           folder / "selection_manifest.parquet", compression="zstd")
            for name in ("ni.parquet", "dev_full_grouped.parquet", "interest_token_mapping.parquet"):
                shutil.copyfile(original / name, folder / name)
                if sha256(folder / name) != metadata["files"][name]:
                    raise ValueError(f"Frozen original data differs: {name}")
            write_json(folder / "statistics.json", stats)
            write_json(folder / "metadata.json", {
                "version": VERSION, "arm": arm, "prompt": PROMPT,
                "original_metadata": portable_path(original / "metadata.json"),
                "original_metadata_sha256": sha256(original / "metadata.json"),
                "source_files": metadata["source_files"], "protected_files": metadata["protected_files"],
                "heldout_files": metadata["heldout_files"], "paired_users_sha256": report["paired_users_sha256"],
                "files": {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}})
            # Independent disk readback reconstructs exactly which behaviors
            # should remain, with no use of the view constructor or selection hash.
            actual = pq.read_table(folder / "cir.parquet").to_pylist()
            manifest_by_id = {m["source_sample_id"]: m for m in
                              pq.read_table(folder / "selection_manifest.parquet").to_pylist()}
            if len(actual) != n or [r["user_id"] for r in actual] != paired_users:
                raise ValueError("Serialized pairing differs")
            for row in actual:
                check_view(source_by_id[row["source_sample_id"]], row,
                           manifest_by_id[row["source_sample_id"]], mapping, arm)
                if history_key(history_from_record(row)) in protected:
                    raise ValueError("Serialized view overlaps protected input")
            report["arms"][arm] = {"statistics": stats, "cir_sha256": sha256(folder / "cir.parquet"),
                                    "metadata_sha256": sha256(folder / "metadata.json"),
                                    "independent_view_readback": "passed"}
        write_json(staging / "paired_audit.json", report)
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--original", type=Path, default=Path("data/training/mind_cir_108_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/mind_cir_frequency_paired_v1"))
    args = parser.parse_args()
    result = build(args.project.resolve(), args.original.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
