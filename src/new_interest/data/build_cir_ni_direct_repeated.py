"""Filter already-paired direct supervision by the frozen CIR target frequency."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import SOURCE, sha256, write_json
from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-ni-direct-repeated-20260929-v1"


def select_pairs(cir, ni, pairs, minimum=2):
    if not isinstance(minimum, int) or minimum < 2:
        raise ValueError("Minimum frequency must be an integer >= 2")
    if not (len(cir) == len(ni) == len(pairs)):
        raise ValueError("Pair lengths differ")
    seen = set(); selected = []; audit = []
    for i, (c,n,p) in enumerate(zip(cir, ni, pairs)):
        key = p["source_sample_id"]
        if (key in seen or key != c["source_sample_id"] or key != n["source_sample_id"]
                or c["user_id"] != n["user_id"] or c["user_id"] != p["user_id"]
                or p["pair_index"] != i or p["cir_target"] != c["target_subcategory"]
                or p["ni_target"] != n["target_subcategory"]):
            raise ValueError("Pair identity or order differs")
        seen.add(key)
        count = p["erased_target_clicks_h50"]
        if not isinstance(count, int) or count < 1:
            raise ValueError("Invalid historical target frequency")
        keep = count >= minimum
        audit.append({"source_sample_id": key, "parent_pair_index": i,
                      "erased_target_clicks_h50": count, "kept": keep})
        if keep:
            selected.append(i)
    if not selected:
        raise ValueError("No repeated-target pairs remain")
    return selected, audit


def build(source: Path, output: Path, structured: Path = SOURCE):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((source/"metadata.json").read_text())
    for name,digest in meta["files"].items():
        if sha256(source/name) != digest:
            raise ValueError("Parent data hash differs: " + name)
    source_entry = next(x for x in meta["source_files"] if x["path"] == portable_path(structured))
    if sha256(structured) != source_entry["sha256"]:
        raise ValueError("Structured history hash differs")
    tables = {n:pq.read_table(source/f"{n}.parquet") for n in ("cir_as_ni", "original_ni_paired", "pairs")}
    rows = {n:t.to_pylist() for n,t in tables.items()}
    histories = {r["sample_id"]:r["history"] for r in pq.read_table(structured).to_pylist()}
    for p in rows["pairs"]:
        actual = sum(x["subcategory"] == p["cir_target"] for x in histories[p["source_sample_id"]][-50:])
        if actual != p["erased_target_clicks_h50"]:
            raise ValueError("Historical frequency differs from structured H50")
    indices, selection = select_pairs(rows["cir_as_ni"], rows["original_ni_paired"], rows["pairs"])
    out_tables = {n:tables[n].take(pa.array(indices)) for n in ("cir_as_ni", "original_ni_paired")}
    pairs = [{**rows["pairs"][i], "parent_pair_index": i, "pair_index": j} for j,i in enumerate(indices)]
    allowed = set(pq.read_table(source/"interest_token_mapping.parquet", columns=["target_subcategory"]).column(0).to_pylist())
    stats = {"version": VERSION, "paired_rows": len(pairs), "paired_users": len({p['user_id'] for p in pairs}),
        "parent_rows": len(rows["pairs"]), "excluded_singleton_pairs": len(rows["pairs"])-len(pairs),
        "minimum_target_occurrences_H50": 2, "target_reselection": False,
        "count_unit": "category-labelled history entries in original visible H50",
        "same_source_users_and_order": True, "prompt_and_training_rows_unchanged_from_parent": True,
        "training_submitted": False, "mapping_classes": len(allowed), "arms": {}}
    for name,table in out_tables.items():
        records = table.to_pylist();counts = Counter(r["target_subcategory"] for r in records)
        lengths = [r["history_length"] for r in records]
        stats["arms"][name] = {"rows": len(records), "classes": len(counts),
            "class_counts": dict(sorted(counts.items())), "missing_training_classes": sorted(allowed-set(counts)),
            "min_class_count": min(counts.values()), "max_class_count": max(counts.values()),
            "mean_history_length": sum(lengths)/len(lengths), "min_history_length": min(lengths), "max_history_length": max(lengths)}
    stats["erased_target_clicks_h50"] = dict(sorted(Counter(p["erased_target_clicks_h50"] for p in pairs).items()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cir-ni-repeated-",dir=output.parent) as tmp:
        staging = Path(tmp)/"data";staging.mkdir()
        for name,table in out_tables.items():
            pq.write_table(table,staging/f"{name}.parquet",compression="zstd")
            if not pq.read_table(staging/f"{name}.parquet").equals(table):
                raise ValueError("Filtered training readback differs")
        for name,records in (("pairs",pairs),("selection_manifest",selection)):
            pq.write_table(pa.Table.from_pylist(records),staging/f"{name}.parquet",compression="zstd")
            if pq.read_table(staging/f"{name}.parquet").to_pylist()!=records:
                raise ValueError("Pair sidecar readback differs")
        for name in ("dev_full_grouped.parquet","interest_token_mapping.parquet"):
            shutil.copyfile(source/name,staging/name)
            if sha256(staging/name)!=sha256(source/name):raise ValueError("Evaluation companion changed")
        write_json(staging/"statistics.json",stats)
        inputs = [source/name for name in meta["files"]]+[source/"metadata.json",structured]
        write_json(staging/"metadata.json",{"version":VERSION,"parent_dataset":portable_path(source),
            "rule":"Keep existing selected CIR targets occurring >=2 times in original H50; identical index filter on original NI.",
            "source_files":[{"path":portable_path(p),"sha256":sha256(p)} for p in inputs],
            "builder_sha256":sha256(Path(__file__)),"prompt_template":meta["prompt_template"],
            "files":{p.name:sha256(p) for p in staging.iterdir() if p.is_file()},
            "scope":"data_only", "training_submitted":False})
        staging.rename(output)
    return stats


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("data/training/mind_cir_ni_direct_20260929_v1"))
    p.add_argument("--output",type=Path,default=Path("data/training/mind_cir_ni_direct_repeated_20260929_v1"))
    p.add_argument("--structured",type=Path,default=SOURCE)
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.structured),ensure_ascii=False,indent=2))
