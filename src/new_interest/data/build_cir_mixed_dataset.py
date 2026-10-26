"""Exact CIR/NI union for uniform, single-stage mixed SFT."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.infrastructure.paths import portable_path


def build(cir_path: Path, ni_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    cir, ni = pq.read_table(cir_path), pq.read_table(ni_path)
    if "training_task" not in cir.column_names or set(cir["training_task"].to_pylist()) != {"cross_interest_reconstruction"}:
        raise ValueError("CIR task attribution differs")
    if "training_task" in ni.column_names:
        if set(ni["training_task"].to_pylist()) != {"new_interest"}:
            raise ValueError("NI task attribution differs")
        ni_tagged = ni
    else:
        ni_tagged = ni.append_column("training_task", pa.array(["new_interest"] * len(ni)))
    # Widen timestamp[ms] to [us] and int32 to int64; text and IDs stay exact.
    aligned_ni = ni_tagged.select(cir.column_names).cast(cir.schema, safe=True)
    mixed = pa.concat_tables([cir, aligned_ni])
    ids = mixed["sample_id"].to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("CIR/NI sample IDs collide")
    source_rows = cir.to_pylist() + ni.to_pylist()
    for original, actual in zip(source_rows, mixed.to_pylist()):
        if any(actual[k] != value for k, value in original.items()):
            raise ValueError("A source field changed during schema alignment")
    counts = Counter(mixed["training_task"].to_pylist())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        pq.write_table(mixed, staging / "mixed.parquet", compression="zstd")
        if not pq.read_table(staging / "mixed.parquet").equals(mixed):
            raise ValueError("Mixed Parquet readback differs")
        metadata = {"version": "mind-cir-ni-exact-union-v1", "rows": len(mixed), "by_task": dict(counts),
            "seed": 42, "sampling": "uniform global permutation each epoch, one exposure per row, no task balancing",
            "source_order": ["cir", "ni"], "optimizer_scheduler_reset": "none within mixed training",
            "source_fields_preserved": True, "duplicate_sample_ids": 0,
            "source_files": [{"path": portable_path(p), "sha256": sha256(p)} for p in (cir_path, ni_path)],
            "output_files": [{"path": portable_path(output / "mixed.parquet"), "sha256": sha256(staging / "mixed.parquet")}],
            "class_counts_by_task": {task: dict(Counter(row["target_subcategory"] for row in mixed.to_pylist()
                                            if row["training_task"] == task)) for task in counts}}
        write_json(staging / "metadata.json", metadata)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cir", type=Path, required=True)
    parser.add_argument("--ni", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(build(args.cir, args.ni, args.output), indent=2))
