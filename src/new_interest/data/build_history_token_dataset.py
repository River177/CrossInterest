"""Matched NI views with append-only, train-derived history category tokens."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.infrastructure.paths import portable_path
from new_interest.training.semantic_id_tokens import load_interest_token_mapping

RULE_VERSION = "ni-history-token-125-v1"
ENTRY = re.compile(r"^(\d+\. \[)([^\]\n]+)(\] )", re.M)


def history_categories(row):
    labels = [match[1] for match in ENTRY.findall(row["prompt"])]
    if len(labels) != row["history_length"]:
        raise ValueError("History parser does not cover every event")
    return labels


def replace_history(row, mapping):
    history_categories(row)
    result = dict(row)
    result["prompt"] = ENTRY.sub(lambda m: m[1] + mapping[m[2]] + m[3], row["prompt"])
    if "messages" in row:
        if row["messages"] != [{"role": "user", "content": row["prompt"]},
                               {"role": "assistant", "content": row["response"]}]:
            raise ValueError("Unexpected chat topology")
        result["messages"] = [{"role": "user", "content": result["prompt"]}, row["messages"][1]]
    # Exact inverse proves that titles, whitespace, order and instructions survived.
    inverse = {v: k for k, v in mapping.items()}
    restored = ENTRY.sub(lambda m: m[1] + inverse[m[2]] + m[3], result["prompt"])
    if restored != row["prompt"]:
        raise ValueError("Non-category input text changed")
    return result


def build(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    mapping_path = source / "interest_token_mapping.parquet"
    original = load_interest_token_mapping(mapping_path)
    train_table = pq.read_table(source / "ni.parquet")
    dev_table = pq.read_table(source / "dev_full_grouped.parquet")
    train, dev = train_table.to_pylist(), dev_table.to_pylist()
    if len(train) != 21600 or len(dev) != 31392 or len(original) != 108:
        raise ValueError("Unexpected frozen NI dataset")
    source_labels = {x for row in train for x in history_categories(row)}
    output_labels = {r["target_subcategory"] for r in original}
    extras = sorted(source_labels - output_labels)
    if len(source_labels) != 125 or len(extras) != 17:
        raise ValueError("Unexpected train-derived history vocabulary")
    extended = [{**r, "is_output_candidate": True} for r in original]
    extended += [{"interest_index": i, "interest_token": f"<interest_{i:03d}>",
                  "target_subcategory": label, "is_output_candidate": False}
                 for i, label in enumerate(extras, 108)]
    by_label = {r["target_subcategory"]: r["interest_token"] for r in extended}
    if any(x not in by_label for row in dev for x in history_categories(row)):
        raise ValueError("Dev history contains a category absent from frozen train vocabulary")
    counts = Counter(r["target_subcategory"] for r in train)
    if set(counts) != output_labels or set(counts.values()) != {200}:
        raise ValueError("Target balance changed")
    if len({r["source_sample_id"] for r in train}) != 21600:
        raise ValueError("Duplicate training sources")
    if {r["user_id"] for r in train} & {r["user_id"] for r in dev}:
        raise ValueError("Train/Dev user overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        schema = pq.read_schema(mapping_path).append(pa.field("is_output_candidate", pa.bool_(), nullable=False))
        pq.write_table(pa.Table.from_pylist(extended, schema=schema), staging / "model_token_mapping.parquet", compression="zstd")
        shutil.copyfile(mapping_path, staging / "interest_token_mapping.parquet")
        assert load_interest_token_mapping(staging / "model_token_mapping.parquet") == extended
        for arm in ("text", "token"):
            (staging / arm).mkdir()
            for name, table, rows in [("ni.parquet", train_table, train), ("dev_full_grouped.parquet", dev_table, dev)]:
                target = staging / arm / name
                if arm == "text":
                    shutil.copyfile(source / name, target)
                else:
                    updated = [replace_history(row, by_label) for row in rows]
                    pq.write_table(pa.Table.from_pylist(updated, schema=table.schema), target, compression="zstd")
                    loaded = pq.read_table(target).to_pylist()
                    if loaded != updated:
                        raise ValueError("Transformed data readback differs")
                    for before, after in zip(rows, loaded):
                        assert {k:v for k,v in before.items() if k not in {"prompt","messages"}} == {k:v for k,v in after.items() if k not in {"prompt","messages"}}
        stats = {"train_rows": len(train), "dev_rows": len(dev), "model_token_count":125,
                 "candidate_count":108, "appended_labels":extras, "mapping_source":"training history only",
                 "train_target_counts":dict(counts), "all_history_events_preserved":True,
                 "category_replacement_exactly_reversible":True, "prompt_instructions_unchanged":True,
                 "dev_supported_classes":len({x for row in dev for x in row["acceptable_subcategories"]})}
        write_json(staging / "statistics.json", stats)
        files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        write_json(staging / "metadata.json", {"rule_version":RULE_VERSION,"source":portable_path(source),
            "sources":{name:sha256(source/name) for name in ["ni.parquet","dev_full_grouped.parquet","interest_token_mapping.parquet"]},
            "files":files})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return stats


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/training/mind_cir_108_v1'))
    parser.add_argument('--output', type=Path, default=Path('data/training/mind_history_token_20260930_v1'))
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output),indent=2))
