"""Fixed within-user history permutations, preserving the accepted NI-only rows."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path

RULE_VERSION = "ni-history-order-v1"
TRAIN = Path("train/behavior_interest/21600.parquet")
DEV = Path("eval/behavior_interest/dev_full_grouped.parquet")
MAPPING = Path("interest_token_mapping.parquet")
DATA = Path("data/training/mind_history_order_20260923_v1")
SOURCE = Path("data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1")
SOURCE_HASHES = {
    str(TRAIN): "e76acd152ed6e0e92f8b8015dfbc599603267a480d9321b41b5acb95d10487f0",
    str(DEV): "c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd",
    str(MAPPING): "45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a",
}
MUTABLE = {"prompt", "messages", "history_subcategories"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_history(row):
    prefix, rest = row["prompt"].split("\n\n", 1)
    body, suffix = rest.rsplit("\n\n", 1)
    lines = body.splitlines()
    matches = [re.fullmatch(r"(\d+)\. (\[([^\]]+)\] .+)", line) for line in lines]
    if not 1 <= len(lines) <= 50 or len(lines) != row["history_length"]:
        raise ValueError("History length differs")
    if any(m is None or int(m[1]) != i for i, m in enumerate(matches, 1)):
        raise ValueError("History numbering differs")
    entries, categories = [m[2] for m in matches], [m[3] for m in matches]
    if "history_subcategories" in row and row["history_subcategories"] != categories:
        raise ValueError("History categories disagree with prompt")
    if "messages" in row:
        users = [m for m in row["messages"] if m["role"] == "user"]
        if len(users) != 1 or users[0]["content"] != row["prompt"]:
            raise ValueError("Messages disagree with prompt")
    return prefix, entries, categories, suffix


def permutation(row, split):
    if split not in {"train", "dev"}:
        raise ValueError("Expected train or dev domain")
    entries = parse_history(row)[1]
    history_digest = hashlib.sha256(canonical(entries)).hexdigest()
    keys = [hashlib.sha256(canonical([RULE_VERSION, 42, split, str(row["user_id"]), history_digest, i])).digest()
            for i in range(len(entries))]
    return sorted(range(len(entries)), key=lambda i: (keys[i], i)), history_digest


def shuffle_record(row, split):
    prefix, entries, categories, suffix = parse_history(row)
    order, digest = permutation(row, split)
    result = copy.deepcopy(row)
    result["prompt"] = prefix + "\n\n" + "\n".join(f"{i}. {entries[j]}" for i, j in enumerate(order, 1)) + "\n\n" + suffix
    if "history_subcategories" in result:
        result["history_subcategories"] = [categories[i] for i in order]
    if "messages" in result:
        next(m for m in result["messages"] if m["role"] == "user")["content"] = result["prompt"]
    return result, {"user_id": str(row["user_id"]), "history_sha256": digest, "indices": order}


def audit_pair(source, changed, split):
    if not source or len(source) != len(changed):
        raise ValueError("Row coverage differs")
    counts = Counter()
    seen = {}
    for a, b in zip(source, changed):
        pa_, ea, ca, sa = parse_history(a)
        pb, eb, cb, sb = parse_history(b)
        if set(a) != set(b) or any(a[k] != b[k] for k in a.keys() - MUTABLE):
            raise ValueError("Frozen identity or labels changed")
        if (pa_, sa) != (pb, sb) or Counter(ea) != Counter(eb) or Counter(ca) != Counter(cb):
            raise ValueError("Permutation changed content, frequency or instructions")
        if a.get("messages") and [m for m in a["messages"] if m["role"] != "user"] != [m for m in b["messages"] if m["role"] != "user"]:
            raise ValueError("Supervision changed")
        expected, ledger = shuffle_record(a, split)
        if b != expected:
            raise ValueError("Not the frozen permutation")
        key = (ledger["user_id"], ledger["history_sha256"])
        if key in seen and seen[key] != b["prompt"]:
            raise ValueError("Same user history has different permutations")
        seen[key] = b["prompt"]
        counts["rows"] += 1
        counts["changed_prompts"] += a["prompt"] != b["prompt"]
        counts["length_one"] += len(ea) == 1
        counts["identity_permutation_length_ge_2"] += len(ea) >= 2 and ledger["indices"] == list(range(len(ea)))
        counts["unchanged_text_nonidentity_permutation"] += a["prompt"] == b["prompt"] and ledger["indices"] != list(range(len(ea)))
    return {**dict(counts), "all_rows_input_only": True, "unique_user_histories": len(seen),
            "history_length_counts": dict(sorted(Counter(r["history_length"] for r in source).items()))}


def build(source=SOURCE, output=DATA):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    for name, expected in SOURCE_HASHES.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Baseline input identity differs: {name}")
    tables = {split: pq.read_table(source / rel) for split, rel in (("train", TRAIN), ("dev", DEV))}
    rows = {split: t.to_pylist() for split, t in tables.items()}
    tr, dv = rows["train"], rows["dev"]
    cls = Counter(r["target_subcategory"] for r in tr)
    if len(tr) != 21600 or len(cls) != 108 or set(cls.values()) != {200}:
        raise ValueError("Training balance differs")
    if len({r["sample_id"] for r in tr}) != 21600 or len({r["user_id"] for r in tr}) != 21600:
        raise ValueError("Training identities differ")
    if len(dv) != 31392 or len({r["impression_id"] for r in dv}) != 31392:
        raise ValueError("Dev coverage differs")
    support = Counter(t for r in dv for t in r["acceptable_subcategories"])
    if len(support) != 104 or sum(support.values()) != 37728:
        raise ValueError("Dev targets differ")
    if {r["user_id"] for r in tr} & {r["user_id"] for r in dv}:
        raise ValueError("Train/Dev user overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".history-order-", dir=output.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        report = {"rule_version": RULE_VERSION, "source": portable_path(source), "seed": 42,
                  "canonical_json": {"encoding": "utf-8", "ensure_ascii": False, "separators": [",", ":"], "position_base": 0},
                  "source_files": SOURCE_HASHES, "conditions": {}, "frozen_prompt_including_chronological_order": True}
        for split, rel in (("train", TRAIN), ("dev", DEV)):
            results = [shuffle_record(r, split) for r in rows[split]]
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([r[0] for r in results], schema=tables[split].schema), dest, compression="zstd")
            audit = audit_pair(rows[split], pq.read_table(dest).to_pylist(), split)
            ledger = [{**r[1], "record_index": i} for i, r in enumerate(results)]
            lp = stage / f"{split}_permutations.parquet"
            pq.write_table(pa.Table.from_pylist(ledger), lp, compression="zstd")
            report["conditions"][split] = {**audit, "sha256": sha256(dest), "permutations_sha256": sha256(lp)}
        shutil.copyfile(source / MAPPING, stage / MAPPING)
        (stage / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=DATA)
    args = p.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
