"""Truncate frozen H50 inputs without changing sample selection or supervision."""
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

RULE_VERSION = "frozen-h50-input-only-history-ablation-v1"
TRAIN = Path("train/behavior_interest/21600.parquet")
DEV = Path("eval/behavior_interest/dev_full_grouped.parquet")
MAPPING = Path("interest_token_mapping.parquet")
MUTABLE_FIELDS = {"prompt", "messages", "history_length", "history_subcategories"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncate_record(record: dict, limit: int) -> dict:
    if limit not in (5, 10, 30, 50):
        raise ValueError("Expected H5/H10/H30/H50")
    before, body = record["prompt"].split("\n\n", 1)
    history, after = body.rsplit("\n\n", 1)
    lines = history.splitlines()
    parsed = [re.fullmatch(r"(\d+)\. (\[([^\]]+)\] .+)", line) for line in lines]
    if (not lines or len(lines) != record["history_length"] or len(lines) > 50
            or any(match is None or int(match[1]) != i for i, match in enumerate(parsed, 1))):
        raise ValueError("Frozen H50 prompt does not have the expected numbered history")
    categories = [match[3] for match in parsed]
    if "history_subcategories" in record and record["history_subcategories"] != categories:
        raise ValueError("Prompt and structured history categories disagree")
    targets = record.get("acceptable_subcategories", [record.get("target_subcategory")])
    if any(target in categories for target in targets):
        raise ValueError("Frozen NI target occurs in H50 history")
    result = copy.deepcopy(record)
    retained = parsed[-limit:]
    rendered = "\n".join(f"{i}. {match[2]}" for i, match in enumerate(retained, 1))
    result["prompt"] = before + "\n\n" + rendered + "\n\n" + after
    result["history_length"] = len(retained)
    if "history_subcategories" in result:
        result["history_subcategories"] = categories[-limit:]
    if "messages" in result:
        users = [message for message in result["messages"] if message["role"] == "user"]
        if len(users) != 1 or users[0]["content"] != record["prompt"]:
            raise ValueError("Training messages do not match the frozen prompt")
        users[0]["content"] = result["prompt"]
    return result


def audit_pair(source: list[dict], target: list[dict], limit: int) -> dict:
    if len(source) != len(target) or not source:
        raise ValueError("Ablation changed row coverage")
    for original, changed in zip(source, target):
        if changed != truncate_record(original, limit):
            raise ValueError("Ablation changed fields beyond the last-H input operation")
        if any(original[k] != changed[k] for k in original.keys() - MUTABLE_FIELDS):
            raise ValueError("Frozen sample or label changed")
    return {"rows": len(target), "all_rows_exact_input_only_transform": True,
            "source_max_history": max(row["history_length"] for row in source),
            "max_history": max(row["history_length"] for row in target),
            "changed_rows": sum(a["prompt"] != b["prompt"] for a, b in zip(source, target)),
            "history_length_counts": dict(sorted(Counter(row["history_length"] for row in target).items())),
            "source_h50_rows": sum(row["history_length"] == 50 for row in source)}


def build(source: Path, output: Path, limits=(30, 10, 5)) -> dict:
    if output.exists():
        raise FileExistsError(output)
    tables = {"train": pq.read_table(source / TRAIN), "dev": pq.read_table(source / DEV)}
    records = {split: table.to_pylist() for split, table in tables.items()}
    if len(records["train"]) != 21600 or len(records["dev"]) != 31392:
        raise ValueError("Wrong NI-only baseline row counts")
    counts = Counter(row["target_subcategory"] for row in records["train"])
    if len(counts) != 108 or set(counts.values()) != {200}:
        raise ValueError("Wrong NI-only class balance")
    if len({row["sample_id"] for row in records["train"]}) != 21600:
        raise ValueError("Duplicate train sample")
    if len({row["impression_id"] for row in records["dev"]}) != 31392:
        raise ValueError("Duplicate Dev impression")
    if {row["user_id"] for row in records["train"]} & {row["user_id"] for row in records["dev"]}:
        raise ValueError("Train/Dev user overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".history-ablation-", dir=output.parent) as directory:
        stage = Path(directory) / "data"
        stage.mkdir()
        report = {"rule_version": RULE_VERSION, "source": portable_path(source),
                  "source_files": {str(name): sha256(source / name) for name in (TRAIN, DEV, MAPPING)},
                  "label_policy": "Keep original H50 samples and NI labels; never reclassify novelty",
                  "history_policy": "Recent min(H, original_length), preserve order and repetitions; renumber from one",
                  "conditions": {}}
        for limit in limits:
            root = stage / f"h{limit}"
            condition = {}
            for split, relative in (("train", TRAIN), ("dev", DEV)):
                rows = [truncate_record(row, limit) for row in records[split]]
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows, schema=tables[split].schema), dest, compression="zstd")
                condition[split] = audit_pair(records[split], pq.read_table(dest).to_pylist(), limit)
                condition[split]["sha256"] = sha256(dest)
            shutil.copyfile(source / MAPPING, root / MAPPING)
            (root / "metadata.json").write_text(json.dumps({**report, "history_limit": limit}, indent=2) + "\n")
            (root / "statistics.json").write_text(json.dumps(condition, indent=2) + "\n")
            report["conditions"][f"h{limit}"] = condition
        cohort = {"selection": "source H50 history_length == 50 (original history at least 50)",
                  "impression_ids": [row["impression_id"] for row in records["dev"] if row["history_length"] == 50]}
        (stage / "dev_h50_cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")
        (stage / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
