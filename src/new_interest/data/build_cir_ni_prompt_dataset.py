"""Replace only CIR task wording with the exact frozen NI prompt template."""
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

from new_interest.data.build_cir_dataset import history_from_record, sha256, write_json
from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-ni-prompt-20260928-v1"


def template_parts(row: dict) -> tuple[str, str]:
    prefix, rest = row["prompt"].split("\n\n", 1)
    body, suffix = rest.rsplit("\n\n", 1)
    if body != history_from_record(row):
        raise ValueError("History serialization differs")
    return prefix, suffix


def replace_prompt(row: dict, template: tuple[str, str]) -> dict:
    if row["messages"] != [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]}]:
        raise ValueError("Source prompt/messages differ")
    body = history_from_record(row)
    categories = [re.match(r"\d+\. \[([^\]]+)\] ", line).group(1)
                  for line in body.splitlines()]
    if row["target_subcategory"] in categories:
        raise ValueError("CIR target still appears in visible history")
    prompt = template[0] + "\n\n" + body + "\n\n" + template[1]
    return {**row, "prompt": prompt,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": row["response"]}]}


def build(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    cir = pq.read_table(source / "cir.parquet")
    ni = pq.read_table(source / "ni.parquet")
    originals, ni_rows = cir.to_pylist(), ni.to_pylist()
    templates = {template_parts(row) for row in ni_rows}
    if len(templates) != 1:
        raise ValueError("NI source does not have one frozen template")
    template = next(iter(templates))
    if len(originals) != 17271 or len(ni_rows) != 21600:
        raise ValueError("Frozen CIR/NI row counts changed")
    transformed = [replace_prompt(row, template) for row in originals]
    counts = Counter(row["target_subcategory"] for row in transformed)
    if len(counts) != 108:
        raise ValueError("Frozen target vocabulary changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cir-ni-prompt-", dir=output.parent) as tmp:
        staging = Path(tmp) / "data"
        staging.mkdir()
        pq.write_table(pa.Table.from_pylist(transformed, schema=cir.schema),
                       staging / "cir.parquet", compression="zstd")
        unchanged = ("ni.parquet", "dev_full_grouped.parquet", "interest_token_mapping.parquet",
                     "selection_manifest.parquet")
        for name in unchanged:
            shutil.copyfile(source / name, staging / name)
            if sha256(source / name) != sha256(staging / name):
                raise ValueError("Unchanged companion differs: " + name)
        actual = pq.read_table(staging / "cir.parquet")
        if not actual.schema.equals(cir.schema):
            raise ValueError("CIR schema changed")
        for old, new in zip(originals, actual.to_pylist()):
            changed = {key for key in old if old[key] != new[key]}
            if changed != {"prompt", "messages"}:
                raise ValueError("Changed non-prompt fields: " + repr(changed))
            if (history_from_record(old) != history_from_record(new)
                    or template_parts(new) != template or new != replace_prompt(old, template)):
                raise ValueError("Serialized template/history audit failed")
        stats = json.loads((source / "statistics.json").read_text())
        stats.update(version=VERSION, prompt_variant="exact_frozen_ni_template",
                     source_selection_version=stats["version"],
                     prompt_changed_rows=len(actual), history_and_targets_unchanged=True)
        write_json(staging / "statistics.json", stats)
        input_names = ("cir.parquet", "ni.parquet", "dev_full_grouped.parquet",
                       "interest_token_mapping.parquet", "selection_manifest.parquet",
                       "metadata.json", "statistics.json")
        metadata = {"version": VERSION, "prompt": "\n\n".join((template[0], "{history}", template[1])),
                    "source_files": [{"path": portable_path(source / name),
                                      "sha256": sha256(source / name)} for name in input_names],
                    "changed_fields": ["prompt", "messages"],
                    "row_order_ids_history_targets_and_task_labels_unchanged": True,
                    "synthetic_targets_are_not_actual_future_click_labels": True,
                    "files": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()}}
        write_json(staging / "metadata.json", metadata)
        staging.rename(output)
    return {"status": "passed", "version": VERSION, "cir_rows": len(originals),
            "ni_rows": len(ni_rows), "classes": len(counts),
            "changed_fields": metadata["changed_fields"],
            "metadata_sha256": sha256(output / "metadata.json"),
            "cir_sha256": sha256(output / "cir.parquet"),
            "unchanged_companions": {n: sha256(output / n) for n in unchanged}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
