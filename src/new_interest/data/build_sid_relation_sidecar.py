"""Freeze exact existing B+I text plus event/category token sidecars, without truncation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from new_interest.training.sft_data import apply_chat_template, tokenize_training_example


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def event_spans(prompt):
    hits = list(re.finditer(r"(?m)^(\d+)\. \[([^\]\n]+)\] ", prompt))
    if not hits or [int(x[1]) for x in hits] != list(range(1, len(hits)+1)):
        raise ValueError("Noncanonical numbered B+I history")
    end_history = prompt.index("\n\n", hits[-1].end())
    spans = []
    for i, hit in enumerate(hits):
        end = hits[i+1].start() if i+1 < len(hits) else end_history
        while end > hit.start() and prompt[end-1].isspace():
            end -= 1
        spans.append((hit.start(), end, hit[2]))
    return spans


def encode_row(row, tokenizer, category_index, training):
    prompt = row["prompt"]
    messages = [{"role": "user", "content": prompt}]
    prefix = apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=False)
    if prefix.count(prompt) != 1:
        raise ValueError("Prompt cannot be uniquely located in chat template")
    if training:
        messages.append({"role": "assistant", "content": row["response"]})
        text = apply_chat_template(tokenizer, messages, add_generation_prompt=False, enable_thinking=False)
        encoded = tokenize_training_example(tokenizer, messages, max_seq_length=2048, enable_thinking=False)
    else:
        text = prefix
        encoded = tokenizer(text, add_special_tokens=False)
    if len(encoded["input_ids"]) > 2048:
        raise ValueError(f"Overlength {row.get('source_sample_id', row.get('impression_id'))}")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if offsets["input_ids"] != encoded["input_ids"]:
        raise ValueError("Offset tokenization changes input")
    start = text.index(prompt)
    spans = [(a+start, b+start, c) for a,b,c in event_spans(prompt)]
    if "history_subcategories" in row and [c for _,_,c in spans] != row["history_subcategories"]:
        raise ValueError("Structured history categories differ from prompt")
    event_ids, source_ids = [], []
    for a,b in offsets["offset_mapping"]:
        intersect = [(i,c) for i,(s,e,c) in enumerate(spans)
                     if a < e and b > s and text[max(a,s):min(b,e)].strip()]
        if len(intersect) > 1:
            raise ValueError("Nonwhitespace token spans multiple events")
        if intersect:
            i,c = intersect[0];event_ids.append(i);source_ids.append(category_index[c])
        else:
            event_ids.append(-1);source_ids.append(-1)
    for i in range(len(spans)):
        if i not in event_ids:
            raise ValueError("Historical event lost during token binding")
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if encoded["input_ids"][:len(prefix_ids)] != prefix_ids:
        raise ValueError("Prefix mismatch")
    if training and any(x >= 0 for x in event_ids[len(prefix_ids):]):
        raise ValueError("Assistant token included in history")
    return {
        "source_sample_id": str(row.get("source_sample_id", row.get("impression_id"))),
        "user_id": row["user_id"], "impression_id": str(row["impression_id"]),
        "prompt": prompt, "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
        "labels": encoded.get("labels", [-100]*len(encoded["input_ids"])),
        "event_ids": event_ids, "source_ids": source_ids, "prefix_length": len(prefix_ids),
        "targets": row.get("acceptable_subcategories", [row.get("target_subcategory")]),
        "history_subcategories": [c for _,_,c in spans],
    }


def build(protocol_path, output):
    from transformers import AutoTokenizer
    protocol = json.loads(Path(protocol_path).read_text())
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    mapping = pq.read_table(protocol["mapping_file"]).to_pylist()
    token_by_name = {r["target_subcategory"]:r["interest_token"] for r in mapping}
    mixed = pq.read_table(protocol["training_source"]).to_pylist()
    train = [r for r in mixed if r["training_task"] == "new_interest"]
    dev = pq.read_table(protocol["dev_file"]).to_pylist()
    selection = json.loads(Path(protocol["validation_selection_file"]).read_text())["rows_selected"]
    requested = {r["sample_id"]:r for r in selection}
    selected = {}
    pf = pq.ParquetFile("data/processed/new_interest_static_user_disjoint_v1/train.parquet")
    value_set = pa.array(list(requested))
    for batch in pf.iter_batches(batch_size=8192):
        sub = batch.filter(pc.is_in(batch.column(batch.schema.get_field_index("sample_id")), value_set=value_set))
        for row in sub.to_pylist():
            selected[row["sample_id"]] = row
    if set(selected) != set(requested):
        raise ValueError("Selected validation IDs not found")
    # Exact NI instruction is inherited from the frozen existing input, not reworded.
    example = train[0]["prompt"]
    first = event_spans(example)[0][0]
    end = event_spans(example)[-1][1]
    header, suffix = example[:first], example[end:]
    validation = []
    for request in selection:
        raw = selected[request["sample_id"]]
        target = raw["target_interest"]["subcategory"]
        if target != request["target_subcategory"] or raw["user_id"] != request["user_id"]:
            raise ValueError("Validation selection identity differs")
        if target in {h["subcategory"] for h in raw["history"]}:
            raise ValueError("Validation target is history seen")
        hist = raw["history"][-50:]
        text = "\n".join(f'{i+1}. [{h["subcategory"]}] {h["title"]}' for i,h in enumerate(hist))
        validation.append(dict(source_sample_id=raw["sample_id"], user_id=raw["user_id"],
            impression_id=raw["impression_id"], prompt=header+text+suffix,
            acceptable_subcategories=[target], history_subcategories=[h["subcategory"] for h in hist]))
    categories = sorted({c for row in train+validation+dev for _,_,c in event_spans(row["prompt"])})
    if len(categories) != 125:
        raise ValueError(f"Source taxonomy changed: {len(categories)}")
    category_index = {c:i for i,c in enumerate(categories)}
    tokenizer = AutoTokenizer.from_pretrained(protocol["initial_adapter_dir"], local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token = tokenizer.eos_token
    source_map = [{"name":c,"sid_index":next((int(r['interest_index']) for r in mapping if r['target_subcategory']==c), -1)} for c in categories]
    json_write(output/"source_categories.json", source_map)
    audit = {}
    for split,rows in [("train",train),("validation",validation),("dev",dev)]:
        result = []
        for i,row in enumerate(rows):
            result.append(encode_row(row, tokenizer, category_index, split=="train"))
            if i%5000==0:print(f"ENCODE {split} {i}/{len(rows)}", flush=True)
        if split=="train":
            counts={sum(x!=-100 for x in r['labels']) for r in result}
            if counts!={3}:raise ValueError(f"Native supervision suffix changed: {counts}")
        pq.write_table(pa.Table.from_pylist(result), output/f"{split}.parquet", compression="zstd")
        audit[split] = {"rows":len(result),"max_tokens":max(len(r['input_ids']) for r in result),
            "users":len({r['user_id'] for r in result}),"targets":sum(len(r['targets']) for r in result),
            "sha256":sha(output/f"{split}.parquet")}
    if (audit['train']['rows'],audit['validation']['rows'],audit['dev']['rows'])!=(21600,5354,31392):
        raise ValueError("Wrong cohort sizes")
    if {r['user_id'] for r in validation}&({r['user_id'] for r in mixed}|{r['user_id'] for r in dev}):
        raise ValueError("Validation user leakage")
    tokenizer.save_pretrained(output/"tokenizer")
    json_write(output/"audit.json", {"protocol_sha256":sha(protocol_path),"splits":audit,
        "source_categories":125,"sid_classes":108,"no_truncation":True,"validation_user_disjoint":True})
    print(json.dumps(audit,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();build(args.protocol,args.output)


if __name__=='__main__':main()
