"""Tokenize every shortened Train and Dev input using the pinned H50 tokenizer."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from new_interest.data.build_history_length_ablation import sha256
from new_interest.training.sft_data import audit_token_lengths
from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
snapshot = json.loads((args.bundle / "manifests/h50_snapshot.json").read_text())
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json"):
    assert sha256(args.tokenizer / name) == snapshot["files"][name], name
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
mapping = args.bundle / "data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1/interest_token_mapping.parquet"
register_semantic_interest_tokens(tokenizer, mapping)
reports = {}
for limit in (30, 10, 5):
    data = args.bundle / f"data/training/mind_history_length_ablation_20260915_v1/h{limit}"
    train = audit_token_lengths(data / "train/behavior_interest/21600.parquet", tokenizer,
                                max_seq_length=2048, enable_thinking=False)
    assert train["samples"] == 21600 and train["over_max_seq_length"] == 0
    assert train["min_assistant_tokens"] == train["max_assistant_tokens"] == 3
    rows = pq.read_table(data / "eval/behavior_interest/dev_full_grouped.parquet", columns=["prompt"]).to_pylist()
    lengths = []
    for row in rows:
        text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    assert len(lengths) == 31392 and 10 < min(lengths) <= max(lengths) <= 2048
    reports[f"h{limit}"] = {"train": train, "dev": {"rows": len(lengths), "max_tokens": max(lengths),
                            "min_tokens": min(lengths), "over_max_seq_length": sum(n > 2048 for n in lengths)}}
    print(limit, train["max_tokens"], max(lengths), flush=True)
args.output.write_text(json.dumps(reports, indent=2) + "\n")
