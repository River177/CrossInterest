from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    canonical_sha256,
    sha256_directory,
    sha256_file,
)
from new_interest.experiments.build_crossfit_weights import OOF_SCORE_SCHEMA
from new_interest.training.sft_data import apply_chat_template


SCORER_VERSION = "category-token-oof-nll-v2"


def category_token_encoding(
    tokenizer: Any,
    messages: list[dict[str, str]],
    target_subcategory: str,
    *,
    enable_thinking: bool = False,
) -> dict[str, object]:
    """Tokenize a full chat and locate only category-name tokens by offsets."""
    if len(messages) != 2 or messages[1].get("role") != "assistant":
        raise ValueError("Expected a user/assistant message pair")
    response = str(messages[1]["content"])
    expected = f"New topic: {target_subcategory}"
    if response != expected:
        raise ValueError(
            f"Assistant response does not exactly match target: {response!r}"
        )
    full_text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    response_start = full_text.rfind(response)
    if response_start < 0:
        raise ValueError("Rendered chat does not contain the assistant response")
    local_target_start = response.find(target_subcategory)
    category_start = response_start + local_target_start
    category_end = category_start + len(target_subcategory)
    try:
        encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError) as error:
        raise ValueError(
            "OOF category-token scoring requires tokenizer offset mappings"
        ) from error
    input_ids = [int(value) for value in encoding["input_ids"]]
    attention_mask = [
        int(value)
        for value in encoding.get("attention_mask", [1] * len(input_ids))
    ]
    offsets = [tuple(int(item) for item in pair) for pair in encoding["offset_mapping"]]
    if not (len(input_ids) == len(attention_mask) == len(offsets)):
        raise ValueError("Tokenizer output lengths differ")
    category_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > category_start and start < category_end
    ]
    if not category_positions or min(category_positions) <= 0:
        raise ValueError("Could not locate valid category-name token positions")
    covered_start = min(offsets[index][0] for index in category_positions)
    covered_end = max(offsets[index][1] for index in category_positions)
    if covered_start > category_start or covered_end < category_end:
        raise ValueError("Category token offsets do not cover the full name")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "category_positions": category_positions,
        "full_text": full_text,
    }


def _atomic_write_table(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=OOF_SCORE_SCHEMA),
            temporary,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_fold_identity(
    fold_manifest: Path,
) -> dict[str, dict[str, object]]:
    rows = pq.read_table(fold_manifest).to_pylist()
    indexed = {str(row["source_sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Fold manifest contains duplicate source identities")
    return indexed


def score_crossfit_sft(
    *,
    model_name: str,
    adapter_path: Path | str,
    heldout_file: Path | str,
    fold_manifest: Path | str,
    output_dir: Path | str,
    expected_fold: int,
    enable_thinking: bool = False,
    max_samples: int | None = None,
) -> dict[str, object]:
    import torch
    import torch.distributed as distributed
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    heldout_file = Path(heldout_file)
    fold_manifest = Path(fold_manifest)
    output_dir = Path(output_dir)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not distributed.is_initialized():
        distributed.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device: Any = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("Distributed OOF scoring requires CUDA")
        device = torch.device("cpu")

    rows = pq.read_table(heldout_file).to_pylist()
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        rows = rows[:max_samples]
    fold_identity = _load_fold_identity(fold_manifest)
    seen: set[str] = set()
    for row in rows:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"Held-out file duplicates {source_id}")
        seen.add(source_id)
        identity = fold_identity.get(source_id)
        if identity is None:
            raise ValueError(f"{source_id}: absent from fold manifest")
        for column in ("user_id", "target_subcategory"):
            if row[column] != identity[column]:
                raise ValueError(f"{source_id}: mismatch in {column}")
        if int(identity["fold"]) != expected_fold:
            raise ValueError(
                f"{source_id}: expected fold {expected_fold}, got "
                f"{identity['fold']}"
            )

    adapter_hash = sha256_directory(adapter_path)
    identity_payload = {
        "scorer_version": SCORER_VERSION,
        "model_name": model_name,
        "adapter_sha256": adapter_hash,
        "heldout_sha256": sha256_file(heldout_file),
        "fold_manifest_sha256": sha256_file(fold_manifest),
        "expected_fold": expected_fold,
        "heldout_rows": len(rows),
        "enable_thinking": enable_thinking,
    }
    teacher_identity = canonical_sha256(identity_payload)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()

    local_rows: list[dict[str, object]] = []
    for global_index, row in enumerate(rows):
        if global_index % world_size != rank:
            continue
        messages = [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in row["messages"]
        ]
        encoded = category_token_encoding(
            tokenizer,
            messages,
            str(row["target_subcategory"]),
            enable_thinking=enable_thinking,
        )
        input_ids = torch.tensor(
            [encoded["input_ids"]], dtype=torch.long, device=device
        )
        attention_mask = torch.tensor(
            [encoded["attention_mask"]], dtype=torch.long, device=device
        )
        positions = torch.tensor(
            encoded["category_positions"], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction_positions = positions - 1
            category_logits = outputs.logits[0, prediction_positions, :].float()
            category_targets = input_ids[0, positions]
            nll = functional.cross_entropy(
                category_logits,
                category_targets,
                reduction="mean",
            )
        nll_value = float(nll.detach().cpu())
        if not math.isfinite(nll_value) or nll_value < 0.0:
            raise ValueError(f"{row['source_sample_id']}: invalid NLL {nll_value}")
        identity = fold_identity[str(row["source_sample_id"])]
        local_rows.append(
            {
                "source_sample_id": str(row["source_sample_id"]),
                "user_id": str(row["user_id"]),
                "target_subcategory": str(row["target_subcategory"]),
                "fold": int(identity["fold"]),
                "category_token_count": len(encoded["category_positions"]),
                "category_token_nll": nll_value,
                "teacher_identity": teacher_identity,
            }
        )

    rank_path = output_dir / "rank_shards" / f"rank_{rank:02d}.parquet"
    _atomic_write_table(rank_path, local_rows)
    if world_size > 1:
        distributed.barrier()
    summary: dict[str, object] = {
        **identity_payload,
        "teacher_identity": teacher_identity,
        "world_size": world_size,
        "rank": rank,
        "rank_rows": len(local_rows),
        "rank_output": str(rank_path),
    }
    if rank == 0:
        merged: list[dict[str, object]] = []
        for shard_rank in range(world_size):
            shard_path = (
                output_dir / "rank_shards" / f"rank_{shard_rank:02d}.parquet"
            )
            merged.extend(pq.read_table(shard_path).to_pylist())
        merged_by_id = {
            str(row["source_sample_id"]): row for row in merged
        }
        if len(merged_by_id) != len(merged) or set(merged_by_id) != seen:
            raise ValueError("Merged OOF rank shards do not cover held-out rows")
        ordered = [merged_by_id[str(row["source_sample_id"])] for row in rows]
        merged_path = output_dir / "oof_scores.parquet"
        _atomic_write_table(merged_path, ordered)
        nll_values = [float(row["category_token_nll"]) for row in ordered]
        token_counts = [int(row["category_token_count"]) for row in ordered]
        summary.update(
            {
                "rows": len(ordered),
                "category_token_nll_min": min(nll_values),
                "category_token_nll_mean": sum(nll_values) / len(nll_values),
                "category_token_nll_max": max(nll_values),
                "category_token_count_min": min(token_counts),
                "category_token_count_max": max(token_counts),
                "oof_scores_sha256": sha256_file(merged_path),
            }
        )
        _atomic_write_json(output_dir / "score_summary.json", summary)
    if world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--heldout-file", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-fold", type=int, required=True)
    parser.add_argument("--max-samples", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = score_crossfit_sft(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        heldout_file=args.heldout_file,
        fold_manifest=args.fold_manifest,
        output_dir=args.output_dir,
        expected_fold=args.expected_fold,
        max_samples=args.max_samples,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
