from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pyarrow.parquet as pq

from new_interest.data.build_wang_stage_a_dataset import RESPONSE_PREFIX
from new_interest.evaluation.evaluate_beam import (
    _atomic_write_json,
    _atomic_write_parquet,
    canonical_sha256,
    record_input_sha256,
    select_logical_shard,
    sha256_adapter_semantics,
    sha256_file,
)
from new_interest.training.sft_data import apply_chat_template


EVALUATION_IDENTITY_VERSION = "wang-stage-a-free-generation-v1"


def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    token_ids = [int(tokenizer.eos_token_id)]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        im_end_id = int(im_end_id)
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if im_end_id != unknown_id and im_end_id not in token_ids:
            token_ids.append(im_end_id)
    return token_ids


def parse_cluster_generation(
    output: str,
    *,
    labels: set[str],
    response_prefix: str = RESPONSE_PREFIX,
) -> dict[str, object]:
    stripped = output.strip()
    has_prefix = stripped.startswith(response_prefix)
    label = stripped[len(response_prefix) :].strip() if has_prefix else None
    valid_label = label in labels if label is not None else False
    return {
        "output": output,
        "stripped_output": stripped,
        "has_prefix": has_prefix,
        "parsed_label": label,
        "valid_label": valid_label,
        "under_30_words": len(stripped.split()) < 30,
    }


def evaluate_generations(
    records: Sequence[dict[str, object]],
    outputs: Sequence[str],
    *,
    labels: set[str],
    response_prefix: str = RESPONSE_PREFIX,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not records or len(records) != len(outputs):
        raise ValueError("records and outputs must have the same nonzero length")
    target_support: Counter[str] = Counter()
    target_hits: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    details: list[dict[str, object]] = []
    set_hits = 0
    recall_values: list[float] = []
    prefix_hits = 0
    match_hits = 0
    novelty_hits = 0
    length_hits = 0

    for record, output in zip(records, outputs):
        parsed = parse_cluster_generation(
            output,
            labels=labels,
            response_prefix=response_prefix,
        )
        targets = [str(value) for value in record["acceptable_subcategories"]]
        if not targets:
            raise ValueError("Evaluation record has no acceptable target")
        target_set = set(targets)
        history = {str(value) for value in record["history_subcategories"]}
        label = parsed["parsed_label"]
        hit = label in target_set if label is not None else False
        recall = (1.0 / len(target_set)) if hit else 0.0
        novelty = bool(parsed["valid_label"] and label not in history)
        set_hits += int(hit)
        recall_values.append(recall)
        prefix_hits += int(parsed["has_prefix"])
        match_hits += int(parsed["valid_label"])
        novelty_hits += int(novelty)
        length_hits += int(parsed["under_30_words"])
        if label is not None:
            prediction_counts[str(label)] += 1
        for target in target_set:
            target_support[target] += 1
            if label == target:
                target_hits[target] += 1
        details.append(
            {
                "impression_id": str(record["impression_id"]),
                "prompt": str(record["prompt"]),
                "history_subcategories": list(record["history_subcategories"]),
                "acceptable_subcategories": targets,
                **parsed,
                "set_hit_at_1": hit,
                "recall_at_1": recall,
                "novelty_compliant": novelty,
            }
        )

    per_class_recall = {
        target: target_hits[target] / support
        for target, support in sorted(target_support.items())
    }
    total = len(records)
    probabilities = [count / total for count in prediction_counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalization = math.log(len(labels)) if len(labels) > 1 else 1.0
    metrics = {
        "samples": total,
        "match_rate": match_hits / total,
        "valid_label_rate": match_hits / total,
        "prefix_compliance_rate": prefix_hits / total,
        "under_30_words_rate": length_hits / total,
        "set_hit_at_1": set_hits / total,
        "recall_at_1": mean(recall_values),
        "interest_macro_recall_at_1": mean(per_class_recall.values()),
        "novelty_compliance_rate": novelty_hits / total,
        "supported_interest_count": len(target_support),
        "per_class_recall_at_1": per_class_recall,
        "per_class_support": dict(sorted(target_support.items())),
        "prediction_counts": dict(
            sorted(prediction_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "unique_generated_labels": sum(
            label in labels for label in prediction_counts
        ),
        "generation_entropy": entropy,
        "normalized_generation_entropy": entropy / normalization,
    }
    return metrics, details


def generate_outputs(
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, object]],
    *,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> list[str]:
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    stop_token_ids = generation_stop_token_ids(tokenizer)
    outputs: list[str] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [
            apply_chat_template(
                tokenizer,
                [{"role": "user", "content": str(row["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for row in batch
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        if int(encoded["attention_mask"].sum(dim=1).max()) > max_seq_length:
            raise ValueError("Generation prompt exceeds max_seq_length")
        prompt_width = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )
        completions = generated[:, prompt_width:]
        outputs.extend(
            tokenizer.batch_decode(completions, skip_special_tokens=True)
        )
    return outputs


def _validate_checkpoint_rows(
    rows: Sequence[dict[str, object]],
    records: Sequence[dict[str, object]],
    indices: Sequence[int],
    input_hashes: Sequence[str],
    *,
    identity: str,
) -> None:
    if not (len(rows) == len(records) == len(indices) == len(input_hashes)):
        raise ValueError("Generation checkpoint row count mismatch")
    for row, record, index, input_hash in zip(
        rows, records, indices, input_hashes
    ):
        if int(row["record_index"]) != index:
            raise ValueError("Generation checkpoint index mismatch")
        if str(row["impression_id"]) != str(record["impression_id"]):
            raise ValueError("Generation checkpoint impression mismatch")
        if str(row["input_sha256"]) != input_hash:
            raise ValueError("Generation checkpoint input hash mismatch")
        if str(row["evaluation_identity"]) != identity:
            raise ValueError("Generation checkpoint identity mismatch")


def evaluate_wang_stage_a_generation(
    config: dict[str, Any],
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    model_revision = config.get("model_revision")
    adapter_value = config.get("adapter_path")
    adapter_path = Path(str(adapter_value)) if adapter_value else None
    eval_file = Path(config["eval_file"])
    labels_file = Path(config["labels_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    response_prefix = str(config.get("response_prefix", RESPONSE_PREFIX))
    batch_size = int(config.get("batch_size", 8))
    checkpoint_size = int(config.get("checkpoint_size", 64))
    max_seq_length = int(config.get("max_seq_length", 512))
    max_new_tokens = int(config.get("max_new_tokens", 32))
    enable_thinking = bool(config.get("enable_thinking", False))
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")

    tokenizer_source: str | Path = adapter_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=model_revision if adapter_path is None else None,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(torch.device("cuda", 0))
    model.eval()

    labels = sorted(
        {
            str(value)
            for value in pq.read_table(
                labels_file,
                columns=["target_subcategory"],
            )["target_subcategory"].to_pylist()
        }
    )
    all_records = pq.read_table(eval_file).to_pylist()
    max_examples = config.get("max_examples")
    if max_examples is not None:
        all_records = all_records[: int(max_examples)]
    indices, records = select_logical_shard(
        all_records,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if not records:
        raise ValueError("Logical generation shard is empty")

    adapter_sha256 = (
        sha256_adapter_semantics(adapter_path)
        if adapter_path is not None
        else None
    )
    identity_payload = {
        "identity_version": EVALUATION_IDENTITY_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "eval_sha256": sha256_file(eval_file),
        "labels_sha256": sha256_file(labels_file),
        "labels": labels,
        "response_prefix": response_prefix,
        "full_record_count": len(all_records),
        "max_examples": int(max_examples) if max_examples is not None else None,
        "num_shards": num_shards,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": enable_thinking,
        "decode": "greedy",
    }
    identity = canonical_sha256(identity_payload)
    input_hashes = [record_input_sha256(record) for record in records]
    checkpoint_dir = output_dir / "shards"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()
    for local_start in range(0, len(records), checkpoint_size):
        local_stop = min(local_start + checkpoint_size, len(records))
        local_records = records[local_start:local_stop]
        local_indices = indices[local_start:local_stop]
        local_hashes = input_hashes[local_start:local_stop]
        checkpoint_path = (
            checkpoint_dir / f"{local_start:09d}-{local_stop:09d}.parquet"
        )
        if checkpoint_path.is_file():
            rows = pq.read_table(checkpoint_path).to_pylist()
            _validate_checkpoint_rows(
                rows,
                local_records,
                local_indices,
                local_hashes,
                identity=identity,
            )
            resumed_samples += len(rows)
        else:
            batch_outputs = generate_outputs(
                model,
                tokenizer,
                local_records,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            rows = [
                {
                    "record_index": index,
                    "impression_id": str(record["impression_id"]),
                    "input_sha256": input_hash,
                    "evaluation_identity": identity,
                    "output": output,
                }
                for index, record, input_hash, output in zip(
                    local_indices,
                    local_records,
                    local_hashes,
                    batch_outputs,
                )
            ]
            _atomic_write_parquet(checkpoint_path, rows)
            generated_samples += len(rows)
        outputs.extend(str(row["output"]) for row in rows)

    metrics, details = evaluate_generations(
        records,
        outputs,
        labels=set(labels),
        response_prefix=response_prefix,
    )
    elapsed = time.time() - started_at
    metrics.update(
        {
            **identity_payload,
            "evaluation_identity": identity,
            "generated_samples_this_run": generated_samples,
            "resumed_samples": resumed_samples,
            "elapsed_seconds": elapsed,
            "examples_per_second": (
                generated_samples / elapsed if generated_samples else None
            ),
            "logical_shard_record_count": len(records),
        }
    )
    prediction_rows = [
        {
            "record_index": index,
            "input_sha256": input_hash,
            "evaluation_identity": identity,
            **detail,
        }
        for index, input_hash, detail in zip(indices, input_hashes, details)
    ]
    prediction_path = output_dir / "generation_predictions.parquet"
    _atomic_write_parquet(prediction_path, prediction_rows)
    _atomic_write_json(output_dir / "metrics.json", metrics)
    _atomic_write_json(output_dir / "resolved_config.json", config)
    _atomic_write_json(
        output_dir / "shard_manifest.json",
        {
            **identity_payload,
            "evaluation_identity": identity,
            "shard_index": shard_index,
            "prediction_sha256": sha256_file(prediction_path),
            "first_record_index": indices[0],
            "last_record_index": indices[-1],
            "logical_shard_record_count": len(records),
        },
    )
    return metrics


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Wang Stage-A free cluster generation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    config = _load_config(args.config)
    for key in (
        "adapter_path",
        "eval_file",
        "labels_file",
        "output_dir",
        "max_examples",
        "num_shards",
        "shard_index",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    metrics = evaluate_wang_stage_a_generation(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
