from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    _atomic_write_json,
    _atomic_write_parquet,
    canonical_sha256,
    record_input_sha256,
    select_logical_shard,
    sha256_directory,
    sha256_file,
)
from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.training.semantic_id_tokens import (
    load_interest_token_mapping,
    prepare_semantic_adapter_base,
    validate_saved_interest_tokens,
)
from new_interest.training.sft_data import apply_chat_template


EVALUATION_IDENTITY_VERSION = "semantic-id-token-next-logits-v2"


def rank_restricted_logits(
    logits: Any,
    token_ids: list[int],
    labels: list[str],
    tokens: list[str],
    *,
    top_k: int,
) -> tuple[list[list[dict[str, object]]], list[float]]:
    import torch

    if logits.ndim != 2:
        raise ValueError("Expected [batch, vocabulary] logits")
    if not (len(token_ids) == len(labels) == len(tokens)):
        raise ValueError("Token IDs, labels, and tokens must have equal length")
    if not 0 < top_k <= len(token_ids):
        raise ValueError("top_k must be within the restricted vocabulary")
    index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    restricted = logits.index_select(1, index).float()
    log_probabilities = torch.log_softmax(restricted, dim=1)
    probabilities = log_probabilities.exp()
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    values, offsets = torch.topk(
        log_probabilities,
        k=top_k,
        dim=1,
        largest=True,
        sorted=True,
    )
    ranked: list[list[dict[str, object]]] = []
    for row_offsets, row_values in zip(
        offsets.detach().cpu().tolist(),
        values.detach().cpu().tolist(),
    ):
        ranked.append(
            [
                {
                    "label": labels[offset],
                    "response": tokens[offset],
                    "token": tokens[offset],
                    "token_id": token_ids[offset],
                    "score": float(score),
                }
                for offset, score in zip(row_offsets, row_values)
            ]
        )
    return ranked, [float(value) for value in entropies.cpu().tolist()]


def generate_id_token_rankings(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    *,
    token_ids: list[int],
    labels: list[str],
    tokens: list[str],
    batch_size: int,
    top_k: int,
    enable_thinking: bool,
) -> tuple[list[list[dict[str, object]]], list[float]]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    outputs: list[list[dict[str, object]]] = []
    entropies: list[float] = []
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
        encoded = {
            key: value.to(model.device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, -1, :]
        ranked, batch_entropies = rank_restricted_logits(
            logits,
            token_ids,
            labels,
            tokens,
            top_k=top_k,
        )
        outputs.extend(ranked)
        entropies.extend(batch_entropies)
    return outputs, entropies


def _validate_checkpoint_rows(
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    global_indices: list[int],
    input_hashes: list[str],
    *,
    evaluation_identity: str,
    token_by_label: dict[str, str],
    top_k: int,
    context: str,
) -> None:
    if len(rows) != len(records):
        raise ValueError(f"{context}: checkpoint row count mismatch")
    for offset, (row, record) in enumerate(zip(rows, records)):
        if int(row["record_index"]) != global_indices[offset]:
            raise ValueError(f"{context}: record index mismatch")
        if str(row["impression_id"]) != str(record["impression_id"]):
            raise ValueError(f"{context}: impression ID mismatch")
        if str(row["input_sha256"]) != input_hashes[offset]:
            raise ValueError(f"{context}: input hash mismatch")
        if str(row["evaluation_identity"]) != evaluation_identity:
            raise ValueError(f"{context}: evaluation identity mismatch")
        labels = [str(value) for value in row["ranked_labels"]]
        tokens = [str(value) for value in row["ranked_tokens"]]
        scores = [float(value) for value in row["ranked_scores"]]
        if not len(labels) == len(tokens) == len(scores) == top_k:
            raise ValueError(f"{context}: invalid Top-{top_k} output")
        if len(set(labels)) != top_k or len(set(tokens)) != top_k:
            raise ValueError(f"{context}: duplicate ranked output")
        if tokens != [token_by_label.get(label) for label in labels]:
            raise ValueError(f"{context}: token/label mapping mismatch")
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"{context}: non-finite restricted score")


def evaluate_id_token_ranking(config: dict[str, Any]) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    adapter_path = Path(config["adapter_path"])
    eval_file = Path(config["eval_file"])
    mapping_file = Path(config["interest_token_mapping_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k = int(config.get("top_k", 20))
    cutoffs = tuple(
        int(value) for value in config.get("cutoffs", (1, 5, 10, 20))
    )
    if max(cutoffs) > top_k:
        raise ValueError("Metric cutoff cannot exceed top_k")
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    batch_size = int(config.get("batch_size", 8))
    checkpoint_size = int(config.get("checkpoint_size", 256))
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_mapping_file = Path(config.get("model_interest_token_mapping_file", mapping_file))
    spec = validate_saved_interest_tokens(
        tokenizer,
        model_mapping_file,
        expected_count=int(config.get("semantic_id_token_count", 108)),
    )
    mapping_rows = load_interest_token_mapping(mapping_file)
    labels = [str(row["target_subcategory"]) for row in mapping_rows]
    tokens = [str(row["interest_token"]) for row in mapping_rows]
    entries = {entry.interest_token: entry for entry in spec.entries}
    if any(token not in entries or entries[token].target_subcategory != label for token, label in zip(tokens, labels)):
        raise ValueError("Candidate mapping does not match model vocabulary")
    token_ids = [entries[token].token_id for token in tokens]
    token_by_label = dict(zip(labels, tokens))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=config.get("model_revision"),
        token=config.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    semantic_initialization_mode = str(
        config.get(
            "semantic_id_initialization_mode",
            "mean_norm_matched",
        )
    )
    semantic_base_audit = prepare_semantic_adapter_base(
        model,
        spec,
        initialization_mode=semantic_initialization_mode,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(torch.device("cuda", 0))
    model.eval()

    all_records = pq.read_table(eval_file).to_pylist()
    max_examples = config.get("max_examples")
    if max_examples is not None:
        all_records = all_records[: int(max_examples)]
    global_indices, records = select_logical_shard(
        all_records,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if not records:
        raise ValueError(f"Logical shard {shard_index}/{num_shards} is empty")

    identity_payload = {
        "identity_version": EVALUATION_IDENTITY_VERSION,
        "model_name": model_name,
        "adapter_sha256": sha256_directory(adapter_path),
        "eval_sha256": sha256_file(eval_file),
        "mapping_sha256": sha256_file(mapping_file),
        "full_record_count": len(all_records),
        "labels": labels,
        "tokens": tokens,
        "token_ids": token_ids,
        "semantic_initialization_version": semantic_base_audit[
            "initialization_version"
        ],
        "model_load_path": "cpu-construct-adapter-then-cuda",
        "semantic_base_tied_embeddings": bool(
            semantic_base_audit["tie_word_embeddings"]
        ),
        "max_examples": int(max_examples) if max_examples is not None else None,
        "num_shards": num_shards,
        "top_k": top_k,
        "cutoffs": list(cutoffs),
        "enable_thinking": bool(config.get("enable_thinking", False)),
    }
    if semantic_initialization_mode != "mean_norm_matched":
        identity_payload["semantic_initialization_mode"] = (
            semantic_initialization_mode
        )
    if config.get("model_revision") is not None:
        identity_payload["model_revision"] = str(config["model_revision"])
    identity_payload["model_mapping_sha256"] = sha256_file(model_mapping_file)
    identity_payload["model_interest_token_count"] = len(spec.entries)
    evaluation_identity = canonical_sha256(identity_payload)
    input_hashes = [record_input_sha256(record) for record in records]
    checkpoint_dir = output_dir / "shards"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ranked_outputs: list[list[dict[str, object]]] = []
    restricted_entropies: list[float] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()

    for local_start in range(0, len(records), checkpoint_size):
        local_stop = min(local_start + checkpoint_size, len(records))
        checkpoint_path = (
            checkpoint_dir / f"{local_start:09d}-{local_stop:09d}.parquet"
        )
        local_records = records[local_start:local_stop]
        local_indices = global_indices[local_start:local_stop]
        local_hashes = input_hashes[local_start:local_stop]
        if checkpoint_path.is_file():
            rows = pq.read_table(checkpoint_path).to_pylist()
            _validate_checkpoint_rows(
                rows,
                local_records,
                local_indices,
                local_hashes,
                evaluation_identity=evaluation_identity,
                token_by_label=token_by_label,
                top_k=top_k,
                context=str(checkpoint_path),
            )
            resumed_samples += len(rows)
        else:
            batch_ranked, batch_entropies = generate_id_token_rankings(
                model,
                tokenizer,
                local_records,
                token_ids=token_ids,
                labels=labels,
                tokens=tokens,
                batch_size=batch_size,
                top_k=top_k,
                enable_thinking=bool(config.get("enable_thinking", False)),
            )
            rows = [
                {
                    "record_index": record_index,
                    "impression_id": str(record["impression_id"]),
                    "input_sha256": input_hash,
                    "evaluation_identity": evaluation_identity,
                    "ranked_tokens": [
                        str(item["token"]) for item in ranked
                    ],
                    "ranked_token_ids": [
                        int(item["token_id"]) for item in ranked
                    ],
                    "ranked_labels": [
                        str(item["label"]) for item in ranked
                    ],
                    "ranked_scores": [
                        float(item["score"]) for item in ranked
                    ],
                    "restricted_entropy": float(entropy),
                }
                for record_index, record, input_hash, ranked, entropy in zip(
                    local_indices,
                    local_records,
                    local_hashes,
                    batch_ranked,
                    batch_entropies,
                )
            ]
            _atomic_write_parquet(checkpoint_path, rows)
            generated_samples += len(rows)
        ranked_outputs.extend(
            [
                [
                    {
                        "label": label,
                        "response": token,
                        "score": score,
                    }
                    for label, token, score in zip(
                        row["ranked_labels"],
                        row["ranked_tokens"],
                        row["ranked_scores"],
                    )
                ]
                for row in rows
            ]
        )
        restricted_entropies.extend(
            float(row["restricted_entropy"]) for row in rows
        )

    elapsed = time.time() - started_at
    metrics, details = evaluate_ranked_outputs(
        records,
        ranked_outputs,
        cutoffs=cutoffs,
    )
    metrics.update(
        {
            **identity_payload,
            "evaluation_identity": evaluation_identity,
            "samples": len(records),
            "shard_index": shard_index,
            "logical_num_shards": num_shards,
            "generated_samples": generated_samples,
            "resumed_samples": resumed_samples,
            "elapsed_seconds": elapsed,
            "examples_per_second": len(records) / elapsed if elapsed else None,
            "mean_restricted_entropy": (
                sum(restricted_entropies) / len(restricted_entropies)
            ),
            "label_vocabulary_size": len(labels),
            "history_prediction_filtering": False,
        }
    )
    prediction_rows = [
        {
            "record_index": global_index,
            "input_sha256": input_hash,
            "evaluation_identity": evaluation_identity,
            **detail,
            "ranked_tokens": [
                str(item["response"]) for item in ranked
            ],
            "restricted_entropy": entropy,
        }
        for global_index, input_hash, detail, ranked, entropy in zip(
            global_indices,
            input_hashes,
            details,
            ranked_outputs,
            restricted_entropies,
        )
    ]
    prediction_path = output_dir / "id_token_predictions.parquet"
    _atomic_write_parquet(prediction_path, prediction_rows)
    _atomic_write_json(output_dir / "metrics.json", metrics)
    _atomic_write_json(output_dir / "resolved_config.json", config)
    shard_manifest = {
        **identity_payload,
        "evaluation_identity": evaluation_identity,
        "shard_index": shard_index,
        "shard_record_count": len(prediction_rows),
        "first_record_index": global_indices[0],
        "last_record_index": global_indices[-1],
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": elapsed,
        "generated_samples": generated_samples,
        "resumed_samples": resumed_samples,
    }
    _atomic_write_json(output_dir / "shard_manifest.json", shard_manifest)
    return metrics


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank Semantic ID Tokens with one next-token forward pass."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--interest-token-mapping-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    for key in (
        "adapter_path",
        "eval_file",
        "interest_token_mapping_file",
        "output_dir",
        "run_name",
        "max_examples",
        "num_shards",
        "shard_index",
        "batch_size",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    metrics = evaluate_id_token_ranking(config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
