from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    _atomic_write_json,
    _atomic_write_parquet,
    _load_labels,
    canonical_sha256,
    record_input_sha256,
    select_logical_shard,
    sha256_adapter_semantics,
    sha256_file,
)
from new_interest.evaluation.metrics import (
    RESPONSE_PREFIX,
    evaluate_ranked_outputs,
)
from new_interest.training.sft_data import apply_chat_template


EVALUATION_IDENTITY_VERSION = "text-label-exhaustive-label-mean-v1"


@dataclass(frozen=True)
class LabelEncoding:
    label: str
    response_token_ids: tuple[int, ...]
    label_response_positions: tuple[int, ...]


def encode_label_response(
    tokenizer: Any,
    label: str,
    *,
    response_prefix: str = RESPONSE_PREFIX,
) -> LabelEncoding:
    """Tokenize the exact Beam response and identify category-bearing tokens."""

    response = f"{response_prefix}{label}"
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets):
        raise ValueError(f"Tokenizer offsets do not align for label {label!r}")
    label_start = len(response_prefix)
    label_stop = len(response)
    label_positions = tuple(
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > label_start and start < label_stop
    )
    if not label_positions:
        raise ValueError(f"Label produced no category-bearing tokens: {label!r}")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    return LabelEncoding(
        label=label,
        response_token_ids=tuple([*token_ids, int(eos_token_id)]),
        label_response_positions=label_positions,
    )


def mean_selected_token_log_probabilities(
    logits: Any,
    predictor_positions: list[list[int]],
    target_token_ids: list[list[int]],
) -> list[float]:
    """Return the mean target-token log probability for each batch row."""

    import torch

    if logits.ndim != 3:
        raise ValueError("Expected logits with shape [batch, time, vocabulary]")
    if not (
        logits.shape[0]
        == len(predictor_positions)
        == len(target_token_ids)
    ):
        raise ValueError("Batch dimensions do not match")
    scores: list[float] = []
    for row_index, (positions, targets) in enumerate(
        zip(predictor_positions, target_token_ids)
    ):
        if not positions or len(positions) != len(targets):
            raise ValueError("Every row needs aligned non-empty score positions")
        if min(positions) < 0 or max(positions) >= logits.shape[1]:
            raise ValueError("Predictor position is outside returned logits")
        row_positions = torch.tensor(
            positions,
            dtype=torch.long,
            device=logits.device,
        )
        row_targets = torch.tensor(
            targets,
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits[row_index].index_select(0, row_positions).float()
        token_log_probs = torch.log_softmax(selected, dim=-1).gather(
            1,
            row_targets.unsqueeze(1),
        )
        score = float(token_log_probs.mean().detach().cpu())
        if not math.isfinite(score):
            raise ValueError("Non-finite label mean log probability")
        scores.append(score)
    return scores


def _score_record_labels(
    model: Any,
    tokenizer: Any,
    prompt: str,
    encodings: list[LabelEncoding],
    *,
    batch_size: int,
    max_seq_length: int,
    enable_thinking: bool,
) -> tuple[list[float], float]:
    import torch

    prompt_text = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = [
        int(value)
        for value in tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
    ]
    scores: list[float] = []
    for start in range(0, len(encodings), batch_size):
        batch = encodings[start : start + batch_size]
        sequences = [
            [*prompt_ids, *encoding.response_token_ids]
            for encoding in batch
        ]
        if any(len(sequence) > max_seq_length for sequence in sequences):
            raise ValueError(
                "Text-label likelihood input exceeds max_seq_length; "
                "truncation is disabled"
            )
        max_length = max(map(len, sequences))
        max_response_length = max(
            len(encoding.response_token_ids) for encoding in batch
        )
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        predictor_positions: list[list[int]] = []
        target_token_ids: list[list[int]] = []
        first_kept_position = max_length - max_response_length
        for sequence, encoding in zip(sequences, batch):
            padding = max_length - len(sequence)
            input_ids.append([tokenizer.pad_token_id] * padding + sequence)
            attention_masks.append([0] * padding + [1] * len(sequence))
            absolute_targets = [
                padding + len(prompt_ids) + response_position
                for response_position in encoding.label_response_positions
            ]
            predictor_positions.append(
                [
                    target_position - 1 - first_kept_position
                    for target_position in absolute_targets
                ]
            )
            target_token_ids.append(
                [
                    encoding.response_token_ids[response_position]
                    for response_position in encoding.label_response_positions
                ]
            )
        tensors = {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
                device=model.device,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
                device=model.device,
            ),
        }
        with torch.inference_mode():
            logits = model(
                **tensors,
                use_cache=False,
                logits_to_keep=max_response_length,
            ).logits
        scores.extend(
            mean_selected_token_log_probabilities(
                logits,
                predictor_positions,
                target_token_ids,
            )
        )
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    log_probabilities = torch.log_softmax(score_tensor, dim=0)
    probabilities = log_probabilities.exp()
    entropy = float(-(probabilities * log_probabilities).sum())
    return scores, entropy


def common_non_label_prefix_length(
    encodings: list[LabelEncoding],
) -> int:
    """Return the shared response prefix before any category-bearing token."""

    if not encodings:
        raise ValueError("At least one label encoding is required")
    prefix_length = min(
        min(encoding.label_response_positions) for encoding in encodings
    )
    reference = encodings[0].response_token_ids[:prefix_length]
    if any(
        encoding.response_token_ids[:prefix_length] != reference
        for encoding in encodings[1:]
    ):
        raise ValueError("Text labels do not share a stable response prefix")
    if any(
        any(position < prefix_length for position in encoding.label_response_positions)
        for encoding in encodings
    ):
        raise ValueError("Shared prefix contains a category-bearing token")
    return prefix_length


def _score_record_labels_prefix_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    encodings: list[LabelEncoding],
    *,
    batch_size: int,
    max_seq_length: int,
    enable_thinking: bool,
) -> tuple[list[float], float]:
    """Score labels exactly while reusing the prompt and fixed-prefix cache."""

    import torch

    prompt_text = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = [
        int(value)
        for value in tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
    ]
    prefix_length = common_non_label_prefix_length(encodings)
    fixed_prefix = list(encodings[0].response_token_ids[:prefix_length])
    base_ids = [*prompt_ids, *fixed_prefix]
    if not base_ids or len(base_ids) > max_seq_length:
        raise ValueError("Cached text-label prefix exceeds max_seq_length")
    base_tensor = torch.tensor(
        [base_ids],
        dtype=torch.long,
        device=model.device,
    )
    base_mask = torch.ones_like(base_tensor)
    with torch.inference_mode():
        base_output = model(
            input_ids=base_tensor,
            attention_mask=base_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    if base_output.past_key_values is None:
        raise RuntimeError("Model did not return a reusable prompt cache")
    first_token_logits = base_output.logits[0, -1].float()

    scores: list[float] = []
    for start in range(0, len(encodings), batch_size):
        batch = encodings[start : start + batch_size]
        tails = [
            list(encoding.response_token_ids[prefix_length:])
            for encoding in batch
        ]
        if any(not tail for tail in tails):
            raise ValueError("A label tail is unexpectedly empty")
        if any(len(base_ids) + len(tail) > max_seq_length for tail in tails):
            raise ValueError(
                "Text-label likelihood input exceeds max_seq_length; "
                "truncation is disabled"
            )
        continuation_lengths = [len(tail) - 1 for tail in tails]
        max_continuation = max(continuation_lengths)
        continuation_logits = None
        if max_continuation > 0:
            input_ids: list[list[int]] = []
            continuation_masks: list[list[int]] = []
            for tail, length in zip(tails, continuation_lengths):
                padding = max_continuation - length
                input_ids.append(
                    tail[:-1] + [tokenizer.pad_token_id] * padding
                )
                continuation_masks.append([1] * length + [0] * padding)
            cache = copy.deepcopy(base_output.past_key_values)
            # `Cache.batch_repeat_interleave` is missing on the Qwen3.5
            # LinearAttentionLayer in Transformers 5.14.1. Reordering with
            # repeated source index zero is the equivalent supported path for
            # Dynamic, linear-attention, and hybrid cache layers.
            cache.reorder_cache(
                torch.zeros(
                    len(batch),
                    dtype=torch.long,
                    device=model.device,
                )
            )
            continuation_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=model.device,
            )
            continuation_mask = torch.tensor(
                continuation_masks,
                dtype=torch.long,
                device=model.device,
            )
            attention_mask = torch.cat(
                [
                    torch.ones(
                        len(batch),
                        len(base_ids),
                        dtype=torch.long,
                        device=model.device,
                    ),
                    continuation_mask,
                ],
                dim=1,
            )
            with torch.inference_mode():
                continuation_logits = model(
                    input_ids=continuation_tensor,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=max_continuation,
                ).logits.float()

        for row_index, (encoding, tail) in enumerate(zip(batch, tails)):
            token_log_probabilities = []
            for response_position in encoding.label_response_positions:
                tail_position = response_position - prefix_length
                target_token_id = tail[tail_position]
                if tail_position == 0:
                    predictor = first_token_logits
                else:
                    if continuation_logits is None:
                        raise RuntimeError("Missing continuation logits")
                    predictor = continuation_logits[
                        row_index,
                        tail_position - 1,
                    ]
                token_log_probabilities.append(
                    torch.log_softmax(predictor, dim=-1)[target_token_id]
                )
            score = float(torch.stack(token_log_probabilities).mean().cpu())
            if not math.isfinite(score):
                raise ValueError("Non-finite cached label mean log probability")
            scores.append(score)

    score_tensor = torch.tensor(scores, dtype=torch.float64)
    log_probabilities = torch.log_softmax(score_tensor, dim=0)
    probabilities = log_probabilities.exp()
    entropy = float(-(probabilities * log_probabilities).sum())
    return scores, entropy


def generate_text_label_rankings(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    labels: list[str],
    *,
    batch_size: int,
    top_k: int,
    max_seq_length: int,
    enable_thinking: bool,
    scoring_backend: str = "prefix_cache",
    response_prefix: str = RESPONSE_PREFIX,
) -> tuple[list[list[dict[str, object]]], list[float], list[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < top_k <= len(labels):
        raise ValueError("top_k must be within the label vocabulary")
    encodings = [
        encode_label_response(
            tokenizer,
            label,
            response_prefix=response_prefix,
        )
        for label in labels
    ]
    label_token_counts = [
        len(encoding.label_response_positions) for encoding in encodings
    ]
    model.eval()
    rankings: list[list[dict[str, object]]] = []
    entropies: list[float] = []
    if scoring_backend not in {"full_sequence", "prefix_cache"}:
        raise ValueError("Unknown text-label scoring backend")
    for record in records:
        scorer = (
            _score_record_labels_prefix_cache
            if scoring_backend == "prefix_cache"
            else _score_record_labels
        )
        scores, entropy = scorer(
            model,
            tokenizer,
            str(record["prompt"]),
            encodings,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            enable_thinking=enable_thinking,
        )
        order = sorted(
            range(len(labels)),
            key=lambda index: (-scores[index], index),
        )[:top_k]
        rankings.append(
            [
                {
                    "label": labels[index],
                    "response": f"{response_prefix}{labels[index]}",
                    "score": scores[index],
                }
                for index in order
            ]
        )
        entropies.append(entropy)
    return rankings, entropies, label_token_counts


def _validate_checkpoint_rows(
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    global_indices: list[int],
    input_hashes: list[str],
    *,
    evaluation_identity: str,
    label_set: set[str],
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
        scores = [float(value) for value in row["ranked_scores"]]
        if len(labels) != top_k or len(scores) != top_k:
            raise ValueError(f"{context}: invalid Top-{top_k} output")
        if len(set(labels)) != top_k:
            raise ValueError(f"{context}: duplicate ranked labels")
        if set(labels).difference(label_set):
            raise ValueError(f"{context}: unknown ranked label")
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"{context}: non-finite score")


def evaluate_text_label_ranking(config: dict[str, Any]) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    model_revision = config.get("model_revision")
    adapter_value = config.get("adapter_path")
    adapter_path = Path(str(adapter_value)) if adapter_value else None
    eval_file = Path(config["eval_file"])
    labels_file = Path(config["retained_interests_file"])
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
    checkpoint_size = int(config.get("checkpoint_size", 16))
    max_seq_length = int(config.get("max_seq_length", 2048))
    scoring_backend = str(config.get("scoring_backend", "prefix_cache"))
    response_prefix = str(config.get("response_prefix", RESPONSE_PREFIX))
    if not response_prefix:
        raise ValueError("response_prefix must not be empty")
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name,
        revision=model_revision if adapter_path is None else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    labels = _load_labels(labels_file)
    label_encodings = [
        encode_label_response(
            tokenizer,
            label,
            response_prefix=response_prefix,
        )
        for label in labels
    ]
    label_token_counts = [
        len(encoding.label_response_positions)
        for encoding in label_encodings
    ]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        token=config.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if adapter_path is not None:
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
        "model_revision": model_revision,
        "adapter_sha256": (
            sha256_adapter_semantics(adapter_path)
            if adapter_path is not None
            else None
        ),
        "eval_sha256": sha256_file(eval_file),
        "labels_sha256": sha256_file(labels_file),
        "full_record_count": len(all_records),
        "labels": labels,
        "label_token_counts": label_token_counts,
        "scoring": "label_token_mean_log_probability",
        "scoring_backend": scoring_backend,
        "response_prefix": response_prefix,
        "fixed_prefix_scored": False,
        "eos_scored": False,
        "max_examples": int(max_examples) if max_examples is not None else None,
        "num_shards": num_shards,
        "top_k": top_k,
        "cutoffs": list(cutoffs),
        "max_seq_length": max_seq_length,
        "enable_thinking": bool(config.get("enable_thinking", False)),
    }
    evaluation_identity = canonical_sha256(identity_payload)
    input_hashes = [record_input_sha256(record) for record in records]
    checkpoint_dir = output_dir / "shards"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ranked_outputs: list[list[dict[str, object]]] = []
    score_entropies: list[float] = []
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
                label_set=set(labels),
                top_k=top_k,
                context=str(checkpoint_path),
            )
            resumed_samples += len(rows)
        else:
            batch_ranked, batch_entropies, observed_counts = (
                generate_text_label_rankings(
                    model,
                    tokenizer,
                    local_records,
                    labels,
                    batch_size=batch_size,
                    top_k=top_k,
                    max_seq_length=max_seq_length,
                    enable_thinking=bool(
                        config.get("enable_thinking", False)
                    ),
                    scoring_backend=scoring_backend,
                    response_prefix=response_prefix,
                )
            )
            if observed_counts != label_token_counts:
                raise RuntimeError("Label tokenization changed during evaluation")
            rows = [
                {
                    "record_index": record_index,
                    "impression_id": str(record["impression_id"]),
                    "input_sha256": input_hash,
                    "evaluation_identity": evaluation_identity,
                    "ranked_labels": [
                        str(item["label"]) for item in ranked
                    ],
                    "ranked_scores": [
                        float(item["score"]) for item in ranked
                    ],
                    "score_softmax_entropy": float(entropy),
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
                        "response": f"{response_prefix}{label}",
                        "score": score,
                    }
                    for label, score in zip(
                        row["ranked_labels"], row["ranked_scores"]
                    )
                ]
                for row in rows
            ]
        )
        score_entropies.extend(
            float(row["score_softmax_entropy"]) for row in rows
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
            "mean_score_softmax_entropy": (
                sum(score_entropies) / len(score_entropies)
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
            "score_softmax_entropy": entropy,
        }
        for global_index, input_hash, detail, entropy in zip(
            global_indices,
            input_hashes,
            details,
            score_entropies,
        )
    ]
    prediction_path = output_dir / "text_label_predictions.parquet"
    _atomic_write_parquet(prediction_path, prediction_rows)
    _atomic_write_json(output_dir / "metrics.json", metrics)
    _atomic_write_json(output_dir / "resolved_config.json", config)
    _atomic_write_json(
        output_dir / "shard_manifest.json",
        {
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
        },
    )
    return metrics


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exhaustively rank text labels by label-token likelihood."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--retained-interests-file", type=Path)
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
        "retained_interests_file",
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
    metrics = evaluate_text_label_ranking(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
