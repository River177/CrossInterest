from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.candidate_metrics import evaluate_candidate_outputs
from new_interest.training.sft_data import apply_chat_template


def _pilot_records(
    records: list[dict[str, object]],
    pilot_size: int | None,
    seed: int,
    *,
    cover_categories: bool,
) -> list[dict[str, object]]:
    if pilot_size is None:
        return records
    if pilot_size <= 0 or pilot_size > len(records):
        raise ValueError("pilot_size must be within the dataset size")

    def priority(row: dict[str, object]) -> tuple[str, str]:
        selection_id = str(row["selection_id"])
        material = f"{seed}:hf-pilot:{selection_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), selection_id

    selected_ids: set[str] = set()
    if cover_categories:
        by_category: dict[str, list[dict[str, object]]] = {}
        for row in records:
            by_category.setdefault(
                str(row["answer_category"]),
                [],
            ).append(row)
        if len(by_category) > pilot_size:
            raise ValueError(
                "pilot_size is smaller than the number of answer categories"
            )
        for category_rows in by_category.values():
            selected_ids.add(
                str(min(category_rows, key=priority)["selection_id"])
            )

    remaining = sorted(
        (
            row
            for row in records
            if str(row["selection_id"]) not in selected_ids
        ),
        key=priority,
    )
    for row in remaining[: pilot_size - len(selected_ids)]:
        selected_ids.add(str(row["selection_id"]))
    return [
        row
        for row in records
        if str(row["selection_id"]) in selected_ids
    ]


def _validate_shard_settings(
    num_shards: int,
    shard_index: int,
) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            "shard_index must be within [0, num_shards)"
        )


def _shard_records(
    source_records: list[dict[str, object]],
    selected_records: list[dict[str, object]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, object]]:
    _validate_shard_settings(num_shards, shard_index)
    source_indices = {
        str(row["selection_id"]): index
        for index, row in enumerate(source_records)
    }
    return [
        row
        for row in selected_records
        if source_indices[str(row["selection_id"])] % num_shards
        == shard_index
    ]


class PresencePenaltyLogitsProcessor:
    def __init__(
        self,
        penalty: float,
        prompt_attention_mask: Any | None = None,
    ) -> None:
        if not math.isfinite(penalty):
            raise ValueError("presence penalty must be finite")
        self.penalty = penalty
        self.prompt_attention_mask = prompt_attention_mask

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        token_mask = torch.ones_like(input_ids, dtype=torch.int32)
        if self.prompt_attention_mask is not None:
            prompt_length = self.prompt_attention_mask.shape[1]
            generated_length = input_ids.shape[1] - prompt_length
            if generated_length < 0:
                raise ValueError(
                    "input IDs are shorter than the prompt attention mask"
                )
            token_mask[:, :prompt_length] = (
                self.prompt_attention_mask.to(
                    device=input_ids.device,
                    dtype=torch.int32,
                )
            )
        seen_counts = torch.zeros_like(scores, dtype=torch.int32)
        seen_counts.scatter_add_(1, input_ids, token_mask)
        seen_tokens = seen_counts > 0
        return scores - seen_tokens.to(scores.dtype) * self.penalty


def apply_candidate_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    model_name: str,
) -> str:
    try:
        return apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError as error:
        if (
            "qwen" in model_name.lower()
            or "enable_thinking" not in str(error)
        ):
            raise
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def parse_generated_output(raw_output: str) -> dict[str, object]:
    opening_tag = "<think>"
    closing_tag = "</think>"
    has_opening_tag = opening_tag in raw_output
    closed_think_block = closing_tag in raw_output
    has_think_block = has_opening_tag or closed_think_block
    think_content = ""
    final_output = raw_output
    if closed_think_block:
        before_close, _, after_close = raw_output.partition(closing_tag)
        think_content = before_close.rsplit(opening_tag, 1)[-1].strip()
        final_output = after_close.strip()
    elif has_opening_tag:
        think_content = raw_output.rsplit(opening_tag, 1)[-1].strip()
        final_output = ""

    return {
        "raw_output": raw_output,
        "output": final_output,
        "final_output": final_output,
        "think_content": think_content,
        "has_think_block": has_think_block,
        "closed_think_block": closed_think_block,
        "missing_final_answer": (
            not final_output.strip()
            or (has_opening_tag and not closed_think_block)
        ),
    }


def _generated_token_metadata(
    token_ids: Any,
    *,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
    pad_token_id: int | None,
    max_new_tokens: int,
) -> tuple[int, bool]:
    ids = [int(token_id) for token_id in token_ids]
    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = {int(token_id) for token_id in eos_token_id}

    for index, token_id in enumerate(ids):
        if token_id in eos_ids:
            return index + 1, False
    if pad_token_id is not None:
        while ids and ids[-1] == pad_token_id:
            ids.pop()
    generated_token_count = len(ids)
    return generated_token_count, generated_token_count >= max_new_tokens


def generate_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    *,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    seed: int | None,
    record_offset: int,
) -> list[dict[str, object]]:
    import torch

    outputs: list[dict[str, object]] = []
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.min_p = None
    for start in range(0, len(records), batch_size):
        if seed is not None:
            batch_seed = seed + record_offset + start
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
        batch = records[start : start + batch_size]
        texts = [
            apply_candidate_chat_template(
                tokenizer,
                [{"role": "user", "content": str(row["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                model_name=model_name,
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
            key: value.to(model.device)
            for key, value in encoded.items()
        }
        input_length = encoded["input_ids"].shape[1]
        generation_kwargs: dict[str, object] = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                }
            )
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if presence_penalty:
            generation_kwargs["logits_processor"] = [
                PresencePenaltyLogitsProcessor(
                    presence_penalty,
                    prompt_attention_mask=encoded["attention_mask"],
                )
            ]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )
        generated_tokens = generated[:, input_length:]
        raw_outputs = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        for raw_output, token_ids in zip(raw_outputs, generated_tokens):
            parsed = parse_generated_output(raw_output)
            token_count, reached_limit = _generated_token_metadata(
                token_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
            )
            parsed.update(
                {
                    "generated_token_count": token_count,
                    "reached_max_new_tokens": reached_limit,
                }
            )
            outputs.append(parsed)
    return outputs


def _normalize_shard_output(row: dict[str, object]) -> dict[str, object]:
    if "raw_output" in row:
        raw_output = str(row["raw_output"])
    else:
        raw_output = str(row["output"])
    parsed = parse_generated_output(raw_output)
    persisted_keys = [
        "final_output",
        "think_content",
        "has_think_block",
        "closed_think_block",
        "missing_final_answer",
        "generated_token_count",
        "reached_max_new_tokens",
    ]
    if "raw_output" in row:
        persisted_keys.append("output")
    for key in persisted_keys:
        if key in row:
            parsed[key] = row[key]
    parsed.setdefault("generated_token_count", None)
    parsed.setdefault("reached_max_new_tokens", None)
    return parsed


def _aggregate_generation_stats(
    outputs: list[dict[str, object]],
) -> dict[str, object]:
    sample_count = len(outputs)
    token_counts = [
        int(row["generated_token_count"])
        for row in outputs
        if row.get("generated_token_count") is not None
    ]

    def count_true(key: str) -> int:
        return sum(row.get(key) is True for row in outputs)

    def rate(count: int) -> float:
        return count / sample_count if sample_count else 0.0

    think_count = count_true("has_think_block")
    closed_count = count_true("closed_think_block")
    missing_count = count_true("missing_final_answer")
    reached_limit_values = [
        bool(row["reached_max_new_tokens"])
        for row in outputs
        if row.get("reached_max_new_tokens") is not None
    ]
    max_token_count = sum(reached_limit_values)
    return {
        "think_block_count": think_count,
        "think_block_rate": rate(think_count),
        "closed_think_block_count": closed_count,
        "closed_think_block_rate": rate(closed_count),
        "missing_final_answer_count": missing_count,
        "missing_final_answer_rate": rate(missing_count),
        "reached_max_new_tokens_count": max_token_count,
        "reached_max_new_tokens_known": len(reached_limit_values),
        "reached_max_new_tokens_rate": (
            max_token_count / len(reached_limit_values)
            if reached_limit_values
            else None
        ),
        "generated_token_count_known": len(token_counts),
        "mean_generated_tokens": (
            sum(token_counts) / len(token_counts) if token_counts else None
        ),
        "max_generated_tokens": max(token_counts) if token_counts else None,
    }


def evaluate_qwen(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config.get("model_name", "Qwen/Qwen3-8B"))
    adapter_path = config.get("adapter_path")
    data_file = Path(config["data_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("batch_size", 8))
    checkpoint_size = int(config.get("checkpoint_size", 256))
    max_new_tokens = int(config.get("max_new_tokens", 32))
    enable_thinking = bool(config.get("enable_thinking", False))
    do_sample = bool(config.get("do_sample", False))
    temperature = (
        float(config.get("temperature", 0.7)) if do_sample else None
    )
    top_p = float(config.get("top_p", 0.8)) if do_sample else None
    top_k = int(config.get("top_k", 20)) if do_sample else None
    min_p = float(config.get("min_p", 0.0)) if do_sample else None
    presence_penalty = (
        float(config["presence_penalty"])
        if config.get("presence_penalty") is not None
        else None
    )
    repetition_penalty = (
        float(config["repetition_penalty"])
        if config.get("repetition_penalty") is not None
        else None
    )
    seed = int(config["seed"]) if config.get("seed") is not None else None
    max_examples = config.get("max_examples")
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    pilot_size = config.get("pilot_size")
    if pilot_size is not None:
        pilot_size = int(pilot_size)
    pilot_seed = int(config.get("pilot_seed", 20260808))
    pilot_cover_categories = bool(
        config.get("pilot_cover_categories", False)
    )
    _validate_shard_settings(num_shards, shard_index)
    if batch_size <= 0 or checkpoint_size <= 0:
        raise ValueError("batch_size and checkpoint_size must be positive")
    if do_sample:
        if temperature is None or temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if top_p is None or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when sampling")
        if top_k is None or top_k <= 0:
            raise ValueError("top_k must be positive when sampling")
        if min_p is None or not 0 <= min_p <= 1:
            raise ValueError("min_p must be in [0, 1] when sampling")
        if seed is None:
            raise ValueError("seed is required for reproducible sampling")
        if checkpoint_size % batch_size:
            raise ValueError(
                "checkpoint_size must be divisible by batch_size "
                "for resume-stable sampling"
            )
    if presence_penalty is not None and not math.isfinite(
        presence_penalty
    ):
        raise ValueError("presence_penalty must be finite")
    if repetition_penalty is not None and (
        not math.isfinite(repetition_penalty)
        or repetition_penalty <= 0
    ):
        raise ValueError("repetition_penalty must be finite and positive")

    records = pq.read_table(data_file).to_pylist()
    source_records = records
    records = _pilot_records(
        records,
        pilot_size,
        pilot_seed,
        cover_categories=pilot_cover_categories,
    )
    if max_examples is not None:
        records = records[: int(max_examples)]
    pre_shard_selected_count = len(records)
    records = _shard_records(
        source_records,
        records,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    shard_sample_count = len(records)
    resolved_config = {
        **config,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "pre_shard_selected_count": pre_shard_selected_count,
        "shard_sample_count": shard_sample_count,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    generated_outputs: list[dict[str, object]] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()
    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_ids = [
                str(row["selection_id"]) for row in records[start:stop]
            ]
            actual_ids = [
                str(row["selection_id"]) for row in shard_rows
            ]
            if actual_ids != expected_ids:
                raise ValueError(
                    f"{shard_path}: selection IDs do not match data"
                )
            generated_outputs.extend(
                _normalize_shard_output(row) for row in shard_rows
            )
            resumed_samples += len(shard_rows)
            continue
        batch_outputs = generate_outputs(
            model,
            tokenizer,
            records[start:stop],
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            record_offset=start,
        )
        shard_rows = [
            {
                "record_index": start + offset,
                "selection_id": str(
                    records[start + offset]["selection_id"]
                ),
                **output,
            }
            for offset, output in enumerate(batch_outputs)
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=f".{shard_path.name}.",
            dir=shards_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(
                pa.Table.from_pylist(shard_rows),
                temporary,
                compression="zstd",
                use_dictionary=True,
            )
            os.replace(temporary, shard_path)
        finally:
            temporary.unlink(missing_ok=True)
        generated_outputs.extend(batch_outputs)
        generated_samples += len(batch_outputs)

    outputs = [str(row["output"]) for row in generated_outputs]
    metrics, details = evaluate_candidate_outputs(records, outputs)
    for detail, generated_output in zip(details, generated_outputs):
        detail.update(
            {
                key: value
                for key, value in generated_output.items()
                if key != "output"
            }
        )
    runtime_seconds = time.time() - started_at
    generation_config = {
        "enable_thinking": enable_thinking,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
    }
    metrics.update(
        {
            "model_name": model_name,
            "adapter_path": (
                str(Path(adapter_path).resolve())
                if adapter_path
                else None
            ),
            "data_file": str(data_file.resolve()),
            "enable_thinking": enable_thinking,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "pilot_size": pilot_size,
            "pilot_seed": pilot_seed,
            "pilot_cover_categories": pilot_cover_categories,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "pre_shard_selected_count": pre_shard_selected_count,
            "shard_sample_count": shard_sample_count,
            "generation_config": generation_config,
            "generation_stats": _aggregate_generation_stats(
                generated_outputs
            ),
            "runtime_seconds": runtime_seconds,
            "samples_per_second": (
                generated_samples / runtime_seconds
                if generated_samples
                else None
            ),
            "generated_samples_this_run": generated_samples,
            "resumed_samples": resumed_samples,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(details),
        output_dir / "predictions.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    wandb_mode = str(config.get("wandb_mode", "offline"))
    if wandb_mode != "disabled":
        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-generative-sft",
                )
            ),
            name=str(config["run_name"]),
            job_type="candidate-evaluation",
            mode=wandb_mode,
            config=resolved_config,
        )
        run.log(
            {
                f"candidate/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        artifact = wandb.Artifact(
            f"{config['run_name']}-predictions",
            type="candidate-evaluation",
        )
        artifact.add_file(str(output_dir / "predictions.parquet"))
        artifact.add_file(str(output_dir / "metrics.json"))
        run.log_artifact(artifact)
        run.finish()
    return metrics


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HF chat models on candidate selection."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--pilot-size", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_json(args.config)
    for key in (
        "data_file",
        "adapter_path",
        "output_dir",
        "run_name",
        "max_examples",
        "pilot_size",
        "num_shards",
        "shard_index",
        "wandb_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    print(json.dumps(evaluate_qwen(config), indent=2))


if __name__ == "__main__":
    main()
