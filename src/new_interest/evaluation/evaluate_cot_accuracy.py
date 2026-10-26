from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_grpo_dataset import (
    COT_INSTRUCTION,
    LEGACY_COT_INSTRUCTION,
    append_cot_instruction,
)
from new_interest.evaluation.metrics import RESPONSE_PREFIX
from new_interest.training.grpo_reward import (
    extract_final_answer,
    parse_structured_output,
)
from new_interest.training.sft_data import apply_chat_template

EvaluationMode = Literal["cot", "no_cot"]


@dataclass(frozen=True)
class ParsedAccuracyOutput:
    loose_answer: str | None
    loose_label: str | None
    strict_answer: str | None
    strict_label: str | None
    strict_format: bool


def _normalize_mode(mode: str) -> EvaluationMode:
    normalized = mode.replace("-", "_")
    if normalized not in ("cot", "no_cot"):
        raise ValueError(f"Unknown evaluation mode: {mode!r}")
    return normalized


def _label_from_answer(answer: str | None) -> str | None:
    if answer is None or not answer.startswith(RESPONSE_PREFIX):
        return None
    label = answer[len(RESPONSE_PREFIX) :].strip()
    return label or None


def parse_accuracy_output(
    output: str,
    *,
    mode: str,
) -> ParsedAccuracyOutput:
    mode = _normalize_mode(mode)
    loose_answer = extract_final_answer(output)
    if mode == "cot":
        structured = parse_structured_output(output)
        strict_answer = (
            structured.answer if structured is not None else None
        )
    elif mode == "no_cot":
        stripped = output.strip()
        strict_answer = (
            stripped
            if (
                stripped.startswith(RESPONSE_PREFIX)
                and "\n" not in stripped
                and "\r" not in stripped
                and _label_from_answer(stripped) is not None
            )
            else None
        )
    return ParsedAccuracyOutput(
        loose_answer=loose_answer,
        loose_label=_label_from_answer(loose_answer),
        strict_answer=strict_answer,
        strict_label=_label_from_answer(strict_answer),
        strict_format=strict_answer is not None,
    )


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _prediction_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def _summarize_group(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    return {
        "samples": total,
        "set_accuracy_at_1": sum(
            bool(row["set_hit_at_1"]) for row in rows
        )
        / total,
        "strict_set_accuracy_at_1": sum(
            bool(row["strict_set_hit_at_1"]) for row in rows
        )
        / total,
        "valid_label_rate": sum(
            bool(row["valid_label"]) for row in rows
        )
        / total,
        "novelty_compliance_rate": sum(
            bool(row["novelty_compliant"]) for row in rows
        )
        / total,
        "strict_format_rate": sum(
            bool(row["strict_format"]) for row in rows
        )
        / total,
        "truncation_rate": sum(
            bool(row["reached_max_new_tokens"]) for row in rows
        )
        / total,
    }


def summarize_accuracy_rows(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("Evaluation rows must be non-empty")
    summary = _summarize_group(rows)
    class_support: Counter[str] = Counter()
    class_hits: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    by_count: dict[int, list[dict[str, object]]] = defaultdict(list)
    token_counts = [int(row["response_tokens"]) for row in rows]

    for row in rows:
        predicted = row["parsed_label"]
        if predicted is not None:
            prediction_counts[str(predicted)] += 1
        for target in row["acceptable_subcategories"]:
            target = str(target)
            class_support[target] += 1
            class_hits[target] += int(predicted == target)
        by_count[int(row["acceptable_target_count"])].append(row)

    per_class_accuracy = {
        target: class_hits[target] / support
        for target, support in sorted(class_support.items())
    }
    summary.update(
        {
            "response_tokens": {
                "mean": mean(token_counts),
                "median": median(token_counts),
                "p95": _percentile(token_counts, 0.95),
                "max": max(token_counts),
            },
            "mean_response_tokens": mean(token_counts),
            "median_response_tokens": median(token_counts),
            "p95_response_tokens": _percentile(token_counts, 0.95),
            "max_response_tokens": max(token_counts),
            "macro_per_class_accuracy": mean(
                per_class_accuracy.values()
            ),
            "per_class_accuracy": per_class_accuracy,
            "per_class_support": dict(sorted(class_support.items())),
            "by_acceptable_target_count": {
                str(count): _summarize_group(group)
                for count, group in sorted(by_count.items())
            },
            "prediction_counts": dict(
                sorted(
                    prediction_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "prediction_entropy_bits": _prediction_entropy(
                prediction_counts,
            ),
        }
    )
    return summary


def generate_greedy_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    *,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    cot_instruction: str | None = COT_INSTRUCTION,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    seed: int | None = None,
    record_offset: int = 0,
) -> list[dict[str, object]]:
    import torch

    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("Generation sizes must be positive")
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
            raise ValueError("seed is required when sampling")
    mode = _normalize_mode(mode)
    rendered_prompts = [
        apply_chat_template(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": (
                        (
                            append_cot_instruction(
                                str(record["prompt"]),
                                instruction=cot_instruction,
                            )
                            if cot_instruction is not None
                            else str(record["prompt"])
                        )
                        if mode == "cot"
                        else str(record["prompt"])
                    ),
                }
            ],
            add_generation_prompt=True,
            enable_thinking=mode == "cot",
        )
        for record in records
    ]
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
        batch_prompts = rendered_prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
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
            "use_cache": True,
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
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )
        for prompt, token_row in zip(
            batch_prompts,
            generated[:, input_length:],
        ):
            token_ids = token_row.tolist()
            if tokenizer.eos_token_id in token_ids:
                token_count = token_ids.index(tokenizer.eos_token_id)
                reached_max = False
            else:
                token_count = len(token_ids)
                reached_max = token_count >= max_new_tokens
            outputs.append(
                {
                    "rendered_prompt": prompt,
                    "raw_output": tokenizer.decode(
                        token_ids[:token_count],
                        skip_special_tokens=True,
                    ),
                    "response_tokens": token_count,
                    "reached_max_new_tokens": reached_max,
                }
            )
    return outputs


def evaluate_accuracy_outputs(
    records: list[dict[str, object]],
    outputs: list[str],
    valid_labels: set[str],
    *,
    mode: str,
    record_indices: list[int] | None = None,
    response_tokens: list[int] | None = None,
    reached_max_new_tokens: list[bool] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mode = _normalize_mode(mode)
    if not records:
        raise ValueError("Evaluation records must be non-empty")
    if len(records) != len(outputs):
        raise ValueError(
            f"Got {len(records)} records and {len(outputs)} outputs"
        )
    total = len(records)
    record_indices = record_indices or list(range(total))
    response_tokens = response_tokens or [0] * total
    reached_max_new_tokens = reached_max_new_tokens or [False] * total
    if not (
        len(record_indices)
        == len(response_tokens)
        == len(reached_max_new_tokens)
        == total
    ):
        raise ValueError("Generation metadata must align with records")

    rows: list[dict[str, object]] = []
    for index, (record, output) in enumerate(zip(records, outputs)):
        acceptable = [
            str(value) for value in record["acceptable_subcategories"]
        ]
        if not acceptable:
            raise ValueError(
                f"{record['impression_id']}: no acceptable subcategories"
            )
        history = [
            str(value) for value in record["history_subcategories"]
        ]
        parsed = parse_accuracy_output(output, mode=mode)
        valid_label = parsed.loose_label in valid_labels
        rows.append(
            {
                "record_index": int(record_indices[index]),
                "impression_id": str(record["impression_id"]),
                "prompt": str(record["prompt"]),
                "mode": mode,
                "raw_output": output,
                "response_tokens": int(response_tokens[index]),
                "reached_max_new_tokens": bool(
                    reached_max_new_tokens[index]
                ),
                "parsed_answer": parsed.loose_answer,
                "parsed_label": parsed.loose_label,
                "strict_answer": parsed.strict_answer,
                "strict_label": parsed.strict_label,
                "strict_format": parsed.strict_format,
                "valid_label": valid_label,
                "set_hit_at_1": parsed.loose_label in acceptable,
                "strict_set_hit_at_1": (
                    parsed.strict_label in acceptable
                ),
                "novelty_compliant": (
                    valid_label and parsed.loose_label not in set(history)
                ),
                "acceptable_subcategories": acceptable,
                "acceptable_target_count": len(acceptable),
                "history_subcategories": history,
            }
        )
    return summarize_accuracy_rows(rows), rows


def load_labels(path: Path | str) -> set[str]:
    labels = {
        str(row["target_subcategory"])
        for row in pq.read_table(
            path,
            columns=["target_subcategory"],
        ).to_pylist()
    }
    if not labels:
        raise ValueError("Label vocabulary must be non-empty")
    return labels


def _write_rows_atomically(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}"
    try:
        pq.write_table(
            pa.Table.from_pylist(rows),
            temporary,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_resume_shard(
    path: Path,
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    record_indices: list[int],
    *,
    mode: EvaluationMode,
) -> None:
    if len(rows) != len(records):
        raise ValueError(
            f"{path}: expected {len(records)} rows, got {len(rows)}"
        )
    expected_ids = [str(record["impression_id"]) for record in records]
    actual_ids = [str(row["impression_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{path}: impression IDs do not match evaluation data"
        )
    if [int(row["record_index"]) for row in rows] != record_indices:
        raise ValueError(f"{path}: record indices do not match")
    if any(str(row["mode"]) != mode for row in rows):
        raise ValueError(f"{path}: evaluation mode does not match")


def evaluate_cot_accuracy(
    config: dict[str, Any],
) -> dict[str, object]:
    mode = _normalize_mode(str(config.get("mode", "cot")))
    typed_mode: EvaluationMode = mode
    eval_file = Path(config["eval_file"])
    labels_file = Path(config["retained_interests_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_file)
    all_records = pq.read_table(eval_file).to_pylist()
    start_index = int(config.get("start_index", 0))
    if start_index < 0 or start_index >= len(all_records):
        raise ValueError("start_index is outside the evaluation data")
    max_examples = config.get("max_examples")
    stop_index = (
        min(start_index + int(max_examples), len(all_records))
        if max_examples is not None
        else len(all_records)
    )
    records = all_records[start_index:stop_index]
    if not records:
        raise ValueError("Evaluation data must be non-empty")
    for record in records:
        for column in (
            "impression_id",
            "prompt",
            "acceptable_subcategories",
            "history_subcategories",
        ):
            if column not in record:
                raise ValueError(f"Evaluation data is missing {column!r}")
    impression_ids = [str(record["impression_id"]) for record in records]
    if len(impression_ids) != len(set(impression_ids)):
        raise ValueError(
            "Grouped evaluation data has duplicate impression IDs"
        )

    checkpoint_size = int(config.get("checkpoint_size", 64))
    batch_size = int(config.get("batch_size", 4))
    max_new_tokens = int(
        config.get("max_new_tokens", 768 if mode == "cot" else 64)
    )
    instruction_version = str(
        config.get("cot_instruction_version", "current")
    )
    do_sample = bool(config.get("do_sample", False))
    temperature = (
        float(config.get("temperature", 0.6)) if do_sample else None
    )
    top_p = float(config.get("top_p", 0.95)) if do_sample else None
    top_k = int(config.get("top_k", 20)) if do_sample else None
    min_p = float(config.get("min_p", 0.0)) if do_sample else None
    seed = int(config["seed"]) if config.get("seed") is not None else None
    instruction_by_version = {
        "current": COT_INSTRUCTION,
        "legacy": LEGACY_COT_INSTRUCTION,
        "native": None,
    }
    if instruction_version not in instruction_by_version:
        raise ValueError(
            f"Unknown CoT instruction version: {instruction_version}"
        )
    if checkpoint_size <= 0 or batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("Generation and checkpoint sizes must be positive")
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
            raise ValueError("seed is required when sampling")
        if checkpoint_size % batch_size:
            raise ValueError(
                "checkpoint_size must be divisible by batch_size "
                "for resume-stable sampling"
            )

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing: list[tuple[int, int, Path]] = []
    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        shard_records = records[start:stop]
        indices = list(range(start_index + start, start_index + stop))
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            _validate_resume_shard(
                shard_path,
                shard_rows,
                shard_records,
                indices,
                mode=typed_mode,
            )
            rows.extend(shard_rows)
        else:
            missing.append((start, stop, shard_path))

    generated_count = 0
    started_at = time.time()
    if missing:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = str(config.get("model_name", "Qwen/Qwen3-8B"))
        adapter_value = config.get("adapter_path")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=str(
                config.get("attn_implementation", "sdpa")
            ),
            device_map={"": int(config.get("device", 0))},
        )
        if adapter_value:
            model = PeftModel.from_pretrained(model, adapter_value)
        model.eval()

        for start, stop, shard_path in missing:
            shard_records = records[start:stop]
            generations = generate_greedy_outputs(
                model,
                tokenizer,
                shard_records,
                mode=typed_mode,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                cot_instruction=instruction_by_version[
                    instruction_version
                ],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                record_offset=start_index + start,
            )
            _, shard_rows = evaluate_accuracy_outputs(
                shard_records,
                [str(row["raw_output"]) for row in generations],
                labels,
                mode=typed_mode,
                record_indices=list(
                    range(start_index + start, start_index + stop)
                ),
                response_tokens=[
                    int(row["response_tokens"]) for row in generations
                ],
                reached_max_new_tokens=[
                    bool(row["reached_max_new_tokens"])
                    for row in generations
                ],
            )
            _write_rows_atomically(shard_path, shard_rows)
            rows.extend(shard_rows)
            generated_count += len(shard_rows)

    rows.sort(key=lambda row: int(row["record_index"]))
    expected_indices = list(range(start_index, stop_index))
    if [int(row["record_index"]) for row in rows] != expected_indices:
        raise RuntimeError("Final prediction indices do not match eval data")
    metrics = summarize_accuracy_rows(rows)
    runtime_seconds = time.time() - started_at
    metrics.update(
        {
            "mode": mode,
            "cot_instruction_version": instruction_version,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "seed": seed,
            "start_index": start_index,
            "stop_index": stop_index,
            "label_count": len(labels),
            "generated_samples_this_run": generated_count,
            "resumed_samples": len(rows) - generated_count,
            "checkpoint_size": checkpoint_size,
            "shard_count": (
                len(records) + checkpoint_size - 1
            )
            // checkpoint_size,
            "runtime_seconds": runtime_seconds,
            "samples_per_second": (
                generated_count / runtime_seconds
                if generated_count
                else None
            ),
        }
    )
    _write_rows_atomically(output_dir / "predictions.parquet", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_config = {
        **config,
        "mode": mode,
        "eval_file": str(eval_file),
        "retained_interests_file": str(labels_file),
        "output_dir": str(output_dir),
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    wandb_mode = str(config.get("wandb_mode", "disabled"))
    if wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-cot-accuracy",
                )
            ),
            name=str(config["run_name"]),
            job_type="cot-accuracy-evaluation",
            mode=wandb_mode,
            config=resolved_config,
        )
        run.log(
            {
                f"eval/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        artifact = wandb.Artifact(
            f"{config['run_name']}-predictions",
            type="accuracy-evaluation",
        )
        artifact.add_file(str(output_dir / "predictions.parquet"))
        artifact.add_file(str(output_dir / "metrics.json"))
        run.log_artifact(artifact)
        run.finish()
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate grouped Dev accuracy with or without CoT."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument(
        "--retained-interests-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("cot", "no_cot", "no-cot"),
        required=True,
    )
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--cot-instruction-version",
        choices=("current", "legacy", "native"),
        default="current",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb-project",
        default="cross-interest-cot-accuracy",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = {
        key: value
        for key, value in vars(args).items()
        if value is not None
    }
    print(json.dumps(evaluate_cot_accuracy(config), indent=2))


if __name__ == "__main__":
    main()
