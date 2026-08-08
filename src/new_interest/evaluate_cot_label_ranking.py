from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.build_grpo_dataset import append_cot_instruction
from new_interest.grpo_reward import (
    extract_final_answer,
    parse_structured_output,
)
from new_interest.metrics import (
    RESPONSE_PREFIX,
    evaluate_ranked_outputs,
)
from new_interest.sft_data import apply_chat_template


@dataclass(frozen=True)
class RepairedReasoning:
    reasoning: str
    repair_type: str


def load_labels(path: Path | str) -> list[str]:
    labels = sorted(
        {
            str(row["target_subcategory"])
            for row in pq.read_table(
                path,
                columns=["target_subcategory"],
            ).to_pylist()
        }
    )
    if not labels:
        raise ValueError("Label vocabulary must be non-empty")
    return labels


def repair_reasoning(output: str) -> RepairedReasoning:
    parsed = parse_structured_output(output)
    if parsed is not None:
        return RepairedReasoning(parsed.think, "none")

    stripped = output.strip()
    if "<think>" in stripped:
        reasoning = stripped.split("<think>", 1)[1]
        if "</think>" in reasoning:
            reasoning = reasoning.split("</think>", 1)[0]
            repair_type = "rebuild_after_closed_think"
        elif RESPONSE_PREFIX in reasoning:
            reasoning = reasoning.split(RESPONSE_PREFIX, 1)[0]
            repair_type = "close_unclosed_think_before_answer"
        else:
            repair_type = "close_unclosed_think"
    else:
        reasoning = (
            stripped.split(RESPONSE_PREFIX, 1)[0]
            if RESPONSE_PREFIX in stripped
            else ""
        )
        repair_type = (
            "wrap_text_before_answer"
            if reasoning.strip()
            else "empty_reasoning"
        )

    reasoning = (
        reasoning.replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )
    return RepairedReasoning(reasoning, repair_type)


def build_reasoning_prefix(
    rendered_prompt: str,
    reasoning: str,
) -> str:
    return (
        f"{rendered_prompt}<think>\n{reasoning}\n</think>\n"
        f"{RESPONSE_PREFIX.rstrip()}"
    )


def tokenize_label_continuations(
    tokenizer: Any,
    prefix: str,
    labels: list[str],
) -> tuple[list[int], list[list[int]]]:
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
    )["input_ids"]
    continuations: list[list[int]] = []
    for label in labels:
        full_ids = tokenizer(
            f"{prefix} {label}",
            add_special_tokens=False,
        )["input_ids"]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"Tokenizer boundary changed before label {label!r}"
            )
        continuation = full_ids[len(prefix_ids) :]
        if not continuation:
            raise ValueError(f"Label {label!r} has no continuation tokens")
        continuations.append(continuation)
    return prefix_ids, continuations


def _repeat_past_key_values(
    past_key_values: Any,
    repeats: int,
) -> Any:
    if hasattr(past_key_values, "batch_repeat_interleave"):
        past_key_values.batch_repeat_interleave(repeats)
        return past_key_values
    return tuple(
        tuple(
            tensor.repeat_interleave(repeats, dim=0)
            for tensor in layer
        )
        for layer in past_key_values
    )


def _crop_past_key_values(
    past_key_values: Any,
    prefix_length: int,
) -> Any:
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(prefix_length)
        return past_key_values
    return tuple(
        tuple(
            tensor[:, :, :prefix_length, :]
            for tensor in layer
        )
        for layer in past_key_values
    )


def score_label_continuations(
    model: Any,
    tokenizer: Any,
    prefix: str,
    labels: list[str],
    *,
    candidate_batch_size: int = 22,
) -> list[float]:
    import torch

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("Tokenizer must define EOS and PAD token IDs")

    prefix_ids, continuation_ids = tokenize_label_continuations(
        tokenizer,
        prefix,
        labels,
    )
    device = model.device
    prefix_tensor = torch.tensor(
        [prefix_ids],
        dtype=torch.long,
        device=device,
    )
    prefix_attention = torch.ones_like(prefix_tensor)

    with torch.inference_mode():
        prefix_output = model(
            input_ids=prefix_tensor,
            attention_mask=prefix_attention,
            use_cache=True,
            return_dict=True,
        )
    prefix_log_probs = torch.log_softmax(
        prefix_output.logits[:, -1, :].float(),
        dim=-1,
    )
    working_cache = _repeat_past_key_values(
        prefix_output.past_key_values,
        candidate_batch_size,
    )
    prefix_length = len(prefix_ids)
    scores: list[float] = []

    for start in range(0, len(labels), candidate_batch_size):
        real_chunk = continuation_ids[
            start : start + candidate_batch_size
        ]
        padded_chunk = list(real_chunk)
        while len(padded_chunk) < candidate_batch_size:
            padded_chunk.append(real_chunk[0])
        max_length = max(len(tokens) for tokens in padded_chunk)
        input_ids = torch.full(
            (candidate_batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        continuation_mask = torch.zeros(
            (candidate_batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        for index, tokens in enumerate(padded_chunk):
            length = len(tokens)
            input_ids[index, :length] = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            )
            continuation_mask[index, :length] = 1
        attention_mask = torch.cat(
            [
                torch.ones(
                    (candidate_batch_size, prefix_length),
                    dtype=torch.long,
                    device=device,
                ),
                continuation_mask,
            ],
            dim=1,
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=working_cache,
                use_cache=True,
                return_dict=True,
            )
        token_log_probs = torch.log_softmax(
            output.logits.float(),
            dim=-1,
        )
        for index, tokens in enumerate(real_chunk):
            total = float(prefix_log_probs[0, tokens[0]].item())
            for position in range(1, len(tokens)):
                total += float(
                    token_log_probs[
                        index,
                        position - 1,
                        tokens[position],
                    ].item()
                )
            total += float(
                token_log_probs[
                    index,
                    len(tokens) - 1,
                    eos_token_id,
                ].item()
            )
            scores.append(total / (len(tokens) + 1))
        working_cache = _crop_past_key_values(
            output.past_key_values,
            prefix_length,
        )

    if len(scores) != len(labels):
        raise RuntimeError(
            f"Expected {len(labels)} scores, got {len(scores)}"
        )
    return scores


def rank_labels(
    labels: list[str],
    scores: list[float],
) -> list[dict[str, object]]:
    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have equal length")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("All label scores must be finite")
    ranked = sorted(
        zip(labels, scores),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        {"label": label, "score": float(score)}
        for label, score in ranked
    ]


def generate_greedy_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    import torch

    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("Generation sizes must be positive")
    rendered_prompts = [
        apply_chat_template(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": append_cot_instruction(
                        str(record["prompt"])
                    ),
                }
            ],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for record in records
    ]
    outputs: list[dict[str, object]] = []
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    for start in range(0, len(records), batch_size):
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
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, input_length:]
        for offset, token_row in enumerate(new_tokens):
            token_ids = token_row.tolist()
            reached_eos = tokenizer.eos_token_id in token_ids
            if reached_eos:
                token_count = token_ids.index(tokenizer.eos_token_id)
            else:
                token_count = len(token_ids)
            output = tokenizer.decode(
                token_ids[:token_count],
                skip_special_tokens=True,
            )
            outputs.append(
                {
                    "rendered_prompt": batch_prompts[offset],
                    "raw_output": output,
                    "response_tokens": token_count,
                    "reached_max_new_tokens": (
                        not reached_eos
                        and token_count >= max_new_tokens
                    ),
                }
            )
    return outputs


def _label_from_response(response: str | None) -> str | None:
    if response is None or not response.startswith(RESPONSE_PREFIX):
        return None
    return response[len(RESPONSE_PREFIX) :].strip()


def build_evaluation_row(
    *,
    record_index: int,
    record: dict[str, object],
    generation: dict[str, object],
    labels: list[str],
    ranked: list[dict[str, object]],
) -> dict[str, object]:
    output = str(generation["raw_output"])
    parsed = parse_structured_output(output)
    repair = repair_reasoning(output)
    loose_answer = extract_final_answer(output)
    loose_label = _label_from_response(loose_answer)
    strict_label = (
        _label_from_response(parsed.answer)
        if parsed is not None
        else None
    )
    acceptable = {
        str(value)
        for value in record["acceptable_subcategories"]
    }
    history = {
        str(value) for value in record["history_subcategories"]
    }
    valid_labels = set(labels)
    return {
        "record_index": record_index,
        "impression_id": str(record["impression_id"]),
        "prompt": str(record["prompt"]),
        "raw_output": output,
        "response_tokens": int(generation["response_tokens"]),
        "reached_max_new_tokens": bool(
            generation["reached_max_new_tokens"]
        ),
        "strict_format": parsed is not None,
        "strict_reasoning": parsed.think if parsed is not None else None,
        "repaired_reasoning": repair.reasoning,
        "repair_type": repair.repair_type,
        "greedy_answer": loose_answer,
        "greedy_label": loose_label,
        "greedy_valid_label": loose_label in valid_labels,
        "greedy_set_hit": loose_label in acceptable,
        "greedy_strict_hit": strict_label in acceptable,
        "novelty_compliant": (
            loose_label in valid_labels and loose_label not in history
        ),
        "acceptable_subcategories": sorted(acceptable),
        "history_subcategories": sorted(history),
        "ranked_labels": [str(item["label"]) for item in ranked],
        "ranked_scores": [float(item["score"]) for item in ranked],
    }


def summarize_greedy(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Rows must be non-empty")
    total = len(rows)
    repair_counts = Counter(str(row["repair_type"]) for row in rows)
    return {
        "samples": total,
        "strict_format_rate": sum(
            bool(row["strict_format"]) for row in rows
        )
        / total,
        "valid_label_rate": sum(
            bool(row["greedy_valid_label"]) for row in rows
        )
        / total,
        "greedy_set_hit_at_1": sum(
            bool(row["greedy_set_hit"]) for row in rows
        )
        / total,
        "greedy_strict_hit_at_1": sum(
            bool(row["greedy_strict_hit"]) for row in rows
        )
        / total,
        "novelty_compliance_rate": sum(
            bool(row["novelty_compliant"]) for row in rows
        )
        / total,
        "truncation_rate": sum(
            bool(row["reached_max_new_tokens"]) for row in rows
        )
        / total,
        "mean_response_tokens": mean(
            int(row["response_tokens"]) for row in rows
        ),
        "repair_counts": dict(sorted(repair_counts.items())),
    }


def _predictions_from_row(
    row: dict[str, object],
) -> list[dict[str, object]]:
    return [
        {"label": label, "score": score}
        for label, score in zip(
            row["ranked_labels"],
            row["ranked_scores"],
        )
    ]


def compute_all_metrics(
    records: list[dict[str, object]],
    rows: list[dict[str, object]],
    *,
    cutoffs: tuple[int, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    repaired_rankings = [_predictions_from_row(row) for row in rows]
    strict_rankings = [
        ranking if bool(row["strict_format"]) else []
        for row, ranking in zip(rows, repaired_rankings)
    ]
    strict_metrics, strict_details = evaluate_ranked_outputs(
        records,
        strict_rankings,
        cutoffs=cutoffs,
    )
    repaired_metrics, repaired_details = evaluate_ranked_outputs(
        records,
        repaired_rankings,
        cutoffs=cutoffs,
    )
    prediction_rows = []
    for row, strict, repaired in zip(
        rows,
        strict_details,
        repaired_details,
    ):
        merged = dict(row)
        for key, value in strict.items():
            if key not in {
                "impression_id",
                "ranked_labels",
                "ranked_scores",
                "acceptable_subcategories",
            }:
                merged[f"strict_{key}"] = value
        for key, value in repaired.items():
            if key not in {
                "impression_id",
                "ranked_labels",
                "ranked_scores",
                "acceptable_subcategories",
            }:
                merged[f"repaired_{key}"] = value
        prediction_rows.append(merged)
    return {
        "greedy": summarize_greedy(rows),
        "strict_ranking": strict_metrics,
        "repaired_ranking": repaired_metrics,
    }, prediction_rows


def _write_rows_atomically(
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
        temporary_path = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows),
            temporary_path,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_cot_label_ranking(
    config: dict[str, Any],
) -> dict[str, object]:
    print("CoT label evaluation: importing model dependencies", flush=True)
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("CoT label evaluation: dependencies imported", flush=True)
    model_name = str(config["model_name"])
    adapter_path = Path(config["adapter_path"])
    eval_file = Path(config["eval_file"])
    labels_file = Path(config["retained_interests_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = tuple(
        sorted({int(value) for value in config.get("cutoffs", (10, 20, 50))})
    )
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("Metric cutoffs must be positive")

    labels = load_labels(labels_file)
    if max(cutoffs) > len(labels):
        raise ValueError("Metric cutoff exceeds label vocabulary")
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
    print(
        "CoT label evaluation: loaded",
        len(records),
        "records and",
        len(labels),
        "labels",
        flush=True,
    )

    checkpoint_size = int(config.get("checkpoint_size", 16))
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, object]] = []
    missing_ranges: list[tuple[int, int, Path]] = []
    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_ids = [
                str(record["impression_id"])
                for record in records[start:stop]
            ]
            actual_ids = [
                str(row["impression_id"]) for row in shard_rows
            ]
            if len(shard_rows) != stop - start or actual_ids != expected_ids:
                raise ValueError(f"{shard_path}: invalid resume shard")
            existing_rows.extend(shard_rows)
        else:
            missing_ranges.append((start, stop, shard_path))

    generated_count = 0
    started_at = time.time()
    if missing_ranges:
        print("CoT label evaluation: loading tokenizer", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print("CoT label evaluation: loading base model", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        print("CoT label evaluation: loading adapter", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        model.config.use_cache = True
        print("CoT label evaluation: model ready", flush=True)

        for start, stop, shard_path in missing_ranges:
            print(
                f"CoT label evaluation: shard {start}:{stop}",
                flush=True,
            )
            shard_records = records[start:stop]
            generations = generate_greedy_outputs(
                model,
                tokenizer,
                shard_records,
                batch_size=int(config.get("batch_size", 4)),
                max_new_tokens=int(config.get("max_new_tokens", 768)),
            )
            shard_rows = []
            for offset, (record, generation) in enumerate(
                zip(shard_records, generations)
            ):
                repair = repair_reasoning(str(generation["raw_output"]))
                scoring_prefix = build_reasoning_prefix(
                    str(generation["rendered_prompt"]),
                    repair.reasoning,
                )
                scores = score_label_continuations(
                    model,
                    tokenizer,
                    scoring_prefix,
                    labels,
                    candidate_batch_size=int(
                        config.get("candidate_batch_size", 22)
                    ),
                )
                ranked = rank_labels(labels, scores)
                shard_rows.append(
                    build_evaluation_row(
                        record_index=start_index + start + offset,
                        record=record,
                        generation=generation,
                        labels=labels,
                        ranked=ranked,
                    )
                )
            _write_rows_atomically(shard_path, shard_rows)
            existing_rows.extend(shard_rows)
            generated_count += len(shard_rows)

    rows = sorted(existing_rows, key=lambda row: int(row["record_index"]))
    if len(rows) != len(records):
        raise RuntimeError(
            f"Expected {len(records)} rows, got {len(rows)}"
        )
    if [
        str(row["impression_id"]) for row in rows
    ] != [
        str(record["impression_id"]) for record in records
    ]:
        raise RuntimeError("Final prediction order does not match eval data")
    if any(len(row["ranked_labels"]) != len(labels) for row in rows):
        raise RuntimeError("A ranking does not contain the full vocabulary")

    metric_groups, prediction_rows = compute_all_metrics(
        records,
        rows,
        cutoffs=cutoffs,
    )
    runtime_seconds = time.time() - started_at
    metrics = {
        "samples": len(records),
        "start_index": start_index,
        "stop_index": stop_index,
        "label_count": len(labels),
        "cutoffs": list(cutoffs),
        "condition": str(config["condition"]),
        "task_type": str(config["task_type"]),
        "generated_samples_this_run": generated_count,
        "resumed_samples": len(records) - generated_count,
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
        **metric_groups,
    }
    _write_rows_atomically(
        output_dir / "predictions.parquet",
        prediction_rows,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_config = {
        **config,
        "adapter_path": str(adapter_path),
        "eval_file": str(eval_file),
        "retained_interests_file": str(labels_file),
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    wandb_mode = str(config.get("wandb_mode", "disabled"))
    if wandb_mode != "disabled":
        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-generative-grpo-eval",
                )
            ),
            name=str(config["run_name"]),
            job_type="cot-label-ranking-evaluation",
            mode=wandb_mode,
            config=resolved_config,
        )
        scalars = {
            "eval/strict_format_rate": metrics["greedy"][
                "strict_format_rate"
            ],
            "eval/greedy_strict_hit_at_1": metrics["greedy"][
                "greedy_strict_hit_at_1"
            ],
        }
        for group_name in ("strict_ranking", "repaired_ranking"):
            for cutoff in cutoffs:
                for metric_name in ("recall", "ndcg"):
                    key = f"{metric_name}_at_{cutoff}"
                    scalars[f"eval/{group_name}/{key}"] = metrics[
                        group_name
                    ][key]
        run.log(scalars)
        artifact = wandb.Artifact(
            f"{config['run_name']}-cot-label-ranking",
            type="evaluation",
        )
        artifact.add_file(str(output_dir / "metrics.json"))
        artifact.add_file(str(output_dir / "predictions.parquet"))
        artifact.add_file(str(output_dir / "resolved_config.json"))
        run.log_artifact(artifact)
        run.finish()
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate greedy CoT followed by full-label scoring."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument(
        "--retained-interests-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--condition",
        choices=("sft22k", "grpo1k"),
        required=True,
    )
    parser.add_argument(
        "--task-type",
        choices=("behavior_only", "interest_only"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-batch-size", type=int, default=22)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--checkpoint-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[10, 20, 50],
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb-project",
        default="cross-interest-generative-grpo-eval",
    )
    return parser


def main() -> None:
    print("CoT label evaluation: CLI start", flush=True)
    args = _build_parser().parse_args()
    config = vars(args)
    metrics = evaluate_cot_label_ranking(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
