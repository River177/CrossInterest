from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.training.grpo_reward import (
    compute_score,
    extract_final_answer,
    parse_structured_output,
)
from new_interest.training.sft_data import apply_chat_template


def run_preflight(
    *,
    model_name: str,
    adapter_path: Path,
    data_path: Path,
    output_dir: Path,
    max_prompts: int,
    num_generations: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(data_path).slice(0, max_prompts).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    output_rows = []
    mixed_groups = 0
    all_zero_groups = 0
    all_one_groups = 0
    total_positive = 0
    total_nonzero = 0
    total_correct_answers = 0
    total_strict_correct = 0
    total_reward = 0.0
    total_structured = 0
    constant_partial_groups = 0
    response_lengths = []
    started_at = time.time()

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            apply_chat_template(
                tokenizer,
                [
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                    for message in row["prompt"]
                ],
                add_generation_prompt=True,
                enable_thinking=True,
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
        prompt_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_generations,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, prompt_length:]
        completions = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
        for batch_index, row in enumerate(batch):
            group = completions[
                batch_index * num_generations :
                (batch_index + 1) * num_generations
            ]
            ground_truth = str(row["reward_model"]["ground_truth"])
            rewards = [
                compute_score(
                    row["data_source"],
                    completion,
                    ground_truth,
                    row["extra_info"],
                )
                for completion in group
            ]
            structured = [
                parse_structured_output(completion) is not None
                for completion in group
            ]
            acceptable = {
                str(value)
                for value in row["extra_info"][
                    "acceptable_subcategories"
                ]
            }
            retained = {
                str(value)
                for value in row["extra_info"]["retained_subcategories"]
            }
            history = {
                str(value)
                for value in row["extra_info"]["history_subcategories"]
            }
            labels = []
            for completion in group:
                answer = extract_final_answer(completion)
                labels.append(
                    answer.removeprefix("New topic: ")
                    if answer is not None
                    else None
                )
            correct = [
                label is not None
                and label in acceptable
                and label in retained
                and label not in history
                for label in labels
            ]
            positive = sum(reward == 1.0 for reward in rewards)
            total_positive += positive
            total_nonzero += sum(reward > 0.0 for reward in rewards)
            total_correct_answers += sum(correct)
            total_strict_correct += sum(
                is_correct and is_structured
                for is_correct, is_structured in zip(
                    correct,
                    structured,
                )
            )
            total_reward += sum(rewards)
            total_structured += sum(structured)
            distinct_rewards = set(rewards)
            if len(distinct_rewards) > 1:
                mixed_groups += 1
            elif rewards[0] == 0.0:
                all_zero_groups += 1
            elif rewards[0] == 1.0:
                all_one_groups += 1
            else:
                constant_partial_groups += 1
            token_lengths = [
                len(
                    tokenizer(
                        completion,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for completion in group
            ]
            response_lengths.extend(token_lengths)
            output_rows.append(
                {
                    "source_sample_id": str(
                        row["extra_info"]["source_sample_id"]
                    ),
                    "ground_truth": ground_truth,
                    "acceptable_subcategories": sorted(acceptable),
                    "completions": group,
                    "parsed_labels": labels,
                    "correct": correct,
                    "rewards": rewards,
                    "structured": structured,
                    "response_token_lengths": token_lengths,
                }
            )

    total_completions = len(rows) * num_generations
    ordered_lengths = sorted(response_lengths)

    def percentile(fraction: float) -> int:
        index = min(
            len(ordered_lengths) - 1,
            max(0, int(len(ordered_lengths) * fraction + 0.999999) - 1),
        )
        return ordered_lengths[index]

    metrics = {
        "prompts": len(rows),
        "num_generations": num_generations,
        "total_completions": total_completions,
        "positive_completions": total_positive,
        "positive_completion_rate": total_positive / total_completions,
        "nonzero_completions": total_nonzero,
        "nonzero_completion_rate": total_nonzero / total_completions,
        "correct_answer_completions": total_correct_answers,
        "correct_answer_rate": total_correct_answers / total_completions,
        "strict_correct_completions": total_strict_correct,
        "strict_correct_rate": total_strict_correct / total_completions,
        "correctness_reward_mass": 0.8 * total_correct_answers,
        "format_bonus_reward_mass": 0.2 * total_strict_correct,
        "mean_reward": total_reward / total_completions,
        "structured_completions": total_structured,
        "structured_cot_rate": total_structured / total_completions,
        "mixed_reward_groups": mixed_groups,
        "mixed_reward_group_rate": mixed_groups / len(rows),
        "all_zero_groups": all_zero_groups,
        "all_zero_group_rate": all_zero_groups / len(rows),
        "all_one_groups": all_one_groups,
        "all_one_group_rate": all_one_groups / len(rows),
        "constant_partial_groups": constant_partial_groups,
        "constant_partial_group_rate": constant_partial_groups / len(rows),
        "mean_response_tokens": (
            sum(response_lengths) / len(response_lengths)
        ),
        "p95_response_tokens": percentile(0.95),
        "max_response_tokens": max(response_lengths),
        "truncated_completions": sum(
            length >= max_new_tokens for length in response_lengths
        ),
        "truncation_rate": sum(
            length >= max_new_tokens for length in response_lengths
        )
        / total_completions,
        "runtime_seconds": time.time() - started_at,
        "model_name": model_name,
        "adapter_path": str(adapter_path),
        "data_path": str(data_path),
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        output_dir / "preflight_samples.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight correctness-gated GRPO reward sparsity."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = run_preflight(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
