from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import sha256_directory, sha256_file
from new_interest.training.semantic_id_tokens import (
    prepare_semantic_adapter_base,
    validate_saved_interest_tokens,
)
from new_interest.training.sft_data import apply_chat_template


RULE_VERSION = "reasoning-stability-semantic-id-v1"
DIRECT_SUFFIX = (
    "Respond with only the learned interest token assigned to the predicted "
    "category, in the form <interest_NNN>."
)
REASONING_SUFFIX = (
    "Reason briefly about two or more plausible new-interest transitions. "
    "Then put exactly one learned interest token on the final line, in the "
    "form <interest_NNN>, with no text after it."
)
TOKEN_PATTERN = re.compile(r"<interest_\d{3}>")


def reasoning_prompt(prompt: str) -> str:
    if not prompt.endswith(DIRECT_SUFFIX):
        raise ValueError("Prompt does not use the canonical Semantic-ID suffix")
    return prompt[: -len(DIRECT_SUFFIX)] + REASONING_SUFFIX


def summarize_reasoning_group(
    predicted_tokens: Sequence[str | None],
    *,
    acceptable_tokens: set[str],
    vocabulary: set[str],
) -> dict[str, object]:
    if not predicted_tokens:
        raise ValueError("A reasoning group must contain at least one rollout")
    valid = [token for token in predicted_tokens if token in vocabulary]
    correct = [token for token in valid if token in acceptable_tokens]
    counts = Counter(valid)
    entropy = 0.0
    if valid:
        for count in counts.values():
            probability = count / len(valid)
            entropy -= probability * math.log(probability)
    return {
        "rollouts": len(predicted_tokens),
        "valid_rollouts": len(valid),
        "correct_rollouts": len(correct),
        "valid_rate": len(valid) / len(predicted_tokens),
        "correct_rate": len(correct) / len(predicted_tokens),
        "unique_valid_predictions": len(counts),
        "prediction_entropy": entropy,
        "mixed_correctness": 0 < len(correct) < len(predicted_tokens),
    }


def _selected_records(path: Path, maximum: int) -> list[dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    single_positive = [
        row
        for row in rows
        if len(row["acceptable_subcategories"]) == 1
        and len(row["acceptable_responses"]) == 1
    ]
    ordered = sorted(
        single_positive,
        key=lambda row: (
            hashlib.sha256(
                f"20260901:reasoning-stability:{row['impression_id']}".encode()
            ).hexdigest(),
            str(row["impression_id"]),
        ),
    )
    if len(ordered) < maximum:
        raise ValueError(
            f"Requested {maximum} single-positive records, found {len(ordered)}"
        )
    return ordered[:maximum]


def _atomic_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=path.parent, delete=False
    ) as handle:
        pending = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows),
            pending,
            compression="zstd",
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent,
        delete=False
    ) as handle:
        pending = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def evaluate_reasoning_stability(
    *,
    model_name: str,
    adapter_path: Path | str,
    eval_file: Path | str,
    mapping_file: Path | str,
    output_dir: Path | str,
    max_examples: int = 16,
    rollouts: int = 8,
    max_new_tokens: int = 192,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Reasoning-stability evaluation requires CUDA")
    if max_examples <= 0 or rollouts < 2 or max_new_tokens <= 0:
        raise ValueError("Invalid evaluation size or rollout configuration")
    if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("Invalid sampling configuration")

    adapter_path = Path(adapter_path)
    eval_file = Path(eval_file)
    mapping_file = Path(mapping_file)
    output_dir = Path(output_dir)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    spec = validate_saved_interest_tokens(tokenizer, mapping_file)
    vocabulary = {entry.interest_token for entry in spec.entries}
    label_by_token = {
        entry.interest_token: entry.target_subcategory for entry in spec.entries
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    semantic_base = prepare_semantic_adapter_base(model, spec)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(torch.device("cuda", 0))
    model.eval()

    records = _selected_records(eval_file, max_examples)
    output_rows: list[dict[str, object]] = []
    for record_index, row in enumerate(records):
        prompt = reasoning_prompt(str(row["prompt"]))
        rendered = apply_chat_template(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=rollouts,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        completions = tokenizer.batch_decode(
            generated[:, prompt_tokens:],
            skip_special_tokens=False,
        )
        predicted_tokens: list[str | None] = []
        for completion in completions:
            matches = TOKEN_PATTERN.findall(completion)
            predicted_tokens.append(matches[-1] if matches else None)
        acceptable_tokens = {
            str(value) for value in row["acceptable_responses"]
        }
        group = summarize_reasoning_group(
            predicted_tokens,
            acceptable_tokens=acceptable_tokens,
            vocabulary=vocabulary,
        )
        output_rows.append(
            {
                "record_index": record_index,
                "impression_id": str(row["impression_id"]),
                "user_id": str(row["user_id"]),
                "acceptable_subcategories": [
                    str(value) for value in row["acceptable_subcategories"]
                ],
                "acceptable_tokens": sorted(acceptable_tokens),
                "reasoning_prompt": prompt,
                "completions": completions,
                "predicted_tokens": predicted_tokens,
                "predicted_labels": [
                    label_by_token.get(token) if token is not None else None
                    for token in predicted_tokens
                ],
                **group,
            }
        )

    prediction_path = output_dir / "reasoning_stability.parquet"
    _atomic_table(prediction_path, output_rows)
    total_rollouts = max_examples * rollouts
    total_valid = sum(int(row["valid_rollouts"]) for row in output_rows)
    total_correct = sum(int(row["correct_rollouts"]) for row in output_rows)
    summary = {
        "rule_version": RULE_VERSION,
        "model_name": model_name,
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_directory(adapter_path),
        "eval_file": str(eval_file),
        "eval_sha256": sha256_file(eval_file),
        "mapping_sha256": sha256_file(mapping_file),
        "max_examples": max_examples,
        "rollouts_per_example": rollouts,
        "total_rollouts": total_rollouts,
        "valid_rollouts": total_valid,
        "correct_rollouts": total_correct,
        "valid_rate": total_valid / total_rollouts,
        "correct_rate": total_correct / total_rollouts,
        "mixed_correctness_groups": sum(
            bool(row["mixed_correctness"]) for row in output_rows
        ),
        "mean_prediction_entropy": sum(
            float(row["prediction_entropy"]) for row in output_rows
        )
        / max_examples,
        "sampling": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
        "semantic_base": semantic_base,
        "prediction_sha256": sha256_file(prediction_path),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_reasoning_stability(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        eval_file=args.eval_file,
        mapping_file=args.mapping_file,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        rollouts=args.rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
