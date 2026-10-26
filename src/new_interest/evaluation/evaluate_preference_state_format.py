from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.training.semantic_id_tokens import (
    prepare_semantic_adapter_base,
    validate_saved_interest_tokens,
)
from new_interest.training.sft_data import apply_chat_template


_FORMAT = re.compile(
    r"\A<think>\s*(?P<reasoning>[\s\S]*?\S)\s*</think>\s*"
    r"<answer>\s*(?P<token><interest_\d{3}>)\s*</answer>\s*\Z"
)
_SID = re.compile(r"<interest_\d{3}>")


def parse_preference_state_completion(
    generated_text: str,
    *,
    legal_tokens: set[str],
    label_by_token: dict[str, str],
) -> dict[str, object]:
    full = generated_text.strip()
    if not full.startswith("<think>"):
        full = "<think>\n" + full
    match = _FORMAT.fullmatch(full)
    if match is None:
        return {
            "full_completion": full,
            "valid_format": False,
            "legal_token": False,
            "predicted_token": None,
            "predicted_subcategory": None,
            "reasoning_hides_prediction": False,
        }
    token = match.group("token")
    legal = token in legal_tokens
    label = label_by_token.get(token)
    reasoning = match.group("reasoning")
    hidden = (
        legal
        and label is not None
        and label.casefold() not in reasoning.casefold()
        and len(_SID.findall(reasoning)) == 0
    )
    return {
        "full_completion": full,
        "valid_format": True,
        "legal_token": legal,
        "predicted_token": token,
        "predicted_subcategory": label,
        "reasoning_hides_prediction": hidden,
    }


def evaluate_format(
    *,
    model_name: str,
    adapter_path: Path | str,
    eval_file: Path | str,
    mapping_file: Path | str,
    output_dir: Path | str,
    start_index: int = 108,
    max_examples: int = 16,
    max_new_tokens: int = 256,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    eval_file = Path(eval_file)
    mapping_file = Path(mapping_file)
    output_dir = Path(output_dir)
    if start_index < 0 or max_examples <= 0 or max_new_tokens <= 0:
        raise ValueError("Invalid format-evaluation range")
    rows = pq.read_table(eval_file).slice(start_index, max_examples).to_pylist()
    if len(rows) != max_examples:
        raise ValueError("Format-evaluation range exceeds the dataset")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    spec = validate_saved_interest_tokens(tokenizer, mapping_file)
    legal_tokens = {entry.interest_token for entry in spec.entries}
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
    stop_ids = {int(tokenizer.eos_token_id)}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)

    output_rows = []
    for offset, row in enumerate(rows, start=start_index):
        rendered = apply_chat_template(
            tokenizer,
            [{"role": "user", "content": str(row["prompt"])}],
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
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(stop_ids),
                use_cache=True,
            )
        prompt_length = int(encoded["input_ids"].shape[1])
        generated_ids = generated[0, prompt_length:].tolist()
        while generated_ids and generated_ids[-1] in stop_ids:
            generated_ids.pop()
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        parsed = parse_preference_state_completion(
            text,
            legal_tokens=legal_tokens,
            label_by_token=label_by_token,
        )
        target_token = str(row["response"])
        output_rows.append(
            {
                "record_index": offset,
                "source_sample_id": str(row["source_sample_id"]),
                "target_token": target_token,
                "target_subcategory": str(row["target_subcategory"]),
                "generated_text": text,
                "generated_tokens": len(generated_ids),
                **parsed,
                "exact": parsed["predicted_token"] == target_token,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / "format_predictions.parquet"
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        predictions,
        compression="zstd",
    )
    summary = {
        "model_name": model_name,
        "adapter_path": str(adapter_path),
        "eval_file": str(eval_file),
        "start_index": start_index,
        "samples": len(output_rows),
        "max_new_tokens": max_new_tokens,
        "valid_format": sum(bool(row["valid_format"]) for row in output_rows),
        "legal_token": sum(bool(row["legal_token"]) for row in output_rows),
        "reasoning_hides_prediction": sum(
            bool(row["reasoning_hides_prediction"]) for row in output_rows
        ),
        "exact": sum(bool(row["exact"]) for row in output_rows),
        "semantic_base": semantic_base,
    }
    (output_dir / "format_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=108)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            evaluate_format(
                model_name=args.model_name,
                adapter_path=args.adapter_path,
                eval_file=args.eval_file,
                mapping_file=args.mapping_file,
                output_dir=args.output_dir,
                start_index=args.start_index,
                max_examples=args.max_examples,
                max_new_tokens=args.max_new_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
