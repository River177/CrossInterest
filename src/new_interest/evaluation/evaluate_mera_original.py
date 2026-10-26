from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from new_interest.experiments.mera_original_core import (
    inference_text,
    load_categories,
    load_json,
    load_records,
    load_resolved_config,
    precision_recall_at_k,
    quote_category,
    parse_quoted_values,
    tensor_mapping_manifest,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def load_model(config: Any, adapter_dir: Path) -> tuple[Any, Any, tuple[str, ...], tuple[int, ...]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    categories = load_categories(config.data_path("category_map"))
    category_tokens = tuple(quote_category(value) for value in categories)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = int(config.model["pad_token_id"])
    token_ids = tuple(tokenizer.convert_tokens_to_ids(value) for value in category_tokens)
    for token, token_id in zip(category_tokens, token_ids):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise RuntimeError(f"Reloaded category is not one token: {token}")
    base = AutoModelForCausalLM.from_pretrained(
        config.model["name"],
        revision=config.model["revision"],
        cache_dir=os.environ.get("HF_HOME"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        load_in_8bit=bool(config.model["load_in_8bit"]),
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()
    return model, tokenizer, categories, token_ids


def score_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    token_ids: Sequence[int],
    *,
    cutoff_len: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    import torch

    predictions: list[dict[str, Any]] = []
    all_candidate_logits: list[list[float]] = []
    token_index = torch.tensor(token_ids, dtype=torch.long, device=model.device)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [inference_text(row) for row in batch_rows]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cutoff_len,
        )
        batch = move_batch(batch, model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1]
            candidate_logits = logits.index_select(1, token_index).float()
            top_values, top_offsets = torch.topk(candidate_logits, k=4, dim=1)
        all_candidate_logits.extend(candidate_logits.cpu().tolist())
        for local_index, row in enumerate(batch_rows):
            row_index = start + local_index
            offsets = top_offsets[local_index].cpu().tolist()
            predicted = [quote_category(categories[offset]) for offset in offsets]
            targets = [
                quote_category(value)
                for value in parse_quoted_values(str(row["output"]))
            ]
            if len(set(predicted)) != 4:
                raise RuntimeError(f"Duplicate Top-4 prediction at row {row_index}")
            predictions.append(
                {
                    "row_index": row_index,
                    "inference_text": texts[local_index],
                    "history_categories": list(
                        parse_quoted_values(str(row["input"]))
                    ),
                    "target_categories": targets,
                    "top4_categories": predicted,
                    "top4_token_ids": [token_ids[offset] for offset in offsets],
                    "top4_logits": top_values[local_index].float().cpu().tolist(),
                }
            )
    return predictions, all_candidate_logits


def score_reference_logits(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    token_ids: Sequence[int],
    *,
    cutoff_len: int,
) -> list[list[float]]:
    import torch

    outputs: list[list[float]] = []
    for row in rows:
        batch = tokenizer(
            inference_text(row),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cutoff_len,
        )
        batch = move_batch(batch, model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1]
            index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
            outputs.append(logits.index_select(1, index).float().cpu()[0].tolist())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("arts", "grocery"))
    parser.add_argument(
        "--contract", required=True, choices=("code-faithful", "paper-spec")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_resolved_config(args.config, args.dataset, args.contract)
    adapter_dir = Path(args.adapter_dir).resolve()
    training_dir = Path(args.training_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, categories, token_ids = load_model(config, adapter_dir)
    rows = load_records(config.data_path("test_file"))
    if args.max_eval_examples is not None:
        rows = rows[: args.max_eval_examples]
    predictions, _ = score_rows(
        model,
        tokenizer,
        rows,
        categories,
        token_ids,
        cutoff_len=int(config.training["cutoff_len"]),
        batch_size=args.batch_size,
    )
    metrics = precision_recall_at_k(
        [row["top4_categories"] for row in predictions],
        [row["target_categories"] for row in predictions],
    )
    per_example = metrics.pop("per_example")
    for prediction, values in zip(predictions, per_example):
        prediction["precision_at_k"] = values["precision_at_k"]
        prediction["recall_at_k"] = values["recall_at_k"]

    reference = load_json(training_dir / "reload_reference.json")
    reference_rows = reference["row_indices"]
    if reference_rows != list(range(len(reference_rows))):
        raise RuntimeError("Unexpected reload reference row indices")
    reload_logits = score_reference_logits(
        model,
        tokenizer,
        rows[: len(reference_rows)],
        token_ids,
        cutoff_len=int(config.training["cutoff_len"]),
    )
    trainer_to_fresh_max_abs_error = 0.0
    for expected, actual in zip(
        reference["candidate_logits"], reload_logits
    ):
        for expected_value, actual_value in zip(expected, actual):
            trainer_to_fresh_max_abs_error = max(
                trainer_to_fresh_max_abs_error,
                abs(float(expected_value) - float(actual_value)),
            )
    reference_topk = [
        sorted(range(len(row)), key=lambda index: row[index], reverse=True)[:4]
        for row in reference["candidate_logits"]
    ]
    reload_topk = [
        sorted(range(len(row)), key=lambda index: row[index], reverse=True)[:4]
        for row in reload_logits
    ]
    trainer_to_fresh_topk_exact = reference_topk == reload_topk
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors

    state_reference = load_json(training_dir / "adapter_state_reference.json")
    saved_state = load_safetensors(adapter_dir / "adapter_model.safetensors")
    saved_state_manifest = tensor_mapping_manifest(saved_state)
    saved_state_exact = state_reference == saved_state_manifest
    live_state = get_peft_model_state_dict(model)
    live_keys_exact = set(saved_state) == set(live_state)
    live_state_exact_after_dtype_cast = live_keys_exact
    live_mismatches: list[str] = []
    if live_keys_exact:
        for name, saved_tensor in saved_state.items():
            live_tensor = live_state[name].detach().cpu()
            expected = saved_tensor.to(dtype=live_tensor.dtype)
            if expected.shape != live_tensor.shape or not torch.equal(expected, live_tensor):
                live_state_exact_after_dtype_cast = False
                live_mismatches.append(name)
    reloaded_state = tensor_mapping_manifest(live_state)
    tolerance = float(config.evaluation["reload_max_abs_error"])
    if not saved_state_exact:
        raise RuntimeError("Saved adapter file differs from the training state manifest")
    if not live_state_exact_after_dtype_cast:
        raise RuntimeError(
            f"Fresh reload PEFT state differs after exact dtype cast: {live_mismatches[:5]}"
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    repeat_model, repeat_tokenizer, repeat_categories, repeat_token_ids = load_model(
        config, adapter_dir
    )
    if repeat_categories != categories or repeat_token_ids != token_ids:
        raise RuntimeError("Second fresh reload changed category token identity")
    repeat_logits = score_reference_logits(
        repeat_model,
        repeat_tokenizer,
        rows[: len(reference_rows)],
        repeat_token_ids,
        cutoff_len=int(config.training["cutoff_len"]),
    )
    fresh_repeat_max_abs_error = 0.0
    for first, second in zip(reload_logits, repeat_logits):
        for first_value, second_value in zip(first, second):
            fresh_repeat_max_abs_error = max(
                fresh_repeat_max_abs_error,
                abs(float(first_value) - float(second_value)),
            )
    repeat_topk = [
        sorted(range(len(row)), key=lambda index: row[index], reverse=True)[:4]
        for row in repeat_logits
    ]
    fresh_repeat_topk_exact = reload_topk == repeat_topk
    repeat_live_state = get_peft_model_state_dict(repeat_model)
    repeat_keys_exact = set(saved_state) == set(repeat_live_state)
    repeat_state_exact_after_dtype_cast = repeat_keys_exact
    if repeat_keys_exact:
        for name, saved_tensor in saved_state.items():
            repeat_tensor = repeat_live_state[name].detach().cpu()
            expected = saved_tensor.to(dtype=repeat_tensor.dtype)
            if expected.shape != repeat_tensor.shape or not torch.equal(
                expected, repeat_tensor
            ):
                repeat_state_exact_after_dtype_cast = False
                break
    if not repeat_state_exact_after_dtype_cast:
        raise RuntimeError("Second fresh reload PEFT state differs after dtype cast")
    if (
        bool(config.evaluation["reload_require_topk_exact"])
        and not fresh_repeat_topk_exact
    ):
        raise RuntimeError("Two independent fresh reloads have different Top-4")
    repeat_tolerance = float(config.evaluation["fresh_repeat_max_abs_error"])
    if fresh_repeat_max_abs_error > repeat_tolerance:
        raise RuntimeError(
            f"Two independent fresh reload logits differ: {fresh_repeat_max_abs_error}"
        )
    reload_audit = {
        "audit_version": "mera-original-reload-audit-v1",
        "reference_examples": len(reference_rows),
        "candidate_categories": len(categories),
        "trainer_to_fresh_max_abs_error": trainer_to_fresh_max_abs_error,
        "trainer_to_fresh_tolerance": tolerance,
        "trainer_to_fresh_top4_exact": trainer_to_fresh_topk_exact,
        "fresh_repeat_max_abs_error": fresh_repeat_max_abs_error,
        "fresh_repeat_tolerance": repeat_tolerance,
        "fresh_repeat_top4_exact": fresh_repeat_topk_exact,
        "saved_adapter_state_exact": saved_state_exact,
        "live_adapter_keys_exact": live_keys_exact,
        "live_adapter_state_exact_after_dtype_cast": live_state_exact_after_dtype_cast,
        "live_adapter_mismatches": live_mismatches,
        "repeat_live_state_exact_after_dtype_cast": repeat_state_exact_after_dtype_cast,
        "adapter_tensor_count": reloaded_state["tensor_count"],
        "adapter_value_count": reloaded_state["value_count"],
        "forward_path": "batch-one-compatible-raw-forward-v1",
    }
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics_payload = {
        "metrics_version": "mera-original-pr-at-k-v1",
        "identity": config.identity(),
        "dataset": args.dataset,
        "contract": args.contract,
        "examples": metrics["examples"],
        "precision_at_k": metrics["precision_at_k"],
        "recall_at_k": metrics["recall_at_k"],
        "precision_at_k_rounded": [round(value, 4) for value in metrics["precision_at_k"]],
        "recall_at_k_rounded": [round(value, 4) for value in metrics["recall_at_k"]],
        "invalid_predictions": 0,
        "duplicate_predictions": 0,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    write_json(output_dir / "reload_audit.json", reload_audit)
    print("MERA_ORIGINAL_EVAL_OK", json.dumps(metrics_payload, sort_keys=True))


if __name__ == "__main__":
    main()
