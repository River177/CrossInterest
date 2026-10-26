from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.training.semantic_id_tokens import (
    initialize_semantic_interest_token_rows,
    register_semantic_interest_tokens,
    trainable_token_indices,
    write_json,
)
from new_interest.training.mera_regularization import build_mera_regularizer
from new_interest.training.sft_data import (
    CompletionOnlyCollator,
    tokenize_training_example,
)
from new_interest.training.train_sft import (
    _load_config,
    _set_seed,
    taxonomy_restricted_causal_lm_loss,
)


def _representative_messages(
    train_file: Path,
    expected_tokens: set[str],
) -> list[tuple[str, list[dict[str, str]]]]:
    selected: dict[str, list[dict[str, str]]] = {}
    for batch in pq.ParquetFile(train_file).iter_batches(
        batch_size=2048,
        columns=["response", "messages"],
    ):
        for row in batch.to_pylist():
            token = str(row["response"])
            if token not in expected_tokens or token in selected:
                continue
            selected[token] = [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
                for message in row["messages"]
            ]
    missing = sorted(expected_tokens.difference(selected))
    if missing:
        raise ValueError(f"Training data misses ID tokens: {missing}")
    return [(token, selected[token]) for token in sorted(selected)]


def audit_semantic_id_token_gradients(
    config: dict[str, Any],
    output_path: Path | str,
    *,
    batch_size: int = 2,
) -> dict[str, object]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_path = Path(output_path)
    mapping_file = Path(config["semantic_id_token_mapping_file"])
    train_file = Path(config["train_file"])
    seed = int(config.get("seed", 42))
    _set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        revision=config.get("model_revision"),
        token=config.get("hf_token"),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    spec = register_semantic_interest_tokens(
        tokenizer,
        mapping_file,
        expected_count=int(config.get("semantic_id_token_count", 108)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(config["model_name"]),
        revision=config.get("model_revision"),
        token=config.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        device_map={"": 0},
    )
    model.resize_token_embeddings(spec.resized_vocabulary_size)
    initialization_audit, _ = initialize_semantic_interest_token_rows(
        model,
        spec,
        initialization_mode=str(
            config.get(
                "semantic_id_initialization_mode",
                "mean_norm_matched",
            )
        ),
    )
    loss_scope = str(config.get("semantic_id_loss_scope", "full_vocabulary"))
    if loss_scope not in {"full_vocabulary", "taxonomy"}:
        raise ValueError(
            "semantic_id_loss_scope must be full_vocabulary or taxonomy"
        )
    indices = trainable_token_indices(
        model,
        spec,
        output_only=(loss_scope == "taxonomy"),
    )
    lora = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
        bias="none",
        trainable_token_indices=indices,
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.train()
    representatives = _representative_messages(
        train_file,
        {entry.interest_token for entry in spec.entries},
    )
    examples = [
        (
            token,
            tokenize_training_example(
                tokenizer,
                messages,
                max_seq_length=int(config.get("max_seq_length", 2048)),
                enable_thinking=bool(config.get("enable_thinking", False)),
            ),
        )
        for token, messages in representatives
    ]
    examples.sort(key=lambda item: len(item[1]["input_ids"]))
    collator = CompletionOnlyCollator(tokenizer)
    model.zero_grad(set_to_none=True)
    loss_values = []
    for start in range(0, len(examples), batch_size):
        features = [value for _, value in examples[start : start + batch_size]]
        batch = {
            key: value.to(model.device)
            for key, value in collator(features).items()
        }
        outputs = model(**batch, use_cache=False)
        if loss_scope == "taxonomy":
            loss = taxonomy_restricted_causal_lm_loss(
                outputs.logits,
                batch["labels"],
                spec.token_ids,
            )
        elif loss_scope == "full_vocabulary":
            loss = outputs.loss
        else:
            raise ValueError(
                "semantic_id_loss_scope must be full_vocabulary or taxonomy"
            )
        if not torch.isfinite(loss):
            raise ValueError("Gradient audit produced non-finite loss")
        loss.backward()
        loss_values.append(float(loss.detach().cpu()))

    mera_audit = None
    mera_losses = None
    if config.get("mera_relations_file") is not None:
        regularizer = build_mera_regularizer(config, spec)
        relation_losses = regularizer.compute(model, step=0)
        relation_loss = relation_losses["weighted_loss"]
        if not torch.isfinite(relation_loss):
            raise ValueError("MeRA gradient audit produced non-finite loss")
        relation_loss.backward()
        mera_audit = regularizer.audit()
        mera_losses = {
            key: float(value.detach().cpu())
            for key, value in relation_losses.items()
        }

    parameter_audits = []
    for name, parameter in model.named_parameters():
        if "trainable_tokens_delta" not in name or not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            raise ValueError(f"No gradient for {name}")
        gradient = gradient.detach().float().cpu()
        if gradient.ndim != 2 or gradient.shape[0] != len(spec.entries):
            raise ValueError(f"Unexpected trainable token gradient shape: {name}")
        row_norms = gradient.norm(p=2, dim=1)
        finite = torch.isfinite(row_norms)
        nonzero = row_norms > 0
        parameter_audits.append(
            {
                "parameter": name,
                "shape": list(gradient.shape),
                "finite_rows": int(finite.sum()),
                "nonzero_rows": int(nonzero.sum()),
                "min_row_gradient_norm": float(row_norms.min()),
                "max_row_gradient_norm": float(row_norms.max()),
            }
        )
    if len(parameter_audits) != len(indices):
        raise ValueError(
            "Expected one trainable-token parameter per targeted module: "
            f"expected {len(indices)}, got {len(parameter_audits)}"
        )
    failures = [
        row
        for row in parameter_audits
        if row["finite_rows"] != len(spec.entries)
        or row["nonzero_rows"] != len(spec.entries)
    ]
    if failures:
        raise ValueError(f"ID token gradient audit failed: {failures}")
    result = {
        "model_name": str(config["model_name"]),
        "model_revision": config.get("model_revision"),
        "train_file": str(train_file),
        "mapping_file": str(mapping_file),
        "representative_samples": len(examples),
        "interest_token_count": len(spec.entries),
        "batch_size": batch_size,
        "semantic_id_loss_scope": loss_scope,
        "batches": math.ceil(len(examples) / batch_size),
        "mean_loss": sum(loss_values) / len(loss_values),
        "min_loss": min(loss_values),
        "max_loss": max(loss_values),
        "semantic_initialization": initialization_audit,
        "trainable_token_indices": indices,
        "gradient_parameters": parameter_audits,
        "mera_regularization": mera_audit,
        "mera_gradient_losses": mera_losses,
        "all_token_rows_finite_and_nonzero": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit gradients for all Semantic ID Token rows."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    if args.train_file is not None:
        config["train_file"] = str(args.train_file)
    result = audit_semantic_id_token_gradients(
        config,
        args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
