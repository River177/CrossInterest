from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from new_interest.training.mera_regularization import build_mera_regularizer
from new_interest.training.semantic_id_tokens import (
    initialize_semantic_interest_token_rows,
    register_semantic_interest_tokens,
    trainable_token_indices,
    write_json,
)
from new_interest.training.train_sft import _load_config, _set_seed


def audit_mera_regularization_gradients(
    config: dict[str, Any],
    output_path: Path | str,
) -> dict[str, object]:
    """Audit MeRA loss and every trainable Semantic-ID row on one GPU."""

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.get("mera_relations_file") is None:
        raise ValueError("Config does not enable MeRA regularization")
    output_path = Path(output_path)
    mapping_file = Path(config["semantic_id_token_mapping_file"])
    seed = int(config.get("seed", 42))
    _set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        token=config.get("hf_token"),
        trust_remote_code=False,
    )
    spec = register_semantic_interest_tokens(
        tokenizer,
        mapping_file,
        expected_count=int(config.get("semantic_id_token_count", 108)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(config["model_name"]),
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
    indices = trainable_token_indices(model, spec)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=config.get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
            bias="none",
            trainable_token_indices=indices,
        ),
    )
    model.train()
    model.zero_grad(set_to_none=True)
    regularizer = build_mera_regularizer(config, spec)
    losses = regularizer.compute(model, step=0)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise ValueError("MeRA regularization produced a non-finite loss")
    losses["weighted_loss"].backward()

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
        raise ValueError(f"MeRA token-row gradient audit failed: {failures}")

    result = {
        "audit_version": "mera-regularization-gradient-v1",
        "model_name": str(config["model_name"]),
        "mapping_file": str(mapping_file),
        "relations_file": str(config["mera_relations_file"]),
        "interest_token_count": len(spec.entries),
        "semantic_initialization": initialization_audit,
        "regularizer": regularizer.audit(),
        "losses": {
            key: float(value.detach().cpu()) for key, value in losses.items()
        },
        "trainable_token_indices": indices,
        "gradient_parameters": parameter_audits,
        "all_token_rows_finite_and_nonzero": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit MeRA regularization gradients for Semantic-ID rows."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = audit_mera_regularization_gradients(
        _load_config(args.config),
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
