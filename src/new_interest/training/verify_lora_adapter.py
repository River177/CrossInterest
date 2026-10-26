from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_interest.training.semantic_id_tokens import write_json
from new_interest.training.verify_semantic_id_token_adapter import (
    compare_adapter_state_dicts,
)


def verify_lora_adapter(
    model_name: str,
    adapter_path: Path | str,
    output_path: Path | str,
    *,
    model_revision: str | None = None,
    expected_tensors: int | None = None,
    hf_token: str | None = None,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    saved_adapter_state = load_file(
        adapter_path / "adapter_model.safetensors",
        device="cpu",
    )
    loaded_adapter_state = get_peft_model_state_dict(
        model,
        adapter_name="default",
        save_embedding_layers=False,
    )
    integrity = compare_adapter_state_dicts(
        saved_adapter_state,
        loaded_adapter_state,
    )
    if not integrity["all_tensors_exact"]:
        raise ValueError(f"Reloaded adapter state is not bit-exact: {integrity}")
    if expected_tensors is not None and integrity["tensor_count"] != int(
        expected_tensors
    ):
        raise ValueError(
            "Unexpected adapter tensor count: "
            f"expected {expected_tensors}, got {integrity['tensor_count']}"
        )
    result = {
        "model_name": model_name,
        "model_revision": model_revision,
        "adapter_path": str(adapter_path),
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        "expected_tensors": expected_tensors,
        "adapter_state_integrity": integrity,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a standard PEFT LoRA adapter reload bit-exactly."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tensors", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = verify_lora_adapter(
        args.model_name,
        args.adapter_path,
        args.output,
        model_revision=args.model_revision,
        expected_tensors=args.expected_tensors,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
