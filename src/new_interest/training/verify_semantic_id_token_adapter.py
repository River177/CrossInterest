from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from new_interest.evaluation.evaluate_beam import canonical_sha256
from new_interest.training.semantic_id_tokens import (
    MERA_MEAN_INITIALIZATION_VERSION,
    RELOAD_REFERENCE_FORWARD_PATH,
    SEMANTIC_INITIALIZATION_REDUCTION_DEVICE,
    SEMANTIC_INITIALIZATION_REDUCTION_DTYPE,
    SEMANTIC_INITIALIZATION_VERSION,
    SEMANTIC_INITIALIZATION_MODE_MEAN,
    SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED,
    apply_torch_numeric_backend,
    deterministic_sdpa_math_context,
    prepare_semantic_adapter_base,
    validate_saved_interest_tokens,
    write_json,
)


def compare_adapter_state_dicts(
    saved_state: dict[str, object],
    loaded_state: dict[str, object],
) -> dict[str, object]:
    """Verify that PEFT reloaded every serialized adapter tensor bit-exactly."""

    import torch

    saved_keys = set(saved_state)
    loaded_keys = set(loaded_state)
    missing = sorted(saved_keys.difference(loaded_keys))
    unexpected = sorted(loaded_keys.difference(saved_keys))
    mismatched = []
    saved_digest = hashlib.sha256()
    loaded_digest = hashlib.sha256()
    total_numel = 0

    def update_digest(digest, key: str, tensor) -> None:
        value = tensor.detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())

    for key in sorted(saved_keys.intersection(loaded_keys)):
        saved = saved_state[key]
        loaded = loaded_state[key]
        total_numel += int(saved.numel())
        update_digest(saved_digest, key, saved)
        update_digest(loaded_digest, key, loaded)
        if (
            saved.shape != loaded.shape
            or saved.dtype != loaded.dtype
            or not torch.equal(saved.detach().cpu(), loaded.detach().cpu())
        ):
            mismatched.append(key)

    all_tensors_exact = not missing and not unexpected and not mismatched
    return {
        "tensor_count": len(saved_keys),
        "total_numel": total_numel,
        "all_tensors_exact": all_tensors_exact,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "saved_state_sha256": saved_digest.hexdigest(),
        "loaded_state_sha256": loaded_digest.hexdigest(),
    }
from new_interest.training.sft_data import apply_chat_template


def verify_semantic_id_token_adapter(
    model_name: str,
    adapter_path: Path | str,
    mapping_file: Path | str,
    reference_file: Path | str,
    output_path: Path | str,
    *,
    enable_thinking: bool = False,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    hf_token: str | None = None,
    model_revision: str | None = None,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    mapping_file = Path(mapping_file)
    reference_file = Path(reference_file)
    output_path = Path(output_path)
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    numeric_backend = reference.get("torch_numeric_backend")
    if not isinstance(numeric_backend, dict):
        raise ValueError(
            "Reload reference does not record the torch numeric backend"
        )
    apply_torch_numeric_backend(torch, numeric_backend)
    if reference.get("sdpa_backend") != "math":
        raise ValueError("Reload reference does not require math SDPA")
    if reference.get("forward_path") != RELOAD_REFERENCE_FORWARD_PATH:
        raise ValueError("Reload reference does not use the raw forward path")
    initialization_version = reference.get(
        "semantic_initialization_version"
    )
    initialization_mode = reference.get("semantic_initialization_mode")
    if initialization_mode is None:
        initialization_mode = SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED
    expected_versions = {
        SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED: (
            SEMANTIC_INITIALIZATION_VERSION
        ),
        SEMANTIC_INITIALIZATION_MODE_MEAN: MERA_MEAN_INITIALIZATION_VERSION,
    }
    if expected_versions.get(str(initialization_mode)) != initialization_version:
        raise ValueError("Reload reference semantic initialization mismatch")
    if (
        reference.get("semantic_initialization_reduction_device")
        != SEMANTIC_INITIALIZATION_REDUCTION_DEVICE
        or reference.get("semantic_initialization_reduction_dtype")
        != SEMANTIC_INITIALIZATION_REDUCTION_DTYPE
    ):
        raise ValueError("Reload reference semantic reduction mismatch")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    spec = validate_saved_interest_tokens(
        tokenizer,
        mapping_file,
        expected_count=len(reference["token_ids"]),
    )
    if spec.mapping_sha256 != str(reference["mapping_sha256"]):
        raise ValueError("Reference mapping SHA-256 mismatch")
    if spec.token_ids != [int(value) for value in reference["token_ids"]]:
        raise ValueError("Reference token IDs mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    semantic_base_audit = prepare_semantic_adapter_base(
        model,
        spec,
        initialization_mode=str(initialization_mode),
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    # Match both training and the production ranking evaluator: construct the
    # base model and PEFT wrappers on CPU, then move the complete model to the
    # target GPU.  Loading the base directly onto CUDA follows a different
    # Transformers initialization path for non-persistent buffers and does not
    # reproduce the model whose reference logits were captured after Trainer
    # moved it from CPU to CUDA.
    model.to(torch.device("cuda", 0))
    model.eval()
    saved_adapter_state = load_file(
        adapter_path / "adapter_model.safetensors",
        device="cpu",
    )
    loaded_adapter_state = get_peft_model_state_dict(
        model,
        adapter_name="default",
        save_embedding_layers=False,
    )
    adapter_state_integrity = compare_adapter_state_dicts(
        saved_adapter_state,
        loaded_adapter_state,
    )
    if not adapter_state_integrity["all_tensors_exact"]:
        raise ValueError(
            "Reloaded adapter state is not bit-exact: "
            f"{adapter_state_integrity}"
        )
    token_index = torch.tensor(
        spec.token_ids,
        dtype=torch.long,
        device=model.device,
    )
    rows = []
    with torch.inference_mode(), deterministic_sdpa_math_context(torch):
        for sample in reference["samples"]:
            prompt = str(sample["prompt"])
            if canonical_sha256(prompt) != str(sample["prompt_sha256"]):
                raise ValueError("Reference prompt hash mismatch")
            text = apply_chat_template(
                tokenizer,
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            logits = model(**encoded, use_cache=False).logits[0, -1]
            actual = logits.index_select(0, token_index).float().cpu()
            expected = torch.tensor(
                sample["restricted_logits"], dtype=torch.float32
            )
            difference = (actual - expected).abs()
            close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
            expected_order = torch.argsort(expected, descending=True)
            actual_order = torch.argsort(actual, descending=True)
            expected_top20 = expected_order[:20].tolist()
            actual_top20 = actual_order[:20].tolist()
            rows.append(
                {
                    "source_sample_id": str(sample["source_sample_id"]),
                    "all_close": bool(close.all()),
                    "max_absolute_difference": float(difference.max()),
                    "mean_absolute_difference": float(difference.mean()),
                    "top1_match": bool(
                        expected_order[0].item() == actual_order[0].item()
                    ),
                    "top20_exact_match": expected_top20 == actual_top20,
                    "top20_set_overlap": len(
                        set(expected_top20).intersection(actual_top20)
                    ),
                }
            )
    all_close = all(row["all_close"] for row in rows)
    result = {
        "model_name": model_name,
        "model_revision": model_revision,
        "adapter_path": str(adapter_path),
        "mapping_file": str(mapping_file),
        "reference_file": str(reference_file),
        "sample_count": len(rows),
        "interest_token_count": len(spec.entries),
        "atol": atol,
        "rtol": rtol,
        "torch_numeric_backend": numeric_backend,
        "sdpa_backend": "math",
        "forward_path": RELOAD_REFERENCE_FORWARD_PATH,
        "model_load_path": "cpu-construct-adapter-then-cuda",
        "semantic_initialization_version": initialization_version,
        "semantic_initialization_mode": initialization_mode,
        "semantic_initialization_reduction_device": (
            SEMANTIC_INITIALIZATION_REDUCTION_DEVICE
        ),
        "semantic_initialization_reduction_dtype": (
            SEMANTIC_INITIALIZATION_REDUCTION_DTYPE
        ),
        "adapter_state_integrity": adapter_state_integrity,
        "semantic_base_initialization": {
            "initialization_version": semantic_base_audit[
                "initialization_version"
            ],
            "mapping_sha256": semantic_base_audit["mapping_sha256"],
            "interest_token_count": semantic_base_audit[
                "interest_token_count"
            ],
            "tie_word_embeddings": semantic_base_audit[
                "tie_word_embeddings"
            ],
        },
        "all_close": all_close,
        "samples": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    if not all_close:
        raise ValueError(f"Adapter reload logits mismatch: {rows}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Semantic ID Token adapter save/reload logits."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = verify_semantic_id_token_adapter(
        args.model_name,
        args.adapter_path,
        args.mapping_file,
        args.reference_file,
        args.output,
        enable_thinking=args.enable_thinking,
        atol=args.atol,
        rtol=args.rtol,
        model_revision=args.model_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
