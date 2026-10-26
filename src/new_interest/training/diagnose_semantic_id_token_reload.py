from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from new_interest.evaluation.evaluate_beam import canonical_sha256
from new_interest.training.semantic_id_tokens import (
    RELOAD_REFERENCE_FORWARD_PATH,
    apply_torch_numeric_backend,
    deterministic_reload_numeric_backend,
    deterministic_sdpa_math_context,
    prepare_semantic_adapter_base,
    validate_saved_interest_tokens,
    write_json,
)
from new_interest.training.sft_data import apply_chat_template
from new_interest.training.verify_semantic_id_token_adapter import (
    compare_adapter_state_dicts,
)


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolve_dtype(torch_module: Any, name: str) -> Any:
    values = {
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"Unsupported diagnostic dtype: {name}") from error


def capture_reloaded_adapter_logits(
    model_name: str,
    adapter_path: Path | str,
    mapping_file: Path | str,
    prompt_reference_file: Path | str,
    output_path: Path | str,
    *,
    inference_dtype: str,
    enable_thinking: bool = False,
    hf_token: str | None = None,
    max_samples: int | None = None,
) -> dict[str, object]:
    """Capture one independent reload forward for differential diagnosis."""

    import torch
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    mapping_file = Path(mapping_file)
    prompt_reference_file = Path(prompt_reference_file)
    output_path = Path(output_path)
    prompt_reference = json.loads(
        prompt_reference_file.read_text(encoding="utf-8")
    )
    reference_samples = list(prompt_reference["samples"])
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        reference_samples = reference_samples[:max_samples]
    backend = deterministic_reload_numeric_backend()
    apply_torch_numeric_backend(torch, backend)
    dtype = _resolve_dtype(torch, inference_dtype)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    spec = validate_saved_interest_tokens(
        tokenizer,
        mapping_file,
        expected_count=len(prompt_reference["token_ids"]),
    )
    if spec.mapping_sha256 != str(prompt_reference["mapping_sha256"]):
        raise ValueError("Prompt reference mapping SHA-256 mismatch")
    if spec.token_ids != [int(value) for value in prompt_reference["token_ids"]]:
        raise ValueError("Prompt reference token IDs mismatch")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    semantic_audit = prepare_semantic_adapter_base(model, spec)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(torch.device("cuda", 0))
    model.eval()

    saved_state = load_file(
        adapter_path / "adapter_model.safetensors",
        device="cpu",
    )
    loaded_state = get_peft_model_state_dict(
        model,
        adapter_name="default",
        save_embedding_layers=False,
    )
    adapter_state = compare_adapter_state_dicts(saved_state, loaded_state)
    if not adapter_state["all_tensors_exact"]:
        raise ValueError(f"Adapter state is not bit-exact: {adapter_state}")

    token_index = torch.tensor(
        spec.token_ids,
        dtype=torch.long,
        device=model.device,
    )
    samples = []
    with torch.inference_mode(), deterministic_sdpa_math_context(torch):
        for sample in reference_samples:
            prompt = str(sample["prompt"])
            if canonical_sha256(prompt) != str(sample["prompt_sha256"]):
                raise ValueError("Prompt SHA-256 mismatch")
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
            input_ids = encoded["input_ids"]
            encoded = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            logits = model(**encoded, use_cache=False).logits[0, -1]
            restricted = logits.index_select(0, token_index).float().cpu()
            samples.append(
                {
                    "source_sample_id": str(sample["source_sample_id"]),
                    "prompt": prompt,
                    "prompt_sha256": canonical_sha256(prompt),
                    "input_ids_sha256": _tensor_sha256(input_ids),
                    "restricted_logits": restricted.tolist(),
                }
            )

    result = {
        "model_name": model_name,
        "inference_dtype": inference_dtype,
        "mapping_sha256": spec.mapping_sha256,
        "token_ids": spec.token_ids,
        "torch_numeric_backend": backend,
        "sdpa_backend": "math",
        "forward_path": RELOAD_REFERENCE_FORWARD_PATH,
        "model_load_path": "cpu-construct-adapter-then-cuda",
        "adapter_state_integrity": adapter_state,
        "semantic_base_initialization": {
            "initialization_version": semantic_audit["initialization_version"],
            "reduction_device": semantic_audit["reduction_device"],
            "reduction_dtype": semantic_audit["reduction_dtype"],
            "mapping_sha256": semantic_audit["mapping_sha256"],
            "tie_word_embeddings": semantic_audit["tie_word_embeddings"],
            "input_initial_sha256": [
                row["input_initial_sha256"] for row in semantic_audit["tokens"]
            ],
            "output_initial_sha256": [
                row["output_initial_sha256"] for row in semantic_audit["tokens"]
            ],
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def compare_capture_to_training_reference(
    reference: dict[str, object],
    capture: dict[str, object],
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict[str, object]:
    """Compare a fresh reload directly with the in-memory training reference."""

    import torch

    identity_mismatches = []
    for field in (
        "mapping_sha256",
        "token_ids",
        "torch_numeric_backend",
        "sdpa_backend",
        "forward_path",
    ):
        if reference[field] != capture[field]:
            identity_mismatches.append(field)

    semantic = capture["semantic_base_initialization"]
    semantic_identity = {
        "semantic_initialization_version": semantic["initialization_version"],
        "semantic_initialization_reduction_device": semantic["reduction_device"],
        "semantic_initialization_reduction_dtype": semantic["reduction_dtype"],
    }
    for field, actual in semantic_identity.items():
        if reference[field] != actual:
            identity_mismatches.append(field)

    reference_by_id = {
        str(row["source_sample_id"]): row for row in reference["samples"]
    }
    rows = []
    for actual_row in capture["samples"]:
        sample_id = str(actual_row["source_sample_id"])
        expected_row = reference_by_id.get(sample_id)
        if expected_row is None:
            identity_mismatches.append(f"missing_reference_sample:{sample_id}")
            continue
        if actual_row["prompt_sha256"] != expected_row["prompt_sha256"]:
            identity_mismatches.append(f"prompt_sha256:{sample_id}")
        expected = torch.tensor(
            expected_row["restricted_logits"], dtype=torch.float32
        )
        actual = torch.tensor(
            actual_row["restricted_logits"], dtype=torch.float32
        )
        if expected.shape != actual.shape:
            identity_mismatches.append(f"restricted_logits_shape:{sample_id}")
            continue
        signed_difference = actual - expected
        absolute_difference = signed_difference.abs()
        max_index = int(absolute_difference.argmax().item())
        close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
        expected_order = torch.argsort(expected, descending=True)
        actual_order = torch.argsort(actual, descending=True)
        expected_top20 = expected_order[:20].tolist()
        actual_top20 = actual_order[:20].tolist()
        rows.append(
            {
                "source_sample_id": sample_id,
                "prompt_sha256": str(actual_row["prompt_sha256"]),
                "input_ids_sha256": str(actual_row["input_ids_sha256"]),
                "all_close": bool(close.all()),
                "max_absolute_difference": float(absolute_difference.max()),
                "mean_absolute_difference": float(absolute_difference.mean()),
                "mean_signed_difference": float(signed_difference.mean()),
                "max_difference_index": max_index,
                "expected_at_max_difference": float(expected[max_index]),
                "actual_at_max_difference": float(actual[max_index]),
                "top1_match": bool(
                    expected_order[0].item() == actual_order[0].item()
                ),
                "top20_exact_match": expected_top20 == actual_top20,
                "top20_set_overlap": len(
                    set(expected_top20).intersection(actual_top20)
                ),
            }
        )

    all_close = not identity_mismatches and bool(rows) and all(
        row["all_close"] for row in rows
    )
    return {
        "inference_dtype": capture["inference_dtype"],
        "atol": atol,
        "rtol": rtol,
        "identity_mismatches": sorted(set(identity_mismatches)),
        "all_close": all_close,
        "samples": rows,
    }


def compare_reload_captures(
    left: dict[str, object],
    right: dict[str, object],
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict[str, object]:
    import torch

    identity_fields = (
        "model_name",
        "inference_dtype",
        "mapping_sha256",
        "token_ids",
        "torch_numeric_backend",
        "sdpa_backend",
        "forward_path",
        "model_load_path",
        "semantic_base_initialization",
    )
    identity_mismatches = [
        field for field in identity_fields if left[field] != right[field]
    ]
    left_samples = left["samples"]
    right_samples = right["samples"]
    if len(left_samples) != len(right_samples):
        identity_mismatches.append("sample_count")

    rows = []
    for left_row, right_row in zip(left_samples, right_samples):
        identity = {
            "source_sample_id": left_row["source_sample_id"],
            "prompt_sha256": left_row["prompt_sha256"],
            "input_ids_sha256": left_row["input_ids_sha256"],
        }
        if any(left_row[key] != right_row[key] for key in identity):
            identity_mismatches.append(
                f"sample:{left_row['source_sample_id']}"
            )
        expected = torch.tensor(
            left_row["restricted_logits"], dtype=torch.float32
        )
        actual = torch.tensor(
            right_row["restricted_logits"], dtype=torch.float32
        )
        difference = (expected - actual).abs()
        close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
        rows.append(
            {
                **identity,
                "all_close": bool(close.all()),
                "max_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
            }
        )
    all_close = not identity_mismatches and all(
        row["all_close"] for row in rows
    )
    return {
        "inference_dtype": left["inference_dtype"],
        "atol": atol,
        "rtol": rtol,
        "identity_mismatches": sorted(set(identity_mismatches)),
        "all_close": all_close,
        "samples": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose cross-process Semantic ID reload logits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--model-name", required=True)
    capture.add_argument("--adapter-path", type=Path, required=True)
    capture.add_argument("--mapping-file", type=Path, required=True)
    capture.add_argument("--prompt-reference-file", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--inference-dtype",
        choices=("bfloat16", "float32"),
        required=True,
    )
    capture.add_argument("--enable-thinking", action="store_true")
    capture.add_argument("--max-samples", type=int)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=5e-3)
    compare.add_argument("--rtol", type=float, default=5e-3)

    compare_reference = subparsers.add_parser("compare-reference")
    compare_reference.add_argument("--reference", type=Path, required=True)
    compare_reference.add_argument("--capture", type=Path, required=True)
    compare_reference.add_argument("--output", type=Path, required=True)
    compare_reference.add_argument("--atol", type=float, default=5e-3)
    compare_reference.add_argument("--rtol", type=float, default=5e-3)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "capture":
        result = capture_reloaded_adapter_logits(
            args.model_name,
            args.adapter_path,
            args.mapping_file,
            args.prompt_reference_file,
            args.output,
            inference_dtype=args.inference_dtype,
            enable_thinking=args.enable_thinking,
            max_samples=args.max_samples,
        )
    elif args.command == "compare":
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
        result = compare_reload_captures(
            left,
            right,
            atol=args.atol,
            rtol=args.rtol,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        result = compare_capture_to_training_reference(
            reference,
            capture,
            atol=args.atol,
            rtol=args.rtol,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
