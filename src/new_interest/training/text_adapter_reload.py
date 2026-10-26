from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.training.sft_data import apply_chat_template
from new_interest.training.semantic_id_tokens import (
    apply_torch_numeric_backend,
    capture_torch_numeric_backend,
    deterministic_reload_numeric_backend,
    deterministic_sdpa_math_context,
)


REFERENCE_VERSION = "text-adapter-reload-reference-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_texts(
    tokenizer: Any,
    train_file: Path,
    *,
    enable_thinking: bool,
    sample_count: int,
) -> tuple[list[str], list[str]]:
    rows = pq.read_table(train_file, columns=["messages"]).slice(
        0, sample_count
    ).to_pylist()
    if len(rows) != sample_count:
        raise ValueError("Text reload reference has too few training samples")
    texts: list[str] = []
    hashes: list[str] = []
    for row in rows:
        messages = row["messages"]
        if len(messages) != 2 or messages[0]["role"] != "user":
            raise ValueError("Invalid SFT messages for text reload reference")
        text = apply_chat_template(
            tokenizer,
            [{"role": "user", "content": str(messages[0]["content"])}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        texts.append(text)
        hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return texts, hashes


def _capture_logits(model: Any, tokenizer: Any, texts: list[str]) -> Any:
    import torch

    model.eval()
    logits = []
    with torch.inference_mode(), deterministic_sdpa_math_context(torch):
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            output = model(**encoded, use_cache=False).logits[0, -1]
            logits.append(output.float().cpu())
    return torch.stack(logits)


def capture_text_reload_reference(
    model: Any,
    tokenizer: Any,
    train_file: Path | str,
    output_dir: Path | str,
    *,
    model_name: str,
    model_revision: str | None,
    enable_thinking: bool,
    sample_count: int = 4,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    train_file = Path(train_file)
    output_dir = Path(output_dir)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    texts, prompt_hashes = _prompt_texts(
        tokenizer,
        train_file,
        enable_thinking=enable_thinking,
        sample_count=sample_count,
    )
    import torch

    original_backend = capture_torch_numeric_backend(torch)
    reload_backend = deterministic_reload_numeric_backend()
    apply_torch_numeric_backend(torch, reload_backend)
    try:
        logits = _capture_logits(model, tokenizer, texts)
    finally:
        apply_torch_numeric_backend(torch, original_backend)
    tensor_path = output_dir / "text_reload_reference.safetensors"
    save_file({"next_token_logits": logits.contiguous()}, tensor_path)
    payload = {
        "reference_version": REFERENCE_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "train_file_sha256": _sha256_file(train_file),
        "enable_thinking": enable_thinking,
        "sample_count": sample_count,
        "prompt_sha256": prompt_hashes,
        "tensor_shape": list(logits.shape),
        "tensor_dtype": str(logits.dtype),
        "tensor_file": tensor_path.name,
        "tensor_sha256": _sha256_file(tensor_path),
        "forward_path": "batch-one-raw-unwrapped-no-accelerate-autocast-v1",
        "torch_numeric_backend": reload_backend,
        "sdpa_backend": "math",
    }
    (output_dir / "text_reload_reference.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_text_adapter_reload(
    *,
    model_name: str,
    model_revision: str | None,
    adapter_path: Path | str,
    train_file: Path | str,
    reference_json: Path | str,
    output_path: Path | str,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    train_file = Path(train_file)
    reference_json = Path(reference_json)
    output_path = Path(output_path)
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    if reference["reference_version"] != REFERENCE_VERSION:
        raise ValueError("Unsupported text reload reference version")
    if reference["model_name"] != model_name:
        raise ValueError("Text reload reference model mismatch")
    if reference.get("model_revision") != model_revision:
        raise ValueError("Text reload reference revision mismatch")
    if reference["train_file_sha256"] != _sha256_file(train_file):
        raise ValueError("Text reload reference Train hash mismatch")
    if reference.get("sdpa_backend") != "math":
        raise ValueError("Text reload reference does not require math SDPA")
    apply_torch_numeric_backend(torch, reference["torch_numeric_backend"])
    tensor_path = reference_json.parent / reference["tensor_file"]
    if reference["tensor_sha256"] != _sha256_file(tensor_path):
        raise ValueError("Text reload tensor hash mismatch")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts, prompt_hashes = _prompt_texts(
        tokenizer,
        train_file,
        enable_thinking=bool(reference["enable_thinking"]),
        sample_count=int(reference["sample_count"]),
    )
    if prompt_hashes != reference["prompt_sha256"]:
        raise ValueError("Text reload prompt hash mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(torch.device("cuda", 0))
    observed = _capture_logits(model, tokenizer, texts)
    expected = load_file(tensor_path)["next_token_logits"].float()
    if observed.shape != expected.shape:
        raise ValueError("Text reload logit shape mismatch")
    max_absolute_error = float((observed - expected).abs().max())
    passed = max_absolute_error <= tolerance
    payload = {
        "verification_version": "text-adapter-reload-verification-v1",
        "model_name": model_name,
        "model_revision": model_revision,
        "adapter_path": str(adapter_path.resolve()),
        "train_file_sha256": _sha256_file(train_file),
        "sample_count": len(texts),
        "tensor_shape": list(observed.shape),
        "max_absolute_error": max_absolute_error,
        "tolerance": tolerance,
        "passed": passed,
        "forward_path": reference["forward_path"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(
            f"Text adapter reload differs by {max_absolute_error}, tolerance={tolerance}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify text LoRA adapter reload.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    result = verify_text_adapter_reload(
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=args.adapter_path,
        train_file=args.train_file,
        reference_json=args.reference_json,
        output_path=args.output,
        tolerance=args.tolerance,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
