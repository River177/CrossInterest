"""Real-model operator bindings and long-sample memory probes, without updates."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
import time


def _callable_name(value):
    if value is None:
        return None
    return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__name__', type(value).__name__)}"


def attention_kernel_audit(model) -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    names = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update", "FusedRMSNormGated", "torch_chunk_gated_delta_rule")
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "triton", "flash-linear-attention", "fla-core", "causal-conv1d", "einops"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    modules = []
    for name, layer in model.named_modules():
        if layer.__class__.__name__ == "Qwen3_5GatedDeltaNet":
            modules.append({"name": name, "class": _callable_name(type(layer)),
                            "instance_functions": {field: _callable_name(getattr(layer, field, None)) for field in names},
                            "normalization_class": _callable_name(type(layer.norm))})
    if not modules:
        raise ValueError("No actual Qwen3.5 gated-delta layers found")
    import torch
    return {"packages": versions, "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "is_fast_path_available": bool(getattr(implementation, "is_fast_path_available", False)),
            "module_bindings": {name: _callable_name(getattr(implementation, name, None)) for name in names},
            "linear_attention_modules": modules}


def run_model_numerical_probe(model, optimizer, tokenizer, args, source_file: str, mapping_file: str, output_dir: Path) -> None:
    """Fixed real inputs at the source checkpoint; no parameter updates."""
    import torch
    import pyarrow.parquet as pq
    from safetensors.torch import save_file
    from new_interest.training.sft_data import CompletionOnlyCollator, ParquetSFTDataset
    dataset = ParquetSFTDataset(source_file, tokenizer, max_seq_length=2048, enable_thinking=False)
    index = int(args.process_index)
    feature = dataset[index]
    batch = {key: value.to(args.device) for key, value in CompletionOnlyCollator(tokenizer)([feature]).items()}
    token_ids = [tokenizer.convert_tokens_to_ids(row["interest_token"]) for row in pq.read_table(mapping_file).to_pylist()]
    prompt_tokens = next(i for i, label in enumerate(feature["labels"]) if label != -100)
    was_training = model.training
    model.train()
    dropouts = [(module, module.p) for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    for module, _ in dropouts:
        module.p = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            scores = outputs.logits[0, prompt_tokens - 1, token_ids].detach().float().cpu()
            loss = outputs.loss
        loss.backward()
        gradients = {name: parameter.grad.detach().float().cpu().contiguous()
                     for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None}
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients.values()):
            raise ValueError("Numerical probe has missing or nonfinite gradients")
        result = {"rank": index, "input_row": index, "input_tokens": len(feature["input_ids"]),
                  "logits": scores.tolist(), "loss": float(loss.detach().float().cpu()),
                  "gradient_tensor_count": len(gradients), "dropout_disabled_for_diagnostic": True,
                  "optimizer_updates": 0, "source_file": Path(source_file).name}
        (output_dir / f"numerical_probe_rank_{index}.json").write_text(json.dumps(result, indent=2) + "\n")
        if index == 0:
            save_file(gradients, str(output_dir / "numerical_probe_gradients.safetensors"))
        del outputs, loss, gradients
    finally:
        for module, probability in dropouts:
            module.p = probability
        model.train(was_training)
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        del batch
        torch.cuda.empty_cache()


def run_long_sample_stress(model, optimizer, tokenizer, args, files: list[str], output_dir: Path) -> dict:
    import torch
    import torch.distributed as dist
    from new_interest.training.execution_migration import file_sha256
    from new_interest.training.sft_data import CompletionOnlyCollator, ParquetSFTDataset
    records = []
    collator = CompletionOnlyCollator(tokenizer)
    microbatch = int(args.per_device_train_batch_size)
    rank = int(args.process_index)
    was_training = model.training
    model.train()
    for filename in files:
        path = Path(filename)
        dataset = ParquetSFTDataset(path, tokenizer, max_seq_length=2048, enable_thinking=False)
        if len(dataset) < microbatch * int(args.world_size):
            raise ValueError("Long-sample probe does not cover every rank")
        start = rank * microbatch
        features = [dataset[index] for index in range(start, start + microbatch)]
        batch = {key: value.to(args.device) for key, value in collator(features).items()}
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        outcome = {"rank": rank, "source_file": path.name, "source_sha256": file_sha256(path),
                   "row_indices": list(range(start, start + microbatch)),
                   "batch_size": microbatch, "input_shape": list(batch["input_ids"].shape),
                   "real_token_lengths": [len(row["input_ids"]) for row in features]}
        try:
            # This is before the first DDP training forward. The unwrapped model
            # has no active DDP reduction; no optimizer or scheduler is stepped.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.cuda.synchronize(args.device)
            outcome.update({"status": "passed", "loss": float(loss.detach().float().cpu()),
                            "finite_loss": bool(torch.isfinite(loss).all())})
            if not outcome["finite_loss"]:
                outcome["status"] = "nonfinite"
            del loss
        except torch.OutOfMemoryError:
            outcome.update({"status": "oom", "finite_loss": None})
        finally:
            outcome.update({"seconds": time.perf_counter() - started,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                            "device_total_bytes": torch.cuda.get_device_properties(args.device).total_memory,
                            "optimizer_updates": 0})
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            del batch
            torch.cuda.empty_cache()
        gathered = [None] * int(args.world_size)
        dist.all_gather_object(gathered, outcome)
        records.append({"source_file": path.name, "ranks": gathered})
        if any(row["status"] != "passed" for row in gathered):
            break
    report = {"status": "passed" if all(row["status"] == "passed" for record in records for row in record["ranks"]) else "failed",
              "records": records, "optimizer_updates": 0,
              "scope": "Real longest samples, forward/backward only; checkpoint RNG is restored before training."}
    model.train(was_training)
    if rank == 0:
        (output_dir / "long_sample_stress.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise RuntimeError("EXECUTION_PROBE_CAPACITY_FAILED")
    return report
