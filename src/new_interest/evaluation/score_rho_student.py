"""Score a live student's own training users without changing its training state.

These are in-sample student losses, not cross-fitted reference losses. The
reference scorer's independent-user guard remains unchanged. No model is
constructed, optimizer stepped, file written, or distributed collective used.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np

from new_interest.evaluation.evaluate_beam import canonical_sha256, sha256_file
from new_interest.evaluation.score_rho_reference import (
    LOSS_DEFINITION, RECORD_FIELDS, effective_output_weight, encode_prefix,
    kernel_selection_context, kernel_selection_identity, loss_rows, score_batch,
    source_identity, validate_rows,
)
from new_interest.training.semantic_id_tokens import (
    apply_torch_numeric_backend, capture_torch_numeric_backend,
)

VERSION = "rho-current-student-first-sid-full-effective-vocab-fp32-v1"
ROLE = "current_student_in_sample"


def _tensor_sha(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _adapter_state(model):
    """Fingerprint compact student parameters without cloning the frozen base."""
    selected = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or "lora_" in name or "trainable_tokens_delta" in name:
            selected[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                              "sha256": _tensor_sha(parameter)}
    if not selected:
        raise ValueError("Live student has no identifiable trainable/adapter parameters")
    return {"sha256": canonical_sha256(selected), "tensor_count": len(selected)}


def _gradient_state(model):
    return {name: None if p.grad is None else {"object_id": id(p.grad), "sha256": _tensor_sha(p.grad),
            "dtype": str(p.grad.dtype), "shape": list(p.grad.shape)} for name, p in model.named_parameters()}


@contextmanager
def preserve_student_runtime(model):
    """Restore per-module flags, buffers, numeric flags and all global RNGs."""
    import torch
    modules = list(model.modules())
    training = [(module, module.training) for module in modules]
    buffer_maps = [(module, dict(module._buffers)) for module in modules]
    buffers = [(module, name, value, value.detach().clone()) for module in modules
               for name, value in module._buffers.items() if value is not None]
    mutable = [(module, name, getattr(module, name)) for module in modules
               for name in ("max_seq_len_cached", "attention_scaling") if hasattr(module, name)]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    backend = capture_torch_numeric_backend(torch)
    device_type = next(model.parameters()).device.type
    try:
        model.eval()
        # Match the reference scorer's FP32 projection and BF16 forward policy.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        try:
            with torch.no_grad():
                for module, name, original, saved in buffers:
                    if not torch.equal(original, saved):
                        original.copy_(saved)
                    module._buffers[name] = original
                for module, original in buffer_maps:
                    module._buffers.clear()
                    module._buffers.update(original)
                for module, name, value in mutable:
                    setattr(module, name, value)
        finally:
            # Assign each flag directly: recursive train() would overwrite
            # intentionally mixed submodule train/eval modes present on entry.
            for module, flag in training:
                module.training = flag
            try:
                apply_torch_numeric_backend(torch, backend)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.random.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)


def score_current_student(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    spec,
    *,
    student_state: dict[str, Any],
    shard_index: int = 0,
    num_shards: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    enable_thinking: bool = False,
    kernel_tuners=None,
):
    """Return one modulo shard of a live student's first-SID losses.

    ``student_state`` binds the caller's run identity and optimizer/epoch cursor.
    The actual current adapter bytes are also hashed. ``records`` must be the
    entire ordered training cohort on every rank. The caller owns gathering and
    persistence. Pass an already unwrapped raw PEFT model, temporarily removing
    and later restoring any Accelerate forward wrapper outside this function.
    ``kernel_tuners`` is for CPU mechanism fixtures only; production discovers
    and guards the real FLA autotuners through ``stable_fla_v1``.
    """
    if not isinstance(student_state, dict) or not student_state:
        raise ValueError("Explicit live-student run/step identity is required")
    if min(num_shards, batch_size, max_seq_length) < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid student scoring batch/shard configuration")
    if enable_thinking:
        raise ValueError("Current-student RHO scoring requires direct non-thinking SID supervision")
    if hasattr(model, "_original_forward") or hasattr(model, "module"):
        raise ValueError("Student scorer requires a raw unwrapped model without an Accelerate forward wrapper")
    if not records or any(any(key not in row or row[key] is None for key in RECORD_FIELDS) for row in records):
        raise ValueError("Empty student cohort or missing source identity")
    for key in ("source_sample_id", "user_id"):
        if len({str(row[key]) for row in records}) != len(records):
            raise ValueError(f"Duplicate student {key}; expected one training example per user")
    indices = list(range(shard_index, len(records), num_shards))
    if not indices:
        raise ValueError("Empty current-student logical shard")
    if tokenizer.pad_token_id is None:
        raise ValueError("Student tokenizer must already have a padding token")
    record_hashes = [source_identity(row) for row in records]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    model_config = getattr(base, "config", None)
    config_dict = model_config.to_dict() if hasattr(model_config, "to_dict") else {}
    parameters = list(model.named_parameters())
    parameter_versions = {name: (id(p), p._version, p.requires_grad) for name, p in parameters}
    before_adapter, before_gradients = _adapter_state(model), _gradient_state(model)
    identity = {"identity_version": VERSION, "role": ROLE, "loss_definition": LOSS_DEFINITION,
        "student_state": copy.deepcopy(student_state), "adapter_state_sha256": before_adapter["sha256"],
        "model_config_sha256": canonical_sha256(config_dict),
        "ordered_source_sha256": canonical_sha256(record_hashes), "full_record_count": len(records),
        "mapping_sha256": spec.mapping_sha256, "labels": [e.target_subcategory for e in spec.entries],
        "tokens": [e.interest_token for e in spec.entries], "token_ids": spec.token_ids,
        "num_shards": num_shards, "batch_size": batch_size, "max_seq_length": max_seq_length,
        "enable_thinking": False, "padding_side": "right", "projection_dtype": "float32",
        "tf32": False, "allow_bf16_reduced_precision_reduction": True,
        "attn_implementation": getattr(model_config, "_attn_implementation", None),
        "scorer_sha256": sha256_file(Path(__file__)), **kernel_selection_identity("stable_fla_v1")}
    identity["evaluation_identity"] = canonical_sha256(identity)
    rows = []
    with preserve_student_runtime(model):
        with kernel_selection_context("stable_fla_v1", tuners=kernel_tuners) as kernels:
            # One effective full-vocabulary copy per scoring call, never per batch.
            weight = effective_output_weight(model).detach().float()
            vocabulary_size = int(weight.shape[0])
            if vocabulary_size != len(tokenizer) or max(spec.token_ids) >= vocabulary_size:
                raise ValueError("Student tokenizer and effective output vocabulary differ")
            for start in range(0, len(indices), batch_size):
                local_indices = indices[start:start+batch_size]
                local = [records[index] for index in local_indices]
                prefixes = [encode_prefix(tokenizer, record, spec, max_seq_length=max_seq_length,
                                          enable_thinking=False) for record in local]
                logits = score_batch(model, prefixes, tokenizer.pad_token_id, weight)
                part = loss_rows(logits, [p["target_interest_index"] for p in prefixes], spec.token_ids)
                for row, record, prefix, index in zip(part, local, prefixes, local_indices):
                    row.update(record_index=index, source_sample_id=str(record["source_sample_id"]),
                        user_id=str(record["user_id"]), target_subcategory=str(record["target_subcategory"]),
                        input_sha256=record_hashes[index], evaluation_identity=identity["evaluation_identity"],
                        score_role=ROLE, target_interest_index=prefix["target_interest_index"],
                        target_token_id=prefix["target_token_id"], prefix_tokens=len(prefix["input_ids"]),
                        full_vocabulary_size=vocabulary_size)
                validate_rows(part, local, local_indices, identity)
                kernels.verify_coverage()
                rows.extend(part)
            # Cache entries already accumulate all selected keys. One final
            # snapshot avoids quadratic receipt growth over thousands of batches.
            kernels.capture("student_shard_complete")
            receipt = kernels.receipt(identity["evaluation_identity"])
    if {name: (id(p), p._version, p.requires_grad) for name, p in model.named_parameters()} != parameter_versions:
        raise ValueError("Student scoring changed a parameter object/version/gradient flag")
    after_adapter = _adapter_state(model)
    if before_adapter != after_adapter or _gradient_state(model) != before_gradients:
        raise ValueError("Student scoring changed adapter parameters or existing gradients")
    if [source_identity(row) for row in records] != record_hashes:
        raise ValueError("Student source records changed during scoring")
    if [row["record_index"] for row in rows] != indices or len({r["source_sample_id"] for r in rows}) != len(indices):
        raise ValueError("Current-student modulo shard coverage differs")
    return {"identity": identity, "rows": rows, "kernel_receipt": receipt,
        "coverage": {"shard_index": shard_index, "num_shards": num_shards, "scored_rows": len(rows),
                     "full_record_count": len(records), "record_indices_sha256": canonical_sha256(indices),
                     "exact_modulo_coverage": True},
        "state_audit": {"adapter_state_before": before_adapter, "adapter_state_after": after_adapter,
                        "parameters_unchanged": True, "gradients_unchanged": True,
                        "training_flags_restored": True, "rng_restored": True, "model_updates": 0}}
