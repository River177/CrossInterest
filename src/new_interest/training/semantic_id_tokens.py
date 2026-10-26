from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SEMANTIC_INITIALIZATION_VERSION = (
    "label-token-mean-norm-matched-cpu-fp32-v2"
)
MERA_MEAN_INITIALIZATION_VERSION = "label-token-mean-cpu-fp32-v1"
SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED = "mean_norm_matched"
SEMANTIC_INITIALIZATION_MODE_MEAN = "mean"
SEMANTIC_INITIALIZATION_REDUCTION_DEVICE = "cpu"
SEMANTIC_INITIALIZATION_REDUCTION_DTYPE = "float32"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
RELOAD_REFERENCE_FORWARD_PATH = "raw-unwrapped-no-accelerate-autocast-v1"


@dataclass(frozen=True)
class InterestTokenEntry:
    interest_index: int
    interest_token: str
    target_subcategory: str
    label_token_ids: tuple[int, ...]
    token_id: int


@dataclass(frozen=True)
class InterestTokenSpec:
    mapping_file: Path
    mapping_sha256: str
    original_vocabulary_size: int
    resized_vocabulary_size: int
    entries: tuple[InterestTokenEntry, ...]

    @property
    def token_ids(self) -> list[int]:
        return [entry.token_id for entry in self.entries]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_interest_token_mapping(
    path: Path | str,
) -> list[dict[str, object]]:
    path = Path(path)
    required = {
        "interest_index",
        "interest_token",
        "target_subcategory",
    }
    missing = required.difference(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    has_roles = "is_output_candidate" in pq.ParquetFile(path).schema_arrow.names
    columns = required | ({"is_output_candidate"} if has_roles else set())
    rows = sorted(
        pq.read_table(path, columns=sorted(columns)).to_pylist(),
        key=lambda row: int(row["interest_index"]),
    )
    indices = [int(row["interest_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"{path}: interest_index must be contiguous from zero")
    tokens = [str(row["interest_token"]) for row in rows]
    labels = [str(row["target_subcategory"]) for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{path}: duplicate interest_token")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{path}: duplicate target_subcategory")
    expected_tokens = [f"<interest_{index:03d}>" for index in indices]
    if tokens != expected_tokens:
        raise ValueError(f"{path}: interest tokens do not match canonical IDs")
    if has_roles:
        roles = [row["is_output_candidate"] for row in rows]
        if any(type(role) is not bool for role in roles):
            raise ValueError(f"{path}: candidate roles must be boolean")
        count = sum(roles)
        if not count or roles != [True] * count + [False] * (len(rows) - count):
            raise ValueError(f"{path}: output candidates must be a nonempty prefix")
        # Preserve established output IDs when appending history-only categories.
        if labels[:count] != sorted(labels[:count]) or labels[count:] != sorted(labels[count:]):
            raise ValueError(f"{path}: each category role must be sorted")
    elif labels != sorted(labels):
        raise ValueError(f"{path}: target_subcategory is not sorted")
    return rows


def register_semantic_interest_tokens(
    tokenizer: Any,
    mapping_file: Path | str,
    *,
    expected_count: int = 108,
) -> InterestTokenSpec:
    mapping_file = Path(mapping_file)
    rows = load_interest_token_mapping(mapping_file)
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} interest tokens, got {len(rows)}"
        )
    original_vocabulary_size = len(tokenizer)
    label_token_ids: list[tuple[int, ...]] = []
    for row in rows:
        ids = tuple(
            int(value)
            for value in tokenizer(
                str(row["target_subcategory"]),
                add_special_tokens=False,
            )["input_ids"]
        )
        if not ids:
            raise ValueError(
                f"Empty base tokenization for {row['target_subcategory']}"
            )
        if any(value >= original_vocabulary_size for value in ids):
            raise ValueError("Label tokenization contains a non-base token")
        label_token_ids.append(ids)

    tokens = [str(row["interest_token"]) for row in rows]
    added = int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens}
        )
    )
    if added != len(tokens):
        raise ValueError(
            f"Expected to add {len(tokens)} new tokens, tokenizer added {added}"
        )
    entries: list[InterestTokenEntry] = []
    for row, source_ids in zip(rows, label_token_ids):
        token = str(row["interest_token"])
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"{token} does not encode as exactly one token")
        token_id = int(encoded[0])
        converted = int(tokenizer.convert_tokens_to_ids(token))
        if converted != token_id:
            raise ValueError(f"{token}: tokenizer ID mismatch")
        entries.append(
            InterestTokenEntry(
                interest_index=int(row["interest_index"]),
                interest_token=token,
                target_subcategory=str(row["target_subcategory"]),
                label_token_ids=source_ids,
                token_id=token_id,
            )
        )
    if len({entry.token_id for entry in entries}) != len(entries):
        raise ValueError("Interest token IDs are not unique")
    if any(
        entry.token_id < original_vocabulary_size for entry in entries
    ):
        raise ValueError("An interest token reused a base vocabulary ID")
    return InterestTokenSpec(
        mapping_file=mapping_file,
        mapping_sha256=sha256_file(mapping_file),
        original_vocabulary_size=original_vocabulary_size,
        resized_vocabulary_size=len(tokenizer),
        entries=tuple(entries),
    )


def validate_saved_interest_tokens(
    tokenizer: Any,
    mapping_file: Path | str,
    *,
    expected_count: int = 108,
) -> InterestTokenSpec:
    mapping_file = Path(mapping_file)
    rows = load_interest_token_mapping(mapping_file)
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} interest tokens, got {len(rows)}"
        )
    entries = []
    for row in rows:
        label_ids = tuple(
            int(value)
            for value in tokenizer(
                str(row["target_subcategory"]),
                add_special_tokens=False,
            )["input_ids"]
            if int(value) != int(tokenizer.unk_token_id or -1)
        )
        token = str(row["interest_token"])
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"{token} does not encode as exactly one token")
        entries.append(
            InterestTokenEntry(
                interest_index=int(row["interest_index"]),
                interest_token=token,
                target_subcategory=str(row["target_subcategory"]),
                label_token_ids=label_ids,
                token_id=int(encoded[0]),
            )
        )
    if len({entry.token_id for entry in entries}) != len(entries):
        raise ValueError("Interest token IDs are not unique")
    original_vocabulary_size = min(entry.token_id for entry in entries)
    return InterestTokenSpec(
        mapping_file=mapping_file,
        mapping_sha256=sha256_file(mapping_file),
        original_vocabulary_size=original_vocabulary_size,
        resized_vocabulary_size=len(tokenizer),
        entries=tuple(entries),
    )


def _module_name(model: Any, module: Any) -> str:
    matches = [name for name, candidate in model.named_modules() if candidate is module]
    if not matches:
        raise ValueError(f"Could not resolve module name for {type(module).__name__}")
    return min(matches, key=lambda value: (value.count("."), len(value)))


def resolve_token_modules(model: Any) -> dict[str, object]:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if input_module is None:
        raise ValueError("Model has no input embedding module")
    if output_module is None:
        raise ValueError("Model has no output embedding module")
    input_name = _module_name(model, input_module)
    output_name = _module_name(model, output_module)
    def base_weight(module: Any) -> Any:
        token_adapter = getattr(module, "token_adapter", None)
        if token_adapter is not None:
            return token_adapter.get_base_layer().weight
        get_base_layer = getattr(module, "get_base_layer", None)
        if callable(get_base_layer):
            base_layer = get_base_layer()
            if hasattr(base_layer, "weight"):
                return base_layer.weight
        return module.weight

    # PEFT's TrainableTokensWrapper exposes an effective ``weight`` made with
    # index_copy. Two tied wrappers therefore return different temporary
    # tensors even though their underlying base weights are shared. Resolve
    # sharing from those base parameters instead of the effective tensors.
    tied = (
        base_weight(input_module).data_ptr()
        == base_weight(output_module).data_ptr()
    )
    configured_tied = bool(
        getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    )
    if configured_tied != tied:
        raise ValueError(
            "tie_word_embeddings does not match the actual parameter sharing"
        )
    return {
        "input_module": input_module,
        "output_module": output_module,
        "input_module_name": input_name,
        "output_module_name": output_name,
        "tied": tied,
    }


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(dtype=__import__("torch").float32, device="cpu")
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def semantic_initialization_version(mode: str) -> str:
    if mode == SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED:
        return SEMANTIC_INITIALIZATION_VERSION
    if mode == SEMANTIC_INITIALIZATION_MODE_MEAN:
        return MERA_MEAN_INITIALIZATION_VERSION
    raise ValueError(f"Unsupported Semantic-ID initialization mode: {mode}")


def _semantic_row(
    weight: Any,
    source_ids: tuple[int, ...],
    *,
    initialization_mode: str = SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED,
) -> Any:
    import torch

    indices = torch.tensor(source_ids, device=weight.device, dtype=torch.long)
    # Training constructs the base model on CPU, while reload/evaluation often
    # use ``device_map`` and construct it directly on CUDA.  Performing the
    # mean/norm reductions on the model's current device therefore makes a
    # handful of BF16 rows cross different rounding boundaries even when the
    # checkpoint weights and formula are identical.  Move only the small set
    # of label rows to CPU FP32 so semantic initialization is independent of
    # model placement and can be reconstructed bit-exactly after adapter save.
    source = weight.index_select(0, indices).to(
        device=SEMANTIC_INITIALIZATION_REDUCTION_DEVICE,
        dtype=torch.float32,
    )
    mean = source.mean(dim=0)
    if initialization_mode == SEMANTIC_INITIALIZATION_MODE_MEAN:
        return mean.to(device=weight.device, dtype=weight.dtype)
    semantic_initialization_version(initialization_mode)
    mean_norm = mean.norm(p=2)
    if not torch.isfinite(mean_norm) or float(mean_norm) <= 0.0:
        raise ValueError("Semantic initialization produced a zero/invalid mean")
    target_norm = source.norm(p=2, dim=1).mean()
    row = mean * (target_norm / mean_norm)
    return row.to(device=weight.device, dtype=weight.dtype)


def initialize_semantic_interest_token_rows(
    model: Any,
    spec: InterestTokenSpec,
    *,
    initialization_mode: str = SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED,
) -> tuple[dict[str, object], dict[str, Any]]:
    import torch

    if model.get_input_embeddings().weight.shape[0] != spec.resized_vocabulary_size:
        raise ValueError("Model input vocabulary was not resized to the tokenizer")
    modules = resolve_token_modules(model)
    input_module = modules["input_module"]
    output_module = modules["output_module"]
    per_token: list[dict[str, object]] = []
    initial_rows: dict[str, list[Any]] = {"input": [], "output": []}
    with torch.no_grad():
        for entry in spec.entries:
            input_row = _semantic_row(
                input_module.weight,
                entry.label_token_ids,
                initialization_mode=initialization_mode,
            )
            input_module.weight[entry.token_id].copy_(input_row)
            if modules["tied"]:
                output_row = input_row
            else:
                output_row = _semantic_row(
                    output_module.weight,
                    entry.label_token_ids,
                    initialization_mode=initialization_mode,
                )
                output_module.weight[entry.token_id].copy_(output_row)
            initial_rows["input"].append(input_row.detach().float().cpu())
            initial_rows["output"].append(output_row.detach().float().cpu())
            per_token.append(
                {
                    "interest_index": entry.interest_index,
                    "interest_token": entry.interest_token,
                    "target_subcategory": entry.target_subcategory,
                    "token_id": entry.token_id,
                    "label_token_ids": list(entry.label_token_ids),
                    "input_initial_norm": float(input_row.float().norm()),
                    "output_initial_norm": float(output_row.float().norm()),
                    "input_initial_sha256": _tensor_sha256(input_row),
                    "output_initial_sha256": _tensor_sha256(output_row),
                }
            )
    audit = {
        "initialization_version": semantic_initialization_version(
            initialization_mode
        ),
        "initialization_mode": initialization_mode,
        "reduction_device": SEMANTIC_INITIALIZATION_REDUCTION_DEVICE,
        "reduction_dtype": SEMANTIC_INITIALIZATION_REDUCTION_DTYPE,
        "mapping_file": str(spec.mapping_file),
        "mapping_sha256": spec.mapping_sha256,
        "original_vocabulary_size": spec.original_vocabulary_size,
        "resized_vocabulary_size": spec.resized_vocabulary_size,
        "interest_token_count": len(spec.entries),
        "input_module_name": modules["input_module_name"],
        "output_module_name": modules["output_module_name"],
        "tie_word_embeddings": modules["tied"],
        "tokens": per_token,
    }
    return audit, initial_rows


def prepare_semantic_adapter_base(
    model: Any,
    spec: InterestTokenSpec,
    *,
    initialization_mode: str = SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED,
) -> dict[str, object]:
    """Reconstruct frozen semantic rows before loading a row-only adapter.

    PEFT stores the learned ``trainable_token_indices`` deltas, not the full
    resized embedding or LM-head rows.  A fresh base model must therefore
    recreate the same semantic initialization used before training.  Loading
    the deltas on top of randomly resized rows would not reproduce the trained
    model, even when every saved adapter tensor is intact.
    """

    if (
        model.get_input_embeddings().weight.shape[0]
        != spec.resized_vocabulary_size
    ):
        model.resize_token_embeddings(spec.resized_vocabulary_size)
    audit, _ = initialize_semantic_interest_token_rows(
        model,
        spec,
        initialization_mode=initialization_mode,
    )
    return audit


def trainable_token_indices(
    model: Any,
    spec: InterestTokenSpec,
    *,
    output_only: bool = False,
) -> dict[str, list[int]]:
    modules = resolve_token_modules(model)
    if output_only and not modules["tied"]:
        return {
            str(modules["output_module_name"]): spec.token_ids,
        }
    result = {
        str(modules["input_module_name"]): spec.token_ids,
    }
    if not modules["tied"]:
        result[str(modules["output_module_name"])] = spec.token_ids
    return result


def audit_interest_token_drift(
    model: Any,
    spec: InterestTokenSpec,
    initial_rows: dict[str, list[Any]],
) -> dict[str, object]:
    import torch

    modules = resolve_token_modules(model)
    per_token = []
    with torch.no_grad():
        for offset, entry in enumerate(spec.entries):
            current_input = modules["input_module"].weight[
                entry.token_id
            ].detach().float().cpu()
            current_output = modules["output_module"].weight[
                entry.token_id
            ].detach().float().cpu()
            initial_input = initial_rows["input"][offset]
            initial_output = initial_rows["output"][offset]
            per_token.append(
                {
                    "interest_token": entry.interest_token,
                    "target_subcategory": entry.target_subcategory,
                    "token_id": entry.token_id,
                    "input_drift_l2": float(
                        (current_input - initial_input).norm()
                    ),
                    "output_drift_l2": float(
                        (current_output - initial_output).norm()
                    ),
                    "input_final_sha256": _tensor_sha256(current_input),
                    "output_final_sha256": _tensor_sha256(current_output),
                }
            )
    input_drifts = [float(row["input_drift_l2"]) for row in per_token]
    output_drifts = [float(row["output_drift_l2"]) for row in per_token]
    return {
        "interest_token_count": len(per_token),
        "changed_input_rows": sum(value > 0.0 for value in input_drifts),
        "changed_output_rows": sum(value > 0.0 for value in output_drifts),
        "all_input_rows_changed": all(value > 0.0 for value in input_drifts),
        "all_output_rows_changed": all(value > 0.0 for value in output_drifts),
        "min_input_drift_l2": min(input_drifts),
        "max_input_drift_l2": max(input_drifts),
        "min_output_drift_l2": min(output_drifts),
        "max_output_drift_l2": max(output_drifts),
        "tokens": per_token,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def capture_torch_numeric_backend(torch_module: Any) -> dict[str, object]:
    """Record inference-relevant torch precision switches.

    ``TrainingArguments(tf32=True)`` mutates process-global CUDA backend
    switches.  Adapter reload verification runs in a fresh process, so it must
    restore the same switches before comparing logits.  Otherwise FP32 LoRA
    matrix multiplications can follow a different TF32 path even though the
    adapter tensors were saved and loaded correctly.
    """

    return {
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(
            torch_module.get_float32_matmul_precision()
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
            torch_module.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cudnn_deterministic": bool(
            torch_module.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }


def deterministic_reload_numeric_backend() -> dict[str, object]:
    """Return the strict backend used by both sides of reload verification.

    Matching only the TF32 switches is insufficient for BF16 CUDA forwards:
    independent processes may select different cuBLAS workspaces or reduced-
    precision accumulation paths.  The reload gate therefore runs its small
    four-prompt audit with deterministic algorithms and full-precision
    reductions, independently of the faster training backend.
    """

    return {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    }


def apply_torch_numeric_backend(
    torch_module: Any,
    backend: dict[str, object],
) -> None:
    required = {
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "float32_matmul_precision",
        "cuda_matmul_allow_bf16_reduced_precision_reduction",
        "cuda_matmul_allow_fp16_reduced_precision_reduction",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cublas_workspace_config",
    }
    missing = required.difference(backend)
    if missing:
        raise ValueError(
            "Reload reference is missing torch numeric backend fields: "
            f"{sorted(missing)}"
        )
    workspace_config = backend["cublas_workspace_config"]
    current_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        torch_module.cuda.is_initialized()
        and workspace_config != current_workspace_config
    ):
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization: "
            f"current={current_workspace_config!r}, "
            f"requested={workspace_config!r}"
        )
    if workspace_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(workspace_config)
    torch_module.backends.cuda.matmul.allow_tf32 = bool(
        backend["cuda_matmul_allow_tf32"]
    )
    torch_module.backends.cudnn.allow_tf32 = bool(
        backend["cudnn_allow_tf32"]
    )
    torch_module.set_float32_matmul_precision(
        str(backend["float32_matmul_precision"])
    )
    torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
        backend["cuda_matmul_allow_bf16_reduced_precision_reduction"]
    )
    torch_module.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
        backend["cuda_matmul_allow_fp16_reduced_precision_reduction"]
    )
    torch_module.backends.cudnn.deterministic = bool(
        backend["cudnn_deterministic"]
    )
    torch_module.backends.cudnn.benchmark = bool(
        backend["cudnn_benchmark"]
    )
    torch_module.use_deterministic_algorithms(
        bool(backend["deterministic_algorithms"]),
        warn_only=bool(backend["deterministic_warn_only"]),
    )


@contextmanager
def deterministic_sdpa_math_context(torch_module: Any):
    """Force the deterministic math SDPA kernel for reload audit forwards.

    ``torch.use_deterministic_algorithms`` does not by itself pin PyTorch's
    scaled-dot-product-attention backend.  Two fresh CUDA processes can pick
    different fused SDPA kernels and produce BF16 logits that differ by one or
    more representable steps even when every model and adapter tensor is
    bit-identical.  The four-example reload audit is small enough to use the
    math kernel explicitly; training and full evaluation retain their normal
    faster kernels.
    """

    attention = getattr(torch_module.nn, "attention", None)
    sdpa_kernel = getattr(attention, "sdpa_kernel", None)
    sdp_backend = getattr(attention, "SDPBackend", None)
    if sdpa_kernel is not None and sdp_backend is not None:
        with sdpa_kernel(backends=[sdp_backend.MATH]):
            yield
        return

    # Compatibility fallback for older PyTorch releases used by some local
    # CPU test environments.  Both remote Semantic ID workers currently use
    # the torch.nn.attention API above.
    with torch_module.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    ):
        yield


def capture_reload_reference(
    model: Any,
    tokenizer: Any,
    train_file: Path | str,
    spec: InterestTokenSpec,
    *,
    enable_thinking: bool,
    sample_count: int = 4,
    initialization_mode: str = SEMANTIC_INITIALIZATION_MODE_NORM_MATCHED,
) -> dict[str, object]:
    import torch

    from new_interest.evaluation.evaluate_beam import canonical_sha256
    from new_interest.training.sft_data import apply_chat_template

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if hasattr(model, "_original_forward"):
        raise ValueError(
            "Reload reference must use the raw model forward without "
            "Accelerate's mixed-precision wrapper"
        )
    rows = pq.read_table(
        train_file,
        columns=["source_sample_id", "prompt"],
    ).slice(0, sample_count).to_pylist()
    if len(rows) != sample_count:
        raise ValueError(
            f"Expected {sample_count} reload reference rows, got {len(rows)}"
        )
    was_training = model.training
    original_backend = capture_torch_numeric_backend(torch)
    reload_backend = deterministic_reload_numeric_backend()
    apply_torch_numeric_backend(torch, reload_backend)
    model.eval()
    output_rows = []
    token_index = torch.tensor(
        spec.token_ids,
        dtype=torch.long,
        device=model.device,
    )
    try:
        with torch.inference_mode(), deterministic_sdpa_math_context(torch):
            for row in rows:
                prompt = str(row["prompt"])
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
                    key: value.to(model.device)
                    for key, value in encoded.items()
                }
                logits = model(**encoded, use_cache=False).logits[0, -1]
                restricted = logits.index_select(0, token_index).float().cpu()
                output_rows.append(
                    {
                        "source_sample_id": str(row["source_sample_id"]),
                        "prompt": prompt,
                        "prompt_sha256": canonical_sha256(prompt),
                        "restricted_logits": restricted.tolist(),
                    }
                )
    finally:
        if was_training:
            model.train()
        apply_torch_numeric_backend(torch, original_backend)
    return {
        "mapping_sha256": spec.mapping_sha256,
        "token_ids": spec.token_ids,
        "semantic_initialization_version": semantic_initialization_version(
            initialization_mode
        ),
        "semantic_initialization_mode": initialization_mode,
        "semantic_initialization_reduction_device": (
            SEMANTIC_INITIALIZATION_REDUCTION_DEVICE
        ),
        "semantic_initialization_reduction_dtype": (
            SEMANTIC_INITIALIZATION_REDUCTION_DTYPE
        ),
        "torch_numeric_backend": reload_backend,
        "sdpa_backend": "math",
        "forward_path": RELOAD_REFERENCE_FORWARD_PATH,
        "interest_tokens": [
            entry.interest_token for entry in spec.entries
        ],
        "sample_count": sample_count,
        "samples": output_rows,
    }
