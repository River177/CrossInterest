"""Exact-autograd Qwen3.5 backend for the OI/NI meta-learning workflow.

This keeps Qwen3.5's gated-delta and full-attention layers, but selects its
ordinary PyTorch kernels. Fused first-order kernels and module-internal
activation checkpointing are deliberately outside this audited contract.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


BACKEND_VERSION = "qwen35-exact-meta-torch-v1"
AUDITED_TRANSFORMERS_VERSION = "5.14.1"


def _text_causal_lm(model: nn.Module) -> nn.Module:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ != "Qwen3_5ForCausalLM":
        raise TypeError("Exact OI/NI backend requires Qwen3_5ForCausalLM (text model)")
    if getattr(base.config, "model_type", None) != "qwen3_5_text":
        raise TypeError("Unexpected Qwen3.5 configuration; multimodal wrappers are not supported")
    return base


def configure_exact_meta_backend(model: nn.Module) -> dict[str, Any]:
    """Select native differentiable kernels without editing installed packages.

    Call after attaching PEFT and before creating the optimizer. Parameter
    objects, names, values, and tying are preserved, including gated-norm
    weights when replacing an installed FLA norm. No dropout policy is changed.
    """
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as impl

    if transformers.__version__ != AUDITED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Exact meta backend was audited with Transformers {AUDITED_TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    base = _text_causal_lm(model)
    checkpointing_was_enabled = any(
        bool(getattr(module, "gradient_checkpointing", False)) for module in model.modules()
    )
    base.gradient_checkpointing_disable()
    base.config.use_cache = False
    base.config._attn_implementation = "eager"
    linear_layers: list[str] = []
    full_layers: list[str] = []
    replaced_norms: list[str] = []
    tied_aliases: list[str] = []
    # PEFT 0.18 registers the SAME ParameterDict beneath input and tied output
    # adapters. torch.func's tied expansion then swaps the same leaf twice and
    # restores a fast tensor instead of the original Parameter. Keep the tied
    # output's alias as an ordinary Python reference, just as PEFT already does
    # for its _tied_adapter pointer. The sole owner and all unique parameter
    # names/objects remain unchanged; output reads still share the same dict.
    for name, module in list(model.named_modules()):
        if type(module).__name__ != "TrainableTokensLayer" or module.tied_adapter is None:
            continue
        for attribute in ("trainable_tokens_delta", "trainable_tokens_original"):
            value = getattr(module, attribute)
            if value is not getattr(module.tied_adapter, attribute):
                raise RuntimeError(f"Unexpected nonshared tied-token state at {name}.{attribute}")
            if attribute in module._modules:
                del module._modules[attribute]
                object.__setattr__(module, attribute, value)
                tied_aliases.append(f"{name}.{attribute}")
    for name, module in list(model.named_modules()):
        if isinstance(module, impl.Qwen3_5GatedDeltaNet):
            module.causal_conv1d_fn = None
            module.causal_conv1d_update = impl.torch_causal_conv1d_update
            module.chunk_gated_delta_rule = impl.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = impl.torch_recurrent_gated_delta_rule
            if not isinstance(module.norm, impl.Qwen3_5RMSNormGated):
                if set(module.norm.state_dict()) != {"weight"}:
                    raise RuntimeError(f"Unsupported fused gated norm state at {name}.norm")
                original_norm = module.norm
                replacement = impl.Qwen3_5RMSNormGated(
                    module.head_v_dim, eps=module.layer_norm_epsilon
                )
                replacement.weight = original_norm.weight
                replacement.train(original_norm.training)
                module.norm = replacement
                replaced_norms.append(f"{name}.norm")
            linear_layers.append(name)
        elif isinstance(module, impl.Qwen3_5Attention):
            module.config._attn_implementation = "eager"
            full_layers.append(name)
    if not linear_layers or not full_layers:
        raise ValueError("The audited Qwen3.5 structure must contain both gated-delta and full-attention layers")
    return {
        "version": BACKEND_VERSION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attention": "eager",
        "gated_delta": "transformers.torch_chunk_gated_delta_rule",
        "convolution": "torch.nn.Conv1d + torch.nn.functional.silu",
        "gated_norm": "transformers.Qwen3_5RMSNormGated",
        "linear_attention_modules": linear_layers,
        "full_attention_modules": full_layers,
        "replaced_norm_modules": replaced_norms,
        "unregistered_tied_alias_modules": tied_aliases,
        "tied_state_alias_policy": "one registered input owner; tied output holds the same objects by reference",
        "checkpointing_was_enabled": checkpointing_was_enabled,
        "internal_gradient_checkpointing": False,
        "use_cache": False,
        "projection_dtype": "float32",
        "dropout_policy": "unchanged; caller must control stochasticity for derivative comparisons",
    }


def effective_sid_rows(causal_lm: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """Read effective SID rows, including PEFT's tied replacement-row adapter.

    Despite its name, PEFT 0.18's ``trainable_tokens_delta`` stores replacement
    rows. Matching ``module.weight`` requires casting those rows to the base
    dtype, not adding them to base rows. Selecting before index-copy avoids
    materializing a second full-vocabulary embedding matrix for the head.
    """
    output = causal_lm.get_output_embeddings()
    adapter = getattr(output, "token_adapter", None)
    if adapter is None:
        return output.weight.index_select(0, token_ids.to(output.weight.device))
    base_weight = adapter.get_base_layer().weight
    selected_ids = token_ids.detach().cpu().tolist()
    rows = base_weight.index_select(0, token_ids.to(base_weight.device))
    if adapter.disable_adapters or adapter.merged:
        return rows
    for name in adapter.active_adapters:
        available = {int(token): index for index, token in enumerate(adapter.token_indices[name])}
        pairs = [(i, available[token]) for i, token in enumerate(selected_ids) if token in available]
        if not pairs:
            continue
        destination = torch.tensor([i for i, _ in pairs], device=rows.device)
        source = torch.tensor([j for _, j in pairs], device=rows.device)
        replacement = adapter.trainable_tokens_delta[name].to(base_weight).index_select(0, source)
        rows = rows.index_copy(0, destination, replacement)
    return rows


class RestrictedSIDModel(nn.Module):
    """Qwen3.5 text body plus its effective SID head; returns [batch, SID].

    ``torch.func.functional_call`` targets this wrapper, so fast-weight names
    include ``causal_lm.``. The body still contains the original PEFT layers.
    Only the final projection is performed in FP32. The same wrapper must be
    used by all compared training conditions and their evaluation.
    """

    def __init__(self, causal_lm: nn.Module, sid_token_ids: Sequence[int]) -> None:
        super().__init__()
        ids = tuple(int(value) for value in sid_token_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("SID IDs must contain at least two distinct tokens")
        base = _text_causal_lm(causal_lm)
        if min(ids) < 0 or max(ids) >= base.config.vocab_size:
            raise ValueError("SID ID lies outside the resized model vocabulary")
        self.causal_lm = causal_lm
        self.register_buffer("sid_token_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        self.backend_audit = configure_exact_meta_backend(causal_lm)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if any(bool(getattr(module, "gradient_checkpointing", False)) for module in self.causal_lm.modules()):
            raise RuntimeError(
                "Internal gradient checkpointing is unaudited with functional fast weights; "
                "keep it disabled and recompute each microbatch's higher-order graph explicitly"
            )
        base = _text_causal_lm(self.causal_lm)
        outputs = base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            last_hidden = hidden[:, -1, :]
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError("Expected a 2D padding mask aligned with input_ids")
            positions = torch.arange(input_ids.shape[1], device=hidden.device)
            last = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).amax(dim=1)
            if bool((last < 0).any()):
                raise ValueError("Every example needs at least one visible input token")
            last_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        rows = effective_sid_rows(self.causal_lm, self.sid_token_ids)
        # Explicit .float() alone does not override an enclosing AMP context.
        with torch.autocast(device_type=last_hidden.device.type, enabled=False):
            return F.linear(last_hidden.float(), rows.float())
