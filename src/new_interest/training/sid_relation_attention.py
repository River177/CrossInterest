"""One-layer, shared-SID category bias; native Qwen causal-LM loss is retained."""
from __future__ import annotations

from types import MethodType
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


VERSION = "sid-relation-attention-v1"


def text_model(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(f"Expected Qwen3_5ForCausalLM, found {type(base).__name__}")
    return base


def effective_sid_rows(model, token_ids):
    """Read compact learned replacement rows in their stored precision, with gradients."""
    output = model.get_output_embeddings()
    adapter = getattr(output, "token_adapter", None)
    if adapter is None:
        return output.weight.index_select(0, token_ids.to(output.weight.device)).float()
    if adapter.disable_adapters or adapter.merged or len(adapter.active_adapters) != 1:
        raise ValueError("SID relation requires one active, unmerged token adapter")
    name = adapter.active_adapters[0]
    indices = {int(t): i for i, t in enumerate(adapter.token_indices[name])}
    requested = token_ids.detach().cpu().tolist()
    if not set(requested).issubset(indices):
        raise ValueError("Requested SID rows missing from compact token adapter")
    rows = adapter.trainable_tokens_delta[name]
    order = torch.tensor([indices[t] for t in requested], device=rows.device)
    return rows.index_select(0, order).float()


def _biased_forward(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kwargs):
    bias = kwargs.pop("sid_relation_bias", None)
    if bias is None:
        raise ValueError("Missing explicit SID relation sidecar/bias")
    if bias.shape[-2:] != (hidden_states.shape[1], hidden_states.shape[1]):
        raise ValueError("Relation forward currently requires a complete prefill")
    if attention_mask is None:
        n = hidden_states.shape[1]
        previous = 0 if past_key_values is None else past_key_values.get_seq_length(self.layer_idx)
        attention_mask = torch.zeros(n, previous+n, device=hidden_states.device, dtype=torch.float32)
        future = torch.arange(previous+n, device=hidden_states.device)[None, :] > (
            previous + torch.arange(n, device=hidden_states.device)[:, None])
        attention_mask.masked_fill_(future, -torch.inf)
        attention_mask = attention_mask[None, None]
    if attention_mask.shape[-1] != bias.shape[-1]:
        if torch.any(bias != 0):
            raise ValueError("Cannot add new historical events to an existing cache")
        bias = torch.zeros_like(attention_mask, dtype=torch.float32)
    combined = attention_mask.float() + bias.float()
    return self._sid_relation_original_forward(
        hidden_states, position_embeddings, attention_mask=combined,
        past_key_values=past_key_values, **kwargs)


class CategoryRelation(nn.Module):
    def __init__(self, dimension, rank, source_to_sid, fallback_rows, enabled, seed=42):
        super().__init__()
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.register_buffer("source_to_sid", torch.tensor(source_to_sid, dtype=torch.long))
        self.register_buffer("fallback_rows", fallback_rows.detach().float().clone())
        if tuple(self.fallback_rows.shape) != (len(source_to_sid), dimension):
            raise ValueError("Source fallback shape differs")
        generator = torch.Generator(device="cpu").manual_seed(seed + 731791)
        self.U = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension),
                              requires_grad=enabled)
        self.V = nn.Parameter(torch.zeros(dimension, rank), requires_grad=enabled)

    def forward(self, sid_rows, event_ids, source_ids):
        if event_ids.shape != source_ids.shape or event_ids.ndim != 2:
            raise ValueError("Event/category sidecars must have equal [batch, sequence] shapes")
        valid = event_ids.ge(0)
        if torch.any(valid & ((source_ids < 0) | (source_ids >= len(self.source_to_sid)))):
            raise ValueError("History token has invalid source category")
        n = event_ids.shape[1]
        if not self.enabled:
            return torch.zeros(event_ids.shape[0], 1, n, n, dtype=torch.float32,
                               device=event_ids.device)
        with torch.autocast(device_type=sid_rows.device.type, enabled=False):
            mapped = self.source_to_sid.ge(0)
            rows = self.fallback_rows.index_copy(
                0, mapped.nonzero().flatten(), sid_rows[self.source_to_sid[mapped]].float())
            rows = rows / torch.sqrt(rows.square().mean(-1, keepdim=True) + 1e-6)
            matrix = (rows @ self.U) @ (rows @ self.V).T / math.sqrt(self.rank)
            categories = source_ids.clamp_min(0)
            bias = matrix[categories[:, :, None], categories[:, None, :]]
            allowed = valid[:, :, None] & valid[:, None, :] & event_ids[:, :, None].ne(event_ids[:, None, :])
            return (bias * allowed).unsqueeze(1)


class SIDRelationModel(nn.Module):
    def __init__(self, causal_lm, token_ids, source_to_sid, fallback_rows, *,
                 enabled, layer_idx=11, rank=16, seed=42):
        super().__init__()
        self.causal_lm = causal_lm
        self.register_buffer("sid_token_ids", torch.tensor(token_ids, dtype=torch.long))
        base = text_model(causal_lm)
        if base.config.layer_types[layer_idx] != "full_attention" or layer_idx >= len(base.model.layers) - 1:
            raise ValueError("Injection must be a full-attention layer before a subsequent mixer")
        base.config._attn_implementation = "eager"
        self.layer_idx = layer_idx
        self.relation = CategoryRelation(base.config.hidden_size, rank, source_to_sid,
                                          fallback_rows, enabled, seed)
        selected = base.model.layers[layer_idx].self_attn
        if hasattr(selected, "_sid_relation_original_forward"):
            raise ValueError("Relation injection already installed")
        selected._sid_relation_original_forward = selected.forward
        selected.forward = MethodType(_biased_forward, selected)

    def _kwargs(self, input_ids, attention_mask, event_ids, source_ids, use_cache=False, past_key_values=None):
        if event_ids is None or source_ids is None or event_ids.shape != input_ids.shape:
            raise ValueError("Missing or misaligned per-token event/category sidecar")
        if attention_mask is not None and torch.any(~attention_mask[:, -input_ids.shape[1]:].bool() & event_ids.ge(0)):
            raise ValueError("Padding token assigned to history")
        rows = effective_sid_rows(self.causal_lm, self.sid_token_ids)
        return dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache,
                    past_key_values=past_key_values,
                    sid_relation_bias=self.relation(rows, event_ids, source_ids), return_dict=True)

    def forward(self, input_ids, attention_mask=None, labels=None, event_ids=None, source_ids=None,
                use_cache=False, past_key_values=None):
        return self.causal_lm(**self._kwargs(input_ids, attention_mask, event_ids, source_ids,
                                            use_cache, past_key_values), labels=labels)

    def score(self, input_ids, attention_mask=None, event_ids=None, source_ids=None):
        kwargs = self._kwargs(input_ids, attention_mask, event_ids, source_ids)
        hidden = text_model(self.causal_lm).model(**kwargs).last_hidden_state
        if attention_mask is None:
            last = torch.full((len(input_ids),), input_ids.shape[1]-1, device=input_ids.device)
        else:
            last = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
            last = last.masked_fill(~attention_mask.bool(), -1).amax(1)
        rows = effective_sid_rows(self.causal_lm, self.sid_token_ids)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden[torch.arange(len(input_ids), device=hidden.device), last].float(), rows.float())
