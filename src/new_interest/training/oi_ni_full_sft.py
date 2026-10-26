"""Completion-only full-vocabulary SFT for exact OI -> NI meta learning.

History/padding remain masked. All completion tokens, including the template
suffix, compete against the entire resized vocabulary. Training uses the native
causal-LM forward and loss, including its full head and original tensor shapes.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from new_interest.training.oi_ni_meta_backend import (
    RestrictedSIDModel, _text_causal_lm,
)
from new_interest.training.oi_ni_meta_objective import LossChunk
from new_interest.training.sft_data import tokenize_training_example


def full_vocabulary_projection(hidden, frozen_weight, token_ids, replacement_rows):
    """Diagnostic: mathematically equivalent to a materialized full head.

    Keep the output projection in the base weight dtype, as in ordinary SFT;
    cross entropy promotes the resulting logits to float32. The vocabulary is
    never restricted. The separate trainable columns preserve second derivatives
    without creating a differentiable full embedding matrix. This is NOT used
    for training: split BF16 head gradients failed the native-SFT parity gate.
    """
    logits = F.linear(hidden, frozen_weight)
    replacement = F.linear(hidden, replacement_rows.to(frozen_weight))
    return logits.index_copy(-1, token_ids, replacement)


class FullVocabularySFTModel(RestrictedSIDModel):
    def __init__(self, causal_lm: nn.Module, sid_token_ids: Sequence[int]):
        super().__init__(causal_lm, sid_token_ids)
        self.backend_audit.update(
            training_loss_scope="full_vocabulary",
            training_projection_dtype="base_weight_dtype_bfloat16",
            loss_dtype="float32", loss_reduction="token_mean",
            supervision="complete_chat_template_suffix_after_prompt",
            masked_position_projection=True,
            training_forward="native_causal_lm_forward_and_loss",
            split_sid_projection=False,
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        if labels is None:
            return super().forward(input_ids, attention_mask, position_ids)
        if labels.shape != input_ids.shape:
            raise ValueError("labels must align with input_ids")
        base = _text_causal_lm(self.causal_lm)
        if any(bool(getattr(m, "gradient_checkpointing", False)) for m in base.modules()):
            raise RuntimeError("Internal checkpointing is unsupported with functional fast parameters")
        if not bool(labels[:, 1:].ne(-100).any()):
            raise ValueError("Completion must contain supervised next tokens")
        return self.causal_lm(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels, use_cache=False, return_dict=True).loss


def encode_completions(tokenizer: Any, records: list[dict], max_length: int):
    result = []
    for row in records:
        messages = row.get("messages") or [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        encoded = tokenize_training_example(tokenizer, messages, max_seq_length=max_length,
                                            enable_thinking=False)
        count = sum(value != -100 for value in encoded["labels"][1:])
        if count <= 0:
            raise ValueError("Empty supervision")
        encoded["supervised_tokens"] = count
        result.append(encoded)
    return result


def completion_batch(examples, pad_token, device):
    width = max(len(row["input_ids"]) for row in examples)
    # Right padding matches the original SFT collator. Training microbatch is 1.
    return {key: torch.tensor([
        row[key] + [padding] * (width - len(row[key])) for row in examples
    ], dtype=torch.long, device=device)
        for key, padding in (("input_ids", pad_token), ("attention_mask", 0), ("labels", -100))}


def pooled_role_scaling(oi_tokens: int, ni_tokens: int, alpha: float):
    """Map token-pooled weights to the established .5/.25/.25 engine.

    p = T_OI / (T_OI + T_NI). Passing 2p*F and 2(1-p)*G with
    alpha/(2p) gives p*F + (1-p)/2*(G + G_after), where the actual virtual
    update is theta - alpha*grad(F). At alpha=0 this is ordinary pooled SFT.
    """
    if oi_tokens <= 0 or ni_tokens <= 0 or alpha < 0:
        raise ValueError("Both roles need positive token counts and alpha must be nonnegative")
    oi_scale = 2 * oi_tokens / (oi_tokens + ni_tokens)
    ni_scale = 2 * ni_tokens / (oi_tokens + ni_tokens)
    return oi_scale, ni_scale, alpha / oi_scale


def completion_chunks(indices, examples, pad_token, device, *, scale, step_seed):
    chunks = []
    for position, index in enumerate(indices):
        example = examples[index]
        seed = int(step_seed) + 8191 * position
        def loss_fn(view, example=example, seed=seed):
            devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed(seed)
                return scale * view(**completion_batch([example], pad_token, device))
        chunks.append(LossChunk(loss_fn, weight=example["supervised_tokens"]))
    return chunks
