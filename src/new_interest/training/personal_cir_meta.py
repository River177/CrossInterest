"""Exact per-user CIR virtual update, followed by a single NI outer objective.

This module never mutates model parameters or accumulates user-specific state.
Only outer gradients may be combined across users. SID remains differentiable
through the inner gradient, although only LoRA has fast parameter values.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from new_interest.training.oi_ni_meta_objective import (
    FunctionalModel, LossChunk, LossSpec, _chunks, _prepare, _guard_identity,
    _grad, _loss, _chunk_gradients, _graph_loss, _replay_rng, _norm,
)


@contextmanager
def forward_mode(model, training):
    states = [(m, m.training) for m in model.modules()]
    model.train(training)
    try:
        yield
    finally:
        for module, value in states:
            module.training = value


@dataclass(frozen=True)
class PersonalGradients:
    loss: Tensor
    support_loss: Tensor
    gradients: Mapping[str, Tensor]
    inner_grad_norm: Tensor
    hvp_norm: Tensor
    support_count: int


def graph_objective(model: nn.Module, support: Sequence[LossChunk], query: LossSpec,
                    *, alpha: float, query_training=True):
    """Small-model reference: retain the full exact unrolled computation graph."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid inner alpha")
    parameters, buffers, _, inner, devices = _prepare(model, alpha)
    view = FunctionalModel(model, parameters, buffers)
    fast = dict(parameters)
    if support and alpha:
        with forward_mode(model, False):
            f, _ = _graph_loss(_chunks(support, "support"), view, devices)
            gradients = _grad(f, [parameters[n] for n in inner], create_graph=True)
        fast.update({n: parameters[n] - alpha * g for n, g in zip(inner, gradients)})
    with forward_mode(model, query_training):
        result, _ = _graph_loss(_chunks(query, "query"), FunctionalModel(model, fast, buffers), devices)
    return result


def compute_personal_gradients(model: nn.Module, support: Sequence[LossChunk], query: LossSpec,
                               *, alpha: float, query_training=True, check_identity=True):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid inner alpha")
    with _guard_identity(model, check_identity):
        parameters, buffers, trainable, inner, devices = _prepare(model, alpha)
        view = FunctionalModel(model, parameters, buffers)
        queries = _chunks(query, "query")
        if not support or alpha == 0:
            with forward_mode(model, query_training):
                loss, gradients, _ = _chunk_gradients(queries, view, trainable, devices)
            zero = loss.new_zeros((), dtype=torch.float64)
            return PersonalGradients(loss, zero, gradients, zero, zero, len(support))
        supports = _chunks(support, "support")
        with forward_mode(model, False):
            f, inner_grads, states = _chunk_gradients(supports, view, inner, devices, create_graph=True)
        fast = dict(parameters)
        fast.update({n: (parameters[n].detach() - alpha * inner_grads[n]).requires_grad_(True) for n in inner})
        with forward_mode(model, query_training):
            loss, gradients, _ = _chunk_gradients(queries, FunctionalModel(model, fast, buffers), trainable, devices)
        vector = {n: gradients[n].clone() for n in inner}
        del fast
        all_targets = tuple(parameters[n] for n in trainable)
        inner_targets = tuple(parameters[n] for n in inner)
        hvp = {n: torch.zeros_like(parameters[n]) for n in trainable}
        repeated = {n: torch.zeros_like(parameters[n]) for n in inner}
        repeated_loss = f.new_zeros(())
        with forward_mode(model, False):
            for chunk, state in zip(supports, states):
                with _replay_rng(state):
                    part = _loss(chunk, view)
                    first = _grad(part, inner_targets, create_graph=True)
                    product = sum(((g * vector[n]).sum().to(part.device) for n, g in zip(inner, first)), part.new_zeros(()))
                    second = _grad(product, all_targets)
                repeated_loss.add_(part.detach(), alpha=chunk.weight)
                for n, g in zip(inner, first):
                    repeated[n].add_(g.detach(), alpha=chunk.weight)
                for n, g in zip(trainable, second):
                    hvp[n].add_(g.detach(), alpha=chunk.weight)
                del part, first, product, second, g
        if not torch.equal(f, repeated_loss):
            raise RuntimeError("Support loss changed during exact HVP recomputation")
        for n in inner:
            tolerance = 8 * torch.finfo(inner_grads[n].dtype).eps
            if not torch.allclose(inner_grads[n], repeated[n], rtol=tolerance, atol=tolerance):
                raise RuntimeError(f"Support gradient changed during recomputation: {n}")
        for n in trainable:
            gradients[n].add_(hvp[n], alpha=-alpha)
            if not bool(torch.isfinite(gradients[n]).all()):
                raise FloatingPointError(f"Nonfinite personal meta gradient: {n}")
        return PersonalGradients(loss, f, gradients, _norm(inner_grads, f), _norm(hvp, f), len(support))
