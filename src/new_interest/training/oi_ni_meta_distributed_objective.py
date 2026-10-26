"""Replicated-model, globally normalized exact OI -> NI meta gradients.

Every process owns an ordinary (not DDP-wrapped) model with identical outer
parameters. The caller must synchronize those parameters at initialization and
apply the returned identical global gradients on every rank. Only detached
values are communicated: no distributed-autograd or DDP hooks are involved.

For globally normalized F, G and inner F_i, the algorithm is

    u = sum_rank d_LoRA F_i,rank;  phi' = phi - alpha*u
    v = sum_rank d_phi' G_rank(phi', psi)
    h = sum_rank d_theta <d_phi F_i,rank, stopgrad(v)>
    dJ = sum_rank [.5*dF_rank + .25*dG_rank + .25*dG_fast,rank]
         - .25*alpha*h.

The HVP differentiates with respect to ALL trainable parameters, preserving
LoRA/SID cross-Hessians. Averaging independent per-rank meta-gradients would
implement a different objective; in particular u and v must be global before
forming the virtual parameters and the local HVP, respectively.

LossChunk.weight is a raw contribution mass. Each role is normalized by the
sum of its weights across the entire process group. Thus sixteen weights of
1 and sixteen weights of 1/16 are equivalent, including unequal/empty shards.

RNG contract: a rank seed is NOT a global serial-dropout plan. The runner must
bind each loss callable to its original global chunk's explicit RNG start/end
state (or an explicitly versioned global-chunk RNG scheme). This module does
not invent a new seed policy. Its additional local replay, including BF16
create_graph=True inner replay, composes with such a bound callable. The
runner owns the final per-episode RNG cursor and its checkpointing.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor, nn

from new_interest.training.oi_ni_meta_objective import (
    FunctionalModel,
    LossChunk,
    LossFn,
    MetaGradients,
    _RNGState,
    _capture_rng,
    _grad,
    _guard_identity,
    _loss,
    _norm,
    _prepare,
    _replay_rng,
    _validate_alpha_mode,
)


LocalLossSpec = LossFn | Sequence[LossChunk]
_T = TypeVar("_T")
_BUCKET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _Group:
    process_group: object | None
    world_size: int
    rank: int
    device: torch.device

    def sum(self, value: Tensor) -> Tensor:
        if value.requires_grad:
            raise ValueError("Only detached values may enter distributed collectives")
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.process_group)
        return value

    def gather(self, value: object) -> list[object]:
        if self.world_size == 1:
            return [value]
        result: list[object] = [None] * self.world_size
        dist.all_gather_object(result, value, group=self.process_group)
        return result

    def broadcast(self, value: Tensor, owner: int) -> Tensor:
        if value.requires_grad:
            raise ValueError("Only detached values may enter distributed collectives")
        if self.world_size > 1:
            source = owner if self.process_group is None else dist.get_global_rank(self.process_group, owner)
            dist.broadcast(value, src=source, group=self.process_group)
        return value


def _group(model: nn.Module, process_group: object | None) -> _Group:
    if not dist.is_available() or not dist.is_initialized():
        if process_group is not None:
            raise ValueError("process_group was supplied without initialized torch.distributed")
        first = next(model.parameters(), None)
        return _Group(None, 1, 0, first.device if first is not None else torch.device("cpu"))
    backend = str(dist.get_backend(process_group)).lower()
    if backend == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Audited distributed exact-meta backends are gloo and nccl, found {backend}")
    return _Group(process_group, dist.get_world_size(process_group), dist.get_rank(process_group), device)


def _collectively(group: _Group, phase: str, function: Callable[[], _T]) -> _T:
    """Coordinate local failures before the next tensor collective.

    A bad shard must not leave its peers waiting forever in a gradient reduce.
    Process death/OOM still requires the runner's process-group timeout/abort.
    """
    if group.world_size == 1:
        return function()
    result: _T | None = None
    error: str | None = None
    try:
        result = function()
    except Exception as exception:
        error = f"rank {group.rank}: {type(exception).__name__}: {exception}"
    errors = [error for error in group.gather(error) if error is not None]
    if errors:
        raise RuntimeError(f"Distributed exact-meta {phase} failed: {'; '.join(errors)}")
    return result  # type: ignore[return-value]


def _local_chunks(spec: LocalLossSpec, label: str) -> tuple[LossChunk, ...]:
    supplied = (LossChunk(spec),) if callable(spec) else tuple(spec)
    if not all(isinstance(chunk, LossChunk) and callable(chunk.loss_fn) for chunk in supplied):
        raise TypeError(f"{label} must be a callable or a sequence of LossChunk objects")
    chunks: list[LossChunk] = []
    for chunk in supplied:
        weight = float(chunk.weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{label} chunk weights must be finite and nonnegative")
        if weight:
            chunks.append(LossChunk(chunk.loss_fn, weight))
    return tuple(chunks)


def _normalize_roles(
    group: _Group,
    oi: Sequence[LossChunk],
    ni: Sequence[LossChunk],
    no_history: Sequence[LossChunk],
    *, need_no_history: bool,
) -> tuple[tuple[LossChunk, ...], tuple[LossChunk, ...], tuple[LossChunk, ...]]:
    roles = (oi, ni, no_history)
    mass = torch.tensor(
        [math.fsum(chunk.weight for chunk in role) for role in roles],
        device=group.device, dtype=torch.float64,
    )
    group.sum(mass)
    required = (True, True, need_no_history)
    normalized: list[tuple[LossChunk, ...]] = []
    for name, role, total, needed in zip(("oi", "ni", "no_history_oi"), roles, mass.tolist(), required):
        if not math.isfinite(total) or (needed and total <= 0):
            raise ValueError(f"Global {name} chunk weights must have a positive finite sum")
        normalized.append(tuple(LossChunk(chunk.loss_fn, chunk.weight / total) for chunk in role) if total else ())
    return normalized[0], normalized[1], normalized[2]


def _zeros(parameters: Mapping[str, Tensor], names: Sequence[str]) -> dict[str, Tensor]:
    return {name: torch.zeros_like(parameters[name]) for name in names}


def _reduce_gradients(group: _Group, values: dict[str, Tensor]) -> dict[str, Tensor]:
    """Sum detached gradients in bounded, deterministic dtype/device buckets."""
    if group.world_size == 1:
        return values
    bucket: list[Tensor] = []
    bucket_size = 0
    bucket_key: tuple[torch.device, torch.dtype] | None = None

    def flush() -> None:
        if not bucket:
            return
        flat = torch.cat([value.reshape(-1) for value in bucket])
        group.sum(flat)
        start = 0
        for value in bucket:
            stop = start + value.numel()
            value.copy_(flat[start:stop].reshape_as(value))
            start = stop

    for value in values.values():
        if value.requires_grad:
            raise ValueError("Gradient reduction received a live autograd graph")
        size = value.numel() * value.element_size()
        key = (value.device, value.dtype)
        if bucket and (key != bucket_key or bucket_size + size > _BUCKET_BYTES):
            flush()
            bucket = []
            bucket_size = 0
        bucket.append(value)
        bucket_size += size
        bucket_key = key
    flush()
    return values


def _local_gradient_pass(
    chunks: Sequence[LossChunk],
    view: FunctionalModel,
    names: Sequence[str],
    cuda_devices: Sequence[int],
    *, create_graph: bool = False,
    replay: Sequence[_RNGState] | None = None,
) -> tuple[Tensor, dict[str, Tensor], tuple[_RNGState, ...]]:
    reference = next(iter(view.parameters.values()))
    total = reference.new_zeros((), dtype=torch.float64)
    targets = tuple(view.parameters[name] for name in names)
    gradients = _zeros(view.parameters, names)
    states: list[_RNGState] = []
    for index, chunk in enumerate(chunks):
        state = _capture_rng(cuda_devices) if replay is None else replay[index]
        states.append(state)
        with nullcontext() if replay is None else _replay_rng(state):
            loss = _loss(chunk, view)
            values = _grad(loss, targets, create_graph=create_graph)
        total.add_(loss.detach().double(), alpha=chunk.weight)
        for name, gradient in zip(names, values):
            gradients[name].add_(gradient.detach(), alpha=chunk.weight)
        del loss, values, gradient
    return total, gradients, tuple(states)


def _local_hvp(
    chunks: Sequence[LossChunk], view: FunctionalModel,
    trainable: Sequence[str], inner: Sequence[str],
    replay: Sequence[_RNGState], v: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    reference = next(iter(view.parameters.values()))
    total = reference.new_zeros((), dtype=torch.float64)
    inner_gradients = _zeros(view.parameters, inner)
    hessian_product = _zeros(view.parameters, trainable)
    inner_targets = tuple(view.parameters[name] for name in inner)
    outer_targets = tuple(view.parameters[name] for name in trainable)
    for chunk, state in zip(chunks, replay):
        with _replay_rng(state):
            loss = _loss(chunk, view)
            first = _grad(loss, inner_targets, create_graph=True)
            dot = sum(
                ((gradient * v[name]).sum().to(loss.device) for name, gradient in zip(inner, first)),
                loss.new_zeros(()),
            )
            values = _grad(dot, outer_targets)
        total.add_(loss.detach().double(), alpha=chunk.weight)
        for name, gradient in zip(inner, first):
            inner_gradients[name].add_(gradient.detach(), alpha=chunk.weight)
        for name, value in zip(trainable, values):
            hessian_product[name].add_(value.detach(), alpha=chunk.weight)
        del loss, first, dot, values, gradient, value
    return total, inner_gradients, hessian_product


def _check_recomputed(
    expected_loss: Tensor, actual_loss: Tensor,
    expected_gradients: Mapping[str, Tensor], actual_gradients: Mapping[str, Tensor],
) -> None:
    for name, expected, actual in (
        ("inner loss", expected_loss, actual_loss),
        *((name, value, actual_gradients[name]) for name, value in expected_gradients.items()),
    ):
        tolerance = 8 * torch.finfo(expected.dtype).eps
        if not torch.allclose(expected, actual, rtol=tolerance, atol=tolerance):
            raise RuntimeError(
                f"Inner recomputation changed {name}; global-chunk RNG binding and "
                "pure repeatable losses are required. No approximate meta-gradient was returned."
            )


@dataclass(frozen=True)
class _Contribution:
    loss: Tensor
    gradients: Mapping[str, Tensor]


@dataclass(frozen=True)
class _Position:
    global_index: int
    owner: int
    local_index: int
    weight: float


def _ordered_plans(
    group: _Group,
    roles: Mapping[str, Sequence[LossChunk]],
    global_positions: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[_Position, ...]]:
    def local_metadata():
        if global_positions is None and group.world_size > 1:
            raise ValueError("ordered reduction requires explicit global_positions on every rank")
        metadata = {}
        for role, chunks in roles.items():
            if global_positions is None:
                positions = tuple(range(len(chunks)))
            else:
                positions = tuple(global_positions.get(role, ()))
            if len(positions) != len(chunks):
                raise ValueError(f"global_positions[{role}] must match the local positive-weight chunks")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in positions):
                raise ValueError("global_positions must contain nonnegative integer indices")
            metadata[role] = tuple((position, chunk.weight) for position, chunk in zip(positions, chunks))
        return metadata

    local = _collectively(group, "ordered chunk positions", local_metadata)
    peers = group.gather(local)
    result = {}
    for role in roles:
        plan = sorted(
            (_Position(position, owner, local_index, weight)
             for owner, peer in enumerate(peers)
             for local_index, (position, weight) in enumerate(peer[role])),
            key=lambda value: value.global_index,
        )
        if [value.global_index for value in plan] != list(range(len(plan))):
            raise ValueError(f"Global {role} positions must cover 0..N-1 exactly once")
        result[role] = tuple(plan)
    return result


def _gradient_contributions(
    chunks: Sequence[LossChunk], view: FunctionalModel, names: Sequence[str],
    cuda_devices: Sequence[int], *, create_graph: bool = False,
    replay: Sequence[_RNGState] | None = None,
) -> tuple[tuple[_Contribution, ...], tuple[_RNGState, ...]]:
    """Keep only detached raw per-chunk gradients for ordered communication."""
    targets = tuple(view.parameters[name] for name in names)
    contributions: list[_Contribution] = []
    states: list[_RNGState] = []
    for index, chunk in enumerate(chunks):
        state = _capture_rng(cuda_devices) if replay is None else replay[index]
        states.append(state)
        with nullcontext() if replay is None else _replay_rng(state):
            loss = _loss(chunk, view)
            values = _grad(loss, targets, create_graph=create_graph)
        contributions.append(_Contribution(loss.detach(), {
            name: gradient.detach().clone() for name, gradient in zip(names, values)
        }))
        del loss, values
    return tuple(contributions), tuple(states)


def _parameter_buckets(parameters: Mapping[str, Tensor], names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    buckets: list[tuple[str, ...]] = []
    current: list[str] = []
    size = 0
    dtype = None
    for name in names:
        value = parameters[name]
        count = value.numel() * value.element_size()
        if current and (value.dtype != dtype or size + count > _BUCKET_BYTES):
            buckets.append(tuple(current))
            current, size = [], 0
        current.append(name)
        size += count
        dtype = value.dtype
    if current:
        buckets.append(tuple(current))
    return tuple(buckets)


def _ordered_sum(
    group: _Group, parameters: Mapping[str, Tensor], names: Sequence[str],
    contributions: Sequence[_Contribution], plan: Sequence[_Position],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Broadcast one global chunk at a time and reproduce serial add_ order.

    Each rank stores only its local detached chunk contributions. No rank needs
    to gather a full episode's gradients. Buckets reduce communication latency,
    but accumulation remains per original parameter, in global chunk order.
    Loss multiplication/addition also uses the original scalar dtype/order.
    """
    loss_dtypes = group.gather(tuple(value.loss.dtype for value in contributions))
    gradients = _zeros(parameters, names)
    total: Tensor | None = None
    buckets = _parameter_buckets(parameters, names)
    for position in plan:
        own = contributions[position.local_index] if group.rank == position.owner else None
        raw_loss = (own.loss.clone() if own is not None else
                    torch.empty((), device=group.device, dtype=loss_dtypes[position.owner][position.local_index]))
        group.broadcast(raw_loss, position.owner)
        weighted = position.weight * raw_loss
        total = weighted if total is None else total + weighted
        for bucket in buckets:
            if own is not None:
                flat = torch.cat([own.gradients[name].reshape(-1) for name in bucket])
            else:
                flat = torch.empty(sum(parameters[name].numel() for name in bucket),
                                   device=group.device, dtype=parameters[bucket[0]].dtype)
            group.broadcast(flat, position.owner)
            start = 0
            for name in bucket:
                stop = start + parameters[name].numel()
                gradients[name].add_(flat[start:stop].reshape_as(parameters[name]), alpha=position.weight)
                start = stop
    if total is None:
        raise ValueError("An ordered global objective may not be empty")
    return total, gradients


def _hvp_contributions(
    chunks: Sequence[LossChunk], view: FunctionalModel, trainable: Sequence[str],
    inner: Sequence[str], replay: Sequence[_RNGState], v: Mapping[str, Tensor],
    expected: Sequence[_Contribution], check_recompute: bool,
) -> tuple[_Contribution, ...]:
    inner_targets = tuple(view.parameters[name] for name in inner)
    outer_targets = tuple(view.parameters[name] for name in trainable)
    contributions: list[_Contribution] = []
    for chunk, state, original in zip(chunks, replay, expected):
        with _replay_rng(state):
            loss = _loss(chunk, view)
            first = _grad(loss, inner_targets, create_graph=True)
            if check_recompute:
                _check_recomputed(original.loss, loss.detach(), original.gradients,
                                  {name: value.detach() for name, value in zip(inner, first)})
            dot = sum(
                ((gradient * v[name]).sum().to(loss.device) for name, gradient in zip(inner, first)),
                loss.new_zeros(()),
            )
            values = _grad(dot, outer_targets)
        contributions.append(_Contribution(loss.detach(), {
            name: gradient.detach().clone() for name, gradient in zip(trainable, values)
        }))
        del loss, first, dot, values
    return tuple(contributions)


def _finite_gradients(gradients: Mapping[str, Tensor]) -> None:
    if any(not bool(torch.isfinite(value).all().item()) for value in gradients.values()):
        raise FloatingPointError("distributed exact meta objective produced a non-finite gradient")


def _compute_ordered(
    group: _Group, view: FunctionalModel, trainable: Sequence[str], inner: Sequence[str],
    devices: Sequence[int], oi: Sequence[LossChunk], ni: Sequence[LossChunk], empty: Sequence[LossChunk],
    plans: Mapping[str, Sequence[_Position]], alpha: float, mode: str, check_recompute: bool,
) -> MetaGradients:
    parameters = view.parameters
    f_local, f_rng = _collectively(group, "ordered ordinary OI", lambda: _gradient_contributions(oi, view, trainable, devices))
    f, df = _ordered_sum(group, parameters, trainable, f_local, plans["oi"])
    del f_local
    g_local, g_rng = _collectively(group, "ordered ordinary NI", lambda: _gradient_contributions(ni, view, trainable, devices))
    g, dg = _ordered_sum(group, parameters, trainable, g_local, plans["ni"])
    del g_local
    if alpha == 0:
        gradients = {name: .5 * df[name] + .5 * dg[name] for name in trainable}
        _collectively(group, "ordered global gradient validation", lambda: _finite_gradients(gradients))
        zero = f.new_zeros((), dtype=torch.float64)
        return MetaGradients(.5 * f + .5 * g, f, g, g, f, tuple(inner), gradients, zero, zero)
    # Match core operation order, rather than adding already weighted local
    # outer terms. The latter changes FP32 rounding before BF16 fast forwards.
    direct = {name: .5 * df[name] + .25 * dg[name] for name in trainable}
    del df, dg
    inner_chunks = empty if mode == "meta_no_history" else oi
    inner_plan = plans["no_history_oi"] if mode == "meta_no_history" else plans["oi"]
    inner_local, inner_rng = _collectively(
        group, "ordered differentiable inner gradient",
        lambda: _gradient_contributions(inner_chunks, view, inner, devices, create_graph=True,
                                        replay=None if mode == "meta_no_history" else f_rng),
    )
    inner_f, u = _ordered_sum(group, parameters, inner, inner_local, inner_plan)
    fast = dict(parameters)
    fast.update({name: (parameters[name].detach() - alpha * u[name]).requires_grad_(True) for name in inner})
    fast_view = FunctionalModel(view.module, fast, view._buffers)
    fast_local, _ = _collectively(
        group, "ordered adapted query gradient",
        lambda: _gradient_contributions(ni, fast_view, trainable, devices, replay=g_rng),
    )
    fast_g, fast_direct = _ordered_sum(group, parameters, trainable, fast_local, plans["ni"])
    del fast_local, fast_view, fast
    for name in trainable:
        direct[name].add_(fast_direct[name], alpha=.25)
    v = {name: fast_direct[name] for name in inner}
    del fast_direct
    h_local = _collectively(
        group, "ordered inner HVP recomputation",
        lambda: _hvp_contributions(inner_chunks, view, trainable, inner, inner_rng, v,
                                   inner_local, check_recompute),
    )
    del inner_local
    _, h = _ordered_sum(group, parameters, trainable, h_local, inner_plan)
    del h_local
    for name in trainable:
        direct[name].add_(h[name], alpha=-.25 * alpha)
    _collectively(group, "ordered global gradient validation", lambda: _finite_gradients(direct))
    return MetaGradients(
        .5 * f + .25 * g + .25 * fast_g, f, g, fast_g, inner_f, tuple(inner),
        direct, _norm(u, f), _norm(h, f),
    )


def compute_distributed_meta_gradients(
    model: nn.Module,
    local_oi: LocalLossSpec,
    local_ni: LocalLossSpec,
    *,
    alpha: float,
    mode: str = "meta",
    local_no_history_oi: LocalLossSpec | None = None,
    process_group: object | None = None,
    check_identity: bool = True,
    check_recompute: bool = True,
    reduction: str = "all_reduce",
    global_positions: Mapping[str, Sequence[int]] | None = None,
) -> MetaGradients:
    """Compute global exact gradients on replicated models and sharded chunks.

    Local shards may be empty; global OI/NI (and active no-history inner)
    objectives may not. ``LossChunk.weight`` is normalized globally, never
    locally. Without an initialized process group this is a world-size-one
    reference using the same algorithm. Every returned gradient and diagnostic
    is global and identical on all ranks up to the collective's floating-point
    arithmetic; the caller must NOT average the returned gradients again.

    ``reduction='ordered'`` preserves the serial per-role/per-parameter
    accumulation order, avoiding an all-reduce tree's different FP32 rounding
    before the BF16 virtual forward. ``global_positions`` maps ``oi``, ``ni``,
    and active ``no_history_oi`` to each local positive-weight chunk's original
    zero-based global position. Each role must cover 0..N-1 exactly once. Raw
    detached local chunk gradients are temporarily retained (about 4 chunks per
    rank in the pilot); owner broadcasts are streamed in global order. This
    preserves accumulation order, not an unaudited promise of cross-device
    kernel bitwise equivalence. World-size one may omit positions.

    Initial and HVP inner gradients both use create_graph=True and are released
    chunk by chunk. Ordinary outer F/G gradients use create_graph=False. Only
    LoRA fast parameters are replaced, while non-inner SID parameters remain
    part of both direct differentiation and the inner cross-Hessian.
    """
    group = _group(model, process_group)

    def prepare():
        _validate_alpha_mode(alpha, mode, local_no_history_oi)
        if reduction not in {"all_reduce", "ordered"}:
            raise ValueError("reduction must be all_reduce or ordered")
        if isinstance(model, nn.parallel.DistributedDataParallel):
            raise ValueError("Pass an ordinary replicated model, not DistributedDataParallel")
        parameters, buffers, trainable, inner, devices = _prepare(model, alpha)
        if any(parameters[name].device != group.device for name in trainable):
            raise ValueError("Trainable tensors must be on this rank's collective device")
        oi = _local_chunks(local_oi, "local_oi")
        ni = _local_chunks(local_ni, "local_ni")
        empty = () if local_no_history_oi is None else _local_chunks(local_no_history_oi, "local_no_history_oi")
        return parameters, buffers, trainable, inner, devices, oi, ni, empty

    with _guard_identity(model, check_identity):
        parameters, buffers, trainable, inner, devices, oi, ni, empty = _collectively(group, "setup", prepare)
        signature = (
            mode, float(alpha), bool(check_identity), bool(check_recompute), reduction,
            tuple((name, tuple(parameters[name].shape), str(parameters[name].dtype)) for name in trainable),
            inner,
        )
        if any(other != signature for other in group.gather(signature)):
            raise ValueError("Ranks disagree on mode, alpha, checks, or trainable parameter names/shapes/dtypes")
        oi, ni, empty = _normalize_roles(group, oi, ni, empty, need_no_history=mode == "meta_no_history" and alpha > 0)
        view = FunctionalModel(model, parameters, buffers)
        if reduction == "ordered":
            roles = {"oi": oi, "ni": ni}
            if mode == "meta_no_history" and alpha > 0:
                roles["no_history_oi"] = empty
            plans = _ordered_plans(group, roles, global_positions)
            return _compute_ordered(group, view, trainable, inner, devices, oi, ni, empty,
                                    plans, alpha, mode, check_recompute)

        def ordinary_passes():
            f, df, f_rng = _local_gradient_pass(oi, view, trainable, devices)
            g, dg, g_rng = _local_gradient_pass(ni, view, trainable, devices)
            return f, df, f_rng, g, dg, g_rng

        f, df, f_rng, g, dg, g_rng = _collectively(group, "ordinary outer gradients", ordinary_passes)
        if alpha == 0:
            gradients = {name: .5 * df[name] + .5 * dg[name] for name in trainable}
            _reduce_gradients(group, gradients)
            _collectively(group, "global gradient validation", lambda: _finite_gradients(gradients))
            totals = group.sum(torch.stack((f, g)))
            f_global, g_global = totals.unbind()
            zero = f_global.new_zeros(())
            return MetaGradients(
                .5 * f_global + .5 * g_global, f_global, g_global, g_global,
                f_global, inner, gradients, zero, zero,
            )
        direct = {name: .5 * df[name] + .25 * dg[name] for name in trainable}
        del df, dg
        inner_chunks = empty if mode == "meta_no_history" else oi
        inner_f, local_u, inner_rng = _collectively(
            group, "initial differentiable inner gradient",
            lambda: _local_gradient_pass(
                inner_chunks, view, inner, devices, create_graph=True,
                replay=None if mode == "meta_no_history" else f_rng,
            ),
        )
        # Keep the local initial gradient for the strict per-shard replay check.
        # The global u, not an average of local virtual steps, defines phi'.
        u = _reduce_gradients(group, {name: value.clone() for name, value in local_u.items()})
        fast = dict(parameters)
        fast.update({name: (parameters[name].detach() - alpha * u[name]).requires_grad_(True) for name in inner})
        fast_g, fast_direct, _ = _collectively(
            group, "adapted query gradient",
            lambda: _local_gradient_pass(ni, FunctionalModel(model, fast, buffers), trainable, devices, replay=g_rng),
        )
        for name in trainable:
            direct[name].add_(fast_direct[name], alpha=.25)
        # v must be global BEFORE every rank computes its local cross-Hessian.
        v = _reduce_gradients(group, {name: fast_direct[name].clone() for name in inner})
        del fast_direct, fast

        def recompute():
            repeated_f, repeated_u, hessian_product = _local_hvp(
                inner_chunks, view, trainable, inner, inner_rng, v,
            )
            if check_recompute:
                _check_recomputed(inner_f, repeated_f, local_u, repeated_u)
            return hessian_product

        h = _collectively(group, "inner HVP recomputation", recompute)
        _reduce_gradients(group, h)
        _reduce_gradients(group, direct)
        gradients = {name: direct[name] - .25 * alpha * h[name] for name in trainable}

        _collectively(group, "global gradient validation", lambda: _finite_gradients(gradients))
        f_global, g_global, fast_global, inner_global = group.sum(torch.stack((f, g, fast_g, inner_f))).unbind()
        return MetaGradients(
            .5 * f_global + .25 * g_global + .25 * fast_global,
            f_global, g_global, fast_global, inner_global, inner,
            gradients, _norm(u, f_global), _norm(h, f_global),
        )
