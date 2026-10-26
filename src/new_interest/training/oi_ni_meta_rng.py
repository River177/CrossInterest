"""Plan the original serial dropout stream for parallel episode execution.

Rank zero previews the actual F, G and optional empty-history inputs in their
original order, with model.train() and no_grad(). Workers replay each chunk's
CPU/CUDA starting state and verify its ending state. The preview overhead must
be included in training timings. No synthetic dropout or RNG-offset estimate is
used. A serial-versus-parallel gradient comparison is still required on CUDA.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, Callable, Hashable, Mapping, Sequence

import torch
from torch import nn

from new_interest.training.oi_ni_meta_objective import FunctionalModel, LossChunk


PLANNER_VERSION = "serial-real-forward-rng-v1"


def _device(model: nn.Module) -> torch.device:
    devices = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    if len(devices) != 1:
        raise ValueError("RNG planning requires one model device per process")
    device = next(iter(devices))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("RNG planning supports CPU and CUDA only")
    return device


def _capture(device: torch.device) -> dict[str, torch.Tensor | None]:
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def _equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return torch.equal(a["cpu"], b["cpu"]) and (
        a["cuda"] is None and b["cuda"] is None or
        a["cuda"] is not None and b["cuda"] is not None and torch.equal(a["cuda"], b["cuda"]))


@contextmanager
def _using(state: Mapping[str, Any], device: torch.device):
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.random.set_rng_state(state["cpu"])
        if device.type == "cuda":
            if state["cuda"] is None:
                raise ValueError("CUDA loss received a CPU-only RNG plan")
            torch.cuda.set_rng_state(state["cuda"], device)
        elif state["cuda"] is not None:
            raise ValueError("CPU loss received a CUDA RNG plan")
        yield


def plan_episode_rng(
    model: nn.Module,
    episode_seed: int,
    global_oi_chunks: Sequence[LossChunk],
    global_ni_chunks: Sequence[LossChunk],
    global_no_history_chunks: Sequence[LossChunk] | None = None,
    *,
    mode: str = "meta",
    cache_keys: Mapping[str, Sequence[Hashable]] | None = None,
    planning_mode: str = "real_preview",
) -> dict[str, Any]:
    """Return serial per-chunk start/end states, preserving the caller's RNG.

    ``cache_keys`` are optional layout metadata for callers; no states or
    stochastic operations are cached. The audited implementation accepts only
    real previews, so functional random operations are also captured.
    """
    if mode not in {"mixed", "meta", "meta_no_history"}:
        raise ValueError("Unknown meta mode")
    if planning_mode != "real_preview":
        raise ValueError("Only real-forward RNG preview is supported")
    if not model.training:
        raise ValueError("Plan the training stream with model.train()")
    roles = {"oi": tuple(global_oi_chunks), "ni": tuple(global_ni_chunks)}
    if not all(roles.values()):
        raise ValueError("Global OI and NI must both be nonempty")
    if mode == "meta_no_history":
        if not global_no_history_chunks:
            raise ValueError("No-history meta requires its global inner chunks")
        roles["no_history"] = tuple(global_no_history_chunks)
    device = _device(model)
    view = FunctionalModel(model, dict(model.named_parameters()),
                           {n: b.detach().clone() for n, b in model.named_buffers()})
    result: dict[str, Any] = {"oi": [], "ni": [], "no_history": []}
    digest = hashlib.sha256()
    outer = _capture(device)
    with _using(outer, device):
        torch.random.default_generator.manual_seed(int(episode_seed))
        if device.type == "cuda":
            torch.cuda.default_generators[device.index].manual_seed(int(episode_seed))
        for role, chunks in roles.items():
            keys = None if cache_keys is None else cache_keys.get(role)
            if keys is not None and len(keys) != len(chunks):
                raise ValueError("RNG cache keys must cover every global chunk")
            for chunk in chunks:
                before = _capture(device)
                with torch.no_grad():
                    loss = chunk.loss_fn(view)
                    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss).item():
                        raise ValueError("Preview must return a finite scalar loss")
                    del loss
                after = _capture(device)
                item = {"cpu_before": before["cpu"], "cuda_before": before["cuda"],
                        "cpu_after": after["cpu"], "cuda_after": after["cuda"]}
                result[role].append(item)
                for state in item.values():
                    if state is not None:
                        digest.update(state.numpy().tobytes())
    if not _equal(outer, _capture(device)):
        raise RuntimeError("RNG planner changed the caller's stream")
    result["audit"] = {"version": PLANNER_VERSION, "episode_seed": int(episode_seed),
                       "state_sha256": digest.hexdigest(), "planning_mode": planning_mode,
                       "preview_forward_count": sum(map(len, roles.values())), "caller_rng_preserved": True,
                       "role_chunk_counts": {role: len(chunks) for role, chunks in roles.items()},
                       "device_type": device.type}
    return result


def bind_planned_rng(loss_fn: Callable, state: Mapping[str, Any]) -> Callable:
    """Replay a planned original loss realization on the worker's local device."""
    def bound(view: FunctionalModel):
        devices = {value.device for value in view.parameters.values()}
        if len(devices) != 1:
            raise ValueError("A bound loss requires exactly one model device")
        device = next(iter(devices))
        before = {"cpu": state["cpu_before"], "cuda": state["cuda_before"]}
        expected = {"cpu": state["cpu_after"], "cuda": state["cuda_after"]}
        with _using(before, device):
            loss = loss_fn(view)
            if not _equal(_capture(device), expected):
                raise RuntimeError("Actual loss RNG consumption differs from the serial dropout plan")
        return loss
    return bound
