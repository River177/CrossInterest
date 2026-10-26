"""Exact OI -> NI virtual-step objectives, without mutating a live model.

``LossChunk`` callables receive a ``FunctionalModel`` and return a scalar mean
loss. Chunk weights are normalized within each objective: use sample/token
counts, rather than equal weights, when chunk means have unequal denominators.
Callables must be pure, except for PyTorch CPU/CUDA randomness (which is replayed
for repeated evaluations). Model buffers are copied for every functional call.

The memory-bounded implementation differentiates the *same* one-step objective
as :func:`build_meta_objective`. Write theta=(phi, psi), with phi the trainable
LoRA parameters, and phi'=phi-alpha*d_phi F_inner. For v=d_phi' G(phi', psi),

    d_theta J = .5*d_theta F_real + .25*d_theta G_real
                + .25*d_(phi',psi) G_adapted
                - .25*alpha*d_theta <d_phi F_inner, stopgrad(v)>.

The last term is recomputed one inner chunk at a time. Differentiating it with
respect to *all* trainable parameters retains the cross-Hessian for non-inner
parameters such as SID rows. No FOMAML approximation, finite differences,
optimizer step, parameter assignment, or accumulation into ``parameter.grad``
is performed. The returned gradients are ready for the caller's outer update.

The initial inner gradient and the recomputed inner gradient both use
``create_graph=True``. Low-precision kernels can take numerically different
backward paths when graph construction is disabled, even if the adapter
parameters themselves are FP32. Each initial inner chunk's graph is discarded
immediately after detaching its gradient; ordinary outer F/G gradients retain
their first-order path.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
from typing import Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call


class FunctionalModel:
    """Call a module with explicit parameters and fresh, isolated buffers.

    ``parameters`` is exposed for loss terms that directly regularize a named
    parameter. Do not mutate its tensors or the underlying module in a loss.
    """

    def __init__(
        self,
        module: nn.Module,
        parameters: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
    ) -> None:
        self.module = module
        self.parameters = dict(parameters)
        self._buffers = buffers

    def __call__(self, *args: object, **kwargs: object) -> object:
        buffers = {name: value.clone() for name, value in self._buffers.items()}
        return functional_call(
            self.module,
            (self.parameters, buffers),
            args,
            kwargs,
            tie_weights=True,
            strict=True,
        )


LossFn = Callable[[FunctionalModel], Tensor]


@dataclass(frozen=True)
class LossChunk:
    loss_fn: LossFn
    weight: float = 1.0


LossSpec = LossFn | Sequence[LossChunk]


@dataclass(frozen=True)
class MetaObjective:
    loss: Tensor
    oi_loss: Tensor
    ni_loss: Tensor
    adapted_ni_loss: Tensor
    inner_loss: Tensor
    inner_parameter_names: tuple[str, ...]

    @property
    def ni_fast_loss(self) -> Tensor:
        return self.adapted_ni_loss


@dataclass(frozen=True)
class MetaGradients(MetaObjective):
    gradients: Mapping[str, Tensor]
    inner_grad_norm: Tensor
    hvp_norm: Tensor


@dataclass(frozen=True)
class _RNGState:
    cpu: Tensor
    cuda: Mapping[int, Tensor]


def _capture_rng(cuda_devices: Sequence[int]) -> _RNGState:
    return _RNGState(
        cpu=torch.random.get_rng_state().clone(),
        cuda={device: torch.cuda.get_rng_state(device).clone() for device in cuda_devices},
    )


@contextmanager
def _replay_rng(state: _RNGState) -> Iterator[None]:
    # fork_rng restores the *current* outer state after recomputation. It does
    # not rewind the training stream to the earlier inner forward pass.
    with torch.random.fork_rng(devices=list(state.cuda)):
        torch.random.set_rng_state(state.cpu)
        for device, value in state.cuda.items():
            torch.cuda.set_rng_state(value, device)
        yield


def _tensor_identity(value: Tensor) -> tuple[object, ...]:
    return (
        id(value), value.data_ptr(), value._version, value.shape,
        value.dtype, value.device, value.requires_grad,
    )


def _model_identity(module: nn.Module) -> tuple[object, ...]:
    return (
        tuple((name, _tensor_identity(value)) for name, value in module.named_parameters()),
        tuple((name, _tensor_identity(value)) for name, value in module.named_buffers()),
        tuple(
            (name, None if value.grad is None else _tensor_identity(value.grad))
            for name, value in module.named_parameters()
        ),
        tuple((name, child.training) for name, child in module.named_modules()),
    )


@contextmanager
def _guard_identity(module: nn.Module, enabled: bool) -> Iterator[None]:
    before = _model_identity(module) if enabled else None
    try:
        yield
    finally:
        if enabled and _model_identity(module) != before:
            raise RuntimeError(
                "The meta objective changed model parameter/buffer identity, "
                "storage, version, .grad, or training mode. Loss callables must be pure."
            )


def _chunks(spec: LossSpec, label: str) -> tuple[LossChunk, ...]:
    supplied = (LossChunk(spec),) if callable(spec) else tuple(spec)
    if not supplied:
        raise ValueError(f"{label} must contain at least one loss chunk")
    if not all(isinstance(chunk, LossChunk) and callable(chunk.loss_fn) for chunk in supplied):
        raise TypeError(f"{label} must be a callable or a sequence of LossChunk objects")
    weights = [float(chunk.weight) for chunk in supplied]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError(f"{label} chunk weights must be finite and nonnegative")
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{label} chunk weights must have a positive finite sum")
    return tuple(
        LossChunk(chunk.loss_fn, weight / total)
        for chunk, weight in zip(supplied, weights)
        if weight > 0
    )


def _validate_alpha_mode(alpha: float, mode: str, no_history_oi: LossSpec | None) -> None:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if mode not in {"mixed", "meta", "meta_no_history"}:
        raise ValueError("mode must be mixed, meta, or meta_no_history")
    if mode == "mixed" and alpha != 0:
        raise ValueError("mixed mode requires alpha=0")
    if mode == "meta_no_history" and no_history_oi is None:
        raise ValueError("meta_no_history requires no_history_oi")
    if mode != "meta_no_history" and no_history_oi is not None:
        raise ValueError("no_history_oi is only accepted in meta_no_history mode")


def _prepare(
    module: nn.Module, alpha: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor], tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    parameters = dict(module.named_parameters())
    trainable = tuple(name for name, value in parameters.items() if value.requires_grad)
    if not trainable:
        raise ValueError("model has no trainable parameters")
    inner = tuple(name for name in trainable if "lora_A" in name or "lora_B" in name)
    if alpha > 0 and not inner:
        raise ValueError("a nonzero virtual step requires trainable lora_A/lora_B parameters")
    buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    cuda_devices = tuple(sorted({
        value.device.index for value in (*parameters.values(), *buffers.values())
        if value.device.type == "cuda" and value.device.index is not None
    }))
    return parameters, buffers, trainable, inner, cuda_devices


def _loss(chunk: LossChunk, view: FunctionalModel) -> Tensor:
    loss = chunk.loss_fn(view)
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError("each loss callable must return a scalar floating-point Tensor")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("loss callable returned a non-finite loss")
    return loss


def _grad(
    loss: Tensor, targets: Sequence[Tensor], *, create_graph: bool = False,
) -> tuple[Tensor, ...]:
    if not targets:
        return ()
    if not loss.requires_grad:
        return tuple(torch.zeros_like(value) for value in targets)
    values = torch.autograd.grad(
        loss, tuple(targets), allow_unused=True, create_graph=create_graph,
        retain_graph=create_graph,
    )
    return tuple(torch.zeros_like(target) if grad is None else grad for target, grad in zip(targets, values))


def _norm(values: Mapping[str, Tensor], reference: Tensor) -> Tensor:
    # Accumulate the diagnostic in float64 to avoid overflow in low precision.
    return sum(
        (value.detach().double().square().sum().to(reference.device) for value in values.values()),
        reference.new_zeros((), dtype=torch.float64),
    ).sqrt()


def _graph_loss(
    chunks: Sequence[LossChunk], view: FunctionalModel, cuda_devices: Sequence[int],
    *, replay: Sequence[_RNGState] | None = None,
) -> tuple[Tensor, tuple[_RNGState, ...]]:
    pieces: list[Tensor] = []
    states: list[_RNGState] = []
    for index, chunk in enumerate(chunks):
        state = _capture_rng(cuda_devices) if replay is None else replay[index]
        states.append(state)
        with nullcontext() if replay is None else _replay_rng(state):
            pieces.append(chunk.weight * _loss(chunk, view))
    return sum(pieces[1:], pieces[0]), tuple(states)


def _chunk_gradients(
    chunks: Sequence[LossChunk], view: FunctionalModel, names: Sequence[str],
    cuda_devices: Sequence[int], *, replay: Sequence[_RNGState] | None = None,
    create_graph: bool = False,
) -> tuple[Tensor, dict[str, Tensor], tuple[_RNGState, ...]]:
    targets = tuple(view.parameters[name] for name in names)
    gradients = {name: torch.zeros_like(value) for name, value in zip(names, targets)}
    total: Tensor | None = None
    states: list[_RNGState] = []
    for index, chunk in enumerate(chunks):
        state = _capture_rng(cuda_devices) if replay is None else replay[index]
        states.append(state)
        with nullcontext() if replay is None else _replay_rng(state):
            loss = _loss(chunk, view)
            values = _grad(loss, targets, create_graph=create_graph)
        value = chunk.weight * loss.detach()
        total = value if total is None else total + value
        for name, gradient in zip(names, values):
            gradients[name].add_(gradient.detach(), alpha=chunk.weight)
        # The loop variable otherwise retains the final differentiable gradient
        # and its entire inner graph into the next chunk, despite deleting the
        # tuple. Keep activation memory bounded by one inner chunk.
        del loss, values, gradient
    assert total is not None
    return total, gradients, tuple(states)


def build_meta_objective(
    model: nn.Module,
    oi: LossSpec,
    ni: LossSpec,
    *,
    alpha: float,
    mode: str = "meta",
    no_history_oi: LossSpec | None = None,
    check_identity: bool = True,
) -> MetaObjective:
    """Build the exact differentiable graph (reference or small-episode path).

    This retains all chunk graphs. Use ``compute_meta_gradients`` to bound
    activation memory by one chunk while retaining the exact second derivative.
    At alpha=0, all modes compute precisely .5*F_real + .5*G_real and do not
    evaluate the no-history callable. No backward/optimizer operation is done.
    """
    _validate_alpha_mode(alpha, mode, no_history_oi)
    oi_chunks, ni_chunks = _chunks(oi, "oi"), _chunks(ni, "ni")
    with _guard_identity(model, check_identity):
        parameters, buffers, _, inner, devices = _prepare(model, alpha)
        view = FunctionalModel(model, parameters, buffers)
        f, _ = _graph_loss(oi_chunks, view, devices)
        g, ni_rng = _graph_loss(ni_chunks, view, devices)
        if alpha == 0:
            return MetaObjective(.5 * f + .5 * g, f, g, g, f, inner)
        if mode == "meta_no_history":
            assert no_history_oi is not None
            inner_f, _ = _graph_loss(_chunks(no_history_oi, "no_history_oi"), view, devices)
        else:
            inner_f = f
        inner_grad = _grad(inner_f, [parameters[name] for name in inner], create_graph=True)
        fast = dict(parameters)
        fast.update({name: parameters[name] - alpha * grad for name, grad in zip(inner, inner_grad)})
        adapted_g, _ = _graph_loss(
            ni_chunks, FunctionalModel(model, fast, buffers), devices, replay=ni_rng,
        )
        return MetaObjective(.5 * f + .25 * g + .25 * adapted_g, f, g, adapted_g, inner_f, inner)


def compute_meta_gradients(
    model: nn.Module,
    oi: LossSpec,
    ni: LossSpec,
    *,
    alpha: float,
    mode: str = "meta",
    no_history_oi: LossSpec | None = None,
    check_identity: bool = True,
    check_recompute: bool = True,
) -> MetaGradients:
    """Return exact named outer gradients, using chunkwise HVP recomputation.

    Returned tensors are detached; this function never changes model values,
    ``.grad`` fields, training mode, or an optimizer. Each forward's PyTorch RNG
    state is replayed for adapted NI and inner recomputation. Other randomness
    (Python/NumPy), stateful data iterators, and external mutable state are not
    allowed inside a loss callable. Initial and recomputed inner gradients use
    the same create_graph=True path, independently of the ordinary outer F
    gradient; initial inner graphs are detached and released chunk by chunk.
    ``check_recompute`` detects a changed inner loss or gradient, rather than
    silently returning a different objective.
    """
    _validate_alpha_mode(alpha, mode, no_history_oi)
    oi_chunks, ni_chunks = _chunks(oi, "oi"), _chunks(ni, "ni")
    with _guard_identity(model, check_identity):
        parameters, buffers, trainable, inner, devices = _prepare(model, alpha)
        view = FunctionalModel(model, parameters, buffers)
        f, f_grads, oi_rng = _chunk_gradients(oi_chunks, view, trainable, devices)
        g, g_grads, ni_rng = _chunk_gradients(ni_chunks, view, trainable, devices)
        if alpha == 0:
            gradients = {name: .5 * f_grads[name] + .5 * g_grads[name] for name in trainable}
            zero = f.new_zeros((), dtype=torch.float64)
            return MetaGradients(.5 * f + .5 * g, f, g, g, f, inner, gradients, zero, zero)
        gradients = {name: .5 * f_grads[name] + .25 * g_grads[name] for name in trainable}
        if mode == "meta_no_history":
            assert no_history_oi is not None
            inner_chunks = _chunks(no_history_oi, "no_history_oi")
            inner_f, inner_grads, inner_rng = _chunk_gradients(
                inner_chunks, view, inner, devices, create_graph=True,
            )
        else:
            inner_chunks = oi_chunks
            # Reusing f_grads here is wrong for BF16 bodies: those gradients
            # were computed with create_graph=False, while the HVP below must
            # differentiate the create_graph=True backward. Replay the same F
            # realization but calculate its inner gradient through that path.
            inner_f, inner_grads, inner_rng = _chunk_gradients(
                inner_chunks, view, inner, devices, replay=oi_rng, create_graph=True,
            )
        del f_grads, g_grads
        fast = dict(parameters)
        fast.update({
            name: (parameters[name].detach() - alpha * inner_grads[name]).requires_grad_(True)
            for name in inner
        })
        adapted_g, direct, _ = _chunk_gradients(
            ni_chunks, FunctionalModel(model, fast, buffers), trainable, devices, replay=ni_rng,
        )
        for name in trainable:
            gradients[name].add_(direct[name], alpha=.25)
        # v is detached, making this a Hessian-vector product, not a derivative
        # of v. The direct derivative of G has already been accumulated above.
        v = {name: direct[name] for name in inner}
        del direct, fast
        original_targets = tuple(parameters[name] for name in trainable)
        inner_targets = tuple(parameters[name] for name in inner)
        recomputed_f: Tensor | None = None
        recomputed_inner = {name: torch.zeros_like(parameters[name]) for name in inner}
        hvp_sum = {name: torch.zeros_like(parameters[name]) for name in trainable}
        for chunk, state in zip(inner_chunks, inner_rng):
            with _replay_rng(state):
                loss = _loss(chunk, view)
                first = _grad(loss, inner_targets, create_graph=True)
                product = sum(
                    ((grad * v[name]).sum().to(loss.device) for name, grad in zip(inner, first)),
                    loss.new_zeros(()),
                )
                hvp = _grad(product, original_targets)
            value = chunk.weight * loss.detach()
            recomputed_f = value if recomputed_f is None else recomputed_f + value
            for name, grad in zip(inner, first):
                recomputed_inner[name].add_(grad.detach(), alpha=chunk.weight)
            for name, value in zip(trainable, hvp):
                hvp_sum[name].add_(value.detach(), alpha=chunk.weight)
            del loss, first, product, hvp, grad
        if check_recompute:
            assert recomputed_f is not None
            for name, expected, actual in (
                ("inner loss", inner_f, recomputed_f),
                *((name, inner_grads[name], recomputed_inner[name]) for name in inner),
            ):
                # Forward values should match exactly; allow only a small
                # dtype-relative rounding tolerance in the accumulated grads.
                tolerance = 8 * torch.finfo(expected.dtype).eps
                if not torch.allclose(expected, actual, rtol=tolerance, atol=tolerance):
                    raise RuntimeError(
                        f"Inner recomputation changed {name}; loss callables must be repeatable "
                        "with PyTorch RNG replay. No approximate meta-gradient was returned."
                    )
        for name in trainable:
            gradients[name].add_(hvp_sum[name], alpha=-.25 * alpha)
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients.values()):
            raise FloatingPointError("exact meta objective produced a non-finite gradient")
        return MetaGradients(
            .5 * f + .25 * g + .25 * adapted_g, f, g, adapted_g, inner_f, inner, gradients,
            _norm(inner_grads, f), _norm(hvp_sum, f),
        )
