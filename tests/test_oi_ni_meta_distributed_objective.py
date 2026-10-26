from __future__ import annotations

from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("CPU distributed parity requires Gloo", allow_module_level=True)

from torch import nn
from torch.nn import functional as F

from new_interest.training.oi_ni_meta_distributed_objective import compute_distributed_meta_gradients
from new_interest.training.oi_ni_meta_objective import (
    FunctionalModel, LossChunk, build_meta_objective, compute_meta_gradients,
)


class TinyModel(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.base = nn.Parameter(torch.randn(3, 3, generator=generator, dtype=torch.double), requires_grad=False)
        self.lora_A = nn.Parameter(.3 * torch.randn(2, 3, generator=generator, dtype=torch.double))
        self.lora_B = nn.Parameter(.3 * torch.randn(3, 2, generator=generator, dtype=torch.double))
        self.sid_rows = nn.Parameter(.5 * torch.randn(4, 3, generator=generator, dtype=torch.double))
        self.sid_bias = nn.Parameter(.2 * torch.randn(4, generator=generator, dtype=torch.double))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("offset", torch.zeros((), dtype=torch.double))

    def forward(self, x):
        hidden = torch.tanh(F.linear(x, self.base + self.lora_B @ self.lora_A))
        hidden = self.dropout(hidden + .2 * torch.sin(hidden))
        logits = F.linear(hidden, self.sid_rows)
        return logits + .1 * logits.square() + self.sid_bias + self.offset


def _loss(x, y):
    return lambda view: F.cross_entropy(view(x), y)


def _raw_roles(dtype=torch.double):
    generator = torch.Generator().manual_seed(39)
    roles = []
    for sizes in ([1, 2, 1, 3, 2, 1, 4], [2, 1, 3]):
        chunks = []
        for size in sizes:
            x = torch.randn(size, 3, generator=generator, dtype=torch.double).to(dtype)
            y = torch.randint(4, (size,), generator=generator)
            chunks.append(LossChunk(_loss(x, y), size))
        roles.append(chunks)
    empty = [LossChunk(_loss(torch.full((int(chunk.weight), 3), .3, dtype=dtype),
                                  torch.full((int(chunk.weight),), index % 4, dtype=torch.long)), chunk.weight)
             for index, chunk in enumerate(roles[0])]
    return roles[0], roles[1], empty


def _bind_rng(function, before, after):
    def bound(view):
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(before)
            result = function(view)
            assert torch.equal(torch.random.get_rng_state(), after)
            return result
    return bound


def _plan_original_serial_rng(model, roles, seed):
    """Independent real-forward planner preserving the old global chunk stream.

    The distributed module must work with explicit states from this planner;
    assigning each rank the same starting seed would fail dropout parity.
    """
    view = FunctionalModel(model, dict(model.named_parameters()), dict(model.named_buffers()))
    planned = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        for role in roles:
            chunks = []
            for chunk in role:
                before = torch.random.get_rng_state().clone()
                chunk.loss_fn(view)
                after = torch.random.get_rng_state().clone()
                chunks.append(LossChunk(_bind_rng(chunk.loss_fn, before, after), chunk.weight))
            planned.append(chunks)
    return planned


def _named(model):
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def _assert_gradients(left, right, rtol=2e-10, atol=2e-11):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=rtol, atol=atol,
                                   msg=lambda text: f"{name}: {text}")


def _manual_value(model, roles, alpha, *, first_order=False):
    parameters = dict(model.named_parameters())
    view = FunctionalModel(model, parameters, dict(model.named_buffers()))

    def average(chunks, current_view):
        denominator = sum(chunk.weight for chunk in chunks)
        return sum(chunk.weight / denominator * chunk.loss_fn(current_view) for chunk in chunks)

    f, g = average(roles[0], view), average(roles[1], view)
    inner = [(name, value) for name, value in parameters.items() if "lora_A" in name or "lora_B" in name]
    gradients = torch.autograd.grad(f, [value for _, value in inner], create_graph=True)
    fast = dict(parameters)
    for (name, value), gradient in zip(inner, gradients):
        fast[name] = value - alpha * (gradient.detach() if first_order else gradient)
    return .5 * f + .25 * g + .25 * average(roles[1], FunctionalModel(model, fast, dict(model.named_buffers())))


def _worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=40))
    try:
        cases = [
            ("mixed", 0., .25, 0, torch.double),
            ("meta", .23, 0., 0, torch.double),
            ("meta", .23, .25, 0, torch.double),
            ("meta_no_history", .23, .25, 0, torch.double),
            ("meta", .23, .25, 1, torch.double),
        ]
        cases = [(reduction, *case) for reduction in ("all_reduce", "ordered") for case in cases]
        cases += [("ordered", "meta", .23, .25, 0, torch.float32),
                  ("ordered", "meta_no_history", .23, .25, 1, torch.float32)]
        for reduction, mode, alpha, dropout, partition, dtype in cases:
            model = TinyModel(dropout).to(dtype).train()
            original = {name: value.detach().clone() for name, value in model.state_dict().items()}
            identities = {name: id(value) for name, value in model.named_parameters()}
            raw = _raw_roles(dtype)
            seed = 42 + 104729 * 3
            planned = _plan_original_serial_rng(model, raw, seed)

            def local(role):
                # First partition has unequal shards and rank 3 completely
                # empty; the alternate partition changes every chunk owner.
                return [chunk for index, chunk in enumerate(role)
                        if (index % 3 if not partition else (index + 1) % world_size) == rank]

            def positions(role):
                return [index for index in range(len(role))
                        if (index % 3 if not partition else (index + 1) % world_size) == rank]

            kwargs = {"local_no_history_oi": local(planned[2])} if mode == "meta_no_history" else {}
            if reduction == "ordered":
                kwargs["global_positions"] = {"oi": positions(planned[0]), "ni": positions(planned[1]),
                                               "no_history_oi": positions(planned[2])}
            torch.manual_seed(9000 + rank)  # intentionally different local ambient streams
            result = compute_distributed_meta_gradients(
                model, local(planned[0]), local(planned[1]), alpha=alpha, mode=mode, reduction=reduction, **kwargs,
            )
            assert all(not value.requires_grad for value in result.gradients.values())
            flat = torch.cat([value.flatten() for value in result.gradients.values()])
            peers = [torch.empty_like(flat) for _ in range(world_size)]
            dist.all_gather(peers, flat)
            assert all(torch.equal(flat, peer) for peer in peers)
            for name, value in model.state_dict().items():
                assert torch.equal(value, original[name])
            for name, value in model.named_parameters():
                assert id(value) == identities[name] and value.grad is None and value.grad_fn is None
            if rank == 0:
                serial_kwargs = {"no_history_oi": raw[2]} if mode == "meta_no_history" else {}
                torch.manual_seed(seed)
                serial = compute_meta_gradients(model, raw[0], raw[1], alpha=alpha, mode=mode, **serial_kwargs)
                if reduction == "ordered":
                    _assert_gradients(result.gradients, serial.gradients, rtol=0, atol=0)
                else:
                    _assert_gradients(result.gradients, serial.gradients)
                for name in ("loss", "oi_loss", "ni_loss", "ni_fast_loss", "inner_loss", "inner_grad_norm", "hvp_norm"):
                    torch.testing.assert_close(getattr(result, name), getattr(serial, name),
                                               rtol=0 if reduction == "ordered" else 2e-10,
                                               atol=0 if reduction == "ordered" else 2e-11)
                torch.manual_seed(seed)
                unrolled = build_meta_objective(model, raw[0], raw[1], alpha=alpha, mode=mode, **serial_kwargs)
                parameters = _named(model)
                full_gradient = dict(zip(parameters, torch.autograd.grad(unrolled.loss, tuple(parameters.values()))))
                _assert_gradients(result.gradients, full_gradient,
                                  rtol=3e-6 if dtype == torch.float32 else 2e-10,
                                  atol=3e-7 if dtype == torch.float32 else 2e-11)
                if mode == "meta" and dropout == 0:
                    # A SID gradient differs from FOMAML: the global cross-
                    # Hessian was retained, rather than just LoRA-only HVP.
                    first_order = _manual_value(model, planned, alpha, first_order=True)
                    fo = dict(zip(parameters, torch.autograd.grad(first_order, tuple(parameters.values()))))
                    assert (result.gradients["sid_rows"] - fo["sid_rows"]).norm().item() > 1e-5
                    epsilon = 1e-5
                    for name in ("lora_A", "lora_B", "sid_rows", "sid_bias"):
                        parameter = parameters[name]
                        old = parameter.detach().flatten()[0].item()
                        with torch.no_grad():
                            parameter.view(-1)[0] = old + epsilon
                        plus = _manual_value(model, planned, alpha).item()
                        with torch.no_grad():
                            parameter.view(-1)[0] = old - epsilon
                        minus = _manual_value(model, planned, alpha).item()
                        with torch.no_grad():
                            parameter.view(-1)[0] = old
                        torch.testing.assert_close(result.gradients[name].flatten()[0],
                                                   torch.tensor((plus - minus) / (2 * epsilon), dtype=torch.double),
                                                   rtol=3e-6, atol=3e-9)
            dist.barrier()

        # A failure on one shard must reach all ranks before the next reduce.
        model = TinyModel()
        roles = _raw_roles()
        oi = [roles[0][rank]]
        ni = [roles[1][rank % 3]]
        if rank == 1:
            oi = [LossChunk(lambda _: torch.tensor(float("nan"), dtype=torch.double))]
        with pytest.raises(RuntimeError, match="rank 1.*non-finite"):
            compute_distributed_meta_gradients(model, oi, ni, alpha=.1)
        with pytest.raises(ValueError, match="Global ni"):
            compute_distributed_meta_gradients(model, [roles[0][rank]], [], alpha=.1)
        with pytest.raises(ValueError, match="positions must cover"):
            compute_distributed_meta_gradients(
                model, [roles[0][rank]], [roles[1][rank % 3]], alpha=.1, reduction="ordered",
                global_positions={"oi": [0], "ni": [rank]},
            )
    finally:
        dist.destroy_process_group()


def test_four_rank_gloo_matches_serial_full_unroll_with_unequal_and_empty_shards(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(4, str(tmp_path / "gloo-init")), nprocs=4, join=True)


def _bf16_qwen_worker(rank, world_size, rendezvous):
    # Reuse the already audited real-model fixture, including PEFT's tied SID
    # registration fix; no production/test fixture file is changed here.
    from test_oi_ni_meta_objective import tiny_tied_qwen35

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=50))
    try:
        model = tiny_tied_qwen35(base_dtype=torch.bfloat16, dropout=.05).train()
        roles = []
        for lengths in ([7, 9, 11, 33, 36, 8, 10], [6, 7, 10, 35, 13]):
            role = []
            for index, length in enumerate(lengths):
                x = (torch.arange(length) + index).remainder(35).add(1).unsqueeze(0)
                role.append(LossChunk(_loss(x, torch.tensor([index % 4])), 1))
            roles.append(role)
        roles.append([LossChunk(_loss(torch.tensor([[2, 3, 4]]), torch.tensor([index % 4])), 1)
                      for index in range(len(roles[0]))])
        for mode in ("meta", "meta_no_history"):
            seed = 42 + 104729 * 3
            planned = _plan_original_serial_rng(model, roles, seed)
            positions = {name: list(range(rank, len(role), world_size))
                         for name, role in zip(("oi", "ni", "no_history_oi"), roles)}
            local = [[chunk for index, chunk in enumerate(role) if index % world_size == rank] for role in planned]
            kwargs = {"local_no_history_oi": local[2]} if mode == "meta_no_history" else {}
            torch.manual_seed(1000 + rank)
            result = compute_distributed_meta_gradients(
                model, local[0], local[1], alpha=.001, mode=mode, reduction="ordered",
                global_positions=positions, **kwargs,
            )
            if rank == 0:
                torch.manual_seed(seed)
                serial = compute_meta_gradients(
                    model, roles[0], roles[1], alpha=.001, mode=mode,
                    **({"no_history_oi": roles[2]} if mode == "meta_no_history" else {}),
                )
                _assert_gradients(result.gradients, serial.gradients, rtol=0, atol=0)
                for name in ("loss", "oi_loss", "ni_loss", "ni_fast_loss", "inner_loss", "inner_grad_norm", "hvp_norm"):
                    torch.testing.assert_close(getattr(result, name), getattr(serial, name), rtol=0, atol=0)
                assert result.hvp_norm.item() > 0
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ordered_four_rank_real_bf16_qwen_matches_original_serial_bitwise(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if transformers.__version__ != "5.14.1":
        pytest.skip("Qwen3.5 exact backend is pinned to Transformers 5.14.1")
    torch.multiprocessing.spawn(_bf16_qwen_worker, args=(4, str(tmp_path / "gloo-qwen")), nprocs=4, join=True)


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("reduction", ["all_reduce", "ordered"])
def test_uninitialized_world_one_matches_serial_and_global_weight_scaling(one_thread, reduction):
    model = TinyModel()
    oi, ni, _ = _raw_roles()
    result = compute_distributed_meta_gradients(model, oi, ni, alpha=.2, reduction=reduction)
    serial = compute_meta_gradients(model, oi, ni, alpha=.2)
    _assert_gradients(result.gradients, serial.gradients)
    scaled = compute_distributed_meta_gradients(
        model, [LossChunk(chunk.loss_fn, chunk.weight / 14) for chunk in oi],
        [LossChunk(chunk.loss_fn, chunk.weight / 6) for chunk in ni], alpha=.2, reduction=reduction,
    )
    _assert_gradients(result.gradients, scaled.gradients)


def test_world_one_rejects_empty_global_role_and_invalid_weights(one_thread):
    model = TinyModel()
    oi, ni, _ = _raw_roles()
    with pytest.raises(ValueError, match="Global oi"):
        compute_distributed_meta_gradients(model, [], ni, alpha=.1)
    with pytest.raises(ValueError, match="nonnegative"):
        compute_distributed_meta_gradients(model, [LossChunk(oi[0].loss_fn, -1)], ni, alpha=.1)
