from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from new_interest.training.oi_ni_meta_objective import (
    LossChunk,
    build_meta_objective,
    compute_meta_gradients,
)


class NonlinearLoRAModel(nn.Module):
    def __init__(self, *, dropout: float = 0.0, touch_buffer: bool = False):
        super().__init__()
        generator = torch.Generator().manual_seed(81)
        self.base = nn.Parameter(torch.randn(3, 3, generator=generator, dtype=torch.double), requires_grad=False)
        self.lora_A = nn.Parameter(.3 * torch.randn(2, 3, generator=generator, dtype=torch.double))
        self.lora_B = nn.Parameter(.3 * torch.randn(3, 2, generator=generator, dtype=torch.double))
        self.sid_rows = nn.Parameter(.5 * torch.randn(4, 3, generator=generator, dtype=torch.double))
        self.sid_bias = nn.Parameter(.2 * torch.randn(4, generator=generator, dtype=torch.double))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.double))
        self.touch_buffer = touch_buffer

    def forward(self, x):
        if self.touch_buffer:
            self.forward_count.add_(1)
        hidden = torch.tanh(F.linear(x, self.base + self.lora_B @ self.lora_A))
        hidden = self.dropout(hidden + .2 * torch.sin(hidden))
        logits = F.linear(hidden, self.sid_rows)
        return logits + .1 * logits.square() + self.sid_bias


@pytest.fixture
def data():
    generator = torch.Generator().manual_seed(713)
    x_oi = torch.randn(6, 3, generator=generator, dtype=torch.double)
    x_ni = torch.randn(5, 3, generator=generator, dtype=torch.double) + .4
    y_oi = torch.tensor([0, 1, 2, 0, 3, 1])
    y_ni = torch.tensor([2, 0, 3, 1, 2])
    return x_oi, y_oi, x_ni, y_ni


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def loss_fn(x, y):
    return lambda view: F.cross_entropy(view(x), y)


def specs(data):
    x_oi, y_oi, x_ni, y_ni = data
    return loss_fn(x_oi, y_oi), loss_fn(x_ni, y_ni)


def named_trainable(model):
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def graph_gradients(model, result):
    parameters = named_trainable(model)
    return dict(zip(parameters, torch.autograd.grad(result.loss, tuple(parameters.values()))))


def assert_gradient_close(left, right, *, rtol=1e-10, atol=1e-11):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}")


def manual_objective(model, data, alpha, *, inner_x=None, first_order=False):
    """Independent, single-batch mathematical objective for tiny CPU tests."""
    x_oi, y_oi, x_ni, y_ni = data
    parameters = dict(model.named_parameters())
    f = F.cross_entropy(model(x_oi), y_oi)
    g = F.cross_entropy(model(x_ni), y_ni)
    inner_f = f if inner_x is None else F.cross_entropy(model(inner_x), y_oi)
    names = [name for name in parameters if "lora_A" in name or "lora_B" in name]
    inner = torch.autograd.grad(inner_f, [parameters[name] for name in names], create_graph=True)
    fast = dict(parameters)
    for name, gradient in zip(names, inner):
        fast[name] = parameters[name] - alpha * (gradient.detach() if first_order else gradient)
    logits = torch.func.functional_call(model, fast, (x_ni,))
    return .5 * f + .25 * g + .25 * F.cross_entropy(logits, y_ni)


def test_exact_recomputed_gradients_match_finite_differences(data):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    alpha = .23
    result = compute_meta_gradients(model, oi, ni, alpha=alpha)
    epsilon = 1e-5
    for name, parameter in named_trainable(model).items():
        numerical = torch.zeros_like(parameter)
        for index in range(parameter.numel()):
            value = parameter.detach().flatten()[index].item()
            with torch.no_grad():
                parameter.view(-1)[index] = value + epsilon
            plus = manual_objective(model, data, alpha).item()
            with torch.no_grad():
                parameter.view(-1)[index] = value - epsilon
            minus = manual_objective(model, data, alpha).item()
            with torch.no_grad():
                parameter.view(-1)[index] = value
            numerical.view(-1)[index] = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close(result.gradients[name], numerical, rtol=3e-7, atol=3e-9)


@pytest.mark.parametrize("mode", ["meta", "meta_no_history"])
def test_recompute_matches_full_second_order_graph(data, mode):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    kwargs = {}
    if mode == "meta_no_history":
        kwargs["no_history_oi"] = loss_fn(torch.full_like(data[0], .3), data[1])
    graph = build_meta_objective(model, oi, ni, alpha=.17, mode=mode, **kwargs)
    expected = graph_gradients(model, graph)
    result = compute_meta_gradients(model, oi, ni, alpha=.17, mode=mode, **kwargs)
    assert_gradient_close(result.gradients, expected)
    torch.testing.assert_close(result.loss, graph.loss.detach())
    assert result.inner_grad_norm.item() > 0
    assert result.hvp_norm.item() > 0
    assert result.ni_fast_loss is result.adapted_ni_loss
    assert all(not gradient.requires_grad for gradient in result.gradients.values())


@pytest.mark.parametrize("mode", ["mixed", "meta", "meta_no_history"])
def test_alpha_zero_is_exact_mixed_and_never_calls_no_history(data, mode):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    kwargs = {}
    if mode == "meta_no_history":
        def forbidden(_):
            raise AssertionError("alpha=0 must not run an inner-only callable")
        kwargs["no_history_oi"] = forbidden
    expected = .5 * F.cross_entropy(model(data[0]), data[1]) + .5 * F.cross_entropy(model(data[2]), data[3])
    parameters = named_trainable(model)
    expected_grads = dict(zip(parameters, torch.autograd.grad(expected, tuple(parameters.values()))))
    result = compute_meta_gradients(model, oi, ni, alpha=0, mode=mode, **kwargs)
    graph = build_meta_objective(model, oi, ni, alpha=0, mode=mode, **kwargs)
    torch.testing.assert_close(result.loss, expected.detach(), rtol=0, atol=0)
    torch.testing.assert_close(graph.loss, expected, rtol=0, atol=0)
    assert_gradient_close(result.gradients, expected_grads, rtol=0, atol=0)
    assert_gradient_close(graph_gradients(model, graph), expected_grads, rtol=0, atol=0)
    assert result.hvp_norm.item() == 0


def test_non_inner_sid_cross_hessian_is_not_dropped(data):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    result = compute_meta_gradients(model, oi, ni, alpha=.6)
    parameters = named_trainable(model)
    exact = manual_objective(model, data, .6)
    expected = dict(zip(parameters, torch.autograd.grad(exact, tuple(parameters.values()))))
    first_order = manual_objective(model, data, .6, first_order=True)
    fo = dict(zip(parameters, torch.autograd.grad(first_order, tuple(parameters.values()))))
    assert_gradient_close(result.gradients, expected)
    assert result.gradients["sid_rows"].norm().item() > .01
    assert (expected["sid_rows"] - fo["sid_rows"]).norm().item() > 1e-4
    assert (expected["sid_bias"] - fo["sid_bias"]).norm().item() > 1e-5
    assert result.inner_parameter_names == ("lora_A", "lora_B")


def test_no_history_changes_inner_only_and_keeps_real_outer_oi(data):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    no_history_x = torch.full_like(data[0], .3)
    result = compute_meta_gradients(
        model, oi, ni, alpha=.3, mode="meta_no_history",
        no_history_oi=loss_fn(no_history_x, data[1]),
    )
    normal = compute_meta_gradients(model, oi, ni, alpha=.3)
    torch.testing.assert_close(result.oi_loss, F.cross_entropy(model(data[0]), data[1]))
    torch.testing.assert_close(result.ni_loss, normal.ni_loss)
    torch.testing.assert_close(result.inner_loss, F.cross_entropy(model(no_history_x), data[1]))
    parameters = named_trainable(model)
    expected_loss = manual_objective(model, data, .3, inner_x=no_history_x)
    expected = dict(zip(parameters, torch.autograd.grad(expected_loss, tuple(parameters.values()))))
    assert_gradient_close(result.gradients, expected)
    assert (result.gradients["lora_A"] - normal.gradients["lora_A"]).norm().item() > 1e-4


def test_unequal_microchunks_equal_full_batch_exact_gradient(data):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    x_oi, y_oi, x_ni, y_ni = data
    oi_chunks = [LossChunk(loss_fn(x_oi[:2], y_oi[:2]), 2), LossChunk(loss_fn(x_oi[2:], y_oi[2:]), 4)]
    ni_chunks = [LossChunk(loss_fn(x_ni[:1], y_ni[:1]), 1), LossChunk(loss_fn(x_ni[1:], y_ni[1:]), 4)]
    full = compute_meta_gradients(model, oi, ni, alpha=.23)
    micro = compute_meta_gradients(model, oi_chunks, ni_chunks, alpha=.23)
    reference = build_meta_objective(model, oi_chunks, ni_chunks, alpha=.23)
    assert_gradient_close(micro.gradients, full.gradients)
    assert_gradient_close(micro.gradients, graph_gradients(model, reference))
    torch.testing.assert_close(micro.loss, full.loss)
    normalized = compute_meta_gradients(
        model, [LossChunk(chunk.loss_fn, chunk.weight / 6) for chunk in oi_chunks],
        [LossChunk(chunk.loss_fn, chunk.weight / 5) for chunk in ni_chunks], alpha=.23,
    )
    assert_gradient_close(normalized.gradients, micro.gradients, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["meta", "meta_no_history"])
def test_dropout_rng_replay_matches_exact_graph_and_preserves_stream(data, mode):
    model = NonlinearLoRAModel(dropout=.35)
    x_oi, y_oi, x_ni, y_ni = data
    oi = [LossChunk(loss_fn(x_oi[:2], y_oi[:2]), 2), LossChunk(loss_fn(x_oi[2:], y_oi[2:]), 4)]
    ni = [LossChunk(loss_fn(x_ni[:2], y_ni[:2]), 2), LossChunk(loss_fn(x_ni[2:], y_ni[2:]), 3)]
    kwargs = {}
    if mode == "meta_no_history":
        kwargs["no_history_oi"] = loss_fn(torch.full_like(x_oi, .3), y_oi)
    torch.manual_seed(294)
    reference = build_meta_objective(model, oi, ni, alpha=.2, mode=mode, **kwargs)
    expected = graph_gradients(model, reference)
    expected_state = torch.random.get_rng_state().clone()
    torch.manual_seed(294)
    result = compute_meta_gradients(model, oi, ni, alpha=.2, mode=mode, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), expected_state)
    assert_gradient_close(result.gradients, expected)
    torch.testing.assert_close(result.loss, reference.loss.detach())


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_no_parameter_grad_buffer_or_optimizer_mutation(data):
    model = NonlinearLoRAModel(touch_buffer=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    F.cross_entropy(model(data[0]), data[1]).backward()
    optimizer.step()
    parameters = dict(model.named_parameters())
    before_ids = {name: (id(value), value.data_ptr(), value._version) for name, value in parameters.items()}
    before_state = copy.deepcopy(model.state_dict())
    before_grads = {name: None if value.grad is None else value.grad.clone() for name, value in parameters.items()}
    before_grad_ids = {name: id(value.grad) for name, value in parameters.items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    oi, ni = specs(data)
    compute_meta_gradients(model, oi, ni, alpha=.3)
    reference = build_meta_objective(model, oi, ni, alpha=.3)
    graph_gradients(model, reference)
    _assert_nested_equal(model.state_dict(), before_state)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    for name, value in model.named_parameters():
        assert (id(value), value.data_ptr(), value._version) == before_ids[name]
        assert id(value.grad) == before_grad_ids[name]
        _assert_nested_equal(value.grad, before_grads[name])
    assert model.training


def test_repeated_objectives_and_external_optimizer_step_do_not_retain_fast_graph(data):
    model = NonlinearLoRAModel(touch_buffer=True)
    oi, ni = specs(data)
    optimizer = torch.optim.AdamW(named_trainable(model).values(), lr=.01)
    identities = {name: id(value) for name, value in model.named_parameters()}
    first = compute_meta_gradients(model, oi, ni, alpha=.3)
    second = compute_meta_gradients(model, oi, ni, alpha=.3)
    assert_gradient_close(first.gradients, second.gradients, rtol=0, atol=0)
    assert first.loss.item() == second.loss.item()
    for name, parameter in named_trainable(model).items():
        parameter.grad = second.gradients[name].clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # No retain_graph=True is needed across objective calls, and the third
    # result must use the new outer parameters rather than a stale fast tensor.
    third = compute_meta_gradients(model, oi, ni, alpha=.3)
    reference = build_meta_objective(model, oi, ni, alpha=.3)
    assert_gradient_close(third.gradients, graph_gradients(model, reference))
    assert third.loss.item() != first.loss.item()
    for name, parameter in model.named_parameters():
        assert id(parameter) == identities[name]
        assert parameter.grad is None
        assert parameter.grad_fn is None
        assert parameter.is_leaf
    assert model.forward_count.item() == 0


def tiny_tied_qwen35(*, base_dtype=torch.float32, dropout=0.0):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    if transformers.__version__ != "5.14.1":
        pytest.skip("Qwen3.5 exact backend is pinned to Transformers 5.14.1")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from new_interest.training.oi_ni_meta_backend import RestrictedSIDModel

    torch.manual_seed(71)
    config = Qwen3_5TextConfig(
        vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
        tie_word_embeddings=True, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [2, 1, 1]},
    )
    config._attn_implementation = "eager"
    causal_lm = peft.get_peft_model(Qwen3_5ForCausalLM(config).to(dtype=base_dtype, device="cpu"), peft.LoraConfig(
        r=2, lora_alpha=4, lora_dropout=dropout, target_modules="all-linear", task_type="CAUSAL_LM",
        trainable_token_indices={"model.embed_tokens": [36, 37, 38, 39]},
    ))
    with torch.no_grad():
        for name, parameter in causal_lm.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, .01)
    return RestrictedSIDModel(causal_lm, [36, 37, 38, 39])


def test_recomputed_objective_integrates_with_tied_peft_qwen35_across_steps():
    model = tiny_tied_qwen35().eval()
    oi = loss_fn(torch.tensor([[3, 36, 7, 8, 9, 10]]), torch.tensor([1]))
    ni = loss_fn(torch.tensor([[4, 37, 7, 8, 9, 10]]), torch.tensor([1]))
    original = copy.deepcopy(model.state_dict())
    identities = {name: id(value) for name, value in model.named_parameters()}
    first = compute_meta_gradients(model, oi, ni, alpha=.03)
    second = compute_meta_gradients(model, oi, ni, alpha=.03)
    assert_gradient_close(first.gradients, second.gradients, rtol=0, atol=0)
    _assert_nested_equal(model.state_dict(), original)
    reference = build_meta_objective(model, oi, ni, alpha=.03)
    assert_gradient_close(first.gradients, graph_gradients(model, reference), rtol=5e-4, atol=3e-6)
    sid_names = [name for name in first.gradients if "trainable_tokens_delta" in name]
    assert len(sid_names) == 1
    assert first.gradients[sid_names[0]].norm().item() > 0
    optimizer = torch.optim.SGD(named_trainable(model).values(), lr=.001)
    for name, parameter in named_trainable(model).items():
        parameter.grad = second.gradients[name].clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    third = compute_meta_gradients(model, oi, ni, alpha=.03)
    reference = build_meta_objective(model, oi, ni, alpha=.03)
    assert_gradient_close(third.gradients, graph_gradients(model, reference), rtol=5e-4, atol=3e-6)
    for name, parameter in model.named_parameters():
        assert id(parameter) == identities[name]
        assert parameter.is_leaf and parameter.grad_fn is None
        assert parameter.grad is None


@pytest.mark.parametrize("mode", ["meta", "meta_no_history"])
def test_bf16_body_uses_same_inner_backward_path_for_fast_step_and_hvp(mode):
    # Regression for the native gate: BF16 body kernels can produce different
    # FP32 LoRA gradients with create_graph=False and create_graph=True, despite
    # identical forward values and RNG. Initial phi' and its HVP must use the
    # SAME differentiable backward path, not a relaxed dtype tolerance.
    model = tiny_tied_qwen35(base_dtype=torch.bfloat16, dropout=.05).train()
    parameters = dict(model.named_parameters())
    inner = {name: value for name, value in parameters.items()
             if value.requires_grad and ("lora_A" in name or "lora_B" in name)}
    assert {value.dtype for value in inner.values()} == {torch.float32}
    assert {value.dtype for name, value in parameters.items() if "trainable_tokens_delta" in name} == {torch.float32}
    assert any(value.dtype == torch.bfloat16 and not value.requires_grad for value in parameters.values())
    x = torch.arange(193).remainder(35).add(1).unsqueeze(0)
    y = torch.tensor([1])
    oi, ni = loss_fn(x, y), loss_fn(x.roll(1, dims=1), y)
    kwargs = {"no_history_oi": loss_fn(x[:, :6], y)} if mode == "meta_no_history" else {}
    original = copy.deepcopy(model.state_dict())
    torch.manual_seed(294)
    first = compute_meta_gradients(model, oi, ni, alpha=.001, mode=mode, **kwargs)
    after_first = torch.random.get_rng_state().clone()
    torch.manual_seed(294)
    second = compute_meta_gradients(model, oi, ni, alpha=.001, mode=mode, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), after_first)
    assert_gradient_close(first.gradients, second.gradients, rtol=0, atol=0)
    _assert_nested_equal(model.state_dict(), original)
    torch.manual_seed(294)
    reference = build_meta_objective(model, oi, ni, alpha=.001, mode=mode, **kwargs)
    torch.testing.assert_close(first.loss, reference.loss.detach(), rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), after_first)
    assert first.inner_grad_norm.item() > 0 and first.hvp_norm.item() > 0


def test_only_lora_parameters_are_replaced_even_with_nested_names(data):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.causal_lm = NonlinearLoRAModel()

        def forward(self, x):
            return self.causal_lm(x)

    model = Wrapper()
    original = dict(model.named_parameters())
    calls = []

    def ni(view):
        calls.append({name: value.detach().clone() for name, value in view.parameters.items()})
        return F.cross_entropy(view(data[2]), data[3])

    result = compute_meta_gradients(model, loss_fn(data[0], data[1]), ni, alpha=.3)
    assert result.inner_parameter_names == ("causal_lm.lora_A", "causal_lm.lora_B")
    assert len(calls) == 2
    for name, value in original.items():
        assert torch.equal(calls[0][name], value)
        if name in result.inner_parameter_names:
            assert not torch.equal(calls[1][name], value)
        else:
            assert torch.equal(calls[1][name], value)


def test_fail_closed_on_nonrepeatable_inner_loss(data):
    model = NonlinearLoRAModel()
    counter = 0

    def stateful(view):
        nonlocal counter
        counter += 1
        return counter * F.cross_entropy(view(data[0]), data[1])

    with pytest.raises(RuntimeError, match="Inner recomputation changed"):
        compute_meta_gradients(model, stateful, specs(data)[1], alpha=.2)


def test_identity_guard_detects_loss_changing_model_mode(data):
    model = NonlinearLoRAModel()

    def impure(view):
        model.eval()
        return F.cross_entropy(view(data[0]), data[1])

    with pytest.raises(RuntimeError, match="identity"):
        compute_meta_gradients(model, impure, specs(data)[1], alpha=.2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"alpha": -.1}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": .1, "mode": "mixed"}, "mixed mode"),
        ({"alpha": .1, "mode": "wrong"}, "mode"),
        ({"alpha": .1, "mode": "meta_no_history"}, "requires no_history_oi"),
    ],
)
def test_invalid_configuration_is_rejected(data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_meta_gradients(NonlinearLoRAModel(), *specs(data), **kwargs)


def test_invalid_chunks_and_missing_trainable_lora_are_rejected(data):
    model = NonlinearLoRAModel()
    oi, ni = specs(data)
    with pytest.raises(ValueError, match="at least one"):
        compute_meta_gradients(model, [], ni, alpha=.1)
    with pytest.raises(ValueError, match="positive finite sum"):
        compute_meta_gradients(model, [LossChunk(oi, 0)], ni, alpha=.1)
    model.lora_A.requires_grad_(False)
    model.lora_B.requires_grad_(False)
    with pytest.raises(ValueError, match="lora_A/lora_B"):
        compute_meta_gradients(model, oi, ni, alpha=.1)
