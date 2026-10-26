from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
if transformers.__version__ != "5.14.1":
    pytest.skip("Exact Qwen3.5 backend is pinned to Transformers 5.14.1", allow_module_level=True)

from torch.func import functional_call
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from new_interest.training.oi_ni_meta_backend import RestrictedSIDModel, effective_sid_rows


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def tiny_base():
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=40, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [2, 1, 1]},
    )
    config._attn_implementation = "eager"
    return Qwen3_5ForCausalLM(config).cpu()


def tiny_model() -> RestrictedSIDModel:
    model = tiny_base()
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0., target_modules="all-linear", task_type="CAUSAL_LM",
        trainable_token_indices={"model.embed_tokens": [36, 37, 38, 39]},
    ))
    # Nonzero A and B exercise derivatives throughout both factor paths.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0, std=0.01)
    return RestrictedSIDModel(model, [36, 37, 38, 39]).eval()


def batch(offset: int = 0):
    return torch.tensor([[3 + offset, 5, 7, 8, 9, 10]])


def trainable(model):
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_restricted_projection_and_effective_tied_rows_match_original():
    model = tiny_model()
    p = model.causal_lm
    ids = model.sid_token_ids
    original = p(input_ids=batch(), use_cache=False).logits[:, -1].index_select(-1, ids)
    actual = model(batch())
    torch.testing.assert_close(actual, original, rtol=2e-6, atol=2e-7)
    rows = effective_sid_rows(p, ids)
    torch.testing.assert_close(rows, p.get_output_embeddings().weight.index_select(0, ids), rtol=0, atol=0)
    params = trainable(model)
    sid_names = [name for name in params if "trainable_tokens_delta" in name]
    assert len(sid_names) == 1
    assert sid_names[0].startswith("causal_lm.")
    gradients = torch.autograd.grad(actual.square().sum(), tuple(params.values()))
    original_gradients = torch.autograd.grad(original.square().sum(), tuple(params.values()))
    for current, reference in zip(gradients, original_gradients):
        torch.testing.assert_close(current, reference, rtol=3e-5, atol=2e-6)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[list(params).index(sid_names[0])].abs().sum() > 0
    assert len(model.backend_audit["linear_attention_modules"]) == 3
    assert len(model.backend_audit["full_attention_modules"]) == 1


def test_functional_fast_weights_replace_lora_and_sid_without_mutation():
    model = tiny_model()
    params = trainable(model)
    snapshots = {name: value.detach().clone() for name, value in params.items()}
    fast = dict(params)
    for name, value in params.items():
        if "trainable_tokens_delta" in name or "lora_B" in name:
            fast[name] = value + 0.02 * torch.randn_like(value)
    expected = model(batch())
    actual = functional_call(model, fast, (batch(),))
    assert not torch.allclose(actual, expected)
    current = trainable(model)
    for name, value in params.items():
        assert current[name] is value, name
        torch.testing.assert_close(value, snapshots[name], rtol=0, atol=0)


def test_strict_functional_call_restores_all_parameter_and_buffer_aliases():
    model = tiny_model()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    fast = {name: value + 0.01 * torch.randn_like(value) if value.requires_grad else value
            for name, value in params.items()}
    scores = functional_call(model, (fast, buffers), (batch(),), strict=True)
    assert torch.isfinite(scores).all()
    for name, value in model.named_parameters():
        assert value is params[name], name
    for name, value in model.named_buffers():
        assert value is buffers[name], name
    # A new gradient after the swapped graph is consumed must use fresh state.
    torch.autograd.grad(scores.square().sum(), tuple(trainable(model).values()))
    gradients = torch.autograd.grad(model(batch()).square().sum(), tuple(trainable(model).values()))
    assert all(torch.isfinite(value).all() for value in gradients)


def test_safe_adapter_reload_preserves_canonical_sid_and_tied_output(tmp_path):
    from safetensors.torch import load_file

    model = tiny_model()
    expected = model(batch()).detach()
    model.causal_lm.save_pretrained(tmp_path, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(tmp_path / "adapter_model.safetensors"))
    sid_keys = [name for name in saved if "trainable_tokens_delta" in name]
    assert len(sid_keys) == 1
    assert "embed_tokens" in sid_keys[0] and "lm_head" not in sid_keys[0]
    fresh = peft.PeftModel.from_pretrained(tiny_base(), tmp_path, is_trainable=True)
    reloaded = RestrictedSIDModel(fresh, [36, 37, 38, 39]).eval()
    torch.testing.assert_close(reloaded(batch()), expected, rtol=0, atol=0)
    inp = fresh.get_input_embeddings().token_adapter
    out = fresh.get_output_embeddings().token_adapter
    assert inp.trainable_tokens_delta is out.trainable_tokens_delta
    assert out.tied_adapter is inp
    assert "trainable_tokens_delta" not in out._modules
    assert len([name for name in trainable(reloaded) if "trainable_tokens_delta" in name]) == 1


def test_true_hessian_vector_product_matches_finite_difference():
    model = tiny_model()
    params = trainable(model)
    direction = {name: torch.randn_like(value) for name, value in params.items()}
    norm = sum(value.square().sum() for value in direction.values()).sqrt()
    direction = {name: value / norm for name, value in direction.items()}

    def loss(values):
        return torch.nn.functional.cross_entropy(functional_call(model, values, (batch(),)), torch.tensor([2]))

    gradients = torch.autograd.grad(loss(params), tuple(params.values()), create_graph=True)
    dot = sum((gradient * direction[name]).sum() for name, gradient in zip(params, gradients))
    hvp = torch.autograd.grad(dot, tuple(params.values()))
    epsilon = 0.003
    sides = []
    for sign in (1., -1.):
        values = {name: (value.detach() + sign * epsilon * direction[name]).requires_grad_()
                  for name, value in params.items()}
        sides.append(torch.autograd.grad(loss(values), tuple(values.values())))
    fd = [(plus - minus) / (2 * epsilon) for plus, minus in zip(*sides)]
    exact_flat = torch.cat([value.flatten() for value in hvp])
    fd_flat = torch.cat([value.flatten() for value in fd])
    assert torch.isfinite(exact_flat).all()
    assert exact_flat.norm() > 1e-4
    relative_error = (exact_flat - fd_flat).norm() / fd_flat.norm()
    assert relative_error < 0.015, relative_error.item()


def test_unrolled_meta_gradient_matches_finite_difference():
    model = tiny_model()
    params = trainable(model)
    phi = torch.tensor(0.3, requires_grad=True)

    def outer(phi_value, create_graph):
        support = torch.nn.functional.cross_entropy(model(batch()), torch.tensor([1]))
        gradient = torch.autograd.grad(phi_value.sigmoid() * support, tuple(params.values()), create_graph=create_graph)
        fast = {name: value - 0.05 * g for (name, value), g in zip(params.items(), gradient)}
        logits = functional_call(model, fast, (batch(1),))
        return torch.nn.functional.cross_entropy(logits, torch.tensor([3]))

    exact = torch.autograd.grad(outer(phi, True), phi)[0]
    epsilon = 0.003
    fd = (outer(phi.detach() + epsilon, False) - outer(phi.detach() - epsilon, False)) / (2 * epsilon)
    assert torch.isfinite(exact) and exact.abs() > 1e-5
    torch.testing.assert_close(exact, fd, rtol=0.02, atol=2e-5)


def test_internal_checkpointing_is_rejected_instead_of_silently_using_base_weights():
    model = tiny_model()
    model.causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with pytest.raises(RuntimeError, match="functional fast weights"):
        model(batch())


def test_last_visible_position_handles_right_padding():
    model = tiny_model()
    x = batch()
    padded = torch.cat([x, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones_like(x), torch.zeros(1, 2, dtype=torch.long)], dim=1)
    torch.testing.assert_close(model(padded, mask), model(x), rtol=2e-5, atol=2e-6)


def test_projection_stays_float32_inside_an_outer_autocast_context():
    model = tiny_model()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = model(batch())
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()


def test_qwen_backend_integrates_with_exact_recomputed_objective_across_updates():
    from new_interest.training.oi_ni_meta_objective import build_meta_objective, compute_meta_gradients

    model = tiny_model()
    oi = lambda view: torch.nn.functional.cross_entropy(view(batch()), torch.tensor([1]))
    ni = lambda view: torch.nn.functional.cross_entropy(view(batch(1)), torch.tensor([3]))
    optimizer = torch.optim.AdamW(trainable(model).values(), lr=1e-4)
    for iteration in range(3):
        params = trainable(model)
        full = build_meta_objective(model, oi, ni, alpha=0.05, mode="meta")
        reference = torch.autograd.grad(full.loss, tuple(params.values()))
        recomputed = compute_meta_gradients(model, oi, ni, alpha=0.05, mode="meta")
        for (name, parameter), expected in zip(params.items(), reference):
            torch.testing.assert_close(recomputed.gradients[name], expected, rtol=3e-4, atol=2e-6)
            assert parameter is trainable(model)[name]
        assert torch.isfinite(recomputed.hvp_norm) and recomputed.hvp_norm > 0
        if iteration == 1:
            for name, parameter in params.items():
                parameter.grad = recomputed.gradients[name].clone()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
