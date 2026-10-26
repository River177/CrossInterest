from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from new_interest.training.oi_ni_meta_objective import (
    FunctionalModel, LossChunk, compute_meta_gradients,
)
from new_interest.training.oi_ni_meta_rng import bind_planned_rng, plan_episode_rng


class StochasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.tensor([[.2, .3], [-.4, .1], [.5, -.2]], dtype=torch.float64))
        self.lora_B = nn.Parameter(torch.tensor([[.6, -.3], [.2, .7]], dtype=torch.float64))
        self.sid = nn.Parameter(torch.tensor([.3, -.2], dtype=torch.float64))
        self.dropout = nn.Dropout(.25)

    def forward(self, x):
        # Functional randomness is deliberately included: module-hook-only RNG
        # plans would miss this operation.
        x = F.dropout(self.dropout(x), .2, training=self.training)
        return torch.tanh(x @ self.lora_A @ self.lora_B + self.sid)


def chunks(offset, count):
    result = []
    for i in range(count):
        x = torch.arange((i + 1) * 3, dtype=torch.float64).reshape(i + 1, 3) / 7 + offset
        target = torch.tensor([.4, -.5], dtype=torch.float64) + i / 10
        result.append(LossChunk(lambda view, x=x, target=target: (view(x)-target).square().mean(), i + 1))
    return result


def bind(chunks_, states):
    return [LossChunk(bind_planned_rng(chunk.loss_fn, state), chunk.weight)
            for chunk, state in zip(chunks_, states)]


@pytest.mark.parametrize("mode,alpha", [("mixed", 0.), ("meta", .07), ("meta_no_history", .07)])
def test_original_serial_gradients_match_preview_bound_reordered_losses(mode, alpha):
    model = StochasticModel().train()
    oi, ni, empty = chunks(.1, 3), chunks(.4, 2), chunks(-.2, 3)
    before = copy.deepcopy(model.state_dict())
    torch.manual_seed(918)
    expected = compute_meta_gradients(model, oi, ni, alpha=alpha, mode=mode,
                                     no_history_oi=empty if mode == "meta_no_history" else None)
    external = torch.random.get_rng_state().clone()
    plan = plan_episode_rng(model, 918, oi, ni, empty, mode=mode)
    assert torch.equal(external, torch.random.get_rng_state())
    actual = compute_meta_gradients(model, bind(oi, plan["oi"]), bind(ni, plan["ni"]),
                                   alpha=alpha, mode=mode,
                                   no_history_oi=bind(empty, plan["no_history"]) if mode == "meta_no_history" else None)
    assert torch.equal(external, torch.random.get_rng_state())
    for name in expected.gradients:
        torch.testing.assert_close(actual.gradients[name], expected.gradients[name], rtol=0, atol=0)
    for name in ("loss", "oi_loss", "ni_loss", "adapted_ni_loss", "inner_grad_norm", "hvp_norm"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert plan["audit"]["preview_forward_count"] == (8 if mode == "meta_no_history" else 5)


def test_plan_recomputes_absolute_states_for_each_episode_seed_and_rejects_unknown_preview():
    model = StochasticModel().train()
    oi, ni = chunks(.1, 2), chunks(.2, 2)
    first = plan_episode_rng(model, 42, oi, ni)
    second = plan_episode_rng(model, 42 + 104729, oi, ni)
    assert first["audit"]["state_sha256"] != second["audit"]["state_sha256"]
    assert not torch.equal(first["oi"][0]["cpu_before"], second["oi"][0]["cpu_before"])
    with pytest.raises(ValueError, match="Only real-forward"):
        plan_episode_rng(model, 42, oi, ni, planning_mode="dummy")


def test_bound_loss_fails_if_actual_rng_consumption_changes_and_restores_outer_stream():
    model = StochasticModel().train()
    oi, ni = chunks(.1, 1), chunks(.2, 1)
    plan = plan_episode_rng(model, 42, oi, ni)
    view = FunctionalModel(model, dict(model.named_parameters()), {})
    external = torch.random.get_rng_state().clone()

    def changed(view):
        torch.rand(1)
        return oi[0].loss_fn(view)

    with pytest.raises(RuntimeError, match="RNG consumption differs"):
        bind_planned_rng(changed, plan["oi"][0])(view)
    assert torch.equal(external, torch.random.get_rng_state())


def test_previews_require_training_mode_and_complete_global_roles():
    model = StochasticModel().eval()
    with pytest.raises(ValueError, match="model.train"):
        plan_episode_rng(model, 42, chunks(.1, 1), chunks(.2, 1))
    model.train()
    with pytest.raises(ValueError, match="nonempty"):
        plan_episode_rng(model, 42, [], chunks(.2, 1))
    with pytest.raises(ValueError, match="No-history"):
        plan_episode_rng(model, 42, chunks(.1, 1), chunks(.2, 1), mode="meta_no_history")
