import copy

import pytest
import torch
from torch import nn

from new_interest.training.oi_ni_meta_objective import LossChunk
from new_interest.training.personal_cir_meta import compute_personal_gradients, graph_objective


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.tensor([[.2, -.3], [.4, .1]], dtype=torch.double))
        self.lora_B = nn.Parameter(torch.tensor([[.6, .2], [-.1, .5]], dtype=torch.double))
        self.sid = nn.Parameter(torch.tensor([[.2, -.4], [.3, .6]], dtype=torch.double))
        self.dropout = nn.Dropout(.25)
        self.register_buffer('constant', torch.ones(2, dtype=torch.double))

    def forward(self, x):
        return torch.tanh(self.dropout(x) @ self.lora_A @ self.lora_B + self.constant) @ self.sid


def chunk(x, y):
    x, y = torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double)
    return LossChunk(lambda model: (model(x) - y).square().mean())


def episode(offset=0):
    support = [chunk([[.7 + offset, -.4]], [[.2, .9]]), chunk([[-.2, .8]], [[.9, -.3 + offset]])]
    return support, [chunk([[.5, .4 + offset]], [[-.8, .1]])]


@pytest.mark.parametrize('support_count', [0, 1, 2])
@pytest.mark.parametrize('alpha', [0., .13])
def test_exact_hvp_matches_full_unroll_and_does_not_mutate(support_count, alpha):
    model = Toy()
    initial = copy.deepcopy(model.state_dict())
    support, query = episode()
    result = compute_personal_gradients(model, support[:support_count], query, alpha=alpha, query_training=False)
    loss = graph_objective(model, support[:support_count], query, alpha=alpha, query_training=False)
    expected = torch.autograd.grad(loss, tuple(model.parameters()))
    torch.testing.assert_close(result.loss, loss)
    for (name, param), grad in zip(model.named_parameters(), expected):
        torch.testing.assert_close(result.gradients[name], grad, atol=1e-12, rtol=1e-12)
        assert param.grad is None
    assert model.training and model.dropout.training
    for name, value in initial.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_sid_mixed_second_derivative_and_finite_difference():
    model = Toy()
    support, query = episode()
    result = compute_personal_gradients(model, support, query, alpha=.2, query_training=False)
    eps = 1e-6
    values = []
    for delta in [-eps, eps]:
        other = copy.deepcopy(model)
        with torch.no_grad():
            other.sid[0, 0].add_(delta)
        values.append(float(graph_objective(other, support, query, alpha=.2, query_training=False).detach()))
    assert result.gradients['sid'][0, 0].item() == pytest.approx((values[1] - values[0]) / (2 * eps), rel=1e-7, abs=1e-9)
    # Detaching the SID dependence of the support gradient must change the answer.
    first = torch.autograd.grad(graph_objective(model, support, query, alpha=0, query_training=False), model.sid)[0]
    assert not torch.allclose(first, result.gradients['sid'])


def test_user_isolation_order_and_not_a_pooled_inner_update():
    model = Toy()
    f1, q1 = episode()
    f2, q2 = episode(1.5)
    r1 = compute_personal_gradients(model, f1, q1, alpha=.2, query_training=False)
    r2 = compute_personal_gradients(model, f2, q2, alpha=.2, query_training=False)
    repeat = compute_personal_gradients(model, f1, q1, alpha=.2, query_training=False)
    pooled = compute_personal_gradients(model, f1 + f2, q1 + q2, alpha=.2, query_training=False)
    for name in r1.gradients:
        torch.testing.assert_close(repeat.gradients[name], r1.gradients[name], rtol=0, atol=0)
    assert any(not torch.allclose((r1.gradients[n] + r2.gradients[n]) / 2, pooled.gradients[n]) for n in r1.gradients)


def test_support_dropout_disabled_and_query_rng_reproducible():
    model = Toy()
    support, query = episode()
    torch.manual_seed(77)
    result = compute_personal_gradients(model, support, query, alpha=.1)
    torch.manual_seed(77)
    loss = graph_objective(model, support, query, alpha=.1)
    expected = torch.autograd.grad(loss, tuple(model.parameters()))
    for n, g in zip(dict(model.named_parameters()), expected):
        torch.testing.assert_close(result.gradients[n], g, atol=1e-12, rtol=1e-12)


def test_query_weight_does_not_depend_on_support_count():
    model = Toy()
    results = [compute_personal_gradients(model, episode(i / 7)[0][:i % 3], episode(i / 7)[1],
                                          alpha=.1, query_training=False) for i in range(7)]
    full = {n: sum(r.gradients[n] for r in results) / len(results) for n in results[0].gradients}
    ranks = [{n: sum((results[i].gradients[n] for i in range(rank, 7, 4)), torch.zeros_like(full[n]))
              for n in full} for rank in range(4)]
    for n in full:
        torch.testing.assert_close(sum(r[n] for r in ranks) / 7, full[n], rtol=1e-12, atol=1e-12)


def test_invalid_alpha_and_mode_restored_on_failure():
    model = Toy()
    support, query = episode()
    with pytest.raises(ValueError):
        compute_personal_gradients(model, support, query, alpha=-1)
    def failing(view):
        assert not view.module.training
        raise RuntimeError('expected')
    with pytest.raises(RuntimeError, match='expected'):
        compute_personal_gradients(model, [LossChunk(failing)], query, alpha=.1)
    assert model.training and model.dropout.training
