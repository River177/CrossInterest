from __future__ import annotations

import pytest
torch = pytest.importorskip('torch')
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from new_interest.training.oi_ni_full_sft import full_vocabulary_projection, pooled_role_scaling
from new_interest.training.oi_ni_meta_objective import LossChunk
from new_interest.training.oi_ni_meta_distributed_objective import compute_distributed_meta_gradients
from new_interest.data.build_oi_ni_full_sft import epoch_batches, validate_epoch


def test_full_projection_loss_gradients_and_hessian_match_materialized_head():
    torch.manual_seed(4)
    h = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    base = torch.randn(11, 5, dtype=torch.double)
    rows = torch.randn(2, 5, dtype=torch.double, requires_grad=True)
    ids = torch.tensor([7, 10]); labels = torch.tensor([7, 2, 4])
    a = F.cross_entropy(full_vocabulary_projection(h, base, ids, rows), labels)
    b = F.cross_entropy(F.linear(h, base.index_copy(0, ids, rows)), labels)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(a, (h, rows), create_graph=True)
    gb = torch.autograd.grad(b, (h, rows), create_graph=True)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
    vectors = [torch.randn_like(x) for x in ga]
    ha = torch.autograd.grad(sum((x*v).sum() for x,v in zip(ga,vectors)), (h, rows))
    hb = torch.autograd.grad(sum((x*v).sum() for x,v in zip(gb,vectors)), (h, rows))
    for x, y in zip(ha, hb):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-12)
    changed = base.clone(); changed[3] += 20
    assert not torch.allclose(a, F.cross_entropy(full_vocabulary_projection(h, changed, ids, rows), labels))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(9)
        self.lora_A = nn.Parameter(torch.randn(3,3,dtype=torch.double)*.1)
        self.lora_B = nn.Parameter(torch.randn(3,3,dtype=torch.double)*.1)
        self.sid_rows = nn.Parameter(torch.randn(7,3,dtype=torch.double))
    def forward(self, x):
        return F.linear(torch.tanh(F.linear(x, self.lora_B @ self.lora_A)), self.sid_rows)


@pytest.mark.parametrize('alpha', [0., .01])
def test_pooled_meta_gradient_matches_direct_unroll_with_unequal_token_counts(alpha):
    m = Toy(); torch.manual_seed(12)
    x = torch.randn(5,3,dtype=torch.double); y = torch.tensor([0,6,4,2,1])
    fscale,gscale,effective_alpha = pooled_role_scaling(2,3,alpha)
    f = [LossChunk(lambda v: fscale*F.cross_entropy(v(x[:2]),y[:2]),2)]
    g = [LossChunk(lambda v: gscale*F.cross_entropy(v(x[2:]),y[2:]),3)]
    result = compute_distributed_meta_gradients(m,f,g,alpha=effective_alpha,
                                                mode='meta' if alpha else 'mixed',reduction='ordered')
    params = dict(m.named_parameters())
    F0 = F.cross_entropy(m(x[:2]),y[:2]); G0 = F.cross_entropy(m(x[2:]),y[2:])
    inner = torch.autograd.grad(F0,(m.lora_A,m.lora_B),create_graph=True)
    fast = {**params,'lora_A':m.lora_A-alpha*inner[0],'lora_B':m.lora_B-alpha*inner[1]}
    G1 = F.cross_entropy(functional_call(m,fast,(x[2:],)),y[2:])
    exact = .4*F0+.3*G0+.3*G1
    expected = torch.autograd.grad(exact,tuple(params.values()))
    torch.testing.assert_close(result.loss,exact,rtol=1e-12,atol=1e-12)
    for name, grad in zip(params,expected):
        torch.testing.assert_close(result.gradients[name],grad,rtol=1e-10,atol=1e-11)
    if alpha == 0:
        torch.testing.assert_close(result.loss,F.cross_entropy(m(x),y),rtol=1e-12,atol=1e-12)


def test_complete_epoch_no_resampling_and_nonzero_epoch_resume():
    rows=[{'training_task':'old_interest' if i%2 else 'new_interest', 'user_id':str(i)} for i in range(202)]
    epochs=[epoch_batches(rows,42,e) for e in range(3)]
    for batches in epochs: validate_epoch(rows,batches)
    assert epochs[0] != epochs[1]
    flat=[(e,b) for e,batches in enumerate(epochs) for b in batches]
    cursor=len(epochs[0])+2
    assert flat[:cursor]+flat[cursor:] == flat
    assert flat[cursor][0] == 1


def test_real_tiny_qwen_completion_matches_original_causal_lm_loss_and_gradients():
    pytest.importorskip('peft')
    transformers=pytest.importorskip('transformers')
    if transformers.__version__ != '5.14.1': pytest.skip('pinned worker backend')
    from test_oi_ni_meta_backend import tiny_model
    from new_interest.training.oi_ni_full_sft import FullVocabularySFTModel
    torch.set_num_threads(1)
    original=tiny_model()
    model=FullVocabularySFTModel(original.causal_lm,[36,37,38,39]).eval()
    ids=torch.tensor([[3,5,7,36,2,10]])
    labels=torch.tensor([[-100,-100,-100,36,2,10]])
    params=tuple(p for p in model.parameters() if p.requires_grad)
    got=model(ids,labels=labels)
    expected=model.causal_lm(input_ids=ids,labels=labels,use_cache=False).loss
    torch.testing.assert_close(got,expected,rtol=2e-6,atol=2e-7)
    gg=torch.autograd.grad(got,params);eg=torch.autograd.grad(expected,params)
    for a,b in zip(gg,eg):torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-6)
    sid_only=labels.clone();sid_only[:,4:]=-100
    assert not torch.allclose(model(ids,labels=sid_only),got)


def test_bfloat16_full_sft_uses_identical_native_loss_and_gradient_path():
    peft=pytest.importorskip('peft')
    transformers=pytest.importorskip('transformers')
    if transformers.__version__ != '5.14.1':pytest.skip('pinned worker backend')
    from test_oi_ni_meta_backend import tiny_base
    from new_interest.training.oi_ni_full_sft import FullVocabularySFTModel
    torch.set_num_threads(1)
    base=tiny_base().to(torch.bfloat16)
    p=peft.get_peft_model(base,peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
        target_modules='all-linear',task_type='CAUSAL_LM',
        trainable_token_indices={'model.embed_tokens':[36,37,38,39]}))
    model=FullVocabularySFTModel(p,[36,37,38,39]).eval()
    ids=torch.tensor([[3,5,7,36,2,10]])
    labels=torch.tensor([[-100,-100,-100,36,2,10]])
    parameters=tuple(x for x in model.parameters() if x.requires_grad)
    a=model(ids,labels=labels)
    ga=torch.autograd.grad(a,parameters,create_graph=True)
    b=model.causal_lm(input_ids=ids,labels=labels,use_cache=False).loss
    gb=torch.autograd.grad(b,parameters,create_graph=True)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for actual,expected in zip(ga,gb):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.manual_seed(81)
    vectors=[torch.randn_like(x) for x in ga]
    ha=torch.autograd.grad(sum((x*v).sum() for x,v in zip(ga,vectors)),parameters)
    hb=torch.autograd.grad(sum((x*v).sum() for x,v in zip(gb,vectors)),parameters)
    for actual,expected in zip(ha,hb):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
