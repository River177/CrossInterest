import pytest
import torch
from transformers import TemperatureLogitsWarper, TopPLogitsWarper

from new_interest.training.short_think_grpo_objective import (
    group_advantages,grpo_loss,nucleus_selected_logprobs,raw_forward_kl)


def test_nucleus_probabilities_match_the_actual_hf_warpers():
    torch.manual_seed(12)
    logits=torch.randn(17,37)
    selected=logits.argmax(-1)
    scores=TemperatureLogitsWarper(.8)(None,logits)
    scores=TopPLogitsWarper(.95)(None,scores)
    expected=scores.log_softmax(-1).gather(-1,selected[:,None])[:,0]
    actual=nucleus_selected_logprobs(logits,selected)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
    low=logits.argmin(-1)
    assert torch.isneginf(nucleus_selected_logprobs(logits,low)).all()


def test_same_policy_has_unit_ratio_and_prompt_padding_have_no_gradient():
    current=torch.tensor([[.0,-2.,-1.,-4.],[.0,-3.,-2.,-1.]],requires_grad=True)
    behavior=current.detach().clone()
    mask=torch.tensor([[0,1,1,0],[0,1,1,1]],dtype=torch.bool)
    loss,stats=grpo_loss(current,behavior,torch.tensor([1.,-1.]),mask,kl_coef=0)
    assert stats['ratio_mean']==1 and stats['clip_fraction']==0
    loss.backward()
    assert (current.grad[~mask]==0).all()
    torch.testing.assert_close(current.grad[0,1:3],torch.tensor([-.25,-.25]))
    torch.testing.assert_close(current.grad[1,1:],torch.full((3,),1/6))


def test_support_miss_is_zero_ratio_not_a_dropped_trajectory():
    logits=torch.tensor([[10.,0.,-10.]],requires_grad=True)
    current=nucleus_selected_logprobs(logits,torch.tensor([2])).reshape(1,1)
    loss,stats=grpo_loss(current,torch.tensor([[-5.]]),torch.tensor([-1.]),torch.ones(1,1,dtype=torch.bool),kl_coef=0)
    torch.testing.assert_close(loss,torch.tensor(.8))
    assert stats['current_support_miss_fraction']==1
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_reference_is_frozen_and_equal_logits_have_zero_kl():
    torch.manual_seed(7)
    current=torch.randn(2,3,11,requires_grad=True)
    reference=current.detach().clone().requires_grad_()
    kl=raw_forward_kl(current,reference)
    torch.testing.assert_close(kl,torch.zeros(2,3),atol=1e-7,rtol=0)
    kl.sum().backward()
    assert reference.grad is None


def test_reward_floor_and_behavior_errors():
    torch.testing.assert_close(group_advantages(torch.zeros(2,8)),torch.zeros(2,8))
    assert (group_advantages(torch.full((2,8),.2))==0).all()
    assert group_advantages(torch.tensor([[1.,.999]])).abs().max()<.01
    with pytest.raises(ValueError,match='Behavior'):
        grpo_loss(torch.zeros(1,2),torch.tensor([[float('-inf'),0.]]),torch.ones(1),torch.ones(1,2,dtype=torch.bool),kl_coef=0)
