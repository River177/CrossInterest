import math

import torch

from new_interest.training.short_think_grpo_engine import pack_trajectories,permutation,train_indices,trajectory_probabilities
from new_interest.training.short_think_grpo_objective import grpo_loss


class Tokenizer:
    def decode(self,ids,**kwargs):
        return ''.join({1:'<answer>',2:'\n',3:'SID_A',4:'SID_B',7:'</answer>'}[x] for x in ids)
    def pad(self,features,**kwargs):
        width=max(map(len,features['input_ids']))
        return {'input_ids':torch.tensor([[0]*(width-len(x))+x for x in features['input_ids']]),
                'attention_mask':torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in features['input_ids']])}


def test_response_alignment_and_phase_masks_include_both_stages():
    rows=[{'prompt_ids':[10,11],'response_ids':[8,9,1,3],'thought_length':2,'behavior_logprobs':[-1]*4},
          {'prompt_ids':[10,11],'response_ids':[8,9,2,1,4,7],'thought_length':2,'behavior_logprobs':[-1]*6}]
    batch,mask,old,ban,only=pack_trajectories(Tokenizer(),rows)
    assert mask.tolist()==[[False,False,True,True,True,True],[True]*6]
    assert ban.tolist()==[[False,False,True,True,False,False],[True,True,False,False,False,False]]
    assert only.tolist()==[[False]*5+[True],[False]*4+[True,False]]
    targets=batch['input_ids'][:,-6:]
    logits=torch.zeros(2,6,16);logits.scatter_(-1,targets[...,None],1.);logits.requires_grad_()
    current,kl=trajectory_probabilities(logits,logits.detach().clone(),targets,ban,only,[3,4])
    assert torch.isfinite(current[mask]).all()
    loss,diagnostic=grpo_loss(current,current.detach(),torch.tensor([1.,-1.]),mask,raw_kl=kl,kl_coef=0)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[mask].abs().sum()>0
    assert (logits.grad[~mask]==0).all()
    assert diagnostic['ratio_mean']==1


def test_global_prompt_cursor_and_restart_are_exact():
    order=permutation(21600,42)
    assert len(set(order))==21600
    first=[i for rank in range(4) for i in train_indices(order,0,rank)]
    next_step=[i for rank in range(4) for i in train_indices(permutation(21600,42),1,rank)]
    assert first+next_step==order[:64]
    assert len(set(first+next_step))==64
    assert [i for rank in range(4) for i in train_indices(order,674,rank)]==order[-32:]
