from types import SimpleNamespace

import torch

from new_interest.evaluation.preflight_short_think_grpo import stage, through_first_stop


def test_stage_keeps_generated_closure_and_uses_independent_budget():
    class Tokenizer:
        pad_token_id=0
        def pad(self, data, **kwargs):
            assert data['input_ids']==[[10,11],[10,11]]
            return {'input_ids':torch.tensor(data['input_ids']), 'attention_mask':torch.ones(2,2,dtype=torch.long)}
    class Model:
        device=torch.device('cpu')
        def generate(self, **kwargs):
            assert kwargs['max_new_tokens']==32
            assert kwargs['eos_token_id']==[99]
            assert kwargs['top_k']==0 and kwargs['temperature']==.8
            # One completion terminates early; padding must not enter the action mask.
            return SimpleNamespace(sequences=torch.tensor([[10,11,5,99,0],[10,11,6,7,99]]))
    rows, diagnostics=stage(Model(),Tokenizer(),[[10,11],[10,11]],{99},32,42,False)
    assert rows==[([5,99],True),([6,7,99],True)]
    assert diagnostics==[]


def test_no_closure_is_injected_when_thought_budget_exhausts():
    assert through_first_stop([1,2,3],{99})==([1,2,3],False)
    assert through_first_stop([1,98,0],{98,99})==([1,98],True)
