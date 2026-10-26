import math
from types import SimpleNamespace
import torch

from new_interest.evaluation.preflight_short_think_grpo import stage


def test_sampled_behavior_scores_exclude_padding_after_eos():
    class Tokenizer:
        pad_token_id=0
        def pad(self,data,**kwargs):
            return {'input_ids':torch.tensor(data['input_ids']),'attention_mask':torch.ones(2,2,dtype=torch.long)}
    class Model:
        device=torch.device('cpu')
        def generate(self,**kwargs):
            assert kwargs['output_scores'] and not kwargs['output_logits']
            return SimpleNamespace(sequences=torch.tensor([[10,11,5,99,0],[10,11,6,7,99]]),scores=tuple(torch.zeros(2,100) for _ in range(3)))
    responses,diagnostics=stage(Model(),Tokenizer(),[[10,11]]*2,{99},32,42,False,capture_behavior=True)
    assert [len(x['cached_behavior_logprobs']) for x in diagnostics]==[2,3]
    assert diagnostics[0]['generated_ids']==[5,99]
    assert all(abs(x+math.log(100))<1e-6 for row in diagnostics for x in row['cached_behavior_logprobs'])
