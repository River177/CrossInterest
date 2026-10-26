import torch

from new_interest.training.short_think_grpo_sampling import PhaseSIDProcessor


class Tokenizer:
    def decode(self,ids,**kwargs):
        return ''.join({1:'\n',2:'<answer>',3:'<interest_000>',4:'</answer>',5:'free text'}[x] for x in ids)


def test_thought_excludes_only_sid_tokens_and_does_not_inject_a_close():
    scores=torch.arange(7,dtype=torch.float)[None]
    proc=PhaseSIDProcessor(Tokenizer(),phase='thought',prefix_width=1,sid_ids=[3,6])
    result=proc(torch.tensor([[0,5]]),scores)
    assert torch.isneginf(result[0,[3,6]]).all()
    torch.testing.assert_close(result[0,[0,1,2,4,5]],scores[0,[0,1,2,4,5]])
    assert torch.isfinite(scores).all()


def test_answer_only_constrains_after_naturally_generated_open():
    proc=PhaseSIDProcessor(Tokenizer(),phase='answer',prefix_width=1,sid_ids=[3,6])
    scores=torch.arange(7,dtype=torch.float)[None]
    torch.testing.assert_close(proc(torch.tensor([[0,1]]),scores),scores)
    result=proc(torch.tensor([[0,1,2]]),scores)
    assert torch.isfinite(result[0,[3,6]]).all()
    assert torch.isneginf(result[0,[0,1,2,4,5]]).all()
    # No forced closing tag or extra tokens after the sampled SID.
    torch.testing.assert_close(proc(torch.tensor([[0,1,2,3]]),scores),scores)
    # Malformed model output remains a real model failure, not repaired text.
    torch.testing.assert_close(proc(torch.tensor([[0,5,2]]),scores),scores)
