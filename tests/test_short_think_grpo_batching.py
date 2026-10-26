from types import SimpleNamespace

from new_interest.training import short_think_grpo_engine as engine


def test_rank_batch_preserves_all_groups_and_failed_trajectory_positions(monkeypatch):
    tokens=['<interest_000>','<interest_001>']
    records=[{'prompt':[{'role':'user','content':str(i)}],'extra_info':{
        'source_sample_id':f'test-{i}','target_token':tokens[i%2],'interest_tokens':tokens,
        'interest_categories':['P','Q'],'interest_subcategories':['A','B'],'history_subcategories':[]}}
        for i in range(8)]
    class Tokenizer:
        eos_token_id=100
        def encode(self,text,**kwargs):return [99] if text=='</think>' else [50+int(text[0])]
        def convert_tokens_to_ids(self,text):return 101
    calls=[]
    def sample(model,tokenizer,prefixes,stops,budget,seed,parity,**kwargs):
        calls.append((kwargs['phase'],prefixes,seed))
        if kwargs['phase']=='thought':
            result=[([10,99],True) for _ in prefixes]
            result[1]=([10]*480,False);result[2]=([10,100],True)
        else:result=[([1,20+(p[0]-50)%2,2,100],True) for p in prefixes]
        return result,[{'cached_behavior_logprobs':[-.1]*len(ids)} for ids,ended in result]
    def parse(tokenizer,ids,**kwargs):
        sid=next((tokens[x-20] for x in ids if x in [20,21]),None)
        return {'valid_format':sid is not None,'generated_sid':sid,'think_tokens':2 if sid else None}
    monkeypatch.setattr(engine,'stage',sample)
    monkeypatch.setattr(engine,'analyze_generation',parse)
    monkeypatch.setattr(engine,'apply_chat_template',lambda tokenizer,messages,**kwargs:messages[0]['content']+'<think>\n')
    model=SimpleNamespace(eval=lambda:None)
    spec=SimpleNamespace(token_ids=[20,21],entries=[SimpleNamespace(interest_token=t) for t in tokens])
    rows=engine.batched_rollouts(model,Tokenizer(),spec,records,list(range(8)),0,1)
    assert len(calls)==2 and len(calls[0][1])==64 and len(calls[1][1])==62
    assert calls[0][2]==100042 and calls[1][2]==150042
    assert len(rows)==64 and not rows[1]['valid'] and not rows[2]['valid']
    for i,row in enumerate(rows):
        assert row['source_sample_id']==f'test-{i//8}' and row['generation_index']==i%8
        assert row['prompt_ids']==[50+i//8]
        assert len(row['behavior_logprobs'])==len(row['response_ids'])
        if i not in [1,2]:assert row['reward']['exact']
