import copy

import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('peft')
from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
from peft import LoraConfig,get_peft_model

from new_interest.training.sid_relation_attention import CategoryRelation,SIDRelationModel,effective_sid_rows
from new_interest.data.build_sid_relation_sidecar import event_spans


def tiny(dtype=torch.float32, hybrid=False):
    torch.manual_seed(19)
    cfg=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3 if hybrid else 2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        layer_types=['full_attention','linear_attention','full_attention'] if hybrid else ['full_attention','full_attention'],
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
        tie_word_embeddings=True,attention_dropout=0,use_cache=False)
    cfg._attn_implementation='eager'
    base=Qwen3_5ForCausalLM(cfg).to(dtype=dtype)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',trainable_token_indices=[60,61,62,63]))
    return model


def sample():
    x=torch.tensor([[2,5,6,7,8,9,60]])
    y=torch.full_like(x,-100);y[:,-1]=60
    return dict(input_ids=x,attention_mask=torch.ones_like(x),labels=y,
        event_ids=torch.tensor([[-1,0,0,1,1,-1,-1]]),
        source_ids=torch.tensor([[-1,0,0,1,1,-1,-1]]))


def wrap(m):
    return SIDRelationModel(m,[60,61,62,63],[0,1,2,3],torch.zeros(4,32),enabled=True,layer_idx=0,rank=3)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_native_zero_bias_loss_logits_and_gradient_parity(dtype):
    m=tiny(dtype);m.eval();b=sample()
    raw={k:v for k,v in b.items() if k not in ['event_ids','source_ids']}
    o=m(**raw);ref=o.logits.detach().clone();loss=o.loss.detach().clone();o.loss.backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None};m.zero_grad(set_to_none=True)
    w=wrap(m);w.eval();o=w(**b)
    assert torch.equal(o.logits,ref) and torch.equal(o.loss,loss)
    o.loss.backward()
    for n,p in m.named_parameters():
        if n in grads:assert torch.equal(p.grad,grads[n]),n
    assert w.relation.V.grad.norm()>0
    assert w.relation.U.grad.norm()==0


def test_nonzero_relation_changes_sid_and_checkpointing_gradient():
    w=wrap(tiny());b=sample();w.eval();ref=w(**b).logits.detach()
    with torch.no_grad():w.relation.V.normal_(std=.1)
    changed=w(**b).logits
    assert not torch.equal(changed[:,-2],ref[:,-2])
    w.causal_lm.enable_input_require_grads()
    w.causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    w.train();w(**b).loss.backward()
    assert w.relation.U.grad.norm()>0 and w.relation.V.grad.norm()>0


def test_source_mapping_and_shared_sid_gradient():
    r=CategoryRelation(12,3,[2,-1,0],torch.ones(3,12),True)
    with torch.no_grad():r.V.normal_(std=.1)
    e=torch.randn(4,12,requires_grad=True)
    events=torch.tensor([[-1,0,0,1,1,-1]])
    source=torch.tensor([[-1,0,0,2,2,-1]])
    out=r(e,events,source)
    out.square().sum().backward()
    assert e.grad[[0,2]].norm()>0
    assert e.grad[[1,3]].norm()==0
    valid=(events[:,:,None]>=0)&(events[:,None,:]>=0)&events[:,:,None].ne(events[:,None,:])
    assert torch.count_nonzero(out[:,0][~valid])==0
    assert torch.count_nonzero(r(e,torch.zeros_like(events),source.clamp_min(0)))==0


def test_cache_prefill_and_incremental_equivalence():
    w=wrap(tiny(hybrid=True));w.eval();b=sample()
    with torch.no_grad():
        w.relation.V.normal_(std=.05)
        prefix={k:v[:,:-1] for k,v in b.items() if k!='labels'}
        plain=w(**prefix,use_cache=False)
        cached=w(**prefix,use_cache=True)
        assert torch.equal(plain.logits,cached.logits)
        step=w(input_ids=b['input_ids'][:,-1:],attention_mask=b['attention_mask'],
            event_ids=b['event_ids'][:,-1:],source_ids=b['source_ids'][:,-1:],
            use_cache=True,past_key_values=cached.past_key_values)
        full=w(**{k:v for k,v in b.items() if k!='labels'})
        torch.testing.assert_close(step.logits[:,-1],full.logits[:,-1],atol=1e-6,rtol=1e-5)


def test_sidecar_and_injection_location_fail_closed():
    m=tiny()
    with pytest.raises(ValueError,match='subsequent mixer'):
        SIDRelationModel(m,[60,61,62,63],[0,1,2,3],torch.zeros(4,32),enabled=True,layer_idx=1)
    w=wrap(m);b=sample();b.pop('event_ids')
    with pytest.raises(ValueError,match='sidecar'):w(**b)


def test_event_spans_keep_repeated_classes_and_order():
    p='Header\n\n1. [Film] A title\n2. [Film] Another title\n3. [Tech] News\n\nPredict SID.'
    s=event_spans(p)
    assert [c for _,_,c in s]==['Film','Film','Tech']
    assert [p[a:b] for a,b,_ in s]==['1. [Film] A title','2. [Film] Another title','3. [Tech] News']
    with pytest.raises(ValueError):event_spans(p.replace('2. [Film]','4. [Film]'))


def test_runtime_registers_original_tokenizer_not_saved_extended_one(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer,models
    from transformers import PreTrainedTokenizerFast,AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    from new_interest.experiments.sid_relation_bias import load_tokenizer
    names=[f'c{i:03d}' for i in range(108)]
    vocab={name:i for i,name in enumerate(['[UNK]','[PAD]','[EOS]']+names)}
    backend=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    tok.chat_template='{{ messages[0]["content"] }}'
    base=tmp_path/'base';tok.save_pretrained(base)
    project=tmp_path/'project';(project/'inputs').mkdir(parents=True)
    mapping=project/'inputs/interest_token_mapping.parquet'
    pq.write_table(pa.Table.from_pylist([{'interest_index':i,'interest_token':f'<interest_{i:03d}>',
        'target_subcategory':name} for i,name in enumerate(names)]),mapping)
    register_semantic_interest_tokens(tok,mapping);tok.save_pretrained(project/'inputs/tokenizer')
    with pytest.raises(ValueError,match='added 0'):
        register_semantic_interest_tokens(AutoTokenizer.from_pretrained(project/'inputs/tokenizer'),mapping)
    loaded,spec=load_tokenizer(project,base)
    assert loaded.get_vocab()==tok.get_vocab() and len(spec.entries)==108
