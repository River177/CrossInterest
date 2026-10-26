import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
from peft import LoraConfig,get_peft_model
from transformers import LlamaConfig,LlamaForCausalLM

from new_interest.training.semantic_id_rollout_export import export_merged_sid_state,merged_sid_lora_context


def make_model():
    torch.manual_seed(42)
    config=LlamaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True)
    model=get_peft_model(LlamaForCausalLM(config),LoraConfig(task_type='CAUSAL_LM',r=2,
        lora_alpha=4,target_modules=['q_proj','v_proj'],trainable_token_indices={'embed_tokens':[126,127]})).eval()
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'trainable_tokens_delta' in name:value.copy_(torch.arange(value.numel()).reshape(value.shape)/1000)
            if 'lora_B' in name:value.fill_(.02)
    return config,model


def test_real_tied_peft_rows_export_and_actor_restores_exactly():
    config,model=make_model()
    original={k:v.clone() for k,v in model.state_dict().items()}
    export=export_merged_sid_state(model)
    assert not any('token_adapter' in k or 'lora_' in k for k in export)
    delta=next(v for k,v in original.items() if 'trainable_tokens_delta' in k)
    torch.testing.assert_close(export['model.embed_tokens.weight'][[126,127]],delta,atol=0,rtol=0)
    torch.testing.assert_close(export['lm_head.weight'],export['model.embed_tokens.weight'],atol=0,rtol=0)
    plain=LlamaForCausalLM(config).eval()
    plain.load_state_dict(export,strict=True)
    ids=torch.tensor([[3,126,7,127]])
    with torch.no_grad():
        torch.testing.assert_close(plain(ids).logits,model(ids).logits,atol=1e-6,rtol=1e-6)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,original[name],atol=0,rtol=0)
    # Repeated exports cannot accumulate merge/unmerge rounding or return aliases.
    again=export_merged_sid_state(model)
    for name in export:torch.testing.assert_close(again[name],export[name],atol=0,rtol=0)


def test_failure_inside_context_restores_live_actor():
    _,model=make_model();before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(RuntimeError,match='injected'):
        with merged_sid_lora_context(model):raise RuntimeError('injected transfer failure')
    for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    # Failure must not leave a hidden merged flag behind.
    export_merged_sid_state(model)
