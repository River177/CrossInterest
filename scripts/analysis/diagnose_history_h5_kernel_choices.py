"""Bounded golden-output comparison across official fused-norm launch choices."""
import json
import sys
from pathlib import Path


def tuners():
    from triton.runtime.autotuner import Autotuner
    result = {}
    seen = set()
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith('fla.') or module is None:
            continue
        for value in list(vars(module).values()):
            chain_seen = set()
            while value is not None and id(value) not in chain_seen:
                chain_seen.add(id(value))
                if isinstance(value, Autotuner):
                    if id(value) not in seen:
                        seen.add(id(value))
                        fn = value.fn
                        while hasattr(fn, 'fn'):
                            fn = fn.fn
                        name = fn.__module__ + '.' + fn.__name__
                        if value.cache:
                            result[name] = value
                    break
                value = getattr(value, 'fn', None)
    return result


def configuration(config):
    return {'kwargs': config.kwargs, 'num_warps': config.num_warps, 'num_stages': config.num_stages,
            'num_ctas': config.num_ctas, 'maxnreg': config.maxnreg}


def profile(objects):
    return {name: [{'key': list(key), 'config': configuration(cfg)} for key, cfg in obj.cache.items()]
            for name, obj in objects.items()}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from new_interest.training.semantic_id_tokens import (validate_saved_interest_tokens, prepare_semantic_adapter_base,
        apply_torch_numeric_backend, deterministic_sdpa_math_context)
    from new_interest.training.sft_data import apply_chat_template
    from diagnose_history_h5_trainer import tensors
    source=Path('/task/source');out=Path('/task/outputs/kernel_choices');out.mkdir(parents=True,exist_ok=True)
    reference=json.loads((source/'reference.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(source/'adapter')
    spec=validate_saved_interest_tokens(tokenizer,source/'adapter/interest_token_mapping.parquet',expected_count=108)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-2B',revision='15852e8c16360a2fea060d615a32b45270f8a8fc',torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    prepare_semantic_adapter_base(model,spec,initialization_mode='mean_norm_matched')
    model=PeftModel.from_pretrained(model,source/'adapter').to('cuda').eval()
    apply_torch_numeric_backend(torch,reference['torch_numeric_backend'])
    encoded=[]
    for row in reference['samples']:
        text=apply_chat_template(tokenizer,[{'role':'user','content':row['prompt']}],add_generation_prompt=True,enable_thinking=False)
        encoded.append({k:v.to('cuda') for k,v in tokenizer(text,add_special_tokens=False,return_tensors='pt').items()})
    indices=torch.tensor(spec.token_ids,device='cuda')
    target=torch.tensor([x['restricted_logits'] for x in reference['samples']],dtype=torch.float32)
    def score():
        with torch.inference_mode(),deterministic_sdpa_math_context(torch):
            values=torch.stack([model(**x,use_cache=False).logits[0,-1].index_select(0,indices).float().cpu() for x in encoded])
        delta=(values-target).abs()
        assert torch.isfinite(values).all()
        return {'max_error':float(delta.max()),'mismatches':int((delta!=0).sum()),'sum_absolute_error':float(delta.sum())}
    baseline=score();objects=tuners();before=tensors(model)
    (out/'baseline_profile.json').write_text(json.dumps(profile(objects),indent=2,default=str)+'\n')
    baseline_cache={name:dict(obj.cache) for name,obj in objects.items()}
    results=[]
    selected={name:obj for name,obj in objects.items() if 'fwd' in name and ('fused_norm_gate' in name or 'l2norm' in name)}
    for name,obj in selected.items():
        for i,config in enumerate(obj.configs):
            if config.pre_hook is not None:continue
            obj.cache={key:config for key in baseline_cache[name]}
            try:
                result={'kernel':name,'index':i,'config':configuration(config),**score()}
            except Exception as exc:
                result={'kernel':name,'index':i,'error_type':type(exc).__name__}
            results.append(result)
            print('KERNEL_CHOICE',json.dumps(result),flush=True)
            (out/'results.json').write_text(json.dumps({'baseline':baseline,'cases':results},indent=2)+'\n')
            if result.get('max_error')==0:
                assert tensors(model)==before
                (out/'matching_profile.json').write_text(json.dumps(profile(objects),indent=2,default=str)+'\n')
                return
            obj.cache=dict(baseline_cache[name])
    assert tensors(model)==before
    (out/'complete.json').write_text(json.dumps({'matching_profile_found':False,'tested':len(results),'state_unchanged':True},indent=2)+'\n')


if __name__=='__main__':
    main()
