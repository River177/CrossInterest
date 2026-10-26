"""Test training-forward warm state without any optimizer step or parameter update."""
import argparse
import json
import sys
from pathlib import Path


def kernel_caches(clear=False):
    result = {}
    seen = set()
    for name, module in list(sys.modules.items()):
        if not name.startswith('fla.') or module is None:
            continue
        for attr, obj in list(vars(module).items()):
            if id(obj) in seen or not hasattr(obj, 'configs') or not isinstance(getattr(obj, 'cache', None), dict):
                continue
            seen.add(id(obj))
            entries = {}
            for key, value in obj.cache.items():
                if hasattr(value, 'kwargs'):
                    entries[str(key)] = {'kwargs': value.kwargs, 'num_warps': value.num_warps,
                                         'num_stages': value.num_stages}
            if entries:
                result[name + '.' + attr] = entries
            if clear:
                obj.cache.clear()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cold_first','warm_first'), required=True)
    args = parser.parse_args()
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from new_interest.training.semantic_id_tokens import (validate_saved_interest_tokens, prepare_semantic_adapter_base,
        capture_reload_reference, deterministic_reload_numeric_backend, apply_torch_numeric_backend)
    from new_interest.training.sft_data import CompletionOnlyCollator, tokenize_training_example
    from diagnose_history_h5_trainer import tensors
    root = Path('/task/source')
    out = Path('/task/outputs/warm') / args.mode
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(root / 'adapter')
    tokenizer.padding_side = 'right'
    spec = validate_saved_interest_tokens(tokenizer, root / 'adapter/interest_token_mapping.parquet', expected_count=108)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
                                               torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    prepare_semantic_adapter_base(model,spec,initialization_mode='mean_norm_matched')
    model=PeftModel.from_pretrained(model,root/'adapter',is_trainable=True).to('cuda')
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    original=json.loads((root/'reference.json').read_text())
    def capture(label):
        ref=capture_reload_reference(model,tokenizer,root/'train.parquet',spec,enable_thinking=False,sample_count=4)
        (out/f'{label}.json').write_text(json.dumps(ref,indent=2)+'\n')
        (out/f'{label}-kernels.json').write_text(json.dumps(kernel_caches(),indent=2,default=str)+'\n')
        error=max(abs(a-b) for x,y in zip(ref['samples'],original['samples']) for a,b in zip(x['restricted_logits'],y['restricted_logits']))
        print('WARM_STATE',args.mode,label,error,flush=True)
        return ref,error
    reports={}
    if args.mode=='cold_first':reports['cold']=capture('cold')[1]
    before=tensors(model)
    order=torch.randperm(21600,generator=torch.Generator().manual_seed(42)).tolist()[16000:16008]
    rows=pq.read_table(root/'train.parquet',columns=['messages']).take(order).to_pylist()
    examples=[tokenize_training_example(tokenizer,r['messages'],max_seq_length=2048,enable_thinking=False) for r in rows]
    batch={k:v.to('cuda') for k,v in CompletionOnlyCollator(tokenizer)(examples).items()}
    backend=deterministic_reload_numeric_backend()
    backend.update(cuda_matmul_allow_tf32=True,cudnn_allow_tf32=True,float32_matmul_precision='high',
                   cuda_matmul_allow_bf16_reduced_precision_reduction=True,
                   cuda_matmul_allow_fp16_reduced_precision_reduction=True,
                   cudnn_deterministic=False,deterministic_algorithms=False)
    apply_torch_numeric_backend(torch,backend)
    model.train();torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
    loss.backward();model.zero_grad(set_to_none=True)
    after=tensors(model)
    changed={k:{'before':before[k],'after':after[k]} for k in before if before[k]!=after[k]}
    assert not any(v['before']['kind']=='parameter' for v in changed.values()), 'Unexpected parameter change without optimizer'
    reports['after_training_forward']=capture('after_training_forward')[1]
    kernel_caches(clear=True)
    reports['after_cache_clear']=capture('after_cache_clear')[1]
    (out/'summary.json').write_text(json.dumps({'errors':reports,'changed_state':changed,'optimizer_updates':0,
                                              'warm_batch_indices':order,'warm_loss':float(loss)},indent=2)+'\n')


if __name__=='__main__':
    main()
