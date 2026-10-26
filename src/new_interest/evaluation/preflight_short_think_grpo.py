"""No-update, four-shard reward-density and numerical-path preflight.

Stage one stops on a model-generated </think>; stage two gets its own budget.
No token is injected. Raw sampled tokens, stage boundaries, rewards and selected
token log-probs are retained. A preflight completion never authorizes training
without the separate coordinator acceptance and training-engine checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import _atomic_write_json, _atomic_write_parquet
from new_interest.evaluation.evaluate_short_think_ranking import analyze_generation
from new_interest.training.short_think_grpo_reward import group_statistics, score
from new_interest.training.sft_data import apply_chat_template


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(root, files):
    for name, expected in files.items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f'File identity differs: {name}')


def through_first_stop(ids, stops):
    for i, value in enumerate(ids):
        if value in stops:
            return ids[:i + 1], True
    return ids, False


def selected_logprobs(logits, targets):
    """Bound full-vocabulary float32 temporaries to sixteen positions."""
    import torch
    result=[]
    for start in range(0, targets.shape[1], 16):
        values=logits[:, start:start+16].float()
        chosen=targets[:, start:start+16]
        result.append(values.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)-torch.logsumexp(values, dim=-1))
    return torch.cat(result, dim=1)


def stage(model, tokenizer, prefixes, stops, budget, seed, parity, *, phase=None, sid_ids=None,capture_behavior=False):
    import torch
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
    from new_interest.training.short_think_grpo_sampling import PhaseSIDProcessor
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    encoded=tokenizer.pad({'input_ids':prefixes}, padding=True, return_tensors='pt')
    encoded={k:v.to(model.device) for k,v in encoded.items()}
    width=encoded['input_ids'].shape[1]
    processors=LogitsProcessorList([PhaseSIDProcessor(tokenizer,phase=phase,prefix_width=width,sid_ids=sid_ids)] if phase else [])
    with torch.inference_mode():
        output=model.generate(**encoded, do_sample=True, temperature=.8, top_p=.95, top_k=0,
            repetition_penalty=1., max_new_tokens=budget, eos_token_id=sorted(stops),
            pad_token_id=tokenizer.pad_token_id, use_cache=True, return_dict_in_generate=True,
            output_logits=parity, output_scores=parity or capture_behavior,logits_processor=processors)
    continuation=output.sequences[:,width:]
    trimmed=[through_first_stop(x, stops) for x in continuation.tolist()]
    behavior_rows=None
    if parity or capture_behavior:
        # Reduce the vocabulary for the whole batch, then transfer one compact
        # matrix. Per-token .item() would serialize thousands of GPU reductions.
        matrix=torch.stack([scores.float().gather(-1,continuation[:,j:j+1]).squeeze(-1)
            -torch.logsumexp(scores.float(),dim=-1) for j,scores in enumerate(output.scores)],dim=1).cpu()
        behavior_rows=[]
        for i,(ids,_) in enumerate(trimmed):
            values=matrix[i,:len(ids)]
            if not torch.isfinite(values).all():raise ValueError('Sampled action has nonfinite behavior logprob')
            behavior_rows.append(values.tolist())
    diagnostic=[]
    if parity:
        # Compare the raw model policy, before sampling temperature/top-p. Do not
        # mistake these for the truncated sampling distribution's log-probability.
        cached=torch.stack([v.float().gather(-1,continuation[:,j:j+1]).squeeze(-1)
                            -torch.logsumexp(v.float(),dim=-1) for j,v in enumerate(output.logits)],dim=1)
        attention=torch.cat([encoded['attention_mask'],torch.zeros_like(continuation)],dim=1)
        for i,(ids,_) in enumerate(trimmed): attention[i,width:width+len(ids)]=1
        with torch.inference_mode():
            logits=model(input_ids=output.sequences,attention_mask=attention,use_cache=False,
                logits_to_keep=continuation.shape[1]+1).logits[:,:-1]
            fresh=selected_logprobs(logits,continuation)
            fresh_behavior=[]
            for j in range(continuation.shape[1]):
                scores=processors(output.sequences[:,:width+j],logits[:,j].float())
                scores=TemperatureLogitsWarper(.8)(None,scores)
                scores=TopPLogitsWarper(.95)(None,scores)
                fresh_behavior.append(scores.gather(-1,continuation[:,j:j+1]).squeeze(-1)-torch.logsumexp(scores,dim=-1))
            fresh_behavior=torch.stack(fresh_behavior,dim=1)
            del logits
            logits=model(input_ids=output.sequences,attention_mask=attention,use_cache=False,
                logits_to_keep=continuation.shape[1]+1).logits[:,:-1]
            repeated=selected_logprobs(logits,continuation)
            del logits
        if not torch.isfinite(cached).all() or not torch.isfinite(fresh).all():
            raise ValueError('Nonfinite selected log probability')
        for i,(ids,_) in enumerate(trimmed):
            length=len(ids)
            behavior=behavior_rows[i]
            fb=fresh_behavior[i,:length].cpu().tolist()
            diagnostic.append({'prefix_ids':prefixes[i], 'generated_ids':ids,
                'cached_raw_logprobs':cached[i,:length].cpu().tolist(),
                'fresh_raw_logprobs':fresh[i,:length].cpu().tolist(),
                'fresh_repeat_max_abs':float((fresh[i,:length]-repeated[i,:length]).abs().max()),
                'cached_behavior_logprobs':behavior,
                'fresh_behavior_logprobs':[x if x!=float('-inf') else None for x in fb],
                'fresh_behavior_support_misses':sum(x==float('-inf') for x in fb),
                'phase_constraint':phase,
                'sampling_distribution':'temperature=.8,top_p=.95,top_k=0; raw logprobs are not behavior logprobs'})
    if capture_behavior and not parity:
        for i,(ids,_) in enumerate(trimmed):
            diagnostic.append({'generated_ids':ids,'cached_behavior_logprobs':behavior_rows[i],'phase_constraint':phase})
    return trimmed,diagnostic


def evaluate(project, root, shard_index):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from new_interest.training.semantic_id_tokens import prepare_semantic_adapter_base, validate_saved_interest_tokens
    project=Path(project); root=Path(root)
    contract=read(project/'contract.json'); identity=sha(project/'contract.json')
    verify_files(project,read(project/'source_manifest.json')['files'])
    adapter=project/'inputs/short_sft_adapter'
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    spec=validate_saved_interest_tokens(tokenizer,project/'inputs/interest_token_mapping.parquet',expected_count=108)
    close=tokenizer.encode('</think>',add_special_tokens=False)
    if len(close)!=1: raise ValueError('Expected single-token think closure')
    stops={int(tokenizer.eos_token_id),int(tokenizer.convert_tokens_to_ids('<|im_end|>'))}
    if close[0] in stops: raise ValueError('Think closure collides with EOS')
    model=AutoModelForCausalLM.from_pretrained((root/'model_path.txt').read_text().strip(),
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    prepare_semantic_adapter_base(model,spec)
    model=PeftModel.from_pretrained(model,adapter).to(torch.device('cuda',0)).eval()
    expected=load_file(str(adapter/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(model)
    if set(expected)!=set(actual) or len(actual)!=373:
        raise ValueError('Inherited adapter tensor inventory differs')
    for name,value in expected.items():
        if not torch.equal(actual[name].cpu(),value.to(actual[name].dtype)):
            raise ValueError(f'Inherited adapter tensor differs: {name}')
    shard=root/f'shard-{shard_index}'; shard.mkdir(parents=True,exist_ok=True)
    _atomic_write_json(shard/'model_audit.json',{'contract_sha256':identity,
        'adapter_file_sha256':sha(adapter/'adapter_model.safetensors'),
        'inherited_tensor_count':373,'all_tensors_exact_after_dtype_cast':True,
        'gpu':torch.cuda.get_device_name(0),'model_updates':0})
    records=pq.read_table(project/'inputs/preflight.parquet').to_pylist()
    token_by_id=dict(zip(spec.token_ids,[e.interest_token for e in spec.entries]))
    result=[]; started=time.monotonic()
    for group_index,record in enumerate(records):
        if group_index % 4 != shard_index: continue
        target=record['extra_info']['target_token']
        info=record['extra_info']
        parents=dict(zip(info['interest_tokens'],info['interest_categories']))
        by_label=dict(zip(info['interest_subcategories'],info['interest_tokens']))
        history=[by_label[x] for x in info['history_subcategories'] if x in by_label]
        prompt=apply_chat_template(tokenizer,record['prompt'],add_generation_prompt=True,enable_thinking=True)
        if not prompt.endswith('<think>\n'): raise ValueError('Native think prefix differs')
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        if len(prefix)>contract['max_prompt_tokens']: raise ValueError('Overlong prompt; no filtering allowed')
        path=shard/f'group-{group_index:03d}.parquet'
        parity_path=shard/f'parity-{group_index:03d}.json'
        if path.exists():
            rows=pq.read_table(path).to_pylist()
            if len(rows)!=16 or any(r['contract_sha256']!=identity or r['source_sample_id']!=info['source_sample_id'] for r in rows):
                raise ValueError('Resumed group identity differs')
            if group_index<4 and not parity_path.exists(): raise ValueError('Committed first group has no parity audit')
            result.extend(rows);continue
        rows=[]; parity_rows=[]
        for start in [0,8]:
            parity=group_index<4 and start==0
            constrained=contract.get('rollout_constraints')=='phase_sid_v1'
            think,diag=stage(model,tokenizer,[prefix]*8,stops|{close[0]},480,
                             contract['seed']+group_index*100+start,parity,
                             phase='thought' if constrained else None,sid_ids=spec.token_ids)
            for j,d in enumerate(diag): parity_rows.append({**d,'stage':'thought','generation_index':start+j})
            closed=[i for i,(ids,ended) in enumerate(think) if ended and ids[-1]==close[0]]
            answers={}
            if closed:
                answer,diag=stage(model,tokenizer,[prefix+think[i][0] for i in closed],stops,32,
                                  contract['seed']+group_index*100+start+50,parity,
                                  phase='answer' if constrained else None,sid_ids=spec.token_ids)
                answers=dict(zip(closed,answer))
                for j,d in zip(closed,diag): parity_rows.append({**d,'stage':'answer','generation_index':start+j})
            for i,(thought,thought_ended) in enumerate(think):
                answer,answer_ended=answers.get(i,([],False))
                generated=thought+answer
                row=analyze_generation(tokenizer,generated,stop_ids=stops,token_by_id=token_by_id)
                exhausted=(not thought_ended) or (i in answers and not answer_ended)
                if exhausted: row['valid_format']=False
                row.update(source_sample_id=info['source_sample_id'],group_index=group_index,
                    generation_index=start+i,contract_sha256=identity,prompt_token_ids=prefix,
                    rollout_token_ids=generated,thought_stage_tokens=len(thought),answer_stage_tokens=len(answer),
                    thought_closed=i in answers,budget_exhausted=exhausted,target_token=target,
                    response_mask=[1]*len(generated),prompt_mask=[0]*len(prefix))
                row['reward']=score(valid=row['valid_format'],predicted=row['generated_sid'],target=target,
                    history=history,parent_by_token=parents,thought_tokens=row['think_tokens'])
                rows.append(row)
        if group_index<4: _atomic_write_json(parity_path,{'contract_sha256':identity,'rows':parity_rows})
        _atomic_write_parquet(path,rows)
        result.extend(rows)
        print(f'GRPO_PREFLIGHT shard={shard_index} groups={len(result)//16}/81 trajectories={len(result)} elapsed={time.monotonic()-started:.1f}',flush=True)
    statistics=group_statistics(result,generations=16)
    if statistics['groups']!=81: raise ValueError('Missing shard groups')
    _atomic_write_json(shard/'statistics.json',statistics)
    _atomic_write_json(shard/'complete.json',{'contract_sha256':identity,'groups':81,'trajectories':1296})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shard-index',type=int,choices=range(4),required=True)
    args=parser.parse_args();evaluate(args.project,args.root,args.shard_index)


if __name__=='__main__': main()
