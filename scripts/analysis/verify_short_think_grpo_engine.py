"""Independent bytes, optimizer, sample-cursor and reward audit of GRPO oracle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def tensor_digest(tensors):
    h=hashlib.sha256()
    for name in sorted(tensors):
        value=tensors[name].detach().cpu().contiguous()
        if not torch.isfinite(value).all():raise ValueError('Nonfinite state tensor')
        item={'shape':list(value.shape),'dtype':str(value.dtype),'numel':value.numel(),
              'sha256':hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        h.update(json.dumps([name,item],sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()


def verify(bundle,result):
    contract=read(bundle/'engine_contract.json');identity=sha(bundle/'engine_contract.json')
    manifest=read(result/'artifact_manifest.json');assert manifest['engine_identity']==identity
    for name,digest in manifest['files'].items():assert sha(result/name)==digest,name
    records=pq.read_table(bundle/'inputs/train.parquet').to_pylist()
    order=list(range(21600));random.Random(42).shuffle(order)
    order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    tokenizer=AutoTokenizer.from_pretrained(bundle/'inputs/short_sft_adapter',local_files_only=True)
    initial=tensor_digest(load_file(str(bundle/'inputs/short_sft_adapter/adapter_model.safetensors')))
    assert initial==contract['initial_adapter_tensor_state_sha256']
    stop_ids={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}
    grammar=re.compile(r'\A(?P<think>\s*\S[\s\S]*?)</think>\s*<answer>\s*(?P<sid><interest_\d{3}>)\s*</answer>\s*\Z')
    states=[];all_ids=[];timing=[];previous=None
    for step in [1,2]:
        folder=result/f'checkpoint-{step}'
        cp=read(folder/'checkpoint_manifest.json');assert cp['identity']==identity
        for name,digest in cp['files'].items():assert sha(folder/name)==digest
        state=read(folder/'state.json');weights=load_file(str(folder/'adapter/adapter_model.safetensors'))
        assert len(weights)==373 and state['identity']==identity and state['step']==step
        assert tensor_digest(weights)==state['adapter_state_sha256']
        assert state['adapter_state_sha256']!=(initial if previous is None else previous['adapter_state_sha256'])
        assert state['rank_state_hashes']==[state['adapter_state_sha256']]*4
        assert state['order_sha256']==order_hash and state['consumed_prompts']==step*32 and state['consumed_trajectories']==step*256
        assert state['reference_sha256']==initial
        opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=False)
        ids=[i for group in opt['param_groups'] for i in group['params']]
        names=state['trainable_parameter_names'];assert len(ids)==len(set(ids))==len(names)==len(opt['state'])==373
        tensors={}
        for index,name in zip(ids,names):
            entry=opt['state'][index];assert int(entry['step'])==step
            normalized=name.replace('.default.','.')
            if normalized.endswith('.default'):normalized=normalized[:-len('.default')]
            assert normalized in weights
            for slot in ['exp_avg','exp_avg_sq']:assert entry[slot].shape==weights[normalized].shape
            assert (entry['exp_avg_sq']>=0).all()
            for key,value in entry.items():tensors[f'{name}/{key}']=value
        groups=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']]
        opt_hash=hashlib.sha256(json.dumps({'tensors':tensor_digest(tensors),'groups':groups},sort_keys=True).encode()).hexdigest()
        assert opt_hash==state['optimizer_state_sha256']
        assert all(g['lr']==5e-6 and g['weight_decay']==0 and tuple(g['betas'])==(.9,.999) for g in groups)
        action_stats=[]
        for rank in range(4):
            start=read(folder/f'start-{rank}.json');metric=read(folder/f'step-{step:04d}-rank-{rank}.json')
            assert start['identity']==identity and start['start_step']==step-1 and start['stop_at']==step
            assert start['world_size']==4 and start['dropout_disabled'] and start['reference_sha256']==initial
            assert start['policy_sha256']==(initial if previous is None else previous['adapter_state_sha256'])
            assert start['optimizer_states']==(0 if step==1 else 373)
            if step==2:assert start['resume_probe_exact'] and start['optimizer_state_sha256']==previous['optimizer_state_sha256']
            expected=order[(step-1)*32+rank*8:(step-1)*32+(rank+1)*8]
            assert metric['train_indices']==expected and metric['reference_state_unchanged'] and metric['trajectories']==64
            assert metric['source_ids']==[records[i]['extra_info']['source_sample_id'] for i in expected]
            assert math.isfinite(metric['gradient_norm']) and metric['gradient_norm']>0
            all_ids.extend(metric['source_ids']);timing.append({'step':step,'rank':rank,'seconds':metric['elapsed_seconds'],'generation_seconds':metric['generation_seconds'],'peak_bytes':metric['max_cuda_allocated_bytes']})
            rng=torch.load(folder/f'rng-{rank}.pt',map_location='cpu',weights_only=False)
            assert rng['torch'].dtype==torch.uint8 and rng['cuda'].dtype==torch.uint8
            random.Random().setstate(rng['python'])
            rows=pq.read_table(folder/f'rollout-{step-1:04d}-rank-{rank}.parquet').to_pylist();assert len(rows)==64
            for local,index in enumerate(expected):
                info=records[index]['extra_info'];local_rows=rows[local*8:(local+1)*8];rs=[]
                prompt=tokenizer.apply_chat_template(records[index]['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                prefix=tokenizer.encode(prompt,add_special_tokens=False)
                token_ids={tokenizer.convert_tokens_to_ids(t) for t in info['interest_tokens']};assert len(token_ids)==108
                parents=dict(zip(info['interest_tokens'],info['interest_categories']));labels=dict(zip(info['interest_tokens'],info['interest_subcategories']))
                for generation,row in enumerate(local_rows):
                    assert row['train_index']==index and row['source_sample_id']==info['source_sample_id'] and row['generation_index']==generation
                    assert row['prompt_ids']==prefix and row['step']==step-1 and row['rank']==rank
                    if contract.get('rollout_batch_size_per_gpu')==64:
                        seed=42+((step-1)*4+rank)*100000
                        assert row['sampling_layout']=='rank64-phase-batch-v1' and row['rollout_seed']==seed and row['answer_seed']==seed+50000
                    response=row['response_ids'];n=row['thought_length'];assert 0<n<=480 and 0<=len(response)-n<=32
                    assert len(response)==len(row['behavior_logprobs']) and all(math.isfinite(x) and x<=1e-5 for x in row['behavior_logprobs'])
                    assert not set(response[:n])&token_ids
                    answer=response[n:]
                    for j,t in enumerate(answer):
                        if re.fullmatch(r'\s*<answer>\s*',tokenizer.decode(answer[:j],skip_special_tokens=False)):assert t in token_ids
                    natural=response[-1] in stop_ids
                    text=tokenizer.decode(response[:-1] if natural else response,skip_special_tokens=False)
                    match=grammar.fullmatch(text) if natural else None
                    valid=bool(match and '<think>' not in match['think'] and not re.search(r'<interest_\d{3}>',match['think']) and match['sid'] in parents)
                    assert row['valid']==valid
                    sid=match['sid'] if valid else None;exact=bool(valid and sid==info['target_token']);seen=bool(valid and labels[sid] in info['history_subcategories']);same=bool(valid and parents[sid]==info['target_category'])
                    if not valid:outcome,h='invalid',-.2
                    elif seen:outcome,h='history_seen',-.1
                    elif exact:outcome,h='exact',1.
                    elif same:outcome,h='novel_same_parent',.2
                    else:outcome,h='novel_other_parent',0.
                    penalty=.1*min(1,max(0,(len(tokenizer.encode(match['think'],add_special_tokens=False))-160)/160)) if exact else 0.
                    assert row['reward']=={'outcome':outcome,'H':h,'E':float(exact),'HL':h-penalty,'length_penalty':penalty,'exact':exact,'history_seen':seen,'same_parent':same}
                    rs.append(h)
                mean=statistics.mean(rs);scale=max(statistics.pstdev(rs),.1)
                assert all(abs(row['advantage']-(r-mean)/scale)<1e-5 for row,r in zip(local_rows,rs))
            action_stats.extend(metric['microbatches'])
        total=sum(x['actions'] for x in action_stats)
        assert total==sum(len(r['response_ids']) for rank in range(4) for r in pq.read_table(folder/f'rollout-{step-1:04d}-rank-{rank}.parquet').to_pylist())
        assert all(math.isfinite(x['loss']) and math.isfinite(x['kl']) for x in action_stats)
        if step==1:
            assert all(x['kl']==0 for x in action_stats)
            assert sum(x['clip_fraction']*x['actions'] for x in action_stats)/total<=.01
            assert sum(x['support_miss_fraction']*x['actions'] for x in action_stats)/total<=.005
        states.append(state);previous=state
    assert len(all_ids)==len(set(all_ids))==64
    fresh=read(result/'fresh_reload.json');assert fresh['passed'] and fresh['fresh_probe_max_abs']==0 and fresh['state_sha256']==states[-1]['adapter_state_sha256'] and fresh['optimizer_state_sha256']==states[-1]['optimizer_state_sha256']
    hydration=read(result/'parent_hydration.json');assert hydration['all_bytes_verified'] and hydration['parent_manifest_sha256']==sha(result/'checkpoint-1/checkpoint_manifest.json')
    recovery_verified=False
    if (result/'recovery_identity.json').exists():
        recovery=read(result/'recovery_identity.json');recovered=read(bundle/'recovered_probe_plan.json')
        assert recovery['source_checkpoint_manifest_sha256']==sha(result/'checkpoint-1/checkpoint_manifest.json')
        assert recovery['recovered_plan_sha256']==sha(bundle/'recovered_probe_plan.json')
        assert recovered['original_probe']==states[0]['reload_probe']
        for rank in range(4):
            audit=read(result/f'checkpoint-2/resume-probe-{rank}.json')
            assert audit['original_reference']==audit['actual']==states[0]['reload_probe'] and audit['max_abs']==0
            assert audit['execution_plan']==recovered['execution_plan']
        assert states[1]['probe_execution_plan']['version']=='fla-probe-execution-plan-v1'
        completed=read(result/'engine_complete.json')
        assert completed['new_optimizer_updates_this_job']==1 and not completed['first_update_retrained']
        recovery_verified=True
    return {'passed':True,'engine_identity':identity,'artifact_files':len(manifest['files']),'artifact_bytes':sum((result/n).stat().st_size for n in manifest['files']),
        'actual_updates':2,'prompts':64,'trajectories':512,'exact_storage_model_optimizer_resume':True,'fresh_reload_max_abs':0,
        'execution_plan_recovery_verified':recovery_verified,
        'final_adapter_state_sha256':states[-1]['adapter_state_sha256'],'timings':timing,'verifier_sha256':sha(Path(__file__)),
        'scope':'Independent native-HF correctness oracle acceptance; not formal production or recommendation quality acceptance'}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['bundle','result','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();report=verify(a.bundle,a.result);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
