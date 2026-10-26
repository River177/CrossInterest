"""Independent formal GRPO checkpoint-chain and Dev metric verification.

Supply all ancestor segment roots explicitly. A missing ancestor, incomplete
checkpoint, or conflicting same-step publication is an error, never a skipped
portion of the exposure ledger. Without --evaluation-root this accepts only the
durable training prefix, not a completed experiment.
"""
import argparse
from collections import Counter,defaultdict
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


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def tensor_digest(values):
    digest=hashlib.sha256()
    for name in sorted(values):
        x=values[name].detach().cpu().contiguous()
        if not torch.isfinite(x).all():raise ValueError('Nonfinite checkpoint tensor')
        info={'shape':list(x.shape),'dtype':str(x.dtype),'numel':x.numel(),
              'sha256':hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        digest.update(json.dumps([name,info],sort_keys=True,separators=(',',':')).encode())
    return digest.hexdigest()


def files_match(folder,files):
    for name,digest in files.items():
        if sha(folder/name)!=digest:raise ValueError(f'Artifact mismatch: {folder/name}')


def resolve_chain(roots,identity,end,allowed_parent=None):
    index={};by_hash={}
    permitted={identity}
    if allowed_parent:permitted.add(allowed_parent['identity'])
    for root in roots:
        for path in root.glob('checkpoint-*/checkpoint_manifest.json'):
            manifest=read(path);state=read(path.parent/'state.json')
            if manifest['identity'] not in permitted:raise ValueError('Unrelated arm in supplied roots')
            if state['identity']!=manifest['identity']:raise ValueError('State/manifest identity mismatch')
            key=(state['identity'],state['step']);digest=sha(path)
            if key in index and index[key]['digest']!=digest:raise ValueError('Conflicting checkpoint publications')
            item={'path':path.parent,'root':root,'state':state,'manifest':manifest,'digest':digest}
            index[key]=item;by_hash[digest]=item
    if (identity,end) not in index:raise ValueError('Requested complete checkpoint is missing')
    chain=[];current=index[(identity,end)];seen=set()
    while True:
        if current['digest'] in seen:raise ValueError('Checkpoint ancestry cycle')
        seen.add(current['digest']);chain.append(current)
        start=read(current['path']/'start-0.json')['start_step']
        if start==0:break
        reference_path=current['root']/'initial_parent_reference.json'
        reference=read(reference_path) if reference_path.exists() else None
        if reference and reference['step']==start:
            parent=by_hash.get(reference['manifest_sha256'])
            if parent is None:raise ValueError(f'Missing explicit ancestor: {reference["path"]}')
        else:parent=index.get((current['state']['identity'],start))
        if parent is None:raise ValueError(f'Missing ancestor checkpoint at step {start}')
        if parent['state']['step']!=start:raise ValueError('Parent step mismatch')
        if parent['state']['identity']!=current['state']['identity']:
            if (not allowed_parent or start!=100 or parent['state'].get('arm')!='H'
                or current['state'].get('arm')!='HL' or parent['state']['identity']!=allowed_parent['identity']
                or sha(parent['path']/'state.json')!=allowed_parent['state_file_sha256']):
                raise ValueError('Unapproved cross-arm ancestry')
        current=parent
    return list(reversed(chain))


GRAMMAR=re.compile(r'\A(?P<think>\s*\S[\s\S]*?)</think>\s*<answer>\s*(?P<sid><interest_\d{3}>)\s*</answer>\s*\Z')


def verify_group(rows,record,index,step,rank,arm,tokenizer):
    info=record['extra_info'];tokens=info['interest_tokens']
    token_ids={tokenizer.convert_tokens_to_ids(t) for t in tokens};assert len(token_ids)==108
    parents=dict(zip(tokens,info['interest_categories']));labels=dict(zip(tokens,info['interest_subcategories']))
    assert labels[info['target_token']] not in info['history_subcategories']
    prompt=tokenizer.apply_chat_template(record['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    stops={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}
    rewards=[]
    for generation,row in enumerate(rows):
        assert row['source_sample_id']==info['source_sample_id'] and row['train_index']==index and row['generation_index']==generation
        assert row['step']==step-1 and row['rank']==rank and row['prompt_ids']==prefix
        seed=42+((step-1)*4+rank)*100000
        assert row['rollout_seed']==seed and row['answer_seed']==seed+50000 and row['sampling_layout']=='rank64-phase-batch-v1'
        response=row['response_ids'];n=row['thought_length'];assert 0<n<=480 and 0<=len(response)-n<=32
        assert len(response)==len(row['behavior_logprobs']) and all(math.isfinite(x) and x<=1e-5 for x in row['behavior_logprobs'])
        assert not set(response[:n])&token_ids
        answer=response[n:]
        for j,t in enumerate(answer):
            if re.fullmatch(r'\s*<answer>\s*',tokenizer.decode(answer[:j],skip_special_tokens=False)):assert t in token_ids
        natural=response[-1] in stops
        match=GRAMMAR.fullmatch(tokenizer.decode(response[:-1] if natural else response,skip_special_tokens=False)) if natural else None
        valid=bool(match and match['sid'] in parents and '<think>' not in match['think'] and not re.search(r'<interest_\d{3}>',match['think']))
        assert row['valid']==valid
        sid=match['sid'] if valid else None
        exact=bool(valid and sid==info['target_token']);seen=bool(valid and labels[sid] in info['history_subcategories'])
        same=bool(valid and parents[sid]==parents[info['target_token']])
        if not valid:outcome,h='invalid',-.2
        elif seen:outcome,h='history_seen',-.1
        elif exact:outcome,h='exact',1.
        elif same:outcome,h='novel_same_parent',.2
        else:outcome,h='novel_other_parent',0.
        penalty=.1*min(1,max(0,(len(tokenizer.encode(match['think'],add_special_tokens=False))-160)/160)) if exact else 0.
        expected={'outcome':outcome,'H':h,'E':float(exact),'HL':h-penalty,'length_penalty':penalty,'exact':exact,'history_seen':seen,'same_parent':same}
        assert row['reward']==expected
        column='H' if arm=='HL' and step<=100 else arm
        rewards.append(expected[column])
    mean=statistics.mean(rewards);scale=max(statistics.pstdev(rewards),.1)
    assert all(abs(row['advantage']-(r-mean)/scale)<1e-5 for row,r in zip(rows,rewards))


def verify_training(bundle,roots,end):
    contract=read(bundle/'engine_contract.json');identity=sha(bundle/'engine_contract.json')
    assert contract['role']=='production' and end%25==0 and 0<end<=675
    records=pq.read_table(bundle/'inputs/train.parquet').to_pylist();assert len(records)==21600
    order=list(range(21600));random.Random(42).shuffle(order)
    order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    tokenizer=AutoTokenizer.from_pretrained(bundle/'inputs/short_sft_adapter',local_files_only=True)
    initial=tensor_digest(load_file(str(bundle/'inputs/short_sft_adapter/adapter_model.safetensors')))
    assert initial==contract['initial_adapter_tensor_state_sha256']
    chain=resolve_chain(roots,identity,end,contract.get('allowed_parent'))
    previous=None;total_actions=0;updates=[];accepted=[]
    for item in chain:
        folder=item['path'];state=item['state'];stop=state['step'];arm=state['arm'];cid=state['identity']
        start=0 if previous is None else previous['step']
        assert stop-start==25 and state['consumed_prompts']==stop*32 and state['consumed_trajectories']==stop*256
        assert state['order_sha256']==order_hash and state['reference_sha256']==initial
        files_match(folder,item['manifest']['files'])
        weights=load_file(str(folder/'adapter/adapter_model.safetensors'));assert len(weights)==373
        assert tensor_digest(weights)==state['adapter_state_sha256'] and state['rank_state_hashes']==[state['adapter_state_sha256']]*4
        assert state['probe_execution_plan']['version']=='fla-probe-execution-plan-v1'
        opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=False)
        ids=[i for group in opt['param_groups'] for i in group['params']];names=state['trainable_parameter_names']
        assert len(ids)==len(set(ids))==len(names)==len(opt['state'])==373
        tensors={}
        for index,name in zip(ids,names):
            entry=opt['state'][index];assert int(entry['step'])==stop
            normalized=name.replace('.default.','.')
            if normalized.endswith('.default'):normalized=normalized[:-len('.default')]
            assert normalized in weights and (entry['exp_avg_sq']>=0).all()
            for slot in ['exp_avg','exp_avg_sq']:assert entry[slot].shape==weights[normalized].shape
            tensors.update({f'{name}/{k}':v for k,v in entry.items()})
        groups=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']]
        opt_hash=hashlib.sha256(json.dumps({'tensors':tensor_digest(tensors),'groups':groups},sort_keys=True).encode()).hexdigest()
        assert opt_hash==state['optimizer_state_sha256']
        assert all(g['lr']==5e-6 and g['weight_decay']==0 and tuple(g['betas'])==(.9,.999) for g in groups)
        for rank in range(4):
            header=read(folder/f'start-{rank}.json')
            assert header['identity']==cid and header['start_step']==start and header['stop_at']==stop
            assert header['world_size']==4 and header['dropout_disabled'] and header['reference_sha256']==initial
            assert header['policy_sha256']==(initial if previous is None else previous['adapter_state_sha256'])
            if previous is not None:
                assert header['optimizer_states']==373 and header['optimizer_state_sha256']==previous['optimizer_state_sha256'] and header['resume_probe_exact']
                replay=read(folder/f'resume-probe-{rank}.json')
                assert replay['original_reference']==replay['actual']==previous['reload_probe'] and replay['max_abs']==0
                assert replay['execution_plan']==previous['probe_execution_plan']
            else:assert header['optimizer_states']==0
            rng=torch.load(folder/f'rng-{rank}.pt',map_location='cpu',weights_only=False)
            assert rng['torch'].dtype==rng['cuda'].dtype==torch.uint8
            random.Random().setstate(rng['python'])
        for step in range(start+1,stop+1):
            stats=[];step_actions=0;times=[]
            for rank in range(4):
                metric=read(folder/f'step-{step:04d}-rank-{rank}.json')
                expected=order[(step-1)*32+rank*8:(step-1)*32+(rank+1)*8]
                assert metric['identity']==cid and metric['step']==step and metric['train_indices']==expected
                assert metric['source_ids']==[records[i]['extra_info']['source_sample_id'] for i in expected]
                assert metric['arm']==arm and metric['reward_column']==('H' if arm=='HL' and step<=100 else arm)
                assert metric['trajectories']==64 and metric['reference_state_unchanged'] and math.isfinite(metric['gradient_norm']) and metric['gradient_norm']>0
                rows=pq.read_table(folder/f'rollout-{step-1:04d}-rank-{rank}.parquet').to_pylist();assert len(rows)==64
                for local,index in enumerate(expected):verify_group(rows[local*8:(local+1)*8],records[index],index,step,rank,arm,tokenizer)
                step_actions+=sum(len(r['response_ids']) for r in rows);stats.extend(metric['microbatches']);times.append(metric['elapsed_seconds'])
            assert sum(x['actions'] for x in stats)==step_actions
            assert all(math.isfinite(x['loss']) and math.isfinite(x['kl']) for x in stats)
            if step==1:
                assert all(x['kl']==0 for x in stats)
                assert sum(x['actions']*x['clip_fraction'] for x in stats)/step_actions<=.01
                assert sum(x['actions']*x['support_miss_fraction'] for x in stats)/step_actions<=.005
            total_actions+=step_actions;updates.append({'step':step,'arm':arm,'seconds':max(times),'actions':step_actions})
        accepted.append({'identity':cid,'arm':arm,'step':stop,'manifest_sha256':item['digest'],'state_file_sha256':sha(folder/'state.json'),
            'adapter_state_sha256':state['adapter_state_sha256'],'optimizer_state_sha256':opt_hash,'checkpoint':str(folder)})
        previous=state
    assert [u['step'] for u in updates]==list(range(1,end+1))
    return {'training_chain_passed':True,'engine_identity':identity,'arm':contract['arm'],'end_step':end,'planned_steps':675,
        'verified_prompts':end*32,'verified_trajectories':end*256,'verified_generated_actions':total_actions,
        'checkpoints':accepted,'updates':updates,'final_state':previous,'full_experiment_complete':False}


def directory_digest(folder):
    h=hashlib.sha256()
    for path in sorted(p for p in folder.rglob('*') if p.is_file()):
        relative=path.relative_to(folder).as_posix().encode()
        h.update(len(relative).to_bytes(8,'big'));h.update(relative);h.update(bytes.fromhex(sha(path)))
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def verify_evaluation(bundle,evaluation,adapter,full):
    data=bundle/'inputs'/('dev.parquet' if full else 'diagnostic_dev.parquet')
    records=pq.read_table(data).to_pylist();expected=31392 if full else 2048
    assert len(records)==expected and len({r['impression_id'] for r in records})==expected
    mapping=pq.read_table(bundle/'inputs/interest_token_mapping.parquet').to_pylist()
    by_label={r['target_subcategory']:r['interest_token'] for r in mapping};assert len(by_label)==108
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
    token_by_id={tokenizer.convert_tokens_to_ids(t):t for t in by_label.values()}
    combined={};shard_hashes={}
    for rank in range(4):
        folder=evaluation/f'shard-{rank}';m=read(folder/'shard_manifest.json')
        assert m['eval_sha256']==sha(data) and m['adapter_sha256']==directory_digest(adapter)
        assert m['mapping_sha256']==sha(bundle/'inputs/interest_token_mapping.parquet')
        assert m['model_revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
        assert m['batch_size']==8 and m['num_shards']==4 and m['max_new_tokens']==512 and not m['do_sample'] and m['enable_thinking']
        assert m['shard_index']==rank and m['full_record_count']==expected
        assert m['evaluation_source_sha256']==sha(bundle/'src/new_interest/evaluation/evaluate_short_think_ranking.py')
        predictions=folder/'id_token_predictions.parquet';assert sha(predictions)==m['prediction_sha256']
        assert sha(folder/'metrics.json')==m['metrics_sha256']
        rows=pq.read_table(predictions).to_pylist();assert [r['record_index'] for r in rows]==list(range(rank,expected,4))
        shard_hashes[str(rank)]=sha(predictions)
        for r in rows:
            index=r['record_index'];record=records[index]
            assert r['impression_id']==record['impression_id'] and r['input_sha256']==canonical(record)
            assert r['evaluation_identity']==m['evaluation_identity'] and index not in combined
            text=tokenizer.decode(r['generated_token_ids'],skip_special_tokens=False)
            assert text==r['generated_text']
            match=GRAMMAR.fullmatch(text) if r['natural_end'] else None
            positions=[i for i,t in enumerate(r['generated_token_ids']) if t in token_by_id]
            valid=bool(match and len(positions)==1 and token_by_id[r['generated_token_ids'][positions[0]]]==match['sid'])
            if match and ('<think>' in match['think'] or re.search(r'<interest_\d{3}>',match['think'])):valid=False
            assert r['valid_format']==valid
            labels=r['ranked_labels'];scores=r['ranked_scores'];tokens=r['ranked_tokens']
            assert len(labels)==len(set(labels))==len(scores)==len(tokens)==(20 if valid else 0)
            assert tokens==[by_label[label] for label in labels]
            assert all(math.isfinite(x) for x in scores) and all(a>=b for a,b in zip(scores,scores[1:]))
            if valid:
                assert r['answer_token_offset']==positions[0] and r['generated_sid']==match['sid']
                assert tokenizer.decode(r['generated_token_ids'][:positions[0]],skip_special_tokens=False)==text[:match.start('sid')]
            combined[index]=r
    merged=evaluation/'merged';manifest=read(merged/'merge_manifest.json');metrics=read(merged/'metrics.json')
    assert manifest['shard_prediction_sha256']==shard_hashes and manifest['merged_record_count']==expected
    assert sha(merged/'id_token_predictions.parquet')==manifest['merged_prediction_sha256'] and sha(merged/'metrics.json')==manifest['metrics_sha256']
    rows=pq.read_table(merged/'id_token_predictions.parquet').to_pylist();assert [r['record_index'] for r in rows]==list(range(expected))
    for i,r in enumerate(rows):
        for key in ['ranked_labels','ranked_tokens','ranked_scores','generated_token_ids','valid_format','input_sha256']:
            assert r[key]==combined[i][key]
    support=Counter();hits={k:Counter() for k in [10,20]};gains={k:defaultdict(list) for k in [10,20]}
    for record,row in zip(records,rows):
        targets=set(record['acceptable_subcategories']);assert targets
        support.update(targets);positions={label:i+1 for i,label in enumerate(row['ranked_labels'])}
        for label in targets:
            rank=positions.get(label,21)
            for k in [10,20]:
                if rank<=k:hits[k][label]+=1;gains[k][label].append(1/math.log2(rank+1))
    assert len(support)==104 and metrics['samples']==expected
    result={}
    for k in [10,20]:
        result[f'interest_macro_recall_at_{k}']=math.fsum(hits[k][label]/n for label,n in support.items())/104
        result[f'interest_macro_ndcg_at_{k}']=math.fsum(math.fsum(gains[k][label])/n for label,n in support.items())/104
    for key,value in result.items():assert math.isclose(value,metrics[key],abs_tol=1e-12,rel_tol=0)
    return {'passed':True,'rows':expected,'supported_classes':104,'target_occurrences':sum(support.values()),
        'macro':result,'prediction_sha256':sha(merged/'id_token_predictions.parquet'),'scope':'full Dev' if full else 'diagnostic Dev'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--end-step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation-root',type=Path)
    a=p.parse_args();report=verify_training(a.bundle,a.roots,a.end_step);report['verifier_sha256']=sha(Path(__file__))
    if a.evaluation_root:
        segment=a.evaluation_root.parent;manifest=read(segment/'artifact_manifest.json')
        assert manifest['engine_identity']==report['engine_identity'];files_match(segment,manifest['files'])
        finished=read(segment/'segment_complete.json');assert finished['engine_identity']==report['engine_identity'] and finished['end_step']==a.end_step
        assert finished['final_state']==report['final_state']
        fresh=read(segment/'fresh_reload.json');assert fresh['passed'] and fresh['fresh_probe_max_abs']==0 and fresh['state_sha256']==report['final_state']['adapter_state_sha256']
        adapter=Path(report['checkpoints'][-1]['checkpoint'])/'adapter'
        report['evaluation']=verify_evaluation(a.bundle,a.evaluation_root,adapter,a.end_step==675)
        report['full_experiment_complete']=a.end_step==675
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['updates','final_state']}))


if __name__=='__main__':main()
