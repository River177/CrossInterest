"""Two-condition full-vocabulary SFT, complete epochs, four A100 replicas.

The deployed state is always the outer model. Startup audits are part of each
job; they do not create additional experimental conditions. Platform retries
write distinct attempt trees and adopt only fully committed same-identity state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from new_interest.experiments import oi_ni_meta as core
from new_interest.experiments import oi_ni_meta_distributed as execution
from new_interest.data.build_oi_ni_full_sft import validate_epoch
from new_interest.training.oi_ni_full_sft import (
    FullVocabularySFTModel, completion_batch, completion_chunks, encode_completions,
    pooled_role_scaling,
)
from new_interest.training.oi_ni_meta_distributed_objective import compute_distributed_meta_gradients
from new_interest.experiments.oi_ni_full_sft_storage import LocalTrainingJournal, hydrate_recovery_parent


_PHASE = itertools.count()
_SHARED = None


@contextmanager
def primary_phase(context, label):
    """Keep NCCL idle while rank zero performs mounted-storage operations."""
    marker = _SHARED / f'phase-{next(_PHASE):06d}.json'
    error = None
    try:
        yield
    except Exception as exception:
        if not context.primary:
            raise
        error = f'{type(exception).__name__}: {exception}'
    if context.primary:
        core.write_json(marker, {'label': label, 'error': error})
    deadline = time.monotonic() + 2700
    while not marker.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Primary phase: {label}')
        time.sleep(.2)
    state = json.loads(marker.read_text())
    if state['label'] != label or state['error']:
        raise RuntimeError(f"{label}: {state['error']}")
    dist.barrier()


def validate_inputs(config, data):
    metadata = json.loads((data / 'metadata.json').read_text())
    if core.sha256_file(data / 'metadata.json') != config['data_metadata_sha256']:
        raise ValueError('Changed data metadata')
    for name, digest in metadata['files'].items():
        if core.sha256_file(data / name) != digest:
            raise ValueError(f'Changed frozen data: {name}')
    rows = pq.read_table(data / 'mixed.parquet').to_pylist()
    plan = pq.read_table(data / 'episodes.parquet').to_pylist()
    if len(rows) != 38698 or len(plan) != config['steps'] or config['steps'] != 3630:
        raise ValueError('Full-data or step count differs')
    if Counter(r['training_task'] for r in rows) != {'old_interest':17098,'new_interest':21600}:
        raise ValueError('Training roles differ')
    for epoch in range(3):
        validate_epoch(rows,[x['row_indices'] for x in plan if x['epoch']==epoch])
    for step, episode in enumerate(plan,1):
        expected_f=[i for i in episode['row_indices'] if rows[i]['training_task']=='old_interest']
        expected_g=[i for i in episode['row_indices'] if rows[i]['training_task']=='new_interest']
        if episode['step']!=step or episode['support_indices']!=expected_f or episode['query_indices']!=expected_g:
            raise ValueError('Frozen role indices differ')
    return rows,plan


def read_prefix(path, plan, step):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != step:
        raise ValueError('Checkpoint log prefix length differs')
    for i,row in enumerate(rows):
        if row['step']!=i+1 or row['row_indices']!=plan[i]['row_indices']:
            raise ValueError('Actual consumed sample order differs from frozen complete epochs')
        if not math.isfinite(row['loss']):
            raise ValueError('Nonfinite historical loss')
    return rows


def learning_rate_multiplier(step, total_steps, warmup_ratio):
    warmup=math.ceil(total_steps*warmup_ratio)
    if step<warmup:return step/max(1,warmup)
    return .5*(1+math.cos(math.pi*min(1,(step-warmup)/(total_steps-warmup))))


def find_parent(root,identity):
    candidates=[]
    for p in root.glob('attempts/*/checkpoints/checkpoint-*/manifest.json'):
        state=json.loads(p.with_name('state.json').read_text())
        if state['identity']!=identity:
            raise ValueError('Different identity in checkpoint tree')
        candidates.append((state['step'],str(p.parent)))
    if not candidates:return None
    _,name=max(candidates)
    parent=Path(name);core.verify_manifest(parent,identity)
    return parent


def checkpoint(context,model,tokenizer,optimizer,scheduler,root,identity,step,probes,log_rows,baseline,journal):
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    tensor_hash=execution._replica_digest(params,context)
    rank_rng=[None]*4;dist.all_gather_object(rank_rng,execution._rank_rng())
    destination=root/'checkpoints'/f'checkpoint-{step:06d}'
    with primary_phase(context,'save checkpoint'):
        if context.primary:
            local=_SHARED/f'checkpoint-{step:06d}';local.mkdir()
            model.causal_lm.save_pretrained(local/'adapter',safe_serialization=True,save_embedding_layers=False)
            tokenizer.save_pretrained(local/'adapter')
            probe=core.capture_probe(model,tokenizer,probes)
            torch.save({'identity':identity,'step':step,'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'parameter_names':list(params),'rng_by_rank':rank_rng,
                'baseline':baseline,'world_size':4},local/'training_state.pt')
            core.write_json(local/'state.json',{'identity':identity,'step':step,'tensor_sha256':tensor_hash,
                'world_size':4,'deployed_outer_parameters':True,'supervised_scope':'full_vocabulary'})
            core.write_json(local/'reload_probe.json',{'input_ids':probes,'scores':probe})
            (local/'training_prefix.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in log_rows))
            core.publish(local,destination,identity)
            journal.snapshot(root/'training.jsonl')
            shutil.rmtree(local)
            core.log('checkpoint_committed',step=step,checkpoint=str(destination),identity=identity)
    return destination


def audit_full_loss(model,example,pad_token):
    """Real-model comparison against the original causal-LM training path."""
    model.eval();batch=completion_batch([example],pad_token,torch.device('cuda',torch.cuda.current_device()))
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    actual=model(**batch)
    actual_grad=torch.autograd.grad(actual,parameters)
    actual_value=float(actual.detach())
    del actual
    expected=model.causal_lm(**batch,use_cache=False).loss
    expected_grad=torch.autograd.grad(expected,parameters)
    difference=sum(float((a.double()-b.double()).square().sum()) for a,b in zip(actual_grad,expected_grad))
    reference=sum(float(b.double().square().sum()) for b in expected_grad)
    relative=math.sqrt(difference/max(reference,1e-30))
    loss_error=abs(actual_value-float(expected.detach()))
    # FP64 mathematical equivalence and full-model FP32 parity are CPU tests.
    # BF16 GEMM shape can change rounding; these fixed tolerances bind this audit.
    if loss_error>1e-3 or relative>0.01:
        raise ValueError(f'Original full-vocabulary loss parity failed: loss={loss_error}, grad={relative}')
    result={'full_sft_loss':actual_value,'original_causal_lm_loss':float(expected.detach()),
        'loss_abs_error':loss_error,'gradient_relative_l2':relative,
        'loss_atol':1e-3,'gradient_relative_l2_limit':.01,
        'supervised_tokens':example['supervised_tokens'],'vocabulary_size':model.causal_lm.config.vocab_size,
        'all_trainable_gradients_checked':len(parameters),'accepted':True}
    del expected,actual_grad,expected_grad
    model.train();torch.cuda.empty_cache()
    return result


def run(args,context):
    global _SHARED
    config=json.loads(args.config.read_text())
    if config['modes']!=['mixed','meta'] or config['semantic_id_loss_scope']!='full_vocabulary':
        raise ValueError('Only full-vocabulary Mixed and Meta are authorized')
    manifest=json.loads(Path('full_sft_bundle_manifest.json').read_text())
    for relative,digest in manifest['files'].items():
        if core.sha256_file(Path(relative))!=digest:raise ValueError(f'Changed bundled file: {relative}')
    source=core.sha256_file(Path('full_sft_bundle_manifest.json'))
    identity=execution.digest_json({'source':source,'config':config,'mode':args.mode})
    data=Path(config['data_dir']);rows,plan=validate_inputs(config,data)
    mode_root=args.root/args.mode
    attempt=execution._broadcast(uuid.uuid4().hex if context.primary else None,context)
    _SHARED=Path(os.environ.get('META_LOCAL_SCRATCH','/scratch/oi-ni-full-sft'))/attempt
    _SHARED.mkdir(parents=True,exist_ok=True)
    root=mode_root/'attempts'/attempt
    parent=None;snapshot=None;completed=False;parent_identity=identity;recovery_binding=None
    with primary_phase(context,'initialize attempt'):
        if context.primary:
            root.mkdir(parents=True)
            pointer=mode_root/'completed_attempt.json'
            if pointer.exists():
                saved=json.loads(pointer.read_text())
                completed_root=mode_root/'attempts'/saved['attempt']
                if saved['identity']!=identity or core.sha256_file(completed_root/'manifest.json')!=saved['manifest_sha256']:
                    raise ValueError('Completed attempt identity differs')
                core.verify_manifest(completed_root,identity)
                completed=True
            parent=find_parent(mode_root,identity)
            if parent is None and args.recovery_manifest is not None:
                recovery_binding=json.loads(args.recovery_manifest.read_text())
                parent=hydrate_recovery_parent(recovery_binding,config,args.mode,_SHARED)
                parent_identity=recovery_binding['identity']
                core.write_json(root/'recovery_lineage.json',recovery_binding)
            core.write_json(root/'resolved_config.json',{'identity':identity,'source_sha256':source,
                'mode':args.mode,'config':config,'parent_checkpoint':str(parent) if parent else None,
                'parent_identity':parent_identity if parent else None,
                'storage_policy':'node_local_journal_checkpoint_snapshots'})
            snapshot=core.prepare_snapshot(config,root)
            core.write_json(root/'environment.json',{'world_size':4,'gpu':torch.cuda.get_device_name(),
                'cuda':torch.version.cuda,'packages':{n:importlib.metadata.version(n) for n in
                    ['torch','transformers','peft','accelerate','tokenizers','pyarrow','safetensors']}})
            shutil.copyfile('full_sft_bundle_manifest.json',root/'source_manifest.json')
    if execution._broadcast(completed,context):
        core.log('existing_completed_attempt_verified',mode=args.mode,identity=identity)
        return
    parent_string,snapshot,parent_identity,recovery_binding=execution._broadcast(
        (str(parent) if parent else None,snapshot,parent_identity,recovery_binding),context)
    parent=Path(parent_string) if parent_string else None
    model,tokenizer,spec,initialization=core.make_model(config,snapshot,data/'interest_token_mapping.parquet',
        parent/'adapter' if parent else None,wrapper_class=FullVocabularySFTModel)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial_hash=execution._replica_digest(params,context)
    optimizer=torch.optim.AdamW(list(params.values()),lr=config['outer_learning_rate'],
                                weight_decay=config['weight_decay'],fused=True)
    def multiplier(step):
        return learning_rate_multiplier(step,config['steps'],config['warmup_ratio'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    examples=encode_completions(tokenizer,rows,config['max_seq_length'])
    completion_identity=execution.digest_json(examples)
    if completion_identity!=config['completion_encoding_sha256']:
        raise ValueError('Worker completion labels differ from local frozen tokenization')
    prompt_tokens=core.encode_records(tokenizer,[rows[i] for i in config['probe_indices']],max_length=config['max_seq_length'])
    log_rows=[];start_step=0;baseline=None
    if parent:
        state=torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
        if state['identity']!=parent_identity or state['parameter_names']!=list(params) or state['world_size']!=4:
            raise ValueError('Parent state identity differs')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        execution._restore_rank_rng(state['rng_by_rank'][context.rank])
        start_step=int(state['step']);baseline=state['baseline']
        log_rows=read_prefix(parent/'training_prefix.jsonl',plan,start_step)
        saved=json.loads((parent/'state.json').read_text())
        if initial_hash!=saved['tensor_sha256']:raise ValueError('Parent trainable tensors differ after reload')
        probe=json.loads((parent/'reload_probe.json').read_text())
        if core.capture_probe(model,tokenizer,probe['input_ids'])!=probe['scores']:
            raise ValueError('Parent GPU reload logits differ')
        if scheduler.last_epoch!=start_step or any(int(s['step'])!=start_step for s in optimizer.state.values()):
            raise ValueError('Parent optimizer/scheduler cursors differ')
        if len(optimizer.state)!=len(params):
            raise ValueError('Parent optimizer does not cover every trainable parameter')
        for parameter,moment in optimizer.state.items():
            for field in ('exp_avg','exp_avg_sq'):
                if moment[field].shape!=parameter.shape or not bool(torch.isfinite(moment[field]).all()):
                    raise ValueError('Parent optimizer moments differ in shape or are nonfinite')
            if not bool((moment['exp_avg_sq']>=0).all()):
                raise ValueError('Parent optimizer second moment is negative')
        if recovery_binding and start_step!=recovery_binding['step']:
            raise ValueError('Storage recovery cursor differs')
        if recovery_binding:
            execution.execution_audit(context,model,_SHARED,identity,'recovery',start_step,
                                      optimizer=optimizer,reload_error=0.)
    journal=None
    with primary_phase(context,'model and token audits'):
        if context.primary:
            core.write_json(root/'model_audit.json',{'identity':identity,'initial_tensor_sha256':initial_hash,
                'resumed_from_step':start_step,'backend':model.backend_audit,'initialization':initialization,
                'trainable_shapes':{n:list(p.shape) for n,p in params.items()}})
            core.write_json(root/'token_audit.json',{'rows':len(rows),'maximum_length':max(len(x['input_ids']) for x in examples),
                'over_length':0,'supervised_token_counts':dict(Counter(x['supervised_tokens'] for x in examples)),
                'completion_encoding_sha256':completion_identity,'original_sft_tokenization':True})
            journal=LocalTrainingJournal(_SHARED,log_rows)
            journal.snapshot(root/'training.jsonl')
            if recovery_binding:
                rank_audit=_SHARED/'execution_audits'/f'recovery-{start_step:06d}.json'
                shutil.copyfile(rank_audit,root/'recovery_rank_audit.json')
                core.write_json(root/'recovery_acceptance.json',{'identity':identity,
                    'parent_identity':parent_identity,'step':start_step,'payload_hashes_verified':True,
                    'optimizer_scheduler_prefix_verified':True,'parent_tensor_sha256':initial_hash,
                    'fresh_reload_max_abs_error':0.,'world_size':4,
                    'rank_audit_sha256':core.sha256_file(rank_audit)})
            core.log('full_sft_startup',identity=identity,mode=args.mode,attempt=attempt,
                     resumed_from_step=start_step,steps=config['steps'],initial_tensor_sha256=initial_hash)
    with primary_phase(context,'original full SFT loss parity'):
        if context.primary:
            audit=audit_full_loss(model,examples[config['parity_sample_index']],tokenizer.pad_token_id)
            core.write_json(root/'full_loss_parity.json',audit)
            if baseline is None:
                baseline={'initial_probe_loss':audit['full_sft_loss'],
                          'initial_tensor_sha256':initial_hash,'sample_index':config['parity_sample_index']}
    baseline=execution._broadcast(baseline,context)
    alpha=config['inner_alpha'] if args.mode=='meta' else 0.
    device=torch.device('cuda',context.local_rank)
    model.train()
    last_checkpoint=parent
    if args.mode=='meta' and start_step==0:
        longest={role:max((i for i,r in enumerate(rows) if r['training_task']==role),
                         key=lambda i:len(examples[i]['input_ids'])) for role in ('old_interest','new_interest')}
        fi,gi=longest['old_interest'],longest['new_interest']
        fs,gs,ea=pooled_role_scaling(examples[fi]['supervised_tokens'],examples[gi]['supervised_tokens'],alpha)
        f=completion_chunks([fi],examples,tokenizer.pad_token_id,device,scale=fs,step_seed=42)
        g=completion_chunks([gi],examples,tokenizer.pad_token_id,device,scale=gs,step_seed=10000061)
        torch.cuda.reset_peak_memory_stats()
        gate=compute_distributed_meta_gradients(model,f if context.rank==0 else [],g if context.rank==1 else [],
            alpha=ea,mode='meta',reduction='ordered',global_positions={
                'oi':[0] if context.rank==0 else [],'ni':[0] if context.rank==1 else []})
        unchanged=execution._replica_digest(params,context)
        if unchanged!=initial_hash:raise ValueError('Gradient-only HVP check changed parameters')
        peak=torch.tensor(torch.cuda.max_memory_allocated(),device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        with primary_phase(context,'longest full-vocabulary HVP audit'):
            if context.primary:
                core.write_json(root/'longest_hvp_audit.json',{'accepted':True,'loss':float(gate.loss),
                    'hvp_norm':float(gate.hvp_norm),'oi_length':len(examples[fi]['input_ids']),
                    'ni_length':len(examples[gi]['input_ids']),'peak_gpu_bytes':int(peak),
                    'parameter_update_performed':False,'scope':'numerical check only'})
        del gate,f,g;torch.cuda.empty_cache()
    for index in range(start_step,config['steps']):
        episode=plan[index];fids=episode['support_indices'];gids=episode['query_indices']
        tf=sum(examples[i]['supervised_tokens'] for i in fids)
        tg=sum(examples[i]['supervised_tokens'] for i in gids)
        fs,gs,effective_alpha=pooled_role_scaling(tf,tg,alpha)
        seed=config['seed']+104729*index
        f=completion_chunks(fids,examples,tokenizer.pad_token_id,device,scale=fs,step_seed=seed)
        g=completion_chunks(gids,examples,tokenizer.pad_token_id,device,scale=gs,step_seed=seed+10000019)
        fp=execution.owned_positions(len(f),context.rank);gp=execution.owned_positions(len(g),context.rank)
        optimizer.zero_grad(set_to_none=True)
        dist.barrier();torch.cuda.synchronize();begin=time.monotonic()
        result=compute_distributed_meta_gradients(model,[f[i] for i in fp],[g[i] for i in gp],
            alpha=effective_alpha,mode=args.mode,reduction='ordered',global_positions={'oi':fp,'ni':gp})
        for name,parameter in params.items():parameter.grad=result.gradients[name]
        norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),config['max_grad_norm'],error_if_nonfinite=True))
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        torch.cuda.synchronize();seconds=execution._rank_max_seconds(time.monotonic()-begin)
        row={'step':index+1,'epoch':episode['epoch'],'row_indices':episode['row_indices'],
            'oi_samples':len(fids),'ni_samples':len(gids),'oi_supervised_tokens':tf,'ni_supervised_tokens':tg,
            'loss':float(result.loss),'oi_loss':float(result.oi_loss)/fs,'ni_loss':float(result.ni_loss)/gs,
            'ni_fast_loss':float(result.adapted_ni_loss)/gs,'alpha':alpha,'engine_alpha':effective_alpha,
            'hvp_norm':float(result.hvp_norm),'grad_norm':norm,'lr':lr,'seconds':seconds,'seed':seed,
            'world_size':4,'loss_scope':'full_vocabulary'}
        log_rows.append(row)
        if context.primary:
            journal.append(row)
            core.log('full_sft_update',**{k:v for k,v in row.items() if k!='row_indices'})
        del result,f,g
        step=index+1
        if step in {2,config['steps']} or step%config['checkpoint_every']==0:
            last_checkpoint=checkpoint(context,model,tokenizer,optimizer,scheduler,root,identity,step,
                                       prompt_tokens,log_rows,baseline,journal)
        if step==2:
            # Checkpoint 2 is real progress, not an extra training condition.
            before=execution._replica_digest(params,context)
            del params,model;gc.collect();torch.cuda.empty_cache()
            model,tokenizer,spec,_=core.make_model(config,snapshot,data/'interest_token_mapping.parquet',
                last_checkpoint/'adapter',wrapper_class=FullVocabularySFTModel)
            params={n:p for n,p in model.named_parameters() if p.requires_grad}
            after=execution._replica_digest(params,context)
            probe=json.loads((last_checkpoint/'reload_probe.json').read_text())
            if before!=after or core.capture_probe(model,tokenizer,probe['input_ids'])!=probe['scores']:
                raise ValueError('Startup four-rank fresh reload failed')
            # Rebind optimizer to the reconstructed Parameters before continuing.
            state=torch.load(last_checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            optimizer=torch.optim.AdamW(list(params.values()),lr=config['outer_learning_rate'],
                                        weight_decay=config['weight_decay'],fused=True)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            execution._restore_rank_rng(state['rng_by_rank'][context.rank]);del state
            with primary_phase(context,'startup reload receipt'):
                if context.primary:
                    core.write_json(root/'startup_gate.json',{'identity':identity,'steps':2,'world_size':4,
                        'full_vocabulary_loss':True,'reload_max_abs_error':0.,'accepted':True})
            core.log('full_sft_startup_gate_passed',step=2,rank=context.rank)
            # Small real ranking/persistence check within the same two jobs.
            small=pq.read_table(data/'dev_full_grouped.parquet').slice(0,64).to_pylist()
            small_tokens=core.encode_records(tokenizer,small,max_length=config['max_seq_length'])
            small_labels=[str(r['target_subcategory']) for r in core.load_interest_token_mapping(data/'interest_token_mapping.parquet')]
            execution.parallel_predict(context,model,small_tokens,small,tokenizer,small_labels,
                                       root/'startup_dev64',identity)
        if step==20 and args.mode=='mixed':
            with primary_phase(context,'SFT learning check'):
                if context.primary:
                    model.eval()
                    with torch.no_grad():
                        probe_loss=float(model(**completion_batch([examples[baseline['sample_index']]],tokenizer.pad_token_id,device)))
                    model.train()
                    learning={'initial':baseline['initial_probe_loss'],'after20':probe_loss,
                              'accepted':math.isfinite(probe_loss) and probe_loss<baseline['initial_probe_loss']}
                    core.write_json(root/'learning_check.json',learning)
                    if not learning['accepted']:raise ValueError('Full SFT fixed training probe did not improve after20 updates')
    final_hash=execution._replica_digest(params,context)
    execution.execution_audit(context,model,root,identity,'deployment',config['steps'],optimizer=optimizer)
    del params,model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    model,tokenizer,spec,_=core.make_model(config,snapshot,data/'interest_token_mapping.parquet',
        last_checkpoint/'adapter',wrapper_class=FullVocabularySFTModel)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if execution._replica_digest(params,context)!=final_hash:raise ValueError('Final fresh tensors differ')
    probe=json.loads((last_checkpoint/'reload_probe.json').read_text())
    if core.capture_probe(model,tokenizer,probe['input_ids'])!=probe['scores']:
        raise ValueError('Final fresh reload logits differ')
    execution.execution_audit(context,model,root,identity,'fresh_reload',config['steps'],reload_error=0.)
    dev=pq.read_table(data/'dev_full_grouped.parquet').to_pylist()
    tokens=core.encode_records(tokenizer,dev,max_length=config['max_seq_length'])
    labels=[str(r['target_subcategory']) for r in core.load_interest_token_mapping(data/'interest_token_mapping.parquet')]
    metrics=execution.parallel_predict(context,model,tokens,dev,tokenizer,labels,root/'dev',identity)
    if metrics['samples']!=31392 or metrics['supported_interest_count']!=104:
        raise ValueError('Full Dev coverage differs')
    with primary_phase(context,'final commit'):
        if context.primary:
            read_prefix(root/'training.jsonl',plan,config['steps'])
            result={'identity':identity,'mode':args.mode,'steps':config['steps'],'epochs':3,'world_size':4,
                'loss_scope':'full_vocabulary','oi_presentations':sum(x['oi_samples'] for x in log_rows),
                'ni_presentations':sum(x['ni_samples'] for x in log_rows),'final_tensor_sha256':final_hash,
                'fresh_reload_max_abs_error':0.,'training_seconds':math.fsum(x['seconds'] for x in log_rows),
                'dev_metrics':metrics,'checkpoint_relative_to_mode':str(last_checkpoint.relative_to(mode_root))}
            core.write_json(root/'success.json',result)
            core.write_json(root/'manifest.json',{'identity':identity,'files':core.inventory(root)})
            core.write_json(mode_root/'completed_attempt.json',{'identity':identity,'attempt':attempt,
                'manifest_sha256':core.sha256_file(root/'manifest.json')})
            core.log('full_sft_complete',mode=args.mode,identity=identity,attempt=attempt)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['mixed','meta'],required=True)
    parser.add_argument('--recovery-manifest',type=Path)
    args=parser.parse_args();context=execution.initialize_distributed()
    try:run(args,context)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
