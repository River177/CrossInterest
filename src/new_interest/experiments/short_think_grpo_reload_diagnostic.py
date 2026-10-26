"""Four read-only GPU conditions on the preserved engine-v2 step1 checkpoint."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import _atomic_write_json
from new_interest.evaluation.preflight_short_think_grpo import read,sha,verify_files,stage
from new_interest.training.short_think_grpo_engine import model_load,probe,pack_trajectories,state_hash,optimizer_hash,trajectory_probabilities
from new_interest.training.short_think_grpo_objective import grpo_loss
from new_interest.training.probe_kernel_diagnostic import discover_tuners,snapshot,fixed_selection


def child(project,root,rank):
    import torch
    from new_interest.training.sft_state_audit import tensor_state_manifest
    os.environ['LOCAL_RANK']='0'
    os.environ['TRITON_CACHE_DIR']=f'/scratch/reload-diagnostic-triton-{rank}'
    config=read(project/'diagnostic_contract.json')
    checkpoint=Path((root/'checkpoint_path.txt').read_text().strip())
    model,tokenizer,spec=model_load(project,(root/'model_path.txt').read_text().strip(),checkpoint/'adapter',True)
    reference,_,_=model_load(project,(root/'model_path.txt').read_text().strip(),project/'inputs/short_sft_adapter',False)
    saved=read(checkpoint/'state.json')
    if state_hash(model)!=saved['adapter_state_sha256']:raise ValueError('Checkpoint adapter changed')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-6,fused=True)
    optimizer.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False))
    if optimizer_hash(model,optimizer)!=saved['optimizer_state_sha256']:raise ValueError('Checkpoint optimizer differs')
    rng=torch.load(checkpoint/'rng-0.pt',map_location='cpu',weights_only=False)
    import random
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python'])
    reference_before=state_hash(reference)
    records=pq.read_table(project/'inputs/train.parquet').to_pylist()
    rows=pq.read_table(checkpoint/'rollout-0000-rank-0.parquet').to_pylist()
    def flags():
        return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'matmul_precision':torch.get_float32_matmul_precision(),'default_dtype':str(torch.get_default_dtype()),
            'deterministic':torch.are_deterministic_algorithms_enabled()}
    def fingerprint():
        buffers=tensor_state_manifest(dict(model.named_buffers()))
        return {'parameters':tensor_state_manifest(dict(model.named_parameters()))['state_sha256'],
                'buffers':buffers['state_sha256'],'buffer_tensors':buffers['tensors']}
    tuners=discover_tuners()
    out=root/f'rank-{rank}';out.mkdir()
    report={'rank':rank,'mode':['fresh','warm64','warm64_backward8','fresh_replica'][rank],
            'contract_sha256':sha(project/'diagnostic_contract.json'),'checkpoint_state_sha256':sha(checkpoint/'state.json'),
            'adapter_state_sha256':state_hash(model),'before':fingerprint(),'flags_before':flags(),'tuners_before':snapshot(tuners)}
    _atomic_write_json(out/'start.json',report)
    if rank in [1,2]:
        stops={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>'),*tokenizer.encode('</think>',add_special_tokens=False)}
        stage(model,tokenizer,[r['prompt_ids'] for r in rows],stops,2,42,False,phase='thought',sid_ids=spec.token_ids)
    if rank==2:
        model.train()
        ordered=sorted(rows,key=lambda r:len(r['prompt_ids'])+len(r['response_ids']))
        for start in range(0,64,8):
            local_rows=ordered[start:start+8]
            batch,mask,old,ban,only=pack_trajectories(tokenizer,local_rows);batch={k:v.to(model.device) for k,v in batch.items()}
            mask,old,ban,only=[v.to(model.device) for v in (mask,old,ban,only)];length=mask.shape[1]
            with torch.no_grad():ref=reference(**batch,use_cache=False,logits_to_keep=length+1).logits[:,:-1]
            logits=model(**batch,use_cache=False,logits_to_keep=length+1).logits[:,:-1]
            current,kl=trajectory_probabilities(logits,ref,batch['input_ids'][:,-length:],ban,only,spec.token_ids)
            loss,_=grpo_loss(current,old,torch.tensor([r['advantage'] for r in local_rows],device=model.device),mask,raw_kl=kl)
            if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic backward')
            (loss/8).backward()
            del logits,ref,current,kl,loss
    tuners=discover_tuners()
    report['after_warm']=fingerprint();report['flags_after_warm']=flags();report['tuners_after_warm']=snapshot(tuners)
    if report['before']['parameters']!=report['after_warm']['parameters']:
        _atomic_write_json(out/'unexpected_parameter_mutation.json',report)
        raise ValueError('Read-only warm path mutated parameters')
    report['native_probe']=probe(model,tokenizer,records,spec.token_ids)
    report['native_repeat']=probe(model,tokenizer,records,spec.token_ids)
    report['native_vs_saved_max_abs']=float((torch.tensor(report['native_probe'])-torch.tensor(saved['reload_probe'])).abs().max())
    report['native_repeat_max_abs']=float((torch.tensor(report['native_probe'])-torch.tensor(report['native_repeat'])).abs().max())
    tuners=discover_tuners()
    report['tuners_after_native_probe']=snapshot(tuners)
    _atomic_write_json(out/'native_probe.json',report)
    with fixed_selection(tuners):
        report['fixed_probe']=probe(model,tokenizer,records,spec.token_ids)
        report['fixed_repeat']=probe(model,tokenizer,records,spec.token_ids)
        report['fixed_choices']=snapshot(tuners)
    report['restored_native_probe']=probe(model,tokenizer,records,spec.token_ids)
    report['restored_vs_native_max_abs']=float((torch.tensor(report['restored_native_probe'])-torch.tensor(report['native_probe'])).abs().max())
    report['fixed_repeat_max_abs']=float((torch.tensor(report['fixed_probe'])-torch.tensor(report['fixed_repeat'])).abs().max())
    report['tf32_controls']={}
    original_precision=torch.get_float32_matmul_precision()
    try:
        for precision in ['highest','high']:
            torch.set_float32_matmul_precision(precision)
            values=probe(model,tokenizer,records,spec.token_ids)
            report['tf32_controls'][precision]={'probe':values,
                'vs_saved_max_abs':float((torch.tensor(values)-torch.tensor(saved['reload_probe'])).abs().max())}
    finally:torch.set_float32_matmul_precision(original_precision)
    report['after']=fingerprint();report['flags_after']=flags();report['model_updates']=0
    report['reference_unchanged']=state_hash(reference)==reference_before
    report['optimizer_unchanged']=optimizer_hash(model,optimizer)==saved['optimizer_state_sha256']
    if not report['reference_unchanged'] or not report['optimizer_unchanged']:raise ValueError('Read-only diagnostic changed reference/optimizer')
    if report['before']['parameters']!=report['after']['parameters']:raise ValueError('Diagnostic changed parameters')
    _atomic_write_json(out/'complete.json',report)
    print('RELOAD_DIAGNOSTIC_DONE',rank,report['native_vs_saved_max_abs'],report['fixed_repeat_max_abs'],flush=True)


def worker(project,output):
    import torch
    from huggingface_hub import snapshot_download
    verify_files(project,read(project/'source_manifest.json')['files'])
    config=read(project/'diagnostic_contract.json')
    if torch.cuda.device_count()!=4:raise ValueError('Expected four GPUs')
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    root=output/'attempts'/attempt;root.mkdir(parents=True,exist_ok=False)
    parent=Path('/mnt/default')/config['parent_blob']
    if sha(parent/'checkpoint_manifest.json')!=config['parent_manifest_sha256']:raise ValueError('Parent manifest differs')
    manifest=read(parent/'checkpoint_manifest.json');checkpoint=Path('/scratch')/f'reload-diagnostic-parent-{attempt}';checkpoint.mkdir()
    for name in manifest['files']:
        target=checkpoint/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,target)
    verify_files(checkpoint,manifest['files'])
    (root/'checkpoint_path.txt').write_text(str(checkpoint)+'\n')
    _atomic_write_json(root/'parent_manifest.json',manifest)
    spec=read(project/'source_snapshot.json');snapshot_path=Path(snapshot_download(spec['model_name'],revision=spec['model_revision'],local_files_only=True))
    verify_files(snapshot_path,spec['files'])
    local=Path('/scratch')/f'reload-diagnostic-model-{attempt}';local.mkdir()
    for name in spec['files']:
        dest=local/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(snapshot_path/name,dest)
    verify_files(local,spec['files']);(root/'model_path.txt').write_text(str(local)+'\n')
    processes=[];streams=[]
    for rank in range(4):
        stream=(root/f'worker-{rank}.log').open('w');streams.append(stream)
        processes.append(subprocess.Popen([sys.executable,'-m','new_interest.experiments.short_think_grpo_reload_diagnostic',
            '--project',str(project),'--root',str(root),'--rank',str(rank)],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(rank),'LOCAL_RANK':'0'},stdout=stream,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for stream in streams:stream.close()
    if any(codes):raise RuntimeError(f'Reload diagnostic conditions failed: {codes}')
    reports=[read(root/f'rank-{rank}/complete.json') for rank in range(4)]
    fixed=[torch.tensor(r['fixed_probe']) for r in reports]
    summary={'diagnostic_contract_sha256':sha(project/'diagnostic_contract.json'),'model_updates':0,
        'native_saved_max_abs':[r['native_vs_saved_max_abs'] for r in reports],
        'fixed_cross_condition_max_abs':max(float((value-fixed[0]).abs().max()) for value in fixed),
        'restored_native_max_abs':[r['restored_vs_native_max_abs'] for r in reports],
        'parameter_hashes':[r['before']['parameters'] for r in reports],
        'buffer_hashes_before':[r['before']['buffers'] for r in reports],
        'scope':'Read-only diagnosis; no checkpoint reference replacement or training continuation approved by this report alone'}
    _atomic_write_json(root/'diagnostic_complete.json',summary)
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and not str(p.relative_to(root)).startswith('checkpoint/')}
    _atomic_write_json(root/'artifact_manifest.json',{'diagnostic_identity':summary['diagnostic_contract_sha256'],'files':files})
    archive=Path('/scratch')/f'reload-diagnostic-{attempt}.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*files,'artifact_manifest.json']:tar.add(root/name,arcname=name,recursive=False)
    shutil.copyfile(archive,root/'results.tar.gz')
    package={'attempt':attempt,'sha256':sha(root/'results.tar.gz'),'bytes':archive.stat().st_size,
        'contract_sha256':summary['diagnostic_contract_sha256'],'manifest_sha256':sha(root/'artifact_manifest.json')}
    _atomic_write_json(root/'package.json',package);_atomic_write_json(output/f'completed-{attempt}.json',package)
    print('RELOAD_DIAGNOSTIC_COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--rank',type=int)
    a=p.parse_args()
    if a.rank is None:worker(a.project,a.root)
    else:child(a.project,a.root,a.rank)


if __name__=='__main__':main()
