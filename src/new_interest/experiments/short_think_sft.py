"""Frozen short-think SFT from the accepted MIND Mixed adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import _atomic_write_json, canonical_sha256
from new_interest.evaluation.evaluate_short_think_ranking import merge


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(base, files):
    size=0
    for name, expected in files.items():
        path=Path(base)/name
        if sha(path)!=expected:
            raise ValueError(f'File hash mismatch: {name}')
        size+=path.stat().st_size
    return size


def validate_contract(contract):
    """The initial-state expectation requires its producer and reset audit."""
    if contract['sft'].get('record_initial_adapter_state') is not True:
        raise ValueError('Short-think SFT requires record_initial_adapter_state=true')
    if not contract['sft'].get('enable_thinking') or contract['sft'].get('epochs') != 2:
        raise ValueError('Short-think SFT contract changed')


def command(args, project, log, env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        proc=subprocess.Popen(args,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            stream.write(line);stream.flush()
            if any(s in line for s in ['loss','Error','Traceback','%|','CHECKPOINT_','SHORT_THINK_']):
                print(line.strip()[:1400],flush=True)
        code=proc.wait()
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log}')


def training_acceptance(root, phase, contract, source_adapter):
    from safetensors.torch import load_file
    from new_interest.training.sft_state_audit import tensor_state_manifest
    import torch
    source_weights=load_file(str(Path(source_adapter)/'adapter_model.safetensors'))
    kinds=['single','ddp'] if phase=='smoke' else ['sft']
    report={}
    for kind in kinds:
        folder=root/kind
        summary=read(folder/'run_summary.json')
        expected_rows=32 if kind=='single' else (108 if kind=='ddp' else 1080)
        expected_steps=8 if phase=='smoke' else 68
        if summary['train_samples']!=expected_rows or summary['actual_optimizer_steps']!=expected_steps:
            raise ValueError('Training sample or step count differs')
        if summary['configured_epochs']!=2 or not summary['enable_thinking']:
            raise ValueError('Training epochs/thinking differ')
        if summary['world_size']!=(1 if kind=='single' else 4):
            raise ValueError('Actual training world size differs')
        initial=read(folder/'initial_adapter_state.json')
        if initial['state_sha256']!=contract['initial_adapter_state_sha256']:
            raise ValueError('Initial Mixed parameters changed')
        start=read(folder/'training_start_audit.json')
        if not start['fresh_optimization_state']:
            raise ValueError('New SFT phase must reset optimization state')
        losses=[float(item['loss']) for item in summary['loss_history'] if 'loss' in item]
        if not losses or not all(math.isfinite(x) for x in losses):
            raise ValueError('Missing or nonfinite training loss')
        if phase=='smoke' and sum(losses[-2:])>=sum(losses[:2]):
            raise ValueError('Smoke loss did not decrease')
        weights=load_file(str(folder/'final_adapter/adapter_model.safetensors'))
        actual=tensor_state_manifest(weights)
        if actual['tensor_count']!=373 or not all(torch.isfinite(x).all().item() for x in weights.values()):
            raise ValueError('Final compact adapter invalid')
        delta=[x for k,x in weights.items() if 'trainable_tokens_delta' in k]
        if len(delta)!=1 or list(delta[0].shape)!=[108,2048]:
            raise ValueError('Wrong tied SID matrix')
        sid_key=next(k for k in weights if 'trainable_tokens_delta' in k)
        row_drift=(weights[sid_key].float()-source_weights[sid_key].float()).norm(dim=1)
        if not torch.isfinite(row_drift).all() or not (row_drift>0).all():
            raise ValueError('Not all108 inherited SID rows have finite nonzero updates')
        reload=read(folder/'semantic_adapter_reload_audit.json')
        if not reload['all_close']:
            raise ValueError('Fresh reload failed')
        integrity=reload['adapter_state_integrity']
        if not integrity.get('all_tensors_exact',False):
            raise ValueError('Reload tensor mismatch')
        report[kind]={'rows':expected_rows,'steps':expected_steps,'epochs':2,
            'initial_state_sha256':initial['state_sha256'],'final_state_sha256':actual['state_sha256'],
            'tensor_count':373,'finite_losses':True,'fresh_optimizer':True,'reload_exact':True,
            'adapter_sha256':sha(folder/'final_adapter/adapter_model.safetensors')}
    return report


def worker(project,root,phase):
    import torch
    from huggingface_hub import snapshot_download
    project=Path(project);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    contract=read(project/'contract.json')
    validate_contract(contract)
    sources=read(project/'source_manifest.json')
    verify_files(project,sources['files'])
    identity={'contract_sha256':sha(project/'contract.json'),'runtime_sha256':sources['runtime_sha256'],'phase':phase}
    if (root/'run_identity.json').exists() and read(root/'run_identity.json')!=identity:
        raise ValueError('Run identity differs')
    _atomic_write_json(root/'run_identity.json',identity)
    if phase=='full':
        accepted=read(project/'smoke_acceptance.json')
        if not accepted['passed'] or accepted['identity']['runtime_sha256']!=identity['runtime_sha256']:
            raise ValueError('Missing corresponding independent smoke acceptance')
    if torch.cuda.device_count()!=4:
        raise ValueError('Expected four GPUs')
    environment={'gpu_count':4,'gpus':[torch.cuda.get_device_name(i) for i in range(4)],
        'packages':{name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','flash-linear-attention','causal-conv1d']}}
    _atomic_write_json(root/'environment.json',environment)
    snapshot=Path(snapshot_download(contract['model_name'],revision=contract['model_revision'],local_files_only=True))
    expected=read(project/'source_snapshot.json')
    verify_files(snapshot,expected['files'])
    local=Path('/scratch/short-think-model')
    local.mkdir(exist_ok=True)
    for name in expected['files']:
        dest=local/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha(dest)!=expected['files'][name]:
            shutil.copyfile(snapshot/name,dest)
    verify_files(local,expected['files'])
    _atomic_write_json(root/'snapshot_audit.json',{**expected,'local_path':str(local)})
    source=project/'inputs/train.parquet';mapping=project/'inputs/interest_token_mapping.parquet'
    data=pq.read_table(source)
    if data.num_rows!=1080:
        raise ValueError('Wrong short reasoning dataset')
    config={**contract['sft'],'model_name':str(local),'model_revision':contract['model_revision'],
        'adapter_path':str(project/'inputs/mixed_adapter'),'train_file':str(source),
        'semantic_id_token_mapping_file':str(mapping)}
    (root/'configs').mkdir(exist_ok=True)
    for kind in (['single','ddp'] if phase=='smoke' else ['sft']):
        folder=root/kind
        if not (folder/'run_summary.json').exists():
            train=source
            if kind!='sft':
                train=root/f'{kind}_train.parquet'
                pq.write_table(data.slice(0,32 if kind=='single' else 108),train)
            cfg={**config,'train_file':str(train),'output_dir':str(folder),
                'run_name':f'mind-short-think-{phase}-{kind}-seed42',
                'per_device_train_batch_size':8,'gradient_accumulation_steps':1}
            cp=root/'configs'/f'{kind}.json';_atomic_write_json(cp,cfg)
            n=1 if kind=='single' else 4
            command([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={n}',
                '-m','new_interest.training.train_sft','--config',str(cp),
                '--expected-initial-adapter-state-sha256',contract['initial_adapter_state_sha256'],
                '--checkpoint-dir',f'/scratch/short-think-{phase}-{kind}-checkpoints',
                '--persistent-checkpoint-dir',str(root/f'checkpoints-{kind}'),'--resume-from-checkpoint','auto',
                '--save-steps','10','--wandb-mode','disabled'],project,root/'logs'/f'{kind}.log')
        reload=folder/'semantic_adapter_reload_audit.json'
        if not reload.exists():
            command([sys.executable,'-m','new_interest.training.verify_semantic_id_token_adapter',
                '--model-name',str(local),'--model-revision',contract['model_revision'],
                '--adapter-path',str(folder/'final_adapter'),'--mapping-file',str(mapping),
                '--reference-file',str(folder/'semantic_reload_reference.json'),'--output',str(reload),
                '--enable-thinking','--atol','0','--rtol','0'],project,root/'logs'/f'{kind}-reload.log')
    training=training_acceptance(root,phase,contract,project/'inputs/mixed_adapter')
    _atomic_write_json(root/'training_acceptance.json',training)
    adapter=root/('ddp' if phase=='smoke' else 'sft')/'final_adapter'
    evaluation={'model_name':str(local),'model_revision':contract['model_revision'],'adapter_path':str(adapter),
        'eval_file':str(project/'inputs/dev.parquet'),'interest_token_mapping_file':str(mapping),
        'batch_size':8,'checkpoint_size':16,'max_new_tokens':512,'num_shards':4,
        'max_examples':64 if phase=='smoke' else None}
    processes=[];streams=[];shards=[]
    for i in range(4):
        shard=root/'real_eval'/f'shard-{i}';shards.append(shard)
        cfg={**evaluation,'shard_index':i,'output_dir':str(shard)}
        cp=root/'configs'/f'eval-{i}.json';_atomic_write_json(cp,cfg)
        stream=(root/'logs'/f'eval-{i}.log').open('a');streams.append(stream)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i)}
        processes.append(subprocess.Popen([sys.executable,'-m','new_interest.evaluation.evaluate_short_think_ranking',
            'evaluate','--config',str(cp)],cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for stream in streams:stream.close()
    if any(codes):
        raise RuntimeError(f'Evaluation processes failed: {codes}')
    metrics=merge(shards,project/'inputs/dev.parquet',mapping,root/'real_eval/merged',64 if phase=='smoke' else 31392)
    _atomic_write_json(root/'success.json',{'identity':identity,'training':training,'evaluation':metrics})
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
        and p.name not in {'artifact_manifest.json','results.tar.gz','package.json'} and 'checkpoints-' not in str(p.relative_to(root))}
    _atomic_write_json(root/'artifact_manifest.json',{'identity':identity,'files':files})
    archive=Path('/scratch')/f'short-think-{phase}.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*files,'artifact_manifest.json']:
            tar.add(root/name,arcname=name,recursive=False)
    shutil.copyfile(archive,root/'results.tar.gz')
    _atomic_write_json(root/'package.json',{'sha256':sha(root/'results.tar.gz'),'bytes':archive.stat().st_size,'files':len(files)})
    print('SHORT_THINK_JOB_COMPLETE',phase,flush=True)


def readback(project,root,phase,output):
    project,root=Path(project),Path(root)
    contract=read(project/'contract.json');manifest=read(root/'artifact_manifest.json')
    if manifest['identity']['contract_sha256']!=sha(project/'contract.json') or manifest['identity']['phase']!=phase:
        raise ValueError('Artifact contract/phase mismatch')
    if manifest['identity']['runtime_sha256']!=read(project/'source_manifest.json')['runtime_sha256']:
        raise ValueError('Artifact runtime mismatch')
    size=verify_files(root,manifest['files'])
    training=training_acceptance(root,phase,contract,project/'inputs/mixed_adapter')
    # Re-run the real merger in an independent output directory; every shard and record is checked.
    verify_dir=Path(output).parent/(Path(output).stem+'-metrics-readback')
    metrics=merge([root/'real_eval'/f'shard-{i}' for i in range(4)],project/'inputs/dev.parquet',
        project/'inputs/interest_token_mapping.parquet',verify_dir,64 if phase=='smoke' else 31392)
    worker_metrics=read(root/'real_eval/merged/metrics.json')
    for key in ('interest_macro_recall_at_10','interest_macro_recall_at_20','interest_macro_ndcg_at_10','interest_macro_ndcg_at_20'):
        if abs(metrics[key]-worker_metrics[key])>1e-12:
            raise ValueError('Recomputed metric mismatch')
    receipt={'passed':True,'identity':manifest['identity'],'training':training,'evaluation':metrics,
        'artifact_files':len(manifest['files']),'artifact_bytes':size,'artifact_manifest_sha256':sha(root/'artifact_manifest.json')}
    _atomic_write_json(Path(output),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['worker','readback'])
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['smoke','full'],required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='worker':worker(args.project,args.root,args.phase)
    else:
        if args.output is None:parser.error('--output required for readback')
        print(json.dumps(readback(args.project,args.root,args.phase,args.output),default=str))


if __name__=='__main__':main()
