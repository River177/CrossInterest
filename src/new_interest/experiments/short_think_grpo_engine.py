"""Publish a two-update four-A100 GRPO oracle with a real storage resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid

from new_interest.evaluation.evaluate_beam import _atomic_write_json
from new_interest.evaluation.preflight_short_think_grpo import read,sha,verify_files


def publish(local,destination,identity):
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    names=[str(p.relative_to(local)) for p in local.rglob('*') if p.is_file()]
    def copy_verified(name):
        expected=sha(local/name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(local/name,target)
        if sha(target)!=expected:raise ValueError(f'Published file differs: {name}')
        return name,expected
    with ThreadPoolExecutor(max_workers=4) as pool:files=dict(pool.map(copy_verified,names))
    _atomic_write_json(destination/'checkpoint_manifest.json',{'identity':identity,'files':files})


def command(args,project,log,env=None,failure_source=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as out:
        process=subprocess.Popen(args,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            out.write(line);out.flush()
            if any(x in line for x in ['GRPO_ENGINE','Error','Traceback']):print(line.rstrip()[:1500],flush=True)
        if process.wait():
            # Preserve sampled trajectories and pre-update diagnostics even if
            # a strict numeric or backward gate fails. These are NOT checkpoints.
            if failure_source is not None and failure_source.exists():
                dest=log.parent/f'failure-{log.stem}';dest.mkdir(exist_ok=False)
                files={str(p.relative_to(failure_source)):sha(p) for p in failure_source.rglob('*') if p.is_file()}
                for name in files:
                    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(failure_source/name,target)
                verify_files(dest,files)
                _atomic_write_json(dest/'failure_manifest.json',{'not_resumable':True,'files':files})
            raise RuntimeError(f'Engine command failed; see {log.name}')


def worker(project,output):
    import torch
    from huggingface_hub import snapshot_download
    project=Path(project);output=Path(output);verify_files(project,read(project/'source_manifest.json')['files'])
    contract=read(project/'engine_contract.json');identity=sha(project/'engine_contract.json')
    acceptance=read(project/'preflight_acceptance.json')
    if (not acceptance['artifact_integrity_passed'] or not acceptance['reward_density_gate']
        or acceptance['contract_sha256']!=sha(project/'contract.json')
        or sha(project/'preflight_acceptance.json')!=contract['preflight_acceptance_sha256']):
        raise ValueError('Missing independent successful preflight')
    if torch.cuda.device_count()!=4:raise ValueError('Four GPUs required')
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    root=output/'attempts'/attempt;root.mkdir(parents=True,exist_ok=False)
    local=Path('/scratch')/f'grpo-engine-{attempt}';local.mkdir()
    _atomic_write_json(root/'identity.json',{'engine_identity':identity,'attempt':attempt,'role':'two-update correctness oracle'})
    _atomic_write_json(root/'environment.json',{'gpus':[torch.cuda.get_device_name(i) for i in range(4)],
        'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','flash-linear-attention']}})
    source=read(project/'source_snapshot.json');snapshot=Path(snapshot_download(source['model_name'],revision=source['model_revision'],local_files_only=True))
    verify_files(snapshot,source['files']);model_path=local/'model';model_path.mkdir()
    for name in source['files']:
        dest=model_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(snapshot/name,dest)
    verify_files(model_path,source['files'])
    first=local/'checkpoint-1'
    base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','new_interest.training.short_think_grpo_engine','--project',str(project),'--model-path',str(model_path)]
    command([*base,'--root',str(first),'--stop-at','1'],project,root/'step1.log',failure_source=first)
    publish(first,root/'checkpoint-1',identity)
    # Hydrate the PUBLISHED checkpoint, not a shortcut to the original scratch.
    parent=read(root/'checkpoint-1/checkpoint_manifest.json');hydrated=local/'hydrated-parent';hydrated.mkdir()
    if parent['identity']!=identity:raise ValueError('Parent identity changed')
    for name in parent['files']:
        dest=hydrated/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/'checkpoint-1'/name,dest)
    verify_files(hydrated,parent['files'])
    _atomic_write_json(root/'parent_hydration.json',{'parent_manifest_sha256':sha(root/'checkpoint-1/checkpoint_manifest.json'),
        'identity':identity,'all_bytes_verified':True,'files':len(parent['files'])})
    second=local/'checkpoint-2'
    command([*base,'--root',str(second),'--stop-at','2','--resume',str(hydrated)],project,root/'step2.log',failure_source=second)
    publish(second,root/'checkpoint-2',identity)
    command([sys.executable,'-m','new_interest.training.short_think_grpo_engine','--project',str(project),
        '--root',str(second),'--model-path',str(model_path),'--stop-at','2','--verify-only',
        '--verification-output',str(root/'fresh_reload.json')],project,root/'fresh_reload.log',env={**os.environ,'LOCAL_RANK':'0'})
    first_state=read(first/'state.json');second_state=read(second/'state.json')
    if first_state['step']!=1 or second_state['step']!=2 or first_state['adapter_state_sha256']==second_state['adapter_state_sha256']:
        raise ValueError('Expected two distinct real policy updates')
    if first_state['adapter_state_sha256']==contract['initial_adapter_tensor_state_sha256']:
        raise ValueError('First optimizer update did not change the policy')
    for i in range(4):
        start=read(second/f'start-{i}.json')
        if not start['resume_probe_exact'] or start['optimizer_state_sha256']!=first_state['optimizer_state_sha256']:
            raise ValueError('Resume model/optimizer probe failed')
    _atomic_write_json(root/'engine_complete.json',{'engine_identity':identity,'actual_optimizer_steps':2,'global_prompts':64,
        'global_trajectories':512,'four_rank_policy_hash_agreement':True,'storage_resume_verified':True,
        'fresh_reload':read(root/'fresh_reload.json'),'production_training_started':False,
        'scope':'Correctness oracle only; throughput and accelerated production integration still require assessment'})
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    _atomic_write_json(root/'artifact_manifest.json',{'engine_identity':identity,'files':files})
    archive=local/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*files,'artifact_manifest.json']:tar.add(root/name,arcname=name,recursive=False)
    shutil.copyfile(archive,root/'results.tar.gz')
    package={'attempt':attempt,'sha256':sha(root/'results.tar.gz'),'bytes':archive.stat().st_size,
        'contract_sha256':identity,'manifest_sha256':sha(root/'artifact_manifest.json')}
    _atomic_write_json(root/'package.json',package);_atomic_write_json(output/f'completed-{attempt}.json',package)
    print('GRPO_ENGINE_COMPLETE',json.dumps(package),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();worker(args.project,args.root)


if __name__=='__main__':main()
