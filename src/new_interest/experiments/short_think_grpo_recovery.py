"""Resume the preserved batch64 CP1 without rerunning its completed update."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import uuid

from new_interest.evaluation.evaluate_beam import _atomic_write_json
from new_interest.evaluation.preflight_short_think_grpo import read,sha,verify_files
from new_interest.experiments.short_think_grpo_engine import command,publish


def worker(project,output):
    import torch
    from huggingface_hub import snapshot_download
    verify_files(project,read(project/'source_manifest.json')['files'])
    recovery=read(project/'recovery_contract.json');identity=sha(project/'engine_contract.json')
    if identity!=recovery['engine_identity'] or sha(project/'recovered_probe_plan.json')!=recovery['recovered_plan_sha256']:
        raise ValueError('Recovery binding differs')
    if torch.cuda.device_count()!=4:raise ValueError('Four GPUs required')
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    root=output/'attempts'/attempt;root.mkdir(parents=True,exist_ok=False)
    local=Path('/scratch')/f'grpo-recovery-{attempt}';local.mkdir()
    parent=Path('/mnt/default')/recovery['parent_blob']
    manifest=read(parent/'checkpoint_manifest.json')
    if sha(parent/'checkpoint_manifest.json')!=recovery['parent_manifest_sha256'] or manifest['identity']!=identity:
        raise ValueError('Wrong committed parent')
    first=local/'parent';first.mkdir();published=root/'checkpoint-1';published.mkdir()
    for name in manifest['files']:
        p=first/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,p)
        q=published/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    verify_files(first,manifest['files']);verify_files(published,manifest['files'])
    shutil.copyfile(parent/'checkpoint_manifest.json',published/'checkpoint_manifest.json')
    if read(first/'state.json')['step']!=1:raise ValueError('Recovery must continue step1')
    _atomic_write_json(root/'parent_hydration.json',{'identity':identity,'all_bytes_verified':True,
        'parent_manifest_sha256':sha(published/'checkpoint_manifest.json'),'files':len(manifest['files'])})
    _atomic_write_json(root/'recovery_identity.json',{'engine_identity':identity,'recovery_contract_sha256':sha(project/'recovery_contract.json'),
        'source_checkpoint_manifest_sha256':recovery['parent_manifest_sha256'],'recovered_plan_sha256':sha(project/'recovered_probe_plan.json'),
        'original_reference_unchanged':True,'new_updates_requested':1})
    source=read(project/'source_snapshot.json');snapshot=Path(snapshot_download(source['model_name'],revision=source['model_revision'],local_files_only=True))
    verify_files(snapshot,source['files']);model_path=local/'model';model_path.mkdir()
    for name in source['files']:
        p=model_path/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(snapshot/name,p)
    verify_files(model_path,source['files'])
    second=local/'checkpoint-2'
    command([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        '-m','new_interest.training.short_think_grpo_engine','--project',str(project),'--root',str(second),
        '--model-path',str(model_path),'--stop-at','2','--resume',str(first)],project,root/'step2.log',failure_source=second)
    publish(second,root/'checkpoint-2',identity)
    command([sys.executable,'-m','new_interest.training.short_think_grpo_engine','--project',str(project),
        '--root',str(second),'--model-path',str(model_path),'--stop-at','2','--verify-only',
        '--verification-output',str(root/'fresh_reload.json')],project,root/'fresh_reload.log',env={**os.environ,'LOCAL_RANK':'0'})
    _atomic_write_json(root/'engine_complete.json',{'engine_identity':identity,'actual_optimizer_steps_total':2,
        'new_optimizer_updates_this_job':1,'first_update_retrained':False,'global_prompts_total':64,'global_trajectories_total':512,
        'fresh_reload':read(root/'fresh_reload.json'),'production_training_started':False})
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    _atomic_write_json(root/'artifact_manifest.json',{'engine_identity':identity,'files':files})
    archive=local/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*files,'artifact_manifest.json']:tar.add(root/name,arcname=name,recursive=False)
    shutil.copyfile(archive,root/'results.tar.gz')
    package={'attempt':attempt,'sha256':sha(root/'results.tar.gz'),'bytes':archive.stat().st_size,
        'contract_sha256':identity,'manifest_sha256':sha(root/'artifact_manifest.json')}
    _atomic_write_json(root/'package.json',package);_atomic_write_json(output/f'completed-{attempt}.json',package)
    print('GRPO_RECOVERY_COMPLETE',json.dumps(package),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();worker(a.project,a.root)


if __name__=='__main__':main()
