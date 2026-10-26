"""Preserve CIR5 and resume NI3 from its exact committed checkpoint in a new job."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from cir_storage_retry import retry_eio


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


sha = retry_eio(sha)


def copy_verified(source, target, expected):
    if target.exists():
        if target.stat().st_size == expected['size'] and sha(target) == expected['sha256']:
            return
        raise ValueError('Existing recovery file differs: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.recovery-partial')
    def copy():
        with source.open('rb') as src, temp.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=1024*1024)
        if temp.stat().st_size != expected['size'] or sha(temp) != expected['sha256']:
            raise ValueError('Recovery source hash mismatch: ' + str(source))
        temp.replace(target)
    retry_eio(copy)()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(p)


def prepare(project, source, output):
    import torch
    from new_interest.training.checkpoint_persistence import validate_persistent_checkpoint
    from new_interest.training import old_new_execution as cursor
    from new_interest.experiments import cir_schedule, cir_warmup as core
    if source.resolve() == output.resolve():
        raise ValueError('Recovery must use a new output root')
    manifest = read(project/'recovery/manifest.json')
    for rel, digest in manifest['recovery_files'].items():
        if sha(project/rel) != digest:
            raise ValueError('Recovery source changed: ' + rel)
    if sha(project/'manifests/cir_bundle.json') != manifest['bundle_sha256']:
        raise ValueError('Frozen bundle changed')
    for rel, digest in read(project/'manifests/cir_bundle.json')['files'].items():
        if sha(project/rel) != digest:
            raise ValueError('Frozen payload changed: ' + rel)
    identity = read(source/'identity.json')
    if identity != manifest['source_identity']:
        raise ValueError('Source run identity changed')
    complete = output/'recovery/seed_complete.json'
    if complete.exists():
        if read(complete)['recovery_manifest_sha256'] != sha(project/'recovery/manifest.json'):
            raise ValueError('Another recovery owns output')
        print('CIR_RECOVERY_SEED_ALREADY_COMMITTED', flush=True)
        return
    # Validate all source content in private scratch before writing any seed marker.
    scratch = Path('/scratch')/('cir-recovery-seed-'+hashlib.sha256(str(output).encode()).hexdigest()[:12])
    scratch.mkdir(parents=True, exist_ok=True)
    pinned = {}
    for rel, spec in manifest['seed_files'].items():
        path = source/rel
        if path.stat().st_size != spec['size']:
            raise ValueError('Source size changed: ' + rel)
        digest = spec.get('sha256') or sha(path)
        pinned[rel] = {'size': spec['size'], 'sha256': digest}
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: copy_verified(source/item[0], scratch/item[0], item[1]), pinned.items()))
    cp = scratch/'formal/checkpoints-ni3/checkpoint-900'
    marker = read(cp/'checkpoint_complete.json')
    validate_persistent_checkpoint(cp, manifest['checkpoint_identity'])
    if marker['global_step'] != 900:
        raise ValueError('Wrong resume step')
    opt = torch.load(cp/'optimizer.pt', map_location='cpu', weights_only=False)
    scheduler = torch.load(cp/'scheduler.pt', map_location='cpu', weights_only=False)
    steps = [int(s['step']) for s in opt['state'].values()]
    if len(steps) != 373 or set(steps) != {900} or scheduler['last_epoch'] != 900:
        raise ValueError('Adam or scheduler state differs')
    if [float(g['lr']) for g in opt['param_groups']] != [float(x) for x in scheduler['_last_lr']]:
        raise ValueError('Optimizer and scheduler LR differ')
    if not all(torch.isfinite(v).all().item() for state in opt['state'].values() for v in state.values() if torch.is_tensor(v)):
        raise ValueError('Nonfinite checkpoint optimizer state')
    cursor.STAGES.add('cir/ni3')
    verified = cursor.make_cursor_contract(project/'data/training/mind_cir_108_v1/ni.parquet', stage='cir/ni3',
        world_size=4, per_device_batch_size=2, source_checkpoint=cp, epochs=3, global_batch_size=32)
    if verified['consumed_total'] != 28800 or verified['total_steps'] != 2025:
        raise ValueError('NI3 cursor changed')
    if sha(scratch/'formal/cir5/final_adapter/adapter_model.safetensors') != manifest['cir5_adapter_sha256']:
        raise ValueError('CIR5 parent changed')
    original, shared = core.bind()
    cirspec = cir_schedule.stage_specs(project, 'staged')[0]
    cir = original.audit_stage(project, scratch/'formal', cirspec)
    shared.audit_adapter(project, '2b', scratch/'formal/cir5')
    shared.audit_reload(project, '2b', scratch/'formal/cir5')
    if cir['optimizer_steps'] != 2700:
        raise ValueError('CIR5 is not completed')
    # Commit identical stage and checkpoint bytes; source paths are never modified.
    # The seed receipt is last; after any platform retry, never reseed over training.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: copy_verified(scratch/item[0], output/item[0], item[1]), pinned.items()))
    validate_persistent_checkpoint(output/'formal/checkpoints-ni3/checkpoint-900', manifest['checkpoint_identity'])
    receipt = {'status':'accepted', 'recovery_manifest_sha256':sha(project/'recovery/manifest.json'),
        'source_prefix':manifest['source_results_prefix'], 'bundle_sha256':manifest['bundle_sha256'],
        'cir5_adapter_sha256':manifest['cir5_adapter_sha256'], 'ni3_step':900, 'optimizer_states':len(steps),
        'optimizer_step_min':min(steps), 'optimizer_step_max':max(steps), 'scheduler_last_epoch':scheduler['last_epoch'],
        'consumed_presentations':verified['consumed_total'], 'source_epoch':verified['source_epoch'],
        'cir5_updates_repeated':0, 'large_payloads':'remote_only', 'files':pinned,
        'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    write(complete, receipt)
    print('CIR_RECOVERY_SEED_ACCEPTED ni3=900 adam=900 scheduler=900 CIR5_REUSED', flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();project=Path.cwd()
    prepare(project,a.source,a.root)
    env=dict(os.environ);env['CIR_STORAGE_RETRY_ENABLED']='1'
    env['PYTHONPATH']=str(project/'recovery')+os.pathsep+str(project/'src')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-m','new_interest.experiments.cir_schedule','--project',str(project),'--root',str(a.root),'--mode','staged'],env=env,check=True)


if __name__=='__main__':
    main()
