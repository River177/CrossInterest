"""Orchestrate and collect the approved no-update MIND GRPO preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import _atomic_write_json, _atomic_write_parquet
from new_interest.evaluation.preflight_short_think_grpo import read, sha, verify_files
from new_interest.training.short_think_grpo_reward import group_statistics


def collect(project, root):
    identity=sha(project/'contract.json'); rows=[]; parities=[]
    expected=pq.read_table(project/'inputs/preflight.parquet').to_pylist()
    for i in range(4):
        shard=root/f'shard-{i}'
        complete=read(shard/'complete.json')
        if complete!={'contract_sha256':identity,'groups':81,'trajectories':1296}:
            raise ValueError('Shard completion identity differs')
        for group in range(i,324,4):
            local=pq.read_table(shard/f'group-{group:03d}.parquet').to_pylist()
            if len(local)!=16: raise ValueError('Group coverage differs')
            for j,r in enumerate(local):
                if (r['contract_sha256']!=identity or r['group_index']!=group or r['generation_index']!=j
                    or r['source_sample_id']!=expected[group]['extra_info']['source_sample_id']):
                    raise ValueError('Row identity/order differs')
            rows.extend(local)
        parities.extend(read(shard/f'parity-{i:03d}.json')['rows'])
    rows.sort(key=lambda r:(r['group_index'],r['generation_index']))
    metrics=group_statistics(rows,generations=16)
    metrics['first8']=group_statistics([r for r in rows if r['generation_index']<8],generations=8)
    deviations=[abs(a-b) for row in parities for a,b in zip(row['cached_raw_logprobs'],row['fresh_raw_logprobs'])]
    import math
    if not deviations or not all(math.isfinite(x) for x in deviations): raise ValueError('Missing/nonfinite parity diagnostic')
    deviations.sort()
    numerical={'stages':len(parities),'sampled_tokens':len(deviations),
        'mean_absolute_logprob_difference':sum(deviations)/len(deviations),
        'p95_absolute_logprob_difference':deviations[math.ceil(.95*len(deviations))-1],
        'max_absolute_logprob_difference':max(deviations),
        'ratio_in_0_8_to_1_2_fraction':sum(.8 <= math.exp(max(-50,min(50,a-b))) <= 1.2
            for row in parities for a,b in zip(row['fresh_raw_logprobs'],row['cached_raw_logprobs']))/len(deviations),
        'fresh_repeat_max_abs':max(r['fresh_repeat_max_abs'] for r in parities),
        'scope':'raw model policy only; temperature/top-p behavior correction and backward/optimizer remain unvalidated'}
    if all('cached_behavior_logprobs' in row for row in parities):
        pairs=[(a,b) for row in parities for a,b in zip(row['cached_behavior_logprobs'],row['fresh_behavior_logprobs'])]
        finite=[abs(b-a) for a,b in pairs if b is not None]
        if not pairs or not all(math.isfinite(a) and (b is None or math.isfinite(b)) for a,b in pairs):
            raise ValueError('Invalid behavior probability diagnostic')
        numerical['behavior']={'tokens':len(pairs),'current_support_misses':sum(b is None for a,b in pairs),
            'mean_abs_logprob_difference_on_shared_support':sum(finite)/len(finite) if finite else None,
            'ratio_in_0_8_to_1_2_fraction':sum(b is not None and .8<=math.exp(max(-50,min(50,b-a)))<=1.2 for a,b in pairs)/len(pairs),
            'scope':'Post-phase-mask, temperature .8 and top-p .95; same sampled actions; not a backward/optimizer check'}
    density=(metrics['valid_rate']>=.95 and metrics['length_limit_rate']<=.05
             and metrics['variable_group_rates']['E']>=.20
             and metrics['first8']['variable_group_rates']['E']>=.10)
    metrics.update(contract_sha256=identity,model_updates=0,reward_density_gate=density,
        numerical_diagnostic=numerical,training_authorized_by_this_artifact=False,
        pending_gates=['independent_download_readback','rollout_training_policy_logprob_contract',
                       'four_gpu_backward_optimizer_save_reload_and_resume'])
    _atomic_write_parquet(root/'trajectories.parquet',rows)
    _atomic_write_json(root/'metrics.json',metrics)
    _atomic_write_json(root/'preflight_complete.json',{'contract_sha256':identity,'trajectories':5184,'model_updates':0})
    return metrics


def worker(project,output):
    import torch
    from huggingface_hub import snapshot_download
    project=Path(project);output=Path(output)
    verify_files(project,read(project/'source_manifest.json')['files'])
    contract=read(project/'contract.json')
    if torch.cuda.device_count()!=4: raise ValueError('Expected four GPUs')
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    root=output/'attempts'/attempt; root.mkdir(parents=True,exist_ok=False)
    _atomic_write_json(root/'identity.json',{'contract_sha256':sha(project/'contract.json'),
        'source_manifest_sha256':sha(project/'source_manifest.json'),'attempt':attempt})
    _atomic_write_json(root/'environment.json',{'gpus':[torch.cuda.get_device_name(i) for i in range(4)],
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','flash-linear-attention','causal-conv1d']}})
    expected=read(project/'source_snapshot.json')
    snapshot=Path(snapshot_download(contract['model_name'],revision=contract['model_revision'],local_files_only=True))
    verify_files(snapshot,expected['files'])
    local=Path('/scratch')/f'grpo-preflight-{attempt}';local.mkdir()
    for name in expected['files']:
        dest=local/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(snapshot/name,dest)
    verify_files(local,expected['files']);(root/'model_path.txt').write_text(str(local)+'\n')
    processes=[];streams=[]
    try:
        for i in range(4):
            stream=(root/f'worker-{i}.log').open('a');streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-m','new_interest.evaluation.preflight_short_think_grpo',
                '--project',str(project),'--root',str(root),'--shard-index',str(i)],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i)},stdout=stream,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        if any(codes): raise RuntimeError(f'Preflight shards failed: {codes}; attempt={attempt}')
        metrics=collect(project,root)
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        _atomic_write_json(root/'artifact_manifest.json',{'contract_sha256':sha(project/'contract.json'),'files':files})
        archive=Path('/scratch')/f'grpo-preflight-{attempt}.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name in [*files,'artifact_manifest.json']:tar.add(root/name,arcname=name,recursive=False)
        shutil.copyfile(archive,root/'results.tar.gz')
        package={'attempt':attempt,'sha256':sha(root/'results.tar.gz'),'bytes':archive.stat().st_size,
                 'contract_sha256':sha(project/'contract.json'),'manifest_sha256':sha(root/'artifact_manifest.json')}
        _atomic_write_json(root/'package.json',package)
        _atomic_write_json(output/f'completed-{attempt}.json',package)
        print('GRPO_PREFLIGHT_COMPLETE',json.dumps(metrics),flush=True)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for stream in streams:stream.close()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['worker','collect'])
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='worker':worker(args.project,args.root)
    else:print(json.dumps(collect(args.project,args.root)))


if __name__=='__main__':main()
