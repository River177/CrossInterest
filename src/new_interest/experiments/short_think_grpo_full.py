"""Formal GRPO segments with committed25-update checkpoints and canonical Dev eval.

Prepared offline; the worker requires independently accepted throughput-engine
artifacts before executing. A100 H segments stop at100 then continue the SAME
optimizer/data cursor to675. HL may branch only from its bound H100 parent.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import uuid

from new_interest.evaluation.evaluate_beam import _atomic_write_json
from new_interest.evaluation.preflight_short_think_grpo import read,sha,verify_files
from new_interest.experiments.short_think_grpo_engine import command,publish
from new_interest.evaluation.evaluate_short_think_ranking import merge


def choose_checkpoint(output,identity,stop_at):
    candidates=[]
    for p in output.glob('attempts/*/checkpoint-*/checkpoint_manifest.json'):
        match=re.fullmatch(r'checkpoint-(\d+)',p.parent.name)
        if not match:continue
        m=read(p)
        if m['identity']!=identity:raise ValueError('Output root contains another experiment identity')
        state=read(p.parent/'state.json');step=int(match[1])
        if state['identity']!=identity or state['step']!=step or step>stop_at:raise ValueError('Checkpoint identity/cursor differs')
        candidates.append((step,sha(p),p.parent))
    if not candidates:return None
    best=max(x[0] for x in candidates);top=[x for x in candidates if x[0]==best]
    if len({x[1] for x in top})!=1:raise ValueError('Conflicting committed attempts at same step')
    return sorted(top,key=lambda x:str(x[2]))[0][2]


def hydrate(source,destination):
    manifest=read(source/'checkpoint_manifest.json');destination.mkdir(parents=True,exist_ok=False)
    def copy_verified(name):
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe checkpoint member')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
        if sha(target)!=manifest['files'][name]:raise ValueError(f'Hydrated file differs: {name}')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy_verified,manifest['files']))
    return read(destination/'state.json')


def evaluate(project,root,model_path,adapter,full):
    dataset=project/'inputs'/('dev.parquet' if full else 'diagnostic_dev.parquet')
    expected=31392 if full else 2048
    configs=root/'eval_configs';configs.mkdir()
    folders=[];streams=[];processes=[]
    for rank in range(4):
        dest=root/'evaluation'/f'shard-{rank}';folders.append(dest)
        cfg={'model_name':str(model_path),'model_revision':read(project/'contract.json')['model_revision'],
             'adapter_path':str(adapter),'eval_file':str(dataset),'interest_token_mapping_file':str(project/'inputs/interest_token_mapping.parquet'),
             'output_dir':str(dest),'batch_size':8,'checkpoint_size':16,'max_new_tokens':512,'num_shards':4,'shard_index':rank}
        path=configs/f'{rank}.json';_atomic_write_json(path,cfg)
        stream=(root/f'eval-{rank}.log').open('w');streams.append(stream)
        processes.append(subprocess.Popen([sys.executable,'-m','new_interest.evaluation.evaluate_short_think_ranking','evaluate','--config',str(path)],
            cwd=project,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(rank)},stdout=stream,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for stream in streams:stream.close()
    if any(codes):raise RuntimeError(f'Dev evaluator failed: {codes}')
    return merge(folders,dataset,project/'inputs/interest_token_mapping.parquet',root/'evaluation/merged',expected)


def worker(project,output,stop_at,parent_blob=None):
    import torch
    from huggingface_hub import snapshot_download
    project=Path(project);output=Path(output)
    sources=read(project/'source_manifest.json');verify_files(project,sources['files'])
    contract=read(project/'engine_contract.json');identity=sha(project/'engine_contract.json')
    if contract.get('role')!='production' or stop_at not in [100,675]:raise ValueError('Wrong formal segment')
    accepted=read(project/'throughput_engine_acceptance.json')
    if not accepted['passed'] or sha(project/'throughput_engine_acceptance.json')!=contract['throughput_acceptance_sha256']:
        raise ValueError('No independent throughput-engine acceptance')
    if not read(project/'preflight_acceptance.json')['reward_density_gate']:raise ValueError('No reward-density acceptance')
    if torch.cuda.device_count()!=4:raise ValueError('Four GPUs required')
    for marker in output.glob('completed-*.json'):
        package=read(marker)
        if package['contract_sha256']!=identity:raise ValueError('Completed artifact identity differs')
        committed=output/'attempts'/package['attempt']
        segment=read(committed/'segment_complete.json')
        if segment['end_step']==stop_at:
            manifest=committed/'artifact_manifest.json'
            if sha(manifest)!=package['manifest_sha256']:raise ValueError('Completed manifest binding differs')
            verify_files(committed,read(manifest)['files'])
            archive=committed/'results.tar.gz'
            if archive.stat().st_size!=package['bytes'] or sha(archive)!=package['sha256']:raise ValueError('Completed archive differs')
            print('GRPO_FULL_REUSE_COMMITTED',package['attempt'],stop_at,flush=True)
            return
    existing=choose_checkpoint(output,identity,stop_at)
    if parent_blob:
        relative=PurePosixPath(parent_blob)
        if relative.is_absolute() or '..' in relative.parts or not parent_blob.startswith('projects/sft4safety/amlt-results/'):
            raise ValueError('Invalid parent blob path')
        if existing is None:existing=Path('/mnt/default')/parent_blob
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    root=output/'attempts'/attempt;root.mkdir(parents=True,exist_ok=False)
    local=Path('/scratch')/f'grpo-full-{attempt}';local.mkdir()
    _atomic_write_json(root/'identity.json',{'engine_identity':identity,'attempt':attempt,'arm':contract['arm'],'stop_at':stop_at,'planned_steps':675})
    source=read(project/'source_snapshot.json');snapshot=Path(snapshot_download(source['model_name'],revision=source['model_revision'],local_files_only=True))
    verify_files(snapshot,source['files']);model_path=local/'model';model_path.mkdir()
    for name in source['files']:
        dest=model_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(snapshot/name,dest)
    verify_files(model_path,source['files'])
    parent=None;start=0
    if existing is not None:
        parent=local/'initial_parent';saved=hydrate(existing,parent);start=saved['step']
        from new_interest.training.short_think_grpo_engine import validate_resume_identity
        validate_resume_identity(saved,identity,contract,parent)
        _atomic_write_json(root/'initial_parent_reference.json',{'path':str(existing),
            'manifest_sha256':sha(existing/'checkpoint_manifest.json'),'state_sha256':sha(existing/'state.json'),'step':start,'identity':saved['identity']})
    if start>stop_at or start%25:raise ValueError('Unapproved checkpoint boundary')
    if contract['arm']=='HL' and start<100:raise ValueError('HL must use the approved H100 parent')
    base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','new_interest.training.short_think_grpo_engine',
        '--project',str(project),'--model-path',str(model_path)]
    final=parent
    for end in range(start+25,stop_at+1,25):
        current=local/f'checkpoint-{end}'
        args=[*base,'--root',str(current),'--stop-at',str(end)]
        if parent:args.extend(['--resume',str(parent)])
        command(args,project,root/f'block-{end}.log',failure_source=current)
        persisted=root/f'checkpoint-{end}';publish(current,persisted,identity)
        # Rehydrate the committed data for the next process, exercising the same
        # exact model/optimizer/RNG/cursor checks at every boundary.
        parent=local/f'hydrated-{end}';hydrate(persisted,parent);final=parent
        _atomic_write_json(root/f'committed-{end}.json',{'engine_identity':identity,'step':end,
            'checkpoint_manifest_sha256':sha(persisted/'checkpoint_manifest.json'),'planned_steps':675})
        print(f'GRPO_FULL_COMMITTED arm={contract["arm"]} step={end}/675',flush=True)
    if final is None:raise ValueError('No model to evaluate')
    command([sys.executable,'-m','new_interest.training.short_think_grpo_engine','--project',str(project),
        '--root',str(final),'--model-path',str(model_path),'--stop-at',str(stop_at),'--verify-only',
        '--verification-output',str(root/'fresh_reload.json')],project,root/'fresh_reload.log',env={**os.environ,'LOCAL_RANK':'0'})
    metrics=evaluate(project,root,model_path,final/'adapter',stop_at==675)
    _atomic_write_json(root/'segment_complete.json',{'engine_identity':identity,'arm':contract['arm'],
        'start_step':start,'end_step':stop_at,'planned_steps':675,'full_training_complete':stop_at==675,
        'evaluation_scope':'full Dev' if stop_at==675 else 'fixed2048 Dev diagnostic; tuning data, not Test',
        'metrics':metrics,'final_state':read(final/'state.json')})
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    _atomic_write_json(root/'artifact_manifest.json',{'engine_identity':identity,'files':files})
    archive=local/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*files,'artifact_manifest.json']:tar.add(root/name,arcname=name,recursive=False)
    shutil.copyfile(archive,root/'results.tar.gz')
    package={'attempt':attempt,'sha256':sha(root/'results.tar.gz'),'bytes':archive.stat().st_size,
        'contract_sha256':identity,'manifest_sha256':sha(root/'artifact_manifest.json')}
    _atomic_write_json(root/'package.json',package);_atomic_write_json(output/f'completed-{attempt}.json',package)
    print('GRPO_FULL_SEGMENT_COMPLETE',contract['arm'],stop_at,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stop-at',type=int,choices=[100,675],required=True);p.add_argument('--parent-blob')
    a=p.parse_args();worker(a.project,a.root,a.stop_at,a.parent_blob)


if __name__=='__main__':main()
