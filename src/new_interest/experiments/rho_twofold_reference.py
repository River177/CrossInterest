"""Bounded two-fold reference SFT and remote-only, independently read scoring."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import pyarrow.parquet as pq


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def verify_manifest(root, identity=None):
    root = Path(root)
    manifest = read(root / 'artifact_manifest.json')
    if not isinstance(manifest, dict) or not isinstance(manifest.get('identity'), dict):
        raise ValueError('Malformed committed training identity')
    if identity is not None and manifest['identity'] != identity:
        raise ValueError('Committed training identity differs')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Training artifact manifest is empty or malformed')
    required = {'run_summary.json', 'resolved_config.json', 'semantic_reload_reference.json',
                'semantic_adapter_reload_audit.json', 'final_adapter/adapter_model.safetensors',
                'final_adapter/adapter_config.json', 'final_adapter/interest_token_mapping.parquet',
                'final_adapter/tokenizer.json', 'final_adapter/tokenizer_config.json'}
    if required.difference(files):
        raise ValueError('Training manifest omits required artifacts: ' +
                         ', '.join(sorted(required.difference(files))))
    for name, item in files.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Unsafe training manifest path')
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or relative.as_posix() != name or name == 'artifact_manifest.json'):
            raise ValueError('Unsafe training manifest path: ' + name)
        if (not isinstance(item, dict) or not isinstance(item.get('bytes'), int)
                or isinstance(item['bytes'], bool) or item['bytes'] < 0
                or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Malformed training manifest entry: ' + name)
        p = root / name
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Training artifact is missing or unsafe: ' + name)
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('Artifact bytes differ: ' + name)
    # Consumers discover tokenizer/adapter files by name, so none of those
    # files may exist outside the authenticated inventory.
    adapter_files = {p.relative_to(root).as_posix() for p in (root / 'final_adapter').rglob('*')
                     if p.is_file()}
    if adapter_files.difference(files):
        raise ValueError('Unauthenticated adapter artifacts: ' + ', '.join(sorted(adapter_files.difference(files))))
    summary = read(root / 'run_summary.json')
    world = summary.get('world_size')
    if type(world) is not int or world not in (1, 4):
        raise ValueError('Training summary has unsupported world size')
    rank_files = {f'rank-{rank}-training-runtime.json' for rank in range(world)}
    if rank_files.difference(files):
        raise ValueError('Training manifest omits rank runtime audits')
    return manifest


def publish_training(local, target, identity):
    """Publish closed byte streams; final manifest is the commit marker."""
    if (target / 'artifact_manifest.json').exists():
        return verify_manifest(target, identity)
    files = {}
    for p in sorted(local.rglob('*')):
        if not p.is_file() or 'checkpoints' in p.relative_to(local).parts:
            continue
        name = p.relative_to(local).as_posix()
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(p)
        with p.open('rb') as source, dest.open('wb') as stream:
            shutil.copyfileobj(source, stream, 4 * 1024 * 1024)
        if sha(dest) != digest:
            raise ValueError('Training publication mismatch: ' + name)
        files[name] = {'bytes': p.stat().st_size, 'sha256': digest}
    write(target / 'artifact_manifest.json', {'identity': identity, 'files': files})
    return verify_manifest(target, identity)


def command(args, project, logfile, env=None):
    """Logs stay on node-local scratch during a subprocess, mirrored to native stdout."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w') as stream:
        p = subprocess.Popen(args, cwd=project, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            stream.write(line)
            stream.flush()
            if any(x in line for x in ('loss', 'Error', 'Traceback', 'CHECKPOINT', 'RHO_', '%|')):
                print(line.strip()[:1800], flush=True)
        code = p.wait()
    if code:
        print('\n'.join(logfile.read_text(errors='replace').splitlines()[-60:]), flush=True)
        raise RuntimeError(f'Worker subprocess exited {code}: {logfile.name}')


def training_config(protocol, model, mapping, train, output, persistent, stage, fold):
    cfg = dict(protocol['sft'])
    cfg.update(model_name=str(model), model_revision=protocol['model_revision'],
               train_file=str(train), semantic_id_token_mapping_file=str(mapping),
               output_dir=str(output), checkpoint_dir=str(output / 'checkpoints'),
               persistent_checkpoint_dir=str(persistent), resume_from_checkpoint='auto',
               run_name=f'rho-reference-fold{fold}-{stage}-seed{cfg["seed"]}')
    if stage == 'single':
        cfg.update(epochs=5, save_steps=10)
    elif stage == 'ddp':
        cfg.update(epochs=2, save_steps=2)
    return cfg


def stage_subset(source, rows, target, local):
    pq.write_table(pq.read_table(source).slice(0, rows), local)
    digest = sha(local)
    if target.exists():
        if sha(target) != digest:
            raise ValueError('Persisted gate subset differs from deterministic source')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    shutil.copyfile(local, temporary)
    if sha(temporary) != digest:
        raise ValueError('Gate subset publication failed')
    os.replace(temporary, target)
    return target


def accept_training(folder, rows, steps, world, epochs, require_decrease=False, *, expected_train_sha=None):
    import torch
    from safetensors.torch import load_file
    summary = read(folder / 'run_summary.json')
    if (summary['train_samples'], summary['world_size'], summary['configured_epochs']) != (rows, world, epochs):
        raise ValueError('Training sample/world/epoch mismatch')
    if expected_train_sha is not None and summary['train_file_sha256'] != expected_train_sha:
        raise ValueError('Actual training file differs from frozen fold')
    if (summary['seed'], summary['per_device_train_batch_size'], summary['gradient_accumulation_steps'],
            summary['expected_optimizer_steps']) != (42, 1, 8, steps):
        raise ValueError('Training seed/batch/expected steps differ')
    if summary['input_adapter'] is not None or summary['sample_weight_file'] is not None:
        raise ValueError('Reference cannot inherit supervised adapters or weights')
    if summary['semantic_id_loss_scope'] != 'full_vocabulary' or summary['loss_reduction'] != 'token_mean':
        raise ValueError('Reference native SFT loss changed')
    if summary['token_audit']['over_max_seq_length']:
        raise ValueError('Overlength training sample')
    audits = [read(folder / f'rank-{i}-training-runtime.json') for i in range(world)]
    initial = {x['initial_state']['state_sha256'] for x in audits}
    if len(initial) != 1:
        raise ValueError('Ranks did not start from identical parameters')
    for audit in audits:
        end = audit['train_end']
        if end['global_step'] != steps or abs(end['epoch'] - epochs) > 1e-8:
            raise ValueError('Actual optimizer/epoch cursor differs')
        if end['optimizer_step_min'] != steps or end['optimizer_step_max'] != steps:
            raise ValueError('Optimizer state is incomplete')
    losses = [float(x['loss']) for x in summary['loss_history']]
    if not losses or not all(math.isfinite(x) for x in losses):
        raise ValueError('Training losses missing or nonfinite')
    if require_decrease and sum(losses[-2:]) >= sum(losses[:2]):
        raise ValueError('Single-GPU repeated-sample loss did not decrease')
    state = load_file(str(folder / 'final_adapter/adapter_model.safetensors'))
    sid = [x for k, x in state.items() if 'trainable_tokens_delta' in k]
    if len(state) != 373 or len(sid) != 1 or list(sid[0].shape) != [108, 2048]:
        raise ValueError('Wrong compact LoRA/tied SID structure')
    if not all(torch.isfinite(x).all().item() for x in state.values()):
        raise ValueError('Nonfinite saved weights')
    reload = read(folder / 'semantic_adapter_reload_audit.json')
    if not reload['all_close'] or not reload['adapter_state_integrity']['all_tensors_exact']:
        raise ValueError('Fresh reload not exact')
    return {'passed': True, 'rows': rows, 'steps': steps, 'epochs': epochs, 'world_size': world,
            'initial_state_sha256': next(iter(initial)), 'adapter_tensor_count': len(state),
            'adapter_sha256': sha(folder / 'final_adapter/adapter_model.safetensors'),
            'train_file_sha256': summary['train_file_sha256'], 'fresh_reload_exact': True,
            'loss_first': losses[0], 'loss_last': losses[-1]}


def run_scores(project, output, adapter, model, train, score, mapping, protocol):
    configs = output / 'configs'
    configs.mkdir(parents=True, exist_ok=True)
    handles, processes, dirs = [], [], []
    for shard in range(4):
        directory = output / f'shard-{shard}'
        dirs.append(directory)
        cfg = {'model_name': str(model), 'model_revision': protocol['model_revision'],
               'adapter_path': str(adapter), 'teacher_train_file': str(train), 'score_file': str(score),
               'interest_token_mapping_file': str(mapping), 'output_dir': str(directory),
               'batch_size': 4, 'checkpoint_size': 256, 'num_shards': 4, 'shard_index': shard,
               'attn_implementation': 'sdpa', 'enable_thinking': False, 'max_seq_length': 2048}
        path = configs / f'score-{shard}.json'
        write(path, cfg)
        logfile = Path('/scratch') / f'rho-score-{uuid.uuid4().hex}.log'
        stream = logfile.open('w')
        handles.append((stream, logfile))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(shard)}
        processes.append(subprocess.Popen([sys.executable, '-m',
            'new_interest.evaluation.score_rho_reference', '--config', str(path)],
            cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT))
    try:
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('OOF scoring subprocess failed')
            print('RHO_REFERENCE_SCORING_ACTIVE ' + json.dumps({'rows': pq.read_metadata(score).num_rows}), flush=True)
            time.sleep(30)
        if any(p.returncode for p in processes):
            raise RuntimeError('OOF scoring failed')
    except BaseException:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for stream, log in handles:
            stream.flush()
            print('\n'.join(log.read_text(errors='replace').splitlines()[-40:]), flush=True)
        raise
    finally:
        for stream, log in handles:
            stream.close()
    return dirs


def worker(project, root, phase, fold):
    import torch
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    protocol = read(project / 'protocol.json')
    source = read(project / 'source_manifest.json')
    for name, digest in source['files'].items():
        if sha(project / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    inputs = {k: v for k, v in source['files'].items()
              if k.startswith('inputs/') or k in ('source_snapshot.json', 'token_audit.json',
                                                 'recovery_parent.json', 'reused_token_gradient_audit.json')}
    inputs_sha = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    identity = {'protocol_sha256': sha(project / 'protocol.json'), 'experiment_inputs_sha256': inputs_sha,
                'runtime_sha256': source['runtime_sha256'], 'phase': phase, 'fold': fold}
    if (root / 'run_identity.json').exists() and read(root / 'run_identity.json') != identity:
        raise ValueError('Run output identity conflict')
    write(root / 'run_identity.json', identity)
    if phase == 'full':
        gate = read(project / 'gate_acceptance.json')
        if (not gate['passed'] or gate['runtime_sha256'] != source['runtime_sha256']
                or gate['identity']['protocol_sha256'] != identity['protocol_sha256']
                or gate['identity']['experiment_inputs_sha256'] != inputs_sha
                or gate['identity']['phase'] != 'gate' or gate['expected_oof_rows'] != 64):
            raise ValueError('Formal reference requires matching accepted gate')
    required = {'torch': '2.8.0+cu126', 'transformers': '5.14.1', 'peft': '0.18.0',
                'accelerate': '1.12.0', 'tokenizers': '0.22.2', 'safetensors': '0.8.0'}
    actual = {name: importlib.metadata.version(name) for name in required}
    if actual != required or torch.cuda.device_count() != 4:
        raise ValueError('Worker environment differs')
    write(root / 'compact/environment.json', {'packages': actual, 'gpus': [torch.cuda.get_device_name(i) for i in range(4)]})
    expected = read(project / 'source_snapshot.json')
    snapshot = Path(snapshot_download(protocol['model_name'], revision=protocol['model_revision'], local_files_only=True))
    model = Path('/scratch/rho-reference-model')
    for name, digest in expected['files'].items():
        dest = model / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or sha(dest) != digest:
            if sha(snapshot / name) != digest:
                raise ValueError('Original pretrained snapshot bytes differ')
            shutil.copyfile(snapshot / name, dest)
        if sha(dest) != digest:
            raise ValueError('Snapshot staging failed')
    write(root / 'compact/snapshot_audit.json', expected)
    train = project / 'inputs' / f'fold{fold}.parquet'
    opposite = project / 'inputs' / f'fold{1-fold}.parquet'
    mapping = project / 'inputs/interest_token_mapping.parquet'
    stages = ('single', 'ddp') if phase == 'gate' else ('full',)
    scratch = Path('/scratch') / f'rho-reference-{fold}-{phase}'
    scratch.mkdir(parents=True, exist_ok=True)
    recovery_parent = None
    if phase == 'gate' and (project / 'recovery_parent.json').exists():
        recovery_parent = read(project / 'recovery_parent.json')
        if recovery_parent['protocol_sha256'] != identity['protocol_sha256']:
            raise ValueError('Recovery changes the single-GPU training protocol')
        for name, digest in recovery_parent['unchanged_source_files'].items():
            if source['files'].get(name) != digest:
                raise ValueError('Recovery changes a bound mathematical/data input: ' + name)
        reused_gradient = project / 'reused_token_gradient_audit.json'
        if sha(reused_gradient) != recovery_parent['gradient_audit_sha256']:
            raise ValueError('Reused gradient receipt changed')
        if sha(project / 'src/new_interest/training/audit_semantic_id_token_gradients.py') != recovery_parent['gradient_audit_source_sha256']:
            raise ValueError('Gradient audit implementation changed')
        shutil.copyfile(reused_gradient, root / 'compact/token_gradient_audit.json')
    train_acceptances = {}
    for stage in stages:
        world = 1 if stage == 'single' else 4
        rows = 32 if stage == 'single' else (64 if stage == 'ddp' else 10800)
        epochs = 5 if stage == 'single' else (2 if stage == 'ddp' else 3)
        steps = math.ceil(rows / (world * 8)) * epochs
        local = scratch / stage
        local.mkdir(parents=True, exist_ok=True)
        current_train = train
        if stage != 'full':
            current_train = stage_subset(train, rows, root / 'inputs' / f'{stage}-train.parquet',
                                         scratch / f'{stage}-train.parquet')
        stage_identity = {**identity, 'stage': stage, 'train_sha256': sha(current_train)}
        target = root / stage
        if not (target / 'artifact_manifest.json').exists():
            cfg = training_config(protocol, model, mapping, current_train, local,
                                  root / f'checkpoints-{stage}', stage, fold)
            prepared_recovery = None
            if stage == 'single' and recovery_parent is not None:
                from new_interest.experiments.rho_reference_recovery import prepare_single_recovery
                prepared_recovery = prepare_single_recovery(project, cfg, scratch / 'restored-parent')
                cfg['resume_from_checkpoint'] = prepared_recovery['resume_from_checkpoint']
            cfg_path = scratch / f'{stage}-config.json'
            write(cfg_path, cfg)
            if stage == 'single' and not (root / 'compact/token_gradient_audit.json').exists():
                gradient_config = {**cfg, 'train_file': str(train)}
                gradient_path = scratch / 'gradient-config.json'
                write(gradient_path, gradient_config)
                command([sys.executable, '-m', 'new_interest.training.audit_semantic_id_token_gradients',
                         '--config', str(gradient_path), '--batch-size', '1', '--output',
                         str(root / 'compact/token_gradient_audit.json')], project, scratch / 'gradient.log')
            env = {**os.environ, 'RHO_SYNC_DIR': str(scratch / f'sync-{stage}-{uuid.uuid4().hex}')}
            command([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}',
                     '-m', 'new_interest.training.rho_reference_sft', '--config', str(cfg_path)],
                    project, scratch / f'train-{stage}.log', env)
            command([sys.executable, '-m', 'new_interest.training.verify_semantic_id_token_adapter',
                     '--model-name', str(model), '--adapter-path', str(local / 'final_adapter'),
                     '--mapping-file', str(mapping), '--reference-file', str(local / 'semantic_reload_reference.json'),
                     '--output', str(local / 'semantic_adapter_reload_audit.json'), '--atol', '0', '--rtol', '0'],
                    project, scratch / f'reload-{stage}.log')
            if prepared_recovery is not None:
                from new_interest.experiments.rho_reference_recovery import validate_single_recovery
                validate_single_recovery(prepared_recovery, local, root / 'compact/single_recovery_acceptance.json')
            accept_training(local, rows, steps, world, epochs, stage == 'single', expected_train_sha=sha(current_train))
            publish_training(local, target, stage_identity)
        verify_manifest(target, stage_identity)
        train_acceptances[stage] = accept_training(target, rows, steps, world, epochs, stage == 'single', expected_train_sha=sha(current_train))
        print('RHO_REFERENCE_TRAINING_ACCEPTED ' + json.dumps(train_acceptances[stage]), flush=True)
    write(root / 'compact/training_acceptance.json', train_acceptances)
    stage = stages[-1]
    score = opposite
    if phase == 'gate':
        if recovery_parent is not None:
            recovered = read(root / 'compact/single_recovery_acceptance.json')
            if not recovered['accepted'] or recovered['new_optimizer_updates'] != 0:
                raise ValueError('Single-GPU recovery did not preserve the completed endpoint')
        gradient = read(root / 'compact/token_gradient_audit.json')
        if gradient['representative_samples'] != 108 or gradient['interest_token_count'] != 108:
            raise ValueError('Incomplete SID gradient coverage')
        if len(gradient['gradient_parameters']) != 1 or any(
                r['finite_rows'] != 108 or r['nonzero_rows'] != 108 for r in gradient['gradient_parameters']):
            raise ValueError('SID gradients are not all finite and nonzero')
        score = stage_subset(opposite, 64, root / 'inputs/gate-score.parquet', scratch / 'gate-score.parquet')
    dirs = run_scores(project, root / 'oof', root / stage / 'final_adapter', model,
                      current_train, score, mapping, protocol)
    # Independent process rereads all remote score shards and recomputes coverage/NLL/ranks.
    acceptance = root / 'compact/reference_acceptance.json'
    command([sys.executable, '-m', 'new_interest.experiments.read_rho_reference', '--score-dirs',
             *map(str, dirs), '--teacher-train-file', str(current_train), '--score-file', str(score),
             '--mapping-file', str(mapping), '--output', str(acceptance)], project, scratch / 'readback.log')
    result = read(acceptance)
    if not result['accepted'] or result['metrics']['rows'] != (64 if phase == 'gate' else 10800):
        raise ValueError('Independent OOF reader rejected scoring')
    success = {'passed': True, 'identity': identity, 'runtime_sha256': source['runtime_sha256'],
               'training': train_acceptances, 'reference_acceptance_sha256': sha(acceptance),
               'expected_oof_rows': 64 if phase == 'gate' else 10800,
               'train_file_sha256': sha(current_train), 'parent_fold_sha256': sha(train), 'score_file_sha256': sha(score),
               'remote_output': str(root), 'large_artifact_policy': 'remote_only',
               'next_stage': 'reference_scoring_only; no automatic student training'}
    write(root / 'compact/worker_success.json', success)
    if phase == 'gate':
        write(root / 'compact/gate_acceptance.json', success)
    print('RHO_REFERENCE_WORKER_SUCCESS ' + json.dumps(success), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('gate', 'full'), required=True)
    parser.add_argument('--fold', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    worker(args.project, args.root, args.phase, args.fold)


if __name__ == '__main__':
    main()
