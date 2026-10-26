"""Recover the last 25 H5 updates from the committed parent and complete Full Dev."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pyarrow.parquet as pq

from diagnose_history_h5_trainer import stage_checkpoint
from new_interest.experiments.history_length_ablation import bind, subset_metrics, verify_bundle
from new_interest.experiments import backbone_four_gpu as execution
from new_interest.training.execution_migration import checkpoint_state_audit


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.partial')
    temp.write_text(json.dumps(v, indent=2) + '\n')
    temp.replace(p)


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def publish_tree(source, destination):
    files = {}
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.name.endswith('.partial'):
            continue
        target = destination / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.partial')
        shutil.copyfile(p, temp)
        assert sha(temp) == sha(p)
        temp.replace(target)
        files[str(p.relative_to(source))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return files


def main():
    project = Path(os.environ['AMLT_CODE_DIR']).resolve()
    root = Path(os.environ['AMLT_OUTPUT_DIR']).resolve()
    bundle_hash = verify_bundle(project)
    if (root / 'complete.json').is_file():
        completed = read(root / 'complete.json')
        assert completed['bundle_sha256'] == bundle_hash
        existing = root / completed['attempt']
        assert sha(existing / 'artifact_manifest.json') == completed['artifact_manifest_sha256']
        for name, entry in read(existing / 'artifact_manifest.json')['files'].items():
            assert sha(existing / name) == entry['sha256']
        print('HISTORY_H5_RECOVERY_REUSED', completed['attempt'], flush=True)
        return
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = root / 'attempts' / attempt_id
    scratch = Path('/scratch') / ('h5-recovery-' + attempt_id)
    scratch.mkdir(parents=True)
    out.mkdir(parents=True)
    shutil.copyfile(root / 'kernel_environment.json', out / 'kernel_environment.json')
    baseline, contract = bind(project, 5)
    source_info = read(project / 'trainer_diagnostic_source.json')
    source = Path('/mnt/default') / source_info['results_prefix']
    parent = scratch / 'parent/checkpoint-2000'
    marker = stage_checkpoint(source / 'final/checkpoints/checkpoint-2000', parent,
                              source_info['checkpoint_marker_sha256']['2000'])
    parent_audit = checkpoint_state_audit(parent, marker['identity'])
    assert parent_audit['global_step'] == 2000 and parent_audit['max_steps'] == 2025
    write(out / 'recovery_parent.json', {'source': source_info, 'selected_step': 2000,
                                       'audit': parent_audit, 'bundle_sha256': bundle_hash})
    execution.environment_audit(project, out / 'environment.json')
    baseline.prepare_snapshot(project, '2b', out / 'snapshot_audit.json')
    assert read(out / 'snapshot_audit.json')['files'] == read(project / 'manifests/h50_snapshot.json')['files']
    sft = scratch / 'sft'
    sft.mkdir()
    shutil.copyfile(source / 'final/sft/execution_migration_start.json', sft / 'execution_migration_start.json')
    config = read(source / 'final/training_config.json')
    config.update(train_file=str(project / baseline.TRAIN),
                  semantic_id_token_mapping_file=str(project / baseline.MAPPING),
                  output_dir=str(sft), checkpoint_dir=str(scratch / 'checkpoints'),
                  persistent_checkpoint_dir=str(out / 'checkpoints'), resume_from_checkpoint=str(parent),
                  execution_migration={'source_checkpoint': str(source / 'single/checkpoints/checkpoint-500'),
                                       'source_stage_artifacts': str(source / 'single/source-stage'),
                                       'cursor_contract_file': str(source / 'single/cursor_contract.json')})
    write(scratch / 'config.json', config)
    write(out / 'resolved_recovery_config.json', config)

    def run(command, log, devices=None):
        env = dict(os.environ)
        if devices is not None:
            env['CUDA_VISIBLE_DEVICES'] = devices
        code = execution.run(command, project, scratch / 'logs' / log, env=env)
        destination = out / 'logs' / log
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scratch / 'logs' / log, destination)
        if code:
            raise RuntimeError(f'Recovery phase failed: {log}, exit={code}')

    try:
        print('HISTORY_H5_RECOVERY_TRAIN_START', attempt_id, flush=True)
        run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '4',
             '-m', 'new_interest.training.train_sft', '--config', str(scratch / 'config.json')], 'train.log')
        summary = read(sft / 'run_summary.json')
        assert summary['actual_optimizer_steps'] == summary['expected_optimizer_steps'] == 2025
        assert summary['train_samples'] == 21600 and summary['configured_epochs'] == 3
        assert summary['world_size'] == 4 and summary['effective_global_batch_size'] == 32
        expected_recipe = {**contract['recipe'], 'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 1}
        baseline.audit_resolved_recipe(read(sft / 'resolved_config.json'), summary['checkpoint_identity'], expected_recipe)
        assert summary['train_file_sha256'] == marker['identity']['train_file_sha256']
        assert summary['loss_reduction'] == 'token_mean' and summary['semantic_id_loss_scope'] == 'full_vocabulary'
        assert summary['token_audit']['samples'] == 21600 and summary['token_audit']['over_max_seq_length'] == 0
        cursor = read(sft / 'training_sample_cursor.json')
        assert cursor['consumed_global_samples'] == 64800 and cursor['actual_updates_audited_this_attempt'] == 25
        assert cursor['consumed_prefix_sha256'] == contract['h50_final_sample_prefix_sha256']
        assert cursor['all_rank_real_sample_order_exact']
        for attempt in (sft / 'execution_attempts').glob('attempt-*'):
            start = read(attempt / 'execution_start.json')
            assert start['global_step'] == 2000 and all(not rank['differences'] for rank in start['rank_restores'])
            for rank in range(4):
                rng = read(attempt / f'rng_rank_{rank}.json')
                assert rng['status'] == 'passed' and rng['expected_rng_sha256'] == rng['actual_rng_sha256']
        run([sys.executable, '-m', 'new_interest.training.verify_semantic_id_token_adapter',
             '--model-name', config['model_name'], '--model-revision', config['model_revision'],
             '--adapter-path', str(sft / 'final_adapter'), '--mapping-file', str(project / baseline.MAPPING),
             '--reference-file', str(sft / 'semantic_reload_reference.json'),
             '--output', str(sft / 'semantic_adapter_reload_audit.json'), '--atol', '0', '--rtol', '0'], 'reload.log', '0')
        training = {'status': 'passed', 'parent_step': 2000, 'new_optimizer_updates': 25,
                    'final_step': 2025, 'samples': 21600, 'epochs': 3,
                    'adapter': baseline.audit_adapter(project, '2b', sft),
                    'reload': baseline.audit_reload(project, '2b', sft), 'cursor': cursor}
        write(sft / 'reference_binding.json', {'parent_marker_sha256': source_info['checkpoint_marker_sha256']['2000'],
            'final_step': 2025, 'adapter_sha256': sha(sft / 'final_adapter/adapter_model.safetensors'),
            'reference_sha256': sha(sft / 'semantic_reload_reference.json'),
            'reload_audit_sha256': sha(sft / 'semantic_adapter_reload_audit.json')})
        sft_manifest = publish_tree(sft, out / 'sft')
        write(out / 'sft_manifest.json', sft_manifest)
        write(out / 'training_accepted.json', training)
        print('HISTORY_H5_RECOVERY_RELOAD_PASSED', attempt_id, flush=True)

        def evaluate(shard):
            run([sys.executable, '-m', 'new_interest.evaluation.evaluate_id_token_ranking',
                 '--config', str(project / 'configs/id_token/history_h5.json'),
                 '--adapter-path', str(sft / 'final_adapter'), '--output-dir', str(out / f'real_eval/shard-{shard:02d}'),
                 '--num-shards', '4', '--shard-index', str(shard), '--batch-size', '8'], f'eval-{shard}.log', str(shard))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(evaluate, range(4)))
        from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
        merge_id_token_shards([out / f'real_eval/shard-{i:02d}' for i in range(4)], project / baseline.DEV,
                             project / baseline.MAPPING, out / 'real_eval/merged', expected_records=31392)
        evaluation = baseline.audit_evaluation(project, '2b', out, 'full')
        ids = set(read(project / 'data/training/mind_history_length_ablation_20260915_v1/dev_h50_cohort.json')['impression_ids'])
        cohort = subset_metrics(pq.read_table(project / baseline.DEV).to_pylist(),
                                pq.read_table(out / 'real_eval/merged/id_token_predictions.parquet').to_pylist(), ids)
        write(out / 'evaluation_readback.json', evaluation)
        write(out / 'cohort_metrics.json', cohort)
        manifest = {str(p.relative_to(out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                    for p in out.rglob('*') if p.is_file() and 'checkpoints' not in p.parts and not p.name.endswith('.partial')}
        write(out / 'artifact_manifest.json', {'files': manifest})
        for name, info in manifest.items():
            assert sha(out / name) == info['sha256']
        complete = {'status': 'recovery_training_reload_full_dev_accepted', 'attempt': str(out.relative_to(root)),
                    'history_limit': 5, 'bundle_sha256': bundle_hash, 'training': training, 'evaluation': evaluation,
                    'cohort': cohort, 'artifact_manifest_sha256': sha(out / 'artifact_manifest.json')}
        write(out / 'complete.json', complete)
        write(root / 'complete.json', complete)
        print('HISTORY_H5_RECOVERY_COMPLETE', attempt_id, flush=True)
    except BaseException:
        publish_tree(sft, out / 'failed_sft')
        raise


if __name__ == '__main__':
    main()
