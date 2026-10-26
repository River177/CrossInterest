"""Evaluate the unchanged H5 adapter with its verified FLA launch profile.

Run against the original frozen bundle, with staged inputs under --project.
The profile is inside final_adapter so the native evaluation identity binds it.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from new_interest.experiments.history_length_ablation import bind, read, write, require, sha256, subset_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'shard', 'verify', 'finish'])
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    project, source, root = args.project.resolve(), args.source.resolve(), args.root.resolve()
    baseline, contract = bind(project, 5)
    sft = root / 'sft'
    adapter = sft / 'final_adapter'
    profile = adapter / 'fla_norm_profile.json'
    if args.mode == 'run':
        root.mkdir(parents=True, exist_ok=True)
        if not adapter.exists():
            shutil.copytree(source / 'adapter', adapter)
            shutil.copyfile(source / 'norm_profile_w16.json', profile)
        require(sha256(profile) == sha256(source / 'norm_profile_w16.json'), 'Profile changed')
        for src, dst in [('reference.json', 'semantic_reload_reference.json'), ('run_summary.json', 'run_summary.json')]:
            shutil.copyfile(source / src, sft / dst)
        for item in contract['input_files']:
            require(sha256(project / item['path']) == item['sha256'], 'Frozen input changed')
        write(root / 'selection.json', {
            'selection': 'original H5 final step2025, fixed norm launch geometry',
            'replacement_recovery_model_used': False,
            'original_adapter_sha256': sha256(source / 'adapter/adapter_model.safetensors'),
            'original_reference_sha256': sha256(source / 'reference.json'),
            'profile_sha256': sha256(profile),
            'selection_basis': 'Two fresh processes match all432 original reference logits exactly; no Dev metrics used',
        })
        for mode, shard in [('verify', None)] + [('shard', i) for i in range(4)] + [('finish', None)]:
            command = [sys.executable, __file__, mode, '--project', str(project), '--source', str(source), '--root', str(root)]
            if shard is not None:
                command += ['--shard', str(shard)]
            log = root / 'logs' / f'{mode}-{shard}.log'
            log.parent.mkdir(exist_ok=True)
            with log.open('a') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            print('H5_PROFILE_PHASE_COMPLETE', mode, shard, flush=True)
    elif args.mode in {'shard', 'verify'}:
        from new_interest.training.fla_norm_profile import apply_fla_norm_profile
        receipt = apply_fla_norm_profile(profile, adapter_path=adapter, reference_file=sft / 'semantic_reload_reference.json')
        if args.mode == 'verify':
            from new_interest.training.verify_semantic_id_token_adapter import verify_semantic_id_token_adapter
            verify_semantic_id_token_adapter(contract['models']['2b']['model_name'], adapter,
                project / baseline.MAPPING, sft / 'semantic_reload_reference.json',
                sft / 'semantic_adapter_reload_audit.json',
                model_revision=contract['models']['2b']['model_revision'], atol=0, rtol=0)
            write(sft / 'profile_applied.json', receipt)
        else:
            from new_interest.evaluation.evaluate_id_token_ranking import evaluate_id_token_ranking
            config = read(project / 'configs/id_token/history_h5.json')
            output = root / f'real_eval/shard-{args.shard:02d}'
            config.update(adapter_path=str(adapter), eval_file=str(project / baseline.DEV),
                interest_token_mapping_file=str(project / baseline.MAPPING), output_dir=str(output),
                num_shards=4, shard_index=args.shard, batch_size=8)
            require(config.get('max_examples') is None, 'Full evaluation must not be capped')
            write(output / 'profile_applied.json', receipt)
            write(output / 'execution_config.json', config)
            evaluate_id_token_ranking(config)
    else:
        import pyarrow.parquet as pq
        from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
        merge_id_token_shards([root / f'real_eval/shard-{i:02d}' for i in range(4)], project / baseline.DEV,
            project / baseline.MAPPING, root / 'real_eval/merged', expected_records=31392)
        evaluation = baseline.audit_evaluation(project, '2b', root, 'full')
        training = {'adapter': baseline.audit_adapter(project, '2b', sft), 'reload': baseline.audit_reload(project, '2b', sft)}
        cohort = set(read(project / 'data/training/mind_history_length_ablation_20260915_v1/dev_h50_cohort.json')['impression_ids'])
        cohort_result = subset_metrics(pq.read_table(project / baseline.DEV).to_pylist(),
            pq.read_table(root / 'real_eval/merged/id_token_predictions.parquet').to_pylist(), cohort)
        write(root / 'evaluation_readback.json', evaluation)
        write(root / 'cohort_metrics.json', cohort_result)
        write(root / 'training_reload_readback.json', training)
        for i in range(4):
            require(read(root / f'real_eval/shard-{i:02d}/profile_applied.json')['profile_sha256'] == sha256(profile), 'Shard profile differs')
        files = {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in root.rglob('*') if p.is_file() and 'logs' not in p.parts
                 and p.name not in {'artifact_manifest.json', 'complete.json'} and not p.name.endswith('.partial')}
        write(root / 'artifact_manifest.json', {'files': files})
        for relative, item in files.items():
            require(sha256(root / relative) == item['sha256'], 'Remote readback failed')
        write(root / 'complete.json', {'status': 'original_h5_profiled_full_dev_verified',
            'history_limit': 5, 'selection': read(root / 'selection.json'), 'training': training,
            'evaluation': evaluation, 'h50_cohort': cohort_result,
            'artifact_manifest_sha256': sha256(root / 'artifact_manifest.json')})
        print('H5_ORIGINAL_PROFILE_FULL_DEV_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
