"""Freeze an evaluation-only diagnostic; reuse accepted numerical source files."""
from pathlib import Path
import copy
import json
import shutil

import pyarrow.parquet as pq
import yaml

from new_interest.experiments import cir_warmup as core
from new_interest.experiments.cir_candidate_competition import audit_data, TIE_POLICY


def main():
    root = Path.cwd()
    base = root/'artifacts/mind-cir-candidate-competition-20260924'
    source = root/'artifacts/mind-cir-parameter-20260923/bundle-v2'
    ni_root = Path('/Users/river/.codex/worktrees/3357/CrossInterest')
    ni_bundle = ni_root/'artifacts/mind-qwen35-four-gpu-migration-v1/bundle-full-2b-v1'
    bundle = base/'bundle-v1'
    if bundle.exists():
        raise FileExistsError(bundle)
    old = core.read(source/'manifests/transfer_bundle.json')
    core.require(core.sha256(source/'manifests/transfer_bundle.json') == '4af6c06788ae2b3365ec49319cc314d83dc1a50a3f44e827cb0e5c6fa4b93b05', 'Parent bundle changed')
    for rel, digest in old['files'].items():
        core.require(core.sha256(source/rel) == digest, 'Parent payload changed '+rel)
    baseline = core.read(ni_root/'manifests/mind_shared_2b_ni_only_final_baseline_v1.json')
    core.require(baseline['status'] == 'independently_accepted', 'NI-only not accepted')
    core.require(core.sha256(ni_root/baseline['acceptance_receipt']) == baseline['acceptance_receipt_sha256'], 'NI acceptance changed')
    transfer = core.read(source/'manifests/cir_parameter_transfer.json')
    models = {}
    for arm, prefix, adapter, afile, prediction, pred_hash, profile, batch in [
        ('ni_only', 'projects/sft4safety/amlt-results/7210843083.71371-a21f4d1c-d098', 'sft/final_adapter', baseline['adapter_sha256'],
         'real_eval/merged/id_token_predictions.parquet', baseline['predictions_sha256'], 'fla-0.5.2-causal-conv1d-1.7.0', 8),
        ('cir5_ni3', transfer['source_results_prefix'], 'formal/ni3/final_adapter', transfer['reference_ni3_adapter_sha256'],
         transfer['reference_prediction_relative_path'], transfer['reference_prediction_sha256'], 'torch-fallback-explicit', 2),
    ]:
        metrics_path = base/'source-readback'/f'{arm}-merged-metrics.json'
        metrics = core.read(metrics_path)
        expected_metric = baseline['metrics_sha256'] if arm == 'ni_only' else 'a002ac12daddd6e41d309319d9d8cac12a40ebfaabca898ded02cb3881f6d6c5'
        core.require(core.sha256(metrics_path) == expected_metric, 'Remote accepted metrics changed')
        models[arm] = {'results_prefix': prefix, 'adapter_path': adapter, 'adapter_file_sha256': afile,
                       'adapter_directory_sha256': metrics['adapter_sha256'], 'predictions_path': prediction,
                       'predictions_sha256': pred_hash, 'metrics_sha256': expected_metric,
                       'kernel_profile': profile, 'batch_size': batch}
    data = source/core.DATA
    audit = audit_data(pq.read_table(data/'dev_full_grouped.parquet').to_pylist(),
                       pq.read_table(data/'interest_token_mapping.parquet')['target_subcategory'].to_pylist())
    core.require((audit['records'], audit['target_occurrences'], audit['supported_classes']) == (31392,37728,104), 'Data coverage changed')
    contract = {'version': 'cir-candidate-competition-20260924-v1', 'models': models,
                'training_updates': 0, 'automatic_monitoring': False,
                'rankings': ['original-108-candidates', 'exclude-visible-history-intersect-108'],
                'tie_policy': TIE_POLICY, 'cutoffs': [1,5,10,20], 'data_audit': audit,
                'source_transfer_bundle_sha256': core.sha256(source/'manifests/transfer_bundle.json'),
                'comparison_limit': 'Mechanism screening: NI-only retains historical training and FLA execution differences. No additive causal attribution from gap attenuation.'}
    shutil.copytree(source, bundle, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.amltignore'))
    for rel in ['src/new_interest/experiments/cir_candidate_competition.py', 'scripts/amlt/sft/run_cir_candidate_competition.sh']:
        (bundle/rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(root/rel, bundle/rel)
    for name in ['install_manifest.json', 'install_fast_kernels.py']:
        dest = bundle/'migration_kernel'/name; dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(ni_bundle/'migration_kernel'/name, dest)
    for rel in ['src/new_interest/evaluation/evaluate_id_token_ranking.py', 'src/new_interest/training/semantic_id_tokens.py', 'src/new_interest/training/sft_data.py']:
        core.require(core.sha256(bundle/rel) == core.sha256(ni_bundle/rel), 'NI inference source differs')
    for arm in models:
        p = bundle/'manifests/sources'/f'{arm}-metrics.json'; p.parent.mkdir(exist_ok=True)
        shutil.copyfile(base/'source-readback'/f'{arm}-merged-metrics.json', p)
    core.write(bundle/'manifests/candidate_competition.json', contract)
    files = {str(p.relative_to(bundle)): core.sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    core.write(bundle/'manifests/competition_bundle.json', {'version':contract['version'], 'files':files})
    cfg = yaml.safe_load((root/'configs/amlt/mind_cir_parameter_transfer_0923_v2.yaml').read_text())
    cfg['description'] = 'Evaluation only: accepted NI-only and CIR5->NI3; full108 scores, exact original Top20 reproduction, paired visible-history exclusion.'
    cfg['code']['local_dir'] = '$CONFIG_DIR/../../artifacts/mind-cir-candidate-competition-20260924/bundle-v1'
    template = cfg['jobs'][0];cfg['jobs'] = []
    for arm in models:
        job = copy.deepcopy(template);job['name'] = 'score-'+arm.replace('_','-')
        job['submit_args']['env'].pop('CIR_TRANSFER_ARM')
        job['submit_args']['env']['COMPETITION_ARM'] = arm
        job['command'][-1] = 'bash scripts/amlt/sft/run_cir_candidate_competition.sh'
        cfg['jobs'].append(job)
    config_path = root/'configs/amlt/mind_cir_candidate_competition_0924_v1.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tracking = {'status':'prepared', 'experiment':'mind-cir-candidate-competition-0924-v1',
                'bundle':str(bundle.relative_to(root)), 'bundle_sha256':core.sha256(bundle/'manifests/competition_bundle.json'),
                'config':str(config_path.relative_to(root)), 'contract':contract,
                'resources':{'jobs':2,'gpus_per_job':4,'max_gpus':8,'target':'Recall','sla':'Standard'},
                'jobs':{arm:{'job':'score-'+arm.replace('_','-')} for arm in models}}
    core.write(root/'manifests/mind_cir_candidate_competition_20260924_v1.json',tracking)
    core.write(base/'local_preflight.json', {'status':'passed','data':audit,'unchanged_parent_payloads':len(old['files']),
                                          'bundle_sha256':tracking['bundle_sha256']})
    print(json.dumps({'status':'prepared','bundle_sha256':tracking['bundle_sha256'],'data':audit},indent=2))


if __name__ == '__main__':
    main()
