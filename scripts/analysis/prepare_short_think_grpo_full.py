"""Freeze formal E/H/HL bundles only after the resumed engine is accepted."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--arm',choices=['E','H','HL'],required=True)
    parser.add_argument('--engine-acceptance',type=Path,required=True)
    parser.add_argument('--bundle-name',required=True)
    parser.add_argument('--parent-info',type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];root=repo/'artifacts/mind-short-think-grpo-20260914'
    source=root/'bundle-engine-recovery-v1';bundle=root/args.bundle_name
    if bundle.exists() or Path(args.bundle_name).name!=args.bundle_name:raise ValueError('New plain bundle name required')
    acceptance=json.loads(args.engine_acceptance.read_text())
    if not acceptance['passed'] or not acceptance.get('execution_plan_recovery_verified'):
        raise ValueError('Full recovered-engine independent acceptance required')
    parent=None
    if args.arm=='HL':
        if args.parent_info is None:raise ValueError('HL requires the independently accepted H100 parent')
        parent=json.loads(args.parent_info.read_text())
        if not parent['accepted'] or parent['arm']!='H' or parent['step']!=100:raise ValueError('Wrong HL parent')
    shutil.copytree(source,bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Recovery-only metadata is not a fallback for any formal checkpoint.
    for name in ['recovered_probe_plan.json','recovery_contract.json','reload_diagnostic_acceptance.json']:
        (bundle/name).unlink(missing_ok=True)
    for name in ['src/new_interest/training/short_think_grpo_engine.py','src/new_interest/training/probe_execution_plan.py',
                 'src/new_interest/experiments/short_think_grpo_engine.py','src/new_interest/experiments/short_think_grpo_full.py']:
        shutil.copyfile(repo/name,bundle/name)
    shutil.copyfile(args.engine_acceptance,bundle/'throughput_engine_acceptance.json')
    diagnostic=root/'production-preparation/diagnostic2048'
    shutil.copyfile(diagnostic/'diagnostic_dev.parquet',bundle/'inputs/diagnostic_dev.parquet')
    shutil.copyfile(diagnostic/'manifest.json',bundle/'diagnostic_manifest.json')
    shutil.copyfile(diagnostic/'short_sft_baseline_metrics.json',bundle/'diagnostic_sft_baseline_metrics.json')
    contract=json.loads((bundle/'engine_contract.json').read_text())
    for name in ['actual_oracle_updates','sequence','no_production_training_authorized_by_oracle_alone','reference_engine_acceptance_sha256']:
        contract.pop(name,None)
    contract.update(version='mind-thinking-grpo-formal-v1',role='production',arm=args.arm,
        full_training_planned_updates=675,checkpoint_interval=25,
        throughput_acceptance_sha256=sha(bundle/'throughput_engine_acceptance.json'),
        probe_protocol='Capture and replay actual FLA execution plans; unchanged exact reference and zero-tolerance comparison',
        diagnostic_dev_sha256=sha(bundle/'inputs/diagnostic_dev.parquet'),
        final_evaluation='unchanged unconstrained greedy512, original4 shards and batch8,31392 Dev,108 SIDs; invalid retained')
    if parent:
        contract['allowed_parent']={'arm':'H','identity':parent['identity'],'state_file_sha256':parent['state_file_sha256']}
        shutil.copyfile(args.parent_info,bundle/'parent_info.json')
    (bundle/'engine_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    shell=(source/'scripts/amlt/grpo/run_short_think_grpo_engine.sh').read_text()
    shell=shell[:shell.rfind('python -m new_interest.experiments.short_think_grpo_engine')]
    shell+='args=(--project "$AMLT_CODE_DIR" --root "$AMLT_OUTPUT_DIR" --stop-at "${1:?}")\n'
    shell+='if [[ -n "${2:-}" ]]; then args+=(--parent-blob "$2"); fi\n'
    shell+='python -m new_interest.experiments.short_think_grpo_full "${args[@]}"\n'
    (bundle/'scripts/amlt/grpo/run_short_think_grpo_full.sh').write_text(shell)
    files={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file() and p.name!='source_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    runtime={k:v for k,v in files.items() if not k.startswith('inputs/')}
    manifest={'files':files,'runtime_sha256':hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest(),
              'parent_verified_engine_source_sha256':sha(source/'source_manifest.json')}
    (bundle/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'arm':args.arm,'bundle':str(bundle.relative_to(repo)),
        'engine_identity':sha(bundle/'engine_contract.json'),'runtime_sha256':manifest['runtime_sha256'],'files':len(files),
        'initial_adapter_file_sha256':contract['initial_adapter_file_sha256'],'planned_steps':675,'submitted':False}
    (root/f'{args.bundle_name}-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
