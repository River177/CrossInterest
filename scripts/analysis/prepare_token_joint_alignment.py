"""Freeze one auxiliary-alignment arm from the accepted Token125 runtime."""
from pathlib import Path
import difflib
import json
import shutil
import yaml

from new_interest.data.build_cir_dataset import sha256,write_json

P=Path.cwd();A=P/'artifacts/mind-token-joint-alignment-20261001'
SOURCE=P/'artifacts/mind-history-token-20260930/bundles-v1/token'
DATA=P/'data/training/mind_token_joint_alignment_20261001_v1'
REL=Path('data/training/mind_cir_108_v1');CONTRACT=Path('configs/experiments/mind_cir_108_v1.json')


def replace(path,before,after,count=1):
    s=path.read_text()
    if s.count(before)!=count:raise ValueError(f'Anchor count differs: {path} {before}')
    path.write_text(s.replace(before,after))


def patch_cursor(path):
    replace(path,'import pyarrow.parquet as pq','import pyarrow.parquet as pq\nfrom new_interest.training.alignment_token_cursor import expected_supervised_tokens')
    replace(path,'    rows = pq.read_table(train_file, columns=columns).to_pylist()',
        '    columns.append("expected_assistant_tokens")\n'
        '    rows = pq.read_table(train_file, columns=columns).to_pylist()\n'
        '    supervised_lengths = [r["expected_assistant_tokens"] for r in rows]\n'
        '    expected_supervised_tokens([1]*len(rows), supervised_lengths)')
    replace(path,'exposure["assistant_tokens"] != [3 * count for count in counts]:',
        'exposure["assistant_tokens"] != expected_supervised_tokens(counts, supervised_lengths):',count=2)
    # Only the standalone source-check uses the local lengths; checkpoint reads
    # use lengths bound into the immutable cursor contract.
    s=path.read_text();i=s.index('def audit_checkpoint_cursor(')
    s=s[:i]+s[i:].replace('expected_supervised_tokens(counts, supervised_lengths)','expected_supervised_tokens(counts, contract["expected_assistant_tokens"])');path.write_text(s)
    replace(path,'exposure.assistant_tokens != [3 * count for count in counts]:',
        'exposure.assistant_tokens != expected_supervised_tokens(counts, contract["expected_assistant_tokens"]):')
    replace(path,'"row_identity_sha256": row_identity, "samples": len(rows),',
        '"row_identity_sha256": row_identity, "expected_assistant_tokens": supervised_lengths, "samples": len(rows),')
    replace(path,'if output_count != 3 or input_count < output_count:',
        'if not 0 <= index < samples or output_count != contract["expected_assistant_tokens"][index] or input_count < output_count:')
    replace(path,'    if any(value != 3 * len(expected) for value in denominators):',
        '    supervised_total = sum(contract["expected_assistant_tokens"][i] for i in expected)\n'
        '    if any(value != supervised_total for value in denominators):')
    replace(path,'"supervised_tokens": 3 * len(indices)', '"supervised_tokens": supervised_total')
    replace(path,'"global_supervised_token_denominator": 3 * len(indices)', '"global_supervised_token_denominator": supervised_total')


def prepare():
    parent=json.loads((SOURCE/'manifests/cir_bundle.json').read_text())
    accepted=json.loads((P/'artifacts/mind-history-token-20260930/token/final_acceptance.json').read_text())
    assert accepted['final_accepted'] and accepted['cold_state_sha256']=='28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565'
    for rel,h in parent['files'].items():assert sha256(SOURCE/rel)==h,rel
    metadata=json.loads((DATA/'metadata.json').read_text())
    for rel,h in metadata['files'].items():assert sha256(DATA/rel)==h,rel
    bundle=A/'bundle-v1'
    if bundle.exists():raise FileExistsError(bundle)
    shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['ni.parquet','cir.parquet']:shutil.copyfile(DATA/'joint.parquet',bundle/REL/name)
    for name in ['dev_full_grouped.parquet','model_token_mapping.parquet','interest_token_mapping.parquet','statistics.json','metadata.json','alignment_pairs.parquet','ni_reference.parquet']:
        shutil.copyfile(DATA/name,bundle/REL/name)
    for rel in ['src/new_interest/experiments/token_joint_alignment.py','src/new_interest/data/build_token_joint_alignment.py','src/new_interest/training/alignment_token_cursor.py']:
        shutil.copyfile(P/rel,bundle/rel)
    patch_cursor(bundle/'src/new_interest/training/old_new_execution.py')
    replace(bundle/'src/new_interest/training/sft_exposure.py',
        '"old_interest", "new_interest", "cross_interest_reconstruction"',
        '"old_interest", "new_interest", "cross_interest_reconstruction", "interest_name_to_id", "interest_id_to_name"')
    audit=bundle/'src/new_interest/experiments/old_new_interest.py'
    replace(audit,'columns=["sample_id", "target_subcategory"] +','columns=["sample_id", "target_subcategory", "expected_assistant_tokens"] +')
    replace(audit,'row["assistant_tokens"] == 3 * expected_presentations[row["sample_id"]]',
        'row["assistant_tokens"] == source["expected_assistant_tokens"] * expected_presentations[row["sample_id"]]')
    core=bundle/'src/new_interest/experiments/cir_warmup.py'
    replace(core,'audits[task]["min_assistant_tokens"] == audits[task]["max_assistant_tokens"] == 3',
        'audits[task]["min_assistant_tokens"] >= 3')
    replace(core,'    gradient_file = root / "gradient_audit.json"',
        '    from new_interest.experiments.token_joint_alignment import validate_completion_lengths\n'
        '    write(root / "joint_length_audit.json", validate_completion_lengths(project, tokenizer))\n'
        '    gradient_file = root / "gradient_audit.json"')
    replace(core,'"new_interest.experiments.history_token","gradient"','"new_interest.experiments.token_joint_alignment","gradient"')
    cfg=json.loads((bundle/CONTRACT).read_text());cfg.pop('history_token',None)
    cfg.update(version='mind-token125-joint-alignment-20261001-v1',method='NI plus bidirectional category name/ID auxiliary joint SFT',
        direct_training={'samples':24100,'epochs':3,'optimizer_steps':2262,'presentations':72300,'parent':None},
        joint_alignment={'NI_per_epoch':21600,'name_to_id_per_epoch':1250,'id_to_name_per_epoch':1250,'pair_repeats_per_epoch':10,'directions':2,'classes':125},
        data=json.loads((DATA/'statistics.json').read_text()),data_metadata_sha256=sha256(DATA/'metadata.json'),
        comparison_limit='Single new cold joint arm versus accepted Token125 NI-only. NI exposures unchanged; auxiliary supervision, update count and LR horizon increase. No isolated alignment-mechanism claim.')
    write_json(bundle/CONTRACT,cfg)
    base=json.loads((bundle/'configs/sft/mind_cir_base_v1.json').read_text())
    base['expected_initial_adapter_state_sha256']=accepted['cold_state_sha256']
    write_json(bundle/'configs/sft/mind_cir_base_v1.json',base)
    baseline=json.loads((SOURCE/'manifests/h50_baseline.json').read_text())
    tracking=json.loads((P/'manifests/mind_history_token_token_20260930_v1.json').read_text())
    baseline={'baseline_id':'Token125-NI-only-seed42','status':'accepted','active_run':tracking['active_run'],
        'acceptance_sha256':sha256(P/'artifacts/mind-history-token-20260930/token/final_acceptance.json'),
        'cold_state_sha256':accepted['cold_state_sha256'],'evaluation':accepted['evaluation']}
    write_json(bundle/'manifests/h50_baseline.json',baseline)
    wrapper_rel='scripts/amlt/sft/run_token_joint_alignment.sh'
    wrapper=(SOURCE/'scripts/amlt/sft/run_history_token.sh').read_text().replace('new_interest.experiments.history_token worker','new_interest.experiments.token_joint_alignment worker')
    (P/wrapper_rel).write_text(wrapper);(bundle/wrapper_rel).write_text(wrapper)
    changed=[rel for rel,h in parent['files'].items() if sha256(bundle/rel)!=h]
    immutable=['src/new_interest/training/train_sft.py','src/new_interest/training/sft_data.py','src/new_interest/training/checkpoint_persistence.py',
        'src/new_interest/training/semantic_id_tokens.py','src/new_interest/training/verify_semantic_id_token_adapter.py',
        'src/new_interest/evaluation/evaluate_id_token_ranking.py','src/new_interest/evaluation/merge_id_token_shards.py']
    assert not set(changed)&set(immutable)
    diff=[]
    for rel in changed:
        if rel.endswith('.py'):diff.extend(difflib.unified_diff((SOURCE/rel).read_text().splitlines(True),(bundle/rel).read_text().splitlines(True),fromfile='Token125/'+rel,tofile='joint/'+rel))
    (A/'runtime.patch').write_text(''.join(diff))
    write_json(bundle/'manifests/joint_preflight.json',{'parent_bundle_sha256':sha256(SOURCE/'manifests/cir_bundle.json'),'changed_files':changed,
        'unchanged_numerical_files':{rel:sha256(bundle/rel) for rel in immutable},'baseline':baseline})
    payload={str(f.relative_to(bundle)):sha256(f) for f in sorted(bundle.rglob('*')) if f.is_file() and f.name not in ['.amltignore','cir_bundle.json'] and '__pycache__' not in f.parts and f.suffix!='.pyc'}
    write_json(bundle/'manifests/cir_bundle.json',{'version':cfg['version'],'files':payload})
    y=yaml.safe_load((P/'configs/amlt/mind_history_token_token_0930_v1.yaml').read_text())
    y['description']='One new cold Token125 NI plus bidirectional name/ID auxiliary joint SFT; NI exposures unchanged, 2262 updates, real mixed gates and Full Dev.'
    y['code']['local_dir']='$CONFIG_DIR/../../'+str(bundle.relative_to(P));job=y['jobs'][0];job['name']='qwen35-2b-token125-joint';job['command'][-1]='bash '+wrapper_rel
    yp=P/'configs/amlt/mind_token_joint_alignment_1001_v1.yaml';assert not yp.exists();yp.write_text(yaml.safe_dump(y,sort_keys=False))
    tracking={'status':'prepared','artifact_root':str(A.relative_to(P)),'report':'docs/experiments/2026-10-01/mind_token_joint_alignment.md',
        'active_run':{'experiment':'mind-token125-joint-alignment-1001-v1','job':job['name'],'yaml':str(yp.relative_to(P)),
            'bundle':str(bundle.relative_to(P)),'bundle_sha256':sha256(bundle/'manifests/cir_bundle.json')},
        'budget':cfg['direct_training'],'tasks':cfg['joint_alignment'],'baseline':baseline,'gpu_count':4,'seed':42}
    write_json(P/'manifests/mind_token_joint_alignment_20261001_v1.json',tracking)
    print(json.dumps(tracking['active_run'],indent=2))


if __name__=='__main__':prepare()
