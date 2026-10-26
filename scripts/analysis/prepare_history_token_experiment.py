"""Freeze matched 125-token NI jobs without modifying earlier experiment bundles."""
from pathlib import Path
import difflib
import json
import shutil

import yaml

from new_interest.data.build_cir_dataset import sha256, write_json

P=Path.cwd()
SOURCE=P/'artifacts/mind-cir-ni-direct-training-20260929/bundles-v1/original_ni_paired'
DATA=P/'data/training/mind_history_token_20260930_v1'
ART=P/'artifacts/mind-history-token-20260930'
REL_DATA=Path('data/training/mind_cir_108_v1')
CONTRACT=Path('configs/experiments/mind_cir_108_v1.json')


def replace_once(path, before, after):
    text=path.read_text()
    if text.count(before)!=1:
        raise ValueError(f'Patch anchor not unique: {path}: {before[:80]}')
    path.write_text(text.replace(before,after))


def patch_runtime(bundle):
    # Candidate IDs keep their original order; a separate full mapping rebuilds
    # all model rows before adapter loading. Scores remain 108-way at evaluation.
    evaluator=bundle/'src/new_interest/evaluation/evaluate_id_token_ranking.py'
    replace_once(evaluator, '    spec = validate_saved_interest_tokens(\n        tokenizer,\n        mapping_file,',
        '    model_mapping_file = Path(config.get("model_interest_token_mapping_file", mapping_file))\n'
        '    spec = validate_saved_interest_tokens(\n        tokenizer,\n        model_mapping_file,')
    replace_once(evaluator, '    token_ids = spec.token_ids\n',
        '    entries = {entry.interest_token: entry for entry in spec.entries}\n'
        '    if any(token not in entries or entries[token].target_subcategory != label for token, label in zip(tokens, labels)):\n'
        '        raise ValueError("Candidate mapping does not match model vocabulary")\n'
        '    token_ids = [entries[token].token_id for token in tokens]\n')
    replace_once(evaluator, '    evaluation_identity = canonical_sha256(identity_payload)',
        '    identity_payload["model_mapping_sha256"] = sha256_file(model_mapping_file)\n'
        '    identity_payload["model_interest_token_count"] = len(spec.entries)\n'
        '    evaluation_identity = canonical_sha256(identity_payload)')
    shared=bundle/'src/new_interest/experiments/backbone_scaling.py'
    replace_once(shared, 'all(s == [108, model["hidden_size"]] for s in token_shapes.values())',
        'all(s == [125, model["hidden_size"]] for s in token_shapes.values())')
    text=shared.read_text();begin=text.index('def audit_evaluation(');end=text.index('\ndef verify_manifest(',begin)
    text=text[:begin]+text[begin:end].replace('project / MAPPING','project / CANDIDATE_MAPPING')+text[end:]
    shared.write_text(text)
    original=bundle/'src/new_interest/experiments/old_new_interest.py'
    replace_once(original, 'semantic["count"] == 108 and semantic["save_embedding_layers"]',
        'semantic["count"] == 125 and semantic["save_embedding_layers"]')
    core=bundle/'src/new_interest/experiments/cir_warmup.py'
    replace_once(core, 'shared.TRAIN, shared.DEV, shared.MAPPING = DATA / "ni.parquet", DATA / "dev_full_grouped.parquet", DATA / "interest_token_mapping.parquet"',
        'shared.TRAIN, shared.DEV, shared.MAPPING = DATA / "ni.parquet", DATA / "dev_full_grouped.parquet", DATA / "model_token_mapping.parquet"\n'
        '    shared.CANDIDATE_MAPPING = DATA / "interest_token_mapping.parquet"')
    replace_once(core, 'project / shared.DEV, project / shared.MAPPING, root / "real_eval/merged"',
        'project / shared.DEV, project / shared.CANDIDATE_MAPPING, root / "real_eval/merged"')
    replace_once(core, 'register_semantic_interest_tokens(tokenizer, project / shared.MAPPING)',
        'register_semantic_interest_tokens(tokenizer, project / shared.MAPPING, expected_count=125)')
    replace_once(core, '    gate = root / "gates"\n',
        '    from new_interest.training.sft_data import apply_chat_template\n'
        '    dev_rows = pq.read_table(project / shared.DEV, columns=["prompt"]).to_pylist()\n'
        '    lengths = [len(tokenizer(apply_chat_template(tokenizer, [{"role":"user","content":r["prompt"]}], add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]) for r in dev_rows]\n'
        '    require(max(lengths) <= 2048, "Dev prompt exceeds context boundary")\n'
        '    write(root / "dev_token_audit.json", {"records":len(lengths), "max_tokens":max(lengths), "over_max_seq_length":sum(n>2048 for n in lengths)})\n'
        '    gradient_file = root / "gradient_audit.json"\n'
        '    if not gradient_file.exists():\n'
        '        run([sys.executable,"-m","new_interest.experiments.history_token","gradient","--project",str(project),"--output",str(gradient_file)], project, root/"logs/gradient.log", "0")\n'
        '    grad = read(gradient_file)\n'
        '    require(grad["interest_token_count"] == 125 and grad["all_token_rows_finite_and_nonzero"], "125-row gradient gate failed")\n'
        '    gate = root / "gates"\n')


def prepare():
    acceptance=json.loads((P/'artifacts/mind-cir-ni-direct-training-20260929/original_ni_paired/final_acceptance.json').read_text())
    assert acceptance['status']=='accepted' and acceptance['bundle_sha256']==sha256(SOURCE/'manifests/cir_bundle.json')
    old=json.loads((SOURCE/'manifests/cir_bundle.json').read_text())
    for rel,digest in old['files'].items():
        assert sha256(SOURCE/rel)==digest, rel
    meta=json.loads((DATA/'metadata.json').read_text())
    for rel,digest in meta['files'].items():
        assert sha256(DATA/rel)==digest,rel
    ART.mkdir(exist_ok=True)
    reports={}
    for arm in ('text','token'):
        bundle=ART/'bundles-v1'/arm
        if bundle.exists():raise FileExistsError(bundle)
        shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('ni.parquet','cir.parquet'):
            shutil.copyfile(DATA/arm/'ni.parquet',bundle/REL_DATA/name)
        shutil.copyfile(DATA/arm/'dev_full_grouped.parquet',bundle/REL_DATA/'dev_full_grouped.parquet')
        for name in ('interest_token_mapping.parquet','model_token_mapping.parquet','statistics.json','metadata.json'):
            shutil.copyfile(DATA/name,bundle/REL_DATA/name)
        for rel in ('src/new_interest/training/semantic_id_tokens.py','src/new_interest/experiments/history_token.py',
                    'src/new_interest/data/build_history_token_dataset.py'):
            shutil.copyfile(P/rel,bundle/rel)
        patch_runtime(bundle)
        cfg=json.loads((bundle/CONTRACT).read_text())
        for key in ('paired_control','reuse_verified_gates','training_schedule'):
            cfg.pop(key,None)
        cfg.update(version='mind-ni-history-token-125-'+arm+'-20260930-v1',
            method='NI-only; shared expanded vocabulary; '+arm+' category input',
            direct_training={'samples':21600,'epochs':3,'optimizer_steps':2025,'presentations':64800,'parent':None},
            data=json.loads((DATA/'statistics.json').read_text()),data_metadata_sha256=sha256(DATA/'metadata.json'),
            history_token={'arm':arm,'model_tokens':125,'output_candidates':108,'append_only':True,'seed':42},
            comparison_limit='Text125 versus Token125 is matched representation comparison. Original Text108 NI-only is historical context with vocabulary/runtime differences.')
        cfg['recipe']['save_steps']=100
        write_json(bundle/CONTRACT,cfg)
        base=json.loads((bundle/'configs/sft/mind_cir_base_v1.json').read_text())
        base.update(semantic_id_token_mapping_file=str(REL_DATA/'model_token_mapping.parquet'),semantic_id_token_count=125,
                    train_file=str(REL_DATA/'ni.parquet'),save_steps=100)
        base.pop('expected_initial_adapter_state_sha256',None)
        write_json(bundle/'configs/sft/mind_cir_base_v1.json',base)
        evaluation=json.loads((bundle/'configs/id_token/mind_cir_v1.json').read_text())
        evaluation.update(model_interest_token_mapping_file=str(REL_DATA/'model_token_mapping.parquet'),semantic_id_token_count=125)
        write_json(bundle/'configs/id_token/mind_cir_v1.json',evaluation)
        wrapper_rel='scripts/amlt/sft/run_history_token.sh'
        wrapper=(SOURCE/'scripts/amlt/sft/run_cir_ni_direct.sh').read_text().replace(
            'python -m new_interest.experiments.cir_ni_direct', 'python -m new_interest.experiments.history_token worker')
        (bundle/wrapper_rel).write_text(wrapper)
        (P/wrapper_rel).write_text(wrapper)
        changed={rel for rel,digest in old['files'].items() if sha256(bundle/rel)!=digest}
        diff=[]
        for rel in sorted(changed):
            if rel.endswith('.py'):
                diff.extend(difflib.unified_diff((SOURCE/rel).read_text().splitlines(True),(bundle/rel).read_text().splitlines(True),fromfile='accepted/'+rel,tofile=arm+'/'+rel))
        (ART/f'{arm}-runtime.patch').write_text(''.join(diff))
        immutable=('src/new_interest/training/train_sft.py','src/new_interest/training/old_new_execution.py',
                   'src/new_interest/training/sft_data.py','src/new_interest/training/sft_exposure.py',
                   'src/new_interest/training/checkpoint_persistence.py','src/new_interest/training/verify_semantic_id_token_adapter.py',
                   'src/new_interest/evaluation/merge_id_token_shards.py')
        assert not changed.intersection(immutable)
        write_json(bundle/'manifests/history_token_preflight.json',{'status':'passed','parent_bundle_sha256':acceptance['bundle_sha256'],
            'changed_parent_files':sorted(changed),'unchanged_numerical_files':{rel:sha256(bundle/rel) for rel in immutable},
            'data_metadata_sha256':sha256(DATA/'metadata.json'),'model_tokens':125,'candidate_tokens':108})
        payload={str(p.relative_to(bundle)):sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.amltignore' and p!=bundle/'manifests/cir_bundle.json'}
        write_json(bundle/'manifests/cir_bundle.json',{'version':cfg['version'],'files':payload})
        config=yaml.safe_load((P/'configs/amlt/mind_cir_ni_direct_ni_0929_v1.yaml').read_text())
        config['description']='NI-only '+arm+' history categories, shared 125-token vocabulary, original 108 candidates, cold SFT3, gated full Dev.'
        config['code']['local_dir']='$CONFIG_DIR/../../'+str(bundle.relative_to(P))
        job=config['jobs'][0];job['name']='qwen35-2b-ni-'+arm+'125';job['command'][-1]='bash '+wrapper_rel
        yaml_path=P/f'configs/amlt/mind_history_token_{arm}_0930_v1.yaml'
        assert not yaml_path.exists();yaml_path.write_text(yaml.safe_dump(config,sort_keys=False))
        (ART/arm).mkdir()
        report={'status':'prepared','arm':arm,'artifact_root':str((ART/arm).relative_to(P)),
            'report':'docs/experiments/2026-09-30/mind_history_token.md','monitor_step_totals':{'ni':2025},
            'active_run':{'experiment':f'mind-ni-history-{arm}125-0930-v1','job':job['name'],
                'yaml':str(yaml_path.relative_to(P)),'bundle':str(bundle.relative_to(P)),
                'bundle_sha256':sha256(bundle/'manifests/cir_bundle.json')}}
        manifest=P/f'manifests/mind_history_token_{arm}_20260930_v1.json'
        assert not manifest.exists();write_json(manifest,report)
        reports[arm]={'manifest':str(manifest.relative_to(P)),**report['active_run']}
    write_json(P/'manifests/mind_history_token_20260930_v1.json',{'status':'prepared','arms':reports,
        'model':'Qwen/Qwen3.5-2B','seed':42,'gpu_per_job':4,'maximum_concurrent_gpus':8,
        'train_rows':21600,'epochs':3,'updates_each':2025,'dev_rows':31392,
        'data':'data/training/mind_history_token_20260930_v1','historical_baseline':'Original NI-only Text108, reference only'})
    print(json.dumps(reports,indent=2))


if __name__=='__main__':prepare()
