"""Freeze two single-stage direct-supervision jobs from an accepted runtime."""
from pathlib import Path
import importlib.util,json,shutil
import yaml
from new_interest.data.build_cir_dataset import sha256,write_json
from new_interest.experiments import cir_warmup as core
from new_interest.experiments.cir_ni_direct import stage_specs

P=Path.cwd();source=P/'artifacts/mind-cir-ni-matched-20260928/bundles-v1/ni_prefix'
accepted=core.read(P/'artifacts/mind-cir-ni-matched-20260928/ni_prefix/final_acceptance.json')
old=core.read(source/'manifests/cir_bundle.json')
assert accepted['status']=='accepted' and accepted['bundle_sha256']==sha256(source/'manifests/cir_bundle.json')
for rel,digest in old['files'].items():assert sha256(source/rel)==digest,rel
D=P/'data/training/mind_cir_ni_direct_balanced_20260929_v1';dm=core.read(D/'metadata.json')
for rel,digest in dm['files'].items():assert sha256(D/rel)==digest
A=P/'artifacts/mind-cir-ni-direct-training-20260929';A.mkdir(exist_ok=True)
modules={}
for name in ['sft_exposure','old_new_execution']:
 spec=importlib.util.spec_from_file_location('direct_frozen_'+name,source/f'src/new_interest/training/{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
reports={}
for arm,short in [('cir_as_ni','cir'),('original_ni_paired','ni')]:
 bundle=A/'bundles-v1'/arm
 if bundle.exists():raise FileExistsError(bundle)
 shutil.copytree(source,bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 # cir.parquet is only a compatibility alias for common runtime token/gate audit;
 # the sole formal stage reads ni.parquet. Both aliases contain this same arm.
 for name in ['cir.parquet','ni.parquet']:shutil.copyfile(D/f'{arm}.parquet',bundle/core.DATA/name)
 for name in ['dev_full_grouped.parquet','interest_token_mapping.parquet']:shutil.copyfile(D/name,bundle/core.DATA/name)
 shutil.copyfile(D/'pairs.parquet',bundle/core.DATA/'selection_manifest.parquet')
 shutil.copyfile(D/'metadata.json',bundle/core.DATA/'metadata.json');shutil.copyfile(D/'statistics.json',bundle/core.DATA/'statistics.json')
 cfg=core.read(source/core.CONTRACT)
 for key in ['training_schedule','paired_control','reuse_verified_gates','existing_cir1_reference','existing_cir3_reference']:cfg.pop(key,None)
 cfg.update(version='mind-cir-ni-direct-balanced-20260929-v1-'+arm,method=arm+' standalone NI SFT',
   direct_training={'samples':1055,'epochs':3,'optimizer_steps':99,'presentations':3165,'parent':None},
   data=core.read(D/'statistics.json'),data_metadata_sha256=sha256(bundle/core.DATA/'metadata.json'),
   paired_control={'arm':arm,'rows':1055,'classes':98,'same_source_users_and_class_counts':True,'pair_manifest_sha256':sha256(D/'pairs.parquet')},
   comparison_limit='Single-seed standalone SFT: users/label histogram/count matched; history length/token compute differ; full 108-candidate Dev unchanged.')
 cfg['recipe']['save_steps']=25
 write_json(bundle/core.CONTRACT,cfg)
 basecfg=core.read(bundle/'configs/sft/mind_cir_base_v1.json');basecfg['save_steps']=25;write_json(bundle/'configs/sft/mind_cir_base_v1.json',basecfg)
 rel='src/new_interest/experiments/cir_ni_direct.py';shutil.copyfile(P/rel,bundle/rel)
 wrapper=(source/'scripts/amlt/sft/run_cir_ni_matched.sh').read_text().replace('python -m new_interest.experiments.cir_warmup worker','python -m new_interest.experiments.cir_ni_direct')
 wrapper_rel='scripts/amlt/sft/run_cir_ni_direct.sh';(bundle/wrapper_rel).write_text(wrapper)
 local=P/wrapper_rel
 if local.exists():assert local.read_text()==wrapper
 else:local.write_text(wrapper)
 stages={}
 for stage in stage_specs(bundle,bundle/'formal',gates=False):
  ledger=modules['sft_exposure'].TrainingExposure.from_parquet(stage['train_file'],train_sha256=sha256(stage['train_file']),expected_epochs=3,max_samples=None,default_task='new_interest')
  cursor=modules['old_new_execution'];cursor.STAGES.add('cir/'+stage['key'])
  contract=cursor.make_cursor_contract(stage['train_file'],stage='cir/'+stage['key'],world_size=4,per_device_batch_size=2,epochs=3,global_batch_size=32)
  assert contract['total_steps']==99
  stages[stage['key']]={'samples':1055,'epochs':3,'steps':99,'presentations':3165,'parent':None,'cursor':contract,'exposure_identity':ledger.identity}
 changed=[rel for rel,h in old['files'].items() if sha256(bundle/rel)!=h]
 expected={str(core.CONTRACT),'configs/sft/mind_cir_base_v1.json',*(str(core.DATA/n) for n in ['cir.parquet','ni.parquet','statistics.json','metadata.json','selection_manifest.parquet'])}
 assert set(changed)==expected,changed
 write_json(bundle/'manifests/direct_preflight.json',{'status':'passed','stages':stages,'changed_prior_files':changed,'source_bundle_sha256':accepted['bundle_sha256'],'source_training_runtime':'all numerical trainer/sampler/loss/checkpoint/evaluator files byte-identical'})
 payload={str(p.relative_to(bundle)):sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.amltignore' and p!=bundle/'manifests/cir_bundle.json'}
 write_json(bundle/'manifests/cir_bundle.json',{'version':cfg['version'],'files':payload})
 config=yaml.safe_load((P/'configs/amlt/mind_cir_ni_matched_ni_prefix_0928_v1.yaml').read_text())
 config['description']=f'Direct supervision {arm}: 1055 matched users, same 98-class histogram, cold SFT3 only.'
 config['code']['local_dir']='$CONFIG_DIR/../../'+str(bundle.relative_to(P));job=config['jobs'][0];job['name']='qwen35-2b-'+short+'-direct3';job['command'][-1]='bash '+wrapper_rel
 yaml_path=P/f'configs/amlt/mind_cir_ni_direct_{short}_0929_v1.yaml';assert not yaml_path.exists();yaml_path.write_text(yaml.safe_dump(config,sort_keys=False))
 experiment=f'mind-cir-ni-direct-{short}-0929-v1';armroot=A/arm;armroot.mkdir()
 record={'status':'prepared','artifact_root':str(armroot.relative_to(P)),'monitor_step_totals':{'standalone':99},'report':'docs/experiments/2026-09-29/mind_cir_ni_direct_training.md','active_run':{'experiment':experiment,'job':job['name'],'bundle':str(bundle.relative_to(P)),'bundle_sha256':sha256(bundle/'manifests/cir_bundle.json'),'yaml':str(yaml_path.relative_to(P))},'arm':arm,'training_schedule':cfg['direct_training']}
 mp=P/f'manifests/mind_cir_ni_direct_{short}_20260929_v1.json';assert not mp.exists();write_json(mp,record)
 reports[arm]={'manifest':str(mp.relative_to(P)),**record['active_run'],'stages':stages}
parent={'status':'prepared','authorized_scope':'two independent cold standalone SFT3 runs on accepted 1055-pair same-user/class-count data','arms':reports,'data_manifest':'manifests/mind_cir_ni_direct_balanced_20260929_v1.json','seed':42,'gpu_per_job':4,'maximum_concurrent_gpus':8,'target':'Recall','sla':'Standard','full_dev_records':31392,'training_steps_each':99,'presentations_each':3165,'checkpoint_every_steps':25,'no_second_NI_stage':True,'no_extra_seeds_or_sweeps':True}
write_json(P/'manifests/mind_cir_ni_direct_training_20260929_v1.json',parent);write_json(A/'local_preflight.json',{'status':'passed',**parent})
print(json.dumps({a:{k:r[k] for k in ['manifest','experiment','job','yaml','bundle_sha256']} for a,r in reports.items()},indent=2))
