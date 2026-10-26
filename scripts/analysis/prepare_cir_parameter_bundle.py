"""Freeze the two approved CIR parameter-transfer arms and evaluation-only probes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from new_interest.data.build_cir_transfer_probes import build
from new_interest.experiments import cir_warmup as core


def main():
    root=Path.cwd();base=root/'artifacts/mind-cir-parameter-20260923';bundle=base/'bundle-v1'
    if bundle.exists():raise FileExistsError(bundle)
    tracking=core.read(root/'manifests/mind_cir_four_108_20260918_v1.json');job=tracking['jobs']['staged']
    core.require(tracking['status']=='all_four_accepted','Source experiment is not accepted')
    source=root/tracking['bundle'];sha=core.sha256
    core.require(sha(source/'manifests/cir_bundle.json')==tracking['bundle_sha256'],'Source bundle hash differs')
    old=core.read(source/'manifests/cir_bundle.json')
    for p,h in old['files'].items():core.require(sha(source/p)==h,'Source payload changed: '+p)
    final=root/tracking['artifact_root']/'final-readback'/job['experiment']/job['job']
    acceptance=core.read(root/tracking['conditions']['cir5_ni3']['acceptance'])
    core.require(acceptance['status']=='accepted','Full inheritance reference is not accepted')
    initial=core.read(final/'formal/cir5/initial_adapter_state.json');cir=core.read(final/'formal/ni3/initial_adapter_state.json')
    seed=core.read(final/'recovery/seed_complete.json')
    prefix='formal/cir5/final_adapter/'
    cirfiles={k[len(prefix):]:v for k,v in seed['files'].items() if k.startswith(prefix)}
    core.require(cirfiles['adapter_model.safetensors']['sha256']==job['stage_acceptance']['cir5']['adapter_sha256'],'Parent adapter differs')
    shutil.copytree(source,bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in ['src/new_interest/data/build_cir_transfer_probes.py','src/new_interest/experiments/cir_parameter_transfer.py','scripts/amlt/sft/run_cir_parameter_transfer.sh']:
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dest)
    refs=bundle/'manifests/transfer_sources';refs.mkdir()
    for src,name in [(final/'formal/cir5/initial_adapter_state.json','cold_state.json'),(final/'formal/ni3/initial_adapter_state.json','cir5_state.json'),(root/tracking['conditions']['cir5_ni3']['acceptance'],'full_inheritance_acceptance.json')]:shutil.copyfile(src,refs/name)
    rt=bundle/'transfer_runtime';rt.mkdir()
    shutil.copyfile(root/'scripts/analysis/cir_storage_retry.py',rt/'cir_storage_retry.py')
    shutil.copyfile(root/'artifacts/mind-cir-four-20260918/bundle-staged-recovery-0919-v1/recovery/sitecustomize.py',rt/'sitecustomize.py')
    probe=build(bundle/core.DATA/'ni.parquet',bundle/core.DATA/'dev_full_grouped.parquet',bundle/core.DATA/'interest_token_mapping.parquet',bundle/'data/diagnostics/cir_parameter_transfer_v1')
    probe['source_sha256']={str((root/Path(p)).relative_to(bundle)):h for p,h in probe['source_sha256'].items()}
    core.write(bundle/'data/diagnostics/cir_parameter_transfer_v1/metadata.json',probe)
    contract={'version':'cir-parameter-transfer-108-0923-v1','arms':['keep_lora','keep_sid'],'source_results_prefix':job['results_prefix'],
      'source_backend':job['backend_id'],'source_bundle_sha256':tracking['bundle_sha256'],
      'cir5_adapter_sha256':job['stage_acceptance']['cir5']['adapter_sha256'],'cir5_state_sha256':cir['state_sha256'],
      'cold_state_sha256':initial['state_sha256'],'cir5_files':cirfiles,
      'reference_ni3_adapter_sha256':job['stage_acceptance']['ni3']['adapter_sha256'],
      'reference_ni3_adapter_directory_sha256':core.read(final/'conditions/cir5_ni3/real_eval/merged/metrics.json')['adapter_sha256'],
      'reference_prediction_relative_path':'conditions/cir5_ni3/real_eval/merged/id_token_predictions.parquet',
      'reference_prediction_sha256':acceptance['evaluation']['prediction_sha256'],
      'ni_epochs':3,'ni_steps':2025,'ni_samples':21600,'ni_presentations':64800,'both_groups_trainable_during_NI':True,
      'automatic_monitoring':False,'probes':probe,'no_extra_training_conditions':True}
    core.write(bundle/'manifests/cir_parameter_transfer.json',contract)
    for p,h in old['files'].items():core.require(sha(bundle/p)==h,'Original numerical runtime changed')
    files={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.amltignore'}
    core.write(bundle/'manifests/transfer_bundle.json',{'version':contract['version'],'source_bundle_sha256':tracking['bundle_sha256'],'files':files})
    config=yaml.safe_load((root/'configs/amlt/mind_cir_four_conditions_full_0918_v1.yaml').read_text());config['description']='Two approved parameter-transfer experiments: CIR5 LoRA with original SID, or original LoRA with CIR5 SID; both train all parameters on unchanged NI3. Pre/post functional probes are evaluation only.'
    config['code']['local_dir']='$CONFIG_DIR/../../artifacts/mind-cir-parameter-20260923/bundle-v1'
    template=config['jobs'][0];config['jobs']=[]
    import copy
    for arm in contract['arms']:
        j=copy.deepcopy(template);j['name']='qwen35-2b-'+arm.replace('_','-')+'-ni3';j['submit_args']['env'].pop('CIR_MODE',None);j['submit_args']['env']['CIR_TRANSFER_ARM']=arm;j['command'][-1]='bash scripts/amlt/sft/run_cir_parameter_transfer.sh';config['jobs'].append(j)
    (root/'configs/amlt/mind_cir_parameter_transfer_0923_v1.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    core.write(base/'local_preflight.json',{'status':'passed','unchanged_original_payload_files':len(old['files']),'source_bundle_sha256':tracking['bundle_sha256'],'transfer_bundle_sha256':sha(bundle/'manifests/transfer_bundle.json'),'probe_rows':{'sid_name':108,'heldout_cir':512},'probe_user_overlap':0,'formal_updates_per_arm':2025,'no_CIR_training':True})
    print(json.dumps(core.read(base/'local_preflight.json'),indent=2))


if __name__=='__main__':main()
