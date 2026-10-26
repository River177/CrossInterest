"""Bind a CPU-only paired analysis to both independently accepted SFT runs."""
from pathlib import Path
import hashlib
import json
import shutil
import yaml

P=Path.cwd();A=P/'artifacts/mind-history-token-20260930'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')


def prepare():
    bundle=A/'analysis-bundle-v1'
    if bundle.exists():raise FileExistsError(bundle)
    accepted={arm:read(A/arm/'final_acceptance.json') for arm in ['text','token']}
    assert all(x['final_accepted'] and x['cross_arm_cold_state_equal'] for x in accepted.values())
    assert accepted['text']['cold_state_sha256']==accepted['token']['cold_state_sha256']
    shutil.copytree(P/'artifacts/mind-cir-ni-direct-training-20260929/analysis-bundle-v1',bundle)
    for rel in ['src/new_interest/experiments/cir_ni_direct_analysis.py','src/new_interest/experiments/history_token_analysis.py']:
        shutil.copyfile(P/rel,bundle/rel)
    arms={}
    for arm in ['text','token']:
        track=read(P/f'manifests/mind_history_token_{arm}_20260930_v1.json')
        source=Path(accepted[arm]['receipt']).parent/'compact'
        metrics=read(source/'formal/real_eval/merged/metrics.json');merge=read(source/'formal/real_eval/merged/merge_manifest.json')
        assert sha(source/'formal/real_eval/merged/metrics.json')==merge['metrics_sha256']
        arms[arm]={key:track['active_run'][key] for key in ['results_prefix','bundle_sha256']}
        arms[arm].update(prediction_sha256=merge['merged_prediction_sha256'],metrics_sha256=merge['metrics_sha256'],
            evaluation_identity=metrics['evaluation_identity'],complete_sha256=sha(source/'complete.json'),
            acceptance_sha256=sha(A/arm/'final_acceptance.json'),metrics=accepted[arm]['evaluation']['metrics'])
        (bundle/'inputs'/arm).mkdir()
        shutil.copyfile(P/f'data/training/mind_history_token_20260930_v1/{arm}/dev_full_grouped.parquet',bundle/'inputs'/arm/'dev_full_grouped.parquet')
    shutil.copyfile(P/'data/training/mind_history_token_20260930_v1/interest_token_mapping.parquet',bundle/'inputs/interest_token_mapping.parquet')
    # Remove the inherited common-Dev input: the two views have different prompts.
    (bundle/'inputs/dev_full_grouped.parquet').unlink()
    baseline=read(A/'bundles-v1/text/manifests/h50_baseline.json')
    contract={'arms':arms,'draws':2000,'seed':42,'cold_state_sha256':accepted['text']['cold_state_sha256'],
        'historical_baseline':{'baseline_id':baseline['baseline_id'],'metrics':baseline['evaluation']['metrics'],
            'metrics_sha256':baseline['metrics_sha256'],'prediction_sha256':baseline['predictions_sha256'],
            'boundary':'Previously accepted Text108 NI-only; vocabulary/runtime differ; descriptive comparison only.'}}
    write(bundle/'analysis_contract.json',contract)
    (bundle/'run_analysis.py').write_text('''from pathlib import Path
import os,json,hashlib,shutil,tempfile,platform
import numpy,pyarrow
from new_interest.experiments.history_token_analysis import analyze
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
code=Path(os.environ['AMLT_CODE_DIR']);out=Path(os.environ['AMLT_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
payload=json.loads((code/'payload.json').read_text())
for rel,digest in payload.items():assert sha(code/rel)==digest,rel
assert numpy.__version__=='2.2.6' and pyarrow.__version__=='20.0.0'
contract=json.loads((code/'analysis_contract.json').read_text());assert contract['draws']==2000 and contract['seed']==42
local=Path(tempfile.mkdtemp(prefix='history-token-analysis-',dir='/scratch'))
result=analyze(code,local/'comparison.json',contract)
receipt={'status':'accepted','python':platform.python_version(),'numpy':numpy.__version__,'pyarrow':pyarrow.__version__,
 'device':'CPU','comparison_sha256':sha(local/'comparison.json'),'payload_sha256':sha(code/'payload.json'),
 'analysis_source_sha256':sha(code/'src/new_interest/experiments/history_token_analysis.py'),
 'full_checkpoint_and_prediction_hashes_verified_remotely':True}
(local/'analysis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
for name in ['comparison.json','analysis_receipt.json']:
 shutil.copyfile(local/name,out/name);assert sha(local/name)==sha(out/name)
(out/'complete.json').write_text(json.dumps({'status':'accepted','comparison_sha256':sha(out/'comparison.json'),
 'analysis_receipt_sha256':sha(out/'analysis_receipt.json'),'payload_sha256':sha(code/'payload.json')},indent=2)+'\\n')
print('HISTORY_TOKEN_PAIRED_ANALYSIS_ACCEPTED',json.dumps(result['differences']),flush=True)
''')
    payload={str(f.relative_to(bundle)):sha(f) for f in sorted(bundle.rglob('*')) if f.is_file() and f.name not in {'payload.json','.amltignore'} and '__pycache__' not in f.parts}
    write(bundle/'payload.json',payload)
    cfg=yaml.safe_load((P/'configs/amlt/mind_cir_ni_direct_analysis_0929_v2.yaml').read_text())
    cfg['description']='CPU-only Text125/Token125 full checkpoint hashes, independently recomputed Full Dev metrics and paired-user comparison; no model training.'
    cfg['code']['local_dir']='$CONFIG_DIR/../../'+str(bundle.relative_to(P));cfg['jobs'][0]['name']='history-token-paired-analysis'
    yp=P/'configs/amlt/mind_history_token_analysis_1001_v1.yaml';assert not yp.exists();yp.write_text(yaml.safe_dump(cfg,sort_keys=False))
    out=A/'analysis';out.mkdir()
    tracking={'status':'prepared','artifact_root':str(out.relative_to(P)),'active_run':{'experiment':'mind-history-token-analysis-1001-v1',
        'job':cfg['jobs'][0]['name'],'yaml':str(yp.relative_to(P)),'bundle':str(bundle.relative_to(P)),'payload_sha256':sha(bundle/'payload.json')}}
    write(P/'manifests/mind_history_token_analysis_20261001_v1.json',tracking)
    print(json.dumps(tracking,indent=2))


if __name__=='__main__':prepare()
