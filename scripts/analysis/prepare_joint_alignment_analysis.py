"""Freeze remote paired analysis of accepted Token125 baseline and joint SFT."""
from pathlib import Path
import hashlib
import json
import shutil
import yaml

P=Path.cwd();A=P/'artifacts/mind-token-joint-alignment-20261001'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')


def prepare():
    b=A/'analysis-bundle-v1'
    if b.exists():raise FileExistsError(b)
    refs={'baseline':(P/'artifacts/mind-history-token-20260930/token/final_acceptance.json',P/'manifests/mind_history_token_token_20260930_v1.json'),
          'joint':(A/'final_acceptance.json',P/'manifests/mind_token_joint_alignment_20261001_v1.json')}
    arms={};acceptances={}
    for arm,(ap,mp) in refs.items():
        accepted=read(ap);track=read(mp);assert accepted['final_accepted']
        acceptances[arm]=accepted
        root=Path(accepted['receipt']).parent/'compact';merged=root/'formal/real_eval/merged'
        metric=read(merged/'metrics.json');merge=read(merged/'merge_manifest.json')
        assert sha(merged/'metrics.json')==merge['metrics_sha256']
        arms[arm]={key:track['active_run'][key] for key in ['results_prefix','bundle_sha256']}
        arms[arm].update(acceptance_path=str(ap.relative_to(P)),acceptance_sha256=sha(ap),
            metrics=accepted['evaluation']['metrics'],prediction_sha256=merge['merged_prediction_sha256'],
            metrics_sha256=merge['metrics_sha256'],evaluation_identity=metric['evaluation_identity'],complete_sha256=sha(root/'complete.json'),
            optimizer_steps=2025 if arm=='baseline' else 2262,
            presentations_by_task={'new_interest':64800} if arm=='baseline' else {'new_interest':64800,'interest_name_to_id':3750,'interest_id_to_name':3750})
    assert acceptances['joint']['cold_state_matches_baseline']
    shutil.copytree(P/'artifacts/mind-history-token-20260930/analysis-bundle-v2',b)
    # Reuse numerical paired-user bootstrap, specialize identities and budgets.
    for rel in ['src/new_interest/experiments/cir_ni_direct_analysis.py','src/new_interest/experiments/token_joint_alignment_analysis.py']:
        shutil.copyfile(P/rel,b/rel)
    for name in ['text','token']:
        shutil.rmtree(b/'inputs'/name)
    for name in ['baseline','joint']:
        (b/'inputs'/name).mkdir()
        shutil.copyfile(P/'data/training/mind_token_joint_alignment_20261001_v1/dev_full_grouped.parquet',b/'inputs'/name/'dev_full_grouped.parquet')
    shutil.copyfile(P/'data/training/mind_token_joint_alignment_20261001_v1/interest_token_mapping.parquet',b/'inputs/interest_token_mapping.parquet')
    contract={'arms':arms,'draws':2000,'seed':42,'cold_state_sha256':'28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565'}
    write(b/'analysis_contract.json',contract)
    code=(b/'run_analysis.py').read_text().replace('new_interest.experiments.history_token_analysis','new_interest.experiments.token_joint_alignment_analysis').replace('experiments/history_token_analysis.py','experiments/token_joint_alignment_analysis.py').replace('HISTORY_TOKEN_PAIRED_ANALYSIS_ACCEPTED','JOINT_ALIGNMENT_PAIRED_ANALYSIS_ACCEPTED')
    (b/'run_analysis.py').write_text(code)
    # Remove unused analysis implementation, keeping the closure narrow.
    (b/'src/new_interest/experiments/history_token_analysis.py').unlink()
    payload={str(p.relative_to(b)):sha(p) for p in sorted(b.rglob('*')) if p.is_file() and p.name not in ['payload.json','.amltignore'] and '__pycache__' not in p.parts}
    write(b/'payload.json',payload)
    cfg=yaml.safe_load((P/'configs/amlt/mind_history_token_analysis_1001_v2.yaml').read_text())
    cfg['description']='CPU-only accepted Token125 baseline versus bidirectional joint SFT: complete checkpoint hashes, Full Dev recomputation and paired user bootstrap; no training.'
    cfg['code']['local_dir']='$CONFIG_DIR/../../'+str(b.relative_to(P));cfg['jobs'][0]['name']='token125-joint-paired-analysis'
    yp=P/'configs/amlt/mind_token_joint_analysis_1002_v1.yaml';assert not yp.exists();yp.write_text(yaml.safe_dump(cfg,sort_keys=False))
    (A/'analysis').mkdir()
    tracking={'status':'prepared','artifact_root':str((A/'analysis').relative_to(P)),'active_run':{
        'experiment':'mind-token125-joint-analysis-1002-v1','job':cfg['jobs'][0]['name'],'yaml':str(yp.relative_to(P)),
        'bundle':str(b.relative_to(P)),'payload_sha256':sha(b/'payload.json')}}
    write(P/'manifests/mind_token_joint_analysis_20261002_v1.json',tracking);print(json.dumps(tracking,indent=2))


if __name__=='__main__':prepare()
