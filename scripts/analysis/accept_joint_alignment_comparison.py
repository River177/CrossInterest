"""Accept the remote paired comparison against both frozen SFT acceptances."""
import argparse
import hashlib
import json
import math
from pathlib import Path

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def accept(receipt_path):
    tracking=read(Path('manifests/mind_token_joint_analysis_20261002_v1.json'))
    job=tracking['active_run'];receipt=read(receipt_path);compact=receipt_path.parent/'compact';bundle=Path(job['bundle'])
    assert receipt['native_status']=='Completed' and receipt['backend']==job['backend_id'] and receipt['results_prefix']==job['results_prefix']
    for rel,item in receipt['compact_files'].items():assert sha(compact/rel)==item['sha256']
    done=read(compact/'complete.json');env=read(compact/'analysis_receipt.json');data=read(compact/'comparison.json')
    assert done['status']==env['status']=='accepted' and data['status']=='passed'
    assert sha(compact/'comparison.json')==done['comparison_sha256']==env['comparison_sha256']
    assert sha(compact/'analysis_receipt.json')==done['analysis_receipt_sha256']
    assert sha(bundle/'payload.json')==done['payload_sha256']==env['payload_sha256']==job['payload_sha256']
    assert sha(bundle/'src/new_interest/experiments/token_joint_alignment_analysis.py')==data['analysis_source_sha256']==env['analysis_source_sha256']
    assert env['device']=='CPU' and env['numpy']=='2.2.6' and env['pyarrow']=='20.0.0'
    assert env['full_checkpoint_and_prediction_hashes_verified_remotely']
    assert tuple(data[k] for k in ['records','users','target_occurrences','supported_classes'])==(31392,27082,37728,104)
    assert data['comparison']=='joint_minus_baseline'
    assert data['uncertainty']=={'method':'paired-user-Exp(1)-Bayesian-bootstrap','draws':2000,'seed':42,'training_seed_uncertainty_included':False}
    contract=read(bundle/'analysis_contract.json');root=Path('artifacts/mind-token-joint-alignment-20261001')
    for arm in ['baseline','joint']:
        expected=contract['arms'][arm];accepted=read(Path(expected['acceptance_path']))
        assert accepted['final_accepted']
        assert accepted.get('cold_state_matches_baseline',accepted.get('cross_arm_cold_state_equal',False))
        assert sha(Path(expected['acceptance_path']))==expected['acceptance_sha256']
        if arm=='baseline':assert accepted['cold_state_sha256']==contract['cold_state_sha256']
        for key,value in data['sources'][arm].items():assert value==expected[key]
        for key,value in data['metrics'][arm].items():
            assert abs(value-accepted['evaluation']['metrics'][key])<1e-12 and abs(value-expected['metrics'][key])<1e-12
        audit=data['checkpoint_audits'][arm]
        assert audit['step']==expected['optimizer_steps'] and audit['files']>10 and audit['bytes']>0 and audit['all_payload_hashes_verified']
    for key,item in data['differences'].items():
        assert abs(item['difference_pp']-100*(data['metrics']['joint'][key]-data['metrics']['baseline'][key]))<1e-10
        lo,hi=item['paired_user_bayesian_95_interval_pp'];assert all(math.isfinite(v) for v in [lo,hi]) and -100<=lo<=hi<=100
    result={'status':'accepted','receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),
        'comparison_sha256':sha(compact/'comparison.json'),'analysis_backend':job['backend_id'],
        'full_checkpoints_and_predictions_verified_remotely':True,'paired_users':27082,
        'arm_acceptance_sha256':{arm:sha(Path(contract['arms'][arm]['acceptance_path'])) for arm in ['baseline','joint']},
        'training_seed_uncertainty_included':False}
    for name in ['comparison.json','analysis_receipt.json']:
        target=root/name
        if target.exists():assert target.read_bytes()==(compact/name).read_bytes()
        else:target.write_bytes((compact/name).read_bytes())
    (root/'paired_final_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(data['differences'],indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();accept(a.receipt)
