"""Verify a completed remote CPU comparison against the two accepted SFT arms."""
import argparse,json,hashlib,math
from pathlib import Path

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def accept(receipt_path):
 parent_path=Path('manifests/mind_cir_ni_direct_training_20260929_v1.json');parent=read(parent_path);job=parent['paired_analysis'];r=read(receipt_path);c=receipt_path.parent/'compact'
 assert r['native_status']=='Completed' and r['backend']==job['backend_id'] and r['results_prefix']==job['results_prefix']
 for rel,record in r['compact_files'].items():assert sha(c/rel)==record['sha256']
 complete=read(c/'complete.json');env=read(c/'analysis_receipt.json');data=read(c/'comparison.json')
 assert complete['status']==env['status']=='accepted' and data['status']=='passed'
 assert sha(c/'comparison.json')==complete['comparison_sha256']==env['comparison_sha256']
 assert sha(c/'analysis_receipt.json')==complete['analysis_receipt_sha256']
 bundle=Path(job['bundle']);assert sha(bundle/'payload.json')==complete['payload_sha256']==env['payload_sha256']==job['payload_sha256']
 assert sha(bundle/'src/new_interest/experiments/cir_ni_direct_analysis.py')==data['analysis_source_sha256']==env['analysis_source_sha256']
 assert env['numpy']=='2.2.6' and env['pyarrow']=='20.0.0' and env['device']=='CPU'
 assert (data['records'],data['users'],data['target_occurrences'],data['supported_classes'])==(31392,27082,37728,104)
 assert data['uncertainty']=={'method':'paired-user-Exp(1)-Bayesian-bootstrap','draws':2000,'seed':42,'training_seed_uncertainty_included':False}
 assert data['comparison']=='cir_as_ni_minus_original_ni_paired'
 base=Path('artifacts/mind-cir-ni-direct-training-20260929')
 for arm in ('cir_as_ni','original_ni_paired'):
  accepted=read(base/arm/'final_acceptance.json');assert accepted['status']=='accepted'
  assert data['sources'][arm]['prediction_sha256']==accepted['evaluation']['prediction_sha256']
  assert data['sources'][arm]['metrics_sha256']==accepted['evaluation']['metrics_sha256']
  assert data['sources'][arm]['evaluation_identity']==accepted['evaluation']['evaluation_identity']
  assert all(abs(v-accepted['metrics'][k])<1e-12 for k,v in data['metrics'][arm].items())
 for key,item in data['differences'].items():
  assert abs(item['difference_pp']-100*(data['metrics']['cir_as_ni'][key]-data['metrics']['original_ni_paired'][key]))<1e-10
  lo,hi=item['paired_user_bayesian_95_interval_pp'];assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
 for name in ['comparison.json','analysis_receipt.json','complete.json']:
  dst=base/('analysis_complete.json' if name=='complete.json' else name);payload=(c/name).read_bytes()
  if dst.exists():assert dst.read_bytes()==payload
  else:dst.write_bytes(payload)
 result={'status':'accepted','source_receipt':str(receipt_path),'source_receipt_sha256':sha(receipt_path),'analysis_backend':r['backend'],'analysis_native_status':'Completed','comparison_sha256':sha(c/'comparison.json'),'analysis_receipt_sha256':sha(c/'analysis_receipt.json'),'both_arm_acceptance_sha256':{a:sha(base/a/'final_acceptance.json') for a in ('cir_as_ni','original_ni_paired')},'full_predictions_recomputed_remotely':True,'users':27082,'records':31392,'bootstrap_draws':2000,'seed':42,'training_seed_uncertainty_included':False,'large_payloads':'remote_only'}
 (base/'paired_final_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(data['differences'],indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();accept(a.receipt)
