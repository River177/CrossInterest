"""Accept one standalone direct-supervision arm after native terminal completion."""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def accept(manifest_path,receipt_path):
 m=read(manifest_path);run=m['active_run'];r=read(receipt_path);root=receipt_path.parent/'compact';bundle=Path(run['bundle'])
 assert r['native_status']=='Completed' and r['backend']==run['backend_id'] and r['results_prefix']==run['results_prefix']
 assert sha(bundle/'manifests/cir_bundle.json')==run['bundle_sha256']
 for rel,info in r['compact_files'].items():assert sha(root/rel)==info['sha256']
 complete=read(root/'complete.json');assert complete['status']=='accepted' and complete['bundle_sha256']==run['bundle_sha256'] and len(complete['stages'])==1
 contract=read(bundle/'configs/experiments/mind_cir_108_v1.json');assert contract['direct_training']=={'samples':1055,'epochs':3,'optimizer_steps':99,'presentations':3165,'parent':None}
 stage=read(root/'formal/ni/stage_accepted.json');summary=read(root/'formal/ni/run_summary.json');reload=read(root/'formal/ni/semantic_adapter_reload_audit.json');start=read(root/'formal/ni/training_start_audit.json')
 assert complete['stages'][0]==stage and stage['samples']==1055 and stage['epochs']==3 and stage['optimizer_steps']==99 and stage['stage_parent'] is None
 assert stage['initial_state_sha256']==start['actual_adapter_state_sha256']==start['source_adapter_state_sha256']=='f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e'
 assert start['global_step']==start['optimizer_state_entries']==0 and start['fresh_optimization_state']
 assert summary['input_adapter'] is None and summary['actual_optimizer_steps']==99 and summary['observed_training_exposure']['presentations']==3165
 assert summary['observed_training_exposure']['min_presentations_per_sample']==summary['observed_training_exposure']['max_presentations_per_sample']==3
 assert summary['train_file_sha256']==sha(bundle/'data/training/mind_cir_108_v1/ni.parquet')
 assert reload['all_close'] and reload['atol']==reload['rtol']==0 and reload['adapter_state_integrity']['all_tensors_exact'] and reload['adapter_state_integrity']['tensor_count']==373
 assert reload['adapter_state_integrity']['saved_state_sha256']==reload['adapter_state_integrity']['loaded_state_sha256']
 cp=root/'formal/checkpoints-ni/checkpoint-99';marker=read(cp/'checkpoint_complete.json');exposure=read(cp/'training_exposure_state.json');cursor=read(cp/'old_new_execution_cursor.json')
 assert marker['global_step']==exposure['optimizer_step']==cursor['global_step']==99
 assert len(exposure['counts'])==1055 and set(exposure['counts'])=={3} and cursor['consumed_total']==3165
 for name in ['trainer_state.json','training_exposure_state.json','old_new_execution_cursor.json']:assert sha(cp/name)==marker['files'][name]['sha256']
 merged=root/'formal/real_eval/merged';metrics=read(merged/'metrics.json');merge=read(merged/'merge_manifest.json');remote=read(root/'formal/evaluation_readback.json')
 assert complete['evaluation']==remote and remote['records']==metrics['samples']==merge['record_count']==merge['expected_record_count']==31392
 assert remote['prediction_sha256']==merge['merged_prediction_sha256'] and remote['metrics_sha256']==merge['metrics_sha256']==sha(merged/'metrics.json')
 assert remote['evaluation_identity']==metrics['evaluation_identity']==merge['evaluation_identity']
 data=bundle/'data/training/mind_cir_108_v1';assert metrics['eval_sha256']==sha(data/'dev_full_grouped.parquet') and metrics['mapping_sha256']==sha(data/'interest_token_mapping.parquet')
 assert metrics['label_vocabulary_size']==108 and not metrics['history_prediction_filtering'] and metrics['top_k']==20
 dev=pq.read_table(data/'dev_full_grouped.parquet').to_pylist();support=Counter(c for row in dev for c in row['acceptable_subcategories'])
 assert len(support)==104 and sum(support.values())==37728 and dict(support)==metrics['per_class_support']
 shards=[]
 for i in range(4):
  folder=root/f'formal/real_eval/shard-{i:02d}';s=read(folder/'metrics.json');sm=read(folder/'shard_manifest.json');assert s['samples']==sm['shard_record_count']==7848 and s['shard_index']==sm['shard_index']==i
  assert sm['first_record_index']==i and sm['last_record_index']==31388+i
  for key in ['model_name','model_revision','adapter_sha256','eval_sha256','mapping_sha256','evaluation_identity']:assert s[key]==sm[key]==metrics[key]
  assert sm['prediction_sha256']==merge['shard_prediction_sha256'][str(i)]
  assert s['per_class_support']==dict(Counter(c for row in dev[i::4] for c in row['acceptable_subcategories']))
  shards.append(s)
 recomputed={}
 for k in (1,5,10,20):
  for kind in ('recall','ndcg'):
   per={c:math.fsum(s['per_class_support'].get(c,0)*s[f'per_class_{kind}'][str(k)].get(c,0) for s in shards)/n for c,n in support.items()}
   assert all(abs(v-metrics[f'per_class_{kind}'][str(k)][c])<1e-12 for c,v in per.items())
   key=f'interest_macro_{kind}_at_{k}';recomputed[key]=math.fsum(per.values())/104
   assert abs(recomputed[key]-metrics[key])<1e-12 and abs(recomputed[key]-remote['metrics'][key])<1e-12
 result={'status':'accepted','arm':m['arm'],'native_status':'Completed','source_receipt':str(receipt_path),'source_receipt_sha256':sha(receipt_path),'bundle_sha256':run['bundle_sha256'],'complete_sha256':sha(root/'complete.json'),'single_cold_stage':True,'training_steps':99,'presentations':3165,'exact_reload_tensors':373,'full_dev_records':31392,'supported_classes':104,'target_occurrences':37728,'metrics':recomputed,'evaluation':remote,'large_artifacts':'remote_only'}
 out=Path(m['artifact_root'])/'final_acceptance.json';out.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();print(json.dumps(accept(a.manifest,a.receipt),indent=2))
