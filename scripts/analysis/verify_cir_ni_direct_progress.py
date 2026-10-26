"""Cross-check direct-training startup and compact checkpoint receipts."""
import argparse,ast,hashlib,json,math
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

COLD='f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(mp,rp):
 m=read(mp);run=m['active_run'];r=read(rp);c=rp.parent/'compact';b=Path(run['bundle']);cfg=read(b/'configs/experiments/mind_cir_108_v1.json')
 assert r['backend']==run['backend_id'] and r['results_prefix']==run['results_prefix'] and r['native_status'] in ('Running','Completed')
 assert read(c/'identity.json')['bundle_sha256']==run['bundle_sha256']==sha(b/'manifests/cir_bundle.json')
 for rel,record in r['compact_files'].items():assert sha(c/rel)==record['sha256'] and (c/rel).stat().st_size==record['bytes']
 env=read(c/'environment.json');assert env['packages']==cfg['worker_packages'] and env['kernel_profile']=='torch-fallback-explicit'
 assert len(env['devices'])==4 and all('A100' in x['name'] and x['bytes']>=79*1024**3 for x in env['devices'])
 assert read(c/'snapshot_audit.json')['files']==read(b/'manifests/h50_snapshot.json')['files']
 gate=read(c/'gates/accepted.json');assert gate['bundle_sha256']==run['bundle_sha256'] and len(gate['stages'])==2
 for i,(name,n,e,steps,world) in enumerate([('single_ni',32,2,8,1),('ddp_ni',41,1,2,4)]):
  f=c/'gates'/name;s=read(f/'stage_accepted.json');summary=read(f/'run_summary.json');reload=read(f/'semantic_adapter_reload_audit.json')
  assert s==gate['stages'][i] and (s['samples'],s['epochs'],s['optimizer_steps'])==(n,e,steps) and s['stage_parent'] is None
  assert s['initial_state_sha256']==COLD and summary['world_size']==world and summary['actual_optimizer_steps']==steps
  assert summary['observed_training_exposure']['presentations']==n*e
  assert reload['all_close'] and reload['atol']==reload['rtol']==0 and reload['adapter_state_integrity']['all_tensors_exact']
  assert reload['adapter_state_integrity']['tensor_count']==373 and reload['adapter_state_integrity']['saved_state_sha256']==reload['adapter_state_integrity']['loaded_state_sha256']
  if name.startswith('single'):
   losses=summary['loss_history'];assert len(losses)>=2 and losses[-1]['loss']<losses[0]['loss'] and all(math.isfinite(x['loss']) for x in losses)
 dev=pq.read_table(b/'data/training/mind_cir_108_v1/dev_full_grouped.parquet').slice(0,64).to_pylist();support=Counter(x for row in dev for x in row['acceptable_subcategories'])
 metrics=read(c/'gates/real_eval/merged/metrics.json');merge=read(c/'gates/real_eval/merged/merge_manifest.json');remote=read(c/'gates/evaluation_readback.json')
 assert remote==gate['evaluation'] and remote['records']==metrics['samples']==64 and remote['supported_classes']==len(support)==36 and remote['target_occurrences']==sum(support.values())==75
 assert remote['metrics_sha256']==merge['metrics_sha256']==sha(c/'gates/real_eval/merged/metrics.json') and remote['prediction_sha256']==merge['merged_prediction_sha256']
 assert metrics['per_class_support']==dict(support)
 shards=[]
 for i in range(4):
  s=read(c/f'gates/real_eval/shard-{i:02d}/metrics.json');sm=read(c/f'gates/real_eval/shard-{i:02d}/shard_manifest.json')
  assert s['samples']==sm['shard_record_count']==16 and s['shard_index']==sm['shard_index']==i
  assert s['per_class_support']==dict(Counter(x for row in dev[i::4] for x in row['acceptable_subcategories']))
  assert sm['prediction_sha256']==merge['shard_prediction_sha256'][str(i)]
  for key in ['model_name','model_revision','adapter_sha256','eval_sha256','mapping_sha256','evaluation_identity']:assert s[key]==sm[key]==metrics[key]
  shards.append(s)
 for k in (1,5,10,20):
  for kind in ('recall','ndcg'):
   score=math.fsum(math.fsum(s['per_class_support'].get(label,0)*s[f'per_class_{kind}'][str(k)].get(label,0) for s in shards)/n for label,n in support.items())/len(support)
   key=f'interest_macro_{kind}_at_{k}';assert abs(score-metrics[key])<1e-12 and abs(score-remote['metrics'][key])<1e-12
 initial=read(c/'formal/ni/initial_adapter_state.json');start=read(c/'formal/ni/training_start_audit.json');identity=start['checkpoint_identity']
 assert initial['state_sha256']==start['actual_adapter_state_sha256']==start['source_adapter_state_sha256']==COLD and start['fresh_optimization_state'] and start['optimizer_state_entries']==start['global_step']==0
 assert identity['input_adapter_sha256'] is None and identity==initial['checkpoint_identity'] and identity['train_file_sha256']==sha(b/'data/training/mind_cir_108_v1/ni.parquet')
 assert identity['world_size']==4 and identity['epochs']==3 and identity['seed']==42 and identity['effective_global_batch_size']==32
 step=r['checkpoints'].get('formal/checkpoints-ni');presentations=None
 if step is not None:
  cp=c/'formal/checkpoints-ni'/f'checkpoint-{step}';marker=read(cp/'checkpoint_complete.json');exp=read(cp/'training_exposure_state.json');cursor=read(cp/'old_new_execution_cursor.json');state=read(cp/'trainer_state.json');assert marker['identity']==identity
  assert marker['global_step']==exp['optimizer_step']==cursor['global_step']==state['global_step']==step and state['max_steps']==99
  epochs,rem=divmod(step,33);presentations=epochs*1055+rem*32
  assert len(exp['counts'])==1055 and set(exp['counts'])<={epochs,epochs+1} and sum(exp['counts'])==cursor['consumed_total']==presentations
  assert cursor['exposure_sha256']==sha(cp/'training_exposure_state.json') and cursor['trainer_state_sha256']==sha(cp/'trainer_state.json')
  inv={x['name']:x for x in read(rp.parent/'blob_inventory.json')}
  for name,entry in marker['files'].items():
   assert inv[f'formal/checkpoints-ni/checkpoint-{step}/'+name]['size']==entry['size']
   if (cp/name).exists():assert sha(cp/name)==entry['sha256']
  assert all(name in marker['files'] for name in ['optimizer.pt','scheduler.pt','adapter_model.safetensors']+[f'rng_state_{i}.pth' for i in range(4)])
 logs=[x for x in r['logs'] if x['name'].startswith('user_logs/')];assert len(logs)==1 and not any(x['has_traceback'] or x['signal15'] for x in r['logs'])
 for log in logs:
  for line in Path(log['file']).read_text().splitlines():
   if line.startswith("{'loss':"):
    x=ast.literal_eval(line);assert all(math.isfinite(float(x[k])) for k in ['loss','grad_norm','learning_rate'])
 result={'status':'startup_and_progress_verified','arm':m['arm'],'native_status':r['native_status'],'source_receipt':str(rp),'source_receipt_sha256':sha(rp),'observed_utc':r['observed_utc'],'single_GPU_steps':8,'four_GPU_steps':2,'gate_exact_reload_tensors':373,'gate_eval_records':64,'gate_metrics_independently_recomputed':True,'formal_cold_state_sha256':COLD,'formal_optimizer_fresh':True,'latest_logged_step':logs[0]['logged_steps'].get('standalone'),'last_committed_checkpoint_step':step,'committed_presentations':presentations,'checkpoint_boundary':'Compact hashes and remote inventory sizes checked; large payloads remain remote, not independently rehashed locally.','final_experiment_accepted':False}
 out=rp.parent/'startup_progress_readback.json';out.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.manifest,a.receipt),indent=2))
