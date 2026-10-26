"""Independent compact evidence reader for the matched history-token jobs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation(root,bundle,phase):
    base=root/phase;merged=base/'real_eval/merged'
    metrics=read(merged/'metrics.json');manifest=read(merged/'merge_manifest.json')
    remote=read(base/'evaluation_readback.json')
    n=64 if phase=='gates' else 31392
    data=bundle/'data/training/mind_cir_108_v1'
    rows=pq.read_table(data/'dev_full_grouped.parquet').slice(0,n).to_pylist()
    support=Counter(c for r in rows for c in r['acceptable_subcategories'])
    assert metrics['samples']==manifest['record_count']==manifest['expected_record_count']==remote['records']==n
    assert metrics['label_vocabulary_size']==108 and metrics['history_prediction_filtering'] is False
    assert metrics['top_k']==20 and metrics['cutoffs']==[1,5,10,20]
    assert metrics['per_class_support']==dict(support)
    assert metrics['eval_sha256']==sha(data/'dev_full_grouped.parquet')
    assert metrics['mapping_sha256']==sha(data/'interest_token_mapping.parquet')
    assert sha(merged/'metrics.json')==manifest['metrics_sha256']==remote['metrics_sha256']
    assert remote['prediction_sha256']==manifest['merged_prediction_sha256']
    shards=[]
    for i in range(4):
        folder=base/f'real_eval/shard-{i:02d}'
        s=read(folder/'metrics.json');m=read(folder/'shard_manifest.json')
        assert s['samples']==m['shard_record_count']==n//4
        assert s['shard_index']==m['shard_index']==i
        assert s['per_class_support']==dict(Counter(c for r in rows[i::4] for c in r['acceptable_subcategories']))
        assert m['prediction_sha256']==manifest['shard_prediction_sha256'][str(i)]
        for key in ['model_name','model_revision','adapter_sha256','eval_sha256','mapping_sha256','evaluation_identity']:
            assert s[key]==m[key]==metrics[key]
        shards.append(s)
    recomputed={}
    for k in [1,5,10,20]:
        for kind in ['recall','ndcg']:
            per={c:math.fsum(s['per_class_support'].get(c,0)*s[f'per_class_{kind}'][str(k)].get(c,0) for s in shards)/count for c,count in support.items()}
            assert all(abs(v-metrics[f'per_class_{kind}'][str(k)][c])<1e-12 for c,v in per.items())
            key=f'interest_macro_{kind}_at_{k}';recomputed[key]=math.fsum(per.values())/len(support)
            assert abs(recomputed[key]-metrics[key])<1e-12 and abs(recomputed[key]-remote['metrics'][key])<1e-12
    return {'records':n,'supported_classes':len(support),'target_occurrences':sum(support.values()),'metrics':recomputed}


def stage(root,name,n,epochs,steps,world,cold):
    folder=root/name;s=read(folder/'stage_accepted.json');r=read(folder/'semantic_adapter_reload_audit.json')
    summary=read(folder/'run_summary.json');initial=read(folder/'initial_adapter_state.json');start=read(folder/'training_start_audit.json')
    assert (s['samples'],s['epochs'],s['optimizer_steps'],s['stage_parent'])==(n,epochs,steps,None)
    assert initial['state_sha256']==s['initial_state_sha256']==start['actual_adapter_state_sha256']==start['source_adapter_state_sha256']==cold
    assert start['fresh_optimization_state'] and start['global_step']==start['optimizer_state_entries']==0
    assert summary['actual_optimizer_steps']==steps and summary['world_size']==world and summary['input_adapter'] is None
    assert summary['model_audit']['semantic_id_tokens']['count']==125
    assert summary['observed_training_exposure']['presentations']==n*epochs
    assert r['all_close'] and r['atol']==r['rtol']==0 and r['adapter_state_integrity']['all_tensors_exact']
    assert r['adapter_state_integrity']['tensor_count']==373
    assert r['adapter_state_integrity']['saved_state_sha256']==r['adapter_state_integrity']['loaded_state_sha256']
    assert all(shape==[125,2048] for shape in s['adapter']['token_shapes'].values())
    assert math.isfinite(summary['train_metrics']['train_loss'])
    if name.startswith('single'):
        history=summary['loss_history'];assert all(math.isfinite(x['loss']) for x in history)
        assert history[-1]['loss']<history[0]['loss']


def verify(manifest_path,receipt_path):
    tracking=read(manifest_path);run=tracking['active_run'];receipt=read(receipt_path)
    bundle=Path(run['bundle']);root=receipt_path.parent/'compact'
    assert receipt['backend']==run['backend_id'] and receipt['results_prefix']==run['results_prefix']
    assert sha(bundle/'manifests/cir_bundle.json')==run['bundle_sha256']
    for rel,item in receipt['compact_files'].items():
        assert sha(root/rel)==item['sha256'] and (root/rel).stat().st_size==item['bytes']
    result={'arm':tracking['arm'],'observed_utc':receipt['observed_utc'],'native_status':receipt['native_status'],
            'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'stage':'pending_startup','final_accepted':False,
            'committed_checkpoints':receipt['checkpoints']}
    if (root/'identity.json').exists():
        assert read(root/'identity.json')['bundle_sha256']==run['bundle_sha256']
        result['stage']='initializing'
    if (root/'environment.json').exists():
        env=read(root/'environment.json');cfg=read(bundle/'configs/experiments/mind_cir_108_v1.json')
        assert env['packages']==cfg['worker_packages'] and env['kernel_profile']=='torch-fallback-explicit'
        assert len(env['devices'])==4 and all('A100' in x['name'] and x['bytes']>=79*1024**3 for x in env['devices'])
    if (root/'snapshot_audit.json').exists():
        assert read(root/'snapshot_audit.json')['files']==read(bundle/'manifests/h50_snapshot.json')['files']
    if (root/'gradient_audit.json').exists():
        g=read(root/'gradient_audit.json');assert g['interest_token_count']==125 and g['all_token_rows_finite_and_nonzero']
        assert g['semantic_initialization']['mapping_sha256']==sha(bundle/'data/training/mind_cir_108_v1/model_token_mapping.parquet')
        assert len(g['gradient_parameters'])==1
        assert all(x['shape']==[125,2048] and x['finite_rows']==x['nonzero_rows']==125 and x['min_row_gradient_norm']>0 for x in g['gradient_parameters'])
        result['stage']='gradient_passed';result['finite_nonzero_gradient_rows']=125
    if (root/'gates/single_ni/stage_accepted.json').exists():
        cold=read(root/'gates/single_ni/initial_adapter_state.json')['state_sha256']
        stage(root/'gates','single_ni',32,2,8,1,cold)
        result.update(stage='single_gpu_passed',cold_state_sha256=cold)
    if (root/'gates/accepted.json').exists():
        gate=read(root/'gates/accepted.json');assert gate['bundle_sha256']==run['bundle_sha256']
        cold=read(root/'gates/single_ni/initial_adapter_state.json')['state_sha256']
        stage(root/'gates','single_ni',32,2,8,1,cold)
        stage(root/'gates','ddp_ni',41,1,2,4,cold)
        result.update(stage='gates_passed',cold_state_sha256=cold,gate_evaluation=evaluation(root,bundle,'gates'))
        for task in ['cir','ni']:
            a=read(root/f'token_audit_{task}.json');assert a['samples']==21600 and a['over_max_seq_length']==0
        dev=read(root/'dev_token_audit.json');assert dev['records']==31392 and dev['over_max_seq_length']==0
        if (root/'formal/ni/training_start_audit.json').exists():
            start=read(root/'formal/ni/training_start_audit.json')
            assert start['actual_adapter_state_sha256']==cold and start['fresh_optimization_state']
            assert start['checkpoint_identity']['train_file_sha256']==sha(bundle/'data/training/mind_cir_108_v1/ni.parquet')
            result['stage']='formal_training'
    step=receipt['checkpoints'].get('formal/checkpoints-ni')
    if step is not None:
        import torch
        checkpoint=root/'formal/checkpoints-ni'/f'checkpoint-{step}'
        marker=read(checkpoint/'checkpoint_complete.json')
        exposure=read(checkpoint/'training_exposure_state.json')
        cursor=read(checkpoint/'old_new_execution_cursor.json')
        state=read(checkpoint/'trainer_state.json')
        start=read(root/'formal/ni/training_start_audit.json')
        assert marker['identity']==start['checkpoint_identity']
        assert marker['global_step']==exposure['optimizer_step']==cursor['global_step']==state['global_step']==step
        assert 0<step<=2025 and state['max_steps']==2025
        epochs,remainder=divmod(step,675)
        expected=[epochs]*21600
        if remainder:
            order=torch.randperm(21600,generator=torch.Generator(device='cpu').manual_seed(42+epochs)).tolist()
            for i in order[:remainder*32]:expected[i]+=1
        assert exposure['counts']==expected
        assert cursor['consumed_total']==sum(expected)==step*32
        assert cursor['exposure_sha256']==sha(checkpoint/'training_exposure_state.json')
        assert cursor['trainer_state_sha256']==sha(checkpoint/'trainer_state.json')
        inventory={x['name']:x for x in read(receipt_path.parent/'blob_inventory.json')}
        for name,entry in marker['files'].items():
            assert inventory[f'formal/checkpoints-ni/checkpoint-{step}/'+name]['size']==entry['size']
            if (checkpoint/name).exists():assert sha(checkpoint/name)==entry['sha256']
        assert {'optimizer.pt','scheduler.pt','adapter_model.safetensors',*[f'rng_state_{i}.pth' for i in range(4)]}<=set(marker['files'])
        result['checkpoint_readback']={'step':step,'presentations':sum(expected),'per_sample_counts_exact':True,
            'boundary':'Compact hashes and remote payload sizes checked; large model/optimizer bytes remain remote.'}
        if step==2025:
            assert state['epoch']==3
            result.update(stage='training_updates_complete_pending_stage_acceptance',
                          formal_optimizer_steps=2025,formal_presentations=64800)
    if (root/'formal/ni/stage_accepted.json').exists():
        cold=result['cold_state_sha256']
        stage(root/'formal','ni',21600,3,2025,4,cold)
        assert result.get('checkpoint_readback',{}).get('step')==2025
        result.update(stage='training_accepted_evaluating',formal_training_accepted=True,
                      formal_optimizer_steps=2025,formal_presentations=64800)
    if (root/'complete.json').exists():
        complete=read(root/'complete.json');assert complete['bundle_sha256']==run['bundle_sha256'] and complete['status']=='accepted'
        cold=result['cold_state_sha256'];stage(root/'formal','ni',21600,3,2025,4,cold)
        result['evaluation']=evaluation(root,bundle,'formal')
        checkpoint=root/'formal/checkpoints-ni/checkpoint-2025'
        marker=read(checkpoint/'checkpoint_complete.json')
        exposure=read(checkpoint/'training_exposure_state.json');cursor=read(checkpoint/'old_new_execution_cursor.json')
        assert marker['global_step']==exposure['optimizer_step']==cursor['global_step']==2025
        assert len(exposure['counts'])==21600 and set(exposure['counts'])=={3} and cursor['consumed_total']==64800
        for name in ['trainer_state.json','training_exposure_state.json','old_new_execution_cursor.json']:
            assert sha(checkpoint/name)==marker['files'][name]['sha256']
        if receipt['native_status']=='Completed':
            result.update(stage='accepted',final_accepted=True)
        else:result['stage']='artifacts_complete_pending_platform'
    sibling='token' if tracking['arm']=='text' else 'text'
    sibling_root=Path(tracking['artifact_root']).parent/sibling
    others=sorted(sibling_root.glob('progress/*/independent_readback.json'),reverse=True)
    if result.get('cold_state_sha256'):
        for path in others:
            other=read(path)
            if other.get('cold_state_sha256'):
                assert other['cold_state_sha256']==result['cold_state_sha256'], 'Text125/Token125 initial parameters differ'
                result['cross_arm_cold_state_equal']=True
                break
    if result['final_accepted']:
        assert result.get('cross_arm_cold_state_equal'), 'Cross-arm initialization check is still pending'
        Path(tracking['artifact_root'],'final_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    (receipt_path.parent/'independent_readback.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.manifest,args.receipt),indent=2))
