"""Independent compact readback for the mixed NI/bidirectional alignment arm."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import math
import sys

import pyarrow.parquet as pq

from read_history_token_state import read,sha,stage,evaluation

COLD='28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565'


def verify(manifest_path,receipt_path):
    tracking=read(manifest_path);run=tracking['active_run'];r=read(receipt_path)
    bundle=Path(run['bundle']);root=receipt_path.parent/'compact';data=bundle/'data/training/mind_cir_108_v1'
    assert r['backend']==run['backend_id'] and r['results_prefix']==run['results_prefix']
    assert sha(bundle/'manifests/cir_bundle.json')==run['bundle_sha256']
    for rel,item in r['compact_files'].items():assert sha(root/rel)==item['sha256'] and (root/rel).stat().st_size==item['bytes']
    result={'status':'pending','stage':'pending_startup','native_status':r['native_status'],'observed_utc':r['observed_utc'],
            'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'final_accepted':False}
    cfg=read(bundle/'configs/experiments/mind_cir_108_v1.json')
    if (root/'identity.json').exists():
        assert read(root/'identity.json')['bundle_sha256']==run['bundle_sha256'];result['stage']='initializing'
    if (root/'environment.json').exists():
        env=read(root/'environment.json');assert env['packages']==cfg['worker_packages'] and env['kernel_profile']=='torch-fallback-explicit'
        assert len(env['devices'])==4 and all('A100' in x['name'] and x['bytes']>=79*1024**3 for x in env['devices'])
    if (root/'snapshot_audit.json').exists():assert read(root/'snapshot_audit.json')['files']==read(bundle/'manifests/h50_snapshot.json')['files']
    if (root/'joint_length_audit.json').exists():
        lengths=read(root/'joint_length_audit.json');stats=read(data/'statistics.json')
        assert lengths['all_lengths_exact'] and lengths['tasks']==stats['tasks']
        assert lengths['assistant_tokens_per_epoch']==stats['output_tokens_per_epoch_by_task']
        result['lengths_verified']=True
    if (root/'gradient_audit.json').exists():
        g=read(root/'gradient_audit.json');assert g['representative_samples']==258 and g['interest_token_count']==125
        assert g['all_token_rows_finite_and_nonzero'] and len(g['gradient_parameters'])==1
        assert all(x['shape']==[125,2048] and x['finite_rows']==x['nonzero_rows']==125 for x in g['gradient_parameters'])
        result['stage']='gradient_passed'
    for name,n,epochs,steps,world in [('single_ni',32,2,8,1),('ddp_ni',41,1,2,4)]:
        if (root/f'gates/{name}/stage_accepted.json').exists():
            stage(root/'gates',name,n,epochs,steps,world,COLD)
            result['stage']=name+'_passed'
    if (root/'gates/accepted.json').exists():
        gate=read(root/'gates/accepted.json');assert gate['bundle_sha256']==run['bundle_sha256'] and len(gate['stages'])==2
        result['gate_evaluation']=evaluation(root,bundle,'gates');result['stage']='gates_passed'
        for name in ['ni','cir']:
            audit=read(root/f'token_audit_{name}.json');assert audit['samples']==24100 and audit['over_max_seq_length']==0
        assert result['lengths_verified']
    if (root/'formal/ni/training_start_audit.json').exists():
        start=read(root/'formal/ni/training_start_audit.json');initial=read(root/'formal/ni/initial_adapter_state.json')
        assert initial['state_sha256']==start['source_adapter_state_sha256']==start['actual_adapter_state_sha256']==COLD
        assert start['fresh_optimization_state'] and start['optimizer_state_entries']==start['global_step']==0
        assert start['checkpoint_identity']['train_file_sha256']==sha(data/'ni.parquet')
        result.update(stage='formal_training',cold_state_matches_baseline=True)
    cpstep=r['checkpoints'].get('formal/checkpoints-ni')
    if cpstep is not None:
        sys.path.insert(0,str(bundle/'src'))
        module_path=bundle/'src/new_interest/training/old_new_execution.py'
        spec=importlib.util.spec_from_file_location('joint_cursor_readback',module_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        cp=root/f'formal/checkpoints-ni/checkpoint-{cpstep}';contract=read(root/'formal/ni/cursor_contract.json')
        assert contract['samples']==24100 and contract['total_steps']==2262 and contract['epochs']==3
        mod.audit_checkpoint_cursor(cp,contract)
        e=read(cp/'training_exposure_state.json');marker=read(cp/'checkpoint_complete.json');cursor=read(cp/'old_new_execution_cursor.json')
        assert marker['global_step']==cursor['global_step']==e['optimizer_step']==cpstep
        assert marker['identity']==read(root/'formal/ni/training_start_audit.json')['checkpoint_identity']
        rows=pq.read_table(data/'ni.parquet',columns=['training_task','expected_assistant_tokens']).to_pylist()
        assert contract['expected_assistant_tokens']==[x['expected_assistant_tokens'] for x in rows]
        by_task=Counter()
        for row,count in zip(rows,e['counts']):by_task[row['training_task']]+=count
        inv={x['name']:x for x in read(receipt_path.parent/'blob_inventory.json')}
        for rel,item in marker['files'].items():
            assert inv[f'formal/checkpoints-ni/checkpoint-{cpstep}/'+rel]['size']==item['size']
            if (cp/rel).exists():assert sha(cp/rel)==item['sha256']
        result['checkpoint']={'step':cpstep,'presentations':sum(e['counts']),'presentations_by_task':dict(by_task),
            'per_sample_and_supervised_token_counts_exact':True,'large_payloads':'remote only; local check covers compact hashes and inventory sizes'}
    if (root/'formal/ni/stage_accepted.json').exists():
        stage(root/'formal','ni',24100,3,2262,4,COLD)
        s=read(root/'formal/ni/stage_accepted.json');expected={'new_interest':64800,'interest_name_to_id':3750,'interest_id_to_name':3750}
        assert s['exposure_by_task']==expected and result['checkpoint']['presentations_by_task']==expected
        result['stage']='training_accepted_evaluating'
    if (root/'complete.json').exists():
        complete=read(root/'complete.json');assert complete['status']=='accepted' and complete['bundle_sha256']==run['bundle_sha256']
        result['evaluation']=evaluation(root,bundle,'formal')
        assert result['checkpoint']['step']==2262 and result['checkpoint']['presentations']==72300
        if r['native_status']=='Completed':
            result.update(status='accepted',stage='accepted',final_accepted=True)
            Path(tracking['artifact_root'],'final_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        else:result['stage']='artifacts_complete_pending_platform'
    (receipt_path.parent/'independent_readback.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=Path('manifests/mind_token_joint_alignment_20261001_v1.json'));p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.manifest,a.receipt),indent=2))
