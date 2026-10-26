"""Independent compact readback for one approved CIR parameter-transfer condition."""
from collections import Counter
import argparse,hashlib,json,math
from pathlib import Path
import pyarrow.parquet as pq


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def evaluation(folder, data, mapping):
    metrics=read(folder/'real_eval/merged/metrics.json');merge=read(folder/'real_eval/merged/merge_manifest.json');remote=read(folder/'evaluation_readback.json')
    rows=pq.read_table(data).to_pylist();support=Counter(c for row in rows for c in row['acceptable_subcategories'])
    require(metrics['samples']==merge['record_count']==merge['expected_record_count']==remote['records']==len(rows),'Evaluation coverage')
    require(metrics['eval_sha256']==sha(data) and metrics['mapping_sha256']==sha(mapping),'Evaluation data identity')
    require(metrics['top_k']==20 and metrics['label_vocabulary_size']==108 and not metrics['history_prediction_filtering'],'Ranking protocol')
    require(metrics['per_class_support']==dict(support) and len(support)==remote['supported_classes'] and sum(support.values())==remote['target_occurrences'],'Class denominators')
    require(sha(folder/'real_eval/merged/metrics.json')==merge['metrics_sha256']==remote['metrics_sha256'],'Metrics hash')
    require(merge['merged_prediction_sha256']==remote['prediction_sha256'] and metrics['evaluation_identity']==merge['evaluation_identity']==remote['evaluation_identity'],'Evaluation binding')
    shards=[]
    for i in range(4):
        sr=folder/f'real_eval/shard-{i:02d}';s=read(sr/'metrics.json');sm=read(sr/'shard_manifest.json')
        require(s['samples']==sm['shard_record_count']==len(rows[i::4]) and sm['shard_index']==s['shard_index']==i,'Shard coverage')
        require(sm['first_record_index']==i and sm['last_record_index']==list(range(i,len(rows),4))[-1],'Shard topology')
        require(s['per_class_support']==dict(Counter(c for row in rows[i::4] for c in row['acceptable_subcategories'])),'Shard class support')
        require(sm['prediction_sha256']==merge['shard_prediction_sha256'][str(i)],'Shard hash')
        for k in ('adapter_sha256','eval_sha256','mapping_sha256','evaluation_identity','model_name','model_revision'):require(s[k]==sm[k]==metrics[k],'Shard identity')
        shards.append(s)
    values={}
    for k in (1,5,10,20):
        for metric in ('recall','ndcg'):
            per={c:math.fsum(s['per_class_support'].get(c,0)*s[f'per_class_{metric}'][str(k)].get(c,0) for s in shards)/n for c,n in support.items()}
            require(all(abs(per[c]-metrics[f'per_class_{metric}'][str(k)][c])<1e-12 for c in support),'Per-class recomputation')
            key=f'interest_macro_{metric}_at_{k}';values[key]=math.fsum(per.values())/len(per)
            require(abs(values[key]-metrics[key])<1e-12 and abs(values[key]-remote['metrics'][key])<1e-12,'Macro recomputation')
    return {'scope':{'records':len(rows),'targets':sum(support.values()),'classes':len(support)},'metrics':values,'adapter_directory_sha256':metrics['adapter_sha256']}


def accept(manifest_path,arm):
    project=Path.cwd();tracking=read(manifest_path);bundle=project/tracking['bundle'];base=project/tracking['artifact_root'];job=tracking['jobs'][arm]
    require(sha(bundle/'manifests/transfer_bundle.json')==tracking['bundle_sha256'],'Bundle identity')
    contract=read(bundle/'manifests/cir_parameter_transfer.json');root=base/'final-readback'/tracking['experiment']/job['job']
    complete=read(root/'complete.json');composition=read(root/'formal/composite/composition.json')
    require(complete['status']=='accepted' and complete['arm']==arm and complete['transfer_bundle_sha256']==tracking['bundle_sha256'],'Completion identity')
    require(complete['composition']==composition and composition['arm']==arm and composition['optimization_steps']==0,'Composition receipt')
    sources={k:read(bundle/f'manifests/transfer_sources/{v}_state.json') for k,v in [('cold','cold'),('cir5','cir5')]}
    expected=read(root/'formal/composite/expected_state.json')
    require(len(composition['provenance'])==len(expected['tensors'])==373,'Tensor inventory')
    digest=hashlib.sha256()
    for key,info in sorted(expected['tensors'].items()):
        source='cir5' if (('trainable_tokens_delta' in key)==(arm=='keep_sid')) else 'cold'
        require(info==sources[source]['tensors'][key] and composition['provenance'][key]=={'source':source,**info},'Tensor source selection')
        digest.update(json.dumps([key,info],sort_keys=True,separators=(',',':')).encode())
    require(digest.hexdigest()==expected['state_sha256']==composition['state_sha256'],'Composed state digest')
    require(composition['trainable_lora_tensors']==372 and composition['trainable_sid_tensors']==1,'Both groups trainable')
    for sub in ('formal/composite','formal/ni3'):
        audit=read(root/sub/'semantic_adapter_reload_audit.json');integrity=audit['adapter_state_integrity']
        require(audit['all_close'] and audit['atol']==audit['rtol']==0 and integrity['all_tensors_exact'] and integrity['tensor_count']==373,'Exact reload')
        require(integrity['saved_state_sha256']==integrity['loaded_state_sha256'] and all(x['max_absolute_difference']==0 and x['top20_exact_match'] for x in audit['samples']),'Tensor/logit reload agreement')
    initial=read(root/'formal/ni3/initial_adapter_state.json');start=read(root/'formal/ni3/training_start_audit.json');summary=read(root/'formal/ni3/run_summary.json');stage=read(root/'formal/ni3/stage_accepted.json')
    require(initial['tensors']==expected['tensors'] and initial['state_sha256']==start['source_adapter_state_sha256']==start['actual_adapter_state_sha256']==composition['state_sha256'],'NI inheritance')
    require(start['fresh_optimization_state'] and start['global_step']==start['optimizer_state_entries']==start['scheduler_last_epoch']==0,'Optimizer reset')
    require(start['checkpoint_identity']['input_adapter_sha256']==composition['adapter_sha256'],'Composite file identity')
    require(stage==complete['stage'] and summary['actual_optimizer_steps']==stage['optimizer_steps']==2025 and summary['configured_epochs']==3,'Training horizon')
    exposure=summary['observed_training_exposure'];require(exposure['presentations']==64800 and exposure['min_presentations_per_sample']==exposure['max_presentations_per_sample']==3,'Exposure')
    require(summary['train_file_sha256']==sha(bundle/'data/training/mind_cir_108_v1/ni.parquet'),'Training inputs')
    require(all(n==600 for n in exposure['by_task_and_class_presentations']['new_interest'].values()) and len(exposure['by_task_and_class_presentations']['new_interest'])==108,'Class exposures')
    cp=read(root/'final_checkpoint_readback.json');require(cp==complete['final_checkpoint'] and cp['full_remote_byte_readback'] and cp['global_step']==cp['optimizer_step_min']==cp['optimizer_step_max']==cp['scheduler_last_epoch']==2025,'Final checkpoint')
    require(cp['optimizer_states']==373 and cp['presentations']==64800 and cp['adapter_sha256']==stage['adapter']['adapter_sha256'],'Final adapter/exposure')
    mapping=bundle/'data/training/mind_cir_108_v1/interest_token_mapping.parquet'
    full=evaluation(root/'formal',bundle/'data/training/mind_cir_108_v1/dev_full_grouped.parquet',mapping)
    diagnostics={}
    phases=['pre','post']+(['reference_full/pre','reference_full/post'] if arm=='keep_lora' else [])
    for phase in phases:
        diagnostics[phase]={key:evaluation(root/'diagnostics'/phase/key,bundle/f'data/diagnostics/cir_parameter_transfer_v1/{key}.parquet',mapping) for key in ['sid_name','heldout_cir']}
    result={'status':'accepted','arm':arm,'backend':job['backend_id'],'composition_state_sha256':composition['state_sha256'],'full_dev':full,'diagnostics':diagnostics,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*.json')},'large_payloads':'remote_only'}
    dest=base/'acceptance'/f'{arm}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    return {'status':'accepted','arm':arm,'metrics':full['metrics'],'receipt_sha256':sha(dest)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--arm',choices=['keep_lora','keep_sid'],required=True);a=p.parse_args();print(json.dumps(accept(a.manifest,a.arm),indent=2))
