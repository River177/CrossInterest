"""Independent downloaded-artifact reader; recomputes rankings and macro metrics."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_sid_relation_sidecar import sha


def read(path):return json.loads(Path(path).read_text())


def fingerprints(state):
    import torch
    return {name:{'shape':list(value.shape),'dtype':str(value.dtype),
        'sha256':hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name,value in sorted(state.items())}


def checked_files(root, expected_identity=None):
    m=read(root/'manifest.json')
    if expected_identity is not None and m['identity']!=expected_identity:raise ValueError('Foreign artifact identity')
    for name,record in m['files'].items():
        p=root/name
        if p.resolve().is_relative_to(root.resolve()) is False:raise ValueError('Unsafe manifest path')
        if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError(f'Hash mismatch {name}')
    return m


def recompute(project,folder,split,expected_rows):
    merged=folder/'merged';manifest=read(merged/'merge_manifest.json')
    if sha(merged/'predictions.parquet')!=manifest['prediction_sha256']:raise ValueError('Prediction hash mismatch')
    if sha(merged/'metrics.json')!=manifest['metrics_sha256']:raise ValueError('Metrics hash mismatch')
    original=pq.read_table(project/f'inputs/{split}.parquet').slice(0,expected_rows).to_pylist()
    table=pq.read_table(merged/'predictions.parquet')
    if table.schema.field('scores').type!=pa.list_(pa.float32()):raise ValueError('Scores not stored as FP32')
    records=table.to_pylist()
    if [r['record_index'] for r in records]!=list(range(expected_rows)):raise ValueError('Incomplete record coverage')
    shard_records=[]
    if len(manifest['shards'])!=4:raise ValueError('Expected exactly four evaluation shards')
    for i,entry in enumerate(manifest['shards']):
        shard=folder/f'shard-{i}'
        if Path(entry['path']).name!=shard.name or sha(shard/'complete.json')!=entry['complete_sha256']:
            raise ValueError('Shard completion binding differs')
        done=read(shard/'complete.json');identity=read(shard/'identity.json')
        if done['identity']!=identity or identity['shard']!=i or identity['shards']!=4:
            raise ValueError('Shard topology differs')
        parts=[]
        for f in sorted(shard.glob('part-*.parquet')):
            receipt=read(f.with_suffix('.json'))
            if sha(f)!=receipt['sha256']:raise ValueError('Shard payload corrupted')
            chunk=pq.read_table(f).to_pylist()
            if [r['record_index'] for r in chunk]!=receipt['indices'] or any(r['record_index']%4!=i for r in chunk):
                raise ValueError('Shard assignment differs')
            parts+=chunk
        if len(parts)!=done['rows']:raise ValueError('Shard row count differs')
        shard_records+=parts
    if sorted(shard_records,key=lambda r:r['record_index'])!=records:raise ValueError('Merged rows differ from source shards')
    mapping=sorted(pq.read_table(project/'inputs/interest_token_mapping.parquet').to_pylist(),key=lambda r:r['interest_index'])
    names=[r['target_subcategory'] for r in mapping];index={c:i for i,c in enumerate(names)}
    counts=Counter();sums=defaultdict(lambda:np.zeros(4))
    for actual,source in zip(records,original):
        for k in ['user_id','source_sample_id','impression_id','input_sha256','targets','history_subcategories']:
            if actual[k]!=source[k]:raise ValueError(f'Changed source binding: {k}')
        values=np.asarray(actual['scores'],dtype=np.float32)
        if values.shape!=(108,) or not np.isfinite(values).all():raise ValueError('Invalid score vector')
        ranks=sorted(range(108),key=lambda i:(-float(values[i]),i))
        if ranks!=actual['ranking']:raise ValueError('Wrong tie-break or ranking')
        positions={i:j+1 for j,i in enumerate(ranks)}
        for c in actual['targets']:
            counts[c]+=1;r=positions[index[c]]
            sums[c]+=np.asarray([r<=10,r<=20,(1/math.log2(r+1)) if r<=10 else 0,
                                (1/math.log2(r+1)) if r<=20 else 0])
    names_metrics=['macro_recall_at_10','macro_recall_at_20','macro_ndcg_at_10','macro_ndcg_at_20']
    expected=read(merged/'metrics.json');report={}
    for label,minimum in [('all',1)]+([('selection_107',20)] if split=='validation' else []):
        selected=[c for c in counts if counts[c]>=minimum]
        result=sum((sums[c]/counts[c] for c in selected),np.zeros(4))/len(selected)
        ref=expected if label=='all' else expected[label]
        for key,value in zip(names_metrics,result):
            if abs(float(value)-ref[key])>1e-12:raise ValueError(f'Metric mismatch {key}')
        report[label]={'classes':len(selected),**dict(zip(names_metrics,result.tolist()))}
    if sum(counts.values())!=expected['target_occurrences']:raise ValueError('Target denominator mismatch')
    return {'rows':len(records),'users':len({r['user_id'] for r in records}),
            'target_occurrences':sum(counts.values()),'metrics':report}


def inspect(project,root,output):
    from safetensors.torch import load_file
    import torch
    success=read(root/'worker_success.json');runtime=read(project/'source_manifest.json')['runtime_sha256']
    if success['runtime_sha256']!=runtime or success['protocol_sha256']!=sha(project/'protocol.json'):
        raise ValueError('Worker runtime/protocol differs')
    protocol=read(project/'protocol.json')
    original=load_file(str(project/'inputs/mixed_adapter/adapter_model.safetensors'))
    expected_original=fingerprints(original)
    snapshot=read(root/'snapshot_audit.json')
    if snapshot['revision']!=protocol['model_revision'] or snapshot['files']!=read(project/'source_snapshot.json')['files']:
        raise ValueError('Pretrained model snapshot identity differs')
    phase=success['phase'];condition=success['condition'];training={}
    for name,steps,world in ([('single',20,1),('ddp',4,4)] if phase=='gate' else [('full',675,4)]):
        pointer=read(root/name/'final_pointer.json');final=root/name/pointer['path']
        if sha(final/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Final pointer hash differs')
        m=checked_files(final,pointer['identity']);accept=read(final/'acceptance.json')
        if accept['identity']!=pointer['identity'] or accept['initial_adapter_tensors']!=expected_original:
            raise ValueError('Initial adapter/acceptance identity mismatch')
        loading=accept['pretrained_loading']
        if not loading['pretrained_text_loaded'] or loading['missing_text_keys'] or loading['unexpected_text_keys']:
            raise ValueError('Pretrained text tensors were not completely loaded')
        if accept['steps']!=steps or accept['identity']['world_size']!=world or accept['identity']['condition']!=condition:
            raise ValueError('Training budget/condition differs')
        history=read(final/'training.json')['steps']
        if [r['step'] for r in history]!=list(range(1,steps+1)):raise ValueError('Training step coverage differs')
        if any(not math.isfinite(r['loss']) or not math.isfinite(r['gradient_norm']) for r in history):
            raise ValueError('Nonfinite loss/gradient')
        if sorted(r['rank'] for r in accept['reload'])!=list(range(world)) or any(not r['tensor_exact'] or not r['sid_scores_exact'] for r in accept['reload']):
            raise ValueError('Incomplete rank reload validation')
        adapter=load_file(str(final/'adapter/adapter_model.safetensors'))
        if len(adapter)!=373 or any(not torch.isfinite(v).all() for v in adapter.values()):raise ValueError('Adapter tensor inventory invalid')
        sid=[v for k,v in adapter.items() if 'trainable_tokens_delta' in k]
        if len(sid)!=1 or tuple(sid[0].shape)!=(108,2048):raise ValueError('Wrong compact SID shape')
        relation=load_file(str(final/'relation.safetensors'))
        tensor_state=read(final/'model_state.json')
        if tensor_state['condition']!=condition or tensor_state['tensors']!={'adapter':fingerprints(adapter),'relation':fingerprints(relation)}:
            raise ValueError('Stored tensor payload differs from model-state manifest')
        if set(relation)!={'U','V','source_to_sid','fallback_rows'}:raise ValueError('Relation state incomplete')
        if tuple(relation['U'].shape)!=(2048,16) or tuple(relation['V'].shape)!=(2048,16):raise ValueError('Wrong projection shapes')
        if any(not torch.isfinite(x).all() for x in relation.values()):raise ValueError('Nonfinite relation state')
        if condition=='relation' and (relation['V'].norm()==0 or not any(r['relation_u_gradient_norm']>0 for r in history)):
            raise ValueError('Relation parameters did not learn')
        if name=='single':
            native=accept['native_parity']
            if any(native[k]!=0 for k in ['native_loss_error','native_logit_error','native_existing_gradient_max_error']):
                raise ValueError('Native zero-bias parity failed')
            if native['sid_only_initial_v_gradient_norm']<=0 or accept['probe_loss_after']>=accept['probe_loss_before']:
                raise ValueError('Learning/SID-only gradient gate failed')
        if name=='full' and (accept['rows']!=21600 or accept['formal_unique_samples']!=21600):
            raise ValueError('Formal sample exposure differs')
        training[name]={'steps':steps,'world_size':world,'adapter_sha256':sha(final/'adapter/adapter_model.safetensors'),
            'relation_sha256':sha(final/'relation.safetensors'),'reload_all_ranks_exact':True}
    evaluations={}
    for split,number in ([('dev',64)] if phase=='gate' else [('validation',5354),('dev',31392)]):
        evaluations[split]=recompute(project,root/'evaluation'/split,split,number)
    if phase=='full' and condition=='baseline':
        evaluations['frozen_reference']={s:recompute(project,root/'frozen_reference'/s,s,n)
            for s,n in [('validation',5354),('dev',31392)]}
    result={'passed':True,'phase':phase,'condition':condition,'runtime_sha256':runtime,
        'reader_source_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(project/'protocol.json'),'training':training,'evaluations':evaluations,
        'scope':'independent downloaded bytes, topology and score-derived metrics; GPU reload claims bound to checked receipts'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.project,a.root,a.output)


if __name__=='__main__':main()
