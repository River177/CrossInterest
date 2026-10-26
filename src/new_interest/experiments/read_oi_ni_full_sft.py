"""Independent CPU readback for a complete full-vocabulary Mixed/Meta attempt."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.experiments.read_oi_ni_meta import (
    _canonical_name, _inside, _require, inspect_evaluation, sha256, verify_directory,
)


def read_run(mode_root:Path,data:Path):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    pointer=json.loads((mode_root/'completed_attempt.json').read_text())
    root=_inside(mode_root,'attempts/'+pointer['attempt'])
    _require(sha256(root/'manifest.json')==pointer['manifest_sha256'],'Completed attempt pointer hash')
    identity=pointer['identity'];verify_directory(root,identity)
    resolved=json.loads((root/'resolved_config.json').read_text())
    config=resolved['config'];result=json.loads((root/'success.json').read_text())
    _require(result['identity']==resolved['identity']==identity,'Run identity')
    _require(config['modes']==['mixed','meta'] and resolved['mode'] in config['modes'],'Only two modes')
    _require(config['semantic_id_loss_scope']==result['loss_scope']=='full_vocabulary','Full vocabulary loss')
    _require(result['steps']==config['steps']==3630 and result['epochs']==3 and result['world_size']==4,'Full horizon')
    source=sha256(root/'source_manifest.json')
    expected=hashlib.sha256(json.dumps({'source':source,'config':config,'mode':resolved['mode']},
                                      sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    _require(identity==expected and source==resolved['source_sha256'],'Bound source identity')
    _require(sha256(data/'metadata.json')==config['data_metadata_sha256'],'Bound full data')
    metadata=json.loads((data/'metadata.json').read_text())
    for name,digest in metadata['files'].items():_require(sha256(data/name)==digest,'Frozen input '+name)
    records=pq.read_table(data/'mixed.parquet').to_pylist()
    plan=pq.read_table(data/'episodes.parquet').to_pylist()
    training=[json.loads(x) for x in (root/'training.jsonl').read_text().splitlines() if x.strip()]
    _require(len(training)==3630,'Actual training log horizon')
    counts=Counter()
    for epoch in range(3):
        ids=[i for x in training if x['epoch']==epoch for i in x['row_indices']]
        _require(sorted(ids)==list(range(38698)),'Each row once per epoch')
    for index,(row,batch) in enumerate(zip(training,plan)):
        _require(row['step']==index+1 and row['row_indices']==batch['row_indices'],'Actual batch order')
        _require(math.isfinite(row['loss']) and row['seconds']>0,'Finite loss and measured time')
        n=Counter(records[i]['training_task'] for i in row['row_indices']);counts.update(n)
        _require(row['oi_samples']==n['old_interest'] and row['ni_samples']==n['new_interest'],'Role exposure')
        tf,tg=row['oi_supervised_tokens'],row['ni_supervised_tokens'];p=tf/(tf+tg)
        expected_loss=p*row['oi_loss']+(1-p)/2*(row['ni_loss']+row['ni_fast_loss'])
        _require(math.isclose(row['loss'],expected_loss,rel_tol=3e-6,abs_tol=3e-6),'Pooled objective value')
        _require(row['alpha']==(config['inner_alpha'] if resolved['mode']=='meta' else 0.),'Virtual step alpha')
        warmup=math.ceil(3630*.03)
        mult=index/warmup if index<warmup else .5*(1+math.cos(math.pi*(index-warmup)/(3630-warmup)))
        _require(math.isclose(row['lr'],config['outer_learning_rate']*mult,rel_tol=1e-12,abs_tol=1e-18),'Applied learning rate')
    _require(counts=={'old_interest':51294,'new_interest':64800},'Full role presentations')
    cp=_inside(mode_root,result['checkpoint_relative_to_mode']);verify_directory(cp,identity)
    state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False)
    _require(state['identity']==identity and state['step']==3630 and state['world_size']==4,'Checkpoint identity')
    _require(state['scheduler']['last_epoch']==3630 and len(state['rng_by_rank'])==4,'Optimizer/RNG cursor')
    shapes=json.loads((root/'model_audit.json').read_text())['trainable_shapes']
    names=state['parameter_names'];_require(names==list(shapes) and len(names)==373,'Parameter identity')
    tensors=load_file(str(cp/'adapter'/'adapter_model.safetensors'))
    saved={_canonical_name(n):v for n,v in tensors.items()}
    _require(set(saved)=={_canonical_name(n) for n in names},'Serialized parameter names')
    param_ids=[i for g in state['optimizer']['param_groups'] for i in g['params']]
    _require(len(param_ids)==len(names) and set(param_ids)==set(state['optimizer']['state']),'Adam state coverage')
    digest=hashlib.sha256()
    restored={}
    for name,key in zip(names,param_ids):
        value=saved[_canonical_name(name)];restored[name]=value
        _require(list(value.shape)==shapes[name] and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Saved tensor')
        moment=state['optimizer']['state'][key]
        _require(int(moment['step'])==3630,'Every Adam cursor')
        for field in ('exp_avg','exp_avg_sq'):
            _require(moment[field].shape==value.shape and bool(torch.isfinite(moment[field]).all()),'Adam moment')
    for name in sorted(restored):
        value=restored[name].contiguous()
        digest.update(name.encode());digest.update(str((value.shape,value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    _require(digest.hexdigest()==result['final_tensor_sha256'],'Final tensor bytes digest')
    for event in ('deployment','fresh_reload'):
        audit=json.loads((root/'execution_audits'/f'{event}-003630.json').read_text())
        _require(audit['accepted'] is True and len(audit['ranks'])==4,'Four-rank deployed/reload evidence')
        _require(all(r['tensor_sha256']==digest.hexdigest() for r in audit['ranks']),'Every rank same saved tensor')
        if event=='fresh_reload':_require(all(r['reload_max_abs_error']==0 for r in audit['ranks']),'GPU fresh reload')
    tokenizer=AutoTokenizer.from_pretrained(cp/'adapter',local_files_only=True)
    mapping=pq.read_table(data/'interest_token_mapping.parquet').to_pylist()
    labels=[r['target_subcategory'] for r in mapping]
    ids=set(tokenizer.convert_tokens_to_ids(r['interest_token']) for r in mapping)
    dev=inspect_evaluation(root/'dev',pq.read_table(data/'dev_full_grouped.parquet').to_pylist(),
        labels,tokenizer,ids,2048,identity,(31392,37728,104))
    return {'accepted':True,'identity':identity,'mode':resolved['mode'],'steps':3630,'epochs':3,
            'presentations':dict(counts),'tensor_sha256':digest.hexdigest(),'dev':dev,
            'manifest_sha256':sha256(root/'manifest.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode-root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=read_run(a.mode_root,a.data_dir)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
