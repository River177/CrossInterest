"""Isolated SID-relation pilot: fixed-order native SFT, durable state, and full scores."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_sid_relation_sidecar import sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def score_table(records):
    table=pa.Table.from_pylist(records)
    return table.set_column(table.schema.get_field_index('scores'),'scores',
        pa.array([r['scores'] for r in records],type=pa.list_(pa.float32())))


def tensor_manifest(state):
    result = {}
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous()
        result[k]={'shape':list(a.shape),'dtype':str(a.dtype),
                   'sha256':hashlib.sha256(a.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()}
    return result


def model_manifest(model):
    from peft import get_peft_model_state_dict
    return {'adapter':tensor_manifest(get_peft_model_state_dict(model.causal_lm,save_embedding_layers=False)),
            'relation':tensor_manifest(model.relation.state_dict())}


def seal(folder, identity):
    files = {str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(folder/'manifest.json', {'identity':identity,'files':files})


def verify(folder, identity=None):
    folder=Path(folder);m=read(folder/'manifest.json')
    if identity is not None and m['identity']!=identity:raise ValueError('Artifact identity differs')
    for name,rec in m['files'].items():
        p=folder/name
        if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:
            raise ValueError(f'Artifact bytes differ: {name}')
    return m


def publish(source, target, identity):
    """Each attempt has its own path; manifest is the final commit marker."""
    if (target/'manifest.json').exists():
        verify(target,identity)
        if read(source/'manifest.json')!=read(target/'manifest.json'):
            raise ValueError('Refusing to replace committed artifact')
        return
    target.mkdir(parents=True,exist_ok=True)
    m=verify(source,identity)
    for name,rec in m['files'].items():
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with (source/name).open('rb') as f, dest.open('wb') as out:shutil.copyfileobj(f,out,4*1024*1024)
        if sha(dest)!=rec['sha256']:raise ValueError('Publication hash mismatch')
    write(target/'manifest.json',m)


def load_tokenizer(project,model_path):
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    # Registration is deliberately strict: start from the original tokenizer,
    # then verify against the already-extended tokenizer used to freeze data.
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    spec=register_semantic_interest_tokens(tok,project/'inputs/interest_token_mapping.parquet')
    frozen=AutoTokenizer.from_pretrained(project/'inputs/tokenizer',local_files_only=True)
    if tok.get_vocab()!=frozen.get_vocab() or tok.chat_template!=frozen.chat_template:
        raise ValueError('Registered tokenizer differs from frozen input tokenizer')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if tok.pad_token_id!=frozen.pad_token_id:raise ValueError('Padding token differs from frozen data')
    return tok,spec


def load_model(project, model_path, device, condition, snapshot=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens,prepare_semantic_adapter_base,_semantic_row
    from new_interest.training.sid_relation_attention import SIDRelationModel,text_model
    cfg=read(project/'protocol.json')
    tok,spec=load_tokenizer(project,model_path)
    model,loading=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True,output_loading_info=True)
    missing=list(loading.get('missing_keys',[]))
    if getattr(model.config,'tie_word_embeddings',False) and model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr():
        missing=[k for k in missing if k!='lm_head.weight']
    unexpected=[k for k in loading.get('unexpected_keys',[]) if not k.startswith(('model.visual.','visual.','mtp.'))]
    if missing or unexpected or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'Pretrained text loading mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}')
    prepare_semantic_adapter_base(model,spec)
    adapter=project/'inputs/mixed_adapter' if snapshot is None else Path(snapshot)/'adapter'
    model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    base=text_model(model)
    base.config.use_cache=False
    if base.config.hidden_size!=2048 or not base.config.tie_word_embeddings:raise ValueError('Unexpected backbone')
    sources=read(project/'inputs/source_categories.json')
    fallback=torch.zeros(len(sources),2048)
    emb=model.get_input_embeddings().token_adapter.get_base_layer().weight
    for i,row in enumerate(sources):
        if row['sid_index']<0:
            ids=tuple(tok(row['name'],add_special_tokens=False)['input_ids'])
            fallback[i]=_semantic_row(emb,ids).detach().float().cpu()
    wrapped=SIDRelationModel(model,spec.token_ids,[x['sid_index'] for x in sources],fallback,
        enabled=condition=='relation',layer_idx=cfg['relation']['layer_idx'],rank=cfg['relation']['rank'])
    if snapshot is not None:
        if not (Path(snapshot)/'relation.safetensors').exists():raise ValueError('Missing relation weights')
        meta=read(Path(snapshot)/'model_state.json')
        if meta['condition']!=condition:raise ValueError('Checkpoint condition differs')
        wrapped.relation.load_state_dict(load_file(str(Path(snapshot)/'relation.safetensors')),strict=True)
    wrapped.to(device)
    wrapped.loading_audit={'missing_text_keys':missing,'unexpected_text_keys':unexpected,
                           'pretrained_text_loaded':True}
    wrapped.causal_lm.enable_input_require_grads()
    if snapshot is not None and model_manifest(wrapped)!=meta['tensors']:
        raise ValueError('Fresh model tensor mismatch')
    return wrapped,tok


def batch(rows,pad,device,prefix=False,padding_side='right'):
    import torch
    fields=['input_ids','attention_mask','event_ids','source_ids']+([] if prefix else ['labels'])
    n=max(r['prefix_length'] if prefix else len(r['input_ids']) for r in rows)
    result={}
    for key in fields:
        out=[]
        fill={'input_ids':pad,'attention_mask':0,'event_ids':-1,'source_ids':-1,'labels':-100}[key]
        for row in rows:
            values=row[key][:row['prefix_length']] if prefix else row[key]
            pads=[fill]*(n-len(values));out.append(values+pads if padding_side=='right' else pads+values)
        result[key]=torch.tensor(out,dtype=torch.long,device=device)
    return result


def save_model(model,path,condition):
    from safetensors.torch import save_file
    path.mkdir(parents=True,exist_ok=True)
    model.causal_lm.save_pretrained(path/'adapter',save_embedding_layers=False)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.relation.state_dict().items()},
              str(path/'relation.safetensors'))
    write(path/'model_state.json',{'condition':condition,'tensors':model_manifest(model)})


def probe_scores(model,rows,pad,device):
    import torch
    was=model.training;model.eval()
    with torch.no_grad():result=torch.cat([model.score(**batch([r],pad,device,True)).cpu() for r in rows])
    model.train(was)
    return result


def loss_probe(model,rows,pad,device):
    import torch
    was=model.training;model.eval()
    with torch.no_grad():value=sum(float(model(**batch([r],pad,device)).loss) for r in rows)/len(rows)
    model.train(was)
    return value


def native_parity(model, row, pad, device):
    """Compare native loss/logits/all existing gradients before any nonzero bias."""
    import torch
    from new_interest.training.sid_relation_attention import text_model
    model.eval();b=batch([row],pad,device)
    selected=text_model(model.causal_lm).model.layers[model.layer_idx].self_attn
    patched=selected.forward;selected.forward=selected._sid_relation_original_forward
    model.zero_grad(set_to_none=True)
    before=model(**b);lv=float(before.loss);logit=before.logits[:,row['prefix_length']-1].detach().cpu()
    before.loss.backward();del before
    grads={k:p.grad.detach().cpu().clone() for k,p in model.causal_lm.named_parameters() if p.requires_grad and p.grad is not None}
    selected.forward=patched;model.zero_grad(set_to_none=True)
    after=model(**b);loss_error=abs(float(after.loss)-lv)
    logits_error=float((after.logits[:,row['prefix_length']-1].detach().cpu()-logit).abs().max())
    after.loss.backward();del after
    gmax=0.0
    for k,p in model.causal_lm.named_parameters():
        if k in grads:
            if p.grad is None:raise ValueError('Existing gradient disappeared')
            gmax=max(gmax,float((p.grad.detach().cpu()-grads[k]).abs().max()))
    if loss_error!=0 or logits_error!=0 or gmax!=0:
        raise ValueError(f'Zero-bias native parity failed: {loss_error},{logits_error},{gmax}')
    model.zero_grad(set_to_none=True)
    sid_only=dict(b);sid_only['labels']=torch.full_like(b['labels'],-100)
    sid_only['labels'][:,row['prefix_length']]=b['labels'][:,row['prefix_length']]
    model(**sid_only).loss.backward()
    vnorm=float(model.relation.V.grad.float().norm());unorm=float(model.relation.U.grad.float().norm())
    if not math.isfinite(vnorm) or vnorm<=0 or unorm!=0:raise ValueError('First SID cannot train relation initialization')
    model.zero_grad(set_to_none=True)
    report={'native_loss_error':loss_error,'native_logit_error':logits_error,
        'native_existing_gradient_max_error':gmax,'existing_gradient_tensors':len(grads),
        'sid_only_initial_v_gradient_norm':vnorm,'initial_u_gradient_norm_expected_zero':unorm}
    print('SID_RELATION_NATIVE_PARITY '+json.dumps(report),flush=True)
    return report


def latest_checkpoint(root,identity):
    found=[]
    for p in root.glob('attempts/*/checkpoints/step-*/manifest.json'):
        m=read(p)
        if m['identity']==identity:
            state=read(p.parent/'cursor.json');found.append((state['step'],p.parent))
    if not found:return None
    step=max(x[0] for x in found);same=[p for s,p in found if s==step]
    models={json.dumps(read(p/'model_state.json')['tensors'],sort_keys=True) for p in same}
    if len(models)!=1:raise ValueError('Conflicting committed attempts at same training step')
    chosen=sorted(same)[-1];verify(chosen,identity);return chosen


def synchronized_publish(action,local_marker,rank,world):
    """No live collective spans the rank-zero network-storage operation."""
    import torch.distributed as dist
    if world>1:dist.barrier()
    if rank==0:
        action();write(local_marker,{'done':True})
    else:
        started=time.monotonic()
        while not local_marker.exists():
            if time.monotonic()-started>1800:raise TimeoutError('Checkpoint publication exceeded30min')
            time.sleep(1)
    if world>1:dist.barrier()


def train(project,root,model_path,condition,phase):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import get_cosine_schedule_with_warmup
    cfg=read(project/'protocol.json');rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local_rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local_rank);device=f'cuda:{local_rank}'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world>1:dist.init_process_group('nccl')
    if (phase=='single' and world!=1) or (phase!='single' and world!=4):raise ValueError('Unexpected world size')
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    data=pq.read_table(project/'inputs/train.parquet').to_pylist()
    permutation=list(range(len(data)));random.Random(42).shuffle(permutation)
    if phase!='full':
        # Diverse, deterministic gate subset. Formal training always uses full permutation.
        per_class={}
        for i,r in enumerate(data):per_class.setdefault(r['targets'][0],i)
        subset=[per_class[c] for c in sorted(per_class)]
        permutation=[subset[i%len(subset)] for i in range(128)]
    steps=675 if phase=='full' else (20 if phase=='single' else 4)
    global_batch=32 if world==4 else 4;accum=global_batch//world
    if phase=='single':permutation=[permutation[i%32] for i in range(steps*global_batch)]
    order_sha=hashlib.sha256(json.dumps(permutation).encode()).hexdigest()
    identity={'protocol_sha256':sha(project/'protocol.json'),'runtime_sha256':read(project/'source_manifest.json')['runtime_sha256'],
        'data_sha256':sha(project/'inputs/train.parquet'),'condition':condition,'phase':phase,
        'steps':steps,'world_size':world,'global_batch':global_batch,'order_sha256':order_sha}
    parent=latest_checkpoint(root,identity) if rank==0 else None
    shared=[str(parent) if parent else None]
    if world>1:dist.broadcast_object_list(shared,src=0)
    parent=Path(shared[0]) if shared[0] else None
    model,tok=load_model(project,model_path,device,condition,parent)
    initial_tensors=model_manifest(model)
    from safetensors.torch import load_file
    expected=tensor_manifest(load_file(str(project/'inputs/mixed_adapter/adapter_model.safetensors')))
    if parent is None:
        if initial_tensors['adapter']!=expected:raise ValueError('Initial adapter differs from frozen Mixed')
    audit={}
    if phase=='single' and parent is None:
        chosen=min((r for r in data if len(set(r['history_subcategories']))>=2),key=lambda r:len(r['input_ids']))
        audit['native_parity']=native_parity(model,chosen,tok.pad_token_id,device)
    relation_ids={id(p) for p in model.relation.parameters()}
    shared_params=[p for p in model.parameters() if p.requires_grad and id(p) not in relation_ids]
    groups=[{'params':shared_params,'lr':1e-5}]
    if condition=='relation':groups.append({'params':[p for p in model.relation.parameters() if p.requires_grad],'lr':1e-4})
    optimizer=torch.optim.AdamW(groups,weight_decay=0.01,fused=True)
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=21 if phase=='full' else 1,num_training_steps=steps)
    probe=[data[i] for i in permutation[:4]]
    before=None if parent else loss_probe(model,probe,tok.pad_token_id,device)
    start=0;history=[]
    if parent:
        state=torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['step'];history=read(parent/'training.json')['steps'];before=state['probe_loss_before']
        audit=state.get('audit',{})
        if read(parent/'cursor.json')['consumed_indices']!=permutation[:start*global_batch]:raise ValueError('Resume prefix differs')
        rng=state['rng'][rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device)
        print(f'SID_RELATION_RESUME step={start} parent={parent}',flush=True)
    model.causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=False) if world>1 else model
    attempt=[uuid.uuid4().hex if rank==0 else None]
    if world>1:dist.broadcast_object_list(attempt,src=0)
    local=Path('/scratch')/f'sid-relation-{condition}-{phase}-{attempt[0]}'
    local.mkdir(parents=True,exist_ok=True);persistent=root/'attempts'/attempt[0]
    sid_rows_initial=None
    last_checkpoint=None
    for step in range(start,steps):
        began=time.monotonic();torch.manual_seed(42+104729*step+8191*rank)
        optimizer.zero_grad(set_to_none=True);loss_sum=0.0
        for micro in range(accum):
            index=permutation[step*global_batch+micro*world+rank]
            scope=wrapped.no_sync() if world>1 and micro<accum-1 else nullcontext()
            with scope:
                loss=wrapped(**batch([data[index]],tok.pad_token_id,device)).loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss/accum).backward();loss_sum+=float(loss.detach())/accum
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
        if condition=='relation':
            for p in [model.relation.U,model.relation.V]:
                if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite relation gradient')
        rates=[g['lr'] for g in optimizer.param_groups]
        optimizer.step();scheduler.step()
        scalar=torch.tensor(loss_sum,device=device)
        if world>1:dist.all_reduce(scalar);scalar/=world
        record={'step':step+1,'loss':float(scalar),'gradient_norm':float(norm),'learning_rates':rates,
            'elapsed_seconds':time.monotonic()-began,'consumed_rows':(step+1)*global_batch}
        if condition=='relation':record.update(relation_u_gradient_norm=float(model.relation.U.grad.norm()),
            relation_v_gradient_norm=float(model.relation.V.grad.norm()))
        history.append(record)
        if rank==0:
            with (local/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('SID_RELATION_TRAIN '+json.dumps(record),flush=True)
        if (step+1)%100==0 or step+1==steps or (phase!='full' and step+1==2):
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)}
            rngs=[None]*world
            if world>1:dist.all_gather_object(rngs,rng)
            else:rngs=[rng]
            checkpoint=local/'checkpoints'/f'step-{step+1:06d}'
            destination=persistent/'checkpoints'/checkpoint.name
            def save():
                save_model(model,checkpoint,condition)
                torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'step':step+1,'rng':rngs,'probe_loss_before':before,'audit':audit},checkpoint/'training_state.pt')
                write(checkpoint/'training.json',{'steps':history})
                write(checkpoint/'cursor.json',{'step':step+1,'consumed_indices':permutation[:(step+1)*global_batch],
                    'parent':str(parent) if parent else None,'order_sha256':order_sha})
                seal(checkpoint,identity);publish(checkpoint,destination,identity)
                print(f'SID_RELATION_CHECKPOINT_COMMITTED step={step+1} attempt={attempt[0]}',flush=True)
            synchronized_publish(save,local/f'published-{step+1}.json',rank,world)
            last_checkpoint=destination
    if last_checkpoint is None:last_checkpoint=parent
    model.causal_lm.gradient_checkpointing_disable();model.eval()
    after=loss_probe(model,probe,tok.pad_token_id,device)
    scores_before_reload=probe_scores(model,probe,tok.pad_token_id,device)
    old_state=model_manifest(model)
    fresh,fresh_tok=load_model(project,model_path,device,condition,last_checkpoint);fresh.eval()
    scores_after_reload=probe_scores(fresh,probe,fresh_tok.pad_token_id,device)
    if model_manifest(fresh)!=old_state or not torch.equal(scores_before_reload,scores_after_reload):
        raise ValueError('Fresh reload differs in tensors or canonical SID logits')
    if phase=='single' and after>=before:raise ValueError('Fixed training probe did not improve')
    reload_audit={'rank':rank,'tensor_exact':True,'sid_scores_exact':True,'max_error':0.0}
    ranks=[None]*world
    if world>1:dist.all_gather_object(ranks,reload_audit)
    else:ranks=[reload_audit]
    def finalize():
        final=local/'final';save_model(fresh,final,condition)
        write(final/'training.json',{'steps':history})
        write(final/'acceptance.json',{'identity':identity,'steps':steps,'rows':steps*global_batch,
            'formal_unique_samples':len(set(permutation)),'initial_adapter_tensors':expected,
            'probe_loss_before':before,'probe_loss_after':after,'reload':ranks,
            'pretrained_loading':model.loading_audit,**audit,
            'checkpoint':str(last_checkpoint),'fresh_optimizer':parent is None,'resume_parent':str(parent) if parent else None})
        write(final/'reload_reference.json',{'source_ids':[r['source_sample_id'] for r in probe],
            'scores':scores_after_reload.tolist()})
        seal(final,identity);publish(final,persistent/'final',identity)
        write(root/'final_pointer.json',{'path':str((persistent/'final').relative_to(root)),
            'manifest_sha256':sha(persistent/'final/manifest.json'),'identity':identity})
        print('SID_RELATION_TRAIN_ACCEPTED '+json.dumps({'phase':phase,'condition':condition,'steps':steps}),flush=True)
    synchronized_publish(finalize,local/'final-published.json',rank,world)
    if world>1:dist.destroy_process_group()


def evaluate(project,root,model_path,condition,snapshot,split,shard,shards,limit=None):
    import torch
    torch.cuda.set_device(0);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tok=load_model(project,model_path,device,condition,snapshot);model.eval()
    rows=pq.read_table(project/f'inputs/{split}.parquet').to_pylist()
    if limit is not None:rows=rows[:limit]
    identity={'input_sha256':sha(project/f'inputs/{split}.parquet'),'condition':condition,'split':split,
        'snapshot_manifest_sha256':sha(Path(snapshot)/'manifest.json') if snapshot else sha(project/'inputs/mixed_adapter/adapter_model.safetensors'),
        'shard':shard,'shards':shards,'limit':limit,'protocol_sha256':sha(project/'protocol.json')}
    root.mkdir(parents=True,exist_ok=True)
    if (root/'identity.json').exists() and read(root/'identity.json')!=identity:raise ValueError('Evaluation identity differs')
    write(root/'identity.json',identity)
    indices=list(range(shard,len(rows),shards))
    for start in range(0,len(indices),64):
        selected=indices[start:start+64];out=root/f'part-{start//64:05d}.parquet';marker=out.with_suffix('.json')
        if marker.exists():
            receipt=read(marker)
            if receipt['sha256']!=sha(out) or receipt['indices']!=selected:raise ValueError('Evaluation checkpoint mismatch')
            continue
        records=[]
        for offset in range(0,len(selected),4):
            idx=selected[offset:offset+4];items=[rows[i] for i in idx]
            with torch.no_grad():scores=model.score(**batch(items,tok.pad_token_id,device,True)).float().cpu()
            if not torch.isfinite(scores).all():raise ValueError('Nonfinite ranking scores')
            order=torch.argsort(scores,dim=-1,descending=True,stable=True)
            for i,s,o in zip(idx,scores.tolist(),order.tolist()):
                r=rows[i];records.append({'record_index':i,'user_id':r['user_id'],'source_sample_id':r['source_sample_id'],
                    'impression_id':r['impression_id'],'input_sha256':r['input_sha256'],'targets':r['targets'],
                    'scores':s,'ranking':o,'history_subcategories':r['history_subcategories']})
        local=Path('/scratch')/f'sid-rel-eval-{uuid.uuid4().hex}.parquet'
        pq.write_table(score_table(records),local,compression='zstd')
        shutil.copyfile(local,out);write(marker,{'sha256':sha(out),'indices':selected});local.unlink()
        print(f'SID_RELATION_EVAL {split} shard={shard} {start+len(selected)}/{len(indices)}',flush=True)
    write(root/'complete.json',{'identity':identity,'rows':len(indices)})


def metrics(records,names,min_support=1):
    support=Counter(c for r in records for c in r['targets']);result={'records':len(records),
        'users':len({r['user_id'] for r in records}),'target_occurrences':sum(support.values())}
    per_class={}
    for c,n in sorted(support.items()):
        vals=[];index=names.index(c)
        for r in records:
            if c in r['targets']:vals.append(r['ranking'].index(index)+1)
        per_class[c]={'support':n}
        for k in [10,20]:
            per_class[c][f'r{k}']=sum(v<=k for v in vals)/n
            per_class[c][f'n{k}']=sum(1/math.log2(v+1) for v in vals if v<=k)/n
    selected=[v for v in per_class.values() if v['support']>=min_support]
    for k in [10,20]:
        result[f'macro_recall_at_{k}']=sum(v[f'r{k}'] for v in selected)/len(selected)
        result[f'macro_ndcg_at_{k}']=sum(v[f'n{k}'] for v in selected)/len(selected)
    result['supported_classes']=len(selected);result['per_class']=per_class
    return result


def merge(project,roots,output,split,limit=None):
    import numpy as np
    original=pq.read_table(project/f'inputs/{split}.parquet').to_pylist()
    if limit is not None:original=original[:limit]
    names=[r['target_subcategory'] for r in sorted(pq.read_table(project/'inputs/interest_token_mapping.parquet').to_pylist(),key=lambda r:r['interest_index'])]
    records=[]
    for root in roots:
        done=read(root/'complete.json');parts=sorted(root.glob('part-*.parquet'))
        shard_rows=[]
        for p in parts:
            receipt=read(p.with_suffix('.json'))
            if sha(p)!=receipt['sha256']:raise ValueError('Shard hash mismatch')
            chunk=pq.read_table(p).to_pylist()
            if [r['record_index'] for r in chunk]!=receipt['indices']:raise ValueError('Shard index receipt differs')
            shard_rows+=chunk
        if len(shard_rows)!=done['rows']:raise ValueError('Shard coverage mismatch')
        records+=shard_rows
    records.sort(key=lambda r:r['record_index'])
    if [r['record_index'] for r in records]!=list(range(len(original))):raise ValueError('Missing/duplicate evaluation records')
    for r,o in zip(records,original):
        if any(r[k]!=o[k] for k in ['user_id','source_sample_id','impression_id','targets','input_sha256']):raise ValueError('Evaluation source mismatch')
        scores=np.asarray(r['scores'],dtype=np.float32)
        if len(scores)!=108 or not np.isfinite(scores).all() or np.argsort(-scores,kind='stable').tolist()!=r['ranking']:
            raise ValueError('Ranking differs from saved108 scores')
    output.mkdir(parents=True,exist_ok=True)
    pq.write_table(score_table(records),output/'predictions.parquet',compression='zstd')
    report=metrics(records,names)
    if split=='validation':report['selection_107']=metrics(records,names,20)
    write(output/'metrics.json',report)
    write(output/'merge_manifest.json',{'rows':len(records),'prediction_sha256':sha(output/'predictions.parquet'),
        'metrics_sha256':sha(output/'metrics.json'),'source_sha256':sha(project/f'inputs/{split}.parquet'),
        'shards':[{'path':str(r),'complete_sha256':sha(r/'complete.json')} for r in roots]})
    return report


def run_command(args,project,log,env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen(args,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            f.write(line);f.flush()
            print(line.rstrip(),flush=True)
        if p.wait():raise RuntimeError(f'Worker subprocess failed; see {log}')


def run_eval(project,root,model_path,condition,snapshot,split,limit=None):
    processes=[];handles=[];shards=[]
    for shard in range(4):
        folder=root/split/f'shard-{shard}';shards.append(folder)
        logfile=Path('/scratch')/f'sid-relation-eval-{uuid.uuid4().hex}.log';stream=logfile.open('w');handles.append(stream)
        args=[sys.executable,'-m','new_interest.experiments.sid_relation_bias','evaluate','--project',str(project),
            '--root',str(folder),'--model-path',str(model_path),'--condition',condition,'--split',split,'--shard',str(shard)]
        if snapshot:args+=['--snapshot',str(snapshot)]
        if limit is not None:args+=['--limit',str(limit)]
        processes.append((subprocess.Popen(args,cwd=project,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard)},stdout=stream,stderr=subprocess.STDOUT),logfile))
    while any(p.poll() is None for p,_ in processes):
        for p,log in processes:
            if p.poll() not in (None,0):
                print(log.read_text()[-6000:],flush=True)
                for other,_ in processes:
                    if other.poll() is None:other.terminate()
                raise RuntimeError('Evaluation child failed')
        print('SID_RELATION_EVAL_ACTIVE '+split,flush=True);time.sleep(30)
    for h in handles:h.close()
    if any(p.returncode for p,_ in processes):raise RuntimeError('Evaluation process failed')
    return merge(project,shards,root/split/'merged',split,limit)


def worker(project,root,phase,condition):
    import torch
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True,exist_ok=True)
    sources=read(project/'source_manifest.json')
    for name,expected in sources['files'].items():
        if sha(project/name)!=expected:raise ValueError(f'Source changed: {name}')
    if torch.cuda.device_count()!=4:raise ValueError('Expected4 GPUs')
    cfg=read(project/'protocol.json')
    if phase=='full':
        accepted=read(project/'gate_acceptance.json')
        if not accepted['passed'] or accepted['runtime_sha256']!=sources['runtime_sha256']:
            raise ValueError('Independent gate acceptance missing or stale')
    packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','tokenizers','safetensors']}
    required={'torch':'2.8.0+cu126','transformers':'5.14.1','peft':'0.18.0',
              'accelerate':'1.12.0','tokenizers':'0.22.2','safetensors':'0.8.0'}
    if packages!=required:raise ValueError(f'Worker package pins differ: {packages}')
    if any(torch.cuda.get_device_capability(i)!=(8,0) for i in range(4)):raise ValueError('Expected A100 GPUs')
    write(root/'environment.json',{'gpu_count':4,'gpus':[torch.cuda.get_device_name(i) for i in range(4)],'packages':packages})
    snapshot=Path(snapshot_download(cfg['model_name'],revision=cfg['model_revision'],local_files_only=True))
    expected=read(project/'source_snapshot.json')['files'];local_model=Path('/scratch/sid-relation-model')
    local_model.mkdir(exist_ok=True)
    for name,expected_sha in expected.items():
        source=snapshot/name;dest=local_model/name
        if not dest.exists() or sha(dest)!=expected_sha:
            if sha(source)!=expected_sha:raise ValueError(f'Cached model changed: {name}')
            shutil.copyfile(source,dest)
        if sha(dest)!=expected_sha:raise ValueError('Model copy hash mismatch')
    write(root/'snapshot_audit.json',{'files':expected,'revision':cfg['model_revision']})
    phases=['single','ddp'] if phase=='gate' else ['full']
    for stage in phases:
        folder=root/stage
        if not (folder/'final_pointer.json').exists():
            n=1 if stage=='single' else 4
            args=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={n}',
                '-m','new_interest.experiments.sid_relation_bias','train','--project',str(project),'--root',str(folder),
                '--model-path',str(local_model),'--condition',condition,'--phase',stage]
            run_command(args,project,Path('/scratch')/f'sid-relation-{stage}.log')
        pointer=read(folder/'final_pointer.json');final=folder/pointer['path']
        if sha(final/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Final pointer changed')
        verify(final,pointer['identity'])
    evaluations={}
    if phase=='full' and condition=='baseline':
        for split in ['validation','dev']:run_eval(project,root/'frozen_reference',local_model,'baseline',None,split)
    for split in (['dev'] if phase=='gate' else ['validation','dev']):
        evaluations[split]=run_eval(project,root/'evaluation',local_model,condition,final,split,64 if phase=='gate' else None)
    write(root/'worker_success.json',{'phase':phase,'condition':condition,'runtime_sha256':sources['runtime_sha256'],
        'protocol_sha256':sha(project/'protocol.json'),'final_relative_path':str(final.relative_to(root)),
        'final_manifest_sha256':sha(final/'manifest.json'),'evaluations':evaluations})
    print('SID_RELATION_WORKER_SUCCESS '+phase,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['worker','train','evaluate'])
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',default='gate');parser.add_argument('--condition',choices=['baseline','relation'],default='relation')
    parser.add_argument('--model-path',type=Path);parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--split',default='dev');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--limit',type=int)
    a=parser.parse_args()
    if a.action=='worker':worker(a.project,a.root,a.phase,a.condition)
    elif a.action=='train':train(a.project,a.root,a.model_path,a.condition,a.phase)
    else:evaluate(a.project,a.root,a.model_path,a.condition,a.snapshot,a.split,a.shard,4,a.limit)


if __name__=='__main__':main()
