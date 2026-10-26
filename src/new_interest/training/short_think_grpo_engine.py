"""Native-HF GRPO correctness oracle: real four-GPU updates and exact resume.

The initial engine check uses two updates of the intended 32x8 global batch.
It is not a claim that an accelerated production engine is ready. Checkpoints
are written to NODE-LOCAL storage; the parent publishes only after torchrun exits.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import _atomic_write_json,_atomic_write_parquet
from new_interest.evaluation.evaluate_short_think_ranking import analyze_generation
from new_interest.evaluation.preflight_short_think_grpo import read,sha,stage
from new_interest.training.short_think_grpo_reward import score
from new_interest.training.short_think_grpo_objective import group_advantages,grpo_loss,nucleus_selected_logprobs,raw_forward_kl
from new_interest.training.sft_data import apply_chat_template


def permutation(count,seed):
    order=list(range(count));random.Random(seed).shuffle(order);return order


def train_indices(order,step,rank,world=4):
    if world!=4 or not 0<=rank<world:raise ValueError('Intended topology is four ranks')
    batch=order[step*32:(step+1)*32]
    if len(batch)!=32:raise ValueError('Missing full global prompt batch')
    return batch[rank*8:(rank+1)*8]


def reward_column(arm,step):
    if arm not in {'E','H','HL'}:raise ValueError('Unknown GRPO arm')
    return 'H' if arm=='HL' and step<100 else arm


def validate_resume_identity(saved,identity,contract,resume):
    if saved['identity']==identity:return
    parent=contract.get('allowed_parent',{})
    if (contract.get('arm')!='HL' or saved['step']!=100 or saved.get('arm')!='H' or parent.get('arm')!='H' or parent.get('identity')!=saved['identity']
        or parent.get('state_file_sha256')!=sha(Path(resume)/'state.json')):
        raise ValueError('Unapproved cross-arm checkpoint parent')


def pack_trajectories(tokenizer,rows):
    """Right-align responses in a left-padded full causal sequence."""
    import torch
    width=max(len(r['response_ids']) for r in rows)
    full=[r['prompt_ids']+r['response_ids'] for r in rows]
    batch=tokenizer.pad({'input_ids':full},padding=True,return_tensors='pt')
    mask=torch.zeros(len(rows),width,dtype=torch.bool)
    ban=torch.zeros_like(mask);only=torch.zeros_like(mask)
    old=torch.zeros(len(rows),width,dtype=torch.float)
    for i,row in enumerate(rows):
        response=row['response_ids'];offset=width-len(response)
        if not response or len(response)!=len(row['behavior_logprobs']):raise ValueError('Action/logprob mismatch')
        mask[i,offset:]=True;old[i,offset:]=torch.tensor(row['behavior_logprobs'])
        n=row['thought_length'];ban[i,offset:offset+n]=True
        answer=response[n:]
        for j in range(len(answer)):
            if re.fullmatch(r'\s*<answer>\s*',tokenizer.decode(answer[:j],skip_special_tokens=False)):
                only[i,offset+n+j]=True
    return batch,mask,old,ban,only


def trajectory_probabilities(current,reference,targets,ban,only,sid_ids):
    """Bound FP32 vocabulary intermediates; use the SAME phase masks as rollout."""
    import torch
    probabilities=[];kls=[]
    allowed=torch.zeros(current.shape[-1],device=current.device,dtype=torch.bool);allowed[sid_ids]=True
    for start in range(0,current.shape[1],16):
        raw=current[:,start:start+16];scores=raw.clone()
        scores[...,sid_ids]=scores[...,sid_ids].masked_fill(ban[:,start:start+16,None],float('-inf'))
        scores=scores.masked_fill(only[:,start:start+16,None]&~allowed,float('-inf'))
        probabilities.append(nucleus_selected_logprobs(scores,targets[:,start:start+16]))
        kls.append(raw_forward_kl(raw,reference[:,start:start+16]))
    return torch.cat(probabilities,1),torch.cat(kls,1)


def model_load(project,model_path,adapter,trainable):
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from new_interest.training.semantic_id_tokens import prepare_semantic_adapter_base,validate_saved_interest_tokens
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True);tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    spec=validate_saved_interest_tokens(tokenizer,project/'inputs/interest_token_mapping.parquet',expected_count=108)
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    prepare_semantic_adapter_base(model,spec)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=trainable).to(torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    expected=load_file(str(adapter/'adapter_model.safetensors'));actual=get_peft_model_state_dict(model,save_embedding_layers=False)
    if set(expected)!=set(actual) or len(actual)!=373:raise ValueError('Adapter inventory changed')
    for name,value in expected.items():
        if not torch.equal(value.to(actual[name].dtype),actual[name].cpu()):raise ValueError(f'Loaded parameter differs: {name}')
    # Policy likelihoods require deterministic dropout behavior in both modes.
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.requires_grad_(False)
    return model.eval(),tokenizer,spec


def rollouts(model,tokenizer,spec,records,indices,step,rank):
    model.eval();rows=[]
    close=tokenizer.encode('</think>',add_special_tokens=False)
    if len(close)!=1:raise ValueError('Think close must be one token')
    stops={int(tokenizer.eos_token_id),int(tokenizer.convert_tokens_to_ids('<|im_end|>'))}
    token_by_id=dict(zip(spec.token_ids,[e.interest_token for e in spec.entries]))
    for local,index in enumerate(indices):
        record=records[index];info=record['extra_info']
        prompt=apply_chat_template(tokenizer,record['prompt'],add_generation_prompt=True,enable_thinking=True)
        if not prompt.endswith('<think>\n'):raise ValueError('Wrong native prompt')
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        seed=42+(step*32+rank*8+local)*100
        thought,td=stage(model,tokenizer,[prefix]*8,stops|set(close),480,seed,False,
                         phase='thought',sid_ids=spec.token_ids,capture_behavior=True)
        closed=[i for i,(ids,ended) in enumerate(thought) if ended and ids[-1]==close[0]]
        answers={};answer_diag={}
        if closed:
            ar,ad=stage(model,tokenizer,[prefix+thought[i][0] for i in closed],stops,32,seed+50,False,
                        phase='answer',sid_ids=spec.token_ids,capture_behavior=True)
            answers=dict(zip(closed,ar));answer_diag=dict(zip(closed,ad))
        parents=dict(zip(info['interest_tokens'],info['interest_categories']))
        labels=dict(zip(info['interest_subcategories'],info['interest_tokens']))
        history=[labels[x] for x in info['history_subcategories'] if x in labels]
        group=[]
        for i,(thought_ids,ended) in enumerate(thought):
            answer,a_end=answers.get(i,([],False));response=thought_ids+answer
            parsed=analyze_generation(tokenizer,response,stop_ids=stops,token_by_id=token_by_id)
            exhausted=not ended or (i in answers and not a_end)
            valid=parsed['valid_format'] and not exhausted
            reward=score(valid=valid,predicted=parsed['generated_sid'],target=info['target_token'],history=history,
                         parent_by_token=parents,thought_tokens=parsed['think_tokens'])
            behavior=td[i]['cached_behavior_logprobs']+answer_diag.get(i,{}).get('cached_behavior_logprobs',[])
            group.append({'source_sample_id':info['source_sample_id'],'train_index':index,'generation_index':i,
                'prompt_ids':prefix,'response_ids':response,'thought_length':len(thought_ids),'behavior_logprobs':behavior,
                'valid':valid,'reward':reward,'step':step,'rank':rank})
        rows.extend(group)
    if len(rows)!=64:raise ValueError('Rank must sample exactly8 histories x8 trajectories')
    return rows


def batched_rollouts(model,tokenizer,spec,records,indices,step,rank):
    """All64 trajectories/rank in one generation batch; still8 per history.

    Batching changes the RNG draw layout. This versioned stream rule is frozen
    before any formal arm and is identical across E/H/HL, with no mid-run switch.
    """
    model.eval()
    if len(indices)!=8:raise ValueError('Rank must own eight histories')
    close=tokenizer.encode('</think>',add_special_tokens=False)
    if len(close)!=1:raise ValueError('Think close must be one token')
    stops={int(tokenizer.eos_token_id),int(tokenizer.convert_tokens_to_ids('<|im_end|>'))}
    token_by_id=dict(zip(spec.token_ids,[e.interest_token for e in spec.entries]))
    prefixes=[]
    for index in indices:
        prompt=apply_chat_template(tokenizer,records[index]['prompt'],add_generation_prompt=True,enable_thinking=True)
        if not prompt.endswith('<think>\n'):raise ValueError('Wrong native prompt')
        prefixes.extend([tokenizer.encode(prompt,add_special_tokens=False)]*8)
    seed=42+(step*4+rank)*100000
    thought,td=stage(model,tokenizer,prefixes,stops|set(close),480,seed,False,
        phase='thought',sid_ids=spec.token_ids,capture_behavior=True)
    closed=[i for i,(ids,ended) in enumerate(thought) if ended and ids[-1]==close[0]]
    answers={};answer_diag={}
    if closed:
        ar,ad=stage(model,tokenizer,[prefixes[i]+thought[i][0] for i in closed],stops,32,seed+50000,False,
            phase='answer',sid_ids=spec.token_ids,capture_behavior=True)
        answers=dict(zip(closed,ar));answer_diag=dict(zip(closed,ad))
    rows=[]
    for i,(thought_ids,ended) in enumerate(thought):
        index=indices[i//8];info=records[index]['extra_info']
        answer,a_end=answers.get(i,([],False));response=thought_ids+answer
        parsed=analyze_generation(tokenizer,response,stop_ids=stops,token_by_id=token_by_id)
        valid=parsed['valid_format'] and ended and (i not in answers or a_end)
        parents=dict(zip(info['interest_tokens'],info['interest_categories']))
        labels=dict(zip(info['interest_subcategories'],info['interest_tokens']))
        reward=score(valid=valid,predicted=parsed['generated_sid'],target=info['target_token'],
            history=[labels[x] for x in info['history_subcategories'] if x in labels],
            parent_by_token=parents,thought_tokens=parsed['think_tokens'])
        rows.append({'source_sample_id':info['source_sample_id'],'train_index':index,'generation_index':i%8,
            'prompt_ids':prefixes[i],'response_ids':response,'thought_length':len(thought_ids),
            'behavior_logprobs':td[i]['cached_behavior_logprobs']+answer_diag.get(i,{}).get('cached_behavior_logprobs',[]),
            'valid':valid,'reward':reward,'step':step,'rank':rank,'rollout_seed':seed,'answer_seed':seed+50000,
            'sampling_layout':'rank64-phase-batch-v1'})
    if len(rows)!=64:raise ValueError('Missing trajectories')
    return rows


def state_hash(model):
    from peft import get_peft_model_state_dict
    from new_interest.training.sft_state_audit import tensor_state_manifest
    return tensor_state_manifest(get_peft_model_state_dict(model,save_embedding_layers=False))['state_sha256']


def optimizer_hash(model,optimizer):
    from new_interest.training.sft_state_audit import tensor_state_manifest
    tensors={f'{name}/{key}':value for name,p in model.named_parameters() if p.requires_grad
             for key,value in optimizer.state.get(p,{}).items()}
    groups=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]
    tensor_hash=tensor_state_manifest(tensors)['state_sha256'] if tensors else None
    return hashlib.sha256(json.dumps({'tensors':tensor_hash,'groups':groups},sort_keys=True).encode()).hexdigest()


def probe(model,tokenizer,records,sid_ids,execution_plan=None):
    import torch
    from new_interest.training.probe_execution_plan import replay_plan
    result=[];model.eval()
    with replay_plan(execution_plan) if execution_plan is not None else nullcontext():
        for r in records[:4]:
            text=apply_chat_template(tokenizer,r['prompt'],add_generation_prompt=True,enable_thinking=True)
            batch=tokenizer(text,return_tensors='pt',add_special_tokens=False).to(model.device)
            with torch.inference_mode():values=model(**batch,use_cache=False,logits_to_keep=1).logits[0,-1,sid_ids]
            result.append(values.float().cpu().tolist())
    return result


def run(project,root,model_path,stop_at,resume=None):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    project=Path(project);root=Path(root);rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    if dist.get_world_size()!=4:raise ValueError('Expected four GPUs')
    root.mkdir(parents=True,exist_ok=True);contract=read(project/'engine_contract.json')
    identity=sha(project/'engine_contract.json');records=pq.read_table(project/'inputs/train.parquet').to_pylist()
    if len(records)!=21600 or not 0<stop_at<=675:raise ValueError('Wrong data or step budget')
    order=permutation(len(records),42);order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    initial=project/'inputs/short_sft_adapter';adapter=Path(resume)/'adapter' if resume else initial
    model,tokenizer,spec=model_load(project,model_path,adapter,True)
    reference,_,_=model_load(project,model_path,initial,False)
    reference_hash=state_hash(reference)
    if reference_hash!=contract['initial_adapter_tensor_state_sha256']:raise ValueError('Reference is not the accepted SFT')
    parameters=[p for p in model.parameters() if p.requires_grad]
    parameter_names=[name for name,p in model.named_parameters() if p.requires_grad]
    if len(parameters)!=373:raise ValueError('Wrong trainable parameter inventory')
    optimizer=torch.optim.AdamW(parameters,lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
    start=0
    if resume:
        saved=read(Path(resume)/'state.json')
        validate_resume_identity(saved,identity,contract,resume)
        if saved['order_sha256']!=order_hash:raise ValueError('Resume order differs')
        start=saved['step']
        if saved['consumed_prompts']!=start*32 or saved['adapter_state_sha256']!=state_hash(model):raise ValueError('Resume cursor/model differs')
        optimizer.load_state_dict(torch.load(Path(resume)/'optimizer.pt',map_location='cpu',weights_only=False))
        if saved['trainable_parameter_names']!=parameter_names or optimizer_hash(model,optimizer)!=saved['optimizer_state_sha256']:
            raise ValueError('Optimizer names/moments/hyperparameters differ after restore')
        if len(optimizer.state)!=373 or any(int(v['step'])!=start for v in optimizer.state.values()):raise ValueError('Optimizer state not restored')
        rng=torch.load(Path(resume)/f'rng-{rank}.pt',map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python'])
        if not torch.equal(torch.get_rng_state(),rng['torch']) or not torch.equal(torch.cuda.get_rng_state().cpu(),rng['cuda']) or random.getstate()!=rng['python']:
            raise ValueError('RNG state restoration differs')
        previous=torch.tensor(saved['reload_probe'])
        from new_interest.training.probe_execution_plan import checkpoint_plan
        plan=checkpoint_plan(project,Path(resume),saved)
        actual_probe=probe(model,tokenizer,records,spec.token_ids,execution_plan=plan)
        _atomic_write_json(root/f'resume-probe-{rank}.json',{'original_reference':saved['reload_probe'],'actual':actual_probe,
            'max_abs':float((torch.tensor(actual_probe)-previous).abs().max()),'execution_plan':plan})
        if not torch.equal(torch.tensor(actual_probe),previous):raise ValueError('Resume fresh logits differ')
    elif state_hash(model)!=contract['initial_adapter_tensor_state_sha256']:
        raise ValueError('Wrong initial policy')
    if start>=stop_at:raise ValueError('No new step requested')
    ddp=DDP(model,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False)
    _atomic_write_json(root/f'start-{rank}.json',{'identity':identity,'start_step':start,'stop_at':stop_at,
        'reference_sha256':reference_hash,'policy_sha256':state_hash(model),'optimizer_states':len(optimizer.state),
        'optimizer_state_sha256':optimizer_hash(model,optimizer),
        'dropout_disabled':True,'rank':rank,'world_size':4,'resume_probe_exact':bool(resume)})
    for step in range(start,stop_at):
        started=time.monotonic();indices=train_indices(order,step,rank)
        generator=batched_rollouts if contract.get('rollout_batch_size_per_gpu',8)==64 else rollouts
        rows=generator(model,tokenizer,spec,records,indices,step,rank)
        generation_seconds=time.monotonic()-started
        key=reward_column(contract.get('arm','H'),step)
        rewards=torch.tensor([[r['reward'][key] for r in rows[i:i+8]] for i in range(0,64,8)])
        advantages=group_advantages(rewards).reshape(-1).tolist()
        for row,adv in zip(rows,advantages):row['advantage']=adv
        _atomic_write_parquet(root/f'rollout-{step:04d}-rank-{rank}.parquet',rows)
        # Sort ONLY the update microbatches. The sampled group order and complete
        # exposure ledger remain immutable and are never success-filtered.
        ordered=sorted(rows,key=lambda r:len(r['prompt_ids'])+len(r['response_ids']))
        optimizer.zero_grad(set_to_none=True);model.train()
        stats=[]
        micro=int(contract.get('microbatch_per_gpu',4))
        if micro not in [4,8]:raise ValueError('Unapproved microbatch')
        for offset in range(0,64,micro):
            local=ordered[offset:offset+micro];batch,mask,old,ban,only=pack_trajectories(tokenizer,local)
            batch={k:v.to(model.device) for k,v in batch.items()}
            mask,old,ban,only=[v.to(model.device) for v in (mask,old,ban,only)]
            length=mask.shape[1];targets=batch['input_ids'][:,-length:]
            with torch.no_grad():ref=reference(**batch,use_cache=False,logits_to_keep=length+1).logits[:,:-1]
            with ddp.no_sync() if offset+len(local)<64 else nullcontext():
                logits=ddp(**batch,use_cache=False,logits_to_keep=length+1).logits[:,:-1]
                current,kl=trajectory_probabilities(logits,ref,targets,ban,only,spec.token_ids)
                adv=torch.tensor([r['advantage'] for r in local],device=model.device)
                loss,diagnostic=grpo_loss(current,old,adv,mask,raw_kl=kl,kl_coef=.01)
                if not torch.isfinite(loss):raise ValueError('Nonfinite GRPO loss')
                (loss*len(local)/64).backward()
            stats.append({'loss':float(loss.detach()),'clip_fraction':float(diagnostic['clip_fraction']),
                'support_miss_fraction':float(diagnostic['current_support_miss_fraction']),
                'kl':float(kl[mask].detach().mean()),'actions':int(mask.sum())})
            del logits,ref,current,kl,loss
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Missing/nonfinite trainable gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('Invalid global gradient norm')
        counts=torch.tensor([sum(x['clip_fraction']*x['actions'] for x in stats),sum(x['actions'] for x in stats),
                             sum(x['support_miss_fraction']*x['actions'] for x in stats)],device=model.device)
        dist.all_reduce(counts)
        _atomic_write_json(root/f'numeric-{step+1:04d}-rank-{rank}.json',{'step':step+1,'rank':rank,
            'global_actions':int(counts[1]),'global_clip_fraction':float(counts[0]/counts[1]),
            'global_support_miss_fraction':float(counts[2]/counts[1]),'microbatches':stats})
        if step==0 and (float(counts[0]/counts[1])>.01 or float(counts[2]/counts[1])>.005):
            raise ValueError('Initial behavior/train probability mismatch exceeds predeclared engine tolerance')
        optimizer.step()
        if state_hash(reference)!=reference_hash:raise ValueError('Fixed SFT reference changed')
        _atomic_write_json(root/f'step-{step+1:04d}-rank-{rank}.json',{'identity':identity,'step':step+1,
            'train_indices':indices,'source_ids':[records[i]['extra_info']['source_sample_id'] for i in indices],
            'trajectories':64,'gradient_norm':float(norm),'microbatches':stats,'elapsed_seconds':time.monotonic()-started,
            'generation_seconds':generation_seconds,'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'arm':contract.get('arm','H'),'reward_column':key,
            'initial_numeric_gate_passed':True,'reference_state_unchanged':state_hash(reference)==reference_hash})
        print(f'GRPO_ENGINE rank={rank} step={step+1} elapsed={time.monotonic()-started:.2f}',flush=True)
    final_hash=state_hash(model);hashes=[None]*4;dist.all_gather_object(hashes,final_hash)
    if len(set(hashes))!=1:raise ValueError('DDP policies differ between ranks')
    torch.save({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'python':random.getstate()},root/f'rng-{rank}.pt')
    dist.barrier()  # Only node-local writes below, never mounted publication.
    if rank==0:
        model.save_pretrained(root/'adapter',save_embedding_layers=False);tokenizer.save_pretrained(root/'adapter')
        torch.save(optimizer.state_dict(),root/'optimizer.pt')
        from new_interest.training.probe_execution_plan import capture_plan
        saved_probe=probe(model,tokenizer,records,spec.token_ids)
        saved_plan=capture_plan()
        _atomic_write_json(root/'state.json',{'identity':identity,'step':stop_at,'consumed_prompts':stop_at*32,
            'arm':contract.get('arm','H'),
            'consumed_trajectories':stop_at*256,'order_sha256':order_hash,'adapter_state_sha256':final_hash,
            'reference_sha256':reference_hash,'reload_probe':saved_probe,'probe_execution_plan':saved_plan,
            'optimizer_state_sha256':optimizer_hash(model,optimizer),'trainable_parameter_names':parameter_names,
            'rank_state_hashes':hashes,'optimizer_states':len(optimizer.state),'initial_source_adapter_sha256':sha(initial/'adapter_model.safetensors')})
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-path',required=True);p.add_argument('--stop-at',type=int,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--verification-output',type=Path)
    a=p.parse_args()
    if a.verify_only:
        import torch
        os.environ.setdefault('LOCAL_RANK','0')
        model,tokenizer,spec=model_load(a.project,a.model_path,a.root/'adapter',True)
        saved=read(a.root/'state.json')
        if state_hash(model)!=saved['adapter_state_sha256']:raise ValueError('Fresh saved state differs')
        records=pq.read_table(a.project/'inputs/train.parquet').to_pylist()
        from new_interest.training.probe_execution_plan import checkpoint_plan
        actual=probe(model,tokenizer,records,spec.token_ids,execution_plan=checkpoint_plan(a.project,a.root,saved))
        if not torch.equal(torch.tensor(actual),torch.tensor(saved['reload_probe'])):raise ValueError('Fresh probe differs')
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-6,fused=True)
        opt.load_state_dict(torch.load(a.root/'optimizer.pt',map_location='cpu',weights_only=False))
        if optimizer_hash(model,opt)!=saved['optimizer_state_sha256']:raise ValueError('Fresh optimizer differs')
        _atomic_write_json(a.verification_output,{'state_sha256':state_hash(model),'fresh_probe_max_abs':0.,
            'optimizer_state_sha256':optimizer_hash(model,opt),'passed':True,'step':saved['step']})
    else:run(a.project,a.root,a.model_path,a.stop_at,a.resume)


if __name__=='__main__':main()
