"""Single cold multitask NI SFT, with real mixed-task gates and exact budgets."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments import cir_warmup as core

COLD='28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565'
TASKS=('new_interest','interest_name_to_id','interest_id_to_name')


def schedule(project):
    value=core.read(project/core.CONTRACT)['direct_training']
    core.require(value=={'samples':24100,'epochs':3,'optimizer_steps':2262,'presentations':72300,'parent':None},'Joint budget changed')
    return value


def stage_specs(project, root, *, gates):
    schedule(project)
    entries=[('single_ni',32,2,1,1,8),('ddp_ni',41,1,4,2,32)] if gates else [('ni',24100,3,4,2,32)]
    return [{'key':k,'task':'ni','parent':None,'samples':n,'epochs':e,'world_size':w,
             'per_device_train_batch_size':micro,'global_batch_size':batch,'accumulation':batch,
             'train_file':root/'gate_data'/f'ni-{n}.parquet' if gates else project/core.DATA/'ni.parquet'}
            for k,n,e,w,micro,batch in entries]


def gate_indices(rows, n):
    counts={32:(24,4,4),41:(29,6,6)}[n];chosen=[]
    for task,count in zip(TASKS,counts):
        indices=[i for i,r in enumerate(rows) if r['training_task']==task]
        # Unique semantic pairs in auxiliaries, longest input/output included.
        seen=set();unique=[]
        for i in sorted(indices,key=lambda i:(-len(rows[i]['prompt']),rows[i]['sample_id'])):
            key=rows[i].get('base_alignment_id') or rows[i]['sample_id']
            if key not in seen:seen.add(key);unique.append(i)
        longest=max(unique,key=lambda i:rows[i]['expected_assistant_tokens'])
        ordered=[longest]+[i for i in unique if i!=longest]
        chosen.extend(ordered[:count])
    core.require(len(chosen)==len(set(chosen))==n,'Invalid mixed gate sampling')
    return chosen


def make_gate_data(project,root):
    table=pq.read_table(project/core.DATA/'ni.parquet');rows=table.to_pylist()
    for n in [32,41]:
        subset=table.take(pa.array(gate_indices(rows,n)))
        for task in ['ni','cir']:
            out=root/'gate_data'/f'{task}-{n}.parquet';out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():core.require(pq.read_table(out).equals(subset),'Gate input changed')
            else:pq.write_table(subset,out,compression='zstd')


def gradient(project,output):
    core.reference_kernels()
    from new_interest.training import audit_semantic_id_token_gradients as audit
    config=core.read(project/'configs/sft/mind_cir_base_v1.json')
    def representatives(train_file,expected_tokens):
        rows=pq.read_table(train_file).to_pylist();selected=[]
        for task in TASKS:
            by_label={}
            for row in sorted(rows,key=lambda r:(len(r['prompt']),r['sample_id'])):
                if row['training_task']==task and row['target_subcategory'] not in by_label:
                    by_label[row['target_subcategory']]=row
            limit=8 if task=='new_interest' else 125
            selected.extend((r['response'],r['messages']) for r in list(by_label.values())[:limit])
        core.require(len(selected)==258,'Incomplete mixed gradient representatives')
        return selected
    audit._representative_messages=representatives
    result=audit.audit_semantic_id_token_gradients(config,output,batch_size=1)
    core.require(result['interest_token_count']==125 and result['all_token_rows_finite_and_nonzero'],'Joint gradient gate failed')
    print('JOINT_ALIGNMENT_GRADIENT_PASSED',flush=True)


def validate_completion_lengths(project,tokenizer):
    from new_interest.training.sft_data import tokenize_training_example
    rows=pq.read_table(project/core.DATA/'ni.parquet').to_pylist()
    tasks=Counter();tokens=Counter()
    for row in rows:
        encoded=tokenize_training_example(tokenizer,row['messages'],max_seq_length=2048,enable_thinking=False)
        count=sum(x!=-100 for x in encoded['labels'])
        core.require(count==row['expected_assistant_tokens'],'Frozen assistant-token length differs')
        tasks[row['training_task']]+=1;tokens[row['training_task']]+=count
    core.require(dict(tasks)==dict(zip(TASKS,[21600,1250,1250])),'Joint task mixture changed')
    return {'tasks':dict(tasks),'assistant_tokens_per_epoch':dict(tokens),'all_lengths_exact':True}


def worker(project,root):
    core.stage_specs=stage_specs;core.formal_schedule=schedule;core.make_gate_data=make_gate_data
    core.worker(project,root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['worker','gradient'])
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='worker':worker(a.project,a.root)
    else:gradient(a.project,a.output)
