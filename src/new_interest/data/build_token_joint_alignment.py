"""Train-only category-name/SID pairs mixed with the exact Token125 NI data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.infrastructure.paths import portable_path
from new_interest.training.semantic_id_tokens import load_interest_token_mapping, register_semantic_interest_tokens
from new_interest.training.sft_data import tokenize_training_example

RULE_VERSION='token125-ni-joint-name-id-v1'
TASKS=('new_interest','interest_name_to_id','interest_id_to_name')
REPEATS=10


def alignment_pairs(mapping):
    result=[]
    for row in mapping:
        label=row['target_subcategory'];token=row['interest_token'];index=row['interest_index']
        for task,question,response in [
            (TASKS[1],f'Which interest type has the name {json.dumps(label,ensure_ascii=False)}? Answer with its interest ID only.',token),
            (TASKS[2],f'What is the name of the interest type with ID {json.dumps(token)}? Answer with the exact name only.',label),
        ]:
            prompt='Answer the question about interest type identification.\n\n'+question
            result.append({'base_alignment_id':f'{task}:{index:03d}','training_task':task,'target_subcategory':label,
                'prompt':prompt,'response':response,'messages':[{'role':'user','content':prompt},{'role':'assistant','content':response}]})
    return result


def build(source,output,tokenizer_path):
    from transformers import AutoTokenizer
    if output.exists():raise FileExistsError(output)
    mapping=load_interest_token_mapping(source/'model_token_mapping.parquet')
    assert len(mapping)==125
    table=pq.read_table(source/'token/ni.parquet');assert table.num_rows==21600
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    spec=register_semantic_interest_tokens(tokenizer,source/'model_token_mapping.parquet',expected_count=125)
    pairs=alignment_pairs(mapping);assert len(pairs)==250
    def length(row):
        encoded=tokenize_training_example(tokenizer,row['messages'],max_seq_length=2048,enable_thinking=False)
        return sum(label!=-100 for label in encoded['labels']),len(encoded['input_ids'])
    rows=[];lengths={};by_task=Counter();max_length=0
    for original in table.to_pylist():
        n,total=length(original);assert n==3
        row={**original,'training_task':TASKS[0],'base_alignment_id':None,'expected_assistant_tokens':n}
        rows.append(row);max_length=max(max_length,total);by_task[TASKS[0]]+=n
    for pair in pairs:
        n,total=length(pair);lengths[pair['base_alignment_id']]=n;max_length=max(max_length,total)
        for repeat in range(REPEATS):
            row={key:None for key in table.column_names}
            sample_id=f"alignment:{pair['base_alignment_id']}:repeat{repeat:02d}"
            row.update(pair,sample_id=sample_id,source_sample_id=pair['base_alignment_id'],split='train',
                       task_type=pair['training_task'],history_length=0,expected_assistant_tokens=n)
            rows.append(row);by_task[pair['training_task']]+=n
    # Preserve original NI row fields/order in the stored prefix. The existing
    # global seeded sampler shuffles all three tasks together every epoch.
    schema=pa.schema([pa.field(f.name,f.type,nullable=True) for f in table.schema]+[
        pa.field('training_task',pa.string(),nullable=False),pa.field('base_alignment_id',pa.string()),
        pa.field('expected_assistant_tokens',pa.int32(),nullable=False)])
    counts=Counter(row['training_task'] for row in rows)
    assert dict(counts)==dict(zip(TASKS,[21600,1250,1250])) and len(rows)==24100
    assert len({r['sample_id'] for r in rows})==len(rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging-',dir=output.parent))
    try:
        pq.write_table(pa.Table.from_pylist(rows,schema=schema),staging/'joint.parquet',compression='zstd')
        loaded=pq.read_table(staging/'joint.parquet').to_pylist();assert loaded==rows
        assert [{k:r[k] for k in table.column_names} for r in loaded[:21600]]==table.to_pylist()
        pq.write_table(pa.Table.from_pylist(pairs),staging/'alignment_pairs.parquet',compression='zstd')
        for name in ['model_token_mapping.parquet','interest_token_mapping.parquet']:
            shutil.copyfile(source/name,staging/name)
        shutil.copyfile(source/'token/dev_full_grouped.parquet',staging/'dev_full_grouped.parquet')
        shutil.copyfile(source/'token/ni.parquet',staging/'ni_reference.parquet')
        stats={'version':RULE_VERSION,'rows_per_epoch':24100,'tasks':dict(counts),'epochs':3,'optimizer_steps':2262,
            'presentations':72300,'presentations_by_task':{k:v*3 for k,v in counts.items()},
            'unique_alignment_pairs':250,'copies_per_pair_per_epoch':10,'presentations_per_pair':30,
            'output_tokens_per_epoch_by_task':dict(by_task),'supervised_token_shares':{k:v/sum(by_task.values()) for k,v in by_task.items()},
            'max_training_tokens':max_length,'over_max_seq_length':0,'candidate_count':108,'model_token_count':125,
            'all_original_NI_fields_exact':True,'alignment_uses_user_or_dev_labels':False,
            'loss':'native full-vocabulary completion-only token_mean; unit weight per supervised token',
            'initialization_token_ids':spec.token_ids,'pair_assistant_lengths':lengths}
        write_json(staging/'statistics.json',stats)
        files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
        write_json(staging/'metadata.json',{'version':RULE_VERSION,'source':portable_path(source),
            'source_NI_sha256':sha256(source/'token/ni.parquet'),'source_dev_sha256':sha256(source/'token/dev_full_grouped.parquet'),
            'source_model_mapping_sha256':sha256(source/'model_token_mapping.parquet'),'files':files})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging);raise
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/training/mind_history_token_20260930_v1'))
    p.add_argument('--output',type=Path,default=Path('data/training/mind_token_joint_alignment_20261001_v1'))
    p.add_argument('--tokenizer-path',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.tokenizer_path),indent=2))
