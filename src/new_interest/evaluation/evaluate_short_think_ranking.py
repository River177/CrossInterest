"""Free-generate short reasoning, then rank SIDs before the generated answer.

Malformed or length-capped model outputs retain their records and receive empty
rankings. Execution errors raise. No forced closing tags or target-aware prefix
construction is permitted.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.data.generate_preference_state_teacher import student_prompt
from new_interest.evaluation.evaluate_beam import (
    _atomic_write_json, _atomic_write_parquet, canonical_sha256,
    record_input_sha256, select_logical_shard, sha256_directory, sha256_file,
)
from new_interest.evaluation.evaluate_id_token_ranking import rank_restricted_logits
from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.training.semantic_id_tokens import prepare_semantic_adapter_base, validate_saved_interest_tokens
from new_interest.training.sft_data import apply_chat_template

VERSION = "short-think-natural-eos-answer-prefix-ranking-v1"
FORMAT = re.compile(r"\A(?P<think>\s*\S[\s\S]*?)</think>\s*<answer>\s*(?P<sid><interest_\d{3}>)\s*</answer>\s*\Z")


def analyze_generation(tokenizer, generated_ids: list[int], *, stop_ids: set[int],
                       token_by_id: dict[int, str]) -> dict[str, Any]:
    stops = [i for i, value in enumerate(generated_ids) if value in stop_ids]
    natural_end = bool(stops)
    ids = generated_ids[:stops[0]] if stops else generated_ids
    text = tokenizer.decode(ids, skip_special_tokens=False)
    match = FORMAT.fullmatch(text) if natural_end else None
    positions = [i for i, value in enumerate(ids) if value in token_by_id]
    valid = bool(match and len(positions) == 1 and token_by_id[ids[positions[0]]] == match['sid'])
    if match and ('<think>' in match['think'] or re.search(r'<interest_\d{3}>', match['think'])):
        valid = False
    if valid:
        prefix = tokenizer.decode(ids[:positions[0]], skip_special_tokens=False)
        valid = prefix == text[:match.start('sid')]
    result = {
        "generated_token_ids": ids, "generated_text": text,
        "generated_tokens": len(ids) + int(natural_end),
        "natural_end": natural_end, "valid_format": valid,
        "generation_status": "valid" if valid else ("invalid_format" if natural_end else "length_limit"),
        "think_tokens": len(tokenizer.encode(match['think'], add_special_tokens=False)) if match else None,
        "generated_sid": match['sid'] if valid else None,
        "answer_token_offset": positions[0] if valid else None,
    }
    return result


def validate_rows(rows, records, indices, identity, token_by_label, top_k):
    if len(rows) != len(records):
        raise ValueError("Checkpoint row count mismatch")
    for row, record, index in zip(rows, records, indices):
        if (row['record_index'] != index or row['impression_id'] != str(record['impression_id'])
                or row['input_sha256'] != record_input_sha256(record)
                or row['evaluation_identity'] != identity):
            raise ValueError("Checkpoint identity/order mismatch")
        expected = top_k if row['valid_format'] else 0
        labels, tokens, scores = row['ranked_labels'], row['ranked_tokens'], row['ranked_scores']
        if not len(labels) == len(set(labels)) == len(tokens) == len(scores) == expected:
            raise ValueError("Invalid ranking cardinality")
        if tokens != [token_by_label[label] for label in labels]:
            raise ValueError("SID mapping differs")
        if not all(math.isfinite(x) for x in scores) or any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError("Invalid ranking scores")
        if row['valid_format'] and (not row['natural_end'] or row['answer_token_offset'] is None):
            raise ValueError("Valid ranking has no natural answer boundary")
        expected_status = 'valid' if row['valid_format'] else ('invalid_format' if row['natural_end'] else 'length_limit')
        if row['generation_status'] != expected_status:
            raise ValueError("Generation status contradicts validity/end marker")


def summarize(records, rows, identity):
    ranked = [[{'label':label, 'response':token, 'score':score}
               for label, token, score in zip(r['ranked_labels'], r['ranked_tokens'], r['ranked_scores'])]
              for r in rows]
    metrics, details = evaluate_ranked_outputs(records, ranked, cutoffs=tuple(identity['cutoffs']))
    statuses = Counter(r['generation_status'] for r in rows)
    metrics.update({**identity, 'samples':len(rows), 'generation_status_counts':dict(statuses),
        'valid_format_rate':statuses['valid']/len(rows),
        'natural_end_rate':sum(r['natural_end'] for r in rows)/len(rows),
        'length_limit_rate':statuses['length_limit']/len(rows),
        'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/len(rows),
        'invalid_generation_scoring':'empty ranking; included in every impression and class denominator',
        'history_prediction_filtering':False, 'label_vocabulary_size':len(identity['labels'])})
    return metrics, [{**detail, **row} for detail, row in zip(details, rows)]


def generate_batch(model, tokenizer, records, spec, *, max_new_tokens, top_k):
    import torch
    prompts = [apply_chat_template(tokenizer, [{'role':'user', 'content':student_prompt(str(r['prompt']))}],
                 add_generation_prompt=True, enable_thinking=True) for r in records]
    if not all(p.endswith('<think>\n') for p in prompts):
        raise ValueError("Native template must supply the opening think tag")
    encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt')
    encoded = {k:v.to(model.device) for k,v in encoded.items()}
    stop_ids = {int(tokenizer.eos_token_id)}
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        stop_ids.add(im_end)
    text_config = getattr(model.config, 'text_config', model.config)
    if text_config.max_position_embeddings < encoded['input_ids'].shape[1] + max_new_tokens:
        raise ValueError("Prompt plus generation budget exceeds model context")
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
            eos_token_id=sorted(stop_ids), pad_token_id=tokenizer.pad_token_id, use_cache=True)
    offset = encoded['input_ids'].shape[1]
    tokens = [e.interest_token for e in spec.entries]
    labels = [e.target_subcategory for e in spec.entries]
    token_by_id = dict(zip(spec.token_ids, tokens))
    rows, prefix_ids, valid_indices = [], [], []
    for i, output_ids in enumerate(output[:, offset:].tolist()):
        row = analyze_generation(tokenizer, output_ids, stop_ids=stop_ids, token_by_id=token_by_id)
        row.update(ranked_labels=[], ranked_tokens=[], ranked_scores=[], restricted_entropy=None)
        if row['valid_format']:
            original = encoded['input_ids'][i][encoded['attention_mask'][i].bool()].tolist()
            # Slice original token IDs; never include the generated SID or any target.
            prefix_ids.append(original + row['generated_token_ids'][:row['answer_token_offset']])
            valid_indices.append(i)
        rows.append(row)
    if prefix_ids:
        batch = tokenizer.pad({'input_ids':prefix_ids}, padding=True, return_tensors='pt')
        batch = {k:v.to(model.device) for k,v in batch.items()}
        with torch.inference_mode():
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        ranked, entropies = rank_restricted_logits(logits, spec.token_ids, labels, tokens, top_k=top_k)
        for i, ranking, entropy in zip(valid_indices, ranked, entropies):
            rows[i].update(ranked_labels=[r['label'] for r in ranking], ranked_tokens=[r['token'] for r in ranking],
                ranked_scores=[r['score'] for r in ranking], restricted_entropy=entropy)
    return rows


def evaluate(config):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    adapter, eval_file, mapping, out = [Path(config[k]) for k in
        ('adapter_path','eval_file','interest_token_mapping_file','output_dir')]
    records = pq.read_table(eval_file).to_pylist()
    if config.get('max_examples') is not None:
        records = records[:int(config['max_examples'])]
    if not records or len({str(r['impression_id']) for r in records}) != len(records):
        raise ValueError("Empty or duplicate evaluation cohort")
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    spec = validate_saved_interest_tokens(tokenizer, mapping, expected_count=108)
    identity = {'identity_version':VERSION, 'model_name':str(config['model_name']),
        'model_revision':str(config['model_revision']), 'adapter_sha256':sha256_directory(adapter),
        'eval_sha256':sha256_file(eval_file), 'mapping_sha256':sha256_file(mapping),
        'labels':[e.target_subcategory for e in spec.entries], 'tokens':[e.interest_token for e in spec.entries],
        'token_ids':spec.token_ids, 'full_record_count':len(records), 'max_examples':config.get('max_examples'),
        'num_shards':int(config.get('num_shards',1)), 'top_k':20, 'cutoffs':[1,5,10,20],
        'max_new_tokens':int(config.get('max_new_tokens',512)), 'do_sample':False, 'enable_thinking':True,
        'batch_size':int(config.get('batch_size',8)), 'checkpoint_size':int(config.get('checkpoint_size',32)),
        'prompt_rule':'preference-state-reasoning-student-prompt-v1', 'evaluation_source_sha256':sha256_file(Path(__file__)),
        'failure_policy':'no repair; invalid model generation remains in denominator; execution errors raise'}
    identity['evaluation_identity'] = canonical_sha256(identity)
    shard_index = int(config.get('shard_index',0))
    indices, shard_records = select_logical_shard(records, num_shards=identity['num_shards'], shard_index=shard_index)
    if not shard_records:
        raise ValueError("Empty logical shard")
    batch_size, checkpoint_size = int(config.get('batch_size',8)), int(config.get('checkpoint_size',32))
    if min(batch_size, checkpoint_size, identity['max_new_tokens']) <= 0:
        raise ValueError("Invalid batch/checkpoint/generation budget")
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['model_revision'],
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', trust_remote_code=False)
    prepare_semantic_adapter_base(model, spec)
    model = PeftModel.from_pretrained(model, adapter).to(torch.device('cuda',0)).eval()
    out.mkdir(parents=True, exist_ok=True)
    (out/'shards').mkdir(exist_ok=True)
    token_by_label = dict(zip(identity['labels'],identity['tokens']))
    all_rows = []
    started = time.monotonic()
    for start in range(0,len(shard_records),checkpoint_size):
        local = shard_records[start:start+checkpoint_size]
        local_indices = indices[start:start+checkpoint_size]
        path = out/'shards'/f'{start:09d}-{start+len(local):09d}.parquet'
        if path.exists():
            rows = pq.read_table(path).to_pylist()
        else:
            rows = []
            for offset in range(0,len(local),batch_size):
                rows.extend(generate_batch(model,tokenizer,local[offset:offset+batch_size],spec,
                    max_new_tokens=identity['max_new_tokens'],top_k=identity['top_k']))
            for row, record, index in zip(rows,local,local_indices):
                row.update(record_index=index, impression_id=str(record['impression_id']),
                    input_sha256=record_input_sha256(record), evaluation_identity=identity['evaluation_identity'])
        validate_rows(rows,local,local_indices,identity['evaluation_identity'],token_by_label,identity['top_k'])
        if not path.exists():
            _atomic_write_parquet(path,rows)
        all_rows.extend(rows)
        print(f'SHORT_THINK_EVAL {len(all_rows)}/{len(shard_records)}',flush=True)
    metrics, detailed = summarize(shard_records,all_rows,identity)
    prediction = out/'id_token_predictions.parquet'
    _atomic_write_parquet(prediction,detailed)
    _atomic_write_json(out/'metrics.json',metrics)
    _atomic_write_json(out/'resolved_config.json',config)
    _atomic_write_json(out/'shard_manifest.json',{**identity,'shard_index':shard_index,
        'shard_record_count':len(all_rows),'prediction_sha256':sha256_file(prediction),
        'metrics_sha256':sha256_file(out/'metrics.json'),'elapsed_seconds':time.monotonic()-started})
    return metrics


def merge(shard_dirs, eval_file, mapping_file, output_dir, expected_records=None):
    dirs = [Path(p) for p in shard_dirs]
    manifests = [json.loads((p/'shard_manifest.json').read_text()) for p in dirs]
    reference = manifests[0]
    count = reference['num_shards']
    if len(dirs)!=count or sorted(m['shard_index'] for m in manifests)!=list(range(count)):
        raise ValueError('Missing or duplicate logical shard')
    if len({m['evaluation_identity'] for m in manifests})!=1 or reference['identity_version']!=VERSION:
        raise ValueError('Mixed evaluation identities')
    if sha256_file(Path(eval_file))!=reference['eval_sha256'] or sha256_file(Path(mapping_file))!=reference['mapping_sha256']:
        raise ValueError('Source identity mismatch')
    records = pq.read_table(eval_file).to_pylist()
    if reference['max_examples'] is not None:
        records=records[:reference['max_examples']]
    if len(records)!=reference['full_record_count'] or (expected_records is not None and len(records)!=expected_records):
        raise ValueError('Wrong complete cohort size')
    rows = []
    for directory, manifest in zip(dirs,manifests):
        pred=directory/'id_token_predictions.parquet'
        if sha256_file(pred)!=manifest['prediction_sha256'] or sha256_file(directory/'metrics.json')!=manifest['metrics_sha256']:
            raise ValueError('Shard artifact hash mismatch')
        part=pq.read_table(pred).to_pylist()
        indices, local=select_logical_shard(records,num_shards=count,shard_index=manifest['shard_index'])
        validate_rows(part,local,indices,reference['evaluation_identity'],dict(zip(reference['labels'],reference['tokens'])),reference['top_k'])
        rows.extend(part)
    rows.sort(key=lambda r:r['record_index'])
    validate_rows(rows,records,list(range(len(records))),reference['evaluation_identity'],dict(zip(reference['labels'],reference['tokens'])),reference['top_k'])
    identity={k:v for k,v in reference.items() if k not in {'shard_index','shard_record_count','prediction_sha256','metrics_sha256','elapsed_seconds'}}
    metrics,detailed=summarize(records,rows,identity)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    prediction=out/'id_token_predictions.parquet'
    _atomic_write_parquet(prediction,detailed)
    _atomic_write_json(out/'metrics.json',metrics)
    _atomic_write_json(out/'merge_manifest.json',{**identity,'merged_record_count':len(rows),
        'merged_prediction_sha256':sha256_file(prediction),'metrics_sha256':sha256_file(out/'metrics.json'),
        'shard_prediction_sha256':{str(m['shard_index']):m['prediction_sha256'] for m in manifests}})
    return metrics


def main():
    parser=argparse.ArgumentParser(__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('evaluate');run.add_argument('--config',type=Path,required=True)
    for flag in ('adapter-path','eval-file','interest-token-mapping-file','output-dir','model-name'):
        run.add_argument('--'+flag)
    for flag in ('max-examples','num-shards','shard-index','batch-size','max-new-tokens'):
        run.add_argument('--'+flag,type=int)
    join=commands.add_parser('merge');join.add_argument('--shard-dirs',nargs='+',required=True)
    for flag in ('eval-file','mapping-file','output-dir'):
        join.add_argument('--'+flag,required=True)
    join.add_argument('--expected-records',type=int)
    args=parser.parse_args()
    if args.command=='evaluate':
        config=json.loads(args.config.read_text())
        config.update({k:v for k,v in vars(args).items() if k not in {'command','config'} and v is not None})
        result=evaluate(config)
    else:
        result=merge(args.shard_dirs,args.eval_file,args.mapping_file,args.output_dir,args.expected_records)
    print(json.dumps(result,ensure_ascii=False,default=str))


if __name__=='__main__':
    main()
