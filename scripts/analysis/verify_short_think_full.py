"""Independent cohort, token-boundary and scalar Macro readback for short-think SFT."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from new_interest.data.generate_preference_state_teacher import student_prompt
from new_interest.evaluation.evaluate_short_think_ranking import analyze_generation
from new_interest.training.sft_data import apply_chat_template

ROOT=Path(__file__).resolve().parents[2]
BASELINE=Path('/Users/river/.codex/worktrees/d00e/CrossInterest/artifacts/mind-old-new-interest-v1/full-mixed-g4-v1-readback/mixed/real_eval/merged')
KEYS=['interest_macro_recall_at_10','interest_macro_recall_at_20','interest_macro_ndcg_at_10','interest_macro_ndcg_at_20']


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def macro(records, predictions):
    support=Counter();hits={k:Counter() for k in (10,20)};gains={k:defaultdict(list) for k in (10,20)}
    for record,row in zip(records,predictions):
        targets=set(record['acceptable_subcategories']);assert targets
        support.update(targets)
        positions={label:i+1 for i,label in enumerate(row['ranked_labels'])}
        for target in targets:
            rank=positions.get(target,21)
            for k in (10,20):
                if rank<=k:hits[k][target]+=1;gains[k][target].append(1/math.log2(rank+1))
    values={}
    for k in (10,20):
        values[f'interest_macro_recall_at_{k}']=math.fsum(hits[k][label]/n for label,n in support.items())/len(support)
        values[f'interest_macro_ndcg_at_{k}']=math.fsum(math.fsum(gains[k][label])/n for label,n in support.items())/len(support)
    return values,support


def verify(bundle, result, output):
    manifest=read(result/'real_eval/merged/merge_manifest.json')
    predfile=result/'real_eval/merged/id_token_predictions.parquet';metricfile=predfile.with_name('metrics.json')
    assert sha(predfile)==manifest['merged_prediction_sha256'] and sha(metricfile)==manifest['metrics_sha256']
    base_manifest=read(BASELINE/'merge_manifest.json')
    assert sha(BASELINE/'id_token_predictions.parquet')==base_manifest['merged_prediction_sha256']
    assert sha(BASELINE/'metrics.json')==base_manifest['metrics_sha256']
    metrics=read(metricfile);base_metrics=read(BASELINE/'metrics.json')
    records=pq.read_table(bundle/'inputs/dev.parquet').to_pylist()
    rows=pq.read_table(predfile).to_pylist();base_rows=pq.read_table(BASELINE/'id_token_predictions.parquet').to_pylist()
    assert len(records)==len(rows)==len(base_rows)==31392
    assert sha(bundle/'inputs/dev.parquet')==manifest['eval_sha256']
    mapping=pq.read_table(bundle/'inputs/interest_token_mapping.parquet').to_pylist()
    token_by_label={r['target_subcategory']:r['interest_token'] for r in mapping}
    assert len(token_by_label)==108 and sha(bundle/'inputs/interest_token_mapping.parquet')==manifest['mapping_sha256']
    tokenizer=AutoTokenizer.from_pretrained(result/'sft/final_adapter',local_files_only=True)
    token_by_id={tokenizer.encode(token,add_special_tokens=False)[0]:token for token in token_by_label.values()}
    stop_ids={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}
    seen=set();status=Counter();valid_ids=[];think_sid=0;argmax_disagreement=0
    for i,(record,row,base) in enumerate(zip(records,rows,base_rows)):
        assert row['record_index']==base['record_index']==i
        assert str(record['impression_id'])==str(row['impression_id'])==str(base['impression_id'])
        assert row['impression_id'] not in seen;seen.add(row['impression_id'])
        assert set(record['acceptable_subcategories'])==set(row['acceptable_subcategories'])==set(base['acceptable_subcategories'])
        ids=row['generated_token_ids'];assert tokenizer.decode(ids,skip_special_tokens=False)==row['generated_text']
        parsed=analyze_generation(tokenizer,ids+([tokenizer.eos_token_id] if row['natural_end'] else []),stop_ids=stop_ids,token_by_id=token_by_id)
        for k in ['valid_format','generation_status','generated_sid','answer_token_offset','generated_tokens']:
            assert parsed[k]==row[k],(i,k)
        labels,tokens,scores=row['ranked_labels'],row['ranked_tokens'],row['ranked_scores']
        expected=20 if row['valid_format'] else 0
        assert len(labels)==len(set(labels))==len(tokens)==len(scores)==expected
        assert tokens==[token_by_label[label] for label in labels]
        assert all(math.isfinite(v) for v in scores) and all(a>=b for a,b in zip(scores,scores[1:]))
        status[row['generation_status']]+=1
        think_sid+=bool(re.search(r'<interest_\d{3}>',row['generated_text'].split('</think>',1)[0]))
        if row['valid_format']:
            prompt=apply_chat_template(tokenizer,[{'role':'user','content':student_prompt(record['prompt'])}],add_generation_prompt=True,enable_thinking=True)
            prefix=tokenizer.encode(prompt,add_special_tokens=False)+ids[:row['answer_token_offset']]
            assert not set(prefix)&set(token_by_id)
            assert tokenizer.decode(prefix,skip_special_tokens=False).rstrip().endswith('<answer>')
            valid_ids.append(i);argmax_disagreement+=row['generated_sid']!=tokens[0]
    calculated,support=macro(records,rows);baseline_calculated,baseline_support=macro(records,base_rows)
    assert support==baseline_support and len(support)==104 and sum(support.values())==37728
    for key in KEYS:
        assert abs(calculated[key]-metrics[key])<1e-12
        assert abs(baseline_calculated[key]-base_metrics[key])<1e-12
    assert dict(status)==metrics['generation_status_counts']
    valid_new,valid_support=macro([records[i] for i in valid_ids],[rows[i] for i in valid_ids])
    valid_base,_=macro([records[i] for i in valid_ids],[base_rows[i] for i in valid_ids])
    generated=[r['generated_tokens'] for r in rows]
    thoughts=[r['think_tokens'] for r in rows if r['think_tokens'] is not None]
    counts=Counter(r['ranked_labels'][0] for r in rows if r['ranked_labels'])
    base_counts=Counter(r['ranked_labels'][0] for r in base_rows)
    reload=read(result/'sft/semantic_adapter_reload_audit.json')
    assert reload['atol']==reload['rtol']==0 and reload['all_close'] and reload['adapter_state_integrity']['all_tensors_exact']
    report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
        'full_dev_impressions':31392,'supported_classes':104,'target_occurrences':37728,
        'same_cohort_order_and_targets_as_mixed':True,'every_generated_prefix_checked_without_answer_sid':True,
        'metric_order':KEYS,'metrics':calculated,'baseline_metrics':baseline_calculated,
        'metrics_percent':[f'{calculated[k]*100:.2f}%' for k in KEYS],
        'delta_percentage_points':{k:(calculated[k]-baseline_calculated[k])*100 for k in KEYS},
        'generation_status_counts':dict(status),'valid_rate':len(valid_ids)/31392,
        'natural_end_count':sum(r['natural_end'] for r in rows),'sid_in_think_rows':think_sid,
        'generation_token_stats':{'mean':float(np.mean(generated)),'median':float(np.median(generated)),
            'p95':float(np.percentile(generated,95)),'max':max(generated)},
        'parsed_think_token_stats':{'count':len(thoughts),'mean':float(np.mean(thoughts)),
            'median':float(np.median(thoughts)),'p95':float(np.percentile(thoughts,95)),'max':max(thoughts)},
        'matched_valid_cohort_diagnostic':{'records':len(valid_ids),'supported_classes':len(valid_support),
            'short_think':valid_new,'mixed':valid_base},
        'top1_counts':dict(counts.most_common()),'baseline_top1_counts':dict(base_counts.most_common()),
        'cached_greedy_vs_canonical_fresh_forward_top1_disagreements':argmax_disagreement,
        'disagreement_cause_not_established':True,'predictions_sha256':sha(predfile),'metrics_sha256':sha(metricfile),
        'adapter_sha256':sha(result/'sft/final_adapter/adapter_model.safetensors'),
        'verifier_sha256':sha(Path(__file__))}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','metrics_percent','generation_status_counts','generation_token_stats','delta_percentage_points']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.bundle,a.result,a.output)
