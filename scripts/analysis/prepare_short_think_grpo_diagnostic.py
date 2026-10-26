"""Freeze2048 Dev diagnostics while preserving original four-shard/batch8 groups."""
from pathlib import Path
import hashlib
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def choose_blocks(records,count=64):
    if len(records)%32:raise ValueError('Original Dev must partition into complete32-row blocks')
    blocks=[set(label for r in records[i:i+32] for label in r['acceptable_subcategories']) for i in range(0,len(records),32)]
    universe=set.union(*blocks);uncovered=set(universe);selected=[]
    tie=lambda i:hashlib.sha256(f'42:block:{i}'.encode()).hexdigest()
    while uncovered:
        remaining=[i for i in range(len(blocks)) if i not in selected]
        best=min(remaining,key=lambda i:(-len(blocks[i]&uncovered),tie(i)))
        if not blocks[best]&uncovered:raise ValueError('Uncovered class')
        selected.append(best);uncovered-=blocks[best]
        if len(selected)>count:raise ValueError('Coverage exceeds declared diagnostic budget')
    coverage_count=len(selected)
    remaining=sorted((i for i in range(len(blocks)) if i not in selected),key=tie)
    selected=sorted(selected+remaining[:count-len(selected)])
    return selected,coverage_count,len(universe)


def main():
    repo=Path(__file__).resolve().parents[2];root=repo/'artifacts/mind-short-think-grpo-20260914'
    out=root/'production-preparation/diagnostic2048';out.mkdir(parents=True,exist_ok=False)
    bundle=root/'bundle-preflight-v3';source=bundle/'inputs/dev.parquet'
    rows=pq.read_table(source).to_pylist();assert len(rows)==31392
    selected,coverage_count,classes=choose_blocks(rows)
    indices=[i for block in selected for i in range(block*32,(block+1)*32)]
    assert len(indices)==2048 and all(i%4==original%4 for i,original in enumerate(indices))
    subset=[rows[i] for i in indices]
    pq.write_table(pa.Table.from_pylist(subset,schema=pq.read_schema(source)),out/'diagnostic_dev.parquet',compression='zstd')
    old=repo/'artifacts/mind-short-think-sft-20260913/readback-full-v1'
    prediction=old/'real_eval/merged/id_token_predictions.parquet'
    assert sha(prediction)==json.loads((old/'artifact_manifest.json').read_text())['files']['real_eval/merged/id_token_predictions.parquet']
    all_predictions=pq.read_table(prediction,columns=['record_index','ranked_labels','ranked_tokens','ranked_scores']).to_pylist()
    assert [r['record_index'] for r in all_predictions]==list(range(31392))
    selected_predictions=[all_predictions[i] for i in indices]
    ranked=[[{'label':label,'response':token,'score':score} for label,token,score in zip(r['ranked_labels'],r['ranked_tokens'],r['ranked_scores'])] for r in selected_predictions]
    sys.path.insert(0,str(bundle/'src'))
    from new_interest.evaluation.metrics import evaluate_ranked_outputs
    baseline,_=evaluate_ranked_outputs(subset,ranked,cutoffs=(1,5,10,20))
    (out/'short_sft_baseline_metrics.json').write_text(json.dumps(baseline,indent=2)+'\n')
    manifest={'scope':'Dev diagnostic for progress/tuning, not a held-out Test or final metric substitute','rows':2048,'classes':classes,
        'selection':'Greedy supported-class coverage with hashed block ties, fill to64 complete32-row blocks by seed42 hash; preserve original order',
        'coverage_blocks':coverage_count,'selected_blocks':selected,'original_record_indices':indices,
        'original_four_shard_batch8_groups_preserved':True,'source_dev_sha256':sha(source),'diagnostic_sha256':sha(out/'diagnostic_dev.parquet'),
        'baseline_prediction_sha256':sha(prediction),'baseline_source':'Accepted full Short-CoT SFT predictions; same original inference batches',
        'baseline_metrics_sha256':sha(out/'short_sft_baseline_metrics.json')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'rows':2048,'supported_classes':classes,'coverage_blocks':coverage_count,'diagnostic_sha256':manifest['diagnostic_sha256']}))


if __name__=='__main__':main()
