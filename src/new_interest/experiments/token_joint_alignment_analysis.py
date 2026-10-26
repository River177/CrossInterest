"""Remote-only paired comparison with arm-specific prompt identity verification."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.experiments.cir_ni_direct_analysis import compare

ARMS=('baseline','joint')


def record_hash(row):
    return hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def validate_pair(records):
    if set(records)!=set(ARMS) or not records['baseline']:
        raise ValueError('Missing paired Dev data')
    if len(records['baseline'])!=len(records['joint']):
        raise ValueError('Dev coverage differs')
    for left,right in zip(records['baseline'],records['joint']):
        if left != right:
            raise ValueError('Paired Dev records differ')


def analyze(code, output, contract):
    records={arm:pq.read_table(code/'inputs'/arm/'dev_full_grouped.parquet').to_pylist() for arm in ARMS}
    validate_pair(records)
    mapping=code/'inputs/interest_token_mapping.parquet'
    labels=pq.read_table(mapping).to_pylist();by_label={r['target_subcategory']:r['interest_token'] for r in labels}
    assert len(by_label)==108
    predictions={};sources={};original={};checkpoint_audits={}
    for arm in ARMS:
        expected=contract['arms'][arm];root=Path('/mnt/default')/expected['results_prefix']
        complete=json.loads((root/'complete.json').read_text())
        assert complete['status']=='accepted' and complete['bundle_sha256']==expected['bundle_sha256']
        assert sha256(root/'complete.json')==expected['complete_sha256']
        checkpoint=root/f"formal/checkpoints-ni/checkpoint-{expected['optimizer_steps']}"
        marker=json.loads((checkpoint/'checkpoint_complete.json').read_text())
        assert marker['global_step']==expected['optimizer_steps']
        total=0
        for name,item in marker['files'].items():
            rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
            path=checkpoint/rel
            assert path.stat().st_size==item['size'] and sha256(path)==item['sha256'],name
            total+=item['size']
        checkpoint_audits[arm]={'step':expected['optimizer_steps'],'files':len(marker['files']),'bytes':total,'all_payload_hashes_verified':True}
        stage=json.loads((root/'formal/ni/stage_accepted.json').read_text())
        assert stage['initial_state_sha256']==contract['cold_state_sha256'] and stage['optimizer_steps']==expected['optimizer_steps']
        assert stage['exposure_by_task']==expected['presentations_by_task']
        assert sha256(root/'formal/ni/final_adapter/adapter_model.safetensors')==stage['adapter']['adapter_sha256']
        folder=root/'formal/real_eval/merged';m=json.loads((folder/'metrics.json').read_text());merge=json.loads((folder/'merge_manifest.json').read_text())
        pred=folder/'id_token_predictions.parquet'
        assert sha256(pred)==merge['merged_prediction_sha256']==expected['prediction_sha256']
        assert sha256(folder/'metrics.json')==merge['metrics_sha256']==expected['metrics_sha256']
        assert m['evaluation_identity']==expected['evaluation_identity']
        assert m['eval_sha256']==sha256(code/'inputs'/arm/'dev_full_grouped.parquet')
        assert m['mapping_sha256']==sha256(mapping) and m['label_vocabulary_size']==108
        assert m['top_k']==20 and m['history_prediction_filtering'] is False
        rows=pq.read_table(pred).to_pylist();assert len(rows)==len(records[arm])
        for i,(record,row) in enumerate(zip(records[arm],rows)):
            assert row['record_index']==i and str(row['impression_id'])==str(record['impression_id'])
            assert row['input_sha256']==record_hash(record) and row['evaluation_identity']==m['evaluation_identity']
            assert row['ranked_tokens']==[by_label[label] for label in row['ranked_labels']]
            scores=row['ranked_scores'];assert len(scores)==20 and all(math.isfinite(v) for v in scores)
            assert all(a>=b for a,b in zip(scores,scores[1:]))
        predictions[arm]=rows;original[arm]=m
        sources[arm]={key:expected[key] for key in ['prediction_sha256','metrics_sha256','evaluation_identity','complete_sha256','bundle_sha256']}
    result=compare(records['baseline'],predictions,set(by_label),draws=contract['draws'],seed=contract['seed'],arms=ARMS)
    assert tuple(result[k] for k in ['records','users','target_occurrences','supported_classes'])==(31392,27082,37728,104)
    for arm in ARMS:
        for key,value in result['metrics'][arm].items():
            assert abs(value-original[arm][key])<1e-12 and abs(value-contract['arms'][arm]['metrics'][key])<1e-12
    result.update(sources=sources,checkpoint_audits=checkpoint_audits,analysis_source_sha256=sha256(Path(__file__)),
        comparison_limits='One cold joint recipe versus fixed Token125 NI-only baseline. Same NI exposures and initial parameters; auxiliary supervision, 237 additional updates and LR schedule horizon change together. User intervals exclude training-seed uncertainty.')
    write_json(output,result)
    return result
