"""Compact ETag-bound readback and independent histogram-based acceptance."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fetch(tracking, arm, destination):
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient
    from azure.core import MatchConditions
    prefix = tracking['jobs'][arm]['results_prefix']
    client = BlobServiceClient('https://csnewsandfeeds4150361735.blob.core.windows.net',
        credential=AzureCliCredential(process_timeout=30), connection_timeout=10, read_timeout=25,
        retry_total=1).get_container_client('amulet')
    index = {b.name[len(prefix)+1:]:b for b in client.list_blobs(name_starts_with=prefix+'/')}
    if 'complete.json' not in index:
        return {'status':'incomplete', 'arm':arm, 'published_full_score_chunks':sum('/shards/' in k and k.endswith('.parquet') for k in index),
                'completed_shards':sum(k.endswith('/shard_manifest.json') for k in index)}
    files = ['complete.json','identity.json','environment.json','snapshot_audit.json','evaluation_config.json',
             'original_metrics.json','unseen_metrics.json','full_scores/merged/metrics.json','full_scores/merged/merge_manifest.json']
    if 'fast_kernel_environment.json' in index: files.append('fast_kernel_environment.json')
    for i in range(4):
        files += [f'full_scores/shard-{i:02d}/{name}' for name in ['metrics.json','shard_manifest.json','resolved_config.json']]
    receipt = {'observed_at_utc':datetime.now(timezone.utc).isoformat(),'prefix':prefix,'files':{}}
    for rel in files:
        meta = index[rel]
        require(meta.size < 8_000_000, 'Unexpected compact file size')
        data = client.get_blob_client(prefix+'/'+rel).download_blob(etag=meta.etag,match_condition=MatchConditions.IfNotModified).readall()
        target = destination/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): require(target.read_bytes()==data, 'Published compact artifact changed')
        else: target.write_bytes(data)
        receipt['files'][rel] = {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'etag':str(meta.etag)}
    write(destination/'download_receipt.json',receipt)
    return {'status':'downloaded','arm':arm,'files':len(files)}


def accept(tracking, arm, folder):
    result = read(folder/'complete.json'); spec = tracking['contract']['models'][arm]
    download = read(folder/'download_receipt.json')
    require(download['prefix']==tracking['jobs'][arm]['results_prefix'],'Readback source prefix differs')
    for rel, item in download['files'].items():
        require(sha(folder/rel)==item['sha256'],'Downloaded compact file changed: '+rel)
    require(result['status']=='accepted' and result['arm']==arm,'Completion differs')
    require(result['bundle_sha256']==tracking['bundle_sha256'],'Bundle identity differs')
    require(result['diagnostic_contract_sha256']==sha(Path(tracking['bundle'])/'manifests/candidate_competition.json'),'Contract differs')
    require(result['adapter_file_sha256']==spec['adapter_file_sha256'] and result['adapter_directory_sha256']==spec['adapter_directory_sha256'],'Adapter differs')
    require(result['accepted_prediction_sha256']==spec['predictions_sha256'],'Reference predictions differ')
    require(result['original_top20_exact_records']==31392 and result['original_top20_max_abs_score_difference']==0,'Original reproduction failed')
    require(result['input_audit']==tracking['contract']['data_audit'],'Data audit differs')
    require(result['records']==31392 and result['target_occurrences']==37728 and result['supported_classes']==104,'Coverage differs')
    source_metrics = read(Path(tracking['bundle'])/'manifests/sources'/f'{arm}-metrics.json')
    computed = {}
    for treatment in ('original','unseen'):
        per_class = result['treatments'][treatment]['per_class']
        require({c:d['support'] for c,d in per_class.items()}==result['per_class_support'],'Class supports differ')
        sums = {f'interest_macro_{m}_at_{k}':[] for m in ('recall','ndcg') for k in (1,5,10,20)}
        rank_sum = 0
        for c,d in per_class.items():
            h = {int(k):v for k,v in d['target_rank_histogram'].items()}
            require(all(1<=r<=108 and type(n)==int and n>0 for r,n in h.items()),'Invalid rank histogram')
            require(sum(h.values())==d['support'],'Histogram coverage differs')
            rs = sum(r*n for r,n in h.items()); require(rs==d['target_rank_sum'],'Rank sum differs');rank_sum+=rs
            for k in (1,5,10,20):
                hits = sum(n for r,n in h.items() if r<=k)
                gain = math.fsum(n/math.log2(r+1) for r,n in h.items() if r<=k)
                require(hits==d[f'hits_at_{k}'] and abs(gain-d[f'discounted_gain_at_{k}'])<1e-8,'Metric counts differ')
                sums[f'interest_macro_recall_at_{k}'].append(hits/d['support'])
                sums[f'interest_macro_ndcg_at_{k}'].append(gain/d['support'])
        computed[treatment] = {k:math.fsum(v)/len(v) for k,v in sums.items()}
        existing = read(folder/f'{treatment}_metrics.json')
        for k,v in computed[treatment].items():
            require(abs(v-result['treatments'][treatment]['metrics'][k])<1e-12 and abs(v-existing[k])<1e-12,'Independent macro mismatch')
            if treatment=='original':require(abs(v-source_metrics[k])<1e-12,'Original macro not reproduced')
        require(abs(rank_sum/37728-result['treatments'][treatment]['mean_target_rank'])<1e-12,'Mean rank differs')
    require(all(computed['unseen'][k]>=v-1e-12 for k,v in computed['original'].items()),'Filtering decreased target metric')
    merged = read(folder/'full_scores/merged/metrics.json')
    require(merged['top_k']==108 and merged['adapter_sha256']==spec['adapter_directory_sha256'],'Full-score binding differs')
    manifests = [read(folder/f'full_scores/shard-{i:02d}/shard_manifest.json') for i in range(4)]
    require({m['shard_index'] for m in manifests}==set(range(4)) and all(m['shard_record_count']==7848 for m in manifests),'Shard coverage differs')
    require(len({m['evaluation_identity'] for m in manifests})==1,'Shard identities differ')
    merge = read(folder/'full_scores/merged/merge_manifest.json')
    require(merge['record_count']==merge['expected_record_count']==31392 and merge['num_shards']==4,'Merge coverage differs')
    require(merge['merged_prediction_sha256']==result['full_prediction_sha256'],'Merged prediction hash differs')
    require(merge['metrics_sha256']==sha(folder/'full_scores/merged/metrics.json'),'Merged metrics hash differs')
    for shard in manifests:
        require(merge['shard_prediction_sha256'][str(shard['shard_index'])]==shard['prediction_sha256'],'Shard prediction hash differs')
        require(shard['evaluation_identity']==merge['evaluation_identity']==merged['evaluation_identity'],'Merge identity differs')
        require(shard['adapter_sha256']==spec['adapter_directory_sha256'],'Shard adapter differs')
        for field in ('eval_sha256','mapping_sha256'):
            require(shard[field]==merged[field]==source_metrics[field],'Frozen input hash differs')
        config=read(folder/f"full_scores/shard-{shard['shard_index']:02d}/resolved_config.json")
        require(config['batch_size']==spec['batch_size'] and config['top_k']==108 and config['num_shards']==4,'Execution config differs')
    snapshot = read(folder/'snapshot_audit.json')
    require(snapshot['files']==read(Path(tracking['bundle'])/'manifests/h50_snapshot.json')['files'],'Base snapshot differs')
    environment=read(folder/'environment.json')
    expected=read(Path(tracking['bundle'])/'configs/experiments/mind_cir_108_v1.json')['worker_packages']
    require(environment['packages']==expected and environment['training_updates']==0 and environment['kernel_profile']==spec['kernel_profile'],'Runtime differs')
    receipt = {'status':'independently_accepted','arm':arm,'complete_sha256':sha(folder/'complete.json'),
               'metrics':computed,'remote_full_prediction_sha256':result['full_prediction_sha256'],
               'original_top20_exact_records':31392,'records':31392,'targets':37728,'supported_classes':104,
               'reader_sha256':sha(__file__), 'compact_etag_hash_binding':True, 'merge_shard_prediction_hash_binding':True,
               'basis':'Remote full prediction audit plus independent local recomputation from all per-class rank histograms'}
    write(folder/'independent_acceptance.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fetch','accept'])
    p.add_argument('--manifest',type=Path,default=Path('manifests/mind_cir_candidate_competition_20260924_v1.json'))
    p.add_argument('--arm',required=True,choices=['ni_only','cir5_ni3']);a=p.parse_args()
    tracking=read(a.manifest)
    folder=Path('artifacts/mind-cir-candidate-competition-20260924/final-readback')/tracking['experiment']/tracking['jobs'][a.arm]['job']
    result=(fetch if a.command=='fetch' else accept)(tracking,a.arm,folder)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
