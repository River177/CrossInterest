"""Evaluation-only candidate competition diagnostic on immutable accepted adapters.

The original native Top20 is retained exactly (including tie order); the tail is
ordered by descending restricted log probability, then mapping index. Filtering
is performed on the full 108 candidates, never on a truncated Top20.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq

from new_interest.experiments import cir_warmup as core
from new_interest.evaluation.evaluate_beam import record_input_sha256, sha256_directory

CONTRACT = Path('manifests/candidate_competition.json')
CUTOFFS = (1, 5, 10, 20)
TIE_POLICY = 'native-top20-prefix_then-descending-score_mapping-index-tail-v1'


def visible_categories(record):
    from new_interest.data.build_cir_dataset import history_from_record
    lines = history_from_record(record).splitlines()
    categories = [re.fullmatch(r'\d+\. \[([^\]]+)\] .+', line)[1] for line in lines]
    core.require(categories == record['history_subcategories'], 'Visible history differs from structured history')
    return set(categories)


def audit_data(records, labels):
    vocabulary = set(labels)
    histories = Counter()
    outside = 0
    support = Counter()
    for row in records:
        history = visible_categories(row)
        targets = row['acceptable_subcategories']
        core.require(len(targets) == len(set(targets)) > 0, 'Invalid target set')
        core.require(set(targets) <= vocabulary - history, 'Filtering removes a positive target')
        histories[len(history & vocabulary)] += 1
        outside += len(history - vocabulary)
        support.update(targets)
    return {'records': len(records), 'target_occurrences': sum(support.values()),
            'supported_classes': len(support), 'per_class_support': dict(sorted(support.items())),
            'history_candidate_count_histogram': dict(sorted(histories.items())),
            'outside_vocabulary_history_occurrences': outside, 'removed_positive_targets': 0,
            'minimum_unseen_candidates': len(labels) - max(histories)}


def full_rankings(logits, token_ids, labels, tokens, *, top_k):
    """Preserve the exact accepted torch.topk(k=20) path and collect its tail."""
    import torch
    core.require(top_k == len(labels), 'Diagnostic must retain every candidate')
    restricted = logits.index_select(1, torch.tensor(token_ids, dtype=torch.long, device=logits.device)).float()
    scores = torch.log_softmax(restricted, dim=1)
    core.require(bool(torch.isfinite(scores).all()), 'Nonfinite candidate score')
    entropy = -(scores.exp() * scores).sum(dim=1)
    _, prefixes = torch.topk(scores, k=20, dim=1, largest=True, sorted=True)
    result = []
    for values, prefix in zip(scores.cpu().tolist(), prefixes.cpu().tolist()):
        prefix_set = set(prefix)
        tail = sorted((i for i in range(len(labels)) if i not in prefix_set), key=lambda i: (-values[i], i))
        result.append([{'label': labels[i], 'response': tokens[i], 'token': tokens[i],
                        'token_id': token_ids[i], 'score': values[i]} for i in prefix + tail])
    return result, entropy.cpu().tolist()


def independently_summarize(records, rows, labels):
    """Compute class-macro metrics and full target ranks without metric-library calls."""
    core.require(len(records) == len(rows) > 0, 'Prediction coverage differs')
    vocabulary = set(labels)
    support = Counter()
    hits = {t: {str(k): Counter() for k in CUTOFFS} for t in ('original', 'unseen')}
    gains = {t: {str(k): Counter() for k in CUTOFFS} for t in hits}
    ranks = {t: defaultdict(Counter) for t in hits}
    historical_slots = Counter()
    historical_records = Counter()
    positive_ties = Counter()
    target_records = []
    for i, (record, row) in enumerate(zip(records, rows)):
        core.require(row['record_index'] == i and row['input_sha256'] == record_input_sha256(record), 'Prediction/input identity differs')
        core.require(str(row['impression_id']) == str(record['impression_id']), 'Impression differs')
        ranked = row['ranked_labels']; values = row['ranked_scores']
        core.require(len(ranked) == len(set(ranked)) == len(values) == len(labels) and set(ranked) == vocabulary, 'Full candidate coverage differs')
        core.require(all(math.isfinite(v) for v in values) and all(a >= b for a, b in zip(values, values[1:])), 'Scores invalid or unsorted')
        history = visible_categories(record) & vocabulary
        unseen = [c for c in ranked if c not in history]
        by_label = dict(zip(ranked, values))
        for k in CUTOFFS:
            n = len(set(ranked[:k]) & history)
            historical_slots[str(k)] += n
            historical_records[str(k)] += int(n > 0)
        for target in record['acceptable_subcategories']:
            core.require(target in unseen, 'Positive was removed')
            support[target] += 1
            target_ranks = {}
            for treatment, order in [('original', ranked), ('unseen', unseen)]:
                rank = order.index(target) + 1
                ranks[treatment][target][rank] += 1
                target_ranks[treatment] = rank
                for k in CUTOFFS:
                    hits[treatment][str(k)][target] += int(rank <= k)
                    gains[treatment][str(k)][target] += 1 / math.log2(rank + 1) if rank <= k else 0
                    if k <= len(order) and by_label[order[k-1]] == by_label[target] and sum(by_label[c] == by_label[target] for c in order) > 1:
                        positive_ties[f'{treatment}@{k}'] += 1
            target_records.append({'record_index': i, 'user_id': record['user_id'], 'target': target,
                                   'original_rank': target_ranks['original'], 'unseen_rank': target_ranks['unseen']})
    result = {'records': len(records), 'target_occurrences': sum(support.values()),
              'supported_classes': len(support), 'per_class_support': dict(sorted(support.items())),
              'tie_policy': TIE_POLICY, 'positive_occurrences_tied_with_cutoff_score': dict(positive_ties),
              'historical_prediction_slots': dict(historical_slots),
              'historical_prediction_slot_fraction': {str(k): historical_slots[str(k)] / (len(records)*k) for k in CUTOFFS},
              'records_with_historical_prediction': dict(historical_records), 'treatments': {}}
    for t in hits:
        per_class = {c: {'support': n, 'target_rank_histogram': dict(sorted(ranks[t][c].items())),
                         'target_rank_sum': sum(rank*count for rank, count in ranks[t][c].items()),
                         **{f'hits_at_{k}': hits[t][str(k)][c] for k in CUTOFFS},
                         **{f'discounted_gain_at_{k}': gains[t][str(k)][c] for k in CUTOFFS}}
                     for c, n in sorted(support.items())}
        metrics = {f'interest_macro_{metric}_at_{k}': math.fsum(counts[str(k)][c]/n for c, n in support.items()) / len(support)
                   for metric, counts in [('recall', hits[t]), ('ndcg', gains[t])] for k in CUTOFFS}
        rank_sum = sum(v['target_rank_sum'] for v in per_class.values())
        result['treatments'][t] = {'metrics': metrics, 'per_class': per_class,
                                  'mean_target_rank': rank_sum / sum(support.values()),
                                  'class_macro_mean_target_rank': math.fsum(v['target_rank_sum']/v['support'] for v in per_class.values())/len(per_class)}
    return result, target_records


def verify_bundle(project):
    manifest = core.read(project/'manifests/competition_bundle.json')
    for rel, digest in manifest['files'].items():
        core.require(core.sha256(project/rel) == digest, 'Diagnostic payload changed: '+rel)
    return core.read(project/CONTRACT)


def infer(project, root, arm, shard):
    from new_interest.evaluation import evaluate_id_token_ranking as evaluator
    contract = verify_bundle(project)
    spec = contract['models'][arm]
    if spec['kernel_profile'] == 'torch-fallback-explicit':
        core.reference_kernels()
    else:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        core.require(q.is_fast_path_available and q.chunk_gated_delta_rule.__module__.startswith('fla.'), 'NI-only fast kernel differs')
    evaluator.rank_restricted_logits = full_rankings
    evaluator.EVALUATION_IDENTITY_VERSION = 'candidate-competition-full108-' + TIE_POLICY
    config = core.read(root/'evaluation_config.json')
    config.update(shard_index=shard, output_dir=str(root/f'full_scores/shard-{shard:02d}'))
    evaluator.evaluate_id_token_ranking(config)
    print('COMPETITION_SHARD_COMPLETE', arm, shard, flush=True)


def audit(project, root, arm):
    from new_interest.evaluation.metrics import evaluate_ranked_outputs
    contract = verify_bundle(project); spec = contract['models'][arm]
    source = Path('/mnt/default')/spec['results_prefix']
    full_path = root/'full_scores/merged/id_token_predictions.parquet'
    old_path = source/spec['predictions_path']
    core.require(core.sha256(old_path) == spec['predictions_sha256'], 'Accepted source predictions changed')
    old = pq.read_table(old_path).to_pylist(); rows = pq.read_table(full_path).to_pylist()
    records = pq.read_table(project/core.DATA/'dev_full_grouped.parquet').to_pylist()
    mapping = pq.read_table(project/core.DATA/'interest_token_mapping.parquet').to_pylist()
    labels = [r['target_subcategory'] for r in mapping]
    tokens = {r['target_subcategory']: r['interest_token'] for r in mapping}
    core.require(len(old) == len(rows) == len(records) == 31392, 'Readback coverage differs')
    score_max = 0.0
    for a, b in zip(old, rows):
        core.require(a['record_index'] == b['record_index'] and a['input_sha256'] == b['input_sha256'], 'Accepted input pairing differs')
        core.require(a['ranked_labels'] == b['ranked_labels'][:20], 'Native Top20 did not reproduce accepted predictions')
        core.require(b['ranked_tokens'] == [tokens[c] for c in b['ranked_labels']], 'SID binding differs')
        score_max = max(score_max, max(abs(x-y) for x,y in zip(a['ranked_scores'], b['ranked_scores'][:20])))
    core.require(score_max == 0, 'Native Top20 scores did not reproduce exactly')
    result, target_rows = independently_summarize(records, rows, labels)
    histories = [visible_categories(record) for record in records]
    for treatment in ('original', 'unseen'):
        outputs = [[{'label': c, 'score': v} for c, v in zip(row['ranked_labels'], row['ranked_scores'])
                    if treatment == 'original' or c not in history] for row, history in zip(rows, histories)]
        metrics, _ = evaluate_ranked_outputs(records, outputs, cutoffs=CUTOFFS)
        for key, value in result['treatments'][treatment]['metrics'].items():
            core.require(abs(value - metrics[key]) < 1e-12, 'Independent metric recomputation differs')
        core.write(root/f'{treatment}_metrics.json', metrics)
    accepted_metrics = core.read(project/'manifests/sources'/f'{arm}-metrics.json')
    for key, value in result['treatments']['original']['metrics'].items():
        core.require(abs(value - accepted_metrics[key]) < 1e-12, 'Accepted metric reproduction differs')
    from new_interest.evaluation.evaluate_beam import _atomic_write_parquet
    _atomic_write_parquet(root/'target_ranks.parquet', target_rows)
    result.update(status='accepted', arm=arm, input_audit=audit_data(records, labels),
                  full_prediction_sha256=core.sha256(full_path), accepted_prediction_sha256=spec['predictions_sha256'],
                  original_top20_exact_records=31392, original_top20_max_abs_score_difference=score_max,
                  adapter_file_sha256=spec['adapter_file_sha256'], adapter_directory_sha256=spec['adapter_directory_sha256'],
                  diagnostic_contract_sha256=core.sha256(project/CONTRACT),
                  bundle_sha256=core.sha256(project/'manifests/competition_bundle.json'),
                  full_target_ranks_sha256=core.sha256(root/'target_ranks.parquet'), large_payloads='remote_only')
    core.write(root/'complete.json', result)
    print('CANDIDATE_COMPETITION_ACCEPTED', arm, flush=True)


def worker(project, root, arm):
    import torch
    from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
    contract = verify_bundle(project); spec = contract['models'][arm]
    root.mkdir(parents=True, exist_ok=True)
    identity = {'arm': arm, 'contract_sha256': core.sha256(project/CONTRACT),
                'bundle_sha256': core.sha256(project/'manifests/competition_bundle.json')}
    if (root/'identity.json').exists():
        core.require(core.read(root/'identity.json') == identity, 'Output identity changed')
    else:
        core.write(root/'identity.json', identity)
    expected = core.read(project/core.CONTRACT)['worker_packages']
    versions = {k: importlib.metadata.version(k) for k in expected}
    core.require(versions == expected, 'Worker packages differ')
    core.require(torch.cuda.device_count() == 4 and all('A100' in torch.cuda.get_device_name(i) for i in range(4)), 'Four A100 required')
    core.write(root/'environment.json', {'packages': versions, 'kernel_profile': spec['kernel_profile'], 'training_updates': 0})
    _, shared = core.bind()
    snapshot = shared.prepare_snapshot(project, '2b', root/'snapshot_audit.json')
    core.require(core.read(root/'snapshot_audit.json')['files'] == core.read(project/'manifests/h50_snapshot.json')['files'], 'Model snapshot changed')
    adapter = Path('/mnt/default')/spec['results_prefix']/spec['adapter_path']
    core.require(core.sha256(adapter/'adapter_model.safetensors') == spec['adapter_file_sha256'], 'Accepted adapter bytes changed')
    core.require(sha256_directory(adapter) == spec['adapter_directory_sha256'], 'Accepted adapter directory changed')
    from safetensors import safe_open
    with safe_open(adapter/'adapter_model.safetensors', framework='pt', device='cpu') as f:
        core.require(len(f.keys()) == 373, 'Accepted adapter tensor count differs')
    config = core.read(project/'configs/id_token/mind_cir_v1.json')
    config.update(model_name=str(snapshot), adapter_path=str(adapter), eval_file=str(project/core.DATA/'dev_full_grouped.parquet'),
                  interest_token_mapping_file=str(project/core.DATA/'interest_token_mapping.parquet'),
                  batch_size=spec['batch_size'], top_k=108, num_shards=4, checkpoint_size=256, cutoffs=list(CUTOFFS))
    core.write(root/'evaluation_config.json', config)
    def one(shard):
        core.run([sys.executable, '-m', 'new_interest.experiments.cir_candidate_competition', 'infer',
                  '--project', project, '--root', root, '--arm', arm, '--shard', str(shard)],
                 project, root/f'logs/infer-{shard}.log', str(shard))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, range(4)))
    merge_id_token_shards([root/f'full_scores/shard-{i:02d}' for i in range(4)], project/core.DATA/'dev_full_grouped.parquet',
                         project/core.DATA/'interest_token_mapping.parquet', root/'full_scores/merged', expected_records=31392)
    audit(project, root, arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'infer', 'audit'])
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['ni_only', 'cir5_ni3'], required=True)
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.command == 'infer':
        infer(args.project.resolve(), args.root.resolve(), args.arm, args.shard)
    else:
        globals()[args.command](args.project.resolve(), args.root.resolve(), args.arm)


if __name__ == '__main__':
    main()
