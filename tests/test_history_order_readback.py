import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('history_order_readback', Path(__file__).resolve().parents[1] / 'scripts/analysis/read_history_order_state.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixture(changed_b_adapter=False):
    objects, source, final_metrics = {}, {}, {}
    for arm in 'BCD':
        metrics = {f'per_class_{kind}': {str(k): {str(i): 0.25 for i in range(104)} for k in (1, 5, 10, 20)} for kind in ('recall', 'ndcg')}
        scores = {f'interest_macro_{kind}_at_{k}': 0.25 for kind in ('recall', 'ndcg') for k in (1, 5, 10, 20)}
        metrics.update(scores)
        final_metrics[arm] = scores
        name = f'evaluations/{arm}/real_eval/merged/metrics.json'
        objects[name] = metrics
        source[arm] = {'metrics_sha256': digest(metrics), 'predictions_sha256': arm}
        adapter = '0a41effcaac398a0ca9d914142b7975dffaad98f435379e436e6132b59c3abf0' if arm == 'B' else 'new'
        if arm == 'B' and changed_b_adapter: adapter = 'foreign'
        objects[f'evaluations/{arm}/accepted.json'] = {'bundle_sha256': 'bundle', 'evaluation': {'prediction_sha256': arm},
            'eval_sha256': 'c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd' if arm == 'C' else 'e53abcf33c9e67a6b8603740ee802671c466ced19586f1cda5b7cf3567ac6354',
            'adapter_file_sha256': adapter}
    source['A'] = {'predictions_sha256': 'c07251bb82843c9caf35a0d6d8d58a57295bd8f4ddb4006e397a6ec1e22dbac6'}
    objects['comparison.json'] = {'status': 'passed', 'source_identity': source,
        'full_dev': {'records': 31392, 'users': 27082, 'target_occurrences': 37728, 'supported_classes': 104, 'metrics': final_metrics}}
    objects['artifact_manifest.json'] = {'files': {k: {'sha256': digest(v)} for k, v in objects.items()}}
    objects['complete.json'] = {'bundle_sha256': 'bundle', 'artifact_manifest_sha256': digest(objects['artifact_manifest.json']),
        'comparison_sha256': digest(objects['comparison.json']), 'status': 'remote_training_reload_full_dev_and_comparison_verified',
        'training': {'steps': 2025, 'samples': 21600, 'epochs': 3, 'adapter': {'adapter_sha256': 'new'}, 'cursor': {'consumed_global_samples': 64800},
                     'reload': {'logits_exact': True, 'all_tensors_exact': True}}}
    return {k: (v, digest(v)) for k, v in objects.items()}, {'bundle_sha256': 'bundle'}


def test_readback_accepts_bound_compact_metrics_and_budget():
    payload, receipt = fixture()
    assert reader.validate_complete(payload, receipt)['status'] == 'compact_independent_readback_passed'


def test_readback_rejects_wrong_baseline_even_with_consistent_manifests():
    payload, receipt = fixture(changed_b_adapter=True)
    with pytest.raises(ValueError, match='original model'):
        reader.validate_complete(payload, receipt)


def test_readback_rejects_changed_metrics_and_incomplete_budget():
    payload, receipt = fixture()
    altered = copy.deepcopy(payload)
    altered['evaluations/B/real_eval/merged/metrics.json'][0]['interest_macro_recall_at_10'] += 0.1
    with pytest.raises(ValueError, match='macro recomputation'):
        reader.validate_complete(altered, receipt)
    altered = copy.deepcopy(payload)
    altered['complete.json'][0]['training']['steps'] = 100
    with pytest.raises(ValueError, match='budget'):
        reader.validate_complete(altered, receipt)
