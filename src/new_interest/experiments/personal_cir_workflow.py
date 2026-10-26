"""Sequential four-A100 workflow: verified sources, two gates, two meta runs, three evaluations."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq

from new_interest.experiments import cir_warmup as cir
from new_interest.experiments import oi_ni_meta as core


def read(path):
    return json.loads(Path(path).read_text())


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    core.log('launch_stage', command=command, log=str(log))
    with log.open('a') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line); stream.flush()
        result = process.wait()
    if result:
        raise RuntimeError(f'Stage failed with exit {result}; see {log}')


def verify_payload():
    manifest = read('personal_cir_bundle_manifest.json')
    for relative, expected in manifest['files'].items():
        if core.sha256_file(Path(relative)) != expected:
            raise ValueError('Frozen payload changed: ' + relative)
    return core.sha256_file(Path('personal_cir_bundle_manifest.json'))


def prepare_sources(project, root):
    """All large state remains on the worker/persistent storage, never the Mac."""
    from safetensors.torch import load_file
    from new_interest.training.sft_state_audit import tensor_state_manifest
    from new_interest.evaluation.evaluate_beam import sha256_directory
    config = read(project / 'configs/experiments/personal_cir_meta_v1.json')
    contract = read(project / 'manifests/cir_parameter_transfer.json')
    identity = verify_payload()
    source = Path('/mnt/default') / contract['source_results_prefix']
    for arm, relative, expected_file, expected_state in [
        ('cir5', 'formal/cir5/final_adapter', contract['cir5_adapter_sha256'], contract['cir5_state_sha256']),
        ('baseline', 'formal/ni3/final_adapter', contract['reference_ni3_adapter_sha256'], None),
    ]:
        target = root / 'sources' / arm
        if (target / 'manifest.json').exists():
            core.verify_manifest(target, identity)
            continue
        local = Path('/scratch/personal-cir-sources') / arm
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(source / relative, local / 'adapter')
        if core.sha256_file(local / 'adapter/adapter_model.safetensors') != expected_file:
            raise ValueError('Parent adapter differs: ' + arm)
        state = tensor_state_manifest(load_file(str(local / 'adapter/adapter_model.safetensors'), device='cpu'))
        if expected_state and state['state_sha256'] != expected_state:
            raise ValueError('Parent parameter manifest differs')
        if arm == 'baseline' and sha256_directory(local / 'adapter') != contract['reference_ni3_adapter_directory_sha256']:
            raise ValueError('Baseline adapter directory differs')
        core.write_json(local / 'source.json', {'arm': arm, 'adapter_sha256': expected_file,
            'state_sha256': state['state_sha256'], 'source_prefix': contract['source_results_prefix'],
            'source_relative': relative, 'tensor_count': len(state['tensors'])})
        core.publish(local, target, identity)
    target = root / 'sources/cold'
    if (target / 'manifest.json').exists():
        core.verify_manifest(target, identity)
        return
    # Invoke the original, byte-frozen SFT constructor, but stop before optimization.
    from new_interest.training import train_sft
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    cir.reference_kernels()
    local = Path('/scratch/personal-cir-sources/cold')
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    original = train_sft.record_initial_adapter_state
    expected = read(project / 'manifests/transfer_sources/cold_state.json')
    class Captured(BaseException):
        pass
    def capture(model, output_dir, checkpoint_identity, *, input_adapter=None):
        if input_adapter is not None:
            raise ValueError('Cold constructor loaded an adapter')
        actual = tensor_state_manifest(get_peft_model_state_dict(model, save_embedding_layers=False))
        if actual['state_sha256'] != expected['state_sha256'] or actual['tensors'] != expected['tensors']:
            raise ValueError('Cold initialization is not the exact original state')
        model.save_pretrained(local / 'adapter', safe_serialization=True, save_embedding_layers=False)
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'], local_files_only=True)
        register_semantic_interest_tokens(tokenizer, project / config['data_dir'] / 'interest_token_mapping.parquet')
        tokenizer.save_pretrained(local / 'adapter')
        core.write_json(local / 'source.json', {'arm': 'cold', 'state_sha256': actual['state_sha256'],
            'tensor_count': len(actual['tensors']), 'optimizer_updates': 0,
            'adapter_sha256': core.sha256_file(local / 'adapter/adapter_model.safetensors')})
        core.publish(local, target, identity)
        raise Captured()
    train_sft.record_initial_adapter_state = capture
    try:
        settings = read(project / 'configs/sft/mind_cir_base_v1.json')
        settings.update(train_file=str(project / config['data_dir'] / 'ni.parquet'),
            output_dir='/scratch/personal-cir-cold-constructor',
            checkpoint_dir='/scratch/personal-cir-cold-unused', wandb_mode='disabled')
        settings.pop('adapter_path', None)
        train_sft.train(settings)
        raise RuntimeError('Original constructor did not stop before optimization')
    except Captured:
        pass
    finally:
        train_sft.record_initial_adapter_state = original
    core.verify_manifest(target, identity)


def completed(root, arm, phase):
    mode = root / arm / phase
    pointer = read(mode / 'completed_attempt.json')
    attempt = mode / 'attempts' / pointer['attempt']
    if core.sha256_file(attempt / 'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Completion pointer hash differs')
    core.verify_manifest(attempt, pointer['identity'])
    success = read(attempt / 'success.json')
    checkpoint = mode / success['checkpoint_relative_to_mode']
    core.verify_manifest(checkpoint, pointer['identity'])
    return attempt, checkpoint, success


def evaluate_adapter(project, destination, adapter, *, gates=False):
    target = destination / 'model/final_adapter'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter, target)
    if core.inventory(target) != core.inventory(adapter):
        raise ValueError('Deployment copy differs')
    return cir.evaluate(project, destination, 'model', gates=gates)


def compare_results(root):
    """Independent target-rank recomputation and paired user Bayesian bootstrap."""
    import numpy as np
    conditions = ['cir5_ni3', 'cir5_meta_ni3', 'cold_meta_ni3']
    records = pq.read_table(cir.DATA / 'dev_full_grouped.parquet').to_pylist()
    users = {u: i for i, u in enumerate(sorted({r['user_id'] for r in records}))}
    classes = {c: i for i, c in enumerate(sorted({c for r in records for c in r['acceptable_subcategories']}))}
    target_user, target_class, ranks = [], [], {}
    for condition in conditions:
        folder = root / 'evaluation' / condition
        predictions = pq.read_table(folder / 'real_eval/merged/id_token_predictions.parquet').to_pylist()
        # The established evaluator binds identities; independently align by impression id here.
        by_id = {str(r['impression_id']): r for r in predictions}
        if len(by_id) != len(records):
            raise ValueError('Prediction coverage differs')
        condition_ranks = []
        for r in records:
            p = by_id[str(r['impression_id'])]
            if str(p['user_id']) != str(r['user_id']):
                raise ValueError('User alignment differs')
            labels = p['ranked_labels']
            for c in r['acceptable_subcategories']:
                condition_ranks.append(labels.index(c) + 1 if c in labels else 21)
                if condition == conditions[0]:
                    target_user.append(users[r['user_id']]); target_class.append(classes[c])
        ranks[condition] = np.asarray(condition_ranks)
    if (len(target_user), len(classes), len(users)) != (37728, 104, 27082):
        raise ValueError('Full Dev target topology differs')
    u, c = np.asarray(target_user), np.asarray(target_class)
    values = {name: np.stack([(r <= 10), (r <= 20),
                (r <= 10) / np.log2(r + 1), (r <= 20) / np.log2(r + 1)]).astype(float)
              for name, r in ranks.items()}
    def metric(v, weight):
        denominator = np.bincount(c, weights=weight, minlength=104)
        return np.asarray([(np.bincount(c, weights=weight * row, minlength=104) / denominator).mean() for row in v])
    actual = {name: metric(v, np.ones(len(u))) for name, v in values.items()}
    keys = ['interest_macro_recall_at_10', 'interest_macro_recall_at_20',
            'interest_macro_ndcg_at_10', 'interest_macro_ndcg_at_20']
    for name in conditions:
        saved = read(root / 'evaluation' / name / 'real_eval/merged/metrics.json')
        if not np.allclose(actual[name], [saved[k] for k in keys], atol=1e-12, rtol=0):
            raise ValueError('Independent metric calculation differs')
    pairs = [(conditions[1], conditions[0]), (conditions[2], conditions[1]), (conditions[2], conditions[0])]
    rng = np.random.default_rng(42)
    draws = {a + '-minus-' + b: [] for a, b in pairs}
    for _ in range(2000):
        weight = rng.exponential(size=len(users))[u]
        result = {name: metric(v, weight) for name, v in values.items()}
        for a, b in pairs:
            draws[a + '-minus-' + b].append(result[a] - result[b])
    result = {'status': 'independently_recomputed', 'metrics': keys,
              'point_estimates': {k: v.tolist() for k, v in actual.items()},
              'bootstrap': {'unit': 'user', 'draws': 2000, 'seed': 42, 'weights': 'Exp(1)',
                            'fixed_classes': 104, 'covers_training_seed_uncertainty': False},
              'comparisons': {a + '-minus-' + b: {'delta': (actual[a] - actual[b]).tolist(),
                  'ci95': np.quantile(draws[a + '-minus-' + b], [.025, .975], axis=0).tolist()} for a, b in pairs}}
    core.write_json(root / 'comparison.json', result)
    return result


def worker(project, root):
    source = verify_payload()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        saved = read(root / 'complete.json')
        if saved['source_sha256'] != source:
            raise ValueError('Completed experiment source differs')
        return
    config = project / 'configs/experiments/personal_cir_meta_v1.json'
    run([sys.executable, '-m', 'new_interest.experiments.personal_cir_workflow', 'prepare',
         '--project', str(project), '--root', str(root)], root / 'logs/prepare.log')
    # Both gates finish before either formal trajectory begins.
    for phase in ['gate', 'formal']:
        for arm in ['cir5', 'cold']:
            run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                 '-m', 'new_interest.experiments.personal_cir_meta', '--config', str(config),
                 '--root', str(root), '--arm', arm, '--phase', phase], root / 'logs' / f'{phase}-{arm}.log')
            attempt, checkpoint, success = completed(root, arm, phase)
            if phase == 'gate':
                evaluate_adapter(project, root / 'gate_evaluation' / arm, checkpoint / 'adapter', gates=True)
            core.log('stage_independently_verified', arm=arm, phase=phase, steps=success['steps'])
    adapters = {'cir5_ni3': root / 'sources/baseline/adapter'}
    for arm in ['cir5', 'cold']:
        adapters[arm + '_meta_ni3'] = completed(root, arm, 'formal')[1] / 'adapter'
    evaluations = {}
    for condition, adapter in adapters.items():
        dest = root / 'evaluation' / condition
        evaluations[condition] = evaluate_adapter(project, dest, adapter)
    comparison = compare_results(root)
    core.write_json(root / 'complete.json', {'status': 'all_three_accepted', 'source_sha256': source,
        'conditions': list(adapters), 'new_training_branches': 2, 'test_time_adaptation': False,
        'evaluation': evaluations, 'comparison_sha256': core.sha256_file(root / 'comparison.json'),
        'large_artifacts': 'remote_only'})
    core.log('PERSONAL_CIR_ALL_THREE_ACCEPTED', comparison= comparison)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['worker', 'prepare'])
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_sources(args.project, args.root)
    else:
        worker(args.project, args.root)


if __name__ == '__main__':
    main()
