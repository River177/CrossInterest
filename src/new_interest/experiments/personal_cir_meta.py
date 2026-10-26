"""Four-rank, resumable per-user CIR meta training; deploy outer weights only."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from new_interest.experiments import oi_ni_meta as core
from new_interest.experiments import oi_ni_meta_distributed as execution
from new_interest.experiments import oi_ni_full_sft as checkpoints
from new_interest.experiments.oi_ni_full_sft_storage import LocalTrainingJournal
from new_interest.training.oi_ni_full_sft import FullVocabularySFTModel, completion_batch, encode_completions
from new_interest.training.oi_ni_meta_objective import LossChunk
from new_interest.training.personal_cir_meta import compute_personal_gradients


def validate_data(config):
    data = Path(config['data_dir'])
    meta = json.loads((data / 'metadata.json').read_text())
    if core.sha256_file(data / 'metadata.json') != config['data_metadata_sha256']:
        raise ValueError('Data metadata changed')
    for name, expected in meta['files'].items():
        if core.sha256_file(data / name) != expected:
            raise ValueError('Data changed: ' + name)
    queries = pq.read_table(data / 'ni.parquet').to_pylist()
    supports = pq.read_table(data / 'supports.parquet').to_pylist()
    users = pq.read_table(data / 'users.parquet').to_pylist()
    plan = pq.read_table(data / 'episodes.parquet').to_pylist()
    if (len(queries), len(supports), len(users), len(plan)) != (21600, 34471, 21600, 2025):
        raise ValueError('Frozen data count changed')
    for i, user in enumerate(users):
        if user['query_index'] != i or user['user_id'] != queries[i]['user_id'] or len(user['support_indices']) > 2:
            raise ValueError('User episode alignment differs')
        for j in user['support_indices']:
            r = supports[j]
            if r['support_index'] != j or r['query_index'] != i or r['user_id'] != user['user_id']:
                raise ValueError('Support/query user identity differs')
            if r['removed_category'] == queries[i]['target_subcategory']:
                raise ValueError('Support and query targets must be disjoint')
    for epoch in range(3):
        indices = [i for row in plan if row['epoch'] == epoch for i in row['row_indices']]
        if Counter(indices) != Counter(range(21600)):
            raise ValueError('Incomplete NI epoch')
    return data, queries, supports, users, plan


def episode_gradient(model, index, queries, supports, users, pad, device, seed, alpha):
    def loss_fn(example, query):
        def compute(view):
            # The query stream is keyed to global user position, never rank.
            devices = [device.index] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed(seed)
                return view(**completion_batch([example], pad, device))
        return LossChunk(compute, weight=1.)
    f = [loss_fn(supports[j], False) for j in users[index]['support_indices']]
    g = [loss_fn(queries[index], True)]
    return compute_personal_gradients(model, f, g, alpha=alpha)


def reduce_gradients(values, denominator):
    """Only detached outer gradients are reduced, in bounded contiguous buckets."""
    names = list(values)
    for start in range(0, len(names), 16):
        chunk = names[start:start + 16]
        flat = torch.cat([values[n].reshape(-1) for n in chunk])
        dist.all_reduce(flat)
        flat.div_(denominator)
        offset = 0
        for n in chunk:
            count = values[n].numel()
            values[n].copy_(flat[offset:offset + count].view_as(values[n]))
            offset += count


def optimizer_for(model, config):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()), lr=config['outer_learning_rate'],
                                 weight_decay=config['weight_decay'], fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        checkpoints.learning_rate_multiplier(step, config['steps'], config['warmup_ratio']))
    return params, optimizer, scheduler


def restore_state(model, params, optimizer, scheduler, checkpoint, identity, context):
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['parameter_names'] != list(params) or state['world_size'] != 4:
        raise ValueError('Checkpoint training identity differs')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    execution._restore_rank_rng(state['rng_by_rank'][context.rank])
    if len(optimizer.state) != len(params) or scheduler.last_epoch != state['step']:
        raise ValueError('Incomplete optimizer/scheduler restore')
    for parameter, saved in optimizer.state.items():
        if int(saved['step']) != state['step']:
            raise ValueError('Adam cursor differs')
        for name in ['exp_avg', 'exp_avg_sq']:
            if saved[name].shape != parameter.shape or not bool(torch.isfinite(saved[name]).all()):
                raise ValueError('Invalid Adam moments')
    expected = json.loads((checkpoint / 'state.json').read_text())
    if execution._replica_digest(params, context) != expected['tensor_sha256']:
        raise ValueError('Checkpoint tensors differ after reload')
    probe = json.loads((checkpoint / 'reload_probe.json').read_text())
    if core.capture_probe(model, None, probe['input_ids']) != probe['scores']:
        raise ValueError('Checkpoint reload logits differ')
    return state


def numerical_gate(model, q, f, users, tokenizer, context, root, config):
    device = torch.device('cuda', context.local_rank)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    selected = [next(i for i, u in enumerate(users) if len(u['support_indices']) == k) for k in (0, 1, 2)]
    longest_support = max(range(len(f)), key=lambda i: len(f[i]['input_ids']))
    owner = next(i for i, u in enumerate(users) if longest_support in u['support_indices'])
    longest_query = max(range(len(q)), key=lambda i: len(q[i]['input_ids']))
    selected.append(owner)
    original = execution._replica_digest(params, context)
    result = episode_gradient(model, selected[context.rank], q, f, users, tokenizer.pad_token_id,
                              device, 907 + context.rank, config['inner_alpha'])
    distributed = {n: g.clone() for n, g in result.gradients.items()}
    reduce_gradients(distributed, 4)
    del result
    with checkpoints.primary_phase(context, 'serial-versus-four-rank per-user gradients'):
        if context.primary:
            serial = {n: torch.zeros_like(p) for n, p in params.items()}
            for position, i in enumerate(selected):
                one = episode_gradient(model, i, q, f, users, tokenizer.pad_token_id,
                                       device, 907 + position, config['inner_alpha'])
                for n in serial:
                    serial[n].add_(one.gradients[n], alpha=.25)
                del one
            diff = sum(float((serial[n].double() - distributed[n].double()).square().sum()) for n in params)
            norm = sum(float(serial[n].double().square().sum()) for n in params)
            relative = math.sqrt(diff / max(norm, 1e-30))
            if relative > 1e-5 or any(not torch.allclose(serial[n], distributed[n], atol=1e-6, rtol=1e-5) for n in params):
                raise ValueError(f'Four-rank meta gradient parity failed: {relative}')
            parity = checkpoints.audit_full_loss(model, q[longest_query], tokenizer.pad_token_id)
            zero = episode_gradient(model, longest_query, q, f, users, tokenizer.pad_token_id, device, 991, 0.)
            plain = episode_gradient(model, longest_query, q, f,
                                     [{**u, 'support_indices': []} for u in users],
                                     tokenizer.pad_token_id, device, 991, config['inner_alpha'])
            for n in params:
                if not torch.equal(zero.gradients[n], plain.gradients[n]):
                    raise ValueError('alpha=0 does not reduce exactly to ordinary NI')
            del zero, plain, serial
            torch.cuda.reset_peak_memory_stats()
            longest = episode_gradient(model, longest_query, q, f, users, tokenizer.pad_token_id,
                                       device, 1001, config['inner_alpha'])
            core.write_json(root / 'numerical_gate.json', {
                'accepted': True, 'serial_four_rank_relative_l2': relative,
                'atol': 1e-6, 'rtol': 1e-5, 'alpha_zero_exact': True,
                'query_max_tokens': len(q[longest_query]['input_ids']),
                'support_max_tokens': len(f[longest_support]['input_ids']),
                'longest_query_loss': float(longest.loss), 'longest_query_hvp_norm': float(longest.hvp_norm),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'native_full_vocabulary': parity,
                'inner_update_is_per_user': True, 'test_time_adaptation': False})
            del longest
    del distributed
    if execution._replica_digest(params, context) != original:
        raise ValueError('Numerical gate mutated shared parameters')
    torch.cuda.empty_cache()


def run(args, context):
    config = json.loads(args.config.read_text())
    bundle = json.loads(Path('personal_cir_bundle_manifest.json').read_text())
    for relative, expected in bundle['files'].items():
        if core.sha256_file(Path(relative)) != expected:
            raise ValueError('Payload changed: ' + relative)
    source = core.sha256_file(Path('personal_cir_bundle_manifest.json'))
    identity = execution.digest_json({'source': source, 'config': config, 'arm': args.arm, 'phase': args.phase})
    data, rows, support_rows, users, plan = validate_data(config)
    total = 8 if args.phase == 'gate' else config['steps']
    mode_root = args.root / args.arm / args.phase
    attempt = execution._broadcast(uuid.uuid4().hex if context.primary else None, context)
    local = Path(os.environ.get('META_LOCAL_SCRATCH', '/scratch/personal-cir')) / attempt
    local.mkdir(parents=True, exist_ok=True)
    checkpoints._SHARED = local
    root = mode_root / 'attempts' / attempt
    parent, parent_source, completed, snapshot = None, None, False, None
    with checkpoints.primary_phase(context, 'prepare immutable inputs and attempt'):
        if context.primary:
            root.mkdir(parents=True)
            pointer = mode_root / 'completed_attempt.json'
            if pointer.exists():
                saved = json.loads(pointer.read_text())
                old = mode_root / 'attempts' / saved['attempt']
                if saved['identity'] != identity or core.sha256_file(old / 'manifest.json') != saved['manifest_sha256']:
                    raise ValueError('Completed stage identity differs')
                core.verify_manifest(old, identity)
                completed = True
            parent = checkpoints.find_parent(mode_root, identity)
            if parent:
                parent_source = str(parent)
                dest = local / 'parent'
                shutil.copytree(parent, dest)
                core.verify_manifest(dest, identity)
                parent = dest
            snapshot = core.prepare_snapshot(config, root)
            core.write_json(root / 'resolved_config.json', {'identity': identity, 'source_sha256': source,
                'arm': args.arm, 'phase': args.phase, 'config': config, 'parent': str(parent) if parent else None})
            core.write_json(root / 'environment.json', {'world_size': 4, 'gpu': torch.cuda.get_device_name(),
                'cuda': torch.version.cuda, 'packages': {n: importlib.metadata.version(n) for n in
                ['torch', 'transformers', 'peft', 'accelerate', 'tokenizers', 'pyarrow', 'safetensors']}})
    if execution._broadcast(completed, context):
        return
    parent, parent_source, snapshot = execution._broadcast((str(parent) if parent else None, parent_source, snapshot), context)
    parent = Path(parent) if parent else None
    source_adapter = args.root / 'sources' / args.arm / 'adapter'
    adapter = parent / 'adapter' if parent else source_adapter
    model, tokenizer, spec, initialization = core.make_model(config, snapshot,
        data / 'interest_token_mapping.parquet', adapter, wrapper_class=FullVocabularySFTModel)
    params, optimizer, scheduler = optimizer_for(model, config)
    original = execution._replica_digest(params, context)
    if not parent:
        from new_interest.training.sft_state_audit import capture_adapter_state
        expected = json.loads((source_adapter.parent / 'source.json').read_text())['state_sha256']
        if capture_adapter_state(model.causal_lm)['state_sha256'] != expected:
            raise ValueError('Loaded initialization tensors differ')
    q = encode_completions(tokenizer, rows, config['max_seq_length'])
    f = encode_completions(tokenizer, support_rows, config['max_seq_length'])
    if execution.digest_json(q) != config['query_encoding_sha256'] or execution.digest_json(f) != config['support_encoding_sha256']:
        raise ValueError('Native completion tokenization changed')
    probes = core.encode_records(tokenizer, [rows[i] for i in config['probe_indices']], max_length=config['max_seq_length'])
    start, log_rows = 0, []
    baseline = {'initial_tensor_sha256': original, 'source_arm': args.arm}
    if parent:
        saved = restore_state(model, params, optimizer, scheduler, parent, identity, context)
        start, baseline = saved['step'], saved['baseline']
        log_rows = checkpoints.read_prefix(parent / 'training_prefix.jsonl', plan, start)
    journal = None
    with checkpoints.primary_phase(context, 'persist model and token audits'):
        if context.primary:
            core.write_json(root / 'model_audit.json', {'identity': identity, 'arm': args.arm,
                'initialization': initialization, 'initial_tensor_sha256': original,
                'backend': model.backend_audit, 'trainable_shapes': {n: list(p.shape) for n, p in params.items()}})
            core.write_json(root / 'token_audit.json', {'queries': len(q), 'supports': len(f),
                'query_max_tokens': max(len(x['input_ids']) for x in q),
                'support_max_tokens': max(len(x['input_ids']) for x in f), 'over_length': 0,
                'query_encoding_sha256': execution.digest_json(q), 'support_encoding_sha256': execution.digest_json(f)})
            journal = LocalTrainingJournal(local, log_rows)
            journal.snapshot(root / 'training.jsonl')
    if args.phase == 'gate' and start == 0:
        numerical_gate(model, q, f, users, tokenizer, context, root, config)
        baseline['numerical_gate'] = execution._broadcast(
            json.loads((root / 'numerical_gate.json').read_text()) if context.primary else None, context)
    model.train()
    device = torch.device('cuda', context.local_rank)
    last = Path(parent_source) if parent_source else None
    for index in range(start, total):
        group = plan[index]
        optimizer.zero_grad(set_to_none=True)
        accumulated = {n: torch.zeros_like(p) for n, p in params.items()}
        stats = torch.zeros(4, dtype=torch.float64, device=device)
        dist.barrier(); torch.cuda.synchronize(); begin = time.monotonic()
        for position in range(context.rank, len(group['row_indices']), 4):
            i = group['row_indices'][position]
            seed = config['seed'] + 104729 * index + 8191 * position
            result = episode_gradient(model, i, q, f, users, tokenizer.pad_token_id, device, seed, config['inner_alpha'])
            for n in accumulated:
                accumulated[n].add_(result.gradients[n])
            stats[0] += result.loss.double(); stats[1] += result.support_loss.double()
            stats[2] += result.inner_grad_norm; stats[3] += result.hvp_norm
            del result
        reduce_gradients(accumulated, len(group['row_indices']))
        dist.all_reduce(stats); stats /= len(group['row_indices'])
        for n, p in params.items():
            p.grad = accumulated[n]
        norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), config['max_grad_norm'], error_if_nonfinite=True))
        lr = optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step()
        torch.cuda.synchronize(); seconds = execution._rank_max_seconds(time.monotonic() - begin)
        row = {'step': index + 1, 'epoch': group['epoch'], 'row_indices': group['row_indices'],
               'loss': float(stats[0]), 'support_loss_mean_per_user': float(stats[1]),
               'inner_grad_norm_mean': float(stats[2]), 'hvp_norm_mean': float(stats[3]),
               'support_count': sum(len(users[i]['support_indices']) for i in group['row_indices']),
               'query_count': len(group['row_indices']), 'lr': lr, 'grad_norm': norm, 'seconds': seconds,
               'world_size': 4, 'inner_alpha': config['inner_alpha'], 'phase': args.phase}
        log_rows.append(row)
        if context.primary:
            journal.append(row)
            core.log('personal_cir_update', arm=args.arm, **{k: v for k, v in row.items() if k != 'row_indices'})
        del accumulated, stats
        step = index + 1
        if step in {2, total} or step % 100 == 0 or step % 675 == 0:
            last = checkpoints.checkpoint(context, model, tokenizer, optimizer, scheduler, root,
                identity, step, probes, log_rows, baseline, journal)
        if step == 2:
            old_hash = execution._replica_digest(params, context)
            del params, model, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()
            model, tokenizer, spec, _ = core.make_model(config, snapshot, data / 'interest_token_mapping.parquet',
                                                       last / 'adapter', wrapper_class=FullVocabularySFTModel)
            params, optimizer, scheduler = optimizer_for(model, config)
            restore_state(model, params, optimizer, scheduler, last, identity, context)
            if execution._replica_digest(params, context) != old_hash:
                raise ValueError('Fresh reload changed shared parameters')
            with checkpoints.primary_phase(context, 'fresh-reload acceptance'):
                if context.primary:
                    core.write_json(root / 'reload_step2.json', {'accepted': True, 'max_abs_error': 0.,
                        'tensor_sha256': old_hash, 'step': 2, 'world_size': 4})
    final_hash = execution._replica_digest(params, context)
    del params, model, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()
    model, tokenizer, spec, _ = core.make_model(config, snapshot, data / 'interest_token_mapping.parquet',
                                               last / 'adapter', wrapper_class=FullVocabularySFTModel)
    params, optimizer, scheduler = optimizer_for(model, config)
    restore_state(model, params, optimizer, scheduler, last, identity, context)
    if execution._replica_digest(params, context) != final_hash:
        raise ValueError('Final fresh reload changed tensors')
    with checkpoints.primary_phase(context, 'stage acceptance'):
        if context.primary:
            checkpoints.read_prefix(root / 'training.jsonl', plan, total)
            success = {'identity': identity, 'arm': args.arm, 'phase': args.phase, 'steps': total,
                'ni_presentations': sum(x['query_count'] for x in log_rows),
                'support_presentations': sum(x['support_count'] for x in log_rows),
                'training_seconds': sum(x['seconds'] for x in log_rows), 'world_size': 4,
                'final_tensor_sha256': final_hash, 'fresh_reload_max_abs_error': 0.,
                'deployed_outer_parameters': True, 'test_time_adaptation': False,
                'checkpoint_relative_to_mode': str(last.relative_to(mode_root)) if last.is_relative_to(mode_root) else str(last)}
            if args.phase == 'gate':
                if not baseline.get('numerical_gate', {}).get('accepted'):
                    raise ValueError('Missing exact-gradient gate evidence')
                success['numerical_gate'] = baseline['numerical_gate']
            if total == 2025 and (success['ni_presentations'], success['support_presentations']) != (64800, 103413):
                raise ValueError('Final exposure count differs')
            core.write_json(root / 'success.json', success)
            core.write_json(root / 'manifest.json', {'identity': identity, 'files': core.inventory(root)})
            core.write_json(mode_root / 'completed_attempt.json', {'identity': identity, 'attempt': attempt,
                'manifest_sha256': core.sha256_file(root / 'manifest.json')})
            core.log('personal_cir_stage_complete', **success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['cir5', 'cold'], required=True)
    parser.add_argument('--phase', choices=['gate', 'formal'], required=True)
    args = parser.parse_args()
    context = execution.initialize_distributed()
    try:
        run(args, context)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
