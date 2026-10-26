"""Zero-update diagnosis of the completed fold1 reference checkpoint.

These are newly measured diagnostic probes. The unavailable original training
probe is never recreated, replaced, or reported as passing by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import pyarrow.parquet as pq

from new_interest.training.checkpoint_persistence import (
    CHECKPOINT_MARKER, hydrate_checkpoint, validate_persistent_checkpoint,
)


CONDITIONS = ("fresh", "warm_forward64", "warm_forward64_backward8", "fresh_replica")
VERSION = "rho-fold1-reload-diagnostic-v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_contract(contract):
    if (contract.get('version') != VERSION or contract.get('conditions') != list(CONDITIONS)
            or contract.get('model_updates') != 0 or contract.get('fold') != 1
            or contract.get('original_training_probe_vectors_available') is not False):
        raise ValueError('Diagnostic contract must preserve four conditions and zero updates')
    if len(contract.get('probe_source_ids', [])) != 4 or len(set(contract['probe_source_ids'])) != 4:
        raise ValueError('Diagnostic must preserve all four original probe identities')
    parent = Path(contract['parent_checkpoint'])
    if not parent.is_absolute() or parent.name != 'checkpoint-1014':
        raise ValueError('Diagnostic parent must be the completed checkpoint-1014')
    for name in ('parent_marker_sha256', 'parent_runtime_sha256'):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid diagnostic parent hash: ' + name)
    relative = Path(contract['train_file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Diagnostic training input must be inside its frozen bundle')
    expected = {'epochs': 3.0, 'world_size': 4, 'seed': 42, 'effective_global_batch_size': 32,
                'input_adapter_sha256': None, 'sample_weight_file_sha256': None,
                'loss_reduction': 'token_mean', 'semantic_id_loss_scope': 'full_vocabulary',
                'model_revision': contract['model_revision']}
    if any(contract['parent_identity'].get(k) != v for k, v in expected.items()):
        raise ValueError('Diagnostic parent is not the original completed fold1 teacher')


def paths_overlap(left, right):
    def canonical_path(value):
        value = Path(value).resolve()
        for mount in (Path('/mnt/default'), Path('/mnt/output')):
            if value.is_relative_to(mount):
                return Path('/__rho_shared_storage__') / value.relative_to(mount)
        return value
    left, right = canonical_path(left), canonical_path(right)
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_output_root(root, parent):
    root, parent = Path(root), Path(parent)
    if not root.is_absolute() or paths_overlap(root, parent.parent.parent):
        raise ValueError('Diagnostic output must be a separate absolute root outside the original run')


def check_endpoint(checkpoint):
    import torch
    checkpoint = Path(checkpoint)
    state = read(checkpoint / 'trainer_state.json')
    optimizer = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
    scheduler = torch.load(checkpoint / 'scheduler.pt', map_location='cpu', weights_only=True)
    entries = optimizer.get('state', {})
    steps = [float(item['step']) for item in entries.values()]
    if (state.get('global_step') != 1014 or state.get('epoch') != 3.0 or len(entries) != 373
            or not all(math.isfinite(value) and value == 1014 for value in steps)
            or scheduler.get('last_epoch') != 1014):
        raise ValueError('Parent Trainer/Adam/scheduler endpoint differs from 1014/epoch3')
    for rank in range(4):
        if not (checkpoint / f'rng_state_{rank}.pth').is_file():
            raise ValueError('Parent is missing a rank RNG state')
    return {'global_step': 1014, 'epoch': 3.0, 'optimizer_state_entries': 373,
            'optimizer_step_min': 1014, 'optimizer_step_max': 1014, 'scheduler_last_epoch': 1014,
            'optimizer_sha256': sha(checkpoint / 'optimizer.pt'),
            'scheduler_sha256': sha(checkpoint / 'scheduler.pt'), 'four_rank_rng_files_present': True}


def stage_parent(project, scratch, contract):
    project, scratch = Path(project), Path(scratch)
    validate_contract(contract)
    parent = Path(contract['parent_checkpoint'])
    if not scratch.is_absolute() or paths_overlap(scratch, parent.parent):
        raise ValueError('Diagnostic staging cannot write into the immutable parent')
    if sha(parent / CHECKPOINT_MARKER) != contract['parent_marker_sha256']:
        raise ValueError('Parent checkpoint marker SHA differs')
    train = project / contract['train_file']
    if (sha(train) != contract['parent_identity']['train_file_sha256']
            or pq.read_metadata(train).num_rows != 10800):
        raise ValueError('Diagnostic input differs from the original training half')
    marker = validate_persistent_checkpoint(parent, contract['parent_identity'])
    expected = {f'rng_state_{rank}.pth' for rank in range(4)}
    if expected.difference(marker['files']):
        raise ValueError('Parent manifest omits four-rank RNG evidence')
    endpoint = check_endpoint(parent)
    checkpoint = hydrate_checkpoint(parent, scratch / 'parent', contract['parent_identity'])
    if check_endpoint(checkpoint) != endpoint:
        raise ValueError('Hydrated optimizer endpoint differs')
    if sha(parent / CHECKPOINT_MARKER) != contract['parent_marker_sha256']:
        raise ValueError('Parent marker changed during read-only hydration')
    return checkpoint, {'parent_marker_sha256': contract['parent_marker_sha256'],
        'parent_identity': marker['identity'], 'all_parent_files_hash_verified': True,
        'files': marker['files'], 'endpoint': endpoint, 'parent_storage_mutated': False}


def tensor_record(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError('Nonfinite tensor in diagnostic state')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def fingerprint(model):
    parameters = {name: tensor_record(value) for name, value in model.named_parameters()}
    buffers = {name: tensor_record(value) for name, value in model.named_buffers()}
    flags = {name: bool(value.requires_grad) for name, value in model.named_parameters()}
    return {'parameter_sha256': canonical(parameters), 'buffer_sha256': canonical(buffers),
            'requires_grad_sha256': canonical(flags), 'parameters': parameters, 'buffers': buffers,
            'requires_grad': flags}


def native_probe(model, tokenizer, spec, records, prefixes):
    import torch
    from new_interest.evaluation.score_rho_reference import effective_output_weight, project_full_vocab
    from new_interest.training.semantic_id_tokens import deterministic_sdpa_math_context
    model.eval()
    token_ids = torch.tensor(spec.token_ids, device=next(model.parameters()).device)
    result = []
    with torch.inference_mode(), deterministic_sdpa_math_context(torch):
        for record, prefix in zip(records, prefixes):
            ids = torch.tensor([prefix['input_ids']], device=token_ids.device)
            captured = []
            hook = model.get_output_embeddings().register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach()))
            try:
                # Keep the original full sequence native head shape. Selecting
                # logits_to_keep=1 could itself change BF16 GEMM rounding.
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError('Diagnostic must observe exactly one native output-head call')
            hidden = captured[0][:, -1]
            weight = effective_output_weight(model)
            native = output.logits[:, -1].index_select(1, token_ids).float()[0]
            fp32 = project_full_vocab(hidden, weight).index_select(1, token_ids)[0]
            if not torch.isfinite(native).all() or not torch.isfinite(fp32).all():
                raise ValueError('Nonfinite diagnostic SID logits')
            result.append({'source_sample_id': str(record['source_sample_id']),
                'prompt_sha256': canonical(record['messages'][0]['content']),
                'input_ids_sha256': tensor_record(ids.cpu())['sha256'], 'input_ids': prefix['input_ids'],
                'padding': 'none-batch1', 'use_cache': False, 'native_full_sequence_head': True,
                'effective_sid_head': tensor_record(weight.index_select(0, token_ids)),
                'last_hidden': tensor_record(hidden), 'native_sid_scores': native.cpu().tolist(),
                'fp32_sid_scores': fp32.cpu().tolist()})
            del output, hidden, weight, captured
    return result


def probe_difference(left, right, field='native_sid_scores'):
    if len(left) != 4 or len(right) != 4:
        raise ValueError('Expected all four diagnostic probes')
    differences, top1, overlaps = [], [], []
    for a, b in zip(left, right):
        if any(a[key] != b[key] for key in ('source_sample_id', 'input_ids_sha256')):
            raise ValueError('Diagnostic probe identities or input tokens differ')
        x, y = a[field], b[field]
        if len(x) != 108 or len(y) != 108 or not all(math.isfinite(float(v)) for v in x + y):
            raise ValueError('Diagnostic SID scores must be finite and cover 108 classes')
        differences.extend(abs(float(u)-float(v)) for u, v in zip(x, y))
        ox = sorted(range(108), key=lambda i: (-x[i], i))
        oy = sorted(range(108), key=lambda i: (-y[i], i))
        top1.append(ox[0] == oy[0])
        overlaps.append(len(set(ox[:20]) & set(oy[:20])))
    return {'max_absolute_difference': max(differences), 'mean_absolute_difference': sum(differences)/len(differences),
            'changed_scores': sum(value != 0 for value in differences), 'top1_match': top1,
            'top20_set_overlap': overlaps}


def summarize(reports):
    if [r['condition'] for r in reports] != list(range(4)):
        raise ValueError('Missing, duplicate, or unordered diagnostic conditions')
    for report in reports:
        if (report['model_updates'] != 0 or not report['original_training_reference_unavailable']
                or not report['parameters_unchanged'] or not report['buffers_unchanged']):
            raise ValueError('Read-only diagnostic mutated state or misrepresented original probes')
    if len({r['contract_sha256'] for r in reports}) != 1:
        raise ValueError('Diagnostic conditions have different contracts')
    same = {name: len({r['before'][name] for r in reports}) == 1
            for name in ('parameter_sha256', 'buffer_sha256', 'requires_grad_sha256')}
    native = [probe_difference(reports[0]['native'], r['native']) for r in reports]
    fixed = [probe_difference(reports[0]['fixed'], r['fixed']) for r in reports]
    repeats = [probe_difference(r['native'], r['native_repeat']) for r in reports]
    fixed_repeats = [probe_difference(r['fixed'], r['fixed_repeat']) for r in reports]
    restored = [probe_difference(r['native'], r['restored_native']) for r in reports]
    fixed_fp32 = [probe_difference(reports[0]['fixed'], r['fixed'], 'fp32_sid_scores') for r in reports]
    fresh_disagreement = native[3]['max_absolute_difference']
    summary = {'version': VERSION, 'diagnostic_completed': True, 'model_updates': 0,
        'contract_sha256': reports[0]['contract_sha256'], 'original_training_reference_unavailable': True,
        'original_training_reference_equivalence_established': False, 'same_starting_state': same,
        'native_vs_fresh': native, 'fixed_vs_fresh': fixed, 'fixed_fp32_vs_fresh': fixed_fp32,
        'native_repeat': repeats, 'fixed_repeat': fixed_repeats, 'restored_vs_native': restored,
        'fresh_replica_max_abs': fresh_disagreement,
        'fixed_cross_condition_max_abs': max(x['max_absolute_difference'] for x in fixed),
        'native_cross_condition_max_abs': max(probe_difference(a['native'], b['native'])['max_absolute_difference']
                                              for a in reports for b in reports),
        'scope': 'New zero-update numerical diagnosis only; no original-probe acceptance or OOF scoring.',
        'large_artifact_policy': 'remote_only'}
    summary['selection_history_supported'] = (
        all(same.values()) and fresh_disagreement == 0
        and any(native[i]['max_absolute_difference'] > 0 for i in (1, 2))
        and summary['fixed_cross_condition_max_abs'] == 0
        and all(x['max_absolute_difference'] == 0 for x in repeats + fixed_repeats + restored))
    summary['fresh_process_variation_observed'] = fresh_disagreement > 0
    summary['read_only_integrity_passed'] = all(same.values())
    summary['numerical_repeat_controls_passed'] = all(
        x['max_absolute_difference'] == 0 for x in repeats + fixed_repeats + restored)
    summary['fixed_controls_passed'] = (
        all(same.values()) and all(x['max_absolute_difference'] == 0 for x in fixed + fixed_fp32 + fixed_repeats))
    return summary


def child(project, root, condition):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from new_interest.evaluation.score_rho_reference import encode_prefix, load_model
    from new_interest.training.semantic_id_tokens import (
        apply_torch_numeric_backend, deterministic_reload_numeric_backend,
        register_semantic_interest_tokens, capture_torch_numeric_backend,
    )
    from new_interest.training.probe_kernel_diagnostic import discover_tuners, snapshot, fixed_selection
    from new_interest.training.sft_data import CompletionOnlyCollator, tokenize_training_example
    from new_interest.training.verify_semantic_id_token_adapter import compare_adapter_state_dicts
    project, root = Path(project), Path(root)
    contract = read(project / 'diagnostic_contract.json')
    validate_contract(contract)
    validate_output_root(root, contract['parent_checkpoint'])
    prepared = read(root / 'prepared.json')
    folder = root / f'condition-{condition}'
    folder.mkdir(parents=True, exist_ok=True)
    report = {'condition': condition, 'mode': CONDITIONS[condition], 'model_updates': 0,
              'optimizer_constructed': False, 'contract_sha256': sha(project / 'diagnostic_contract.json'),
              'diagnostic_seed': 42, 'checkpoint_rng_restored': False,
              'original_training_reference_unavailable': True,
              'original_training_reference_equivalence_established': False}
    try:
        apply_torch_numeric_backend(torch, deterministic_reload_numeric_backend())
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(prepared['model_path'], local_files_only=True)
        mapping = project / 'inputs/interest_token_mapping.parquet'
        spec = register_semantic_interest_tokens(tokenizer, mapping)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        saved_tokenizer = AutoTokenizer.from_pretrained(prepared['checkpoint_path'], local_files_only=True)
        if (tokenizer.get_vocab() != saved_tokenizer.get_vocab()
                or tokenizer.chat_template != saved_tokenizer.chat_template
                or tokenizer.pad_token_id != saved_tokenizer.pad_token_id):
            raise ValueError('Diagnostic tokenizer differs from saved checkpoint')
        model, loading = load_model({'model_name': contract['model_name'],
            'model_path': prepared['model_path'], 'model_revision': contract['model_revision'],
            'adapter_path': prepared['checkpoint_path'], 'device': 'cuda:0', 'attn_implementation': 'sdpa'}, spec, tokenizer)
        saved_state = load_file(str(Path(prepared['checkpoint_path']) / 'adapter_model.safetensors'), device='cpu')
        state_integrity = compare_adapter_state_dicts(saved_state, get_peft_model_state_dict(model, save_embedding_layers=False))
        write(folder / 'adapter_state_integrity.json', state_integrity)
        if not state_integrity['all_tensors_exact'] or state_integrity['tensor_count'] != 373:
            raise ValueError('Diagnostic adapter tensors differ in dtype, shape, keys or exact bytes')
        del saved_state
        # Every condition starts with the same gradient flags. Only its warmup
        # history differs; no optimizer is ever created.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('lora_' in name or 'trainable_tokens_delta' in name)
        if sum(p.requires_grad for p in model.parameters()) != 373:
            raise ValueError('Expected 373 trainable LoRA/SID tensors in every condition')
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.eval()
        records = pq.read_table(project / contract['train_file']).slice(0, 64).to_pylist()
        if [str(r['source_sample_id']) for r in records[:4]] != contract['probe_source_ids'] or len(records) != 64:
            raise ValueError('Diagnostic probe identities differ from original first four rows')
        prefixes = [encode_prefix(tokenizer, row, spec, max_seq_length=2048, enable_thinking=False) for row in records]
        report.update(loading=loading, adapter_state_integrity=state_integrity,
                      before=fingerprint(model), flags_before=capture_torch_numeric_backend(torch),
                      tuners_before=snapshot(discover_tuners()))
        write(folder / 'start.json', report)
        if condition in (1, 2):
            with torch.inference_mode():
                for prefix in prefixes:
                    ids = torch.tensor([prefix['input_ids']], device='cuda:0')
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    del output
        if condition == 2:
            model.train()
            collator = CompletionOnlyCollator(tokenizer)
            for row in records[:8]:
                example = tokenize_training_example(tokenizer, row['messages'], max_seq_length=2048, enable_thinking=False)
                batch = {k: v.to('cuda:0') for k, v in collator([example]).items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite read-only warmup loss')
                (loss / 8).backward()
                del loss, batch
            model.zero_grad(set_to_none=True)
        model.eval()
        report.update(after_warm=fingerprint(model), flags_after_warm=capture_torch_numeric_backend(torch),
                      tuners_after_warm=snapshot(discover_tuners()))
        write(folder / 'after_warm.json', report)
        report['native'] = native_probe(model, tokenizer, spec, records[:4], prefixes[:4])
        report['native_repeat'] = native_probe(model, tokenizer, spec, records[:4], prefixes[:4])
        report['tuners_after_native'] = snapshot(discover_tuners())
        report['fla_kkt_tuners'] = [name for name in discover_tuners() if 'kkt' in name.lower()]
        write(folder / 'native.json', report)
        tuners = discover_tuners()
        with fixed_selection(tuners):
            report['fixed'] = native_probe(model, tokenizer, spec, records[:4], prefixes[:4])
            report['fixed_repeat'] = native_probe(model, tokenizer, spec, records[:4], prefixes[:4])
            report['fixed_selections'] = snapshot(tuners)
        write(folder / 'fixed.json', report)
        report['restored_native'] = native_probe(model, tokenizer, spec, records[:4], prefixes[:4])
        report.update(after=fingerprint(model), flags_after=capture_torch_numeric_backend(torch),
                      tuners_restored=snapshot(discover_tuners()))
        report['parameters_unchanged'] = len({report[k]['parameter_sha256'] for k in ('before', 'after_warm', 'after')}) == 1
        report['buffers_unchanged'] = len({report[k]['buffer_sha256'] for k in ('before', 'after_warm', 'after')}) == 1
        report['gradient_flags_unchanged'] = len({report[k]['requires_grad_sha256'] for k in ('before', 'after_warm', 'after')}) == 1
        write(folder / 'complete.json', report)
        if not all(report[k] for k in ('parameters_unchanged', 'buffers_unchanged', 'gradient_flags_unchanged')):
            raise ValueError('Read-only diagnostic unexpectedly mutated model state')
        print('RHO_RELOAD_DIAGNOSTIC_CHILD_COMPLETE', condition, flush=True)
    except Exception as error:
        write(folder / 'failure.json', {'condition': condition, 'error': type(error).__name__ + ': ' + str(error),
                                      'model_updates': 0, 'original_training_reference_unavailable': True})
        raise


def worker(project, root):
    import torch
    from huggingface_hub import snapshot_download
    project, root = Path(project), Path(root)
    contract = read(project / 'diagnostic_contract.json')
    validate_contract(contract)
    validate_output_root(root, contract['parent_checkpoint'])
    root.mkdir(parents=True, exist_ok=True)
    source = read(project / 'source_manifest.json')
    for name, digest in source['files'].items():
        if sha(project / name) != digest:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    expected_versions = {'torch': '2.8.0+cu126', 'transformers': '5.14.1', 'peft': '0.18.0',
        'accelerate': '1.12.0', 'tokenizers': '0.22.2', 'safetensors': '0.8.0',
        'flash-linear-attention': '0.5.2', 'fla-core': '0.5.2', 'causal-conv1d': '1.7.0'}
    actual = {name: importlib.metadata.version(name) for name in expected_versions}
    write(root / 'compact/environment.json', {'packages': actual,
          'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    if actual != expected_versions or torch.cuda.device_count() != 4:
        raise ValueError('Diagnostic GPU/environment differs from the frozen stack')
    scratch = Path('/scratch') / ('rho-reload-diagnostic-' + uuid.uuid4().hex)
    scratch.mkdir()
    checkpoint, parent_receipt = stage_parent(project, scratch, contract)
    write(root / 'compact/parent_validation.json', parent_receipt)
    snapshot = read(project / 'source_snapshot.json')
    if (snapshot['model_name'], snapshot['model_revision']) != (contract['model_name'], contract['model_revision']):
        raise ValueError('Diagnostic model snapshot identity differs')
    cached = Path(snapshot_download(contract['model_name'], revision=contract['model_revision'], local_files_only=True))
    model = scratch / 'model'
    for name, digest in snapshot['files'].items():
        if sha(cached / name) != digest:
            raise ValueError('Frozen base snapshot changed: ' + name)
        target = model / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached / name, target)
        if sha(target) != digest:
            raise ValueError('Staged base snapshot changed: ' + name)
    write(root / 'prepared.json', {'checkpoint_path': str(checkpoint), 'model_path': str(model),
          'contract_sha256': sha(project / 'diagnostic_contract.json'), 'source_manifest_sha256': sha(project / 'source_manifest.json')})
    handles, processes = [], []
    try:
        for condition in range(4):
            stream = (root / f'condition-{condition}.log').open('w')
            handles.append(stream)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(condition), 'LOCAL_RANK': '0',
                   'TRITON_CACHE_DIR': str(scratch / f'triton-{condition}')}
            for key in ('RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            processes.append(subprocess.Popen([sys.executable, '-m',
                'new_interest.experiments.rho_reference_reload_diagnostic', '--project', str(project),
                '--root', str(root), '--condition', str(condition)], env=env, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 2700
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A zero-update diagnostic child failed; inspect its preserved reports')
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded diagnostic exceeded 45 minutes')
            print('RHO_RELOAD_DIAGNOSTIC_ACTIVE', [p.poll() for p in processes], flush=True)
            time.sleep(20)
        if any(p.returncode for p in processes):
            raise RuntimeError('Diagnostic child failed')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in handles:
            stream.close()
    reports = [read(root / f'condition-{i}/complete.json') for i in range(4)]
    summary = summarize(reports)
    summary.update(parent_marker_sha256=contract['parent_marker_sha256'], parent_endpoint=parent_receipt['endpoint'],
                   report_sha256={str(i): sha(root / f'condition-{i}/complete.json') for i in range(4)})
    write(root / 'compact/diagnostic_summary.json', summary)
    if not all(summary['same_starting_state'].values()):
        raise ValueError('Diagnostic conditions did not start from identical model state')
    if sha(Path(contract['parent_checkpoint']) / CHECKPOINT_MARKER) != contract['parent_marker_sha256']:
        raise ValueError('Original parent marker changed during diagnosis')
    check_endpoint(checkpoint)
    print('RHO_RELOAD_DIAGNOSTIC_COMPLETE ' + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--condition', type=int, choices=range(4))
    args = parser.parse_args()
    if args.condition is None:
        worker(args.project, args.root)
    else:
        child(args.project, args.root, args.condition)


if __name__ == '__main__':
    main()
