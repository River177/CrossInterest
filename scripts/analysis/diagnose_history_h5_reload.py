"""Read-only GPU differential probe of the completed H5 adapter and reference."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def probe(case, adapter, reference, mapping, output):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from new_interest.training.verify_semantic_id_token_adapter import verify_semantic_id_token_adapter
    assert q.is_fast_path_available
    if case == 'fast_seed42':
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    if case in ('torch_all', 'torch_delta'):
        q.chunk_gated_delta_rule = q.fused_recurrent_gated_delta_rule = None
    if case in ('torch_all', 'torch_conv'):
        q.causal_conv1d_fn = q.causal_conv1d_update = None
    if case in ('torch_all', 'torch_norm'):
        q.FusedRMSNormGated = None
    if case == 'torch_all':
        q.is_fast_path_available = False
    original = torch.isclose
    captures = []

    def capture(actual, expected, *args, **kwargs):
        if tuple(actual.shape) == tuple(expected.shape) == (108,):
            captures.append({'actual': actual.detach().cpu().tolist(), 'expected': expected.detach().cpu().tolist()})
        return original(actual, expected, *args, **kwargs)

    torch.isclose = capture
    error = None
    try:
        verify_semantic_id_token_adapter('Qwen/Qwen3.5-2B', adapter, mapping, reference,
                                         output / f'{case}-audit.json',
                                         model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc', atol=0, rtol=0)
    except ValueError as exc:
        if not str(exc).startswith('Adapter reload logits mismatch:'):
            raise
        error = 'Adapter reload logits mismatch'
    finally:
        torch.isclose = original
    assert len(captures) == 4
    write(output / f'{case}-logits.json', {'case': case, 'reference_sha256': digest(reference), 'error': error, 'captures': captures})
    print('H5_RELOAD_DIAGNOSTIC_CASE', case, 'exact' if error is None else 'mismatch', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    project = Path(os.environ['AMLT_CODE_DIR'])
    mapping = project / 'data/training/mind_history_length_ablation_20260915_v1/h5/interest_token_mapping.parquet'
    if args.case:
        probe(args.case, args.adapter, args.reference, mapping, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((project / 'diagnostic_source.json').read_text())
    source = Path('/mnt/default') / spec['results_prefix']
    checkpoint = source / 'final/checkpoints/checkpoint-2025'
    marker = json.loads((checkpoint / 'checkpoint_complete.json').read_text())
    assert digest(checkpoint / 'checkpoint_complete.json') == spec['checkpoint_marker_sha256']
    assert marker['global_step'] == 2025
    verified = {}
    for name, info in marker['files'].items():
        assert digest(checkpoint / name) == info['sha256'], name
        verified[name] = info['sha256']
    scratch = Path('/scratch/h5-reload-diagnostic')
    scratch.mkdir(parents=True, exist_ok=True)
    adapter = scratch / 'adapter'
    shutil.copytree(source / 'final/sft/final_adapter', adapter, dirs_exist_ok=True)
    reference = scratch / 'reference.json'
    shutil.copyfile(source / 'final/sft/semantic_reload_reference.json', reference)
    assert digest(reference) == spec['reference_sha256']
    import torch
    from safetensors.torch import load_file
    left = load_file(str(checkpoint / 'adapter_model.safetensors'))
    right = load_file(str(adapter / 'adapter_model.safetensors'))
    assert left.keys() == right.keys() and len(left) == 373
    assert all(torch.equal(left[k], right[k]) for k in left)
    del left, right
    gc.collect()
    write(args.output / 'source_readback.json', {'checkpoint_step': 2025, 'checkpoint_files_verified': verified,
          'final_adapter_matches_checkpoint_tensors': True, 'tensor_count': 373, 'optimizer_updates': 0,
          'reference_sha256': digest(reference), 'adapter_sha256': digest(adapter / 'adapter_model.safetensors')})
    for case in ('fast_a', 'fast_b', 'fast_seed42', 'torch_all', 'torch_delta', 'torch_conv', 'torch_norm'):
        subprocess.run([sys.executable, __file__, '--case', case, '--adapter', str(adapter),
                        '--reference', str(reference), '--output', str(args.output)], check=True)
    results = {}
    for case in ('fast_a', 'fast_b', 'fast_seed42', 'torch_all', 'torch_delta', 'torch_conv', 'torch_norm'):
        data = json.loads((args.output / f'{case}-logits.json').read_text())
        errors = [abs(a-b) for row in data['captures'] for a,b in zip(row['actual'],row['expected'])]
        results[case] = {'exact': max(errors) == 0, 'max_absolute_difference': max(errors)}
    write(args.output / 'diagnostic_complete.json', {'status': 'diagnostic_complete_not_training_acceptance', 'cases': results,
                                                   'original_artifacts_modified': False, 'optimizer_updates': 0})
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
