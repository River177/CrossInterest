"""Isolated native Trainer endpoint reconstruction and bounded final-25 replay."""
import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def tensors(model):
    import torch
    result = {}
    for kind, entries in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in entries:
            x = value.detach().cpu().contiguous()
            result[name] = {'kind': kind, 'shape': list(x.shape), 'dtype': str(x.dtype),
                            'sha256': hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def check_diagnostic_step(parent_step, step):
    if parent_step != 2000 or not 2000 <= step < 2025:
        raise RuntimeError('Diagnostic attempted an unauthorized optimizer update')


def stage_checkpoint(source, destination, marker_sha256):
    """Stream each immutable file once; deserialize only from node-local storage."""
    marker_bytes = (source / 'checkpoint_complete.json').read_bytes()
    if hashlib.sha256(marker_bytes).hexdigest() != marker_sha256:
        raise ValueError('Source checkpoint marker changed')
    marker = json.loads(marker_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    for name, info in marker['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe checkpoint member')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha(target) == info['sha256']:
            continue
        temporary = target.with_name(target.name + '.partial')
        digest = hashlib.sha256()
        length = 0
        with (source / name).open('rb') as incoming, temporary.open('wb') as outgoing:
            for block in iter(lambda: incoming.read(4 * 1024 * 1024), b''):
                outgoing.write(block)
                digest.update(block)
                length += len(block)
        if digest.hexdigest() != info['sha256'] or length != info['size']:
            raise ValueError('Checkpoint member changed: ' + name)
        temporary.replace(target)
    (destination / 'checkpoint_complete.json').write_bytes(marker_bytes)
    return marker


def train_mode(config_file, parent_step):
    import torch
    from new_interest.training import train_sft as core, execution_migration as migration
    from new_interest.training.semantic_id_tokens import prepare_semantic_adapter_base
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    config = read(config_file)
    single_gpu_probe = bool(config.pop('_diagnostic_single_gpu', False))
    output = Path(config['output_dir'])
    factory = migration.make_execution_migration_trainer

    def guarded_factory(base, runtime):
        native = factory(base, runtime)
        class Guarded(native):
            def training_step(self, model, inputs, num_items_in_batch=None):
                step = int(self.state.global_step)
                check_diagnostic_step(parent_step, step)
                return super().training_step(model, inputs, num_items_in_batch)
        return Guarded

    migration.make_execution_migration_trainer = guarded_factory
    original_capture = core.capture_reload_reference

    def captured(model, tokenizer, train_file, spec, **kwargs):
        # This call is the original post-Trainer reference path, before saving.
        reference = original_capture(model, tokenizer, train_file, spec, **kwargs)
        before = tensors(model)
        adapter = output / 'diagnostic_adapter'
        model.save_pretrained(adapter, save_embedding_layers=False)
        tokenizer.save_pretrained(adapter)
        shutil.copyfile(spec.mapping_file, adapter / 'interest_token_mapping.parquet')
        after_save = original_capture(model, tokenizer, train_file, spec, **kwargs)
        fresh = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['model_revision'],
                                                    torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        prepare_semantic_adapter_base(fresh, spec, initialization_mode='mean_norm_matched')
        fresh = PeftModel.from_pretrained(fresh, adapter)
        fresh.to(model.device)
        fresh.eval()
        restored = original_capture(fresh, tokenizer, train_file, spec, **kwargs)
        fresh_tensors = tensors(fresh)
        differences = {name: {'trainer': before.get(name), 'fresh': fresh_tensors.get(name)}
                       for name in before.keys() | fresh_tensors.keys() if before.get(name) != fresh_tensors.get(name)}
        historical = read(Path(os.environ['H5_SOURCE_ROOT']) / 'final/sft/semantic_reload_reference.json')

        def compare(a, b):
            assert [r['prompt_sha256'] for r in a['samples']] == [r['prompt_sha256'] for r in b['samples']]
            return max(abs(x-y) for ra, rb in zip(a['samples'], b['samples'])
                       for x, y in zip(ra['restricted_logits'], rb['restricted_logits']))

        report = {'parent_step': parent_step, 'optimizer_updates': 2025-parent_step,
                  'trainer_vs_original_reference_max_error': compare(reference, historical),
                  'fresh_vs_original_reference_max_error': compare(restored, historical),
                  'trainer_vs_fresh_max_error': compare(reference, restored),
                  'before_vs_after_save_max_error': compare(reference, after_save),
                  'state_differences': differences}
        write(output / 'trainer_reference.json', reference)
        write(output / 'fresh_reference.json', restored)
        write(output / 'trainer_tensor_manifest.json', before)
        write(output / 'fresh_tensor_manifest.json', fresh_tensors)
        write(output / 'comparison.json', report)
        print('H5_NATIVE_TRAINER_COMPARISON', json.dumps(report), flush=True)
        del fresh
        gc.collect()
        torch.cuda.empty_cache()
        return reference

    core.capture_reload_reference = captured
    def zero_update_factory(base):
        class ZeroUpdate(base):
            def training_step(self, *args, **kwargs):
                raise RuntimeError('Single-GPU endpoint diagnostic forbids all training updates')
        return ZeroUpdate
    summary = core.train(config, trainer_class_transform=zero_update_factory if single_gpu_probe else None)
    assert summary['actual_optimizer_steps'] == 2025


def orchestrate(project, output):
    from new_interest.training.execution_migration import checkpoint_state_audit
    from safetensors.torch import load_file
    import torch
    spec = read(project / 'trainer_diagnostic_source.json')
    source = Path('/mnt/default') / spec['results_prefix']
    os.environ['H5_SOURCE_ROOT'] = str(source)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for parent_step in (2025, 2000):
        case = 'zero' if parent_step == 2025 else 'replay25'
        stage = Path('/scratch/h5-native-diagnostic') / case
        stage.mkdir(parents=True, exist_ok=True)
        original_checkpoint = source / f'final/checkpoints/checkpoint-{parent_step}'
        checkpoint = stage / f'parent/checkpoint-{parent_step}'
        print('HISTORY_DIAGNOSTIC_PHASE', case, 'copy_checkpoint', flush=True)
        marker = stage_checkpoint(original_checkpoint, checkpoint, spec['checkpoint_marker_sha256'][str(parent_step)])
        print('HISTORY_DIAGNOSTIC_PHASE', case, 'audit_local_checkpoint', flush=True)
        checkpoint_state_audit(checkpoint, marker['identity'])
        write(output / f'{case}_staging.json', {'checkpoint_step': parent_step, 'files': len(marker['files']),
                                               'marker_sha256': sha(checkpoint / 'checkpoint_complete.json')})
        sft = stage / 'sft'
        sft.mkdir(exist_ok=True)
        shutil.copyfile(source / 'final/sft/execution_migration_start.json', sft / 'execution_migration_start.json')
        config = read(source / 'final/training_config.json')
        for field in ('train_file', 'semantic_id_token_mapping_file'):
            relative = Path(config[field]).relative_to('/scratch/amlt_code')
            config[field] = str(project / relative)
        config.update(output_dir=str(sft), checkpoint_dir=str(stage / 'checkpoints'),
                      resume_from_checkpoint=str(checkpoint), save_strategy='no',
                      execution_migration={'source_checkpoint': str(source / 'single/checkpoints/checkpoint-500'),
                          'source_stage_artifacts': str(source / 'single/source-stage'),
                          'cursor_contract_file': str(source / 'single/cursor_contract.json')})
        config.pop('persistent_checkpoint_dir', None)
        write(stage / 'config.json', config)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '4',
                   __file__, '--train-config', str(stage / 'config.json'), '--parent-step', str(parent_step)]
        print('HISTORY_DIAGNOSTIC_PHASE', case, 'native_trainer', flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            # Preserve compact evidence before surfacing failure. Original checkpoints remain untouched.
            for p in sft.rglob('*.json'):
                if p.stat().st_size < 500000:
                    dest = output / case / p.relative_to(sft)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dest)
            raise RuntimeError(f'Native diagnostic {case} failed with exit {completed.returncode}')
        report = read(sft / 'comparison.json')
        old = load_file(str(source / 'final/sft/final_adapter/adapter_model.safetensors'))
        new = load_file(str(sft / 'final_adapter/adapter_model.safetensors'))
        assert old.keys() == new.keys()
        changed = [k for k in old if not torch.equal(old[k], new[k])]
        report['adapter_exactly_matches_original_final'] = not changed
        report['different_adapter_tensors'] = changed
        del old, new
        gc.collect()
        for p in sft.rglob('*.json'):
            if p.stat().st_size < 500000:
                dest = output / case / p.relative_to(sft)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
        write(output / case / 'comparison.json', report)
        results[case] = report
        write(output / 'progress.json', results)
        # Replay is only needed if endpoint reconstruction cannot expose the divergence.
        if parent_step == 2025 and report['trainer_vs_fresh_max_error'] != 0:
            break
    write(output / 'diagnostic_complete.json', {'status': 'diagnostic_complete_not_acceptance', 'cases': results,
                                               'original_outputs_unchanged': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-config', type=Path)
    parser.add_argument('--parent-step', type=int, choices=(2000, 2025))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        if args.train_config:
            train_mode(args.train_config, args.parent_step)
        else:
            orchestrate(Path(os.environ['AMLT_CODE_DIR']), args.output)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
