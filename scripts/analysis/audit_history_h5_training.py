"""Accept original H5 training evidence after the independent profiled reload."""
import argparse
import math
from pathlib import Path

from new_interest.experiments.history_length_ablation import bind, read, write, require, sha256
from new_interest.training.execution_migration import validate_execution_transition, checkpoint_state_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source-training', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    project, source, root = args.project, args.source_training, args.root
    baseline, contract = bind(project, 5)
    receipt = read(source / 'source_readback.json')
    for name, item in receipt['files'].items():
        require(sha256(source / name) == item['sha256'], 'Original evidence hash mismatch')
    sft = source / 'final/sft'
    summary = read(sft / 'run_summary.json')
    require(sha256(sft / 'run_summary.json') == sha256(root / 'sft/run_summary.json'), 'Publication summary changed')
    require(sha256(sft / 'semantic_reload_reference.json') == sha256(root / 'sft/semantic_reload_reference.json'), 'Original reference changed')
    for key, expected in {'train_samples':21600, 'actual_optimizer_steps':2025, 'expected_optimizer_steps':2025,
                          'configured_epochs':3, 'world_size':4, 'effective_global_batch_size':32,
                          'loss_reduction':'token_mean', 'semantic_id_loss_scope':'full_vocabulary', 'seed':42}.items():
        require(summary[key] == expected, f'Training contract mismatch: {key}')
    require(math.isfinite(summary['train_metrics']['train_loss']), 'Nonfinite loss')
    require(summary['train_file_sha256'] == sha256(project / baseline.TRAIN), 'Training data changed')
    baseline.audit_resolved_recipe(read(sft / 'resolved_config.json'), summary['checkpoint_identity'],
        {**contract['recipe'], 'per_device_train_batch_size':8, 'gradient_accumulation_steps':1})
    for key in ('model_name', 'model_revision'):
        require(summary[key] == contract['models']['2b'][key], 'Model identity differs')
    require(summary['token_audit']['over_max_seq_length'] == 0, 'Overlength input')
    require(summary['semantic_token_drift']['all_input_rows_changed'] and summary['semantic_token_drift']['all_output_rows_changed'], 'Untrained SID rows')
    migration = read(sft / 'execution_migration_start.json')
    transition = validate_execution_transition(migration['source_identity'], summary['checkpoint_identity'])
    require(migration['status'] == 'passed' and migration['global_step'] == 500 and len(migration['rank_restores']) == 4, 'Migration failed')
    require(read(source / 'single/sft/initial_adapter_state.json')['state_sha256'] == contract['h50_initial_adapter_sha256'], 'Initialization differs')
    cursor = read(sft / 'training_sample_cursor.json')
    require(cursor['global_step'] == 2025 and cursor['consumed_global_samples'] == 64800 and cursor['all_rank_real_sample_order_exact'], 'Sample cursor failed')
    require(cursor['consumed_prefix_sha256'] == contract['h50_final_sample_prefix_sha256'], 'Sample order differs')
    attempts = list((sft / 'execution_attempts').glob('attempt-*'))
    require(bool(attempts), 'Missing execution attempts')
    for attempt in attempts:
        for rank in range(4):
            rng = read(attempt / f'rng_rank_{rank}.json')
            require(rng['status'] == 'passed' and rng['actual_rng_sha256'] == rng['expected_rng_sha256'], 'RNG mismatch')
            require(read(attempt / f'kernels_rank_{rank}.json')['is_fast_path_available'], 'Training kernel path differs')
    state = checkpoint_state_audit(args.checkpoint, summary['checkpoint_identity'])
    require(state['global_step'] == 2025 and state['max_steps'] == 2025, 'Final checkpoint cursor differs')
    # The reload verifier normalizes PEFT key names before hashing; compare the
    # actual saved files directly instead of crossing those hash conventions.
    import torch
    from safetensors.torch import load_file
    checkpoint_tensors = load_file(str(args.checkpoint / 'adapter_model.safetensors'))
    final_tensors = load_file(str(root / 'sft/final_adapter/adapter_model.safetensors'))
    require(set(checkpoint_tensors) == set(final_tensors), 'Checkpoint tensor inventory differs')
    require(all(torch.equal(checkpoint_tensors[k], final_tensors[k]) for k in final_tensors), 'Checkpoint and final adapter differ')
    write(root / 'original_training_accepted.json', {'status':'passed', 'steps':2025,'samples':21600,'epochs':3,
        'source_evidence_manifest_sha256':sha256(source / 'source_readback.json'), 'cursor':cursor,
        'checkpoint':state,'checkpoint_final_tensors_exact':True,'transition':transition,'adapter':baseline.audit_adapter(project,'2b',root / 'sft'),
        'reload':baseline.audit_reload(project,'2b',root / 'sft')})
    print('H5_ORIGINAL_TRAINING_ACCEPTED', flush=True)


if __name__ == '__main__':
    main()
