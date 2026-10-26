import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.rho_reference_reload_diagnostic import (
    CONDITIONS, VERSION, check_endpoint, fingerprint, probe_difference, sha,
    stage_parent, summarize, tensor_record, validate_contract, paths_overlap, validate_output_root, native_probe,
)
from new_interest.training.checkpoint_persistence import CHECKPOINT_MARKER, CheckpointValidationError, publish_checkpoint


def contract(parent, train_sha='a' * 64):
    return {'version': VERSION, 'conditions': list(CONDITIONS), 'model_updates': 0, 'fold': 1,
        'original_training_probe_vectors_available': False,
        'probe_source_ids': ['a', 'b', 'c', 'd'], 'train_file': 'inputs/fold1.parquet',
        'parent_checkpoint': str(parent), 'parent_marker_sha256': 'b' * 64, 'parent_runtime_sha256': 'c' * 64,
        'model_revision': 'revision', 'model_name': 'Qwen/Qwen3.5-2B',
        'parent_identity': {'epochs': 3.0, 'world_size': 4, 'seed': 42, 'effective_global_batch_size': 32,
            'input_adapter_sha256': None, 'sample_weight_file_sha256': None,
            'loss_reduction': 'token_mean', 'semantic_id_loss_scope': 'full_vocabulary',
            'model_revision': 'revision', 'train_file_sha256': train_sha}}


@pytest.mark.parametrize('field,value', [
    ('model_updates', 1), ('original_training_probe_vectors_available', True),
    ('conditions', ['fresh']), ('probe_source_ids', ['a', 'a', 'c', 'd']),
    ('train_file', '../outside.parquet'), ('parent_marker_sha256', 'not-a-hash'),
])
def test_contract_refuses_training_or_replacing_original_reference(tmp_path, field, value):
    cfg = contract(tmp_path / 'checkpoint-1014')
    validate_contract(cfg)
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_contract(cfg)


def checkpoint_fixture(tmp_path):
    torch = pytest.importorskip('torch')
    project = tmp_path / 'project'
    (project / 'inputs').mkdir(parents=True)
    train = project / 'inputs/fold1.parquet'
    pq.write_table(pa.table({'id': list(range(10800))}), train)
    local = tmp_path / 'original/checkpoint-1014'
    local.mkdir(parents=True)
    (local / 'trainer_state.json').write_text(json.dumps({'global_step': 1014, 'epoch': 3.0}))
    torch.save({'state': {i: {'step': torch.tensor(1014.)} for i in range(373)}}, local / 'optimizer.pt')
    torch.save({'last_epoch': 1014}, local / 'scheduler.pt')
    for name in ['training_args.bin', 'adapter_config.json', 'adapter_model.safetensors',
                 *[f'rng_state_{i}.pth' for i in range(4)]]:
        (local / name).write_bytes(b'fixture')
    parent = tmp_path / 'persistent/checkpoint-1014'
    cfg = contract(parent, sha(train))
    publish_checkpoint(local, parent.parent, cfg['parent_identity'])
    cfg['parent_marker_sha256'] = sha(parent / CHECKPOINT_MARKER)
    return project, parent, cfg


def test_parent_hydration_verifies_all_bytes_and_keeps_original_immutable(tmp_path):
    project, parent, cfg = checkpoint_fixture(tmp_path)
    before = {p.name: sha(p) for p in parent.iterdir() if p.is_file()}
    copied, receipt = stage_parent(project, tmp_path / 'scratch', cfg)
    assert copied.name == 'checkpoint-1014'
    assert receipt['all_parent_files_hash_verified'] is True
    assert receipt['endpoint']['optimizer_state_entries'] == 373
    assert receipt['endpoint']['global_step'] == 1014
    assert before == {p.name: sha(p) for p in parent.iterdir() if p.is_file()}


def test_corrupt_parent_payload_is_rejected_before_hydration(tmp_path):
    project, parent, cfg = checkpoint_fixture(tmp_path)
    (parent / 'adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(CheckpointValidationError, match='hash mismatch|size mismatch'):
        stage_parent(project, tmp_path / 'scratch', cfg)
    assert not (tmp_path / 'scratch/parent').exists()


def test_optimizer_endpoint_cannot_be_shorter_than_final_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    _, parent, _ = checkpoint_fixture(tmp_path)
    torch.save({'last_epoch': 1013}, parent / 'scheduler.pt')
    with pytest.raises(ValueError, match='endpoint'):
        check_endpoint(parent)


def probes(value=0.):
    return [{'source_sample_id': name, 'input_ids_sha256': name + '-input',
             'native_sid_scores': [value] + [float(i) for i in range(1, 108)],
             'fp32_sid_scores': [value] + [float(i) for i in range(1, 108)]}
            for name in ('a', 'b', 'c', 'd')]


def reports():
    result = []
    for i in range(4):
        native = probes(.125 if i == 2 else 0.)
        result.append({'condition': i, 'model_updates': 0, 'contract_sha256': 'same',
            'original_training_reference_unavailable': True, 'parameters_unchanged': True,
            'buffers_unchanged': True, 'before': {'parameter_sha256': 'p', 'buffer_sha256': 'b', 'requires_grad_sha256': 'g'},
            'native': native, 'native_repeat': copy.deepcopy(native), 'restored_native': copy.deepcopy(native),
            'fixed': probes(), 'fixed_repeat': probes()})
    return result


def test_history_support_requires_identical_tensors_fixed_agreement_and_restoration():
    summary = summarize(reports())
    assert summary['selection_history_supported'] is True
    assert summary['native_cross_condition_max_abs'] == .125
    assert summary['fixed_cross_condition_max_abs'] == 0
    assert summary['original_training_reference_equivalence_established'] is False


def test_fresh_replica_variation_is_not_misattributed_to_warm_history():
    values = reports()
    values[3]['native'] = probes(.25)
    values[3]['native_repeat'] = probes(.25)
    values[3]['restored_native'] = probes(.25)
    summary = summarize(values)
    assert summary['fresh_replica_max_abs'] == .25
    assert summary['fresh_process_variation_observed'] is True
    assert summary['selection_history_supported'] is False


@pytest.mark.parametrize('mutation', ['state', 'repeat', 'fixed', 'restoration'])
def test_premature_kernel_history_conclusion_is_rejected(mutation):
    values = reports()
    if mutation == 'state':
        values[2]['before']['buffer_sha256'] = 'different'
    elif mutation == 'repeat':
        values[2]['native_repeat'] = probes(.3)
    elif mutation == 'fixed':
        values[2]['fixed'] = probes(.3)
    else:
        values[2]['restored_native'] = probes(.3)
    assert summarize(values)['selection_history_supported'] is False


def test_probe_comparison_requires_same_all_four_inputs_and_complete_sid_scores():
    left, right = probes(), probes()
    right[0]['input_ids_sha256'] = 'changed'
    with pytest.raises(ValueError, match='identities'):
        probe_difference(left, right)
    with pytest.raises(ValueError, match='four'):
        probe_difference(left[:3], probes())
    right = probes()
    right[0]['native_sid_scores'] = [1.]
    with pytest.raises(ValueError, match='108'):
        probe_difference(left, right)


def test_fingerprint_covers_nonpersistent_buffers_and_gradient_flags():
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(2, 2)
    model.register_buffer('inv_freq', torch.ones(2), persistent=False)
    model.register_buffer('counter', torch.tensor(0), persistent=False)
    before = fingerprint(model)
    model.inv_freq.add_(.5)
    after = fingerprint(model)
    assert before['parameter_sha256'] == after['parameter_sha256']
    assert before['buffer_sha256'] != after['buffer_sha256']
    model.weight.requires_grad_(False)
    assert fingerprint(model)['requires_grad_sha256'] != after['requires_grad_sha256']
    assert tensor_record(torch.tensor(1.))['shape'] == []


def test_output_root_guard_recognizes_shared_mount_aliases():
    parent = Path('/mnt/default/projects/task/original/checkpoints-full/checkpoint-1014')
    for root in ['/mnt/output/projects/task/original/diagnostic', '/mnt/default/projects/task',
                 '/mnt/output/projects/task/original', 'relative-output']:
        with pytest.raises(ValueError, match='separate absolute root'):
            validate_output_root(root, parent)
    validate_output_root('/mnt/output/projects/task/separate-diagnostic', parent)
    assert paths_overlap('/mnt/default/a', '/mnt/output/a/child')
    assert not paths_overlap('/mnt/default/a', '/mnt/output/b')


def test_native_probe_keeps_full_sequence_head_and_preserves_parameters():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 4)
            self.head = torch.nn.Linear(4, 128, bias=False)
            self.seen_shapes = []

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, attention_mask, use_cache):
            assert not use_cache
            hidden = self.embedding(input_ids)
            self.seen_shapes.append(list(hidden.shape))
            return SimpleNamespace(logits=self.head(hidden))

    model = TinyModel().to(torch.bfloat16)
    before = fingerprint(model)
    records = [{'source_sample_id': str(i), 'messages': [{'content': 'frozen prompt'}]} for i in range(4)]
    prefixes = [{'input_ids': [1] * (i+2)} for i in range(4)]
    values = native_probe(model, None, SimpleNamespace(token_ids=list(range(108))), records, prefixes)
    assert model.seen_shapes == [[1, i+2, 4] for i in range(4)]
    assert all(len(row['native_sid_scores']) == len(row['fp32_sid_scores']) == 108 for row in values)
    assert all(row['effective_sid_head']['dtype'] == 'torch.bfloat16' for row in values)
    assert all(row['native_full_sequence_head'] and row['padding'] == 'none-batch1' for row in values)
    assert before == fingerprint(model)
