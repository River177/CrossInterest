from __future__ import annotations

import json
from pathlib import Path

import pytest

from new_interest.experiments import rho_twofold_reference as worker


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def training_tree(tmp_path):
    local, target = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir()
    _json(local / 'run_summary.json', {'world_size': 4})
    for name in ('resolved_config.json', 'semantic_reload_reference.json',
                 'semantic_adapter_reload_audit.json', 'final_adapter/adapter_config.json',
                 'final_adapter/tokenizer_config.json', 'final_adapter/tokenizer.json',
                 'final_adapter/special_tokens_map.json'):
        _json(local / name, {'file': name})
    for rank in range(4):
        _json(local / f'rank-{rank}-training-runtime.json', {'rank': rank})
    (local / 'final_adapter/adapter_model.safetensors').write_bytes(b'closed adapter byte stream')
    (local / 'final_adapter/interest_token_mapping.parquet').write_bytes(b'frozen mapping byte stream')
    _json(local / 'checkpoints/checkpoint-2/trainer_state.json', {'global_step': 2})
    identity = {'protocol_sha256': 'p', 'runtime_sha256': 'r', 'phase': 'gate',
                'fold': 0, 'stage': 'ddp', 'train_sha256': 'actual-subset'}
    return local, target, identity


def test_publication_verifies_closed_files_and_excludes_live_checkpoints(training_tree):
    local, target, identity = training_tree
    manifest = worker.publish_training(local, target, identity)
    assert manifest == worker.verify_manifest(target, identity)
    assert not (target / 'checkpoints').exists()
    for name, receipt in manifest['files'].items():
        assert (target / name).read_bytes() == (local / name).read_bytes()
        assert worker.sha(target / name) == receipt['sha256']
    assert set(f'rank-{rank}-training-runtime.json' for rank in range(4)) <= set(manifest['files'])


def test_committed_same_identity_is_immutable_and_different_identity_rejected(training_tree):
    local, target, identity = training_tree
    original = worker.publish_training(local, target, identity)
    before = (target / 'final_adapter/adapter_model.safetensors').read_bytes()
    (local / 'final_adapter/adapter_model.safetensors').write_bytes(b'a new attempt must not overwrite a commit')
    assert worker.publish_training(local, target, identity) == original
    assert (target / 'final_adapter/adapter_model.safetensors').read_bytes() == before
    with pytest.raises(ValueError, match='identity differs'):
        worker.publish_training(local, target, {**identity, 'fold': 1})
    assert (target / 'final_adapter/adapter_model.safetensors').read_bytes() == before


def test_interrupted_publication_has_no_commit_and_can_resume(training_tree, monkeypatch):
    local, target, identity = training_tree
    original_copy = worker.shutil.copyfileobj
    def fail_after_partial_copy(source, destination, length):
        destination.write(source.read(3))
        raise OSError('interrupted remote copy')
    monkeypatch.setattr(worker.shutil, 'copyfileobj', fail_after_partial_copy)
    with pytest.raises(OSError, match='interrupted'):
        worker.publish_training(local, target, identity)
    assert not (target / 'artifact_manifest.json').exists()
    monkeypatch.setattr(worker.shutil, 'copyfileobj', original_copy)
    assert worker.publish_training(local, target, identity)['identity'] == identity


def test_committed_byte_corruption_is_not_overwritten(training_tree):
    local, target, identity = training_tree
    worker.publish_training(local, target, identity)
    path = target / 'final_adapter/adapter_model.safetensors'
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='bytes differ'):
        worker.publish_training(local, target, identity)
    assert path.read_bytes() == b'corrupted'


@pytest.mark.parametrize('name', ['run_summary.json', 'final_adapter/adapter_model.safetensors',
                                  'final_adapter/adapter_config.json', 'rank-3-training-runtime.json'])
def test_manifest_must_authenticate_files_used_by_training_acceptance(training_tree, name):
    local, target, identity = training_tree
    manifest = worker.publish_training(local, target, identity)
    del manifest['files'][name]
    _json(target / 'artifact_manifest.json', manifest)
    # The bytes are still on disk: existence alone must not satisfy acceptance.
    assert (target / name).exists()
    with pytest.raises(ValueError, match='omits|Unauthenticated'):
        worker.verify_manifest(target, identity)


@pytest.mark.parametrize('name', ['../outside', '/tmp/absolute-artifact', 'x/../outside', './ambiguous'])
def test_manifest_rejects_unsafe_or_noncanonical_paths(training_tree, name):
    local, target, identity = training_tree
    manifest = worker.publish_training(local, target, identity)
    manifest['files'][name] = {'bytes': 0, 'sha256': '0' * 64}
    _json(target / 'artifact_manifest.json', manifest)
    with pytest.raises(ValueError, match='Unsafe training manifest path'):
        worker.verify_manifest(target, identity)


def test_manifest_rejects_empty_inventory_and_unauthenticated_tokenizer_file(training_tree):
    local, target, identity = training_tree
    manifest = worker.publish_training(local, target, identity)
    _json(target / 'artifact_manifest.json', {**manifest, 'files': {}})
    with pytest.raises(ValueError, match='empty or malformed'):
        worker.verify_manifest(target, identity)
    _json(target / 'artifact_manifest.json', manifest)
    _json(target / 'final_adapter/added_tokens.json', {'unexpected': True})
    with pytest.raises(ValueError, match='Unauthenticated adapter'):
        worker.verify_manifest(target, identity)


def test_manifest_rejects_missing_bytes_and_symlink_to_outside(training_tree):
    local, target, identity = training_tree
    worker.publish_training(local, target, identity)
    path = target / 'final_adapter/tokenizer.json'
    path.unlink()
    with pytest.raises(ValueError, match='missing or unsafe'):
        worker.verify_manifest(target, identity)
    path.symlink_to(local / 'final_adapter/tokenizer.json')
    with pytest.raises(ValueError, match='missing or unsafe'):
        worker.verify_manifest(target, identity)


@pytest.mark.parametrize('stage,epochs,save_steps', [('single', 5, 10), ('ddp', 2, 2), ('full', 3, 100)])
def test_stage_config_preserves_base_recipe_and_exact_training_file(stage, epochs, save_steps):
    original = {'seed': 42, 'epochs': 3, 'save_steps': 100, 'per_device_train_batch_size': 1,
                'gradient_accumulation_steps': 8, 'semantic_id_loss_scope': 'full_vocabulary',
                'loss_reduction': 'token_mean', 'learning_rate': 2e-5}
    protocol = {'sft': original.copy(), 'model_revision': 'original-revision'}
    selected_train = Path('/remote/inputs') / ('fold0.parquet' if stage == 'full' else f'{stage}-train.parquet')
    cfg = worker.training_config(protocol, Path('/scratch/pretrained'), Path('/remote/mapping.parquet'),
                                 selected_train, Path('/scratch/train'), Path('/remote/checkpoints'), stage, 0)
    assert protocol['sft'] == original
    assert cfg['train_file'] == str(selected_train)
    assert cfg['model_name'] == '/scratch/pretrained'
    assert cfg['model_revision'] == 'original-revision'
    assert cfg.get('adapter_path') is None and cfg.get('sample_weight_file') is None
    assert cfg['resume_from_checkpoint'] == 'auto'
    assert cfg['persistent_checkpoint_dir'] == '/remote/checkpoints'
    assert cfg['epochs'] == epochs and cfg['save_steps'] == save_steps
    for key in ('semantic_id_loss_scope', 'loss_reduction', 'learning_rate',
                'per_device_train_batch_size', 'gradient_accumulation_steps'):
        assert cfg[key] == original[key]
