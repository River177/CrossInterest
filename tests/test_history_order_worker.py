import json
from pathlib import Path

import pytest

from new_interest.experiments.history_order_ablation import verify_bundle


def test_source_identity_gate_rejects_changed_and_missing_payload(tmp_path):
    (tmp_path / 'manifests').mkdir()
    source = tmp_path / 'source.py'
    source.write_text('frozen\n')
    from new_interest.data.build_history_order_ablation import sha256
    (tmp_path / 'manifests/history_order_bundle.json').write_text(json.dumps({'files': {'source.py': sha256(source)}}))
    assert verify_bundle(tmp_path)
    source.write_text('changed\n')
    with pytest.raises(ValueError): verify_bundle(tmp_path)
    source.unlink()
    with pytest.raises(FileNotFoundError): verify_bundle(tmp_path)


def test_stage_config_preserves_budget_and_binds_migration(tmp_path, monkeypatch):
    import sys
    import types
    from new_interest.experiments import history_order_ablation as worker
    baseline = types.ModuleType('new_interest.experiments.backbone_scaling')
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    for folder in ('experiments', 'sft'):
        (tmp_path / 'configs' / folder).mkdir(parents=True)
    (tmp_path / 'configs/experiments/history_order.json').write_text(json.dumps({'shuffled_data': 'shuffle', 'original_data': 'original'}))
    (tmp_path / 'configs/sft/history_order.json').write_text(json.dumps({'epochs': 3, 'seed': 42, 'learning_rate': 2e-5}))
    root = tmp_path / 'run'
    single = worker.stage_config(tmp_path, root, 'single')
    final = worker.stage_config(tmp_path, root, 'final')
    assert single['per_device_train_batch_size'] * single['gradient_accumulation_steps'] == 32
    assert 4 * final['per_device_train_batch_size'] * final['gradient_accumulation_steps'] == 32
    assert single['epochs'] == final['epochs'] == 3
    assert final['execution_migration']['source_checkpoint'] == str(root / 'single/checkpoints/checkpoint-500')
    for arm in ('A', 'C'):
        bound, _ = worker.bind(tmp_path, arm)
        assert str(bound.DEV).startswith('original/')
    for arm in ('B', 'D'):
        bound, _ = worker.bind(tmp_path, arm)
        assert str(bound.DEV).startswith('shuffle/')
    with pytest.raises(ValueError): worker.bind(tmp_path, 'extra')
    with pytest.raises(ValueError): worker.stage_config(tmp_path, root, 'other')
