import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('h5_native_diagnostic', Path(__file__).parents[1] / 'scripts/analysis/diagnose_history_h5_trainer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_step_guard_forbids_zero_case_and_exceeding_replay_budget():
    for step in range(2000, 2025):
        module.check_diagnostic_step(2000, step)
    for parent, step in ((2025, 2025), (2025, 2000), (2000, 1999), (2000, 2025), (0, 0)):
        with pytest.raises(RuntimeError):
            module.check_diagnostic_step(parent, step)


def test_manifest_detects_scalar_buffer_and_parameter_changes():
    torch = pytest.importorskip('torch')
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    model.register_buffer('scalar', torch.tensor(3.0, dtype=torch.bfloat16))
    before = module.tensors(model)
    assert before['scalar']['shape'] == []
    with torch.no_grad():
        model.scalar.add_(1)
    after = module.tensors(model)
    assert before['scalar']['sha256'] != after['scalar']['sha256']
    assert before['weight'] == after['weight']
    with torch.no_grad():
        model.weight.add_(1)
    assert module.tensors(model)['weight']['sha256'] != after['weight']['sha256']


def test_stream_copy_verifies_bytes_before_checkpoint_is_committed(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    payload = b'checkpoint tensor bytes'
    (source / 'optimizer.pt').write_bytes(payload)
    marker = {'files': {'optimizer.pt': {'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}}}
    encoded = json.dumps(marker).encode()
    (source / 'checkpoint_complete.json').write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    dest = tmp_path / 'valid'
    module.stage_checkpoint(source, dest, digest)
    assert (dest / 'optimizer.pt').read_bytes() == payload
    assert (dest / 'checkpoint_complete.json').read_bytes() == encoded
    (source / 'optimizer.pt').write_bytes(b'corrupt')
    broken = tmp_path / 'broken'
    with pytest.raises(ValueError):
        module.stage_checkpoint(source, broken, digest)
    assert not (broken / 'checkpoint_complete.json').exists()
