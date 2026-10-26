import errno
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('cir_storage_retry', Path(__file__).parents[1] / 'scripts/analysis/cir_storage_retry.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def test_eio_read_restarts_whole_operation():
    calls, sleeps = [], []
    def read_bytes():
        calls.append(1)
        if len(calls) < 3:
            raise OSError(errno.EIO, 'injected storage I/O')
        return b'exact checkpoint bytes'
    assert retry.retry_eio(read_bytes, delays=(1, 2), sleep=sleeps.append)() == b'exact checkpoint bytes'
    assert len(calls) == 3 and sleeps == [1, 2]


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'denied'), FileNotFoundError(errno.ENOENT, 'missing'), ValueError('hash mismatch')])
def test_non_eio_and_validation_fail_closed(error):
    calls = []
    def fail():
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        retry.retry_eio(fail, sleep=lambda _: pytest.fail('unexpected retry'))()
    assert len(calls) == 1


def test_eio_retry_exhaustion_fails():
    calls = []
    def fail():
        calls.append(1)
        raise OSError(errno.EIO, 'injected permanent I/O')
    with pytest.raises(OSError):
        retry.retry_eio(fail, delays=(0, 0), sleep=lambda _: None)()
    assert len(calls) == 3


def test_atomic_copy_retries_partial_eio_and_preserves_source(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/analysis'))
    import recover_cir_schedule as recovery
    import hashlib
    source, target = tmp_path/'source', tmp_path/'out'/'target'
    payload = b'checkpoint data' * 100
    source.write_bytes(payload)
    original = recovery.shutil.copyfileobj
    calls = []
    def flaky(src, dst, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            dst.write(src.read(12))
            raise OSError(errno.EIO, 'injected short copy')
        return original(src, dst, **kwargs)
    monkeypatch.setattr(recovery.shutil, 'copyfileobj', flaky)
    monkeypatch.setattr(recovery, 'retry_eio', lambda fn: retry.retry_eio(fn, delays=(0,), sleep=lambda _: None))
    expected={'size':len(payload), 'sha256':hashlib.sha256(payload).hexdigest()}
    recovery.copy_verified(source, target, expected)
    assert source.read_bytes() == target.read_bytes() == payload and len(calls)==2
    recovery.copy_verified(source, target, expected)
    assert len(calls)==2


def test_copy_hash_failure_never_commits(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/analysis'))
    import recover_cir_schedule as recovery
    source,target=tmp_path/'source',tmp_path/'target';source.write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash mismatch'):
        recovery.copy_verified(source,target,{'size':3,'sha256':'0'*64})
    assert not target.exists() and source.read_bytes()==b'bad'


def test_copy_refuses_to_overwrite_other_committed_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/analysis'))
    import recover_cir_schedule as recovery
    source,target=tmp_path/'source',tmp_path/'target';source.write_bytes(b'new');target.write_bytes(b'old')
    with pytest.raises(ValueError, match='Existing recovery file differs'):
        recovery.copy_verified(source,target,{'size':3,'sha256':'0'*64})
    assert target.read_bytes()==b'old'
