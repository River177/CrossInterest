from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path

import pytest

from new_interest.training.checkpoint_persistence import (
    CHECKPOINT_MARKER,
    CheckpointValidationError,
    hydrate_checkpoint,
    prune_persistent_checkpoints,
    publish_checkpoint,
    select_latest_complete_checkpoint,
    validate_persistent_checkpoint,
)


def _identity(*, run_name: str = "test-run") -> dict[str, object]:
    return {
        "identity_version": "trainer-checkpoint-identity-v1",
        "run_name": run_name,
        "world_size": 1,
        "effective_global_batch_size": 32,
    }


def _write_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    files = {
        "trainer_state.json": json.dumps({"global_step": step}).encode(),
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "training_args.bin": b"arguments",
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"adapter",
        "rng_state.pth": b"rng",
    }
    for name, payload in files.items():
        (checkpoint / name).write_bytes(payload)
    return checkpoint


def test_publish_and_hydrate_complete_checkpoint(tmp_path: Path) -> None:
    identity = _identity()
    local = _write_checkpoint(tmp_path / "local", 100)
    persistent_root = tmp_path / "persistent"

    published = publish_checkpoint(local, persistent_root, identity)
    marker = validate_persistent_checkpoint(published, identity)
    assert marker["global_step"] == 100
    assert (published / CHECKPOINT_MARKER).is_file()

    hydrated = hydrate_checkpoint(
        published,
        tmp_path / "hydrated",
        identity,
    )
    assert hydrated.name == "checkpoint-100"
    assert (hydrated / "optimizer.pt").read_bytes() == b"optimizer"
    assert (hydrated / CHECKPOINT_MARKER).read_bytes() == (published / CHECKPOINT_MARKER).read_bytes()
    assert validate_persistent_checkpoint(hydrated, identity) == marker


def test_latest_selection_ignores_partial_checkpoint(
    tmp_path: Path,
) -> None:
    identity = _identity()
    persistent_root = tmp_path / "persistent"
    publish_checkpoint(
        _write_checkpoint(tmp_path / "local", 100),
        persistent_root,
        identity,
    )
    _write_checkpoint(persistent_root, 200)

    selected, diagnostics = select_latest_complete_checkpoint(
        persistent_root,
        identity,
    )
    assert selected is not None
    assert selected.name == "checkpoint-100"
    assert any("completion marker" in value.lower() for value in diagnostics)


def test_latest_selection_falls_back_from_corrupt_checkpoint(
    tmp_path: Path,
) -> None:
    identity = _identity()
    persistent_root = tmp_path / "persistent"
    publish_checkpoint(
        _write_checkpoint(tmp_path / "local-a", 100),
        persistent_root,
        identity,
    )
    newest = publish_checkpoint(
        _write_checkpoint(tmp_path / "local-b", 200),
        persistent_root,
        identity,
    )
    (newest / "optimizer.pt").write_bytes(b"corrupt")

    selected, diagnostics = select_latest_complete_checkpoint(
        persistent_root,
        identity,
    )
    assert selected is not None
    assert selected.name == "checkpoint-100"
    assert any("mismatch" in value.lower() for value in diagnostics)


def test_identity_mismatch_is_not_resumed(tmp_path: Path) -> None:
    persistent_root = tmp_path / "persistent"
    published = publish_checkpoint(
        _write_checkpoint(tmp_path / "local", 100),
        persistent_root,
        _identity(run_name="other-run"),
    )

    with pytest.raises(CheckpointValidationError, match="identity mismatch"):
        validate_persistent_checkpoint(published, _identity())
    selected, diagnostics = select_latest_complete_checkpoint(
        persistent_root,
        _identity(),
    )
    assert selected is None
    assert any("identity mismatch" in value for value in diagnostics)


def test_publish_refuses_different_committed_identity(
    tmp_path: Path,
) -> None:
    persistent_root = tmp_path / "persistent"
    publish_checkpoint(
        _write_checkpoint(tmp_path / "local-a", 100),
        persistent_root,
        _identity(run_name="first"),
    )
    with pytest.raises(
        CheckpointValidationError,
        match="different identity",
    ):
        publish_checkpoint(
            _write_checkpoint(tmp_path / "local-b", 100),
            persistent_root,
            _identity(run_name="second"),
        )


def test_publish_repairs_malformed_uncommitted_marker(
    tmp_path: Path,
) -> None:
    identity = _identity()
    persistent_root = tmp_path / "persistent"
    incomplete = _write_checkpoint(persistent_root, 100)
    (incomplete / CHECKPOINT_MARKER).write_text("{", encoding="utf-8")

    published = publish_checkpoint(
        _write_checkpoint(tmp_path / "local", 100),
        persistent_root,
        identity,
    )
    marker = validate_persistent_checkpoint(published, identity)
    assert marker["global_step"] == 100


def test_prune_removes_only_old_complete_same_identity(
    tmp_path: Path,
) -> None:
    identity = _identity()
    persistent_root = tmp_path / "persistent"
    for step in (100, 200, 300, 400):
        publish_checkpoint(
            _write_checkpoint(tmp_path / f"local-{step}", step),
            persistent_root,
            identity,
        )
    partial = _write_checkpoint(persistent_root, 500)
    different = publish_checkpoint(
        _write_checkpoint(tmp_path / "other", 600),
        persistent_root,
        _identity(run_name="other"),
    )

    removed = prune_persistent_checkpoints(
        persistent_root,
        identity,
        keep=3,
    )
    assert [path.name for path in removed] == ["checkpoint-100"]
    assert not (persistent_root / "checkpoint-100").exists()
    assert partial.exists()
    assert different.exists()


def _published_retention_checkpoints(tmp_path: Path):
    identity = _identity()
    persistent_root = tmp_path / "persistent"
    checkpoints = [
        publish_checkpoint(
            _write_checkpoint(tmp_path / f"local-{step}", step),
            persistent_root,
            identity,
        )
        for step in (400, 500, 600, 700)
    ]
    return identity, persistent_root, checkpoints


@pytest.mark.parametrize("disappearance", ["before_rmtree", "final_rmdir", "child_unlink"])
def test_prune_tolerates_paths_disappearing_during_real_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disappearance: str,
) -> None:
    """Drive real publish/prune/rmtree while another deletion wins the race."""
    identity, persistent_root, checkpoints = _published_retention_checkpoints(tmp_path)
    oldest, *retained = checkpoints
    fired = False
    if disappearance == "before_rmtree":
        real_rmtree = shutil.rmtree
        def competing_rmtree(path, *args, **kwargs):
            nonlocal fired
            if Path(path) == oldest and not fired:
                fired = True
                real_rmtree(path)
            return real_rmtree(path, *args, **kwargs)
        monkeypatch.setattr(shutil, "rmtree", competing_rmtree)
    elif disappearance == "final_rmdir":
        real_rmdir = os.rmdir
        def disappearing_prefix(path, *args, **kwargs):
            nonlocal fired
            if Path(path) == oldest and not fired:
                # All children have already been removed by the real rmtree.
                assert not list(oldest.iterdir())
                fired = True
                real_rmdir(path, *args, **kwargs)
            return real_rmdir(path, *args, **kwargs)
        monkeypatch.setattr(os, "rmdir", disappearing_prefix)
    else:
        real_unlink = os.unlink
        def competing_unlink(path, *args, **kwargs):
            nonlocal fired
            if Path(path).name == "optimizer.pt" and not fired:
                fired = True
                real_unlink(path, *args, **kwargs)
            return real_unlink(path, *args, **kwargs)
        monkeypatch.setattr(os, "unlink", competing_unlink)

    removed = prune_persistent_checkpoints(persistent_root, identity, keep=3)

    assert fired
    assert removed == [oldest]
    assert not oldest.exists()
    for checkpoint in retained:
        assert validate_persistent_checkpoint(checkpoint, identity)["global_step"] >= 500
    assert validate_persistent_checkpoint(retained[-1], identity)["global_step"] == 700


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_prune_does_not_suppress_permission_or_io_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int,
) -> None:
    identity, persistent_root, checkpoints = _published_retention_checkpoints(tmp_path)
    real_unlink = os.unlink
    def failing_unlink(path, *args, **kwargs):
        if Path(path).name == "optimizer.pt":
            raise OSError(error_number, "injected deletion failure", os.fspath(path))
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", failing_unlink)

    with pytest.raises(OSError) as caught:
        prune_persistent_checkpoints(persistent_root, identity, keep=3)

    assert caught.value.errno == error_number
    assert checkpoints[0].exists()
    assert validate_persistent_checkpoint(checkpoints[-1], identity)["global_step"] == 700
