from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_MARKER = "checkpoint_complete.json"
CHECKPOINT_PROTOCOL_VERSION = "persistent-trainer-checkpoint-v1"


class CheckpointValidationError(RuntimeError):
    """Raised when a persisted checkpoint is incomplete or inconsistent."""


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def checkpoint_identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix):
        raise CheckpointValidationError(
            f"Checkpoint directory has an invalid name: {path.name}"
        )
    value = path.name[len(prefix) :]
    if not value.isdigit():
        raise CheckpointValidationError(
            f"Checkpoint directory has an invalid step: {path.name}"
        )
    return int(value)


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    marker = checkpoint_dir / CHECKPOINT_MARKER
    return sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path != marker
    )


def _validate_required_files(
    checkpoint_dir: Path,
    *,
    world_size: int,
) -> None:
    required = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "adapter_config.json",
    )
    missing = [
        name
        for name in required
        if not (checkpoint_dir / name).is_file()
        or (checkpoint_dir / name).stat().st_size <= 0
    ]
    adapter_candidates = (
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    if not any(
        (checkpoint_dir / name).is_file()
        and (checkpoint_dir / name).stat().st_size > 0
        for name in adapter_candidates
    ):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if world_size <= 1:
        rng_names = ("rng_state.pth",)
    else:
        rng_names = tuple(
            f"rng_state_{rank}.pth" for rank in range(world_size)
        )
    missing.extend(
        name
        for name in rng_names
        if not (checkpoint_dir / name).is_file()
        or (checkpoint_dir / name).stat().st_size <= 0
    )
    if missing:
        raise CheckpointValidationError(
            f"{checkpoint_dir}: missing or empty required files: "
            + ", ".join(missing)
        )


def _file_manifest(checkpoint_dir: Path) -> dict[str, dict[str, object]]:
    files = _checkpoint_files(checkpoint_dir)
    if not files:
        raise CheckpointValidationError(
            f"Checkpoint contains no files: {checkpoint_dir}"
        )
    return {
        path.relative_to(checkpoint_dir).as_posix(): {
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_file_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.partial-{uuid.uuid4().hex}"
    )
    try:
        with source.open("rb") as input_handle, temporary.open(
            "wb"
        ) as output_handle:
            shutil.copyfileobj(
                input_handle,
                output_handle,
                length=1024 * 1024,
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_marker(checkpoint_dir: Path) -> dict[str, Any]:
    marker_path = checkpoint_dir / CHECKPOINT_MARKER
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(
            f"Cannot read completion marker: {marker_path}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointValidationError(
            f"Completion marker is not an object: {marker_path}"
        )
    return payload


def validate_persistent_checkpoint(
    checkpoint_dir: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_marker(checkpoint_dir)
    if payload.get("protocol_version") != CHECKPOINT_PROTOCOL_VERSION:
        raise CheckpointValidationError(
            f"Unsupported checkpoint protocol: {checkpoint_dir}"
        )
    if payload.get("checkpoint_name") != checkpoint_dir.name:
        raise CheckpointValidationError(
            f"Checkpoint name mismatch: {checkpoint_dir}"
        )
    step = _checkpoint_step(checkpoint_dir)
    if int(payload.get("global_step", -1)) != step:
        raise CheckpointValidationError(
            f"Checkpoint step mismatch: {checkpoint_dir}"
        )
    identity = payload.get("identity")
    if identity != expected_identity:
        raise CheckpointValidationError(
            f"Checkpoint identity mismatch: {checkpoint_dir}"
        )
    expected_identity_hash = checkpoint_identity_sha256(expected_identity)
    if payload.get("identity_sha256") != expected_identity_hash:
        raise CheckpointValidationError(
            f"Checkpoint identity hash mismatch: {checkpoint_dir}"
        )
    world_size = int(expected_identity.get("world_size", 1))
    _validate_required_files(checkpoint_dir, world_size=world_size)
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise CheckpointValidationError(
            f"Checkpoint manifest is empty: {checkpoint_dir}"
        )
    for relative_name, expected in files.items():
        relative = Path(str(relative_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointValidationError(
                f"Unsafe checkpoint manifest path: {relative_name}"
            )
        if not isinstance(expected, dict):
            raise CheckpointValidationError(
                f"Invalid checkpoint manifest entry: {relative_name}"
            )
        path = checkpoint_dir / relative
        if not path.is_file():
            raise CheckpointValidationError(
                f"Checkpoint file is missing: {path}"
            )
        expected_size = int(expected.get("size", -1))
        if path.stat().st_size != expected_size:
            raise CheckpointValidationError(
                f"Checkpoint file size mismatch: {path}"
            )
        expected_hash = str(expected.get("sha256", ""))
        if _sha256_file(path) != expected_hash:
            raise CheckpointValidationError(
                f"Checkpoint file hash mismatch: {path}"
            )
    return payload


def publish_checkpoint(
    local_checkpoint_dir: Path,
    persistent_root: Path,
    identity: dict[str, Any],
) -> Path:
    """Publish a locally completed Trainer checkpoint with marker-last commit."""

    step = _checkpoint_step(local_checkpoint_dir)
    world_size = int(identity.get("world_size", 1))
    _validate_required_files(local_checkpoint_dir, world_size=world_size)
    source_manifest = _file_manifest(local_checkpoint_dir)
    destination = persistent_root / local_checkpoint_dir.name
    marker_path = destination / CHECKPOINT_MARKER
    if marker_path.exists():
        try:
            validate_persistent_checkpoint(destination, identity)
        except CheckpointValidationError:
            try:
                existing = _load_marker(destination)
            except CheckpointValidationError:
                existing = None
            if (
                existing is not None
                and existing.get("identity") != identity
            ):
                raise CheckpointValidationError(
                    "Refusing to replace a committed checkpoint with a "
                    f"different identity: {destination}"
                )
            marker_path.unlink(missing_ok=True)
        else:
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    marker_path.unlink(missing_ok=True)
    for relative_name, expected in source_manifest.items():
        source = local_checkpoint_dir / relative_name
        target = destination / relative_name
        _copy_file_bytes(source, target)
        if target.stat().st_size != int(expected["size"]):
            raise CheckpointValidationError(
                f"Published checkpoint file size mismatch: {target}"
            )
        if _sha256_file(target) != str(expected["sha256"]):
            raise CheckpointValidationError(
                f"Published checkpoint file hash mismatch: {target}"
            )
    marker = {
        "protocol_version": CHECKPOINT_PROTOCOL_VERSION,
        "checkpoint_name": destination.name,
        "global_step": step,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": checkpoint_identity_sha256(identity),
        "files": source_manifest,
    }
    _atomic_write_bytes(
        marker_path,
        json.dumps(marker, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    validate_persistent_checkpoint(destination, identity)
    return destination


def select_latest_complete_checkpoint(
    persistent_root: Path,
    identity: dict[str, Any],
) -> tuple[Path | None, list[str]]:
    """Select newest valid same-identity checkpoint, ignoring partial ones."""

    if not persistent_root.is_dir():
        return None, []
    candidates: list[tuple[int, Path]] = []
    diagnostics: list[str] = []
    for path in persistent_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = _checkpoint_step(path)
        except CheckpointValidationError as error:
            diagnostics.append(str(error))
            continue
        candidates.append((step, path))
    for _, path in sorted(candidates, reverse=True):
        try:
            validate_persistent_checkpoint(path, identity)
        except CheckpointValidationError as error:
            diagnostics.append(str(error))
            continue
        return path, diagnostics
    return None, diagnostics


def prune_persistent_checkpoints(
    persistent_root: Path,
    identity: dict[str, Any],
    *,
    keep: int,
) -> list[Path]:
    """Remove older validated checkpoints, tolerating deletion-time absence."""

    def on_delete_error(
        _function: Any, _path: str, exc_info: tuple[Any, Any, Any],
    ) -> None:
        error = exc_info[1]
        # Another deletion or a disappearing mounted prefix can win the race.
        # Only absence is benign; permission, I/O and all other errors still fail.
        if isinstance(error, FileNotFoundError) and error.errno == errno.ENOENT:
            return
        raise error

    if keep <= 0:
        raise ValueError("Persistent checkpoint retention must be positive")
    valid: list[tuple[int, Path]] = []
    if not persistent_root.is_dir():
        return []
    for path in persistent_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = _checkpoint_step(path)
            validate_persistent_checkpoint(path, identity)
        except CheckpointValidationError:
            continue
        valid.append((step, path))
    removed: list[Path] = []
    for _, path in sorted(valid, reverse=True)[keep:]:
        shutil.rmtree(path, onerror=on_delete_error)
        removed.append(path)
    return removed


def hydrate_checkpoint(
    persistent_checkpoint_dir: Path,
    local_checkpoint_root: Path,
    identity: dict[str, Any],
) -> Path:
    """Copy a validated persistent checkpoint into node-local scratch."""

    marker = validate_persistent_checkpoint(
        persistent_checkpoint_dir,
        identity,
    )
    destination = local_checkpoint_root / persistent_checkpoint_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = marker["files"]
    for relative_name in sorted(files):
        _copy_file_bytes(
            persistent_checkpoint_dir / relative_name,
            destination / relative_name,
        )
    world_size = int(identity.get("world_size", 1))
    _validate_required_files(destination, world_size=world_size)
    for relative_name, expected in files.items():
        path = destination / relative_name
        if path.stat().st_size != int(expected["size"]):
            raise CheckpointValidationError(
                f"Hydrated checkpoint file size mismatch: {path}"
            )
        if _sha256_file(path) != str(expected["sha256"]):
            raise CheckpointValidationError(
                f"Hydrated checkpoint file hash mismatch: {path}"
            )
    _copy_file_bytes(
        persistent_checkpoint_dir / CHECKPOINT_MARKER,
        destination / CHECKPOINT_MARKER,
    )
    validate_persistent_checkpoint(destination, identity)
    return destination
