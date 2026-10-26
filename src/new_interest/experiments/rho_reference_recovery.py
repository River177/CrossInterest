"""Recover the finished single-GPU gate without replaying its twenty updates.

The old runtime did not produce an initial-state callback audit. Recovery can
verify the committed endpoint and a newly reconstructed initialization; it must
never present the latter as an observation of the original training start.
All checkpoint hydration and tensor verification run on the remote worker.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import pyarrow.parquet as pq

from new_interest.training.checkpoint_persistence import (
    CHECKPOINT_MARKER, hydrate_checkpoint, validate_persistent_checkpoint,
)
from new_interest.training.train_sft import _training_checkpoint_identity


VERSION = "rho-single-final-checkpoint-zero-update-recovery-v1"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {name}")
    return value


def _overlap(left: Path, right: Path) -> bool:
    def canonical(path: Path) -> Path:
        resolved = path.resolve()
        # The frozen AMLT configuration mounts this same amulet container
        # under both names. Different mount spellings must not permit writes
        # back into a checkpoint that was bound read-only.
        for mount in (Path('/mnt/default'), Path('/mnt/output')):
            if resolved.is_relative_to(mount):
                return Path('/__rho_shared_amulet_storage__') / resolved.relative_to(mount)
        return resolved
    left, right = canonical(left), canonical(right)
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _loss_history(state: dict) -> list[dict]:
    history = [{"step": int(row["step"]), "loss": float(row["loss"])}
               for row in state["log_history"] if "step" in row and "loss" in row]
    if [row["step"] for row in history] != list(range(1, 21)):
        raise ValueError("Parent must retain the original twenty per-update loss records")
    if not all(math.isfinite(row["loss"]) for row in history):
        raise ValueError("Parent loss history is nonfinite")
    return history


def _check_hydrated_files(prepared: dict) -> None:
    local = Path(prepared["resume_from_checkpoint"])
    for name, item in prepared["checkpoint_files"].items():
        path = local / name
        if not path.is_file() or path.stat().st_size != item["size"] or _sha(path) != item["sha256"]:
            raise ValueError(f"Hydrated parent checkpoint bytes changed: {name}")


def _optimizer_endpoint(local: Path) -> dict:
    import torch
    optimizer = torch.load(local / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(local / "scheduler.pt", map_location="cpu", weights_only=True)
    entries = optimizer.get("state")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Parent optimizer state is missing")
    steps = [float(entry["step"]) for entry in entries.values()]
    if not all(math.isfinite(step) and step == 20 for step in steps):
        raise ValueError("Parent optimizer did not finish exactly twenty updates")
    if scheduler.get("last_epoch") != 20:
        raise ValueError("Parent scheduler did not finish exactly twenty updates")
    return {"optimizer_state_entries": len(entries), "optimizer_step_min": 20,
            "optimizer_step_max": 20, "scheduler_last_epoch": 20}


def prepare_single_recovery(project: Path | str, config: dict,
                            local_checkpoint_root: Path | str) -> dict:
    """Validate the bound old checkpoint and return an explicit resume path.

    The caller must set ``config['resume_from_checkpoint']`` to the returned
    path, not ``auto``. Its persistent checkpoint destination stays in the new
    job; neither this helper nor the resumed run may publish into the parent.
    """
    project, local_root = Path(project), Path(local_checkpoint_root)
    binding_path = project / "recovery_parent.json"
    binding = _read(binding_path)
    parent = Path(binding["checkpoint_path"])
    if not parent.is_absolute() or parent.name != "checkpoint-20":
        raise ValueError("Recovery parent must be an absolute checkpoint-20 path")
    marker_sha = _digest(binding["checkpoint_marker_sha256"], "checkpoint marker SHA-256")
    runtime_sha = _digest(binding.get("runtime_sha256", binding.get("old_runtime_sha256")), "old runtime SHA-256")
    if "runtime_sha256" in binding and "old_runtime_sha256" in binding and binding["runtime_sha256"] != binding["old_runtime_sha256"]:
        raise ValueError("Conflicting old runtime bindings")
    if _sha(parent / CHECKPOINT_MARKER) != marker_sha:
        raise ValueError("Parent checkpoint marker differs from frozen recovery binding")
    if not local_root.is_absolute() or _overlap(local_root, parent.parent):
        raise ValueError("Recovery scratch must be separate from the immutable parent checkpoint root")
    persistent = config.get("persistent_checkpoint_dir")
    if not persistent or _overlap(Path(persistent), parent.parent):
        raise ValueError("Resumed checkpoint publication must use a separate new-job root")
    for field in ("adapter_path", "sample_weight_file", "mera_relations_file", "max_samples"):
        if config.get(field) is not None:
            raise ValueError(f"Single gate recovery forbids {field}")
    train = Path(config["train_file"])
    identity = _training_checkpoint_identity(config, train_file=train, sample_weight_file=None, adapter_path=None)
    expected = {"world_size": 1, "epochs": 5.0, "seed": 42,
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
                "effective_global_batch_size": 8, "input_adapter_sha256": None,
                "sample_weight_file_sha256": None, "loss_reduction": "token_mean",
                "semantic_id_loss_scope": "full_vocabulary", "enable_thinking": False}
    if any(identity.get(key) != value for key, value in expected.items()) or pq.read_metadata(train).num_rows != 32:
        raise ValueError("Recovery requires the unchanged single-GPU, 32-row, five-epoch native SFT recipe")
    marker = validate_persistent_checkpoint(parent, identity)
    required_names = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "training_args.bin",
                      "adapter_config.json", "adapter_model.safetensors", "rng_state.pth"}
    if required_names.difference(marker["files"]):
        raise ValueError("Parent manifest omits required checkpoint provenance files")
    state = _read(parent / "trainer_state.json")
    if state.get("global_step") != 20 or state.get("epoch") != 5.0:
        raise ValueError("Parent Trainer state is not the completed single gate")
    history = _loss_history(state)
    local = hydrate_checkpoint(parent, local_root, identity)
    prepared = {"version": VERSION, "parent_checkpoint": str(parent),
                "parent_checkpoint_marker_sha256": marker_sha, "parent_runtime_sha256": runtime_sha,
                "recovery_binding_sha256": _sha(binding_path), "checkpoint_identity": identity,
                "checkpoint_files": marker["files"], "resume_from_checkpoint": str(local),
                "parent_global_step": 20, "parent_epoch": 5.0, "parent_loss_history": history,
                "original_initial_runtime_audit_missing": True,
                "initial_state_provenance": "reconstructed_before_resume_not_original_start_observation",
                "parent_storage_mutated": False, "large_artifact_policy": "remote_only"}
    _check_hydrated_files(prepared)
    prepared["parent_optimizer_endpoint"] = _optimizer_endpoint(local)
    if _sha(parent / CHECKPOINT_MARKER) != marker_sha:
        raise ValueError("Parent commit changed during hydration")
    return prepared


def validate_single_recovery(prepared: dict, training_output: Path | str,
                             receipt_path: Path | str) -> dict:
    """Accept zero-update recovery, without reconstructing a historical audit."""
    import torch
    from safetensors.torch import load_file
    output, receipt_path = Path(training_output), Path(receipt_path)
    if prepared.get("version") != VERSION:
        raise ValueError("Unsupported single-gate recovery preparation")
    parent = Path(prepared["parent_checkpoint"])
    if _overlap(receipt_path, parent.parent):
        raise ValueError("Recovery receipt must not modify the immutable parent checkpoint root")
    if _sha(parent / CHECKPOINT_MARKER) != prepared["parent_checkpoint_marker_sha256"]:
        raise ValueError("Parent commit changed after hydration")
    # No operation here changes the parent or hydrates over a source checkpoint.
    _check_hydrated_files(prepared)
    local = Path(prepared["resume_from_checkpoint"])
    endpoint = _optimizer_endpoint(local)
    if endpoint != prepared["parent_optimizer_endpoint"]:
        raise ValueError("Hydrated optimizer endpoint differs from validated parent")
    summary = _read(output / "run_summary.json")
    if Path(str(summary.get("resumed_from_checkpoint", ""))).resolve() != local.resolve():
        raise ValueError("Training did not resume the explicitly hydrated parent")
    if (summary.get("train_file_sha256") != prepared["checkpoint_identity"]["train_file_sha256"]
            or summary.get("checkpoint_identity") != prepared["checkpoint_identity"]
            or summary.get("input_adapter") is not None or summary.get("sample_weight_file") is not None):
        raise ValueError("Recovered training provenance differs from its parent")
    if summary.get("loss_history") != prepared["parent_loss_history"]:
        raise ValueError("Recovery must retain the original parent loss history without new loss steps")
    audit = _read(output / "rank-0-training-runtime.json")
    if (audit.get("rank"), audit.get("world_size"), audit.get("train_samples"),
            audit.get("expected_optimizer_steps"), audit.get("completed_as_configured")) != (0, 1, 32, 20, True):
        raise ValueError("Recovered runtime audit does not describe the completed single gate")
    begin, end = audit["train_begin"], audit["train_end"]
    if begin.get("resumed_from_checkpoint") is not True or begin.get("fresh_optimization_state") is not False:
        raise ValueError("Recovery must observe restored, not freshly empty, optimizer state")
    expected_endpoint = {"global_step": 20, "epoch": 5.0, **endpoint}
    for name, observed in (("begin", begin), ("end", end)):
        if any(observed.get(key) != value for key, value in expected_endpoint.items()):
            raise ValueError(f"Recovery {name} optimizer/scheduler cursor differs; zero new updates required")
    initial = audit["initial_state"]
    _digest(initial.get("state_sha256"), "reconstructed initialization SHA-256")
    if initial.get("lora_b_all_zero") is not True:
        raise ValueError("Recovery did not reconstruct a fresh LoRA initialization before restore")
    checkpoint_state = load_file(str(local / "adapter_model.safetensors"))
    final_path = output / "final_adapter/adapter_model.safetensors"
    final_state = load_file(str(final_path))
    if not checkpoint_state or set(checkpoint_state) != set(final_state):
        raise ValueError("Recovered adapter tensor inventory differs from parent")
    for key, original in checkpoint_state.items():
        actual = final_state[key]
        if (original.dtype != actual.dtype or original.shape != actual.shape
                or not torch.equal(original, actual) or not torch.isfinite(actual).all().item()):
            raise ValueError(f"Recovered adapter tensor differs from parent: {key}")
    reload = _read(output / "semantic_adapter_reload_audit.json")
    if (reload.get("all_close") is not True
            or reload.get("adapter_state_integrity", {}).get("all_tensors_exact") is not True):
        raise ValueError("Recovered adapter fresh GPU reload was not exact")
    receipt = {"accepted": True, "version": VERSION,
               "parent_checkpoint": str(parent),
               "parent_checkpoint_marker_sha256": prepared["parent_checkpoint_marker_sha256"],
               "parent_runtime_sha256": prepared["parent_runtime_sha256"],
               "recovery_binding_sha256": prepared["recovery_binding_sha256"],
               "parent_global_step": 20, "begin_global_step": 20, "end_global_step": 20,
               "new_optimizer_updates": 0, "parent_loss_history_preserved": True,
               "parent_loss_record_count": len(prepared["parent_loss_history"]),
               "parent_loss_first": prepared["parent_loss_history"][0]["loss"],
               "parent_loss_last": prepared["parent_loss_history"][-1]["loss"],
               "original_initial_runtime_audit_missing": True,
               "initial_state_provenance": prepared["initial_state_provenance"],
               "reconstructed_initial_state_sha256": initial["state_sha256"],
               "parent_adapter_sha256": prepared["checkpoint_files"]["adapter_model.safetensors"]["sha256"],
               "recovered_adapter_sha256": _sha(final_path), "parent_adapter_tensors_exact": True,
               "adapter_tensor_count": len(final_state), "fresh_gpu_reload_exact": True,
               "runtime_audit_sha256": _sha(output / "rank-0-training-runtime.json"),
               "reload_audit_sha256": _sha(output / "semantic_adapter_reload_audit.json"),
               "parent_storage_mutated": False, "large_artifact_policy": "remote_only"}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(receipt_path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, receipt_path)
    return receipt
