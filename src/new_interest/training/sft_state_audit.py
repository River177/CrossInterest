"""Observe adapter identity and optimizer resets without changing training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def tensor_state_manifest(state_dict: dict[str, Any]) -> dict[str, object]:
    import torch

    digest = hashlib.sha256()
    tensors = {}
    total_numel = 0
    for key in sorted(state_dict):
        value = state_dict[key].detach().contiguous().cpu()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        item = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "numel": int(value.numel()),
        }
        digest.update(json.dumps([key, item], sort_keys=True, separators=(",", ":")).encode("utf-8"))
        tensors[key] = item
        total_numel += int(value.numel())
    return {
        "version": "peft-tensor-state-v1",
        "state_sha256": digest.hexdigest(),
        "tensor_count": len(tensors),
        "total_numel": total_numel,
        "tensors": tensors,
    }


def capture_adapter_state(model: Any) -> dict[str, object]:
    from peft.utils.save_and_load import get_peft_model_state_dict

    return tensor_state_manifest(get_peft_model_state_dict(model, save_embedding_layers=False))


def record_initial_adapter_state(
    model: Any,
    output_dir: Path,
    checkpoint_identity: dict[str, object],
    *,
    input_adapter: Path | None = None,
) -> dict[str, object]:
    manifest = capture_adapter_state(model)
    manifest["checkpoint_identity"] = checkpoint_identity
    if input_adapter is not None:
        from safetensors.torch import load_file

        source = tensor_state_manifest(load_file(str(input_adapter / "adapter_model.safetensors")))
        if manifest["state_sha256"] != source["state_sha256"]:
            raise ValueError("Initial adapter does not exactly preserve the preceding stage tensors")
        manifest["input_adapter_state_sha256"] = source["state_sha256"]
    target = output_dir / "initial_adapter_state.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous != manifest:
            raise ValueError("Existing initial adapter identity differs; use a fresh output directory")
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest.items() if key not in {"tensors", "checkpoint_identity"}}


def optimizer_start_state(state: Any, optimizer: Any, lr_scheduler: Any) -> dict[str, object]:
    if optimizer is None:
        raise ValueError("Training-start audit requires the actual optimizer")
    steps = []
    for entry in optimizer.state.values():
        if "step" in entry:
            value = entry["step"]
            steps.append(float(value.item() if hasattr(value, "item") else value))
    last_epoch = getattr(lr_scheduler, "last_epoch", None)
    if last_epoch is None and getattr(lr_scheduler, "scheduler", None) is not None:
        last_epoch = getattr(lr_scheduler.scheduler, "last_epoch", None)
    result = {
        "version": "sft-training-start-v1",
        "global_step": int(state.global_step),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_step_min": min(steps) if steps else None,
        "optimizer_step_max": max(steps) if steps else None,
        "scheduler_last_epoch": last_epoch,
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
    }
    result["fresh_optimization_state"] = (
        result["global_step"] == 0
        and result["optimizer_state_entries"] == 0
        and last_epoch in {-1, 0}
    )
    return result


def make_training_start_audit_callback(
    base_callback: type[Any],
    output_dir: Path,
    initial_state_sha256: str,
    checkpoint_identity: dict[str, object],
) -> Any:
    class TrainingStartAuditCallback(base_callback):
        def on_train_begin(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True):
                return control
            start = optimizer_start_state(state, kwargs.get("optimizer"), kwargs.get("lr_scheduler"))
            start["source_adapter_state_sha256"] = initial_state_sha256
            start["checkpoint_identity"] = checkpoint_identity
            target = output_dir / "training_start_audit.json"
            if start["global_step"] == 0:
                if not start["fresh_optimization_state"]:
                    raise ValueError("New SFT stage retained optimizer or scheduler state")
                actual = capture_adapter_state(kwargs["model"])
                if actual["state_sha256"] != initial_state_sha256:
                    raise ValueError("Adapter changed before the first optimizer step")
                start["actual_adapter_state_sha256"] = actual["state_sha256"]
                if target.exists():
                    if json.loads(target.read_text()) != start:
                        raise ValueError("Existing first-start evidence differs")
                else:
                    target.write_text(json.dumps(start, indent=2) + "\n", encoding="utf-8")
            else:
                if not target.exists():
                    raise ValueError("Checkpoint resume lacks original first-start evidence")
                original = json.loads(target.read_text())
                if original["checkpoint_identity"] != checkpoint_identity:
                    raise ValueError("Checkpoint resume has a different stage identity")
                with (output_dir / "training_resume_audits.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(start, sort_keys=True) + "\n")
            return control

    return TrainingStartAuditCallback()
