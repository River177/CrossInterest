"""Scoped runtime guards for independently trained RHO reference models.

The objective, sampler, optimizer, and scheduler remain those of ``train_sft``.
Only checkpoint coordination and observational training callbacks are replaced,
inside this process and for the duration of this invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_reference_config(config: dict[str, Any]) -> None:
    for name in ("adapter_path", "sample_weight_file", "mera_relations_file"):
        if config.get(name) is not None:
            raise ValueError(f"Reference teachers forbid {name}")
    if config.get("loss_reduction", "token_mean") != "token_mean":
        raise ValueError("Reference teachers require native token_mean loss")
    if config.get("semantic_id_loss_scope", "full_vocabulary") != "full_vocabulary":
        raise ValueError("Reference teachers require full_vocabulary loss")
    if bool(config.get("enable_thinking", False)):
        raise ValueError("Reference teachers require non-thinking SID targets")
    if not config.get("semantic_id_token_mapping_file"):
        raise ValueError("Reference teachers require the frozen SID mapping")
    if config.get("gradient_checkpointing_use_reentrant", True) is not False:
        raise ValueError("Reference teachers require non-reentrant checkpointing")


class LocalCheckpointCoordinator:
    """Let peer ranks poll a node-local outcome while rank zero publishes.

    ``root`` must be a fresh directory shared by ranks on the same node. The
    launcher owns its lifetime. No collective is active across ``operation``.
    """

    def __init__(
        self,
        root: Path,
        *,
        distributed: Any,
        timeout_seconds: float = 1800,
        poll_seconds: float = 0.05,
    ) -> None:
        if not root.is_absolute():
            raise ValueError("RHO_SYNC_DIR must be an absolute node-local path")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Checkpoint coordination timeout must be positive")
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Checkpoint poll interval must be positive")
        self.root = root
        self.distributed = distributed
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.sequence = 0

    def __call__(self, operation: Callable[[], Any]) -> str | None:
        distributed = self.distributed
        active = distributed.is_available() and distributed.is_initialized()
        world_size = distributed.get_world_size() if active else 1
        rank = distributed.get_rank() if active else 0
        if world_size > 1 and int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != world_size:
            raise ValueError("RHO checkpoint coordination supports one node only")
        sequence = self.sequence
        self.sequence += 1
        outcome_file = self.root / f"operation-{sequence:06d}.json"
        # The parent allocates RHO_SYNC_DIR per torchrun launch; a prior marker
        # means the caller reused a rendezvous and must not consume stale state.
        if outcome_file.exists():
            raise RuntimeError(f"Checkpoint rendezvous already exists: {outcome_file.name}")
        if world_size > 1:
            distributed.barrier()
        if rank == 0:
            try:
                result = operation()
                outcome = {
                    "sequence": sequence,
                    "value": str(result) if result is not None else None,
                    "error": None,
                }
            except Exception as error:
                outcome = {
                    "sequence": sequence,
                    "value": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            _write_json(outcome_file, outcome)
        else:
            deadline = time.monotonic() + self.timeout_seconds
            while not outcome_file.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Rank-zero checkpoint operation {sequence} timed out "
                        f"after {self.timeout_seconds:g} seconds"
                    )
                time.sleep(self.poll_seconds)
            outcome = json.loads(outcome_file.read_text())
        if outcome.get("sequence") != sequence:
            raise RuntimeError("Checkpoint outcome sequence mismatch")
        if outcome.get("error") is not None:
            raise RuntimeError(str(outcome["error"]))
        return outcome["value"]


def adapter_initial_state(model: Any) -> dict[str, Any]:
    import torch
    from peft.utils.save_and_load import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    tensors: dict[str, Any] = {}
    lora_b_count = 0
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite initial adapter tensor: {name}")
        if "lora_B" in name:
            lora_b_count += 1
            if int(torch.count_nonzero(value)):
                raise ValueError(f"Reference teacher LoRA B is not fresh: {name}")
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    if not tensors or not lora_b_count:
        raise ValueError("Reference teacher has no freshly initialized LoRA B tensors")
    return {
        "state_sha256": hashlib.sha256(_canonical_bytes(tensors)).hexdigest(),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "lora_b_tensors": lora_b_count,
        "lora_b_all_zero": True,
    }


def _optimizer_snapshot(optimizer: Any, scheduler: Any, state: Any) -> dict[str, Any]:
    if optimizer is None or scheduler is None:
        raise ValueError("Training callback lacks optimizer or scheduler")
    entries = optimizer.state
    steps: list[int] = []
    for entry in entries.values():
        if "step" not in entry:
            raise ValueError("Optimizer entry has no Adam step cursor")
        scalar = float(entry["step"])
        if not math.isfinite(scalar) or scalar < 0 or not scalar.is_integer():
            raise ValueError("Invalid optimizer step cursor")
        steps.append(int(scalar))
    scheduler_epoch = int(scheduler.last_epoch)
    return {
        "global_step": int(state.global_step),
        "epoch": None if state.epoch is None else float(state.epoch),
        "optimizer_state_entries": len(entries),
        "optimizer_step_min": min(steps) if steps else None,
        "optimizer_step_max": max(steps) if steps else None,
        "scheduler_last_epoch": scheduler_epoch,
    }


class TrainingRuntimeAudit:
    """Record real Trainer state at optimizer start and completion on each rank."""

    def __init__(
        self,
        config: dict[str, Any],
        initial_state: dict[str, Any],
        *,
        train_samples: int,
        rank: int,
        world_size: int,
    ) -> None:
        batch = (
            world_size
            * int(config["per_device_train_batch_size"])
            * int(config["gradient_accumulation_steps"])
        )
        self.expected_steps = math.ceil(math.ceil(train_samples / batch) * float(config["epochs"]))
        self.expected_epochs = float(config["epochs"])
        self.output = Path(config["output_dir"]) / f"rank-{rank}-training-runtime.json"
        self.payload: dict[str, Any] = {
            "version": "rho-reference-sft-runtime-v1",
            "rank": rank,
            "world_size": world_size,
            "train_samples": train_samples,
            "expected_optimizer_steps": self.expected_steps,
            "initial_state": initial_state,
        }

    def begin(self, state: Any, optimizer: Any, scheduler: Any, *, resumed: bool) -> None:
        record = _optimizer_snapshot(optimizer, scheduler, state)
        fresh = (
            record["global_step"] == 0
            and record["optimizer_state_entries"] == 0
            and record["scheduler_last_epoch"] in (-1, 0)
        )
        record["fresh_optimization_state"] = fresh
        record["resumed_from_checkpoint"] = resumed
        self.payload["train_begin"] = record
        _write_json(self.output, self.payload)
        if not resumed and not fresh:
            raise ValueError("Fresh reference teacher did not start with fresh optimization state")
        if resumed and record["global_step"] > 0:
            if (
                record["optimizer_step_min"] != record["global_step"]
                or record["optimizer_step_max"] != record["global_step"]
                or record["scheduler_last_epoch"] != record["global_step"]
            ):
                raise ValueError("Resumed optimizer/scheduler cursors differ from Trainer state")
        print("RHO_REFERENCE_TRAIN_BEGIN " + json.dumps({"rank": self.payload["rank"], **record}), flush=True)

    def end(self, state: Any, optimizer: Any, scheduler: Any) -> None:
        record = _optimizer_snapshot(optimizer, scheduler, state)
        self.payload["train_end"] = record
        passed = (
            record["global_step"] == self.expected_steps
            and record["epoch"] is not None
            and abs(record["epoch"] - self.expected_epochs) < 1e-8
            and record["optimizer_step_min"] == self.expected_steps
            and record["optimizer_step_max"] == self.expected_steps
            and record["scheduler_last_epoch"] == self.expected_steps
        )
        self.payload["completed_as_configured"] = passed
        _write_json(self.output, self.payload)
        if not passed:
            raise ValueError("Actual training/optimizer/scheduler endpoint differs from configuration")


def _instrumented_trainer(base: type, callback_base: type, config: dict[str, Any]) -> type:
    class AuditCallback(callback_base):
        def __init__(self, trainer: Any) -> None:
            self.trainer = trainer

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.trainer.rho_runtime_audit.begin(
                state, kwargs["optimizer"], kwargs["lr_scheduler"],
                resumed=self.trainer.rho_resume,
            )
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.trainer.rho_runtime_audit.end(state, kwargs["optimizer"], kwargs["lr_scheduler"])
            return control

    class ReferenceTrainer(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            import torch.distributed as distributed

            super().__init__(*args, **kwargs)
            initial = adapter_initial_state(self.model)
            active = distributed.is_available() and distributed.is_initialized()
            rank = distributed.get_rank() if active else 0
            world_size = distributed.get_world_size() if active else 1
            if world_size > 1:
                all_hashes: list[Any] = [None] * world_size
                distributed.all_gather_object(all_hashes, initial["state_sha256"])
                if len(set(all_hashes)) != 1:
                    raise ValueError("Initial reference adapter differs between ranks")
            self.rho_runtime_audit = TrainingRuntimeAudit(
                config, initial, train_samples=len(self.train_dataset), rank=rank, world_size=world_size,
            )
            self.rho_resume = False
            self.add_callback(AuditCallback(self))
            _write_json(self.rho_runtime_audit.output, self.rho_runtime_audit.payload)

        def train(self, *args: Any, **kwargs: Any) -> Any:
            if not any(isinstance(callback, AuditCallback) for callback in self.callback_handler.callbacks):
                raise RuntimeError("RHO runtime callback was not installed before training")
            resume = kwargs.get("resume_from_checkpoint", args[0] if args else None)
            self.rho_resume = resume is not None and resume is not False
            print("RHO_REFERENCE_RUNTIME_ENABLED " + json.dumps({
                "rank": self.rho_runtime_audit.payload["rank"],
                "initial_state_sha256": self.rho_runtime_audit.payload["initial_state"]["state_sha256"],
            }), flush=True)
            result = super().train(*args, **kwargs)
            if not self.rho_runtime_audit.payload.get("completed_as_configured"):
                raise RuntimeError("RHO runtime callback did not verify the training endpoint")
            return result

    return ReferenceTrainer


def train_reference(config: dict[str, Any]) -> dict[str, object]:
    validate_reference_config(config)
    import torch.distributed as distributed
    from transformers.trainer import Trainer
    from transformers.trainer_callback import TrainerCallback
    from new_interest.training import train_sft

    sync_dir = os.environ.get("RHO_SYNC_DIR")
    if not sync_dir:
        raise ValueError("Launcher must provide a fresh node-local RHO_SYNC_DIR")
    coordinator = LocalCheckpointCoordinator(
        Path(sync_dir), distributed=distributed,
        timeout_seconds=float(os.environ.get("RHO_CHECKPOINT_TIMEOUT_SECONDS", "1800")),
    )
    original_coordinator = train_sft._run_rank_zero_checkpoint_operation
    try:
        train_sft._run_rank_zero_checkpoint_operation = coordinator
        trainer_base = _instrumented_trainer(Trainer, TrainerCallback, config)
        return train_sft.train(config, trainer_base=trainer_base)
    finally:
        train_sft._run_rank_zero_checkpoint_operation = original_coordinator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Reference SFT config must be a JSON object")
    train_reference(config)


if __name__ == "__main__":
    main()
