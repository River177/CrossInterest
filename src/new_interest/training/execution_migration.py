"""Explicit, audited changes of SFT execution layout across a checkpoint.

The original checkpoint and stage evidence are immutable inputs. A migrated
checkpoint has its own complete marker and a hash-linked execution receipt.
Optimizer, scheduler, model, step, epoch, and caller-owned cursor state are kept.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Callable

from new_interest.training.checkpoint_persistence import (
    CHECKPOINT_MARKER, CHECKPOINT_PROTOCOL_VERSION,
    checkpoint_identity_sha256, validate_persistent_checkpoint,
)
from new_interest.training.sft_state_audit import (
    capture_adapter_state, optimizer_start_state, tensor_state_manifest,
)

PROTOCOL = "sft-execution-migration-v1"
EXECUTION_FIELDS = frozenset({
    "world_size", "per_device_train_batch_size", "gradient_accumulation_steps",
    "gradient_checkpointing_enabled",
})
MIGRATION_FILE = "execution_migration.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()).hexdigest()


def write_json_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def state_tree(value: Any) -> Any:
    """Canonical values and tensor bytes, independent of device and pickle."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"tensor": True, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"ndarray": True, "shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): state_tree(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [state_tree(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite checkpoint scalar")
        return value
    raise TypeError(f"Unsupported checkpoint state value: {type(value).__name__}")


def state_sha256(value: Any) -> str:
    return canonical_sha256(state_tree(value))


def validate_execution_transition(source_identity: dict, target_identity: dict) -> dict:
    source = copy.deepcopy(source_identity)
    target = copy.deepcopy(target_identity)
    source.setdefault("gradient_checkpointing_enabled", True)
    target.setdefault("gradient_checkpointing_enabled", True)
    differences = {key: {"source": source.get(key), "target": target.get(key)}
                   for key in source.keys() | target.keys() if source.get(key) != target.get(key)}
    forbidden = set(differences) - EXECUTION_FIELDS
    if forbidden:
        raise ValueError("Migration changes scientific/stage identity: " + ", ".join(sorted(forbidden)))
    if int(source["world_size"]) != 1 or int(target["world_size"]) != 4:
        raise ValueError("This migration protocol requires a 1 to 4 GPU transition")
    for identity in (source, target):
        batch = int(identity["world_size"]) * int(identity["per_device_train_batch_size"]) * int(identity["gradient_accumulation_steps"])
        if batch != 32 or identity["effective_global_batch_size"] != 32:
            raise ValueError("Migration must preserve effective global batch 32")
        if identity.get("gradient_checkpointing_kwargs") != {"use_reentrant": False}:
            raise ValueError("Migration requires the approved non-reentrant checkpointing policy")
    if int(target["per_device_train_batch_size"]) not in {2, 4, 8}:
        raise ValueError("Target microbatch was not in the approved probe set")
    return {"changed_execution_fields": differences,
            "source_identity_sha256": checkpoint_identity_sha256(source_identity),
            "target_identity_sha256": checkpoint_identity_sha256(target_identity)}


def checkpoint_state_audit(checkpoint: Path, identity: dict) -> dict:
    import torch
    from safetensors.torch import load_file
    marker = validate_persistent_checkpoint(checkpoint, identity)
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    step = int(marker["global_step"])
    if step <= 0 or int(trainer["global_step"]) != step:
        raise ValueError("Migration requires a populated, step-aligned checkpoint")
    optimizer_steps = [float(item["step"].item() if hasattr(item["step"], "item") else item["step"])
                       for item in optimizer["state"].values()]
    if not optimizer_steps or any(value != step for value in optimizer_steps):
        raise ValueError("Checkpoint optimizer steps differ from Trainer step")
    if int(scheduler["last_epoch"]) != step:
        raise ValueError("Checkpoint scheduler step differs from Trainer step")
    tensors = tensor_state_manifest(load_file(str(checkpoint / "adapter_model.safetensors")))
    return {"global_step": step, "epoch": trainer["epoch"], "max_steps": int(trainer["max_steps"]),
            "num_train_epochs": trainer["num_train_epochs"], "adapter_state_sha256": tensors["state_sha256"],
            "adapter_tensor_count": tensors["tensor_count"], "optimizer_state_entries": len(optimizer_steps),
            "optimizer_state_sha256": state_sha256(optimizer), "scheduler_state_sha256": state_sha256(scheduler),
            "optimizer_step_min": min(optimizer_steps), "optimizer_step_max": max(optimizer_steps),
            "scheduler_last_epoch": int(scheduler["last_epoch"]),
            "trainer_state_sha256": file_sha256(checkpoint / "trainer_state.json")}


def _stage_inventory(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Stage evidence must contain ordinary files")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    if not {"initial_adapter_state.json", "training_start_audit.json"} <= files.keys():
        raise ValueError("Original initial and first-start stage evidence is required")
    return files


def restore_stage_evidence(source: Path, destination: Path, source_identity: dict) -> dict:
    inventory = _stage_inventory(source)
    first = json.loads((source / "training_start_audit.json").read_text())
    initial = json.loads((source / "initial_adapter_state.json").read_text())
    if first["checkpoint_identity"] != source_identity or initial["checkpoint_identity"] != source_identity:
        raise ValueError("Original stage evidence does not match the source checkpoint identity")
    if not first["fresh_optimization_state"] or first["global_step"] != 0:
        raise ValueError("Original first-start evidence is not a fresh stage start")
    if first["source_adapter_state_sha256"] != initial["state_sha256"]:
        raise ValueError("Original stage parameter fingerprints differ")
    for name, expected in inventory.items():
        target = destination / name
        if target.exists():
            if file_sha256(target) != expected["sha256"]:
                # Resume histories are append-only at the new destination.
                old = (source / name).read_bytes()
                if name != "training_resume_audits.jsonl" or not target.read_bytes().startswith(old):
                    raise ValueError(f"Existing stage evidence differs: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
    return inventory


def _cuda_rng_state(seed: int):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Materializing migration CUDA RNG states requires the target GPU worker")
    return torch.Generator(device="cuda:0").manual_seed(seed).get_state()


def derive_rank_rng_states(source: dict, lineage: str, world_size: int, cuda_state_factory: Callable | None = None) -> tuple[list[dict], dict]:
    import numpy as np
    import torch
    cuda_state_factory = cuda_state_factory or _cuda_rng_state
    if set(source) != {"python", "numpy", "cpu", "cuda"}:
        raise ValueError("Unsupported source RNG schema")
    old_cuda = source["cuda"]
    if isinstance(old_cuda, (list, tuple)):
        if len(old_cuda) != 1:
            raise ValueError("Source RNG has more than one visible CUDA device")
        old_cuda = old_cuda[0]
    if not isinstance(old_cuda, torch.Tensor):
        raise ValueError("Source CUDA RNG state is not a tensor")
    states, records = [], []
    for rank in range(world_size):
        seed = int.from_bytes(hashlib.sha256(f"{lineage}:rank:{rank}".encode()).digest()[:8], "little") % (2**63 - 1)
        current = ({key: copy.deepcopy(source[key]) for key in ("python", "numpy", "cpu")}
                   if rank == 0 else {"python": random.Random(seed).getstate(),
                     "numpy": np.random.RandomState(seed % (2**32)).get_state(),
                     "cpu": torch.Generator(device="cpu").manual_seed(seed).get_state()})
        # Transformers 5.14.1 DDP restores CUDA state with set_rng_state_all.
        # Every rank sees four devices; only its local rank is used for training.
        cuda_states = []
        for device in range(world_size):
            device_seed = int.from_bytes(hashlib.sha256(f"{lineage}:rank:{rank}:device:{device}".encode()).digest()[:8], "little") % (2**63 - 1)
            cuda_states.append(old_cuda.clone() if rank == 0 and device == 0 else cuda_state_factory(device_seed))
        current["cuda"] = cuda_states
        states.append(current)
        records.append({"rank": rank, "cpu_seed": None if rank == 0 else seed,
                        "state_sha256": state_sha256(current),
                        "local_cuda_sha256": state_sha256(cuda_states[rank])})
    for key in ("python", "numpy", "cpu"):
        if state_sha256(states[0][key]) != state_sha256(source[key]):
            raise ValueError("Rank-zero source RNG was not preserved")
    if state_sha256(states[0]["cuda"][0]) != state_sha256(old_cuda):
        raise ValueError("Rank-zero source CUDA RNG was not preserved")
    return states, {"policy": "preserve-source-rank0-derive-new-ranks-v1", "visible_cuda_devices": world_size,
                    "rank0_source_rng_preserved": True, "ranks": records,
                    "trajectory_bitwise_equality_claimed": False,
                    "boundary": "Global sample order and restored optimizer state are exact; batch shape, DDP reductions and new-rank dropout streams change numerical execution."}


def prepare_execution_migration(
    source_checkpoint: Path, destination: Path, source_identity: dict, target_identity: dict,
    source_stage_artifacts: Path, cursor_contract: dict, *, cuda_state_factory: Callable | None = None,
) -> dict:
    """Create a new migration root containing checkpoint, stage evidence, receipt.

    ``cursor_contract`` is caller-owned and must bind dataset, source global
    step/epoch, and total optimizer budget. The core never rewrites sidecars.
    """
    import torch
    transition = validate_execution_transition(source_identity, target_identity)
    audit = checkpoint_state_audit(source_checkpoint, source_identity)
    for key, expected in (("dataset_sha256", source_identity["train_file_sha256"]),
                          ("source_global_step", audit["global_step"]), ("total_steps", audit["max_steps"])):
        if cursor_contract.get(key) != expected:
            raise ValueError(f"Cursor contract differs from checkpoint: {key}")
    if not cursor_contract.get("protocol") or not math.isclose(float(cursor_contract["source_epoch"]), float(audit["epoch"]), abs_tol=1e-10):
        raise ValueError("Cursor contract lacks the exact source epoch")
    source_marker = json.loads((source_checkpoint / CHECKPOINT_MARKER).read_text())
    stage_files = _stage_inventory(source_stage_artifacts)
    chain = {"protocol": PROTOCOL, "source_checkpoint_marker_sha256": file_sha256(source_checkpoint / CHECKPOINT_MARKER),
             "source_identity": source_identity, "target_identity": target_identity,
             "source_state_audit": audit, "cursor_contract": cursor_contract,
             "source_stage_files": stage_files, "source_checkpoint_files": source_marker["files"], **transition}
    lineage = canonical_sha256(chain)
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".migration-", dir=destination.parent))
    try:
        checkpoint = temporary / source_checkpoint.name
        checkpoint.mkdir()
        for name, expected in source_marker["files"].items():
            target = checkpoint / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_checkpoint / name, target)
            if file_sha256(target) != expected["sha256"]:
                raise ValueError(f"Copied checkpoint differs: {name}")
        shutil.copyfile(source_checkpoint / CHECKPOINT_MARKER, checkpoint / "source_checkpoint_complete.json")
        # These RNG files are trusted, hash-verified artifacts of our own worker.
        old_rng = torch.load(source_checkpoint / "rng_state.pth", map_location="cpu", weights_only=False)
        rng_states, rng_audit = derive_rank_rng_states(old_rng, lineage, 4, cuda_state_factory)
        for rank, state in enumerate(rng_states):
            torch.save(state, checkpoint / f"rng_state_{rank}.pth")
        stage = temporary / "source-stage"
        restore_stage_evidence(source_stage_artifacts, stage, source_identity)
        receipt = {**chain, "lineage_id": lineage, "rng_transition": rng_audit,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "cuda_rng_materialized_on_gpu": cuda_state_factory is None and torch.cuda.is_available(),
                   "status": "prepared_requires_live_restore_audit"}
        (checkpoint / MIGRATION_FILE).write_text(json.dumps(receipt, indent=2) + "\n")
        inventory = {path.relative_to(checkpoint).as_posix(): {"sha256": file_sha256(path), "size": path.stat().st_size}
                     for path in sorted(checkpoint.rglob("*")) if path.is_file()}
        marker = {"protocol_version": CHECKPOINT_PROTOCOL_VERSION, "checkpoint_name": checkpoint.name,
                  "global_step": audit["global_step"], "created_at_utc": receipt["created_utc"],
                  "identity": target_identity, "identity_sha256": checkpoint_identity_sha256(target_identity), "files": inventory}
        (checkpoint / CHECKPOINT_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
        target_audit = checkpoint_state_audit(checkpoint, target_identity)
        if target_audit != audit:
            raise ValueError("Migration changed model, optimizer, scheduler, step or Trainer state")
        (temporary / "migration_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.rename(temporary, destination)
    except BaseException:
        # Preserve incomplete construction for diagnosis; never touch the source.
        raise
    return {"checkpoint_dir": destination / source_checkpoint.name,
            "stage_dir": destination / "source-stage", "manifest_path": destination / "migration_receipt.json",
            "lineage_id": lineage, "receipt": receipt}


def load_migration_receipt(checkpoint: Path, target_identity: dict) -> dict:
    receipt = json.loads((checkpoint / MIGRATION_FILE).read_text())
    if receipt.get("protocol") != PROTOCOL or receipt["target_identity"] != target_identity:
        raise ValueError("Loaded checkpoint lacks its exact execution migration lineage")
    validate_execution_transition(receipt["source_identity"], target_identity)
    if not receipt.get("cuda_rng_materialized_on_gpu"):
        raise ValueError("A CPU RNG fixture cannot authorize GPU execution")
    return receipt


def make_execution_migration_trainer(base_trainer, runtime: dict):
    class ExecutionMigrationTrainer(base_trainer):
        def reuse_completed_sample_cursor(self, state, destination):
            from new_interest.training.global_sample_cursor import reuse_completed_cursor
            if int(state.global_step) != int(runtime["cursor_contract"]["total_steps"]):
                raise ValueError("Cannot use a final cursor before the complete stage")
            return reuse_completed_cursor(Path(runtime["resume_checkpoint"]), runtime["target_identity"],
                                          runtime["cursor_contract"], Path(destination))

        def _load_rng_state(self, checkpoint):
            import numpy as np
            import torch
            if checkpoint is None:
                raise ValueError("Migrated SFT cannot start with missing RNG/checkpoint state")
            rank = int(self.args.process_index)
            path = Path(checkpoint) / f"rng_state_{rank}.pth"
            if not path.is_file():
                raise ValueError(f"Migrated checkpoint lacks rank {rank} RNG state")
            expected = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(expected.get("cuda"), (list, tuple)) or len(expected["cuda"]) != torch.cuda.device_count():
                raise ValueError("DDP CUDA RNG state does not cover exactly the visible devices")
            super()._load_rng_state(checkpoint)
            actual = {"python": random.getstate(), "numpy": np.random.get_state(),
                      "cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
            if state_sha256(actual) != state_sha256(expected):
                raise ValueError(f"Actual RNG restore differs on rank {rank}")
            record = {"rank": rank, "checkpoint_step": int(self.state.global_step),
                      "rng_file_sha256": file_sha256(path), "expected_rng_sha256": state_sha256(expected),
                      "actual_rng_sha256": state_sha256(actual), "status": "passed"}
            record["lineage_id"] = runtime["lineage_id"]
            out = Path(runtime["output_dir"])
            write_json_once(Path(runtime["attempt_dir"]) / f"rng_rank_{rank}.json", record)
            first = out / f"execution_rng_restore_rank_{rank}.json"
            if not first.exists():
                write_json_once(first, record)
            self._execution_rng_restored = True

        def training_step(self, model, inputs, num_items_in_batch=None):
            if not getattr(self, "_execution_rng_restored", False):
                raise ValueError("No model forward is allowed before audited per-rank RNG restore")
            return super().training_step(model, inputs, num_items_in_batch)

    return ExecutionMigrationTrainer


def make_execution_migration_callback(base_callback, runtime: dict, *, cursor_audit_hook=None):
    class ExecutionMigrationCallback(base_callback):
        def on_train_begin(self, args, state, control, **kwargs):
            import torch.distributed as dist
            checkpoint = Path(runtime["resume_checkpoint"])
            target_identity = runtime["target_identity"]
            receipt = load_migration_receipt(checkpoint, target_identity)
            expected = checkpoint_state_audit(checkpoint, target_identity)
            optimizer, scheduler = kwargs["optimizer"], kwargs["lr_scheduler"]
            model = kwargs["model"]
            if hasattr(model, "module") and not hasattr(model, "peft_config"):
                model = model.module
            actual = optimizer_start_state(state, optimizer, scheduler)
            actual.update({"epoch": state.epoch, "max_steps": state.max_steps,
                           "adapter_state_sha256": capture_adapter_state(model)["state_sha256"],
                           "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
                           "scheduler_state_sha256": state_sha256(scheduler.state_dict())})
            compared = ("global_step", "epoch", "max_steps", "adapter_state_sha256", "optimizer_state_entries",
                        "optimizer_state_sha256", "scheduler_state_sha256", "optimizer_step_min", "optimizer_step_max",
                        "scheduler_last_epoch")
            differences = {key: {"expected": expected[key], "actual": actual[key]}
                           for key in compared if actual[key] != expected[key]}
            gathered = [None] * int(args.world_size)
            local = {"rank": int(args.process_index), "differences": differences,
                     "actual": actual, "status": "passed" if not differences else "failed"}
            dist.all_gather_object(gathered, local)
            if any(record["differences"] for record in gathered):
                raise ValueError("Actual distributed checkpoint restore differs: " + json.dumps(gathered))
            original_start = json.loads((Path(runtime["output_dir"]) / "training_start_audit.json").read_text())
            if original_start["checkpoint_identity"] != receipt["source_identity"] or not original_start["fresh_optimization_state"]:
                raise ValueError("Original first-start evidence was replaced or changed")
            state.train_batch_size = args.train_batch_size
            self._initial_step = int(state.global_step)
            self._step_metrics = []
            if state.is_world_process_zero:
                audit = {"protocol": PROTOCOL, "lineage_id": receipt["lineage_id"], "status": "passed",
                         "checkpoint_name": checkpoint.name, "global_step": state.global_step, "epoch": state.epoch,
                         "source_identity": receipt["source_identity"], "target_identity": target_identity,
                         "original_first_start_sha256": file_sha256(Path(runtime["output_dir"]) / "training_start_audit.json"),
                         "rank_restores": gathered, "rng_verification": "required before first forward",
                         "sample_cursor_verification": "required on actual dataloader batches"}
                out = Path(runtime["output_dir"]) / "execution_migration_start.json"
                write_json_once(Path(runtime["attempt_dir"]) / "execution_start.json", audit)
                if not out.exists():
                    if state.global_step != receipt["source_state_audit"]["global_step"]:
                        raise ValueError("Missing original 1-to-4 migration start evidence")
                    write_json_once(out, audit)
                else:
                    original_migration = json.loads(out.read_text())
                    if original_migration["lineage_id"] != receipt["lineage_id"]:
                        raise ValueError("Existing first migration evidence has a different lineage")
                    continuation = {"attempt_dir": Path(runtime["attempt_dir"]).name,
                                    "original_migration_start_sha256": file_sha256(out), **audit}
                    with (Path(runtime["output_dir"]) / "execution_resume_audits.jsonl").open("a") as stream:
                        stream.write(json.dumps(continuation, sort_keys=True) + "\n")
                print("EXECUTION_MIGRATION_STATE_RESTORED", state.global_step, receipt["lineage_id"], flush=True)
            from new_interest.training.execution_probe import attention_kernel_audit, run_long_sample_stress, run_model_numerical_probe
            kernels = attention_kernel_audit(model)
            write_json_once(Path(runtime["attempt_dir"]) / f"kernels_rank_{int(args.process_index)}.json", kernels)
            if runtime.get("numerical_probe_file"):
                run_model_numerical_probe(model, optimizer, runtime["tokenizer"], args,
                                          runtime["numerical_probe_file"], runtime["mapping_file"],
                                          Path(runtime["output_dir"]))
            if runtime.get("long_sample_files"):
                run_long_sample_stress(model, optimizer, runtime["tokenizer"], args,
                                       runtime["long_sample_files"], Path(runtime["output_dir"]))
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            import torch
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            self._step_start = time.perf_counter()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            import torch
            import torch.distributed as dist
            torch.cuda.synchronize(args.device)
            free, total = torch.cuda.mem_get_info(args.device)
            local = {"rank": int(args.process_index), "seconds": time.perf_counter() - self._step_start,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                     "device_free_bytes": free, "device_total_bytes": total}
            gathered = [None] * int(args.world_size)
            dist.all_gather_object(gathered, local)
            if state.is_world_process_zero:
                record = {"global_step": int(state.global_step), "ranks": gathered,
                          "seconds": max(row["seconds"] for row in gathered), "nominal_global_batch_size": 32}
                self._step_metrics.append(record)
            stop = runtime.get("stop_after_global_step")
            if stop is not None and state.global_step >= int(stop):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            from new_interest.training.train_sft import _run_rank_zero_checkpoint_operation
            def publish_auxiliary():
                checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
                source = Path(runtime["resume_checkpoint"]) / MIGRATION_FILE
                destination = checkpoint / MIGRATION_FILE
                if source != destination:
                    shutil.copyfile(source, destination)
                hook = cursor_audit_hook or getattr(runtime["trainer"], "save_migration_cursor", None)
                if hook is not None:
                    hook(state, checkpoint)
                self._publish_metrics()
            _run_rank_zero_checkpoint_operation(publish_auxiliary)
            return control

        def _publish_metrics(self):
            target = Path(runtime["attempt_dir"]) / "step_metrics.jsonl"
            temporary = target.with_suffix(".jsonl.partial")
            temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in self._step_metrics))
            temporary.replace(target)

        def on_train_end(self, args, state, control, **kwargs):
            from new_interest.training.train_sft import _run_rank_zero_checkpoint_operation
            def publish_final_evidence():
                self._publish_metrics()
                hook = cursor_audit_hook or getattr(runtime["trainer"], "save_migration_cursor", None)
                if hook is not None:
                    hook(state, Path(runtime["output_dir"]))
            _run_rank_zero_checkpoint_operation(publish_final_evidence)
            return control

    return ExecutionMigrationCallback()
