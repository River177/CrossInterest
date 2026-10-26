"""Topology-independent sample order and stage cursor for old/new-interest SFT.

The dataloader represents a complete epoch. Transformers alone skips the consumed
microbatches using the restored optimizer step and the new accumulation factor.
Masked execution slots make the final ten real rows safe across four ranks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterator

import pyarrow.parquet as pq
from new_interest.training.alignment_token_cursor import expected_supervised_tokens


CURSOR_VERSION = "old-new-global-epoch-cursor-v1"
PADDING_INDEX = -1
STAGES = {"two_stage/old_sft", "two_stage/sft", "mixed/sft"}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_permutation(samples: int, epoch: int, seed: int = 42) -> list[int]:
    import torch

    if samples <= 0 or epoch < 0:
        raise ValueError("Invalid epoch/sample count")
    return torch.randperm(samples, generator=torch.Generator(device="cpu").manual_seed(seed + epoch)).tolist()


def expected_counts(samples: int, optimizer_step: int, *, global_batch: int = 32,
                    epochs: int = 3, seed: int = 42) -> tuple[list[int], dict[str, int]]:
    updates = math.ceil(samples / global_batch)
    if optimizer_step < 0 or optimizer_step > updates * epochs:
        raise ValueError("Checkpoint step is outside the complete stage budget")
    completed_epochs, epoch_step = divmod(optimizer_step, updates)
    consumed = min(epoch_step * global_batch, samples)
    counts = [completed_epochs] * samples
    if consumed:
        for index in epoch_permutation(samples, completed_epochs, seed)[:consumed]:
            counts[index] += 1
    return counts, {"completed_epochs": completed_epochs, "epoch_optimizer_step": epoch_step,
                    "consumed_in_current_epoch": consumed, "consumed_total": sum(counts),
                    "updates_per_epoch": updates}


def make_cursor_contract(
    train_file: Path, *, stage: str, world_size: int, per_device_batch_size: int,
    source_checkpoint: Path | None = None, epochs: int = 3, seed: int = 42,
    global_batch_size: int = 32, token_lengths_file: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("Unknown old/new training stage")
    width = world_size * per_device_batch_size
    if min(world_size, per_device_batch_size) <= 0 or global_batch_size % width:
        raise ValueError("Execution shape must divide the frozen global batch")
    columns = ["sample_id", "target_subcategory"]
    if "training_task" in pq.ParquetFile(train_file).schema_arrow.names:
        columns.append("training_task")
    # Legacy SID-only datasets have a fixed three-token completion. Joint
    # datasets freeze each completion length explicitly (including name text).
    if "expected_assistant_tokens" in pq.ParquetFile(train_file).schema_arrow.names:
        columns.append("expected_assistant_tokens")
    rows = pq.read_table(train_file, columns=columns).to_pylist()
    supervised_lengths = [r.get("expected_assistant_tokens", 3) for r in rows]
    expected_supervised_tokens([1]*len(rows), supervised_lengths)
    default_task = "old_interest" if stage == "two_stage/old_sft" else "new_interest"
    rows = [{"sample_id": str(row["sample_id"]), "target_subcategory": str(row["target_subcategory"]),
             "training_task": str(row.get("training_task", default_task))} for row in rows]
    row_identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    step = 0
    source_epoch = 0.0
    source = None
    if source_checkpoint is not None:
        exposure_path = source_checkpoint / "training_exposure_state.json"
        exposure = json.loads(exposure_path.read_text())
        trainer = json.loads((source_checkpoint / "trainer_state.json").read_text())
        step = int(trainer["global_step"])
        source_epoch = float(trainer.get("epoch", 0.0))
        if step != exposure["optimizer_step"]:
            raise ValueError("Checkpoint exposure and optimizer cursors disagree")
        identity = exposure["identity"]
        if identity["train_sha256"] != _sha(train_file) or identity["row_identity_sha256"] != row_identity:
            raise ValueError("Checkpoint belongs to another stage dataset")
        if identity["expected_epochs"] != epochs:
            raise ValueError("Stage epoch budget changed")
        counts, progress = expected_counts(len(rows), step, global_batch=global_batch_size, epochs=epochs, seed=seed)
        if counts != exposure["counts"]:
            raise ValueError("Consumed samples are not the frozen global permutation prefix")
        if exposure["assistant_tokens"] != expected_supervised_tokens(counts, supervised_lengths):
            raise ValueError("Supervised-token cursor differs from the frozen SID task")
        if token_lengths_file is not None:
            lengths = {row["sample_id"]: int(row["tokens"]) for row in pq.read_table(token_lengths_file).to_pylist()}
            expected_tokens = [lengths[row["sample_id"]] * count for row, count in zip(rows, counts)]
            if expected_tokens != exposure["input_tokens"]:
                raise ValueError("Input-token cursor differs from the actual frozen samples")
        source = {"checkpoint_step": step, "training_exposure_sha256": _sha(exposure_path),
                  "trainer_state_sha256": _sha(source_checkpoint / "trainer_state.json"),
                  "input_tokens_consumed": sum(exposure["input_tokens"]),
                  "supervised_tokens_consumed": sum(exposure["assistant_tokens"])}
    else:
        _, progress = expected_counts(len(rows), 0, global_batch=global_batch_size, epochs=epochs, seed=seed)
    order = epoch_permutation(len(rows), progress["completed_epochs"], seed) if progress["completed_epochs"] < epochs else []
    remaining = order[progress["consumed_in_current_epoch"]:]
    return {"version": CURSOR_VERSION, "protocol": CURSOR_VERSION,
            "dataset_sha256": _sha(train_file), "source_global_step": step,
            "source_epoch": source_epoch, "total_steps": progress["updates_per_epoch"] * epochs,
            "stage": stage, "train_sha256": _sha(train_file),
            "row_identity_sha256": row_identity, "expected_assistant_tokens": supervised_lengths, "samples": len(rows), "epochs": epochs, "seed": seed,
            "global_batch_size": global_batch_size, "world_size": world_size,
            "per_device_batch_size": per_device_batch_size, "gradient_accumulation_steps": global_batch_size // width,
            "optimizer_step": step, **progress, "source": source,
            "remaining_current_epoch_order_sha256": hashlib.sha256(json.dumps(remaining, separators=(",", ":")).encode()).hexdigest(),
            "sampler_mode": "complete_epoch_global_permutation_then_native_hf_skip",
            "ignore_data_skip": False, "split_batches": True, "even_batches": False,
            "padding_index": PADDING_INDEX, "padding_is_training_exposure": False,
            "token_normalization": "global real shifted-label count across ranks and accumulation microbatches",
            "rng_boundary": "Source RNG retained; rank/dropout streams after topology change need not be bit-identical to single GPU"}


class EpochPermutationSampler:
    def __init__(self, samples: int, seed: int = 42) -> None:
        self.samples, self.seed, self.epoch = samples, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[int]:
        yield from epoch_permutation(self.samples, self.epoch, self.seed)


class PaddedGlobalBatchSampler:
    def __init__(self, samples: int, *, world_size: int, per_device_batch_size: int,
                 global_batch_size: int = 32, seed: int = 42) -> None:
        self.sampler = EpochPermutationSampler(samples, seed)
        self.batch_size = world_size * per_device_batch_size
        self.global_batch_size = global_batch_size
        self.drop_last = False
        if self.batch_size <= 0 or global_batch_size % self.batch_size:
            raise ValueError("Global microbatch width must divide the optimizer batch")

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch + [PADDING_INDEX] * (self.batch_size - len(batch))


def audit_update_batches(contract: dict[str, Any], step: int,
                         rank_batches: list[list[dict[str, Any]]],
                         *, order: list[int] | None = None) -> dict[str, Any]:
    """Verify actual rank/microbatch order and the loss denominator for one update."""
    samples, global_batch = contract["samples"], contract["global_batch_size"]
    updates = math.ceil(samples / global_batch)
    if not 1 <= step <= updates * contract["epochs"]:
        raise ValueError("Actual optimizer update is outside the stage budget")
    epoch, epoch_step = divmod(step - 1, updates)
    order = epoch_permutation(samples, epoch, contract["seed"]) if order is None else order
    expected = order[epoch_step * global_batch:(epoch_step + 1) * global_batch]
    world, micro = contract["world_size"], contract["per_device_batch_size"]
    microbatches = math.ceil(len(expected) / (world * micro))
    if len(rank_batches) != world or any(len(batches) != microbatches for batches in rank_batches):
        raise ValueError("Actual rank/microbatch coverage differs from the global update")
    indices, input_tokens, denominators = [], [], []
    padding = 0
    for offset in range(microbatches):
        for rank in range(world):
            batch = rank_batches[rank][offset]
            if any(len(batch[key]) != micro for key in ("indices", "assistant_tokens", "input_tokens")):
                raise ValueError("Actual per-rank batch width differs")
            denominators.append(batch["global_supervised_token_denominator"])
            for index, output_count, input_count in zip(batch["indices"], batch["assistant_tokens"], batch["input_tokens"]):
                if index == PADDING_INDEX:
                    if output_count != 0:
                        raise ValueError("Masked padding contributed supervised tokens")
                    padding += 1
                else:
                    if not 0 <= index < samples or output_count != contract["expected_assistant_tokens"][index] or input_count < output_count:
                        raise ValueError("Actual SID sample token counts differ")
                    indices.append(index)
                    input_tokens.append(input_count)
    if indices != expected:
        raise ValueError("Actual dataloader order differs from the frozen global permutation")
    supervised_total = sum(contract["expected_assistant_tokens"][i] for i in expected)
    if any(value != supervised_total for value in denominators):
        raise ValueError("Actual loss denominator differs from the global real shifted-label count")
    return {"global_step": step, "epoch_index": epoch, "epoch_optimizer_step": epoch_step + 1,
            "indices": indices, "indices_sha256": _canonical_sha(indices),
            "real_presentations": len(indices), "input_tokens": sum(input_tokens),
            "supervised_tokens": supervised_total, "masked_padding_slots": padding,
            "microbatches_per_rank": microbatches, "epoch_tail": epoch_step + 1 == updates,
            "global_supervised_token_denominator": supervised_total, "status": "passed"}


def audit_checkpoint_cursor(checkpoint: Path, contract: dict[str, Any]) -> dict[str, Any]:
    exposure = json.loads((checkpoint / "training_exposure_state.json").read_text())
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    step = int(trainer["global_step"])
    identity = exposure["identity"]
    if (identity["train_sha256"] != contract["dataset_sha256"]
            or identity["row_identity_sha256"] != contract["row_identity_sha256"]
            or identity["expected_epochs"] != contract["epochs"]
            or exposure["optimizer_step"] != step):
        raise ValueError("Saved exposure cursor belongs to another stage/step")
    counts, progress = expected_counts(contract["samples"], step, global_batch=contract["global_batch_size"],
                                       epochs=contract["epochs"], seed=contract["seed"])
    if exposure["counts"] != counts or exposure["assistant_tokens"] != expected_supervised_tokens(counts, contract["expected_assistant_tokens"]):
        raise ValueError("Saved actual exposure is not the exact stage cursor")
    return {"version": CURSOR_VERSION, "stage": contract["stage"], "global_step": step,
            "contract_sha256": _canonical_sha(contract), **progress,
            "exposure_sha256": _sha(checkpoint / "training_exposure_state.json"),
            "trainer_state_sha256": _sha(checkpoint / "trainer_state.json"), "status": "passed"}


def reuse_endpoint_cursor(checkpoint: Path, contract: dict[str, Any], *,
                         step: int, start_step: int, end_step: int) -> dict[str, Any]:
    if not 0 < start_step == step == end_step <= contract["total_steps"]:
        raise ValueError("Cursor reuse requires an already-completed exact endpoint with zero new updates")
    expected = audit_checkpoint_cursor(checkpoint, contract)
    saved = json.loads((checkpoint / "old_new_execution_cursor.json").read_text())
    if expected["global_step"] != step or any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed checkpoint cursor differs from its exact exposure state")
    last = saved["last_update_audit"]
    if (last["global_step"] != step or last["status"] != "passed"
            or last["record_sha256"] != _canonical_sha({key: value for key, value in last.items() if key != "record_sha256"})):
        raise ValueError("Completed checkpoint has an invalid final update audit")
    return saved


def make_cursor_callback(base_callback: type[Any], trainer: Any, exposure: Any,
                         contract: dict[str, Any], output_dir: Path,
                         *, stop_after_global_step: int | None = None) -> Any:
    class CursorCallback(base_callback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            counts, _ = expected_counts(contract["samples"], int(state.global_step),
                global_batch=contract["global_batch_size"], epochs=contract["epochs"], seed=contract["seed"])
            if exposure.counts != counts or exposure.assistant_tokens != expected_supervised_tokens(counts, contract["expected_assistant_tokens"]):
                raise ValueError("Actual restored exposure does not match the stage cursor")
            trainer._cursor_microbatches = []
            trainer._cursor_last_record = None
            trainer._cursor_start_step = int(state.global_step)
            trainer._cursor_end_step = stop_after_global_step if stop_after_global_step is not None else contract["total_steps"]
            if trainer._cursor_start_step > trainer._cursor_end_step:
                raise ValueError("Restored checkpoint is beyond the requested endpoint")
            if trainer._cursor_start_step == trainer._cursor_end_step:
                control.should_training_stop = True
            self.previous_sha = None
            self.orders = {}
            self.records = []
            def initialize_attempt() -> Path:
                root = output_dir / "cursor_attempts"
                root.mkdir(parents=True, exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
                (directory / "start.json").write_text(json.dumps({"contract": contract,
                    "restored_global_step": int(state.global_step), "status": "passed"}, indent=2) + "\n")
                return directory
            from new_interest.training.train_sft import _run_rank_zero_checkpoint_operation
            self.attempt_dir = Path(_run_rank_zero_checkpoint_operation(initialize_attempt))
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            import torch.distributed as distributed
            local = trainer._cursor_microbatches
            if distributed.is_initialized():
                gathered = [None] * int(args.world_size)
                distributed.all_gather_object(gathered, local)
            else:
                gathered = [local]
            epoch = (int(state.global_step) - 1) // math.ceil(contract["samples"] / contract["global_batch_size"])
            if epoch not in self.orders:
                self.orders[epoch] = epoch_permutation(contract["samples"], epoch, contract["seed"])
            record = audit_update_batches(contract, int(state.global_step), gathered, order=self.orders[epoch])
            record["previous_record_sha256"] = self.previous_sha
            record["record_sha256"] = _canonical_sha(record)
            self.previous_sha = record["record_sha256"]
            trainer._cursor_last_record = record
            trainer._cursor_microbatches = []
            if state.is_world_process_zero:
                self.records.append(record)
                if record["epoch_tail"]:
                    print("OLD_NEW_EPOCH_TAIL_VERIFIED", contract["stage"], state.global_step,
                          record["real_presentations"], record["masked_padding_slots"], flush=True)
            if stop_after_global_step is not None and int(state.global_step) >= stop_after_global_step:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.flush()
            if state.is_world_process_zero:
                trainer.save_migration_cursor(state, Path(args.output_dir) / f"checkpoint-{int(state.global_step)}")
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.flush()
            checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            if state.is_world_process_zero and checkpoint.exists():
                trainer.save_migration_cursor(state, output_dir)
            return control

        def flush(self) -> None:
            from new_interest.training.train_sft import _run_rank_zero_checkpoint_operation
            def persist() -> Path:
                destination = self.attempt_dir / "updates.jsonl"
                temporary = self.attempt_dir / "updates.partial"
                temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in self.records))
                temporary.replace(destination)
                return destination
            _run_rank_zero_checkpoint_operation(persist)

    return CursorCallback()


def wrap_trainer(base_trainer: type[Any], cursor_contract: dict[str, Any]) -> type[Any]:
    """Inject a full-epoch sampler; do not pre-slice or double-skip the cursor."""
    contract = dict(cursor_contract)
    if contract["version"] != CURSOR_VERSION or contract["ignore_data_skip"] is not False:
        raise ValueError("Unsupported stage cursor protocol")

    class StageCursorTrainer(base_trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if len(self.train_dataset) != contract["samples"]:
                raise ValueError("Trainer dataset differs from its stage cursor")
            for actual, expected, name in [
                (self.args.world_size, contract["world_size"], "world size"),
                (self.args.per_device_train_batch_size, contract["per_device_batch_size"], "microbatch"),
                (self.args.gradient_accumulation_steps, contract["gradient_accumulation_steps"], "accumulation"),
            ]:
                if actual != expected:
                    raise ValueError(f"Stage cursor execution {name} differs")
            self.args.ignore_data_skip = False
            self.args.dataloader_drop_last = False
            self.args.average_tokens_across_devices = True
            if not self.model_accepts_loss_kwargs or self.accelerator.gradient_accumulation_steps != 1:
                raise ValueError("Global token mean requires the validated Trainer loss/accumulation path")
            self._loss_shifts_labels = True
            self._cursor_microbatches = []
            self._cursor_last_record = None
            self._cursor_start_step = None
            self._cursor_end_step = contract["total_steps"]
            self._cursor_reused_endpoint = False
            self._old_new_contract = contract

        def _run_epoch(self, model: Any, epoch: int, train_dataloader: Any,
                       *args: Any, **kwargs: Any) -> Any:
            # A completed bounded probe may still have unused Full epochs.
            # Preserve the original scheduler horizon while finalizing without
            # an accidental extra update after a publication interruption.
            if int(self.state.global_step) >= self._cursor_end_step:
                return None
            # Trainer sets the epoch after Accelerate adds SkipBatchSampler.
            # That extra wrapper hides our sampler from its nested set_epoch
            # forwarding, so set the original loader before the native skip.
            train_dataloader.set_epoch(epoch)
            return super()._run_epoch(model, epoch, train_dataloader, *args, **kwargs)

        def record_cursor_microbatch(self, metadata: dict[str, Any]) -> None:
            self._cursor_microbatches.append({**{key: list(value) for key, value in metadata.items()},
                "global_supervised_token_denominator": self._cursor_last_denominator})

        def save_migration_cursor(self, state: Any, destination: Path) -> dict[str, Any]:
            # Shared end hooks write to the stage directory, while on_save
            # writes to the checkpoint. Both derive evidence from the same
            # already-saved checkpoint payload.
            checkpoint = destination
            if not (checkpoint / "training_exposure_state.json").exists():
                checkpoint = Path(self.args.output_dir) / f"checkpoint-{int(state.global_step)}"
            record = audit_checkpoint_cursor(checkpoint, contract)
            if self._cursor_last_record is None:
                record = reuse_endpoint_cursor(checkpoint, contract, step=int(state.global_step),
                    start_step=self._cursor_start_step, end_step=self._cursor_end_step)
                self._cursor_reused_endpoint = True
            else:
                if self._cursor_last_record["global_step"] != int(state.global_step):
                    raise ValueError("Checkpoint lacks a verified actual update cursor")
                record["last_update_audit"] = self._cursor_last_record
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / "old_new_execution_cursor.json"
            text = json.dumps(record, sort_keys=True, indent=2) + "\n"
            if path.exists() and path.read_text() != text:
                previous = json.loads(path.read_text())
                if (destination.resolve() == checkpoint.resolve()
                        or previous["contract_sha256"] != record["contract_sha256"]
                        or previous["global_step"] >= record["global_step"]):
                    raise ValueError("Existing checkpoint cursor audit differs")
                history = destination / "cursor_completion_history" / f"step-{previous['global_step']}-{_sha(path)[:16]}.json"
                history.parent.mkdir(parents=True, exist_ok=True)
                if history.exists() and history.read_bytes() != path.read_bytes():
                    raise ValueError("Previous endpoint evidence changed")
                history.write_bytes(path.read_bytes())
            path.write_text(text)
            if self._cursor_reused_endpoint and destination.resolve() != checkpoint.resolve():
                (destination / "cursor_completion_recovery.json").write_text(json.dumps({
                    "global_step": int(state.global_step), "new_optimizer_updates": 0,
                    "source_cursor_sha256": _sha(checkpoint / "old_new_execution_cursor.json"),
                    "reused_endpoint_cursor": True, "full_stage_completed": int(state.global_step) == contract["total_steps"],
                }, indent=2) + "\n")
            return record

        def get_train_dataloader(self) -> Any:
            import torch
            from torch.utils.data import DataLoader
            from accelerate.data_loader import prepare_data_loader

            sampler = PaddedGlobalBatchSampler(len(self.train_dataset), world_size=contract["world_size"],
                per_device_batch_size=contract["per_device_batch_size"], global_batch_size=contract["global_batch_size"], seed=contract["seed"])
            loader = DataLoader(self.train_dataset, batch_sampler=sampler, collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
                generator=torch.Generator(device="cpu").manual_seed(contract["seed"]))
            return prepare_data_loader(loader, device=self.accelerator.device,
                num_processes=self.accelerator.num_processes, process_index=self.accelerator.process_index,
                split_batches=True, put_on_device=True, even_batches=False, dispatch_batches=False,
                rng_types=[], use_seedable_sampler=False)

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                         num_items_in_batch: Any | None = None) -> Any:
            if num_items_in_batch is None or float(num_items_in_batch) <= 0:
                raise ValueError("Global real-token denominator is missing or empty")
            self._cursor_last_denominator = int(num_items_in_batch)
            # The native Trainer/model token-mean path sums local valid losses,
            # divides by the gathered real-token denominator, and compensates
            # for DDP's gradient average. A fully masked rank contributes zero.
            result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                          num_items_in_batch=num_items_in_batch)
            return result

    return StageCursorTrainer
