"""Audit the exact global sample cursor for the divisible NI SFT dataset.

The frozen HF/Accelerate seedable sampler and its ordinary data skip are kept.
Every real microbatch is checked across ranks before any model forward pass.
Non-divisible datasets must supply their own masked-tail trainer extension.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

from new_interest.training.execution_migration import canonical_sha256

CURSOR_PROTOCOL = "hf-seedable-global-sample-cursor-v1"
ROW_KEY = "_migration_row_indices"


def epoch_order(rows: int, seed: int, epoch: int) -> list[int]:
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return torch.randperm(rows, generator=generator).tolist()


def index_sha256(indices: list[int]) -> str:
    import numpy as np
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def build_hf_cursor_contract(dataset_sha256: str, rows: int, seed: int, epochs: int, trainer_state: dict) -> dict:
    if rows <= 0 or rows % 32:
        raise ValueError("The NI cursor requires a row count divisible by global batch 32")
    updates = rows // 32
    step = int(trainer_state["global_step"])
    if int(trainer_state["max_steps"]) != epochs * updates:
        raise ValueError("Checkpoint optimizer budget differs from dataset cursor")
    if not math.isclose(float(trainer_state["epoch"]), step / updates, abs_tol=1e-10):
        raise ValueError("Checkpoint epoch and consumed sample cursor differ")
    orders = [epoch_order(rows, seed, epoch) for epoch in range(epochs)]
    prefix = [index for order in orders for index in order][:step * 32]
    return {"protocol": CURSOR_PROTOCOL, "dataset_sha256": dataset_sha256, "rows": rows,
            "seed": seed, "epochs": epochs, "global_batch_size": 32,
            "updates_per_epoch": updates, "total_steps": epochs * updates,
            "source_global_step": step, "source_epoch": trainer_state["epoch"],
            "consumed_global_samples": step * 32, "consumed_prefix_sha256": index_sha256(prefix),
            "epoch_order_sha256": [index_sha256(order) for order in orders],
            "ignore_data_skip": False, "use_seedable_sampler": True,
            "sampler_seed_rule": "torch.randperm(rows, cpu_generator_seed=seed+epoch)",
            "distributed_partition": "microbatch-major then rank-major contiguous slices"}


def reuse_completed_cursor(checkpoint: Path, target_identity: dict, contract: dict, destination: Path) -> dict:
    from new_interest.training.checkpoint_persistence import validate_persistent_checkpoint
    from new_interest.training.execution_migration import file_sha256, load_migration_receipt
    marker = validate_persistent_checkpoint(checkpoint, target_identity)
    load_migration_receipt(checkpoint, target_identity)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    total = int(contract["total_steps"])
    if marker["global_step"] != total or state["global_step"] != total or state["max_steps"] != total or state["epoch"] != contract["epochs"]:
        raise ValueError("Only a fully completed same-stage checkpoint can reuse its final cursor")
    source = checkpoint / "training_sample_cursor.json"
    cursor = json.loads(source.read_text())
    expected = [index for epoch in range(int(contract["epochs"]))
                for index in epoch_order(int(contract["rows"]), int(contract["seed"]), epoch)]
    if (cursor["global_step"] != total or cursor["dataset_sha256"] != contract["dataset_sha256"]
            or cursor["consumed_global_samples"] != len(expected)
            or cursor["consumed_prefix_sha256"] != index_sha256(expected)
            or not cursor["all_rank_real_sample_order_exact"]):
        raise ValueError("Completed checkpoint cursor does not cover the exact full sequence")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "training_sample_cursor.json"
    if target.resolve() != source.resolve():
        shutil.copyfile(source, target)
    (destination / "cursor_completion_recovery.json").write_text(json.dumps({
        "checkpoint_name": checkpoint.name, "global_step": total, "new_optimizer_updates": 0,
        "source_cursor_sha256": file_sha256(source), "reused_final_cursor": True,
    }, indent=2) + "\n")
    return cursor


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], ROW_KEY: int(index)}


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        import torch
        indices = [item[ROW_KEY] for item in features]
        clean = [{key: value for key, value in item.items() if key != ROW_KEY} for item in features]
        batch = self.collator(clean)
        batch[ROW_KEY] = torch.tensor(indices, dtype=torch.long)
        return batch


def make_hf_cursor_trainer(base_trainer, contract: dict):
    if contract.get("protocol") != CURSOR_PROTOCOL:
        raise ValueError("Use the owning workflow's trainer for this cursor protocol")
    rows, seed, epochs = int(contract["rows"]), int(contract["seed"]), int(contract["epochs"])
    orders = [epoch_order(rows, seed, epoch) for epoch in range(epochs)]
    if [index_sha256(order) for order in orders] != contract["epoch_order_sha256"]:
        raise ValueError("Current Torch sampler order differs from the frozen cursor")
    full_order = [index for order in orders for index in order]
    if index_sha256(full_order[:int(contract["source_global_step"]) * 32]) != contract["consumed_prefix_sha256"]:
        raise ValueError("Original consumed prefix does not match the cursor contract")

    class HFCursorTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            dataset = kwargs["train_dataset"]
            if len(dataset) != rows:
                raise ValueError("Cursor dataset row count differs")
            kwargs["train_dataset"] = IndexedDataset(dataset)
            kwargs["data_collator"] = IndexedCollator(kwargs["data_collator"])
            super().__init__(*args, **kwargs)
            if self.args.ignore_data_skip or not self.accelerator.use_seedable_sampler:
                raise ValueError("The NI cursor requires exactly one HF data-skip operation")
            if self.args.data_seed != seed or self.args.train_sampling_strategy != "random":
                raise ValueError("The NI cursor requires the original seeded random sampler")
            if self.args.dataloader_drop_last:
                raise ValueError("The NI cursor cannot drop real rows")
            self._cursor_step = None
            self._cursor_microbatch = 0
            self._cursor_checked_updates = 0
            self._cursor_first_indices = None

        def training_step(self, model, inputs, num_items_in_batch=None):
            import torch.distributed as dist
            if ROW_KEY not in inputs:
                raise ValueError("Actual batch lacks its sample cursor indices")
            indices = inputs.pop(ROW_KEY).detach().cpu().tolist()
            step = int(self.state.global_step)
            if self._cursor_step != step:
                if self._cursor_step is not None and self._cursor_microbatch != self.args.gradient_accumulation_steps:
                    raise ValueError("An optimizer update did not consume its complete global batch")
                self._cursor_step, self._cursor_microbatch = step, 0
            gathered = [None] * int(self.args.world_size)
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(gathered, indices)
            else:
                gathered = [indices]
            actual = [index for rank_indices in gathered for index in rank_indices]
            width = int(self.args.world_size) * int(self.args.per_device_train_batch_size)
            offset = step * 32 + self._cursor_microbatch * width
            expected = full_order[offset:offset + width]
            if actual != expected:
                raise ValueError(f"Global sample cursor differs at update {step}, microbatch {self._cursor_microbatch}")
            if self._cursor_first_indices is None:
                self._cursor_first_indices = {"global_step": step, "indices": actual}
            self._cursor_microbatch += 1
            if self._cursor_microbatch == self.args.gradient_accumulation_steps:
                self._cursor_checked_updates += 1
            return super().training_step(model, inputs, num_items_in_batch)

        def save_migration_cursor(self, state, checkpoint_dir):
            step = int(state.global_step)
            if self._cursor_step is None and step == int(contract["total_steps"]):
                reuse = getattr(self, "reuse_completed_sample_cursor", None)
                if reuse is None:
                    raise ValueError("Completed checkpoint recovery lacks a validated final-cursor source")
                return reuse(state, Path(checkpoint_dir))
            if self._cursor_microbatch != self.args.gradient_accumulation_steps or self._cursor_step != step - 1:
                raise ValueError("Cannot publish a cursor before its optimizer update is complete")
            evidence = {"protocol": CURSOR_PROTOCOL, "dataset_sha256": contract["dataset_sha256"],
                        "contract_sha256": canonical_sha256(contract), "global_step": step,
                        "epoch": step / int(contract["updates_per_epoch"]),
                        "consumed_global_samples": step * 32,
                        "consumed_prefix_sha256": index_sha256(full_order[:step * 32]),
                        "next_global_batch_sha256": index_sha256(full_order[step * 32:(step + 1) * 32]),
                        "actual_updates_audited_this_attempt": self._cursor_checked_updates,
                        "first_actual_global_microbatch": self._cursor_first_indices,
                        "all_rank_real_sample_order_exact": True}
            Path(checkpoint_dir, "training_sample_cursor.json").write_text(json.dumps(evidence, indent=2) + "\n")
            return evidence

    return HFCursorTrainer
