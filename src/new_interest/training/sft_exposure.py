"""Count successful training microbatches without changing SFT inputs or loss."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPOSURE_STATE = "training_exposure_state.json"


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


class TrainingExposure:
    def __init__(
        self, rows: list[dict[str, str]], *, train_sha256: str, expected_epochs: int
    ) -> None:
        if not rows or expected_epochs < 1:
            raise ValueError("Exposure audit requires rows and a positive integer epoch count")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Exposure audit sample IDs must be unique")
        if any(row["training_task"] not in {
            "old_interest", "new_interest", "cross_interest_reconstruction", "interest_name_to_id", "interest_id_to_name"
        } for row in rows):
            raise ValueError("Unknown interest training task")
        self.rows = rows
        self.expected_epochs = expected_epochs
        self.counts = [0] * len(rows)
        self.assistant_tokens = [0] * len(rows)
        self.input_tokens = [0] * len(rows)
        self.successful_microbatches = 0
        self.padding_slots_ignored = 0
        self._synced_counts = list(self.counts)
        self._synced_assistant_tokens = list(self.assistant_tokens)
        self._synced_input_tokens = list(self.input_tokens)
        self._synced_microbatches = 0
        self._synced_padding_slots = 0
        row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.identity = {
            "version": "sft-observed-exposure-v1",
            "train_sha256": train_sha256,
            "row_identity_sha256": row_hash,
            "samples": len(rows),
            "expected_epochs": expected_epochs,
        }

    @classmethod
    def from_parquet(
        cls, path: Path, *, train_sha256: str, expected_epochs: float,
        max_samples: int | None, default_task: str,
    ) -> TrainingExposure:
        if not float(expected_epochs).is_integer():
            raise ValueError("Exact exposure audit requires complete integer epochs")
        columns = ["sample_id", "target_subcategory"]
        if "training_task" in pq.ParquetFile(path).schema_arrow.names:
            columns.append("training_task")
        rows = pq.read_table(path, columns=columns).to_pylist()
        if max_samples is not None:
            rows = rows[:max_samples]
        normalized = [{
            "sample_id": str(row["sample_id"]),
            "target_subcategory": str(row["target_subcategory"]),
            "training_task": str(row.get("training_task", default_task)),
        } for row in rows]
        return cls(normalized, train_sha256=train_sha256, expected_epochs=int(expected_epochs))

    def record(self, indices: list[int], assistant_tokens: list[int], input_tokens: list[int]) -> None:
        if not (len(indices) == len(assistant_tokens) == len(input_tokens)) or not indices:
            raise ValueError("Exposure microbatch metadata lengths differ")
        for index, output_count, input_count in zip(indices, assistant_tokens, input_tokens):
            if index == -1 and output_count == 0 and input_count > 0:
                continue
            if not 0 <= index < len(self.rows) or not 0 < output_count <= input_count:
                raise ValueError("Invalid exposure index or token count")
        for index, output_count, input_count in zip(indices, assistant_tokens, input_tokens):
            if index == -1:
                self.padding_slots_ignored += 1
                continue
            self.counts[index] += 1
            self.assistant_tokens[index] += output_count
            self.input_tokens[index] += input_count
        self.successful_microbatches += 1

    @staticmethod
    def _is_primary() -> bool:
        try:
            import torch.distributed as distributed
        except ImportError:
            return True
        return not distributed.is_initialized() or distributed.get_rank() == 0

    def synchronize(self) -> None:
        """Sum new rank-local deltas once; restored global counters are not summed."""
        try:
            import torch
            import torch.distributed as distributed
        except ImportError:
            return
        if not distributed.is_initialized() or distributed.get_world_size() == 1:
            return
        device = torch.device("cuda", torch.cuda.current_device()) if distributed.get_backend() == "nccl" else torch.device("cpu")
        vectors = [(self.counts, self._synced_counts),
                   (self.assistant_tokens, self._synced_assistant_tokens),
                   (self.input_tokens, self._synced_input_tokens)]
        values = [value - previous for current, base in vectors for value, previous in zip(current, base)]
        values += [self.successful_microbatches - self._synced_microbatches,
                   self.padding_slots_ignored - self._synced_padding_slots]
        if any(value < 0 for value in values):
            raise ValueError("Rank-local exposure counters regressed")
        tensor = torch.tensor(values, dtype=torch.int64, device=device)
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        merged = tensor.cpu().tolist()
        n = len(self.rows)
        for offset, (current, base) in enumerate(vectors):
            current[:] = [previous + delta for previous, delta in zip(base, merged[offset * n:(offset + 1) * n])]
            base[:] = current
        self.successful_microbatches = self._synced_microbatches + merged[-2]
        self.padding_slots_ignored = self._synced_padding_slots + merged[-1]
        self._synced_microbatches = self.successful_microbatches
        self._synced_padding_slots = self.padding_slots_ignored

    def save_checkpoint(self, directory: Path, step: int) -> None:
        self.synchronize()
        if not self._is_primary():
            return
        _atomic_json(directory / EXPOSURE_STATE, {
            "identity": self.identity,
            "optimizer_step": step,
            "counts": self.counts,
            "assistant_tokens": self.assistant_tokens,
            "input_tokens": self.input_tokens,
            "successful_microbatches": self.successful_microbatches,
            "padding_slots_ignored": self.padding_slots_ignored,
        })

    def restore_checkpoint(self, directory: Path) -> None:
        data = json.loads((directory / EXPOSURE_STATE).read_text())
        trainer_state = json.loads((directory / "trainer_state.json").read_text())
        if data["identity"] != self.identity or data["optimizer_step"] != trainer_state["global_step"]:
            raise ValueError("Exposure checkpoint identity/step mismatch")
        vectors = [data[key] for key in ("counts", "assistant_tokens", "input_tokens")]
        if any(len(values) != len(self.rows) for values in vectors):
            raise ValueError("Exposure checkpoint vector length mismatch")
        if any(type(value) is not int or value < 0 for values in vectors for value in values):
            raise ValueError("Exposure checkpoint counters must be nonnegative integers")
        if type(data["successful_microbatches"]) is not int or data["successful_microbatches"] < 0:
            raise ValueError("Invalid successful microbatch counter")
        if any(output > input_count for output, input_count in zip(vectors[1], vectors[2])):
            raise ValueError("Exposure checkpoint token counters are inconsistent")
        self.counts, self.assistant_tokens, self.input_tokens = vectors
        self.successful_microbatches = data["successful_microbatches"]
        self.padding_slots_ignored = int(data.get("padding_slots_ignored", 0))
        self._synced_counts = list(self.counts)
        self._synced_assistant_tokens = list(self.assistant_tokens)
        self._synced_input_tokens = list(self.input_tokens)
        self._synced_microbatches = self.successful_microbatches
        self._synced_padding_slots = self.padding_slots_ignored

    def finish(self, directory: Path, *, optimizer_steps: int) -> dict[str, Any]:
        return self._finish(directory, optimizer_steps=optimizer_steps,
                            expected_counts=[self.expected_epochs] * len(self.rows), complete=True)

    def finish_cursor(self, directory: Path, *, optimizer_steps: int,
                      expected_counts: list[int]) -> dict[str, Any]:
        """Accept a bounded migration probe without claiming full training."""
        return self._finish(directory, optimizer_steps=optimizer_steps,
                            expected_counts=expected_counts, complete=False)

    def _finish(self, directory: Path, *, optimizer_steps: int,
                expected_counts: list[int], complete: bool) -> dict[str, Any]:
        self.synchronize()
        if len(expected_counts) != len(self.rows) or any(
            type(count) is not int or not 0 <= count <= self.expected_epochs for count in expected_counts
        ):
            raise ValueError("Invalid expected exposure cursor")
        wrong = [self.rows[index]["sample_id"] for index, count in enumerate(self.counts)
                 if count != expected_counts[index]]
        if wrong:
            raise ValueError(f"Actual exposure differs from the frozen epoch budget for {len(wrong)} samples")
        by_task: dict[str, Counter[str]] = defaultdict(Counter)
        by_class: dict[str, Counter[str]] = defaultdict(Counter)
        records = []
        for index, row in enumerate(self.rows):
            task = row["training_task"]
            by_task[task]["unique_samples"] += 1
            by_task[task]["presentations"] += self.counts[index]
            by_task[task]["assistant_tokens"] += self.assistant_tokens[index]
            by_task[task]["input_tokens"] += self.input_tokens[index]
            by_class[task][row["target_subcategory"]] += self.counts[index]
            records.append({**row, "presentations": self.counts[index], "assistant_tokens": self.assistant_tokens[index], "input_tokens": self.input_tokens[index]})
        summary = {
            "identity": self.identity,
            "status": "verified_observed_training" if complete else "verified_observed_cursor",
            "optimizer_steps": optimizer_steps,
            "successful_microbatches": self.successful_microbatches,
            "padding_slots_ignored": self.padding_slots_ignored,
            "presentations": sum(self.counts),
            "min_presentations_per_sample": min(self.counts),
            "max_presentations_per_sample": max(self.counts),
            "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
            "by_task_and_class_presentations": {key: dict(value) for key, value in sorted(by_class.items())},
        }
        if self._is_primary():
            name = "training_exposure.json" if complete else "training_exposure_partial.json"
            _atomic_json(directory / name, {**summary, "samples": records})
        return summary


class ExposureIndexedDataset:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self._dummy_index = min(range(len(dataset)), key=lambda index: len(dataset[index]["input_ids"])) if len(dataset) else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index == -1:
            if self._dummy_index is None:
                raise ValueError("Cannot pad an empty dataset")
            example = dict(self.dataset[self._dummy_index])
            example["labels"] = [-100] * len(example["labels"])
            example["_exposure_index"] = -1
            return example
        return {**self.dataset[index], "_exposure_index": index}


class ExposureCollator:
    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [{key: value for key, value in feature.items() if key != "_exposure_index"} for feature in features]
        batch = self.collator(inputs)
        batch["_exposure_metadata"] = {
            "indices": [int(feature["_exposure_index"]) for feature in features],
            "assistant_tokens": [sum(label != -100 for label in feature["labels"][1:]) for feature in features],
            "input_tokens": [sum(feature["attention_mask"]) for feature in features],
        }
        return batch


def make_exposure_trainer(base_trainer: type[Any], exposure: TrainingExposure) -> type[Any]:
    class ExposureTrainer(base_trainer):
        def training_step(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            model_inputs = dict(inputs)
            metadata = model_inputs.pop("_exposure_metadata")
            result = super().training_step(model, model_inputs, *args, **kwargs)
            # Count only microbatches whose actual forward/backward completed.
            exposure.record(**metadata)
            cursor_hook = getattr(self, "record_cursor_microbatch", None)
            if cursor_hook is not None:
                cursor_hook(metadata)
            return result

    return ExposureTrainer


def make_exposure_callback(base_callback: type[Any], exposure: TrainingExposure) -> Any:
    class ExposureCallback(base_callback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            exposure.save_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", int(state.global_step))
            return control

    return ExposureCallback()
