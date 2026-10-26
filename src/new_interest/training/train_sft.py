from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

from new_interest.training.checkpoint_persistence import (
    hydrate_checkpoint,
    prune_persistent_checkpoints,
    publish_checkpoint,
    select_latest_complete_checkpoint,
)
from new_interest.training.mera_regularization import (
    build_mera_regularizer,
)
from new_interest.training.sft_data import (
    CompletionOnlyCollator,
    ParquetSFTDataset,
    audit_token_lengths,
)
from new_interest.training.sft_exposure import (
    ExposureCollator,
    ExposureIndexedDataset,
    TrainingExposure,
    make_exposure_callback,
    make_exposure_trainer,
)
from new_interest.training.sft_state_audit import (
    make_training_start_audit_callback,
    record_initial_adapter_state,
)
from new_interest.training.text_adapter_reload import (
    capture_text_reload_reference,
)
from new_interest.training.semantic_id_tokens import (
    audit_interest_token_drift,
    capture_reload_reference,
    initialize_semantic_interest_token_rows,
    register_semantic_interest_tokens,
    trainable_token_indices,
    write_json,
)


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a configuration object")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_checkpoint_identity(
    config: dict[str, Any],
    *,
    train_file: Path,
    sample_weight_file: Path | None,
    adapter_path: Path | None,
) -> dict[str, Any]:
    """Build the frozen identity required for safe checkpoint reuse."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_batch_size = int(
        config.get("per_device_train_batch_size", 4)
    )
    gradient_accumulation_steps = int(
        config.get("gradient_accumulation_steps", 8)
    )
    adapter_weights = (
        adapter_path / "adapter_model.safetensors"
        if adapter_path is not None
        else None
    )
    mera_relations_value = config.get("mera_relations_file")
    mera_relations_file = (
        Path(str(mera_relations_value))
        if mera_relations_value is not None
        else None
    )
    identity = {
        "identity_version": "trainer-checkpoint-identity-v1",
        "model_name": str(config["model_name"]),
        "model_revision": config.get("model_revision"),
        "run_name": str(config["run_name"]),
        "train_file_sha256": _sha256_file(train_file),
        "sample_weight_file_sha256": (
            _sha256_file(sample_weight_file)
            if sample_weight_file is not None
            else None
        ),
        "loss_reduction": str(config.get("loss_reduction", "token_mean")),
        "semantic_id_loss_scope": str(
            config.get("semantic_id_loss_scope", "full_vocabulary")
        ),
        "input_adapter_sha256": (
            _sha256_file(adapter_weights)
            if adapter_weights is not None
            else None
        ),
        "max_samples": (
            int(config["max_samples"])
            if config.get("max_samples") is not None
            else None
        ),
        "epochs": float(config.get("epochs", 3)),
        "seed": int(config.get("seed", 42)),
        "enable_thinking": bool(config.get("enable_thinking", False)),
        "max_seq_length": int(config.get("max_seq_length", 2048)),
        "world_size": world_size,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_size": (
            world_size
            * per_device_batch_size
            * gradient_accumulation_steps
        ),
        "learning_rate": float(config.get("learning_rate", 2e-4)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(
            config.get("lr_scheduler_type", "cosine")
        ),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
        "optim": str(config.get("optim", "adamw_torch_fused")),
        "lora_r": int(config.get("lora_r", 16)),
        "lora_alpha": int(config.get("lora_alpha", 32)),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "target_modules": config.get("target_modules", "all-linear"),
        "gradient_checkpointing_kwargs": (
            _gradient_checkpointing_kwargs(config)
        ),
    }
    if "semantic_id_initialization_mode" in config:
        identity["semantic_id_initialization_mode"] = str(
            config["semantic_id_initialization_mode"]
        )
    if bool(config.get("audit_training_exposure", False)):
        identity["training_exposure_audit"] = "sft-observed-exposure-v1"
        identity["exposure_default_task"] = str(
            config.get("exposure_default_task", "new_interest")
        )
    if "gradient_checkpointing_enabled" in config:
        identity["gradient_checkpointing_enabled"] = bool(config["gradient_checkpointing_enabled"])
    if mera_relations_file is not None:
        identity.update(
            {
                "mera_relations_file_sha256": _sha256_file(
                    mera_relations_file
                ),
                "mera_structure_weight": float(
                    config.get("mera_structure_weight", 0.05)
                ),
                "mera_collaborative_weight": float(
                    config.get("mera_collaborative_weight", 0.1)
                ),
                "mera_negative_samples": int(
                    config.get("mera_negative_samples", 10)
                ),
                "mera_cooccurrence_threshold": float(
                    config.get("mera_cooccurrence_threshold", 0.05)
                ),
                "mera_temperature": float(
                    config.get("mera_temperature", 1.0)
                ),
            }
        )
    return identity


def _run_rank_zero_checkpoint_operation(
    operation: Any,
) -> str | None:
    """Run checkpoint I/O on rank zero and propagate its outcome to all ranks."""

    import torch.distributed as distributed

    is_distributed = (
        distributed.is_available() and distributed.is_initialized()
    )
    rank = distributed.get_rank() if is_distributed else 0
    outcome: list[str | None] = [None, None]
    single_node = (is_distributed and distributed.get_world_size() > 1
                   and int(os.environ.get("LOCAL_WORLD_SIZE", "0")) == distributed.get_world_size())
    if single_node:
        import tempfile
        import time
        # Only the small rendezvous announcement is collective. Mounted I/O
        # happens while other ranks wait on this shared node-local file.
        rendezvous = [tempfile.mkdtemp(prefix="sft-checkpoint-rendezvous-") if rank == 0 else None]
        distributed.broadcast_object_list(rendezvous, src=0)
        directory = Path(str(rendezvous[0]))
        finished = directory / "outcome.json"
        if rank == 0:
            try:
                value = operation()
                outcome[0] = str(value) if value is not None else None
            except Exception as error:
                outcome[1] = f"{type(error).__name__}: {error}"
            temporary = directory / "outcome.partial"
            temporary.write_text(json.dumps(outcome) + "\n")
            temporary.replace(finished)
        else:
            deadline = time.monotonic() + 900
            while not finished.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Rank-zero checkpoint operation did not publish its outcome within 900 seconds")
                time.sleep(0.05)
            outcome = json.loads(finished.read_text())
        # Rank zero has finished I/O before it can enter this short collective.
        distributed.barrier()
        if rank == 0 and outcome[1] is None:
            shutil.rmtree(directory)
        if outcome[1] is not None:
            raise RuntimeError(outcome[1])
        return outcome[0]
    if is_distributed:
        # Every rank writes its own RNG state before the save callback. Wait
        # until all of those files are closed before rank zero publishes.
        distributed.barrier()
    if rank == 0:
        try:
            value = operation()
            outcome[0] = str(value) if value is not None else None
        except Exception as error:  # propagated so every rank exits together
            outcome[1] = f"{type(error).__name__}: {error}"
    if is_distributed:
        distributed.broadcast_object_list(outcome, src=0)
    if outcome[1] is not None:
        raise RuntimeError(outcome[1])
    return outcome[0]


def _prepare_automatic_resume(
    *,
    persistent_checkpoint_root: Path,
    local_checkpoint_root: Path,
    identity: dict[str, Any],
) -> str | None:
    def prepare() -> Path | None:
        selected, diagnostics = select_latest_complete_checkpoint(
            persistent_checkpoint_root,
            identity,
        )
        for diagnostic in diagnostics:
            print("CHECKPOINT_IGNORED", diagnostic, flush=True)
        if selected is None:
            if local_checkpoint_root.exists():
                shutil.rmtree(local_checkpoint_root)
            local_checkpoint_root.mkdir(parents=True, exist_ok=True)
            print("CHECKPOINT_AUTO_RESUME_NONE", flush=True)
            return None
        if local_checkpoint_root.exists():
            shutil.rmtree(local_checkpoint_root)
        local_checkpoint_root.mkdir(parents=True, exist_ok=True)
        hydrated = hydrate_checkpoint(
            selected,
            local_checkpoint_root,
            identity,
        )
        print(
            "CHECKPOINT_AUTO_RESUME_READY",
            selected,
            hydrated,
            flush=True,
        )
        return hydrated

    return _run_rank_zero_checkpoint_operation(prepare)


def _resolve_adapter_path(
    config: dict[str, Any],
) -> Path | None:
    value = config.get("adapter_path")
    if value is None:
        return None
    path = Path(str(value))
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Input adapter not found: {path}")
    return path


def _gradient_checkpointing_kwargs(
    config: dict[str, Any],
) -> dict[str, bool]:
    return {
        "use_reentrant": bool(
            config.get(
                "gradient_checkpointing_use_reentrant",
                True,
            )
        )
    }


def _restore_configured_save_steps(args: Any, state: Any) -> None:
    """Keep a resumed Trainer on the current checkpoint cadence.

    Transformers restores ``TrainerState.save_steps`` from the checkpoint.
    That serialized value can be stale when a recovery run intentionally
    changes ``TrainingArguments.save_steps``.  The default flow callback uses
    the state value to decide when to save, so refresh it after resume loading.
    """

    state.save_steps = args.save_steps


def _unwrap_model_for_reload_reference(trainer: Any) -> Any:
    """Remove Accelerate's mixed-precision forward wrapper after training."""

    model = trainer.accelerator.unwrap_model(
        trainer.model_wrapped,
        keep_fp32_wrapper=False,
        keep_torch_compile=True,
    )
    if hasattr(model, "_original_forward"):
        raise RuntimeError(
            "Accelerate mixed-precision forward wrapper was not removed"
        )
    return model


def _trainable_parameter_group(name: str) -> str:
    for group in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        if f".{group}." in name:
            return group
    return "other"


def audit_trainable_parameters(
    model: Any,
    *,
    forbidden_substrings: tuple[str, ...] = (),
) -> dict[str, object]:
    total_parameters = 0
    trainable_parameters = 0
    trainable_groups: Counter[str] = Counter()
    trainable_names: list[str] = []
    forbidden = tuple(value.lower() for value in forbidden_substrings)

    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameters += count
        if not parameter.requires_grad:
            continue
        trainable_parameters += count
        trainable_names.append(name)
        trainable_groups[_trainable_parameter_group(name)] += count

    forbidden_matches = sorted(
        name
        for name in trainable_names
        if any(value in name.lower() for value in forbidden)
    )
    base_model = (
        model.get_base_model()
        if hasattr(model, "get_base_model")
        else model
    )
    config = getattr(base_model, "config", None)
    architectures = getattr(config, "architectures", None)
    return {
        "wrapped_model_class": type(model).__name__,
        "base_model_class": type(base_model).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": (
            list(architectures)
            if architectures is not None
            else None
        ),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_ratio": (
            trainable_parameters / total_parameters
            if total_parameters
            else 0.0
        ),
        "trainable_parameter_groups": dict(
            sorted(trainable_groups.items())
        ),
        "trainable_parameter_tensors": len(trainable_names),
        "forbidden_substrings": list(forbidden_substrings),
        "forbidden_trainable_parameters": forbidden_matches,
    }


def configure_row_only_adapter_saves(model: Any) -> None:
    """Keep Semantic ID checkpoints limited to adapter and token-row state.

    PEFT's default ``save_embedding_layers='auto'`` saves the complete input
    and output matrices whenever the tokenizer vocabulary was resized. The
    trainable-token adapter already stores replacement values for every new
    row, so those full matrices are unnecessary and would make each checkpoint
    several gigabytes. Patching the instance method also covers Trainer's
    intermediate checkpoints, not only the final save.
    """

    if bool(getattr(model, "_semantic_row_only_saves", False)):
        return
    original_save_pretrained = model.save_pretrained

    def save_pretrained_row_only(
        _model: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        requested = kwargs.get("save_embedding_layers")
        if requested not in (None, False, "auto"):
            raise ValueError(
                "Semantic ID Token adapters must not save full embedding "
                "layers"
            )
        kwargs["save_embedding_layers"] = False
        return original_save_pretrained(*args, **kwargs)

    model.save_pretrained = MethodType(save_pretrained_row_only, model)
    model._semantic_row_only_saves = True


def sample_weighted_causal_lm_loss(
    logits: Any,
    labels: Any,
    sample_weight: Any,
) -> Any:
    """Apply a fixed sample weight while averaging over target tokens.

    The denominator is deliberately the unweighted count of valid target
    tokens. Dividing by a weight sum would cancel the intervention whenever a
    per-device batch contains one sample.
    """
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label batch/sequence shapes differ")
    if sample_weight.ndim != 1 or sample_weight.shape[0] != labels.shape[0]:
        raise ValueError("sample_weight must have shape [B]")
    if not bool(torch.isfinite(sample_weight).all()) or bool(
        (sample_weight <= 0).any()
    ):
        raise ValueError("sample_weight must be positive and finite")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_mask = shift_labels.ne(-100)
    valid_token_count = token_mask.sum()
    if int(valid_token_count.detach().cpu()) <= 0:
        raise ValueError("Batch contains no trainable target tokens")
    token_loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    weighted = token_loss * token_mask * sample_weight.to(
        device=token_loss.device,
        dtype=token_loss.dtype,
    ).unsqueeze(1)
    return weighted.sum() / valid_token_count.to(token_loss.dtype)


def per_example_mean_causal_lm_loss(logits: Any, labels: Any) -> Any:
    """Average completion-token CE within each sample, then across samples.

    This gives every training example the same total loss mass even when its
    assistant response contains a different number of tokenizer tokens.
    """
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label batch/sequence shapes differ")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_mask = shift_labels.ne(-100)
    token_counts = token_mask.sum(dim=1)
    if bool((token_counts <= 0).any()):
        raise ValueError("Every sample must contain a trainable target token")
    token_loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    per_example = (token_loss * token_mask).sum(dim=1) / token_counts.to(
        token_loss.dtype
    )
    return per_example.mean()


def taxonomy_restricted_causal_lm_loss(
    logits: Any,
    labels: Any,
    semantic_token_ids: list[int] | tuple[int, ...],
    sample_weight: Any | None = None,
) -> Any:
    """Classify the one Semantic-ID target only within its taxonomy.

    Semantic-ID evaluation ranks exactly the registered interest tokens with
    one next-token forward pass. This loss uses the same output space during
    training and deliberately ignores chat-template closing tokens.
    """
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label batch/sequence shapes differ")
    if len(semantic_token_ids) < 2:
        raise ValueError("At least two Semantic-ID token IDs are required")
    if len(set(int(value) for value in semantic_token_ids)) != len(
        semantic_token_ids
    ):
        raise ValueError("Semantic-ID token IDs must be unique")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_ids = torch.tensor(
        [int(value) for value in semantic_token_ids],
        dtype=torch.long,
        device=shift_labels.device,
    )
    if bool((token_ids < 0).any()) or bool(
        (token_ids >= shift_logits.shape[-1]).any()
    ):
        raise ValueError("Semantic-ID token ID is outside the vocabulary")

    matches = shift_labels.unsqueeze(-1).eq(token_ids.view(1, 1, -1))
    semantic_positions = matches.any(dim=-1)
    counts = semantic_positions.sum(dim=1)
    if bool((counts != 1).any()):
        raise ValueError(
            "Every sample must contain exactly one trainable Semantic-ID target"
        )

    batch_indices, time_indices = semantic_positions.nonzero(as_tuple=True)
    restricted_logits = shift_logits[batch_indices, time_indices].index_select(
        dim=-1,
        index=token_ids,
    )
    targets = matches[batch_indices, time_indices].to(torch.long).argmax(dim=-1)
    per_example = functional.cross_entropy(
        restricted_logits,
        targets,
        reduction="none",
    )
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != labels.shape[0]:
            raise ValueError("sample_weight must have shape [B]")
        if not bool(torch.isfinite(sample_weight).all()) or bool(
            (sample_weight <= 0).any()
        ):
            raise ValueError("sample_weight must be positive and finite")
        per_example = per_example * sample_weight.to(
            device=per_example.device,
            dtype=per_example.dtype,
        )[batch_indices]
    return per_example.mean()


def make_sample_weighted_trainer(base_trainer: type[Any]) -> type[Any]:
    """Create a Transformers-version-tolerant weighted Trainer subclass."""

    class SampleWeightedTrainer(base_trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            model_inputs = dict(inputs)
            sample_weight = model_inputs.pop("sample_weight", None)
            if sample_weight is None:
                return super().compute_loss(
                    model,
                    model_inputs,
                    return_outputs=return_outputs,
                )
            labels = model_inputs.get("labels")
            if labels is None:
                raise ValueError("Weighted SFT requires labels")
            outputs = model(**model_inputs)
            loss = sample_weighted_causal_lm_loss(
                outputs.logits,
                labels,
                sample_weight,
            )
            return (loss, outputs) if return_outputs else loss

    return SampleWeightedTrainer


def make_per_example_mean_trainer(base_trainer: type[Any]) -> type[Any]:
    """Create a Trainer whose completion loss is exactly sample-normalized."""

    class PerExampleMeanTrainer(base_trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Transformers otherwise treats num_items_in_batch as a global
            # token denominator across gradient-accumulation micro-batches.
            # Our custom loss is already normalized per example.
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("Per-example mean SFT requires labels")
            outputs = model(**inputs)
            loss = per_example_mean_causal_lm_loss(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return PerExampleMeanTrainer


def make_taxonomy_restricted_trainer(
    base_trainer: type[Any],
    semantic_token_ids: list[int] | tuple[int, ...],
) -> type[Any]:
    """Create a Trainer aligned to restricted Semantic-ID evaluation."""

    frozen_token_ids = tuple(int(value) for value in semantic_token_ids)

    class TaxonomyRestrictedTrainer(base_trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            model_inputs = dict(inputs)
            sample_weight = model_inputs.pop("sample_weight", None)
            labels = model_inputs.get("labels")
            if labels is None:
                raise ValueError("Taxonomy-restricted SFT requires labels")
            outputs = model(**model_inputs)
            loss = taxonomy_restricted_causal_lm_loss(
                outputs.logits,
                labels,
                frozen_token_ids,
                sample_weight,
            )
            return (loss, outputs) if return_outputs else loss

    return TaxonomyRestrictedTrainer


def make_mera_trainer(
    base_trainer: type[Any],
    regularizer: Any,
    *,
    loss_reduction: str,
) -> type[Any]:
    """Create a Trainer that adds MeRA relation losses to completion SFT."""

    class MeRATrainer(base_trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            model_inputs = dict(inputs)
            sample_weight = model_inputs.pop("sample_weight", None)
            labels = model_inputs.get("labels")
            if labels is None:
                raise ValueError("MeRA SFT requires labels")
            outputs = model(**model_inputs)
            if sample_weight is not None:
                base_loss = sample_weighted_causal_lm_loss(
                    outputs.logits,
                    labels,
                    sample_weight,
                )
            elif loss_reduction == "per_example_mean":
                base_loss = per_example_mean_causal_lm_loss(
                    outputs.logits,
                    labels,
                )
            else:
                base_loss = outputs.loss
                if base_loss is None:
                    raise ValueError("Model did not return a causal-LM loss")
            relation_losses = regularizer.compute(
                model,
                step=int(self.state.global_step),
            )
            loss = base_loss + relation_losses["weighted_loss"]
            return (loss, outputs) if return_outputs else loss

    return MeRATrainer


def train(
    config: dict[str, Any],
    *,
    trainer_base: type[Any] | None = None,
    trainer_class_transform: Any = None,
) -> dict[str, object]:
    import torch
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    # Keep workflow-local instrumentation explicit and compose it with the
    # cursor, migration and exposure wrappers below.
    if trainer_base is not None:
        if not issubclass(trainer_base, Trainer):
            raise TypeError("trainer_base must inherit the native Trainer")
        Trainer = trainer_base

    class RestoreConfiguredSaveStepsCallback(TrainerCallback):
        def on_train_begin(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            del kwargs
            _restore_configured_save_steps(args, state)
            return control

    class PersistentCheckpointCallback(TrainerCallback):
        def __init__(
            self,
            *,
            persistent_root: Path,
            identity: dict[str, Any],
            retention_limit: int,
        ) -> None:
            self.persistent_root = persistent_root
            self.identity = identity
            self.retention_limit = retention_limit

        def on_save(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            del kwargs

            def persist() -> Path:
                local_checkpoint = (
                    Path(args.output_dir)
                    / f"checkpoint-{int(state.global_step)}"
                )
                published = publish_checkpoint(
                    local_checkpoint,
                    self.persistent_root,
                    self.identity,
                )
                removed = prune_persistent_checkpoints(
                    self.persistent_root,
                    self.identity,
                    keep=self.retention_limit,
                )
                print(
                    "CHECKPOINT_PERSISTED",
                    int(state.global_step),
                    published,
                    flush=True,
                )
                for path in removed:
                    print("CHECKPOINT_PRUNED", path, flush=True)
                return published

            _run_rank_zero_checkpoint_operation(persist)
            return control

    seed = int(config.get("seed", 42))
    _set_seed(seed)
    model_name = str(config["model_name"])
    model_revision = config.get("model_revision")
    adapter_path = _resolve_adapter_path(config)
    train_file = Path(config["train_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(
        config.get("checkpoint_dir", output_dir / "checkpoints")
    )
    persistent_checkpoint_value = config.get(
        "persistent_checkpoint_dir"
    )
    persistent_checkpoint_root = (
        Path(str(persistent_checkpoint_value))
        if persistent_checkpoint_value is not None
        else None
    )
    max_seq_length = int(config.get("max_seq_length", 2048))
    enable_thinking = bool(config.get("enable_thinking", False))
    max_samples = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
    sample_weight_value = config.get("sample_weight_file")
    sample_weight_file = (
        Path(str(sample_weight_value))
        if sample_weight_value is not None
        else None
    )
    loss_reduction = str(config.get("loss_reduction", "token_mean"))
    if loss_reduction not in {"token_mean", "per_example_mean"}:
        raise ValueError(
            "loss_reduction must be token_mean or per_example_mean"
        )
    if sample_weight_file is not None and loss_reduction != "token_mean":
        raise ValueError(
            "sample_weight_file cannot be combined with per_example_mean"
        )
    semantic_id_loss_scope = str(
        config.get("semantic_id_loss_scope", "full_vocabulary")
    )
    if semantic_id_loss_scope not in {"full_vocabulary", "taxonomy"}:
        raise ValueError(
            "semantic_id_loss_scope must be full_vocabulary or taxonomy"
        )
    if semantic_id_loss_scope == "taxonomy" and loss_reduction != "token_mean":
        raise ValueError(
            "taxonomy Semantic-ID loss cannot be combined with "
            "per_example_mean"
        )
    checkpoint_identity = _training_checkpoint_identity(
        config,
        train_file=train_file,
        sample_weight_file=sample_weight_file,
        adapter_path=adapter_path,
    )
    is_primary_process = int(os.environ.get("RANK", "0")) == 0
    migration = config.get("execution_migration")
    migration_runtime = None
    initial_checkpoint_identity = checkpoint_identity
    if migration is not None:
        from new_interest.training.execution_migration import (
            restore_stage_evidence, validate_execution_transition,
        )
        source_checkpoint = Path(migration["source_checkpoint"])
        source_marker = json.loads((source_checkpoint / "checkpoint_complete.json").read_text())
        initial_checkpoint_identity = source_marker["identity"]
        validate_execution_transition(initial_checkpoint_identity, checkpoint_identity)
        cursor_contract = json.loads(Path(migration["cursor_contract_file"]).read_text())
        if is_primary_process:
            restore_stage_evidence(Path(migration["source_stage_artifacts"]), output_dir, initial_checkpoint_identity)
        migration_runtime = {"output_dir": output_dir, "target_identity": checkpoint_identity,
                             "stop_after_global_step": config.get("stop_after_global_step"),
                             "long_sample_files": config.get("long_sample_probe_files", []),
                             "numerical_probe_file": config.get("numerical_probe_file"),
                             "mapping_file": config.get("semantic_id_token_mapping_file"),
                             "cursor_contract": cursor_contract}
    stage_cursor_value = config.get("old_new_cursor_contract_file")
    stage_cursor_contract = json.loads(Path(stage_cursor_value).read_text()) if stage_cursor_value else None
    if stage_cursor_contract is not None:
        if (stage_cursor_contract["dataset_sha256"] != checkpoint_identity["train_file_sha256"]
                or stage_cursor_contract["epochs"] != float(config.get("epochs", 3))
                or stage_cursor_contract["seed"] != seed):
            raise ValueError("Old/new cursor does not match the configured stage")
        if migration_runtime is not None and stage_cursor_contract != cursor_contract:
            raise ValueError("Execution migration and exposure cursors differ")
        if trainer_class_transform is None:
            from new_interest.training.old_new_execution import wrap_trainer
            trainer_class_transform = lambda base: wrap_trainer(base, stage_cursor_contract)
    semantic_mapping_value = config.get("semantic_id_token_mapping_file")
    semantic_mapping_file = (
        Path(str(semantic_mapping_value))
        if semantic_mapping_value is not None
        else None
    )
    mera_relations_value = config.get("mera_relations_file")
    mera_relations_file = (
        Path(str(mera_relations_value))
        if mera_relations_value is not None
        else None
    )
    if mera_relations_file is not None and semantic_mapping_file is None:
        raise ValueError(
            "MeRA regularization requires semantic_id_token_mapping_file"
        )
    if semantic_id_loss_scope == "taxonomy" and semantic_mapping_file is None:
        raise ValueError(
            "taxonomy Semantic-ID loss requires "
            "semantic_id_token_mapping_file"
        )
    if semantic_id_loss_scope == "taxonomy" and mera_relations_file is not None:
        raise ValueError(
            "taxonomy Semantic-ID loss and MeRA must be evaluated separately"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        token=config.get("hf_token"),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    semantic_spec = None
    semantic_initialization_mode = str(
        config.get(
            "semantic_id_initialization_mode",
            "mean_norm_matched",
        )
    )
    if semantic_mapping_file is not None:
        semantic_spec = register_semantic_interest_tokens(
            tokenizer,
            semantic_mapping_file,
            expected_count=int(config.get("semantic_id_token_count", 108)),
        )

    token_audit = audit_token_lengths(
        train_file,
        tokenizer,
        max_seq_length=max_seq_length,
        enable_thinking=enable_thinking,
        output_path=(
            output_dir / "token_audit.json"
            if is_primary_process
            else None
        ),
    )
    if token_audit["over_max_seq_length"]:
        raise ValueError(
            f"{token_audit['over_max_seq_length']} samples exceed "
            f"max_seq_length={max_seq_length}"
        )

    dataset = ParquetSFTDataset(
        train_file,
        tokenizer,
        max_seq_length=max_seq_length,
        enable_thinking=enable_thinking,
        max_samples=max_samples,
        sample_weight_file=sample_weight_file,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        token=config.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    semantic_initialization_audit = None
    semantic_initial_rows = None
    semantic_trainable_indices = None
    if semantic_spec is not None:
        model.resize_token_embeddings(semantic_spec.resized_vocabulary_size)
        (
            semantic_initialization_audit,
            semantic_initial_rows,
        ) = initialize_semantic_interest_token_rows(
            model,
            semantic_spec,
            initialization_mode=semantic_initialization_mode,
        )
        semantic_trainable_indices = trainable_token_indices(
            model,
            semantic_spec,
            output_only=(semantic_id_loss_scope == "taxonomy"),
        )
        if is_primary_process:
            write_json(
                output_dir / "semantic_token_initialization.json",
                semantic_initialization_audit,
            )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            revision=model_revision,
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=config.get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
            bias="none",
            trainable_token_indices=semantic_trainable_indices,
        )
        model = get_peft_model(model, lora)
    if semantic_spec is not None:
        configure_row_only_adapter_saves(model)
    mera_regularizer = None
    mera_regularization_audit = None
    if mera_relations_file is not None:
        mera_regularizer = build_mera_regularizer(config, semantic_spec)
        mera_regularization_audit = mera_regularizer.audit()
        if is_primary_process:
            write_json(
                output_dir / "mera_regularization_audit.json",
                mera_regularization_audit,
            )
    model.enable_input_require_grads()
    if is_primary_process:
        model.print_trainable_parameters()
    forbidden_substrings = tuple(
        str(value)
        for value in config.get(
            "forbidden_trainable_module_substrings",
            (),
        )
    )
    model_audit = audit_trainable_parameters(
        model,
        forbidden_substrings=forbidden_substrings,
    )
    model_audit["tokenizer_class"] = type(tokenizer).__name__
    model_audit["tokenizer_name_or_path"] = str(
        getattr(tokenizer, "name_or_path", model_name)
    )
    model_audit["semantic_id_tokens"] = (
        {
            "enabled": True,
            "mapping_file": str(semantic_mapping_file),
            "mapping_sha256": semantic_spec.mapping_sha256,
            "count": len(semantic_spec.entries),
            "token_ids": semantic_spec.token_ids,
            "trainable_token_indices": semantic_trainable_indices,
            "initialization_version": semantic_initialization_audit[
                "initialization_version"
            ],
            "tie_word_embeddings": semantic_initialization_audit[
                "tie_word_embeddings"
            ],
            "save_embedding_layers": False,
            "loss_scope": semantic_id_loss_scope,
        }
        if semantic_spec is not None
        else {"enabled": False}
    )
    model_audit["mera_regularization"] = (
        mera_regularization_audit
        if mera_regularization_audit is not None
        else {"enabled": False}
    )
    if is_primary_process:
        (output_dir / "model_audit.json").write_text(
            json.dumps(model_audit, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    forbidden_parameters = model_audit[
        "forbidden_trainable_parameters"
    ]
    if forbidden_parameters:
        raise ValueError(
            "Forbidden trainable parameters detected: "
            + ", ".join(str(value) for value in forbidden_parameters)
        )

    initial_adapter_state = None
    if bool(config.get("record_initial_adapter_state", False)) and is_primary_process:
        if migration_runtime is not None:
            # The temporary model initialization is replaced by the trained
            # checkpoint before any forward. Keep the historical first stage
            # evidence; the migration callback verifies actual restored tensors.
            original_initial = json.loads((output_dir / "initial_adapter_state.json").read_text())
            initial_adapter_state = {key: value for key, value in original_initial.items()
                                     if key not in {"tensors", "checkpoint_identity"}}
        else:
            initial_adapter_state = record_initial_adapter_state(
                model, output_dir, initial_checkpoint_identity, input_adapter=adapter_path,
            )
    expected_initial_state = config.get("expected_initial_adapter_state_sha256")
    if expected_initial_state is not None and is_primary_process:
        if initial_adapter_state is None or initial_adapter_state["state_sha256"] != expected_initial_state:
            raise ValueError("Initial adapter state differs from the accepted smoke")

    wandb_mode = str(config.get("wandb_mode", "online"))
    report_to = [] if wandb_mode == "disabled" else ["wandb"]
    if wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = wandb_mode
        os.environ["WANDB_PROJECT"] = str(
            config.get(
                "wandb_project",
                "cross-interest-generative-sft",
            )
        )

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(config.get("epochs", 3)),
        per_device_train_batch_size=int(
            config.get("per_device_train_batch_size", 4)
        ),
        gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        bf16=True,
        tf32=True,
        gradient_checkpointing=bool(config.get("gradient_checkpointing_enabled", True)),
        gradient_checkpointing_kwargs=(
            _gradient_checkpointing_kwargs(config)
        ),
        logging_steps=int(config.get("logging_steps", 5)),
        save_strategy=str(config.get("save_strategy", "epoch")),
        save_steps=int(config.get("save_steps", 500)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        optim=str(config.get("optim", "adamw_torch_fused")),
        report_to=report_to,
        run_name=str(config["run_name"]),
        seed=seed,
        data_seed=seed,
        ignore_data_skip=bool(config.get("ignore_data_skip", False)),
        remove_unused_columns=False,
        dataloader_num_workers=int(
            config.get("dataloader_num_workers", 2)
        ),
    )
    if semantic_id_loss_scope == "taxonomy":
        if semantic_spec is None:
            raise ValueError("Taxonomy loss requires registered Semantic IDs")
        trainer_class = make_taxonomy_restricted_trainer(
            Trainer,
            semantic_spec.token_ids,
        )
    elif mera_regularizer is not None:
        trainer_class = make_mera_trainer(
            Trainer,
            mera_regularizer,
            loss_reduction=loss_reduction,
        )
    elif sample_weight_file is not None:
        trainer_class = make_sample_weighted_trainer(Trainer)
    elif loss_reduction == "per_example_mean":
        trainer_class = make_per_example_mean_trainer(Trainer)
    else:
        trainer_class = Trainer
    if migration_runtime is not None:
        from new_interest.training.execution_migration import make_execution_migration_trainer
        trainer_class = make_execution_migration_trainer(trainer_class, migration_runtime)
        if trainer_class_transform is None:
            from new_interest.training.global_sample_cursor import make_hf_cursor_trainer
            trainer_class = make_hf_cursor_trainer(trainer_class, migration_runtime["cursor_contract"])
    if trainer_class_transform is not None:
        trainer_class = trainer_class_transform(trainer_class)
    exposure = None
    training_dataset = dataset
    collator = CompletionOnlyCollator(tokenizer)
    if bool(config.get("audit_training_exposure", False)):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 and stage_cursor_contract is None:
            raise ValueError("Distributed old/new exposure requires its validated full-epoch cursor")
        exposure = TrainingExposure.from_parquet(
            train_file,
            train_sha256=checkpoint_identity["train_file_sha256"],
            expected_epochs=float(config.get("epochs", 3)),
            max_samples=max_samples,
            default_task=str(config.get("exposure_default_task", "new_interest")),
        )
        if len(exposure.rows) != len(dataset):
            raise ValueError("Exposure identities differ from the actual training dataset")
        training_dataset = ExposureIndexedDataset(dataset)
        collator = ExposureCollator(collator)
        trainer_class = make_exposure_trainer(trainer_class, exposure)
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=training_dataset,
        data_collator=collator,
    )
    trainer.add_callback(RestoreConfiguredSaveStepsCallback())
    if initial_adapter_state is not None and migration_runtime is None:
        from transformers import TrainerCallback

        trainer.add_callback(make_training_start_audit_callback(
            TrainerCallback, output_dir,
            str(initial_adapter_state["state_sha256"]), checkpoint_identity,
        ))
    if exposure is not None:
        from transformers import TrainerCallback

        # This must precede persistent checkpoint publication in callback order.
        trainer.add_callback(make_exposure_callback(TrainerCallback, exposure))
    if stage_cursor_contract is not None:
        if exposure is None:
            raise ValueError("An old/new stage cursor requires observed exposure accounting")
        from new_interest.training.old_new_execution import make_cursor_callback
        trainer.add_callback(make_cursor_callback(TrainerCallback, trainer, exposure, stage_cursor_contract, output_dir,
            stop_after_global_step=config.get("stop_after_global_step")))
    if migration_runtime is not None:
        from new_interest.training.execution_migration import make_execution_migration_callback
        migration_runtime["trainer"] = trainer
        migration_runtime["tokenizer"] = tokenizer
        trainer.add_callback(make_execution_migration_callback(TrainerCallback, migration_runtime))
    if persistent_checkpoint_root is not None:
        persistent_checkpoint_limit = int(
            config.get(
                "persistent_checkpoint_limit",
                config.get("save_total_limit", 3),
            )
        )
        if persistent_checkpoint_limit <= 0:
            raise ValueError(
                "persistent_checkpoint_limit must be positive"
            )
        trainer.add_callback(
            PersistentCheckpointCallback(
                persistent_root=persistent_checkpoint_root,
                identity=checkpoint_identity,
                retention_limit=persistent_checkpoint_limit,
            )
        )
    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint == "auto":
        if persistent_checkpoint_root is None:
            raise ValueError(
                "resume_from_checkpoint=auto requires "
                "persistent_checkpoint_dir"
            )
        resume_from_checkpoint = _prepare_automatic_resume(
            persistent_checkpoint_root=persistent_checkpoint_root,
            local_checkpoint_root=checkpoint_dir,
            identity=checkpoint_identity,
        )
    if migration_runtime is not None:
        from new_interest.training.execution_migration import (
            load_migration_receipt, prepare_execution_migration,
        )
        if resume_from_checkpoint is None:
            def prepare_migration() -> Path:
                migration_root = checkpoint_dir.parent / (checkpoint_dir.name + "-execution-migration")
                if migration_root.exists():
                    candidate = migration_root / source_checkpoint.name
                    load_migration_receipt(candidate, checkpoint_identity)
                    return candidate
                prepared = prepare_execution_migration(
                    source_checkpoint, migration_root, initial_checkpoint_identity, checkpoint_identity,
                    Path(migration["source_stage_artifacts"]), cursor_contract,
                )
                return prepared["checkpoint_dir"]
            resume_from_checkpoint = _run_rank_zero_checkpoint_operation(prepare_migration)
        if resume_from_checkpoint is None:
            raise ValueError("Execution migration cannot silently fall back to a fresh optimizer")
        migration_runtime["resume_checkpoint"] = str(resume_from_checkpoint)
        migration_receipt = load_migration_receipt(Path(resume_from_checkpoint), checkpoint_identity)
        migration_runtime["lineage_id"] = migration_receipt["lineage_id"]
        def make_attempt_directory() -> Path:
            import tempfile
            root = output_dir / "execution_attempts"
            root.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        migration_runtime["attempt_dir"] = _run_rank_zero_checkpoint_operation(make_attempt_directory)
    if exposure is not None and resume_from_checkpoint:
        exposure.restore_checkpoint(Path(resume_from_checkpoint))
    result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    exposure_summary = None
    if exposure is not None:
        step = int(trainer.state.global_step)
        if config.get("stop_after_global_step") is not None:
            if stage_cursor_contract is None or step != int(config["stop_after_global_step"]):
                raise ValueError("Bounded migration probe did not stop at its exact cursor")
            from new_interest.training.old_new_execution import expected_counts
            counts, _ = expected_counts(len(exposure.rows), step,
                epochs=stage_cursor_contract["epochs"], seed=seed,
                global_batch=stage_cursor_contract["global_batch_size"])
            exposure_summary = exposure.finish_cursor(output_dir, optimizer_steps=step, expected_counts=counts)
        else:
            exposure_summary = exposure.finish(output_dir, optimizer_steps=step)
    semantic_token_drift = None
    semantic_reload_reference = None
    text_reload_reference = None
    if semantic_spec is not None:
        reload_reference_model = _unwrap_model_for_reload_reference(trainer)
        semantic_token_drift = audit_interest_token_drift(
            reload_reference_model,
            semantic_spec,
            semantic_initial_rows,
        )
        if trainer.is_world_process_zero():
            write_json(
                output_dir / "semantic_token_drift.json",
                semantic_token_drift,
            )
            semantic_reload_reference = capture_reload_reference(
                reload_reference_model,
                tokenizer,
                train_file,
                semantic_spec,
                enable_thinking=enable_thinking,
                sample_count=int(
                    config.get("semantic_reload_reference_samples", 4)
                ),
                initialization_mode=semantic_initialization_mode,
            )
            write_json(
                output_dir / "semantic_reload_reference.json",
                semantic_reload_reference,
            )
    elif trainer.is_world_process_zero():
        reload_reference_model = _unwrap_model_for_reload_reference(trainer)
        text_reload_reference = capture_text_reload_reference(
            reload_reference_model,
            tokenizer,
            train_file,
            output_dir,
            model_name=model_name,
            model_revision=model_revision,
            enable_thinking=enable_thinking,
            sample_count=int(config.get("text_reload_reference_samples", 4)),
        )
    loss_history = [
        {
            "step": int(entry["step"]),
            "loss": float(entry["loss"]),
        }
        for entry in trainer.state.log_history
        if "step" in entry and "loss" in entry
    ]
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        if semantic_mapping_file is not None:
            shutil.copy2(
                semantic_mapping_file,
                final_dir / "interest_token_mapping.parquet",
            )
        if mera_relations_file is not None:
            shutil.copy2(
                mera_relations_file,
                final_dir / "mera_relations.json",
            )
        trainer.save_state()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_batch_size = int(config.get("per_device_train_batch_size", 4))
    gradient_accumulation_steps = int(
        config.get("gradient_accumulation_steps", 8)
    )
    effective_global_batch_size = (
        world_size
        * per_device_batch_size
        * gradient_accumulation_steps
    )
    configured_epochs = float(config.get("epochs", 3))
    expected_optimizer_steps = math.ceil(
        math.ceil(len(dataset) / effective_global_batch_size)
        * configured_epochs
    )
    summary = {
        "model_name": model_name,
        "model_revision": model_revision,
        "initial_adapter_state": initial_adapter_state,
        "input_adapter": (
            str(adapter_path.resolve())
            if adapter_path is not None
            else None
        ),
        "train_file": str(train_file.resolve()),
        "train_file_sha256": _sha256_file(train_file),
        "sample_weight_file": (
            str(sample_weight_file.resolve())
            if sample_weight_file is not None
            else None
        ),
        "sample_weight_file_sha256": (
            _sha256_file(sample_weight_file)
            if sample_weight_file is not None
            else None
        ),
        "loss_reduction": loss_reduction,
        "semantic_id_loss_scope": semantic_id_loss_scope,
        "train_samples": len(dataset),
        "observed_training_exposure": exposure_summary,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "max_seq_length": max_seq_length,
        "configured_epochs": configured_epochs,
        "world_size": world_size,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_size": effective_global_batch_size,
        "expected_optimizer_steps": expected_optimizer_steps,
        "actual_optimizer_steps": int(trainer.state.global_step),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "persistent_checkpoint_dir": (
            str(persistent_checkpoint_root.resolve())
            if persistent_checkpoint_root is not None
            else None
        ),
        "resumed_from_checkpoint": resume_from_checkpoint,
        "gradient_checkpointing_kwargs": (
            _gradient_checkpointing_kwargs(config)
        ),
        "token_audit": token_audit,
        "model_audit": model_audit,
        "semantic_token_initialization": semantic_initialization_audit,
        "semantic_token_drift": semantic_token_drift,
        "semantic_reload_reference": semantic_reload_reference,
        "text_reload_reference": text_reload_reference,
        "mera_regularization_audit": mera_regularization_audit,
        "mera_regularization_training": (
            mera_regularizer.snapshot()
            if mera_regularizer is not None
            else None
        ),
        "loss_history": loss_history,
        "train_metrics": result.metrics,
        "final_adapter": str(final_dir.resolve()),
    }
    if stage_cursor_contract is not None:
        summary["optimizer_updates_in_this_process"] = int(trainer.state.global_step) - trainer._cursor_start_step
        summary["reused_completed_endpoint_cursor"] = trainer._cursor_reused_endpoint
    if trainer.is_world_process_zero():
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    if trainer.is_world_process_zero() and wandb.run is not None:
        wandb.config.update(
            {
                "token_audit": token_audit,
                "resolved_config": config,
            },
            allow_val_change=True,
        )
        wandb.run.summary.update(result.metrics)
        wandb.finish()
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoRA SFT.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-initial-adapter-state-sha256")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--sample-weight-file", type=Path)
    parser.add_argument(
        "--loss-reduction",
        choices=("token_mean", "per_example_mean"),
    )
    parser.add_argument(
        "--semantic-id-loss-scope",
        choices=("full_vocabulary", "taxonomy"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--persistent-checkpoint-dir", type=Path)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--semantic-id-token-mapping-file", type=Path)
    parser.add_argument(
        "--semantic-id-initialization-mode",
        choices=("mean_norm_matched", "mean"),
    )
    parser.add_argument("--mera-relations-file", type=Path)
    parser.add_argument("--mera-structure-weight", type=float)
    parser.add_argument("--mera-collaborative-weight", type=float)
    parser.add_argument("--mera-negative-samples", type=int)
    parser.add_argument("--mera-cooccurrence-threshold", type=float)
    parser.add_argument(
        "--mera-min-collaborative-anchor-fraction",
        type=float,
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    if args.expected_initial_adapter_state_sha256 is not None:
        config["expected_initial_adapter_state_sha256"] = args.expected_initial_adapter_state_sha256
    for key in (
        "train_file",
        "adapter_path",
        "sample_weight_file",
        "loss_reduction",
        "semantic_id_loss_scope",
        "output_dir",
        "run_name",
        "max_samples",
        "epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "resume_from_checkpoint",
        "checkpoint_dir",
        "persistent_checkpoint_dir",
        "save_steps",
        "semantic_id_token_mapping_file",
        "semantic_id_initialization_mode",
        "mera_relations_file",
        "mera_structure_weight",
        "mera_collaborative_weight",
        "mera_negative_samples",
        "mera_cooccurrence_threshold",
        "mera_min_collaborative_anchor_fraction",
        "wandb_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    train(config)


if __name__ == "__main__":
    main()
