from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from types import MethodType
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from new_interest.experiments.mera_original_core import (
    MeRAOriginalRegularizer,
    files_manifest,
    inference_text,
    jsonable_versions,
    load_categories,
    load_json,
    load_records,
    load_resolved_config,
    model_category_rows,
    peft_state_manifest,
    resolve_trainable_weight,
    sha256_file,
    training_prompt,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def seed_everything(seed: int) -> None:
    import torch
    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def capture_rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def remove_accelerate_fp32_forward_wrapper(model: Any) -> Any:
    """Mirror Accelerate 0.29.3's wrapper removal without optional imports."""
    forward = model.forward
    original_forward = model.__dict__.pop("_original_forward", None)
    if original_forward is not None:
        while hasattr(forward, "__wrapped__"):
            forward = forward.__wrapped__
            if forward == original_forward:
                break
        model.forward = MethodType(forward, model)
    if hasattr(model, "_original_forward"):
        raise RuntimeError("Accelerate forward wrapper remains after removal")
    return model


def prepare_model_and_tokenizer(config: Any) -> tuple[Any, Any, tuple[str, ...], tuple[int, ...], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.model
    categories = load_categories(config.data_path("category_map"))
    category_tokens = tuple(f'"{value}"' for value in categories)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        cache_dir=os.environ.get("HF_HOME"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    old_tokenizer_size = len(tokenizer)
    tokenizer.pad_token_id = int(model_config["pad_token_id"])
    tokenizer.padding_side = "left"
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(category_tokens)}
    )
    if added != len(categories):
        raise RuntimeError(
            f"Expected to add {len(categories)} category tokens, added {added}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        cache_dir=os.environ.get("HF_HOME"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        load_in_8bit=bool(model_config["load_in_8bit"]),
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    original_model_vocab_size = int(model.config.vocab_size)
    model.resize_token_embeddings(len(tokenizer))
    resized_input = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if resized_input.weight.data_ptr() != output_module.weight.data_ptr():
        raise RuntimeError("Expected Qwen2 input/output embeddings to be tied after resize")

    category_ids = tuple(tokenizer.convert_tokens_to_ids(value) for value in category_tokens)
    if len(set(category_ids)) != len(category_ids):
        raise RuntimeError("Category token ids are not unique")
    for token, token_id in zip(category_tokens, category_ids):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise RuntimeError(f"Category is not a single token: {token} -> {encoded}")

    old_weight = resized_input.weight
    replacement = torch.nn.Embedding(
        old_weight.shape[0],
        old_weight.shape[1],
        padding_idx=resized_input.padding_idx,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(old_weight)
        for category, token_id in zip(categories, category_ids):
            source_ids = tokenizer.encode(category, add_special_tokens=False)
            if not source_ids:
                raise RuntimeError(f"Category has empty base tokenization: {category}")
            source = torch.tensor(source_ids, dtype=torch.long, device=old_weight.device)
            replacement.weight[token_id].copy_(
                old_weight.index_select(0, source).mean(dim=0)
            )
    model.set_input_embeddings(replacement)
    if model.get_input_embeddings().weight.data_ptr() == output_module.weight.data_ptr():
        raise RuntimeError("Input embedding and LM head did not become untied")
    model.config.use_cache = False
    audit = {
        "audit_version": "mera-original-tokenizer-model-init-v1",
        "model_name": model_config["name"],
        "model_revision": model_config["revision"],
        "original_model_vocab_size": original_model_vocab_size,
        "old_tokenizer_size": old_tokenizer_size,
        "new_tokenizer_size": len(tokenizer),
        "category_tokens": len(category_tokens),
        "category_token_id_min": min(category_ids),
        "category_token_id_max": max(category_ids),
        "input_output_tied_after_author_equivalent_replacement": False,
        "input_category_rows_sha256": tensor_sha256(
            model.get_input_embeddings().weight[list(category_ids)]
        ),
        "output_category_rows_sha256": tensor_sha256(
            model.get_output_embeddings().weight[list(category_ids)]
        ),
        "input_initialization": "mean_of_unquoted_category_base_tokens",
        "output_initialization": "transformers_resize_initialization",
    }
    return model, tokenizer, categories, category_ids, audit


def tokenize_records(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoff_len: int,
    split: str,
) -> tuple[Any, dict[str, Any]]:
    from datasets import Dataset

    lengths: list[int] = []
    tokenized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        full_prompt = training_prompt(row, include_output=True)
        user_prompt = training_prompt(row, include_output=False)
        untruncated = tokenizer(full_prompt, truncation=False, padding=False)
        length = len(untruncated["input_ids"])
        lengths.append(length)
        if length > cutoff_len:
            raise RuntimeError(
                f"{split} row {row_index} exceeds cutoff {cutoff_len}: {length}"
            )
        encoded = tokenizer(
            full_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )
        if encoded["input_ids"][-1] != tokenizer.eos_token_id:
            encoded["input_ids"].append(tokenizer.eos_token_id)
            encoded["attention_mask"].append(1)
        prompt_encoded = tokenizer(
            user_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )
        prompt_len = len(prompt_encoded["input_ids"])
        encoded["labels"] = [-100] * prompt_len + encoded["input_ids"][prompt_len:]
        tokenized.append(encoded)
    return Dataset.from_list(tokenized), {
        "split": split,
        "rows": len(rows),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "over_cutoff": 0,
    }


def _weights_and_grads(model: Any, token_ids: Sequence[int]) -> tuple[Any, Any, Any, Any]:
    import torch

    input_weight = resolve_trainable_weight(model.get_input_embeddings())
    output_weight = resolve_trainable_weight(model.get_output_embeddings())
    input_index = torch.tensor(token_ids, dtype=torch.long, device=input_weight.device)
    output_index = torch.tensor(token_ids, dtype=torch.long, device=output_weight.device)
    input_grad = input_weight.grad.index_select(0, input_index) if input_weight.grad is not None else None
    output_grad = output_weight.grad.index_select(0, output_index) if output_weight.grad is not None else None
    return input_weight, output_weight, input_grad, output_grad


def _nonzero_rows(value: Any) -> int:
    import torch

    if value is None:
        return 0
    return int((torch.linalg.vector_norm(value.float(), dim=1) > 0).sum().item())


def run_gradient_audit(model: Any, regularizer: MeRAOriginalRegularizer, token_ids: Sequence[int]) -> dict[str, Any]:
    import torch

    saved_rng = capture_rng_state()
    model.train()
    model.zero_grad(set_to_none=True)
    token_ids = tuple(token_ids)
    for start in range(0, len(token_ids), 36):
        anchors = token_ids[start : start + 36]
        targets = tuple(token_ids[(token_ids.index(value) + 1) % len(token_ids)] for value in anchors)
        input_ids = torch.tensor(
            [[anchor, target] for anchor, target in zip(anchors, targets)],
            dtype=torch.long,
            device=model.device,
        )
        labels = torch.tensor(
            [[-100, target] for target in targets],
            dtype=torch.long,
            device=model.device,
        )
        attention = torch.ones_like(input_ids)
        loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CE gradient-audit loss")
        loss.backward()
    _, _, input_grad, output_grad = _weights_and_grads(model, token_ids)
    ce_input_nonzero = _nonzero_rows(input_grad)
    ce_output_nonzero = _nonzero_rows(output_grad)
    if ce_input_nonzero != len(token_ids) or ce_output_nonzero != len(token_ids):
        raise RuntimeError("Not every category row receives CE audit gradient")

    model.zero_grad(set_to_none=True)
    losses = regularizer.compute(model)
    losses["weighted_loss"].backward()
    _, _, input_grad, output_grad = _weights_and_grads(model, token_ids)
    regularizer_input_nonzero = _nonzero_rows(input_grad)
    regularizer_output_nonzero = _nonzero_rows(output_grad)
    if not torch.isfinite(losses["weighted_loss"]):
        raise RuntimeError("Non-finite MeRA gradient-audit loss")
    if regularizer_input_nonzero == 0 or regularizer_output_nonzero == 0:
        raise RuntimeError("MeRA regularizer has no category-row gradients")
    payload = {
        "audit_version": "mera-original-gradient-audit-v1",
        "category_rows": len(token_ids),
        "ce_input_nonzero_rows": ce_input_nonzero,
        "ce_output_nonzero_rows": ce_output_nonzero,
        "regularizer_input_nonzero_rows": regularizer_input_nonzero,
        "regularizer_output_nonzero_rows": regularizer_output_nonzero,
        "structure_loss": float(losses["structure_loss"].detach().cpu()),
        "collaborative_loss": float(losses["collaborative_loss"].detach().cpu()),
        "weighted_loss": float(losses["weighted_loss"].detach().cpu()),
    }
    model.zero_grad(set_to_none=True)
    restore_rng_state(saved_rng)
    return payload


def move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def probe_loss(model: Any, collator: Any, dataset: Any) -> float:
    import torch

    saved_rng = capture_rng_state()
    model.eval()
    batch = collator([dataset[index] for index in range(min(4, len(dataset)))])
    batch = move_batch(batch, model.device)
    with torch.no_grad():
        loss = model(**batch).loss
    restore_rng_state(saved_rng)
    return float(loss.detach().cpu())


def candidate_logits(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], token_ids: Sequence[int], cutoff_len: int) -> list[list[float]]:
    import torch

    model.eval()
    outputs: list[list[float]] = []
    for row in rows:
        batch = tokenizer(
            inference_text(row),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cutoff_len,
        )
        batch = move_batch(batch, model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1]
            index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
            selected = logits.index_select(1, index).float().cpu()[0]
        outputs.append(selected.tolist())
    return outputs


def _copy_content(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _copy_content(path, destination / path.relative_to(source))


def _validate_checkpoint(directory: Path, identity: Mapping[str, Any]) -> bool:
    commit = directory / "checkpoint_commit.json"
    manifest_file = directory / "checkpoint_manifest.json"
    if not commit.is_file() or not manifest_file.is_file():
        return False
    commit_payload = load_json(commit)
    if commit_payload["identity"] != identity:
        return False
    if sha256_file(manifest_file) != commit_payload["manifest_sha256"]:
        return False
    manifest = load_json(manifest_file)
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def publish_checkpoint(source: Path, persistent_root: Path, identity: Mapping[str, Any], best_checkpoint: str | None) -> None:
    step = int(source.name.split("-")[-1])
    destination = persistent_root / source.name
    if destination.exists() and _validate_checkpoint(destination, identity):
        print("MERA_ORIGINAL_CHECKPOINT_REUSE_OK", destination)
        return
    staging = persistent_root / f".{source.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _copy_tree(source, staging)
    write_json(staging / "checkpoint_identity.json", identity)
    manifest = {
        "manifest_version": "mera-original-checkpoint-manifest-v1",
        "files": files_manifest(
            staging,
            exclude_names=("checkpoint_manifest.json", "checkpoint_commit.json"),
        ),
    }
    write_json(staging / "checkpoint_manifest.json", manifest)
    write_json(
        staging / "checkpoint_commit.json",
        {
            "step": step,
            "identity": identity,
            "manifest_sha256": sha256_file(staging / "checkpoint_manifest.json"),
        },
    )
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)
    if not _validate_checkpoint(destination, identity):
        raise RuntimeError(f"Published checkpoint did not validate: {destination}")

    committed = sorted(
        (
            value
            for value in persistent_root.glob("checkpoint-*")
            if value.is_dir() and _validate_checkpoint(value, identity)
        ),
        key=lambda value: int(value.name.split("-")[-1]),
    )
    best_name = Path(best_checkpoint).name if best_checkpoint else None
    keep = {destination.name}
    if best_name:
        keep.add(best_name)
    for value in reversed(committed):
        if len(keep) >= 3:
            break
        keep.add(value.name)
    for value in committed:
        if value.name not in keep:
            shutil.rmtree(value)
    print("MERA_ORIGINAL_CHECKPOINT_PUBLISH_OK", destination)


def materialize_auto_resume(local_output: Path, persistent_root: Path, identity: Mapping[str, Any]) -> str | None:
    candidates = sorted(
        (
            value
            for value in persistent_root.glob("checkpoint-*")
            if value.is_dir() and _validate_checkpoint(value, identity)
        ),
        key=lambda value: int(value.name.split("-")[-1]),
    )
    if not candidates:
        return None
    source = candidates[-1]
    destination = local_output / source.name
    if destination.exists():
        shutil.rmtree(destination)
    _copy_tree(source, destination)
    print("MERA_ORIGINAL_RESUME_MATERIALIZED", source, destination)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("arts", "grocery"))
    parser.add_argument(
        "--contract", required=True, choices=("code-faithful", "paper-spec")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--resume-from-checkpoint", default="none")
    parser.add_argument("--persistent-checkpoint-dir")
    args = parser.parse_args()

    import torch
    import transformers.utils.import_utils as transformers_import_utils

    # The Feeds base image exposes an Apex build compiled for its newer system
    # PyTorch. The frozen MeRA stack uses native AMP with PyTorch 2.3 and never
    # requests Apex, but Transformers 4.37 imports it solely because it is
    # discoverable. Disable that unused optional backend before importing Trainer.
    if hasattr(transformers_import_utils, "_apex_available"):
        transformers_import_utils._apex_available = False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("MeRA original training requires exactly one visible GPU")
    config = load_resolved_config(args.config, args.dataset, args.contract)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = config.identity()
    seed = int(config.training["seed"])
    seed_everything(seed)

    model, tokenizer, categories, token_ids, tokenizer_audit = prepare_model_and_tokenizer(config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=int(config.training["lora_r"]),
        lora_alpha=int(config.training["lora_alpha"]),
        lora_dropout=float(config.training["lora_dropout"]),
        target_modules=list(config.training["lora_target_modules"]),
        modules_to_save=list(config.training["modules_to_save"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not any("embed_tokens" in name for name in trainable):
        raise RuntimeError("Input embeddings are not trainable")
    if not any("lm_head" in name for name in trainable):
        raise RuntimeError("LM head is not trainable")

    tree = load_json(config.data_path("category_tree"))
    matrix = np.load(config.data_path("cooccurrence_matrix"))
    regularizer = MeRAOriginalRegularizer(
        token_ids=token_ids,
        categories=categories,
        category_tree=tree,
        cooccurrence_matrix=matrix,
        contract=config.regularizer,
    )
    gradient_audit = run_gradient_audit(model, regularizer, token_ids)

    train_rows = load_records(config.data_path("train_file"))
    valid_rows = load_records(config.data_path("valid_file"))
    if args.max_eval_examples is not None:
        valid_rows = valid_rows[: args.max_eval_examples]
    cutoff = int(config.training["cutoff_len"])
    train_dataset, train_token_audit = tokenize_records(
        tokenizer, train_rows, cutoff_len=cutoff, split="train"
    )
    eval_dataset, eval_token_audit = tokenize_records(
        tokenizer, valid_rows, cutoff_len=cutoff, split="valid"
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    initial_input, initial_output = model_category_rows(model, token_ids)
    initial_input = initial_input.detach().float().cpu().clone()
    initial_output = initial_output.detach().float().cpu().clone()
    before_probe = probe_loss(model, collator, train_dataset)

    class CustomTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            outputs = model(**inputs)
            base_loss = outputs.loss
            extra = regularizer.compute(model)
            loss = base_loss + extra["weighted_loss"]
            return (loss, outputs) if return_outputs else loss

    class PersistentCheckpointCallback(TrainerCallback):
        def on_save(self, training_args: Any, state: Any, control: Any, **_: Any) -> Any:
            if not args.persistent_checkpoint_dir:
                return control
            source = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            publish_checkpoint(
                source,
                Path(args.persistent_checkpoint_dir),
                identity,
                state.best_model_checkpoint,
            )
            return control

    effective_save_steps = int(config.training["save_steps"])
    if args.max_steps is not None and args.max_steps < effective_save_steps:
        effective_save_steps = args.max_steps
    callbacks: list[Any] = [
        EarlyStoppingCallback(
            early_stopping_patience=int(config.training["early_stopping_patience"])
        ),
        PersistentCheckpointCallback(),
    ]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config.training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config.training["gradient_accumulation_steps"]),
        num_train_epochs=float(config.training["epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=float(config.training["learning_rate"]),
        warmup_steps=int(config.training["warmup_steps"]),
        fp16=True,
        optim="adamw_torch",
        logging_steps=int(config.training["logging_steps"]),
        evaluation_strategy="steps",
        eval_steps=int(config.training["eval_steps"]),
        save_strategy="steps",
        save_steps=effective_save_steps,
        save_total_limit=int(config.training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        seed=seed,
        data_seed=seed,
        report_to=[],
        remove_unused_columns=True,
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    resume: str | bool | None = None
    if args.resume_from_checkpoint == "auto":
        if not args.persistent_checkpoint_dir:
            raise RuntimeError("auto resume requires --persistent-checkpoint-dir")
        resume = materialize_auto_resume(
            output_dir,
            Path(args.persistent_checkpoint_dir),
            identity,
        )
    elif args.resume_from_checkpoint != "none":
        resume = args.resume_from_checkpoint
    result = trainer.train(resume_from_checkpoint=resume)

    raw_model = remove_accelerate_fp32_forward_wrapper(trainer.model)
    after_probe = probe_loss(raw_model, collator, train_dataset)
    if args.max_steps is not None and not after_probe < before_probe:
        raise RuntimeError(
            f"Smoke probe loss did not decrease: {before_probe} -> {after_probe}"
        )
    final_adapter = output_dir / "final_adapter"
    raw_model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)
    write_json(
        output_dir / "adapter_state_reference.json",
        peft_state_manifest(raw_model),
    )
    final_input, final_output = model_category_rows(raw_model, token_ids)
    input_drift = torch.linalg.vector_norm(
        final_input.detach().float().cpu() - initial_input, dim=1
    )
    output_drift = torch.linalg.vector_norm(
        final_output.detach().float().cpu() - initial_output, dim=1
    )
    if int((input_drift > 0).sum()) == 0 or int((output_drift > 0).sum()) == 0:
        raise RuntimeError("Category input/output rows did not train")

    test_rows = load_records(config.data_path("test_file"))[
        : int(config.evaluation["reload_reference_examples"])
    ]
    reference_logits = candidate_logits(
        raw_model, tokenizer, test_rows, token_ids, cutoff
    )
    write_json(
        output_dir / "reload_reference.json",
        {
            "audit_version": "mera-original-reload-reference-v1",
            "row_indices": list(range(len(test_rows))),
            "candidate_token_ids": list(token_ids),
            "candidate_logits": reference_logits,
            "forward_path": "batch-one-compatible-raw-forward-v1",
        },
    )
    regularizer_audit = regularizer.audit()
    write_json(output_dir / "resolved_config.json", config.payload)
    write_json(
        output_dir / "environment.json",
        {
            "audit_version": "mera-original-environment-v1",
            "versions": jsonable_versions(),
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "bitsandbytes_fallback": os.environ.get("MERA_BNB_FALLBACK", "0") == "1",
            "apex_optional_backend_disabled": True,
        },
    )
    tokenizer_audit["trainable_parameters"] = trainable
    tokenizer_audit["input_drift_nonzero_rows"] = int((input_drift > 0).sum())
    tokenizer_audit["output_drift_nonzero_rows"] = int((output_drift > 0).sum())
    tokenizer_audit["input_drift_max"] = float(input_drift.max())
    tokenizer_audit["output_drift_max"] = float(output_drift.max())
    write_json(output_dir / "tokenizer_audit.json", tokenizer_audit)
    write_json(output_dir / "regularizer_audit.json", regularizer_audit)
    write_json(output_dir / "gradient_audit.json", gradient_audit)
    write_json(
        output_dir / "token_length_audit.json",
        {"train": train_token_audit, "valid": eval_token_audit},
    )
    write_json(output_dir / "loss_trace.json", trainer.state.log_history)
    summary = {
        "summary_version": "mera-original-train-summary-v1",
        "identity": identity,
        "actual_optimizer_steps": trainer.state.global_step,
        "configured_max_optimizer_steps": int(config.dataset_config["expected_max_steps"]),
        "stopped_early": trainer.state.global_step < int(config.dataset_config["expected_max_steps"]),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "probe_loss_before": before_probe,
        "probe_loss_after": after_probe,
        "resume_source": resume,
    }
    write_json(output_dir / "run_summary.json", summary)
    print("MERA_ORIGINAL_TRAIN_OK", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
