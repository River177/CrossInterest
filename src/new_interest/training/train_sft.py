from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from new_interest.training.sft_data import (
    CompletionOnlyCollator,
    ParquetSFTDataset,
    audit_token_lengths,
)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML object")
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


def train(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    seed = int(config.get("seed", 42))
    _set_seed(seed)
    model_name = str(config["model_name"])
    adapter_path = _resolve_adapter_path(config)
    train_file = Path(config["train_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_seq_length = int(config.get("max_seq_length", 2048))
    enable_thinking = bool(config.get("enable_thinking", False))
    max_samples = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=config.get("hf_token"),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    token_audit = audit_token_lengths(
        train_file,
        tokenizer,
        max_seq_length=max_seq_length,
        enable_thinking=enable_thinking,
        output_path=output_dir / "token_audit.json",
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
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=config.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=config.get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

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
        output_dir=str(output_dir / "checkpoints"),
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
        gradient_checkpointing=True,
        logging_steps=int(config.get("logging_steps", 5)),
        save_strategy=str(config.get("save_strategy", "epoch")),
        save_steps=int(config.get("save_steps", 500)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        optim=str(config.get("optim", "adamw_torch_fused")),
        report_to=report_to,
        run_name=str(config["run_name"]),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=int(
            config.get("dataloader_num_workers", 2)
        ),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
    )
    result = trainer.train(
        resume_from_checkpoint=config.get("resume_from_checkpoint")
    )
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()

    summary = {
        "model_name": model_name,
        "input_adapter": (
            str(adapter_path.resolve())
            if adapter_path is not None
            else None
        ),
        "train_file": str(train_file.resolve()),
        "train_samples": len(dataset),
        "seed": seed,
        "enable_thinking": enable_thinking,
        "max_seq_length": max_seq_length,
        "token_audit": token_audit,
        "train_metrics": result.metrics,
        "final_adapter": str(final_dir.resolve()),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if wandb.run is not None:
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
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    for key in (
        "train_file",
        "adapter_path",
        "output_dir",
        "run_name",
        "max_samples",
        "resume_from_checkpoint",
        "wandb_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    train(config)


if __name__ == "__main__":
    main()
