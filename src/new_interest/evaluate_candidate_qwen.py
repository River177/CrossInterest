from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.candidate_metrics import evaluate_candidate_outputs
from new_interest.sft_data import apply_chat_template


def generate_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    record_offset: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    for start in range(0, len(records), batch_size):
        if seed is not None:
            batch_seed = seed + record_offset + start
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
        batch = records[start : start + batch_size]
        texts = [
            apply_chat_template(
                tokenizer,
                [{"role": "user", "content": str(row["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
        }
        input_length = encoded["input_ids"].shape[1]
        generation_kwargs: dict[str, object] = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )
        outputs.extend(
            tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )
        )
    return outputs


def evaluate_qwen(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config.get("model_name", "Qwen/Qwen3-8B"))
    adapter_path = config.get("adapter_path")
    data_file = Path(config["data_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("batch_size", 8))
    checkpoint_size = int(config.get("checkpoint_size", 256))
    max_new_tokens = int(config.get("max_new_tokens", 32))
    do_sample = bool(config.get("do_sample", False))
    temperature = (
        float(config.get("temperature", 0.7)) if do_sample else None
    )
    top_p = float(config.get("top_p", 0.8)) if do_sample else None
    top_k = int(config.get("top_k", 20)) if do_sample else None
    seed = int(config["seed"]) if config.get("seed") is not None else None
    max_examples = config.get("max_examples")
    if batch_size <= 0 or checkpoint_size <= 0:
        raise ValueError("batch_size and checkpoint_size must be positive")
    if do_sample:
        if temperature is None or temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if top_p is None or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when sampling")
        if top_k is None or top_k <= 0:
            raise ValueError("top_k must be positive when sampling")
        if seed is None:
            raise ValueError("seed is required for reproducible sampling")
        if checkpoint_size % batch_size:
            raise ValueError(
                "checkpoint_size must be divisible by batch_size "
                "for resume-stable sampling"
            )

    records = pq.read_table(data_file).to_pylist()
    if max_examples is not None:
        records = records[: int(max_examples)]
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()
    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_ids = [
                str(row["selection_id"]) for row in records[start:stop]
            ]
            actual_ids = [
                str(row["selection_id"]) for row in shard_rows
            ]
            if actual_ids != expected_ids:
                raise ValueError(
                    f"{shard_path}: selection IDs do not match data"
                )
            outputs.extend(str(row["output"]) for row in shard_rows)
            resumed_samples += len(shard_rows)
            continue
        batch_outputs = generate_outputs(
            model,
            tokenizer,
            records[start:stop],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            record_offset=start,
        )
        shard_rows = [
            {
                "record_index": start + offset,
                "selection_id": str(
                    records[start + offset]["selection_id"]
                ),
                "output": output,
            }
            for offset, output in enumerate(batch_outputs)
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=f".{shard_path.name}.",
            dir=shards_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(
                pa.Table.from_pylist(shard_rows),
                temporary,
                compression="zstd",
                use_dictionary=True,
            )
            os.replace(temporary, shard_path)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.extend(batch_outputs)
        generated_samples += len(batch_outputs)

    metrics, details = evaluate_candidate_outputs(records, outputs)
    runtime_seconds = time.time() - started_at
    metrics.update(
        {
            "model_name": model_name,
            "adapter_path": (
                str(Path(adapter_path).resolve())
                if adapter_path
                else None
            ),
            "data_file": str(data_file.resolve()),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "runtime_seconds": runtime_seconds,
            "samples_per_second": (
                generated_samples / runtime_seconds
                if generated_samples
                else None
            ),
            "generated_samples_this_run": generated_samples,
            "resumed_samples": resumed_samples,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(details),
        output_dir / "predictions.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    wandb_mode = str(config.get("wandb_mode", "offline"))
    if wandb_mode != "disabled":
        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-generative-sft",
                )
            ),
            name=str(config["run_name"]),
            job_type="candidate-evaluation",
            mode=wandb_mode,
            config=config,
        )
        run.log(
            {
                f"candidate/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        artifact = wandb.Artifact(
            f"{config['run_name']}-predictions",
            type="candidate-evaluation",
        )
        artifact.add_file(str(output_dir / "predictions.parquet"))
        artifact.add_file(str(output_dir / "metrics.json"))
        run.log_artifact(artifact)
        run.finish()
    return metrics


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen candidate selection with greedy output."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_json(args.config)
    for key in (
        "data_file",
        "adapter_path",
        "output_dir",
        "run_name",
        "max_examples",
        "wandb_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    print(json.dumps(evaluate_qwen(config), indent=2))


if __name__ == "__main__":
    main()
