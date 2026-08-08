from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.metrics import evaluate_outputs
from new_interest.sft_data import apply_chat_template

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("output", pa.string(), nullable=False),
        pa.field("parsed_label", pa.string()),
        pa.field("has_prefix", pa.bool_(), nullable=False),
        pa.field("valid_label", pa.bool_(), nullable=False),
        pa.field("set_hit_at_1", pa.bool_(), nullable=False),
        pa.field(
            "strict_response_hit_at_1",
            pa.bool_(),
            nullable=False,
        ),
        pa.field("novelty_compliant", pa.bool_(), nullable=False),
        pa.field(
            "acceptable_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "acceptable_responses",
            pa.list_(pa.string()),
            nullable=False,
        ),
    ]
)


def _load_valid_labels(path: Path) -> set[str]:
    return {
        str(row["target_subcategory"])
        for row in pq.read_table(
            path,
            columns=["target_subcategory"],
        ).to_pylist()
    }


def generate_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    *,
    batch_size: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> list[str]:
    import torch

    outputs: list[str] = []
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [
            apply_chat_template(
                tokenizer,
                [{"role": "user", "content": str(row["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
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
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, input_length:]
        outputs.extend(
            tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )
        )
    return outputs


def evaluate(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    adapter_path = config.get("adapter_path")
    eval_file = Path(config["eval_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enable_thinking = bool(config.get("enable_thinking", False))

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name,
        token=config.get("hf_token"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=config.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    records = pq.read_table(eval_file).to_pylist()
    max_examples = config.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]
    outputs = generate_outputs(
        model,
        tokenizer,
        records,
        batch_size=int(config.get("batch_size", 16)),
        max_new_tokens=int(config.get("max_new_tokens", 32)),
        enable_thinking=enable_thinking,
    )
    valid_labels = _load_valid_labels(
        Path(config["retained_interests_file"])
    )
    metrics, details = evaluate_outputs(records, outputs, valid_labels)
    pq.write_table(
        pa.Table.from_pylist(details, schema=PREDICTION_SCHEMA),
        output_dir / "predictions.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    if config.get("wandb_mode", "online") != "disabled":
        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-generative-sft",
                )
            ),
            name=str(config["run_name"]),
            job_type="evaluation",
            mode=str(config.get("wandb_mode", "online")),
            config=config,
        )
        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        run.log({f"eval/{key}": value for key, value in scalar_metrics.items()})
        artifact = wandb.Artifact(
            name=f"{config['run_name']}-predictions",
            type="evaluation",
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
        description="Evaluate SFT generation on grouped Dev."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evaluate(_load_json(args.config))


if __name__ == "__main__":
    main()
