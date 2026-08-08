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

from new_interest.evaluation.constrained_beam import (
    build_response_trie,
    decode_ranked_sequences,
    make_prefix_allowed_tokens_fn,
)
from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.training.sft_data import apply_chat_template


def _load_labels(path: Path) -> list[str]:
    return sorted(
        {
            str(row["target_subcategory"])
            for row in pq.read_table(
                path,
                columns=["target_subcategory"],
            ).to_pylist()
        }
    )


def generate_ranked_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    labels: list[str],
    *,
    batch_size: int,
    num_beams: int,
    top_k: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> list[list[dict[str, object]]]:
    import torch

    if num_beams > len(labels):
        raise ValueError(
            f"num_beams={num_beams} exceeds {len(labels)} labels"
        )
    if top_k <= 0 or top_k > num_beams:
        raise ValueError("top_k must be positive and no greater than num_beams")
    trie, _ = build_response_trie(tokenizer, labels)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    ranked_outputs: list[list[dict[str, object]]] = []

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
        initial_input_length = encoded["input_ids"].shape[1]
        prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
            trie,
            initial_input_length=initial_input_length,
        )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        ranked_batches = decode_ranked_sequences(
            tokenizer,
            trie,
            generated.sequences,
            generated.sequences_scores,
            initial_input_length=initial_input_length,
            num_return_sequences=num_beams,
        )
        ranked_outputs.extend(
            [
                [
                    {
                        "label": prediction.label,
                        "response": prediction.response,
                        "score": prediction.score,
                    }
                    for prediction in predictions[:top_k]
                ]
                for predictions in ranked_batches
            ]
        )
    return ranked_outputs


def evaluate_beam(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    adapter_path = Path(config["adapter_path"])
    eval_file = Path(config["eval_file"])
    labels_file = Path(config["retained_interests_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_beams = int(config.get("num_beams", 40))
    top_k = int(config.get("top_k", 20))
    cutoffs = tuple(int(value) for value in config.get(
        "cutoffs",
        (1, 5, 10, 20),
    ))
    if max(cutoffs) > top_k:
        raise ValueError("Metric cutoff cannot exceed top_k")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    labels = _load_labels(labels_file)
    records = pq.read_table(eval_file).to_pylist()
    max_examples = config.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]

    batch_size = int(config.get("batch_size", 4))
    checkpoint_size = int(config.get("checkpoint_size", 64))
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    ranked_outputs: list[list[dict[str, object]]] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()
    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            if len(shard_rows) != stop - start:
                raise ValueError(
                    f"{shard_path}: expected {stop - start} rows, "
                    f"got {len(shard_rows)}"
                )
            expected_impressions = [
                str(row["impression_id"])
                for row in records[start:stop]
            ]
            actual_impressions = [
                str(row["impression_id"]) for row in shard_rows
            ]
            if actual_impressions != expected_impressions:
                raise ValueError(
                    f"{shard_path}: impression IDs do not match eval data"
                )
            resumed_samples += len(shard_rows)
            ranked_outputs.extend(
                [
                    [
                        {
                            "label": label,
                            "response": response,
                            "score": score,
                        }
                        for label, response, score in zip(
                            row["ranked_labels"],
                            row["ranked_responses"],
                            row["ranked_scores"],
                        )
                    ]
                    for row in shard_rows
                ]
            )
            continue

        batch_outputs = generate_ranked_outputs(
            model,
            tokenizer,
            records[start:stop],
            labels,
            batch_size=batch_size,
            num_beams=num_beams,
            top_k=top_k,
            max_new_tokens=int(config.get("max_new_tokens", 32)),
            enable_thinking=bool(config.get("enable_thinking", False)),
        )
        shard_rows = [
            {
                "record_index": start + offset,
                "impression_id": str(
                    records[start + offset]["impression_id"]
                ),
                "ranked_labels": [
                    item["label"] for item in ranked
                ],
                "ranked_responses": [
                    item["response"] for item in ranked
                ],
                "ranked_scores": [
                    item["score"] for item in ranked
                ],
            }
            for offset, ranked in enumerate(batch_outputs)
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=f".{shard_path.name}.",
            dir=shards_dir,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            pq.write_table(
                pa.Table.from_pylist(shard_rows),
                temporary_path,
                compression="zstd",
                use_dictionary=True,
            )
            os.replace(temporary_path, shard_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        generated_samples += len(batch_outputs)
        ranked_outputs.extend(batch_outputs)

    metrics, details = evaluate_ranked_outputs(
        records,
        ranked_outputs,
        cutoffs=cutoffs,
    )
    runtime_seconds = time.time() - started_at
    metrics["runtime_seconds"] = runtime_seconds
    metrics["samples_per_second"] = (
        generated_samples / runtime_seconds
        if generated_samples
        else None
    )
    metrics["generated_samples_this_run"] = generated_samples
    metrics["resumed_samples"] = resumed_samples
    metrics["shard_count"] = (
        len(records) + checkpoint_size - 1
    ) // checkpoint_size
    metrics["checkpoint_size"] = checkpoint_size
    metrics["num_beams"] = num_beams
    metrics["top_k"] = top_k
    metrics["cutoffs"] = list(cutoffs)
    metrics["model_name"] = model_name
    metrics["adapter_path"] = str(adapter_path.resolve())
    metrics["eval_file"] = str(eval_file.resolve())

    prediction_rows = [
        {
            **detail,
            "ranked_responses": [
                item["response"] for item in ranked
            ],
        }
        for detail, ranked in zip(details, ranked_outputs)
    ]
    pq.write_table(
        pa.Table.from_pylist(prediction_rows),
        output_dir / "beam_predictions.parquet",
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
        os.environ["WANDB_MODE"] = wandb_mode
        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-generative-sft",
                )
            ),
            name=str(config["run_name"]),
            job_type="beam-evaluation",
            config=config,
        )
        run.log(
            {
                f"beam/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        artifact = wandb.Artifact(
            f"{config['run_name']}-beam-predictions",
            type="evaluation",
        )
        artifact.add_file(
            str(output_dir / "beam_predictions.parquet")
        )
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
        description="Run constrained Beam ranking evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_json(args.config)
    for key in (
        "model_name",
        "adapter_path",
        "eval_file",
        "output_dir",
        "run_name",
        "max_examples",
        "wandb_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    metrics = evaluate_beam(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
