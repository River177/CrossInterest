from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.candidate_metrics import evaluate_candidate_outputs
from new_interest.training.sft_data import tokenize_training_example


def evaluate_likelihood(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config.get("model_name", "Qwen/Qwen3-8B"))
    data_file = Path(config["data_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("batch_size", 16))
    max_examples = config.get("max_examples")
    records = pq.read_table(data_file).to_pylist()
    if max_examples is not None:
        records = records[: int(max_examples)]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()

    flattened = []
    for record_index, record in enumerate(records):
        for candidate_index, candidate in enumerate(record["candidates"]):
            response = f"New topic: {candidate}"
            encoded = tokenize_training_example(
                tokenizer,
                [
                    {"role": "user", "content": str(record["prompt"])},
                    {"role": "assistant", "content": response},
                ],
                max_seq_length=int(config.get("max_seq_length", 2048)),
                enable_thinking=False,
            )
            flattened.append(
                {
                    "record_index": record_index,
                    "candidate_index": candidate_index,
                    "candidate": str(candidate),
                    **encoded,
                }
            )

    scores: list[list[float]] = [
        [float("-inf")] * 10 for _ in records
    ]
    started_at = time.time()
    for start in range(0, len(flattened), batch_size):
        batch = flattened[start : start + batch_size]
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            padding = max_length - len(item["input_ids"])
            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * padding
            )
            attention_mask.append(
                item["attention_mask"] + [0] * padding
            )
            labels.append(item["labels"] + [-100] * padding)
        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=model.device,
        )
        mask_tensor = torch.tensor(
            attention_mask,
            dtype=torch.long,
            device=model.device,
        )
        label_tensor = torch.tensor(
            labels,
            dtype=torch.long,
            device=model.device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
            ).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = label_tensor[:, 1:]
        answer_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(
            ~answer_mask,
            0,
        )
        token_log_probs = F.log_softmax(
            shifted_logits,
            dim=-1,
        ).gather(
            -1,
            safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        sequence_scores = (
            (token_log_probs * answer_mask).sum(dim=-1)
            / answer_mask.sum(dim=-1)
        )
        for item, score in zip(batch, sequence_scores.tolist()):
            scores[item["record_index"]][
                item["candidate_index"]
            ] = score

    outputs = []
    prediction_rows = []
    for record, candidate_scores in zip(records, scores):
        best_index = max(
            range(len(candidate_scores)),
            key=lambda index: (
                candidate_scores[index],
                -index,
            ),
        )
        candidate = str(record["candidates"][best_index])
        output = f"New topic: {candidate}"
        outputs.append(output)
        prediction_rows.append(
            {
                "selection_id": str(record["selection_id"]),
                "output": output,
                "selected_candidate": candidate,
                "selected_position": best_index,
                "candidate_scores": candidate_scores,
            }
        )
    metrics, details = evaluate_candidate_outputs(records, outputs)
    metrics.update(
        {
            "scoring": "mean_completion_log_probability",
            "runtime_seconds": time.time() - started_at,
            "model_name": model_name,
        }
    )
    merged = [
        {**prediction, **detail}
        for prediction, detail in zip(prediction_rows, details)
    ]
    pq.write_table(
        pa.Table.from_pylist(merged),
        output_dir / "predictions.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score candidate responses by conditional likelihood."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_likelihood(config), indent=2))


if __name__ == "__main__":
    main()
