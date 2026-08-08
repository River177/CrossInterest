from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_training_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq_length: int,
    enable_thinking: bool,
) -> dict[str, list[int]]:
    if len(messages) != 2:
        raise ValueError(
            f"Expected user/assistant messages, got {len(messages)}"
        )
    if messages[0].get("role") != "user":
        raise ValueError("First message must have role=user")
    if messages[1].get("role") != "assistant":
        raise ValueError("Second message must have role=assistant")

    prompt_text = apply_chat_template(
        tokenizer,
        messages[:1],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_encoding = tokenizer(
        full_text,
        add_special_tokens=False,
    )
    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding.get(
        "attention_mask",
        [1] * len(input_ids),
    )

    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"Tokenized sample has {len(input_ids)} tokens, exceeding "
            f"max_seq_length={max_seq_length}; truncation is disabled"
        )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokens are not a prefix of the full chat template"
        )
    if len(prompt_ids) >= len(input_ids):
        raise ValueError("Assistant response has no trainable tokens")

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class ParquetSFTDataset:
    def __init__(
        self,
        path: Path | str,
        tokenizer: Any,
        *,
        max_seq_length: int,
        enable_thinking: bool,
        max_samples: int | None = None,
    ) -> None:
        path = Path(path)
        rows = pq.read_table(path, columns=["messages"]).to_pylist()
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            rows = rows[:max_samples]
        self.examples = [
            tokenize_training_example(
                tokenizer,
                [
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                    for message in row["messages"]
                ],
                max_seq_length=max_seq_length,
                enable_thinking=enable_thinking,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(
                feature["input_ids"] + [pad_token_id] * padding
            )
            attention_masks.append(
                feature["attention_mask"] + [0] * padding
            )
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def audit_token_lengths(
    path: Path | str,
    tokenizer: Any,
    *,
    max_seq_length: int,
    enable_thinking: bool,
    output_path: Path | str | None = None,
) -> dict[str, object]:
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    rows = pq.read_table(path, columns=["messages"]).to_pylist()
    for row in rows:
        encoded = tokenize_training_example(
            tokenizer,
            [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
                for message in row["messages"]
            ],
            max_seq_length=max_seq_length,
            enable_thinking=enable_thinking,
        )
        lengths.append(len(encoded["input_ids"]))
        assistant_lengths.append(
            sum(label != -100 for label in encoded["labels"])
        )

    sorted_lengths = sorted(lengths)

    def percentile(fraction: float) -> int:
        index = min(
            len(sorted_lengths) - 1,
            max(0, int(len(sorted_lengths) * fraction + 0.999999) - 1),
        )
        return sorted_lengths[index]

    audit = {
        "samples": len(lengths),
        "min_tokens": min(lengths),
        "p50_tokens": percentile(0.50),
        "p90_tokens": percentile(0.90),
        "p95_tokens": percentile(0.95),
        "p99_tokens": percentile(0.99),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "max_assistant_tokens": max(assistant_lengths),
        "over_max_seq_length": sum(
            length > max_seq_length for length in lengths
        ),
        "max_seq_length": max_seq_length,
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(audit, indent=2) + "\n",
            encoding="utf-8",
        )
    return audit
