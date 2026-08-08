from __future__ import annotations

import pytest

from new_interest.sft_data import (
    CompletionOnlyCollator,
    tokenize_training_example,
)


class FakeTokenizer:
    name_or_path = "/tmp/local-adapter"
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert enable_thinking is False
        text = ""
        for message in messages:
            text += f"<{message['role']}>{message['content']}"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }


def test_completion_only_masking() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "history"},
        {"role": "assistant", "content": "New topic: Music"},
    ]
    encoded = tokenize_training_example(
        tokenizer,
        messages,
        max_seq_length=200,
        enable_thinking=False,
    )
    first_trainable = next(
        index
        for index, label in enumerate(encoded["labels"])
        if label != -100
    )
    prompt = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert first_trainable == prompt_length
    assert encoded["labels"][:prompt_length] == [-100] * prompt_length
    assert encoded["labels"][prompt_length:] == encoded["input_ids"][prompt_length:]


def test_tokenization_rejects_silent_truncation() -> None:
    with pytest.raises(ValueError, match="truncation is disabled"):
        tokenize_training_example(
            FakeTokenizer(),
            [
                {"role": "user", "content": "long history"},
                {"role": "assistant", "content": "New topic: Music"},
            ],
            max_seq_length=5,
            enable_thinking=False,
        )


def test_collator_masks_padding() -> None:
    torch = pytest.importorskip("torch")
    collator = CompletionOnlyCollator(FakeTokenizer())
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
            },
            {
                "input_ids": [1, 3],
                "attention_mask": [1, 1],
                "labels": [-100, 3],
            },
        ]
    )
    assert batch["input_ids"].shape == torch.Size([2, 3])
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][1].tolist() == [-100, 3, -100]
