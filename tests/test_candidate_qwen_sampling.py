from __future__ import annotations

from types import SimpleNamespace

import torch

from new_interest.evaluation.evaluate_candidate_qwen import (
    PresencePenaltyLogitsProcessor,
    generate_outputs,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[1, 2] for _ in texts]),
            "attention_mask": torch.tensor([[1, 1] for _ in texts]),
        }

    def batch_decode(self, generated, **kwargs):
        del kwargs
        return [str(int(row[-1])) for row in generated]


class _Model:
    device = torch.device("cpu")

    def __init__(self):
        self.generation_config = SimpleNamespace(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )
        self.calls = []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        self.calls.append(kwargs)
        sampled = torch.randint(3, 100, (len(input_ids), 1))
        return torch.cat([input_ids, sampled], dim=1)


def test_seeded_sampling_is_reproducible_and_passes_parameters() -> None:
    records = [{"prompt": "a"}, {"prompt": "b"}]

    first_model = _Model()
    first = generate_outputs(
        first_model,
        _Tokenizer(),
        records,
        model_name="Qwen/Qwen3.5-9B",
        batch_size=1,
        max_new_tokens=32,
        enable_thinking=False,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.1,
        seed=20260808,
        record_offset=0,
    )
    second_model = _Model()
    second = generate_outputs(
        second_model,
        _Tokenizer(),
        records,
        model_name="Qwen/Qwen3.5-9B",
        batch_size=1,
        max_new_tokens=32,
        enable_thinking=False,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.1,
        seed=20260808,
        record_offset=0,
    )

    assert first == second
    assert first_model.calls[0]["do_sample"] is True
    assert first_model.calls[0]["temperature"] == 0.7
    assert first_model.calls[0]["top_p"] == 0.8
    assert first_model.calls[0]["top_k"] == 20
    assert first_model.calls[0]["min_p"] == 0.0
    assert first_model.calls[0]["repetition_penalty"] == 1.1
    processor = first_model.calls[0]["logits_processor"][0]
    assert isinstance(processor, PresencePenaltyLogitsProcessor)


def test_presence_penalty_is_applied_once_per_seen_token() -> None:
    processor = PresencePenaltyLogitsProcessor(1.5)
    input_ids = torch.tensor([[1, 1, 3], [0, 2, 2]])
    scores = torch.tensor(
        [
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0],
        ]
    )

    adjusted = processor(input_ids, scores)

    assert torch.equal(
        adjusted,
        torch.tensor(
            [
                [10.0, 8.5, 10.0, 8.5],
                [18.5, 20.0, 18.5, 20.0],
            ]
        ),
    )
    assert torch.equal(
        scores,
        torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0, 20.0],
            ]
        ),
    )


def test_presence_penalty_ignores_prompt_padding() -> None:
    processor = PresencePenaltyLogitsProcessor(
        1.5,
        prompt_attention_mask=torch.tensor([[0, 1, 1]]),
    )

    adjusted = processor(
        torch.tensor([[0, 2, 3, 1]]),
        torch.tensor([[10.0, 10.0, 10.0, 10.0]]),
    )

    assert torch.equal(
        adjusted,
        torch.tensor([[10.0, 8.5, 8.5, 8.5]]),
    )
