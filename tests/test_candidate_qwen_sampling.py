from __future__ import annotations

from types import SimpleNamespace

import torch

from new_interest.evaluation.evaluate_candidate_qwen import generate_outputs


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
        batch_size=1,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=20260808,
        record_offset=0,
    )
    second_model = _Model()
    second = generate_outputs(
        second_model,
        _Tokenizer(),
        records,
        batch_size=1,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=20260808,
        record_offset=0,
    )

    assert first == second
    assert first_model.calls[0]["do_sample"] is True
    assert first_model.calls[0]["temperature"] == 0.7
    assert first_model.calls[0]["top_p"] == 0.8
    assert first_model.calls[0]["top_k"] == 20
