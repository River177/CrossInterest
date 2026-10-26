from __future__ import annotations

import pytest

from new_interest.evaluation.constrained_beam import (
    RankedPrediction,
    TokenTrie,
    build_response_trie,
    decode_ranked_sequences,
    fill_unique_predictions,
    make_prefix_allowed_tokens_fn,
)


class FakeTokenizer:
    eos_token_id = 1000

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids):
        return "".join(
            chr(token_id) for token_id in token_ids if token_id != 1000
        )


def test_response_trie_reaches_all_labels() -> None:
    tokenizer = FakeTokenizer()
    labels = ["Music", "Golf", "Film"]
    trie, sequences = build_response_trie(tokenizer, labels)
    assert set(sequences) == set(labels)
    for label, sequence in sequences.items():
        prefix = []
        for token_id in sequence:
            assert token_id in trie.allowed_tokens(prefix)
            prefix.append(token_id)
        assert trie.label_for(prefix) == label
    assert trie.allowed_tokens([123456]) == []


def test_decode_ranked_sequences_sorts_and_deduplicates() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    trie, sequences = build_response_trie(
        tokenizer,
        ["Music", "Golf"],
    )
    prompt = [1, 2, 3]
    generated = torch.tensor(
        [
            prompt + list(sequences["Music"]) + [0, 0],
            prompt + list(sequences["Golf"]) + [0, 0, 0],
            prompt + list(sequences["Music"]) + [0, 0],
        ]
    )
    scores = torch.tensor([0.2, 0.8, 0.1])
    ranked = decode_ranked_sequences(
        tokenizer,
        trie,
        generated,
        scores,
        initial_input_length=len(prompt),
        num_return_sequences=3,
    )
    assert [[item.label for item in row] for row in ranked] == [
        ["Golf", "Music"]
    ]
    assert ranked[0][0].score == pytest.approx(0.8)


def test_duplicate_token_sequences_are_rejected() -> None:
    with pytest.raises(ValueError, match="identical token sequences"):
        build_response_trie(
            FakeTokenizer(),
            ["Same", "Same"],
        )


def test_finished_prefix_continues_to_allow_eos() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    trie, sequences = build_response_trie(tokenizer, ["Music"])
    callback = make_prefix_allowed_tokens_fn(
        trie,
        initial_input_length=2,
    )
    complete = torch.tensor([1, 2, *sequences["Music"]])
    assert callback(0, complete) == [tokenizer.eos_token_id]
    repeated_eos = torch.tensor(
        [1, 2, *sequences["Music"], tokenizer.eos_token_id]
    )
    assert callback(0, repeated_eos) == [tokenizer.eos_token_id]


def test_unique_prediction_fill_retries_excluding_selected_labels() -> None:
    labels = [f"L{index:02d}" for index in range(24)]
    initial = [
        RankedPrediction(
            label=label,
            response=f"New topic: {label}",
            score=float(100 - index),
        )
        for index, label in enumerate(labels[:19])
    ]
    seen_remaining: list[list[str]] = []

    def generate_remaining(remaining: list[str]):
        seen_remaining.append(remaining)
        return [
            RankedPrediction(
                label=remaining[0],
                response=f"New topic: {remaining[0]}",
                score=1.0,
            )
        ]

    filled = fill_unique_predictions(
        initial,
        labels,
        top_k=20,
        generate_remaining=generate_remaining,
    )
    assert len(filled.predictions) == 20
    assert len({item.label for item in filled.predictions}) == 20
    assert filled.initial_unique_count == 19
    assert filled.retry_rounds == 1
    assert seen_remaining == [labels[19:]]


def test_unique_prediction_fill_rejects_retry_without_progress() -> None:
    initial = [RankedPrediction("A", "New topic: A", 1.0)]
    with pytest.raises(ValueError, match="no additional unique labels"):
        fill_unique_predictions(
            initial,
            ["A", "B"],
            top_k=2,
            generate_remaining=lambda remaining: [],
        )
