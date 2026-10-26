from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.training.sft_data import (
    CompletionOnlyCollator,
    ParquetSFTDataset,
    load_sample_weights,
)


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        del tokenize, enable_thinking
        text = "".join(
            f"<{message['role']}>{message['content']}"
            for message in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {
            "input_ids": [ord(character) + 1 for character in text],
            "attention_mask": [1] * len(text),
        }


def _write_train(path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_sample_id": "s1",
                    "messages": [
                        {"role": "user", "content": "history one"},
                        {
                            "role": "assistant",
                            "content": "New topic: A",
                        },
                    ],
                },
                {
                    "source_sample_id": "s2",
                    "messages": [
                        {"role": "user", "content": "history two"},
                        {
                            "role": "assistant",
                            "content": "New topic: B",
                        },
                    ],
                },
            ]
        ),
        path,
    )


def _write_weights(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_dataset_joins_weights_by_source_identity(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    weight_path = tmp_path / "weights.parquet"
    _write_train(train_path)
    _write_weights(
        weight_path,
        [
            {"source_sample_id": "s2", "sample_weight": 0.5},
            {"source_sample_id": "s1", "sample_weight": 1.5},
        ],
    )

    dataset = ParquetSFTDataset(
        train_path,
        _Tokenizer(),
        max_seq_length=512,
        enable_thinking=False,
        sample_weight_file=weight_path,
    )
    assert [row["sample_weight"] for row in dataset.examples] == [1.5, 0.5]
    batch = CompletionOnlyCollator(_Tokenizer())(
        [dataset[0], dataset[1]]
    )
    assert batch["sample_weight"].tolist() == [1.5, 0.5]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"source_sample_id": "s1", "sample_weight": 1.0}],
            "identity mismatch",
        ),
        (
            [
                {"source_sample_id": "s1", "sample_weight": 1.0},
                {"source_sample_id": "s2", "sample_weight": 1.0},
                {"source_sample_id": "extra", "sample_weight": 1.0},
            ],
            "identity mismatch",
        ),
        (
            [
                {"source_sample_id": "s1", "sample_weight": 0.0},
                {"source_sample_id": "s2", "sample_weight": 1.0},
            ],
            "positive and finite",
        ),
    ],
)
def test_weight_loader_rejects_invalid_mappings(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "weights.parquet"
    _write_weights(path, rows)
    with pytest.raises(ValueError, match=message):
        load_sample_weights(
            path,
            expected_source_sample_ids=["s1", "s2"],
        )


def test_all_one_weight_matches_vanilla_loss_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    from new_interest.training.train_sft import (
        sample_weighted_causal_lm_loss,
    )

    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 0]])
    original = torch.randn(2, 4, 5)
    vanilla_logits = original.clone().requires_grad_(True)
    vanilla = functional.cross_entropy(
        vanilla_logits[:, :-1, :].reshape(-1, 5),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    vanilla.backward()

    weighted_logits = original.clone().requires_grad_(True)
    weighted = sample_weighted_causal_lm_loss(
        weighted_logits,
        labels,
        torch.ones(2),
    )
    weighted.backward()
    assert weighted.item() == pytest.approx(vanilla.item())
    assert torch.equal(weighted_logits.grad, vanilla_logits.grad)


def test_batch_size_one_weight_does_not_cancel() -> None:
    torch = pytest.importorskip("torch")

    from new_interest.training.train_sft import (
        sample_weighted_causal_lm_loss,
    )

    labels = torch.tensor([[-100, 1, 2]])
    original = torch.randn(1, 3, 4)
    full_logits = original.clone().requires_grad_(True)
    full_loss = sample_weighted_causal_lm_loss(
        full_logits,
        labels,
        torch.tensor([1.0]),
    )
    full_loss.backward()

    half_logits = original.clone().requires_grad_(True)
    half_loss = sample_weighted_causal_lm_loss(
        half_logits,
        labels,
        torch.tensor([0.5]),
    )
    half_loss.backward()
    assert half_loss.item() == pytest.approx(full_loss.item() * 0.5)
    assert torch.allclose(half_logits.grad, full_logits.grad * 0.5)


def test_per_example_mean_loss_equalizes_different_target_lengths() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    from new_interest.training.train_sft import (
        per_example_mean_causal_lm_loss,
    )

    labels = torch.tensor(
        [
            [-100, 1, -100, -100],
            [-100, 2, 3, 4],
        ]
    )
    original = torch.zeros(2, 4, 5)
    original[0, 0, 1] = -2.0
    original[1, 0, 2] = 4.0
    original[1, 1, 3] = 4.0
    original[1, 2, 4] = 4.0
    logits = original.requires_grad_(True)
    loss = per_example_mean_causal_lm_loss(logits, labels)

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    token_losses = functional.cross_entropy(
        shifted_logits.reshape(-1, 5),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    expected = (
        token_losses[0, :1].mean() + token_losses[1, :3].mean()
    ) / 2
    assert loss.item() == pytest.approx(expected.item())
    token_global_mean = token_losses[shifted_labels != -100].mean()
    assert loss.item() != pytest.approx(token_global_mean.item())


def test_per_example_mean_loss_rejects_empty_target_row() -> None:
    torch = pytest.importorskip("torch")

    from new_interest.training.train_sft import (
        per_example_mean_causal_lm_loss,
    )

    logits = torch.randn(2, 3, 4)
    labels = torch.tensor([[-100, 1, 2], [-100, -100, -100]])
    with pytest.raises(ValueError, match="Every sample"):
        per_example_mean_causal_lm_loss(logits, labels)
