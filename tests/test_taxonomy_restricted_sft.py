from __future__ import annotations

import pytest

from new_interest.training.train_sft import (
    taxonomy_restricted_causal_lm_loss,
)


torch = pytest.importorskip("torch")


def _fixture() -> tuple[object, object, tuple[int, ...]]:
    logits = torch.zeros((2, 4, 8), dtype=torch.float64)
    labels = torch.tensor(
        [
            [-100, -100, 3, 7],
            [-100, 6, 7, -100],
        ],
        dtype=torch.long,
    )
    token_ids = (1, 3, 6)
    logits[0, 1, token_ids[0]] = -1.0
    logits[0, 1, token_ids[1]] = 2.0
    logits[0, 1, token_ids[2]] = 0.5
    logits[1, 0, token_ids[0]] = 1.0
    logits[1, 0, token_ids[1]] = 0.0
    logits[1, 0, token_ids[2]] = 3.0
    return logits, labels, token_ids


def test_taxonomy_loss_matches_manual_restricted_cross_entropy() -> None:
    logits, labels, token_ids = _fixture()
    actual = taxonomy_restricted_causal_lm_loss(
        logits,
        labels,
        token_ids,
    )
    expected = torch.nn.functional.cross_entropy(
        torch.stack(
            [
                logits[0, 1, list(token_ids)],
                logits[1, 0, list(token_ids)],
            ]
        ),
        torch.tensor([1, 2]),
    )
    torch.testing.assert_close(actual, expected)


def test_taxonomy_loss_ignores_full_vocabulary_and_closing_tokens() -> None:
    logits, labels, token_ids = _fixture()
    baseline = taxonomy_restricted_causal_lm_loss(logits, labels, token_ids)
    perturbed = logits.clone()
    perturbed[:, :, 0] = 10_000.0
    perturbed[0, 2, :] = torch.arange(8, dtype=torch.float64) * 1_000.0
    perturbed[1, 1, :] = -torch.arange(8, dtype=torch.float64) * 1_000.0
    actual = taxonomy_restricted_causal_lm_loss(
        perturbed,
        labels,
        token_ids,
    )
    torch.testing.assert_close(actual, baseline)


def test_taxonomy_loss_applies_sample_weights_after_example_reduction() -> None:
    logits, labels, token_ids = _fixture()
    weights = torch.tensor([1.5, 0.5], dtype=torch.float64)
    actual = taxonomy_restricted_causal_lm_loss(
        logits,
        labels,
        token_ids,
        weights,
    )
    individual = torch.nn.functional.cross_entropy(
        torch.stack(
            [
                logits[0, 1, list(token_ids)],
                logits[1, 0, list(token_ids)],
            ]
        ),
        torch.tensor([1, 2]),
        reduction="none",
    )
    expected = (individual * weights).mean()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "labels",
    [
        [[-100, 7, 7, -100], [-100, 6, 7, -100]],
        [[-100, 3, 6, -100], [-100, 6, 7, -100]],
    ],
)
def test_taxonomy_loss_requires_one_semantic_target_per_sample(labels) -> None:
    logits = torch.zeros((2, 4, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly one"):
        taxonomy_restricted_causal_lm_loss(
            logits,
            torch.tensor(labels, dtype=torch.long),
            (1, 3, 6),
        )
