from __future__ import annotations

import numpy as np
import pytest

from new_interest.experiments.sasrec_gnip import (
    Architecture,
    GNIPExample,
    _batch_tensors,
    _candidate_rows,
    _parse_prompt_history,
    _single_target_selection_metrics,
    build_model,
    stratified_split,
)


def _example(example_id: str, target: int, history: tuple[int, ...] = (0, 1)) -> GNIPExample:
    return GNIPExample(
        example_id=example_id,
        user_id=f"user-{example_id}",
        prompt=f"prompt-{example_id}",
        history=history,
        positives=(target,),
    )


def test_parse_prompt_history_preserves_order_and_duplicates() -> None:
    prompt = """History:\n\n1. [Alpha] first\n2. [Beta] second\n3. [Alpha] third\n\nPredict."""

    assert _parse_prompt_history(prompt) == ("Alpha", "Beta", "Alpha")


def test_stratified_split_is_balanced_disjoint_and_deterministic() -> None:
    examples = tuple(
        _example(f"{target}-{index}", target)
        for target in range(3)
        for index in range(10)
    )

    fit, validation = stratified_split(examples, seed=42, validation_fraction=0.2)
    fit_again, validation_again = stratified_split(examples, seed=42, validation_fraction=0.2)

    assert len(fit) == 24
    assert len(validation) == 6
    assert {row.example_id for row in fit}.isdisjoint(
        {row.example_id for row in validation}
    )
    assert [row.example_id for row in fit] == [row.example_id for row in fit_again]
    assert [row.example_id for row in validation] == [row.example_id for row in validation_again]
    assert [sum(row.positives == (target,) for row in validation) for target in range(3)] == [2, 2, 2]


def test_model_forward_backward_and_strict_reload() -> None:
    torch = pytest.importorskip("torch")
    architecture = Architecture(16, 2, 2, 0.0, 5)
    examples = (_example("a", 2), _example("b", 3, (1, 0, 1)))
    model = build_model(input_label_count=4, label_count=4, architecture=architecture)
    input_ids, lengths, targets = _batch_tensors(
        examples,
        np.arange(2),
        maximum_history_length=5,
        device="cpu",
        include_targets=True,
    )

    logits = model(input_ids, lengths)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    clone = build_model(input_label_count=4, label_count=4, architecture=architecture)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()

    assert logits.shape == (2, 4)
    assert torch.isfinite(loss)
    assert torch.equal(model(input_ids, lengths), clone(input_ids, lengths))


def test_selection_metrics_are_class_macro() -> None:
    examples = (_example("a", 0), _example("b", 0), _example("c", 1))
    scores = np.zeros((3, 25), dtype=np.float32)
    scores[0, 0] = 3.0
    scores[1, 1:21] = np.arange(20, 0, -1, dtype=np.float32)
    scores[2, 1] = 3.0

    metrics = _single_target_selection_metrics(examples, scores)

    assert metrics["macro_recall_at_20"] == pytest.approx(0.75)
    assert metrics["macro_ndcg_at_10"] == pytest.approx(0.75)


def test_candidate_export_does_not_mask_history_labels() -> None:
    example = GNIPExample(
        example_id="x",
        user_id="u",
        prompt="p",
        history=(0, 0, 1),
        positives=(2,),
    )
    indices = np.asarray([[0, 2]], dtype=np.int64)
    scores = np.asarray([[10.0, 9.0]], dtype=np.float32)

    rows, metrics = _candidate_rows(
        (example,),
        labels=("history", "other-history", "target"),
        input_labels=("history", "other-history", "target"),
        top_indices=indices,
        top_scores=scores,
        split="train",
        in_sample=True,
    )

    assert rows[0]["candidate_interest_ids"] == ["C000", "C002"]
    assert rows[0]["history_overlap_count"] == 1
    assert rows[0]["target_in_candidates"] is True
    assert metrics["candidate_recall_at_20"] == 1.0


def test_candidate_history_overlap_matches_names_not_cross_vocab_indices() -> None:
    example = GNIPExample(
        example_id="x",
        user_id="u",
        prompt="p",
        history=(0,),
        positives=(1,),
    )

    rows, metrics = _candidate_rows(
        (example,),
        labels=("candidate-a", "target"),
        input_labels=("different-history", "candidate-a", "target"),
        top_indices=np.asarray([[0, 1]], dtype=np.int64),
        top_scores=np.asarray([[2.0, 1.0]], dtype=np.float32),
        split="dev",
        in_sample=False,
    )

    assert rows[0]["history_overlap_count"] == 0
    assert metrics["history_overlap_predictions"] == 0
