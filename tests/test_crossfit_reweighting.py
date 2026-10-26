from __future__ import annotations

import random

import pytest

from new_interest.data.build_crossfit_reweighting_dataset import (
    assign_balanced_hash_folds,
    build_matched_donor_mapping,
)
from new_interest.experiments.build_crossfit_weights import (
    compute_class_normalized_weights,
)
from new_interest.experiments.score_crossfit_sft import (
    SCORER_VERSION,
    category_token_encoding,
)


def _fold_rows() -> list[dict[str, object]]:
    return [
        {
            "source_sample_id": f"{target}-{index:02d}",
            "user_id": f"U-{target}-{index:02d}",
            "target_subcategory": target,
        }
        for target in ("A", "B")
        for index in range(10)
    ]


def test_balanced_hash_folds_are_exact_and_order_invariant() -> None:
    rows = _fold_rows()
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    first = assign_balanced_hash_folds(rows, seed=20260818, num_folds=5)
    second = assign_balanced_hash_folds(
        shuffled, seed=20260818, num_folds=5
    )
    assert first == second
    counts = {
        (target, fold): sum(
            row["target_subcategory"] == target and row["fold"] == fold
            for row in first
        )
        for target in ("A", "B")
        for fold in range(5)
    }
    assert set(counts.values()) == {2}
    assert len({row["source_sample_id"] for row in first}) == 20


def test_matched_donors_are_not_self_and_exclude_recipient_targets() -> None:
    histories = {
        "u1": {"titles": ["1", "2"], "categories": ["A", "B"]},
        "u2": {"titles": ["3", "4"], "categories": ["C", "D"]},
        "u3": {"titles": ["5", "6"], "categories": ["E", "F"]},
        "u4": {"titles": ["7", "8"], "categories": ["G", "H"]},
    }
    targets = {
        "u1": {"C"},
        "u2": {"E"},
        "u3": {"G"},
        "u4": {"A"},
    }
    mapping, audit = build_matched_donor_mapping(
        histories, targets, seed=20260818
    )
    repeated, repeated_audit = build_matched_donor_mapping(
        histories, targets, seed=20260818
    )
    assert mapping == repeated
    assert audit == repeated_audit
    for recipient, donor in mapping.items():
        assert donor != recipient
        assert targets[recipient].isdisjoint(histories[donor]["categories"])
    assert not any(row["is_fixed_point"] for row in audit)
    assert not any(row["target_history_overlap"] for row in audit)


def _score_rows() -> list[dict[str, object]]:
    return [
        {
            "source_sample_id": f"{target}-{index}",
            "user_id": f"u-{target}-{index}",
            "target_subcategory": target,
            "fold": index % 5,
            "category_token_nll": float(index),
        }
        for target in ("A", "B")
        for index in range(5)
    ]


def test_weights_have_class_mean_one_and_ratio_two() -> None:
    weights = compute_class_normalized_weights(_score_rows())
    for target in ("A", "B"):
        values = [
            float(row["sample_weight"])
            for row in weights
            if row["target_subcategory"] == target
        ]
        assert sum(values) / len(values) == pytest.approx(1.0)
        assert max(values) / min(values) == pytest.approx(2.0)
    easiest = next(row for row in weights if row["source_sample_id"] == "A-0")
    hardest = next(row for row in weights if row["source_sample_id"] == "A-4")
    assert easiest["sample_weight"] > hardest["sample_weight"]


def test_oof_scorer_identity_is_model_neutral() -> None:
    assert SCORER_VERSION == "category-token-oof-nll-v2"
    assert "qwen" not in SCORER_VERSION


def test_nll_rank_ties_break_by_source_identity() -> None:
    rows = _score_rows()
    rows[0]["category_token_nll"] = 1.0
    rows[1]["category_token_nll"] = 1.0
    weights = compute_class_normalized_weights(rows)
    ranks = {
        row["source_sample_id"]: row["within_class_nll_rank"]
        for row in weights
    }
    assert ranks["A-0"] < ranks["A-1"]


class _OffsetTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        del tokenize, add_generation_prompt, enable_thinking
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping,
    ):
        del add_special_tokens, return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_category_encoding_excludes_common_prefix_and_eos() -> None:
    target = "Topic A"
    encoded = category_token_encoding(
        _OffsetTokenizer(),
        [
            {"role": "user", "content": "History"},
            {"role": "assistant", "content": f"New topic: {target}"},
        ],
        target,
    )
    positions = encoded["category_positions"]
    text = encoded["full_text"]
    assert "".join(text[index] for index in positions) == target
    assert len(positions) == len(target)
