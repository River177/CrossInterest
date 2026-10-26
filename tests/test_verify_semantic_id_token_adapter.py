from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from new_interest.training.verify_semantic_id_token_adapter import (
    compare_adapter_state_dicts,
)


def test_compare_adapter_state_dicts_requires_bit_exact_tensors() -> None:
    saved = {
        "lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "trainable_tokens_delta": torch.tensor([[3.0, 4.0]]),
    }
    exact = compare_adapter_state_dicts(
        saved,
        {key: value.clone() for key, value in saved.items()},
    )
    assert exact["all_tensors_exact"] is True
    assert exact["saved_state_sha256"] == exact["loaded_state_sha256"]

    changed = {key: value.clone() for key, value in saved.items()}
    changed["trainable_tokens_delta"][0, 0] += 1
    mismatch = compare_adapter_state_dicts(saved, changed)
    assert mismatch["all_tensors_exact"] is False
    assert mismatch["mismatched_keys"] == ["trainable_tokens_delta"]


def test_compare_adapter_state_dicts_rejects_key_drift() -> None:
    result = compare_adapter_state_dicts(
        {"expected": torch.tensor([1.0])},
        {"unexpected": torch.tensor([1.0])},
    )
    assert result["all_tensors_exact"] is False
    assert result["missing_keys"] == ["expected"]
    assert result["unexpected_keys"] == ["unexpected"]
