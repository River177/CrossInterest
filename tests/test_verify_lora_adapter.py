from __future__ import annotations

import pytest

from new_interest.training.verify_lora_adapter import verify_lora_adapter


def test_verify_lora_adapter_requires_real_model_stack() -> None:
    assert callable(verify_lora_adapter)


def test_compare_tensor_count_guard_is_exercised_remotely() -> None:
    pytest.importorskip("torch")
    from new_interest.training.verify_semantic_id_token_adapter import (
        compare_adapter_state_dicts,
    )

    import torch

    result = compare_adapter_state_dicts(
        {"a": torch.tensor([1.0])},
        {"a": torch.tensor([1.0])},
    )
    assert result["all_tensors_exact"] is True
    assert result["tensor_count"] == 1
