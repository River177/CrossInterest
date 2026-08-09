from __future__ import annotations

from pathlib import Path

import pytest

from new_interest.training.train_sft import _resolve_adapter_path


def test_resolve_adapter_path_allows_new_lora() -> None:
    assert _resolve_adapter_path({}) is None
    assert _resolve_adapter_path({"adapter_path": None}) is None


def test_resolve_adapter_path_requires_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Input adapter not found"):
        _resolve_adapter_path({"adapter_path": tmp_path})

    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert _resolve_adapter_path({"adapter_path": str(tmp_path)}) == tmp_path
