from __future__ import annotations

import json

import numpy as np
import pytest

from new_interest.experiments.evaluate_nrms_checkpoint import (
    _architecture_from_selected,
    _load_checkpoint,
)


def test_load_nrms_checkpoint_preserves_identity_and_state(tmp_path) -> None:
    path = tmp_path / "selected_model.npz"
    selected = {
        "hidden_dimension": 8,
        "attention_heads": 2,
        "dropout": 0.1,
    }
    np.savez_compressed(
        path,
        labels=np.asarray(["a", "b"]),
        selected_json=np.asarray([json.dumps(selected)]),
        **{"parameter::classifier.weight": np.ones((2, 8))},
    )

    labels, loaded_selected, state = _load_checkpoint(path)

    assert labels == ("a", "b")
    assert loaded_selected == selected
    assert state["classifier.weight"].shape == (2, 8)
    architecture = _architecture_from_selected(loaded_selected)
    assert architecture.hidden_dimension == 8
    assert architecture.attention_heads == 2
    assert architecture.dropout == pytest.approx(0.1)


def test_load_nrms_checkpoint_rejects_duplicate_labels(tmp_path) -> None:
    path = tmp_path / "selected_model.npz"
    np.savez_compressed(
        path,
        labels=np.asarray(["a", "a"]),
        selected_json=np.asarray(
            [
                json.dumps(
                    {
                        "hidden_dimension": 8,
                        "attention_heads": 2,
                        "dropout": 0.1,
                    }
                )
            ]
        ),
        **{"parameter::classifier.weight": np.ones((2, 8))},
    )

    with pytest.raises(ValueError, match="unique"):
        _load_checkpoint(path)
