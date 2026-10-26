from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from new_interest.training.diagnose_semantic_id_token_reload import (
    compare_capture_to_training_reference,
    compare_reload_captures,
)


def _capture(dtype: str, logits: list[float]) -> dict[str, object]:
    return {
        "model_name": "fixture/model",
        "inference_dtype": dtype,
        "mapping_sha256": "mapping",
        "token_ids": [10, 11],
        "torch_numeric_backend": {"deterministic": True},
        "sdpa_backend": "math",
        "forward_path": "raw-unwrapped-no-accelerate-autocast-v1",
        "model_load_path": "cpu-construct-adapter-then-cuda",
        "semantic_base_initialization": {
            "initialization_version": "fixture-cpu-fp32",
            "reduction_device": "cpu",
            "reduction_dtype": "float32",
            "input_initial_sha256": ["a", "b"],
            "output_initial_sha256": ["c", "d"],
        },
        "samples": [
            {
                "source_sample_id": "sample-0",
                "prompt_sha256": "prompt",
                "input_ids_sha256": "tokens",
                "restricted_logits": logits,
            }
        ],
    }


def test_compare_reload_captures_reports_numeric_mismatch() -> None:
    result = compare_reload_captures(
        _capture("bfloat16", [1.0, 2.0]),
        _capture("bfloat16", [1.25, 2.0]),
    )
    assert result["all_close"] is False
    assert result["identity_mismatches"] == []
    assert result["samples"][0]["max_absolute_difference"] == 0.25


def test_compare_reload_captures_accepts_strict_close_values() -> None:
    result = compare_reload_captures(
        _capture("float32", [1.0, 2.0]),
        _capture("float32", [1.001, 2.001]),
    )
    assert result["all_close"] is True


def test_compare_reload_captures_rejects_identity_drift() -> None:
    left = _capture("float32", [1.0, 2.0])
    right = _capture("float32", [1.0, 2.0])
    right["semantic_base_initialization"] = {
        "input_initial_sha256": ["changed"],
        "output_initial_sha256": ["c", "d"],
    }
    result = compare_reload_captures(left, right)
    assert result["all_close"] is False
    assert result["identity_mismatches"] == [
        "semantic_base_initialization"
    ]


def _training_reference(logits: list[float]) -> dict[str, object]:
    return {
        "mapping_sha256": "mapping",
        "token_ids": [10, 11],
        "torch_numeric_backend": {"deterministic": True},
        "sdpa_backend": "math",
        "forward_path": "raw-unwrapped-no-accelerate-autocast-v1",
        "semantic_initialization_version": "fixture-cpu-fp32",
        "semantic_initialization_reduction_device": "cpu",
        "semantic_initialization_reduction_dtype": "float32",
        "samples": [
            {
                "source_sample_id": "sample-0",
                "prompt_sha256": "prompt",
                "restricted_logits": logits,
            }
        ],
    }


def test_compare_capture_to_training_reference_reproduces_gate_failure() -> None:
    result = compare_capture_to_training_reference(
        _training_reference([1.0, 2.0]),
        _capture("bfloat16", [1.25, 2.0]),
    )
    assert result["all_close"] is False
    assert result["identity_mismatches"] == []
    assert result["samples"] == [
        {
            "source_sample_id": "sample-0",
            "prompt_sha256": "prompt",
            "input_ids_sha256": "tokens",
            "all_close": False,
            "max_absolute_difference": 0.25,
            "mean_absolute_difference": 0.125,
            "mean_signed_difference": 0.125,
            "max_difference_index": 0,
            "expected_at_max_difference": 1.0,
            "actual_at_max_difference": 1.25,
            "top1_match": True,
            "top20_exact_match": True,
            "top20_set_overlap": 2,
        }
    ]


def test_compare_capture_to_training_reference_accepts_strict_match() -> None:
    result = compare_capture_to_training_reference(
        _training_reference([1.0, 2.0]),
        _capture("bfloat16", [1.001, 2.001]),
    )
    assert result["all_close"] is True
