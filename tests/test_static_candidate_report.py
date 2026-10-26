from __future__ import annotations

import numpy as np

from new_interest.experiments.report_static_candidate_results import (
    _comparison_specs,
    _history_bucket,
    bootstrap_interest_macro,
)


def test_interim_manifest_declares_nonfinal_scope() -> None:
    import json
    from pathlib import Path

    manifest = json.loads(
        Path(
            "configs/candidate/static_user_disjoint_v1_gpu_only_interim.json"
        ).read_text()
    )
    assert manifest["report_scope"]["status"] == (
        "exploratory_dev_gpu_only_interim"
    )
    assert manifest["report_scope"]["excluded_api_conditions"] == 12


def test_history_bucket_does_not_mislabel_histories_longer_than_fifty() -> None:
    assert _history_bucket(36) == "36+"
    assert _history_bucket(356) == "36+"


def test_comparison_matrix_contains_requested_paired_contrasts() -> None:
    metadata = {
        "gpt56_medium_behavior_uniform": {"strategy": "uniform"},
        "gpt56_medium_interest_uniform": {"strategy": "uniform"},
        "gpt56_none_behavior_uniform": {"strategy": "uniform"},
        "gpt56_none_interest_uniform": {"strategy": "uniform"},
        "gpt56_medium_behavior_frequency_matched": {
            "strategy": "train_frequency_matched"
        },
        "gpt56_medium_interest_frequency_matched": {
            "strategy": "train_frequency_matched"
        },
    }
    specs = _comparison_specs(metadata)
    families = {row["family"] for row in specs}
    assert families == {
        "model_vs_baseline",
        "interest_vs_behavior",
        "medium_vs_none",
        "frequency_matched_vs_uniform",
    }
    assert len([row for row in specs if row["family"] == "model_vs_baseline"]) == 18


def test_user_cluster_bootstrap_recomputes_interest_macro() -> None:
    records = [
        {"user_id": "u1", "answer_category": "A"},
        {"user_id": "u1", "answer_category": "B"},
        {"user_id": "u2", "answer_category": "A"},
        {"user_id": "u3", "answer_category": "B"},
    ]
    correctness = {
        "left": np.asarray([True, True, False, True]),
        "right": np.asarray([False, True, False, False]),
    }
    specs = [
        {"family": "test", "name": "left_minus_right", "left": "left", "right": "right"}
    ]
    intervals, paired = bootstrap_interest_macro(
        records, correctness, specs, replicates=100, seed=17
    )
    assert intervals["resampling_unit"] == "user"
    assert intervals["conditions"]["left"]["point_estimate"] == 0.75
    assert intervals["conditions"]["right"]["point_estimate"] == 0.25
    assert paired["comparisons"][0]["point_delta"] == 0.5
