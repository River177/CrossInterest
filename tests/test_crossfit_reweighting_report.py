from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import new_interest.experiments.report_crossfit_reweighting as report_module
from new_interest.experiments.report_crossfit_reweighting import (
    bootstrap_interest_macro,
    report_crossfit_reweighting,
)


def test_user_cluster_bootstrap_recomputes_interest_macro() -> None:
    rows = [
        {"user_id": "u1", "target_subcategory": "A"},
        {"user_id": "u1", "target_subcategory": "B"},
        {"user_id": "u2", "target_subcategory": "A"},
        {"user_id": "u3", "target_subcategory": "B"},
    ]
    values = {
        "baseline": {
            "interest_macro_recall_at_10": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "interest_macro_ndcg_at_10": np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
        "weighted": {
            "interest_macro_recall_at_10": np.asarray([1.0, 1.0, 1.0, 0.0]),
            "interest_macro_ndcg_at_10": np.asarray([1.0, 1.0, 1.0, 0.0]),
        },
    }
    result = bootstrap_interest_macro(
        rows,
        values,
        [("weighted_minus_baseline", "weighted", "baseline")],
        replicates=2_000,
        seed=3,
    )
    assert result["resampling_unit"] == "user"
    recall = result["conditions"]["weighted"][
        "interest_macro_recall_at_10"
    ]
    assert recall["point_estimate"] == pytest.approx(0.75)
    delta = result["paired_comparisons"][0]["metrics"][
        "interest_macro_recall_at_10"
    ]
    assert delta["point_delta"] == pytest.approx(0.5)
    assert delta["lower"] <= delta["point_delta"] <= delta["upper"]


def test_real_only_report_does_not_require_shuffled_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    static_train = tmp_path / "train.parquet"
    record = {
        "impression_id": "i1",
        "user_id": "u1",
        "acceptable_subcategories": ["label-00"],
        "history_length": 3,
    }
    for task in ("behavior_only", "interest_only"):
        path = data_dir / "eval" / task / "real.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([record]), path)
    pq.write_table(
        pa.Table.from_pylist(
            [{"target_interest": {"subcategory": "label-00"}}]
        ),
        static_train,
    )
    prediction = {
        "record_index": 0,
        "impression_id": "i1",
        "ranked_labels": [f"label-{index:02d}" for index in range(20)],
    }
    metrics = {
        "samples": 1,
        "interest_macro_recall_at_10": 1.0,
        "interest_macro_ndcg_at_10": 1.0,
    }
    for _, directory in report_module.CONDITIONS.values():
        root = artifact_root / "final" / directory / "eval" / "real" / "merged"
        root.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([prediction]),
            root / "beam_predictions.parquet",
        )
        (root / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )

    monkeypatch.setattr(
        report_module,
        "bootstrap_interest_macro",
        lambda *args, **kwargs: {"replicates": 2_000},
    )
    output = report_crossfit_reweighting(
        data_dir,
        artifact_root,
        static_train,
        weight_dir=tmp_path / "llama-weights",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        real_only=True,
    )

    assert not (output / "real_minus_shuffled.json").exists()
    condition_rows = pq.read_table(output / "condition_metrics.parquet").to_pylist()
    assert {row["eval_name"] for row in condition_rows} == {"real"}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation_names"] == ["real"]
    assert summary["shuffled_controls_included"] is False
    assert summary["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert summary["weight_dir"].endswith("llama-weights")
