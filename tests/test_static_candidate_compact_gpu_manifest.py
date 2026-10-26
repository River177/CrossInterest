from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/candidate/static_user_disjoint_compact_single_view_gpu_v4.json"


def test_compact_gpu_manifest_covers_exact_missing_conditions() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["data"] == {
        "full_root": "data/training/candidate_selection_static_user_disjoint_k10_compact_v4",
        "pilot_root": "data/training/candidate_selection_static_user_disjoint_k10_compact_v4_pilot1000",
        "full_filename": "dev_expanded.parquet",
        "pilot_filename": "dev_pilot_1000.parquet",
        "full_samples": 37817,
        "pilot_samples": 1000,
        "supported_categories": 111,
    }
    conditions = {row["id"]: row for row in manifest["conditions"]}
    assert set(conditions) == {
        "qwen35_behavior_uniform",
        "qwen35_interest_uniform",
        "llama31_behavior_uniform",
        "llama31_interest_uniform",
    }
    for condition_id, condition in conditions.items():
        assert condition["evaluator"] == "hf"
        assert condition["strategy"] == "uniform"
        assert condition["task_type"] == (
            "behavior_only" if "behavior" in condition_id else "interest_only"
        )
        parameters = condition["parameters"]
        assert parameters["adapter_path"] is None
        assert parameters["wandb_mode"] == "disabled"
        assert parameters["batch_size"] == 4
        assert parameters["checkpoint_size"] == 128
        assert parameters["max_new_tokens"] == 64
        assert parameters["enable_thinking"] is False
        assert parameters["do_sample"] is False
        assert parameters["seed"] == 20260815


def test_compact_gpu_gcr_scripts_keep_pilot_gate_and_allowlist() -> None:
    condition_script = (
        ROOT / "scripts/gcr/run_static_candidate_gpu_condition.sh"
    ).read_text(encoding="utf-8")
    pilot_queue = (
        ROOT / "scripts/gcr/run_static_candidate_gpu_pilot_queue.sh"
    ).read_text(encoding="utf-8")
    full_queue = (
        ROOT / "scripts/gcr/run_static_candidate_gpu_full_queue.sh"
    ).read_text(encoding="utf-8")

    for condition_id in (
        "qwen35_behavior_uniform",
        "qwen35_interest_uniform",
        "llama31_behavior_uniform",
        "llama31_interest_uniform",
    ):
        assert condition_id in condition_script
        assert condition_id in pilot_queue
        assert condition_id in full_queue
    assert "--pilot-gate" in condition_script
    assert "finalize_static_candidate_pilots" in pilot_queue
