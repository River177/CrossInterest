from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_independent_pilots_are_separate_four_gpu_experiments() -> None:
    conditions = {
        "taxonomy": (
            "static_sud_taxonomy_softmax_qwen35_pilot_0901_v1.yaml",
            "qwen35-bi-taxonomy-softmax-5400",
        ),
        "frontier": (
            "static_sud_frontier_reweighting_qwen35_pilot_0901_v1.yaml",
            "qwen35-bi-frontier-reweighting-5400",
        ),
    }
    for method, (filename, expected_job) in conditions.items():
        config = yaml.safe_load(
            (ROOT / "configs/amlt/archive/2026-10-26" / filename).read_text(encoding="utf-8")
        )
        assert config["code"]["local_dir"].endswith(
            "amlt_inputs/static_sud_independent_pilots_0901_v1"
        )
        assert len(config["jobs"]) == 1
        job = config["jobs"][0]
        assert job["name"] == expected_job
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert job["command"][0] == "set -euo pipefail"
        assert job["submit_args"]["env"]["SEMANTIC_ID_PILOT_METHOD"] == method
        assert "run_static_sud_independent_pilot.sh" in "\n".join(
            job["command"]
        )


def test_independent_pilot_runner_is_full_dev_and_fail_closed() -> None:
    script = (
        ROOT / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_independent_pilot.sh"
    ).read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "--nproc_per_node=4" in script
    assert "--expected-records 31392" in script
    assert "expected_optimizer_steps\"] != 507" in script
    assert "SEMANTIC_ID_INDEPENDENT_PILOT_GATE_OK" in script
    assert "SEMANTIC_ID_PILOT_METHOD" in script


def test_taxonomy_cache_recovery_localizes_a_dedicated_snapshot() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_taxonomy_softmax_qwen35_pilot_0902_v2.yaml"
        ).read_text(encoding="utf-8")
    )
    job = config["jobs"][0]
    env = job["submit_args"]["env"]
    assert job["name"] == "qwen35-bi-taxonomy-softmax-5400"
    assert job["sku"] == "80G4-A100"
    assert env["SEMANTIC_ID_PILOT_METHOD"] == "taxonomy"
    assert env["SEMANTIC_ID_LOCALIZE_MODEL_CACHE"] == "1"
    assert env["HF_HOME"].endswith("huggingface-cache-taxonomy-pilot-v2")
    assert config["code"]["local_dir"].endswith(
        "amlt_inputs/static_sud_taxonomy_softmax_pilot_0902_v2"
    )

    script = (
        ROOT / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_independent_pilot.sh"
    ).read_text(encoding="utf-8")
    assert 'rsync -a "$PERSISTENT_MODEL_CACHE/" "$LOCAL_MODEL_CACHE/"' in script
    assert "model_cache_audit.json" in script
    assert "SEMANTIC_ID_INDEPENDENT_PILOT_LOCAL_CACHE_OK" in script
