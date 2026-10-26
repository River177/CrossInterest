from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_reasoning_stability_preflight_is_single_gpu_and_fail_closed() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / 'configs/amlt/archive/2026-10-26/static_sud_reasoning_stability_qwen35_preflight_0901_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    job = config["jobs"][0]
    assert job["name"] == "qwen35-bi-reasoning-stability-preflight"
    assert job["sku"] == "80G1-A100"
    assert job["sla_tier"] == "Standard"
    assert job["command"][0] == "set -euo pipefail"
    assert "WANDB_API_KEY" not in str(config)

    script = (
        ROOT
        / "scripts/archive/amlt/2026-10-26/evaluation/run_reasoning_stability_preflight.sh"
    ).read_text(encoding="utf-8")
    assert "--max-examples 16" in script
    assert "--rollouts 8" in script
    assert 'REASONING_STABILITY_MAX_NEW_TOKENS:-192' in script
    assert "mixed_correctness_groups" in script
    assert "raise SystemExit(2)" in script
    assert "b1d698724a90d929f28973433ba41ca8b95f2a5dc15523b4aead627e3c901f2f" in script


def test_reasoning_stability_format_repair_changes_only_generation_budget() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / 'configs/amlt/archive/2026-10-26/static_sud_reasoning_stability_qwen35_preflight_0901_v2.yaml'
        ).read_text(encoding="utf-8")
    )
    job = config["jobs"][0]
    assert job["name"] == "qwen35-bi-reasoning-stability-preflight"
    assert job["sku"] == "80G1-A100"
    assert job["submit_args"]["env"][
        "REASONING_STABILITY_MAX_NEW_TOKENS"
    ] == "384"
    assert "run_reasoning_stability_preflight.sh" in "\n".join(
        job["command"]
    )
