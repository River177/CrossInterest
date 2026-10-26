from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_preference_state_reasoning_sft_smoke_is_fail_closed() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_preference_state_reasoning_sft_smoke_0902_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["code"]["local_dir"].endswith(
        "amlt_inputs/static_sud_preference_state_reasoning_sft_smoke_0902_v1"
    )
    job = config["jobs"][0]
    assert job["name"] == "qwen35-bi-preference-state-reasoning-sft-smoke"
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert job["command"][0] == "set -euo pipefail"
    assert "WANDB_API_KEY" not in str(config)
    script = (
        ROOT / "scripts/archive/amlt/2026-10-26/sft/run_preference_state_reasoning_sft_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "PREFERENCE_STATE_REASONING_SFT_TOKEN_AUDIT_OK" in script
    assert "--nproc_per_node=4" in script
    assert "single_reload_audit.json" in script
    assert "ddp_reload_audit.json" in script
    assert "evaluate_preference_state_format" in script
    assert "valid_format\"] < 12" in script
    assert "PREFERENCE_STATE_REASONING_SFT_SMOKE_GATE_OK" in script


def test_reasoning_sft_config_continues_the_frozen_21600_adapter() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/sft/experiments/"
            "static_sud_preference_state_reasoning_qwen35_behavior_interest_1080_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["model_name"] == "Qwen/Qwen3.5-9B"
    assert config["model_revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert "7212029499.70786-e4538499-b9a2" in config["adapter_path"]
    assert config["enable_thinking"] is True
    assert config["epochs"] == 2
    assert config["learning_rate"] == 1e-5
    assert config["gradient_checkpointing_use_reentrant"] is False
