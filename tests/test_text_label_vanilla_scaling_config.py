from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIZES = (5400, 10800, 16200, 21600)


def test_text_label_scaling_configs_are_matched() -> None:
    for model, model_name, learning_rate in (
        ("qwen35", "Qwen/Qwen3.5-9B", 2e-5),
        ("llama31", "meta-llama/Llama-3.1-8B-Instruct", 2e-4),
    ):
        for size in SIZES:
            path = ROOT / (
                "configs/sft/experiments/"
                f"static_sud_text_label_{model}_behavior_interest_"
                f"vanilla_{size}_v1.yaml"
            )
            config = yaml.safe_load(path.read_text())
            assert config["model_name"] == model_name
            assert config["train_file"].endswith(
                f"behavior_interest_scaling_v1/train/behavior_interest/{size}.parquet"
            )
            assert config["loss_reduction"] == "token_mean"
            assert "sample_weight_file" not in config
            assert config["enable_thinking"] is False
            assert config["epochs"] == 3
            assert config["per_device_train_batch_size"] == 1
            assert config["gradient_accumulation_steps"] == 8
            assert config["gradient_checkpointing_use_reentrant"] is False
            assert config["learning_rate"] == learning_rate
            assert config["seed"] == 42


def test_text_label_scaling_amlt_contract() -> None:
    smoke = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_text_label_behavior_interest_vanilla_scaling_smoke_0830_v1.yaml"
        ).read_text()
    )
    full = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_text_label_behavior_interest_vanilla_scaling_full_0830_v1.yaml"
        ).read_text()
    )
    assert [job["name"] for job in smoke["jobs"]] == [
        "qwen35-bi-text-label-vanilla-scaling-smoke",
        "llama31-bi-text-label-vanilla-scaling-smoke",
    ]
    assert [job["name"] for job in full["jobs"]] == [
        f"{model}-bi-text-label-vanilla-{size}"
        for model in ("qwen35", "llama31")
        for size in SIZES
    ]
    for job in smoke["jobs"] + full["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert job["command"][0] == "set -euo pipefail"
        command = "\n".join(job["command"])
        assert "run_static_sud_text_label_vanilla_scaling.sh" in command
        assert "HF_TOKEN" not in job["submit_args"]["env"]
    assert smoke["code"]["local_dir"] == full["code"]["local_dir"]


def test_text_label_scaling_wrapper_is_fail_closed() -> None:
    wrapper = (
        ROOT / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_text_label_vanilla_scaling.sh"
    ).read_text()
    assert "loss_reduction\": \"token_mean" in wrapper
    assert "New topic: " in wrapper
    assert "verify_lora_adapter" in wrapper
    assert "select_latest_complete_checkpoint" in wrapper
    assert "TEXT_LABEL_RECOVERY_CHECKPOINT_GATE_OK" in wrapper
    assert "recovery_checkpoint_audit.json" in wrapper
    assert "expected_steps - selected_step >= save_steps" in wrapper
    assert "Text-label training did not complete all configured epochs" in wrapper
    assert 'int(summary["expected_optimizer_steps"])' in wrapper
    assert "configured_epochs - epoch_tolerance" in wrapper
    assert "completed_epochs" in wrapper
    assert 'FINAL_CHECKPOINT="$PERSISTENT_CHECKPOINT_DIR/checkpoint-${EXPECTED_FULL_STEPS}"' not in wrapper
    assert "--nproc_per_node=\"$NUM_GPUS\"" in wrapper
    assert "TEXT_LABEL_BEAM_SMOKE_GATE_OK" in wrapper
    assert 'if len(labels) != 108 or int(metrics["top_k"]) != 20:' in wrapper
    assert "TEXT_LABEL_BEAM_FULL_GATE_OK" in wrapper
    assert "TEXT_LABEL_VANILLA_SCALING_GATE_OK" in wrapper


def test_llama_5400_recovery_is_scoped_to_fixed_bundle() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_text_label_llama31_5400_recovery_0830_v1.yaml"
        ).read_text()
    )
    assert config["code"]["local_dir"].endswith(
        "static_sud_text_label_vanilla_scaling_0830_v3"
    )
    assert [job["name"] for job in config["jobs"]] == [
        "llama31-bi-text-label-vanilla-5400"
    ]
    job = config["jobs"][0]
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert job["command"][0] == "set -euo pipefail"
    assert "llama31 full 5400" in "\n".join(job["command"])
