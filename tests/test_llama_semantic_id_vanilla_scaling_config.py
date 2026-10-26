from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIZES = (5400, 10800, 16200, 21600)
PRIMARY_SIZES = (10800, 21600)
SUPPLEMENT_SIZES = (5400, 16200)


def test_llama_scaling_train_and_eval_configs_are_aligned() -> None:
    for size in SIZES:
        config = yaml.safe_load(
            (
                ROOT
                / "configs/sft/experiments"
                / f"static_sud_semantic_id_llama31_behavior_interest_{size}.yaml"
            ).read_text()
        )
        assert config["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert config["train_file"].endswith(
            f"behavior_interest_scaling_v1/train/behavior_interest/{size}.parquet"
        )
        assert "sample_weight_file" not in config
        assert config["semantic_id_token_count"] == 108
        assert config["enable_thinking"] is False
        assert config["epochs"] == 3
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 8
        assert config["gradient_checkpointing_use_reentrant"] is False
        assert config["learning_rate"] == 2e-4
        assert config["seed"] == 42
    evaluation = json.loads(
        (
            ROOT
            / "configs/id_token/static_sud_llama31_behavior_interest_full.json"
        ).read_text()
    )
    assert evaluation["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert evaluation["top_k"] == 20
    assert evaluation["cutoffs"] == [1, 5, 10, 20]


def test_generic_semantic_id_runner_supports_combined_scaling() -> None:
    runner = (
        ROOT
        / "scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_semantic_id_token_sft.sh"
    ).read_text()
    assert '*behavior-interest*)' in runner
    assert 'TRAIN_SIZE="${SEMANTIC_ID_TRAIN_SIZE:-21600}"' in runner
    assert 'GRADIENT_AUDIT_BATCH_SIZE="${SEMANTIC_ID_GRADIENT_AUDIT_BATCH_SIZE:-2}"' in runner
    assert '--batch-size "$GRADIENT_AUDIT_BATCH_SIZE"' in runner
    assert '--save-steps "$FULL_SAVE_STEPS"' in runner
    assert 'summary["sample_weight_file"] is not None' in runner
    assert 'summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}' in runner


def test_llama_scaling_amlt_contract_and_gates() -> None:
    smoke = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_llama31_behavior_interest_vanilla_scaling_smoke_0829_v1.yaml"
        ).read_text()
    )
    full = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_llama31_behavior_interest_vanilla_scaling_full_0829_v1.yaml"
        ).read_text()
    )
    supplement = yaml.safe_load(
        (
            ROOT
            / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_llama31_behavior_interest_vanilla_scaling_supplement_0829_v1.yaml"
        ).read_text()
    )
    assert [job["name"] for job in smoke["jobs"]] == [
        "llama31-behavior-interest-semantic-id-vanilla-scaling-smoke"
    ]
    assert [job["name"] for job in full["jobs"]] == [
        f"llama31-behavior-interest-semantic-id-vanilla-{size}"
        for size in PRIMARY_SIZES
    ]
    assert [job["name"] for job in supplement["jobs"]] == [
        f"llama31-behavior-interest-semantic-id-vanilla-{size}"
        for size in SUPPLEMENT_SIZES
    ]
    assert supplement["code"]["local_dir"].endswith(
        "amlt_inputs/static_sud_semantic_id_llama_vanilla_scaling_0829_v2"
    )
    for job in smoke["jobs"] + full["jobs"] + supplement["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert job["command"][0] == "set -euo pipefail"
        assert "HF_TOKEN" not in job["submit_args"]["env"]
        command = "\n".join(job["command"])
        assert "install_semantic_id_token_llama_worker_deps.sh" in command
        assert "run_static_sud_semantic_id_llama_vanilla_scaling.sh" in command
    wrapper = (
        ROOT
        / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_llama_vanilla_scaling.sh"
    ).read_text()
    assert "SEMANTIC_ID_GRADIENT_AUDIT_BATCH_SIZE=1" in wrapper
    assert "EXPECTED_FULL_STEPS=507" in wrapper
    assert "EXPECTED_FULL_STEPS=1521" in wrapper
    assert "--max-examples 64" in wrapper
    assert 'integrity["tensor_count"] != 450' in wrapper
    assert "LLAMA_SEMANTIC_ID_VANILLA_SCALING_GATE_OK" in wrapper
