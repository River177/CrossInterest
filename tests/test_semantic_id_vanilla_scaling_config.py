from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIZES = (5400, 10800, 16200, 21600)


def test_vanilla_scaling_configs_are_aligned() -> None:
    for size in SIZES:
        config_file = ROOT / (
            "configs/sft/experiments/"
            f"static_sud_semantic_id_qwen35_behavior_interest_vanilla_{size}_v1.yaml"
        )
        config = yaml.safe_load(config_file.read_text())
        assert config["model_name"] == "Qwen/Qwen3.5-9B"
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
        assert config["learning_rate"] == 2e-5
        assert config["seed"] == 42


def test_vanilla_scaling_amlt_jobs_are_fail_closed() -> None:
    smoke = yaml.safe_load(
        (ROOT / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_qwen35_behavior_interest_vanilla_scaling_smoke_0829_v3.yaml").read_text()
    )
    full = yaml.safe_load(
        (ROOT / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_qwen35_behavior_interest_vanilla_scaling_full_0829_v1.yaml").read_text()
    )
    assert [job["name"] for job in smoke["jobs"]] == [
        "qwen35-bi-semantic-id-vanilla-scaling-smoke"
    ]
    assert [job["name"] for job in full["jobs"]] == [
        f"qwen35-bi-semantic-id-vanilla-{size}" for size in SIZES
    ]
    assert full["code"]["local_dir"].endswith(
        "amlt_inputs/static_sud_semantic_id_vanilla_scaling_0829_v3"
    )
    assert smoke["code"]["local_dir"].endswith(
        "amlt_inputs/static_sud_semantic_id_vanilla_scaling_0829_v3"
    )
    assert smoke["jobs"][0]["submit_args"]["env"][
        "SEMANTIC_ID_VANILLA_SCALING_REUSE_SINGLE_ROOT"
    ].endswith("7212034008.69907-6804a11f-919a")
    for job in smoke["jobs"] + full["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert job["command"][0] == "set -euo pipefail"
        assert job["submit_args"]["env"][
            "SEMANTIC_ID_VANILLA_SCALING_NUM_GPUS"
        ] == "4"
    assert full["target"] == smoke["target"] == {
        "service": "sing",
        "name": "Feeds",
        "workspace_name": (
            "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
        ),
    }


def test_vanilla_scaling_wrapper_and_eval_contract() -> None:
    wrapper = (
        ROOT / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_vanilla_scaling.sh"
    ).read_text()
    assert "sample-weight-file" not in wrapper
    assert 'gradient_checkpointing_kwargs"] != {"use_reentrant": False}' in wrapper
    assert "audit_semantic_id_token_gradients" in wrapper
    assert "--batch-size 1" in wrapper
    assert "verify_semantic_id_token_adapter" in wrapper
    assert "--nproc_per_node=\"$NUM_GPUS\"" in wrapper
    assert "--expected-records 31392" in wrapper
    assert "SEMANTIC_ID_VANILLA_SCALING_GATE_OK" in wrapper
    eval_config = json.loads(
        (ROOT / "configs/id_token/static_sud_qwen35_behavior_interest_vanilla_scaling_v1.json").read_text()
    )
    assert eval_config["model_name"] == "Qwen/Qwen3.5-9B"
    assert eval_config["top_k"] == 20
    assert eval_config["cutoffs"] == [1, 5, 10, 20]
