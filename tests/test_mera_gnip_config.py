from __future__ import annotations

from pathlib import Path
import json

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/sft/experiments/"
    "static_sud_mera_faithful_qwen35_interest_sid_21600_t005_v1.yaml"
)
EVAL_CONFIG = ROOT / (
    "configs/id_token/"
    "static_sud_mera_faithful_qwen35_interest_sid_t005_v1.json"
)

ABLATION_CONFIGS = {
    ("sid", "vanilla"): "static_sud_mera_qwen35_sid_sid_mean_vanilla_21600_v1.yaml",
    ("sid", "structure"): "static_sud_mera_qwen35_sid_sid_mean_structure_21600_v1.yaml",
    ("sid", "collaborative"): "static_sud_mera_qwen35_sid_sid_mean_collaborative_21600_v1.yaml",
    ("behavior-interest", "vanilla"): "static_sud_mera_qwen35_behavior_interest_mean_vanilla_21600_v1.yaml",
    ("behavior-interest", "structure"): "static_sud_mera_qwen35_behavior_interest_mean_structure_21600_v1.yaml",
    ("behavior-interest", "collaborative"): "static_sud_mera_qwen35_behavior_interest_mean_collaborative_21600_v1.yaml",
    ("behavior-interest", "full"): "static_sud_mera_qwen35_behavior_interest_mean_full_21600_v1.yaml",
}


def test_mera_gnip_config_is_aligned_with_semantic_id_vanilla() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["model_name"] == "Qwen/Qwen3.5-9B"
    assert config["train_file"].endswith(
        "mera_semantic_id_108_200_h50_v1/"
        "train/interest_sid/21600.parquet"
    )
    assert config["semantic_id_token_count"] == 108
    assert config["semantic_id_initialization_mode"] == "mean"
    assert config["mera_structure_weight"] == 0.05
    assert config["mera_collaborative_weight"] == 0.1
    assert config["mera_negative_samples"] == 10
    assert config["mera_cooccurrence_threshold"] == 0.05
    assert config["mera_min_collaborative_anchor_fraction"] == 0.0
    assert config["enable_thinking"] is False
    assert config["gradient_checkpointing_use_reentrant"] is False
    assert "sample_weight_file" not in config


def test_mera_gnip_eval_reconstructs_plain_mean_token_rows() -> None:
    config = json.loads(EVAL_CONFIG.read_text(encoding="utf-8"))
    assert config["model_name"] == "Qwen/Qwen3.5-9B"
    assert config["semantic_id_initialization_mode"] == "mean"
    assert config["top_k"] == 20
    assert config["cutoffs"] == [1, 5, 10, 20]


def test_remaining_mera_ablation_configs_are_exact_pure_mean_matrix() -> None:
    expected_weights = {
        "vanilla": (0.0, 0.0),
        "structure": (0.05, 0.0),
        "collaborative": (0.0, 0.1),
        "full": (0.05, 0.1),
    }
    for (view, condition), filename in ABLATION_CONFIGS.items():
        path = ROOT / "configs/sft/experiments" / filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["model_name"] == "Qwen/Qwen3.5-9B"
        assert config["semantic_id_token_count"] == 108
        assert config["semantic_id_initialization_mode"] == "mean"
        assert config["epochs"] == 3
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 8
        assert config["gradient_checkpointing_use_reentrant"] is False
        assert "sample_weight_file" not in config
        if view == "sid":
            assert "/train/interest_sid/21600.parquet" in config["train_file"]
        else:
            assert "/train/behavior_interest/21600.parquet" in config["train_file"]
        structure_weight, collaborative_weight = expected_weights[condition]
        if condition == "vanilla":
            assert "mera_relations_file" not in config
            assert "mera_structure_weight" not in config
            assert "mera_collaborative_weight" not in config
        else:
            assert config["mera_structure_weight"] == structure_weight
            assert config["mera_collaborative_weight"] == collaborative_weight
            assert config["mera_negative_samples"] == 10
            assert config["mera_cooccurrence_threshold"] == 0.05


def test_mera_ablation_full_amlt_contains_exactly_seven_missing_jobs() -> None:
    path = ROOT / "configs/amlt/archive/2026-10-26/static_sud_mera_qwen35_mean_ablation_full_0901_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = {job["name"] for job in config["jobs"]}
    assert names == {
        "qwen35-mera-sid-mean-vanilla-full",
        "qwen35-mera-sid-mean-structure-full",
        "qwen35-mera-sid-mean-collaborative-full",
        "qwen35-mera-bi-mean-vanilla-full",
        "qwen35-mera-bi-mean-structure-full",
        "qwen35-mera-bi-mean-collaborative-full",
        "qwen35-mera-bi-mean-full-full",
    }
    assert "qwen35-mera-sid-mean-full-full" not in names
    for job in config["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert job["command"][0] == "set -euo pipefail"


def test_behavior_interest_mera_eval_uses_pure_mean() -> None:
    path = ROOT / "configs/id_token/static_sud_mera_qwen35_behavior_interest_mean_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["semantic_id_initialization_mode"] == "mean"
    assert config["semantic_id_token_count"] == 108
    assert config["top_k"] == 20
    assert config["cutoffs"] == [1, 5, 10, 20]
