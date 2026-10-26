from __future__ import annotations

from pathlib import Path

import yaml

from new_interest.experiments.wang_stage_a import (
    evaluation_config,
    training_config,
)


MATRIX = Path("configs/wang_stage_a/experiment_matrix_v1.yaml")


def test_matrix_freezes_stage_a_and_exact_models() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert matrix["method"]["scope"] == "high-level language policy only"
    assert matrix["method"]["context_clusters"] == 2
    assert matrix["method"]["pairs_per_label"] == 10
    assert matrix["models"]["qwen35"]["name"] == "Qwen/Qwen3.5-9B"
    assert matrix["models"]["qwen2"]["name"] == "Qwen/Qwen2-1.5B"
    assert matrix["training"]["effective_global_batch_size"] == 16
    assert matrix["training"]["gradient_checkpointing_use_reentrant"] is False


def test_resolved_training_configs_preserve_model_specific_revisions() -> None:
    qwen35 = training_config(
        MATRIX,
        model_key="qwen35",
        dataset_key="mind",
        condition="balanced",
        output_dir="/tmp/qwen35",
    )
    qwen2 = training_config(
        MATRIX,
        model_key="qwen2",
        dataset_key="arts",
        condition="random",
        output_dir="/tmp/qwen2",
    )
    assert qwen35["model_revision"] == (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert qwen2["model_revision"] == (
        "8a16abf2848eda07cc5253dec660bf1ce007ad7a"
    )
    assert qwen35["learning_rate"] == 2e-5
    assert qwen2["learning_rate"] == 1e-4
    assert qwen35["train_file"].endswith("mind/train/balanced.parquet")
    assert qwen2["train_file"].endswith("arts/train/random.parquet")


def test_resolved_evaluation_configs_use_wang_prefix() -> None:
    generation = evaluation_config(
        MATRIX,
        model_key="qwen2",
        dataset_key="grocery",
        split="validation",
        view="supported",
        output_dir="/tmp/generation",
        adapter_path=None,
        kind="generation",
    )
    ranking = evaluation_config(
        MATRIX,
        model_key="qwen35",
        dataset_key="mind",
        split="validation",
        view="full",
        output_dir="/tmp/ranking",
        adapter_path="/tmp/adapter",
        kind="ranking",
    )
    assert generation["response_prefix"] == "New cluster: "
    assert "adapter_path" not in generation
    assert generation["labels_file"].endswith(
        "grocery/labels_supported.parquet"
    )
    assert ranking["response_prefix"] == "New cluster: "
    assert ranking["adapter_path"] == "/tmp/adapter"
    assert ranking["cutoffs"] == [1, 5, 10]


def test_ranking_config_accepts_audited_top20_override() -> None:
    ranking = evaluation_config(
        MATRIX,
        model_key="qwen35",
        dataset_key="mind",
        split="validation",
        view="supported",
        output_dir="/tmp/ranking-top20",
        adapter_path="/tmp/adapter",
        kind="ranking",
        top_k=20,
        cutoffs=(1, 5, 10, 20),
    )
    assert ranking["top_k"] == 20
    assert ranking["cutoffs"] == [1, 5, 10, 20]


def test_amlt_smoke_and_main_configs_preserve_stage_a_matrix() -> None:
    smoke = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/wang_stage_a_dual_model_smoke_0902_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    main = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/wang_stage_a_main_validation_0902_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert [job["name"] for job in smoke["jobs"]] == [
        "qwen35-mind-balanced-smoke",
        "qwen2-mind-balanced-smoke",
    ]
    assert len(main["jobs"]) == 6
    assert all(job["sku"] == "80G1-A100" for job in smoke["jobs"])
    assert all(job["sku"] == "80G4-A100" for job in main["jobs"])
    assert all(
        job["sla_tier"] == "Standard"
        for job in [*smoke["jobs"], *main["jobs"]]
    )
    commands = "\n".join(
        line for job in main["jobs"] for line in job["command"]
    )
    assert "full-validation" in commands
    assert "Stage B" not in commands
    assert all(
        job["submit_args"]["env"]["WANG_NUM_GPUS"] == "4"
        for job in main["jobs"]
    )


def test_top20_amlt_configs_are_eval_only_and_reuse_accepted_adapters() -> None:
    smoke = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/wang_stage_a_top20_smoke_0903_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    qwen2 = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/wang_stage_a_top20_qwen2_0903_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    qwen35 = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/wang_stage_a_top20_qwen35_0903_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert len(smoke["jobs"]) == 1
    assert smoke["jobs"][0]["submit_args"]["env"]["WANG_MAX_EXAMPLES"] == "64"
    jobs = [*qwen2["jobs"], *qwen35["jobs"]]
    assert len(jobs) == 6
    assert {job["name"] for job in jobs} == {
        "qwen2-arts-top20",
        "qwen2-grocery-top20",
        "qwen2-mind-top20",
        "qwen35-arts-top20",
        "qwen35-grocery-top20",
        "qwen35-mind-top20",
    }
    for job in jobs:
        command = "\n".join(job["command"])
        environment = job["submit_args"]["env"]
        assert "run_wang_stage_a_top20.sh" in command
        assert "train_sft" not in command
        assert "WANG_REUSE_SFT_DIR" in environment
        assert len(environment["WANG_EXPECTED_ADAPTER_SHA256"]) == 64
    assert all(job["sku"] == "80G4-A100" for job in qwen2["jobs"])
    assert all(job["sku"] == "80G8-A100" for job in qwen35["jobs"])
