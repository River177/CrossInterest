from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_combined_training_and_evaluation_configs_are_aligned() -> None:
    train_config = yaml.safe_load(
        (
            ROOT
            / "configs/sft/experiments/"
            "static_sud_semantic_id_class_crossfit_"
            "qwen35_behavior_interest_weighted_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    eval_config = json.loads(
        (
            ROOT
            / "configs/id_token/"
            "static_sud_qwen35_behavior_interest_full.json"
        ).read_text(encoding="utf-8")
    )

    assert train_config["model_name"] == "Qwen/Qwen3.5-9B"
    assert train_config["train_file"].endswith(
        "semantic_id_token_200_h50_behavior_interest_v1/"
        "train/behavior_interest/21600.parquet"
    )
    assert train_config["sample_weight_file"].endswith(
        "semantic_id_class_crossfit_qwen35_behavior_interest_v1/"
        "sample_weights.parquet"
    )
    assert "global_crossfit" not in train_config["sample_weight_file"]
    assert train_config["semantic_id_token_count"] == 108
    assert train_config["enable_thinking"] is False
    assert train_config["gradient_checkpointing_use_reentrant"] is False
    assert train_config["per_device_train_batch_size"] == 1
    assert train_config["gradient_accumulation_steps"] == 8

    assert eval_config["model_name"] == train_config["model_name"]
    assert eval_config["eval_file"].endswith(
        "semantic_id_token_200_h50_behavior_interest_v1/"
        "eval/behavior_interest/dev_full_grouped.parquet"
    )
    assert (
        eval_config["interest_token_mapping_file"]
        == train_config["semantic_id_token_mapping_file"]
    )
    assert eval_config["top_k"] == 20
    assert eval_config["cutoffs"] == [1, 5, 10, 20]


def test_combined_amlt_configs_define_one_exact_four_gpu_job_each() -> None:
    expected = {
        "smoke": (
            "static_sud_semantic_id_class_crossfit_qwen35_"
            "behavior_interest_smoke_0826_v1.yaml",
            "qwen35-behavior-interest-semantic-id-class-crossfit-oof-smoke",
        ),
        "full": (
            "static_sud_semantic_id_class_crossfit_qwen35_"
            "behavior_interest_final_real_0826_v1.yaml",
            "qwen35-behavior-interest-semantic-id-class-crossfit-final",
        ),
    }
    for mode, (filename, job_name) in expected.items():
        config = yaml.safe_load(
            (ROOT / "configs/amlt/archive/2026-10-26" / filename).read_text(encoding="utf-8")
        )
        assert config["target"]["name"] == "Feeds"
        assert config["target"]["workspace_name"].endswith(
            "@rg-cs-newsandfeeds-singularity"
        )
        assert len(config["jobs"]) == 1
        job = config["jobs"][0]
        assert job["name"] == job_name
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert (
            job["submit_args"]["env"][
            (
                "SEMANTIC_ID_BEHAVIOR_INTEREST_CROSSFIT_NUM_GPUS"
                if mode == "smoke"
                else "SEMANTIC_ID_CLASS_CROSSFIT_NUM_GPUS"
            )
            ]
            == "4"
        )
        command = "\n".join(job["command"])
        assert "set -euo pipefail" in command
        if mode == "smoke":
            assert (
                "run_static_sud_semantic_id_class_crossfit_"
                "behavior_interest_smoke.sh"
            ) in command
        else:
            assert (
                "run_static_sud_semantic_id_class_crossfit_"
                "behavior_interest.sh full"
            ) in command


def test_combined_wrapper_freezes_class_weight_and_real_eval_contracts() -> None:
    script = (
        ROOT
        / 'scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_class_crossfit_behavior_interest.sh'
    ).read_text(encoding="utf-8")

    assert (
        "sft_static_user_disjoint_semantic_id_class_crossfit_"
        "qwen35_behavior_interest_v1"
    ) in script
    assert 'WEIGHT_FILE="$WEIGHT_ROOT/sample_weights.parquet"' in script
    assert "semantic_id_global_crossfit_qwen35_v1" not in script
    assert 'row["within_class_rank"]' in script
    assert 'weight_summary["formula"]["class_normalization"] is not True' in script
    assert '"weight_source_view": "behavior-plus-interest-semantic-id-oof"' in script
    assert '"weight_application_view": "behavior-plus-interest-semantic-id"' in script
    assert "--nproc_per_node=\"$NUM_GPUS\"" in script
    assert "--sample-weight-file \"$ACTIVE_WEIGHT_FILE\"" in script
    assert "--max-examples 64" in script
    assert "--expected-records 31392" in script
    assert '"shuffled_evaluation_used": False' in script
    assert '"history_prediction_filtering": False' in script


def test_combined_oof_config_and_wrapper_cover_exactly_five_folds() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / 'configs/amlt/archive/2026-10-26/static_sud_semantic_id_class_crossfit_qwen35_behavior_interest_oof_0826_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    jobs = config["jobs"]
    assert [job["name"] for job in jobs] == [
        f"qwen35-behavior-interest-semantic-id-class-crossfit-fold-{fold}"
        for fold in range(5)
    ]
    assert all(job["sku"] == "80G4-A100" for job in jobs)
    assert all(job["sla_tier"] == "Standard" for job in jobs)
    assert all(
        "run_static_sud_semantic_id_class_crossfit_behavior_interest_oof.sh"
        in "\n".join(job["command"])
        for job in jobs
    )

    script = (
        ROOT
        / 'scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_class_crossfit_behavior_interest_oof.sh'
    ).read_text(encoding="utf-8")
    assert "train/behavior_interest/21600.parquet" in script
    assert "static_sud_semantic_id_qwen35_behavior_interest_21600.yaml" in script
    assert "--sample-weight-file" not in script
    assert 'summary["expected_optimizer_steps"] != 1620' in script
    assert 'score["rows"] != 4320' in script
    assert 'set(counts.values()) != {40}' in script
    assert '"weighted_training_used": False' in script


def test_combined_crossfit_smoke_builds_its_own_within_class_weights() -> None:
    script = (
        ROOT
        / 'scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_class_crossfit_behavior_interest_smoke.sh'
    ).read_text(encoding="utf-8")
    assert "--max-per-class 2" in script
    assert "semantic_id_crossfit class-weights" in script
    assert 'split["train_rows"] != 216' in script
    assert 'split["heldout_rows"] != 216' in script
    assert 'summary["expected_optimizer_steps"] != 14' in script
    assert 'weights["formula"]["class_normalization"] is not True' in script
    assert '"weight_source_view": "behavior-plus-interest-semantic-id-oof"' in script


def test_combined_global_training_config_changes_only_weight_source() -> None:
    class_config = yaml.safe_load(
        (
            ROOT
            / "configs/sft/experiments/"
            "static_sud_semantic_id_class_crossfit_"
            "qwen35_behavior_interest_weighted_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    global_config = yaml.safe_load(
        (
            ROOT
            / "configs/sft/experiments/"
            "static_sud_semantic_id_global_crossfit_"
            "qwen35_behavior_interest_weighted_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    ignored = {"sample_weight_file", "output_dir", "run_name"}
    assert {
        key: value for key, value in global_config.items() if key not in ignored
    } == {
        key: value for key, value in class_config.items() if key not in ignored
    }
    assert global_config["sample_weight_file"].endswith(
        "semantic_id_global_crossfit_qwen35_behavior_interest_v1/"
        "sample_weights.parquet"
    )


def test_combined_global_amlt_configs_define_exact_smoke_and_full_jobs() -> None:
    expected = {
        "smoke": (
            "static_sud_semantic_id_global_crossfit_qwen35_"
            "behavior_interest_smoke_0828_v1.yaml",
            "qwen35-behavior-interest-semantic-id-global-crossfit-smoke",
        ),
        "full": (
            "static_sud_semantic_id_global_crossfit_qwen35_"
            "behavior_interest_final_real_0828_v1.yaml",
            "qwen35-behavior-interest-semantic-id-global-crossfit-final",
        ),
    }
    for mode, (filename, job_name) in expected.items():
        config = yaml.safe_load(
            (ROOT / "configs/amlt/archive/2026-10-26" / filename).read_text(encoding="utf-8")
        )
        assert config["target"] == {
            "service": "sing",
            "name": "Feeds",
            "workspace_name": (
                "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
            ),
        }
        assert len(config["jobs"]) == 1
        job = config["jobs"][0]
        assert job["name"] == job_name
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert (
            job["submit_args"]["env"]["SEMANTIC_ID_GLOBAL_CROSSFIT_NUM_GPUS"]
            == "4"
        )
        command = "\n".join(job["command"])
        assert command.startswith("set -euo pipefail")
        assert (
            "run_static_sud_semantic_id_global_crossfit_"
            f"behavior_interest.sh {mode}"
        ) in command


def test_combined_global_wrapper_freezes_fold_global_contract() -> None:
    script = (
        ROOT
        / 'scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_global_crossfit_behavior_interest.sh'
    ).read_text(encoding="utf-8")

    assert (
        "sft_static_user_disjoint_semantic_id_global_crossfit_"
        "qwen35_behavior_interest_v1"
    ) in script
    assert "4d41c6daae4f03a968e5e05effa232dda593a0ba8055523df9bed78c148c33e0" in script
    assert "2828bed1d6f1a6a485275272070673419f6f53535b4e0e7074a4b2d71336c3cc" in script
    assert 'row["fold_global_rank"]' in script
    assert 'row["fold_global_difficulty_quantile"]' in script
    assert 'row["within_class_rank"]' not in script
    assert "rank / 4319" in script
    assert "set(range(4320))" in script
    assert "math.fsum(values)" in script
    assert '"weight_rank_scope": "within-heldout-fold-across-target-classes"' in script
    assert '"class_normalization": False' in script
    assert '"class_normalization_used": False' in script
    assert '--sample-weight-file "$ACTIVE_WEIGHT_FILE"' in script
    assert 'expected_steps = 8 if mode == "smoke" else 2025' in script
    assert "--expected-records 31392" in script
    assert '"evaluation_scope": "real-dev-only"' in script
