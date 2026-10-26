from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from new_interest.experiments.semantic_id_crossfit import (
    build_semantic_id_class_weights,
    build_semantic_id_global_weights,
    compute_fold_global_weights,
    compute_within_class_weights,
    prepare_semantic_id_fold,
    prior_adjusted_score_rows,
)


def _semantic_rows() -> list[dict[str, object]]:
    return [
        {
            "source_sample_id": f"{label}-{index}",
            "user_id": f"u-{label}-{index}",
            "target_subcategory": label,
            "prompt": f"history {label} {index}",
            "messages": [
                {"role": "user", "content": f"history {label} {index}"},
                {"role": "assistant", "content": token},
            ],
        }
        for label, token in (("A", "<interest_000>"), ("B", "<interest_001>"))
        for index in range(4)
    ]


def test_prepare_semantic_fold_is_disjoint_and_class_balanced(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    manifest = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(_semantic_rows()), train)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_sample_id": row["source_sample_id"],
                    "user_id": row["user_id"],
                    "target_subcategory": row["target_subcategory"],
                    "fold": index % 2,
                }
                for index, row in enumerate(_semantic_rows())
            ]
        ),
        manifest,
    )
    result = prepare_semantic_id_fold(
        train, manifest, tmp_path / "split", fold=0, max_per_class=1
    )
    assert result["train_rows"] == 2
    assert result["heldout_rows"] == 2
    assert result["overlapping_source_ids"] == 0
    assert {
        row["target_subcategory"]
        for row in pq.read_table(tmp_path / "split/train.parquet").to_pylist()
    } == {"A", "B"}


def test_prior_adjustment_removes_fold_output_bias() -> None:
    identities = [
        {"source_sample_id": "a", "user_id": "ua", "target_subcategory": "A"},
        {"source_sample_id": "b", "user_id": "ub", "target_subcategory": "B"},
    ]
    rows, summary = prior_adjusted_score_rows(
        np.asarray([[4.0, 0.0], [4.0, 0.0]]),
        [0, 1],
        identities,
        labels=["A", "B"],
        tokens=["<interest_000>", "<interest_001>"],
        fold=0,
        teacher_identity="teacher",
    )
    assert summary["fold_prior_sum"] == pytest.approx(1.0)
    assert rows[0]["prior_adjusted_nll"] == pytest.approx(np.log(2.0))
    assert rows[1]["prior_adjusted_nll"] == pytest.approx(np.log(2.0))
    assert rows[0]["target_raw_logprob"] > rows[1]["target_raw_logprob"]


def _score(source_id: str, label: str, difficulty: float) -> dict[str, object]:
    return {
        "source_sample_id": source_id,
        "user_id": f"u-{source_id}",
        "target_subcategory": label,
        "fold": 0,
        "prior_adjusted_nll": difficulty,
        "prior_adjusted_margin": difficulty - 0.5,
    }


def test_global_weights_have_global_mean_one_but_not_class_mean_one() -> None:
    weights = compute_fold_global_weights(
        [
            _score("a0", "A", 0.0),
            _score("a1", "A", 1.0),
            _score("b0", "B", 2.0),
            _score("b1", "B", 3.0),
        ]
    )
    values = [float(row["sample_weight"]) for row in weights]
    assert sum(values) / len(values) == pytest.approx(1.0)
    assert max(values) / min(values) == pytest.approx(2.0)
    class_means = {
        label: sum(float(row["sample_weight"]) for row in weights if row["target_subcategory"] == label) / 2
        for label in ("A", "B")
    }
    assert class_means["A"] > 1.0
    assert class_means["B"] < 1.0


def test_global_weight_normalization_uses_cross_python_fsum() -> None:
    scores = [
        _score(f"sample-{index:05d}", "A" if index % 2 == 0 else "B", float(index))
        for index in range(4320)
    ]
    weights = compute_fold_global_weights(scores)
    raw_mean = math.fsum(float(row["raw_weight"]) for row in weights) / len(weights)
    assert all(
        float(row["sample_weight"])
        == float(row["raw_weight"]) / raw_mean
        for row in weights
    )
    assert "math.fsum" in inspect.getsource(compute_fold_global_weights)


def test_within_class_weights_match_original_crossfit_recipe() -> None:
    weights = compute_within_class_weights(
        [
            _score("a0", "A", 0.0),
            _score("a1", "A", 1.0),
            _score("a2", "A", 2.0),
            _score("b0", "B", 10.0),
            _score("b1", "B", 11.0),
            _score("b2", "B", 12.0),
        ]
    )
    by_class = {
        label: [row for row in weights if row["target_subcategory"] == label]
        for label in ("A", "B")
    }
    values = [float(row["sample_weight"]) for row in weights]
    assert sum(values) / len(values) == pytest.approx(1.0)
    assert max(values) / min(values) == pytest.approx(2.0)
    for rows in by_class.values():
        assert sum(float(row["sample_weight"]) for row in rows) / 3 == pytest.approx(1.0)
        assert sorted(int(row["within_class_rank"]) for row in rows) == [0, 1, 2]
        assert sorted(float(row["within_class_difficulty_quantile"]) for row in rows) == [0.0, 0.5, 1.0]


def test_within_class_weight_normalization_uses_cross_python_fsum() -> None:
    scores = [
        _score(f"a{index:03d}", "A", float(index))
        for index in range(200)
    ]
    weights = compute_within_class_weights(scores)
    raw_mean = math.fsum(float(row["raw_weight"]) for row in weights) / 200
    assert all(
        float(row["sample_weight"])
        == float(row["raw_weight"]) / raw_mean
        for row in weights
    )
    assert "math.fsum" in inspect.getsource(compute_within_class_weights)


def test_class_weight_builder_requires_exact_train_coverage(tmp_path: Path) -> None:
    scores = tmp_path / "scores.parquet"
    train = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist([_score("a", "A", 0.0), _score("b", "B", 1.0)]),
        scores,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_sample_id": "a", "user_id": "u-a", "target_subcategory": "A"},
                {"source_sample_id": "extra", "user_id": "u-extra", "target_subcategory": "B"},
            ]
        ),
        train,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        build_semantic_id_class_weights([scores], train, tmp_path / "weights")


def test_weight_builder_requires_exact_train_coverage(tmp_path: Path) -> None:
    scores = tmp_path / "scores.parquet"
    train = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([_score("a", "A", 0.0), _score("b", "B", 1.0)]), scores)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_sample_id": "a", "user_id": "u-a", "target_subcategory": "A"},
                {"source_sample_id": "extra", "user_id": "u-extra", "target_subcategory": "B"},
            ]
        ),
        train,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        build_semantic_id_global_weights([scores], train, tmp_path / "weights")


def test_smoke_training_checkpoints_are_transient_and_distinct() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_global_crossfit_smoke.sh"
    ).read_text(encoding="utf-8")
    checkpoint_dirs = re.findall(r"--checkpoint-dir\s+([^\s\\]+)", script)
    assert checkpoint_dirs == [
        "/scratch/semantic-id-global-crossfit-teacher-checkpoints",
        "/scratch/semantic-id-global-crossfit-student-checkpoints",
    ]
    assert all("AMLT_OUTPUT_DIR" not in path for path in checkpoint_dirs)


def test_oof_amlt_config_has_exact_five_qwen_interest_jobs() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_global_crossfit_qwen35_interest_oof_0824_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    jobs = config["jobs"]
    assert [job["name"] for job in jobs] == [
        f"qwen35-interest-semantic-id-global-crossfit-fold-{fold}"
        for fold in range(5)
    ]
    assert config["target"]["name"] == "Feeds"
    assert all(job["sku"] == "80G4-A100" for job in jobs)
    assert all(job["sla_tier"] == "Standard" for job in jobs)
    assert all(
        job["submit_args"]["env"]["SEMANTIC_ID_GLOBAL_CROSSFIT_NUM_GPUS"]
        == "4"
        for job in jobs
    )
    assert all(
        "run_static_sud_semantic_id_global_crossfit_oof.sh"
        in "\n".join(job["command"])
        for job in jobs
    )


def test_oof_wrapper_enforces_unweighted_prior_adjusted_semantic_id_protocol() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_global_crossfit_oof.sh"
    ).read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-9B" in script
    assert "semantic_id_crossfit prepare" in script
    assert "semantic_id_crossfit score" in script
    assert "--sample-weight-file" not in script
    assert "--persistent-checkpoint-dir \"$PERSISTENT_CHECKPOINT_ROOT\"" in script
    assert "--resume-from-checkpoint auto" in script
    assert "--save-steps 100" in script
    assert "--epochs 3" in script
    assert "--gradient-accumulation-steps 8" in script
    assert 'summary["expected_optimizer_steps"] != 1620' in script
    assert 'score["rows"] != 4320' in script
    assert 'set(counts.values()) != {40}' in script
    assert '"shuffled_evaluation_used": False' in script
    assert '"weighted_training_used": False' in script
    assert "SEMANTIC_ID_GLOBAL_CROSSFIT_OOF_GATE_OK" in script


def test_final_config_consumes_only_audited_global_weights() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs/sft/experiments/static_sud_semantic_id_global_crossfit_qwen35_interest_weighted_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["model_name"] == "Qwen/Qwen3.5-9B"
    assert config["train_file"].endswith(
        "semantic_id_token_200_h50_v1/train/interest_only/21600.parquet"
    )
    assert config["sample_weight_file"].endswith(
        "semantic_id_global_crossfit_qwen35_v1/sample_weights.parquet"
    )
    assert config["semantic_id_token_count"] == 108
    assert config["enable_thinking"] is False
    assert config["epochs"] == 3
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 8
    assert config["gradient_checkpointing_use_reentrant"] is False
    assert config["save_steps"] == 100


def test_final_amlt_config_has_one_exact_qwen_interest_job() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs/amlt/archive/2026-10-26/static_sud_semantic_id_global_crossfit_qwen35_interest_final_real_0824_v1.yaml"
        ).read_text(encoding="utf-8")
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
    assert job["name"] == "qwen35-interest-semantic-id-global-crossfit-final"
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert (
        job["submit_args"]["env"]["SEMANTIC_ID_GLOBAL_CROSSFIT_NUM_GPUS"]
        == "4"
    )
    command = "\n".join(job["command"])
    assert command.startswith("set -euo pipefail")
    assert "run_static_sud_semantic_id_global_crossfit_final.sh" in command


def test_final_wrapper_enforces_weighted_real_only_semantic_id_protocol() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_global_crossfit_final.sh"
    ).read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-9B" in script
    assert (
        "685657ea1893d8cf5ca088ceca88c619e45f16e95631bd11edf83b434a66ff4f"
        in script
    )
    assert '--sample-weight-file "$WEIGHT_FILE"' in script
    assert '--persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT"' in script
    assert "--resume-from-checkpoint auto" in script
    assert "--epochs 3" in script
    assert "--gradient-accumulation-steps 8" in script
    assert 'summary["expected_optimizer_steps"] != 2025' in script
    assert "merge_id_token_shards" in script
    assert "--expected-records 31392" in script
    assert '--num-shards 4' in script
    assert 'metrics["logical_num_shards"] != 4' in script
    assert 'metrics["supported_interest_count"] != 104' in script
    assert 'metrics["history_prediction_filtering"] is not False' in script
    assert '"evaluation_scope": "real-dev-only"' in script
    assert '"shuffled_evaluation_used": False' in script
    assert '"global_class_normalization_used": False' in script
    assert '"checkpoints/"' in script
    assert "SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_GATE_OK" in script


def test_class_crossfit_final_config_consumes_only_class_rank_weights() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs/sft/experiments/static_sud_semantic_id_class_crossfit_qwen35_interest_weighted_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["model_name"] == "Qwen/Qwen3.5-9B"
    assert config["train_file"].endswith(
        "semantic_id_token_200_h50_v1/train/interest_only/21600.parquet"
    )
    assert config["sample_weight_file"].endswith(
        "semantic_id_class_crossfit_qwen35_v1/sample_weights.parquet"
    )
    assert config["semantic_id_token_count"] == 108
    assert config["enable_thinking"] is False
    assert config["epochs"] == 3
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 8
    assert config["gradient_checkpointing_use_reentrant"] is False
    assert config["save_steps"] == 100


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_class_crossfit_final_amlt_config_has_one_exact_job(
    version: str,
) -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            root
            / f"configs/amlt/archive/2026-10-26/static_sud_semantic_id_class_crossfit_qwen35_interest_final_real_0826_{version}.yaml"
        ).read_text(encoding="utf-8")
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
    assert job["name"] == "qwen35-interest-semantic-id-class-crossfit-final"
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert job["submit_args"]["env"]["SEMANTIC_ID_CLASS_CROSSFIT_NUM_GPUS"] == "4"
    command = "\n".join(job["command"])
    assert command.startswith("set -euo pipefail")
    assert "run_static_sud_semantic_id_class_crossfit_final.sh" in command


def test_class_crossfit_final_wrapper_enforces_original_rank_recipe() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_semantic_id_class_crossfit_final.sh"
    ).read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-9B" in script
    assert "9cca060a957ffc60f81dbda1f36ab7f7d282aca89882bdbd5aa1c6536cb04f06" in script
    assert "within_class_rank" in script
    assert "within_class_difficulty_quantile" in script
    assert "rank / 199" in script
    assert "set(range(200))" in script
    assert "raw_mean = math.fsum" in script
    assert '"class_normalization": True' in script
    assert '"class_normalization_used": True' in script
    assert '"weight_rank_scope": "within-target-class-over-all-oof-rows"' in script
    assert "fold_global_rank" not in script
    assert "fold_global_difficulty_quantile" not in script
    assert "global_weight_summary.json" not in script
    assert '--sample-weight-file "$WEIGHT_FILE"' in script
    assert '--persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT"' in script
    assert "--resume-from-checkpoint auto" in script
    assert 'summary["expected_optimizer_steps"] != 2025' in script
    assert "merge_id_token_shards" in script
    assert "--expected-records 31392" in script
    assert '--num-shards 4' in script
    assert 'metrics["history_prediction_filtering"] is not False' in script
    assert '"evaluation_scope": "real-dev-only"' in script
    assert '"shuffled_evaluation_used": False' in script
    assert '"checkpoints/"' in script
    assert "SEMANTIC_ID_CLASS_CROSSFIT_FINAL_GATE_OK" in script
