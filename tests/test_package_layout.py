from __future__ import annotations

import importlib
from pathlib import Path

import tomllib

from new_interest.infrastructure.paths import portable_path


MODULES = (
    "new_interest.data.mind_io",
    "new_interest.data.build_dataset",
    "new_interest.data.build_static_user_disjoint_dataset",
    "new_interest.data.build_dynamic_dataset",
    "new_interest.data.build_dynamic_generative_dataset",
    "new_interest.data.build_future_interest_benchmark",
    "new_interest.data.build_semantic_id_token_sft_dataset",
    "new_interest.data.build_generative_dataset",
    "new_interest.data.build_sft_subsets",
    "new_interest.data.build_candidate_selection",
    "new_interest.data.build_grpo_dataset",
    "new_interest.data.build_wang_stage_a_dataset",
    "new_interest.training.sft_data",
    "new_interest.training.grpo_reward",
    "new_interest.training.text_adapter_reload",
    "new_interest.training.global_sample_cursor",
    "new_interest.training.old_new_execution",
    "new_interest.training.execution_migration",
    "new_interest.training.execution_probe",
    "new_interest.training.sft_state_audit",
    "new_interest.data.build_old_new_interest_dataset",
    "new_interest.data.restore_static_mind_projection",
    "new_interest.evaluation.metrics",
    "new_interest.evaluation.ranking_metrics",
    "new_interest.evaluation.evaluate_future_interest",
    "new_interest.evaluation.evaluate_id_token_ranking",
    "new_interest.evaluation.merge_id_token_shards",
    "new_interest.evaluation.evaluate_text_label_ranking",
    "new_interest.evaluation.merge_text_label_ranking_shards",
    "new_interest.evaluation.evaluate_wang_stage_a_generation",
    "new_interest.evaluation.merge_wang_stage_a_generation_shards",
    "new_interest.evaluation.candidate_metrics",
    "new_interest.experiments.statistical_future_interest",
    "new_interest.experiments.embedding_future_interest",
    "new_interest.experiments.multilabel_future_interest",
    "new_interest.experiments.run_future_interest_baselines",
    "new_interest.experiments.report_future_interest_baselines",
    "new_interest.experiments.analyze_sft_training_similarity",
    "new_interest.experiments.report_sft_training_similarity",
    "new_interest.experiments.wang_stage_a",
    "new_interest.experiments.backbone_scaling",
    "new_interest.experiments.backbone_bundle",
    "new_interest.experiments.backbone_four_gpu",
    "new_interest.experiments.backbone_recovery",
    "new_interest.experiments.old_new_interest",
    "new_interest.experiments.old_new_interest_g4",
    "new_interest.experiments.cir_warmup",
    "new_interest.experiments.history_token",
    "new_interest.experiments.token_joint_alignment",
    "new_interest.experiments.run_serial_experiments",
)


def test_reorganized_modules_import() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_console_script_targets_resolve() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert set(scripts) == {
        "build-new-interest-dataset",
        "build-dynamic-new-interest-dataset",
        "build-dynamic-generative-dataset",
        "build-future-interest-benchmark",
        "build-static-user-disjoint-candidate-selection",
        "build-static-candidate-pilot",
        "build-semantic-id-token-sft-dataset",
        "build-mera-relations",
        "build-mera-semantic-id-dataset",
        "build-generative-new-interest-dataset",
        "build-sft-subsets",
        "build-candidate-selection",
        "build-grpo-datasets",
        "build-grpo-acceptable-targets",
        "generate-cot-teacher",
        "build-cot-sft-datasets",
        "build-cot-sft-rehearsal",
        "evaluate-cot-accuracy",
        "aggregate-cot-accuracy",
        "compare-paired-accuracy",
        "select-cot-v2-preview",
        "generate-cot-teacher-v2",
        "render-cot-v2-preview",
        "evaluate-future-interest",
        "run-future-interest-baselines",
        "report-future-interest-baselines",
        "train-semantic-history-linear-ranker",
        "analyze-sft-training-similarity",
        "report-sft-training-similarity",
        "evaluate-static-candidate-baselines",
        "run-static-candidate-condition",
        "finalize-static-candidate-pilots",
        "report-static-candidate-results",
        "evaluate-id-token-ranking",
        "merge-id-token-shards",
    }
    for target in scripts.values():
        module_name, attribute = target.split(":", maxsplit=1)
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_portable_path_prefers_repository_relative_paths(tmp_path: Path) -> None:
    project_file = Path.cwd() / "data" / "example.parquet"
    assert portable_path(project_file) == "data/example.parquet"
    assert portable_path(tmp_path) == str(tmp_path.resolve())
