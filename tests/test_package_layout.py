from __future__ import annotations

import importlib
from pathlib import Path

import tomllib

from new_interest.infrastructure.paths import portable_path


MODULES = (
    "new_interest.data.build_dataset",
    "new_interest.data.build_generative_dataset",
    "new_interest.data.build_sft_subsets",
    "new_interest.data.build_candidate_selection",
    "new_interest.data.build_grpo_dataset",
    "new_interest.training.sft_data",
    "new_interest.training.grpo_reward",
    "new_interest.evaluation.metrics",
    "new_interest.evaluation.candidate_metrics",
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
        "build-generative-new-interest-dataset",
        "build-sft-subsets",
        "build-candidate-selection",
        "build-grpo-datasets",
    }
    for target in scripts.values():
        module_name, attribute = target.split(":", maxsplit=1)
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_portable_path_prefers_repository_relative_paths(tmp_path: Path) -> None:
    project_file = Path.cwd() / "data" / "example.parquet"
    assert portable_path(project_file) == "data/example.parquet"
    assert portable_path(tmp_path) == str(tmp_path.resolve())
