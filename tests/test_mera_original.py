from __future__ import annotations

import json
import random
from functools import wraps
from pathlib import Path

import numpy as np
import pytest
import yaml

from new_interest.experiments.mera_original_core import (
    MeRAOriginalRegularizer,
    audit_dataset,
    expected_optimizer_steps,
    inference_text,
    load_resolved_config,
    precision_recall_at_k,
    training_prompt,
)
from new_interest.training.train_mera_original import (
    remove_accelerate_fp32_forward_wrapper,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mera_original/experiment_matrix_v1.yaml"


@pytest.mark.parametrize(
    ("dataset", "rows", "categories", "code_anchors", "paper_anchors"),
    [
        ("arts", 2524, 143, 117, 131),
        ("grocery", 4373, 172, 138, 156),
    ],
)
def test_frozen_source_and_author_results_reproduce(
    dataset: str,
    rows: int,
    categories: int,
    code_anchors: int,
    paper_anchors: int,
) -> None:
    config = load_resolved_config(CONFIG, dataset, "code-faithful")
    audit = audit_dataset(config)
    assert audit["splits"]["test"]["rows"] == rows
    assert audit["categories"] == categories
    assert audit["regularizer_coverage"]["code_collaborative_anchors"] == code_anchors
    assert audit["regularizer_coverage"]["paper_collaborative_anchors"] == paper_anchors
    manifest = json.loads(config.source_manifest_path.read_text())
    expected = manifest["datasets"][dataset]["author_predictions"]
    assert audit["author_metrics"]["precision_at_k_rounded"] == expected["precision_at_k"]
    assert audit["author_metrics"]["recall_at_k_rounded"] == expected["recall_at_k"]


def test_matrix_resolves_both_contracts_without_silent_overrides() -> None:
    arts_code = load_resolved_config(CONFIG, "arts", "code-faithful")
    grocery_code = load_resolved_config(CONFIG, "grocery", "code-faithful")
    arts_paper = load_resolved_config(CONFIG, "arts", "paper-spec")
    assert arts_code.regularizer == {
        "regularizer_semantics": "released-code",
        "cooccurrence_threshold": None,
        "structure_weight": 0.01,
        "collaborative_weight": 0.01,
        "negative_samples": 5,
    }
    assert grocery_code.regularizer["structure_weight"] == 0.1
    assert grocery_code.regularizer["negative_samples"] == 10
    assert arts_paper.regularizer == {
        "regularizer_semantics": "paper-equation",
        "cooccurrence_threshold": 0.05,
        "structure_weight": 0.05,
        "collaborative_weight": 0.1,
        "negative_samples": 10,
    }
    assert expected_optimizer_steps(20186, arts_code.training) == 800
    assert expected_optimizer_steps(34981, grocery_code.training) == 1380


def test_prompt_and_inference_views_match_upstream_contract() -> None:
    row = {
        "instruction": "Predict a new category.",
        "input": 'History: "A", "B"\n',
        "output": '"C"',
    }
    prompt = training_prompt(row)
    assert "### Instruction:\nPredict a new category." in prompt
    assert "### Response:\n\"C\"" in prompt
    assert inference_text(row) == '"A""B"'


def test_multi_positive_precision_recall() -> None:
    result = precision_recall_at_k(
        [["A", "B", "C", "D"], ["D", "C", "B", "A"]],
        [["A", "C"], ["A"]],
    )
    assert result["precision_at_k"] == [0.5, 0.25, pytest.approx(1 / 3), 0.375]
    assert result["recall_at_k"] == [0.25, 0.25, 0.5, 1.0]


def _toy_regularizer(semantics: str) -> MeRAOriginalRegularizer:
    categories = ("A", "B", "C", "D")
    tree = {"P1": ["A", "B"], "P2": ["C", "D"]}
    matrix = np.array(
        [
            [0, 20, 3, 1],
            [20, 0, 2, 1],
            [3, 2, 0, 20],
            [1, 1, 20, 0],
        ],
        dtype=np.float64,
    )
    contract = {
        "regularizer_semantics": semantics,
        "structure_weight": 0.05,
        "collaborative_weight": 0.1,
        "negative_samples": 1,
        "cooccurrence_threshold": None if semantics == "released-code" else 0.05,
    }
    return MeRAOriginalRegularizer(
        token_ids=(4, 5, 6, 7),
        categories=categories,
        category_tree=tree,
        cooccurrence_matrix=matrix,
        contract=contract,
    )


@pytest.mark.parametrize("semantics", ["released-code", "paper-equation"])
def test_regularizer_has_finite_input_and_output_gradients(semantics: str) -> None:
    torch = pytest.importorskip("torch")

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Embedding(8, 6)
            self.output = torch.nn.Linear(6, 8, bias=False)

        def get_input_embeddings(self):
            return self.input

        def get_output_embeddings(self):
            return self.output

    random.seed(2024)
    torch.manual_seed(2024)
    model = ToyModel()
    regularizer = _toy_regularizer(semantics)
    losses = regularizer.compute(model)
    assert torch.isfinite(losses["structure_loss"])
    assert torch.isfinite(losses["collaborative_loss"])
    losses["weighted_loss"].backward()
    assert model.input.weight.grad[4:].norm() > 0
    assert model.output.weight.grad[4:].norm() > 0
    if semantics == "paper-equation":
        assert regularizer.audit()["structure_anchors"] == 4
    else:
        assert regularizer.audit()["structure_anchors"] == 2


def test_amlt_configs_have_four_independent_single_gpu_jobs() -> None:
    expected = {
        "arts-code-smoke",
        "arts-paper-smoke",
        "grocery-code-smoke",
        "grocery-paper-smoke",
    }
    smoke = yaml.safe_load(
        (ROOT / "configs/amlt/archive/2026-10-26/mera_original_qwen2_four_condition_smoke_0902_v1.yaml").read_text()
    )
    assert {job["name"] for job in smoke["jobs"]} == expected
    assert all(job["sku"] == "80G1-A100" for job in smoke["jobs"])
    assert all(job["sla_tier"] == "Standard" for job in smoke["jobs"])
    full = yaml.safe_load(
        (ROOT / "configs/amlt/archive/2026-10-26/mera_original_qwen2_full_0902_v1.yaml").read_text()
    )
    assert {job["name"] for job in full["jobs"]} == {
        value.replace("smoke", "full") for value in expected
    }
    assert all(job["sku"] == "80G1-A100" for job in full["jobs"])
    assert all(job["command"][0] == "set -euo pipefail" for job in full["jobs"])


def test_accelerate_forward_wrapper_removal_does_not_import_optional_backends() -> None:
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def forward(self, value):
            return value + 1

    model = Toy()
    model._original_forward = model.forward

    @wraps(Toy.forward)
    def wrapper(self, *args, **kwargs):
        return Toy.forward(self, *args, **kwargs).float()

    model.forward = wrapper.__get__(model, Toy)
    remove_accelerate_fp32_forward_wrapper(model)
    assert not hasattr(model, "_original_forward")
    assert model(torch.tensor(1)) == 2
