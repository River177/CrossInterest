from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from new_interest.training.mera_regularization import (
    MeRARegularizer,
    load_mera_relations,
)
from new_interest.training.semantic_id_tokens import (
    InterestTokenEntry,
    InterestTokenSpec,
)
from new_interest.training.train_sft import make_mera_trainer


class FakeConfig:
    tie_word_embeddings = False


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.config = FakeConfig()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _write_relations(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "format_version": "mera-gnip-relations-v1",
                "labels": ["Alpha", "Beta", "Gamma", "Zulu"],
                "interest_tokens": [
                    "<interest_000>",
                    "<interest_001>",
                    "<interest_002>",
                    "<interest_003>",
                ],
                "parents": ["P1", "P1", "P2", "P2"],
                "cooccurrence_counts": [
                    [10, 5, 1, 1],
                    [5, 10, 1, 1],
                    [1, 1, 10, 5],
                    [1, 1, 5, 10],
                ],
                "cooccurrence_contract": {"source_split": "train-only"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _semantic_spec(tmp_path: Path) -> InterestTokenSpec:
    labels = ("Alpha", "Beta", "Gamma", "Zulu")
    entries = tuple(
        InterestTokenEntry(
            interest_index=index,
            interest_token=f"<interest_{index:03d}>",
            target_subcategory=label,
            label_token_ids=(index,),
            token_id=index + 4,
        )
        for index, label in enumerate(labels)
    )
    return InterestTokenSpec(
        mapping_file=tmp_path / "mapping.parquet",
        mapping_sha256="mapping-sha",
        original_vocabulary_size=4,
        resized_vocabulary_size=8,
        entries=entries,
    )


def test_mera_regularizer_is_step_deterministic_and_backpropagates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "relations.json"
    _write_relations(path)
    regularizer = MeRARegularizer(
        load_mera_relations(path),
        _semantic_spec(tmp_path),
        structure_weight=0.05,
        collaborative_weight=0.1,
        negative_samples=1,
        cooccurrence_threshold=0.05,
        seed=17,
        min_collaborative_anchor_fraction=1.0,
    )
    model = FakeModel()

    first = regularizer.compute(model, step=3)
    second = regularizer.compute(model, step=3)
    torch.testing.assert_close(first["structure_loss"], second["structure_loss"])
    torch.testing.assert_close(
        first["collaborative_loss"], second["collaborative_loss"]
    )
    assert torch.isfinite(first["weighted_loss"])
    first["weighted_loss"].backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_output_embeddings().weight.grad is not None
    assert float(model.get_input_embeddings().weight.grad.norm()) > 0
    assert float(model.get_output_embeddings().weight.grad.norm()) > 0

    audit = regularizer.audit()
    assert audit["structure_eligible_anchors"] == 4
    assert audit["collaborative_eligible_anchors"] == 4
    assert audit["regularized_modules"] == ["input_embeddings", "lm_head"]
    assert regularizer.snapshot()["calls"] == 2


def test_mera_regularizer_fails_closed_on_sparse_collaborative_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "relations.json"
    _write_relations(path)
    with pytest.raises(ValueError, match="coverage gate failed"):
        MeRARegularizer(
            load_mera_relations(path),
            _semantic_spec(tmp_path),
            structure_weight=0.05,
            collaborative_weight=0.1,
            negative_samples=1,
            cooccurrence_threshold=0.9,
            min_collaborative_anchor_fraction=1.0,
        )


def test_load_mera_relations_rejects_non_square_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "relations.json"
    _write_relations(path)
    payload = json.loads(path.read_text())
    payload["cooccurrence_counts"][0].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="square"):
        load_mera_relations(path)


def test_mera_trainer_adds_relation_loss_to_causal_lm_loss() -> None:
    class BaseTrainer:
        def __init__(self) -> None:
            self.state = SimpleNamespace(global_step=7)

    class Model(torch.nn.Module):
        def forward(self, input_ids, labels):
            del input_ids, labels
            logits = torch.zeros(1, 2, 3, requires_grad=True)
            return SimpleNamespace(logits=logits, loss=logits.sum() + 1.25)

    class Regularizer:
        def __init__(self) -> None:
            self.steps = []

        def compute(self, model, *, step):
            del model
            self.steps.append(step)
            return {"weighted_loss": torch.tensor(0.75)}

    regularizer = Regularizer()
    trainer_class = make_mera_trainer(
        BaseTrainer,
        regularizer,
        loss_reduction="token_mean",
    )
    trainer = trainer_class()
    loss = trainer.compute_loss(
        Model(),
        {
            "input_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[1, 2]]),
        },
    )
    assert float(loss.detach()) == pytest.approx(2.0)
    assert regularizer.steps == [7]


def test_mera_regularizer_unwraps_ddp_style_model(tmp_path: Path) -> None:
    path = tmp_path / "relations.json"
    _write_relations(path)
    regularizer = MeRARegularizer(
        load_mera_relations(path),
        _semantic_spec(tmp_path),
        structure_weight=0.05,
        collaborative_weight=0.1,
        negative_samples=1,
        cooccurrence_threshold=0.05,
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    losses = regularizer.compute(Wrapper(FakeModel()), step=0)
    assert torch.isfinite(losses["weighted_loss"])
