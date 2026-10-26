from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from new_interest.training.semantic_id_tokens import (
    InterestTokenSpec,
    resolve_token_modules,
)


RELATION_FORMAT_VERSION = "mera-gnip-relations-v1"
REGULARIZER_VERSION = "mera-paper-infonce-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MeRARelations:
    path: Path
    sha256: str
    labels: tuple[str, ...]
    interest_tokens: tuple[str, ...]
    parents: tuple[str, ...]
    cooccurrence_counts: tuple[tuple[int, ...], ...]
    cooccurrence_contract: dict[str, object]


def load_mera_relations(path: Path | str) -> MeRARelations:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != RELATION_FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported relation format "
            f"{payload.get('format_version')!r}"
        )
    labels = tuple(str(value) for value in payload["labels"])
    tokens = tuple(str(value) for value in payload["interest_tokens"])
    parents = tuple(str(value) for value in payload["parents"])
    counts = tuple(
        tuple(int(value) for value in row)
        for row in payload["cooccurrence_counts"]
    )
    size = len(labels)
    if size < 2 or len(set(labels)) != size:
        raise ValueError(f"{path}: labels must be unique and non-trivial")
    if len(tokens) != size or len(set(tokens)) != size:
        raise ValueError(f"{path}: interest token count/uniqueness mismatch")
    if len(parents) != size or any(not value for value in parents):
        raise ValueError(f"{path}: parent count/value mismatch")
    if len(counts) != size or any(len(row) != size for row in counts):
        raise ValueError(f"{path}: co-occurrence matrix must be square")
    if any(value < 0 for row in counts for value in row):
        raise ValueError(f"{path}: co-occurrence counts must be non-negative")
    if any(sum(row) <= 0 for row in counts):
        raise ValueError(f"{path}: every co-occurrence row must be non-empty")
    return MeRARelations(
        path=path,
        sha256=_sha256_file(path),
        labels=labels,
        interest_tokens=tokens,
        parents=parents,
        cooccurrence_counts=counts,
        cooccurrence_contract=dict(payload["cooccurrence_contract"]),
    )


class MeRARegularizer:
    """Paper-faithful structural and collaborative token-row InfoNCE.

    Sampling is stateless at the optimizer-step level. This preserves the
    paper's random positive/negative sampling while making interrupted AMLT
    training reproducible after a Trainer checkpoint resume.
    """

    def __init__(
        self,
        relations: MeRARelations,
        semantic_spec: InterestTokenSpec,
        *,
        structure_weight: float,
        collaborative_weight: float,
        negative_samples: int,
        cooccurrence_threshold: float,
        temperature: float = 1.0,
        seed: int = 42,
        min_collaborative_anchor_fraction: float = 0.0,
    ) -> None:
        import torch

        if tuple(
            entry.target_subcategory for entry in semantic_spec.entries
        ) != relations.labels:
            raise ValueError(
                "MeRA relation labels do not match Semantic-ID mapping order"
            )
        if tuple(
            entry.interest_token for entry in semantic_spec.entries
        ) != relations.interest_tokens:
            raise ValueError(
                "MeRA relation tokens do not match Semantic-ID mapping order"
            )
        if structure_weight < 0 or collaborative_weight < 0:
            raise ValueError("MeRA regularization weights must be non-negative")
        if structure_weight == 0 and collaborative_weight == 0:
            raise ValueError("At least one MeRA regularization must be enabled")
        if negative_samples <= 0:
            raise ValueError("negative_samples must be positive")
        if not 0 <= cooccurrence_threshold < 1:
            raise ValueError("cooccurrence_threshold must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= min_collaborative_anchor_fraction <= 1:
            raise ValueError(
                "min_collaborative_anchor_fraction must be in [0, 1]"
            )

        self.relations = relations
        self.semantic_spec = semantic_spec
        self.structure_weight = float(structure_weight)
        self.collaborative_weight = float(collaborative_weight)
        self.negative_samples = int(negative_samples)
        self.cooccurrence_threshold = float(cooccurrence_threshold)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.token_ids = tuple(int(value) for value in semantic_spec.token_ids)
        size = len(self.token_ids)

        parents = relations.parents
        structure_positives: list[tuple[int, ...]] = []
        structure_negatives: list[tuple[int, ...]] = []
        for anchor, parent in enumerate(parents):
            structure_positives.append(
                tuple(
                    candidate
                    for candidate, candidate_parent in enumerate(parents)
                    if candidate != anchor and candidate_parent == parent
                )
            )
            structure_negatives.append(
                tuple(
                    candidate
                    for candidate, candidate_parent in enumerate(parents)
                    if candidate_parent != parent
                )
            )
        self.structure_positives = tuple(structure_positives)
        self.structure_negatives = tuple(structure_negatives)
        self.structure_anchors = tuple(
            anchor
            for anchor, positives in enumerate(self.structure_positives)
            if positives
        )
        if self.structure_weight > 0 and not self.structure_anchors:
            raise ValueError("No interests have a structural sibling positive")
        for anchor in self.structure_anchors:
            if len(self.structure_negatives[anchor]) < self.negative_samples:
                raise ValueError(
                    f"Interest {anchor} has too few structural negatives"
                )

        raw_counts = torch.tensor(
            relations.cooccurrence_counts,
            dtype=torch.float64,
            device="cpu",
        )
        probabilities = raw_counts / raw_counts.sum(dim=1, keepdim=True)
        positive_weights = probabilities.clone()
        positive_weights[
            positive_weights < self.cooccurrence_threshold
        ] = 0
        positive_weights.fill_diagonal_(0)
        self.positive_weights = positive_weights
        self.negative_weights = 1.0 - positive_weights
        self.negative_weights.fill_diagonal_(0)
        self.collaborative_anchors = tuple(
            int(value)
            for value in torch.where(positive_weights.sum(dim=1) > 0)[0]
            .tolist()
        )
        if self.collaborative_weight > 0 and not self.collaborative_anchors:
            raise ValueError(
                "MeRA collaborative coverage gate failed: no anchors survive "
                "the co-occurrence threshold"
            )
        collaborative_fraction = len(self.collaborative_anchors) / size
        if collaborative_fraction < min_collaborative_anchor_fraction:
            raise ValueError(
                "MeRA collaborative coverage gate failed: "
                f"{len(self.collaborative_anchors)}/{size}="
                f"{collaborative_fraction:.6f} < "
                f"{min_collaborative_anchor_fraction:.6f}; lower the "
                "co-occurrence threshold or explicitly relax the gate"
            )
        for anchor in self.collaborative_anchors:
            possible = int((self.negative_weights[anchor] > 0).sum()) - 1
            if possible < self.negative_samples:
                raise ValueError(
                    f"Interest {anchor} has too few collaborative negatives"
                )

        self.calls = 0
        self.structure_loss_sum = 0.0
        self.collaborative_loss_sum = 0.0
        self.weighted_loss_sum = 0.0
        self.last_step: int | None = None

    def audit(self) -> dict[str, object]:
        size = len(self.token_ids)
        positive_counts = [
            int((row > 0).sum()) for row in self.positive_weights
        ]
        return {
            "regularizer_version": REGULARIZER_VERSION,
            "relations_file": str(self.relations.path),
            "relations_sha256": self.relations.sha256,
            "interest_count": size,
            "structure_weight": self.structure_weight,
            "collaborative_weight": self.collaborative_weight,
            "negative_samples": self.negative_samples,
            "cooccurrence_threshold": self.cooccurrence_threshold,
            "temperature": self.temperature,
            "seed": self.seed,
            "structure_eligible_anchors": len(self.structure_anchors),
            "structure_eligible_anchor_fraction": (
                len(self.structure_anchors) / size
            ),
            "collaborative_eligible_anchors": len(
                self.collaborative_anchors
            ),
            "collaborative_eligible_anchor_fraction": (
                len(self.collaborative_anchors) / size
            ),
            "collaborative_positive_edges": sum(positive_counts),
            "min_collaborative_positives_per_anchor": min(positive_counts),
            "max_collaborative_positives_per_anchor": max(positive_counts),
            "sampling": (
                "one sibling positive and N non-sibling negatives per "
                "eligible category; one co-occurrence positive and N "
                "complement-probability negatives per eligible category"
            ),
            "regularized_modules": ["input_embeddings", "lm_head"],
            "similarity": "unnormalized dot product",
            "resume_sampling": "stateless seed derived from optimizer step",
        }

    def snapshot(self) -> dict[str, object]:
        denominator = max(self.calls, 1)
        return {
            "calls": self.calls,
            "last_step": self.last_step,
            "mean_structure_loss": self.structure_loss_sum / denominator,
            "mean_collaborative_loss": (
                self.collaborative_loss_sum / denominator
            ),
            "mean_weighted_regularization_loss": (
                self.weighted_loss_sum / denominator
            ),
        }

    def _generator(self, step: int) -> Any:
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step) * 1_000_003)
        return generator

    def _sample_structure(self, generator: Any) -> tuple[Any, Any]:
        import torch

        samples: list[list[int]] = []
        for anchor in self.structure_anchors:
            positive_pool = self.structure_positives[anchor]
            positive_offset = int(
                torch.randint(
                    len(positive_pool),
                    (1,),
                    generator=generator,
                ).item()
            )
            positive = positive_pool[positive_offset]
            negative_pool = torch.tensor(
                self.structure_negatives[anchor],
                dtype=torch.long,
            )
            order = torch.randperm(
                len(negative_pool), generator=generator
            )[: self.negative_samples]
            negatives = negative_pool.index_select(0, order).tolist()
            samples.append([positive, *negatives])
        return (
            torch.tensor(self.structure_anchors, dtype=torch.long),
            torch.tensor(samples, dtype=torch.long),
        )

    def _sample_collaborative(self, generator: Any) -> tuple[Any, Any]:
        import torch

        samples: list[list[int]] = []
        for anchor in self.collaborative_anchors:
            positive = int(
                torch.multinomial(
                    self.positive_weights[anchor],
                    1,
                    replacement=False,
                    generator=generator,
                ).item()
            )
            negative_weights = self.negative_weights[anchor].clone()
            negative_weights[anchor] = 0
            negative_weights[positive] = 0
            negatives = torch.multinomial(
                negative_weights,
                self.negative_samples,
                replacement=False,
                generator=generator,
            ).tolist()
            samples.append([positive, *negatives])
        return (
            torch.tensor(self.collaborative_anchors, dtype=torch.long),
            torch.tensor(samples, dtype=torch.long),
        )

    def _infonce(
        self,
        category_rows: Any,
        anchors: Any,
        samples: Any,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        anchors = anchors.to(category_rows.device)
        samples = samples.to(category_rows.device)
        anchor_rows = category_rows.index_select(0, anchors)
        sample_rows = category_rows[samples]
        similarity = (
            sample_rows * anchor_rows.unsqueeze(1)
        ).sum(dim=-1) / self.temperature
        targets = similarity.new_zeros(similarity.shape[0], dtype=torch.long)
        return functional.cross_entropy(similarity.float(), targets)

    def compute(self, model: Any, *, step: int) -> dict[str, Any]:
        import torch

        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        modules = resolve_token_modules(unwrapped)
        device = modules["input_module"].weight.device
        token_ids = torch.tensor(
            self.token_ids,
            dtype=torch.long,
            device=device,
        )
        input_rows = modules["input_module"].weight.index_select(
            0, token_ids
        )
        output_rows = modules["output_module"].weight.index_select(
            0, token_ids.to(modules["output_module"].weight.device)
        )
        generator = self._generator(step)
        zero = input_rows.sum() * 0.0
        structure_loss = zero
        collaborative_loss = zero
        if self.structure_weight > 0:
            anchors, samples = self._sample_structure(generator)
            structure_loss = self._infonce(input_rows, anchors, samples)
            structure_loss = structure_loss + self._infonce(
                output_rows, anchors, samples
            )
        if self.collaborative_weight > 0:
            anchors, samples = self._sample_collaborative(generator)
            collaborative_loss = self._infonce(
                input_rows, anchors, samples
            )
            collaborative_loss = collaborative_loss + self._infonce(
                output_rows, anchors, samples
            )
        weighted = (
            self.structure_weight * structure_loss
            + self.collaborative_weight * collaborative_loss
        )
        self.calls += 1
        self.last_step = int(step)
        self.structure_loss_sum += float(structure_loss.detach().cpu())
        self.collaborative_loss_sum += float(
            collaborative_loss.detach().cpu()
        )
        self.weighted_loss_sum += float(weighted.detach().cpu())
        return {
            "structure_loss": structure_loss,
            "collaborative_loss": collaborative_loss,
            "weighted_loss": weighted,
        }


def build_mera_regularizer(
    config: dict[str, Any],
    semantic_spec: InterestTokenSpec,
) -> MeRARegularizer:
    relation_file = Path(str(config["mera_relations_file"]))
    relations = load_mera_relations(relation_file)
    return MeRARegularizer(
        relations,
        semantic_spec,
        structure_weight=float(config.get("mera_structure_weight", 0.05)),
        collaborative_weight=float(
            config.get("mera_collaborative_weight", 0.1)
        ),
        negative_samples=int(config.get("mera_negative_samples", 10)),
        cooccurrence_threshold=float(
            config.get("mera_cooccurrence_threshold", 0.05)
        ),
        temperature=float(config.get("mera_temperature", 1.0)),
        seed=int(config.get("seed", 42)),
        min_collaborative_anchor_fraction=float(
            config.get("mera_min_collaborative_anchor_fraction", 0.0)
        ),
    )
