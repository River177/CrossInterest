from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.experiments.future_interest_prediction import (
    PredictionWriter,
    load_vocabulary,
    rank_label_scores,
)


STATISTICAL_BASELINES = (
    "global_popularity",
    "conditional_frequency",
    "conditional_lift",
)


@dataclass(frozen=True)
class StatisticalFutureInterestModel:
    labels: tuple[str, ...]
    alpha: float
    train_users: int
    target_user_counts: dict[str, int]
    source_user_counts: dict[str, int]
    transition_user_counts: dict[str, dict[str, int]]

    def global_probability(self, target: str) -> float:
        return (
            self.target_user_counts.get(target, 0) + self.alpha
        ) / (
            self.train_users + self.alpha * len(self.labels)
        )

    def conditional_probability(self, source: str, target: str) -> float:
        return (
            self.transition_user_counts.get(source, {}).get(target, 0)
            + self.alpha
        ) / (
            self.source_user_counts.get(source, 0)
            + self.alpha * len(self.labels)
        )

    def score(
        self,
        history_labels: tuple[str, ...],
        target: str,
        *,
        baseline: str,
    ) -> float:
        if baseline == "global_popularity":
            return math.log(self.global_probability(target))
        if baseline not in {
            "conditional_frequency",
            "conditional_lift",
        }:
            raise ValueError(f"Unknown statistical baseline {baseline!r}")
        sources = tuple(dict.fromkeys(history_labels))
        if not sources:
            return math.log(self.global_probability(target))
        conditional = sum(
            self.conditional_probability(source, target)
            for source in sources
        ) / len(sources)
        if baseline == "conditional_frequency":
            return math.log(conditional)
        return math.log(conditional) - math.log(
            self.global_probability(target)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "alpha": self.alpha,
            "train_users": self.train_users,
            "target_user_counts": dict(sorted(self.target_user_counts.items())),
            "source_user_counts": dict(sorted(self.source_user_counts.items())),
            "transition_user_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(
                    self.transition_user_counts.items()
                )
            },
        }


def fit_statistical_future_interest(
    train_path: Path | str,
    vocabulary_path: Path | str,
    *,
    alpha: float = 1.0,
) -> StatisticalFutureInterestModel:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    train_path = Path(train_path)
    labels = load_vocabulary(vocabulary_path)
    label_set = set(labels)
    target_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    seen_users: set[str] = set()
    required = {
        "user_id",
        "history_unique_subcategories",
        "future_novel_subcategories",
    }
    schema = pq.ParquetFile(train_path).schema_arrow
    missing = required - set(schema.names)
    if missing:
        raise ValueError(
            f"{train_path}: missing columns {sorted(missing)}"
        )
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=sorted(required),
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in seen_users:
                raise ValueError(
                    f"{train_path}: duplicate user_id {user_id!r}"
                )
            seen_users.add(user_id)
            sources = {
                str(value)
                for value in row["history_unique_subcategories"]
            }
            targets = {
                str(value)
                for value in row["future_novel_subcategories"]
            }
            if not sources.issubset(label_set):
                raise ValueError(
                    f"{train_path}: {user_id} has invalid history label"
                )
            if not targets.issubset(label_set):
                raise ValueError(
                    f"{train_path}: {user_id} has invalid target label"
                )
            target_counts.update(targets)
            source_counts.update(sources)
            for source in sources:
                transitions[source].update(targets)
    if not seen_users:
        raise ValueError(f"{train_path}: no users")
    return StatisticalFutureInterestModel(
        labels=labels,
        alpha=alpha,
        train_users=len(seen_users),
        target_user_counts=dict(target_counts),
        source_user_counts=dict(source_counts),
        transition_user_counts={
            source: dict(counts)
            for source, counts in transitions.items()
        },
    )


def predict_statistical_future_interest(
    model: StatisticalFutureInterestModel,
    benchmark_path: Path | str,
    output_dir: Path | str,
    *,
    baselines: tuple[str, ...] = STATISTICAL_BASELINES,
    overwrite: bool = False,
) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Statistical full-vocabulary prediction requires NumPy"
        ) from error
    benchmark_path = Path(benchmark_path)
    output_dir = Path(output_dir)
    unknown = set(baselines) - set(STATISTICAL_BASELINES)
    if unknown:
        raise ValueError(f"Unknown statistical baselines {sorted(unknown)}")
    model_path = output_dir / "statistical_model.json"
    prediction_paths = {
        baseline: output_dir / baseline / "predictions.parquet"
        for baseline in baselines
    }
    existing = [
        path
        for path in [model_path, *prediction_paths.values()]
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Statistical outputs already exist: {existing}. "
            "Use overwrite=True."
        )
    required = {"user_id", "history_subcategories"}
    schema = pq.ParquetFile(benchmark_path).schema_arrow
    missing = required - set(schema.names)
    if missing:
        raise ValueError(
            f"{benchmark_path}: missing columns {sorted(missing)}"
        )
    label_to_index = {
        label: index for index, label in enumerate(model.labels)
    }
    global_probabilities = np.asarray(
        [
            model.global_probability(label)
            for label in model.labels
        ],
        dtype=np.float64,
    )
    global_scores = np.log(global_probabilities)
    conditional_probabilities = np.asarray(
        [
            [
                model.conditional_probability(source, target)
                for target in model.labels
            ]
            for source in model.labels
        ],
        dtype=np.float64,
    )
    seen_users: set[str] = set()
    with ExitStack() as stack:
        writers = {
            baseline: stack.enter_context(
                PredictionWriter(
                    prediction_paths[baseline],
                    overwrite=overwrite,
                )
            )
            for baseline in baselines
        }
        for batch in pq.ParquetFile(benchmark_path).iter_batches(
            columns=sorted(required),
            batch_size=8_192,
        ):
            for row in batch.to_pylist():
                user_id = str(row["user_id"])
                if user_id in seen_users:
                    raise ValueError(
                        f"{benchmark_path}: duplicate user_id {user_id!r}"
                    )
                seen_users.add(user_id)
                history = tuple(
                    str(value) for value in row["history_subcategories"]
                )
                for baseline, writer in writers.items():
                    if baseline == "global_popularity":
                        score_vector = global_scores
                    else:
                        source_indices = [
                            label_to_index[label]
                            for label in dict.fromkeys(history)
                            if label in label_to_index
                        ]
                        if source_indices:
                            conditional = conditional_probabilities[
                                source_indices
                            ].mean(axis=0)
                            score_vector = np.log(conditional)
                            if baseline == "conditional_lift":
                                score_vector = (
                                    score_vector - global_scores
                                )
                        else:
                            score_vector = global_scores
                    scores = {
                        label: float(score_vector[index])
                        for index, label in enumerate(model.labels)
                    }
                    ranked = rank_label_scores(model.labels, scores)
                    writer.append(
                        user_id=user_id,
                        ranked_labels=ranked["ranked_labels"],
                        ranked_scores=ranked["ranked_scores"],
                    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(model.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "users": len(seen_users),
        "baselines": list(baselines),
        "prediction_files": {
            baseline: str(prediction_paths[baseline])
            for baseline in baselines
        },
        "model_path": str(model_path),
    }
