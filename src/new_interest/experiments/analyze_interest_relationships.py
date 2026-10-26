from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "history-new-interest-analysis-v1"
DEFAULT_CUTOFFS = (1, 5, 10, 20)
RECENT_WINDOWS = (3, 5, 10)
RELATION_NAMES = (
    "last_interest",
    "recent5_interests",
    "full_history_interests",
)


@dataclass(frozen=True)
class Vocabulary:
    labels: tuple[str, ...]
    parents: tuple[str, ...]
    interest_to_id: dict[tuple[str, str], int]
    parent_to_id: dict[str, int]
    interest_parent_ids: tuple[int, ...]


@dataclass(frozen=True)
class Target:
    interest_id: int
    parent_id: int


@dataclass(frozen=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime
    history_ids: tuple[int, ...]
    history_parent_ids: tuple[int, ...]
    targets: tuple[Target, ...]


@dataclass
class IntegerDistribution:
    counts: Counter[int] = field(default_factory=Counter)
    total: int = 0
    weighted_sum: int = 0

    def add(self, value: int) -> None:
        self.counts[value] += 1
        self.total += 1
        self.weighted_sum += value

    def summary(self) -> dict[str, float | int | None]:
        if not self.total:
            return {
                "count": 0,
                "min": None,
                "p50": None,
                "p90": None,
                "p99": None,
                "max": None,
                "mean": None,
            }
        return {
            "count": self.total,
            "min": min(self.counts),
            "p50": self._quantile(0.50),
            "p90": self._quantile(0.90),
            "p99": self._quantile(0.99),
            "max": max(self.counts),
            "mean": self.weighted_sum / self.total,
        }

    def _quantile(self, fraction: float) -> int:
        rank = max(1, math.ceil(self.total * fraction))
        cumulative = 0
        for value, count in sorted(self.counts.items()):
            cumulative += count
            if cumulative >= rank:
                return value
        raise AssertionError("Unreachable quantile")


def _new_relation_store() -> dict[str, dict[int, Counter[int]]]:
    return {
        name: defaultdict(Counter)
        for name in RELATION_NAMES
    }


@dataclass
class SplitStats:
    split: str
    vocabulary: Vocabulary
    rows: int = 0
    impressions: int = 0
    users: set[str] = field(default_factory=set)
    user_impression_counts: Counter[str] = field(default_factory=Counter)
    user_first_histories: dict[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    users_with_history_variation: set[str] = field(default_factory=set)
    user_target_counts: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    history_lengths: IntegerDistribution = field(
        default_factory=IntegerDistribution
    )
    unique_interest_counts: IntegerDistribution = field(
        default_factory=IntegerDistribution
    )
    unique_parent_counts: IntegerDistribution = field(
        default_factory=IntegerDistribution
    )
    targets_per_impression: IntegerDistribution = field(
        default_factory=IntegerDistribution
    )
    histories_with_repetition: int = 0
    histories_with_adjacent_repetition: int = 0
    repeat_fraction_sum: float = 0.0
    dominant_interest_share_sum: float = 0.0
    target_counts: Counter[int] = field(default_factory=Counter)
    target_counts_by_date: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    target_parent_counts: Counter[int] = field(default_factory=Counter)
    exact_target_history_overlap: int = 0
    relation_counts: dict[str, dict[int, Counter[int]]] = field(
        default_factory=_new_relation_store
    )
    last_parent_to_target: dict[int, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    last_parent_to_target_parent: dict[int, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    parent_source_counts: dict[str, Counter[int]] = field(
        default_factory=lambda: {
            "any": Counter(),
            "last": Counter(),
            **{f"recent_{window}": Counter() for window in RECENT_WINDOWS},
        }
    )
    same_parent_counts: Counter[str] = field(default_factory=Counter)

    def add(self, impression: Impression) -> None:
        target_count = len(impression.targets)
        history_ids = impression.history_ids
        history_parent_ids = impression.history_parent_ids
        history_counts = Counter(history_ids)
        unique_history = set(history_ids)
        unique_parents = set(history_parent_ids)

        self.rows += target_count
        self.impressions += 1
        self.users.add(impression.user_id)
        self.user_impression_counts[impression.user_id] += 1
        first_history = self.user_first_histories.setdefault(
            impression.user_id,
            history_ids,
        )
        if first_history != history_ids:
            self.users_with_history_variation.add(impression.user_id)
        self.timestamp_min = (
            impression.timestamp
            if self.timestamp_min is None
            else min(self.timestamp_min, impression.timestamp)
        )
        self.timestamp_max = (
            impression.timestamp
            if self.timestamp_max is None
            else max(self.timestamp_max, impression.timestamp)
        )
        self.history_lengths.add(len(history_ids))
        self.unique_interest_counts.add(len(unique_history))
        self.unique_parent_counts.add(len(unique_parents))
        self.targets_per_impression.add(target_count)

        repeats = len(history_ids) - len(unique_history)
        self.histories_with_repetition += int(repeats > 0)
        self.repeat_fraction_sum += repeats / len(history_ids)
        self.dominant_interest_share_sum += (
            max(history_counts.values()) / len(history_ids)
        )
        self.histories_with_adjacent_repetition += int(
            any(
                left == right
                for left, right in zip(history_ids, history_ids[1:])
            )
        )

        source_parent_sets = {
            "any": unique_parents,
            "last": {history_parent_ids[-1]},
            **{
                f"recent_{window}": set(history_parent_ids[-window:])
                for window in RECENT_WINDOWS
            },
        }
        for relation_name, parent_ids in source_parent_sets.items():
            for parent_id in parent_ids:
                self.parent_source_counts[relation_name][parent_id] += (
                    target_count
                )

        recent5 = set(history_ids[-5:])
        last_interest = history_ids[-1]
        last_parent = history_parent_ids[-1]
        for target in impression.targets:
            target_id = target.interest_id
            target_parent = target.parent_id
            self.target_counts[target_id] += 1
            self.target_counts_by_date[
                impression.timestamp.date().isoformat()
            ][target_id] += 1
            self.target_parent_counts[target_parent] += 1
            self.user_target_counts[impression.user_id][target_id] += 1
            self.exact_target_history_overlap += int(
                target_id in unique_history
            )
            self.relation_counts["last_interest"][last_interest][
                target_id
            ] += 1
            for source_id in recent5:
                self.relation_counts["recent5_interests"][source_id][
                    target_id
                ] += 1
            for source_id in unique_history:
                self.relation_counts["full_history_interests"][source_id][
                    target_id
                ] += 1
            self.last_parent_to_target[last_parent][target_id] += 1
            self.last_parent_to_target_parent[last_parent][target_parent] += 1
            for relation_name, parent_ids in source_parent_sets.items():
                if target_parent in parent_ids:
                    self.same_parent_counts[relation_name] += 1

    def summary(self) -> dict[str, object]:
        if not self.impressions or not self.rows:
            raise ValueError(f"{self.split}: no eligible impressions")
        unique_user_target_pairs = sum(
            len(counts) for counts in self.user_target_counts.values()
        )
        users_with_repeated_targets = sum(
            any(count > 1 for count in counts.values())
            for counts in self.user_target_counts.values()
        )
        return {
            "target_rows": self.rows,
            "impressions": self.impressions,
            "unique_users": len(self.users),
            "users_with_multiple_eligible_impressions": sum(
                count > 1 for count in self.user_impression_counts.values()
            ),
            "users_with_multiple_history_sequences": len(
                self.users_with_history_variation
            ),
            "unique_user_target_pairs": unique_user_target_pairs,
            "repeated_user_target_memberships": (
                self.rows - unique_user_target_pairs
            ),
            "repeated_user_target_membership_rate": (
                (self.rows - unique_user_target_pairs) / self.rows
            ),
            "users_with_repeated_target_memberships": (
                users_with_repeated_targets
            ),
            "timestamp_min": _isoformat(self.timestamp_min),
            "timestamp_max": _isoformat(self.timestamp_max),
            "rows_per_impression": self.rows / self.impressions,
            "multi_target_impressions": sum(
                count
                for target_count, count
                in self.targets_per_impression.counts.items()
                if target_count > 1
            ),
            "multi_target_impression_rate": sum(
                count
                for target_count, count
                in self.targets_per_impression.counts.items()
                if target_count > 1
            ) / self.impressions,
            "history": {
                "length": self.history_lengths.summary(),
                "unique_interests": self.unique_interest_counts.summary(),
                "unique_parents": self.unique_parent_counts.summary(),
                "histories_with_repetition": self.histories_with_repetition,
                "histories_with_repetition_rate": (
                    self.histories_with_repetition / self.impressions
                ),
                "histories_with_adjacent_repetition_rate": (
                    self.histories_with_adjacent_repetition
                    / self.impressions
                ),
                "mean_repeat_fraction": (
                    self.repeat_fraction_sum / self.impressions
                ),
                "mean_dominant_interest_share": (
                    self.dominant_interest_share_sum / self.impressions
                ),
            },
            "novelty": {
                "exact_target_history_overlap_count": (
                    self.exact_target_history_overlap
                ),
                "exact_target_history_overlap_rate": (
                    self.exact_target_history_overlap / self.rows
                ),
            },
            "taxonomic_relationship": self._taxonomic_relationship(),
            "target_distribution": _target_distribution(
                self.target_counts,
                self.vocabulary,
            ),
            "target_distribution_by_date": {
                date: _target_distribution(counts, self.vocabulary)
                for date, counts in sorted(self.target_counts_by_date.items())
            },
            "relations": {
                name: _relation_summary(rows)
                for name, rows in self.relation_counts.items()
            },
            "last_parent_to_target_parent": _relation_summary(
                self.last_parent_to_target_parent
            ),
        }

    def _taxonomic_relationship(self) -> dict[str, object]:
        relationships = {}
        for relation_name, source_counts in self.parent_source_counts.items():
            observed_count = self.same_parent_counts[relation_name]
            observed_rate = observed_count / self.rows
            expected_rate = sum(
                (source_counts[parent_id] / self.rows)
                * (self.target_parent_counts[parent_id] / self.rows)
                for parent_id in range(len(self.vocabulary.parents))
            )
            relationships[relation_name] = {
                "observed_count": observed_count,
                "observed_rate": observed_rate,
                "independence_expected_rate": expected_rate,
                "difference": observed_rate - expected_rate,
                "lift": (
                    observed_rate / expected_rate
                    if expected_rate
                    else None
                ),
            }
        return relationships


@dataclass
class RankingMetrics:
    cutoffs: tuple[int, ...]
    impressions: int = 0
    target_memberships: int = 0
    recall_sums: Counter[int] = field(default_factory=Counter)
    ndcg_sums: Counter[int] = field(default_factory=Counter)
    hit_counts: Counter[int] = field(default_factory=Counter)
    target_hit_counts: Counter[int] = field(default_factory=Counter)
    reciprocal_rank_sum: float = 0.0

    def add(self, ranking: list[int], targets: Iterable[int]) -> None:
        acceptable = set(targets)
        if not acceptable:
            return
        self.impressions += 1
        self.target_memberships += len(acceptable)
        rank_by_target = {
            target_id: rank
            for rank, target_id in enumerate(ranking, start=1)
            if target_id in acceptable
        }
        first_rank = min(rank_by_target.values(), default=None)
        if first_rank is not None:
            self.reciprocal_rank_sum += 1.0 / first_rank
        for cutoff in self.cutoffs:
            hits = sum(
                rank <= cutoff for rank in rank_by_target.values()
            )
            self.target_hit_counts[cutoff] += hits
            self.hit_counts[cutoff] += int(hits > 0)
            self.recall_sums[cutoff] += hits / len(acceptable)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in rank_by_target.values()
                if rank <= cutoff
            )
            ideal_hits = min(cutoff, len(acceptable))
            idcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )
            self.ndcg_sums[cutoff] += dcg / idcg

    def as_dict(self) -> dict[str, object]:
        if not self.impressions:
            return {"impressions": 0, "target_memberships": 0}
        return {
            "impressions": self.impressions,
            "target_memberships": self.target_memberships,
            "mrr": self.reciprocal_rank_sum / self.impressions,
            **{
                f"recall_at_{cutoff}": (
                    self.recall_sums[cutoff] / self.impressions
                )
                for cutoff in self.cutoffs
            },
            **{
                f"ndcg_at_{cutoff}": (
                    self.ndcg_sums[cutoff] / self.impressions
                )
                for cutoff in self.cutoffs
            },
            **{
                f"hit_at_{cutoff}": (
                    self.hit_counts[cutoff] / self.impressions
                )
                for cutoff in self.cutoffs
            },
            **{
                f"target_membership_hit_at_{cutoff}": (
                    self.target_hit_counts[cutoff]
                    / self.target_memberships
                )
                for cutoff in self.cutoffs
            },
        }


class BaselineEvaluator:
    def __init__(
        self,
        train: SplitStats,
        allowed_targets: set[int],
        cutoffs: tuple[int, ...],
    ) -> None:
        self.train = train
        self.vocabulary = train.vocabulary
        self.cutoffs = cutoffs
        self.train_targets = {
            target_id
            for target_id in allowed_targets
            if train.target_counts[target_id] > 0
        }
        self.global_order = self._sort_by_counts(train.target_counts)
        self.global_rank = {
            target_id: rank
            for rank, target_id in enumerate(self.global_order)
        }
        all_train_targets = {
            target_id
            for target_id, count in train.target_counts.items()
            if count > 0
        }
        self.candidate_global_order = sorted(
            all_train_targets,
            key=lambda target_id: (
                -train.target_counts[target_id],
                self.vocabulary.labels[target_id],
            ),
        )
        self.candidate_global_rank = {
            target_id: rank
            for rank, target_id in enumerate(self.candidate_global_order)
        }
        latest_train_date = max(train.target_counts_by_date)
        self.latest_train_date = latest_train_date
        self.latest_day_counts = train.target_counts_by_date[latest_train_date]
        self.latest_day_order = self._sort_by_counts(self.latest_day_counts)
        self.conditional_probabilities = {
            relation_name: _conditional_probabilities(rows)
            for relation_name, rows in train.relation_counts.items()
        }
        train_target_total = sum(train.target_counts.values())
        target_probabilities = {
            target_id: train.target_counts[target_id] / train_target_total
            for target_id in self.train_targets
        }
        self.last_interest_lift = {
            source_id: {
                target_id: probability / target_probabilities[target_id]
                for target_id, probability in probabilities.items()
                if target_id in target_probabilities
            }
            for source_id, probabilities
            in self.conditional_probabilities["last_interest"].items()
        }
        self.metrics = {
            baseline: {
                slice_name: RankingMetrics(cutoffs)
                for slice_name in (
                    "all",
                    "seen_user",
                    "unseen_user",
                    "same_parent_target",
                    "new_parent_target",
                )
            }
            for baseline in (
                "global_popularity",
                "latest_train_day_popularity",
                "user_train_target_frequency",
                "last_interest_conditional",
                "last_interest_lift",
                "last_parent_conditional",
                "recent5_union_conditional",
                "full_history_union_conditional",
            )
        }

    def evaluate(self, impression: Impression) -> None:
        excluded = set(impression.history_ids)
        rankings = {
            "global_popularity": self._filter(self.global_order, excluded),
            "latest_train_day_popularity": self._filter(
                self.latest_day_order,
                excluded,
            ),
            "user_train_target_frequency": self._count_order(
                self.train.user_target_counts.get(
                    impression.user_id,
                    Counter(),
                ),
                excluded,
            ),
            "last_interest_conditional": self._count_order(
                self.train.relation_counts["last_interest"].get(
                    impression.history_ids[-1], Counter()
                ),
                excluded,
            ),
            "last_interest_lift": self._score_order(
                self.last_interest_lift.get(
                    impression.history_ids[-1],
                    {},
                ),
                excluded,
            ),
            "last_parent_conditional": self._count_order(
                self.train.last_parent_to_target.get(
                    impression.history_parent_ids[-1], Counter()
                ),
                excluded,
            ),
            "recent5_union_conditional": self._union_order(
                "recent5_interests",
                set(impression.history_ids[-5:]),
                excluded,
            ),
            "full_history_union_conditional": self._union_order(
                "full_history_interests",
                set(impression.history_ids),
                excluded,
            ),
        }
        targets = [target.interest_id for target in impression.targets]
        parent_history = set(impression.history_parent_ids)
        same_parent_targets = [
            target.interest_id
            for target in impression.targets
            if target.parent_id in parent_history
        ]
        new_parent_targets = [
            target.interest_id
            for target in impression.targets
            if target.parent_id not in parent_history
        ]
        user_slice = (
            "seen_user"
            if impression.user_id in self.train.users
            else "unseen_user"
        )
        for baseline, ranking in rankings.items():
            self.metrics[baseline]["all"].add(ranking, targets)
            self.metrics[baseline][user_slice].add(ranking, targets)
            self.metrics[baseline]["same_parent_target"].add(
                ranking,
                same_parent_targets,
            )
            self.metrics[baseline]["new_parent_target"].add(
                ranking,
                new_parent_targets,
            )

    def summary(self) -> dict[str, object]:
        return {
            baseline: {
                slice_name: metrics.as_dict()
                for slice_name, metrics in slices.items()
            }
            for baseline, slices in self.metrics.items()
        }

    def candidate_rankings(
        self,
        *,
        user_id: str,
        history_ids: tuple[int, ...],
        candidates: list[int],
    ) -> dict[str, list[int]]:
        candidate_set = set(candidates)
        relations = self.train.relation_counts
        return {
            "global_popularity": self._candidate_count_order(
                self.train.target_counts,
                candidate_set,
            ),
            "latest_train_day_popularity": self._candidate_count_order(
                self.latest_day_counts,
                candidate_set,
            ),
            "user_train_target_frequency": self._candidate_count_order(
                self.train.user_target_counts.get(user_id, Counter()),
                candidate_set,
            ),
            "last_interest_conditional": self._candidate_count_order(
                relations["last_interest"].get(
                    history_ids[-1], Counter()
                ),
                candidate_set,
            ),
            "last_interest_lift": self._candidate_score_order(
                self.last_interest_lift.get(history_ids[-1], {}),
                candidate_set,
            ),
            "last_parent_conditional": self._candidate_count_order(
                self.train.last_parent_to_target.get(
                    self.vocabulary.interest_parent_ids[history_ids[-1]],
                    Counter(),
                ),
                candidate_set,
            ),
            "recent5_union_conditional": self._candidate_union_order(
                "recent5_interests",
                set(history_ids[-5:]),
                candidate_set,
            ),
            "full_history_union_conditional": self._candidate_union_order(
                "full_history_interests",
                set(history_ids),
                candidate_set,
            ),
        }

    def _sort_by_counts(self, counts: Counter[int]) -> list[int]:
        return sorted(
            self.train_targets,
            key=lambda target_id: (
                -counts[target_id],
                self.vocabulary.labels[target_id],
            ),
        )

    def _filter(
        self,
        order: Iterable[int],
        excluded: set[int],
    ) -> list[int]:
        return [target_id for target_id in order if target_id not in excluded]

    def _count_order(
        self,
        counts: Counter[int],
        excluded: set[int],
    ) -> list[int]:
        candidates = self.train_targets - excluded
        return sorted(
            candidates,
            key=lambda target_id: (
                -counts[target_id],
                self.global_rank[target_id],
                self.vocabulary.labels[target_id],
            ),
        )

    def _union_order(
        self,
        relation_name: str,
        source_ids: set[int],
        excluded: set[int],
    ) -> list[int]:
        scores: Counter[int] = Counter()
        probabilities = self.conditional_probabilities[relation_name]
        for source_id in source_ids:
            scores.update(probabilities.get(source_id, {}))
        candidates = self.train_targets - excluded
        return sorted(
            candidates,
            key=lambda target_id: (
                -scores[target_id],
                self.global_rank[target_id],
                self.vocabulary.labels[target_id],
            ),
        )

    def _score_order(
        self,
        scores: dict[int, float],
        excluded: set[int],
    ) -> list[int]:
        candidates = self.train_targets - excluded
        return sorted(
            candidates,
            key=lambda target_id: (
                -scores.get(target_id, 0.0),
                self.global_rank[target_id],
                self.vocabulary.labels[target_id],
            ),
        )

    def _candidate_count_order(
        self,
        counts: Counter[int],
        candidates: set[int],
    ) -> list[int]:
        return sorted(
            candidates,
            key=lambda target_id: (
                -counts[target_id],
                self.candidate_global_rank.get(
                    target_id,
                    len(self.candidate_global_rank),
                ),
                self.vocabulary.labels[target_id],
            ),
        )

    def _candidate_union_order(
        self,
        relation_name: str,
        source_ids: set[int],
        candidates: set[int],
    ) -> list[int]:
        scores: Counter[int] = Counter()
        probabilities = self.conditional_probabilities[relation_name]
        for source_id in source_ids:
            scores.update(probabilities.get(source_id, {}))
        return sorted(
            candidates,
            key=lambda target_id: (
                -scores[target_id],
                self.candidate_global_rank.get(
                    target_id,
                    len(self.candidate_global_rank),
                ),
                self.vocabulary.labels[target_id],
            ),
        )

    def _candidate_score_order(
        self,
        scores: dict[int, float],
        candidates: set[int],
    ) -> list[int]:
        return sorted(
            candidates,
            key=lambda target_id: (
                -scores.get(target_id, 0.0),
                self.candidate_global_rank.get(
                    target_id,
                    len(self.candidate_global_rank),
                ),
                self.vocabulary.labels[target_id],
            ),
        )


def _load_vocabulary(path: Path) -> Vocabulary:
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: int(row["interest_id"]))
    expected = list(range(len(rows)))
    actual = [int(row["interest_id"]) for row in rows]
    if actual != expected:
        raise ValueError("Vocabulary interest IDs must be contiguous from zero")
    keys = [
        (str(row["category"]), str(row["subcategory"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Vocabulary contains duplicate interests")
    labels = tuple(subcategory for _, subcategory in keys)
    if len(set(labels)) != len(labels):
        raise ValueError("Subcategory labels are not globally unique")
    parents = tuple(sorted({category for category, _ in keys}))
    parent_to_id = {
        parent: parent_id for parent_id, parent in enumerate(parents)
    }
    return Vocabulary(
        labels=labels,
        parents=parents,
        interest_to_id={key: interest_id for interest_id, key in enumerate(keys)},
        parent_to_id=parent_to_id,
        interest_parent_ids=tuple(parent_to_id[category] for category, _ in keys),
    )


def _load_allowed_targets(
    retained_interests_path: Path | None,
    vocabulary: Vocabulary,
) -> set[int]:
    if retained_interests_path is None:
        return set(range(len(vocabulary.labels)))
    rows = pq.read_table(retained_interests_path).to_pylist()
    labels = {
        str(
            row.get("target_subcategory", row.get("subcategory", ""))
        )
        for row in rows
    }
    labels.discard("")
    unknown = labels - set(vocabulary.labels)
    if unknown:
        raise ValueError(
            f"Retained-interest file contains unknown labels: {sorted(unknown)}"
        )
    return {
        interest_id
        for interest_id, label in enumerate(vocabulary.labels)
        if label in labels
    }


def _iter_impressions(
    path: Path,
    split: str,
    vocabulary: Vocabulary,
    allowed_targets: set[int],
    *,
    history_limit: int | None,
    batch_size: int,
    selected_impressions: set[str] | None = None,
) -> Iterator[Impression]:
    columns = [
        "split",
        "impression_id",
        "user_id",
        "timestamp",
        "history.list.element.category",
        "history.list.element.subcategory",
        "target_interest.category",
        "target_interest.subcategory",
    ]
    current_id: str | None = None
    current_user = ""
    current_timestamp: datetime | None = None
    current_history_ids: tuple[int, ...] = ()
    current_history_parent_ids: tuple[int, ...] = ()
    current_targets: list[Target] = []

    def emit() -> Impression | None:
        if current_id is None or not current_targets:
            return None
        if (
            selected_impressions is not None
            and current_id not in selected_impressions
        ):
            return None
        if current_timestamp is None:
            raise AssertionError("Missing impression timestamp")
        return Impression(
            impression_id=current_id,
            user_id=current_user,
            timestamp=current_timestamp,
            history_ids=current_history_ids,
            history_parent_ids=current_history_parent_ids,
            targets=tuple(current_targets),
        )

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            if str(row["split"]) != split:
                raise ValueError(
                    f"{path}: expected split {split!r}, got {row['split']!r}"
                )
            impression_id = str(row["impression_id"])
            history = row["history"]
            if not isinstance(history, list) or not history:
                raise ValueError(f"{split}:{impression_id}: empty history")
            if history_limit is not None:
                history = history[-history_limit:]
            history_keys = tuple(
                (str(item["category"]), str(item["subcategory"]))
                for item in history
            )
            try:
                history_ids = tuple(
                    vocabulary.interest_to_id[key] for key in history_keys
                )
                history_parent_ids = tuple(
                    vocabulary.parent_to_id[key[0]] for key in history_keys
                )
            except KeyError as error:
                raise ValueError(
                    f"{split}:{impression_id}: unknown history interest "
                    f"{error.args[0]!r}"
                ) from error

            if current_id is None or impression_id != current_id:
                completed = emit()
                if completed is not None:
                    yield completed
                current_id = impression_id
                current_user = str(row["user_id"])
                current_timestamp = row["timestamp"]
                current_history_ids = history_ids
                current_history_parent_ids = history_parent_ids
                current_targets = []
            elif (
                current_user != str(row["user_id"])
                or current_timestamp != row["timestamp"]
                or current_history_ids != history_ids
            ):
                raise ValueError(
                    f"{split}:{impression_id}: inconsistent grouped rows"
                )

            target = row["target_interest"]
            target_key = (
                str(target["category"]),
                str(target["subcategory"]),
            )
            try:
                target_id = vocabulary.interest_to_id[target_key]
            except KeyError as error:
                raise ValueError(
                    f"{split}:{impression_id}: unknown target {target_key!r}"
                ) from error
            if target_id in allowed_targets:
                current_targets.append(
                    Target(
                        interest_id=target_id,
                        parent_id=vocabulary.parent_to_id[target_key[0]],
                    )
                )
    completed = emit()
    if completed is not None:
        yield completed


def _conditional_probabilities(
    rows: dict[int, Counter[int]],
) -> dict[int, dict[int, float]]:
    result = {}
    for source_id, counts in rows.items():
        total = sum(counts.values())
        result[source_id] = {
            target_id: count / total
            for target_id, count in counts.items()
        }
    return result


def _relation_summary(
    rows: dict[int, Counter[int]],
) -> dict[str, float | int]:
    source_totals = {
        source_id: sum(counts.values())
        for source_id, counts in rows.items()
    }
    target_totals: Counter[int] = Counter()
    event_count = 0
    nonzero_pairs = 0
    top1_count = 0
    for counts in rows.values():
        event_count += sum(counts.values())
        nonzero_pairs += len(counts)
        top1_count += max(counts.values(), default=0)
        target_totals.update(counts)
    if not event_count:
        return {
            "events": 0,
            "active_sources": 0,
            "nonzero_pairs": 0,
            "conditional_top1_mass": 0.0,
            "mutual_information_bits": 0.0,
            "target_entropy_bits": 0.0,
            "fraction_target_entropy_explained": 0.0,
        }
    mutual_information = 0.0
    for source_id, counts in rows.items():
        for target_id, count in counts.items():
            joint_probability = count / event_count
            expected_probability = (
                source_totals[source_id]
                * target_totals[target_id]
                / (event_count * event_count)
            )
            mutual_information += joint_probability * math.log2(
                joint_probability / expected_probability
            )
    target_entropy = _entropy(target_totals)
    degrees_of_freedom = max(0, (len(source_totals) - 1) * (len(target_totals) - 1))
    miller_madow_bias = (
        degrees_of_freedom / (2 * event_count * math.log(2))
        if event_count
        else 0.0
    )
    bias_corrected = max(0.0, mutual_information - miller_madow_bias)
    return {
        "events": event_count,
        "active_sources": len(source_totals),
        "nonzero_pairs": nonzero_pairs,
        "conditional_top1_mass": top1_count / event_count,
        "mutual_information_bits": mutual_information,
        "mutual_information_miller_madow_bias_bits": miller_madow_bias,
        "mutual_information_bias_corrected_bits": bias_corrected,
        "target_entropy_bits": target_entropy,
        "fraction_target_entropy_explained": (
            mutual_information / target_entropy if target_entropy else 0.0
        ),
        "bias_corrected_fraction_target_entropy_explained": (
            bias_corrected / target_entropy if target_entropy else 0.0
        ),
    }


def _entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def _target_distribution(
    counts: Counter[int],
    vocabulary: Vocabulary,
) -> dict[str, object]:
    total = sum(counts.values())
    order = sorted(
        counts,
        key=lambda target_id: (
            -counts[target_id],
            vocabulary.labels[target_id],
        ),
    )
    return {
        "rows": total,
        "observed_targets": len(order),
        "entropy_bits": _entropy(counts),
        "effective_target_count": 2 ** _entropy(counts),
        "top1_mass": sum(counts[target_id] for target_id in order[:1]) / total,
        "top5_mass": sum(counts[target_id] for target_id in order[:5]) / total,
        "top10_mass": sum(counts[target_id] for target_id in order[:10]) / total,
        "top10": [
            {
                "rank": rank,
                "subcategory": vocabulary.labels[target_id],
                "parent": vocabulary.parents[
                    vocabulary.interest_parent_ids[target_id]
                ],
                "count": counts[target_id],
                "share": counts[target_id] / total,
            }
            for rank, target_id in enumerate(order[:10], start=1)
        ],
    }


def _distribution_shift(
    train_counts: Counter[int],
    dev_counts: Counter[int],
    allowed_targets: set[int],
    vocabulary: Vocabulary,
) -> dict[str, object]:
    train_total = sum(train_counts.values())
    dev_total = sum(dev_counts.values())
    train_probabilities = {
        target_id: train_counts[target_id] / train_total
        for target_id in allowed_targets
    }
    dev_probabilities = {
        target_id: dev_counts[target_id] / dev_total
        for target_id in allowed_targets
    }
    total_variation = sum(
        abs(train_probabilities[target_id] - dev_probabilities[target_id])
        for target_id in allowed_targets
    ) / 2
    jensen_shannon = 0.0
    for target_id in allowed_targets:
        left = train_probabilities[target_id]
        right = dev_probabilities[target_id]
        mixture = (left + right) / 2
        if left:
            jensen_shannon += 0.5 * left * math.log2(left / mixture)
        if right:
            jensen_shannon += 0.5 * right * math.log2(right / mixture)
    train_order = sorted(
        allowed_targets,
        key=lambda target_id: (
            -train_counts[target_id],
            vocabulary.labels[target_id],
        ),
    )
    dev_order = sorted(
        allowed_targets,
        key=lambda target_id: (
            -dev_counts[target_id],
            vocabulary.labels[target_id],
        ),
    )
    return {
        "total_variation_distance": total_variation,
        "jensen_shannon_divergence_bits": jensen_shannon,
        "top10_overlap_count": len(
            set(train_order[:10]).intersection(dev_order[:10])
        ),
        "dev_mass_on_train_top1": sum(
            dev_probabilities[target_id] for target_id in train_order[:1]
        ),
        "dev_mass_on_train_top5": sum(
            dev_probabilities[target_id] for target_id in train_order[:5]
        ),
        "dev_mass_on_train_top10": sum(
            dev_probabilities[target_id] for target_id in train_order[:10]
        ),
    }


def _cross_split_user_summary(
    train: SplitStats,
    dev: SplitStats,
) -> dict[str, object]:
    overlapping_users = train.users.intersection(dev.users)
    identical_history_users = sum(
        train.user_first_histories[user_id]
        == dev.user_first_histories[user_id]
        for user_id in overlapping_users
    )
    dev_seen_user_impressions = sum(
        dev.user_impression_counts[user_id]
        for user_id in overlapping_users
    )
    seen_user_target_memberships = 0
    repeated_target_memberships = 0
    for user_id in overlapping_users:
        train_targets = train.user_target_counts[user_id]
        dev_targets = dev.user_target_counts[user_id]
        seen_user_target_memberships += sum(dev_targets.values())
        repeated_for_user = sum(
            count
            for target_id, count in dev_targets.items()
            if train_targets[target_id] > 0
        )
        repeated_target_memberships += repeated_for_user
    repeated_target_users = sum(
        any(
            train.user_target_counts[user_id][target_id] > 0
            for target_id in dev.user_target_counts[user_id]
        )
        for user_id in overlapping_users
    )

    return {
        "overlapping_users": len(overlapping_users),
        "overlapping_user_rate_over_dev_users": (
            len(overlapping_users) / len(dev.users)
        ),
        "overlapping_users_with_identical_static_history": (
            identical_history_users
        ),
        "identical_static_history_rate_over_overlapping_users": (
            identical_history_users / len(overlapping_users)
            if overlapping_users
            else 0.0
        ),
        "dev_impressions_from_overlapping_users": dev_seen_user_impressions,
        "dev_impression_rate_from_overlapping_users": (
            dev_seen_user_impressions / dev.impressions
        ),
        "dev_target_memberships_from_overlapping_users": (
            seen_user_target_memberships
        ),
        "dev_targets_previously_observed_for_same_user_in_train": (
            repeated_target_memberships
        ),
        "repeat_rate_over_seen_user_dev_target_memberships": (
            repeated_target_memberships / seen_user_target_memberships
            if seen_user_target_memberships
            else 0.0
        ),
        "overlapping_users_with_any_repeated_dev_target": (
            repeated_target_users
        ),
    }
def _top_relation_pairs(
    rows: dict[int, Counter[int]],
    target_counts: Counter[int],
    vocabulary: Vocabulary,
    *,
    minimum_source_support: int = 1_000,
    minimum_pair_count: int = 100,
    limit: int = 20,
) -> list[dict[str, object]]:
    target_total = sum(target_counts.values())
    records = []
    for source_id, counts in rows.items():
        source_support = sum(counts.values())
        if source_support < minimum_source_support:
            continue
        for target_id, count in counts.items():
            if count < minimum_pair_count:
                continue
            conditional = count / source_support
            target_share = target_counts[target_id] / target_total
            records.append(
                {
                    "source": vocabulary.labels[source_id],
                    "target": vocabulary.labels[target_id],
                    "count": count,
                    "source_support": source_support,
                    "conditional_probability": conditional,
                    "target_share": target_share,
                    "lift": conditional / target_share,
                    "same_parent": (
                        vocabulary.interest_parent_ids[source_id]
                        == vocabulary.interest_parent_ids[target_id]
                    ),
                }
            )
    records.sort(
        key=lambda row: (
            -float(row["conditional_probability"]),
            -int(row["count"]),
            str(row["source"]),
            str(row["target"]),
        )
    )
    return records[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _input_description(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": portable_path(path, root=root),
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_target_csv(
    path: Path,
    train: SplitStats,
    dev: SplitStats,
    allowed_targets: set[int],
) -> None:
    fieldnames = (
        "split",
        "parent",
        "subcategory",
        "count",
        "share",
        "rank",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stats in (train, dev):
            total = sum(stats.target_counts.values())
            order = sorted(
                allowed_targets,
                key=lambda target_id: (
                    -stats.target_counts[target_id],
                    stats.vocabulary.labels[target_id],
                ),
            )
            for rank, target_id in enumerate(order, start=1):
                parent_id = stats.vocabulary.interest_parent_ids[target_id]
                writer.writerow(
                    {
                        "split": stats.split,
                        "parent": stats.vocabulary.parents[parent_id],
                        "subcategory": stats.vocabulary.labels[target_id],
                        "count": stats.target_counts[target_id],
                        "share": stats.target_counts[target_id] / total,
                        "rank": rank,
                    }
                )


def _write_relation_csv(path: Path, train: SplitStats) -> None:
    fieldnames = (
        "relation",
        "source_parent",
        "source",
        "target_parent",
        "target",
        "count",
        "source_support",
        "conditional_probability",
        "target_share",
        "lift",
        "same_parent",
    )
    target_total = sum(train.target_counts.values())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for relation_name, rows in train.relation_counts.items():
            for source_id, counts in rows.items():
                source_support = sum(counts.values())
                for target_id, count in counts.items():
                    target_share = train.target_counts[target_id] / target_total
                    conditional = count / source_support
                    source_parent_id = (
                        train.vocabulary.interest_parent_ids[source_id]
                    )
                    target_parent_id = (
                        train.vocabulary.interest_parent_ids[target_id]
                    )
                    writer.writerow(
                        {
                            "relation": relation_name,
                            "source_parent": train.vocabulary.parents[
                                source_parent_id
                            ],
                            "source": train.vocabulary.labels[source_id],
                            "target_parent": train.vocabulary.parents[
                                target_parent_id
                            ],
                            "target": train.vocabulary.labels[target_id],
                            "count": count,
                            "source_support": source_support,
                            "conditional_probability": conditional,
                            "target_share": target_share,
                            "lift": conditional / target_share,
                            "same_parent": source_parent_id == target_parent_id,
                        }
                    )


def _write_baseline_csv(
    path: Path,
    baseline_summary: dict[str, object],
    cutoffs: tuple[int, ...],
) -> None:
    fieldnames = ["baseline", "slice", "impressions", "target_memberships", "mrr"]
    for cutoff in cutoffs:
        fieldnames.extend(
            [
                f"recall_at_{cutoff}",
                f"ndcg_at_{cutoff}",
                f"hit_at_{cutoff}",
                f"target_membership_hit_at_{cutoff}",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for baseline, slices in baseline_summary.items():
            for slice_name, metrics in slices.items():
                writer.writerow(
                    {
                        "baseline": baseline,
                        "slice": slice_name,
                        **metrics,
                    }
                )


def _evaluate_candidate_rows(
    path: Path,
    evaluator: BaselineEvaluator,
    vocabulary: Vocabulary,
    *,
    batch_size: int,
) -> dict[str, object]:
    label_to_id = {
        label: interest_id
        for interest_id, label in enumerate(vocabulary.labels)
    }
    result = {
        baseline: {
            "samples": 0,
            "correct": 0,
            "reciprocal_rank_sum": 0.0,
        }
        for baseline in evaluator.metrics
    }
    columns = [
        "selection_id",
        "user_id",
        "history_subcategories",
        "candidates",
        "answer_category",
    ]
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        for row in batch.to_pylist():
            try:
                history_ids = tuple(
                    label_to_id[str(label)]
                    for label in row["history_subcategories"]
                )
                candidates = [
                    label_to_id[str(label)] for label in row["candidates"]
                ]
                answer_id = label_to_id[str(row["answer_category"])]
            except KeyError as error:
                raise ValueError(
                    f"{row['selection_id']}: unknown label {error.args[0]!r}"
                ) from error
            rankings = evaluator.candidate_rankings(
                user_id=str(row["user_id"]),
                history_ids=history_ids,
                candidates=candidates,
            )
            for baseline, ranking in rankings.items():
                rank = ranking.index(answer_id) + 1
                result[baseline]["samples"] += 1
                result[baseline]["correct"] += int(rank == 1)
                result[baseline]["reciprocal_rank_sum"] += 1.0 / rank
    return {
        baseline: {
            "samples": metrics["samples"],
            "correct": metrics["correct"],
            "accuracy": metrics["correct"] / metrics["samples"],
            "mrr": metrics["reciprocal_rank_sum"] / metrics["samples"],
        }
        for baseline, metrics in result.items()
    }


def _write_candidate_baseline_csv(
    path: Path,
    candidate_results: dict[str, object],
) -> None:
    fieldnames = ("dataset", "baseline", "samples", "correct", "accuracy", "mrr")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_name, baselines in candidate_results.items():
            for baseline, metrics in baselines.items():
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "baseline": baseline,
                        **metrics,
                    }
                )


def analyze_relationships(
    train_path: Path | str,
    dev_path: Path | str,
    vocabulary_path: Path | str,
    output_dir: Path | str,
    *,
    retained_interests_path: Path | str | None = None,
    dev_impressions_path: Path | str | None = None,
    candidate_paths: Iterable[Path | str] = (),
    history_limit: int | None = None,
    batch_size: int = 4_096,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    overwrite: bool = False,
    root: Path | str | None = None,
) -> dict[str, object]:
    train_path = Path(train_path)
    dev_path = Path(dev_path)
    vocabulary_path = Path(vocabulary_path)
    output_dir = Path(output_dir)
    retained_path = (
        Path(retained_interests_path)
        if retained_interests_path is not None
        else None
    )
    dev_impressions = (
        Path(dev_impressions_path)
        if dev_impressions_path is not None
        else None
    )
    candidates = tuple(Path(path) for path in candidate_paths)
    root_path = Path(root) if root is not None else Path.cwd()
    if history_limit is not None and history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive")
    cutoffs = tuple(sorted(set(cutoffs)))
    required = [train_path, dev_path, vocabulary_path]
    if retained_path is not None:
        required.append(retained_path)
    if dev_impressions is not None:
        required.append(dev_impressions)
    required.extend(candidates)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite."
        )

    vocabulary = _load_vocabulary(vocabulary_path)
    allowed_targets = _load_allowed_targets(retained_path, vocabulary)
    selected_dev_impressions = (
        {
            str(value)
            for value in pq.read_table(
                dev_impressions,
                columns=["impression_id"],
            ).column("impression_id").to_pylist()
        }
        if dev_impressions is not None
        else None
    )
    train = SplitStats("train", vocabulary)
    for impression in _iter_impressions(
        train_path,
        "train",
        vocabulary,
        allowed_targets,
        history_limit=history_limit,
        batch_size=batch_size,
        selected_impressions=None,
    ):
        train.add(impression)
    evaluator = BaselineEvaluator(train, allowed_targets, cutoffs)
    dev = SplitStats("dev", vocabulary)
    for impression in _iter_impressions(
        dev_path,
        "dev",
        vocabulary,
        allowed_targets,
        history_limit=history_limit,
        batch_size=batch_size,
        selected_impressions=selected_dev_impressions,
    ):
        dev.add(impression)
        evaluator.evaluate(impression)

    baseline_summary = evaluator.summary()
    candidate_results = {
        path.stem: _evaluate_candidate_rows(
            path,
            evaluator,
            vocabulary,
            batch_size=batch_size,
        )
        for path in candidates
    }
    payload = {
        "analysis": {
            "rule_version": RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "learned_prediction_model_used": False,
            "analysis_unit": "eligible impression with all retained targets",
            "history_limit": history_limit,
            "dev_impression_filter_count": (
                len(selected_dev_impressions)
                if selected_dev_impressions is not None
                else None
            ),
            "allowed_target_count": len(allowed_targets),
            "cutoffs": list(cutoffs),
            "fit_policy": (
                "All conditional counts and rankings are fitted on Train, "
                "frozen, and then evaluated on Dev."
            ),
        },
        "inputs": {
            "train": _input_description(train_path, root_path),
            "dev": _input_description(dev_path, root_path),
            "vocabulary": _input_description(vocabulary_path, root_path),
            "retained_interests": (
                _input_description(retained_path, root_path)
                if retained_path is not None
                else None
            ),
            "dev_impressions": (
                _input_description(dev_impressions, root_path)
                if dev_impressions is not None
                else None
            ),
        },
        "vocabulary": {
            "exact_interest_count": len(vocabulary.labels),
            "parent_count": len(vocabulary.parents),
            "allowed_targets": [
                vocabulary.labels[target_id]
                for target_id in sorted(allowed_targets)
            ],
        },
        "train": train.summary(),
        "dev": dev.summary(),
        "train_dev": {
            **_cross_split_user_summary(train, dev),
            "target_distribution_shift": _distribution_shift(
                train.target_counts,
                dev.target_counts,
                allowed_targets,
                vocabulary,
            ),
        },
        "top_train_relationships": {
            relation_name: _top_relation_pairs(
                rows,
                train.target_counts,
                vocabulary,
            )
            for relation_name, rows in train.relation_counts.items()
        },
        "dev_baselines_fitted_on_train": baseline_summary,
        "candidate_baselines_fitted_on_train": candidate_results,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as staging_name:
        staging = Path(staging_name)
        (staging / "summary.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_target_csv(
            staging / "target_distribution.csv",
            train,
            dev,
            allowed_targets,
        )
        _write_relation_csv(staging / "train_relationships.csv", train)
        _write_baseline_csv(
            staging / "dev_baseline_metrics.csv",
            baseline_summary,
            cutoffs,
        )
        if candidate_results:
            _write_candidate_baseline_csv(
                staging / "candidate_baseline_metrics.csv",
                candidate_results,
            )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze model-free relationships between ordered history interests "
            "and novel clicked interests."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/new_interest/train.parquet"),
    )
    parser.add_argument(
        "--candidate-path",
        action="append",
        type=Path,
        default=[],
        help="Optional expanded K-candidate Parquet to score without a model.",
    )
    parser.add_argument(
        "--dev-path",
        type=Path,
        default=Path("data/processed/new_interest/dev.parquet"),
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=Path(
            "data/processed/new_interest/interest_vocabulary.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-interests-path", type=Path)
    parser.add_argument(
        "--dev-impressions-path",
        type=Path,
        help=(
            "Optional Parquet containing impression_id values that restrict "
            "Dev evaluation, for example the frozen grouped Dev10K."
        ),
    )
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = analyze_relationships(
        args.train_path,
        args.dev_path,
        args.vocabulary_path,
        args.output_dir,
        retained_interests_path=args.retained_interests_path,
        dev_impressions_path=args.dev_impressions_path,
        candidate_paths=args.candidate_path,
        history_limit=args.history_limit,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "train_impressions": result["train"]["impressions"],
                "dev_impressions": result["dev"]["impressions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
