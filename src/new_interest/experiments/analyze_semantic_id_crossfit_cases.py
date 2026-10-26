from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    record_input_sha256,
    sha256_file,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "semantic-id-global-crossfit-case-analysis-v1"
CUTOFFS = (1, 5, 10, 20)
EXPECTED_IMPRESSIONS = 31_392
EXPECTED_TARGET_OCCURRENCES = 37_728
EXPECTED_SUPPORTED_TARGETS = 104

DEFAULT_EVAL = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_v1/"
    "eval/interest_only/dev_full_grouped.parquet"
)
DEFAULT_WEIGHTED = Path(
    "artifacts/static-sud-semantic-id-global-crossfit/qwen35-interest/"
    "final/"
    "static-sud-semantic-id-global-crossfit-qwen-interest-final-real-v2-0825/"
    "qwen35-interest-semantic-id-global-crossfit-final/"
    "real_dev_id_token_predictions.parquet"
)
DEFAULT_VANILLA = Path(
    "artifacts/static-sud-semantic-id-token/"
    "static-semantic-id-token-qwen-interest-eval-recovery-0820-v1/"
    "qwen35-interest-semantic-id-eval-recovery/recovery/eval/merged/"
    "id_token_predictions.parquet"
)
DEFAULT_MAPPING = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_v1/"
    "interest_token_mapping.parquet"
)
DEFAULT_VOCABULARY = Path(
    "data/processed/new_interest_static_user_disjoint_v1/"
    "interest_vocabulary.parquet"
)
DEFAULT_RAW_DEV = Path(
    "data/processed/new_interest_static_user_disjoint_v1/dev.parquet"
)
DEFAULT_WEIGHTS = Path(
    "data/training/"
    "sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_v1/"
    "sample_weights.parquet"
)
DEFAULT_RETAINED = Path(
    "data/training/"
    "generative_new_interest_static_user_disjoint_balanced_200_h50_v1/"
    "retained_interests.parquet"
)
DEFAULT_OUTPUT = Path(
    "artifacts/static-sud-semantic-id-global-crossfit/qwen35-interest/"
    "analysis/case-study-20260828"
)

# Each outcome is selected independently in the short (1-4), medium (5-10),
# and long (11+) history strata. Weighted/Vanilla-only cases maximize the
# absolute rank transition; both-success cases require Weighted rank 1 and
# maximize the Vanilla rank; neither-success cases minimize Weighted entropy.
REPRESENTATIVE_CASES = {
    "weighted_only_at_10": ["273342", "18212", "42111"],
    "vanilla_only_at_10": ["207581", "251428", "151933"],
    "both_at_10_weighted_top1": ["334907", "296897", "16093"],
    "neither_at_20_lowest_weighted_entropy": [
        "7384",
        "179474",
        "356447",
    ],
}


def _rank(labels: Sequence[str], acceptable: Iterable[str]) -> int:
    positions = {label: index + 1 for index, label in enumerate(labels)}
    ranks = [positions[label] for label in acceptable if label in positions]
    return min(ranks) if ranks else 21


def _history_features(history: Sequence[str]) -> dict[str, object]:
    counts = Counter(history)
    length = len(history)
    unique = len(counts)
    probabilities = [count / length for count in counts.values()]
    return {
        "history_unique": unique,
        "history_repeat_rate": 1.0 - unique / length,
        "history_dominant_share": max(counts.values()) / length,
        "history_entropy": -sum(
            probability * math.log(probability)
            for probability in probabilities
        ),
        "last_interest_seen_before": history[-1] in history[:-1],
    }


def _history_length_bucket(length: int) -> str:
    if length <= 4:
        return "1-4"
    if length <= 10:
        return "5-10"
    if length <= 20:
        return "11-20"
    return "21-50"


def _history_unique_bucket(unique: int) -> str:
    if unique <= 2:
        return "1-2"
    if unique <= 4:
        return "3-4"
    if unique <= 7:
        return "5-7"
    if unique <= 12:
        return "8-12"
    return "13-50"


def _repeat_bucket(value: float) -> str:
    if value <= 0.10:
        return "<=0.10"
    if value <= 0.30:
        return "0.10-0.30"
    if value <= 0.50:
        return "0.30-0.50"
    return ">0.50"


def _answer_count_bucket(count: int) -> str:
    return str(count) if count <= 3 else "4+"


def _load_predictions(
    path: Path,
    records: Sequence[Mapping[str, object]],
    *,
    token_by_label: Mapping[str, str],
) -> list[dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    by_index = {int(row["record_index"]): row for row in rows}
    if len(rows) != len(records) or set(by_index) != set(range(len(records))):
        raise ValueError(f"{path}: prediction indices do not cover eval data")
    ordered = [by_index[index] for index in range(len(records))]
    for record, row in zip(records, ordered):
        if str(record["impression_id"]) != str(row["impression_id"]):
            raise ValueError(f"{path}: impression identity mismatch")
        expected_hash = record_input_sha256(record)
        if str(row["input_sha256"]) != expected_hash:
            raise ValueError(f"{path}: input SHA-256 mismatch")
        labels = [str(value) for value in row["ranked_labels"]]
        tokens = [str(value) for value in row["ranked_tokens"]]
        scores = [float(value) for value in row["ranked_scores"]]
        if not len(labels) == len(tokens) == len(scores) == 20:
            raise ValueError(f"{path}: expected exactly 20 ranked outputs")
        if len(set(labels)) != 20 or len(set(tokens)) != 20:
            raise ValueError(f"{path}: duplicate ranked output")
        if tokens != [token_by_label.get(label) for label in labels]:
            raise ValueError(f"{path}: token/label mapping mismatch")
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"{path}: non-finite ranked score")
        if [str(value) for value in row["acceptable_subcategories"]] != [
            str(value) for value in record["acceptable_subcategories"]
        ]:
            raise ValueError(f"{path}: ground-truth mismatch")
    return ordered


def _case_rows(
    records: Sequence[Mapping[str, object]],
    weighted: Sequence[Mapping[str, object]],
    vanilla: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for index, (record, weighted_row, vanilla_row) in enumerate(
        zip(records, weighted, vanilla)
    ):
        history = [str(value) for value in record["history_subcategories"]]
        acceptable = [
            str(value) for value in record["acceptable_subcategories"]
        ]
        weighted_labels = [
            str(value) for value in weighted_row["ranked_labels"]
        ]
        vanilla_labels = [
            str(value) for value in vanilla_row["ranked_labels"]
        ]
        weighted_rank = _rank(weighted_labels, acceptable)
        vanilla_rank = _rank(vanilla_labels, acceptable)
        row: dict[str, object] = {
            "record_index": index,
            "input_sha256": str(weighted_row["input_sha256"]),
            "impression_id": str(record["impression_id"]),
            "user_id": str(record["user_id"]),
            "timestamp": record["timestamp"],
            "history_length": int(record["history_length"]),
            "history_subcategories": history,
            "prompt": str(record["prompt"]),
            "acceptable_responses": [
                str(value) for value in record["acceptable_responses"]
            ],
            "acceptable_subcategories": acceptable,
            "answer_count": len(acceptable),
            "weighted_rank": weighted_rank,
            "vanilla_rank": vanilla_rank,
            "rank_gain": vanilla_rank - weighted_rank,
            "weighted_ranked_labels": weighted_labels,
            "weighted_ranked_tokens": [
                str(value) for value in weighted_row["ranked_tokens"]
            ],
            "weighted_ranked_scores": [
                float(value) for value in weighted_row["ranked_scores"]
            ],
            "weighted_restricted_entropy": float(
                weighted_row["restricted_entropy"]
            ),
            "vanilla_ranked_labels": vanilla_labels,
            "vanilla_ranked_tokens": [
                str(value) for value in vanilla_row["ranked_tokens"]
            ],
            "vanilla_ranked_scores": [
                float(value) for value in vanilla_row["ranked_scores"]
            ],
            "vanilla_restricted_entropy": float(
                vanilla_row["restricted_entropy"]
            ),
            **_history_features(history),
        }
        row["history_length_bucket"] = _history_length_bucket(
            int(row["history_length"])
        )
        row["history_unique_bucket"] = _history_unique_bucket(
            int(row["history_unique"])
        )
        row["history_repeat_bucket"] = _repeat_bucket(
            float(row["history_repeat_rate"])
        )
        row["answer_count_bucket"] = _answer_count_bucket(len(acceptable))
        for cutoff in CUTOFFS:
            for prefix, source in (
                ("weighted", weighted_row),
                ("vanilla", vanilla_row),
            ):
                row[f"{prefix}_hit_at_{cutoff}"] = float(
                    source[f"hit_at_{cutoff}"]
                )
                row[f"{prefix}_recall_at_{cutoff}"] = float(
                    source[f"recall_at_{cutoff}"]
                )
        output.append(row)
    return output


def _occurrence_rows(
    cases: Sequence[Mapping[str, object]],
    *,
    parent_by_subcategory: Mapping[str, str],
) -> list[dict[str, object]]:
    output = []
    for case in cases:
        history = [str(value) for value in case["history_subcategories"]]
        history_parents = {
            parent_by_subcategory.get(label, "Unknown") for label in history
        }
        last_parent = parent_by_subcategory.get(history[-1], "Unknown")
        weighted_positions = {
            str(label): index + 1
            for index, label in enumerate(case["weighted_ranked_labels"])
        }
        vanilla_positions = {
            str(label): index + 1
            for index, label in enumerate(case["vanilla_ranked_labels"])
        }
        for target in case["acceptable_subcategories"]:
            label = str(target)
            target_parent = parent_by_subcategory.get(label, "Unknown")
            weighted_rank = weighted_positions.get(label, 21)
            vanilla_rank = vanilla_positions.get(label, 21)
            row: dict[str, object] = {
                "record_index": int(case["record_index"]),
                "impression_id": str(case["impression_id"]),
                "user_id": str(case["user_id"]),
                "target_subcategory": label,
                "target_parent": target_parent,
                "history_length": int(case["history_length"]),
                "history_unique": int(case["history_unique"]),
                "history_repeat_rate": float(case["history_repeat_rate"]),
                "answer_count": int(case["answer_count"]),
                "same_parent_anywhere": target_parent in history_parents,
                "same_parent_last": target_parent == last_parent,
                "weighted_rank": weighted_rank,
                "vanilla_rank": vanilla_rank,
                "rank_gain": vanilla_rank - weighted_rank,
            }
            for cutoff in CUTOFFS:
                row[f"weighted_recall_at_{cutoff}"] = float(
                    weighted_rank <= cutoff
                )
                row[f"vanilla_recall_at_{cutoff}"] = float(
                    vanilla_rank <= cutoff
                )
            output.append(row)
    return output


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _slice_row(
    *,
    unit: str,
    family: str,
    value: str,
    rows: Sequence[Mapping[str, object]],
    metric: str,
) -> dict[str, object]:
    output: dict[str, object] = {
        "unit": unit,
        "slice_family": family,
        "slice_value": value,
        "n": len(rows),
    }
    for cutoff in CUTOFFS:
        weighted_key = f"weighted_{metric}_at_{cutoff}"
        vanilla_key = f"vanilla_{metric}_at_{cutoff}"
        weighted_value = _mean(rows, weighted_key)
        vanilla_value = _mean(rows, vanilla_key)
        output[f"weighted_{metric}_at_{cutoff}"] = weighted_value
        output[f"vanilla_{metric}_at_{cutoff}"] = vanilla_value
        output[f"delta_{metric}_at_{cutoff}_pp"] = 100.0 * (
            weighted_value - vanilla_value
        )
    return output


def _slice_metrics(
    cases: Sequence[Mapping[str, object]],
    occurrences: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    impression_specs = (
        ("history_length", "history_length_bucket"),
        ("history_unique", "history_unique_bucket"),
        ("history_repeat_rate", "history_repeat_bucket"),
        ("answer_count", "answer_count_bucket"),
    )
    for family, key in impression_specs:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in cases:
            grouped[str(row[key])].append(row)
        for value, rows in sorted(grouped.items()):
            output.append(
                _slice_row(
                    unit="impression",
                    family=family,
                    value=value,
                    rows=rows,
                    metric="hit",
                )
            )
    occurrence_specs = (
        ("same_parent_anywhere", "same_parent_anywhere"),
        ("same_parent_last", "same_parent_last"),
        ("target_parent", "target_parent"),
    )
    for family, key in occurrence_specs:
        grouped = defaultdict(list)
        for row in occurrences:
            grouped[str(row[key])].append(row)
        for value, rows in sorted(grouped.items()):
            output.append(
                _slice_row(
                    unit="target_occurrence",
                    family=family,
                    value=value,
                    rows=rows,
                    metric="recall",
                )
            )
    return output


def _target_metrics(
    occurrences: Sequence[Mapping[str, object]],
    weights: Sequence[Mapping[str, object]],
    retained: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_target: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in occurrences:
        by_target[str(row["target_subcategory"])].append(row)
    weight_by_target: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in weights:
        weight_by_target[str(row["target_subcategory"])].append(row)
    retained_by_target = {
        str(row["target_subcategory"]): row for row in retained
    }
    output = []
    for target, rows in sorted(by_target.items()):
        target_weights = weight_by_target[target]
        retained_row = retained_by_target[target]
        result: dict[str, object] = {
            "target_subcategory": target,
            "target_parent": str(rows[0]["target_parent"]),
            "dev_occurrences": len(rows),
            "source_train_count": int(retained_row["source_train_count"]),
            "selected_train_count": int(
                retained_row["selected_train_count"]
            ),
            "mean_sample_weight": _mean(target_weights, "sample_weight"),
            "mean_oof_prior_adjusted_nll": _mean(
                target_weights, "prior_adjusted_nll"
            ),
            "mean_oof_prior_adjusted_margin": _mean(
                target_weights, "prior_adjusted_margin"
            ),
            "mean_oof_difficulty_quantile": _mean(
                target_weights, "fold_global_difficulty_quantile"
            ),
            "weighted_mean_rank_capped_21": _mean(rows, "weighted_rank"),
            "vanilla_mean_rank_capped_21": _mean(rows, "vanilla_rank"),
            "mean_rank_gain": _mean(rows, "rank_gain"),
        }
        for cutoff in CUTOFFS:
            weighted_value = _mean(rows, f"weighted_recall_at_{cutoff}")
            vanilla_value = _mean(rows, f"vanilla_recall_at_{cutoff}")
            result[f"weighted_recall_at_{cutoff}"] = weighted_value
            result[f"vanilla_recall_at_{cutoff}"] = vanilla_value
            result[f"delta_recall_at_{cutoff}_pp"] = 100.0 * (
                weighted_value - vanilla_value
            )
        output.append(result)
    return output


def _top1_metrics(
    cases: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    target_impressions = Counter()
    for row in cases:
        target_impressions.update(set(row["acceptable_subcategories"]))
    by_label: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in cases:
        label = str(row["weighted_ranked_labels"][0])
        by_label[label].append(row)
    output = []
    for label, rows in sorted(by_label.items()):
        correct = sum(
            label in set(row["acceptable_subcategories"]) for row in rows
        )
        precision = correct / len(rows)
        prevalence = target_impressions[label] / len(cases)
        output.append(
            {
                "predicted_subcategory": label,
                "prediction_count": len(rows),
                "correct_count": correct,
                "precision": precision,
                "target_impression_count": target_impressions[label],
                "target_prevalence": prevalence,
                "precision_lift_over_prevalence": (
                    precision / prevalence if prevalence else None
                ),
            }
        )
    return output


def _history_target_pairs(
    cases: Sequence[Mapping[str, object]],
    occurrences: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    histories = {
        int(row["record_index"]): set(row["history_subcategories"])
        for row in cases
    }
    target_baselines: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    pairs: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in occurrences:
        target = str(row["target_subcategory"])
        target_baselines[target].append(row)
        for history_label in histories[int(row["record_index"])]:
            pairs[(target, str(history_label))].append(row)
    output = []
    for (target, history_label), rows in sorted(pairs.items()):
        weighted = _mean(rows, "weighted_recall_at_10")
        vanilla = _mean(rows, "vanilla_recall_at_10")
        baseline = _mean(
            target_baselines[target], "weighted_recall_at_10"
        )
        output.append(
            {
                "target_subcategory": target,
                "history_subcategory": history_label,
                "n_occurrences": len(rows),
                "weighted_recall_at_10": weighted,
                "vanilla_recall_at_10": vanilla,
                "delta_recall_at_10_pp": 100.0 * (weighted - vanilla),
                "target_baseline_weighted_recall_at_10": baseline,
                "lift_over_target_baseline_pp": 100.0 * (
                    weighted - baseline
                ),
            }
        )
    return output


def _raw_target_news(
    raw_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        target = row["target_interest"]
        output[str(row["impression_id"])].append(
            {
                "target_category": str(target["category"]),
                "target_subcategory": str(target["subcategory"]),
                "target_news": [
                    {
                        "news_id": str(item["news_id"]),
                        "title": str(item["title"]),
                    }
                    for item in row["target_news"]
                ],
            }
        )
    return output


def _representative_cases(
    cases: Sequence[Mapping[str, object]],
    raw_by_impression: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    selection_by_id = {
        impression_id: outcome
        for outcome, impression_ids in REPRESENTATIVE_CASES.items()
        for impression_id in impression_ids
    }
    by_id = {str(row["impression_id"]): row for row in cases}
    if set(selection_by_id).difference(by_id):
        raise ValueError("Representative case is absent from evaluation data")
    output = []
    for outcome, impression_ids in REPRESENTATIVE_CASES.items():
        for impression_id in impression_ids:
            row = dict(by_id[impression_id])
            row["selection_outcome"] = outcome
            row["raw_target_news_not_model_input"] = list(
                raw_by_impression.get(impression_id, [])
            )
            if hasattr(row["timestamp"], "isoformat"):
                row["timestamp"] = row["timestamp"].isoformat()
            output.append(row)
    return output


def _transition_counts(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
) -> dict[str, dict[str, int]]:
    output = {}
    for cutoff in CUTOFFS:
        weighted_key = f"weighted_{metric}_at_{cutoff}"
        vanilla_key = f"vanilla_{metric}_at_{cutoff}"
        both = weighted_only = vanilla_only = neither = 0
        for row in rows:
            weighted = bool(row[weighted_key])
            vanilla = bool(row[vanilla_key])
            if weighted and vanilla:
                both += 1
            elif weighted:
                weighted_only += 1
            elif vanilla:
                vanilla_only += 1
            else:
                neither += 1
        output[str(cutoff)] = {
            "both": both,
            "weighted_only": weighted_only,
            "vanilla_only": vanilla_only,
            "neither": neither,
        }
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def analyze(
    *,
    eval_path: Path,
    weighted_path: Path,
    vanilla_path: Path,
    mapping_path: Path,
    vocabulary_path: Path,
    raw_dev_path: Path,
    weights_path: Path,
    retained_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    source_paths = (
        eval_path,
        weighted_path,
        vanilla_path,
        mapping_path,
        vocabulary_path,
        raw_dev_path,
        weights_path,
        retained_path,
    )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists; pass --overwrite explicitly: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", dir=output_dir.parent
        )
    )
    try:
        records = pq.read_table(eval_path).to_pylist()
        mapping_rows = pq.read_table(mapping_path).to_pylist()
        token_by_label = {
            str(row["target_subcategory"]): str(row["interest_token"])
            for row in mapping_rows
        }
        vocabulary_rows = pq.read_table(vocabulary_path).to_pylist()
        parent_by_subcategory = {
            str(row["subcategory"]): str(row["category"])
            for row in vocabulary_rows
        }
        weighted = _load_predictions(
            weighted_path, records, token_by_label=token_by_label
        )
        vanilla = _load_predictions(
            vanilla_path, records, token_by_label=token_by_label
        )
        cases = _case_rows(records, weighted, vanilla)
        occurrences = _occurrence_rows(
            cases, parent_by_subcategory=parent_by_subcategory
        )
        supported_targets = {
            str(row["target_subcategory"]) for row in occurrences
        }
        if len(cases) != EXPECTED_IMPRESSIONS:
            raise ValueError("Unexpected impression count")
        if len(occurrences) != EXPECTED_TARGET_OCCURRENCES:
            raise ValueError("Unexpected target-occurrence count")
        if len(supported_targets) != EXPECTED_SUPPORTED_TARGETS:
            raise ValueError("Unexpected supported-target count")

        weights = pq.read_table(weights_path).to_pylist()
        retained = pq.read_table(retained_path).to_pylist()
        raw_rows = pq.read_table(raw_dev_path).to_pylist()
        raw_by_impression = _raw_target_news(raw_rows)

        slice_rows = _slice_metrics(cases, occurrences)
        target_rows = _target_metrics(occurrences, weights, retained)
        top1_rows = _top1_metrics(cases)
        pair_rows = _history_target_pairs(cases, occurrences)
        representative_rows = _representative_cases(
            cases, raw_by_impression
        )

        pq.write_table(pa.Table.from_pylist(cases), staging / "case_catalog.parquet")
        pq.write_table(
            pa.Table.from_pylist(occurrences),
            staging / "target_occurrences.parquet",
        )
        _write_csv(staging / "slice_metrics.csv", slice_rows)
        _write_csv(staging / "target_metrics.csv", target_rows)
        _write_csv(staging / "top1_prediction_metrics.csv", top1_rows)
        _write_csv(staging / "history_target_pair_metrics.csv", pair_rows)
        _write_json(
            staging / "representative_cases.json", representative_rows
        )

        summary = {
            "rule_version": RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": {
                "model": "Qwen/Qwen3.5-9B",
                "input_view": "interest_only",
                "output_representation": "semantic_id",
                "supervision": "global_cross_fitted_weighting",
                "comparison": "semantic_id_vanilla",
                "ranking": (
                    "one next-token forward pass; restricted log-softmax over "
                    "108 Semantic-ID tokens; unfiltered Top-20"
                ),
            },
            "counts": {
                "impressions": len(cases),
                "users": len({str(row["user_id"]) for row in cases}),
                "target_occurrences": len(occurrences),
                "supported_targets": len(supported_targets),
                "single_answer_impressions": sum(
                    int(row["answer_count"]) == 1 for row in cases
                ),
            },
            "impression_transitions": _transition_counts(
                cases, metric="hit"
            ),
            "target_occurrence_transitions": _transition_counts(
                occurrences, metric="recall"
            ),
            "mean_restricted_entropy": {
                "weighted": _mean(cases, "weighted_restricted_entropy"),
                "vanilla": _mean(cases, "vanilla_restricted_entropy"),
            },
            "representative_case_selection": {
                "population": "single-answer impressions",
                "history_strata": ["1-4", "5-10", "11+"],
                "rules": (
                    "weighted/vanilla-only maximize rank transition; both "
                    "requires Weighted rank 1 and maximizes Vanilla rank; "
                    "neither minimizes Weighted restricted entropy"
                ),
                "impression_ids": REPRESENTATIVE_CASES,
            },
            "source_sha256": {
                portable_path(path): sha256_file(path) for path in source_paths
            },
        }
        output_files = sorted(
            path for path in staging.iterdir() if path.is_file()
        )
        summary["output_sha256"] = {
            path.name: sha256_file(path) for path in output_files
        }
        _write_json(staging / "analysis_summary.json", summary)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete case catalog and descriptive slices for the "
            "Semantic-ID global Cross-Fitted result."
        )
    )
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--weighted", type=Path, default=DEFAULT_WEIGHTED)
    parser.add_argument("--vanilla", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--raw-dev", type=Path, default=DEFAULT_RAW_DEV)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--retained", type=Path, default=DEFAULT_RETAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = analyze(
        eval_path=args.eval,
        weighted_path=args.weighted,
        vanilla_path=args.vanilla,
        mapping_path=args.mapping,
        vocabulary_path=args.vocabulary,
        raw_dev_path=args.raw_dev,
        weights_path=args.weights,
        retained_path=args.retained,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
