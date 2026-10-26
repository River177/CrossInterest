from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.ranking_metrics import (
    RankingExample,
    RankingPrediction,
    evaluate_rankings,
)
from new_interest.infrastructure.paths import portable_path


TRACKS = ("exposure_restricted", "full_taxonomy")


def _history_length_bucket(length: int) -> str:
    if length <= 10:
        return "1-10"
    if length <= 25:
        return "11-25"
    if length <= 50:
        return "26-50"
    if length <= 100:
        return "51-100"
    return "101+"


def _target_count_bucket(count: int) -> str:
    if count <= 2:
        return str(count)
    if count <= 5:
        return "3-5"
    return "6+"


def _profile_group_bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    return "6+"


def _load_labels(path: Path) -> tuple[str, ...]:
    table = pq.read_table(path)
    if "subcategory" not in table.column_names:
        raise ValueError(f"{path}: missing 'subcategory' column")
    labels = tuple(str(value) for value in table["subcategory"].to_pylist())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"{path}: vocabulary labels must be unique and non-empty")
    return labels


def _load_predictions(
    path: Path,
    vocabulary: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    required = {"user_id", "ranked_labels", "ranked_scores"}
    schema = pq.ParquetFile(path).schema_arrow
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    vocabulary_set = set(vocabulary)
    indexed: dict[str, dict[str, object]] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        columns=sorted(required),
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            if user_id in indexed:
                raise ValueError(f"{path}: duplicate user_id {user_id!r}")
            labels = tuple(str(value) for value in row["ranked_labels"])
            scores = tuple(float(value) for value in row["ranked_scores"])
            if set(labels) != vocabulary_set or len(labels) != len(vocabulary):
                raise ValueError(
                    f"{path}: {user_id} ranking must contain each vocabulary "
                    "label exactly once"
                )
            if len(scores) != len(labels):
                raise ValueError(
                    f"{path}: {user_id} labels and scores differ in length"
                )
            indexed[user_id] = {
                "labels": labels,
                "scores": scores,
            }
    if not indexed:
        raise ValueError(f"{path}: predictions are empty")
    return indexed


def _filtered_ranking(
    full_labels: tuple[str, ...],
    full_scores: tuple[float, ...],
    candidates: set[str],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    selected = [
        (label, score)
        for label, score in zip(full_labels, full_scores)
        if label in candidates
    ]
    return (
        tuple(label for label, _ in selected),
        tuple(score for _, score in selected),
    )


def evaluate_future_interest_predictions(
    benchmark_path: Path | str,
    vocabulary_path: Path | str,
    predictions_path: Path | str,
    output_dir: Path | str,
    *,
    track: str,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
    overwrite: bool = False,
    root: Path | None = None,
) -> dict[str, object]:
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    benchmark_path = Path(benchmark_path)
    vocabulary_path = Path(vocabulary_path)
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)
    for path in (benchmark_path, vocabulary_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation outputs already exist in {output_dir}. "
            "Use --overwrite."
        )
    backup = output_dir.with_name(output_dir.name + ".old")
    if output_dir.exists() and overwrite and backup.exists():
        raise FileExistsError(
            f"Cannot replace {output_dir}: backup exists at {backup}"
        )

    vocabulary = _load_labels(vocabulary_path)
    vocabulary_set = set(vocabulary)
    predictions = _load_predictions(predictions_path, vocabulary)
    required_columns = [
        "user_id",
        "history_subcategories",
        "future_novel_subcategories",
        "future_exposed_subcategories",
    ]
    optional_columns = []
    benchmark_schema = pq.ParquetFile(benchmark_path).schema_arrow
    for column in ("history_length", "profile_group_size"):
        if column in benchmark_schema.names:
            optional_columns.append(column)
    missing = set(required_columns) - set(benchmark_schema.names)
    if missing:
        raise ValueError(
            f"{benchmark_path}: missing benchmark columns {sorted(missing)}"
        )

    examples: list[RankingExample] = []
    track_predictions: list[RankingPrediction] = []
    benchmark_user_ids: set[str] = set()
    total_users = 0
    zero_target_users = 0
    excluded_history_labels = 0

    for batch in pq.ParquetFile(benchmark_path).iter_batches(
        columns=required_columns + optional_columns,
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            total_users += 1
            user_id = str(row["user_id"])
            if user_id in benchmark_user_ids:
                raise ValueError(
                    f"{benchmark_path}: duplicate user_id {user_id!r}"
                )
            benchmark_user_ids.add(user_id)
            prediction = predictions.get(user_id)
            if prediction is None:
                raise ValueError(
                    f"{predictions_path}: missing prediction for {user_id}"
                )
            history = {
                str(value) for value in row["history_subcategories"]
            }
            positives = {
                str(value)
                for value in row["future_novel_subcategories"]
            }
            exposed = {
                str(value)
                for value in row["future_exposed_subcategories"]
            }
            if not history.issubset(vocabulary_set):
                raise ValueError(
                    f"{benchmark_path}: {user_id} history has invalid labels"
                )
            if not positives.issubset(vocabulary_set):
                raise ValueError(
                    f"{benchmark_path}: {user_id} targets have invalid labels"
                )
            if not positives.issubset(exposed):
                raise ValueError(
                    f"{benchmark_path}: {user_id} positive target was not "
                    "exposed"
                )
            if positives.intersection(history):
                raise ValueError(
                    f"{benchmark_path}: {user_id} novel target occurs in "
                    "history"
                )
            if not positives:
                zero_target_users += 1
                continue

            candidates = (
                exposed - history
                if track == "exposure_restricted"
                else vocabulary_set - history
            )
            if not positives.issubset(candidates):
                raise ValueError(
                    f"{benchmark_path}: {user_id} target missing from "
                    f"{track} candidates"
                )
            full_labels = tuple(prediction["labels"])
            full_scores = tuple(prediction["scores"])
            ranked_labels, ranked_scores = _filtered_ranking(
                full_labels,
                full_scores,
                candidates,
            )
            history_length = int(
                row.get("history_length", len(row["history_subcategories"]))
            )
            profile_group_size = int(row.get("profile_group_size", 1))
            attributes = {
                "target_count": _target_count_bucket(len(positives)),
                "history_length": _history_length_bucket(history_length),
                "profile_group_size": _profile_group_bucket(
                    profile_group_size
                ),
            }
            examples.append(
                RankingExample(
                    example_id=user_id,
                    positive_labels=tuple(sorted(positives)),
                    candidate_labels=tuple(
                        label for label in vocabulary if label in candidates
                    ),
                    attributes=attributes,
                )
            )
            track_predictions.append(
                RankingPrediction(
                    example_id=user_id,
                    ranked_labels=ranked_labels,
                    ranked_scores=ranked_scores,
                )
            )
            excluded_history_labels += len(history)

    unexpected_predictions = set(predictions) - benchmark_user_ids
    if unexpected_predictions:
        raise ValueError(
            f"{predictions_path}: contains predictions outside benchmark, "
            f"for example {sorted(unexpected_predictions)[:5]}"
        )

    evaluation = evaluate_rankings(
        examples,
        track_predictions,
        cutoffs=cutoffs,
    )
    metrics = {
        "track": track,
        "total_users": total_users,
        "positive_target_users": len(examples),
        "zero_target_users": zero_target_users,
        "evaluated_user_rate": len(examples) / total_users,
        "mean_excluded_history_labels": (
            excluded_history_labels / len(examples)
            if examples
            else 0.0
        ),
        "benchmark_path": portable_path(benchmark_path, root=root),
        "vocabulary_path": portable_path(vocabulary_path, root=root),
        "predictions_path": portable_path(predictions_path, root=root),
        **evaluation.metrics,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".future-interest-evaluation-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        pq.write_table(
            pa.Table.from_pylist(evaluation.details),
            staging / "details.parquet",
            compression="zstd",
            use_dictionary=True,
            row_group_size=2_000,
        )
        (staging / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full-vocabulary user scores on exposure-restricted or "
            "full-taxonomy future novel-interest retrieval."
        )
    )
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--vocabulary-file", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = evaluate_future_interest_predictions(
        args.benchmark_file,
        args.vocabulary_file,
        args.predictions,
        args.output_dir,
        track=args.track,
        cutoffs=tuple(args.cutoffs),
        overwrite=args.overwrite,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
