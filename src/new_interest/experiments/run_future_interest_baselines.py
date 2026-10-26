from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_future_interest import (
    TRACKS,
    evaluate_future_interest_predictions,
)
from new_interest.experiments.embedding_future_interest import (
    SentenceTransformerEncoder,
    TextEncoder,
    predict_embedding_future_interest,
    write_history_embeddings,
)
from new_interest.experiments.multilabel_future_interest import (
    run_multilabel_baseline,
)
from new_interest.experiments.statistical_future_interest import (
    STATISTICAL_BASELINES,
    fit_statistical_future_interest,
    predict_statistical_future_interest,
)
from new_interest.infrastructure.paths import portable_path


BASELINE_IDS = (
    *STATISTICAL_BASELINES,
    "embedding_cosine",
    "multilabel_topic_logistic",
    "multilabel_embedding_logistic",
)


def _load_config(
    config_path_or_config: Path | str | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(config_path_or_config, Mapping):
        return dict(config_path_or_config)
    path = Path(config_path_or_config)
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_prefix(
    source: Path,
    destination: Path,
    *,
    max_rows: int,
) -> int:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    parquet = pq.ParquetFile(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        destination,
        parquet.schema_arrow,
        compression="zstd",
        use_dictionary=True,
    )
    written = 0
    try:
        for batch in parquet.iter_batches(batch_size=min(8_192, max_rows)):
            if written >= max_rows:
                break
            remaining = max_rows - written
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            writer.write_batch(batch)
            written += batch.num_rows
    finally:
        writer.close()
    if written == 0:
        raise ValueError(f"{source}: no rows for smoke subset")
    return written


def _prepare_benchmark_paths(
    benchmark_dir: Path,
    *,
    max_users_per_split: int | None,
    temporary_root: Path,
) -> dict[str, Path]:
    paths = {
        split: benchmark_dir / f"{split}.parquet"
        for split in ("train", "validation", "test")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if max_users_per_split is None:
        return paths
    subset_dir = temporary_root / "benchmark-subset"
    for split, source in list(paths.items()):
        destination = subset_dir / f"{split}.parquet"
        _copy_prefix(
            source,
            destination,
            max_rows=max_users_per_split,
        )
        paths[split] = destination
    return paths


def _ensure_artifact_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Artifact directory already exists: {path}. "
            "Use overwrite=True."
        )
    backup = path.with_name(path.name + ".old")
    if path.exists() and overwrite and backup.exists():
        raise FileExistsError(
            f"Cannot replace {path}: backup exists at {backup}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _relocate_paths(
    value: object,
    *,
    staging_dir: Path,
    artifact_dir: Path,
) -> object:
    if isinstance(value, dict):
        return {
            key: _relocate_paths(
                item,
                staging_dir=staging_dir,
                artifact_dir=artifact_dir,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _relocate_paths(
                item,
                staging_dir=staging_dir,
                artifact_dir=artifact_dir,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    replacements = (
        (str(staging_dir.resolve()), str(artifact_dir.resolve())),
        (portable_path(staging_dir), portable_path(artifact_dir)),
        (str(staging_dir), str(artifact_dir)),
    )
    for source, destination in replacements:
        if value == source or value.startswith(source + os.sep):
            return destination + value[len(source) :]
    return value


def _publish_directory(
    staging_dir: Path,
    artifact_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if artifact_dir.exists() and overwrite:
        backup = artifact_dir.with_name(artifact_dir.name + ".old")
        os.replace(artifact_dir, backup)
        try:
            os.replace(staging_dir, artifact_dir)
        except BaseException:
            os.replace(backup, artifact_dir)
            raise
        shutil.rmtree(backup)
        return
    if artifact_dir.exists():
        raise FileExistsError(f"Artifact directory exists: {artifact_dir}")
    os.replace(staging_dir, artifact_dir)


def _evaluate_condition(
    *,
    baseline_id: str,
    predictions_path: Path,
    test_path: Path,
    vocabulary_path: Path,
    artifact_dir: Path,
    cutoffs: tuple[int, ...],
    overwrite: bool,
) -> dict[str, object]:
    tracks: dict[str, object] = {}
    for track in TRACKS:
        output_dir = artifact_dir / baseline_id / track
        metrics = evaluate_future_interest_predictions(
            test_path,
            vocabulary_path,
            predictions_path,
            output_dir,
            track=track,
            cutoffs=cutoffs,
            overwrite=overwrite,
        )
        tracks[track] = {
            "metrics_path": portable_path(output_dir / "metrics.json"),
            "details_path": portable_path(output_dir / "details.parquet"),
            "metrics": metrics,
        }
    return {
        "baseline_id": baseline_id,
        "predictions_path": portable_path(predictions_path),
        "tracks": tracks,
    }


def run_future_interest_baseline_suite(
    config_path_or_config: Path | str | Mapping[str, object],
    *,
    overwrite: bool = False,
    max_users_per_split: int | None = None,
    encoder: TextEncoder | None = None,
) -> dict[str, object]:
    config = _load_config(config_path_or_config)
    benchmark_dir = Path(str(config["benchmark_dir"]))
    artifact_dir = Path(str(config["artifact_dir"]))
    vocabulary_path = benchmark_dir / "interest_vocabulary.parquet"
    if not vocabulary_path.is_file():
        raise FileNotFoundError(vocabulary_path)
    cutoffs = tuple(int(value) for value in config["cutoffs"])
    _ensure_artifact_dir(artifact_dir, overwrite=overwrite)

    embedding_config = dict(config["embedding"])
    statistical_config = dict(config["statistical"])
    multilabel_config = dict(config["multilabel"])
    model_name = str(embedding_config["model_name"])
    revision = str(embedding_config["revision"])
    device_value = embedding_config.get("device")
    embedding_device = (
        str(device_value) if device_value is not None else None
    )
    embedding_batch_size = int(embedding_config["batch_size"])
    shared_encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=embedding_device,
        batch_size=embedding_batch_size,
    )

    with tempfile.TemporaryDirectory(
        prefix=".future-interest-baseline-suite-",
        dir=str(artifact_dir.parent),
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        working_dir = temporary_root / "publish"
        working_dir.mkdir()
        paths = _prepare_benchmark_paths(
            benchmark_dir,
            max_users_per_split=max_users_per_split,
            temporary_root=temporary_root,
        )

        embeddings_dir = working_dir / "embeddings"
        embedding_paths: dict[str, Path] = {}
        embedding_summaries: dict[str, object] = {}
        for split in ("train", "validation", "test"):
            path = embeddings_dir / f"{split}.parquet"
            embedding_summaries[split] = write_history_embeddings(
                paths[split],
                path,
                encoder=shared_encoder,
                model_name=model_name,
                revision=revision,
                device=embedding_device,
                batch_size=embedding_batch_size,
                overwrite=overwrite,
            )
            embedding_paths[split] = path

        statistical_model = fit_statistical_future_interest(
            paths["train"],
            vocabulary_path,
            alpha=float(statistical_config["alpha"]),
        )
        statistical_result = predict_statistical_future_interest(
            statistical_model,
            paths["test"],
            working_dir,
            overwrite=False,
        )

        embedding_prediction_path = (
            working_dir / "embedding_cosine" / "predictions.parquet"
        )
        embedding_result = predict_embedding_future_interest(
            paths["test"],
            vocabulary_path,
            embedding_paths["test"],
            embedding_prediction_path,
            encoder=shared_encoder,
            model_name=model_name,
            revision=revision,
            device=embedding_device,
            batch_size=embedding_batch_size,
            metric=str(embedding_config["metric"]),
            overwrite=False,
        )

        common_multilabel: dict[str, Any] = {
            "epochs": int(multilabel_config["epochs"]),
            "lr": float(multilabel_config["learning_rate"]),
            "weight_decay": float(multilabel_config["weight_decay"]),
            "patience": int(multilabel_config["patience"]),
            "batch_size": int(multilabel_config["batch_size"]),
            "prediction_batch_size": int(multilabel_config["batch_size"]),
            "seed": int(multilabel_config["seed"]),
            "device": str(multilabel_config["device"]),
            "prior_alpha": float(multilabel_config["prior_alpha"]),
            "overwrite": False,
        }
        topic_result = run_multilabel_baseline(
            paths["train"],
            paths["validation"],
            paths["test"],
            vocabulary_path,
            working_dir / "multilabel_topic_logistic",
            feature_mode="topic_frequency",
            **common_multilabel,
        )
        embedding_logistic_result = run_multilabel_baseline(
            paths["train"],
            paths["validation"],
            paths["test"],
            vocabulary_path,
            working_dir / "multilabel_embedding_logistic",
            feature_mode="history_embedding",
            train_embedding_path=embedding_paths["train"],
            validation_embedding_path=embedding_paths["validation"],
            test_embedding_path=embedding_paths["test"],
            **common_multilabel,
        )

        prediction_paths = {
            baseline: Path(
                statistical_result["prediction_files"][baseline]
            )
            for baseline in STATISTICAL_BASELINES
        }
        prediction_paths.update(
            {
                "embedding_cosine": embedding_prediction_path,
                "multilabel_topic_logistic": Path(
                    topic_result["prediction_path"]
                ),
                "multilabel_embedding_logistic": Path(
                    embedding_logistic_result["prediction_path"]
                ),
            }
        )
        conditions = [
            _evaluate_condition(
                baseline_id=baseline_id,
                predictions_path=prediction_paths[baseline_id],
                test_path=paths["test"],
                vocabulary_path=vocabulary_path,
                artifact_dir=working_dir,
                cutoffs=cutoffs,
                overwrite=False,
            )
            for baseline_id in BASELINE_IDS
        ]

        resolved_config = {
            **config,
            "overwrite": overwrite,
            "max_users_per_split": max_users_per_split,
            "benchmark_paths": {
                split: portable_path(benchmark_dir / f"{split}.parquet")
                for split in ("train", "validation", "test")
            },
            "vocabulary_path": portable_path(vocabulary_path),
        }
        manifest = {
            "rule_version": config["rule_version"],
            "artifact_dir": portable_path(artifact_dir),
            "benchmark_dir": portable_path(benchmark_dir),
            "conditions": conditions,
            "embedding_artifacts": {
                split: portable_path(path)
                for split, path in embedding_paths.items()
            },
            "run_summaries": {
                "embeddings": embedding_summaries,
                "statistical": statistical_result,
                "embedding_cosine": embedding_result,
                "multilabel_topic_logistic": topic_result,
                "multilabel_embedding_logistic": (
                    embedding_logistic_result
                ),
            },
        }
        resolved_config = _relocate_paths(
            resolved_config,
            staging_dir=working_dir,
            artifact_dir=artifact_dir,
        )
        manifest = _relocate_paths(
            manifest,
            staging_dir=working_dir,
            artifact_dir=artifact_dir,
        )
        for metrics_path in working_dir.glob("*/**/metrics.json"):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = _relocate_paths(
                metrics,
                staging_dir=working_dir,
                artifact_dir=artifact_dir,
            )
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (working_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (working_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish_directory(
            working_dir,
            artifact_dir,
            overwrite=overwrite,
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run statistical, embedding, and masked multi-label future "
            "novel-interest baselines and evaluate both tracks."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/future_interest/baselines_v1.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-users-per-split", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = run_future_interest_baseline_suite(
        args.config,
        overwrite=args.overwrite,
        max_users_per_split=args.max_users_per_split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
