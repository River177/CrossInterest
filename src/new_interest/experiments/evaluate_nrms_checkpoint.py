from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from new_interest.experiments.build_nrms_interest_embeddings import (
    load_nrms_interest_embeddings,
)
from new_interest.experiments.build_nrms_title_embeddings import (
    load_nrms_title_embeddings,
)
from new_interest.experiments.tune_semantic_history_linear_ranker import (
    _resolve_device,
)
from new_interest.experiments.tune_semantic_history_nrms import (
    Architecture,
    NRMSInterestRanker,
    _device_split,
    _environment_metadata,
    _evaluation_view,
    _load_dataset_row_splits,
    _load_history_splits,
    _score_split,
    _torch,
    _train_targets,
    _write_nrms_condition,
)
from new_interest.experiments.tune_semantic_history_ridge import (
    _load_config,
    _publish,
    _sha256,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-nrms-checkpoint-unfiltered-eval-v1"


def _load_checkpoint(
    path: Path,
) -> tuple[tuple[str, ...], dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        labels = tuple(str(value) for value in payload["labels"].tolist())
        selected_values = payload["selected_json"].reshape(-1).tolist()
        if len(selected_values) != 1:
            raise ValueError("checkpoint must contain exactly one selected config")
        selected = json.loads(str(selected_values[0]))
        if not isinstance(selected, dict):
            raise ValueError("checkpoint selected_json must contain an object")
        state = {
            key.removeprefix("parameter::"): np.asarray(payload[key]).copy()
            for key in payload.files
            if key.startswith("parameter::")
        }
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be non-empty and unique")
    if not state:
        raise ValueError("checkpoint contains no model parameters")
    return labels, selected, state


def _architecture_from_selected(
    selected: Mapping[str, object],
) -> Architecture:
    architecture = Architecture(
        hidden_dimension=int(selected["hidden_dimension"]),
        attention_heads=int(selected["attention_heads"]),
        dropout=float(selected["dropout"]),
    )
    if architecture.hidden_dimension % architecture.attention_heads:
        raise ValueError("checkpoint hidden dimension is not divisible by heads")
    return architecture


def _positive_count(split: Any) -> int:
    return int(
        sum(
            int(bucket.positives.sum())
            for bucket in split.buckets
            if bucket.positives is not None
        )
    )


def evaluate_nrms_checkpoint(
    path_or_config: Path | str | Mapping[str, object],
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    requested_device: str = "auto",
    overwrite: bool = False,
) -> Path:
    config, config_path = _load_config(path_or_config)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    dataset_dir = Path(str(config["dataset_dir"]))
    train_path = dataset_dir / "train.parquet"
    dev_path = dataset_dir / "dev.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    checkpoint_path = checkpoint_dir / "selected_model.npz"
    masked_metrics_path = (
        checkpoint_dir
        / "conditions"
        / "dev_exploratory"
        / "selected_nrms"
        / "metrics.json"
    )

    data_config = dict(config["data"])
    loader = str(data_config.get("loader", "semantic_users"))
    history_input = str(
        data_config.get("history_input", "behavior_titles")
    )
    if loader not in {"semantic_users", "dataset_rows"}:
        raise ValueError("unsupported NRMS data loader")
    if history_input not in {"behavior_titles", "interest_names"}:
        raise ValueError("unsupported NRMS History input")
    if loader == "semantic_users" and history_input != "behavior_titles":
        raise ValueError("semantic_users supports only behavior titles")

    boundaries = tuple(int(value) for value in data_config["bucket_boundaries"])
    batch_sizes = {
        int(key): int(value)
        for key, value in dict(
            data_config["batch_size_by_boundary"]
        ).items()
    }
    if set(boundaries) != set(batch_sizes):
        raise ValueError("batch sizes must cover every History bucket")

    labels, train_targets, _ = _train_targets(train_path)
    vocabulary = {
        str(value)
        for value in pq.read_table(
            vocabulary_path, columns=["subcategory"]
        )["subcategory"].to_pylist()
    }
    if not set(labels).issubset(vocabulary):
        raise ValueError("checkpoint taxonomy falls outside the vocabulary")

    if history_input == "behavior_titles":
        embedding_dir = Path(str(config["title_embedding_dir"]))
        history_values, history_matrix, embedding_metadata = (
            load_nrms_title_embeddings(embedding_dir)
        )
        embedding_paths = (
            embedding_dir / "title_embeddings.parquet",
            embedding_dir / "metadata.json",
            embedding_dir / "manifest.json",
        )
    else:
        embedding_dir = Path(str(config["interest_embedding_dir"]))
        history_values, history_matrix, embedding_metadata = (
            load_nrms_interest_embeddings(embedding_dir)
        )
        embedding_paths = (
            embedding_dir / "interest_embeddings.parquet",
            embedding_dir / "metadata.json",
            embedding_dir / "manifest.json",
        )
    input_to_id = {
        value: index for index, value in enumerate(history_values)
    }
    if len(input_to_id) != len(history_values):
        raise ValueError("frozen History embedding cache contains duplicates")
    padding_id = len(history_values)
    seed = int(config["seed"])

    benchmark_value = config.get("semantic_benchmark_dir")
    benchmark_dir = (
        Path(str(benchmark_value)) if benchmark_value is not None else None
    )
    if loader == "semantic_users":
        if benchmark_dir is None:
            raise ValueError("semantic_benchmark_dir is required")
        _, dev, data_audit = _load_history_splits(
            benchmark_dir=benchmark_dir,
            train_targets=train_targets,
            labels=labels,
            title_to_id=input_to_id,
            boundaries=boundaries,
            seed=seed,
            padding_id=padding_id,
        )
        data_paths = tuple(
            benchmark_dir / f"{split}.parquet"
            for split in ("fit", "validation", "dev")
        )
    else:
        _, dev, data_audit = _load_dataset_row_splits(
            train_path=train_path,
            dev_path=dev_path,
            train_targets=train_targets,
            labels=labels,
            input_to_id=input_to_id,
            history_input=history_input,
            boundaries=boundaries,
            seed=seed,
            padding_id=padding_id,
        )
        data_paths = (dev_path,)

    checkpoint_labels, selected, checkpoint_state = _load_checkpoint(
        checkpoint_path
    )
    if checkpoint_labels != labels:
        raise ValueError("checkpoint label order differs from dataset taxonomy")

    torch = _torch()
    device = _resolve_device(requested_device)
    precision = "float32"
    current_device = torch.device(device)
    frozen_matrix = np.concatenate(
        [
            history_matrix,
            np.zeros((1, history_matrix.shape[1]), dtype=np.float32),
        ],
        axis=0,
    )
    embedding_tensor = torch.as_tensor(
        frozen_matrix, dtype=torch.float32, device=current_device
    )
    model = NRMSInterestRanker.build(
        embedding_tensor,
        padding_id=padding_id,
        label_count=len(labels),
        architecture=_architecture_from_selected(selected),
    ).to(current_device)
    model.title_embeddings = embedding_tensor
    incompatible = model.load_state_dict(
        {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in checkpoint_state.items()
        },
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"checkpoint state mismatch: {incompatible}")

    dev_device = _device_split(dev, device)
    result = _score_split(
        model,
        dev,
        dev_device,
        batch_sizes=batch_sizes,
        precision=precision,
        device=device,
        labels=labels,
        k=int(config["k"]),
        history_prediction_filtering=False,
    )
    metrics = dict(result.metrics)
    expected = dict(config.get("expected_counts", {}))
    expected_units = int(expected.get("dev_evaluation_units", dev.user_count))
    expected_targets = int(
        expected.get("dev_target_occurrences", _positive_count(dev))
    )
    expected_supported = int(
        expected.get(
            "dev_supported_interests",
            metrics["supported_interests"],
        )
    )
    if int(metrics["evaluation_units"]) != expected_units:
        raise ValueError("unfiltered evaluation unit count mismatch")
    if int(metrics["target_occurrences"]) != expected_targets:
        raise ValueError("unfiltered target occurrence count mismatch")
    if int(metrics["supported_interests"]) != expected_supported:
        raise ValueError("unfiltered Interest-Macro denominator mismatch")
    if metrics["history_prediction_filtering"] is not False:
        raise ValueError("unfiltered evaluation unexpectedly enabled filtering")

    view = _evaluation_view(dev)
    overlap_flags = [
        view.history_mask[index, result.top_k[index]]
        for index in range(dev.user_count)
    ]
    top_k_unique = all(
        len(set(row.tolist())) == len(row) for row in result.top_k
    )
    history_overlap_rows = sum(bool(values.any()) for values in overlap_flags)
    history_overlap_predictions = sum(
        int(values.sum()) for values in overlap_flags
    )
    if not top_k_unique:
        raise ValueError("unfiltered NRMS Top-K contains duplicates")

    masked_reference = None
    if masked_metrics_path.is_file():
        masked_reference = json.loads(
            masked_metrics_path.read_text(encoding="utf-8")
        )
    training_resolved_path = checkpoint_dir / "resolved_config.json"
    training_resolved = (
        json.loads(training_resolved_path.read_text(encoding="utf-8"))
        if training_resolved_path.is_file()
        else {}
    )

    source_paths = [
        train_path,
        vocabulary_path,
        *data_paths,
        *embedding_paths,
        checkpoint_path,
    ]
    for optional in (
        config_path,
        masked_metrics_path,
        training_resolved_path,
    ):
        if optional is not None and optional.is_file():
            source_paths.append(optional)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".nrms-unfiltered-eval-", dir=str(output_dir.parent)
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        condition_paths = _write_nrms_condition(
            staging,
            "dev_exploratory",
            "selected_nrms_unfiltered",
            dev,
            result,
            labels,
        )
        delta = None
        if masked_reference is not None:
            delta = {
                key: float(metrics[key]) - float(masked_reference[key])
                for key in (
                    "interest_macro_recall_at_10",
                    "interest_macro_ndcg_at_10",
                    "impression_macro_recall_at_10",
                    "impression_macro_ndcg_at_10",
                )
            }
        summary = {
            "condition": "selected_nrms_unfiltered",
            "dev_exploratory": metrics,
            "masked_reference": masked_reference,
            "unfiltered_minus_masked": delta,
            "condition_paths": condition_paths,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit = {
            **data_audit,
            "evaluation_units": dev.user_count,
            "target_occurrences": _positive_count(dev),
            "supported_interests": int(metrics["supported_interests"]),
            "label_vocabulary_size": len(labels),
            "top_k": int(config["k"]),
            "top_k_unique": top_k_unique,
            "history_prediction_filtering": False,
            "history_overlap_rows_in_top_k": history_overlap_rows,
            "history_overlap_predictions_in_top_k": (
                history_overlap_predictions
            ),
            "training_candidate_mask": training_resolved.get(
                "training_candidate_mask"
            ),
        }
        (staging / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        resolved = {
            "rule_version": RULE_VERSION,
            "source_config": portable_path(config_path)
            if config_path is not None
            else "inline",
            "checkpoint_dir": portable_path(checkpoint_dir),
            "checkpoint": portable_path(checkpoint_path),
            "dataset_dir": portable_path(dataset_dir),
            "history_embedding_dir": portable_path(embedding_dir),
            "history_input": history_input,
            "history_encoder": embedding_metadata["model"],
            "selected": selected,
            "evaluation_device": device,
            "evaluation_precision": precision,
            "history_prediction_filtering": False,
            "training_candidate_mask": training_resolved.get(
                "training_candidate_mask"
            ),
        }
        (staging / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "environment.json").write_text(
            json.dumps(
                _environment_metadata(
                    torch, device=device, precision=precision
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "rule_version": RULE_VERSION,
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "output_sha256": output_hashes,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen NRMS checkpoint on the full output taxonomy "
            "without filtering History interests at prediction time."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = evaluate_nrms_checkpoint(
        args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        requested_device=args.device,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
