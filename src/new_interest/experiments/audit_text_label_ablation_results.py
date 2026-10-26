from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    _load_labels,
    record_input_sha256,
    sha256_adapter_semantics,
    sha256_directory,
    sha256_file,
)
from new_interest.evaluation.metrics import (
    RESPONSE_PREFIX,
    evaluate_ranked_outputs,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_equal(
    expected: object,
    actual: object,
    name: str,
    *,
    tolerance: float,
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        _require(expected.keys() == actual.keys(), f"{name}: key mismatch")
        for key in expected:
            _assert_equal(
                expected[key],
                actual[key],
                f"{name}.{key}",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        _require(len(expected) == len(actual), f"{name}: length mismatch")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_equal(
                left,
                right,
                f"{name}[{index}]",
                tolerance=tolerance,
            )
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        _require(
            math.isfinite(float(expected))
            and math.isfinite(float(actual))
            and abs(float(expected) - float(actual)) <= tolerance,
            f"{name}: expected {expected}, got {actual}",
        )
        return
    _require(expected == actual, f"{name}: expected {expected!r}, got {actual!r}")


def audit_t1_training(
    sft_dir: Path | str,
    *,
    expected_train_samples: int = 21_600,
    expected_optimizer_steps: int = 2_025,
) -> dict[str, object]:
    sft_dir = Path(sft_dir)
    summary = _load_json(sft_dir / "run_summary.json")
    token_audit = _load_json(sft_dir / "token_audit.json")
    model_audit = _load_json(sft_dir / "model_audit.json")
    checkpoint_root = sft_dir / "persistent_checkpoints"
    if not checkpoint_root.is_dir():
        checkpoint_root = sft_dir.parent / "persistent_checkpoints"
    trainer_state = _load_json(
        checkpoint_root
        / f"checkpoint-{expected_optimizer_steps}"
        / "trainer_state.json"
    )
    adapter_dir = sft_dir / "final_adapter"
    adapter_file = adapter_dir / "adapter_model.safetensors"

    _require(
        int(summary["train_samples"]) == expected_train_samples,
        "Training sample count mismatch",
    )
    _require(summary["loss_reduction"] == "per_example_mean", "Loss mismatch")
    _require(summary.get("sample_weight_file") is None, "Unexpected weights")
    _require(float(summary["configured_epochs"]) == 3.0, "Epoch mismatch")
    _require(int(summary["world_size"]) == 4, "World size mismatch")
    _require(
        int(summary["effective_global_batch_size"]) == 32,
        "Global batch mismatch",
    )
    _require(
        int(summary["expected_optimizer_steps"]) == expected_optimizer_steps,
        "Expected optimizer-step mismatch",
    )
    _require(
        int(trainer_state["global_step"]) == expected_optimizer_steps,
        "Completed optimizer-step mismatch",
    )
    _require(
        int(trainer_state["max_steps"]) == expected_optimizer_steps,
        "Trainer max-step mismatch",
    )
    _require(float(trainer_state["epoch"]) == 3.0, "Completed epoch mismatch")
    _require(
        math.isfinite(float(summary["train_metrics"]["train_loss"])),
        "Non-finite training loss",
    )
    _require(
        float(summary["train_metrics"]["epoch"]) == 3.0,
        "Training metric epoch mismatch",
    )
    _require(
        int(token_audit["samples"]) == expected_train_samples,
        "Token-audit sample mismatch",
    )
    _require(
        int(token_audit["over_max_seq_length"]) == 0,
        "Token audit contains over-length rows",
    )
    _require(
        not model_audit["forbidden_trainable_parameters"],
        "Forbidden trainable parameters found",
    )
    _require(adapter_file.is_file(), f"Missing {adapter_file}")
    _require(adapter_file.stat().st_size > 0, "Adapter is empty")
    return {
        "status": "passed",
        "train_samples": expected_train_samples,
        "configured_epochs": 3.0,
        "optimizer_steps": expected_optimizer_steps,
        "world_size": 4,
        "effective_global_batch_size": 32,
        "loss_reduction": "per_example_mean",
        "train_loss": float(summary["train_metrics"]["train_loss"]),
        "adapter_bytes": adapter_file.stat().st_size,
        "adapter_semantic_sha256": sha256_adapter_semantics(adapter_dir),
    }


def audit_text_label_evaluation(
    merged_dir: Path | str,
    eval_file: Path | str,
    labels_file: Path | str,
    adapter_dir: Path | str,
    *,
    prediction_kind: str,
    expected_records: int = 31_392,
    expected_targets: int = 37_728,
    expected_supported_interests: int = 104,
    expected_label_count: int = 108,
    expected_top_k: int = 20,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
    metric_tolerance: float = 1e-12,
) -> dict[str, object]:
    if prediction_kind not in {"likelihood", "beam"}:
        raise ValueError("prediction_kind must be likelihood or beam")
    merged_dir = Path(merged_dir)
    eval_file = Path(eval_file)
    labels_file = Path(labels_file)
    adapter_dir = Path(adapter_dir)
    prediction_name = (
        "text_label_predictions.parquet"
        if prediction_kind == "likelihood"
        else "beam_predictions.parquet"
    )
    prediction_path = merged_dir / prediction_name
    metrics_path = merged_dir / "metrics.json"
    per_interest_path = merged_dir / "per_interest_metrics.parquet"
    manifest_path = merged_dir / "merge_manifest.json"
    for path in (
        prediction_path,
        metrics_path,
        per_interest_path,
        manifest_path,
        eval_file,
        labels_file,
    ):
        _require(path.is_file(), f"Missing {path}")

    labels = _load_labels(labels_file)
    label_set = set(labels)
    records = pq.read_table(eval_file).to_pylist()
    predictions = pq.read_table(prediction_path).to_pylist()
    _require(len(labels) == expected_label_count, "Label count mismatch")
    _require(len(records) == expected_records, "Eval row count mismatch")
    _require(len(predictions) == expected_records, "Prediction row count mismatch")

    identities: set[str] = set()
    ranked_outputs: list[list[dict[str, object]]] = []
    target_occurrences = 0
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        _require(
            int(prediction["record_index"]) == index,
            f"Record {index}: index/order mismatch",
        )
        _require(
            str(prediction["impression_id"]) == str(record["impression_id"]),
            f"Record {index}: impression mismatch",
        )
        _require(
            str(prediction["input_sha256"]) == record_input_sha256(record),
            f"Record {index}: input hash mismatch",
        )
        acceptable = [str(value) for value in record["acceptable_subcategories"]]
        _require(
            [str(value) for value in prediction["acceptable_subcategories"]]
            == acceptable,
            f"Record {index}: target mismatch",
        )
        target_occurrences += len(acceptable)
        identities.add(str(prediction["evaluation_identity"]))
        ranked_labels = [str(value) for value in prediction["ranked_labels"]]
        ranked_scores = [float(value) for value in prediction["ranked_scores"]]
        _require(
            len(ranked_labels) == len(ranked_scores) == expected_top_k,
            f"Record {index}: expected Top-{expected_top_k}",
        )
        _require(
            len(set(ranked_labels)) == expected_top_k,
            f"Record {index}: duplicate labels",
        )
        _require(
            set(ranked_labels).issubset(label_set),
            f"Record {index}: unknown label",
        )
        _require(
            all(math.isfinite(score) for score in ranked_scores),
            f"Record {index}: non-finite score",
        )
        responses = [f"{RESPONSE_PREFIX}{label}" for label in ranked_labels]
        if prediction_kind == "beam":
            _require(
                [str(value) for value in prediction["ranked_responses"]]
                == responses,
                f"Record {index}: response/label mismatch",
            )
        ranked_outputs.append(
            [
                {"label": label, "response": response, "score": score}
                for label, response, score in zip(
                    ranked_labels,
                    responses,
                    ranked_scores,
                )
            ]
        )

    _require(len(identities) == 1, "Multiple evaluation identities")
    _require(target_occurrences == expected_targets, "Target count mismatch")
    recomputed, _ = evaluate_ranked_outputs(
        records,
        ranked_outputs,
        cutoffs=cutoffs,
    )
    _require(
        int(recomputed["supported_interest_count"])
        == expected_supported_interests,
        "Supported-interest count mismatch",
    )

    remote = _load_json(metrics_path)
    for key, value in recomputed.items():
        _require(key in remote, f"Remote metrics missing {key}")
        _assert_equal(
            value,
            remote[key],
            key,
            tolerance=metric_tolerance,
        )
    _require(int(remote["label_vocabulary_size"]) == expected_label_count, "Remote label count mismatch")
    _require(int(remote["top_k"]) == expected_top_k, "Remote Top-K mismatch")
    _require(list(remote["cutoffs"]) == list(cutoffs), "Remote cutoff mismatch")
    if prediction_kind == "likelihood":
        _require(remote["history_prediction_filtering"] is False, "History filtering enabled")
        _require(remote["scoring"] == "label_token_mean_log_probability", "Scoring mismatch")
        _require(remote["scoring_backend"] == "prefix_cache", "Backend mismatch")
        _require(remote["fixed_prefix_scored"] is False, "Prefix was scored")
        _require(remote["eos_scored"] is False, "EOS was scored")
    else:
        _require(int(remote["num_beams"]) == 32, "Beam width mismatch")

    semantic_adapter_hash = sha256_adapter_semantics(adapter_dir)
    legacy_adapter_hash = sha256_directory(adapter_dir)
    remote_adapter_hash = str(remote["adapter_sha256"])
    _require(
        remote_adapter_hash in {semantic_adapter_hash, legacy_adapter_hash},
        "Adapter identity mismatch",
    )
    adapter_hash_scheme = (
        "peft-evaluation-adapter-v1"
        if remote_adapter_hash == semantic_adapter_hash
        else "legacy-directory-hash"
    )
    _require(remote["eval_sha256"] == sha256_file(eval_file), "Eval hash mismatch")
    _require(remote["labels_sha256"] == sha256_file(labels_file), "Label hash mismatch")
    identity = next(iter(identities))
    _require(remote["evaluation_identity"] == identity, "Metric identity mismatch")

    manifest = _load_json(manifest_path)
    _require(int(manifest["record_count"]) == expected_records, "Manifest row mismatch")
    _require(manifest["evaluation_identity"] == identity, "Manifest identity mismatch")
    _require(
        manifest["merged_prediction_sha256"] == sha256_file(prediction_path),
        "Prediction hash mismatch",
    )
    _require(
        manifest["per_interest_metrics_sha256"] == sha256_file(per_interest_path),
        "Per-interest hash mismatch",
    )
    _require(
        manifest["metrics_sha256"] == sha256_file(metrics_path),
        "Metrics hash mismatch",
    )

    metric_subset = {
        key: float(remote[key])
        for cutoff in cutoffs
        for key in (
            f"recall_at_{cutoff}",
            f"ndcg_at_{cutoff}",
            f"hit_at_{cutoff}",
            f"interest_macro_recall_at_{cutoff}",
            f"interest_macro_ndcg_at_{cutoff}",
        )
    }
    metric_subset[f"mrr_at_{cutoffs[-1]}"] = float(
        remote[f"mrr_at_{cutoffs[-1]}"]
    )
    return {
        "status": "passed",
        "prediction_kind": prediction_kind,
        "records": expected_records,
        "target_occurrences": target_occurrences,
        "supported_interest_count": expected_supported_interests,
        "label_vocabulary_size": expected_label_count,
        "top_k": expected_top_k,
        "history_prediction_filtering": False,
        "evaluation_identity": identity,
        "adapter_sha256": remote_adapter_hash,
        "adapter_hash_scheme": adapter_hash_scheme,
        "prediction_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(metrics_path),
        "per_interest_metrics_sha256": sha256_file(per_interest_path),
        "merge_manifest_sha256": sha256_file(manifest_path),
        "metrics": metric_subset,
    }


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently audit one text-label ranking condition."
    )
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-kind",
        choices=("likelihood", "beam"),
        required=True,
    )
    parser.add_argument("--sft-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    audit: dict[str, object] = {
        "status": "passed",
        "evaluation": audit_text_label_evaluation(
            args.merged_dir,
            args.eval_file,
            args.labels_file,
            args.adapter_dir,
            prediction_kind=args.prediction_kind,
        ),
    }
    if args.sft_dir is not None:
        audit["training"] = audit_t1_training(args.sft_dir)
    _atomic_write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
