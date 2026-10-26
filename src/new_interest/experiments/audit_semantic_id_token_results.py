from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    record_input_sha256,
    sha256_file,
)
from new_interest.training.semantic_id_tokens import (
    load_interest_token_mapping,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    _require(bool(files), f"{path}: adapter directory is empty")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _audit_training(
    sft_dir: Path,
    *,
    expected_train_samples: int,
    expected_optimizer_steps: int,
    expected_interest_count: int,
) -> dict[str, object]:
    required = [
        "run_summary.json",
        "model_audit.json",
        "token_audit.json",
        "semantic_token_drift.json",
        "semantic_adapter_state_audit.json",
        "semantic_adapter_reload_audit.json",
    ]
    for name in required:
        _require((sft_dir / name).is_file(), f"Missing {sft_dir / name}")

    summary = _load_json(sft_dir / "run_summary.json")
    trainer_state_path = sft_dir / "checkpoints" / "trainer_state.json"
    _require(
        trainer_state_path.is_file(),
        f"Missing {trainer_state_path}",
    )
    trainer_state = _load_json(trainer_state_path)
    model_audit = _load_json(sft_dir / "model_audit.json")
    token_audit = _load_json(sft_dir / "token_audit.json")
    drift = _load_json(sft_dir / "semantic_token_drift.json")
    state = _load_json(sft_dir / "semantic_adapter_state_audit.json")
    reload_audit = _load_json(
        sft_dir / "semantic_adapter_reload_audit.json"
    )

    _require(
        int(summary["train_samples"]) == expected_train_samples,
        "Training sample count mismatch",
    )
    _require(float(summary["configured_epochs"]) == 3.0, "Epoch mismatch")
    _require(int(summary["world_size"]) == 4, "World size mismatch")
    _require(
        int(summary["effective_global_batch_size"]) == 32,
        "Global batch mismatch",
    )
    _require(
        int(summary["expected_optimizer_steps"]) == expected_optimizer_steps,
        "Optimizer step count mismatch",
    )
    _require(
        int(trainer_state["global_step"]) == expected_optimizer_steps,
        "Completed optimizer step count mismatch",
    )
    _require(
        int(trainer_state["max_steps"]) == expected_optimizer_steps,
        "Trainer maximum step count mismatch",
    )
    _require(
        float(trainer_state["epoch"]) == 3.0,
        "Completed epoch mismatch",
    )
    _require(
        int(token_audit["samples"]) == expected_train_samples,
        "Token-audit sample count mismatch",
    )
    _require(
        int(token_audit["over_max_seq_length"]) == 0,
        "Token audit contains over-length samples",
    )
    _require(
        int(token_audit["max_tokens"]) <= int(token_audit["max_seq_length"]),
        "Maximum token length exceeds configured limit",
    )
    semantic = model_audit["semantic_id_tokens"]
    _require(bool(semantic["enabled"]), "Semantic ID tokens are disabled")
    _require(
        int(semantic["count"]) == expected_interest_count,
        "Model-audit interest count mismatch",
    )
    _require(
        not model_audit["forbidden_trainable_parameters"],
        "Forbidden trainable parameters were found",
    )
    _require(
        bool(drift["all_input_rows_changed"])
        and bool(drift["all_output_rows_changed"]),
        "At least one Semantic-ID row did not change during training",
    )
    _require(
        int(drift["changed_input_rows"]) == expected_interest_count
        and int(drift["changed_output_rows"]) == expected_interest_count,
        "Semantic-ID drift row count mismatch",
    )
    _require(int(state["adapter_bytes"]) > 0, "Adapter is empty")
    _require(
        not state["full_embedding_tensors"],
        "Adapter contains a full embedding tensor",
    )
    _require(
        int(state["trainable_token_tensor_count"]) in (1, 2),
        "Unexpected number of compact trainable-token tensors",
    )
    for shape in state["trainable_token_shapes"].values():
        _require(
            int(shape[0]) == expected_interest_count,
            "Compact token tensor has the wrong row count",
        )
    _require(bool(reload_audit["all_close"]), "Adapter reload audit failed")
    _require(
        int(reload_audit["interest_token_count"]) == expected_interest_count,
        "Reload-audit interest count mismatch",
    )
    integrity = reload_audit["adapter_state_integrity"]
    _require(bool(integrity["all_tensors_exact"]), "Reload state mismatch")
    _require(
        not integrity["missing_keys"]
        and not integrity["unexpected_keys"]
        and not integrity["mismatched_keys"],
        "Reload state contains key mismatches",
    )

    adapter_dir = sft_dir / "final_adapter"
    adapter_model = adapter_dir / "adapter_model.safetensors"
    _require(adapter_model.is_file(), f"Missing {adapter_model}")
    _require(adapter_model.stat().st_size > 0, "Adapter weights are empty")
    _require(
        adapter_model.stat().st_size == int(state["adapter_bytes"]),
        "Adapter byte count does not match state audit",
    )
    return {
        "train_samples": expected_train_samples,
        "configured_epochs": 3.0,
        "optimizer_steps": expected_optimizer_steps,
        "completed_optimizer_steps": int(trainer_state["global_step"]),
        "world_size": 4,
        "effective_global_batch_size": 32,
        "token_overflow_count": 0,
        "adapter_bytes": adapter_model.stat().st_size,
        "adapter_sha256": _sha256_directory(adapter_dir),
        "compact_token_tensor_count": int(
            state["trainable_token_tensor_count"]
        ),
        "reload_all_close": True,
    }


def _independent_interest_metrics(
    records: list[dict[str, object]],
    predictions: list[dict[str, object]],
    cutoffs: tuple[int, ...],
) -> tuple[dict[str, float], int, int]:
    support: Counter[str] = Counter()
    hits: dict[int, Counter[str]] = {
        cutoff: Counter() for cutoff in cutoffs
    }
    gains: dict[int, defaultdict[str, float]] = {
        cutoff: defaultdict(float) for cutoff in cutoffs
    }
    target_occurrences = 0
    for record, prediction in zip(records, predictions):
        targets = [str(value) for value in record["acceptable_subcategories"]]
        ranked = [str(value) for value in prediction["ranked_labels"]]
        rank_by_label = {label: index + 1 for index, label in enumerate(ranked)}
        target_occurrences += len(targets)
        for target in targets:
            support[target] += 1
            rank = rank_by_label.get(target)
            if rank is None:
                continue
            for cutoff in cutoffs:
                if rank <= cutoff:
                    hits[cutoff][target] += 1
                    gains[cutoff][target] += 1.0 / math.log2(rank + 1)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"interest_macro_recall_at_{cutoff}"] = mean(
            hits[cutoff][label] / count for label, count in support.items()
        )
        metrics[f"interest_macro_ndcg_at_{cutoff}"] = mean(
            gains[cutoff][label] / count for label, count in support.items()
        )
    return metrics, target_occurrences, len(support)


def audit_semantic_id_token_condition(
    condition_dir: Path | str,
    eval_file: Path | str,
    mapping_file: Path | str,
    *,
    expected_records: int = 31_392,
    expected_targets: int = 37_728,
    expected_supported_interests: int = 104,
    expected_interest_count: int = 108,
    expected_train_samples: int = 21_600,
    expected_optimizer_steps: int = 2_025,
    expected_top_k: int = 20,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
    metric_tolerance: float = 1e-12,
) -> dict[str, object]:
    condition_dir = Path(condition_dir)
    eval_file = Path(eval_file)
    mapping_file = Path(mapping_file)
    training = _audit_training(
        condition_dir / "sft",
        expected_train_samples=expected_train_samples,
        expected_optimizer_steps=expected_optimizer_steps,
        expected_interest_count=expected_interest_count,
    )

    mapping_rows = load_interest_token_mapping(mapping_file)
    _require(
        len(mapping_rows) == expected_interest_count,
        "Mapping interest count mismatch",
    )
    labels = [str(row["target_subcategory"]) for row in mapping_rows]
    tokens = [str(row["interest_token"]) for row in mapping_rows]
    _require(
        len(set(labels)) == len(labels) and len(set(tokens)) == len(tokens),
        "Mapping labels or tokens are not unique",
    )
    token_by_label = dict(zip(labels, tokens))

    merged_dir = condition_dir / "eval" / "merged"
    prediction_path = merged_dir / "id_token_predictions.parquet"
    metrics_path = merged_dir / "metrics.json"
    per_interest_path = merged_dir / "per_interest_metrics.parquet"
    manifest_path = merged_dir / "merge_manifest.json"
    for path in (
        prediction_path,
        metrics_path,
        per_interest_path,
        manifest_path,
    ):
        _require(path.is_file(), f"Missing {path}")
    records = pq.read_table(eval_file).to_pylist()
    predictions = pq.read_table(prediction_path).to_pylist()
    _require(len(records) == expected_records, "Eval row count mismatch")
    _require(
        len(predictions) == expected_records,
        "Merged prediction row count mismatch",
    )

    identities: set[str] = set()
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        _require(
            int(prediction["record_index"]) == index,
            f"Record {index}: ordering/index mismatch",
        )
        _require(
            str(prediction["impression_id"]) == str(record["impression_id"]),
            f"Record {index}: impression ID mismatch",
        )
        _require(
            str(prediction["input_sha256"]) == record_input_sha256(record),
            f"Record {index}: input hash mismatch",
        )
        identities.add(str(prediction["evaluation_identity"]))
        ranked_labels = [str(value) for value in prediction["ranked_labels"]]
        ranked_tokens = [str(value) for value in prediction["ranked_tokens"]]
        ranked_scores = [float(value) for value in prediction["ranked_scores"]]
        _require(
            len(ranked_labels)
            == len(ranked_tokens)
            == len(ranked_scores)
            == expected_top_k,
            f"Record {index}: output is not Top-{expected_top_k}",
        )
        _require(
            len(set(ranked_labels)) == expected_top_k
            and len(set(ranked_tokens)) == expected_top_k,
            f"Record {index}: duplicate Top-{expected_top_k} output",
        )
        _require(
            all(label in token_by_label for label in ranked_labels),
            f"Record {index}: unknown ranked label",
        )
        _require(
            ranked_tokens == [token_by_label[label] for label in ranked_labels],
            f"Record {index}: token/label mapping mismatch",
        )
        _require(
            all(math.isfinite(score) for score in ranked_scores),
            f"Record {index}: non-finite ranked score",
        )
        _require(
            math.isfinite(float(prediction["restricted_entropy"])),
            f"Record {index}: non-finite entropy",
        )
    _require(len(identities) == 1, "Merged evaluation identity mismatch")

    recomputed, target_count, supported_count = _independent_interest_metrics(
        records,
        predictions,
        cutoffs,
    )
    _require(target_count == expected_targets, "Target occurrence mismatch")
    _require(
        supported_count == expected_supported_interests,
        "Supported-interest count mismatch",
    )
    remote_metrics = _load_json(metrics_path)
    _require(
        remote_metrics.get("history_prediction_filtering") is False,
        "History prediction filtering was enabled",
    )
    _require(
        int(remote_metrics["label_vocabulary_size"])
        == expected_interest_count,
        "Remote label-vocabulary size mismatch",
    )
    for key, value in recomputed.items():
        remote = float(remote_metrics[key])
        _require(
            abs(remote - value) <= metric_tolerance,
            f"Metric mismatch for {key}: remote={remote}, local={value}",
        )

    manifest = _load_json(manifest_path)
    _require(
        int(manifest["record_count"]) == expected_records,
        "Merge-manifest row count mismatch",
    )
    _require(
        str(manifest["merged_prediction_sha256"])
        == sha256_file(prediction_path),
        "Merged prediction hash mismatch",
    )
    _require(
        str(manifest["per_interest_metrics_sha256"])
        == sha256_file(per_interest_path),
        "Per-interest metric hash mismatch",
    )
    _require(
        str(manifest["metrics_sha256"]) == sha256_file(metrics_path),
        "Metrics hash mismatch",
    )

    return {
        "status": "passed",
        "condition_dir": str(condition_dir.resolve()),
        "training": training,
        "evaluation": {
            "records": expected_records,
            "target_occurrences": target_count,
            "supported_interest_count": supported_count,
            "label_vocabulary_size": expected_interest_count,
            "top_k": expected_top_k,
            "history_prediction_filtering": False,
            "evaluation_identity": next(iter(identities)),
            "metrics": recomputed,
            "prediction_sha256": sha256_file(prediction_path),
            "metrics_sha256": sha256_file(metrics_path),
            "per_interest_metrics_sha256": sha256_file(per_interest_path),
            "merge_manifest_sha256": sha256_file(manifest_path),
            "eval_sha256": sha256_file(eval_file),
            "mapping_sha256": sha256_file(mapping_file),
        },
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
        description="Independently audit a downloaded Semantic-ID condition."
    )
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    audit = audit_semantic_id_token_condition(
        args.condition_dir,
        args.eval_file,
        args.mapping_file,
    )
    _atomic_write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
