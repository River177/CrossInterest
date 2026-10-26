from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    record_input_sha256,
    sha256_file,
)
from new_interest.evaluation.metrics import evaluate_ranked_outputs


RULE_VERSION = "static-sud-crossfit-real-independent-audit-v2"
DEFAULT_ARTIFACT_ROOT = Path(
    "artifacts/static-sud-crossfit-reweighting/qwen35-v1"
)
DEFAULT_DATA_DIR = Path(
    "data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1"
)
DEFAULT_LABELS_FILE = Path(
    "data/training/"
    "generative_new_interest_static_user_disjoint_balanced_200_h50_v1/"
    "retained_interests.parquet"
)
CONDITIONS = {
    "behavior-vanilla": "behavior_only",
    "behavior-weighted": "behavior_only",
    "interest-vanilla": "interest_only",
    "interest-weighted": "interest_only",
}
REMOTE_SOURCES = {
    "behavior-vanilla": {
        "source": {
            "experiment": "static-sud-crossfit-final-v1-0818",
            "job": "behavior-vanilla-final",
            "result_id": "7212788530.09261-77e8fede-8ba2",
            "status": "failed_during_evaluation_after_completed_training",
        },
        "recovery": {
            "experiment": "static-sud-crossfit-behavior-eval-recovery-v1-0821",
            "job": "behavior-vanilla-eval-recovery",
            "cloud_job_id": "good_cheetah_vr4plgsswq",
            "result_id": "7212738953.66495-4a25a0ef-9ccc",
            "status": "killed_by_user_after_real_merged",
        },
    },
    "behavior-weighted": {
        "source": {
            "experiment": "static-sud-crossfit-final-v1-0818",
            "job": "behavior-weighted-final",
            "result_id": "7212788530.09227-18411bb2-ac34",
            "status": "failed_during_evaluation_after_completed_training",
        },
        "recovery": {
            "experiment": "static-sud-crossfit-behavior-eval-recovery-v1-0821",
            "job": "behavior-weighted-eval-recovery",
            "cloud_job_id": "willing_quince_dbs2c8smwr",
            "result_id": "7212738953.66418-d30ebdc1-c487",
            "status": "killed_by_user_after_real_merged",
        },
    },
    "interest-vanilla": {
        "source": {
            "experiment": "static-sud-crossfit-final-v1-0818",
            "job": "interest-vanilla-final",
            "cloud_job_id": "honest_atemoya_r2htmht5gw",
            "result_id": "7212788530.09200-da8d2f7e-5551",
            "status": "passed",
        }
    },
    "interest-weighted": {
        "source": {
            "experiment": "static-sud-crossfit-final-v1-0818",
            "job": "interest-weighted-final",
            "cloud_job_id": "loving_cheese_9q0h80dssm",
            "result_id": "7212788530.09170-1390b448-a950",
            "status": "passed",
        }
    },
}


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{context}: expected a mapping")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{context}: missing key {key}")
            _assert_equal(actual[key], value, f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{context}: list mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equal(left, right, f"{context}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{context}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def _assert_finite(value: Any, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{context}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context}: non-finite value {value!r}")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _verify_training(
    condition: str,
    condition_root: Path,
    metrics: Mapping[str, object],
    *,
    expected_model_name: str,
    expected_world_size: int,
) -> dict[str, object]:
    source_root = condition_root / "source_result"
    summary_path = source_root / "sft" / "run_summary.json"
    environment_path = source_root / "environment.json"
    if not summary_path.is_file() or not environment_path.is_file():
        raise ValueError(f"{condition}: missing source summary/environment")
    json.loads(environment_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "train_samples": 21_600,
        "configured_epochs": 3.0,
        "expected_optimizer_steps": 2_025,
        "world_size": expected_world_size,
        "effective_global_batch_size": 32,
        "model_name": expected_model_name,
    }
    _assert_equal(summary, expected, f"{condition}.run_summary")
    _assert_finite(summary["train_metrics"], f"{condition}.train_metrics")
    if int(summary["loss_history"][-1]["step"]) != 2_025:
        raise ValueError(f"{condition}: loss history does not reach step 2025")
    _assert_finite(summary["loss_history"], f"{condition}.loss_history")

    final_adapter = source_root / "sft" / "final_adapter"
    recovery_root = condition_root / "recovery_result"
    if recovery_root.is_dir():
        recovery_environment = recovery_root / "environment.json"
        audit_path = recovery_root / "source_training_audit.json"
        if not recovery_environment.is_file() or not audit_path.is_file():
            raise ValueError(f"{condition}: missing recovery audit/environment")
        json.loads(recovery_environment.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        _assert_equal(
            audit,
            {
                "source_train_samples": 21_600,
                "source_checkpoint_step": 2_025,
                "source_expected_optimizer_steps": 2_025,
                "source_world_size": expected_world_size,
                "source_effective_global_batch_size": 32,
                "source_configured_epochs": 3.0,
            },
            f"{condition}.source_training_audit",
        )
        _assert_finite(
            audit["source_train_loss"], f"{condition}.source_train_loss"
        )
        if sha256_file(summary_path) != audit["source_run_summary_sha256"]:
            raise ValueError(f"{condition}: source run summary hash mismatch")
        for name, identity in audit["adapter_files"].items():
            path = final_adapter / name
            if not path.is_file():
                raise ValueError(f"{condition}: missing final adapter file {name}")
            if path.stat().st_size != int(identity["size"]):
                raise ValueError(f"{condition}: final adapter size mismatch {name}")
            if sha256_file(path) != identity["sha256"]:
                raise ValueError(f"{condition}: final adapter hash mismatch {name}")
        if audit["adapter_semantic_sha256"] != metrics["adapter_sha256"]:
            raise ValueError(f"{condition}: evaluation adapter identity mismatch")
        return {
            "source_training_audit": str(audit_path),
            "checkpoint_marker_sha256_remote_evidence": audit[
                "source_checkpoint_marker_sha256"
            ],
            "final_adapter_semantic_sha256": audit["adapter_semantic_sha256"],
        }

    artifact_manifest_path = source_root / "artifact_manifest.json"
    manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )["files"]
    for path in sorted(final_adapter.iterdir()):
        if path.is_file():
            relative = path.relative_to(source_root).as_posix()
            if sha256_file(path) != manifest[relative]:
                raise ValueError(f"{condition}: final adapter manifest mismatch")
    return {
        "source_artifact_manifest": str(artifact_manifest_path),
        "final_adapter_sha256": sha256_file(
            final_adapter / "adapter_model.safetensors"
        ),
    }


def _verify_artifact_manifest_subset(
    condition: str, condition_root: Path
) -> dict[str, object] | None:
    manifest_path = condition_root / "source_result" / "artifact_manifest.json"
    if not manifest_path.is_file():
        return None
    source_root = manifest_path.parent
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    missing = []
    verified = 0
    for relative, expected in sorted(files.items()):
        path = source_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"{condition}: artifact manifest mismatch: {relative}")
        verified += 1
    unexpected_missing = [
        relative
        for relative in missing
        if not relative.startswith("sft/checkpoints/")
        and not relative.startswith("eval/shuffled_")
    ]
    if unexpected_missing:
        raise ValueError(
            f"{condition}: unexpected missing manifest files: "
            f"{unexpected_missing[:5]}"
        )
    return {
        "entries": len(files),
        "verified_local_files": verified,
        "missing_local_files": len(missing),
        "missing_checkpoint_files": sum(
            relative.startswith("sft/checkpoints/") for relative in missing
        ),
        "missing_excluded_shuffled_files": sum(
            relative.startswith("eval/shuffled_") for relative in missing
        ),
        "mismatches": 0,
    }


def _condition_file_hashes(condition_root: Path) -> dict[str, str]:
    output = {}
    for base_name in ("source_result", "recovery_result"):
        base = condition_root / base_name
        if not base.is_dir():
            continue
        for path in sorted(value for value in base.rglob("*") if value.is_file()):
            output[path.relative_to(condition_root).as_posix()] = sha256_file(path)
    return output


def audit_crossfit_real_results(
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    labels_file: Path | str = DEFAULT_LABELS_FILE,
    *,
    remote_sources: Mapping[str, object] = REMOTE_SOURCES,
    expected_model_name: str = "Qwen/Qwen3.5-9B",
    expected_world_size: int = 8,
) -> Path:
    artifact_root = Path(artifact_root)
    data_dir = Path(data_dir)
    labels_file = Path(labels_file)
    final_root = artifact_root / "final"
    checkpoint_paths = list(final_root.glob("**/sft/checkpoints"))
    if checkpoint_paths:
        raise ValueError(f"Local checkpoints are prohibited: {checkpoint_paths}")

    labels_sha256 = sha256_file(labels_file)
    labels = {
        str(row["target_subcategory"])
        for row in pq.read_table(labels_file).to_pylist()
    }
    if len(labels) != 108:
        raise ValueError(f"Expected 108 vocabulary labels, got {len(labels)}")

    records_by_task = {
        task: pq.read_table(data_dir / "eval" / task / "real.parquet").to_pylist()
        for task in ("behavior_only", "interest_only")
    }
    paired_columns = ("impression_id", "user_id", "acceptable_subcategories")
    paired = {
        task: [
            (
                str(row["impression_id"]),
                str(row["user_id"]),
                tuple(str(value) for value in row["acceptable_subcategories"]),
            )
            for row in records
        ]
        for task, records in records_by_task.items()
    }
    if paired["behavior_only"] != paired["interest_only"]:
        raise ValueError("Behavior and Interest real source identities differ")
    if len(set(paired["behavior_only"])) != 31_392:
        raise ValueError("Real source identities are missing or duplicated")

    condition_audits: dict[str, dict[str, object]] = {}
    for condition, task in CONDITIONS.items():
        condition_root = final_root / condition
        merged_root = condition_root / "eval" / "real" / "merged"
        metrics_path = merged_root / "metrics.json"
        predictions_path = merged_root / "beam_predictions.parquet"
        per_interest_path = merged_root / "per_interest_metrics.parquet"
        merge_manifest_path = merged_root / "merge_manifest.json"
        for path in (
            metrics_path,
            predictions_path,
            per_interest_path,
            merge_manifest_path,
        ):
            if not path.is_file():
                raise ValueError(f"{condition}: missing {path}")

        records = records_by_task[task]
        rows = pq.read_table(predictions_path).to_pylist()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        merge_manifest = json.loads(
            merge_manifest_path.read_text(encoding="utf-8")
        )
        _assert_finite(metrics, f"{condition}.metrics")
        if len(rows) != 31_392:
            raise ValueError(f"{condition}: expected 31,392 predictions")
        by_index = {int(row["record_index"]): row for row in rows}
        if len(by_index) != len(rows) or set(by_index) != set(range(31_392)):
            raise ValueError(f"{condition}: missing/duplicate prediction indices")
        ordered = [by_index[index] for index in range(31_392)]
        if len({str(row["impression_id"]) for row in ordered}) != 31_392:
            raise ValueError(f"{condition}: duplicate impression identities")

        identities = {str(row["evaluation_identity"]) for row in ordered}
        if identities != {
            str(metrics["evaluation_identity"]),
        } or identities != {str(merge_manifest["evaluation_identity"])}:
            raise ValueError(f"{condition}: evaluation identity mismatch")
        if int(merge_manifest["num_shards"]) != 8:
            raise ValueError(f"{condition}: expected eight logical shards")
        if int(merge_manifest["record_count"]) != 31_392:
            raise ValueError(f"{condition}: merge record count mismatch")
        if metrics["labels_sha256"] != labels_sha256:
            raise ValueError(f"{condition}: vocabulary hash mismatch")
        eval_path = data_dir / "eval" / task / "real.parquet"
        if metrics["eval_sha256"] != sha256_file(eval_path):
            raise ValueError(f"{condition}: eval input hash mismatch")
        for key, name in (
            ("merged_prediction_sha256", "beam_predictions.parquet"),
            ("per_interest_metrics_sha256", "per_interest_metrics.parquet"),
            ("metrics_sha256", "metrics.json"),
        ):
            if sha256_file(merged_root / name) != merge_manifest[key]:
                raise ValueError(f"{condition}: merged artifact hash mismatch")

        ranked_outputs = []
        for record, row in zip(records, ordered):
            if str(record["impression_id"]) != str(row["impression_id"]):
                raise ValueError(f"{condition}: impression identity mismatch")
            if row["input_sha256"] != record_input_sha256(record):
                raise ValueError(f"{condition}: record input hash mismatch")
            expected_targets = [
                str(value) for value in record["acceptable_subcategories"]
            ]
            if [str(value) for value in row["acceptable_subcategories"]] != expected_targets:
                raise ValueError(f"{condition}: target identity mismatch")
            ranked_labels = [str(value) for value in row["ranked_labels"]]
            if len(ranked_labels) != 20 or len(set(ranked_labels)) != 20:
                raise ValueError(f"{condition}: prediction is not unique Top-20")
            if not set(ranked_labels).issubset(labels):
                raise ValueError(f"{condition}: prediction is outside vocabulary")
            ranked_scores = [float(value) for value in row["ranked_scores"]]
            if not all(math.isfinite(value) for value in ranked_scores):
                raise ValueError(f"{condition}: non-finite prediction score")
            ranked_outputs.append(
                [
                    {"label": label, "score": score}
                    for label, score in zip(ranked_labels, ranked_scores)
                ]
            )

        recomputed, details = evaluate_ranked_outputs(records, ranked_outputs)
        _assert_equal(metrics, recomputed, f"{condition}.recomputed_metrics")
        for index, (stored, detail) in enumerate(zip(ordered, details)):
            _assert_equal(stored, detail, f"{condition}.row[{index}]")

        per_interest = {
            str(row["interest"]): row
            for row in pq.read_table(per_interest_path).to_pylist()
        }
        supported = set(recomputed["per_class_support"])
        if len(supported) != 104 or set(per_interest) != supported:
            raise ValueError(f"{condition}: expected 104 supported interests")
        for label, row in per_interest.items():
            expected = {"support": recomputed["per_class_support"][label]}
            for cutoff in (1, 5, 10, 20):
                expected[f"recall_at_{cutoff}"] = recomputed[
                    "per_class_recall"
                ][str(cutoff)][label]
                expected[f"ndcg_at_{cutoff}"] = recomputed[
                    "per_class_ndcg"
                ][str(cutoff)][label]
            _assert_equal(row, expected, f"{condition}.per_interest.{label}")

        training = _verify_training(
            condition,
            condition_root,
            metrics,
            expected_model_name=expected_model_name,
            expected_world_size=expected_world_size,
        )
        remote_manifest = _verify_artifact_manifest_subset(
            condition, condition_root
        )
        condition_audits[condition] = {
            "predictions": len(rows),
            "unique_record_identities": len(rows),
            "top_k": 20,
            "logical_shards": 8,
            "supported_interests": len(supported),
            "evaluation_identity": metrics["evaluation_identity"],
            "metrics_recomputed_from_predictions": True,
            "merged_manifest_verified": True,
            "environment_verified": True,
            "training": training,
            "source_artifact_manifest_subset": remote_manifest,
            "metrics": {
                key: metrics[key]
                for key in (
                    "hit_at_1",
                    "recall_at_10",
                    "ndcg_at_10",
                    "interest_macro_recall_at_10",
                    "interest_macro_ndcg_at_10",
                    "recall_at_20",
                    "interest_macro_recall_at_20",
                    "interest_macro_ndcg_at_20",
                    "ndcg_at_20",
                    "unique_top1_labels",
                )
            },
        }

        local_files = _condition_file_hashes(condition_root)
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "condition": condition,
            "analysis_evaluation": "real",
            "remote_sources": remote_sources[condition],
            "recovery_job_passed": (
                False
                if (condition_root / "recovery_result").is_dir()
                else None
            ),
            "shuffled_controls_used": False,
            "checkpoint_files_downloaded": 0,
            "checkpoint_policy": (
                "No local checkpoint files by explicit user request; "
                "only remote marker/hash evidence is retained."
            ),
            "analysis_link": (
                os.readlink(condition_root / "eval" / "real")
                if (condition_root / "eval" / "real").is_symlink()
                else str(condition_root / "eval" / "real")
            ),
            "local_file_sha256": local_files,
        }
        _write_json(condition_root / "local_provenance_manifest.json", provenance)

    metrics = {
        condition: audit["metrics"] for condition, audit in condition_audits.items()
    }
    deltas = {}
    for view in ("behavior", "interest"):
        left = metrics[f"{view}-weighted"]
        right = metrics[f"{view}-vanilla"]
        deltas[f"{view}_weighted_minus_vanilla"] = {
            key: float(left[key]) - float(right[key])
            for key in (
                "hit_at_1",
                "recall_at_10",
                "ndcg_at_10",
                "interest_macro_recall_at_10",
                "interest_macro_ndcg_at_10",
                "recall_at_20",
                "interest_macro_recall_at_20",
                "interest_macro_ndcg_at_20",
                "ndcg_at_20",
                "unique_top1_labels",
            )
        }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule_version": RULE_VERSION,
        "evaluation_scope": "real_only",
        "exploratory_dev": True,
        "shuffled_controls_used": False,
        "checkpoint_files_downloaded": 0,
        "model_name": expected_model_name,
        "paired_behavior_interest_source_identities": True,
        "conditions": condition_audits,
        "weighted_minus_vanilla": deltas,
    }
    output = artifact_root / "verification" / "real_results_audit.json"
    _write_json(output, result)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--remote-sources-file", type=Path)
    parser.add_argument(
        "--expected-model-name", default="Qwen/Qwen3.5-9B"
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    remote_sources = REMOTE_SOURCES
    if args.remote_sources_file is not None:
        remote_sources = json.loads(
            args.remote_sources_file.read_text(encoding="utf-8")
        )
    print(
        audit_crossfit_real_results(
            args.artifact_root,
            args.data_dir,
            args.labels_file,
            remote_sources=remote_sources,
            expected_model_name=args.expected_model_name,
            expected_world_size=args.expected_world_size,
        )
    )


if __name__ == "__main__":
    main()
