#!/usr/bin/env python3
"""Independently verify downloaded MeRA Full artifacts and recompute P/R@1-4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


QUOTED_VALUE_RE = re.compile(r'"([^"]+)"')
JOB_ORDER = (
    "arts-code-full",
    "arts-paper-full",
    "grocery-code-full",
    "grocery-paper-full",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quoted_values(value: str) -> list[str]:
    return [f'"{item}"' for item in QUOTED_VALUE_RE.findall(value)]


def categories_from_map(path: Path) -> set[str]:
    categories: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != 2:
            raise RuntimeError(f"{path}:{line_number}: malformed category mapping")
        categories.add(f'"{parts[0]}"')
    return categories


def assert_close_list(actual: list[float], expected: list[float], label: str) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: length mismatch")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"{label}[{index}]: {left} != {right}")


def verify_run(
    job_root: Path,
    source_root: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = job_root / "artifact_manifest.json"
    success_path = job_root / "full_success.json"
    run_root = job_root / "run"
    manifest = load_json(manifest_path)
    success = load_json(success_path)
    if success["status"] != "pass" or success["mode"] != "full":
        raise RuntimeError(f"{job_root.name}: invalid success marker")
    if sha256_file(manifest_path) != success["artifact_manifest_sha256"]:
        raise RuntimeError(f"{job_root.name}: manifest digest mismatch")
    if manifest["identity"] != success["identity"]:
        raise RuntimeError(f"{job_root.name}: identity mismatch")
    if success["artifact_files"] != len(manifest["files"]):
        raise RuntimeError(f"{job_root.name}: artifact count mismatch")
    verified_files = 0
    for relative, expected_digest in sorted(manifest["files"].items()):
        local_path = run_root / relative
        if not local_path.is_file():
            raise RuntimeError(f"{job_root.name}: missing {relative}")
        actual_digest = sha256_file(local_path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"{job_root.name}: SHA-256 mismatch for {relative}: "
                f"{actual_digest} != {expected_digest}"
            )
        verified_files += 1

    identity = manifest["identity"]
    dataset = identity["dataset"]
    contract = identity["contract"]
    dataset_manifest = source_manifest["datasets"][dataset]
    if identity["dataset_files"] != dataset_manifest["files"]:
        raise RuntimeError(f"{job_root.name}: source data identity mismatch")
    dataset_root = source_root / dataset_manifest["directory"]
    for filename, expected_digest in dataset_manifest["files"].items():
        if sha256_file(dataset_root / filename) != expected_digest:
            raise RuntimeError(f"{job_root.name}: local source mismatch for {filename}")

    legal_categories = categories_from_map(dataset_root / "layer2_map.txt")
    test_rows = load_json(dataset_root / "test.json")
    predictions_path = run_root / "evaluation" / "predictions.jsonl"
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_rows = int(dataset_manifest["rows"]["test"])
    if len(test_rows) != expected_rows or len(predictions) != expected_rows:
        raise RuntimeError(f"{job_root.name}: Test row count mismatch")

    precision_sums = [0.0, 0.0, 0.0, 0.0]
    recall_sums = [0.0, 0.0, 0.0, 0.0]
    normalized_rows: list[dict[str, Any]] = []
    for row_index, (source_row, prediction) in enumerate(zip(test_rows, predictions)):
        if prediction["row_index"] != row_index:
            raise RuntimeError(f"{job_root.name}: row_index mismatch at {row_index}")
        predicted = prediction["top4_categories"]
        targets = prediction["target_categories"]
        expected_targets = quoted_values(str(source_row["output"]))
        expected_history = QUOTED_VALUE_RE.findall(str(source_row["input"]))
        if targets != expected_targets:
            raise RuntimeError(f"{job_root.name}: target mismatch at {row_index}")
        if prediction["history_categories"] != expected_history:
            raise RuntimeError(f"{job_root.name}: history mismatch at {row_index}")
        if len(predicted) != 4 or len(set(predicted)) != 4:
            raise RuntimeError(f"{job_root.name}: invalid Top-4 at {row_index}")
        if not set(predicted).issubset(legal_categories):
            raise RuntimeError(f"{job_root.name}: unknown prediction at {row_index}")
        if len(prediction["top4_token_ids"]) != 4 or len(
            set(prediction["top4_token_ids"])
        ) != 4:
            raise RuntimeError(f"{job_root.name}: invalid token IDs at {row_index}")
        logits = [float(value) for value in prediction["top4_logits"]]
        if len(logits) != 4 or any(not math.isfinite(value) for value in logits):
            raise RuntimeError(f"{job_root.name}: invalid logits at {row_index}")
        if any(left < right for left, right in zip(logits, logits[1:])):
            raise RuntimeError(f"{job_root.name}: unsorted logits at {row_index}")

        target_set = set(targets)
        row_precision: list[float] = []
        row_recall: list[float] = []
        for offset in range(4):
            hits = len(set(predicted[: offset + 1]) & target_set)
            p_value = hits / (offset + 1)
            r_value = hits / len(target_set)
            precision_sums[offset] += p_value
            recall_sums[offset] += r_value
            row_precision.append(p_value)
            row_recall.append(r_value)
        assert_close_list(
            row_precision,
            prediction["precision_at_k"],
            f"{job_root.name}: row {row_index} precision",
        )
        assert_close_list(
            row_recall,
            prediction["recall_at_k"],
            f"{job_root.name}: row {row_index} recall",
        )
        normalized_rows.append(
            {
                "row_index": row_index,
                "history_categories": expected_history,
                "target_categories": targets,
                "top4_categories": predicted,
                "hits_at_4": len(set(predicted) & target_set),
            }
        )

    recomputed_precision = [value / expected_rows for value in precision_sums]
    recomputed_recall = [value / expected_rows for value in recall_sums]
    metrics = load_json(run_root / "evaluation" / "metrics.json")
    if metrics["examples"] != expected_rows:
        raise RuntimeError(f"{job_root.name}: metrics row count mismatch")
    if metrics["invalid_predictions"] != 0 or metrics["duplicate_predictions"] != 0:
        raise RuntimeError(f"{job_root.name}: evaluator reported invalid predictions")
    assert_close_list(
        recomputed_precision,
        metrics["precision_at_k"],
        f"{job_root.name}: aggregate precision",
    )
    assert_close_list(
        recomputed_recall,
        metrics["recall_at_k"],
        f"{job_root.name}: aggregate recall",
    )

    reload_audit = load_json(run_root / "evaluation" / "reload_audit.json")
    required_reload_flags = (
        "saved_adapter_state_exact",
        "live_adapter_keys_exact",
        "live_adapter_state_exact_after_dtype_cast",
        "repeat_live_state_exact_after_dtype_cast",
        "fresh_repeat_top4_exact",
    )
    if not all(reload_audit.get(flag) is True for flag in required_reload_flags):
        raise RuntimeError(f"{job_root.name}: reload gate failed")
    if float(reload_audit["fresh_repeat_max_abs_error"]) > float(
        reload_audit["fresh_repeat_tolerance"]
    ):
        raise RuntimeError(f"{job_root.name}: fresh reload logits exceed tolerance")

    run_summary = load_json(run_root / "run_summary.json")
    actual_steps = int(run_summary["actual_optimizer_steps"])
    maximum_steps = int(run_summary["configured_max_optimizer_steps"])
    if not 0 < actual_steps <= maximum_steps:
        raise RuntimeError(f"{job_root.name}: invalid optimizer step count")

    author_precision = dataset_manifest["author_predictions"]["precision_at_k"]
    author_recall = dataset_manifest["author_predictions"]["recall_at_k"]
    return {
        "job": job_root.name,
        "dataset": dataset,
        "contract": contract,
        "status": "verified",
        "verified_manifest_files": verified_files,
        "prediction_rows": expected_rows,
        "row_indices_complete_unique": True,
        "top4_legal_unique": True,
        "targets_match_frozen_test": True,
        "metrics_exact": True,
        "precision_at_k": recomputed_precision,
        "recall_at_k": recomputed_recall,
        "delta_vs_author_precision_at_k": [
            value - author for value, author in zip(recomputed_precision, author_precision)
        ],
        "delta_vs_author_recall_at_k": [
            value - author for value, author in zip(recomputed_recall, author_recall)
        ],
        "actual_optimizer_steps": actual_steps,
        "configured_max_optimizer_steps": maximum_steps,
        "stopped_early": bool(run_summary["stopped_early"]),
        "best_metric": float(run_summary["best_metric"]),
        "best_model_checkpoint": run_summary["best_model_checkpoint"],
        "fresh_repeat_max_abs_error": float(
            reload_audit["fresh_repeat_max_abs_error"]
        ),
        "fresh_repeat_top4_exact": bool(reload_audit["fresh_repeat_top4_exact"]),
        "trainer_to_fresh_max_abs_error_record_only": float(
            reload_audit["trainer_to_fresh_max_abs_error"]
        ),
        "rows": normalized_rows,
    }


def select_case(
    code_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    *,
    prefer: str,
) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]] = []
    for code_row, paper_row in zip(code_rows, paper_rows):
        if code_row["row_index"] != paper_row["row_index"]:
            raise RuntimeError("Contract prediction rows are not aligned")
        if len(code_row["target_categories"]) > 4:
            continue
        delta = paper_row["hits_at_4"] - code_row["hits_at_4"]
        signed = delta if prefer == "paper" else -delta
        if signed > 0:
            candidates.append(
                (
                    signed,
                    -len(code_row["target_categories"]),
                    -code_row["row_index"],
                    code_row,
                    paper_row,
                )
            )
    if not candidates:
        return None
    _, _, _, code_row, paper_row = max(candidates)
    return {
        "selection_rule": (
            f"largest_{prefer}_hit_at_4_advantage_among_rows_with_at_most_4_targets_"
            "then_fewest_targets_then_lowest_row_index"
        ),
        "row_index": code_row["row_index"],
        "history_categories": code_row["history_categories"],
        "target_categories": code_row["target_categories"],
        "code_faithful_top4": code_row["top4_categories"],
        "code_faithful_hits_at_4": code_row["hits_at_4"],
        "paper_spec_top4": paper_row["top4_categories"],
        "paper_spec_hits_at_4": paper_row["hits_at_4"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readback-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    readback_root = Path(args.readback_root).resolve()
    source_root = Path(args.source_root).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    source_manifest = load_json(source_manifest_path)
    runs = [
        verify_run(readback_root / job, source_root, source_manifest)
        for job in JOB_ORDER
    ]
    cases: dict[str, Any] = {}
    for dataset in ("arts", "grocery"):
        code = next(
            run for run in runs if run["dataset"] == dataset and run["contract"] == "code-faithful"
        )
        paper = next(
            run for run in runs if run["dataset"] == dataset and run["contract"] == "paper-spec"
        )
        cases[dataset] = {
            "paper_better": select_case(code["rows"], paper["rows"], prefer="paper"),
            "code_better": select_case(code["rows"], paper["rows"], prefer="code"),
        }
    for run in runs:
        del run["rows"]
    payload = {
        "audit_version": "mera-original-full-local-readback-v1",
        "experiment": readback_root.name,
        "status": "verified",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "runs": runs,
        "illustrative_cases": cases,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "MERA_ORIGINAL_FULL_READBACK_OK",
        json.dumps(
            {
                run["job"]: {
                    "steps": run["actual_optimizer_steps"],
                    "precision_at_k": run["precision_at_k"],
                    "recall_at_k": run["recall_at_k"],
                }
                for run in runs
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
