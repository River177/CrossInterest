from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from new_interest.experiments.audit_wang_stage_a_results import (
    _assert_close,
    _load_json,
    _ordered_rows,
    _sha256,
    recompute_ranking,
)


def _baseline_roots(root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for marker_path in root.rglob("full-validation_success.json"):
        marker = _load_json(marker_path)
        contract = marker["contract"]
        key = (str(contract["model_name"]), str(contract["dataset"]))
        if key in result:
            raise ValueError(f"Duplicate baseline identity {key}")
        result[key] = marker_path.parent
    return result


def audit_top20_readback(
    readback_root: Path | str,
    baseline_root: Path | str,
) -> dict[str, Any]:
    readback_root = Path(readback_root)
    baselines = _baseline_roots(Path(baseline_root))
    marker_paths = sorted(readback_root.rglob("top20_success.json"))
    if not marker_paths:
        raise ValueError("No Top-20 success markers found")
    jobs: list[dict[str, Any]] = []
    for marker_path in marker_paths:
        root = marker_path.parent
        marker = _load_json(marker_path)
        contract = _load_json(root / "run_contract.json")
        _assert_close(marker["contract"], contract, f"{root}.contract")
        if contract["top_k"] != 20 or contract["cutoffs"] != [1, 5, 10, 20]:
            raise ValueError(f"{root}: not a Top-20 contract")
        for relative, expected_hash in marker["artifact_sha256"].items():
            path = root / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"{root}: missing {relative}")
            if _sha256(path) != expected_hash:
                raise ValueError(f"{root}: hash mismatch for {relative}")

        metrics_path = root / "eval/ranking-primary-top20/metrics.json"
        predictions_path = (
            root / "eval/ranking-primary-top20/text_label_predictions.parquet"
        )
        metrics = _load_json(metrics_path)
        rows = _ordered_rows(predictions_path)
        recomputed = recompute_ranking(
            rows,
            cutoffs=tuple(int(value) for value in metrics["cutoffs"]),
        )
        for key, value in recomputed.items():
            _assert_close(metrics[key], value, f"{root}.metrics.{key}")
        if any(len(row["ranked_labels"]) != 20 for row in rows):
            raise ValueError(f"{root}: a prediction is not Top-20")

        identity = (str(contract["model_name"]), str(contract["dataset"]))
        if identity not in baselines:
            raise ValueError(f"{root}: no accepted Top-10 baseline for {identity}")
        baseline_view = (
            "ranking-supported"
            if contract["dataset"] == "mind"
            else "ranking-test-supported"
        )
        baseline_rows = _ordered_rows(
            baselines[identity]
            / "eval"
            / baseline_view
            / "text_label_predictions.parquet"
        )
        baseline_by_index = {
            int(row["record_index"]): row for row in baseline_rows
        }
        for row in rows:
            baseline = baseline_by_index[int(row["record_index"])]
            _assert_close(
                row["ranked_labels"][:10],
                baseline["ranked_labels"],
                f"{root}.top10_labels.{row['record_index']}",
            )
            _assert_close(
                row["ranked_scores"][:10],
                baseline["ranked_scores"],
                f"{root}.top10_scores.{row['record_index']}",
            )
            _assert_close(
                row["input_sha256"],
                baseline["input_sha256"],
                f"{root}.input_sha256.{row['record_index']}",
            )
        jobs.append(
            {
                "model_name": contract["model_name"],
                "model_revision": contract["model_revision"],
                "dataset": contract["dataset"],
                "samples": metrics["samples"],
                "adapter_sha256": contract["adapter_sha256"],
                "artifact_hash_count": len(marker["artifact_sha256"]),
                "top10_prefix_exact_rows": len(rows),
                "interest_macro_recall_at_10": metrics[
                    "interest_macro_recall_at_10"
                ],
                "interest_macro_ndcg_at_10": metrics[
                    "interest_macro_ndcg_at_10"
                ],
                "interest_macro_recall_at_20": metrics[
                    "interest_macro_recall_at_20"
                ],
                "interest_macro_ndcg_at_20": metrics[
                    "interest_macro_ndcg_at_20"
                ],
            }
        )
    return {
        "audit_version": "wang-stage-a-top20-local-readback-v1",
        "passed": True,
        "job_count": len(jobs),
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Wang Stage-A Top-20 results")
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_top20_readback(args.readback_root, args.baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
