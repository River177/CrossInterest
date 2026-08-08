from __future__ import annotations

import json
from pathlib import Path

from new_interest.experiments.report_22k_results import collect_results


def test_collect_results_reports_missing_and_complete(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    output = tmp_path / "output"
    first = root / "qwen3-interest-22k"
    (first / "sft").mkdir(parents=True)
    (first / "beam").mkdir()
    (first / "sft/run_summary.json").write_text(
        json.dumps(
            {
                "train_samples": 22000,
                "train_metrics": {
                    "train_loss": 0.5,
                    "train_runtime": 10.0,
                },
                "token_audit": {
                    "max_tokens": 500,
                    "p99_tokens": 480,
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "recall_at_1": 0.1,
        "recall_at_5": 0.2,
        "recall_at_10": 0.3,
        "recall_at_20": 0.4,
        "ndcg_at_1": 0.1,
        "ndcg_at_5": 0.2,
        "ndcg_at_10": 0.3,
        "ndcg_at_20": 0.4,
        "hit_at_20": 0.5,
        "mrr_at_20": 0.2,
        "ranking_length": 20.0,
        "runtime_seconds": 100.0,
        "samples_per_second": 1.0,
    }
    (first / "beam/metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    result = collect_results(root, output)

    assert result["complete"] is False
    assert len(result["conditions"]) == 1
    assert len(result["missing_jobs"]) == 3
    assert (output / "qwen_llama_22k_results.csv").is_file()
