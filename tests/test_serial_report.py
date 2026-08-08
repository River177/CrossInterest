from __future__ import annotations

import json
from pathlib import Path

from new_interest.report_serial_experiments import build_report


def test_build_report_writes_comparison_files(tmp_path: Path) -> None:
    state = {
        "experiments": {
            "interest_only-1k": {
                "task_type": "interest_only",
                "size": "1k",
                "metrics_10k": {
                    "recall_at_1": 0.1,
                    "recall_at_5": 0.2,
                    "recall_at_10": 0.3,
                    "recall_at_20": 0.4,
                    "ndcg_at_1": 0.1,
                    "ndcg_at_5": 0.15,
                    "ndcg_at_10": 0.2,
                    "ndcg_at_20": 0.25,
                    "hit_at_20": 0.5,
                    "mrr_at_20": 0.2,
                    "runtime_seconds": 10.0,
                    "samples_per_second": 1.0,
                },
            }
        },
        "best_models": {"interest_only": "interest_only-1k"},
        "completed_at": "now",
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    output = tmp_path / "report"

    summary = build_report(state_path, output)

    assert len(summary["experiments"]) == 1
    assert (output / "dev10k_metrics.csv").is_file()
    assert (output / "dev10k_metrics.json").is_file()
    assert "interest_only-1k" in (output / "report.md").read_text()
