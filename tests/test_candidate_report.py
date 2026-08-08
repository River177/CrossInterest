from __future__ import annotations

import json
from pathlib import Path

import new_interest.experiments.report_candidate_results as report


def test_candidate_report_supports_partial_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "samples": 10,
                "strict_accuracy": 0.2,
                "category_accuracy": 0.3,
                "candidate_valid_rate": 0.9,
                "prefix_compliance_rate": 0.8,
                "empty_output_rate": 0.1,
                "multiple_candidates_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report,
        "DEFAULT_RESULTS",
        {
            "qwen_base_behavior": str(metrics_path),
            "qwen_base_interest": str(tmp_path / "missing.json"),
        },
    )
    rows = report.build_report(
        tmp_path / "report",
        require_complete=False,
    )
    assert len(rows) == 1
    assert rows[0]["strict_accuracy"] == 0.2
    assert (tmp_path / "report/comparison.csv").is_file()
    assert "missing.json" in (
        tmp_path / "report/report.md"
    ).read_text()
