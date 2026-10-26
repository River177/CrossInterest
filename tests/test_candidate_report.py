from __future__ import annotations

import csv
import json
from pathlib import Path

import new_interest.experiments.report_candidate_results as report
import pytest


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
                "per_category_accuracy": {
                    "Golf": 0.2,
                    "Music": 0.4,
                },
                "per_category_support": {
                    "Golf": 5,
                    "Music": 5,
                },
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
    assert rows[0]["macro_category_accuracy"] == pytest.approx(0.3)
    assert rows[0]["macro_category_count"] == 2
    assert (tmp_path / "report/comparison.csv").is_file()
    assert "missing.json" in (
        tmp_path / "report/report.md"
    ).read_text()


def test_candidate_report_supports_extended_manifest(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "extended-metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "samples": 20,
                "strict_accuracy": 0.6,
                "category_accuracy": 0.7,
                "macro_category_accuracy": 0.65,
                "macro_category_count": 4,
                "per_category_accuracy": {
                    "Golf": 0.5,
                    "Music": 0.8,
                },
                "candidate_valid_rate": 0.9,
                "prefix_compliance_rate": 0.85,
                "empty_output_rate": 0.05,
                "multiple_candidates_rate": 0.1,
                "request_failures": 2,
                "enable_thinking": True,
                "generation_stats": {
                    "reached_max_new_tokens_count": 3,
                    "reached_max_new_tokens_rate": 0.15,
                    "missing_final_answer_count": 1,
                    "missing_final_answer_rate": 0.05,
                    "think_block_count": 18,
                    "think_block_rate": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "conditions.json"
    manifest_path.write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "condition_id": "qwen35-thinking-sampling",
                        "model": "Qwen3.5-9B",
                        "input": "behavior_only",
                        "thinking_reasoning": "Thinking enabled",
                        "decoding": "Sampling",
                        "metrics_path": str(metrics_path),
                        "source_experiment": "candidate-qwen35-v2",
                        "source_job": "qwen35-thinking-behavior",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = report.build_report(
        tmp_path / "report",
        manifest_path=manifest_path,
    )

    assert rows[0]["condition_id"] == "qwen35-thinking-sampling"
    assert rows[0]["thinking_reasoning"] == "Thinking enabled"
    assert rows[0]["decoding"] == "Sampling"
    assert rows[0]["request_failures"] == 2
    assert rows[0]["truncation_count"] == 3
    assert rows[0]["missing_final_answer_rate"] == pytest.approx(0.05)
    assert rows[0]["enable_thinking"] is True
    assert rows[0]["source_job"] == "qwen35-thinking-behavior"

    markdown = (tmp_path / "report/report.md").read_text()
    assert "Thinking/Reasoning" in markdown
    assert "failures: 2; truncated: 3 (15.00%)" in markdown
    with (tmp_path / "report/comparison.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["source_experiment"] == "candidate-qwen35-v2"
    assert csv_row["metrics_path"] == str(metrics_path.resolve())
