from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.report_future_interest_baselines import (
    build_future_interest_report,
)


METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_20",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
    "mrr_at_20",
)
TRACKS = ("exposure_restricted", "full_taxonomy")


def _write_fixture(root: Path, *, mismatch: bool = False) -> Path:
    vocabulary = root / "vocabulary.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"subcategory": label} for label in ("a", "b", "c")]
        ),
        vocabulary,
    )
    conditions = []
    for condition_id, value in (("global_popularity", 0.0), ("better", 1.0)):
        tracks = {}
        for track in TRACKS:
            rows = []
            ids = ("u1", "u2")
            if mismatch and condition_id == "better" and track == TRACKS[0]:
                ids = ("u1", "u3")
            for example_id in ids:
                row = {
                    "example_id": example_id,
                    "positive_labels": ["a"],
                }
                row.update({metric: value for metric in METRICS})
                rows.append(row)
            details_path = root / f"{condition_id}-{track}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), details_path)
            metrics = {
                "track": track,
                "samples": 2,
                "vocabulary_path": str(vocabulary),
                "per_label_support": {"a": 2},
                "per_label_recall": {
                    str(cutoff): {"a": value, "b": 0.0, "c": 0.0}
                    for cutoff in (1, 5, 10, 20)
                },
                "macro_recall_at_k": {str(cutoff): value for cutoff in (1, 5, 10, 20)},
            }
            metrics.update({metric: value for metric in METRICS})
            metrics_path = root / f"{condition_id}-{track}.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            tracks[track] = {
                "metrics_path": str(metrics_path),
                "details_path": str(details_path),
            }
        conditions.append({"baseline_id": condition_id, "tracks": tracks})
    manifest = {
        "rule_version": "test",
        "benchmark_dir": "data/benchmark",
        "artifact_dir": str(root),
        "bootstrap": {
            "reference_baseline": "global_popularity",
            "replicates": 2000,
            "seed": 7,
            "confidence": 0.95,
        },
        "conditions": conditions,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_builds_track_tables_and_paired_bootstrap(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    result = build_future_interest_report(manifest, tmp_path / "report")

    assert len(result["rows"]) == 4
    delta = result["bootstrap_deltas"][0]["result"]["metrics"]["recall_at_1"]
    assert delta["observed_delta"] == 1.0
    assert result["label_support"]["exposure_restricted"] == {
        "a": 2,
        "b": 0,
        "c": 0,
    }
    assert (
        tmp_path
        / "report/future_novel_interest_baselines_metrics.csv"
    ).is_file()
    payload = json.loads(
        (
            tmp_path
            / "report/future_novel_interest_baselines_metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["bootstrap_deltas"]
    assert payload["label_support"]["full_taxonomy"]["a"] == 2
    assert {row["tier"] for row in payload["label_tier_results"]} == {
        "head",
        "torso",
        "tail",
    }
    report = (
        tmp_path
        / "report/future_novel_interest_baselines_generated.md"
    ).read_text(encoding="utf-8")
    assert "exposure_restricted" in report
    assert "full_taxonomy" in report
    assert "Never-exposed unknown" in report
    assert "no causal claim" in report
    assert "no LLM judge" in report


def test_requires_complete_tracks_and_paired_user_ids(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="identical evaluated user IDs"):
        build_future_interest_report(manifest, tmp_path / "report")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["conditions"][0]["tracks"]["full_taxonomy"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="tracks must be exactly"):
        build_future_interest_report(manifest, tmp_path / "report-2")


def test_protects_existing_report_outputs(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    output = tmp_path / "report"
    build_future_interest_report(manifest, output)
    with pytest.raises(FileExistsError, match="Use overwrite=True"):
        build_future_interest_report(manifest, output)


def test_overwrite_replaces_complete_report_directory(
    tmp_path: Path,
) -> None:
    manifest = _write_fixture(tmp_path)
    output = tmp_path / "report"
    build_future_interest_report(manifest, output)
    (output / "stale.txt").write_text("old generation", encoding="utf-8")

    build_future_interest_report(manifest, output, overwrite=True)

    assert not (output / "stale.txt").exists()
    assert {path.name for path in output.iterdir()} == {
        "future_novel_interest_baselines_metrics.json",
        "future_novel_interest_baselines_metrics.csv",
        "future_novel_interest_baselines_generated.md",
    }
