from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.run_future_interest_baselines import (
    run_future_interest_baseline_suite,
)


class FakeEncoder:
    def encode(self, texts, *, batch_size):
        vectors = []
        for text in texts:
            value = float(sum(ord(character) for character in text) % 17 + 1)
            vectors.append([value, 1.0, value / 2.0])
        return vectors


class FailingTopicEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, *, batch_size):
        self.calls += 1
        if self.calls == 4:
            raise RuntimeError("topic encoding failed")
        return super().encode(texts, batch_size=batch_size)


def _write_fixture(root: Path) -> Path:
    benchmark = root / "benchmark"
    benchmark.mkdir()
    vocabulary = [
        {"interest_id": 0, "category": "A", "subcategory": "a"},
        {"interest_id": 1, "category": "B", "subcategory": "b"},
        {"interest_id": 2, "category": "C", "subcategory": "c"},
    ]
    pq.write_table(
        pa.Table.from_pylist(vocabulary),
        benchmark / "interest_vocabulary.parquet",
    )
    rows = [
        {
            "user_id": f"u{index}",
            "history_titles": [f"title {index}", "shared"],
            "history_subcategories": ["a"],
            "history_unique_subcategories": ["a"],
            "future_novel_subcategories": ["b"] if index % 2 == 0 else ["c"],
            "future_exposed_subcategories": ["b", "c"],
            "future_exposed_unclicked_subcategories": (
                ["c"] if index % 2 == 0 else ["b"]
            ),
            "history_length": 2,
            "profile_group_size": 1,
        }
        for index in range(12)
    ]
    for split, selected in (
        ("train", rows[:8]),
        ("validation", rows[8:10]),
        ("test", rows[10:]),
    ):
        pq.write_table(
            pa.Table.from_pylist(selected),
            benchmark / f"{split}.parquet",
        )
    config = {
        "rule_version": "test-v1",
        "benchmark_dir": str(benchmark),
        "artifact_dir": str(root / "artifacts"),
        "cutoffs": [1, 2],
        "bootstrap": {
            "reference_baseline": "global_popularity",
            "replicates": 2000,
            "seed": 7,
            "confidence": 0.95
        },
        "statistical": {"alpha": 1.0},
        "embedding": {
            "model_name": "fake",
            "revision": "fake-revision",
            "device": None,
            "batch_size": 4,
            "metric": "cosine"
        },
        "multilabel": {
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "patience": 1,
            "batch_size": 2,
            "seed": 7,
            "device": "cpu",
            "prior_alpha": 1.0
        }
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_runs_all_baselines_and_both_tracks(tmp_path: Path) -> None:
    manifest = run_future_interest_baseline_suite(
        _write_fixture(tmp_path),
        encoder=FakeEncoder(),
    )

    assert len(manifest["conditions"]) == 6
    assert {
        condition["baseline_id"]
        for condition in manifest["conditions"]
    } == {
        "global_popularity",
        "conditional_frequency",
        "conditional_lift",
        "embedding_cosine",
        "multilabel_topic_logistic",
        "multilabel_embedding_logistic",
    }
    for condition in manifest["conditions"]:
        assert set(condition["tracks"]) == {
            "exposure_restricted",
            "full_taxonomy",
        }
        for track in condition["tracks"].values():
            assert Path(track["metrics_path"]).is_file()
    assert (tmp_path / "artifacts/manifest.json").is_file()


def test_failed_suite_does_not_publish_partial_artifacts(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    try:
        run_future_interest_baseline_suite(
            config,
            encoder=FailingTopicEncoder(),
        )
    except RuntimeError as error:
        assert "topic encoding failed" in str(error)
    else:
        raise AssertionError("Expected the injected encoder failure")
    assert not (tmp_path / "artifacts").exists()


def test_overwrite_replaces_the_complete_suite_directory(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    run_future_interest_baseline_suite(
        config,
        encoder=FakeEncoder(),
    )
    artifacts = tmp_path / "artifacts"
    (artifacts / "stale.txt").write_text("old generation", encoding="utf-8")

    manifest = run_future_interest_baseline_suite(
        config,
        overwrite=True,
        encoder=FakeEncoder(),
    )

    assert not (artifacts / "stale.txt").exists()
    for condition in manifest["conditions"]:
        assert ".future-interest-baseline-suite-" not in str(condition)
        assert Path(condition["predictions_path"]).is_file()
