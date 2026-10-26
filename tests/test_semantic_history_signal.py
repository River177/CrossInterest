from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.semantic_history_signal import (
    RULE_VERSION,
    run_semantic_history_signal_experiment,
)


class FakeEncoder:
    def encode(self, texts, *, batch_size):
        vectors = []
        for text in texts:
            total = sum(ord(character) for character in str(text))
            vectors.append(
                [
                    float(total % 17 + 1),
                    float(total % 13 + 1),
                    float(total % 7 + 1),
                    1.0,
                ]
            )
        return vectors


def _row(index: int, split: str) -> dict[str, object]:
    labels = ("a", "b", "c", "d")
    history = labels[index % len(labels)]
    positive = labels[(index + 1) % len(labels)]
    second = labels[(index + 2) % len(labels)]
    positives = [positive, second] if index % 3 == 0 else [positive]
    return {
        "user_id": f"{split}-{index}",
        "split": split,
        "history_length": 2 + index % 2,
        "history_unique_subcategory_count": 1,
        "history_titles": [f"title {index}", f"shared {history}"],
        "history_subcategories": [history, history],
        "history_unique_subcategories": [history],
        "target_count": len(positives),
        "future_novel_subcategories": positives,
        "future_exposed_subcategories": positives,
        "future_exposed_unclicked_subcategories": [],
    }


def _fixture(root: Path) -> dict[str, object]:
    benchmark = root / "benchmark"
    benchmark.mkdir()
    vocabulary = [
        {"interest_id": index, "category": "C", "subcategory": label}
        for index, label in enumerate(("a", "b", "c", "d"))
    ]
    pq.write_table(
        pa.Table.from_pylist(vocabulary),
        benchmark / "interest_vocabulary.parquet",
    )
    for split, count in (("fit", 12), ("validation", 6), ("dev", 8)):
        pq.write_table(
            pa.Table.from_pylist([_row(index, split) for index in range(count)]),
            benchmark / f"{split}.parquet",
        )
    return {
        "rule_version": RULE_VERSION,
        "benchmark_dir": str(benchmark),
        "artifact_dir": str(root / "artifacts"),
        "cutoffs": [1, 5, 10, 20],
        "embedding": {
            "model_name": "fake",
            "revision": "fake-revision",
            "device": None,
            "batch_size": 4,
        },
        "model": {
            "prior_alpha": 1.0,
            "ridge_alphas": [1.0, 10.0],
            "lambda_grid": [0.0, 0.5, 1.0],
        },
        "shuffle": {"seeds": [11, 12, 13]},
        "bootstrap": {"replicates": 2000, "seed": 17, "confidence": 0.95},
        "acceptance": {"minimum_absolute_delta": 0.01},
    }


def test_runs_actual_global_and_matched_shuffle_conditions(tmp_path: Path) -> None:
    manifest = run_semantic_history_signal_experiment(
        _fixture(tmp_path), encoder=FakeEncoder()
    )
    artifact = tmp_path / "artifacts"
    assert artifact.is_dir()
    assert manifest["users"] == {"fit": 12, "validation": 6, "dev": 8}
    assert len(manifest["conditions"]) == 9
    assert (artifact / "shuffle_manifest.parquet").is_file()
    shuffles = pq.read_table(artifact / "shuffle_manifest.parquet").to_pylist()
    assert {row["seed"] for row in shuffles} == {11, 12, 13}
    assert not any(row["is_fixed_point"] for row in shuffles)
    assert (artifact / "comparisons.json").is_file()
    assert (artifact / "acceptance.json").is_file()
    assert (artifact / "linear_model.npz").is_file()


def test_failed_overwrite_is_non_destructive(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    run_semantic_history_signal_experiment(config, encoder=FakeEncoder())
    marker = tmp_path / "artifacts" / "marker"
    marker.write_text("keep", encoding="utf-8")
    try:
        run_semantic_history_signal_experiment(config, encoder=FakeEncoder())
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected overwrite protection")
    assert marker.read_text(encoding="utf-8") == "keep"
