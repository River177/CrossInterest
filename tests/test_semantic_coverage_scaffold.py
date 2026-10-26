from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.semantic_coverage_scaffold import (
    RULE_VERSION,
    _interest_metrics_at_k,
    _internal_partition,
    _load_split,
    _ranking_context,
    _scaffold_priority,
    _split_priority,
    _topic_texts,
    ScaffoldParameters,
    run_semantic_coverage_scaffold_experiment,
)
from new_interest.experiments.semantic_history_signal import SplitData


LABELS = tuple(f"topic-{index:02d}" for index in range(24))


def _row(index: int, split: str) -> dict[str, object]:
    history = LABELS[index % len(LABELS)]
    first = LABELS[(index + 3) % len(LABELS)]
    second = LABELS[(index + 7) % len(LABELS)]
    positives = [first, second] if index % 4 == 0 else [first]
    return {
        "user_id": f"{split}-{index:04d}",
        "split": split,
        "history_length": 2,
        "history_unique_subcategory_count": 1,
        "history_titles": [f"user title {index}", f"history {history}"],
        "history_subcategories": [history, history],
        "history_unique_subcategories": [history],
        "target_count": len(positives),
        "future_novel_subcategories": positives,
        "future_exposed_subcategories": positives,
        "future_exposed_unclicked_subcategories": [],
    }


def _embedding(index: int) -> list[float]:
    return [
        float(index % 11 + 1),
        float(index % 7 + 1),
        float(index % 5 + 1),
        1.0,
    ]


def _fixture(root: Path) -> dict[str, object]:
    benchmark = root / "benchmark"
    embeddings = root / "semantic" / "embeddings"
    benchmark.mkdir()
    embeddings.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "interest_id": index,
                    "category": "category",
                    "subcategory": label,
                }
                for index, label in enumerate(LABELS)
            ]
        ),
        benchmark / "interest_vocabulary.parquet",
    )
    for split, count in (("fit", 120), ("validation", 30), ("dev", 30)):
        rows = [_row(index, split) for index in range(count)]
        pq.write_table(pa.Table.from_pylist(rows), benchmark / f"{split}.parquet")
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "user_id": row["user_id"],
                        "embedding": _embedding(index),
                    }
                    for index, row in enumerate(rows)
                ]
            ),
            embeddings / f"{split}.parquet",
        )
    return {
        "rule_version": RULE_VERSION,
        "benchmark_dir": str(benchmark),
        "semantic_artifact_dir": str(root / "semantic"),
        "artifact_dir": str(root / "artifacts"),
        "cutoffs": [1, 5, 10, 20],
        "internal_split": {
            "seed": 123,
            "modulus": 100,
            "model_fit_end": 60,
            "tune_end": 80,
        },
        "model": {"prior_alpha": 1.0, "ridge_alpha_grid": [1.0, 10.0]},
        "scaffold": {
            "primary_cutoff": 10,
            "global_pool_size_grid": [10, 12],
            "semantic_expansion_grid": [0, 2],
            "lambda_grid": [0.0, 1.0],
            "minimum_tune_recall_delta": 0.0,
        },
        "shuffle": {"seeds": [11, 12, 13]},
        "bootstrap": {"replicates": 2000, "seed": 17, "confidence": 0.95},
        "acceptance": {},
    }


def test_hash_partition_is_independent_of_input_order(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    benchmark = Path(str(config["benchmark_dir"]))
    semantic = Path(str(config["semantic_artifact_dir"]))
    labels, _ = _topic_texts(benchmark / "interest_vocabulary.parquet")
    data = _load_split(
        benchmark / "fit.parquet",
        semantic / "embeddings" / "fit.parquet",
        labels,
        include_history_mask=True,
    )
    partitions, rows = _internal_partition(
        data,
        seed=123,
        modulus=100,
        model_fit_end=60,
        tune_end=80,
    )
    by_user = {str(row["user_id"]): str(row["role"]) for row in rows}
    assert all(partition.user_ids for partition in partitions.values())
    for user_id, role in by_user.items():
        bucket = int(_split_priority(user_id, seed=123)[:16], 16) % 100
        expected = (
            "model_fit"
            if bucket < 60
            else "scaffold_tune"
            if bucket < 80
            else "internal_holdout"
        )
        assert role == expected


def test_global_top_k_is_preserved_without_semantic_expansion(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    benchmark = Path(str(config["benchmark_dir"]))
    semantic = Path(str(config["semantic_artifact_dir"]))
    labels, _ = _topic_texts(benchmark / "interest_vocabulary.parquet")
    data = _load_split(
        benchmark / "dev.parquet",
        semantic / "embeddings" / "dev.parquet",
        labels,
        include_history_mask=True,
    )
    global_scores = np.linspace(0.0, -3.0, len(labels), dtype=np.float32)
    residual = np.flip(data.embeddings[:, :1], axis=0) @ np.linspace(
        -1.0, 1.0, len(labels), dtype=np.float32
    )[None, :]
    context = _ranking_context(global_scores, residual, data)
    priority = _scaffold_priority(
        global_scores,
        context,
        data,
        ScaffoldParameters(10, 0, 10.0),
    )
    scaffold_top10 = np.argsort(priority, axis=1, kind="stable")[:, :10]
    for scaffold, global_order in zip(
        scaffold_top10, context.global_order, strict=True
    ):
        assert set(scaffold.tolist()) == set(global_order[:10].tolist())


def test_interest_macro_weights_interests_not_users() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    data = SplitData(
        user_ids=("u0", "u1", "u2"),
        embeddings=np.zeros((3, 2), dtype=np.float32),
        positives=scipy_sparse.csr_matrix(
            np.asarray(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ],
                dtype=np.int8,
            )
        ),
        history_mask=np.zeros((3, 3), dtype=np.bool_),
        history_length=np.ones(3, dtype=np.int32),
        history_topic_count=np.ones(3, dtype=np.int32),
    )
    rankings = np.asarray(
        [
            [1, 0, 2],
            [0, 1, 2],
            [2, 0, 1],
        ],
        dtype=np.int64,
    )
    support, recall, ndcg = _interest_metrics_at_k(
        rankings,
        data,
        cutoff=1,
    )
    assert support.tolist() == [3, 1, 0]
    assert recall[:2].tolist() == pytest.approx([1 / 3, 1.0])
    assert ndcg[:2].tolist() == pytest.approx([1 / 3, 1.0])
    assert recall[support > 0].mean() == pytest.approx(2 / 3)
    # The secondary User-Macro value would be (1/2 + 1 + 0) / 3 = 1/2.
    assert recall[support > 0].mean() != pytest.approx(0.5)


def test_runner_writes_interest_macro_primary_metrics_and_protects_output(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    manifest = run_semantic_coverage_scaffold_experiment(config)
    artifact = Path(str(config["artifact_dir"]))
    assert manifest["primary_metrics"] == [
        "interest_macro_recall_at_10",
        "interest_macro_ndcg_at_10",
    ]
    assert manifest["users"]["deployment_train"] == 150
    metrics_path = (
        artifact
        / "conditions"
        / "dev_exploratory"
        / "semantic_coverage_actual"
        / "metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["user_macro_recall_at_10"] == metrics["recall_at_10"]
    assert metrics["user_macro_ndcg_at_10"] == metrics["ndcg_at_10"]
    assert (
        metrics["interest_macro_recall_at_10"]
        == metrics["macro_recall_at_10"]
    )
    assert "interest_macro_ndcg_at_10" in metrics
    assert metrics["supported_interest_count"] > 0
    assert metrics["metric_definitions"]["legacy_aliases"] == {
        "recall_at_k": "user_macro_recall_at_k",
        "ndcg_at_k": "user_macro_ndcg_at_k",
        "macro_recall_at_k": "interest_macro_recall_at_k",
    }
    per_interest = metrics_path.with_name("per_interest_metrics.parquet")
    assert pq.read_table(per_interest).num_rows == len(LABELS)
    tuning = json.loads((artifact / "tuning.json").read_text(encoding="utf-8"))
    assert tuning["selection_unit"] == "interest"
    comparisons = json.loads(
        (artifact / "comparisons.json").read_text(encoding="utf-8")
    )
    bootstrap = comparisons["dev_exploratory"]["semantic_coverage"][
        "actual_vs_global"
    ]
    assert bootstrap["resampling_unit"] == "user"
    assert "recompute per-interest" in bootstrap["aggregation"]
    assert (artifact / "internal_split_manifest.parquet").is_file()
    assert (artifact / "shuffle_manifest.parquet").is_file()
    shuffle = pq.read_table(artifact / "shuffle_manifest.parquet").to_pylist()
    assert not any(row["is_fixed_point"] for row in shuffle)
    with pytest.raises(FileExistsError):
        run_semantic_coverage_scaffold_experiment(config)
