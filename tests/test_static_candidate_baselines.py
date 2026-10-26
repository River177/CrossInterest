from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import DATASET_SCHEMA, VOCABULARY_SCHEMA
from new_interest.data.build_static_user_disjoint_candidate_selection import (
    build_static_user_disjoint_candidate_selection,
)
from new_interest.experiments.evaluate_static_candidate_baselines import (
    _build_parser,
    run_static_candidate_baselines,
)


LABELS = [f"Topic {index:02d}" for index in range(12)]


def _history(user_id: str, label: str) -> list[dict[str, str]]:
    return [
        {
            "news_id": f"N-{user_id}",
            "category": "Category",
            "subcategory": label,
            "title": f"History title for {user_id}",
        }
    ]


def _row(
    split: str,
    impression: str,
    user: str,
    target: str,
    history_label: str,
) -> dict[str, object]:
    target_id = LABELS.index(target)
    return {
        "sample_id": f"{split}:{impression}:{target_id}",
        "split": split,
        "impression_id": impression,
        "user_id": user,
        "timestamp": datetime(2019, 11, 15, 12, 0, 0),
        "history": _history(user, history_label),
        "target_interest": {
            "category": "Category",
            "subcategory": target,
        },
        "target_news": [
            {"news_id": f"T-{impression}", "title": f"Target {target}"}
        ],
    }


def _embedding(label_id: int) -> list[float]:
    vector = np.asarray(
        [1.0, label_id / len(LABELS), (label_id % 3) / 3.0, 0.5],
        dtype=np.float32,
    )
    return (vector / np.linalg.norm(vector)).tolist()


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "dataset"
    embeddings = root / "embeddings"
    candidates = root / "candidates"
    dataset.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    train = []
    train_embeddings = []
    for index in range(240):
        target_id = index % len(LABELS)
        history_id = (target_id + 1) % len(LABELS)
        user = f"TU{index:04d}"
        train.append(
            _row(
                "train",
                f"train-{index}",
                user,
                LABELS[target_id],
                LABELS[history_id],
            )
        )
        train_embeddings.append(
            {"user_id": user, "embedding": _embedding(target_id)}
        )
    dev = []
    dev_embeddings = []
    for index, target in enumerate(LABELS):
        user = f"DU{index:04d}"
        dev.append(
            _row(
                "dev",
                f"dev-{index}",
                user,
                target,
                LABELS[(index + 1) % len(LABELS)],
            )
        )
        dev_embeddings.append(
            {"user_id": user, "embedding": _embedding(index)}
        )
    vocabulary = [
        {
            "interest_id": index,
            "category": "Category",
            "subcategory": label,
        }
        for index, label in enumerate(LABELS)
    ]
    pq.write_table(pa.Table.from_pylist(train, schema=DATASET_SCHEMA), dataset / "train.parquet")
    pq.write_table(pa.Table.from_pylist(dev, schema=DATASET_SCHEMA), dataset / "dev.parquet")
    pq.write_table(
        pa.Table.from_pylist(vocabulary, schema=VOCABULARY_SCHEMA),
        dataset / "interest_vocabulary.parquet",
    )
    embedding_schema = pa.schema(
        [
            pa.field("user_id", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32()), nullable=False),
        ]
    )
    midpoint = len(train_embeddings) // 2
    pq.write_table(
        pa.Table.from_pylist(train_embeddings[:midpoint], schema=embedding_schema),
        embeddings / "fit.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(train_embeddings[midpoint:], schema=embedding_schema),
        embeddings / "validation.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev_embeddings, schema=embedding_schema),
        embeddings / "dev.parquet",
    )
    build_static_user_disjoint_candidate_selection(
        dataset, candidates, k=10, seed=17
    )
    return dataset, candidates, embeddings


def test_runs_train_only_candidate_baselines(tmp_path: Path) -> None:
    dataset, candidates, embeddings = _write_fixture(tmp_path)
    output = tmp_path / "baselines"

    run_static_candidate_baselines(
        dataset,
        candidates,
        embeddings,
        output,
        seed=17,
        shuffle_seeds=(17, 18, 19),
        bootstrap_replicates=10,
        ridge_alphas=(1.0, 10.0),
        semantic_weights=(0.0, 1.0, 4.0),
    )

    summary = json.loads((output / "summary.json").read_text())
    resolved = json.loads((output / "resolved_config.json").read_text())
    tuning = json.loads((output / "semantic_tuning.json").read_text())
    assert summary["train_users"] == 240
    assert summary["train_supported_categories"] == 12
    assert resolved["semantic_supervision"] == "Static Train selected target only"
    assert tuning["tune_users"] > 0
    assert tuning["holdout_users"] > 0
    for strategy in ("uniform", "train_frequency_matched"):
        conditions = summary["strategies"][strategy]
        assert "global_popularity" in conditions
        assert "last_interest_actual" in conditions
        assert "semantic_actual" in conditions
        assert len(
            [name for name in conditions if name.startswith("semantic_shuffled_")]
        ) == 3
        metrics = json.loads(
            (output / strategy / "semantic_actual/metrics.json").read_text()
        )
        assert metrics["samples"] == 12
        assert metrics["macro_category_count"] == 12
        assert 0.0 <= metrics["macro_category_accuracy"] <= 1.0
        comparisons = json.loads(
            (output / strategy / "bootstrap_comparisons.json").read_text()
        )
        assert comparisons["replicates"] == 10

    shuffle = pq.read_table(output / "shuffle_manifest.parquet").to_pylist()
    assert shuffle
    assert all(row["user_id"] != row["donor_user_id"] for row in shuffle)


def test_cli_accepts_explicit_semantic_search_grid() -> None:
    args = _build_parser().parse_args(
        [
            "--ridge-alphas",
            "0.01,0.1,1,10",
            "--semantic-weights",
            "0,16,64,256",
        ]
    )

    assert args.ridge_alphas == (0.01, 0.1, 1.0, 10.0)
    assert args.semantic_weights == (0.0, 16.0, 64.0, 256.0)
