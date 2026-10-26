from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_dataset import DATASET_SCHEMA, VOCABULARY_SCHEMA
from new_interest.experiments.analyze_interest_relationships import (
    analyze_relationships,
)


def _history(*items: tuple[str, str, str]) -> list[dict[str, str]]:
    return [
        {
            "news_id": news_id,
            "category": category,
            "subcategory": subcategory,
            "title": subcategory,
        }
        for news_id, category, subcategory in items
    ]


def _row(
    split: str,
    impression_id: str,
    user_id: str,
    history: list[dict[str, str]],
    target: tuple[str, str],
) -> dict[str, object]:
    category, subcategory = target
    return {
        "sample_id": f"{split}:{impression_id}:{subcategory}",
        "split": split,
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": datetime(2026, 1, 1),
        "history": history,
        "target_interest": {
            "category": category,
            "subcategory": subcategory,
        },
        "target_news": [
            {"news_id": f"target-{subcategory}", "title": subcategory}
        ],
    }


def test_analyze_relationships_uses_train_counts_and_groups_targets(
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "vocabulary.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"interest_id": 0, "category": "A", "subcategory": "a"},
                {"interest_id": 1, "category": "A", "subcategory": "b"},
                {"interest_id": 2, "category": "B", "subcategory": "c"},
                {"interest_id": 3, "category": "C", "subcategory": "d"},
            ],
            schema=VOCABULARY_SCHEMA,
        ),
        vocabulary_path,
    )
    train_history_a = _history(("n1", "A", "a"), ("n2", "A", "a"))
    train_rows = [
        _row("train", "1", "seen", train_history_a, ("A", "b")),
        _row("train", "1", "seen", train_history_a, ("B", "c")),
        _row(
            "train",
            "2",
            "other",
            _history(("n3", "B", "c")),
            ("C", "d"),
        ),
    ]
    dev_rows = [
        _row(
            "dev",
            "3",
            "seen",
            _history(("n4", "A", "a")),
            ("A", "b"),
        ),
        _row(
            "dev",
            "4",
            "unseen",
            _history(("n5", "B", "c")),
            ("C", "d"),
        ),
    ]
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    pq.write_table(
        pa.Table.from_pylist(train_rows, schema=DATASET_SCHEMA),
        train_path,
    )
    pq.write_table(
        pa.Table.from_pylist(dev_rows, schema=DATASET_SCHEMA),
        dev_path,
    )

    result = analyze_relationships(
        train_path,
        dev_path,
        vocabulary_path,
        tmp_path / "output",
        root=tmp_path,
    )

    assert result["train"]["impressions"] == 2
    assert result["train"]["target_rows"] == 3
    assert result["train"]["multi_target_impressions"] == 1
    assert result["train"]["repeated_user_target_memberships"] == 0
    assert result["train"]["novelty"][
        "exact_target_history_overlap_count"
    ] == 0
    assert result["train"]["taxonomic_relationship"]["any"][
        "observed_count"
    ] == 1
    last_metrics = result["dev_baselines_fitted_on_train"][
        "last_interest_conditional"
    ]["all"]
    assert last_metrics["impressions"] == 2
    assert last_metrics["recall_at_1"] == 1.0
    assert result["dev_baselines_fitted_on_train"][
        "last_interest_conditional"
    ]["seen_user"]["impressions"] == 1
    assert result["train_dev"]["target_distribution_shift"] is not None
    assert (tmp_path / "output/summary.json").is_file()
    assert (tmp_path / "output/train_relationships.csv").is_file()

    candidate_path = tmp_path / "candidate.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "selection_id": "dev:3:b",
                    "user_id": "seen",
                    "history_subcategories": ["a"],
                    "candidates": ["d", "b"],
                    "answer_category": "b",
                }
            ]
        ),
        candidate_path,
    )
    candidate_result = analyze_relationships(
        train_path,
        dev_path,
        vocabulary_path,
        tmp_path / "candidate-output",
        candidate_paths=[candidate_path],
        root=tmp_path,
    )
    assert candidate_result["candidate_baselines_fitted_on_train"][
        "candidate"
    ]["last_interest_conditional"]["accuracy"] == 1.0


def test_analyze_relationships_can_filter_dev_impressions(
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "vocabulary.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"interest_id": 0, "category": "A", "subcategory": "a"},
                {"interest_id": 1, "category": "B", "subcategory": "b"},
            ],
            schema=VOCABULARY_SCHEMA,
        ),
        vocabulary_path,
    )
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _row(
                    "train",
                    "1",
                    "train-user",
                    _history(("n1", "A", "a")),
                    ("B", "b"),
                )
            ],
            schema=DATASET_SCHEMA,
        ),
        train_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _row(
                    "dev",
                    impression_id,
                    f"user-{impression_id}",
                    _history((f"n-{impression_id}", "A", "a")),
                    ("B", "b"),
                )
                for impression_id in ("2", "3")
            ],
            schema=DATASET_SCHEMA,
        ),
        dev_path,
    )
    filter_path = tmp_path / "filter.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"impression_id": "3"}]),
        filter_path,
    )

    result = analyze_relationships(
        train_path,
        dev_path,
        vocabulary_path,
        tmp_path / "filtered-output",
        dev_impressions_path=filter_path,
        root=tmp_path,
    )

    assert result["analysis"]["dev_impression_filter_count"] == 1
    assert result["dev"]["impressions"] == 1
