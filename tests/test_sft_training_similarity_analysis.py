from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_dataset import DATASET_SCHEMA
from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.build_sft_subsets import GROUPED_EVAL_SCHEMA
from new_interest.experiments.analyze_sft_training_similarity import (
    bootstrap_mean_interval,
    compute_parent_permutation_baseline,
    compute_prompt_overlap,
    compute_similarity_features,
    compute_taxonomy_features,
    run_sft_training_similarity_analysis,
    write_sft_analysis_rows,
    write_category_name_embeddings,
    write_sft_enriched_rows,
    write_sft_semantic_rows,
    write_sft_similarity_cases,
    write_sft_taxonomy_rows,
    write_sft_training_statistics,
    write_title_embeddings,
)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def _news(
    news_id: str,
    category: str,
    subcategory: str,
    title: str,
) -> list[str]:
    return [
        news_id,
        category,
        subcategory,
        title,
        "",
        f"https://example.test/{news_id}",
        "[]",
        "[]",
    ]


def _history_item(
    news_id: str,
    category: str,
    subcategory: str,
    title: str,
) -> dict[str, str]:
    return {
        "news_id": news_id,
        "category": category,
        "subcategory": subcategory,
        "title": title,
    }


def _source_row(
    split: str,
    impression_id: str,
    user_id: str,
    history: list[dict[str, str]],
    category: str,
    subcategory: str,
    news_id: str,
    title: str,
) -> dict[str, object]:
    return {
        "sample_id": f"{split}:{impression_id}:{subcategory}",
        "split": split,
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": datetime(2019, 11, 13, 12, 0, 0),
        "history": history,
        "target_interest": {
            "category": category,
            "subcategory": subcategory,
        },
        "target_news": [{"news_id": news_id, "title": title}],
    }


def _train_sft_row(
    source_sample_id: str,
    impression_id: str,
    user_id: str,
    task_type: str,
    target: str,
) -> dict[str, object]:
    prompt = f"{task_type} prompt for {user_id}"
    response = f"New topic: {target}"
    return {
        "sample_id": f"{source_sample_id}:{task_type}",
        "source_sample_id": source_sample_id,
        "split": "train",
        "task_type": task_type,
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": datetime(2019, 11, 13, 12, 0, 0),
        "history_length": 2,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "target_subcategory": target,
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    raw = root / "raw"
    source = root / "source"
    sft = root / "sft"
    news_rows = [
        _news("H0", "Parent 0", "History 0", "oldest history"),
        _news("H1", "Parent A", "History A", "recent history one"),
        _news("H2", "Parent B", "History B", "recent history two"),
        _news("P1", "Parent C", "Target C", "clicked train title"),
        _news("P2", "Parent A", "Target A2", "clicked dev title one"),
        _news("P3", "Parent D", "Target D", "clicked dev title two"),
        _news("C1", "Parent B", "History B", "known control"),
        _news("C2", "Parent E", "Control E", "novel control"),
    ]
    for split in ("train", "dev"):
        _write_tsv(raw / split / "news.tsv", news_rows)
    _write_tsv(
        raw / "train" / "behaviors.tsv",
        [
            [
                "1",
                "U1",
                "11/13/2019 12:00:00 PM",
                "H0 H1 H2",
                "P1-1 C1-0 C2-0 C2-0",
            ]
        ],
    )
    _write_tsv(
        raw / "dev" / "behaviors.tsv",
        [
            [
                "2",
                "U2",
                "11/13/2019 12:00:00 PM",
                "H0 H1 H2",
                "P2-1 P3-1 C1-0 C2-0",
            ]
        ],
    )

    history = [
        _history_item("H0", "Parent 0", "History 0", "oldest history"),
        _history_item("H1", "Parent A", "History A", "recent history one"),
        _history_item("H2", "Parent B", "History B", "recent history two"),
    ]
    train_source = _source_row(
        "train",
        "1",
        "U1",
        history,
        "Parent C",
        "Target C",
        "P1",
        "clicked train title",
    )
    dev_source = [
        _source_row(
            "dev",
            "2",
            "U2",
            history,
            "Parent A",
            "Target A2",
            "P2",
            "clicked dev title one",
        ),
        _source_row(
            "dev",
            "2",
            "U2",
            history,
            "Parent D",
            "Target D",
            "P3",
            "clicked dev title two",
        ),
    ]
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist([train_source], schema=DATASET_SCHEMA),
        source / "train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(dev_source, schema=DATASET_SCHEMA),
        source / "dev.parquet",
    )

    source_sample_id = str(train_source["sample_id"])
    for task_type in ("behavior_only", "interest_only"):
        train_path = sft / "train" / task_type / "22k.parquet"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _train_sft_row(
                        source_sample_id,
                        "1",
                        "U1",
                        task_type,
                        "Target C",
                    )
                ],
                schema=GENERATIVE_SCHEMA,
            ),
            train_path,
        )
        dev_path = (
            sft / "eval" / task_type / "dev_10k_grouped.parquet"
        )
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "impression_id": "2",
                        "user_id": "U2",
                        "timestamp": datetime(2019, 11, 13, 12, 0, 0),
                        "history_length": 2,
                        "prompt": f"{task_type} dev prompt",
                        "acceptable_responses": [
                            "New topic: Target A2",
                            "New topic: Target D",
                        ],
                        "acceptable_subcategories": [
                            "Target A2",
                            "Target D",
                        ],
                        "history_subcategories": [
                            "History A",
                            "History B",
                        ],
                    }
                ],
                schema=GROUPED_EVAL_SCHEMA,
            ),
            dev_path,
        )
    return sft, source, raw


def test_builds_aligned_train_and_dev_analysis_rows(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    output = tmp_path / "analysis_rows.parquet"

    summary = write_sft_analysis_rows(
        sft,
        source,
        raw,
        output,
        history_limit=2,
    )

    rows = pq.read_table(output).to_pylist()
    assert summary["rows"] == {"train": 1, "dev": 1}
    assert [row["split"] for row in rows] == ["train", "dev"]

    train = rows[0]
    assert [item["news_id"] for item in train["history"]] == ["H1", "H2"]
    assert [item["news_id"] for item in train["positive_news"]] == ["P1"]
    assert [item["news_id"] for item in train["control_news"]] == [
        "C1",
        "C2",
        "C2",
    ]
    assert [item["news_id"] for item in train["novel_control_news"]] == [
        "C2",
        "C2",
    ]

    dev = rows[1]
    assert dev["target_subcategories"] == ["Target A2", "Target D"]
    assert [item["news_id"] for item in dev["positive_news"]] == ["P2", "P3"]
    assert [item["news_id"] for item in dev["novel_control_news"]] == ["C2"]


def test_computes_row_weighted_similarity_auc_and_hit_at_one() -> None:
    diagonal = math.sqrt(0.5)
    features = compute_similarity_features(
        history_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        positive_vectors=np.asarray(
            [[1.0, 0.0], [diagonal, diagonal]]
        ),
        control_vectors=np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        novel_control_vectors=np.asarray([[-1.0, 0.0]]),
    )

    nearest = features["nearest_50"]
    assert nearest["positive_mean"] == pytest.approx(
        (1.0 + diagonal) / 2
    )
    assert nearest["control_mean"] == pytest.approx(0.5)
    assert nearest["novel_control_mean"] == pytest.approx(0.0)
    assert nearest["delta"] == pytest.approx(diagonal / 2)
    assert nearest["novel_delta"] == pytest.approx(
        (1.0 + diagonal) / 2
    )
    assert nearest["auc"] == pytest.approx(0.625)
    assert nearest["novel_auc"] == pytest.approx(1.0)
    assert nearest["hit_at_1"] == pytest.approx(0.5)
    assert nearest["novel_hit_at_1"] == pytest.approx(1.0)

    assert features["top3_mean"]["positive_mean"] == pytest.approx(
        (0.5 + diagonal) / 2
    )
    assert features["centroid"]["positive_mean"] == pytest.approx(
        (diagonal + 1.0) / 2
    )
    assert features["nearest_5"] == features["nearest_50"]
    assert features["last"]["positive_mean"] == pytest.approx(diagonal / 2)


class _FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        del batch_size
        return [self.vectors[text] for text in texts]


def test_writes_deduplicated_embeddings_and_semantic_rows(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    input_rows = tmp_path / "input_rows.parquet"
    embeddings = tmp_path / "title_embeddings.parquet"
    semantic_rows = tmp_path / "semantic_rows.parquet"
    write_sft_analysis_rows(
        sft,
        source,
        raw,
        input_rows,
        history_limit=2,
    )
    diagonal = math.sqrt(0.5)
    encoder = _FakeEncoder(
        {
            "recent history one": [1.0, 0.0],
            "recent history two": [0.0, 1.0],
            "clicked train title": [diagonal, diagonal],
            "clicked dev title one": [1.0, 0.0],
            "clicked dev title two": [-1.0, 0.0],
            "known control": [0.0, 1.0],
            "novel control": [-1.0, 0.0],
            "History A": [1.0, 0.0],
            "History B": [0.0, 1.0],
            "Target C": [diagonal, diagonal],
            "Target A2": [1.0, 0.0],
            "Target D": [-1.0, 0.0],
            "Control E": [-1.0, 0.0],
            "Parent A": [1.0, 0.0],
            "Parent B": [0.0, 1.0],
            "Parent C": [diagonal, diagonal],
            "Parent D": [-1.0, 0.0],
            "Parent E": [-1.0, 0.0],
        }
    )

    embedding_summary = write_title_embeddings(
        input_rows,
        embeddings,
        encoder=encoder,
        model_name="fake",
        revision="test",
        batch_size=3,
    )
    semantic_summary = write_sft_semantic_rows(
        input_rows,
        embeddings,
        semantic_rows,
    )

    embedding_rows = pq.read_table(embeddings).to_pylist()
    embedding_metadata = json.loads(
        pq.ParquetFile(embeddings).schema_arrow.metadata[b"metadata"]
    )
    assert embedding_summary["unique_titles"] == 7
    assert embedding_summary["dimensions"] == 2
    assert embedding_metadata["unique_titles"] == 7
    assert len(embedding_metadata["title_corpus_sha256"]) == 64
    assert "source_snapshot_sha256" not in embedding_metadata
    assert len(embedding_rows) == 7
    assert len({row["title"] for row in embedding_rows}) == 7

    rows = pq.read_table(semantic_rows).to_pylist()
    assert semantic_summary["rows"] == {"train": 1, "dev": 1}
    assert rows[0]["nearest_50_positive_mean"] == pytest.approx(diagonal)
    assert rows[0]["nearest_50_control_mean"] == pytest.approx(1.0 / 3.0)
    assert rows[0]["nearest_50_novel_control_mean"] == pytest.approx(0.0)
    assert rows[0]["nearest_50_novel_auc"] == pytest.approx(1.0)
    assert rows[1]["nearest_50_positive_mean"] == pytest.approx(0.5)


def test_writes_category_name_embeddings_and_similarity_rows(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    input_rows = tmp_path / "input_rows.parquet"
    title_embeddings = tmp_path / "title_embeddings.parquet"
    category_embeddings = tmp_path / "category_embeddings.parquet"
    semantic_rows = tmp_path / "semantic_rows.parquet"
    write_sft_analysis_rows(
        sft,
        source,
        raw,
        input_rows,
        history_limit=2,
    )
    diagonal = math.sqrt(0.5)
    encoder = _FakeEncoder(
        {
            "recent history one": [1.0, 0.0],
            "recent history two": [0.0, 1.0],
            "clicked train title": [diagonal, diagonal],
            "clicked dev title one": [1.0, 0.0],
            "clicked dev title two": [-1.0, 0.0],
            "known control": [0.0, 1.0],
            "novel control": [-1.0, 0.0],
            "History A": [1.0, 0.0],
            "History B": [0.0, 1.0],
            "Target C": [diagonal, diagonal],
            "Target A2": [1.0, 0.0],
            "Target D": [-1.0, 0.0],
            "Control E": [-1.0, 0.0],
            "Parent A": [1.0, 0.0],
            "Parent B": [0.0, 1.0],
            "Parent C": [diagonal, diagonal],
            "Parent D": [-1.0, 0.0],
            "Parent E": [-1.0, 0.0],
        }
    )
    write_title_embeddings(
        input_rows,
        title_embeddings,
        encoder=encoder,
        model_name="fake",
        revision="test",
    )

    category_summary = write_category_name_embeddings(
        input_rows,
        category_embeddings,
        encoder=encoder,
        model_name="fake",
        revision="test",
    )
    write_sft_semantic_rows(
        input_rows,
        title_embeddings,
        semantic_rows,
        category_name_embeddings_path=category_embeddings,
    )

    assert category_summary["unique_names"] == 11
    rows = pq.read_table(semantic_rows).to_pylist()
    assert rows[0][
        "subcategory_name_nearest_50_positive_mean"
    ] == pytest.approx(diagonal)
    assert rows[0][
        "subcategory_name_nearest_50_control_mean"
    ] == pytest.approx(1.0 / 3.0)
    assert rows[0][
        "subcategory_name_nearest_50_novel_control_mean"
    ] == pytest.approx(0.0)
    assert rows[0][
        "parent_name_nearest_50_positive_mean"
    ] == pytest.approx(diagonal)
    assert rows[1][
        "subcategory_name_nearest_50_positive_mean"
    ] == pytest.approx(0.5)
    assert rows[1][
        "parent_name_nearest_50_positive_mean"
    ] == pytest.approx(0.5)


def test_computes_multi_target_taxonomy_and_history_breadth() -> None:
    history = [
        _history_item("H1", "Parent A", "Sub A", "one"),
        _history_item("H2", "Parent B", "Sub B", "two"),
        _history_item("H2", "Parent B", "Sub B", "two"),
    ]

    features = compute_taxonomy_features(
        history=history,
        target_categories=["Parent A", "Parent C"],
    )

    assert features["same_parent_fraction"] == pytest.approx(0.5)
    assert features["any_target_same_parent"] == 1.0
    assert features["all_targets_same_parent"] == 0.0
    assert features["same_parent_recent5_fraction"] == pytest.approx(0.5)
    assert features["same_parent_last_fraction"] == pytest.approx(0.0)
    assert features["history_length"] == 3
    assert features["history_unique_news"] == 2
    assert features["history_unique_subcategories"] == 2
    assert features["history_unique_parents"] == 2
    assert features["history_repetition_count"] == 1
    assert features["history_repetition_rate"] == pytest.approx(1.0 / 3.0)
    expected_entropy = -(
        (1.0 / 3.0) * math.log(1.0 / 3.0)
        + (2.0 / 3.0) * math.log(2.0 / 3.0)
    )
    assert features["history_subcategory_entropy"] == pytest.approx(
        expected_entropy
    )


def test_writes_one_taxonomy_row_per_sft_row(tmp_path: Path) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    input_rows = tmp_path / "input_rows.parquet"
    taxonomy_rows = tmp_path / "taxonomy_rows.parquet"
    write_sft_analysis_rows(
        sft,
        source,
        raw,
        input_rows,
        history_limit=2,
    )

    summary = write_sft_taxonomy_rows(input_rows, taxonomy_rows)

    rows = pq.read_table(taxonomy_rows).to_pylist()
    assert summary["rows"] == {"train": 1, "dev": 1}
    assert rows[0]["same_parent_fraction"] == 0.0
    assert rows[1]["same_parent_fraction"] == pytest.approx(0.5)
    assert rows[1]["any_target_same_parent"] == 1.0
    assert rows[1]["all_targets_same_parent"] == 0.0
    assert rows[1]["same_parent_last_fraction"] == 0.0


def test_bootstrap_and_parent_permutation_are_deterministic() -> None:
    first_interval = bootstrap_mean_interval(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        replicates=500,
        seed=19,
        confidence=0.95,
    )
    second_interval = bootstrap_mean_interval(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        replicates=500,
        seed=19,
        confidence=0.95,
    )
    assert first_interval == second_interval
    assert first_interval["mean"] == pytest.approx(1.5)
    assert (
        first_interval["ci_low"]
        <= first_interval["mean"]
        <= first_interval["ci_high"]
    )

    histories = [
        ["Parent A"],
        ["Parent A"],
        ["Parent B"],
        ["Parent B"],
    ]
    targets = [
        ["Parent A"],
        ["Parent A"],
        ["Parent B"],
        ["Parent B"],
    ]
    first_baseline = compute_parent_permutation_baseline(
        history_categories=histories,
        target_categories=targets,
        replicates=2_000,
        seed=23,
        confidence=0.95,
    )
    second_baseline = compute_parent_permutation_baseline(
        history_categories=histories,
        target_categories=targets,
        replicates=2_000,
        seed=23,
        confidence=0.95,
    )
    assert first_baseline == second_baseline
    assert first_baseline["same_parent_fraction"]["observed"] == 1.0
    assert first_baseline["same_parent_fraction"]["expected"] == pytest.approx(
        0.5,
        abs=0.03,
    )
    assert first_baseline["same_parent_fraction"]["lift"] == pytest.approx(
        2.0,
        abs=0.15,
    )


def test_audits_prompt_duplicates_conflicts_and_overlap() -> None:
    rows = [
        {
            "split": "train",
            "user_id": "U1",
            "behavior_prompt_sha256": "behavior-shared",
            "interest_prompt_sha256": "interest-shared",
            "target_subcategories": ["Target A"],
        },
        {
            "split": "train",
            "user_id": "U2",
            "behavior_prompt_sha256": "behavior-shared",
            "interest_prompt_sha256": "interest-shared",
            "target_subcategories": ["Target B"],
        },
        {
            "split": "dev",
            "user_id": "U1",
            "behavior_prompt_sha256": "behavior-shared",
            "interest_prompt_sha256": "interest-shared",
            "target_subcategories": ["Target A", "Target C"],
        },
        {
            "split": "dev",
            "user_id": "U3",
            "behavior_prompt_sha256": "behavior-dev-only",
            "interest_prompt_sha256": "interest-dev-only",
            "target_subcategories": ["Target D"],
        },
    ]

    audit = compute_prompt_overlap(rows)

    assert audit["user_overlap"]["users"] == 1
    assert audit["user_overlap"]["dev_rows"] == 1
    for task_type in ("behavior", "interest"):
        task = audit["tasks"][task_type]
        assert task["train_duplicate_row_excess"] == 1
        assert task["train_conflicting_prompt_count"] == 1
        assert task["dev_exact_prompt_overlap_rows"] == 1
        assert task["dev_any_target_seen_for_prompt_rows"] == 1
        assert task["dev_all_targets_seen_for_prompt_rows"] == 0


def test_merges_semantic_and_taxonomy_features_by_row_id(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    base_rows = tmp_path / "base_rows.parquet"
    embeddings = tmp_path / "embeddings.parquet"
    semantic_rows = tmp_path / "semantic.parquet"
    taxonomy_rows = tmp_path / "taxonomy.parquet"
    enriched_rows = tmp_path / "analysis_rows.parquet"
    write_sft_analysis_rows(
        sft,
        source,
        raw,
        base_rows,
        history_limit=2,
    )
    diagonal = math.sqrt(0.5)
    write_title_embeddings(
        base_rows,
        embeddings,
        encoder=_FakeEncoder(
            {
                "recent history one": [1.0, 0.0],
                "recent history two": [0.0, 1.0],
                "clicked train title": [diagonal, diagonal],
                "clicked dev title one": [1.0, 0.0],
                "clicked dev title two": [-1.0, 0.0],
                "known control": [0.0, 1.0],
                "novel control": [-1.0, 0.0],
            }
        ),
        model_name="fake",
        revision="test",
    )
    write_sft_semantic_rows(base_rows, embeddings, semantic_rows)
    write_sft_taxonomy_rows(base_rows, taxonomy_rows)

    summary = write_sft_enriched_rows(
        base_rows,
        semantic_rows,
        taxonomy_rows,
        enriched_rows,
    )

    rows = pq.read_table(enriched_rows).to_pylist()
    assert summary["rows"] == {"train": 1, "dev": 1}
    assert rows[0]["history"][0]["news_id"] == "H1"
    assert rows[0]["nearest_50_positive_mean"] == pytest.approx(diagonal)
    assert rows[0]["same_parent_fraction"] == 0.0
    assert rows[1]["same_parent_fraction"] == pytest.approx(0.5)


def test_writes_bootstrapped_statistics_and_machine_artifacts(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    base_rows = tmp_path / "base.parquet"
    embeddings = tmp_path / "embeddings.parquet"
    category_embeddings = tmp_path / "category_embeddings.parquet"
    semantic_rows = tmp_path / "semantic.parquet"
    taxonomy_rows = tmp_path / "taxonomy.parquet"
    enriched_rows = tmp_path / "analysis.parquet"
    output_dir = tmp_path / "statistics"
    write_sft_analysis_rows(
        sft,
        source,
        raw,
        base_rows,
        history_limit=2,
    )
    diagonal = math.sqrt(0.5)
    encoder = _FakeEncoder(
        {
            "recent history one": [1.0, 0.0],
            "recent history two": [0.0, 1.0],
            "clicked train title": [diagonal, diagonal],
            "clicked dev title one": [1.0, 0.0],
            "clicked dev title two": [-1.0, 0.0],
            "known control": [0.0, 1.0],
            "novel control": [-1.0, 0.0],
            "History A": [1.0, 0.0],
            "History B": [0.0, 1.0],
            "Target C": [diagonal, diagonal],
            "Target A2": [1.0, 0.0],
            "Target D": [-1.0, 0.0],
            "Control E": [-1.0, 0.0],
            "Parent A": [1.0, 0.0],
            "Parent B": [0.0, 1.0],
            "Parent C": [diagonal, diagonal],
            "Parent D": [-1.0, 0.0],
            "Parent E": [-1.0, 0.0],
        }
    )
    write_title_embeddings(
        base_rows,
        embeddings,
        encoder=encoder,
        model_name="fake",
        revision="test",
    )
    write_category_name_embeddings(
        base_rows,
        category_embeddings,
        encoder=encoder,
        model_name="fake",
        revision="test",
    )
    write_sft_semantic_rows(
        base_rows,
        embeddings,
        semantic_rows,
        category_name_embeddings_path=category_embeddings,
    )
    write_sft_taxonomy_rows(base_rows, taxonomy_rows)
    write_sft_enriched_rows(
        base_rows,
        semantic_rows,
        taxonomy_rows,
        enriched_rows,
    )

    result = write_sft_training_statistics(
        enriched_rows,
        output_dir,
        bootstrap_replicates=100,
        permutation_replicates=100,
        seed=31,
        confidence=0.95,
    )

    assert result["rows"] == {"train": 1, "dev": 1}
    expected_files = {
        "summary.json",
        "similarity_metrics.csv",
        "category_name_similarity_metrics.csv",
        "taxonomy_metrics.csv",
        "slice_metrics.csv",
        "parent_transition_matrix.csv",
        "prompt_overlap.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files

    with (output_dir / "similarity_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        similarity = list(csv.DictReader(handle))
    train_primary = next(
        row
        for row in similarity
        if row["split"] == "train"
        and row["score_name"] == "nearest_50"
        and row["control_scope"] == "novel"
    )
    assert int(train_primary["rows"]) == 1
    assert float(train_primary["delta_mean"]) == pytest.approx(diagonal)

    with (output_dir / "category_name_similarity_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        category_similarity = list(csv.DictReader(handle))
    train_subcategory = next(
        row
        for row in category_similarity
        if row["split"] == "train"
        and row["name_level"] == "subcategory"
        and row["score_name"] == "nearest_50"
        and row["control_scope"] == "novel"
    )
    assert float(train_subcategory["delta_mean"]) == pytest.approx(diagonal)

    with (output_dir / "taxonomy_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        taxonomy = list(csv.DictReader(handle))
    dev_full = next(
        row
        for row in taxonomy
        if row["split"] == "dev"
        and row["history_scope"] == "full"
        and row["metric"] == "same_parent_fraction"
    )
    assert float(dev_full["observed"]) == pytest.approx(0.5)

    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["rows"] == {"dev": 1, "train": 1}
    assert summary["primary_rows"] == {"dev": 1, "train": 1}


def test_selects_deterministic_provenance_rich_cases(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    base_rows = tmp_path / "base.parquet"
    embeddings = tmp_path / "embeddings.parquet"
    semantic_rows = tmp_path / "semantic.parquet"
    taxonomy_rows = tmp_path / "taxonomy.parquet"
    enriched_rows = tmp_path / "analysis.parquet"
    cases_path = tmp_path / "cases.json"
    write_sft_analysis_rows(
        sft,
        source,
        raw,
        base_rows,
        history_limit=2,
    )
    diagonal = math.sqrt(0.5)
    write_title_embeddings(
        base_rows,
        embeddings,
        encoder=_FakeEncoder(
            {
                "recent history one": [1.0, 0.0],
                "recent history two": [0.0, 1.0],
                "clicked train title": [diagonal, diagonal],
                "clicked dev title one": [1.0, 0.0],
                "clicked dev title two": [-1.0, 0.0],
                "known control": [0.0, 1.0],
                "novel control": [-1.0, 0.0],
            }
        ),
        model_name="fake",
        revision="test",
    )
    write_sft_semantic_rows(base_rows, embeddings, semantic_rows)
    write_sft_taxonomy_rows(base_rows, taxonomy_rows)
    write_sft_enriched_rows(
        base_rows,
        semantic_rows,
        taxonomy_rows,
        enriched_rows,
    )

    result = write_sft_similarity_cases(
        enriched_rows,
        embeddings,
        cases_path,
        cases_per_split_quadrant=1,
    )

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    assert result["cases"] == 1
    case = payload["cases"][0]
    assert case["split"] == "train"
    assert case["quadrant"] == "cross_parent__higher_than_control"
    assert case["row_id"].startswith("train:1:")
    assert case["history"][0]["title"] == "recent history one"
    assert case["positive_news"][0]["title"] == "clicked train title"
    assert case["positive_news"][0]["nearest_history"]["news_id"] == "H1"
    assert case["top_novel_controls"][0]["title"] == "novel control"
    assert case["metrics"]["nearest_50_novel_auc"] == 1.0


def test_runs_complete_analysis_with_manifest_and_atomic_overwrite(
    tmp_path: Path,
) -> None:
    sft, source, raw = _write_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "v1"
    diagonal = math.sqrt(0.5)
    encoder = _FakeEncoder(
        {
            "recent history one": [1.0, 0.0],
            "recent history two": [0.0, 1.0],
            "clicked train title": [diagonal, diagonal],
            "clicked dev title one": [1.0, 0.0],
            "clicked dev title two": [-1.0, 0.0],
            "known control": [0.0, 1.0],
            "novel control": [-1.0, 0.0],
            "History A": [1.0, 0.0],
            "History B": [0.0, 1.0],
            "Target C": [diagonal, diagonal],
            "Target A2": [1.0, 0.0],
            "Target D": [-1.0, 0.0],
            "Control E": [-1.0, 0.0],
            "Parent A": [1.0, 0.0],
            "Parent B": [0.0, 1.0],
            "Parent C": [diagonal, diagonal],
            "Parent D": [-1.0, 0.0],
            "Parent E": [-1.0, 0.0],
        }
    )
    config = {
        "rule_version": "sft-training-similarity-v1",
        "sft_dir": str(sft),
        "source_dir": str(source),
        "raw_dir": str(raw),
        "artifact_dir": str(artifact_dir),
        "train_name": "22k.parquet",
        "dev_name": "dev_10k_grouped.parquet",
        "history_limit": 2,
        "embedding": {
            "model_name": "fake",
            "revision": "test",
            "device": None,
            "batch_size": 3,
        },
        "bootstrap": {
            "replicates": 50,
            "seed": 37,
            "confidence": 0.95,
        },
        "permutation": {
            "replicates": 50,
            "seed": 37,
            "confidence": 0.95,
        },
        "cases": {
            "per_split_quadrant": 1,
            "top_controls": 2,
        },
    }

    manifest = run_sft_training_similarity_analysis(
        config,
        encoder=encoder,
    )

    expected_files = {
        "manifest.json",
        "resolved_config.json",
        "summary.json",
        "analysis_rows.parquet",
        "title_embeddings.parquet",
        "category_name_embeddings.parquet",
        "similarity_metrics.csv",
        "category_name_similarity_metrics.csv",
        "taxonomy_metrics.csv",
        "slice_metrics.csv",
        "parent_transition_matrix.csv",
        "prompt_overlap.json",
        "cases.json",
    }
    assert artifact_dir.is_dir()
    assert {path.name for path in artifact_dir.iterdir()} == expected_files
    assert manifest["join"]["rows"] == {"train": 1, "dev": 1}
    assert manifest["statistics"]["primary_rows"] == {
        "train": 1,
        "dev": 1,
    }
    assert manifest["category_name_embedding"]["unique_names"] == 11
    assert set(manifest["output_sha256"]) == expected_files - {
        "manifest.json"
    }
    with pytest.raises(FileExistsError):
        run_sft_training_similarity_analysis(config, encoder=encoder)

    replaced = run_sft_training_similarity_analysis(
        config,
        encoder=encoder,
        overwrite=True,
    )
    assert replaced["join"]["rows"] == {"train": 1, "dev": 1}
    assert artifact_dir.is_dir()
