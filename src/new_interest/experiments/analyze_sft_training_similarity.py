from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.mind_io import (
    BehaviorEvent,
    NewsCatalog,
    NewsRecord,
    iter_behavior_events,
    load_news_catalog,
)
from new_interest.experiments.embedding_future_interest import (
    SentenceTransformerEncoder,
    TextEncoder,
)
from new_interest.infrastructure.paths import portable_path

RULE_VERSION = "sft-training-similarity-v1"
DEFAULT_TITLE_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TITLE_ENCODER_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
NEWS_ITEM_TYPE = pa.struct(
    [
        pa.field("news_id", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("subcategory", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
    ]
)

ANALYSIS_ROW_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field(
            "source_sample_ids",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("history", pa.list_(NEWS_ITEM_TYPE), nullable=False),
        pa.field(
            "target_categories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "target_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "positive_news",
            pa.list_(NEWS_ITEM_TYPE),
            nullable=False,
        ),
        pa.field(
            "control_news",
            pa.list_(NEWS_ITEM_TYPE),
            nullable=False,
        ),
        pa.field(
            "novel_control_news",
            pa.list_(NEWS_ITEM_TYPE),
            nullable=False,
        ),
        pa.field("behavior_prompt", pa.string(), nullable=False),
        pa.field("interest_prompt", pa.string(), nullable=False),
        pa.field(
            "behavior_prompt_sha256",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "interest_prompt_sha256",
            pa.string(),
            nullable=False,
        ),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("target_count", pa.int32(), nullable=False),
        pa.field("positive_title_count", pa.int32(), nullable=False),
        pa.field("control_count", pa.int32(), nullable=False),
        pa.field("novel_control_count", pa.int32(), nullable=False),
        pa.field(
            "duplicate_control_token_count",
            pa.int32(),
            nullable=False,
        ),
    ]
)

TITLE_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("title_sha256", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field(
            "embedding",
            pa.list_(pa.float32()),
            nullable=False,
        ),
    ]
)

CATEGORY_NAME_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("name_sha256", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field(
            "embedding",
            pa.list_(pa.float32()),
            nullable=False,
        ),
    ]
)

SIMILARITY_SCORE_NAMES = (
    "nearest_50",
    "top3_mean",
    "centroid",
    "nearest_5",
    "last",
)

_SIMILARITY_REQUIRED_FIELDS = (
    "positive_mean",
    "control_mean",
    "delta",
    "auc",
    "hit_at_1",
    "positive_le_control_mean",
    "positive_le_control_max",
)
_SIMILARITY_NOVEL_FIELDS = (
    "novel_control_mean",
    "novel_delta",
    "novel_auc",
    "novel_hit_at_1",
    "positive_le_novel_control_mean",
    "positive_le_novel_control_max",
)
_SEMANTIC_REPRESENTATION_PREFIXES = (
    "",
    "subcategory_name_",
    "parent_name_",
)
CATEGORY_NAME_LEVEL_PREFIXES = {
    "subcategory": "subcategory_name_",
    "parent": "parent_name_",
}
SEMANTIC_ROW_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        *[
            pa.field(
                f"{prefix}{score_name}_{field_name}",
                pa.float64(),
                nullable=(
                    prefix != ""
                    or field_name in _SIMILARITY_NOVEL_FIELDS
                ),
            )
            for prefix in _SEMANTIC_REPRESENTATION_PREFIXES
            for score_name in SIMILARITY_SCORE_NAMES
            for field_name in (
                *_SIMILARITY_REQUIRED_FIELDS,
                *_SIMILARITY_NOVEL_FIELDS,
            )
        ],
    ]
)

TAXONOMY_ROW_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("same_parent_fraction", pa.float64(), nullable=False),
        pa.field("any_target_same_parent", pa.float64(), nullable=False),
        pa.field("all_targets_same_parent", pa.float64(), nullable=False),
        pa.field(
            "same_parent_recent5_fraction",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "any_target_same_parent_recent5",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "all_targets_same_parent_recent5",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "same_parent_last_fraction",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "any_target_same_parent_last",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "all_targets_same_parent_last",
            pa.float64(),
            nullable=False,
        ),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("history_unique_news", pa.int32(), nullable=False),
        pa.field(
            "history_unique_subcategories",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "history_unique_parents",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "history_repetition_count",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "history_repetition_rate",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "history_subcategory_entropy",
            pa.float64(),
            nullable=False,
        ),
        pa.field("target_count", pa.int32(), nullable=False),
        pa.field("positive_title_count", pa.int32(), nullable=False),
        pa.field("control_count", pa.int32(), nullable=False),
        pa.field("novel_control_count", pa.int32(), nullable=False),
        pa.field(
            "duplicate_control_token_count",
            pa.int32(),
            nullable=False,
        ),
        pa.field("has_novel_controls", pa.float64(), nullable=False),
    ]
)

_BASE_FIELD_NAMES = set(ANALYSIS_ROW_SCHEMA.names)
_ENRICHMENT_FIELDS = [
    *[
        field
        for field in SEMANTIC_ROW_SCHEMA
        if field.name not in _BASE_FIELD_NAMES
    ],
    *[
        field
        for field in TAXONOMY_ROW_SCHEMA
        if field.name not in _BASE_FIELD_NAMES
    ],
]
ENRICHED_ANALYSIS_ROW_SCHEMA = pa.schema(
    [*ANALYSIS_ROW_SCHEMA, *_ENRICHMENT_FIELDS]
)


def _normalise_matrix(
    values: np.ndarray,
    *,
    name: str,
    dimensions: int | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if dimensions is not None and matrix.shape[1] != dimensions:
        raise ValueError(
            f"{name} has {matrix.shape[1]} dimensions; expected {dimensions}"
        )
    if matrix.shape[0] == 0:
        if allow_empty:
            return matrix
        raise ValueError(f"{name} must contain at least one vector")
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains empty or non-finite vectors")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero-norm vector")
    return matrix / norms[:, np.newaxis]


def _title_scores(
    query_vectors: np.ndarray,
    history_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    cosine = query_vectors @ history_vectors.T
    top_count = min(3, history_vectors.shape[0])
    centroid = history_vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if not math.isfinite(centroid_norm) or centroid_norm == 0.0:
        raise ValueError("History centroid has zero norm")
    recent = history_vectors[-min(5, history_vectors.shape[0]) :]
    return {
        "nearest_50": cosine.max(axis=1),
        "top3_mean": np.sort(cosine, axis=1)[:, -top_count:].mean(
            axis=1
        ),
        "centroid": query_vectors @ (centroid / centroid_norm),
        "nearest_5": (query_vectors @ recent.T).max(axis=1),
        "last": query_vectors @ history_vectors[-1],
    }


def _auc(positive_scores: np.ndarray, control_scores: np.ndarray) -> float:
    comparisons = positive_scores[:, np.newaxis] - control_scores
    return float(
        (
            np.count_nonzero(comparisons > 0.0)
            + 0.5 * np.count_nonzero(comparisons == 0.0)
        )
        / comparisons.size
    )


def _hit_at_one(
    positive_scores: np.ndarray,
    control_scores: np.ndarray,
) -> float:
    positive_max = float(positive_scores.max())
    control_max = float(control_scores.max())
    if positive_max > control_max:
        return 1.0
    if positive_max < control_max:
        return 0.0
    return 0.5


def _score_comparison(
    positive_scores: np.ndarray,
    control_scores: np.ndarray,
) -> dict[str, float]:
    positive_mean = float(positive_scores.mean())
    control_mean = float(control_scores.mean())
    control_max = float(control_scores.max())
    return {
        "control_mean": control_mean,
        "delta": positive_mean - control_mean,
        "auc": _auc(positive_scores, control_scores),
        "hit_at_1": _hit_at_one(positive_scores, control_scores),
        "positive_le_control_mean": float(
            positive_mean <= control_mean
        ),
        "positive_le_control_max": float(positive_mean <= control_max),
    }


def compute_similarity_features(
    *,
    history_vectors: np.ndarray,
    positive_vectors: np.ndarray,
    control_vectors: np.ndarray,
    novel_control_vectors: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Compute row-level title similarity and within-impression ranking."""

    history = _normalise_matrix(history_vectors, name="history_vectors")
    dimensions = history.shape[1]
    positives = _normalise_matrix(
        positive_vectors,
        name="positive_vectors",
        dimensions=dimensions,
    )
    controls = _normalise_matrix(
        control_vectors,
        name="control_vectors",
        dimensions=dimensions,
    )
    novel_controls = _normalise_matrix(
        novel_control_vectors,
        name="novel_control_vectors",
        dimensions=dimensions,
        allow_empty=True,
    )

    positive_scores = _title_scores(positives, history)
    control_scores = _title_scores(controls, history)
    novel_scores = (
        _title_scores(novel_controls, history) if len(novel_controls) else {}
    )
    features: dict[str, dict[str, float | None]] = {}
    for score_name in SIMILARITY_SCORE_NAMES:
        positive = positive_scores[score_name]
        result: dict[str, float | None] = {
            "positive_mean": float(positive.mean()),
            **_score_comparison(positive, control_scores[score_name]),
        }
        if len(novel_controls):
            novel_comparison = _score_comparison(
                positive,
                novel_scores[score_name],
            )
            result.update(
                {
                    "novel_control_mean": novel_comparison[
                        "control_mean"
                    ],
                    "novel_delta": novel_comparison["delta"],
                    "novel_auc": novel_comparison["auc"],
                    "novel_hit_at_1": novel_comparison["hit_at_1"],
                    "positive_le_novel_control_mean": novel_comparison[
                        "positive_le_control_mean"
                    ],
                    "positive_le_novel_control_max": novel_comparison[
                        "positive_le_control_max"
                    ],
                }
            )
        else:
            result.update(
                {
                    "novel_control_mean": None,
                    "novel_delta": None,
                    "novel_auc": None,
                    "novel_hit_at_1": None,
                    "positive_le_novel_control_mean": None,
                    "positive_le_novel_control_max": None,
                }
            )
        features[score_name] = result
    return features


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _title_corpus_sha256(titles: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for title in titles:
        encoded = title.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _analysis_titles(path: Path) -> list[str]:
    required = {"history", "positive_news", "control_news"}
    parquet = pq.ParquetFile(_require_file(path))
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{path}: missing title-bearing columns {sorted(missing)}"
        )
    unique_titles: dict[str, None] = {}
    rows = 0
    for batch in parquet.iter_batches(
        columns=["history", "positive_news", "control_news"],
        batch_size=1_024,
    ):
        for row in batch.to_pylist():
            rows += 1
            for collection_name in (
                "history",
                "positive_news",
                "control_news",
            ):
                collection = row[collection_name]
                if not collection:
                    raise ValueError(
                        f"{path}: row {rows} has empty {collection_name}"
                    )
                for item in collection:
                    title = str(item["title"])
                    if not title.strip():
                        raise ValueError(
                            f"{path}: row {rows} contains an empty title"
                        )
                    unique_titles.setdefault(title, None)
    if rows == 0:
        raise ValueError(f"{path}: no analysis rows")
    return list(unique_titles)


def _analysis_category_names(path: Path) -> list[str]:
    required = {
        "history",
        "target_categories",
        "target_subcategories",
        "control_news",
    }
    parquet = pq.ParquetFile(_require_file(path))
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{path}: missing category-name columns {sorted(missing)}"
        )
    unique_names: dict[str, None] = {}
    rows = 0
    for batch in parquet.iter_batches(
        columns=[
            "history",
            "target_categories",
            "target_subcategories",
            "control_news",
        ],
        batch_size=1_024,
    ):
        for row in batch.to_pylist():
            rows += 1
            names = [
                *(str(item["category"]) for item in row["history"]),
                *(str(item["subcategory"]) for item in row["history"]),
                *(str(value) for value in row["target_categories"]),
                *(str(value) for value in row["target_subcategories"]),
                *(str(item["category"]) for item in row["control_news"]),
                *(str(item["subcategory"]) for item in row["control_news"]),
            ]
            if any(not name.strip() for name in names):
                raise ValueError(
                    f"{path}: row {rows} contains an empty category name"
                )
            for name in names:
                unique_names.setdefault(name, None)
    if rows == 0:
        raise ValueError(f"{path}: no analysis rows")
    return list(unique_names)


def _encode_texts(
    encoder: TextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
    context: str,
) -> np.ndarray:
    try:
        values = encoder.encode(texts, batch_size=batch_size)
    except (RuntimeError, ValueError):
        raise
    except Exception as error:
        raise RuntimeError(f"{context} encoder failed") from error
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError(
            f"{context} encoder returned shape "
            f"{matrix.shape}; expected ({len(texts)}, dimensions)"
        )
    return np.asarray(
        _normalise_matrix(matrix, name=f"{context} embeddings"),
        dtype=np.float32,
    )


def write_title_embeddings(
    analysis_rows_path: Path,
    output_path: Path,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = DEFAULT_TITLE_ENCODER_NAME,
    revision: str = DEFAULT_TITLE_ENCODER_REVISION,
    device: str | None = None,
    batch_size: int = 64,
    write_batch_size: int = 2_048,
    source_path_label: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Encode every unique title in the aligned analysis rows once."""

    analysis_rows_path = Path(analysis_rows_path)
    output_path = Path(output_path)
    if batch_size <= 0 or write_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Title embedding output already exists: {output_path}"
        )
    titles = _analysis_titles(analysis_rows_path)
    encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    embeddings = _encode_texts(
        encoder,
        titles,
        batch_size=batch_size,
        context="Title",
    )
    dimensions = int(embeddings.shape[1])
    metadata = {
        "artifact_type": "sft_title_embeddings",
        "artifact_version": 1,
        "source_path": str(source_path_label or analysis_rows_path),
        "source_title_columns": [
            "history.title",
            "positive_news.title",
            "control_news.title",
        ],
        "title_corpus_sha256": _title_corpus_sha256(titles),
        "unique_titles": len(titles),
        "model": {
            "name": model_name,
            "revision": revision,
            "device": device,
            "batch_size": batch_size,
        },
        "embedding_dimensions": dimensions,
        "normalised": True,
    }
    schema = TITLE_EMBEDDING_SCHEMA.with_metadata(
        {
            b"metadata": json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        }
    )

    def rows() -> Iterator[dict[str, Any]]:
        for title, embedding in zip(titles, embeddings, strict=True):
            yield {
                "title_sha256": _sha256_text(title),
                "title": title,
                "embedding": embedding.tolist(),
            }

    written = _write_rows_atomically(
        output_path,
        rows(),
        schema=schema,
        batch_size=write_batch_size,
    )
    return {
        "unique_titles": written,
        "dimensions": dimensions,
        "path": str(output_path),
        "metadata": metadata,
    }


def write_category_name_embeddings(
    analysis_rows_path: Path,
    output_path: Path,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = DEFAULT_TITLE_ENCODER_NAME,
    revision: str = DEFAULT_TITLE_ENCODER_REVISION,
    device: str | None = None,
    batch_size: int = 64,
    write_batch_size: int = 2_048,
    source_path_label: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Encode every unique parent and subcategory name once."""

    analysis_rows_path = Path(analysis_rows_path)
    output_path = Path(output_path)
    if batch_size <= 0 or write_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Category-name embedding output already exists: {output_path}"
        )
    names = _analysis_category_names(analysis_rows_path)
    encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    embeddings = _encode_texts(
        encoder,
        names,
        batch_size=batch_size,
        context="Category-name",
    )
    dimensions = int(embeddings.shape[1])
    metadata = {
        "artifact_type": "sft_category_name_embeddings",
        "artifact_version": 1,
        "source_path": str(source_path_label or analysis_rows_path),
        "source_name_columns": [
            "history.category",
            "history.subcategory",
            "target_categories",
            "target_subcategories",
            "control_news.category",
            "control_news.subcategory",
        ],
        "name_corpus_sha256": _title_corpus_sha256(names),
        "unique_names": len(names),
        "model": {
            "name": model_name,
            "revision": revision,
            "device": device,
            "batch_size": batch_size,
        },
        "embedding_dimensions": dimensions,
        "normalised": True,
    }
    schema = CATEGORY_NAME_EMBEDDING_SCHEMA.with_metadata(
        {
            b"metadata": json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        }
    )

    def rows() -> Iterator[dict[str, Any]]:
        for name, embedding in zip(names, embeddings, strict=True):
            yield {
                "name_sha256": _sha256_text(name),
                "name": name,
                "embedding": embedding.tolist(),
            }

    written = _write_rows_atomically(
        output_path,
        rows(),
        schema=schema,
        batch_size=write_batch_size,
    )
    return {
        "unique_names": written,
        "dimensions": dimensions,
        "path": str(output_path),
        "metadata": metadata,
    }


def _load_title_embeddings(
    path: Path,
) -> tuple[dict[str, np.ndarray], int]:
    parquet = pq.ParquetFile(_require_file(path))
    missing = set(TITLE_EMBEDDING_SCHEMA.names) - set(
        parquet.schema_arrow.names
    )
    if missing:
        raise ValueError(
            f"{path}: missing embedding columns {sorted(missing)}"
        )
    title_vectors: dict[str, np.ndarray] = {}
    dimensions: int | None = None
    for row in _iter_parquet_rows(
        path,
        columns=TITLE_EMBEDDING_SCHEMA.names,
    ):
        title = str(row["title"])
        if row["title_sha256"] != _sha256_text(title):
            raise ValueError(f"{path}: title hash mismatch for {title!r}")
        if title in title_vectors:
            raise ValueError(f"{path}: duplicate embedded title {title!r}")
        vector = _normalise_matrix(
            np.asarray([row["embedding"]], dtype=np.float64),
            name=f"embedding for {title!r}",
            dimensions=dimensions,
        )[0]
        if dimensions is None:
            dimensions = int(vector.shape[0])
        title_vectors[title] = vector
    if not title_vectors or dimensions is None:
        raise ValueError(f"{path}: no title embeddings")
    return title_vectors, dimensions


def _load_category_name_embeddings(
    path: Path,
) -> tuple[dict[str, np.ndarray], int]:
    parquet = pq.ParquetFile(_require_file(path))
    missing = set(CATEGORY_NAME_EMBEDDING_SCHEMA.names) - set(
        parquet.schema_arrow.names
    )
    if missing:
        raise ValueError(
            f"{path}: missing category embedding columns {sorted(missing)}"
        )
    name_vectors: dict[str, np.ndarray] = {}
    dimensions: int | None = None
    for row in _iter_parquet_rows(
        path,
        columns=CATEGORY_NAME_EMBEDDING_SCHEMA.names,
    ):
        name = str(row["name"])
        if row["name_sha256"] != _sha256_text(name):
            raise ValueError(
                f"{path}: category name hash mismatch for {name!r}"
            )
        if name in name_vectors:
            raise ValueError(
                f"{path}: duplicate embedded category name {name!r}"
            )
        vector = _normalise_matrix(
            np.asarray([row["embedding"]], dtype=np.float64),
            name=f"embedding for category name {name!r}",
            dimensions=dimensions,
        )[0]
        if dimensions is None:
            dimensions = int(vector.shape[0])
        name_vectors[name] = vector
    if not name_vectors or dimensions is None:
        raise ValueError(f"{path}: no category-name embeddings")
    return name_vectors, dimensions


def _vectors_for_news(
    news_items: Sequence[Mapping[str, Any]],
    title_vectors: Mapping[str, np.ndarray],
    *,
    dimensions: int,
) -> np.ndarray:
    if not news_items:
        return np.empty((0, dimensions), dtype=np.float64)
    vectors = []
    for item in news_items:
        title = str(item["title"])
        try:
            vectors.append(title_vectors[title])
        except KeyError as error:
            raise ValueError(
                f"Missing title embedding for {title!r}"
            ) from error
    return np.asarray(vectors, dtype=np.float64)


def _vectors_for_names(
    names: Sequence[str],
    name_vectors: Mapping[str, np.ndarray],
    *,
    dimensions: int,
) -> np.ndarray:
    if not names:
        return np.empty((0, dimensions), dtype=np.float64)
    vectors = []
    for name in names:
        try:
            vectors.append(name_vectors[str(name)])
        except KeyError as error:
            raise ValueError(
                f"Missing category-name embedding for {name!r}"
            ) from error
    return np.asarray(vectors, dtype=np.float64)


def _flatten_similarity_features(
    row_id: str,
    split: str,
    features: Mapping[str, Mapping[str, float | None]],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    output: dict[str, Any] = {"row_id": row_id, "split": split}
    for score_name in SIMILARITY_SCORE_NAMES:
        values = features[score_name]
        for field_name in (
            *_SIMILARITY_REQUIRED_FIELDS,
            *_SIMILARITY_NOVEL_FIELDS,
        ):
            output[f"{prefix}{score_name}_{field_name}"] = values[
                field_name
            ]
    return output


def write_sft_semantic_rows(
    analysis_rows_path: Path,
    title_embeddings_path: Path,
    output_path: Path,
    *,
    category_name_embeddings_path: Path | None = None,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one row of semantic comparison metrics per SFT row."""

    analysis_rows_path = Path(analysis_rows_path)
    title_embeddings_path = Path(title_embeddings_path)
    category_name_embeddings_path = (
        None
        if category_name_embeddings_path is None
        else Path(category_name_embeddings_path)
    )
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Semantic row output already exists: {output_path}"
        )
    title_vectors, dimensions = _load_title_embeddings(
        title_embeddings_path
    )
    category_vectors: dict[str, np.ndarray] | None = None
    category_dimensions: int | None = None
    if category_name_embeddings_path is not None:
        category_vectors, category_dimensions = (
            _load_category_name_embeddings(
                category_name_embeddings_path
            )
        )
    split_counts: Counter[str] = Counter()
    missing_novel_counts: Counter[str] = Counter()

    def rows() -> Iterator[dict[str, Any]]:
        seen_ids: set[str] = set()
        for row in _iter_parquet_rows(
            _require_file(analysis_rows_path),
            columns=[
                "row_id",
                "split",
                "history",
                "positive_news",
                "control_news",
                "novel_control_news",
                "target_categories",
                "target_subcategories",
            ],
        ):
            row_id = str(row["row_id"])
            split = str(row["split"])
            if row_id in seen_ids:
                raise ValueError(
                    f"Duplicate analysis row_id {row_id!r}"
                )
            seen_ids.add(row_id)
            split_counts[split] += 1
            if not row["novel_control_news"]:
                missing_novel_counts[split] += 1
            features = compute_similarity_features(
                history_vectors=_vectors_for_news(
                    row["history"],
                    title_vectors,
                    dimensions=dimensions,
                ),
                positive_vectors=_vectors_for_news(
                    row["positive_news"],
                    title_vectors,
                    dimensions=dimensions,
                ),
                control_vectors=_vectors_for_news(
                    row["control_news"],
                    title_vectors,
                    dimensions=dimensions,
                ),
                novel_control_vectors=_vectors_for_news(
                    row["novel_control_news"],
                    title_vectors,
                    dimensions=dimensions,
                ),
            )
            output = _flatten_similarity_features(
                row_id,
                split,
                features,
            )
            if (
                category_vectors is not None
                and category_dimensions is not None
            ):
                for prefix, history_field, target_field in (
                    (
                        "subcategory_name_",
                        "subcategory",
                        "target_subcategories",
                    ),
                    (
                        "parent_name_",
                        "category",
                        "target_categories",
                    ),
                ):
                    category_features = compute_similarity_features(
                        history_vectors=_vectors_for_names(
                            [
                                str(item[history_field])
                                for item in row["history"]
                            ],
                            category_vectors,
                            dimensions=category_dimensions,
                        ),
                        positive_vectors=_vectors_for_names(
                            [
                                str(value)
                                for value in row[target_field]
                            ],
                            category_vectors,
                            dimensions=category_dimensions,
                        ),
                        control_vectors=_vectors_for_names(
                            [
                                str(item[history_field])
                                for item in row["control_news"]
                            ],
                            category_vectors,
                            dimensions=category_dimensions,
                        ),
                        novel_control_vectors=_vectors_for_names(
                            [
                                str(item[history_field])
                                for item in row["novel_control_news"]
                            ],
                            category_vectors,
                            dimensions=category_dimensions,
                        ),
                    )
                    output.update(
                        _flatten_similarity_features(
                            row_id,
                            split,
                            category_features,
                            prefix=prefix,
                        )
                    )
            yield output

    written = _write_rows_atomically(
        output_path,
        rows(),
        schema=SEMANTIC_ROW_SCHEMA,
        batch_size=batch_size,
    )
    return {
        "rows": dict(split_counts),
        "total_rows": written,
        "rows_without_novel_controls": dict(missing_novel_counts),
        "embedding_dimensions": dimensions,
        "category_name_embedding_dimensions": category_dimensions,
        "path": str(output_path),
    }


def _parent_relation(
    target_categories: Sequence[str],
    history_categories: set[str],
) -> tuple[float, float, float]:
    matches = [
        float(category in history_categories)
        for category in target_categories
    ]
    return (
        float(np.mean(matches)),
        float(any(matches)),
        float(all(matches)),
    )


def compute_taxonomy_features(
    *,
    history: Sequence[Mapping[str, Any]],
    target_categories: Sequence[str],
) -> dict[str, float | int]:
    """Compute row-equal parent overlap and H-history breadth features."""

    if not history:
        raise ValueError("history must be non-empty")
    if not target_categories:
        raise ValueError("target_categories must be non-empty")
    cleaned_targets = [str(value) for value in target_categories]
    if any(not value.strip() for value in cleaned_targets):
        raise ValueError("target_categories contains an empty value")
    history_parents = [str(item["category"]) for item in history]
    history_subcategories = [
        str(item["subcategory"]) for item in history
    ]
    history_news_ids = [str(item["news_id"]) for item in history]
    if any(not value.strip() for value in history_parents):
        raise ValueError("history contains an empty parent category")
    if any(not value.strip() for value in history_subcategories):
        raise ValueError("history contains an empty subcategory")
    if any(not value.strip() for value in history_news_ids):
        raise ValueError("history contains an empty news ID")

    full = _parent_relation(cleaned_targets, set(history_parents))
    recent = _parent_relation(
        cleaned_targets,
        set(history_parents[-min(5, len(history_parents)) :]),
    )
    last = _parent_relation(cleaned_targets, {history_parents[-1]})
    subcategory_counts = Counter(history_subcategories)
    history_length = len(history)
    entropy = -sum(
        (count / history_length) * math.log(count / history_length)
        for count in subcategory_counts.values()
    )
    unique_news = len(set(history_news_ids))
    repetition_count = history_length - unique_news
    return {
        "same_parent_fraction": full[0],
        "any_target_same_parent": full[1],
        "all_targets_same_parent": full[2],
        "same_parent_recent5_fraction": recent[0],
        "any_target_same_parent_recent5": recent[1],
        "all_targets_same_parent_recent5": recent[2],
        "same_parent_last_fraction": last[0],
        "any_target_same_parent_last": last[1],
        "all_targets_same_parent_last": last[2],
        "history_length": history_length,
        "history_unique_news": unique_news,
        "history_unique_subcategories": len(subcategory_counts),
        "history_unique_parents": len(set(history_parents)),
        "history_repetition_count": repetition_count,
        "history_repetition_rate": repetition_count / history_length,
        "history_subcategory_entropy": entropy,
    }


def write_sft_taxonomy_rows(
    analysis_rows_path: Path,
    output_path: Path,
    *,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write taxonomy, history-breadth, and exposure features per row."""

    analysis_rows_path = Path(analysis_rows_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Taxonomy row output already exists: {output_path}"
        )
    split_counts: Counter[str] = Counter()

    def rows() -> Iterator[dict[str, Any]]:
        seen_ids: set[str] = set()
        for row in _iter_parquet_rows(
            _require_file(analysis_rows_path),
            columns=[
                "row_id",
                "split",
                "history",
                "target_categories",
                "target_count",
                "positive_title_count",
                "control_count",
                "novel_control_count",
                "duplicate_control_token_count",
            ],
        ):
            row_id = str(row["row_id"])
            if row_id in seen_ids:
                raise ValueError(
                    f"Duplicate analysis row_id {row_id!r}"
                )
            seen_ids.add(row_id)
            split = str(row["split"])
            split_counts[split] += 1
            features = compute_taxonomy_features(
                history=row["history"],
                target_categories=row["target_categories"],
            )
            yield {
                "row_id": row_id,
                "split": split,
                **features,
                "target_count": int(row["target_count"]),
                "positive_title_count": int(
                    row["positive_title_count"]
                ),
                "control_count": int(row["control_count"]),
                "novel_control_count": int(
                    row["novel_control_count"]
                ),
                "duplicate_control_token_count": int(
                    row["duplicate_control_token_count"]
                ),
                "has_novel_controls": float(
                    int(row["novel_control_count"]) > 0
                ),
            }

    written = _write_rows_atomically(
        output_path,
        rows(),
        schema=TAXONOMY_ROW_SCHEMA,
        batch_size=batch_size,
    )
    return {
        "rows": dict(split_counts),
        "total_rows": written,
        "path": str(output_path),
    }


def write_sft_enriched_rows(
    base_rows_path: Path,
    semantic_rows_path: Path,
    taxonomy_rows_path: Path,
    output_path: Path,
    *,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge aligned base, semantic, and taxonomy rows by row ID."""

    base_rows_path = Path(base_rows_path)
    semantic_rows_path = Path(semantic_rows_path)
    taxonomy_rows_path = Path(taxonomy_rows_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Enriched analysis output already exists: {output_path}"
        )
    split_counts: Counter[str] = Counter()
    missing = object()

    def rows() -> Iterator[dict[str, Any]]:
        base_iterator = _iter_parquet_rows(_require_file(base_rows_path))
        semantic_iterator = _iter_parquet_rows(
            _require_file(semantic_rows_path)
        )
        taxonomy_iterator = _iter_parquet_rows(
            _require_file(taxonomy_rows_path)
        )
        for row_number, (base, semantic, taxonomy) in enumerate(
            zip_longest(
                base_iterator,
                semantic_iterator,
                taxonomy_iterator,
                fillvalue=missing,
            ),
            start=1,
        ):
            if missing in (base, semantic, taxonomy):
                raise ValueError(
                    "Base, semantic, and taxonomy row counts differ"
                )
            assert isinstance(base, dict)
            assert isinstance(semantic, dict)
            assert isinstance(taxonomy, dict)
            identities = {
                (str(row["row_id"]), str(row["split"]))
                for row in (base, semantic, taxonomy)
            }
            if len(identities) != 1:
                raise ValueError(
                    f"Enrichment row {row_number} identities differ: "
                    f"{sorted(identities)}"
                )
            output = dict(base)
            for enrichment in (semantic, taxonomy):
                for name, value in enrichment.items():
                    if name not in _BASE_FIELD_NAMES:
                        output[name] = value
            split_counts[str(base["split"])] += 1
            yield output

    written = _write_rows_atomically(
        output_path,
        rows(),
        schema=ENRICHED_ANALYSIS_ROW_SCHEMA,
        batch_size=batch_size,
    )
    return {
        "rows": dict(split_counts),
        "total_rows": written,
        "path": str(output_path),
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int = 128,
) -> dict[str, float | int]:
    """Estimate a deterministic row-bootstrap interval for a mean."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional")
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        raise ValueError("values has no finite observations")
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("replicates and batch_size must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    offset = 0
    while offset < replicates:
        current = min(batch_size, replicates - offset)
        indices = rng.integers(
            0,
            observations.size,
            size=(current, observations.size),
        )
        estimates[offset : offset + current] = observations[indices].mean(
            axis=1
        )
        offset += current
    alpha = (1.0 - confidence) / 2.0
    return {
        "rows": int(observations.size),
        "mean": float(observations.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _parent_match_rows(
    history_matrix: np.ndarray,
    target_ids: np.ndarray,
    target_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ids = np.arange(history_matrix.shape[0])[:, np.newaxis]
    matches = history_matrix[row_ids, target_ids] & target_valid
    target_counts = target_valid.sum(axis=1)
    fractions = matches.sum(axis=1) / target_counts
    any_matches = matches.any(axis=1).astype(np.float64)
    all_matches = (
        matches.sum(axis=1) == target_counts
    ).astype(np.float64)
    return fractions, any_matches, all_matches


def compute_parent_permutation_baseline(
    *,
    history_categories: Sequence[Sequence[str]],
    target_categories: Sequence[Sequence[str]],
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float | None]]:
    """Shuffle target sets across rows while preserving their composition."""

    if len(history_categories) != len(target_categories):
        raise ValueError("history and target row counts differ")
    if not history_categories:
        raise ValueError("permutation baseline requires rows")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if any(not values for values in history_categories):
        raise ValueError("history category sets must be non-empty")
    if any(not values for values in target_categories):
        raise ValueError("target category sets must be non-empty")

    vocabulary = sorted(
        {
            str(category)
            for rows in (*history_categories, *target_categories)
            for category in rows
        }
    )
    category_to_id = {
        category: index for index, category in enumerate(vocabulary)
    }
    row_count = len(history_categories)
    history_matrix = np.zeros(
        (row_count, len(vocabulary)),
        dtype=np.bool_,
    )
    for row_index, categories in enumerate(history_categories):
        history_matrix[
            row_index,
            [category_to_id[str(value)] for value in categories],
        ] = True
    max_targets = max(len(values) for values in target_categories)
    target_ids = np.zeros((row_count, max_targets), dtype=np.intp)
    target_valid = np.zeros((row_count, max_targets), dtype=np.bool_)
    for row_index, categories in enumerate(target_categories):
        count = len(categories)
        target_ids[row_index, :count] = [
            category_to_id[str(value)] for value in categories
        ]
        target_valid[row_index, :count] = True

    observed_rows = _parent_match_rows(
        history_matrix,
        target_ids,
        target_valid,
    )
    null_means = np.empty((replicates, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        permutation = rng.permutation(row_count)
        permuted_rows = _parent_match_rows(
            history_matrix,
            target_ids[permutation],
            target_valid[permutation],
        )
        null_means[replicate] = [
            float(values.mean()) for values in permuted_rows
        ]

    alpha = (1.0 - confidence) / 2.0
    names = (
        "same_parent_fraction",
        "any_target_same_parent",
        "all_targets_same_parent",
    )
    output: dict[str, dict[str, float | None]] = {}
    for metric_index, name in enumerate(names):
        observed = float(observed_rows[metric_index].mean())
        expected = float(null_means[:, metric_index].mean())
        output[name] = {
            "observed": observed,
            "expected": expected,
            "null_ci_low": float(
                np.quantile(null_means[:, metric_index], alpha)
            ),
            "null_ci_high": float(
                np.quantile(
                    null_means[:, metric_index],
                    1.0 - alpha,
                )
            ),
            "observed_minus_expected": observed - expected,
            "lift": observed / expected if expected else None,
        }
    return output


def compute_prompt_overlap(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit prompt duplication, conflicts, and Train/Dev leakage."""

    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if len(train_rows) + len(dev_rows) != len(rows):
        raise ValueError("Prompt audit rows must use train/dev splits")
    train_users = {str(row["user_id"]) for row in train_rows}
    dev_users = {str(row["user_id"]) for row in dev_rows}
    overlapping_users = train_users.intersection(dev_users)

    tasks: dict[str, Any] = {}
    for task_name in ("behavior", "interest"):
        hash_field = f"{task_name}_prompt_sha256"
        train_by_prompt: dict[str, list[Mapping[str, Any]]] = {}
        dev_by_prompt: dict[str, list[Mapping[str, Any]]] = {}
        for row in train_rows:
            train_by_prompt.setdefault(str(row[hash_field]), []).append(row)
        for row in dev_rows:
            dev_by_prompt.setdefault(str(row[hash_field]), []).append(row)

        conflicting_hashes = {
            prompt_hash
            for prompt_hash, prompt_rows in train_by_prompt.items()
            if len(
                {
                    frozenset(
                        str(value)
                        for value in row["target_subcategories"]
                    )
                    for row in prompt_rows
                }
            )
            > 1
        }
        exact_overlap_rows = [
            row
            for row in dev_rows
            if str(row[hash_field]) in train_by_prompt
        ]
        any_target_seen = 0
        all_targets_seen = 0
        for row in exact_overlap_rows:
            train_targets = {
                str(value)
                for train_row in train_by_prompt[str(row[hash_field])]
                for value in train_row["target_subcategories"]
            }
            dev_targets = {
                str(value) for value in row["target_subcategories"]
            }
            any_target_seen += int(bool(dev_targets & train_targets))
            all_targets_seen += int(dev_targets <= train_targets)

        tasks[task_name] = {
            "train_rows": len(train_rows),
            "train_unique_prompts": len(train_by_prompt),
            "train_duplicate_prompt_count": sum(
                len(prompt_rows) > 1
                for prompt_rows in train_by_prompt.values()
            ),
            "train_duplicate_row_excess": sum(
                max(0, len(prompt_rows) - 1)
                for prompt_rows in train_by_prompt.values()
            ),
            "train_conflicting_prompt_count": len(conflicting_hashes),
            "train_conflicting_rows": sum(
                len(train_by_prompt[prompt_hash])
                for prompt_hash in conflicting_hashes
            ),
            "dev_rows": len(dev_rows),
            "dev_unique_prompts": len(dev_by_prompt),
            "dev_duplicate_prompt_count": sum(
                len(prompt_rows) > 1
                for prompt_rows in dev_by_prompt.values()
            ),
            "dev_duplicate_row_excess": sum(
                max(0, len(prompt_rows) - 1)
                for prompt_rows in dev_by_prompt.values()
            ),
            "dev_exact_prompt_overlap_rows": len(exact_overlap_rows),
            "dev_any_target_seen_for_prompt_rows": any_target_seen,
            "dev_all_targets_seen_for_prompt_rows": all_targets_seen,
        }

    return {
        "rows": {"train": len(train_rows), "dev": len(dev_rows)},
        "user_overlap": {
            "train_users": len(train_users),
            "dev_users": len(dev_users),
            "users": len(overlapping_users),
            "dev_rows": sum(
                str(row["user_id"]) in overlapping_users
                for row in dev_rows
            ),
        },
        "tasks": tasks,
    }


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _bootstrap_column_intervals(
    columns: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int = 32,
) -> dict[str, dict[str, float | int]]:
    if not columns:
        raise ValueError("bootstrap columns must be non-empty")
    names = list(columns)
    matrix = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in names]
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("bootstrap matrix must contain rows")
    if any(len(columns[name]) != matrix.shape[0] for name in names):
        raise ValueError("bootstrap column lengths differ")
    valid = np.isfinite(matrix).astype(np.float64)
    if np.any(valid.sum(axis=0) == 0):
        empty = [
            name
            for index, name in enumerate(names)
            if valid[:, index].sum() == 0
        ]
        raise ValueError(f"bootstrap columns have no values: {empty}")
    filled = np.nan_to_num(matrix, nan=0.0)
    estimates = np.empty(
        (replicates, matrix.shape[1]),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    probabilities = np.full(
        matrix.shape[0],
        1.0 / matrix.shape[0],
        dtype=np.float64,
    )
    offset = 0
    while offset < replicates:
        current = min(batch_size, replicates - offset)
        weights = rng.multinomial(
            matrix.shape[0],
            probabilities,
            size=current,
        ).astype(np.float64)
        numerators = weights @ filled
        denominators = weights @ valid
        estimates[offset : offset + current] = np.divide(
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan),
            where=denominators > 0,
        )
        offset += current
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(names):
        values = matrix[:, index]
        finite = values[np.isfinite(values)]
        output[name] = {
            "rows": int(finite.size),
            "mean": float(finite.mean()),
            "ci_low": float(
                np.nanquantile(estimates[:, index], alpha)
            ),
            "ci_high": float(
                np.nanquantile(estimates[:, index], 1.0 - alpha)
            ),
        }
    return output


def _load_compact_stat_rows(
    path: Path,
) -> dict[str, list[dict[str, Any]]]:
    scalar_columns = [
        field.name
        for field in ENRICHED_ANALYSIS_ROW_SCHEMA
        if field.name.startswith(
            (
                "nearest_",
                "top3_",
                "centroid_",
                "last_",
                "subcategory_name_",
                "parent_name_",
                "same_parent",
                "any_target_same_parent",
                "all_targets_same_parent",
                "history_unique",
                "history_repetition",
                "history_subcategory_entropy",
            )
        )
    ]
    columns = [
        "row_id",
        "split",
        "user_id",
        "history",
        "target_categories",
        "target_subcategories",
        "behavior_prompt_sha256",
        "interest_prompt_sha256",
        "history_length",
        "target_count",
        "positive_title_count",
        "control_count",
        "novel_control_count",
        "duplicate_control_token_count",
        *scalar_columns,
    ]
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
    }
    seen_ids: set[str] = set()
    for row in _iter_parquet_rows(
        _require_file(path),
        columns=list(dict.fromkeys(columns)),
    ):
        row_id = str(row["row_id"])
        split = str(row["split"])
        if split not in rows_by_split:
            raise ValueError(f"Unexpected analysis split {split!r}")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate enriched row ID {row_id!r}")
        seen_ids.add(row_id)
        history_parents = [
            str(item["category"]) for item in row["history"]
        ]
        compact = {
            name: row[name]
            for name in row
            if name not in {"history"}
        }
        compact.update(
            {
                "history_parents_full": tuple(
                    sorted(set(history_parents))
                ),
                "history_parents_recent5": tuple(
                    sorted(set(history_parents[-5:]))
                ),
                "history_parents_last": (history_parents[-1],),
                "last_history_parent": history_parents[-1],
            }
        )
        rows_by_split[split].append(compact)
    if not rows_by_split["train"] or not rows_by_split["dev"]:
        raise ValueError("Statistics require non-empty Train and Dev rows")
    return rows_by_split


def _finite_quantiles(
    values: np.ndarray,
) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot compute quantiles without finite values")
    return {
        f"positive_p{percentile}": float(
            np.quantile(finite, percentile / 100.0)
        )
        for percentile in (10, 25, 50, 75, 90)
    }


def _similarity_metric_rows(
    *,
    split: str,
    split_row_count: int,
    bootstrap_columns: Mapping[str, np.ndarray],
    intervals: Mapping[str, Mapping[str, float | int]],
    field_prefix: str,
    identity: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    identity = {} if identity is None else dict(identity)
    for score_name in SIMILARITY_SCORE_NAMES:
        summary[score_name] = {}
        for scope in ("novel", "all"):
            if scope == "all":
                positive_key = (
                    f"{field_prefix}{score_name}_positive_mean"
                )
                control_key = (
                    f"{field_prefix}{score_name}_control_mean"
                )
                delta_key = f"{field_prefix}{score_name}_delta"
                auc_key = f"{field_prefix}{score_name}_auc"
                hit_key = f"{field_prefix}{score_name}_hit_at_1"
                le_mean_key = (
                    f"{field_prefix}{score_name}_"
                    "positive_le_control_mean"
                )
                le_max_key = (
                    f"{field_prefix}{score_name}_"
                    "positive_le_control_max"
                )
            else:
                positive_key = (
                    f"{field_prefix}{score_name}_novel_positive_mean"
                )
                control_key = (
                    f"{field_prefix}{score_name}_novel_control_mean"
                )
                delta_key = (
                    f"{field_prefix}{score_name}_novel_delta"
                )
                auc_key = f"{field_prefix}{score_name}_novel_auc"
                hit_key = (
                    f"{field_prefix}{score_name}_novel_hit_at_1"
                )
                le_mean_key = (
                    f"{field_prefix}{score_name}_"
                    "positive_le_novel_control_mean"
                )
                le_max_key = (
                    f"{field_prefix}{score_name}_"
                    "positive_le_novel_control_max"
                )
            positive_values = bootstrap_columns[positive_key]
            row_count = int(
                np.isfinite(bootstrap_columns[delta_key]).sum()
            )
            metric_row = {
                "split": split,
                **identity,
                "score_name": score_name,
                "control_scope": scope,
                "rows": row_count,
                "excluded_rows": split_row_count - row_count,
                "positive_mean": intervals[positive_key]["mean"],
                "positive_ci_low": intervals[positive_key]["ci_low"],
                "positive_ci_high": intervals[positive_key]["ci_high"],
                **_finite_quantiles(positive_values),
                "control_mean": intervals[control_key]["mean"],
                "control_ci_low": intervals[control_key]["ci_low"],
                "control_ci_high": intervals[control_key]["ci_high"],
                "delta_mean": intervals[delta_key]["mean"],
                "delta_ci_low": intervals[delta_key]["ci_low"],
                "delta_ci_high": intervals[delta_key]["ci_high"],
                "auc_mean": intervals[auc_key]["mean"],
                "auc_ci_low": intervals[auc_key]["ci_low"],
                "auc_ci_high": intervals[auc_key]["ci_high"],
                "hit_at_1_mean": intervals[hit_key]["mean"],
                "hit_at_1_ci_low": intervals[hit_key]["ci_low"],
                "hit_at_1_ci_high": intervals[hit_key]["ci_high"],
                "positive_le_control_mean_rate": intervals[
                    le_mean_key
                ]["mean"],
                "positive_le_control_mean_ci_low": intervals[
                    le_mean_key
                ]["ci_low"],
                "positive_le_control_mean_ci_high": intervals[
                    le_mean_key
                ]["ci_high"],
                "positive_le_control_max_rate": intervals[le_max_key][
                    "mean"
                ],
                "positive_le_control_max_ci_low": intervals[le_max_key][
                    "ci_low"
                ],
                "positive_le_control_max_ci_high": intervals[le_max_key][
                    "ci_high"
                ],
            }
            output_rows.append(metric_row)
            summary[score_name][scope] = metric_row
    return output_rows, summary


def _bin_history_length(value: int) -> str:
    if value <= 10:
        return "01-10"
    if value <= 25:
        return "11-25"
    if value <= 49:
        return "26-49"
    return "50"


def _bin_unique_subcategories(value: int) -> str:
    if value <= 5:
        return "01-05"
    if value <= 10:
        return "06-10"
    if value <= 20:
        return "11-20"
    return "21+"


def _bin_unique_parents(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def _bin_candidate_count(value: int) -> str:
    if value <= 20:
        return "01-20"
    if value <= 40:
        return "21-40"
    if value <= 60:
        return "41-60"
    return "61+"


def _bin_target_count(value: int) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def _slice_memberships(
    row: Mapping[str, Any],
) -> Iterator[tuple[str, str, str]]:
    yield (
        "history_length",
        _bin_history_length(int(row["history_length"])),
        "single",
    )
    yield (
        "history_unique_subcategories",
        _bin_unique_subcategories(
            int(row["history_unique_subcategories"])
        ),
        "single",
    )
    yield (
        "history_unique_parents",
        _bin_unique_parents(int(row["history_unique_parents"])),
        "single",
    )
    yield (
        "control_count",
        _bin_candidate_count(int(row["control_count"])),
        "single",
    )
    yield (
        "novel_control_count",
        (
            "0"
            if int(row["novel_control_count"]) == 0
            else _bin_candidate_count(int(row["novel_control_count"]))
        ),
        "single",
    )
    yield (
        "target_count",
        _bin_target_count(int(row["target_count"])),
        "single",
    )
    for parent in sorted(set(row["target_categories"])):
        yield "target_parent", str(parent), "multi_membership"
    for subcategory in sorted(set(row["target_subcategories"])):
        yield "target_subcategory", str(subcategory), "multi_membership"

    primary_delta = row["nearest_50_novel_delta"]
    if primary_delta is not None and math.isfinite(float(primary_delta)):
        same_parent_fraction = float(row["same_parent_fraction"])
        if same_parent_fraction == 1.0:
            taxonomy = "same_parent"
        elif same_parent_fraction == 0.0:
            taxonomy = "cross_parent"
        else:
            taxonomy = "mixed_parent"
        semantic = (
            "higher_than_control"
            if float(primary_delta) > 0.0
            else "not_higher_than_control"
        )
        yield (
            "taxonomy_semantic_quadrant",
            f"{taxonomy}__{semantic}",
            "single",
        )


def _mean_or_none(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float | None:
    values = np.asarray(
        [
            float(row[field])
            if row[field] is not None
            else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _slice_metric_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split, split_rows in rows_by_split.items():
        groups: dict[
            tuple[str, str, str],
            list[Mapping[str, Any]],
        ] = {}
        for row in split_rows:
            for membership in _slice_memberships(row):
                groups.setdefault(membership, []).append(row)
        for (slice_type, slice_value, policy), rows in sorted(
            groups.items()
        ):
            output.append(
                {
                    "split": split,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "membership_policy": policy,
                    "rows": len(rows),
                    "primary_rows": sum(
                        row["nearest_50_novel_delta"] is not None
                        for row in rows
                    ),
                    "same_parent_fraction": _mean_or_none(
                        rows,
                        "same_parent_fraction",
                    ),
                    "positive_similarity": _mean_or_none(
                        rows,
                        "nearest_50_positive_mean",
                    ),
                    "all_control_delta": _mean_or_none(
                        rows,
                        "nearest_50_delta",
                    ),
                    "novel_control_delta": _mean_or_none(
                        rows,
                        "nearest_50_novel_delta",
                    ),
                    "all_control_auc": _mean_or_none(
                        rows,
                        "nearest_50_auc",
                    ),
                    "novel_control_auc": _mean_or_none(
                        rows,
                        "nearest_50_novel_auc",
                    ),
                    "all_control_hit_at_1": _mean_or_none(
                        rows,
                        "nearest_50_hit_at_1",
                    ),
                    "novel_control_hit_at_1": _mean_or_none(
                        rows,
                        "nearest_50_novel_hit_at_1",
                    ),
                }
            )
    return output


def _transition_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    weights: dict[tuple[str, str, str], float] = {}
    for split, split_rows in rows_by_split.items():
        for row in split_rows:
            targets = [str(value) for value in row["target_categories"]]
            weight = 1.0 / len(targets)
            for target in targets:
                key = (split, str(row["last_history_parent"]), target)
                counts[key] += 1
                weights[key] = weights.get(key, 0.0) + weight
    return [
        {
            "split": split,
            "last_history_parent": history_parent,
            "target_parent": target_parent,
            "target_occurrences": counts[
                (split, history_parent, target_parent)
            ],
            "row_weight": weights[
                (split, history_parent, target_parent)
            ],
            "same_parent": float(history_parent == target_parent),
        }
        for split, history_parent, target_parent in sorted(counts)
    ]


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_sft_training_statistics(
    analysis_rows_path: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 2_000,
    permutation_replicates: int = 2_000,
    seed: int = 17,
    permutation_seed: int | None = None,
    confidence: float = 0.95,
    source_path_label: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Aggregate row-equal semantic, taxonomy, slice, and leakage metrics."""

    analysis_rows_path = Path(analysis_rows_path)
    output_dir = Path(output_dir)
    if bootstrap_replicates <= 0 or permutation_replicates <= 0:
        raise ValueError("replicate counts must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    permutation_seed = seed if permutation_seed is None else permutation_seed
    output_names = (
        "summary.json",
        "similarity_metrics.csv",
        "category_name_similarity_metrics.csv",
        "taxonomy_metrics.csv",
        "slice_metrics.csv",
        "parent_transition_matrix.csv",
        "prompt_overlap.json",
    )
    existing = [
        output_dir / name
        for name in output_names
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Statistics outputs already exist: {existing[0]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = _load_compact_stat_rows(analysis_rows_path)

    similarity_rows: list[dict[str, Any]] = []
    category_similarity_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    similarity_summary: dict[str, Any] = {}
    category_similarity_summary: dict[str, Any] = {}
    taxonomy_summary: dict[str, Any] = {}
    permutation_summary: dict[str, Any] = {}
    for split, split_rows in rows_by_split.items():
        bootstrap_columns: dict[str, np.ndarray] = {}
        for score_name in SIMILARITY_SCORE_NAMES:
            for field_name in (
                *_SIMILARITY_REQUIRED_FIELDS,
                *_SIMILARITY_NOVEL_FIELDS,
            ):
                name = f"{score_name}_{field_name}"
                bootstrap_columns[name] = np.asarray(
                    [
                        (
                            float(row[name])
                            if row[name] is not None
                            else math.nan
                        )
                        for row in split_rows
                    ],
                    dtype=np.float64,
                )
            positive = bootstrap_columns[
                f"{score_name}_positive_mean"
            ]
            novel_mask = np.isfinite(
                bootstrap_columns[f"{score_name}_novel_delta"]
            )
            bootstrap_columns[
                f"{score_name}_novel_positive_mean"
            ] = np.where(novel_mask, positive, np.nan)
        for field_prefix in CATEGORY_NAME_LEVEL_PREFIXES.values():
            for score_name in SIMILARITY_SCORE_NAMES:
                for field_name in (
                    *_SIMILARITY_REQUIRED_FIELDS,
                    *_SIMILARITY_NOVEL_FIELDS,
                ):
                    name = (
                        f"{field_prefix}{score_name}_{field_name}"
                    )
                    bootstrap_columns[name] = np.asarray(
                        [
                            (
                                float(row[name])
                                if row[name] is not None
                                else math.nan
                            )
                            for row in split_rows
                        ],
                        dtype=np.float64,
                    )
                positive = bootstrap_columns[
                    f"{field_prefix}{score_name}_positive_mean"
                ]
                novel_mask = np.isfinite(
                    bootstrap_columns[
                        f"{field_prefix}{score_name}_novel_delta"
                    ]
                )
                bootstrap_columns[
                    f"{field_prefix}{score_name}_novel_positive_mean"
                ] = np.where(novel_mask, positive, np.nan)
        taxonomy_fields = (
            "same_parent_fraction",
            "any_target_same_parent",
            "all_targets_same_parent",
            "same_parent_recent5_fraction",
            "any_target_same_parent_recent5",
            "all_targets_same_parent_recent5",
            "same_parent_last_fraction",
            "any_target_same_parent_last",
            "all_targets_same_parent_last",
        )
        for name in taxonomy_fields:
            bootstrap_columns[name] = np.asarray(
                [float(row[name]) for row in split_rows],
                dtype=np.float64,
            )
        intervals = _bootstrap_column_intervals(
            bootstrap_columns,
            replicates=bootstrap_replicates,
            seed=_stable_seed(seed, f"bootstrap:{split}"),
            confidence=confidence,
        )

        split_rows_output, split_similarity = _similarity_metric_rows(
            split=split,
            split_row_count=len(split_rows),
            bootstrap_columns=bootstrap_columns,
            intervals=intervals,
            field_prefix="",
        )
        similarity_rows.extend(split_rows_output)
        similarity_summary[split] = split_similarity

        split_category_similarity: dict[str, Any] = {}
        for name_level, field_prefix in (
            CATEGORY_NAME_LEVEL_PREFIXES.items()
        ):
            level_rows, level_summary = _similarity_metric_rows(
                split=split,
                split_row_count=len(split_rows),
                bootstrap_columns=bootstrap_columns,
                intervals=intervals,
                field_prefix=field_prefix,
                identity={"name_level": name_level},
            )
            category_similarity_rows.extend(level_rows)
            split_category_similarity[name_level] = level_summary
        category_similarity_summary[split] = split_category_similarity

        history_scopes = {
            "full": (
                "history_parents_full",
                {
                    "same_parent_fraction": "same_parent_fraction",
                    "any_target_same_parent": "any_target_same_parent",
                    "all_targets_same_parent": "all_targets_same_parent",
                },
            ),
            "recent5": (
                "history_parents_recent5",
                {
                    "same_parent_fraction": (
                        "same_parent_recent5_fraction"
                    ),
                    "any_target_same_parent": (
                        "any_target_same_parent_recent5"
                    ),
                    "all_targets_same_parent": (
                        "all_targets_same_parent_recent5"
                    ),
                },
            ),
            "last": (
                "history_parents_last",
                {
                    "same_parent_fraction": (
                        "same_parent_last_fraction"
                    ),
                    "any_target_same_parent": (
                        "any_target_same_parent_last"
                    ),
                    "all_targets_same_parent": (
                        "all_targets_same_parent_last"
                    ),
                },
            ),
        }
        split_taxonomy: dict[str, Any] = {}
        split_permutations: dict[str, Any] = {}
        for scope, (history_field, observed_fields) in (
            history_scopes.items()
        ):
            baseline = compute_parent_permutation_baseline(
                history_categories=[
                    row[history_field] for row in split_rows
                ],
                target_categories=[
                    row["target_categories"] for row in split_rows
                ],
                replicates=permutation_replicates,
                seed=_stable_seed(
                    permutation_seed,
                    f"permutation:{split}:{scope}",
                ),
                confidence=confidence,
            )
            split_permutations[scope] = baseline
            split_taxonomy[scope] = {}
            for metric, observed_field in observed_fields.items():
                observed_interval = intervals[observed_field]
                baseline_values = baseline[metric]
                row = {
                    "split": split,
                    "history_scope": scope,
                    "metric": metric,
                    "rows": len(split_rows),
                    "observed": observed_interval["mean"],
                    "observed_ci_low": observed_interval["ci_low"],
                    "observed_ci_high": observed_interval["ci_high"],
                    "permutation_expected": baseline_values["expected"],
                    "permutation_null_ci_low": baseline_values[
                        "null_ci_low"
                    ],
                    "permutation_null_ci_high": baseline_values[
                        "null_ci_high"
                    ],
                    "observed_minus_expected": baseline_values[
                        "observed_minus_expected"
                    ],
                    "lift": baseline_values["lift"],
                }
                taxonomy_rows.append(row)
                split_taxonomy[scope][metric] = row
        taxonomy_summary[split] = split_taxonomy
        permutation_summary[split] = split_permutations

    prompt_rows = [
        row
        for split_rows in rows_by_split.values()
        for row in split_rows
    ]
    prompt_overlap = compute_prompt_overlap(prompt_rows)
    slices = _slice_metric_rows(rows_by_split)
    transitions = _transition_rows(rows_by_split)
    row_counts = {
        split: len(rows) for split, rows in rows_by_split.items()
    }
    primary_rows = {
        split: sum(
            row["nearest_50_novel_delta"] is not None for row in rows
        )
        for split, rows in rows_by_split.items()
    }
    summary = {
        "artifact_type": "sft_training_similarity_statistics",
        "artifact_version": 1,
        "source_path": str(source_path_label or analysis_rows_path),
        "source_sha256": _sha256_file(analysis_rows_path),
        "rows": row_counts,
        "primary_rows": primary_rows,
        "rows_without_novel_controls": {
            split: row_counts[split] - primary_rows[split]
            for split in row_counts
        },
        "counts": {
            split: {
                "positive_titles": sum(
                    int(row["positive_title_count"]) for row in rows
                ),
                "controls": sum(
                    int(row["control_count"]) for row in rows
                ),
                "novel_controls": sum(
                    int(row["novel_control_count"]) for row in rows
                ),
                "duplicate_control_tokens": sum(
                    int(row["duplicate_control_token_count"])
                    for row in rows
                ),
            }
            for split, rows in rows_by_split.items()
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "confidence": confidence,
            "unit": "sft_row",
        },
        "permutation": {
            "replicates": permutation_replicates,
            "seed": permutation_seed,
            "confidence": confidence,
            "unit": "split_target_set",
        },
        "similarity": similarity_summary,
        "category_name_similarity": category_similarity_summary,
        "taxonomy": taxonomy_summary,
        "prompt_overlap": prompt_overlap,
    }

    similarity_fields = list(similarity_rows[0])
    category_similarity_fields = list(category_similarity_rows[0])
    taxonomy_fields = list(taxonomy_rows[0])
    slice_fields = list(slices[0])
    transition_fields = list(transitions[0])
    _write_csv_atomic(
        output_dir / "similarity_metrics.csv",
        similarity_rows,
        similarity_fields,
    )
    _write_csv_atomic(
        output_dir / "category_name_similarity_metrics.csv",
        category_similarity_rows,
        category_similarity_fields,
    )
    _write_csv_atomic(
        output_dir / "taxonomy_metrics.csv",
        taxonomy_rows,
        taxonomy_fields,
    )
    _write_csv_atomic(
        output_dir / "slice_metrics.csv",
        slices,
        slice_fields,
    )
    _write_csv_atomic(
        output_dir / "parent_transition_matrix.csv",
        transitions,
        transition_fields,
    )
    _write_json_atomic(
        output_dir / "prompt_overlap.json",
        prompt_overlap,
    )
    _write_json_atomic(output_dir / "summary.json", summary)
    return {
        "rows": row_counts,
        "primary_rows": primary_rows,
        "paths": {
            name: str(output_dir / name) for name in output_names
        },
    }


def _case_quadrant(row: Mapping[str, Any]) -> str | None:
    if int(row["target_count"]) != 1:
        return None
    delta = row["nearest_50_novel_delta"]
    if delta is None or not math.isfinite(float(delta)):
        return None
    same_parent = float(row["same_parent_fraction"])
    if same_parent not in {0.0, 1.0}:
        return None
    taxonomy = "same_parent" if same_parent == 1.0 else "cross_parent"
    semantic = (
        "higher_than_control"
        if float(delta) > 0.0
        else "not_higher_than_control"
    )
    return f"{taxonomy}__{semantic}"


def _nearest_scores_for_news(
    news_items: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    title_vectors: Mapping[str, np.ndarray],
    *,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_vectors = _vectors_for_news(
        news_items,
        title_vectors,
        dimensions=dimensions,
    )
    history_vectors = _vectors_for_news(
        history,
        title_vectors,
        dimensions=dimensions,
    )
    cosine = query_vectors @ history_vectors.T
    return cosine.max(axis=1), cosine.argmax(axis=1)


def _case_news_details(
    news_items: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    title_vectors: Mapping[str, np.ndarray],
    *,
    dimensions: int,
) -> list[dict[str, Any]]:
    scores, nearest_indices = _nearest_scores_for_news(
        news_items,
        history,
        title_vectors,
        dimensions=dimensions,
    )
    output = []
    for index, item in enumerate(news_items):
        nearest = history[int(nearest_indices[index])]
        output.append(
            {
                **dict(item),
                "nearest_50": float(scores[index]),
                "nearest_history": {
                    **dict(nearest),
                    "cosine": float(scores[index]),
                },
            }
        )
    return output


def _top_case_controls(
    news_items: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    title_vectors: Mapping[str, np.ndarray],
    *,
    dimensions: int,
    limit: int,
) -> list[dict[str, Any]]:
    details = _case_news_details(
        news_items,
        history,
        title_vectors,
        dimensions=dimensions,
    )
    ranked = sorted(
        enumerate(details),
        key=lambda item: (-item[1]["nearest_50"], item[0]),
    )
    return [
        {**detail, "candidate_index": original_index}
        for original_index, detail in ranked[:limit]
    ]


def write_sft_similarity_cases(
    analysis_rows_path: Path,
    title_embeddings_path: Path,
    output_path: Path,
    *,
    cases_per_split_quadrant: int = 1,
    top_controls: int = 3,
    analysis_path_label: Path | str | None = None,
    title_embeddings_path_label: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select median-representative single-target cases per quadrant."""

    analysis_rows_path = Path(analysis_rows_path)
    title_embeddings_path = Path(title_embeddings_path)
    output_path = Path(output_path)
    if cases_per_split_quadrant <= 0 or top_controls <= 0:
        raise ValueError("case and control limits must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Case output already exists: {output_path}")

    candidates: dict[
        tuple[str, str],
        list[tuple[str, float]],
    ] = {}
    for row in _iter_parquet_rows(
        _require_file(analysis_rows_path),
        columns=[
            "row_id",
            "split",
            "target_count",
            "same_parent_fraction",
            "nearest_50_novel_delta",
        ],
    ):
        quadrant = _case_quadrant(row)
        if quadrant is None:
            continue
        key = (str(row["split"]), quadrant)
        candidates.setdefault(key, []).append(
            (str(row["row_id"]), float(row["nearest_50_novel_delta"]))
        )

    selected: dict[str, tuple[str, str]] = {}
    for (split, quadrant), values in sorted(candidates.items()):
        median = float(np.median([value for _, value in values]))
        ranked = sorted(
            values,
            key=lambda item: (abs(item[1] - median), item[0]),
        )
        for row_id, _ in ranked[:cases_per_split_quadrant]:
            selected[row_id] = (split, quadrant)

    title_vectors, dimensions = _load_title_embeddings(
        title_embeddings_path
    )
    selected_rows: dict[str, dict[str, Any]] = {}
    for row in _iter_parquet_rows(_require_file(analysis_rows_path)):
        row_id = str(row["row_id"])
        if row_id in selected:
            selected_rows[row_id] = row
    missing = selected.keys() - selected_rows.keys()
    if missing:
        raise ValueError(f"Selected case rows are missing: {sorted(missing)}")

    cases: list[dict[str, Any]] = []
    for row_id, (split, quadrant) in sorted(
        selected.items(),
        key=lambda item: (
            item[1][0],
            item[1][1],
            item[0],
        ),
    ):
        row = selected_rows[row_id]
        positive_details = _case_news_details(
            row["positive_news"],
            row["history"],
            title_vectors,
            dimensions=dimensions,
        )
        all_control_scores, _ = _nearest_scores_for_news(
            row["control_news"],
            row["history"],
            title_vectors,
            dimensions=dimensions,
        )
        novel_control_scores, _ = _nearest_scores_for_news(
            row["novel_control_news"],
            row["history"],
            title_vectors,
            dimensions=dimensions,
        )
        for detail in positive_details:
            score = float(detail["nearest_50"])
            detail["rank_among_all_controls"] = 1 + int(
                np.count_nonzero(all_control_scores > score)
            )
            detail["rank_among_novel_controls"] = 1 + int(
                np.count_nonzero(novel_control_scores > score)
            )
        cases.append(
            {
                "case_id": f"{split}:{quadrant}:{row_id}",
                "row_id": row_id,
                "split": split,
                "quadrant": quadrant,
                "source_sample_ids": row["source_sample_ids"],
                "impression_id": row["impression_id"],
                "user_id": row["user_id"],
                "behavior_prompt": row["behavior_prompt"],
                "interest_prompt": row["interest_prompt"],
                "history": row["history"],
                "target_categories": row["target_categories"],
                "target_subcategories": row["target_subcategories"],
                "positive_news": positive_details,
                "top_all_controls": _top_case_controls(
                    row["control_news"],
                    row["history"],
                    title_vectors,
                    dimensions=dimensions,
                    limit=top_controls,
                ),
                "top_novel_controls": _top_case_controls(
                    row["novel_control_news"],
                    row["history"],
                    title_vectors,
                    dimensions=dimensions,
                    limit=top_controls,
                ),
                "metrics": {
                    "same_parent_fraction": float(
                        row["same_parent_fraction"]
                    ),
                    "nearest_50_positive_mean": float(
                        row["nearest_50_positive_mean"]
                    ),
                    "nearest_50_novel_control_mean": float(
                        row["nearest_50_novel_control_mean"]
                    ),
                    "nearest_50_novel_delta": float(
                        row["nearest_50_novel_delta"]
                    ),
                    "nearest_50_novel_auc": float(
                        row["nearest_50_novel_auc"]
                    ),
                    "nearest_50_novel_hit_at_1": float(
                        row["nearest_50_novel_hit_at_1"]
                    ),
                },
            }
        )
    payload = {
        "artifact_type": "sft_training_similarity_cases",
        "artifact_version": 1,
        "selection": {
            "eligible_rows": (
                "single target, non-null novel controls, exact same-parent "
                "or cross-parent status"
            ),
            "priority": (
                "minimum absolute distance from split/quadrant median "
                "nearest_50 novel-control delta, then row_id"
            ),
            "cases_per_split_quadrant": cases_per_split_quadrant,
            "top_controls": top_controls,
        },
        "source_path": str(analysis_path_label or analysis_rows_path),
        "source_sha256": _sha256_file(analysis_rows_path),
        "title_embeddings_path": str(
            title_embeddings_path_label or title_embeddings_path
        ),
        "title_embeddings_sha256": _sha256_file(
            title_embeddings_path
        ),
        "candidate_counts": {
            f"{split}:{quadrant}": len(values)
            for (split, quadrant), values in sorted(candidates.items())
        },
        "cases": cases,
    }
    _write_json_atomic(output_path, payload)
    return {
        "cases": len(cases),
        "candidate_counts": payload["candidate_counts"],
        "path": str(output_path),
    }


def _iter_parquet_rows(
    path: Path,
    *,
    columns: list[str] | None = None,
    batch_size: int = 1_024,
) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        yield from batch.to_pylist()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis input does not exist: {path}")
    return path


def _index_unique(
    rows: Iterable[dict[str, Any]],
    *,
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"Duplicate {label} key: {value}")
        indexed[value] = row
    return indexed


def _load_sft_pairs(
    sft_dir: Path,
    *,
    train_name: str,
    dev_name: str,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    behavior_train = list(
        _iter_parquet_rows(
            _require_file(
                sft_dir / "train" / "behavior_only" / train_name
            )
        )
    )
    interest_train = _index_unique(
        _iter_parquet_rows(
            _require_file(
                sft_dir / "train" / "interest_only" / train_name
            )
        ),
        key="source_sample_id",
        label="interest-only Train source sample",
    )
    train_pairs = []
    for behavior_row in behavior_train:
        source_sample_id = str(behavior_row["source_sample_id"])
        try:
            interest_row = interest_train.pop(source_sample_id)
        except KeyError as error:
            raise ValueError(
                "Behavior-only Train row has no matching interest-only row: "
                f"{source_sample_id}"
            ) from error
        for field in (
            "impression_id",
            "user_id",
            "target_subcategory",
        ):
            if behavior_row[field] != interest_row[field]:
                raise ValueError(
                    f"Train task variants disagree on {field} for "
                    f"{source_sample_id}"
                )
        train_pairs.append((behavior_row, interest_row))
    if interest_train:
        raise ValueError(
            "Interest-only Train contains unmatched rows: "
            f"{sorted(interest_train)[:3]}"
        )

    behavior_dev = list(
        _iter_parquet_rows(
            _require_file(
                sft_dir / "eval" / "behavior_only" / dev_name
            )
        )
    )
    interest_dev = _index_unique(
        _iter_parquet_rows(
            _require_file(
                sft_dir / "eval" / "interest_only" / dev_name
            )
        ),
        key="impression_id",
        label="interest-only Dev impression",
    )
    dev_pairs = []
    for behavior_row in behavior_dev:
        impression_id = str(behavior_row["impression_id"])
        try:
            interest_row = interest_dev.pop(impression_id)
        except KeyError as error:
            raise ValueError(
                "Behavior-only Dev row has no matching interest-only row: "
                f"{impression_id}"
            ) from error
        for field in (
            "user_id",
            "acceptable_subcategories",
            "history_subcategories",
        ):
            if behavior_row[field] != interest_row[field]:
                raise ValueError(
                    f"Dev task variants disagree on {field} for "
                    f"impression {impression_id}"
                )
        dev_pairs.append((behavior_row, interest_row))
    if interest_dev:
        raise ValueError(
            "Interest-only Dev contains unmatched impressions: "
            f"{sorted(interest_dev)[:3]}"
        )
    return train_pairs, dev_pairs


def _load_selected_source_rows(
    source_dir: Path,
    train_sample_ids: set[str],
    dev_impression_ids: set[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    train_rows: dict[str, dict[str, Any]] = {}
    for row in _iter_parquet_rows(
        _require_file(source_dir / "train.parquet")
    ):
        sample_id = str(row["sample_id"])
        if sample_id not in train_sample_ids:
            continue
        if sample_id in train_rows:
            raise ValueError(f"Duplicate Train source sample: {sample_id}")
        train_rows[sample_id] = row
    missing_train = train_sample_ids - train_rows.keys()
    if missing_train:
        raise ValueError(
            "SFT Train rows are missing from the source dataset: "
            f"{sorted(missing_train)[:3]}"
        )

    dev_rows: dict[str, list[dict[str, Any]]] = {
        impression_id: [] for impression_id in dev_impression_ids
    }
    for row in _iter_parquet_rows(
        _require_file(source_dir / "dev.parquet")
    ):
        impression_id = str(row["impression_id"])
        if impression_id in dev_rows:
            dev_rows[impression_id].append(row)
    missing_dev = [
        impression_id
        for impression_id, rows in dev_rows.items()
        if not rows
    ]
    if missing_dev:
        raise ValueError(
            "SFT Dev rows are missing from the source dataset: "
            f"{sorted(missing_dev)[:3]}"
        )
    return train_rows, dev_rows


def _news_item(record: NewsRecord) -> dict[str, str]:
    return {
        "news_id": record.news_id,
        "category": record.category,
        "subcategory": record.subcategory,
        "title": record.title,
    }


def _source_history(
    row: Mapping[str, Any],
    *,
    history_limit: int,
) -> list[dict[str, str]]:
    history = list(row["history"])[-history_limit:]
    if not history:
        raise ValueError(f"Source row has empty H{history_limit} history")
    for item in history:
        for field in ("news_id", "category", "subcategory", "title"):
            if not str(item[field]).strip():
                raise ValueError(
                    f"Source history item has empty {field}: {item}"
                )
    return history


def _validate_event_alignment(
    event: BehaviorEvent,
    source_row: Mapping[str, Any],
    history: list[dict[str, str]],
) -> None:
    if event.user_id != source_row["user_id"]:
        raise ValueError(
            f"User mismatch for impression {event.impression_id}: "
            f"{event.user_id} != {source_row['user_id']}"
        )
    source_history_ids = [
        str(item["news_id"]) for item in source_row["history"]
    ]
    if list(event.history_news_ids) != source_history_ids:
        raise ValueError(
            f"Raw/source history mismatch for impression "
            f"{event.impression_id}"
        )
    if [item["news_id"] for item in history] != list(
        event.history_news_ids[-len(history) :]
    ):
        raise ValueError(
            f"H{len(history)} history mismatch for impression "
            f"{event.impression_id}"
        )


def _load_raw_context(
    raw_dir: Path,
    *,
    split: str,
    needed_impressions: set[str],
    source_by_impression: Mapping[str, Mapping[str, Any]],
    history_by_impression: Mapping[str, list[dict[str, str]]],
) -> dict[str, dict[str, Any]]:
    catalog = load_news_catalog(raw_dir)
    contexts: dict[str, dict[str, Any]] = {}
    for event in iter_behavior_events(
        raw_dir,
        (split,),
        catalog,
    ):
        if event.impression_id not in needed_impressions:
            continue
        if event.impression_id in contexts:
            raise ValueError(
                f"Duplicate raw {split} impression: {event.impression_id}"
            )
        source_row = source_by_impression[event.impression_id]
        history = history_by_impression[event.impression_id]
        _validate_event_alignment(event, source_row, history)
        clicked_ids = {
            candidate.news_id
            for candidate in event.impression_candidates
            if candidate.clicked
        }
        control_ids = [
            candidate.news_id
            for candidate in event.impression_candidates
            if not candidate.clicked and candidate.news_id not in clicked_ids
        ]
        history_subcategories = {
            str(item["subcategory"]) for item in history
        }
        controls = [
            _news_item(catalog.news[news_id]) for news_id in control_ids
        ]
        novel_controls = [
            item
            for item in controls
            if item["subcategory"] not in history_subcategories
        ]
        contexts[event.impression_id] = {
            "catalog": catalog,
            "clicked_ids": clicked_ids,
            "controls": controls,
            "novel_controls": novel_controls,
            "duplicate_control_token_count": (
                len(control_ids) - len(set(control_ids))
            ),
        }
    missing = needed_impressions - contexts.keys()
    if missing:
        raise ValueError(
            f"SFT {split} impressions are missing from raw behaviors: "
            f"{sorted(missing)[:3]}"
        )
    return contexts


def _positive_items(
    source_rows: list[Mapping[str, Any]],
    catalog: NewsCatalog,
    clicked_ids: set[str],
) -> list[dict[str, str]]:
    positive_items: list[dict[str, str]] = []
    for source_row in source_rows:
        target = source_row["target_interest"]
        for target_news in source_row["target_news"]:
            news_id = str(target_news["news_id"])
            if news_id not in clicked_ids:
                raise ValueError(
                    f"Positive news {news_id} is not a clicked candidate"
                )
            record = catalog.news[news_id]
            if (
                record.category != target["category"]
                or record.subcategory != target["subcategory"]
                or record.title != target_news["title"]
            ):
                raise ValueError(
                    f"Source/catalog target mismatch for news {news_id}"
                )
            positive_items.append(_news_item(record))
    if not positive_items:
        raise ValueError("Analysis row has no positive news titles")
    return positive_items


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_output_row(
    *,
    split: str,
    behavior_row: Mapping[str, Any],
    interest_row: Mapping[str, Any],
    source_rows: list[Mapping[str, Any]],
    history: list[dict[str, str]],
    raw_context: Mapping[str, Any],
) -> dict[str, Any]:
    impression_id = str(behavior_row["impression_id"])
    target_categories = [
        str(row["target_interest"]["category"]) for row in source_rows
    ]
    target_subcategories = [
        str(row["target_interest"]["subcategory"]) for row in source_rows
    ]
    history_subcategories = {
        str(item["subcategory"]) for item in history
    }
    overlapping_targets = (
        history_subcategories.intersection(target_subcategories)
    )
    if overlapping_targets:
        raise ValueError(
            f"Target subcategory occurs in H{len(history)} for impression "
            f"{impression_id}: {sorted(overlapping_targets)}"
        )

    positive_news = _positive_items(
        source_rows,
        raw_context["catalog"],
        raw_context["clicked_ids"],
    )
    positive_ids = {item["news_id"] for item in positive_news}
    control_ids = {
        item["news_id"] for item in raw_context["controls"]
    }
    overlap = positive_ids.intersection(control_ids)
    if overlap:
        raise ValueError(
            f"Positive/control news overlap for impression "
            f"{impression_id}: {sorted(overlap)}"
        )

    behavior_prompt = str(behavior_row["prompt"])
    interest_prompt = str(interest_row["prompt"])
    return {
        "row_id": (
            str(source_rows[0]["sample_id"])
            if split == "train"
            else f"dev:{impression_id}"
        ),
        "split": split,
        "source_sample_ids": [
            str(row["sample_id"]) for row in source_rows
        ],
        "impression_id": impression_id,
        "user_id": str(behavior_row["user_id"]),
        "timestamp": behavior_row["timestamp"],
        "history": history,
        "target_categories": target_categories,
        "target_subcategories": target_subcategories,
        "positive_news": positive_news,
        "control_news": raw_context["controls"],
        "novel_control_news": raw_context["novel_controls"],
        "behavior_prompt": behavior_prompt,
        "interest_prompt": interest_prompt,
        "behavior_prompt_sha256": _sha256_text(behavior_prompt),
        "interest_prompt_sha256": _sha256_text(interest_prompt),
        "history_length": len(history),
        "target_count": len(target_subcategories),
        "positive_title_count": len(positive_news),
        "control_count": len(raw_context["controls"]),
        "novel_control_count": len(raw_context["novel_controls"]),
        "duplicate_control_token_count": raw_context[
            "duplicate_control_token_count"
        ],
    }


def _write_rows_atomically(
    output_path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    schema: pa.Schema,
    batch_size: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.staging-{uuid4().hex}"
    )
    writer: pq.ParquetWriter | None = None
    count = 0
    batch: list[dict[str, Any]] = []
    try:
        writer = pq.ParquetWriter(
            temporary_path,
            schema,
            compression="zstd",
        )
        for row in rows:
            batch.append(row)
            if len(batch) < batch_size:
                continue
            writer.write_table(
                pa.Table.from_pylist(batch, schema=schema)
            )
            count += len(batch)
            batch.clear()
        if batch:
            writer.write_table(
                pa.Table.from_pylist(batch, schema=schema)
            )
            count += len(batch)
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
    return count


def write_sft_analysis_rows(
    sft_dir: Path,
    source_dir: Path,
    raw_dir: Path,
    output_path: Path,
    *,
    train_name: str = "22k.parquet",
    dev_name: str = "dev_10k_grouped.parquet",
    history_limit: int = 50,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build aligned SFT/source/raw rows for semantic data analysis."""

    if history_limit <= 0:
        raise ValueError("history_limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Analysis row output already exists: {output_path}"
        )

    train_pairs, dev_pairs = _load_sft_pairs(
        sft_dir,
        train_name=train_name,
        dev_name=dev_name,
    )
    train_sample_ids = {
        str(behavior["source_sample_id"])
        for behavior, _ in train_pairs
    }
    dev_impression_ids = {
        str(behavior["impression_id"]) for behavior, _ in dev_pairs
    }
    train_sources, dev_sources = _load_selected_source_rows(
        source_dir,
        train_sample_ids,
        dev_impression_ids,
    )

    train_source_by_impression: dict[str, Mapping[str, Any]] = {}
    train_history_by_impression: dict[str, list[dict[str, str]]] = {}
    train_source_by_sample: dict[str, Mapping[str, Any]] = {}
    for behavior_row, _ in train_pairs:
        sample_id = str(behavior_row["source_sample_id"])
        source_row = train_sources[sample_id]
        impression_id = str(behavior_row["impression_id"])
        if impression_id in train_source_by_impression:
            raise ValueError(
                "SFT Train contains multiple selected rows for one "
                f"impression: {impression_id}"
            )
        history = _source_history(
            source_row,
            history_limit=history_limit,
        )
        if int(behavior_row["history_length"]) != len(history):
            raise ValueError(
                f"SFT/source history length mismatch for {sample_id}"
            )
        train_source_by_sample[sample_id] = source_row
        train_source_by_impression[impression_id] = source_row
        train_history_by_impression[impression_id] = history

    dev_source_by_impression = {
        impression_id: rows[0]
        for impression_id, rows in dev_sources.items()
    }
    dev_history_by_impression = {
        impression_id: _source_history(
            rows[0],
            history_limit=history_limit,
        )
        for impression_id, rows in dev_sources.items()
    }
    for behavior_row, _ in dev_pairs:
        impression_id = str(behavior_row["impression_id"])
        history = dev_history_by_impression[impression_id]
        if int(behavior_row["history_length"]) != len(history):
            raise ValueError(
                "SFT/source history length mismatch for Dev impression "
                f"{impression_id}"
            )
        expected_subcategories = [
            item["subcategory"] for item in history
        ]
        if behavior_row["history_subcategories"] != expected_subcategories:
            raise ValueError(
                "SFT/source H-history mismatch for Dev impression "
                f"{impression_id}"
            )

    train_context = _load_raw_context(
        raw_dir,
        split="train",
        needed_impressions=set(train_source_by_impression),
        source_by_impression=train_source_by_impression,
        history_by_impression=train_history_by_impression,
    )
    dev_context = _load_raw_context(
        raw_dir,
        split="dev",
        needed_impressions=dev_impression_ids,
        source_by_impression=dev_source_by_impression,
        history_by_impression=dev_history_by_impression,
    )

    def output_rows() -> Iterator[dict[str, Any]]:
        for behavior_row, interest_row in train_pairs:
            sample_id = str(behavior_row["source_sample_id"])
            source_row = train_source_by_sample[sample_id]
            impression_id = str(behavior_row["impression_id"])
            target_subcategory = str(
                source_row["target_interest"]["subcategory"]
            )
            if target_subcategory != behavior_row["target_subcategory"]:
                raise ValueError(
                    f"SFT/source target mismatch for {sample_id}"
                )
            yield _build_output_row(
                split="train",
                behavior_row=behavior_row,
                interest_row=interest_row,
                source_rows=[source_row],
                history=train_history_by_impression[impression_id],
                raw_context=train_context[impression_id],
            )

        for behavior_row, interest_row in dev_pairs:
            impression_id = str(behavior_row["impression_id"])
            rows_by_subcategory = {
                str(row["target_interest"]["subcategory"]): row
                for row in dev_sources[impression_id]
            }
            accepted = [
                str(value)
                for value in behavior_row["acceptable_subcategories"]
            ]
            try:
                selected_sources = [
                    rows_by_subcategory[subcategory]
                    for subcategory in accepted
                ]
            except KeyError as error:
                raise ValueError(
                    "SFT Dev target is missing from source impression "
                    f"{impression_id}: {error.args[0]}"
                ) from error
            yield _build_output_row(
                split="dev",
                behavior_row=behavior_row,
                interest_row=interest_row,
                source_rows=selected_sources,
                history=dev_history_by_impression[impression_id],
                raw_context=dev_context[impression_id],
            )

    written = _write_rows_atomically(
        output_path,
        output_rows(),
        schema=ANALYSIS_ROW_SCHEMA,
        batch_size=batch_size,
    )
    expected = len(train_pairs) + len(dev_pairs)
    if written != expected:
        raise RuntimeError(
            f"Wrote {written} analysis rows; expected {expected}"
        )

    return {
        "rows": {
            "train": len(train_pairs),
            "dev": len(dev_pairs),
        },
        "positive_titles": {
            "train": sum(
                len(
                    train_source_by_sample[
                        str(row["source_sample_id"])
                    ]["target_news"]
                )
                for row, _ in train_pairs
            ),
            "dev": sum(
                sum(
                    len(source["target_news"])
                    for source in dev_sources[str(row["impression_id"])]
                    if source["target_interest"]["subcategory"]
                    in row["acceptable_subcategories"]
                )
                for row, _ in dev_pairs
            ),
        },
        "rows_without_novel_controls": {
            "train": sum(
                not train_context[str(row["impression_id"])][
                    "novel_controls"
                ]
                for row, _ in train_pairs
            ),
            "dev": sum(
                not dev_context[str(row["impression_id"])][
                    "novel_controls"
                ]
                for row, _ in dev_pairs
            ),
        },
        "controls": {
            "train": sum(
                len(train_context[str(row["impression_id"])]["controls"])
                for row, _ in train_pairs
            ),
            "dev": sum(
                len(dev_context[str(row["impression_id"])]["controls"])
                for row, _ in dev_pairs
            ),
        },
        "novel_controls": {
            "train": sum(
                len(
                    train_context[str(row["impression_id"])][
                        "novel_controls"
                    ]
                )
                for row, _ in train_pairs
            ),
            "dev": sum(
                len(
                    dev_context[str(row["impression_id"])][
                        "novel_controls"
                    ]
                )
                for row, _ in dev_pairs
            ),
        },
    }


def _load_analysis_config(
    config_path_or_config: Path | str | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(config_path_or_config, Mapping):
        return dict(config_path_or_config)
    path = Path(config_path_or_config)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    return value


def _lexical_portable_path(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(Path.cwd()))
    try:
        return str(absolute.relative_to(root))
    except ValueError:
        return str(absolute)


def _config_mapping(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} config must be an object")
    return dict(value)


def _analysis_input_paths(
    *,
    sft_dir: Path,
    source_dir: Path,
    raw_dir: Path,
    train_name: str,
    dev_name: str,
) -> dict[str, Path]:
    paths = {
        "sft_train_behavior": (
            sft_dir / "train" / "behavior_only" / train_name
        ),
        "sft_train_interest": (
            sft_dir / "train" / "interest_only" / train_name
        ),
        "sft_dev_behavior": (
            sft_dir / "eval" / "behavior_only" / dev_name
        ),
        "sft_dev_interest": (
            sft_dir / "eval" / "interest_only" / dev_name
        ),
        "source_train": source_dir / "train.parquet",
        "source_dev": source_dir / "dev.parquet",
        "raw_train_news": raw_dir / "train" / "news.tsv",
        "raw_train_behaviors": raw_dir / "train" / "behaviors.tsv",
        "raw_dev_news": raw_dir / "dev" / "news.tsv",
        "raw_dev_behaviors": raw_dir / "dev" / "behaviors.tsv",
    }
    optional = {
        "sft_metadata": sft_dir / "metadata.json",
        "sft_statistics": sft_dir / "statistics.json",
        "sft_subset_manifest": sft_dir / "subset_manifest.parquet",
    }
    for label, path in paths.items():
        _require_file(path)
    paths.update(
        {
            label: path
            for label, path in optional.items()
            if path.is_file()
        }
    )
    return paths


def _publish_analysis_generation(
    generation_dir: Path,
    artifact_dir: Path,
    *,
    overwrite: bool,
) -> Path | None:
    exists = os.path.lexists(artifact_dir)
    if exists and not overwrite:
        raise FileExistsError(
            f"Artifact directory already exists: {artifact_dir}"
        )
    old_target: Path | None = None
    if exists:
        if not artifact_dir.is_symlink():
            raise FileExistsError(
                "Atomic overwrite requires the existing artifact path to "
                f"be a managed symlink: {artifact_dir}"
            )
        link_target = Path(os.readlink(artifact_dir))
        old_target = (
            link_target
            if link_target.is_absolute()
            else artifact_dir.parent / link_target
        ).resolve()

    temporary_link = artifact_dir.with_name(
        f".{artifact_dir.name}.link-{uuid4().hex}"
    )
    relative_target = os.path.relpath(
        generation_dir,
        artifact_dir.parent,
    )
    try:
        os.symlink(
            relative_target,
            temporary_link,
            target_is_directory=True,
        )
        os.replace(temporary_link, artifact_dir)
    finally:
        temporary_link.unlink(missing_ok=True)
    return old_target


def run_sft_training_similarity_analysis(
    config_path_or_config: Path | str | Mapping[str, Any],
    *,
    encoder: TextEncoder | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run and atomically publish the complete SFT data analysis."""

    config = _load_analysis_config(config_path_or_config)
    if config.get("rule_version") != RULE_VERSION:
        raise ValueError(
            f"Expected rule_version {RULE_VERSION!r}, got "
            f"{config.get('rule_version')!r}"
        )
    sft_dir = Path(str(config["sft_dir"]))
    source_dir = Path(str(config["source_dir"]))
    raw_dir = Path(str(config["raw_dir"]))
    artifact_dir = Path(str(config["artifact_dir"]))
    train_name = str(config.get("train_name", "22k.parquet"))
    dev_name = str(
        config.get("dev_name", "dev_10k_grouped.parquet")
    )
    history_limit = int(config.get("history_limit", 50))
    embedding_config = _config_mapping(
        config["embedding"],
        name="embedding",
    )
    bootstrap_config = _config_mapping(
        config["bootstrap"],
        name="bootstrap",
    )
    permutation_config = _config_mapping(
        config["permutation"],
        name="permutation",
    )
    case_config = _config_mapping(config["cases"], name="cases")
    if float(permutation_config["confidence"]) != float(
        bootstrap_config["confidence"]
    ):
        raise ValueError(
            "bootstrap and permutation confidence must match"
        )
    if os.path.lexists(artifact_dir) and not overwrite:
        raise FileExistsError(
            f"Artifact directory already exists: {artifact_dir}"
        )
    if (
        os.path.lexists(artifact_dir)
        and overwrite
        and not artifact_dir.is_symlink()
    ):
        raise FileExistsError(
            "Refusing non-atomic overwrite of an unmanaged artifact "
            f"directory: {artifact_dir}"
        )

    input_paths = _analysis_input_paths(
        sft_dir=sft_dir,
        source_dir=source_dir,
        raw_dir=raw_dir,
        train_name=train_name,
        dev_name=dev_name,
    )
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    generation_root = artifact_dir.parent / (
        f".{artifact_dir.name}.generations"
    )
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_dir = generation_root / uuid4().hex
    generation_dir.mkdir()
    published = False
    old_target: Path | None = None
    canonical_analysis_path = _lexical_portable_path(
        artifact_dir / "analysis_rows.parquet"
    )
    canonical_embeddings_path = _lexical_portable_path(
        artifact_dir / "title_embeddings.parquet"
    )
    canonical_category_embeddings_path = _lexical_portable_path(
        artifact_dir / "category_name_embeddings.parquet"
    )
    try:
        analysis_rows_path = generation_dir / "analysis_rows.parquet"
        semantic_rows_path = generation_dir / ".semantic_rows.parquet"
        taxonomy_rows_path = generation_dir / ".taxonomy_rows.parquet"
        enriched_rows_path = generation_dir / ".enriched_rows.parquet"
        title_embeddings_path = (
            generation_dir / "title_embeddings.parquet"
        )
        category_name_embeddings_path = (
            generation_dir / "category_name_embeddings.parquet"
        )
        join_summary = write_sft_analysis_rows(
            sft_dir,
            source_dir,
            raw_dir,
            analysis_rows_path,
            train_name=train_name,
            dev_name=dev_name,
            history_limit=history_limit,
        )
        shared_encoder = encoder or SentenceTransformerEncoder(
            str(embedding_config["model_name"]),
            revision=str(embedding_config["revision"]),
            device=(
                None
                if embedding_config.get("device") is None
                else str(embedding_config["device"])
            ),
            batch_size=int(embedding_config["batch_size"]),
        )
        embedding_summary = write_title_embeddings(
            analysis_rows_path,
            title_embeddings_path,
            encoder=shared_encoder,
            model_name=str(embedding_config["model_name"]),
            revision=str(embedding_config["revision"]),
            device=(
                None
                if embedding_config.get("device") is None
                else str(embedding_config["device"])
            ),
            batch_size=int(embedding_config["batch_size"]),
            source_path_label=canonical_analysis_path,
        )
        category_embedding_summary = write_category_name_embeddings(
            analysis_rows_path,
            category_name_embeddings_path,
            encoder=shared_encoder,
            model_name=str(embedding_config["model_name"]),
            revision=str(embedding_config["revision"]),
            device=(
                None
                if embedding_config.get("device") is None
                else str(embedding_config["device"])
            ),
            batch_size=int(embedding_config["batch_size"]),
            source_path_label=canonical_analysis_path,
        )
        write_sft_semantic_rows(
            analysis_rows_path,
            title_embeddings_path,
            semantic_rows_path,
            category_name_embeddings_path=category_name_embeddings_path,
        )
        write_sft_taxonomy_rows(
            analysis_rows_path,
            taxonomy_rows_path,
        )
        enriched_summary = write_sft_enriched_rows(
            analysis_rows_path,
            semantic_rows_path,
            taxonomy_rows_path,
            enriched_rows_path,
        )
        os.replace(enriched_rows_path, analysis_rows_path)
        semantic_rows_path.unlink()
        taxonomy_rows_path.unlink()

        statistics_summary = write_sft_training_statistics(
            analysis_rows_path,
            generation_dir,
            bootstrap_replicates=int(
                bootstrap_config["replicates"]
            ),
            permutation_replicates=int(
                permutation_config["replicates"]
            ),
            seed=int(bootstrap_config["seed"]),
            permutation_seed=int(permutation_config["seed"]),
            confidence=float(bootstrap_config["confidence"]),
            source_path_label=canonical_analysis_path,
        )
        cases_summary = write_sft_similarity_cases(
            analysis_rows_path,
            title_embeddings_path,
            generation_dir / "cases.json",
            cases_per_split_quadrant=int(
                case_config["per_split_quadrant"]
            ),
            top_controls=int(case_config["top_controls"]),
            analysis_path_label=canonical_analysis_path,
            title_embeddings_path_label=canonical_embeddings_path,
        )
        cases_summary["path"] = _lexical_portable_path(
            artifact_dir / "cases.json"
        )
        enriched_summary["path"] = canonical_analysis_path

        resolved_config = {
            "rule_version": RULE_VERSION,
            "sft_dir": portable_path(sft_dir),
            "source_dir": portable_path(source_dir),
            "raw_dir": portable_path(raw_dir),
            "artifact_dir": _lexical_portable_path(artifact_dir),
            "train_name": train_name,
            "dev_name": dev_name,
            "history_limit": history_limit,
            "control_policy": {
                "primary": (
                    "same-impression exposed-unclicked news whose "
                    "subcategory is absent from H-history"
                ),
                "secondary": "all same-impression exposed-unclicked news",
                "clicked_news_excluded_from_controls": True,
                "duplicate_candidate_tokens_preserved": True,
            },
            "row_weighting": "one equal weight per SFT row",
            "embedding": {
                "model_name": str(embedding_config["model_name"]),
                "revision": str(embedding_config["revision"]),
                "device": embedding_config.get("device"),
                "batch_size": int(embedding_config["batch_size"]),
                "dimensions": embedding_summary["dimensions"],
                "normalised": True,
            },
            "bootstrap": {
                "replicates": int(bootstrap_config["replicates"]),
                "seed": int(bootstrap_config["seed"]),
                "confidence": float(bootstrap_config["confidence"]),
                "unit": "sft_row",
            },
            "permutation": {
                "replicates": int(permutation_config["replicates"]),
                "seed": int(permutation_config["seed"]),
                "confidence": float(permutation_config["confidence"]),
                "unit": "split_target_set",
            },
            "cases": {
                "per_split_quadrant": int(
                    case_config["per_split_quadrant"]
                ),
                "top_controls": int(case_config["top_controls"]),
            },
        }
        _write_json_atomic(
            generation_dir / "resolved_config.json",
            resolved_config,
        )

        summary_path = generation_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "rule_version": RULE_VERSION,
                "join": join_summary,
                "embedding": {
                    "unique_titles": embedding_summary[
                        "unique_titles"
                    ],
                    "dimensions": embedding_summary["dimensions"],
                    "model": resolved_config["embedding"],
                },
                "category_name_embedding": {
                    "unique_names": category_embedding_summary[
                        "unique_names"
                    ],
                    "dimensions": category_embedding_summary[
                        "dimensions"
                    ],
                    "model": resolved_config["embedding"],
                },
                "cases": cases_summary,
            }
        )
        _write_json_atomic(summary_path, summary)

        output_names = sorted(
            path.name
            for path in generation_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        )
        output_sha256 = {
            name: _sha256_file(generation_dir / name)
            for name in output_names
        }
        manifest = {
            "artifact_type": "sft_training_similarity_analysis",
            "artifact_version": 1,
            "rule_version": RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_dir": _lexical_portable_path(artifact_dir),
            "inputs": {
                label: portable_path(path)
                for label, path in sorted(input_paths.items())
            },
            "input_sha256": {
                label: _sha256_file(path)
                for label, path in sorted(input_paths.items())
            },
            "output_sha256": output_sha256,
            "join": join_summary,
            "enriched_rows": enriched_summary,
            "embedding": {
                "unique_titles": embedding_summary["unique_titles"],
                "dimensions": embedding_summary["dimensions"],
                "model_name": str(embedding_config["model_name"]),
                "revision": str(embedding_config["revision"]),
                "title_corpus_sha256": embedding_summary["metadata"][
                    "title_corpus_sha256"
                ],
            },
            "category_name_embedding": {
                "unique_names": category_embedding_summary[
                    "unique_names"
                ],
                "dimensions": category_embedding_summary["dimensions"],
                "model_name": str(embedding_config["model_name"]),
                "revision": str(embedding_config["revision"]),
                "name_corpus_sha256": category_embedding_summary[
                    "metadata"
                ]["name_corpus_sha256"],
                "path": canonical_category_embeddings_path,
            },
            "statistics": {
                "rows": statistics_summary["rows"],
                "primary_rows": statistics_summary["primary_rows"],
            },
            "cases": cases_summary,
            "policies": {
                "history": f"last {history_limit} ordered clicks",
                "primary_control": (
                    "same-impression -0 with subcategory absent from "
                    "model-visible history"
                ),
                "secondary_control": "all same-impression -0",
                "row_weighting": "sft_row_equal",
                "train_dev_aggregation": "reported_separately",
            },
        }
        _write_json_atomic(generation_dir / "manifest.json", manifest)

        old_target = _publish_analysis_generation(
            generation_dir,
            artifact_dir,
            overwrite=overwrite,
        )
        published = True
        if old_target is not None:
            root = generation_root.resolve()
            if old_target.is_relative_to(root) and old_target != generation_dir:
                shutil.rmtree(old_target)
        return manifest
    finally:
        if not published and generation_dir.exists():
            shutil.rmtree(generation_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze semantic and taxonomy relationships in the active "
            "SFT22K/Dev10K datasets."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/analysis/sft_training_similarity_v1.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_sft_training_similarity_analysis(
        args.config,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact_dir": manifest["artifact_dir"],
                "rows": manifest["join"]["rows"],
                "primary_rows": manifest["statistics"][
                    "primary_rows"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
