from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.future_interest_prediction import (
    PredictionWriter,
    load_vocabulary,
    rank_label_scores,
)
from new_interest.infrastructure.paths import portable_path


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
EMBEDDING_ARTIFACT_VERSION = 1
EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ]
)


class TextEncoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        """Encode texts into one vector per input text."""


EmbeddingEncoder = TextEncoder
Encoder = TextEncoder


def _resolve_revision(
    revision: str | None,
    model_revision: str | None,
) -> str | None:
    if revision is not None and model_revision is not None and revision != model_revision:
        raise ValueError("revision and model_revision disagree")
    return model_revision if model_revision is not None else revision


class SentenceTransformerEncoder:
    """Concrete optional dependency adapter for sentence-transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        revision: str | None = None,
        model_revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.revision = _resolve_revision(revision, model_revision)
        self.device = device
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the default encoder"
            ) from exc
        try:
            kwargs: dict[str, Any] = {}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            if device is not None:
                kwargs["device"] = device
            self._model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load sentence-transformer model {model_name!r}"
            ) from exc

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> Sequence[Sequence[float]]:
        try:
            return self._model.encode(
                list(texts),
                batch_size=batch_size or self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError("Unable to encode text with sentence-transformers") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_input(path: Path, required: set[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    missing = required - set(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _normalise_vector(value: Any, *, context: str) -> list[float]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: encoder returned a non-numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{context}: vector contains non-finite values or is empty")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{context}: vector has a non-finite or zero norm")
    return np.asarray(vector / norm, dtype=np.float32).tolist()


def _encode_normalised(
    encoder: TextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
    context: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    try:
        encoded = np.asarray(
            encoder.encode(texts, batch_size=batch_size),
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: encoder returned non-numeric vectors") from exc
    except (RuntimeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError("Encoder failed while encoding text") from exc
    if encoded.ndim != 2 or encoded.shape[0] != len(texts):
        raise ValueError(
            f"{context}: encoder returned shape {encoded.shape}, expected "
            f"({len(texts)}, dimensions)"
        )
    if encoded.shape[1] == 0 or not np.isfinite(encoded).all():
        raise ValueError(f"{context}: vectors contain non-finite values or are empty")
    norms = np.linalg.norm(encoded.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise ValueError(f"{context}: vector has a non-finite or zero norm")
    return np.ascontiguousarray(
        encoded / norms[:, np.newaxis],
        dtype=np.float32,
    )


def _iter_history_rows(path: Path):
    _check_input(path, {"user_id", "history_titles"})
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["user_id", "history_titles"],
        batch_size=8_192,
    ):
        for row in batch.to_pylist():
            if row["user_id"] is None or not str(row["user_id"]):
                raise ValueError(f"{path}: user_id must be non-empty")
            user_id = str(row["user_id"])
            titles = row["history_titles"]
            if titles is None or not titles:
                raise ValueError(f"{path}: {user_id} has missing history_titles")
            cleaned: list[str] = []
            for title in titles:
                if title is None or not str(title):
                    raise ValueError(f"{path}: {user_id} has a missing title")
                cleaned.append(str(title))
            yield {"user_id": user_id, "history_titles": cleaned}


def _history_rows(path: Path):
    """Compatibility wrapper returning a streaming history-row iterator."""
    return _iter_history_rows(path)


def _metadata(
    benchmark_path: Path,
    *,
    model_name: str,
    revision: str | None,
    device: str | None,
    batch_size: int,
    dimensions: int,
) -> dict[str, object]:
    return {
        "artifact_type": "history_embeddings",
        "artifact_version": EMBEDDING_ARTIFACT_VERSION,
        "model": {
            "name": model_name,
            "revision": revision,
            "device": device,
            "batch_size": batch_size,
        },
        "embedding_dimensions": dimensions,
        "source_sha256": {
            portable_path(benchmark_path): _sha256(benchmark_path)
        },
    }


def write_history_embeddings(
    benchmark_path: Path | str,
    output_path: Path | str,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str | None = None,
    model_revision: str | None = None,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    write_batch_size: int = 8_192,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write normalized, reusable mean-pooled history embeddings."""
    benchmark_path = Path(benchmark_path)
    output_path = Path(output_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    revision = _resolve_revision(revision, model_revision)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Embedding file already exists: {output_path}. Use overwrite=True."
        )
    unique_titles: dict[str, None] = {}
    pass_one_rows = 0
    for row in _iter_history_rows(benchmark_path):
        pass_one_rows += 1
        for title in row["history_titles"]:
            unique_titles.setdefault(title, None)
    if pass_one_rows == 0:
        raise ValueError(f"{benchmark_path}: no benchmark rows")
    titles = list(unique_titles)
    if write_batch_size <= 0:
        raise ValueError("write_batch_size must be positive")
    encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    title_vectors = _encode_normalised(
        encoder,
        titles,
        batch_size=batch_size,
        context="history titles",
    )
    title_to_row = {title: index for index, title in enumerate(titles)}
    dimensions = title_vectors.shape[1]
    metadata = _metadata(
        benchmark_path,
        model_name=model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
        dimensions=dimensions,
    )
    schema = EMBEDDING_SCHEMA.with_metadata(
        {
            b"metadata": json.dumps(
                metadata, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    user_count = 0
    seen_users: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix=".future-interest-embeddings-",
        dir=str(output_path.parent),
    ) as temporary:
        staged = Path(temporary) / output_path.name
        writer = pq.ParquetWriter(
            staged,
            schema,
            compression="zstd",
        )
        pending: list[dict[str, object]] = []
        try:
            for row in _iter_history_rows(benchmark_path):
                user_id = row["user_id"]
                if user_id in seen_users:
                    raise ValueError(
                        f"{benchmark_path}: duplicate user_id {user_id!r}"
                    )
                seen_users.add(user_id)
                title_rows = np.fromiter(
                    (title_to_row[title] for title in row["history_titles"]),
                    dtype=np.intp,
                )
                pooled = title_vectors[title_rows].mean(
                    axis=0,
                    dtype=np.float32,
                )
                embedding = _normalise_vector(
                    pooled,
                    context=f"history for user {user_id!r}",
                )
                pending.append({"user_id": user_id, "embedding": embedding})
                user_count += 1
                if len(pending) >= write_batch_size:
                    writer.write_table(
                        pa.Table.from_pylist(pending, schema=schema),
                        row_group_size=write_batch_size,
                    )
                    pending.clear()
            if pending:
                writer.write_table(
                    pa.Table.from_pylist(pending, schema=schema),
                    row_group_size=write_batch_size,
                )
        finally:
            writer.close()
        if user_count == 0:
            raise ValueError(f"{benchmark_path}: no benchmark rows")
        os.replace(staged, output_path)
    return {
        "users": user_count,
        "unique_titles": len(titles),
        "dimensions": dimensions,
        "path": str(output_path),
        "metadata": metadata,
    }


def _load_embeddings(
    path: Path,
    *,
    benchmark_path: Path,
) -> tuple[Any, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    table_schema = parquet.schema_arrow
    required = {"user_id", "embedding"}
    missing = required - set(table_schema.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    raw_metadata = (table_schema.metadata or {}).get(b"metadata")
    if raw_metadata is None:
        raise ValueError(f"{path}: missing embedding metadata")
    try:
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid embedding metadata") from exc
    source_hashes = metadata.get("source_sha256", {})
    expected_hash = source_hashes.get(portable_path(benchmark_path))
    if expected_hash is not None and expected_hash != _sha256(benchmark_path):
        raise ValueError(f"{path}: benchmark source hash does not match artifact")

    def rows():
        seen_users: set[str] = set()
        dimensions: int | None = None
        row_count = 0
        for batch in parquet.iter_batches(
            columns=["user_id", "embedding"],
            batch_size=8_192,
        ):
            for row in batch.to_pylist():
                user_id = str(row["user_id"])
                if user_id in seen_users:
                    raise ValueError(f"{path}: duplicate user_id {user_id!r}")
                seen_users.add(user_id)
                embedding = _normalise_vector(
                    row["embedding"], context=f"{path} user {user_id!r}"
                )
                if dimensions is None:
                    dimensions = len(embedding)
                elif len(embedding) != dimensions:
                    raise ValueError(f"{path}: inconsistent embedding dimensions")
                row_count += 1
                yield {"user_id": user_id, "embedding": embedding}
        if row_count == 0:
            raise ValueError(f"{path}: no embeddings")

    return rows(), metadata


def _vocabulary_topics(path: Path) -> tuple[tuple[str, ...], list[str]]:
    _check_input(path, {"category", "subcategory"})
    labels = load_vocabulary(path)
    table = pq.read_table(path, columns=["category", "subcategory"])
    topics: list[str] = []
    seen: set[str] = set()
    for row in table.to_pylist():
        category = row["category"]
        label = row["subcategory"]
        if category is None or label is None:
            raise ValueError(f"{path}: vocabulary contains missing category/topic")
        label = str(label)
        if label in seen:
            raise ValueError(f"{path}: duplicate subcategory {label!r}")
        seen.add(label)
        topics.append(f"category: {str(category)}; topic: {label}")
    if tuple(seen) != labels and set(seen) != set(labels):
        raise ValueError(f"{path}: category/subcategory rows disagree with vocabulary")
    by_label = dict(zip((str(row["subcategory"]) for row in table.to_pylist()), topics))
    return labels, [by_label[label] for label in labels]


def predict_embedding_future_interest(
    benchmark_path: Path | str,
    vocabulary_path: Path | str,
    history_embeddings_path: Path | str,
    output_path: Path | str,
    *,
    encoder: TextEncoder | None = None,
    model_name: str | None = None,
    revision: str | None = None,
    model_revision: str | None = None,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    metric: str = "cosine",
    overwrite: bool = False,
) -> dict[str, object]:
    """Rank every vocabulary topic for every benchmark user."""
    benchmark_path = Path(benchmark_path)
    vocabulary_path = Path(vocabulary_path)
    history_embeddings_path = Path(history_embeddings_path)
    output_path = Path(output_path)
    _check_input(benchmark_path, {"user_id", "history_titles"})
    if metric not in {"cosine", "dot"}:
        raise ValueError("metric must be 'cosine' or 'dot'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prediction file already exists: {output_path}. Use overwrite=True."
        )
    revision = _resolve_revision(revision, model_revision)
    embedding_rows, embedding_metadata = _load_embeddings(
        history_embeddings_path, benchmark_path=benchmark_path
    )
    labels, topic_texts = _vocabulary_topics(vocabulary_path)
    model_config = embedding_metadata.get("model", {})
    if not isinstance(model_config, dict) or not isinstance(
        model_config.get("name"), str
    ):
        raise ValueError(
            f"{history_embeddings_path}: embedding metadata lacks model name"
        )
    artifact_model_name = model_config["name"]
    artifact_revision = model_config.get("revision")
    if artifact_revision is not None and not isinstance(artifact_revision, str):
        raise ValueError(
            f"{history_embeddings_path}: embedding metadata has invalid revision"
        )
    if model_name is not None and model_name != artifact_model_name:
        raise ValueError(
            "Predictor model name does not match embedding artifact: "
            f"{model_name!r} != {artifact_model_name!r}"
        )
    if revision is not None and revision != artifact_revision:
        raise ValueError(
            "Predictor model revision does not match embedding artifact: "
            f"{revision!r} != {artifact_revision!r}"
        )
    if encoder is None:
        encoder = SentenceTransformerEncoder(
            model_name or artifact_model_name,
            revision=artifact_revision if revision is None else revision,
            device=device
            if device is not None
            else model_config.get("device"),
            batch_size=batch_size,
        )
    topic_vectors = _encode_normalised(
        encoder,
        topic_texts,
        batch_size=batch_size,
        context="vocabulary topics",
    )
    prediction_user_count = 0
    with PredictionWriter(output_path, overwrite=overwrite) as writer:
        for row in embedding_rows:
            prediction_user_count += 1
            history = np.asarray(row["embedding"], dtype=np.float32)
            scores_array = topic_vectors @ history
            if metric == "cosine":
                history_norm = float(np.linalg.norm(history.astype(np.float64)))
                topic_norms = np.linalg.norm(
                    topic_vectors.astype(np.float64),
                    axis=1,
                )
                scores_array = scores_array / (history_norm * topic_norms)
            scores = {
                label: float(score)
                for label, score in zip(labels, scores_array, strict=True)
            }
            ranked = rank_label_scores(labels, scores)
            writer.append(
                user_id=str(row["user_id"]),
                ranked_labels=ranked["ranked_labels"],
                ranked_scores=ranked["ranked_scores"],
            )
    return {
        "users": prediction_user_count,
        "labels": len(labels),
        "prediction_path": str(output_path),
        "history_embeddings_path": str(history_embeddings_path),
        "model": model_config,
    }


# Descriptive aliases for callers that prefer the artifact terminology.
write_embedding_artifact = write_history_embeddings
predict_full_vocabulary = predict_embedding_future_interest
