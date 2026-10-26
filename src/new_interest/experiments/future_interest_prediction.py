from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("ranked_labels", pa.list_(pa.string()), nullable=False),
        pa.field("ranked_scores", pa.list_(pa.float64()), nullable=False),
    ]
)


def load_vocabulary(path: Path | str) -> tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    if "subcategory" not in table.column_names:
        raise ValueError(f"{path}: missing 'subcategory' column")
    labels = tuple(str(value) for value in table["subcategory"].to_pylist())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"{path}: labels must be unique and non-empty")
    return labels


def rank_label_scores(
    labels: tuple[str, ...],
    scores: Mapping[str, float],
) -> dict[str, object]:
    if set(scores) != set(labels):
        missing = sorted(set(labels) - set(scores))
        unexpected = sorted(set(scores) - set(labels))
        raise ValueError(
            "Scores must cover the vocabulary exactly; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("Scores must be finite")
    label_order = {label: index for index, label in enumerate(labels)}
    ranked = sorted(
        labels,
        key=lambda label: (
            -float(scores[label]),
            label_order[label],
        ),
    )
    return {
        "ranked_labels": ranked,
        "ranked_scores": [float(scores[label]) for label in ranked],
    }


class PredictionWriter:
    def __init__(
        self,
        path: Path | str,
        *,
        overwrite: bool = False,
        batch_size: int = 2_000,
    ) -> None:
        self.path = Path(path)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"Prediction file already exists: {self.path}. "
                "Use overwrite=True."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".future-interest-predictions-",
            dir=str(self.path.parent),
        )
        self._staging_path = Path(self._temporary.name) / self.path.name
        self._writer = pq.ParquetWriter(
            self._staging_path,
            PREDICTION_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        self._batch_size = batch_size
        self._rows: list[dict[str, object]] = []
        self.row_count = 0
        self._closed = False

    def append(
        self,
        *,
        user_id: str,
        ranked_labels: Iterable[str],
        ranked_scores: Iterable[float],
    ) -> None:
        if self._closed:
            raise RuntimeError("PredictionWriter is closed")
        labels = list(ranked_labels)
        scores = [float(score) for score in ranked_scores]
        if len(labels) != len(scores):
            raise ValueError("ranked labels and scores differ in length")
        self._rows.append(
            {
                "user_id": user_id,
                "ranked_labels": labels,
                "ranked_scores": scores,
            }
        )
        if len(self._rows) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        self._writer.write_table(
            pa.Table.from_pylist(self._rows, schema=PREDICTION_SCHEMA),
            row_group_size=self._batch_size,
        )
        self.row_count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._writer.close()
        os.replace(self._staging_path, self.path)
        self._temporary.cleanup()
        self._closed = True

    def __enter__(self) -> PredictionWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
            return
        self._writer.close()
        self._temporary.cleanup()
        self._closed = True
