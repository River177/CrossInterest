from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import new_interest.experiments.embedding_future_interest as embedding_module
from new_interest.experiments.embedding_future_interest import (
    predict_embedding_future_interest,
    write_history_embeddings,
)


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def _write_fixture(root: Path) -> tuple[Path, Path]:
    benchmark = root / "benchmark.parquet"
    vocabulary = root / "vocabulary.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"user_id": "u1", "history_titles": ["one", "two", "one"]},
                {"user_id": "u2", "history_titles": ["two"]},
            ]
        ),
        benchmark,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"category": "A", "subcategory": "a"},
                {"category": "B", "subcategory": "b"},
                {"category": "C", "subcategory": "c"},
            ]
        ),
        vocabulary,
    )
    return benchmark, vocabulary


def test_writes_reusable_embeddings_and_preserves_repeated_titles(
    tmp_path: Path,
) -> None:
    benchmark, _ = _write_fixture(tmp_path)
    encoder = FakeEncoder(
        {
            "one": [1.0, 0.0],
            "two": [0.0, 1.0],
        }
    )
    output = tmp_path / "history.parquet"
    result = write_history_embeddings(
        benchmark,
        output,
        encoder=encoder,
        model_name="fake",
    )

    assert result["users"] == 2
    assert encoder.calls == [["one", "two"]]
    table = pq.read_table(output)
    rows = table.to_pylist()
    assert table.schema.field("embedding").type == pa.list_(pa.float32())
    assert rows[0]["embedding"][0] > rows[0]["embedding"][1]
    metadata = json.loads(table.schema.metadata[b"metadata"])
    assert metadata["model"]["name"] == "fake"
    assert str(benchmark) in metadata["source_sha256"]


def test_predictions_rank_full_vocabulary(tmp_path: Path) -> None:
    benchmark, vocabulary = _write_fixture(tmp_path)
    encoder = FakeEncoder(
        {
            "one": [1.0, 0.0],
            "two": [0.0, 1.0],
            "category: A; topic: a": [1.0, 0.0],
            "category: B; topic: b": [0.0, 1.0],
            "category: C; topic: c": [-1.0, 0.0],
        }
    )
    history = tmp_path / "history.parquet"
    write_history_embeddings(benchmark, history, encoder=encoder)
    predictions = tmp_path / "predictions.parquet"
    predict_embedding_future_interest(
        benchmark,
        vocabulary,
        history,
        predictions,
        encoder=encoder,
    )

    rows = pq.read_table(predictions).to_pylist()
    assert rows[0]["ranked_labels"] == ["a", "b", "c"]
    assert len(rows[0]["ranked_labels"]) == 3
    assert set(rows[0]["ranked_labels"]) == {"a", "b", "c"}


def test_rejects_invalid_vectors_and_overwrite(tmp_path: Path) -> None:
    benchmark, _ = _write_fixture(tmp_path)
    output = tmp_path / "history.parquet"
    with pytest.raises(ValueError, match="zero norm"):
        write_history_embeddings(
            benchmark,
            output,
            encoder=FakeEncoder({"one": [0.0, 0.0], "two": [1.0, 0.0]}),
        )

    write_history_embeddings(
        benchmark,
        output,
        encoder=FakeEncoder({"one": [1.0, 0.0], "two": [0.0, 1.0]}),
    )
    with pytest.raises(FileExistsError):
        write_history_embeddings(
            benchmark,
            output,
            encoder=FakeEncoder({"one": [1.0, 0.0], "two": [0.0, 1.0]}),
        )


def test_prediction_uses_artifact_model_and_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, vocabulary = _write_fixture(tmp_path)
    vectors = {
        "one": [1.0, 0.0],
        "two": [0.0, 1.0],
        "category: A; topic: a": [1.0, 0.0],
        "category: B; topic: b": [0.0, 1.0],
        "category: C; topic: c": [-1.0, 0.0],
    }
    history = tmp_path / "history.parquet"
    write_history_embeddings(
        benchmark,
        history,
        encoder=FakeEncoder(vectors),
        model_name="fake-model",
        revision="rev-1",
    )

    class StubAdapter(FakeEncoder):
        init_args: tuple[str, str | None] | None = None

        def __init__(
            self,
            model_name: str,
            *,
            revision: str | None,
            device: str | None,
            batch_size: int,
        ) -> None:
            type(self).init_args = (model_name, revision)
            super().__init__(vectors)

    monkeypatch.setattr(embedding_module, "SentenceTransformerEncoder", StubAdapter)
    predictions = tmp_path / "predictions.parquet"
    predict_embedding_future_interest(
        benchmark,
        vocabulary,
        history,
        predictions,
    )
    assert StubAdapter.init_args == ("fake-model", "rev-1")

    with pytest.raises(ValueError, match="model name"):
        predict_embedding_future_interest(
            benchmark,
            vocabulary,
            history,
            tmp_path / "mismatch.parquet",
            model_name="other-model",
        )
