from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.build_nrms_interest_embeddings import (
    build_nrms_interest_embeddings,
    load_nrms_interest_embeddings,
)


class FakeEncoder:
    def encode(self, texts, *, batch_size):
        del batch_size
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 19 + 1] for text in texts],
            dtype=np.float32,
        )


def _write_vocabulary(path: Path, values: list[str]) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "interest_id": index,
                    "category": "category",
                    "subcategory": value,
                }
                for index, value in enumerate(values)
            ]
        ),
        path,
    )


def test_interest_embeddings_cover_sorted_vocabulary_and_are_loadable(
    tmp_path: Path,
) -> None:
    vocabulary = tmp_path / "interest_vocabulary.parquet"
    _write_vocabulary(vocabulary, ["zeta", "alpha", "beta"])
    output = tmp_path / "output"

    build_nrms_interest_embeddings(
        vocabulary,
        output,
        encoder=FakeEncoder(),
        model_name="fake",
        revision="r1",
        batch_size=2,
    )
    interests, vectors, metadata = load_nrms_interest_embeddings(output)

    assert interests == ("alpha", "beta", "zeta")
    assert vectors.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert metadata["unique_interests"] == 3
    assert metadata["model"]["revision"] == "r1"
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["output_sha256"]) == {
        "metadata.json",
        "interest_embeddings.parquet",
    }


def test_interest_embedding_builder_rejects_duplicates_and_overwrite(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.parquet"
    _write_vocabulary(duplicate, ["alpha", "alpha"])
    with pytest.raises(ValueError, match="duplicate interest"):
        build_nrms_interest_embeddings(
            duplicate, tmp_path / "duplicate-output", encoder=FakeEncoder()
        )

    vocabulary = tmp_path / "interest_vocabulary.parquet"
    _write_vocabulary(vocabulary, ["alpha", "beta"])
    output = tmp_path / "output"
    build_nrms_interest_embeddings(vocabulary, output, encoder=FakeEncoder())
    with pytest.raises(FileExistsError):
        build_nrms_interest_embeddings(
            vocabulary, output, encoder=FakeEncoder()
        )
