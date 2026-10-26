from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.build_nrms_title_embeddings import (
    build_nrms_title_embeddings,
    load_nrms_title_embeddings,
)


class FakeEncoder:
    def encode(self, texts, *, batch_size):
        del batch_size
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 17 + 1] for text in texts],
            dtype=np.float32,
        )


def _write_benchmark(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_nrms_title_embeddings_are_deduplicated_sorted_and_loadable(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_benchmark(
        benchmark / "fit.parquet",
        [{"user_id": "u1", "history_titles": ["zeta", "alpha", "zeta"]}],
    )
    _write_benchmark(
        benchmark / "validation.parquet",
        [{"user_id": "u2", "history_titles": ["beta"]}],
    )
    _write_benchmark(
        benchmark / "dev.parquet",
        [{"user_id": "u3", "history_titles": ["alpha"]}],
    )
    output = tmp_path / "output"

    build_nrms_title_embeddings(
        benchmark,
        output,
        encoder=FakeEncoder(),
        model_name="fake",
        revision="r1",
        batch_size=2,
    )
    titles, vectors, metadata = load_nrms_title_embeddings(output)

    assert titles == ("alpha", "beta", "zeta")
    assert vectors.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert metadata["unique_titles"] == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["output_sha256"]) == {
        "metadata.json",
        "title_embeddings.parquet",
    }


def test_nrms_title_embedding_builder_refuses_overwrite(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    for split in ("fit", "validation", "dev"):
        _write_benchmark(
            benchmark / f"{split}.parquet",
            [{"user_id": split, "history_titles": [split]}],
        )
    output = tmp_path / "output"
    build_nrms_title_embeddings(benchmark, output, encoder=FakeEncoder())
    with pytest.raises(FileExistsError):
        build_nrms_title_embeddings(benchmark, output, encoder=FakeEncoder())
