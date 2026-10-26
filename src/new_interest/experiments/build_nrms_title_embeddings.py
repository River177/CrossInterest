from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.embedding_future_interest import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEncoder,
    TextEncoder,
    _encode_normalised,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-nrms-title-embeddings-v1"
DEFAULT_BENCHMARK_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_semantic_users_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/future-novel-interest/"
    "static-user-disjoint-nrms-title-embeddings-v1"
)
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_titles(paths: Sequence[Path]) -> tuple[str, ...]:
    titles: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if "history_titles" not in pq.ParquetFile(path).schema_arrow.names:
            raise ValueError(f"{path}: missing history_titles")
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["history_titles"], batch_size=8_192
        ):
            for history in batch["history_titles"].to_pylist():
                if not history:
                    raise ValueError(f"{path}: empty History")
                for title in history:
                    if title is None or not str(title):
                        raise ValueError(f"{path}: missing History title")
                    titles.add(str(title))
    if not titles:
        raise ValueError("no History titles found")
    return tuple(sorted(titles))


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    backup = output.with_name(output.name + ".old")
    if backup.exists():
        raise FileExistsError(f"stale backup blocks publication: {backup}")
    if output.exists():
        os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output)


def build_nrms_title_embeddings(
    benchmark_dir: Path | str = DEFAULT_BENCHMARK_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_REVISION,
    device: str | None = None,
    batch_size: int = 256,
    write_batch_size: int = 4_096,
    overwrite: bool = False,
) -> Path:
    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    if batch_size <= 0 or write_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")
    benchmark_paths = tuple(
        benchmark_dir / f"{split}.parquet"
        for split in ("fit", "validation", "dev")
    )
    titles = _unique_titles(benchmark_paths)
    current_encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    vectors = _encode_normalised(
        current_encoder,
        titles,
        batch_size=batch_size,
        context="NRMS unique History titles",
    )
    if vectors.shape[0] != len(titles):
        raise ValueError("title encoder output is misaligned")
    dimensions = int(vectors.shape[1])
    source_hashes = {
        portable_path(path): _sha256(path) for path in benchmark_paths
    }
    metadata = {
        "rule_version": RULE_VERSION,
        "model": {
            "name": model_name,
            "revision": revision,
            "device": device,
            "batch_size": batch_size,
        },
        "normalization": "each title vector has unit L2 norm",
        "title_order": "UTF-8 lexical sort",
        "unique_titles": len(titles),
        "dimensions": dimensions,
        "source_sha256": source_hashes,
    }
    schema = pa.schema(
        [
            pa.field("title_id", pa.int32(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(pa.float32(), dimensions),
                nullable=False,
            ),
        ],
        metadata={
            b"metadata": json.dumps(
                metadata, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".nrms-title-embeddings-", dir=str(output_dir.parent)
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        output_path = staging / "title_embeddings.parquet"
        writer = pq.ParquetWriter(
            output_path, schema, compression="zstd", use_dictionary=True
        )
        try:
            for start in range(0, len(titles), write_batch_size):
                end = min(start + write_batch_size, len(titles))
                flat = pa.array(
                    vectors[start:end].reshape(-1), type=pa.float32()
                )
                embedding = pa.FixedSizeListArray.from_arrays(flat, dimensions)
                table = pa.Table.from_arrays(
                    [
                        pa.array(range(start, end), type=pa.int32()),
                        pa.array(titles[start:end], type=pa.string()),
                        embedding,
                    ],
                    schema=schema,
                )
                writer.write_table(table, row_group_size=write_batch_size)
        finally:
            writer.close()
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            "metadata.json": _sha256(staging / "metadata.json"),
            "title_embeddings.parquet": _sha256(output_path),
        }
        manifest = {
            "rule_version": RULE_VERSION,
            "source_sha256": source_hashes,
            "output_sha256": output_hashes,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)
    return output_dir


def load_nrms_title_embeddings(
    output_dir: Path | str,
) -> tuple[tuple[str, ...], np.ndarray, Mapping[str, object]]:
    output_dir = Path(output_dir)
    path = output_dir / "title_embeddings.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    raw_metadata = (parquet.schema_arrow.metadata or {}).get(b"metadata")
    if raw_metadata is None:
        raise ValueError(f"{path}: missing metadata")
    metadata = json.loads(raw_metadata.decode("utf-8"))
    dimensions = int(metadata["dimensions"])
    table = pq.read_table(path, columns=["title_id", "title", "embedding"])
    title_ids = np.asarray(table["title_id"].to_numpy(), dtype=np.int64)
    if not np.array_equal(title_ids, np.arange(len(title_ids))):
        raise ValueError("title IDs are not contiguous and ordered")
    titles = tuple(str(value) for value in table["title"].to_pylist())
    embeddings_column = table["embedding"].combine_chunks()
    embeddings = np.asarray(
        embeddings_column.values.to_numpy(zero_copy_only=False),
        dtype=np.float32,
    ).reshape(len(titles), dimensions)
    if not np.isfinite(embeddings).all():
        raise ValueError("title embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("title embeddings are not unit normalized")
    return titles, np.ascontiguousarray(embeddings), metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic frozen MiniLM embeddings per unique History title."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = build_nrms_title_embeddings(
        args.benchmark_dir,
        args.output_dir,
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
