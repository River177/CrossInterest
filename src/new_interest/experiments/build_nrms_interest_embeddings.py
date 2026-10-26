from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.build_nrms_title_embeddings import (
    DEFAULT_REVISION,
    _publish,
    _sha256,
)
from new_interest.experiments.embedding_future_interest import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEncoder,
    TextEncoder,
    _encode_normalised,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-nrms-interest-embeddings-v1"
DEFAULT_VOCABULARY_PATH = Path(
    "data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/"
    "interest_vocabulary.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/future-novel-interest/"
    "static-user-disjoint-nrms-interest-embeddings-v1"
)


def _interest_names(vocabulary_path: Path) -> tuple[str, ...]:
    if not vocabulary_path.is_file():
        raise FileNotFoundError(vocabulary_path)
    parquet = pq.ParquetFile(vocabulary_path)
    if "subcategory" not in parquet.schema_arrow.names:
        raise ValueError(f"{vocabulary_path}: missing subcategory")
    raw = pq.read_table(vocabulary_path, columns=["subcategory"])[
        "subcategory"
    ].to_pylist()
    if any(value is None or not str(value) for value in raw):
        raise ValueError(f"{vocabulary_path}: missing interest name")
    interests = tuple(sorted({str(value) for value in raw}))
    if len(interests) != len(raw):
        raise ValueError(f"{vocabulary_path}: duplicate interest name")
    if not interests:
        raise ValueError(f"{vocabulary_path}: no interests")
    return interests


def build_nrms_interest_embeddings(
    vocabulary_path: Path | str = DEFAULT_VOCABULARY_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_REVISION,
    device: str | None = None,
    batch_size: int = 256,
    overwrite: bool = False,
) -> Path:
    vocabulary_path = Path(vocabulary_path)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    interests = _interest_names(vocabulary_path)
    current_encoder = encoder or SentenceTransformerEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
    )
    vectors = _encode_normalised(
        current_encoder,
        interests,
        batch_size=batch_size,
        context="NRMS interest names",
    )
    if vectors.shape[0] != len(interests):
        raise ValueError("interest encoder output is misaligned")
    dimensions = int(vectors.shape[1])
    source_hashes = {portable_path(vocabulary_path): _sha256(vocabulary_path)}
    metadata = {
        "rule_version": RULE_VERSION,
        "model": {
            "name": model_name,
            "revision": revision,
            "device": device,
            "batch_size": batch_size,
        },
        "normalization": "each interest-name vector has unit L2 norm",
        "interest_order": "UTF-8 lexical sort",
        "unique_interests": len(interests),
        "dimensions": dimensions,
        "source_sha256": source_hashes,
    }
    schema = pa.schema(
        [
            pa.field("interest_id", pa.int16(), nullable=False),
            pa.field("interest", pa.string(), nullable=False),
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
        prefix=".nrms-interest-embeddings-", dir=str(output_dir.parent)
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        output_path = staging / "interest_embeddings.parquet"
        flat = pa.array(vectors.reshape(-1), type=pa.float32())
        embedding = pa.FixedSizeListArray.from_arrays(flat, dimensions)
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(range(len(interests)), type=pa.int16()),
                    pa.array(interests, type=pa.string()),
                    embedding,
                ],
                schema=schema,
            ),
            output_path,
            compression="zstd",
            use_dictionary=True,
        )
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            "metadata.json": _sha256(staging / "metadata.json"),
            "interest_embeddings.parquet": _sha256(output_path),
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


def load_nrms_interest_embeddings(
    output_dir: Path | str,
) -> tuple[tuple[str, ...], np.ndarray, Mapping[str, object]]:
    output_dir = Path(output_dir)
    path = output_dir / "interest_embeddings.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    raw_metadata = (parquet.schema_arrow.metadata or {}).get(b"metadata")
    if raw_metadata is None:
        raise ValueError(f"{path}: missing metadata")
    metadata = json.loads(raw_metadata.decode("utf-8"))
    dimensions = int(metadata["dimensions"])
    table = pq.read_table(
        path, columns=["interest_id", "interest", "embedding"]
    )
    interest_ids = np.asarray(table["interest_id"].to_numpy(), dtype=np.int64)
    if not np.array_equal(interest_ids, np.arange(len(interest_ids))):
        raise ValueError("interest IDs are not contiguous and ordered")
    interests = tuple(str(value) for value in table["interest"].to_pylist())
    embedding_column = table["embedding"].combine_chunks()
    embeddings = np.asarray(
        embedding_column.values.to_numpy(zero_copy_only=False),
        dtype=np.float32,
    ).reshape(len(interests), dimensions)
    if not np.isfinite(embeddings).all():
        raise ValueError("interest embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("interest embeddings are not unit normalized")
    return interests, np.ascontiguousarray(embeddings), metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic frozen MiniLM embeddings for the complete "
            "interest vocabulary used by Interest-NRMS."
        )
    )
    parser.add_argument(
        "--vocabulary-path", type=Path, default=DEFAULT_VOCABULARY_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = build_nrms_interest_embeddings(
        args.vocabulary_path,
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
