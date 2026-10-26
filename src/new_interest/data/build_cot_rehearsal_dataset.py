from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.infrastructure.paths import portable_path

RULE_VERSION = "cot-direct-rehearsal-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cot_rehearsal_datasets(
    direct_dir: Path | str,
    cot_dir: Path | str,
    output_dir: Path | str,
    *,
    size: int = 1_000,
    cot_repeats: int = 3,
    direct_repeats: int = 1,
    overwrite: bool = False,
) -> Path:
    if size <= 0 or cot_repeats <= 0 or direct_repeats <= 0:
        raise ValueError("size and repeat counts must be positive")
    direct_dir = Path(direct_dir)
    cot_dir = Path(cot_dir)
    output_dir = Path(output_dir)
    name = f"{size // 1_000}k" if size % 1_000 == 0 else str(size)
    output_paths = [
        output_dir / task_type / f"{name}.parquet"
        for task_type in ("behavior_only", "interest_only")
    ] + [output_dir / "statistics.json", output_dir / "metadata.json"]
    if any(path.exists() for path in output_paths) and not overwrite:
        raise FileExistsError(
            f"Rehearsal outputs exist in {output_dir}. Use --overwrite."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cot-rehearsal-",
        dir=output_dir.parent,
    ) as staging_name:
        staging = Path(staging_name)
        statistics: dict[str, object] = {"tasks": {}}
        source_hashes: dict[str, str] = {}
        identities: dict[str, list[tuple[str, str, str]]] = {}
        for task_type in ("behavior_only", "interest_only"):
            direct_path = (
                direct_dir / "train" / task_type / f"{name}.parquet"
            )
            cot_path = cot_dir / task_type / f"{name}.parquet"
            for path in (direct_path, cot_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            direct_rows = pq.read_table(direct_path).to_pylist()
            cot_rows = pq.read_table(cot_path).to_pylist()
            if len(direct_rows) != size or len(cot_rows) != size:
                raise ValueError(
                    f"{task_type}: expected {size} direct and CoT rows"
                )
            direct_identity = [
                (
                    str(row["source_sample_id"]),
                    str(row["impression_id"]),
                    str(row["target_subcategory"]),
                )
                for row in direct_rows
            ]
            cot_identity = [
                (
                    str(row["source_sample_id"]),
                    str(row["impression_id"]),
                    str(row["target_subcategory"]),
                )
                for row in cot_rows
            ]
            if direct_identity != cot_identity:
                raise ValueError(
                    f"{task_type}: direct and CoT rows do not align"
                )
            identities[task_type] = direct_identity

            mixed_rows: list[dict[str, object]] = []
            for direct, cot in zip(direct_rows, cot_rows):
                variants = [
                    *[
                        ("cot", repeat, cot)
                        for repeat in range(cot_repeats)
                    ],
                    *[
                        ("direct", repeat, direct)
                        for repeat in range(direct_repeats)
                    ],
                ]
                for variant, repeat, source in variants:
                    mixed_rows.append(
                        {
                            **source,
                            "sample_id": (
                                f"{source['sample_id']}:"
                                f"{variant}-rehearsal-{repeat}"
                            ),
                        }
                    )
            destination = staging / task_type / f"{name}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(
                    mixed_rows,
                    schema=GENERATIVE_SCHEMA,
                ),
                destination,
                compression="zstd",
                use_dictionary=True,
                row_group_size=2_000,
            )
            source_hashes[
                f"direct/{task_type}/{name}.parquet"
            ] = _sha256(direct_path)
            source_hashes[
                f"cot/{task_type}/{name}.parquet"
            ] = _sha256(cot_path)
            statistics["tasks"][task_type] = {
                "unique_sources": size,
                "cot_repeats": cot_repeats,
                "direct_repeats": direct_repeats,
                "samples": len(mixed_rows),
            }
        if identities["behavior_only"] != identities["interest_only"]:
            raise ValueError("Behavior/Interest source identities differ")

        (staging / "statistics.json").write_text(
            json.dumps(statistics, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "direct_dir": portable_path(direct_dir),
            "cot_dir": portable_path(cot_dir),
            "size": size,
            "cot_repeats": cot_repeats,
            "direct_repeats": direct_repeats,
            "ordering": "per-source-cot-repeats-then-direct-repeats",
            "source_sha256": source_hashes,
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path)
                for path in staging.rglob("*")
                if path.is_file()
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_paths:
            destination = output_dir / path.relative_to(output_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / path.relative_to(output_dir), destination)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build mixed CoT/direct rehearsal SFT data."
    )
    parser.add_argument(
        "--direct-dir",
        type=Path,
        default=Path("data/training/sft_experiments"),
    )
    parser.add_argument(
        "--cot-dir",
        type=Path,
        default=Path("data/training/cot_sft_experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/training/cot_sft_rehearsal_experiments"
        ),
    )
    parser.add_argument("--size", type=int, default=1_000)
    parser.add_argument("--cot-repeats", type=int, default=3)
    parser.add_argument("--direct-repeats", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = build_cot_rehearsal_datasets(
        args.direct_dir,
        args.cot_dir,
        args.output_dir,
        size=args.size,
        cot_repeats=args.cot_repeats,
        direct_repeats=args.direct_repeats,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
