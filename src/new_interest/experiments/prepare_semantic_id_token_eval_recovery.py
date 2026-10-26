from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPECTED_OPTIMIZER_STEPS = 2025
SHARD_NAMES = tuple(f"shard-{index:02d}" for index in range(4))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _assert_safe_source_root(
    source_root: Path,
    *,
    expected_source_result_id: str,
) -> Path:
    resolved = source_root.resolve(strict=True)
    if resolved.name != expected_source_result_id:
        raise ValueError(
            "Source result identity mismatch: "
            f"expected {expected_source_result_id!r}, got {resolved.name!r}"
        )
    required = (
        resolved / "sft" / "final_adapter" / "adapter_model.safetensors",
        resolved / "sft" / "run_summary.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Source result is missing completed training artifacts: "
            + ", ".join(missing)
        )
    summary = json.loads(required[1].read_text(encoding="utf-8"))
    if int(summary.get("expected_optimizer_steps", -1)) != EXPECTED_OPTIMIZER_STEPS:
        raise ValueError("Source training summary is not the 2,025-step full run")
    return resolved


def _inspect_cache(cache_dir: Path) -> dict[str, Any]:
    parquet_files = sorted(cache_dir.glob("*.parquet"))
    identities: set[str] = set()
    rows = 0
    total_bytes = 0
    for path in parquet_files:
        total_bytes += path.stat().st_size
        table = pq.read_table(path, columns=["evaluation_identity"])
        rows += table.num_rows
        identities.update(str(value.as_py()) for value in table.column(0))
    return {
        "path": str(cache_dir),
        "parquet_files": len(parquet_files),
        "rows": rows,
        "bytes": total_bytes,
        "evaluation_identities": sorted(identities),
    }


def clean_stale_eval_caches(
    source_root: Path,
    output_file: Path,
    *,
    expected_source_result_id: str,
    expected_stale_identity: str,
) -> dict[str, Any]:
    root = _assert_safe_source_root(
        source_root,
        expected_source_result_id=expected_source_result_id,
    )
    eval_root = root / "eval"
    inspections: list[dict[str, Any]] = []
    removed: list[str] = []
    for shard_name in SHARD_NAMES:
        cache_dir = eval_root / shard_name / "shards"
        if cache_dir.is_symlink():
            raise ValueError(f"Refusing to remove symlinked cache: {cache_dir}")
        if not cache_dir.exists():
            inspections.append(
                {
                    "path": str(cache_dir),
                    "parquet_files": 0,
                    "rows": 0,
                    "bytes": 0,
                    "evaluation_identities": [],
                    "status": "already_absent",
                }
            )
            continue
        if not cache_dir.is_dir():
            raise ValueError(f"Expected a cache directory: {cache_dir}")
        inspection = _inspect_cache(cache_dir)
        identities = set(inspection["evaluation_identities"])
        if identities and identities != {expected_stale_identity}:
            raise ValueError(
                f"Unexpected evaluation identities in {cache_dir}: "
                f"{sorted(identities)}"
            )
        inspection["status"] = "removed"
        inspections.append(inspection)
        shutil.rmtree(cache_dir)
        if cache_dir.exists():
            raise RuntimeError(f"Failed to remove stale cache: {cache_dir}")
        removed.append(str(cache_dir))

    manifest = {
        "source_root": str(root),
        "expected_source_result_id": expected_source_result_id,
        "expected_stale_identity": expected_stale_identity,
        "training_expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "removed_cache_directories": removed,
        "shards": inspections,
        "preserved_final_adapter": str(root / "sft" / "final_adapter"),
        "preserved_checkpoint_root": str(root / "sft" / "checkpoints"),
    }
    _atomic_write_json(output_file, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove only stale Semantic-ID evaluation checkpoint caches from "
            "one completed AMLT result."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-result-id", required=True)
    parser.add_argument("--expected-stale-identity", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = clean_stale_eval_caches(
        args.source_root,
        args.output,
        expected_source_result_id=args.expected_source_result_id,
        expected_stale_identity=args.expected_stale_identity,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
