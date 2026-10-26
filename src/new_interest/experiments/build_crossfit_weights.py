from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_crossfit_reweighting_dataset import (
    DEFAULT_OUTPUT_DIR,
)
from new_interest.evaluation.evaluate_beam import sha256_file


OOF_SCORE_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("category_token_count", pa.int16(), nullable=False),
        pa.field("category_token_nll", pa.float64(), nullable=False),
        pa.field("teacher_identity", pa.string(), nullable=False),
    ]
)

WEIGHT_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("category_token_nll", pa.float64(), nullable=False),
        pa.field("within_class_nll_rank", pa.int16(), nullable=False),
        pa.field("difficulty_quantile", pa.float64(), nullable=False),
        pa.field("raw_weight", pa.float64(), nullable=False),
        pa.field("sample_weight", pa.float64(), nullable=False),
    ]
)


def compute_class_normalized_weights(
    scores: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in scores:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"duplicate OOF score for {source_id}")
        seen.add(source_id)
        nll = float(row["category_token_nll"])
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError(f"invalid category_token_nll for {source_id}: {nll}")
        by_target[str(row["target_subcategory"])].append(row)

    output: list[dict[str, object]] = []
    for target, rows in sorted(by_target.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["category_token_nll"]),
                str(row["source_sample_id"]),
            ),
        )
        denominator = len(ordered) - 1
        if denominator <= 0:
            raise ValueError(f"{target}: at least two scores are required")
        staged = []
        for rank, row in enumerate(ordered):
            quantile = rank / denominator
            raw_weight = 1.0 / (1.0 + quantile)
            staged.append((row, rank, quantile, raw_weight))
        class_mean = sum(item[3] for item in staged) / len(staged)
        for row, rank, quantile, raw_weight in staged:
            output.append(
                {
                    "source_sample_id": str(row["source_sample_id"]),
                    "user_id": str(row["user_id"]),
                    "target_subcategory": target,
                    "fold": int(row["fold"]),
                    "category_token_nll": float(row["category_token_nll"]),
                    "within_class_nll_rank": rank,
                    "difficulty_quantile": quantile,
                    "raw_weight": raw_weight,
                    "sample_weight": raw_weight / class_mean,
                }
            )
    return sorted(output, key=lambda row: str(row["source_sample_id"]))


def _atomic_write_table(
    path: Path,
    rows: list[dict[str, object]],
    schema: pa.Schema,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            temporary,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_and_validate_scores(
    score_files: list[Path],
    fold_manifest: Path,
) -> list[dict[str, object]]:
    manifest_rows = pq.read_table(fold_manifest).to_pylist()
    manifest = {
        str(row["source_sample_id"]): row for row in manifest_rows
    }
    if len(manifest) != len(manifest_rows):
        raise ValueError("Fold manifest contains duplicate source identities")
    scores: list[dict[str, object]] = []
    for path in score_files:
        rows = pq.read_table(path).to_pylist()
        scores.extend(rows)
    score_ids = [str(row["source_sample_id"]) for row in scores]
    if len(set(score_ids)) != len(score_ids):
        raise ValueError("OOF score files contain duplicate source identities")
    if set(score_ids) != set(manifest):
        raise ValueError(
            "OOF score identities do not exactly cover the fold manifest"
        )
    for row in scores:
        source_id = str(row["source_sample_id"])
        expected = manifest[source_id]
        for column in ("user_id", "target_subcategory", "fold"):
            if row[column] != expected[column]:
                raise ValueError(f"{source_id}: OOF mismatch in {column}")
        if int(row["category_token_count"]) <= 0:
            raise ValueError(f"{source_id}: no category tokens were scored")
    return sorted(scores, key=lambda row: str(row["source_sample_id"]))


def build_crossfit_weights(
    view: str,
    score_files: list[Path | str],
    data_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    fold_manifest: Path | str | None = None,
    output_dir: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if view not in {"behavior", "interest"}:
        raise ValueError("view must be behavior or interest")
    data_dir = Path(data_dir)
    score_files = [Path(path) for path in score_files]
    if len(score_files) != 5:
        raise ValueError("Exactly five OOF score files are required")
    fold_manifest = (
        Path(fold_manifest)
        if fold_manifest is not None
        else data_dir / "fold_manifest.parquet"
    )
    output_dir = Path(output_dir) if output_dir is not None else data_dir
    scores = _load_and_validate_scores(score_files, fold_manifest)
    weights = compute_class_normalized_weights(scores)
    score_path = output_dir / "oof_scores" / f"{view}.parquet"
    weight_path = output_dir / "sample_weights" / f"{view}.parquet"
    summary_path = output_dir / "sample_weights" / f"{view}_statistics.json"
    existing = [path for path in (score_path, weight_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Weight outputs already exist: {existing}")
    _atomic_write_table(score_path, scores, OOF_SCORE_SCHEMA)
    _atomic_write_table(weight_path, weights, WEIGHT_SCHEMA)

    by_target: dict[str, list[float]] = defaultdict(list)
    for row in weights:
        by_target[str(row["target_subcategory"])].append(
            float(row["sample_weight"])
        )
    class_means = {
        target: sum(values) / len(values)
        for target, values in sorted(by_target.items())
    }
    ratios = {
        target: max(values) / min(values)
        for target, values in sorted(by_target.items())
    }
    nll_values = [float(row["category_token_nll"]) for row in scores]
    summary = {
        "view": view,
        "rows": len(weights),
        "classes": len(by_target),
        "fold_counts": dict(
            sorted(Counter(str(row["fold"]) for row in scores).items())
        ),
        "category_token_nll": {
            "min": min(nll_values),
            "mean": sum(nll_values) / len(nll_values),
            "max": max(nll_values),
        },
        "class_weight_mean_min": min(class_means.values()),
        "class_weight_mean_max": max(class_means.values()),
        "class_max_min_ratio_min": min(ratios.values()),
        "class_max_min_ratio_max": max(ratios.values()),
        "formula": {
            "q": (
                "one-based (rank-1)/(class_size-1), implemented as "
                "zero-based within_class_nll_rank/(class_size-1)"
            ),
            "raw_weight": "1/(1+q)",
            "sample_weight": "raw_weight / within-class mean(raw_weight)",
        },
        "input_sha256": {
            str(path): sha256_file(path) for path in score_files
        },
        "output_sha256": {
            str(score_path): sha256_file(score_path),
            str(weight_path): sha256_file(weight_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f".{summary_path.name}.",
        dir=summary_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, summary_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("behavior", "interest"), required=True)
    parser.add_argument("--score-files", type=Path, nargs=5, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_crossfit_weights(
        args.view,
        args.score_files,
        args.data_dir,
        fold_manifest=args.fold_manifest,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
