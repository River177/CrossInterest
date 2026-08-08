from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _priority(seed: int, namespace: str, *values: str) -> str:
    material = ":".join((str(seed), namespace, *values))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _replace_candidates(prompt: str, candidates: list[str]) -> str:
    start_marker = "\n\nCandidates:\n"
    end_marker = "\n\nRespond with exactly "
    prefix, remainder = prompt.split(start_marker, 1)
    _, suffix = remainder.split(end_marker, 1)
    block = "\n".join(
        f"{index}. {candidate}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return (
        f"{prefix}{start_marker}{block}{end_marker}{suffix}"
    )


def build_permutations(
    source_path: Path,
    output_path: Path,
    *,
    sample_size: int,
    permutation_seeds: tuple[int, ...],
    selection_seed: int,
) -> Path:
    rows = pq.read_table(source_path).to_pylist()
    selected = sorted(
        rows,
        key=lambda row: (
            _priority(
                selection_seed,
                "permutation-subset",
                str(row["selection_id"]),
            ),
            str(row["selection_id"]),
        ),
    )[:sample_size]
    output_rows = []
    for permutation_seed in permutation_seeds:
        for row in selected:
            candidates = sorted(
                [str(value) for value in row["candidates"]],
                key=lambda candidate: (
                    _priority(
                        permutation_seed,
                        "candidate-permutation",
                        str(row["selection_id"]),
                        candidate,
                    ),
                    candidate,
                ),
            )
            answer_category = str(row["answer_category"])
            output_rows.append(
                {
                    **row,
                    "permutation_seed": permutation_seed,
                    "permutation_id": (
                        f"{row['selection_id']}:p{permutation_seed}"
                    ),
                    "prompt": _replace_candidates(
                        str(row["prompt"]),
                        candidates,
                    ),
                    "candidates": candidates,
                    "correct_position": candidates.index(
                        answer_category
                    ),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        output_path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=1_000,
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build candidate-order permutation diagnostics."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/training/candidate_selection_k10/"
            "interest_only/dev_10k_expanded.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/training/candidate_selection_k10/"
            "diagnostics/interest_1k_permutations.parquet"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument(
        "--permutation-seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505],
    )
    parser.add_argument("--selection-seed", type=int, default=20260806)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        build_permutations(
            args.source,
            args.output,
            sample_size=args.sample_size,
            permutation_seeds=tuple(args.permutation_seeds),
            selection_seed=args.selection_seed,
        )
    )


if __name__ == "__main__":
    main()
