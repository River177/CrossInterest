from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.build_candidate_selection import _render_prompt


def _uniform(seed: int, *parts: str) -> float:
    material = ":".join((str(seed), *parts))
    value = int(hashlib.sha256(material.encode()).hexdigest(), 16)
    return (value + 1) / (2**256 + 1)


def build_frequency_matched(
    source_path: Path,
    train_path: Path,
    output_path: Path,
    *,
    sample_size: int,
    seed: int,
) -> Path:
    source_rows = pq.read_table(source_path).to_pylist()
    selected = sorted(
        source_rows,
        key=lambda row: (
            _uniform(
                seed,
                "frequency-subset",
                str(row["selection_id"]),
            ),
            str(row["selection_id"]),
        ),
    )[:sample_size]
    frequencies: Counter[str] = Counter()
    for batch in pq.ParquetFile(train_path).iter_batches(
        batch_size=8192,
        columns=["target_interest"],
    ):
        for row in batch.to_pylist():
            frequencies[
                str(row["target_interest"]["subcategory"])
            ] += 1
    labels = sorted(frequencies)

    rows = []
    for row in selected:
        selection_id = str(row["selection_id"])
        excluded = set(row["all_acceptable_subcategories"]).union(
            row["history_subcategories"]
        )
        pool = [label for label in labels if label not in excluded]
        negatives = sorted(
            pool,
            key=lambda label: (
                -math.log(
                    _uniform(
                        seed,
                        "frequency-negative",
                        selection_id,
                        label,
                    )
                )
                / frequencies[label],
                label,
            ),
        )[:9]
        candidates = [*negatives, str(row["answer_category"])]
        candidates.sort(
            key=lambda label: (
                _uniform(
                    seed,
                    "frequency-order",
                    selection_id,
                    label,
                ),
                label,
            )
        )
        answer_category = str(row["answer_category"])
        rows.append(
            {
                **row,
                "negative_sampling": "train_frequency_weighted",
                "prompt": _render_prompt(
                    str(row["prompt"]).split(
                        "\n\nSelect the one topic category",
                        1,
                    )[0],
                    candidates,
                    task_type="interest_only",
                ),
                "candidates": candidates,
                "correct_position": candidates.index(answer_category),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows),
        output_path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=1_000,
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build train-frequency-matched candidate negatives."
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
        "--train",
        type=Path,
        default=Path("data/processed/new_interest/train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/training/candidate_selection_k10/diagnostics/"
            "interest_1k_frequency_matched.parquet"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        build_frequency_matched(
            args.source,
            args.train,
            args.output,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
