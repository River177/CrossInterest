from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEBIAS_SENTENCE = (
    "The candidate order is random and carries no information. "
    "Evaluate every candidate equally based only on the history."
)


def _variant_prompt(
    prompt: str,
    candidates: list[str],
    variant: str,
) -> str:
    instruction_marker = "\n\nSelect the one topic category"
    history, remainder = prompt.split(instruction_marker, 1)
    candidate_marker = "\n\nCandidates:\n"
    selection_text, answer_text = remainder.split(candidate_marker, 1)
    _, final_instruction = answer_text.split(
        '\n\nRespond with exactly "New topic: <category>"',
        1,
    )
    if variant == "current":
        candidate_block = "\n".join(
            f"{index}. {candidate}"
            for index, candidate in enumerate(candidates, start=1)
        )
        extra = ""
    elif variant == "debiased_numbered":
        candidate_block = "\n".join(
            f"{index}. {candidate}"
            for index, candidate in enumerate(candidates, start=1)
        )
        extra = f"\n\n{DEBIAS_SENTENCE}"
    elif variant == "debiased_bullets":
        candidate_block = "\n".join(
            f"- {candidate}" for candidate in candidates
        )
        extra = f"\n\n{DEBIAS_SENTENCE}"
    else:
        raise ValueError(variant)
    return (
        f"{history}{instruction_marker}{selection_text}{extra}"
        f"{candidate_marker}{candidate_block}\n\n"
        'Respond with exactly "New topic: <category>"'
        f"{final_instruction}"
    )


def build_variants(
    source_path: Path,
    output_path: Path,
    *,
    permutation_seed: int,
) -> Path:
    source_rows = pq.read_table(source_path).to_pylist()
    selected = [
        row
        for row in source_rows
        if int(row["permutation_seed"]) == permutation_seed
    ]
    variants = ("current", "debiased_numbered", "debiased_bullets")
    rows = []
    for variant in variants:
        for row in selected:
            rows.append(
                {
                    **row,
                    "prompt_variant": variant,
                    "variant_id": (
                        f"{row['selection_id']}:{variant}"
                    ),
                    "prompt": _variant_prompt(
                        str(row["prompt"]),
                        [str(value) for value in row["candidates"]],
                        variant,
                    ),
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
        description="Build candidate prompt variants."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/training/candidate_selection_k10/diagnostics/"
            "interest_1k_permutations.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/training/candidate_selection_k10/diagnostics/"
            "interest_1k_prompt_variants.parquet"
        ),
    )
    parser.add_argument("--permutation-seed", type=int, default=101)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        build_variants(
            args.source,
            args.output,
            permutation_seed=args.permutation_seed,
        )
    )


if __name__ == "__main__":
    main()
