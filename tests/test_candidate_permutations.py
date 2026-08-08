from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.build_candidate_permutations import build_permutations


def test_permutations_preserve_candidate_set_and_answer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    candidates = [f"Topic {index}" for index in range(10)]
    prompt = (
        "History\n\nCandidates:\n"
        + "\n".join(
            f"{index}. {candidate}"
            for index, candidate in enumerate(candidates, start=1)
        )
        + '\n\nRespond with exactly "New topic: <category>".'
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "selection_id": "s1",
                    "prompt": prompt,
                    "candidates": candidates,
                    "answer_category": "Topic 4",
                    "answer": "New topic: Topic 4",
                    "correct_position": 4,
                }
            ]
        ),
        source,
    )

    build_permutations(
        source,
        output,
        sample_size=1,
        permutation_seeds=(1, 2),
        selection_seed=3,
    )
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 2
    for row in rows:
        assert set(row["candidates"]) == set(candidates)
        assert row["candidates"][row["correct_position"]] == "Topic 4"
        assert row["answer"] == "New topic: Topic 4"
        assert f"{row['correct_position'] + 1}. Topic 4" in row["prompt"]
