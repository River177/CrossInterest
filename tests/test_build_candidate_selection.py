from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.build_candidate_selection import (
    build_candidate_selection,
)
from new_interest.build_generative_dataset import RETAINED_INTEREST_SCHEMA
from new_interest.build_sft_subsets import GROUPED_EVAL_SCHEMA


def _grouped_row(
    impression_id: str,
    prompt: str,
    acceptable: list[str],
    history: list[str],
) -> dict[str, object]:
    return {
        "impression_id": impression_id,
        "user_id": f"U{impression_id}",
        "timestamp": datetime(2019, 11, 15, 12, 0, 0),
        "history_length": len(history),
        "prompt": prompt,
        "acceptable_responses": [
            f"New topic: {target}" for target in acceptable
        ],
        "acceptable_subcategories": acceptable,
        "history_subcategories": history,
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    sft_dir = root / "sft"
    retained = root / "retained.parquet"
    labels = [
        "History A",
        "History B",
        "Target A",
        "Target B",
        "Target C",
        "Negative A",
        "Negative B",
        "Negative C",
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "target_subcategory": label,
                    "source_train_count": 200,
                    "selected_train_count": 200,
                    "filtered_dev_count": 1,
                }
                for label in labels
            ],
            schema=RETAINED_INTEREST_SCHEMA,
        ),
        retained,
    )

    specifications = [
        ("1", ["Target A"], ["History A"]),
        ("2", ["Target B", "Target C"], ["History A", "History B"]),
        ("3", ["Target A"], ["History B"]),
    ]
    behavior_rows = []
    interest_rows = []
    for impression_id, acceptable, history in specifications:
        behavior_history = "\n".join(
            f"{index}. Headline {impression_id}-{index}"
            for index in range(1, len(history) + 1)
        )
        interest_history = "\n".join(
            f"{index}. {value}"
            for index, value in enumerate(history, start=1)
        )
        behavior_rows.append(
            _grouped_row(
                impression_id,
                (
                    "The headlines of the news items I recently viewed are "
                    f"listed below:\n\n{behavior_history}\n\n"
                    "Based on these headlines, old instruction."
                ),
                acceptable,
                history,
            )
        )
        interest_rows.append(
            _grouped_row(
                impression_id,
                (
                    "The topic categories associated with the news items I "
                    "recently viewed, in chronological order, are listed "
                    f"below:\n\n{interest_history}\n\n"
                    "Based on this sequence, old instruction."
                ),
                acceptable,
                history,
            )
        )

    for task_type, rows in (
        ("behavior_only", behavior_rows),
        ("interest_only", interest_rows),
    ):
        path = sft_dir / "eval" / task_type / "dev_10k_grouped.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=GROUPED_EVAL_SCHEMA),
            path,
        )
    return sft_dir, retained


def test_builds_paired_candidate_data(tmp_path: Path) -> None:
    sft_dir, retained = _write_fixture(tmp_path)
    output = tmp_path / "output"

    build_candidate_selection(
        sft_dir,
        retained,
        output,
        k=3,
        seed=17,
    )

    manifest = pq.read_table(
        output / "candidate_manifest.parquet"
    ).to_pylist()
    behavior = pq.read_table(
        output / "behavior_only/dev_10k_expanded.parquet"
    ).to_pylist()
    interest = pq.read_table(
        output / "interest_only/dev_10k_expanded.parquet"
    ).to_pylist()

    assert len(manifest) == len(behavior) == len(interest) == 4
    assert [row["selection_id"] for row in behavior] == [
        row["selection_id"] for row in interest
    ]
    assert sum(row["impression_id"] == "2" for row in manifest) == 2

    position_counts = {}
    for common, behavior_row, interest_row in zip(
        manifest,
        behavior,
        interest,
    ):
        candidates = common["candidates"]
        positive = common["positive"]
        negatives = common["negative_candidates"]
        history = set(common["history_subcategories"])
        acceptable = set(common["all_acceptable_subcategories"])

        assert len(candidates) == len(set(candidates)) == 3
        assert candidates.count(positive) == 1
        assert candidates[common["correct_position"]] == positive
        assert len(negatives) == 2
        assert not set(negatives).intersection(history)
        assert not set(negatives).intersection(acceptable)
        assert behavior_row["candidates"] == interest_row["candidates"]
        assert (
            behavior_row["answer_category"]
            == interest_row["answer_category"]
            == positive
        )
        assert (
            behavior_row["answer"]
            == interest_row["answer"]
            == f"New topic: {positive}"
        )
        assert (
            behavior_row["correct_position"]
            == interest_row["correct_position"]
        )
        assert "Headline" in behavior_row["prompt"]
        assert "History A" not in behavior_row["prompt"]
        assert "Headline" not in interest_row["prompt"]
        assert 'Respond with exactly "New topic: <category>"' in (
            behavior_row["prompt"]
        )
        position_counts[common["correct_position"]] = (
            position_counts.get(common["correct_position"], 0) + 1
        )

    assert max(position_counts.values()) - min(position_counts.values()) <= 1


def test_candidate_build_is_deterministic(tmp_path: Path) -> None:
    sft_dir, retained = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_candidate_selection(
        sft_dir,
        retained,
        first,
        k=3,
        seed=17,
    )
    build_candidate_selection(
        sft_dir,
        retained,
        second,
        k=3,
        seed=17,
    )

    for relative in (
        "candidate_manifest.parquet",
        "behavior_only/dev_10k_expanded.parquet",
        "interest_only/dev_10k_expanded.parquet",
    ):
        assert pq.read_table(first / relative).equals(
            pq.read_table(second / relative)
        )
