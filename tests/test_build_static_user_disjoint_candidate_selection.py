from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_dataset import DATASET_SCHEMA, VOCABULARY_SCHEMA
from new_interest.data.build_static_user_disjoint_candidate_selection import (
    build_static_user_disjoint_candidate_selection,
)
from new_interest.data.build_static_candidate_pilot import (
    build_static_candidate_pilot,
)


LABELS = [f"Topic {index:02d}" for index in range(15)]


def _history(user_id: str, labels: list[str]) -> list[dict[str, str]]:
    return [
        {
            "news_id": f"N-{user_id}-{index}",
            "category": "Category",
            "subcategory": label,
            "title": f"Full raw title {user_id} {index}",
        }
        for index, label in enumerate(labels)
    ]


def _row(
    split: str,
    impression_id: str,
    user_id: str,
    target: str,
    history: list[dict[str, str]],
) -> dict[str, object]:
    target_index = LABELS.index(target)
    return {
        "sample_id": f"{split}:{impression_id}:{target_index}",
        "split": split,
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": datetime(2019, 11, 15, 12, 0, 0),
        "history": history,
        "target_interest": {
            "category": "Category",
            "subcategory": target,
        },
        "target_news": [
            {
                "news_id": f"T-{impression_id}-{target_index}",
                "title": f"Target title {target}",
            }
        ],
    }


def _write_dataset(root: Path, *, reverse_dev: bool = False) -> Path:
    root.mkdir(parents=True)
    train = [
        _row(
            "train",
            f"train-{index}",
            f"TU{index}",
            label,
            _history(f"TU{index}", [LABELS[(index + 1) % 14]]),
        )
        for index, label in enumerate(LABELS[:14])
    ]
    dev_history_a = _history("DU1", [LABELS[10], LABELS[11]])
    dev_history_b = _history("DU2", [LABELS[12]])
    dev = [
        _row("dev", "dev-1", "DU1", LABELS[0], dev_history_a),
        _row("dev", "dev-1", "DU1", LABELS[1], dev_history_a),
        _row("dev", "dev-2", "DU1", LABELS[2], dev_history_a),
        _row("dev", "dev-3", "DU2", LABELS[3], dev_history_b),
    ]
    if reverse_dev:
        dev.reverse()
    vocabulary = [
        {
            "interest_id": index,
            "category": "Category",
            "subcategory": label,
        }
        for index, label in enumerate(LABELS)
    ]
    pq.write_table(pa.Table.from_pylist(train, schema=DATASET_SCHEMA), root / "train.parquet")
    pq.write_table(pa.Table.from_pylist(dev, schema=DATASET_SCHEMA), root / "dev.parquet")
    pq.write_table(
        pa.Table.from_pylist(vocabulary, schema=VOCABULARY_SCHEMA),
        root / "interest_vocabulary.parquet",
    )
    return root


def test_builds_paired_static_candidate_data(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"

    build_static_user_disjoint_candidate_selection(
        dataset,
        output,
        k=5,
        seed=17,
    )

    statistics = json.loads((output / "statistics.json").read_text())
    assert statistics["samples"] == 4
    assert statistics["users"] == 2
    assert statistics["impressions"] == 3
    assert statistics["train_supported_interest_count"] == 14
    assert statistics["dev_supported_interest_count"] == 4

    for strategy in ("uniform", "train_frequency_matched"):
        manifest = pq.read_table(
            output / strategy / "candidate_manifest.parquet"
        ).to_pylist()
        behavior = pq.read_table(
            output / strategy / "behavior_only/dev_expanded.parquet"
        ).to_pylist()
        interest = pq.read_table(
            output / strategy / "interest_only/dev_expanded.parquet"
        ).to_pylist()
        behavior_interest = pq.read_table(
            output / strategy / "behavior_interest/dev_expanded.parquet"
        ).to_pylist()
        assert (
            len(manifest)
            == len(behavior)
            == len(interest)
            == len(behavior_interest)
            == 4
        )
        assert [row["selection_id"] for row in behavior] == [
            row["selection_id"] for row in interest
        ] == [row["selection_id"] for row in behavior_interest]
        positions = []
        for common, behavior_row, interest_row, combined_row in zip(
            manifest, behavior, interest, behavior_interest
        ):
            candidates = common["candidates"]
            assert len(candidates) == len(set(candidates)) == 5
            assert candidates.count(common["positive"]) == 1
            assert candidates[common["correct_position"]] == common["positive"]
            assert not set(common["negative_candidates"]).intersection(
                common["history_subcategories"]
            )
            assert not set(common["negative_candidates"]).intersection(
                common["all_acceptable_subcategories"]
            )
            assert LABELS[14] not in candidates
            assert behavior_row["candidates"] == interest_row["candidates"]
            assert behavior_row["correct_position"] == interest_row["correct_position"]
            assert combined_row["candidates"] == behavior_row["candidates"]
            assert combined_row["correct_position"] == behavior_row["correct_position"]
            assert "Full raw title" in behavior_row["prompt"]
            assert "Full raw title" not in interest_row["prompt"]
            for prompt in (
                behavior_row["prompt"],
                interest_row["prompt"],
                combined_row["prompt"],
            ):
                assert "Allowed responses" not in prompt
                assert "This is a closed-set choice" not in prompt
                assert (
                    'Return only "New topic: <complete candidate name>".'
                    in prompt
                )
                for index, candidate in enumerate(candidates, start=1):
                    assert f"{index}. {candidate}" in prompt
            assert (
                'Each item is formatted as "[Topic category] Headline"'
                in combined_row["prompt"]
            )
            assert "[Topic " in combined_row["prompt"]
            assert "] Full raw title" in combined_row["prompt"]
            assert "Based on the headlines and corresponding topic categories" in (
                combined_row["prompt"]
            )
            assert "Select the single most likely new topic" in combined_row["prompt"]
            assert behavior_row["history_length"] == len(
                behavior_row["history_subcategories"]
            )
            positions.append(common["correct_position"])
        counts = [positions.count(index) for index in range(5)]
        assert max(counts) - min(counts) <= 1


def test_selection_is_independent_of_input_order(tmp_path: Path) -> None:
    first_dataset = _write_dataset(tmp_path / "first-dataset")
    second_dataset = _write_dataset(
        tmp_path / "second-dataset", reverse_dev=True
    )
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"

    build_static_user_disjoint_candidate_selection(
        first_dataset, first_output, k=5, seed=17
    )
    build_static_user_disjoint_candidate_selection(
        second_dataset, second_output, k=5, seed=17
    )

    for strategy in ("uniform", "train_frequency_matched"):
        for relative in (
            "candidate_manifest.parquet",
            "behavior_only/dev_expanded.parquet",
            "interest_only/dev_expanded.parquet",
            "behavior_interest/dev_expanded.parquet",
        ):
            assert pq.read_table(first_output / strategy / relative).equals(
                pq.read_table(second_output / strategy / relative)
            )


def test_refuses_overwrite(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    build_static_user_disjoint_candidate_selection(
        dataset, output, k=5, seed=17
    )
    with pytest.raises(FileExistsError):
        build_static_user_disjoint_candidate_selection(
            dataset, output, k=5, seed=17
        )


def test_rejects_dev_target_absent_from_train(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    rows = pq.read_table(dataset / "dev.parquet").to_pylist()
    rows[0]["target_interest"]["subcategory"] = LABELS[14]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=DATASET_SCHEMA),
        dataset / "dev.parquet",
    )
    with pytest.raises(ValueError, match="absent from Train"):
        build_static_user_disjoint_candidate_selection(
            dataset, tmp_path / "output", k=5, seed=17
        )


def test_builds_one_shared_category_covered_pilot(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    candidates = tmp_path / "candidates"
    pilot = tmp_path / "pilot"
    build_static_user_disjoint_candidate_selection(
        dataset, candidates, k=5, seed=17
    )

    build_static_candidate_pilot(
        candidates, pilot, size=4, seed=19
    )

    identities = []
    for strategy in ("uniform", "train_frequency_matched"):
        for task_type in (
            "behavior_only",
            "interest_only",
            "behavior_interest",
        ):
            table = pq.read_table(
                pilot / strategy / task_type / "dev_pilot_1000.parquet"
            )
            assert table.num_rows == 4
            assert len(set(table["answer_category"].to_pylist())) == 4
            identities.append(table["selection_id"].to_pylist())
    assert all(value == identities[0] for value in identities[1:])
