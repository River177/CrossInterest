from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.data.build_wang_stage_a_dataset import (
    RESPONSE_PREFIX,
    DatasetSource,
    Transition,
    build_dataset,
    most_recent_unique_clusters,
    select_balanced_transitions,
    select_random_transitions,
)


def _transition(
    source_id: str,
    context: tuple[str, str],
    target: str,
    *,
    history: tuple[str, ...] | None = None,
) -> Transition:
    return Transition(
        source_id=source_id,
        user_id=f"user-{source_id}",
        history=history or context,
        context=context,
        targets=(target,),
    )


def test_most_recent_unique_clusters_preserves_recent_chronology() -> None:
    assert most_recent_unique_clusters(["A", "B", "A", "C", "C"]) == (
        "A",
        "C",
    )
    assert most_recent_unique_clusters(["A", "A"]) == ("A",)


def test_balanced_selection_uses_frequency_then_lexical_tie_break() -> None:
    rows = [
        _transition("0", ("B", "C"), "T1"),
        _transition("1", ("B", "C"), "T1"),
        _transition("2", ("A", "C"), "T1"),
        _transition("3", ("A", "B"), "T1"),
        _transition("4", ("C", "D"), "T2"),
    ]
    selected, supported = select_balanced_transitions(
        rows,
        pairs_per_label=2,
    )
    assert supported == ("T1",)
    assert [row.context for row, _ in selected] == [
        ("B", "C"),
        ("A", "B"),
    ]
    assert [frequency for _, frequency in selected] == [2, 1]


def test_random_control_is_deterministic_and_support_matched() -> None:
    rows = [
        _transition(str(index), (f"A{index}", "B"), target)
        for index, target in enumerate(("T1", "T1", "T2", "T3", "T2"))
    ]
    first = select_random_transitions(
        rows,
        supported_labels={"T1", "T2"},
        sample_count=3,
        seed=2024,
    )
    second = select_random_transitions(
        rows,
        supported_labels={"T1", "T2"},
        sample_count=3,
        seed=2024,
    )
    assert [row.source_id for row, _ in first] == [
        row.source_id for row, _ in second
    ]
    assert {row.targets[0] for row, _ in first} <= {"T1", "T2"}


def test_build_dataset_writes_balanced_random_and_supported_views(
    tmp_path: Path,
) -> None:
    train = (
        _transition("0", ("A", "B"), "T1"),
        _transition("1", ("A", "C"), "T1"),
        _transition("2", ("B", "C"), "T1"),
        _transition("3", ("A", "B"), "T2"),
        _transition("4", ("A", "C"), "T2"),
        _transition("5", ("B", "C"), "T2"),
        _transition("6", ("A", "D"), "T3"),
    )
    evaluation = (
        Transition(
            source_id="eval-0",
            user_id="u0",
            history=("A", "B"),
            context=("A", "B"),
            targets=("T1", "T3"),
        ),
        Transition(
            source_id="eval-1",
            user_id="u1",
            history=("B", "C"),
            context=("B", "C"),
            targets=("T3",),
        ),
    )
    source_marker = tmp_path / "source.txt"
    source_marker.write_text("frozen", encoding="utf-8")
    source = DatasetSource(
        name="toy",
        domain="toy topics",
        train=train,
        evaluations={"test": evaluation},
        labels=("T1", "T2", "T3"),
        source_files=(source_marker,),
    )
    output = tmp_path / "output"
    result = build_dataset(source, output, pairs_per_label=2, random_seed=7)

    assert result.balanced_rows == 4
    assert result.random_rows == 4
    assert result.supported_labels == ("T1", "T2")

    balanced = pq.read_table(output / "train/balanced.parquet").to_pylist()
    assert len(balanced) == 4
    assert {row["target_subcategory"] for row in balanced} == {"T1", "T2"}
    assert all(
        row["response"]
        == f"{RESPONSE_PREFIX}{row['target_subcategory']}"
        for row in balanced
    )
    assert all(row["messages"][0]["content"] == row["prompt"] for row in balanced)

    full_eval = pq.read_table(output / "eval/test_full.parquet").to_pylist()
    supported_eval = pq.read_table(
        output / "eval/test_supported.parquet"
    ).to_pylist()
    assert len(full_eval) == 2
    assert len(supported_eval) == 1
    assert supported_eval[0]["acceptable_subcategories"] == ["T1"]

    statistics = json.loads((output / "statistics.json").read_text())
    assert statistics["underfilled_labels"] == {"T3": 1}
    assert statistics["balanced"]["min_per_label"] == 2
    assert statistics["balanced"]["max_per_label"] == 2
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["rule_version"].startswith("wang-stage-a")
    assert len(metadata["selection_identity"]) == 64
