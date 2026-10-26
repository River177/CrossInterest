from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_mera_relations import build_mera_relations


def _write_mapping(path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "interest_index": 0,
                    "interest_token": "<interest_000>",
                    "target_subcategory": "Alpha",
                },
                {
                    "interest_index": 1,
                    "interest_token": "<interest_001>",
                    "target_subcategory": "Beta",
                },
                {
                    "interest_index": 2,
                    "interest_token": "<interest_002>",
                    "target_subcategory": "Gamma",
                },
            ]
        ),
        path,
    )


def _event(category: str, subcategory: str) -> dict[str, str]:
    return {
        "news_id": f"news-{category}-{subcategory}",
        "category": category,
        "subcategory": subcategory,
        "title": subcategory,
    }


def _write_source(path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "sample-1",
                    "user_id": "user-1",
                    "history": [
                        _event("Parent One", "Alpha"),
                        _event("Parent One", "Alpha"),
                        _event("Parent Two", "Gamma"),
                    ],
                    "target_interest": {
                        "category": "Parent One",
                        "subcategory": "Beta",
                    },
                },
                {
                    "sample_id": "sample-2",
                    "user_id": "user-2",
                    "history": [
                        _event("Parent One", "Alpha"),
                        _event("Input Only", "Outside Output Vocabulary"),
                    ],
                    "target_interest": {
                        "category": "Parent Two",
                        "subcategory": "Gamma",
                    },
                },
                {
                    "sample_id": "sample-3",
                    "user_id": "user-3",
                    "history": [_event("Parent One", "Beta")],
                    "target_interest": {
                        "category": "Parent One",
                        "subcategory": "Alpha",
                    },
                },
            ]
        ),
        path,
    )


def test_build_mera_relations_matches_released_cooccurrence_contract(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.parquet"
    source = tmp_path / "train.parquet"
    output = tmp_path / "relations"
    _write_mapping(mapping)
    _write_source(source)

    result = build_mera_relations(source, mapping, output, batch_size=1)

    assert result.rows == 3
    assert result.interests == 3
    assert result.parent_groups == 2
    payload = json.loads((output / "relations.json").read_text())
    assert payload["labels"] == ["Alpha", "Beta", "Gamma"]
    assert payload["parents"] == ["Parent One", "Parent One", "Parent Two"]
    # Row 1 contributes the full 3x3 set. Row 2 adds Alpha/Gamma and
    # row 3 adds Alpha/Beta. The repeated Alpha in row 1 is deduplicated.
    assert payload["cooccurrence_counts"] == [
        [3, 2, 2],
        [2, 2, 1],
        [2, 1, 2],
    ]
    statistics = json.loads((output / "statistics.json").read_text())
    assert statistics["users"] == 3
    assert statistics["structure_eligible_anchors"] == 2
    assert statistics["history_only_interest_occurrences_ignored"] == 1
    assert [
        row["threshold"]
        for row in statistics["cooccurrence_threshold_coverage"]
    ] == [0.0, 0.01, 0.02, 0.05]
    metadata = json.loads((output / "metadata.json").read_text())
    assert set(metadata["artifact_sha256"]) == {
        "relations.json",
        "statistics.json",
    }

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        build_mera_relations(source, mapping, output)


def test_build_mera_relations_rejects_ambiguous_parents(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.parquet"
    source = tmp_path / "train.parquet"
    output = tmp_path / "relations"
    _write_mapping(mapping)
    _write_source(source)
    table = pq.read_table(source).to_pylist()
    table.append(
        {
            "sample_id": "sample-4",
            "user_id": "user-4",
            "history": [_event("Parent Two", "Gamma")],
            "target_interest": {
                "category": "Wrong Parent",
                "subcategory": "Alpha",
            },
        }
    )
    pq.write_table(pa.Table.from_pylist(table), source)

    with pytest.raises(ValueError, match="unique parent"):
        build_mera_relations(source, mapping, output)
