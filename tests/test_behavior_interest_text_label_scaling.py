from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_behavior_interest_text_label_scaling_dataset import (
    build_behavior_interest_text_label_scaling_dataset,
)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    train_dir = source / "train" / "behavior_interest"
    eval_dir = source / "eval" / "behavior_interest"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    labels = ("Alpha", "Beta")
    rows = []
    manifest = []
    global_rank = 0
    for within_rank in range(2):
        for label in labels:
            source_id = f"{label}-{within_rank}"
            response = f"New topic: {label}"
            prompt = f"history for {source_id}"
            rows.append(
                {
                    "source_sample_id": source_id,
                    "user_id": f"user-{source_id}",
                    "target_subcategory": label,
                    "task_type": "behavior_interest",
                    "prompt": prompt,
                    "response": response,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                }
            )
            manifest.append(
                {
                    "source_sample_id": source_id,
                    "target_subcategory": label,
                    "within_class_rank": within_rank,
                    "global_rank": global_rank,
                }
            )
            global_rank += 1
    pq.write_table(pa.Table.from_pylist(rows), train_dir / "21600.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"impression_id": "dev-1"}]),
        eval_dir / "dev_full_grouped.parquet",
    )
    manifest_path = root / "subset_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest), manifest_path)
    return source, manifest_path


def test_builds_nested_balanced_text_label_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source, manifest = _write_inputs(tmp_path)
    output = tmp_path / "output"
    result = build_behavior_interest_text_label_scaling_dataset(
        source,
        manifest,
        output,
        train_sizes=(2, 4),
        expected_interests=2,
    )

    small = pq.read_table(
        output / "train/behavior_interest/2.parquet"
    ).to_pylist()
    large = pq.read_table(
        output / "train/behavior_interest/4.parquet"
    ).to_pylist()
    assert result.train_sizes == (2, 4)
    assert {row["source_sample_id"] for row in small}.issubset(
        {row["source_sample_id"] for row in large}
    )
    assert all(
        row["response"] == f"New topic: {row['target_subcategory']}"
        for row in large
    )
    assert (output / "eval/behavior_interest/dev_full_grouped.parquet").is_file()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["output_type"] == "text_label"
    assert metadata["train_sizes"] == [2, 4]
    assert metadata["selection_rule"] == (
        "within_class_rank < samples_per_interest"
    )
    assert not Path(metadata["source_dir"]).is_absolute()


def test_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    source, manifest = _write_inputs(tmp_path)
    output = tmp_path / "output"
    build_behavior_interest_text_label_scaling_dataset(
        source,
        manifest,
        output,
        train_sizes=(2,),
        expected_interests=2,
    )
    with pytest.raises(FileExistsError):
        build_behavior_interest_text_label_scaling_dataset(
            source,
            manifest,
            output,
            train_sizes=(2,),
            expected_interests=2,
        )


def test_rejects_semantic_id_response(tmp_path: Path) -> None:
    source, manifest = _write_inputs(tmp_path)
    path = source / "train/behavior_interest/21600.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["response"] = "<interest_000>"
    rows[0]["messages"][1]["content"] = "<interest_000>"
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="not its Text-label"):
        build_behavior_interest_text_label_scaling_dataset(
            source,
            manifest,
            tmp_path / "output",
            train_sizes=(2,),
            expected_interests=2,
        )
