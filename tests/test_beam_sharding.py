from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.evaluation.evaluate_beam import (
    _atomic_write_parquet,
    canonical_sha256,
    record_input_sha256,
    select_logical_shard,
    sha256_adapter_semantics,
    sha256_file,
)
from new_interest.evaluation.merge_beam_shards import merge_beam_shards


def _records() -> list[dict[str, object]]:
    return [
        {
            "impression_id": str(index),
            "user_id": f"U{index}",
            "prompt": f"Prompt {index}",
            "acceptable_subcategories": ["A" if index % 2 == 0 else "B"],
        }
        for index in range(4)
    ]


def _write_fixture(root: Path) -> tuple[Path, Path, list[Path]]:
    eval_file = root / "dev.parquet"
    labels_file = root / "labels.parquet"
    records = _records()
    pq.write_table(pa.Table.from_pylist(records), eval_file)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"target_subcategory": "A"},
                {"target_subcategory": "B"},
                {"target_subcategory": "C"},
            ]
        ),
        labels_file,
    )
    labels = ["A", "B", "C"]
    identity_payload = {
        "identity_version": "test-v1",
        "model_name": "test-model",
        "adapter_sha256": "a" * 64,
        "eval_sha256": sha256_file(eval_file),
        "labels_sha256": sha256_file(labels_file),
        "full_record_count": len(records),
        "label_count": len(labels),
        "labels": labels,
        "max_examples": None,
        "num_shards": 2,
        "num_beams": 2,
        "top_k": 2,
        "cutoffs": [1, 2],
        "max_new_tokens": 8,
        "enable_thinking": False,
    }
    evaluation_identity = canonical_sha256(identity_payload)
    shard_dirs = []
    for shard_index in range(2):
        shard_dir = root / f"shard-{shard_index:02d}"
        shard_dir.mkdir()
        shard_dirs.append(shard_dir)
        rows = []
        for record_index, record in enumerate(records):
            if record_index % 2 != shard_index:
                continue
            correct = str(record["acceptable_subcategories"][0])
            other = "B" if correct == "A" else "A"
            ranked_labels = [correct, other]
            rows.append(
                {
                    "record_index": record_index,
                    "input_sha256": record_input_sha256(record),
                    "evaluation_identity": evaluation_identity,
                    "impression_id": str(record["impression_id"]),
                    "ranked_labels": ranked_labels,
                    "ranked_responses": [
                        f"New topic: {label}" for label in ranked_labels
                    ],
                    "ranked_scores": [0.9, 0.1],
                    "dedup_retry_rounds": 0,
                    "initial_unique_label_count": 2,
                }
            )
        prediction_path = shard_dir / "beam_predictions.parquet"
        pq.write_table(pa.Table.from_pylist(rows), prediction_path)
        manifest = {
            **identity_payload,
            "evaluation_identity": evaluation_identity,
            "shard_index": shard_index,
            "shard_record_count": len(rows),
            "first_record_index": rows[0]["record_index"],
            "last_record_index": rows[-1]["record_index"],
            "prediction_sha256": sha256_file(prediction_path),
        }
        (shard_dir / "shard_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    return eval_file, labels_file, shard_dirs


def _refresh_manifest(shard_dir: Path) -> None:
    prediction_path = shard_dir / "beam_predictions.parquet"
    manifest_path = shard_dir / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = sha256_file(prediction_path)
    manifest["shard_record_count"] = pq.read_metadata(
        prediction_path
    ).num_rows
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_logical_shards_cover_global_order() -> None:
    records = _records()
    first_indices, first = select_logical_shard(
        records,
        num_shards=3,
        shard_index=0,
    )
    second_indices, second = select_logical_shard(
        records,
        num_shards=3,
        shard_index=1,
    )
    third_indices, third = select_logical_shard(
        records,
        num_shards=3,
        shard_index=2,
    )
    assert first_indices == [0, 3]
    assert second_indices == [1]
    assert third_indices == [2]
    combined = {
        index: record
        for indices, shard in (
            (first_indices, first),
            (second_indices, second),
            (third_indices, third),
        )
        for index, record in zip(indices, shard)
    }
    assert [combined[index] for index in sorted(combined)] == records


def test_adapter_semantic_hash_ignores_resave_only_changes(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"stable weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-9B",
                "r": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (adapter / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}),
        encoding="utf-8",
    )
    (adapter / "tokenizer_config.json").write_text(
        json.dumps({"padding_side": "left"}),
        encoding="utf-8",
    )
    (adapter / "chat_template.jinja").write_text(
        "{{ messages }}\n",
        encoding="utf-8",
    )
    (adapter / "README.md").write_text("first generated card\n")
    (adapter / "training_args.bin").write_bytes(b"first trainer state")

    before = sha256_adapter_semantics(adapter)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": ["v_proj", "q_proj", "k_proj"],
                "r": 16,
                "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (adapter / "README.md").write_text("second generated card\n")
    (adapter / "training_args.bin").write_bytes(b"second trainer state")

    assert sha256_adapter_semantics(adapter) == before
    (adapter / "adapter_model.safetensors").write_bytes(b"changed weights")
    assert sha256_adapter_semantics(adapter) != before


def test_parquet_publish_retries_without_committing_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import new_interest.evaluation.evaluate_beam as evaluate_beam_module

    destination = tmp_path / "mounted" / "checkpoint.parquet"
    staging_dir = tmp_path / "scratch"
    real_copy = evaluate_beam_module.shutil.copyfileobj
    attempts = 0
    target_absent_during_failure = False

    def flaky_copy(source, target, *, length):
        nonlocal attempts, target_absent_during_failure
        attempts += 1
        if attempts == 1:
            target.write(source.read(7))
            target.flush()
            target_absent_during_failure = not destination.exists()
            raise OSError("error closing file")
        return real_copy(source, target, length=length)

    monkeypatch.setattr(
        evaluate_beam_module.shutil,
        "copyfileobj",
        flaky_copy,
    )
    rows = [{"record_index": 0, "value": "complete"}]
    _atomic_write_parquet(
        destination,
        rows,
        staging_dir=staging_dir,
        publish_attempts=2,
        retry_seconds=0.0,
    )

    assert attempts == 2
    assert target_absent_during_failure
    assert pq.read_table(destination).to_pylist() == rows
    assert not list(destination.parent.glob(".*.partial-*"))
    assert not list(staging_dir.glob("*.parquet"))


def test_merge_beam_shards_validates_and_restores_order(tmp_path: Path) -> None:
    eval_file, labels_file, shard_dirs = _write_fixture(tmp_path)
    output_dir = tmp_path / "merged"
    metrics = merge_beam_shards(
        list(reversed(shard_dirs)),
        eval_file,
        labels_file,
        output_dir,
        expected_records=4,
    )
    assert metrics["samples"] == 4
    assert metrics["supported_interest_count"] == 2
    assert metrics["interest_macro_recall_at_1"] == 1.0
    assert metrics["interest_macro_ndcg_at_1"] == 1.0
    assert metrics["dedup_retry_example_count"] == 0
    assert metrics["dedup_retry_round_count"] == 0
    assert metrics["minimum_initial_unique_label_count"] == 2
    predictions = pq.read_table(
        output_dir / "beam_predictions.parquet"
    ).to_pylist()
    assert [row["record_index"] for row in predictions] == [0, 1, 2, 3]
    per_interest = pq.read_table(
        output_dir / "per_interest_metrics.parquet"
    ).to_pylist()
    assert [row["interest"] for row in per_interest] == ["A", "B"]


def test_merge_beam_shards_rejects_missing_record(tmp_path: Path) -> None:
    eval_file, labels_file, shard_dirs = _write_fixture(tmp_path)
    prediction_path = shard_dirs[0] / "beam_predictions.parquet"
    rows = pq.read_table(prediction_path).to_pylist()[:-1]
    pq.write_table(pa.Table.from_pylist(rows), prediction_path)
    _refresh_manifest(shard_dirs[0])
    with pytest.raises(ValueError, match="Missing 1 record"):
        merge_beam_shards(
            shard_dirs,
            eval_file,
            labels_file,
            tmp_path / "merged",
        )


def test_merge_beam_shards_rejects_duplicate_record(tmp_path: Path) -> None:
    eval_file, labels_file, shard_dirs = _write_fixture(tmp_path)
    prediction_path = shard_dirs[0] / "beam_predictions.parquet"
    rows = pq.read_table(prediction_path).to_pylist()
    rows.append(rows[0])
    pq.write_table(pa.Table.from_pylist(rows), prediction_path)
    _refresh_manifest(shard_dirs[0])
    with pytest.raises(ValueError, match="Duplicate record index 0"):
        merge_beam_shards(
            shard_dirs,
            eval_file,
            labels_file,
            tmp_path / "merged",
        )


def test_merge_beam_shards_rejects_identity_mismatch(tmp_path: Path) -> None:
    eval_file, labels_file, shard_dirs = _write_fixture(tmp_path)
    manifest_path = shard_dirs[1] / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_identity"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        merge_beam_shards(
            shard_dirs,
            eval_file,
            labels_file,
            tmp_path / "merged",
        )
