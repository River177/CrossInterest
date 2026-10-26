from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from new_interest.evaluation.evaluate_beam import (
    canonical_sha256,
    record_input_sha256,
    sha256_file,
)
from new_interest.evaluation.evaluate_id_token_ranking import (
    EVALUATION_IDENTITY_VERSION,
    rank_restricted_logits,
)
from new_interest.evaluation.merge_id_token_shards import (
    merge_id_token_shards,
)


def test_rank_restricted_logits_ignores_non_interest_vocabulary() -> None:
    logits = torch.tensor(
        [
            [100.0, 1.0, 3.0, 2.0, 90.0],
            [99.0, 4.0, 2.0, 7.0, 88.0],
        ]
    )
    ranked, entropies = rank_restricted_logits(
        logits,
        [1, 2, 3],
        ["A", "B", "C"],
        ["<interest_000>", "<interest_001>", "<interest_002>"],
        top_k=2,
    )
    assert [row["label"] for row in ranked[0]] == ["B", "C"]
    assert [row["label"] for row in ranked[1]] == ["C", "A"]
    assert all(len(row) == 2 for row in ranked)
    assert all(value > 0 for value in entropies)
    probabilities = [
        sum(torch.exp(torch.tensor(item["score"])) for item in row)
        for row in ranked
    ]
    assert all(value < 1.0 for value in probabilities)


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


def _write_fixture(root: Path):
    eval_file = root / "dev.parquet"
    mapping_file = root / "mapping.parquet"
    records = _records()
    pq.write_table(pa.Table.from_pylist(records), eval_file)
    mapping_rows = [
        {
            "interest_index": index,
            "interest_token": f"<interest_{index:03d}>",
            "target_subcategory": label,
        }
        for index, label in enumerate(["A", "B", "C"])
    ]
    pq.write_table(pa.Table.from_pylist(mapping_rows), mapping_file)
    labels = ["A", "B", "C"]
    tokens = [row["interest_token"] for row in mapping_rows]
    identity_payload = {
        "identity_version": EVALUATION_IDENTITY_VERSION,
        "model_name": "test-model",
        "adapter_sha256": "a" * 64,
        "eval_sha256": sha256_file(eval_file),
        "mapping_sha256": sha256_file(mapping_file),
        "full_record_count": 4,
        "labels": labels,
        "tokens": tokens,
        "token_ids": [100, 101, 102],
        "max_examples": None,
        "num_shards": 2,
        "top_k": 2,
        "cutoffs": [1, 2],
        "enable_thinking": False,
    }
    identity = canonical_sha256(identity_payload)
    shard_dirs = []
    for shard_index in range(2):
        shard_dir = root / f"shard-{shard_index}"
        shard_dir.mkdir()
        shard_dirs.append(shard_dir)
        rows = []
        for record_index, record in enumerate(records):
            if record_index % 2 != shard_index:
                continue
            correct = record["acceptable_subcategories"][0]
            other = "B" if correct == "A" else "A"
            ranked_labels = [correct, other]
            ranked_tokens = [
                tokens[labels.index(label)] for label in ranked_labels
            ]
            rows.append(
                {
                    "record_index": record_index,
                    "impression_id": record["impression_id"],
                    "input_sha256": record_input_sha256(record),
                    "evaluation_identity": identity,
                    "ranked_labels": ranked_labels,
                    "ranked_tokens": ranked_tokens,
                    "ranked_scores": [-0.1, -2.0],
                    "restricted_entropy": 0.5,
                }
            )
        prediction_path = shard_dir / "id_token_predictions.parquet"
        pq.write_table(pa.Table.from_pylist(rows), prediction_path)
        manifest = {
            **identity_payload,
            "evaluation_identity": identity,
            "shard_index": shard_index,
            "shard_record_count": len(rows),
            "prediction_sha256": sha256_file(prediction_path),
            "elapsed_seconds": 2.0,
            "generated_samples": len(rows),
            "resumed_samples": 0,
        }
        (shard_dir / "shard_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return eval_file, mapping_file, shard_dirs


def _refresh(shard_dir: Path) -> None:
    predictions = shard_dir / "id_token_predictions.parquet"
    manifest_path = shard_dir / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = sha256_file(predictions)
    manifest["shard_record_count"] = pq.read_metadata(predictions).num_rows
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_merge_id_token_shards_restores_order(tmp_path: Path) -> None:
    eval_file, mapping_file, shard_dirs = _write_fixture(tmp_path)
    output = tmp_path / "merged"
    metrics = merge_id_token_shards(
        list(reversed(shard_dirs)),
        eval_file,
        mapping_file,
        output,
        expected_records=4,
    )
    assert metrics["samples"] == 4
    assert metrics["interest_macro_recall_at_1"] == 1.0
    assert metrics["interest_macro_ndcg_at_1"] == 1.0
    assert metrics["history_prediction_filtering"] is False
    assert metrics["label_vocabulary_size"] == 3
    rows = pq.read_table(
        output / "id_token_predictions.parquet"
    ).to_pylist()
    assert [row["record_index"] for row in rows] == [0, 1, 2, 3]


def test_merge_id_token_shards_rejects_missing_record(tmp_path: Path) -> None:
    eval_file, mapping_file, shard_dirs = _write_fixture(tmp_path)
    path = shard_dirs[0] / "id_token_predictions.parquet"
    rows = pq.read_table(path).to_pylist()[:-1]
    pq.write_table(pa.Table.from_pylist(rows), path)
    _refresh(shard_dirs[0])
    with pytest.raises(ValueError, match="Missing 1 record"):
        merge_id_token_shards(
            shard_dirs,
            eval_file,
            mapping_file,
            tmp_path / "merged",
        )


def test_merge_id_token_shards_rejects_mapping_mismatch(tmp_path: Path) -> None:
    eval_file, mapping_file, shard_dirs = _write_fixture(tmp_path)
    path = shard_dirs[0] / "id_token_predictions.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["ranked_tokens"] = list(reversed(rows[0]["ranked_tokens"]))
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    _refresh(shard_dirs[0])
    with pytest.raises(ValueError, match="token/label mapping mismatch"):
        merge_id_token_shards(
            shard_dirs,
            eval_file,
            mapping_file,
            tmp_path / "merged",
        )
