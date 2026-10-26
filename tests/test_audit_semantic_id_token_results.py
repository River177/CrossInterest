from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.evaluation.evaluate_beam import (
    record_input_sha256,
    sha256_file,
)
from new_interest.experiments.audit_semantic_id_token_results import (
    audit_semantic_id_token_condition,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    condition = root / "condition"
    sft = condition / "sft"
    adapter = sft / "final_adapter"
    checkpoints = sft / "checkpoints"
    merged = condition / "eval" / "merged"
    adapter.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    merged.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    _write_json(
        sft / "run_summary.json",
        {
            "train_samples": 4,
            "configured_epochs": 3.0,
            "world_size": 4,
            "effective_global_batch_size": 32,
            "expected_optimizer_steps": 2,
        },
    )
    _write_json(
        checkpoints / "trainer_state.json",
        {
            "global_step": 2,
            "max_steps": 2,
            "epoch": 3.0,
        },
    )
    _write_json(
        sft / "model_audit.json",
        {
            "semantic_id_tokens": {"enabled": True, "count": 3},
            "forbidden_trainable_parameters": [],
        },
    )
    _write_json(
        sft / "token_audit.json",
        {
            "samples": 4,
            "over_max_seq_length": 0,
            "max_tokens": 20,
            "max_seq_length": 32,
        },
    )
    _write_json(
        sft / "semantic_token_drift.json",
        {
            "all_input_rows_changed": True,
            "all_output_rows_changed": True,
            "changed_input_rows": 3,
            "changed_output_rows": 3,
        },
    )
    _write_json(
        sft / "semantic_adapter_state_audit.json",
        {
            "adapter_bytes": 7,
            "full_embedding_tensors": [],
            "trainable_token_tensor_count": 2,
            "trainable_token_shapes": {
                "input": [3, 8],
                "output": [3, 8],
            },
        },
    )
    _write_json(
        sft / "semantic_adapter_reload_audit.json",
        {
            "all_close": True,
            "interest_token_count": 3,
            "adapter_state_integrity": {
                "all_tensors_exact": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
            },
        },
    )

    mapping = root / "mapping.parquet"
    mapping_rows = [
        {
            "interest_index": index,
            "interest_token": f"<interest_{index:03d}>",
            "target_subcategory": label,
        }
        for index, label in enumerate(["A", "B", "C"])
    ]
    pq.write_table(pa.Table.from_pylist(mapping_rows), mapping)
    token_by_label = {
        row["target_subcategory"]: row["interest_token"]
        for row in mapping_rows
    }
    records = [
        {
            "impression_id": str(index),
            "prompt": f"Prompt {index}",
            "acceptable_subcategories": ["A" if index % 2 == 0 else "B"],
        }
        for index in range(4)
    ]
    eval_file = root / "dev.parquet"
    pq.write_table(pa.Table.from_pylist(records), eval_file)
    prediction_rows = []
    identity = "e" * 64
    for index, record in enumerate(records):
        correct = record["acceptable_subcategories"][0]
        other = "B" if correct == "A" else "A"
        labels = [correct, other, "C"]
        prediction_rows.append(
            {
                "record_index": index,
                "impression_id": record["impression_id"],
                "input_sha256": record_input_sha256(record),
                "evaluation_identity": identity,
                "ranked_labels": labels,
                "ranked_tokens": [token_by_label[label] for label in labels],
                "ranked_scores": [-0.1, -1.0, -2.0],
                "restricted_entropy": 0.5,
            }
        )
    predictions = merged / "id_token_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(prediction_rows), predictions)
    per_interest = merged / "per_interest_metrics.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"interest": "A", "support": 2}]),
        per_interest,
    )
    metrics = {
        "history_prediction_filtering": False,
        "label_vocabulary_size": 3,
        "interest_macro_recall_at_1": 1.0,
        "interest_macro_ndcg_at_1": 1.0,
        "interest_macro_recall_at_2": 1.0,
        "interest_macro_ndcg_at_2": 1.0,
        "interest_macro_recall_at_3": 1.0,
        "interest_macro_ndcg_at_3": 1.0,
    }
    metrics_path = merged / "metrics.json"
    _write_json(metrics_path, metrics)
    _write_json(
        merged / "merge_manifest.json",
        {
            "record_count": 4,
            "merged_prediction_sha256": sha256_file(predictions),
            "per_interest_metrics_sha256": sha256_file(per_interest),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return condition, eval_file, mapping


def test_independent_semantic_id_condition_audit(tmp_path: Path) -> None:
    condition, eval_file, mapping = _fixture(tmp_path)
    audit = audit_semantic_id_token_condition(
        condition,
        eval_file,
        mapping,
        expected_records=4,
        expected_targets=4,
        expected_supported_interests=2,
        expected_interest_count=3,
        expected_train_samples=4,
        expected_optimizer_steps=2,
        expected_top_k=3,
        cutoffs=(1, 2, 3),
    )
    assert audit["status"] == "passed"
    assert audit["evaluation"]["metrics"][
        "interest_macro_recall_at_1"
    ] == 1.0


def test_independent_audit_rejects_duplicate_topk(tmp_path: Path) -> None:
    condition, eval_file, mapping = _fixture(tmp_path)
    path = condition / "eval" / "merged" / "id_token_predictions.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["ranked_labels"] = ["A", "A", "C"]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="duplicate Top-3 output"):
        audit_semantic_id_token_condition(
            condition,
            eval_file,
            mapping,
            expected_records=4,
            expected_targets=4,
            expected_supported_interests=2,
            expected_interest_count=3,
            expected_train_samples=4,
            expected_optimizer_steps=2,
            expected_top_k=3,
            cutoffs=(1, 2, 3),
        )


def test_independent_audit_rejects_incomplete_training(tmp_path: Path) -> None:
    condition, eval_file, mapping = _fixture(tmp_path)
    trainer_state = condition / "sft" / "checkpoints" / "trainer_state.json"
    state = json.loads(trainer_state.read_text(encoding="utf-8"))
    state["global_step"] = 1
    _write_json(trainer_state, state)
    with pytest.raises(ValueError, match="Completed optimizer step count"):
        audit_semantic_id_token_condition(
            condition,
            eval_file,
            mapping,
            expected_records=4,
            expected_targets=4,
            expected_supported_interests=2,
            expected_interest_count=3,
            expected_train_samples=4,
            expected_optimizer_steps=2,
            expected_top_k=3,
            cutoffs=(1, 2, 3),
        )
