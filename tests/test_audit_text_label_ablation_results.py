from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    record_input_sha256,
    sha256_adapter_semantics,
    sha256_file,
)
from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.experiments.audit_text_label_ablation_results import (
    audit_t1_training,
    audit_text_label_evaluation,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_text_label_evaluation_audit_recomputes_metrics(tmp_path: Path) -> None:
    labels = ["A", "B", "C"]
    labels_file = tmp_path / "labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"target_subcategory": label} for label in labels]
        ),
        labels_file,
    )
    records = [
        {
            "impression_id": str(index),
            "prompt": f"Prompt {index}",
            "acceptable_subcategories": ["A" if index % 2 == 0 else "B"],
        }
        for index in range(4)
    ]
    eval_file = tmp_path / "dev.parquet"
    pq.write_table(pa.Table.from_pylist(records), eval_file)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    _write_json(adapter / "adapter_config.json", {"target_modules": ["b", "a"]})

    identity = "e" * 64
    prediction_rows = []
    ranked_outputs = []
    for index, record in enumerate(records):
        correct = record["acceptable_subcategories"][0]
        other = "B" if correct == "A" else "A"
        ranked = [correct, other, "C"]
        scores = [-0.1, -1.0, -2.0]
        prediction_rows.append(
            {
                "record_index": index,
                "input_sha256": record_input_sha256(record),
                "evaluation_identity": identity,
                "impression_id": record["impression_id"],
                "ranked_labels": ranked,
                "ranked_scores": scores,
                "acceptable_subcategories": record[
                    "acceptable_subcategories"
                ],
            }
        )
        ranked_outputs.append(
            [
                {
                    "label": label,
                    "response": f"New topic: {label}",
                    "score": score,
                }
                for label, score in zip(ranked, scores)
            ]
        )

    merged = tmp_path / "merged"
    merged.mkdir()
    prediction_path = merged / "text_label_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(prediction_rows), prediction_path)
    metrics, _ = evaluate_ranked_outputs(
        records,
        ranked_outputs,
        cutoffs=(1, 2, 3),
    )
    metrics.update(
        {
            "label_vocabulary_size": 3,
            "top_k": 3,
            "cutoffs": [1, 2, 3],
            "history_prediction_filtering": False,
            "scoring": "label_token_mean_log_probability",
            "scoring_backend": "prefix_cache",
            "fixed_prefix_scored": False,
            "eos_scored": False,
            "adapter_sha256": sha256_adapter_semantics(adapter),
            "eval_sha256": sha256_file(eval_file),
            "labels_sha256": sha256_file(labels_file),
            "evaluation_identity": identity,
        }
    )
    metrics_path = merged / "metrics.json"
    _write_json(metrics_path, metrics)
    per_interest_path = merged / "per_interest_metrics.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"interest": "A", "support": 2}]),
        per_interest_path,
    )
    _write_json(
        merged / "merge_manifest.json",
        {
            "evaluation_identity": identity,
            "record_count": 4,
            "merged_prediction_sha256": sha256_file(prediction_path),
            "per_interest_metrics_sha256": sha256_file(per_interest_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )

    audit = audit_text_label_evaluation(
        merged,
        eval_file,
        labels_file,
        adapter,
        prediction_kind="likelihood",
        expected_records=4,
        expected_targets=4,
        expected_supported_interests=2,
        expected_label_count=3,
        expected_top_k=3,
        cutoffs=(1, 2, 3),
    )
    assert audit["status"] == "passed"
    assert audit["metrics"]["interest_macro_recall_at_1"] == 1.0


def test_t1_training_audit_reads_sibling_persistent_checkpoint(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    sft = job / "sft"
    adapter = sft / "final_adapter"
    checkpoint = job / "persistent_checkpoints" / "checkpoint-2"
    adapter.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    _write_json(adapter / "adapter_config.json", {})
    _write_json(
        sft / "run_summary.json",
        {
            "train_samples": 4,
            "loss_reduction": "per_example_mean",
            "sample_weight_file": None,
            "configured_epochs": 3.0,
            "world_size": 4,
            "effective_global_batch_size": 32,
            "expected_optimizer_steps": 2,
            "train_metrics": {"train_loss": 0.5, "epoch": 3.0},
        },
    )
    _write_json(
        sft / "token_audit.json",
        {"samples": 4, "over_max_seq_length": 0},
    )
    _write_json(
        sft / "model_audit.json",
        {"forbidden_trainable_parameters": []},
    )
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": 2, "max_steps": 2, "epoch": 3.0},
    )

    audit = audit_t1_training(
        sft,
        expected_train_samples=4,
        expected_optimizer_steps=2,
    )
    assert audit["status"] == "passed"
    assert audit["optimizer_steps"] == 2
