from __future__ import annotations

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    canonical_sha256,
    record_input_sha256,
    sha256_file,
)

from new_interest.evaluation.evaluate_wang_stage_a_generation import (
    evaluate_generations,
    generation_stop_token_ids,
    parse_cluster_generation,
)
from new_interest.evaluation.merge_wang_stage_a_generation_shards import (
    merge_generation_shards,
)


def _record(
    impression_id: str,
    targets: list[str],
    history: list[str],
) -> dict[str, object]:
    return {
        "impression_id": impression_id,
        "prompt": "prompt",
        "acceptable_subcategories": targets,
        "acceptable_responses": [f"New cluster: {value}" for value in targets],
        "history_subcategories": history,
    }


def test_parse_cluster_generation_requires_exact_predefined_label() -> None:
    labels = {"Music", "Film"}
    assert parse_cluster_generation(
        "New cluster: Music",
        labels=labels,
    )["valid_label"]
    assert not parse_cluster_generation("Music", labels=labels)["has_prefix"]
    assert not parse_cluster_generation(
        "New cluster: Music and Film",
        labels=labels,
    )["valid_label"]


def test_generation_stops_on_chat_end_even_when_base_eos_differs() -> None:
    class Tokenizer:
        eos_token_id = 1
        unk_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "<|im_end|>"
            return 2

    assert generation_stop_token_ids(Tokenizer()) == [1, 2]


def test_free_generation_metrics_separate_match_recall_and_novelty() -> None:
    records = [
        _record("0", ["Music", "Film"], ["Sports"]),
        _record("1", ["Sports"], ["Sports"]),
        _record("2", ["Film"], ["Music"]),
    ]
    metrics, details = evaluate_generations(
        records,
        [
            "New cluster: Film",
            "New cluster: Sports",
            "Film",
        ],
        labels={"Music", "Film", "Sports"},
    )
    assert metrics["match_rate"] == pytest.approx(2 / 3)
    assert metrics["set_hit_at_1"] == pytest.approx(2 / 3)
    assert metrics["recall_at_1"] == pytest.approx((0.5 + 1.0) / 3)
    assert metrics["novelty_compliance_rate"] == pytest.approx(1 / 3)
    assert metrics["interest_macro_recall_at_1"] == pytest.approx(0.5)
    assert details[0]["novelty_compliant"] is True
    assert details[1]["novelty_compliant"] is False


def test_merge_generation_shards_recomputes_metrics(tmp_path) -> None:
    records = [
        _record("0", ["A"], ["H"]),
        _record("1", ["B"], ["H"]),
        _record("2", ["A"], ["H"]),
        _record("3", ["B"], ["H"]),
    ]
    eval_file = tmp_path / "eval.parquet"
    labels_file = tmp_path / "labels.parquet"
    pq.write_table(pa.Table.from_pylist(records), eval_file)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"target_subcategory": "A"},
                {"target_subcategory": "B"},
            ]
        ),
        labels_file,
    )
    identity_payload = {
        "model_name": "model",
        "model_revision": "revision",
        "adapter_sha256": "a" * 64,
        "eval_sha256": sha256_file(eval_file),
        "labels_sha256": sha256_file(labels_file),
        "labels": ["A", "B"],
        "response_prefix": "New cluster: ",
        "full_record_count": 4,
        "max_examples": None,
        "num_shards": 2,
        "decode": "greedy",
    }
    identity = canonical_sha256(identity_payload)
    shard_dirs = []
    outputs = ["New cluster: A", "New cluster: A", "bad", "New cluster: B"]
    for shard_index in range(2):
        shard_dir = tmp_path / f"shard-{shard_index}"
        shard_dir.mkdir()
        shard_dirs.append(shard_dir)
        rows = [
            {
                "record_index": index,
                "impression_id": records[index]["impression_id"],
                "input_sha256": record_input_sha256(records[index]),
                "evaluation_identity": identity,
                "output": outputs[index],
            }
            for index in range(4)
            if index % 2 == shard_index
        ]
        prediction = shard_dir / "generation_predictions.parquet"
        pq.write_table(pa.Table.from_pylist(rows), prediction)
        (shard_dir / "metrics.json").write_text(
            '{"elapsed_seconds": 2, "generated_samples_this_run": 2, "resumed_samples": 0}\n'
        )
        manifest = {
            **identity_payload,
            "evaluation_identity": identity,
            "shard_index": shard_index,
            "prediction_sha256": sha256_file(prediction),
            "logical_shard_record_count": 2,
        }
        (shard_dir / "shard_manifest.json").write_text(
            __import__("json").dumps(manifest) + "\n"
        )
    result = merge_generation_shards(
        shard_dirs,
        eval_file,
        labels_file,
        tmp_path / "merged",
        expected_records=4,
    )
    assert result["samples"] == 4
    assert result["match_rate"] == pytest.approx(3 / 4)
    assert result["set_hit_at_1"] == pytest.approx(2 / 4)
    assert pq.ParquetFile(
        tmp_path / "merged/generation_predictions.parquet"
    ).metadata.num_rows == 4
