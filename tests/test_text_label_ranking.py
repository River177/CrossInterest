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
from new_interest.evaluation.evaluate_text_label_ranking import (
    EVALUATION_IDENTITY_VERSION,
    common_non_label_prefix_length,
    encode_label_response,
    generate_text_label_rankings,
    mean_selected_token_log_probabilities,
)
from new_interest.evaluation.merge_text_label_ranking_shards import (
    merge_text_label_ranking_shards,
)


class _OffsetTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        del tokenize, enable_thinking
        text = "".join(str(message["content"]) for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ):
        assert add_special_tokens is False
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def test_encode_label_response_excludes_prefix_and_eos_from_score() -> None:
    encoded = encode_label_response(_OffsetTokenizer(), "AB")
    assert encoded.response_token_ids[-1] == 99
    assert [
        encoded.response_token_ids[index]
        for index in encoded.label_response_positions
    ] == [ord("A"), ord("B")]


def test_encode_label_response_accepts_wang_cluster_prefix() -> None:
    encoded = encode_label_response(
        _OffsetTokenizer(),
        "AB",
        response_prefix="New cluster: ",
    )
    assert [
        encoded.response_token_ids[index]
        for index in encoded.label_response_positions
    ] == [ord("A"), ord("B")]


def test_mean_selected_token_log_probabilities_average_per_row() -> None:
    logits = torch.tensor(
        [
            [[0.0, 2.0, 0.0], [0.0, 0.0, 4.0]],
            [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    scores = mean_selected_token_log_probabilities(
        logits,
        [[0], [0, 1]],
        [[1], [0, 1]],
    )
    expected_first = torch.log_softmax(logits[0, 0], dim=-1)[1]
    expected_second = torch.stack(
        [
            torch.log_softmax(logits[1, 0], dim=-1)[0],
            torch.log_softmax(logits[1, 1], dim=-1)[1],
        ]
    ).mean()
    assert scores[0] == pytest.approx(float(expected_first))
    assert scores[1] == pytest.approx(float(expected_second))


def test_generate_text_label_rankings_scores_only_label_tokens() -> None:
    from types import SimpleNamespace

    class _CopyNextTokenModel:
        device = torch.device("cpu")

        def eval(self):
            return self

        def __call__(
            self,
            *,
            input_ids,
            attention_mask,
            use_cache,
            logits_to_keep,
        ):
            del attention_mask, use_cache
            batch, sequence_length = input_ids.shape
            first = sequence_length - logits_to_keep
            logits = torch.zeros(batch, logits_to_keep, 128)
            for row in range(batch):
                for offset in range(logits_to_keep - 1):
                    target = int(input_ids[row, first + offset + 1])
                    logits[row, offset, target] = 8.0
            return SimpleNamespace(logits=logits)

    ranked, entropies, counts = generate_text_label_rankings(
        _CopyNextTokenModel(),
        _OffsetTokenizer(),
        [{"prompt": "history"}],
        ["A", "BC"],
        batch_size=2,
        top_k=2,
        max_seq_length=128,
        enable_thinking=False,
        scoring_backend="full_sequence",
    )
    assert counts == [1, 2]
    assert {item["label"] for item in ranked[0]} == {"A", "BC"}
    assert all(item["score"] > -0.1 for item in ranked[0])
    assert len(entropies) == 1 and entropies[0] > 0


def test_common_non_label_prefix_stops_before_category_tokens() -> None:
    tokenizer = _OffsetTokenizer()
    encodings = [
        encode_label_response(tokenizer, label) for label in ("A", "BC")
    ]
    prefix_length = common_non_label_prefix_length(encodings)
    assert prefix_length == len("New topic: ")
    assert all(
        min(encoding.label_response_positions) == prefix_length
        for encoding in encodings
    )


def test_prefix_cache_backend_repeats_cache_by_reorder() -> None:
    from types import SimpleNamespace

    class _Cache:
        def __init__(self):
            self.batch_indices = None

        def reorder_cache(self, indices):
            self.batch_indices = indices.clone()

    class _UniformCacheModel:
        device = torch.device("cpu")

        def eval(self):
            return self

        def __call__(self, *, input_ids, logits_to_keep, **kwargs):
            cache = kwargs.get("past_key_values") or _Cache()
            return SimpleNamespace(
                logits=torch.zeros(input_ids.shape[0], logits_to_keep, 128),
                past_key_values=cache,
            )

    ranked, entropies, counts = generate_text_label_rankings(
        _UniformCacheModel(),
        _OffsetTokenizer(),
        [{"prompt": "history"}],
        ["A", "BC"],
        batch_size=2,
        top_k=2,
        max_seq_length=128,
        enable_thinking=False,
        scoring_backend="prefix_cache",
    )
    assert counts == [1, 2]
    assert [item["label"] for item in ranked[0]] == ["A", "BC"]
    assert all(
        item["score"] == pytest.approx(-torch.log(torch.tensor(128.0)).item())
        for item in ranked[0]
    )
    assert entropies[0] == pytest.approx(torch.log(torch.tensor(2.0)).item())


def _write_merge_fixture(
    root: Path,
) -> tuple[Path, Path, list[Path]]:
    records = [
        {
            "impression_id": str(index),
            "user_id": f"U{index}",
            "prompt": f"Prompt {index}",
            "acceptable_subcategories": ["A" if index % 2 == 0 else "B"],
        }
        for index in range(4)
    ]
    eval_file = root / "dev.parquet"
    labels_file = root / "labels.parquet"
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
        "identity_version": EVALUATION_IDENTITY_VERSION,
        "model_name": "test-model",
        "adapter_sha256": "a" * 64,
        "eval_sha256": sha256_file(eval_file),
        "labels_sha256": sha256_file(labels_file),
        "full_record_count": len(records),
        "labels": labels,
        "label_token_counts": [1, 1, 1],
        "scoring": "label_token_mean_log_probability",
        "scoring_backend": "prefix_cache",
        "fixed_prefix_scored": False,
        "eos_scored": False,
        "max_examples": None,
        "num_shards": 2,
        "top_k": 2,
        "cutoffs": [1, 2],
        "max_seq_length": 128,
        "enable_thinking": False,
    }
    identity = canonical_sha256(identity_payload)
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
            rows.append(
                {
                    "record_index": record_index,
                    "impression_id": record["impression_id"],
                    "input_sha256": record_input_sha256(record),
                    "evaluation_identity": identity,
                    "ranked_labels": [correct, other],
                    "ranked_scores": [-0.1, -2.0],
                    "score_softmax_entropy": 0.5,
                }
            )
        prediction_path = shard_dir / "text_label_predictions.parquet"
        pq.write_table(pa.Table.from_pylist(rows), prediction_path)
        (shard_dir / "shard_manifest.json").write_text(
            json.dumps(
                {
                    **identity_payload,
                    "evaluation_identity": identity,
                    "shard_index": shard_index,
                    "shard_record_count": len(rows),
                    "prediction_sha256": sha256_file(prediction_path),
                    "elapsed_seconds": 2.0,
                    "generated_samples": len(rows),
                    "resumed_samples": 0,
                }
            ),
            encoding="utf-8",
        )
    return eval_file, labels_file, shard_dirs


def test_merge_text_label_shards_restores_global_order(tmp_path: Path) -> None:
    eval_file, labels_file, shard_dirs = _write_merge_fixture(tmp_path)
    output_dir = tmp_path / "merged"
    metrics = merge_text_label_ranking_shards(
        list(reversed(shard_dirs)),
        eval_file,
        labels_file,
        output_dir,
        expected_records=4,
    )
    assert metrics["samples"] == 4
    assert metrics["interest_macro_recall_at_1"] == 1.0
    assert metrics["interest_macro_ndcg_at_1"] == 1.0
    assert metrics["history_prediction_filtering"] is False
    rows = pq.read_table(
        output_dir / "text_label_predictions.parquet"
    ).to_pylist()
    assert [row["record_index"] for row in rows] == [0, 1, 2, 3]


def test_merge_text_label_shards_rejects_duplicate_record(
    tmp_path: Path,
) -> None:
    eval_file, labels_file, shard_dirs = _write_merge_fixture(tmp_path)
    prediction_path = shard_dirs[0] / "text_label_predictions.parquet"
    table = pq.read_table(prediction_path)
    rows = table.to_pylist()
    rows.append(rows[0])
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), prediction_path)
    manifest_path = shard_dirs[0] / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_sha256"] = sha256_file(prediction_path)
    manifest["shard_record_count"] = len(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate record index 0"):
        merge_text_label_ranking_shards(
            shard_dirs,
            eval_file,
            labels_file,
            tmp_path / "merged",
        )
