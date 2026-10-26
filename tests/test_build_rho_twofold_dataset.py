from __future__ import annotations

from collections import Counter
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_rho_twofold_dataset import (
    assign_twofold, build_twofold_dataset, sha256_file,
)
from new_interest.training.sft_data import ParquetSFTDataset


def _write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def inputs(tmp_path):
    rows, raw = [], []
    # Interleaved targets and OI rows exercise membership vs preserved source order.
    for index in range(8):
        target = f"target{index % 2}"
        source_id = f"source:{index}"
        history = [{"subcategory": "old", "title": f"old headline {index}"}]
        prompt = f"History\n\n1. [old] old headline {index}\n\nPredict new interest"
        response = f"<interest_{index % 2:03d}>"
        row = {"sample_id": f"{source_id}:behavior_interest", "source_sample_id": source_id,
               "user_id": f"user{index}", "impression_id": f"impression{index}",
               "split": "train", "task_type": "behavior_interest", "history_length": 1,
               "prompt": prompt, "response": response, "target_subcategory": target,
               "messages": [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}],
               "training_task": "new_interest", "unrelated_preserved_field": index * 17}
        rows.append({**row, "sample_id": f"oi{index}", "source_sample_id": f"oi{index}",
                     "training_task": "old_interest"})
        rows.append(row)
        raw.append({"sample_id": source_id, "user_id": row["user_id"],
                    "impression_id": row["impression_id"], "history": history,
                    "target_interest": {"subcategory": target}})
    _write(tmp_path / "mixed.parquet", rows)
    _write(tmp_path / "mapping.parquet", [
        {"interest_index": i, "interest_token": f"<interest_{i:03d}>", "target_subcategory": f"target{i}"}
        for i in range(2)])
    _write(tmp_path / "validation.parquet", [{"user_id": "validation-user"}])
    _write(tmp_path / "dev.parquet", [{"user_id": "dev-user"}, {"user_id": "dev-user"}])
    _write(tmp_path / "raw.parquet", raw)
    raw_metadata = {"full_history": True, "history_truncation": None,
                    "impression_clicks_appended_to_history": False,
                    "artifact_sha256": {"raw.parquet": sha256_file(tmp_path / "raw.parquet")}}
    (tmp_path / "raw_metadata.json").write_text(json.dumps(raw_metadata))
    return {"source_file": tmp_path / "mixed.parquet", "mapping_file": tmp_path / "mapping.parquet",
            "validation_file": tmp_path / "validation.parquet", "dev_file": tmp_path / "dev.parquet",
            "raw_train_file": tmp_path / "raw.parquet", "raw_metadata_file": tmp_path / "raw_metadata.json",
            "output_dir": tmp_path / "output", "expected_class_count": 2, "rows_per_class": 4}


def _rewrite_raw(inputs, rows):
    _write(inputs["raw_train_file"], rows)
    path = inputs["raw_metadata_file"]
    metadata = json.loads(path.read_text())
    metadata["artifact_sha256"]["raw.parquet"] = sha256_file(inputs["raw_train_file"])
    path.write_text(json.dumps(metadata))


class ToyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not enable_thinking
        text = "".join(f"[{message['role']}]{message['content']}\n" for message in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        ids = [ord(character) for character in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_balanced_exact_union_preserves_every_source_field_and_sft_loading(inputs, monkeypatch):
    monkeypatch.chdir(inputs["source_file"].parent)
    result = build_twofold_dataset(**inputs)
    source = pq.read_table(inputs["source_file"])
    manifest = pq.read_table(inputs["output_dir"] / "fold_manifest.parquet").to_pylist()
    assert [row["source_row_index"] for row in manifest] == list(range(1, 16, 2))
    assert [row["ni_row_index"] for row in manifest] == list(range(8))
    seen = set()
    for fold in range(2):
        path = inputs["output_dir"] / f"fold{fold}.parquet"
        table = pq.read_table(path)
        members = [row for row in manifest if row["fold"] == fold]
        assert table.equals(source.take([row["source_row_index"] for row in members]), check_metadata=True)
        assert Counter(table["target_subcategory"].to_pylist()) == {"target0": 2, "target1": 2}
        users = set(table["user_id"].to_pylist())
        assert not seen & users
        seen |= users
        dataset = ParquetSFTDataset(path, ToyTokenizer(), max_seq_length=2048, enable_thinking=False)
        assert len(dataset) == 4
        assert all(any(value != -100 for value in example["labels"]) for example in dataset.examples)
    assert seen == {f"user{i}" for i in range(8)}
    assert result["sources"]["training_source"]["path"] == "mixed.parquet"
    assert result["novelty_audit"]["rows_checked"] == 8
    assert result["heldout_audit"]["dev"] == {"rows": 2, "users": 1, "training_user_overlap": 0}
    assert (inputs["output_dir"] / "interest_token_mapping.parquet").read_bytes() == inputs["mapping_file"].read_bytes()
    for name, receipt in result["files"].items():
        assert sha256_file(inputs["output_dir"] / name) == receipt["sha256"]


def test_membership_deterministic_independent_of_source_row_order(inputs):
    rows = [row for row in pq.read_table(inputs["source_file"]).to_pylist() if row["training_task"] == "new_interest"]
    first = assign_twofold(rows, seed=42, rows_per_class=4)
    second = assign_twofold(list(reversed(rows)), seed=42, rows_per_class=4)
    assert {row["user_id"]: value for row, value in zip(rows, first)} == {
        row["user_id"]: value for row, value in zip(reversed(rows), second)}
    assert first == assign_twofold(rows, seed=42, rows_per_class=4)


@pytest.mark.parametrize("field", ["sample_id", "source_sample_id", "user_id"])
def test_duplicate_identity_rejected(inputs, field):
    rows = pq.read_table(inputs["source_file"]).to_pylist()
    rows[3][field] = rows[1][field]
    _write(inputs["source_file"], rows)
    with pytest.raises(ValueError, match=f"duplicate {field}"):
        build_twofold_dataset(**inputs)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("split", ["validation", "dev"])
def test_heldout_user_leakage_rejected(inputs, split):
    _write(inputs[f"{split}_file"], [{"user_id": "user0"}])
    with pytest.raises(ValueError, match=f"Train/{split} user overlap"):
        build_twofold_dataset(**inputs)


def test_full_history_novelty_uses_events_outside_recent_prompt(inputs):
    raw = pq.read_table(inputs["raw_train_file"]).to_pylist()
    raw[0]["history"].insert(0, {"subcategory": "target0", "title": "older target evidence"})
    _rewrite_raw(inputs, raw)
    with pytest.raises(ValueError, match="Target appears in full raw history"):
        build_twofold_dataset(**inputs, history_limit=1)


@pytest.mark.parametrize("corruption,expected", [
    ("hash", "SHA-256"), ("truncation", "untruncated"), ("missing", "Missing 1 source IDs"),
    ("user", "user_id mismatch"), ("target", "target mismatch"), ("prompt", "recent history"),
])
def test_raw_history_provenance_and_identity_fail_closed(inputs, corruption, expected):
    raw = pq.read_table(inputs["raw_train_file"]).to_pylist()
    if corruption == "hash":
        raw[0]["history"][0]["title"] += " edited"
        _write(inputs["raw_train_file"], raw)
    elif corruption == "truncation":
        metadata = json.loads(inputs["raw_metadata_file"].read_text())
        metadata["history_truncation"] = 50
        inputs["raw_metadata_file"].write_text(json.dumps(metadata))
    else:
        if corruption == "missing":
            raw.pop()
        elif corruption == "user":
            raw[0]["user_id"] = "wrong-user"
        elif corruption == "target":
            raw[0]["target_interest"]["subcategory"] = "wrong-target"
        elif corruption == "prompt":
            raw[0]["history"][0]["title"] += " edited"
        _rewrite_raw(inputs, raw)
    with pytest.raises(ValueError, match=expected):
        build_twofold_dataset(**inputs)
    assert not inputs["output_dir"].exists()


def test_corrupt_message_and_mapping_rejected(inputs):
    rows = pq.read_table(inputs["source_file"]).to_pylist()
    rows[1]["messages"][1]["content"] = "wrong label"
    _write(inputs["source_file"], rows)
    with pytest.raises(ValueError, match="Prompt/response/messages mismatch"):
        build_twofold_dataset(**inputs)
    rows[1]["messages"][1]["content"] = rows[1]["response"]
    rows[1]["response"] = "<interest_001>"
    _write(inputs["source_file"], rows)
    with pytest.raises(ValueError, match="Target/SID mismatch"):
        build_twofold_dataset(**inputs)


def test_production_profile_and_even_balance_required(inputs):
    with pytest.raises(ValueError, match="Production profile requires"):
        build_twofold_dataset(**inputs, production=True)
    with pytest.raises(ValueError, match="positive even"):
        assign_twofold([], seed=42, rows_per_class=3)
    rows = pq.read_table(inputs["source_file"]).to_pylist()
    rows[1]["target_subcategory"] = "target1"
    rows[1]["response"] = "<interest_001>"
    rows[1]["messages"][1]["content"] = "<interest_001>"
    _write(inputs["source_file"], rows)
    with pytest.raises(ValueError, match="expected 4"):
        build_twofold_dataset(**inputs)


def test_existing_output_is_never_replaced(inputs):
    build_twofold_dataset(**inputs)
    before = (inputs["output_dir"] / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_twofold_dataset(**inputs)
    assert (inputs["output_dir"] / "metadata.json").read_bytes() == before
    assert not list(inputs["output_dir"].parent.glob(".output.staging-*"))
