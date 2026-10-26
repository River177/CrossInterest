from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_cir_mixed_dataset import build
from new_interest.experiments.cir_schedule import conditions, stage_specs


def test_shared_cir5_has_two_independent_ni_branches():
    specs = stage_specs(Path("/project"), "staged")
    assert [(s["key"], s["epochs"], s["parent"]) for s in specs] == [
        ("cir5", 5, None), ("ni3", 3, "cir5"), ("ni5", 5, "cir5")]
    assert [math.ceil(s["samples"] / 32) * s["epochs"] for s in specs] == [2700, 2025, 3375]
    assert conditions("staged") == [("cir5_ni3", ["cir5", "ni3"]), ("cir5_ni5", ["cir5", "ni5"])]


@pytest.mark.parametrize("mode,epochs,steps,presentations", [("mixed3", 3, 3645, 116613), ("mixed5", 5, 6075, 194355)])
def test_mixed_is_one_continuous_cold_stage(mode, epochs, steps, presentations):
    specs = stage_specs(Path("/project"), mode)
    assert len(specs) == 1 and specs[0]["parent"] is None
    assert specs[0]["epochs"] == epochs and specs[0]["task"] == "mixed"
    assert math.ceil(specs[0]["samples"] / 32) * epochs == steps
    assert specs[0]["samples"] * epochs == presentations
    assert specs[0]["world_size"] * specs[0]["per_device_train_batch_size"] * 4 == 32


def fixture_files(tmp_path, *, duplicate=False, wrong_task=False):
    cir = pa.table({"sample_id": ["cir:1", "cir:2"], "prompt": ["erase A", "erase B"],
        "response": ["<interest_000>", "<interest_001>"], "target_subcategory": ["A", "B"],
        "timestamp": pa.array([datetime(2019, 1, 1)] * 2, type=pa.timestamp("us")),
        "history_length": pa.array([5, 6], type=pa.int64()), "training_task": ["cross_interest_reconstruction"] * 2})
    ni = pa.table({"sample_id": ["cir:1" if duplicate else "ni:1"], "prompt": ["next click"],
        "response": ["<interest_001>"], "target_subcategory": ["B"],
        "timestamp": pa.array([datetime(2019, 1, 1)], type=pa.timestamp("ms")),
        "history_length": pa.array([7], type=pa.int32())})
    if wrong_task:
        ni = ni.append_column("training_task", pa.array(["old_interest"]))
    paths = tmp_path / "cir.parquet", tmp_path / "ni.parquet"
    for path, table in zip(paths, (cir, ni)):
        pq.write_table(table, path)
    return paths


def test_mixed_builder_preserves_original_rows_and_task_attribution(tmp_path):
    cir, ni = fixture_files(tmp_path)
    before = cir.read_bytes(), ni.read_bytes()
    output = tmp_path / "mixed"
    metadata = build(cir, ni, output)
    assert metadata["by_task"] == {"cross_interest_reconstruction": 2, "new_interest": 1}
    assert metadata["source_fields_preserved"]
    actual = pq.read_table(output / "mixed.parquet").to_pylist()
    for row, source in zip(actual, pq.read_table(cir).to_pylist() + pq.read_table(ni).to_pylist()):
        assert all(row[k] == v for k, v in source.items())
    assert actual[-1]["training_task"] == "new_interest"
    assert before == (cir.read_bytes(), ni.read_bytes())
    with pytest.raises(FileExistsError):
        build(cir, ni, output)


@pytest.mark.parametrize("key", ["duplicate", "wrong_task"])
def test_mixed_rejects_source_identity_conflicts(tmp_path, key):
    cir, ni = fixture_files(tmp_path, **{key: True})
    with pytest.raises(ValueError):
        build(cir, ni, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_five_epoch_mixed_ledger_and_nonzero_epoch_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from new_interest.training import sft_exposure as exposure, old_new_execution as cursor
    rows = [{"sample_id": f"sample:{i}", "target_subcategory": "A" if i < 20 else "B",
             "training_task": "cross_interest_reconstruction" if i < 20 else "new_interest"} for i in range(41)]
    path = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    digest = cursor._sha(path)
    first = exposure.TrainingExposure(rows, train_sha256=digest, expected_epochs=5)
    for epoch in range(5):
        order = cursor.epoch_permutation(41, epoch)
        for batch_index, offset in enumerate(range(0, 41, 32)):
            indices = order[offset:offset + 32]
            first.record(indices, [3] * len(indices), [10] * len(indices))
            if epoch == 1 and batch_index == 0:
                cp = tmp_path / "checkpoint-3"
                first.save_checkpoint(cp, 3)
                (cp / "trainer_state.json").write_text(json.dumps({"global_step": 3, "epoch": 1.5}))
                cursor.STAGES.add("cir/mixed5")
                checked = cursor.make_cursor_contract(path, stage="cir/mixed5", world_size=4,
                    per_device_batch_size=2, source_checkpoint=cp, epochs=5, global_batch_size=32)
                assert checked["completed_epochs"] == 1 and checked["epoch_optimizer_step"] == 1
                assert checked["consumed_total"] == 73
                restored = exposure.TrainingExposure(rows, train_sha256=digest, expected_epochs=5)
                restored.restore_checkpoint(cp)
                assert restored.counts == first.counts
                first = restored
    result = first.finish(tmp_path / "final", optimizer_steps=10)
    assert result["presentations"] == 205 and first.counts == [5] * 41
    assert result["by_task"]["cross_interest_reconstruction"]["presentations"] == 100
    assert result["by_task"]["new_interest"]["presentations"] == 105
