import json
from pathlib import Path

import pytest

from new_interest.experiments.cir_warmup import formal_schedule, invocation, stage_specs


@pytest.mark.parametrize("count", [108, 125])
def test_bind_separates_model_vocabulary_from_output_candidates(tmp_path, monkeypatch, count):
    from new_interest.experiments import cir_warmup, backbone_scaling, old_new_interest

    for name in ("CONTRACT", "TRAIN", "DEV", "MAPPING", "CANDIDATE_MAPPING"):
        monkeypatch.setattr(backbone_scaling, name, getattr(backbone_scaling, name))
    monkeypatch.setattr(old_new_interest, "CONTRACT", old_new_interest.CONTRACT)
    folder = tmp_path / "configs/sft"
    folder.mkdir(parents=True)
    mapping = cir_warmup.DATA / ("model_token_mapping.parquet" if count == 125 else "interest_token_mapping.parquet")
    (folder / "mind_cir_base_v1.json").write_text(json.dumps({
        "semantic_id_token_mapping_file": str(mapping), "semantic_id_token_count": count,
    }))
    _, shared = cir_warmup.bind(tmp_path)
    assert shared.MAPPING == mapping
    assert shared.CANDIDATE_MAPPING == cir_warmup.DATA / "interest_token_mapping.parquet"


def write_contract(project, schedule=None):
    folder = project / "configs/experiments"
    folder.mkdir(parents=True, exist_ok=True)
    value = {"training_schedule": schedule} if schedule is not None else {}
    (folder / "mind_cir_108_v1.json").write_text(json.dumps(value))


def test_formal_is_only_cir_then_original_ni(tmp_path):
    folder = tmp_path / "data/training/mind_cir_108_v1"
    folder.mkdir(parents=True)
    (folder / "statistics.json").write_text(json.dumps({"cir_rows": 17271}))
    write_contract(tmp_path)
    specs = stage_specs(tmp_path, tmp_path / "outputs/formal", gates=False)
    assert [(s["task"], s["epochs"], s["parent"]) for s in specs] == [("cir", 1, None), ("ni", 3, "cir")]
    assert specs[1]["samples"] == 21600
    for spec in specs:
        assert spec["world_size"] == 4 and spec["global_batch_size"] == 32


def test_three_epoch_duration_keeps_data_and_fresh_cir_start(tmp_path):
    folder = tmp_path / "data/training/mind_cir_108_v1"
    folder.mkdir(parents=True)
    original = json.dumps({"cir_rows": 17271, "cir_epochs": 1})
    (folder / "statistics.json").write_text(original)
    write_contract(tmp_path, {"cir_epochs": 3, "ni_epochs": 3})
    specs = stage_specs(tmp_path, tmp_path / "formal", gates=False)
    assert [(s["epochs"], s["parent"]) for s in specs] == [(3, None), (3, "cir")]
    assert formal_schedule(tmp_path)["cir_optimizer_steps"] == 1620
    assert formal_schedule(tmp_path)["ni_optimizer_steps"] == 2025
    assert (folder / "statistics.json").read_text() == original


@pytest.mark.parametrize("schedule", [
    {"cir_epochs": 0, "ni_epochs": 3}, {"cir_epochs": 1.5, "ni_epochs": 3},
    {"cir_epochs": True, "ni_epochs": 3}, {"cir_epochs": 3, "ni_epochs": 2},
    {"cir_epochs": 3, "ni_epochs": 3, "cir_steps": 540},
])
def test_invalid_duration_rejected_before_training(tmp_path, schedule):
    write_contract(tmp_path, schedule)
    with pytest.raises(ValueError):
        formal_schedule(tmp_path)


def test_gates_exercise_both_handoffs_and_nondivisible_ddp_tails(tmp_path):
    specs = stage_specs(tmp_path, tmp_path / "gates", gates=True)
    assert [s["world_size"] for s in specs] == [1, 1, 4, 4]
    assert [s["parent"] for s in specs] == [None, "single_cir", None, "ddp_cir"]
    assert specs[2]["samples"] % 32 != 0


def test_all_inference_calls_use_explicit_kernel_wrapper():
    assert "new_interest.experiments.cir_warmup" in invocation("verify", "--", "--atol", 0)
    assert invocation("infer", "--", "--batch-size", 2)[-1] == "2"
