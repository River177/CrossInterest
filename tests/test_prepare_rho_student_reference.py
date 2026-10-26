from __future__ import annotations

import copy
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from new_interest.experiments import prepare_rho_student_reference as preparation
from new_interest.experiments.read_rho_reference import verify_reference
from new_interest.training.rho_dynamic_weights import build_classwise_weights
from test_rho_reference import make_scored_fixture, record, stable_fixture, write_json, write_parquet


@pytest.fixture
def accepted(tmp_path):
    halves = [[record(i, "a" if i % 2 == 0 else "b") for i in range(start, start + 4)] for start in (0, 4)]
    input_paths = [tmp_path / "inputs" / f"fold{i}.parquet" for i in range(2)]
    mapping_path = tmp_path / "inputs/mapping.parquet"
    folds, fixture_args = [], []
    for index in range(2):
        root = tmp_path / f"worker{index}"
        args = stable_fixture(make_scored_fixture(root, train_rows=halves[index], score_rows=halves[1-index],
            num_shards=4, train_path=input_paths[index], score_path=input_paths[1-index], mapping_path=mapping_path))
        fixture_args.append(args)
        report_path = root / "compact/reference_acceptance.json"
        reference = verify_reference(**args, output=report_path)
        worker = {"passed": True, "fold": index, "new_training_updates": 0,
            "reference_acceptance_sha256": preparation.sha(report_path),
            "parent_adapter_sha256": preparation.sha(root / "teacher/final_adapter/adapter_model.safetensors"),
            "restoration": {"teacher": str(root / "teacher"), "train_file": str(args["teacher_train_file"]),
                "score_file": str(args["score_file"]), "mapping_file": str(args["mapping_file"]),
                "score_dirs": list(map(str, args["score_dirs"]))}}
        folds.append({"worker": worker, "reference": reference})
    joint = {"passed": True, "rows": 8, "users": 8, "folds": folds,
        "twofold_union_exact_from_verified_complementary_inputs": True,
        "kernel_selection": "stable_fla_v1", "common_kernel_configs_consistent": True,
        "original_fold1_training_reference_equivalence_established": False}
    path = tmp_path / "joint_acceptance.json"
    write_json(path, joint)
    train = tmp_path / "student.parquet"
    source = [halves[1][3], halves[0][0], halves[1][0]]
    write_parquet(train, source)
    return {"acceptance_path": path, "expected_sha": preparation.sha(path), "train_file": train,
            "output_dir": tmp_path / "prepared", "expected_total_rows": 8,
            "joint": joint, "args": fixture_args, "source": source, "halves": halves}


def prepare(case):
    return preparation.prepare_reference(**{key: case[key] for key in
        ("acceptance_path", "expected_sha", "train_file", "output_dir", "expected_total_rows")})


def reseal_joint(case):
    write_json(case["acceptance_path"], case["joint"])
    case["expected_sha"] = preparation.sha(case["acceptance_path"])


def teacher_hashes(case):
    paths = []
    for args in case["args"]:
        for directory in args["score_dirs"]:
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return {str(path): preparation.sha(path) for path in paths}


def test_full_remote_readback_then_real_subset_in_source_order(accepted):
    before = teacher_hashes(accepted)
    result = prepare(accepted)
    rows = pq.read_table(result["file"]).to_pylist()
    assert result["rows"] == 3 and result["sha256"] == preparation.sha(result["file"])
    assert [row["source_sample_id"] for row in rows] == ["s7", "s0", "s4"]
    assert [row["teacher_fold"] for row in rows] == [0, 1, 0]
    assert [row["full_vocab_first_sid_nll"] for row in rows] == [2., 2., 2.]
    assert all(row["joint_acceptance_sha256"] == accepted["expected_sha"] for row in rows)
    assert "sid_scores_fp32" not in rows[0]
    assert result["receipt"]["verified_full_reference_rows"] == 8
    assert result["receipt"]["score_artifacts_verified"] == 8
    assert result["receipt"]["subset"] and result["receipt"]["common_kernel_configs_consistent"]
    assert not result["receipt"]["original_fold1_training_reference_equivalence_established"]
    assert teacher_hashes(accepted) == before
    assert sorted(path.name for path in accepted["output_dir"].iterdir()) == ["reference_preparation.json", "reference_scores.parquet"]
    weights = build_classwise_weights(accepted["source"], mode="static", reference_rows=rows,
                                     expected_classes=None, expected_rows_per_class=None)
    assert [row["sample_weight"] for row in weights["rows"]] == [1., 1., 1.]


def test_full_source_and_committed_cache_reuse_without_second_teacher_readback(accepted, monkeypatch):
    write_parquet(accepted["train_file"], accepted["halves"][0] + accepted["halves"][1])
    first = prepare(accepted)
    assert first["rows"] == 8 and not first["receipt"]["subset"]
    def no_repeated_readback(**kwargs):
        raise AssertionError("Committed cache should not reread the teacher artifacts")
    monkeypatch.setattr(preparation, "verify_reference", no_repeated_readback)
    second = prepare(accepted)
    assert second == first


@pytest.mark.parametrize("corruption", ["joint_hash", "joint_pass", "fold_pass", "duplicate_fold", "missing_artifact", "changed_parent"])
def test_accepted_lineage_and_complete_topology_are_required(accepted, corruption):
    if corruption == "joint_hash":
        accepted["expected_sha"] = "0" * 64
    elif corruption == "joint_pass":
        accepted["joint"]["passed"] = False
        reseal_joint(accepted)
    elif corruption == "fold_pass":
        accepted["joint"]["folds"][0]["worker"]["passed"] = False
        reseal_joint(accepted)
    elif corruption == "duplicate_fold":
        accepted["joint"]["folds"][1]["worker"]["fold"] = 0
        reseal_joint(accepted)
    elif corruption == "missing_artifact":
        accepted["joint"]["folds"][0]["worker"]["restoration"]["score_dirs"].pop()
        reseal_joint(accepted)
    elif corruption == "changed_parent":
        accepted["joint"]["folds"][0]["worker"]["parent_adapter_sha256"] = "0" * 64
        reseal_joint(accepted)
    with pytest.raises(ValueError):
        prepare(accepted)
    assert not accepted["output_dir"].exists()


def test_actual_remote_prediction_bytes_are_reverified(accepted):
    path = accepted["args"][0]["score_dirs"][0] / "scores.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["full_vocab_first_sid_nll"] += .01
    write_parquet(path, rows)
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(accepted)


@pytest.mark.parametrize("mutation", ["messages", "duplicate_user", "unknown", "old_interest"])
def test_student_subset_must_be_exact_real_scored_inputs(accepted, mutation):
    rows = copy.deepcopy(accepted["source"])
    if mutation == "messages":
        rows[0]["messages"][0]["content"] += " modified history"
    elif mutation == "duplicate_user":
        rows[1]["user_id"] = rows[0]["user_id"]
    elif mutation == "unknown":
        rows[0] = record(100)
    elif mutation == "old_interest":
        rows[0]["training_task"] = "old_interest"
    write_parquet(accepted["train_file"], rows)
    with pytest.raises(ValueError):
        prepare(accepted)


def test_crossfold_kernel_config_disagreement_is_rejected_even_if_each_fold_passes(accepted):
    args = accepted["args"][1]
    for directory in args["score_dirs"]:
        receipt = preparation.read(directory / "kernel_selection.json")
        for snapshot in receipt["snapshots"]:
            snapshot["kernels"]["fla.test"]["cache"]["common"] = "BLOCK=128"
        receipt["snapshots_sha256"] = preparation.canonical_sha256(receipt["snapshots"])
        write_json(directory / "kernel_selection.json", receipt)
        manifest = preparation.read(directory / "manifest.json")
        manifest["kernel_selection_sha256"] = preparation.sha(directory / "kernel_selection.json")
        write_json(directory / "manifest.json", manifest)
    report_path = Path(accepted["joint"]["folds"][1]["worker"]["restoration"]["teacher"]).parent / "compact/reference_acceptance.json"
    reference = verify_reference(**args, output=report_path)
    accepted["joint"]["folds"][1]["reference"] = reference
    accepted["joint"]["folds"][1]["worker"]["reference_acceptance_sha256"] = preparation.sha(report_path)
    reseal_joint(accepted)
    with pytest.raises(ValueError, match="identical cache key"):
        prepare(accepted)


def test_duplicate_opposite_half_cannot_masquerade_as_a_complete_twofold_union(accepted):
    accepted["joint"]["folds"][1] = copy.deepcopy(accepted["joint"]["folds"][0])
    accepted["joint"]["folds"][1]["worker"]["fold"] = 1
    reseal_joint(accepted)
    with pytest.raises(ValueError, match="repeats a source sample"):
        prepare(accepted)


def test_cached_reference_corruption_and_output_overlap_are_rejected(accepted):
    result = prepare(accepted)
    path = Path(result["file"])
    rows = pq.read_table(path).to_pylist()
    rows[0]["full_vocab_first_sid_nll"] += .2
    write_parquet(path, rows)
    with pytest.raises(ValueError, match="identity or bytes differ"):
        prepare(accepted)
    accepted["output_dir"] = Path(accepted["joint"]["folds"][0]["worker"]["restoration"]["teacher"])
    with pytest.raises(ValueError, match="overlap immutable"):
        prepare(accepted)


def test_default_full_size_rejects_small_fixture_without_explicit_test_override(accepted):
    with pytest.raises(ValueError, match="complete stable twofold"):
        preparation.prepare_reference(accepted["acceptance_path"], accepted["expected_sha"],
                                      accepted["train_file"], accepted["output_dir"])
