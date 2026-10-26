from __future__ import annotations

import copy
import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from new_interest.evaluation.evaluate_beam import canonical_sha256
from new_interest.evaluation.score_rho_reference import LOSS_DEFINITION, kernel_selection_identity


def finisher(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/analysis"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("rho_finish_scoring_test", scripts/"finish_rho_reference_scoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    return module


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def fixture_fold(pipe, fold=0):
    parents = {str(i): {"checkpoint_marker_sha256": digest(f"marker-{i}"), "adapter_sha256": digest(f"adapter-{i}"),
        "checkpoint_identity": {"train_file_sha256": digest(f"fold-{i}"), "model_name": "Qwen/Qwen3.5-2B",
                                "model_revision": "pinned-revision"}} for i in (0, 1)}
    contract = {"parents": parents, "original_training_runtime_sha256": digest("original-runtime"),
                "expected_initial_state_sha256": digest("initial-state"), "kernel_selection": "stable_fla_v1"}
    runtime, contract_sha = digest("current-runtime"), canonical_sha256(contract)
    kernel = kernel_selection_identity("stable_fla_v1")
    selections = [{"kernel": "fla.kernel", "cache_key": "common", "config": "BLOCK=32"},
                  {"kernel": "fla.kernel", "cache_key": f"fold-{fold}", "config": "BLOCK=64"}]
    ref = {"accepted": True, "teacher_rows": 10800, "full_opposite_fold_coverage": True,
        "metrics": {"rows": 10800, "users": 10800, "class_support": {f"class-{i}": 100 for i in range(108)}},
        "teacher_score_user_overlap": 0, "teacher_train_sha256": parents[str(fold)]["checkpoint_identity"]["train_file_sha256"],
        "score_file_sha256": parents[str(1-fold)]["checkpoint_identity"]["train_file_sha256"],
        "loss_definition": LOSS_DEFINITION, "mapping_sha256": digest("mapping"), **kernel,
        "selected_kernel_configs": selections, "selected_kernel_configs_sha256": canonical_sha256(selections)}
    probe = {"accepted": True, "native_and_fp32_exact": True, "production_batch4_fp32_exact": True,
             "independent_processes": 2, "probe_rows": 4, "overlapping_kernel_cache_keys_consistent": True,
             "model_updates": 0, "historical_training_parity_claimed": False}
    checkpoint = {"all_parent_file_hashes_verified": True, "new_training_updates": 0,
        "optimizer_state_entries": 373, "optimizer_step_min": 1014, "optimizer_step_max": 1014,
        "scheduler_last_epoch": 1014, "parent_marker_sha256": parents[str(fold)]["checkpoint_marker_sha256"],
        "adapter_sha256": parents[str(fold)]["adapter_sha256"], "identity": copy.deepcopy(parents[str(fold)]["checkpoint_identity"]),
        "parent_files_mutated": False}
    success = {"passed": True, "fold": fold, "scored_rows": 10800, "new_training_updates": 0,
        "parent_files_mutated": False, "historical_training_reload_parity_claimed": False,
        "identity": {"fold": fold, "runtime_sha256": runtime, "contract_sha256": contract_sha,
                     "protocol_sha256": digest("protocol"), "kernel": kernel},
        "parent_marker_sha256": parents[str(fold)]["checkpoint_marker_sha256"],
        "parent_adapter_sha256": parents[str(fold)]["adapter_sha256"],
        "original_training_runtime_sha256": contract["original_training_runtime_sha256"],
        "original_observed_initial_state_sha256": contract["expected_initial_state_sha256"]}
    payloads = {"worker_success.json": success, "reference_acceptance.json": ref,
                "fresh_probe_acceptance.json": probe, "checkpoint_verification.json": checkpoint,
                "environment.json": {"packages": {"torch": "2.8.0+cu126", "transformers": "5.14.1", "peft": "0.18.0",
                    "accelerate": "1.12.0", "tokenizers": "0.22.2", "safetensors": "0.8.0"},
                    "gpus": ["NVIDIA A100-SXM4-80GB"]*4}}
    folder = pipe.ROOT/f"fold-{fold}"
    files = {name: folder/name for name in payloads}
    republish(pipe, files, payloads)
    return files, payloads, contract, runtime, contract_sha


def republish(pipe, files, payloads):
    for name, obj in payloads.items():
        if name != "worker_success.json":
            save(files[name], obj)
    worker = payloads["worker_success.json"]
    worker["reference_acceptance_sha256"] = pipe.sha(files["reference_acceptance.json"])
    worker["fresh_probe_acceptance_sha256"] = pipe.sha(files["fresh_probe_acceptance.json"])
    save(files["worker_success.json"], worker)


@pytest.mark.parametrize("value", [None, [], {"中文": "兴趣"},
    [{"kernel": "fla.ops.k:0", "cache_key": "(32, 'torch.bfloat16')", "config": "BLOCK=64, num_warps=4"}]])
def test_canonical_kernel_digest_matches_remote_reader(tmp_path, monkeypatch, value):
    pipe = finisher(tmp_path, monkeypatch)
    assert pipe.canonical(value) == canonical_sha256(value)


def test_compact_download_uses_only_exact_requested_basenames(tmp_path, monkeypatch):
    pipe = finisher(tmp_path, monkeypatch)
    names = ["worker_success.json", "reference_acceptance.json", "fresh_probe_acceptance.json",
             "checkpoint_verification.json", "environment.json"]
    calls = []

    def command(args, *unused, **kwargs):
        calls.append(args)
        destination = Path(args[args.index("--output")+1])/"remote"/"compact"
        for name in names:
            save(destination/name, {})
        return ""

    monkeypatch.setattr(pipe, "command", command)
    files = pipe.compact(pipe.RESCORE, pipe.JOBS[0], "fold0", names)
    assert set(files) == set(names) and len(calls) == 1
    args = calls[0]
    patterns = [args[i+1] for i, token in enumerate(args) if token == "--include"]
    assert patterns == names
    for bad in ["adapter_model.safetensors", "optimizer.pt", "checkpoint_complete.json", "scores.parquet",
                "full_predictions.json", "manifest.json", "all_scores.zip", "results.tar.gz"]:
        assert not any(fnmatch.fnmatchcase(bad, pattern) for pattern in patterns)


@pytest.mark.parametrize("name", ["compact/worker_success.json", "*compact/*.json", "*.json", "x?.json",
                                   "x[01].json", "compact\\worker_success.json", "scores.parquet", "optimizer.pt"])
def test_compact_rejects_path_wildcard_and_non_json_patterns(tmp_path, monkeypatch, name):
    pipe = finisher(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: pytest.fail("Invalid include must not issue a download"))
    with pytest.raises(ValueError, match="exact compact JSON basenames"):
        pipe.compact(pipe.RESCORE, pipe.JOBS[0], "bad", [name])


@pytest.mark.parametrize("field,value", [("diagnostic_completed", False), ("read_only_integrity_passed", False),
    ("fixed_controls_passed", False), ("fixed_cross_condition_max_abs", .125), ("model_updates", 1),
    ("contract_sha256", "foreign"), ("original_training_reference_equivalence_established", True),
    ("parent_endpoint", {"global_step": 1000})])
def test_diagnostic_controls_gate_rescoring(tmp_path, monkeypatch, field, value):
    pipe = finisher(tmp_path, monkeypatch)
    report = {"contract_sha256": "bound", "model_updates": 0, "parent_endpoint": {"global_step": 1014},
              "original_training_reference_equivalence_established": False, "diagnostic_completed": True,
              "read_only_integrity_passed": True, "fixed_controls_passed": True, "fixed_cross_condition_max_abs": 0}
    pipe.check_diagnostic(report, "bound")
    report[field] = value
    with pytest.raises(ValueError):
        pipe.check_diagnostic(report, "bound")


def test_accepts_actual_fresh_probe_field_and_complete_bound_fold(tmp_path, monkeypatch):
    pipe = finisher(tmp_path, monkeypatch)
    files, payloads, contract, runtime, contract_sha = fixture_fold(pipe)
    assert "production_batch4_fp32_exact" in payloads["fresh_probe_acceptance.json"]
    assert "batch4_fp32_exact" not in payloads["fresh_probe_acceptance.json"]
    accepted = pipe.check_fold(files, 0, contract, runtime, contract_sha)
    assert accepted["reference"]["metrics"]["rows"] == 10800


@pytest.mark.parametrize("case", ["runtime", "contract", "fold", "identity_fold", "marker", "adapter", "original_runtime", "initial_state",
    "train_direction", "score_direction", "rows", "users", "class_support", "user_overlap", "probe_batch4", "probe_processes", "probe_rows", "probe_kernel_consistency",
    "checkpoint_identity", "checkpoint_adapter", "optimizer_min", "optimizer_max", "scheduler", "kernel_digest", "kernel_identity"])
def test_rejects_rehashed_but_semantically_wrong_fold_receipts(tmp_path, monkeypatch, case):
    pipe = finisher(tmp_path, monkeypatch)
    files, payloads, contract, runtime, contract_sha = fixture_fold(pipe)
    worker, ref, probe, checkpoint = [payloads[name] for name in
        ("worker_success.json", "reference_acceptance.json", "fresh_probe_acceptance.json", "checkpoint_verification.json")]
    if case in ("runtime", "contract"):
        worker["identity"][case+"_sha256"] = "foreign"
    elif case == "fold": worker["fold"] = 1
    elif case == "identity_fold": worker["identity"]["fold"] = 1
    elif case == "marker": worker["parent_marker_sha256"] = "foreign"
    elif case == "adapter": worker["parent_adapter_sha256"] = "foreign"
    elif case == "original_runtime": worker["original_training_runtime_sha256"] = "foreign"
    elif case == "initial_state": worker["original_observed_initial_state_sha256"] = "foreign"
    elif case == "train_direction": ref["teacher_train_sha256"] = "foreign"
    elif case == "score_direction": ref["score_file_sha256"] = "foreign"
    elif case in ("rows", "users"): ref["metrics"][case] -= 1
    elif case == "class_support": ref["metrics"]["class_support"]["class-0"] = 99
    elif case == "user_overlap": ref["teacher_score_user_overlap"] = 1
    elif case == "probe_batch4": probe["production_batch4_fp32_exact"] = False
    elif case == "probe_processes": probe["independent_processes"] = 1
    elif case == "probe_rows": probe["probe_rows"] = 3
    elif case == "probe_kernel_consistency": probe["overlapping_kernel_cache_keys_consistent"] = False
    elif case == "checkpoint_identity": checkpoint["identity"]["model_revision"] = "foreign"
    elif case == "checkpoint_adapter": checkpoint["adapter_sha256"] = "foreign"
    elif case == "optimizer_min": checkpoint["optimizer_step_min"] = 1000
    elif case == "optimizer_max": checkpoint["optimizer_step_max"] = 1000
    elif case == "scheduler": checkpoint["scheduler_last_epoch"] = 1000
    elif case == "kernel_digest": ref["selected_kernel_configs_sha256"] = "foreign"
    elif case == "kernel_identity": ref["kernel_selector_sha256"] = "foreign"
    republish(pipe, files, payloads)
    with pytest.raises(ValueError):
        pipe.check_fold(files, 0, contract, runtime, contract_sha)


@pytest.mark.parametrize("change", [None, "common_kernel", "direction", "mapping", "loss", "selector"])
def test_twofold_joint_acceptance_compares_common_keys_and_opposite_halves(tmp_path, monkeypatch, change):
    pipe = finisher(tmp_path, monkeypatch)
    folds = []
    for fold in (0, 1):
        files, payloads, *_ = fixture_fold(pipe, fold)
        folds.append({"worker": payloads["worker_success.json"], "reference": payloads["reference_acceptance.json"]})
    ref = folds[1]["reference"]
    if change == "common_kernel": ref["selected_kernel_configs"][0]["config"] = "BLOCK=128"
    elif change == "direction": ref["score_file_sha256"] = "foreign"
    elif change == "mapping": ref["mapping_sha256"] = "foreign"
    elif change == "loss": ref["loss_definition"] = "foreign"
    elif change == "selector": ref["kernel_selector_sha256"] = "foreign"
    if change is None:
        accepted = pipe.combine(folds, {"fixed_controls_passed": True})
        assert accepted["rows"] == accepted["users"] == 21600
        assert accepted["kernel_cache_keys_checked"] == 3
        assert accepted["student_training_started"] is False
    else:
        with pytest.raises(ValueError):
            pipe.combine(folds, {})


def run_once(pipe, monkeypatch, state):
    save(pipe.ROOT/"repair_state.json", state)
    monkeypatch.setattr(sys, "argv", ["finisher", "--once"])
    pipe.main()
    return pipe.read(pipe.ROOT/"repair_state.json")


@pytest.mark.parametrize("phase", ["diagnostic_submitted", "rescore_wait"])
@pytest.mark.parametrize("status", ["failed", "killed", "canceled"])
def test_any_failed_worker_stops_without_readback_or_submission(tmp_path, monkeypatch, phase, status):
    pipe = finisher(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe, "statuses", lambda exp, jobs: {job: status if i == 0 else "running" for i, job in enumerate(jobs)})
    monkeypatch.setattr(pipe, "compact", lambda *args: pytest.fail("Failed job must not collect acceptance"))
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: pytest.fail("Failed job must not resubmit"))
    result = run_once(pipe, monkeypatch, {"phase": phase})
    assert result["phase"] == "needs_attention"
    assert "transport_error" not in result


@pytest.mark.parametrize("already_attempted", [False, True])
@pytest.mark.parametrize("failure", [TimeoutError("lost response"), RuntimeError("submit failed")])
def test_submission_failure_never_duplicates(tmp_path, monkeypatch, already_attempted, failure):
    pipe = finisher(tmp_path, monkeypatch)
    calls = []

    def failed(args, *rest, **kwargs):
        assert pipe.read(pipe.ROOT/"repair_state.json")["rescoring_submission_attempted"] is True
        calls.append(args)
        raise failure

    monkeypatch.setattr(pipe, "command", failed)
    result = run_once(pipe, monkeypatch, {"phase": "rescore_submit", "rescoring_submission_attempted": already_attempted,
                                       "rescore_config": "/tmp/authorized.yaml"})
    assert result["phase"] == "needs_attention" and result["rescoring_submission_attempted"] is True
    assert len(calls) == (0 if already_attempted else 1)
    assert run_once(pipe, monkeypatch, result) == result


@pytest.mark.parametrize("phase", ["diagnostic_submitted", "rescore_wait"])
def test_status_transport_failure_preserves_phase(tmp_path, monkeypatch, phase):
    pipe = finisher(tmp_path, monkeypatch)

    def failed(*args):
        raise TimeoutError("retryable transport")

    monkeypatch.setattr(pipe, "statuses", failed)
    result = run_once(pipe, monkeypatch, {"phase": phase})
    assert result["phase"] == phase and "transport_error" in result
