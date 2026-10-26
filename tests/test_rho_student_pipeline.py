from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import yaml


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pipeline(tmp_path, monkeypatch, *, prepared=True):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/"scripts/analysis"))
    fixtures = load(repo/"tests/test_prepare_rho_student_comparison.py", "comparison_fixture_helpers")
    preparer = fixtures.module(tmp_path)
    monkeypatch.setitem(sys.modules, "prepare_rho_student_comparison", preparer)
    root, frozen, acceptance, accepted, plan = fixtures.fixture(preparer)
    # The real comparison also binds the frozen metrics implementation.
    metric_file = frozen/"src/new_interest/experiments/sid_relation_bias.py"
    fixtures.put(metric_file, "# frozen metric implementation fixture\n")
    source = preparer.read(frozen/"source_manifest.json")
    source["files"]["src/new_interest/experiments/sid_relation_bias.py"] = preparer.sha(metric_file)
    fixtures.put(frozen/"source_manifest.json", source)
    pipe = load(repo/"scripts/analysis/run_rho_student_pipeline.py", "student_pipeline_fixture")
    pipe.ROOT, pipe.REPO = root, tmp_path
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: pytest.fail("No real command is authorized in controller tests"))
    if prepared:
        ready = preparer.prepare_comparison(acceptance, root/"bundle-comparison-v1",
            tmp_path/"configs/amlt/mind_rho_student_comparison_0916_v1.yaml", frozen_bundle=frozen)
        pipe.write(root/"comparison_ready.json", ready)
    return pipe, source, frozen, preparer


def comparison_report(pipe, source):
    plan = pipe.read(pipe.ROOT/"comparison_plan.json")
    results, provenance = {}, {}
    for split, records, users, targets, classes in (("validation", 5354, 5354, 5354, 107), ("dev", 31392, 27082, 37728, 104)):
        means = {arm: {metric: value for metric in pipe.METRICS} for arm, value in zip(pipe.ARMS, (.1, .2, .205))}
        comparisons = {}
        for contrast in pipe.CONTRASTS:
            newer, older = contrast.split("_minus_")
            delta = {metric: means[newer][metric]-means[older][metric] for metric in pipe.METRICS}
            intervals = {metric: [value-.01, value+.01] for metric, value in delta.items()}
            comparisons[contrast] = {"delta": delta, "delta_percentage_points": {m: v*100 for m, v in delta.items()},
                "ci95": intervals, "ci95_percentage_points": {m: [v*100 for v in limits] for m, limits in intervals.items()}}
        results[split] = {"records": records, "users": users, "target_occurrences": targets,
            "main_macro_classes": classes, "cross_arm_kernel_configs_consistent": True, "kernel_cache_keys_checked": 2,
            "bootstrap": {"unit": "user", "draws": 2000, "seed_per_split": 42, "observed_users": users,
                "sampled_users_per_draw": users, "usable_draws": 2000, "draws_missing_any_eligible_class": 2},
            "means": means, "comparisons": comparisons}
        provenance[split] = {"source_sha256": source["files"][f"inputs/{split}.parquet"]}
    results["validation"]["rare_validation_class"] = {"class": "Artificial Intelligence & Robotics", "support": 4,
        "users": 4, "included_in_primary_macro": False, "intervals_computed": False,
        "point_metrics": {arm: {"support": 4, "r10": .25, "r20": .5, "n10": .125, "n20": .25} for arm in pipe.ARMS}}
    return {"passed": True, "version": "rho-three-student-paired-user-bootstrap-v1",
        "acceptance_sha256": pipe.sha(pipe.ROOT/"three_arm_acceptance.json"),
        "comparison_code_sha256": plan["comparison_code_sha256"],
        "mapping_sha256": source["files"]["inputs/interest_token_mapping.parquet"],
        "metric_implementation_sha256": source["files"]["src/new_interest/experiments/sid_relation_bias.py"],
        "numpy_version": "2.2.6", "scipy_version": "1.15.3",
        "primary_comparison": {"split": "validation", "contrast": "dynamic_minus_static", "metric": "macro_ndcg_at_20",
            "dev_role": "consistency check under the same fixed contrast and metric"}, "results": results, "provenance": provenance}


def run_once(pipe, monkeypatch, state):
    pipe.write(pipe.ROOT/"pipeline_state.json", state)
    monkeypatch.setattr(sys, "argv", ["student-pipeline", "--once"])
    pipe.main()
    return pipe.read(pipe.ROOT/"pipeline_state.json")


def test_actual_comparison_schema_and_idempotent_human_report(tmp_path, monkeypatch):
    pipe, source, _, _ = pipeline(tmp_path, monkeypatch)
    report = comparison_report(pipe, source)
    pipe.check_comparison(report, source)
    output = pipe.write_report(report, pipe.ROOT/"compact/comparison.json")
    text = output.read_text()
    assert "Dynamic−Static" in text and "不足以证明动态项优于静态 OOF" in text
    assert "验证集稀有 AI 类" in text and "4 个用户样本" in text
    assert "+0.5000" in text
    modified = output.stat().st_mtime_ns
    assert pipe.write_report(report, pipe.ROOT/"compact/comparison.json") == output
    assert output.stat().st_mtime_ns == modified


@pytest.mark.parametrize("case", ["old_primary_schema", "wrong_code", "rows", "bootstrap_seed", "bootstrap_unit", "nan_mean", "wrong_percentage", "ci_scale", "rare_scope"])
def test_comparison_reader_rejects_identity_population_or_display_corruption(tmp_path, monkeypatch, case):
    pipe, source, _, _ = pipeline(tmp_path, monkeypatch)
    report = comparison_report(pipe, source)
    result = report["results"]["validation"]
    if case == "old_primary_schema": report["primary_comparison"].pop("dev_role")
    elif case == "wrong_code": report["comparison_code_sha256"] = "different"
    elif case == "rows": result["records"] = 5353
    elif case == "bootstrap_seed": result["bootstrap"]["seed_per_split"] = 43
    elif case == "bootstrap_unit": result["bootstrap"]["unit"] = "impression"
    elif case == "nan_mean": result["means"]["dynamic"]["macro_ndcg_at_20"] = float("nan")
    elif case == "wrong_percentage": result["comparisons"]["dynamic_minus_static"]["delta_percentage_points"]["macro_ndcg_at_20"] *= 100
    elif case == "ci_scale": result["comparisons"]["dynamic_minus_static"]["ci95_percentage_points"]["macro_ndcg_at_20"][0] *= 100
    elif case == "rare_scope": result["rare_validation_class"]["included_in_primary_macro"] = True
    with pytest.raises(ValueError):
        pipe.check_comparison(report, source)


@pytest.mark.parametrize("case", ["bundle_bytes", "extra_upload_file", "acceptance", "plan", "rehashed_gpu_command", "rehashed_ignore"])
def test_ready_receipt_cannot_adopt_changed_inputs_or_cpu_contract(tmp_path, monkeypatch, case):
    pipe, source, frozen, _ = pipeline(tmp_path, monkeypatch)
    ready = pipe.read(pipe.ROOT/"comparison_ready.json")
    pipe.check_comparison_ready(ready, source, frozen)
    if case == "bundle_bytes": (Path(ready["bundle"])/"inputs/dev.parquet").write_text("changed")
    elif case == "extra_upload_file": (Path(ready["bundle"])/"untracked_predictions.parquet").write_text("must not upload")
    elif case == "acceptance": (pipe.ROOT/"three_arm_acceptance.json").write_text("{}")
    elif case == "plan": (pipe.ROOT/"comparison_plan.json").write_text("{}")
    else:
        config = yaml.safe_load(Path(ready["config"]).read_text())
        if case == "rehashed_ignore":
            config["code"]["ignore"] = ["*"]
        else:
            config["jobs"][0]["command"].append("nvidia-smi")
        Path(ready["config"]).write_text(yaml.safe_dump(config))
        ready["config_sha256"] = pipe.sha(ready["config"])
    with pytest.raises(ValueError):
        pipe.check_comparison_ready(ready, source, frozen)


def test_recovers_prepared_bundle_when_ready_write_was_interrupted(tmp_path, monkeypatch):
    pipe, source, frozen, preparer = pipeline(tmp_path, monkeypatch)
    original = pipe.read(pipe.ROOT/"comparison_ready.json")
    (pipe.ROOT/"comparison_ready.json").unlink()
    monkeypatch.setattr(preparer, "prepare_comparison", lambda *args, **kwargs: pytest.fail("Must recover authenticated existing preparation"))
    recovered = pipe.prepare_or_recover_comparison(source, frozen)
    assert recovered == original
    assert pipe.read(pipe.ROOT/"comparison_ready.json") == original


def test_incomplete_local_preparation_is_preserved_for_inspection(tmp_path, monkeypatch):
    pipe, source, frozen, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    bundle = pipe.ROOT/"bundle-comparison-v1"
    bundle.mkdir()
    (bundle/"partial").write_text("preserve")
    with pytest.raises(ValueError, match="Partial comparison preparation"):
        pipe.prepare_or_recover_comparison(source, frozen)
    assert (bundle/"partial").read_text() == "preserve"


def test_dump_timeout_is_retryable_before_any_submission_attempt(tmp_path, monkeypatch):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch)
    calls = []

    def timeout(args, *rest, **kwargs):
        calls.append(args)
        assert "--dump" in args
        raise TimeoutError("local config preflight timed out")

    monkeypatch.setattr(pipe, "command", timeout)
    state = run_once(pipe, monkeypatch, {"phase": "comparison_submit"})
    assert state["phase"] == "comparison_submit" and "transport_error" in state
    assert not state.get("comparison_submission_attempted")
    monkeypatch.setattr(pipe, "command", lambda args, *rest, **kwargs: calls.append(args) or "accepted")
    monkeypatch.setattr(pipe, "statuses", lambda exp, jobs: {jobs[0]: "running"})
    state = run_once(pipe, monkeypatch, state)
    assert state["phase"] == "comparison_wait" and state["comparison_submission_attempted"] is True
    assert len([args for args in calls if "-y" in args]) == 1


@pytest.mark.parametrize("failure", [TimeoutError("lost response"), RuntimeError("backend/client failure")])
def test_actual_submission_failure_never_duplicates(tmp_path, monkeypatch, failure):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch)
    submissions = []

    def command(args, *rest, **kwargs):
        if "--dump" in args:
            return "valid"
        assert pipe.read(pipe.ROOT/"pipeline_state.json")["comparison_submission_attempted"] is True
        submissions.append(args)
        raise failure

    monkeypatch.setattr(pipe, "command", command)
    state = run_once(pipe, monkeypatch, {"phase": "comparison_submit"})
    assert state["phase"] == "needs_attention" and state["comparison_submission_attempted"]
    assert len(submissions) == 1
    assert run_once(pipe, monkeypatch, state) == state
    assert len(submissions) == 1


@pytest.mark.parametrize("status", ["failed", "killed", "canceled"])
def test_failed_comparison_stops_without_touching_training_or_collecting(tmp_path, monkeypatch, status):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe, "statuses", lambda exp, jobs: {jobs[0]: status})
    monkeypatch.setattr(pipe, "collect", lambda *args: pytest.fail("A failed comparison cannot be accepted"))
    state = run_once(pipe, monkeypatch, {"phase": "comparison_wait", "comparison_submission_attempted": True})
    assert state["phase"] == "needs_attention" and state["comparison_status"] == status


def test_download_retry_then_complete_with_actual_schema(tmp_path, monkeypatch):
    pipe, source, _, _ = pipeline(tmp_path, monkeypatch)

    def timeout(*args):
        raise TimeoutError("download transport")

    monkeypatch.setattr(pipe, "collect", timeout)
    state = run_once(pipe, monkeypatch, {"phase": "comparison_readback", "comparison_submission_attempted": True})
    assert state["phase"] == "comparison_readback" and "transport_error" in state
    comparison = pipe.ROOT/"compact/comparison/remote/comparison.json"
    pipe.write(comparison, comparison_report(pipe, source))
    monkeypatch.setattr(pipe, "collect", lambda *args: {"comparison.json": comparison})
    state = run_once(pipe, monkeypatch, state)
    assert state["phase"] == "complete" and state["paired_comparison_pending"] is False
    assert state["comparison_sha256"] == pipe.sha(comparison)
    assert state["report_sha256"] == pipe.sha(state["report"])


def test_report_io_retry_reuses_accepted_compact_result_without_download(tmp_path, monkeypatch):
    pipe, source, frozen, _ = pipeline(tmp_path, monkeypatch)
    comparison = pipe.ROOT/"compact/comparison/remote/comparison.json"
    pipe.write(comparison, comparison_report(pipe, source))
    monkeypatch.setattr(pipe, "collect", lambda *args: {"comparison.json": comparison})
    original = pipe.write_report

    def disk_error(*args):
        persisted = pipe.read(pipe.ROOT/"pipeline_state.json")
        assert persisted["phase"] == "comparison_report" and persisted["comparison_sha256"] == pipe.sha(comparison)
        raise OSError("temporary filesystem error")

    monkeypatch.setattr(pipe, "write_report", disk_error)
    state = run_once(pipe, monkeypatch, {"phase": "comparison_readback", "reason": "stale previous failure"})
    assert state["phase"] == "comparison_report" and "report_error" in state
    monkeypatch.setattr(pipe, "collect", lambda *args: pytest.fail("Accepted compact result must be reused"))
    monkeypatch.setattr(pipe, "write_report", original)
    state = run_once(pipe, monkeypatch, state)
    assert state["phase"] == "complete" and "report_error" not in state and "reason" not in state
    # Completion must remain idempotent even if old local bundles are pruned.
    shutil.rmtree(frozen)
    assert run_once(pipe, monkeypatch, state) == state


def test_cached_comparison_hash_is_rechecked_before_report(tmp_path, monkeypatch):
    pipe, source, _, _ = pipeline(tmp_path, monkeypatch)
    path = pipe.ROOT/"compact/comparison.json"
    pipe.write(path, comparison_report(pipe, source))
    state = {"phase": "comparison_report", "comparison": str(path), "comparison_sha256": pipe.sha(path)}
    path.write_text("{}")
    monkeypatch.setattr(pipe, "write_report", lambda *args: pytest.fail("Changed data cannot be reported"))
    assert run_once(pipe, monkeypatch, state)["phase"] == "needs_attention"


@pytest.mark.parametrize("bad", ["*.json", "x?.json", "x[01].json", "compact/report.json", "compact\\report.json", "scores.parquet"])
def test_collect_rejects_unsafe_or_noncompact_patterns(tmp_path, monkeypatch, bad):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    with pytest.raises(ValueError, match="compact JSON"):
        pipe.collect(pipe.COMPARISON, pipe.COMPARISON_JOB, "comparison", [bad])


def test_status_logs_are_scoped_to_this_controller(tmp_path, monkeypatch):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    seen = []

    def command(args, logfile, **kwargs):
        seen.append(logfile)
        return f":0 :{pipe.COMPARISON_JOB} 3m running 4m ago\n"

    monkeypatch.setattr(pipe, "command", command)
    assert pipe.statuses(pipe.COMPARISON, [pipe.COMPARISON_JOB]) == {pipe.COMPARISON_JOB: "running"}
    assert seen == [pipe.ROOT/f"{pipe.COMPARISON}-status.txt"]


def training_receipt(pipe, rule, arm, train_sha, *, parity):
    per_epoch = rule["optimizer_steps"] // rule["epochs"]
    controls = []
    if parity:
        for rank in range(rule["world_size"]):
            controls.append({"rank": rank, "parameters_unchanged": True, "optimizer_updates": 0,
                "comparisons": [{"scale": scale, "all_gradients_exact": True, "tensor_count": 373} for scale in (1., 1.5)],
                "trials": [{"label": label, "gradient_tensors": 373, "gradients_complete_and_finite": True,
                            "numeric_backend": {"deterministic_algorithms": True, "deterministic_warn_only": False}}
                           for label in pipe.PRODUCTION_PARITY_TRIALS]})
    student = {"passed": True, "arm": arm, "rows": rule["rows"], "classes": rule["classes"],
        "rows_per_class": rule["rows_per_class"], "optimizer_steps": rule["optimizer_steps"], "epochs": rule["epochs"],
        "initial_state_sha256": "same-initial", "train_file_sha256": train_sha,
        "class_weight_mass_preserved": True, "all_weights_positive": True, "weight_bounds": [.5, 1.5],
        "current_student_scoring_all_arms": True, "optimizer_reinitialized_at_refresh": False,
        "training_kernel_selection": "stable_fla_v1", "training_deterministic_algorithms": True,
        "training_deterministic_warn_only": False, "refresh_steps": [e * per_epoch for e in range(rule["epochs"])],
        "epoch_receipts": [{"epoch": e, "epoch_start_step": e * per_epoch,
            "new_optimizer_updates_during_refresh": 0, "optimizer_reinitialized": False} for e in range(rule["epochs"])],
        "rank_order_history": [[] for _ in range(rule["world_size"])], "native_gradient_parity": controls}
    return {"passed": True, "rows": rule["rows"], "steps": rule["optimizer_steps"], "epochs": rule["epochs"],
        "world_size": rule["world_size"], "fresh_reload_exact": True, "adapter_tensor_count": 373,
        "initial_state_sha256": "same-initial", "train_file_sha256": train_sha, "student_summary": student,
        "fresh_fp32_probes": {"accepted": True, "production_batch4_fp32_exact": True}}


def deterministic_worker(pipe, phase, arm):
    source = {"runtime_sha256": "runtime", "files": {"protocol.json": "protocol",
        "src/new_interest/experiments/rho_student_experiment.py": "reader",
        **{f"inputs/{name}.parquet": name for name in ("train", "single_train", "ddp_train", "dev", "validation")}}}
    protocol = {"training_numeric_policy": pipe.TRAINING_NUMERIC_POLICY, "reference_acceptance_sha256": "reference",
        "gate": {"single": {"rows": 32, "classes": 4, "rows_per_class": 8, "epochs": 3, "world_size": 1, "optimizer_steps": 12},
                 "ddp": {"rows": 64, "classes": 8, "rows_per_class": 8, "epochs": 3, "world_size": 4, "optimizer_steps": 6}},
        "evaluation": {"validation_rows": 5354, "dev_rows": 31392}}
    full = {"rows": 21600, "classes": 108, "rows_per_class": 200, "epochs": 3, "world_size": 4, "optimizer_steps": 2025}
    stages = ("single", "ddp") if phase == "gate" else ("full",)
    worker = {"passed": True, "identity": {"phase": phase, "arm": arm, "runtime_sha256": "runtime", "protocol_sha256": "protocol",
        "reference_acceptance_sha256": "reference", "inputs_sha256": pipe.canonical({k: v for k, v in source["files"].items() if k.startswith("inputs/")})},
        "training": {stage: training_receipt(pipe, protocol["gate"][stage] if phase == "gate" else full,
            arm, stage + "_train" if phase == "gate" else "train", parity=phase == "gate") for stage in stages}, "evaluations": {}}
    for split in (("dev",) if phase == "gate" else ("validation", "dev")):
        worker["evaluations"][split] = {"accepted": True, "rows": 64 if phase == "gate" else protocol["evaluation"][split+"_rows"],
            "source_sha256": split, "protocol_sha256": "protocol", "reader_sha256": "reader",
            "kernel_configs_consistent": True, "kernel_cache_keys_checked": 1,
            "metrics": {"main_107": {"supported_classes": 107}, "supported_classes": 104}}
    return worker, source, protocol


def test_worker_cannot_silently_accept_an_unplanned_storage_override(tmp_path, monkeypatch):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    worker, source, protocol = deterministic_worker(pipe, "full", "dynamic")
    worker["storage_retention_override"] = {"new_value": 25}
    with pytest.raises(ValueError, match="storage"):
        pipe.check_worker(worker, source, protocol, "full", "dynamic")


def storage_recovery_worker(tmp_path, monkeypatch, arm="dynamic"):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    worker, source, protocol = deterministic_worker(pipe, "full", arm)
    protocol["sft"] = {"epochs": 3, "seed": 42, "learning_rate": 2e-5,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "save_steps": 100, "save_total_limit": 3, "persistent_checkpoint_limit": 3}
    before = {**protocol["sft"], "rho_arm": arm,
        "rho_training_numeric_policy": pipe.TRAINING_NUMERIC_POLICY,
        "model_name": "/remote/frozen-model", "train_file": "/remote/frozen-train.parquet"}
    bundle = tmp_path/"storage-recovery-package"
    for name in ("recovery_manifest.json", "recovery_worker.py", "base/source_manifest.json"):
        path = bundle/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen " + name)
    expected = {"bundle": str(bundle), "arm": arm,
        "base_source_manifest_sha256": pipe.sha(bundle/"base/source_manifest.json"),
        "recovery_manifest_sha256": pipe.sha(bundle/"recovery_manifest.json"),
        "launcher_sha256": pipe.sha(bundle/"recovery_worker.py")}
    worker["storage_retention_override"] = {
        "version": "rho-storage-retention-recovery-v1", "arm": arm, "stage": "full",
        "base_runtime_sha256": source["runtime_sha256"], "base_protocol_sha256": source["files"]["protocol.json"],
        **{key: value for key, value in expected.items() if key not in ("bundle", "arm")},
        "config_before": before, "config_after": {**before, "persistent_checkpoint_limit": 25},
        "changed_fields": ["persistent_checkpoint_limit"], "old_value": 3, "new_value": 25,
        "planned_optimizer_steps": 2025, "planned_max_checkpoint_count": 21,
        "native_training_entrypoint_unchanged": True}
    return pipe, worker, source, protocol, expected


@pytest.mark.parametrize("arm", ["static", "dynamic"])
def test_planned_storage_recovery_preserves_full_training_acceptance(tmp_path, monkeypatch, arm):
    pipe, worker, source, protocol, expected = storage_recovery_worker(tmp_path, monkeypatch, arm)
    pipe.check_worker(worker, source, protocol, "full", arm, expected_storage_override=expected)
    assert worker["training"]["full"]["steps"] == 2025
    worker["training"]["full"]["fresh_fp32_probes"]["accepted"] = False
    with pytest.raises(ValueError, match="FP32 probes"):
        pipe.check_worker(worker, source, protocol, "full", arm, expected_storage_override=expected)


@pytest.mark.parametrize("case", ["missing_receipt", "missing_expected", "base_runtime", "protocol", "manifest",
    "launcher", "base_manifest", "after_learning_rate", "both_learning_rates", "both_accumulation",
    "after_input", "retention", "local_retention", "cadence", "step_budget", "count_bound", "entrypoint", "arm", "expected_arm"])
def test_storage_recovery_cannot_relax_scientific_or_source_contract(tmp_path, monkeypatch, case):
    pipe, worker, source, protocol, expected = storage_recovery_worker(tmp_path, monkeypatch)
    report = worker["storage_retention_override"]
    before, after = report["config_before"], report["config_after"]
    if case == "missing_receipt": worker.pop("storage_retention_override")
    elif case == "missing_expected": expected = None
    elif case == "base_runtime": report["base_runtime_sha256"] = "changed"
    elif case == "protocol": report["base_protocol_sha256"] = "changed"
    elif case == "manifest": report["recovery_manifest_sha256"] = "changed"
    elif case == "launcher": report["launcher_sha256"] = "changed"
    elif case == "base_manifest": report["base_source_manifest_sha256"] = "changed"
    elif case == "after_learning_rate": after["learning_rate"] *= 2
    elif case == "both_learning_rates": before["learning_rate"] = after["learning_rate"] = 4e-5
    elif case == "both_accumulation": before["gradient_accumulation_steps"] = after["gradient_accumulation_steps"] = 4
    elif case == "after_input": after["train_file"] = "/different/input.parquet"
    elif case == "retention": after["persistent_checkpoint_limit"] = report["new_value"] = 100
    elif case == "local_retention": before["save_total_limit"] = after["save_total_limit"] = 25
    elif case == "cadence": before["save_steps"] = after["save_steps"] = 25
    elif case == "step_budget": report["planned_optimizer_steps"] = 2026
    elif case == "count_bound": report["planned_max_checkpoint_count"] = 20
    elif case == "entrypoint": report["native_training_entrypoint_unchanged"] = False
    elif case == "arm": report["arm"] = "static"
    elif case == "expected_arm": expected["arm"] = "static"
    with pytest.raises(ValueError, match="[Ss]torage"):
        pipe.check_worker(worker, source, protocol, "full", "dynamic", expected_storage_override=expected)


@pytest.mark.parametrize("name", ["recovery_manifest.json", "recovery_worker.py", "base/source_manifest.json"])
def test_storage_recovery_package_is_rechecked_at_final_readback(tmp_path, monkeypatch, name):
    pipe, worker, source, protocol, expected = storage_recovery_worker(tmp_path, monkeypatch)
    (Path(expected["bundle"])/name).write_text("changed after launch")
    with pytest.raises(ValueError, match="package changed"):
        pipe.check_worker(worker, source, protocol, "full", "dynamic", expected_storage_override=expected)


@pytest.mark.parametrize("phase,arm", [("gate", "dynamic"), ("full", "uniform"), ("full", "static"), ("full", "dynamic")])
def test_new_gate_and_all_formal_arms_accept_the_same_deterministic_policy(tmp_path, monkeypatch, phase, arm):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    worker, source, protocol = deterministic_worker(pipe, phase, arm)
    pipe.check_worker(worker, source, protocol, phase, arm)
    for field, value in (("training_deterministic_algorithms", False), ("training_deterministic_warn_only", True),
                         ("training_deterministic_algorithms", 1), ("training_deterministic_warn_only", 0)):
        changed = copy.deepcopy(worker)
        next(iter(changed["training"].values()))["student_summary"][field] = value
        with pytest.raises(ValueError, match="training contract differs"):
            pipe.check_worker(changed, source, protocol, phase, arm)
    old_protocol = {k: v for k, v in protocol.items() if k != "training_numeric_policy"}
    with pytest.raises(ValueError, match="deterministic training policy"):
        pipe.check_worker(worker, source, old_protocol, phase, arm)


@pytest.mark.parametrize("case", ["algorithms_off", "warn_only", "missing_flags", "missing_trial", "duplicate_label",
    "missing_gradient", "nonfinite", "inexact_scale1", "inexact_scale1_5", "wrong_scale"])
def test_gate_requires_deterministic_production_trials_and_original_exact_gradient_controls(tmp_path, monkeypatch, case):
    pipe, _, _, _ = pipeline(tmp_path, monkeypatch, prepared=False)
    worker, source, protocol = deterministic_worker(pipe, "gate", "dynamic")
    item = worker["training"]["ddp"]["student_summary"]["native_gradient_parity"][3]
    trial = item["trials"][-1]
    if case == "algorithms_off": trial["numeric_backend"]["deterministic_algorithms"] = False
    elif case == "warn_only": trial["numeric_backend"]["deterministic_warn_only"] = True
    elif case == "missing_flags": trial["numeric_backend"] = {}
    elif case == "missing_trial": item["trials"].pop()
    elif case == "duplicate_label": trial["label"] = item["trials"][0]["label"]
    elif case == "missing_gradient": trial["gradient_tensors"] = 372
    elif case == "nonfinite": trial["gradients_complete_and_finite"] = False
    elif case == "inexact_scale1": item["comparisons"][0]["all_gradients_exact"] = False
    elif case == "inexact_scale1_5": item["comparisons"][1]["all_gradients_exact"] = False
    elif case == "wrong_scale": item["comparisons"][1]["scale"] = 2.
    with pytest.raises(ValueError):
        pipe.check_worker(worker, source, protocol, "gate", "dynamic")
