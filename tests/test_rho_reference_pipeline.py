from __future__ import annotations

import copy
import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def pipeline(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/run_rho_twofold_reference_pipeline.py"
    spec = importlib.util.spec_from_file_location("rho_reference_pipeline_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.ROOT = tmp_path
    return result


@pytest.mark.parametrize("prefix", [":", ":0 :", ""])
@pytest.mark.parametrize("body,state", [("queued 1m ago", "queued"), ("37m running 45m ago", "running"), ("1h pass 2h ago", "pass")])
def test_status_parser_accepts_only_exact_job_rows(tmp_path, monkeypatch, prefix, body, state):
    pipe = pipeline(tmp_path)
    job = pipe.GATE_JOB
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: f"Description {job} failed\n{prefix}{job} {body}\n")
    assert pipe.statuses(pipe.GATE_EXP, [job]) == {job: state}


@pytest.mark.parametrize("body", ["", "description JOB running", ":JOB running\n:JOB pass", ":JOB-other running", ":JOB unknown"])
def test_status_parser_rejects_missing_ambiguous_and_unknown_states(tmp_path, monkeypatch, body):
    pipe = pipeline(tmp_path)
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: body.replace("JOB", pipe.GATE_JOB))
    with pytest.raises(ValueError):
        pipe.statuses(pipe.GATE_EXP, [pipe.GATE_JOB])


@pytest.mark.parametrize("exception", [RuntimeError("CLI failed"), TimeoutError("network timeout")])
def test_status_command_failure_is_retryable(tmp_path, monkeypatch, exception):
    pipe = pipeline(tmp_path)

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(pipe, "command", fail)
    with pytest.raises(TimeoutError, match="Retryable status transport"):
        pipe.statuses(pipe.GATE_EXP, [pipe.GATE_JOB])


def source_fixture(pipe):
    bundle = pipe.ROOT / "bundle-gate-v1"
    (bundle / "inputs").mkdir(parents=True)
    for name, content in {
        "protocol.json": "{}", "source_snapshot.json": "{}", "token_audit.json": "{}",
        "inputs/fold0.parquet": "fold0", "inputs/fold1.parquet": "fold1",
        "inputs/interest_token_mapping.parquet": "mapping",
    }.items():
        (bundle / name).write_text(content)
    files = {str(p.relative_to(bundle)): pipe.sha(p) for p in bundle.rglob("*") if p.is_file()}
    source = {"runtime_sha256": "runtime", "files": files}
    pipe.write(bundle / "source_manifest.json", source)
    inputs = {k: v for k, v in files.items() if k.startswith("inputs/") or k in ("source_snapshot.json", "token_audit.json")}
    inputs_sha = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return source, inputs_sha


def compact_fixture(pipe, *, phase="full", fold=0, label="fold0"):
    source, inputs_sha = source_fixture(pipe)
    folder = pipe.ROOT / "compact" / label / "remote" / "compact"
    folder.mkdir(parents=True)
    train_sha = source["files"][f"inputs/fold{fold}.parquet"]
    score_sha = source["files"][f"inputs/fold{1-fold}.parquet"]
    acceptance = {
        "accepted": True, "teacher_train_sha256": train_sha,
        "score_file_sha256": score_sha,
        "mapping_sha256": source["files"]["inputs/interest_token_mapping.parquet"],
        "teacher_rows": 10800, "teacher_score_user_overlap": 0,
        "full_opposite_fold_coverage": True, "metrics": {"rows": 64 if phase == "gate" else 10800},
    }
    pipe.write(folder / "reference_acceptance.json", acceptance)
    success = {
        "passed": True, "runtime_sha256": source["runtime_sha256"],
        "identity": {
            "phase": phase, "fold": fold, "runtime_sha256": source["runtime_sha256"],
            "protocol_sha256": source["files"]["protocol.json"],
            "experiment_inputs_sha256": inputs_sha,
        },
        "reference_acceptance_sha256": pipe.sha(folder / "reference_acceptance.json"),
        "expected_oof_rows": 64 if phase == "gate" else 10800,
        "train_file_sha256": train_sha, "score_file_sha256": score_sha,
        "parent_fold_sha256": train_sha,
        "training": {"full": {"steps": 1014, "initial_state_sha256": "initial"}},
    }
    pipe.write(folder / "worker_success.json", success)
    if phase == "gate":
        pipe.write(folder / "gate_acceptance.json", success)
    return success, acceptance, folder


@pytest.mark.parametrize("phase", ["full", "gate"])
def test_collect_only_requests_compact_basename_allowlist_never_large_payloads(tmp_path, monkeypatch, phase):
    pipe = pipeline(tmp_path)
    label = "gate" if phase == "gate" else "fold0"
    success, _, folder = compact_fixture(pipe, phase=phase, label=label)
    commands = []
    monkeypatch.setattr(pipe, "command", lambda args, *rest, **kwargs: commands.append(args) or "")
    exp, job = (pipe.GATE_EXP, pipe.GATE_JOB) if phase == "gate" else (pipe.FULL_EXP, pipe.FULL_JOBS[0])
    assert pipe.collect(exp, job, label) == (success, folder)
    assert len(commands) == 1
    args = commands[0]
    patterns = [args[i+1] for i, value in enumerate(args) if value == "--include"]
    expected = {"environment.json", "snapshot_audit.json", "training_acceptance.json",
                "reference_acceptance.json", "worker_success.json"}
    if phase == "gate":
        expected |= {"token_gradient_audit.json", "gate_acceptance.json", "single_recovery_acceptance.json"}
    assert set(patterns) == expected
    assert len(patterns) == len(expected)
    assert all("/" not in pattern and "\\" not in pattern for pattern in patterns)

    # AzCopy --include-pattern applies to basenames. A path-shaped glob such as
    # *compact/*.json, or a literal compact/environment.json, is not supported.
    def included(name):
        return any(fnmatch.fnmatchcase(Path(name).name, pattern) for pattern in patterns)

    for name in expected:
        assert included(f"compact/{name}")
        assert included(f"remote/job/outputs/compact/{name}")
    for name in [
        "full/final_adapter/adapter_model.safetensors", "checkpoints-full/checkpoint-100/optimizer.pt",
        "oof/shard-0/scores.parquet", "results.tar.gz", "full/final_adapter/pytorch_model.bin",
        "oof/shard-0/manifest.json", "full/rank-0-training-runtime.json", "oof/full_scores.json",
        "checkpoints-full/checkpoint-100/scheduler.pt", "checkpoints-full/checkpoint-100/trainer_state.json",
        "oof/shard-0/shards/000000000-000000256.parquet", "full_predictions.zip", "results.tar",
    ]:
        assert not included(name), name
    if phase == "full":
        assert not included("compact/token_gradient_audit.json")


@pytest.mark.parametrize("field", ["runtime_sha256", "protocol_sha256", "experiment_inputs_sha256"])
def test_collect_rejects_report_from_foreign_frozen_source(tmp_path, monkeypatch, field):
    pipe = pipeline(tmp_path)
    success, _, folder = compact_fixture(pipe)
    if field == "runtime_sha256":
        success["runtime_sha256"] = "foreign"
    else:
        success["identity"][field] = "foreign"
    pipe.write(folder / "worker_success.json", success)
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: "")
    with pytest.raises(ValueError):
        pipe.collect(pipe.FULL_EXP, pipe.FULL_JOBS[0], "fold0")


def test_collect_rejects_independent_reader_for_different_training_bytes(tmp_path, monkeypatch):
    pipe = pipeline(tmp_path)
    success, acceptance, folder = compact_fixture(pipe)
    acceptance["teacher_train_sha256"] = "foreign-training-file"
    pipe.write(folder / "reference_acceptance.json", acceptance)
    success["reference_acceptance_sha256"] = pipe.sha(folder / "reference_acceptance.json")
    pipe.write(folder / "worker_success.json", success)
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: "")
    with pytest.raises(ValueError):
        pipe.collect(pipe.FULL_EXP, pipe.FULL_JOBS[0], "fold0")


@pytest.mark.parametrize("exception", [RuntimeError("CLI failed"), TimeoutError("network timeout")])
def test_compact_report_transport_failure_is_retryable(tmp_path, monkeypatch, exception):
    pipe = pipeline(tmp_path)

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(pipe, "command", fail)
    with pytest.raises(TimeoutError, match="Retryable compact-report transport"):
        pipe.collect(pipe.FULL_EXP, pipe.FULL_JOBS[0], "fold0")


def run_once(pipe, monkeypatch, state, extra_args=()):
    pipe.write(pipe.ROOT / "pipeline_state.json", state)
    monkeypatch.setattr(sys, "argv", ["rho-pipeline", "--once", *extra_args])
    pipe.main()
    return pipe.read(pipe.ROOT / "pipeline_state.json")


@pytest.mark.parametrize("gate_status", ["failed", "killed", "canceled"])
def test_failed_gate_stops_immediately_without_collect_retry_or_formal_submit(tmp_path, monkeypatch, gate_status):
    pipe = pipeline(tmp_path)
    calls = []

    def observed(exp, jobs):
        calls.append((exp, jobs))
        return {pipe.GATE_JOB: gate_status}

    monkeypatch.setattr(pipe, "statuses", observed)
    monkeypatch.setattr(pipe, "collect", lambda *args: pytest.fail("A failed gate must not collect missing acceptance"))
    monkeypatch.setattr(pipe, "prepare_full", lambda *args: pytest.fail("A failed gate must not stage formal inputs"))
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: pytest.fail("A failed gate must not resubmit"))
    monkeypatch.setattr(pipe.time, "sleep", lambda *args: pytest.fail("A failed gate must not enter a retry wait"))
    result = run_once(pipe, monkeypatch, {"phase": "gate_wait", "formal_submission_attempted": False})
    assert result["phase"] == "needs_attention"
    assert result["gate_status"] == gate_status
    assert result["formal_submission_attempted"] is False
    assert "Gate ended without acceptance" in result["reason"]
    assert "transport_error" not in result
    assert calls == [(pipe.GATE_EXP, [pipe.GATE_JOB])]
    # Restarting the finite supervisor preserves the terminal state rather than
    # polling the failed gate or treating its missing report as a transport retry.
    assert run_once(pipe, monkeypatch, result) == result
    assert len(calls) == 1


@pytest.mark.parametrize("version,extra_args", [(1, ()), (1, ("--gate-version", "1")), (2, ("--gate-version", "2"))])
def test_main_gate_version_selects_matching_experiment_bundle_and_config(tmp_path, monkeypatch, version, extra_args):
    pipe = pipeline(tmp_path)
    assert pipe.GATE_EXP == "mind-rho-twofold-gate-v1-0915"
    assert pipe.GATE_BUNDLE == "bundle-gate-v1"
    assert pipe.GATE_CONFIG == "mind_rho_twofold_gate_0915_v1.yaml"
    calls = []

    def observed(exp, jobs):
        calls.append((exp, jobs))
        return {pipe.GATE_JOB: "running"}

    monkeypatch.setattr(pipe, "statuses", observed)
    monkeypatch.setattr(pipe, "collect", lambda *args: pytest.fail("Running gate must not collect"))
    result = run_once(pipe, monkeypatch, {"phase": "gate_wait", "formal_submission_attempted": False}, extra_args)
    assert pipe.GATE_EXP == f"mind-rho-twofold-gate-v{version}-0915"
    assert pipe.GATE_BUNDLE == f"bundle-gate-v{version}"
    assert pipe.GATE_CONFIG == f"mind_rho_twofold_gate_0915_v{version}.yaml"
    assert calls == [(pipe.GATE_EXP, [pipe.GATE_JOB])]
    assert result["phase"] == "gate_wait" and result["gate_status"] == "running"


def test_gate_receipt_cannot_be_replaced_between_verification_and_staging(tmp_path, monkeypatch):
    pipe = pipeline(tmp_path)
    success, _, folder = compact_fixture(pipe, phase="gate", label="gate")
    pipe.write(folder / "gate_acceptance.json", {**success, "passed": False})
    monkeypatch.setattr(pipe, "collect", lambda *args: (success, folder))
    monkeypatch.setattr(pipe, "prepare_full", lambda *args: pytest.fail("Wrong gate must not stage formal inputs"))
    result = run_once(pipe, monkeypatch, {"phase": "gate_readback", "formal_submission_attempted": False})
    assert result["phase"] == "needs_attention"
    assert "Gate marker differs" in result["reason"]


def test_formal_staging_preserves_gate_input_bytes_and_two_directions(tmp_path, monkeypatch):
    import yaml

    pipe = pipeline(tmp_path)
    _, _, folder = compact_fixture(pipe, phase="gate", label="gate")
    before = pipe.read(tmp_path / "bundle-gate-v1/source_manifest.json")
    repo = tmp_path / "repo"
    configs = repo / "configs/amlt"
    configs.mkdir(parents=True)
    (configs / "mind_rho_twofold_gate_0915_v1.yaml").write_text(yaml.safe_dump({
        "code": {"local_dir": "original"},
        "jobs": [{"name": pipe.GATE_JOB, "sku": "80G4-A100", "sla_tier": "Premium",
                  "command": ["set -euo pipefail", "bash scripts/amlt/sft/run_rho_twofold_reference.sh gate 0"]}],
    }))
    pipe.REPO = repo
    commands = []
    monkeypatch.setattr(pipe, "command", lambda args, *rest, **kwargs: commands.append(args) or "")
    config_path = pipe.prepare_full(folder)
    after = pipe.read(tmp_path / "bundle-full-v1/source_manifest.json")
    assert after["runtime_sha256"] == before["runtime_sha256"]
    assert set(after["files"]) == set(before["files"]) | {"gate_acceptance.json"}
    for name, digest in before["files"].items():
        assert after["files"][name] == digest
        assert pipe.sha(tmp_path / "bundle-full-v1" / name) == digest
    jobs = yaml.safe_load(config_path.read_text())["jobs"]
    assert [job["name"] for job in jobs] == pipe.FULL_JOBS
    assert all(job["sku"] == "80G4-A100" and job["sla_tier"] == "Premium" for job in jobs)
    assert [job["command"][-1] for job in jobs] == [
        f"bash scripts/amlt/sft/run_rho_twofold_reference.sh full {i}" for i in range(2)
    ]
    assert len(commands) == 1 and commands[0][-1] == "--dump"
    with pytest.raises(ValueError, match="existing formal bundle"):
        pipe.prepare_full(folder)


@pytest.mark.parametrize("phase", ["gate_wait", "formal_wait"])
def test_pipeline_status_failure_preserves_phase_without_resubmit(tmp_path, monkeypatch, phase):
    pipe = pipeline(tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError("Retryable status transport")

    monkeypatch.setattr(pipe, "statuses", fail)
    monkeypatch.setattr(pipe, "command", lambda *args, **kwargs: pytest.fail("Status failure must not submit"))
    result = run_once(pipe, monkeypatch, {"phase": phase, "formal_submission_attempted": phase == "formal_wait"})
    assert result["phase"] == phase
    assert "transport_error" in result


@pytest.mark.parametrize("already_attempted", [False, True])
def test_ambiguous_submission_never_automatically_duplicates(tmp_path, monkeypatch, already_attempted):
    pipe = pipeline(tmp_path)
    calls = []

    def timeout(args, *rest, **kwargs):
        persisted = pipe.read(pipe.ROOT / "pipeline_state.json")
        assert persisted["formal_submission_attempted"] is True
        calls.append(args)
        raise TimeoutError("Submission response was lost")

    monkeypatch.setattr(pipe, "command", timeout)
    result = run_once(pipe, monkeypatch, {"phase": "formal_submit", "formal_submission_attempted": already_attempted,
                                       "formal_config": "/tmp/formal.yaml"})
    assert result["phase"] == "needs_attention"
    assert len(calls) == (0 if already_attempted else 1)
    assert result["formal_submission_attempted"] is True


@pytest.mark.parametrize("bad", [None, "same_half", "different_initialization", "different_inputs", "wrong_fold", "short_training"])
def test_final_twofold_acceptance_checks_directions_initialization_and_scope(tmp_path, monkeypatch, bad):
    pipe = pipeline(tmp_path)
    successes = []
    for fold in range(2):
        successes.append({
            "identity": {"fold": fold, "phase": "full", "experiment_inputs_sha256": "inputs"},
            "expected_oof_rows": 10800,
            "training": {"full": {"steps": 1014, "initial_state_sha256": "initial"}},
            "train_file_sha256": f"half-{fold}", "score_file_sha256": f"half-{1-fold}",
        })
    if bad == "same_half":
        successes[1]["score_file_sha256"] = "half-1"
    elif bad == "different_initialization":
        successes[1]["training"]["full"]["initial_state_sha256"] = "different"
    elif bad == "different_inputs":
        successes[1]["identity"]["experiment_inputs_sha256"] = "different"
    elif bad == "wrong_fold":
        successes[1]["identity"]["fold"] = 0
    elif bad == "short_training":
        successes[1]["training"]["full"]["steps"] = 1000
    monkeypatch.setattr(pipe, "collect", lambda exp, job, label: (copy.deepcopy(successes[pipe.FULL_JOBS.index(job)]), tmp_path))
    result = run_once(pipe, monkeypatch, {"phase": "formal_readback", "formal_submission_attempted": True})
    if bad is None:
        assert result["phase"] == "complete"
        accepted = pipe.read(tmp_path / "twofold_acceptance.json")
        assert accepted["rows"] == 21600
        assert accepted["opposite_halves_exact"] is True
    else:
        assert result["phase"] == "needs_attention"
        assert not (tmp_path / "twofold_acceptance.json").exists()
