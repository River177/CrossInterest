from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gcr/oi_ni_meta_calibration_container.py"
SPEC = importlib.util.spec_from_file_location("oi_ni_meta_host_container", SCRIPT)
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": host.sha256(path)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    task = tmp_path.resolve() / "task"
    task.mkdir()
    monkeypatch.setattr(host, "TASK_ROOT", task)
    bundle = task / "code"
    data = bundle / "data/frozen"
    data.mkdir(parents=True)
    (data / "examples.bin").write_bytes(b"accepted original training data")
    write_json(data / "metadata.json", {"artifacts": {"examples.bin": {"sha256": host.sha256(data / "examples.bin")}}})
    config = {"data_dir": "data/frozen", "data_metadata_sha256": host.sha256(data / "metadata.json"),
              "inner_alpha_candidates": [0.0001, 0.001, 0.01], "calibration_steps": 27, "seed": 42}
    write_json(bundle / host.PROTOCOL, config)
    for relative in (*host.LEGACY_FILES.values(), host.WRAPPER, host.DISPATCHER, "requirements-oi-ni-meta-worker.txt"):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen bytes: {relative}\n")
    source_files = {name: host.sha256(bundle / name) for name in
                    (host.PROTOCOL, *host.LEGACY_FILES.values(), "requirements-oi-ni-meta-worker.txt")}
    write_json(bundle / "source_bundle_manifest.json", {"files": source_files})
    write_json(bundle / "recovery_parent.json", {"old_serial_cancellation": "preserved"})
    for constant, relative in {"SOURCE_MANIFEST_SHA": "source_bundle_manifest.json", "PROTOCOL_SHA": host.PROTOCOL,
                               "RECOVERY_SHA": "recovery_parent.json", "WRAPPER_SHA": host.WRAPPER,
                               "DISPATCHER_SHA": host.DISPATCHER, "REQUIREMENTS_SHA": "requirements-oi-ni-meta-worker.txt"}.items():
        monkeypatch.setattr(host, constant, host.sha256(bundle / relative))
    hashes = {"metadata.json": host.sha256(data / "metadata.json"), "examples.bin": host.sha256(data / "examples.bin"),
              "protocol": host.PROTOCOL_SHA, "worker_bundle": host.SOURCE_MANIFEST_SHA,
              **{key: host.sha256(bundle / relative) for key, relative in host.LEGACY_FILES.items()}}
    monkeypatch.setattr(host, "SOURCE_ID", digest(hashes))
    run_ids = {i: digest({"config": config, "mode": "meta", "alpha": config["inner_alpha_candidates"][i],
                         "steps": 27, "phase": "calibrate", "source_identity": host.SOURCE_ID,
                         "eval_dev": False, "selection_sha256": None, "budget_sha256": None}) for i in (1, 2)}
    monkeypatch.setattr(host, "RUN_IDS", run_ids)
    dispatch = {"version": "oi-ni-meta-calibration-arm-dispatch-v1", "legacy_source_identity": host.SOURCE_ID,
                "legacy_source_manifest_sha256": host.SOURCE_MANIFEST_SHA, "protocol_sha256": host.PROTOCOL_SHA,
                "allowed_alpha_indices": [1, 2], "scientific_protocol_unchanged": True,
                "parent_recovery_sha256": host.RECOVERY_SHA, "expected_run_identities": {str(k): v for k, v in run_ids.items()},
                "files": {host.WRAPPER: host.WRAPPER_SHA, host.DISPATCHER: host.DISPATCHER_SHA}}
    write_json(bundle / "dispatch_manifest.json", dispatch)
    monkeypatch.setattr(host, "DISPATCH_MANIFEST_SHA", host.sha256(bundle / "dispatch_manifest.json"))
    cache = task / "hf-cache"
    prefix = f"hub/models--Qwen--Qwen3.5-2B/snapshots/{host.REVISION}"
    snapshot = cache / prefix
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"accepted":"model config"}')
    (snapshot / "tokenizer.json").write_text('{"accepted":"tokenizer"}')
    (snapshot / "tokenizer_config.json").write_text('{"accepted":"tokenizer config"}')
    # Exercise normal Hugging Face symlinks into a sibling blobs directory.
    blob = snapshot.parents[1] / "blobs" / "original-weights"
    blob.parent.mkdir()
    blob.write_bytes(b"original accepted weights")
    (snapshot / "model-00001.safetensors").symlink_to("../../blobs/original-weights")
    write_json(snapshot / "model.safetensors.index.json", {"weight_map": {"weight": "model-00001.safetensors"}})
    monkeypatch.setattr(host, "MODEL_CONFIG_SHA", host.sha256(snapshot / "config.json"))
    accepted = write_json(task / "control/accepted_snapshot.json", {"model_name": host.MODEL, "revision": host.REVISION,
                          "files": {path.name: host.sha256(path) for path in snapshot.iterdir()}})
    monkeypatch.setattr(host, "ACCEPTED_SNAPSHOT_SHA", accepted["sha256"])
    cache_audit = write_json(task / "control/cache.json", {"version": "oi-ni-meta-model-cache-v1",
                             "cache_root": str(cache), "model_name": host.MODEL, "revision": host.REVISION,
                             "complete": True, "tokenizer_local_load_passed": True,
                             "files": {f"{prefix}/{path.name}": host.sha256(path) for path in snapshot.iterdir()}})
    image = {"prepared_id": "sha256:" + "b" * 64, "base_id": "sha256:" + "a" * 64,
             "base_reference": host.BASE_REFERENCE}
    image_config = {"User": "", "Env": ["HOME=/root", "PATH=/usr/local/bin:/usr/bin:/bin"]}
    prep_audit = write_json(task / "control/prep.json", {"version": "oi-ni-meta-container-preparation-v1",
                            "base_image_id": image["base_id"], "prepared_image_id": image["prepared_id"],
                            "image_config_sha256": digest(image_config), "cuda_available": False,
                            "gpu_visible_count": 0, "pip_check_passed": True, "packages": host.PACKAGES,
                            "original_wrapper_sha256": host.WRAPPER_SHA, "requirements_sha256": host.REQUIREMENTS_SHA,
                            "runtime": {"python_executable": "/usr/local/bin/python", "user_site": "/root/.local/site-packages",
                                        "user_site_enabled": True, "home": "/root", "image_config_user": "", "uid": 0, "gid": 0,
                                        "import_paths": {name: f"/root/.local/site-packages/{name}/__init__.py"
                                                         for name in ("torch", "transformers", "peft")}}})
    ended = "2026-01-01T00:00:00+00:00"
    native = write_json(task / "control/native.json", {"backend": host.AMLT_BACKENDS[1], "native_status": "Canceled",
                        "start_time_utc": None, "end_time_utc": ended})
    cancel = write_json(task / "control/cancel.json", {"version": "oi-ni-meta-terminal-cancel-v1",
                        "backend_run_id": host.AMLT_BACKENDS[1], "job_name": "qwen35-2b-exact-meta-calibrate-alpha1",
                        "alpha_index": 1, "legacy_run_identity": run_ids[1], "state": "Canceled",
                        "ended_at": ended, "started": False, "native_evidence": native})
    operation = {"version": host.VERSION, "alpha_index": 1, "legacy_source_identity": host.SOURCE_ID,
                 "legacy_run_identity": run_ids[1], "source_manifest_sha256": host.SOURCE_MANIFEST_SHA,
                 "dispatch_manifest_sha256": host.DISPATCH_MANIFEST_SHA, "bundle_dir": str(bundle),
                 "host": host.HOST, "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555", "image": image,
                 "container_name": "oi-ni-meta-alpha1", "preparation_audit": prep_audit, "cache_audit": cache_audit,
                 "accepted_snapshot_audit": accepted, "cancellation": cancel,
                 "amlt_source": {"backend_run_id": host.AMLT_BACKENDS[1], "job_name": "qwen35-2b-exact-meta-calibrate-alpha1"},
                 "paths": {"task_root": str(task), "output_root": str(task / "outputs/alpha1"), "hf_cache": str(cache),
                           "scratch": str(task / "scratch/alpha1"), "pip_cache": str(task / "pip-cache"),
                           "evidence_dir": str(task / "control/launch-alpha1")},
                 "host_tools": {name: host.sha256(SCRIPT.with_name(name)) for name in
                                ("oi_ni_meta_calibration_container.py", "run_oi_ni_meta_calibration_container.sh")}}
    manifest = task / "control/operation.json"
    write_json(manifest, operation)
    monkeypatch.setattr(host.socket, "gethostname", lambda: host.HOST)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    return manifest, operation, image_config


def mutate_artifact(operation, label, changes):
    path = Path(operation[label]["path"])
    value = host.load_json(path)
    value.update(changes)
    operation[label] = write_json(path, value)


def write_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": str(path), "sha256": host.sha256(path)}


@pytest.fixture
def prepared_v2(prepared, monkeypatch):
    manifest, operation, image_config = prepared
    folder = manifest.parent / "v2"
    before_bytes = (b"[01/01/2026 00:00:00 +00:00] Created: Reason: Created job\n"
                    b"[01/01/2026 00:00:01 +00:00] Queued: Reason: Added to scheduler queue in location westus3\n"
                    b"[01/01/2026 00:00:02 +00:00] ScheduleNotPossible: Reason: Not enough quota\n")
    after_bytes = before_bytes + (
        b"[01/01/2026 00:00:05 +00:00] Canceling: Reason: Canceling job due to user request\n"
        b"[01/01/2026 00:00:06 +00:00] Canceled: Reason: Deleted/canceled due to a user request.\n")
    refs = {key: write_bytes(folder / filename, value) for key, filename, value in (
        ("pre_cancel_lifecycle_evidence", "before.txt", before_bytes),
        ("terminal_lifecycle_evidence", "after.txt", after_bytes))}
    started = "2026-01-01T00:00:04.0000007+00:00"
    ended = "2026-01-01T00:00:07.0000009+00:00"
    for key, filename, state, observed, start, end, log_key, log_bytes in (
        ("pre_cancel_native_evidence", "before.json", "Queued", "2026-01-01T00:00:03+00:00", None, None,
         "pre_cancel_lifecycle_evidence", before_bytes),
        ("native_evidence", "terminal.json", "Canceled", "2026-01-01T00:00:08+00:00", started, ended,
         "terminal_lifecycle_evidence", after_bytes),
    ):
        refs[key] = write_json(folder / filename, {
            "backend": host.AMLT_BACKENDS[1], "native_status": state, "observed_at_utc": observed,
            "start_time_utc": start, "end_time_utc": end, "available_log_names": [host.LIFECYCLE_LOG],
            "logs": {host.LIFECYCLE_LOG: {"local_file": "/original/mac/path/not/available/on/gcr.txt",
                     "sha256": refs[log_key]["sha256"], "offset": 0, "remote_bytes": len(log_bytes)}},
        })
    refs["results_inventory_evidence"] = write_json(folder / "inventory.json", {
        "version": "oi-ni-meta-result-inventory-v1", "backend_run_id": host.AMLT_BACKENDS[1],
        "job_name": operation["amlt_source"]["job_name"], **host.CANCELLATION_V2_RESULT_SCOPES[1],
        "observed_at_utc": "2026-01-01T00:00:09+00:00", "completed_at_utc": "2026-01-01T00:00:10+00:00",
        "enumeration_complete": True, "files": [],
    })
    monkeypatch.setattr(host, "CANCELLATION_V2_EVIDENCE_SHAS", {
        1: {key: refs[key]["sha256"] for key in host.CANCELLATION_V2_EVIDENCE_SHAS[1]},
    })
    operation["cancellation"] = write_json(folder / "cancellation.json", {
        "version": "oi-ni-meta-terminal-cancel-v2", "backend_run_id": host.AMLT_BACKENDS[1],
        "job_name": operation["amlt_source"]["job_name"], "alpha_index": 1, "legacy_run_identity": host.RUN_IDS[1],
        "state": "Canceled", "ended_at": ended, "native_start_time_utc": started,
        "observation": "no_training_artifacts_observed", **refs,
    })
    write_json(manifest, operation)
    return manifest, operation, image_config


def rewrite_v2_evidence(prepared, monkeypatch, key, changes, *, repin=True):
    """Replace synthetic pinned inputs to test semantics separately from pins."""
    manifest, operation, _ = prepared
    cancel_path = Path(operation["cancellation"]["path"])
    cancel = host.load_json(cancel_path)
    path = Path(cancel[key]["path"])
    document = host.load_json(path)
    document.update(changes)
    cancel[key] = write_json(path, document)
    if repin and key in host.CANCELLATION_V2_EVIDENCE_SHAS[1]:
        pins = deepcopy(host.CANCELLATION_V2_EVIDENCE_SHAS)
        pins[1][key] = cancel[key]["sha256"]
        monkeypatch.setattr(host, "CANCELLATION_V2_EVIDENCE_SHAS", pins)
    operation["cancellation"] = write_json(cancel_path, cancel)
    write_json(manifest, operation)


def rewrite_v2_lifecycle(prepared, monkeypatch, key, value):
    manifest, operation, _ = prepared
    cancel_path = Path(operation["cancellation"]["path"])
    cancel = host.load_json(cancel_path)
    cancel[key] = write_bytes(Path(cancel[key]["path"]), value)
    operation["cancellation"] = write_json(cancel_path, cancel)
    native_key = "pre_cancel_native_evidence" if key == "pre_cancel_lifecycle_evidence" else "native_evidence"
    native = host.load_json(Path(cancel[native_key]["path"]))
    metadata = native["logs"][host.LIFECYCLE_LOG]
    metadata.update(sha256=cancel[key]["sha256"], remote_bytes=len(value))
    rewrite_v2_evidence(prepared, monkeypatch, native_key, {"logs": native["logs"]})


def fake_runner(operation, image_config, overrides=None):
    calls = []
    overrides = overrides or {}
    def run(command):
        calls.append(command)
        if tuple(command) in overrides:
            return overrides[tuple(command)]
        if command[:2] == ["docker", "info"]:
            return json.dumps({"Name": host.HOST, "Runtimes": {"nvidia": {}}, "ServerVersion": "test-only"})
        if command[:3] == ["docker", "image", "inspect"]:
            base = command[-1] == operation["image"]["base_id"]
            return json.dumps([{"Id": command[-1], "RootFS": {"Layers": ["base-layer"] + ([] if base else ["prepared-layer"])},
                                "RepoTags": [host.BASE_REFERENCE] if base else [], "Config": image_config}])
        if command[0] == "nvidia-smi":
            return (f"{operation['gpu_uuid']}, NVIDIA A100 80GB PCIe, 81920"
                    if "--query-gpu=uuid,name,memory.total" in command else "")
        if command[:3] == ["docker", "ps", "-a"]:
            return ""
        if command[:2] == ["docker", "run"]:
            return "c" * 64
        if command[:3] == ["docker", "container", "inspect"]:
            return json.dumps([{"Id": "c" * 64, "Image": operation["image"]["prepared_id"], "State": {"Status": "running"}}])
        raise AssertionError(command)
    return run, calls


def test_real_protocol_keeps_original_run_ids():
    config = json.loads((SCRIPT.parents[2] / host.PROTOCOL).read_text())
    for index, expected in host.RUN_IDS.items():
        assert digest({"config": config, "mode": "meta", "alpha": config["inner_alpha_candidates"][index],
                       "steps": 27, "phase": "calibrate", "source_identity": host.SOURCE_ID,
                       "eval_dev": False, "selection_sha256": None, "budget_sha256": None}) == expected


def test_complete_local_validation_preserves_frozen_bytes_and_supports_cache_symlinks(prepared):
    manifest, operation, _ = prepared
    before = {path: host.sha256(path) for path in Path(operation["bundle_dir"]).rglob("*") if path.is_file()}
    plan = host.validate_operation(manifest, 1)
    assert plan["config"]["calibration_steps"] == 27
    assert before == {path: host.sha256(path) for path in before}
    assert not Path(operation["paths"]["output_root"]).exists()


@pytest.mark.parametrize("index", [0, 3, -1, True])
def test_alpha0_and_invalid_indices_rejected_before_any_io(index):
    with pytest.raises(ValueError, match="Only alpha"):
        host.validate_operation(Path("/does/not/exist.json"), index)


@pytest.mark.parametrize("field,value", [("alpha_index", 2), ("legacy_run_identity", "0" * 64),
                                         ("legacy_source_identity", "0" * 64), ("dispatch_manifest_sha256", "0" * 64)])
def test_identity_substitution_is_rejected(prepared, field, value):
    manifest, operation, _ = prepared
    operation[field] = value
    write_json(manifest, operation)
    with pytest.raises(ValueError):
        host.validate_operation(manifest, 1)


@pytest.mark.parametrize("changes", [{"state": "CancelRequested"}, {"state": "Queued"}, {"ended_at": ""},
                                    {"ended_at": "tomorrow"}, {"ended_at": "2026-01-01"}, {"started": True},
                                    {"backend_run_id": "stoic_oven_mcnfd4v969"}, {"alpha_index": 2}])
def test_nonterminal_or_wrong_arm_cancellation_rejected(prepared, changes):
    manifest, operation, _ = prepared
    mutate_artifact(operation, "cancellation", changes)
    write_json(manifest, operation)
    with pytest.raises(ValueError):
        host.validate_operation(manifest, 1)


def test_claimed_cancellation_cannot_override_native_queued_receipt(prepared):
    manifest, operation, _ = prepared
    cancel_path = Path(operation["cancellation"]["path"])
    cancel = host.load_json(cancel_path)
    native_path = Path(cancel["native_evidence"]["path"])
    native = host.load_json(native_path)
    native.update(native_status="Queued", end_time_utc=None)
    cancel["native_evidence"] = write_json(native_path, native)
    operation["cancellation"] = write_json(cancel_path, cancel)
    write_json(manifest, operation)
    with pytest.raises(ValueError, match="Native cancellation"):
        host.validate_operation(manifest, 1)


def test_v2_retains_actual_native_start_and_uses_portable_complete_evidence(prepared_v2):
    manifest, operation, _ = prepared_v2
    plan = host.validate_operation(manifest, 1)
    receipt = plan["cancellation"]
    assert receipt["native_start_time_utc"] == "2026-01-01T00:00:04.0000007+00:00"
    assert "started" not in receipt
    assert receipt["observation"] == "no_training_artifacts_observed"
    assert not Path(operation["paths"]["output_root"]).exists()


@pytest.mark.parametrize("key", ["pre_cancel_native_evidence", "native_evidence", "pre_cancel_lifecycle_evidence",
                                 "terminal_lifecycle_evidence", "results_inventory_evidence"])
def test_v2_requires_each_evidence_reference(prepared_v2, key):
    manifest, operation, _ = prepared_v2
    cancel_path = Path(operation["cancellation"]["path"])
    cancel = host.load_json(cancel_path)
    del cancel[key]
    operation["cancellation"] = write_json(cancel_path, cancel)
    write_json(manifest, operation)
    with pytest.raises(ValueError, match="Missing cancellation evidence"):
        host.validate_operation(manifest, 1)


@pytest.mark.parametrize("changes", [
    {"started": False}, {"observation": "training_never_started"}, {"state": "CancelRequested"},
    {"native_start_time_utc": None}, {"native_start_time_utc": "2026-01-01T00:00:04.000000+00:00"},
    {"backend_run_id": host.AMLT_BACKENDS[2]}, {"job_name": "qwen35-2b-exact-meta-calibrate-alpha2"},
    {"alpha_index": 2}, {"legacy_run_identity": "0" * 64},
])
def test_v2_rejects_identity_or_observation_substitution(prepared_v2, changes):
    manifest, operation, _ = prepared_v2
    mutate_artifact(operation, "cancellation", changes)
    write_json(manifest, operation)
    with pytest.raises(ValueError):
        host.validate_operation(manifest, 1)


@pytest.mark.parametrize("key", ["pre_cancel_native_evidence", "native_evidence", "results_inventory_evidence"])
def test_v2_cannot_replace_reviewed_original_by_rehashing_a_new_summary(prepared_v2, monkeypatch, key):
    rewrite_v2_evidence(prepared_v2, monkeypatch, key, {"new_summary": True}, repin=False)
    with pytest.raises(ValueError, match="reviewed original evidence"):
        host.validate_operation(prepared_v2[0], 1)


@pytest.mark.parametrize("key,changes", [
    ("pre_cancel_native_evidence", {"native_status": "Running"}),
    ("pre_cancel_native_evidence", {"start_time_utc": "2026-01-01T00:00:00+00:00"}),
    ("pre_cancel_native_evidence", {"end_time_utc": "2026-01-01T00:00:01+00:00"}),
    ("pre_cancel_native_evidence", {"backend": host.AMLT_BACKENDS[2]}),
    ("native_evidence", {"backend": host.AMLT_BACKENDS[2]}),
    ("native_evidence", {"native_status": "CancelRequested"}),
    ("native_evidence", {"observed_at_utc": "2026-01-01T00:00:06+00:00"}),
    ("native_evidence", {"observation_error_type": "TransportError"}),
    ("pre_cancel_native_evidence", {"observed_at_utc": "2026-01-01T00:00:05+00:00"}),
    ("native_evidence", {"available_log_names": [host.LIFECYCLE_LOG, "user_logs/std_log.txt"]}),
    ("pre_cancel_native_evidence", {"available_log_names": [host.LIFECYCLE_LOG, "user_logs/retry_001/std_log.txt"]}),
])
def test_v2_native_semantic_validation_is_independent_of_pins(prepared_v2, monkeypatch, key, changes):
    rewrite_v2_evidence(prepared_v2, monkeypatch, key, changes)
    with pytest.raises(ValueError):
        host.validate_operation(prepared_v2[0], 1)


@pytest.mark.parametrize("changes", [
    {"files": [{"name": "alpha-1/checkpoint/step-000001.pt"}]}, {"files": {}},
    {"enumeration_complete": False}, {"enumeration_complete": 1},
    {"backend_run_id": host.AMLT_BACKENDS[2]}, {"job_name": "another-job"},
    {"storage_account": "another-account"}, {"container": "another-container"},
    {"prefix": "projects/sft4safety/amlt-results/nonexistent-empty-prefix/"},
    {"experiment": "another-experiment"}, {"observed_at_utc": "2026-01-01T00:00:06+00:00"},
    {"completed_at_utc": "2026-01-01T00:00:08+00:00"}, {"completed_at_utc": None},
    {"completed_at_utc": "2099-01-01T00:00:00+00:00"},
])
def test_v2_inventory_semantic_validation_is_independent_of_pins(prepared_v2, monkeypatch, changes):
    rewrite_v2_evidence(prepared_v2, monkeypatch, "results_inventory_evidence", changes)
    with pytest.raises(ValueError):
        host.validate_operation(prepared_v2[0], 1)


@pytest.mark.parametrize("field,value", [("offset", 1), ("offset", False), ("remote_bytes", 1),
                                        ("sha256", "0" * 64)])
def test_v2_rejects_incomplete_or_unbound_native_log(prepared_v2, monkeypatch, field, value):
    cancel = host.load_json(Path(prepared_v2[1]["cancellation"]["path"]))
    native = host.load_json(Path(cancel["native_evidence"]["path"]))
    native["logs"][host.LIFECYCLE_LOG][field] = value
    rewrite_v2_evidence(prepared_v2, monkeypatch, "native_evidence", {"logs": native["logs"]})
    with pytest.raises(ValueError, match="lifecycle"):
        host.validate_operation(prepared_v2[0], 1)


@pytest.mark.parametrize("problem", ["tamper", "prefix", "running", "unknown_line", "wrong_reason", "reverse", "no_newline"])
def test_v2_rejects_changed_or_non_cancellation_lifecycle(prepared_v2, monkeypatch, problem):
    cancel = host.load_json(Path(prepared_v2[1]["cancellation"]["path"]))
    path = Path(cancel["terminal_lifecycle_evidence"]["path"])
    value = path.read_bytes()
    if problem == "tamper":
        path.write_bytes(value + b"tampered")  # Keep every recorded hash unchanged.
    else:
        if problem == "prefix":
            value = value.replace(b"Not enough quota", b"different history")
        elif problem == "running":
            value = value.replace(b"Canceling: Reason:", b"Running: Reason:")
        elif problem == "unknown_line":
            value += b"optimizer step 1\n"
        elif problem == "wrong_reason":
            value = value.replace(b"due to user request", b"due to infrastructure error")
        elif problem == "reverse":
            lines = value.splitlines(keepends=True)
            value = b"".join(lines[:-2] + lines[-2:][::-1])
        else:
            value = value.rstrip(b"\n")
        rewrite_v2_lifecycle(prepared_v2, monkeypatch, "terminal_lifecycle_evidence", value)
    with pytest.raises(ValueError):
        host.validate_operation(prepared_v2[0], 1)


@pytest.mark.parametrize("event", [b"Running: Reason: Worker started", b"Preparing: Reason: Allocated",
                                   b"ScheduleSelected: ", b"Created: Reason: Created job"])
def test_v2_rejects_execution_or_unresolved_selection_before_cancel(prepared_v2, monkeypatch, event):
    cancel = host.load_json(Path(prepared_v2[1]["cancellation"]["path"]))
    before = Path(cancel["pre_cancel_lifecycle_evidence"]["path"]).read_bytes()
    after = Path(cancel["terminal_lifecycle_evidence"]["path"]).read_bytes()
    changed = before + b"[01/01/2026 00:00:03 +00:00] " + event + b"\n"
    rewrite_v2_lifecycle(prepared_v2, monkeypatch, "pre_cancel_lifecycle_evidence", changed)
    rewrite_v2_lifecycle(prepared_v2, monkeypatch, "terminal_lifecycle_evidence", changed + after[len(before):])
    with pytest.raises(ValueError, match="Pre-cancel lifecycle"):
        host.validate_operation(prepared_v2[0], 1)


def test_v1_stays_strict_with_filled_native_start_even_if_v2_fields_are_added(prepared, monkeypatch):
    manifest, operation, _ = prepared
    cancel_path = Path(operation["cancellation"]["path"])
    cancel = host.load_json(cancel_path)
    path = Path(cancel["native_evidence"]["path"])
    native = host.load_json(path)
    native["start_time_utc"] = "2025-12-31T23:59:59+00:00"
    cancel["native_evidence"] = write_json(path, native)
    cancel["observation"] = "no_training_artifacts_observed"
    cancel["enumeration_complete"] = True
    operation["cancellation"] = write_json(cancel_path, cancel)
    write_json(manifest, operation)
    with pytest.raises(ValueError, match="Native cancellation evidence"):
        host.validate_operation(manifest, 1)


def test_v2_launch_copies_every_evidence_byte_with_portable_inventory(prepared_v2):
    manifest, operation, image_config = prepared_v2
    plan = host.validate_operation(manifest, 1)
    runner, _ = fake_runner(operation, image_config)
    host.launch(plan, runner)
    folder = Path(operation["paths"]["evidence_dir"])
    copies = host.load_json(folder / "evidence_copies.json")["files"]
    assert {"pre_cancel_native.json", "native_cancellation.json", "pre_cancel_lifecycle.txt",
            "terminal_lifecycle.txt", "terminal_results_inventory.json"} <= set(copies)
    for name, reference in copies.items():
        assert (folder / name).read_bytes() == Path(reference["path"]).read_bytes()
        assert host.sha256(folder / name) == reference["sha256"]


@pytest.mark.parametrize("changes", [{"cuda_available": True}, {"gpu_visible_count": 1},
                                    {"packages": {**host.PACKAGES, "torch": "2.9.0"}},
                                    {"prepared_image_id": "sha256:" + "d" * 64}])
def test_missing_cpu_preparation_or_changed_numerical_environment_rejected(prepared, changes):
    manifest, operation, _ = prepared
    mutate_artifact(operation, "preparation_audit", changes)
    write_json(manifest, operation)
    with pytest.raises(ValueError):
        host.validate_operation(manifest, 1)


def test_changed_weights_and_rewritten_new_audit_cannot_replace_accepted_snapshot(prepared):
    manifest, operation, _ = prepared
    cache_path = Path(operation["cache_audit"]["path"])
    cache_audit = host.load_json(cache_path)
    relative = next(name for name in cache_audit["files"] if name.endswith("model-00001.safetensors"))
    path = Path(operation["paths"]["hf_cache"]) / relative
    path.write_bytes(b"changed weights")
    cache_audit["files"][relative] = host.sha256(path)
    operation["cache_audit"] = write_json(cache_path, cache_audit)
    write_json(manifest, operation)
    with pytest.raises(ValueError, match="original accepted snapshot"):
        host.validate_operation(manifest, 1)


def test_source_change_cannot_be_hidden_by_operation_manifest(prepared):
    manifest, operation, _ = prepared
    path = Path(operation["bundle_dir"]) / host.LEGACY_FILES["oi_ni_meta.py"]
    path.write_text("different numerical engine")
    with pytest.raises(ValueError, match="File hash mismatch"):
        host.validate_operation(manifest, 1)


def test_cache_symlink_cannot_escape_the_sealed_cache_even_with_identical_bytes(prepared, tmp_path):
    manifest, operation, _ = prepared
    audit = host.load_json(Path(operation["cache_audit"]["path"]))
    relative = next(name for name in audit["files"] if name.endswith("model-00001.safetensors"))
    shard = Path(operation["paths"]["hf_cache"]) / relative
    outside = tmp_path / "weights-outside-mounted-cache"
    outside.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its root"):
        host.validate_operation(manifest, 1)


def test_mutable_image_tag_is_rejected(prepared):
    manifest, operation, _ = prepared
    operation["image"]["prepared_id"] = "prepared:latest"
    write_json(manifest, operation)
    with pytest.raises(ValueError, match="immutable local image IDs"):
        host.validate_operation(manifest, 1)


def test_readonly_input_cannot_overlap_writable_mount(prepared):
    manifest, operation, _ = prepared
    operation["paths"]["output_root"] = str(Path(operation["bundle_dir"]) / "outputs")
    write_json(manifest, operation)
    with pytest.raises(ValueError, match="must not overlap"):
        host.validate_operation(manifest, 1)


def test_output_entire_root_must_be_fresh(prepared):
    manifest, operation, _ = prepared
    write_json(Path(operation["paths"]["output_root"]) / "alpha-0/completion.json", {"from_another_arm": True})
    with pytest.raises(ValueError, match="nonempty output_root"):
        host.validate_operation(manifest, 1)


def test_default_cli_never_calls_subprocess_or_creates_paths(prepared, monkeypatch, capsys):
    manifest, operation, _ = prepared
    monkeypatch.setattr(host.subprocess, "run", lambda *a, **k: pytest.fail("Default mode must be read-only"))
    host.main(["--manifest", str(manifest), "--alpha-index", "1"])
    result = json.loads(capsys.readouterr().out)
    assert result["validated"] is True and result["launched"] is False
    assert not Path(operation["paths"]["output_root"]).exists()
    assert not (host.TASK_ROOT / "control/gcr-calibration-launch-lock.json").exists()


def test_docker_command_preserves_original_wrapper_and_user_environment(prepared):
    manifest, operation, _ = prepared
    command = host.docker_command(host.validate_operation(manifest, 1))
    assert command[-5:] == ["--entrypoint", "/bin/bash", operation["image"]["prepared_id"], host.WRAPPER, "1"]
    assert "--pull=never" in command and command[command.index("--network") + 1] == "none"
    assert "PIP_NO_INDEX=1" in command and "AMLT_CODE_DIR=/workspace" in command and "AMLT_OUTPUT_DIR=/output" in command
    for target in ("/workspace", host.HF_DEST):
        assert any(value.endswith(f"dst={target},readonly") for value in command)
    for target in ("/output", host.SCRATCH_DEST, host.PIP_DEST):
        assert any(value.endswith(f"dst={target}") for value in command)
    assert not any(value in command for value in ("--rm", "--restart", "--privileged", "--user"))
    assert not any(value.startswith("HOME=") or "docker.sock" in value for value in command)


def test_launch_uses_one_gpu_and_keeps_container_and_separate_receipts(prepared):
    manifest, operation, image_config = prepared
    plan = host.validate_operation(manifest, 1)
    runner, calls = fake_runner(operation, image_config)
    result = host.launch(plan, runner)
    assert result["container_id"] == "c" * 64
    assert sum(command[:2] == ["docker", "run"] for command in calls) == 1
    assert (Path(operation["paths"]["evidence_dir"]) / "launch_receipt.json").is_file()
    assert not list(Path(operation["paths"]["output_root"]).iterdir())  # Host did not fabricate worker evidence.
    with pytest.raises(ValueError, match="task-global launch intent"):
        host.validate_operation(manifest, 1)
    with pytest.raises(FileExistsError):
        host.launch(plan, runner)


@pytest.mark.parametrize("problem", ["active_gpu", "stopped_container", "wrong_user", "different_image", "missing_layer"])
def test_host_preflight_rejects_duplicate_or_changed_execution_environment(prepared, problem):
    manifest, operation, image_config = prepared
    plan = host.validate_operation(manifest, 1)
    runner, calls = fake_runner(operation, image_config)
    def failing_runner(command):
        if problem == "active_gpu" and "--query-compute-apps=gpu_uuid,pid,process_name" in command:
            return "GPU-other, 1234, python"
        if problem == "stopped_container" and command[:3] == ["docker", "ps", "-a"]:
            return "stopped-container-id"
        value = runner(command)
        if command[:3] == ["docker", "image", "inspect"] and command[-1] == operation["image"]["prepared_id"]:
            image = json.loads(value)
            if problem == "wrong_user":
                image[0]["Config"]["User"] = "different-user"
            if problem == "different_image":
                image[0]["Id"] = "sha256:" + "f" * 64
            if problem == "missing_layer":
                image[0]["RootFS"]["Layers"] = ["unrelated-layer"]
            return json.dumps(image)
        return value
    with pytest.raises(ValueError):
        host.host_preflight(plan, failing_runner)
    assert not any(command[:2] == ["docker", "run"] for command in calls)


def test_launch_failure_keeps_lock_and_never_automatically_retries(prepared):
    manifest, operation, image_config = prepared
    plan = host.validate_operation(manifest, 1)
    runner, calls = fake_runner(operation, image_config)
    def losing_inspect(command):
        if command[:3] == ["docker", "container", "inspect"]:
            raise RuntimeError("Container launched, inspection transport failed")
        return runner(command)
    with pytest.raises(RuntimeError, match="Container launched"):
        host.launch(plan, losing_inspect)
    assert (Path(operation["paths"]["evidence_dir"]) / "launch_failure.json").is_file()
    with pytest.raises(FileExistsError):
        host.launch(plan, runner)
    assert sum(command[:2] == ["docker", "run"] for command in calls) == 1


def test_nonroot_image_without_output_permissions_fails_before_gpu_launch(prepared):
    manifest, operation, image_config = prepared
    prep = host.load_json(Path(operation["preparation_audit"]["path"]))
    prep["runtime"].update(uid=2147483000, gid=2147483000, image_config_user="2147483000:2147483000")
    image_config["User"] = prep["runtime"]["image_config_user"]
    prep["image_config_sha256"] = digest(image_config)
    operation["preparation_audit"] = write_json(Path(operation["preparation_audit"]["path"]), prep)
    write_json(manifest, operation)
    plan = host.validate_operation(manifest, 1)
    runner, calls = fake_runner(operation, image_config)
    with pytest.raises(ValueError, match="cannot write/traverse output_root"):
        host.launch(plan, runner)
    assert not any(command[:2] == ["docker", "run"] for command in calls)
    assert (Path(operation["paths"]["evidence_dir"]) / "launch_failure.json").is_file()


def test_other_manifest_and_name_cannot_bypass_task_global_launch_lock(prepared):
    manifest, operation, image_config = prepared
    first = host.validate_operation(manifest, 1)
    another = deepcopy(first)
    another["operation"]["container_name"] = "different-container-name"
    another["paths"]["evidence_dir"] = host.TASK_ROOT / "control/other-launch"
    runner, calls = fake_runner(operation, image_config)
    host.launch(first, runner)
    with pytest.raises(FileExistsError):
        host.launch(another, runner)
    assert sum(command[:2] == ["docker", "run"] for command in calls) == 1


@pytest.mark.parametrize("state", [{"Status": "exited", "ExitCode": 1},
                                  {"Status": "running", "OOMKilled": True},
                                  {"Status": "dead", "Error": "runtime failed"}])
def test_detached_container_immediate_failure_is_not_reported_as_success(prepared, state):
    manifest, operation, image_config = prepared
    plan = host.validate_operation(manifest, 1)
    runner, calls = fake_runner(operation, image_config)
    def already_failed(command):
        if command[:3] == ["docker", "container", "inspect"]:
            return json.dumps([{"Id": "c" * 64, "Image": operation["image"]["prepared_id"], "State": state}])
        return runner(command)
    with pytest.raises(ValueError, match="already failed during launch"):
        host.launch(plan, already_failed)
    evidence = Path(operation["paths"]["evidence_dir"])
    assert not (evidence / "launch_receipt.json").exists()
    assert host.load_json(evidence / "launch_failure.json")["container_id"] == "c" * 64
    assert sum(command[:2] == ["docker", "run"] for command in calls) == 1
