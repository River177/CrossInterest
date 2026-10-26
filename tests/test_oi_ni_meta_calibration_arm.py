from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from new_interest.experiments import oi_ni_meta_calibration_arm as arm


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture_bundle(tmp_path, monkeypatch):
    config = {"seed": 42, "world_size": 1, "inner_alpha_candidates": [0.0001, 0.001, 0.01],
              "calibration_steps": 27, "data_dir": "data/frozen"}
    config_path = tmp_path / "configs/experiment.json"
    write_json(config_path, config)
    files = {"configs/experiment.json": arm.sha256(config_path)}
    for relative in arm.LEGACY_FILES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"immutable legacy {relative}")
        files[relative] = arm.sha256(path)
    write_json(tmp_path / "source_bundle_manifest.json", {"version": "legacy", "files": files})
    path = tmp_path / arm.DISPATCHER_PATH
    path.write_text("new dispatcher, separately bound")
    source = arm.legacy_source_receipt(tmp_path, config_path, data_hashes={"metadata.json": "d" * 64})
    monkeypatch.setattr(arm, "ACCEPTED_LEGACY_SOURCE_ID", source["identity"])
    manifest = {"version": arm.DISPATCH_VERSION, "legacy_source_identity": source["identity"],
                "legacy_source_manifest_sha256": source["files"]["worker_bundle"],
                "protocol_sha256": source["files"]["protocol"], "allowed_alpha_indices": [1,2],
                "scientific_protocol_unchanged": True, "files": {arm.DISPATCHER_PATH: arm.sha256(path)},
                "expected_run_identities": {str(i): arm.calibration_identity(config, source["identity"], i) for i in (1,2)},
                "parent_recovery_sha256": None}
    manifest_path = tmp_path / "dispatch_manifest.json"
    write_json(manifest_path, manifest)
    return config_path, config, source, manifest_path


def test_real_original_protocol_retains_both_calibration_identities():
    config_path = Path(__file__).resolve().parents[1] / "configs/experiments/mind_oi_ni_meta_pilot_v1.json"
    config = json.loads(config_path.read_text())
    assert arm.calibration_identity(config, arm.ACCEPTED_LEGACY_SOURCE_ID, 1) == (
        "c72786e0457ca7d4e0202b48dbcf65d752bf6b09eb21b5c5e8f8d15a3ab17a07")
    assert arm.calibration_identity(config, arm.ACCEPTED_LEGACY_SOURCE_ID, 2) == (
        "acb003caf66a463d96f1e45e6031543ffb87041b9a1331311f1f09d073c375f8")


def test_source_formula_matches_legacy_main_and_excludes_dispatch_entry(tmp_path, monkeypatch):
    config_path, _, source, _ = fixture_bundle(tmp_path, monkeypatch)
    assert source["identity"] == hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    (tmp_path / arm.DISPATCHER_PATH).write_text("different operational dispatcher")
    again = arm.legacy_source_receipt(tmp_path, config_path, data_hashes={"metadata.json": "d" * 64})
    assert again == source
    assert arm.DISPATCHER_PATH not in source["files"]


def test_source_rejects_changed_legacy_core(tmp_path, monkeypatch):
    config_path, _, _, _ = fixture_bundle(tmp_path, monkeypatch)
    (tmp_path / arm.LEGACY_FILES["oi_ni_meta.py"]).write_text("changed core")
    with pytest.raises(ValueError, match="Frozen legacy source changed"):
        arm.legacy_source_receipt(tmp_path, config_path, data_hashes={"metadata.json": "d" * 64})


def test_dispatch_is_distinct_while_arm_run_identity_is_unchanged(tmp_path, monkeypatch):
    config_path, config, source, manifest = fixture_bundle(tmp_path, monkeypatch)
    a = arm.validate_dispatch(tmp_path, config_path, manifest, source, 1)
    b = arm.validate_dispatch(tmp_path, config_path, manifest, source, 2)
    assert a["legacy_run_identity"] == arm.calibration_identity(config, source["identity"], 1)
    assert a["dispatch_identity"] != a["legacy_run_identity"]
    assert a["dispatch_identity"] != b["dispatch_identity"]
    assert a["legacy_entry_point_reused"] is False
    assert a["alpha0_parameters_imported"] is False and a["alpha_selection_performed"] is False


@pytest.mark.parametrize("index", [0, 3, -1, True])
def test_completed_or_out_of_scope_alpha_cannot_be_dispatched(index):
    with pytest.raises(ValueError, match="remaining alpha"):
        arm.calibration_settings({"inner_alpha_candidates": [0.0001, 0.001, 0.01], "calibration_steps": 27}, "source", index)


def test_modified_dispatcher_is_not_hidden_behind_legacy_identity(tmp_path, monkeypatch):
    config_path, _, source, manifest = fixture_bundle(tmp_path, monkeypatch)
    (tmp_path / arm.DISPATCHER_PATH).write_text("changed dispatch code")
    with pytest.raises(ValueError, match="Dispatcher/launcher bytes changed"):
        arm.validate_dispatch(tmp_path, config_path, manifest, source, 1)


def test_wrong_source_or_wrong_expected_arm_identity_fails(tmp_path, monkeypatch):
    config_path, _, source, manifest_path = fixture_bundle(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="accepted legacy source"):
        arm.validate_dispatch(tmp_path, config_path, manifest_path, {**source, "identity": "other"}, 1)
    manifest = json.loads(manifest_path.read_text())
    manifest["expected_run_identities"]["1"] = "wrong"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="run identities"):
        arm.validate_dispatch(tmp_path, config_path, manifest_path, source, 1)


def test_parent_recovery_is_hash_bound_and_cannot_supply_model_parameters(tmp_path, monkeypatch):
    config_path, _, source, manifest_path = fixture_bundle(tmp_path, monkeypatch)
    parent = tmp_path / "recovery_parent.json"
    write_json(parent, {"legacy_source_identity": source["identity"], "reason": "SLA eviction", "accepted_alpha0": True})
    manifest = json.loads(manifest_path.read_text()); manifest["parent_recovery_sha256"] = arm.sha256(parent)
    write_json(manifest_path, manifest)
    report = arm.validate_dispatch(tmp_path, config_path, manifest_path, source, 1, parent)
    assert report["parent_recovery_sha256"] == arm.sha256(parent)
    with pytest.raises(ValueError, match="recovery receipt"):
        arm.validate_dispatch(tmp_path, config_path, manifest_path, source, 1)
    parent.write_text(parent.read_text() + " ")
    with pytest.raises(ValueError, match="recovery receipt"):
        arm.validate_dispatch(tmp_path, config_path, manifest_path, source, 1, parent)


@pytest.mark.parametrize("index", [1, 2])
def test_execution_calls_only_the_original_function_with_original_arguments(tmp_path, monkeypatch, index):
    _, config, source, _ = fixture_bundle(tmp_path, monkeypatch)
    calls = []
    def run_training(*args, **kwargs):
        calls.append((args, kwargs))
        return {"identity": arm.calibration_identity(config, source["identity"], index), "mode": "meta",
                "alpha": config["inner_alpha_candidates"][index], "steps": 27, "dev_evaluated": False}
    data, output = tmp_path / "data", tmp_path / "output"
    result = arm.execute_arm(SimpleNamespace(run_training=run_training), config, "snapshot", data, output, index, source["identity"])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (config, "snapshot", data, output / f"alpha-{index}")
    assert kwargs == {"mode": "meta", "alpha": config["inner_alpha_candidates"][index], "steps": 27,
                      "phase": "calibrate", "source_identity": source["identity"]}
    assert result["steps"] == 27
    assert not output.exists()  # Dispatcher did not synthesize/copy any checkpoint.


def test_unexpected_function_result_is_rejected(tmp_path, monkeypatch):
    _, config, source, _ = fixture_bundle(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unexpected calibration"):
        arm.execute_arm(SimpleNamespace(run_training=lambda *a, **k: {"identity": "wrong"}),
                        config, "snapshot", tmp_path, tmp_path / "out", 1, source["identity"])


def test_cli_rejects_alpha0_before_any_cuda_or_training(monkeypatch):
    monkeypatch.setattr("sys.argv", ["dispatcher", "--config", "unused", "--root", "unused", "--alpha-index", "0"])
    with pytest.raises(SystemExit) as result:
        arm.main()
    assert result.value.code == 2
