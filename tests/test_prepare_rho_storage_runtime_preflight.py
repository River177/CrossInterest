from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_rho_storage_recovery import preparation, prepared


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def preflight():
    file = REPO / "scripts/analysis/prepare_rho_storage_runtime_preflight.py"
    spec = importlib.util.spec_from_file_location("storage_runtime_preflight_preparer", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_worker(preflight, package, tmp_path, *, name="run", old_bytes=None):
    outside = tmp_path / name
    outside.mkdir()
    script = outside / "runtime_preflight.py"
    script.write_text(preflight.worker_source())
    old = outside / "original_v1.py"
    old.write_bytes(preflight.OLD_GUARD_FIXTURE.read_bytes() if old_bytes is None else old_bytes)
    output = outside / "receipt.json"
    manifest = json.loads((package / "recovery_manifest.json").read_text())
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(script), "--package", str(package),
         "--old-guard", str(old), "--manifest-sha256", preflight.sha(package / "recovery_manifest.json"),
         "--launcher-sha256", manifest["launcher_sha256"], "--output", str(output), "--wait-seconds", "0"],
        capture_output=True, text=True, timeout=30,
    )
    return process, json.loads(output.read_text()) if output.exists() else None


def runtime_logs(package, *, populated=True):
    logs = package / "outputs/logs"
    logs.mkdir(parents=True)
    if populated:
        (logs / "platform-runner.log").write_text("AMLT runtime log fixture, content is not consumed by the new guard\n")
    return logs


def replace_launcher(preflight, package, suffix):
    launcher = package / "recovery_worker.py"
    launcher.write_text(launcher.read_text() + "\n" + suffix)
    manifest_file = package / "recovery_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["launcher_sha256"] = manifest["files"]["recovery_worker.py"] = preflight.sha(launcher)
    manifest_file.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def test_actual_payload_old_inventory_red_new_runtime_green(preparation, preflight, tmp_path):
    _, helper, _, package, _, _ = prepared(preparation)
    runtime_logs(package)
    result, report = launch_worker(preflight, package, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["passed"] and report["new_guard_passed_before"] and report["new_guard_passed_after"]
    assert report["old_guard_rejection"]["message"] == preflight.OLD_REJECTION
    assert report["old_guard_rejection"]["original_inventory_branch"] is True
    assert report["old_guard_sha256"] == preflight.OLD_GUARD_SHA256
    assert report["strict_packaging_rejection"]["extra"] == ["outputs"]
    assert report["old_inventory_diff"]["extra_paths"] == ["outputs/logs/platform-runner.log"]
    assert report["old_inventory_diff"]["missing"] == []
    assert report["protected_changed_paths"] == []
    assert report["package_tree_before_sha256"] == report["package_tree_after_sha256"]
    assert report["base_tree_before_sha256"] == report["base_tree_after_sha256"]
    assert report["runtime_environment"]["failure_witness_source"] == "preexisting runtime log filename"
    assert report["runtime_environment"]["probe_file"] is None
    for phase in ("new_before", "new_after"):
        assert report["filesystem_audit"][phase] == {"runtime_directory_enumerations": 0, "runtime_file_opens": 0}
    assert report["filesystem_audit"]["old_guard"]["runtime_directory_enumerations"] >= 1
    assert report["torch_modules_before"] == report["torch_modules_after"] == []
    assert report["forbidden_modules_after"] == report["forbidden_import_attempts"] == []
    assert report["subprocess_or_network_attempts"] == []
    assert not list(package.rglob("*.pyc"))


def test_empty_existing_runtime_logs_uses_explicitly_labeled_probe(preparation, preflight, tmp_path):
    _, _, _, package, _, _ = prepared(preparation)
    runtime_logs(package, populated=False)
    result, report = launch_worker(preflight, package, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = report["runtime_environment"]
    assert environment["preexisting_regular_log_file_count"] == 0
    assert environment["failure_witness_source"] == "explicit preflight probe"
    assert environment["runtime_directories_created_by_preflight"] is False
    assert environment["probe_file"].startswith("outputs/logs/rho_storage_guard_preflight_probe_")
    assert json.loads((package / environment["probe_file"]).read_text())["source"] == "CPU preflight generated probe"


def test_missing_runtime_is_not_fabricated(preparation, preflight, tmp_path):
    _, _, _, package, _, _ = prepared(preparation)
    process, report = launch_worker(preflight, package, tmp_path)
    assert process.returncode == 1 and not report["passed"]
    assert "no runtime directories were fabricated" in report["error"]["message"]
    assert not (package / "outputs").exists()


def test_runtime_root_symlink_is_rejected(preparation, preflight, tmp_path):
    _, _, _, package, _, _ = prepared(preparation)
    external = tmp_path / "external-runtime"
    (external / "logs").mkdir(parents=True)
    (external / "logs/platform.log").write_text("external")
    (package / "outputs").symlink_to(external, target_is_directory=True)
    process, report = launch_worker(preflight, package, tmp_path)
    assert process.returncode == 1 and not report["passed"]
    assert "non-symlink" in report["error"]["message"]


def test_packaging_rejects_even_empty_outputs_and_runtime_scopes_exception(preparation):
    _, helper, _, package, _, _ = prepared(preparation)
    (package / "outputs").mkdir()
    with pytest.raises(helper.PackageIntegrityError) as error:
        helper.verify_package(package)
    assert error.value.details["extra"] == ["outputs"]
    assert error.value.details["invalid_nodes"]["outputs"] == "root_outputs_requires_real_directory_and_runtime_mode"
    helper.verify_package(package, runtime=True)
    (package / "base/outputs").mkdir()
    with pytest.raises(helper.PackageIntegrityError):
        helper.verify_package(package, runtime=True)


@pytest.mark.parametrize("mutation", ["unknown-symlink", "unknown-empty-directory", "modified-base"])
def test_runtime_guard_still_rejects_unrelated_package_changes(preparation, mutation, tmp_path):
    _, helper, _, package, _, _ = prepared(preparation)
    runtime_logs(package)
    if mutation == "unknown-symlink":
        (package / "unexpected").symlink_to(tmp_path / "missing")
    elif mutation == "unknown-empty-directory":
        (package / "unexpected").mkdir()
    else:
        (package / "base/requirements-worker.txt").write_text("changed")
    with pytest.raises(helper.PackageIntegrityError):
        helper.verify_package(package, runtime=True)


def test_audit_catches_enumeration_even_without_reading_log_contents(preparation, preflight, tmp_path):
    _, _, _, package, _, _ = prepared(preparation)
    runtime_logs(package)
    replace_launcher(preflight, package, '''
_original_verify_package = verify_package
def verify_package(package, expected_manifest_sha256=None, *, runtime=False):
    if runtime:
        list((Path(package) / "outputs").iterdir())
    return _original_verify_package(package, expected_manifest_sha256, runtime=runtime)
''')
    result, report = launch_worker(preflight, package, tmp_path)
    assert result.returncode == 1 and not report["passed"]
    assert report["error"]["stage"] == "new_before"
    assert report["filesystem_audit"]["new_before"]["runtime_directory_enumerations"] == 1
    assert "New guard accessed runtime outputs contents" in report["error"]["message"]


def test_audit_blocks_model_import_before_loading_torch(preparation, preflight, tmp_path):
    _, _, _, package, _, _ = prepared(preparation)
    runtime_logs(package)
    replace_launcher(preflight, package, "import torch\n")
    result, report = launch_worker(preflight, package, tmp_path)
    assert result.returncode == 1 and not report["passed"]
    assert report["forbidden_import_attempts"] == ["torch"]
    assert report["torch_modules_after"] == []


def test_old_guard_hash_cannot_be_replaced_by_a_reimplementation(preparation, preflight, tmp_path):
    _, _, _, package, _, _ = prepared(preparation)
    runtime_logs(package)
    result, report = launch_worker(preflight, package, tmp_path, old_bytes=b"raise ValueError('fake old guard')\n")
    assert result.returncode == 1 and not report["passed"]
    assert "exact original v1 source" in report["error"]["message"]


def test_generated_bootstrap_runs_actual_payload_without_modifying_bundle(preparation, preflight, tmp_path):
    _, helper, _, original, ready, _ = prepared(preparation)
    package = tmp_path / "runtime-code-copy"
    shutil.copytree(original, package)
    output = tmp_path / "generated"
    receipt = preflight.prepare(original, ready["recovery_manifest_sha256"], output)
    config = json.loads((output / "preflight.yaml").read_text())
    assert config["jobs"][0]["sku"] == "5C3" and config["jobs"][0]["sla_tier"] == "Basic"
    assert config["code"]["local_dir"] == "$CONFIG_DIR/../recovery"
    assert config["jobs"][0]["submit_args"]["env"]["AMLT_NO_RUNTIME_DEPS"] is True
    assert receipt["package_manifest_sha256"] == ready["recovery_manifest_sha256"]
    assert receipt["old_guard_sha256"] == hashlib.sha256(preflight.OLD_GUARD_FIXTURE.read_bytes()).hexdigest()
    assert receipt["embedded_payload_encoding"] == "zlib-level-9+base64"
    table_command = json.dumps(config["jobs"][0]["command"])
    assert receipt["command_table_utf16_code_units"] == len(table_command.encode("utf-16-le")) // 2
    assert receipt["command_table_utf16_code_units"] < 20_000
    runtime_logs(package)
    remote = tmp_path / "amlt-output"
    process = subprocess.run(["bash", "-c", config["jobs"][0]["command"][0]],
        env={**os.environ, "AMLT_CODE_DIR": str(package), "AMLT_OUTPUT_DIR": str(remote)},
        cwd=package, capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((remote / "compact/rho_storage_runtime_preflight.json").read_text())
    assert report["passed"]
    assert (remote / "preflight_assets/original_storage_guard_v1.py").read_bytes() == preflight.OLD_GUARD_FIXTURE.read_bytes()
    assert (remote / "preflight_assets/rho_storage_runtime_preflight.py").read_bytes() == preflight.worker_source().encode()
    assert not (package / "preflight_assets").exists()
    assert not (original / "outputs").exists()
    helper.verify_package(original, ready["recovery_manifest_sha256"])
    with pytest.raises(FileExistsError):
        preflight.prepare(original, ready["recovery_manifest_sha256"], output)


def test_oversized_azure_table_command_is_rejected_before_writing(preparation, preflight, tmp_path, monkeypatch):
    _, _, _, package, ready, _ = prepared(preparation)
    monkeypatch.setattr(preflight, "bootstrap_command", lambda *args: "x" * 20_000)
    output = tmp_path / "oversized-config"
    with pytest.raises(ValueError, match="Azure Table property size budget"):
        preflight.prepare(package, ready["recovery_manifest_sha256"], output)
    assert not output.exists()


def test_preparer_rejects_wrong_ready_hash_and_mutated_runtime_bundle(preparation, preflight, tmp_path):
    _, _, _, package, ready, _ = prepared(preparation)
    with pytest.raises(ValueError, match="immutable ready"):
        preflight.prepare(package, "0" * 64, tmp_path / "wrong")
    runtime_logs(package)
    with pytest.raises(ValueError, match="already-mutated runtime"):
        preflight.prepare(package, ready["recovery_manifest_sha256"], tmp_path / "changed")
