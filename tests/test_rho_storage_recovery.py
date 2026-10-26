from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n" if isinstance(value, dict) else value)


@pytest.fixture
def preparation(tmp_path):
    preparer = load(REPO / "scripts/analysis/prepare_rho_student_storage_recovery.py", "storage_preparer")
    helper = preparer.launcher_module()
    base = tmp_path / "frozen"
    protocol = {
        "version": "mind-rho-student-v1", "arms": ["uniform", "static", "dynamic"],
        "optimizer_steps": 2025, "gpus_per_arm": 4, "training_rows": 21600,
        "training_numeric_policy": "torch-deterministic-algorithms-v1",
        "sft": {"persistent_checkpoint_limit": 3, "save_total_limit": 3,
                "save_strategy": "steps", "save_steps": 100, "epochs": 3,
                "seed": 42, "learning_rate": 2e-5, "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8, "wandb_mode": "disabled"},
    }
    put(base / "protocol.json", protocol)
    put(base / "src/new_interest/experiments/rho_student_experiment.py", "# frozen worker fixture\n")
    put(base / "scripts/amlt/sft/run_rho_student.sh",
        (REPO / "scripts/amlt/sft/run_rho_student.sh").read_text())
    put(base / "requirements-worker.txt", "# pinned dependencies fixture\n")
    files = {p.relative_to(base).as_posix(): helper.sha(p) for p in base.rglob("*") if p.is_file()}
    runtime = {k: v for k, v in files.items()
               if k.startswith(("src/", "scripts/", "configs/")) or k == "requirements-worker.txt"}
    runtime_sha = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    put(base / "gate_acceptance.json", {"passed": True, "identity": {
        "phase": "gate", "runtime_sha256": runtime_sha, "protocol_sha256": helper.sha(base / "protocol.json")}})
    files["gate_acceptance.json"] = helper.sha(base / "gate_acceptance.json")
    put(base / "source_manifest.json", {"runtime_sha256": runtime_sha, "files": files})
    output = tmp_path / "recovery"
    return preparer, helper, base, output


def prepared(preparation):
    preparer, helper, base, output = preparation
    ready = preparer.prepare_bundle(base, output)
    return preparer, helper, base, output, ready, helper.verify_package(output, ready["recovery_manifest_sha256"])


def fake_worker(helper, context, *, change_before=None, fail=False, alter_saved=False, change_command=False):
    module = SimpleNamespace(command=object(), observed_commands=[], used_configs=[])

    def configuration(project, protocol, model, root, local, stage, arm, reference):
        value = copy.deepcopy(protocol["sft"])
        value.update(rho_arm=arm, rho_protocol_file=str(project / "protocol.json"),
                     rho_training_numeric_policy=protocol["training_numeric_policy"],
                     resume_from_checkpoint="auto", output_dir=str(local),
                     persistent_checkpoint_dir=str(root / "checkpoints-full"))
        if change_before:
            change_before(value)
        return value, {"optimizer_steps": 2025, "world_size": 4, "rows": 21600, "epochs": 3}

    module.make_training_config = configuration

    def worker(project, root, phase, arm):
        cfg, _ = module.make_training_config(project, context["protocol"], "model", root,
                                            root / "scratch", phase, arm, {})
        module.used_configs.append(cfg)
        if fail:
            raise RuntimeError("simulated native training interruption")
        if not (root / "full/artifact_manifest.json").exists():
            module.observed_commands.append(["python", "-m", "torch.distributed.run", "--standalone",
                                             "--nproc_per_node=4", "-m", "new_interest.training.rho_student_sft"])
            saved = copy.deepcopy(cfg)
            if alter_saved:
                saved["learning_rate"] *= 2
            helper.write(root / "full/resolved_config.json", saved)
            helper.write(root / "full/artifact_manifest.json", {"fixture": "closed"})
        if change_command:
            module.command = object()
        result = {"passed": True, "identity": {"arm": arm, "phase": phase}}
        helper.write(root / "compact/worker_success.json", result)
        return result

    module.worker = worker
    return module


def test_package_keeps_exact_base_and_original_install_prefix(preparation):
    preparer, helper, base, output, ready, context = prepared(preparation)
    assert ready["arms"] == ["static", "dynamic"] and ready["submitted"] is False
    assert ready["base_source_manifest_sha256"] == helper.sha(base / "source_manifest.json")
    for name in helper.inventory(base):
        assert (base / name).read_bytes() == (output / "base" / name).read_bytes()
    manifest = helper.read(output / "recovery_manifest.json")
    assert set(manifest["files"]) == helper.inventory(output) - {"recovery_manifest.json"}
    original = (base / helper.COMPAT_PATH).read_text()
    compat = (output / helper.COMPAT_PATH).read_text()
    original_install = original.split('cd "${AMLT_CODE_DIR:?}"\n', 1)[1].split('if [[ "${1:?}"', 1)[0]
    assert original_install in compat
    assert compat.count('python "$RHO_STORAGE_RECOVERY_PACKAGE_DIR/recovery_worker.py"') == 1
    assert "rho_student_experiment worker" not in compat
    assert '--phase "$1" --arm "$2"' in compat
    assert context["identity"]["recovery_manifest_sha256"] == ready["recovery_manifest_sha256"]
    subprocess.run(["bash", "-n", str(output / helper.COMPAT_PATH)], check=True)
    for args in ([], ["gate", "dynamic"], ["full", "uniform"]):
        result = subprocess.run(["bash", str(output / helper.COMPAT_PATH), *args],
                                env={"AMLT_CODE_DIR": str(output)}, capture_output=True)
        assert result.returncode == 2
        assert b"Storage recovery only supports" in result.stderr


@pytest.mark.parametrize("case", ["base_bytes", "launcher_bytes", "extra_file", "outer_hash", "wrong_expected_hash"])
def test_tampered_recovery_bundle_is_rejected_before_worker_import(preparation, case):
    _, helper, _, output, ready, _ = prepared(preparation)
    if case == "base_bytes":
        (output / "base/requirements-worker.txt").write_text("changed")
    elif case == "launcher_bytes":
        (output / "recovery_worker.py").write_text("raise RuntimeError('must not execute')")
    elif case == "extra_file":
        (output / "untracked.py").write_text("unbound")
    elif case == "outer_hash":
        manifest = helper.read(output / "recovery_manifest.json")
        manifest["compat_wrapper_sha256"] = "0" * 64
        helper.write(output / "recovery_manifest.json", manifest)
    expected = "f" * 64 if case == "wrong_expected_hash" else None
    with pytest.raises(ValueError):
        helper.verify_package(output, expected)


@pytest.mark.parametrize("case", ["source_change", "symlink", "overlap", "existing_output", "wrapper_shape"])
def test_invalid_inputs_do_not_replace_or_modify_frozen_material(preparation, case):
    preparer, helper, base, output = preparation
    original_protocol = (base / "protocol.json").read_bytes()
    if case == "source_change":
        (base / "requirements-worker.txt").write_text("changed")
    elif case == "symlink":
        target = base.parent / "elsewhere.txt"
        target.write_bytes((base / "requirements-worker.txt").read_bytes())
        (base / "requirements-worker.txt").unlink()
        (base / "requirements-worker.txt").symlink_to(target)
    elif case == "overlap":
        output = base / "nested"
    elif case == "existing_output":
        output.mkdir()
        (output / "preserve").write_text("existing")
    else:
        source = helper.read(base / "source_manifest.json")
        wrapper = base / helper.COMPAT_PATH
        wrapper.write_text(wrapper.read_text().replace("diagnose-native", "different-entry"))
        source["files"][helper.COMPAT_PATH] = helper.sha(wrapper)
        # A changed entrypoint is rejected even if the caller tries to supply it.
        helper.write(base / "source_manifest.json", source)
    with pytest.raises((ValueError, FileExistsError)):
        preparer.prepare_bundle(base, output)
    assert (base / "protocol.json").read_bytes() == original_protocol
    if case == "existing_output":
        assert (output / "preserve").read_text() == "existing"
    else:
        assert not output.exists()


@pytest.mark.parametrize("arm", ["static", "dynamic"])
def test_worker_changes_only_remote_retention_and_keeps_native_entrypoint(preparation, tmp_path, arm):
    _, helper, base, _, _, context = prepared(preparation)
    module = fake_worker(helper, context)
    original = module.make_training_config
    result = helper._run_worker_with_override(module, context, tmp_path / "run", arm=arm)
    receipt = result["storage_retention_override"]
    assert module.make_training_config is original
    assert receipt["arm"] == arm
    assert receipt["changed_fields"] == ["persistent_checkpoint_limit"]
    assert receipt["config_before"]["persistent_checkpoint_limit"] == 3
    assert receipt["config_after"]["persistent_checkpoint_limit"] == 25
    for key in receipt["config_before"]:
        if key != "persistent_checkpoint_limit":
            assert receipt["config_before"][key] == receipt["config_after"][key]
    assert receipt["config_after"]["save_total_limit"] == 3
    assert receipt["config_after"]["save_steps"] == 100
    assert receipt == helper.read(tmp_path / "run/compact/storage_retention_override.json")
    assert result == helper.read(tmp_path / "run/compact/worker_success.json")
    assert module.observed_commands[0][-1] == "new_interest.training.rho_student_sft"
    assert helper.read(base / "protocol.json")["sft"]["persistent_checkpoint_limit"] == 3


@pytest.mark.parametrize("field,value", [("learning_rate", 1e-3), ("gradient_accumulation_steps", 4),
                                         ("save_steps", 50), ("persistent_checkpoint_limit", 25)])
def test_worker_rejects_any_undeclared_before_config_change(preparation, tmp_path, field, value):
    _, helper, _, _, _, context = prepared(preparation)
    module = fake_worker(helper, context, change_before=lambda cfg: cfg.update({field: value}))
    original = module.make_training_config
    with pytest.raises(ValueError):
        helper._run_worker_with_override(module, context, tmp_path / "run", arm="dynamic")
    assert module.make_training_config is original
    assert module.observed_commands == []
    assert not (tmp_path / "run/compact/worker_success.json").exists()


@pytest.mark.parametrize("case", ["interruption", "wrong_saved_config", "changed_command"])
def test_native_failure_or_acceptance_mismatch_never_adds_override_success(preparation, tmp_path, case):
    _, helper, _, _, _, context = prepared(preparation)
    module = fake_worker(helper, context, fail=case == "interruption",
                         alter_saved=case == "wrong_saved_config", change_command=case == "changed_command")
    original = module.make_training_config
    root = tmp_path / "run"
    with pytest.raises((ValueError, RuntimeError)):
        helper._run_worker_with_override(module, context, root, arm="dynamic")
    assert module.make_training_config is original
    success = root / "compact/worker_success.json"
    assert not success.exists() or "storage_retention_override" not in helper.read(success)


def test_closed_training_reuse_keeps_actual_original_override_and_does_not_train_again(preparation, tmp_path):
    _, helper, _, _, _, context = prepared(preparation)
    first = fake_worker(helper, context)
    original = helper._run_worker_with_override(first, context, tmp_path / "run", arm="static")
    shutil.copytree(tmp_path / "run", tmp_path / "copied-run")
    second = fake_worker(helper, context)
    copied = helper._run_worker_with_override(second, context, tmp_path / "copied-run", arm="static")
    assert second.observed_commands == []
    assert copied["storage_retention_override"] == original["storage_retention_override"]


def test_uniform_is_not_implicitly_authorized(preparation, tmp_path):
    _, helper, _, _, _, context = prepared(preparation)
    module = fake_worker(helper, context)
    with pytest.raises(ValueError, match="explicitly supported"):
        helper._run_worker_with_override(module, context, tmp_path / "run", arm="uniform")
    assert module.observed_commands == []


def test_copy_failure_keeps_base_unchanged_and_does_not_publish_partial_package(preparation, monkeypatch):
    preparer, helper, base, output = preparation
    before = {name: (base / name).read_bytes() for name in helper.inventory(base)}
    original_copy = preparer.shutil.copyfile
    count = 0

    def interrupted_copy(source, destination):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("simulated local copy interruption")
        return original_copy(source, destination)

    monkeypatch.setattr(preparer.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="copy interruption"):
        preparer.prepare_bundle(base, output)
    assert not output.exists()
    assert not list(output.parent.glob(output.name + ".staging-*"))
    assert before == {name: (base / name).read_bytes() for name in helper.inventory(base)}


def test_code_package_refuses_declared_model_weights(preparation):
    preparer, helper, base, output = preparation
    put(base / "model.safetensors", "must stay remote")
    source = helper.read(base / "source_manifest.json")
    source["files"]["model.safetensors"] = helper.sha(base / "model.safetensors")
    helper.write(base / "source_manifest.json", source)
    with pytest.raises(ValueError, match="model or optimizer"):
        preparer.prepare_bundle(base, output)
    assert not output.exists()


def test_amlt_runtime_outputs_logs_are_outside_frozen_package(preparation):
    _, helper, _, output, ready, _ = prepared(preparation)
    # AMLT's real runner starts background log sync into this root runtime area.
    put(output / "outputs/logs/amlt_code_runner.txt", "runtime runner log\n")
    context = helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)
    assert context["identity"]["recovery_manifest_sha256"] == ready["recovery_manifest_sha256"]


def test_guard_reports_complete_extra_missing_changed_lists(preparation):
    _, helper, _, output, ready, _ = prepared(preparation)
    put(output / "unexpected.txt", "extra")
    (output / helper.COMPAT_PATH).unlink()
    (output / "base/requirements-worker.txt").write_text("changed source")
    with pytest.raises(ValueError) as caught:
        helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)
    details = caught.value.details
    assert details["extra"] == ["unexpected.txt"]
    assert details["missing"] == [helper.COMPAT_PATH]
    assert details["changed"] == ["base/requirements-worker.txt"]
    assert details["counts"] == {"extra": 1, "missing": 1, "changed": 1}


def test_first_guard_failure_writes_untrusted_no_training_diagnostic(preparation, tmp_path, monkeypatch):
    _, helper, _, output, ready, _ = prepared(preparation)
    put(output / "unbound.py", "must not execute")
    monkeypatch.setattr(helper.importlib, "import_module", lambda *args: pytest.fail("Guard failure must precede worker import"))
    root = tmp_path / "failed-run"
    with pytest.raises(ValueError):
        helper.run_worker(output, root, arm="dynamic", expected_manifest_sha256=ready["recovery_manifest_sha256"])
    report = helper.read(root / "compact/storage_recovery_preflight_failure.json")
    assert report["trusted"] is False and report["acceptance"] is False
    assert report["training_started"] is False and report["optimizer_updates"] == 0
    assert report["phase"] == "before_frozen_worker_import"
    assert report["details"]["extra"] == ["unbound.py"]
    assert not (root / "compact/worker_success.json").exists()


@pytest.mark.parametrize("with_log", [False, True])
def test_packing_mode_rejects_root_outputs_even_when_empty(preparation, with_log):
    _, helper, _, output, ready, _ = prepared(preparation)
    (output / "outputs").mkdir()
    if with_log:
        put(output / "outputs/logs/amlt_code_runner.txt", "runtime log")
    with pytest.raises(ValueError) as caught:
        helper.verify_package(output, ready["recovery_manifest_sha256"])
    assert "outputs" in caught.value.details["extra"]


def test_preparation_does_not_adopt_an_empty_outputs_directory_in_base(preparation):
    preparer, _, base, output = preparation
    (base / "outputs").mkdir()
    with pytest.raises(ValueError):
        preparer.prepare_bundle(base, output)
    assert not output.exists()


def test_runtime_outputs_is_never_enumerated_or_hashed(preparation, monkeypatch):
    _, helper, _, output, ready, _ = prepared(preparation)
    runtime = output / "outputs"
    put(runtime / "logs/amlt_code_runner.txt", "unreadable platform log")
    original_scandir, original_sha = helper.os.scandir, helper.sha

    def no_runtime_scan(path):
        if not isinstance(path, int):
            current = Path(path)
            if current == runtime or runtime in current.parents:
                raise PermissionError("runtime log tree must never be enumerated")
        return original_scandir(path)

    def no_runtime_hash(path):
        current = Path(path)
        if current == runtime or runtime in current.parents:
            raise AssertionError("runtime log bytes must never be hashed")
        return original_sha(path)

    monkeypatch.setattr(helper.os, "scandir", no_runtime_scan)
    monkeypatch.setattr(helper, "sha", no_runtime_hash)
    helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)


@pytest.mark.parametrize("case", ["outputs_file", "outputs_symlink", "outputs_dangling_symlink",
                                  "unknown_root_symlink", "cache_named_symlink", "base_outputs",
                                  "unregistered_empty_directory"])
def test_runtime_exception_does_not_admit_other_nodes(preparation, tmp_path, case):
    _, helper, _, output, ready, _ = prepared(preparation)
    outside = tmp_path / "external"
    outside.mkdir()
    if case == "outputs_file":
        put(output / "outputs", "not a directory")
    elif case == "outputs_symlink":
        (output / "outputs").symlink_to(outside, target_is_directory=True)
    elif case == "outputs_dangling_symlink":
        (output / "outputs").symlink_to(outside / "missing")
    elif case == "unknown_root_symlink":
        (output / "unknown.link").symlink_to(outside, target_is_directory=True)
    elif case == "cache_named_symlink":
        (output / "__pycache__").symlink_to(outside, target_is_directory=True)
    elif case == "base_outputs":
        (output / "base/outputs").mkdir()
    else:
        (output / "unregistered").mkdir()
    with pytest.raises(ValueError):
        helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)


def test_runtime_logs_do_not_hide_source_corruption(preparation):
    _, helper, _, output, ready, _ = prepared(preparation)
    put(output / "outputs/logs/amlt_code_runner.txt", "allowed runtime log")
    (output / "base/requirements-worker.txt").write_text("modified frozen bytes")
    with pytest.raises(ValueError) as caught:
        helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)
    assert caught.value.details["changed"] == ["base/requirements-worker.txt"]


def test_failure_path_lists_are_bounded_but_counts_remain_complete(preparation):
    _, helper, _, output, ready, _ = prepared(preparation)
    for index in range(75):
        put(output / f"extra-{index:03d}.txt", "unexpected")
    with pytest.raises(ValueError) as caught:
        helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)
    details = caught.value.details
    assert details["counts"]["extra"] == 75
    assert len(details["extra"]) <= 50 and details["truncated"]["extra"] is True


def test_worker_uses_runtime_boundary_before_and_after_original_worker(preparation, tmp_path, monkeypatch):
    _, helper, _, output, ready, context = prepared(preparation)
    put(output / "outputs/logs/before.txt", "platform before worker")
    module = fake_worker(helper, context)
    module.__file__ = str(context["base"] / "src/new_interest/experiments/rho_student_experiment.py")
    original_worker, original_verify = module.worker, helper.verify_package
    calls = []

    def original_worker_with_platform_logs(*args):
        result = original_worker(*args)
        put(output / "outputs/logs/after.txt", "platform after worker")
        return result

    def verified(*args, **kwargs):
        calls.append(kwargs.get("runtime", False))
        return original_verify(*args, **kwargs)

    module.worker = original_worker_with_platform_logs
    monkeypatch.setattr(helper, "__file__", str(output / "recovery_worker.py"))
    monkeypatch.setattr(helper, "verify_package", verified)
    monkeypatch.setattr(helper.importlib, "import_module", lambda name: module)
    result = helper.run_worker(output, tmp_path / "run", arm="dynamic",
                               expected_manifest_sha256=ready["recovery_manifest_sha256"])
    assert result["passed"] is True
    assert calls == [True, True]


def test_unwritable_failure_capture_does_not_suppress_guard_rejection(preparation, tmp_path):
    _, helper, _, output, ready, _ = prepared(preparation)
    put(output / "untrusted.py", "invalid package")
    root = tmp_path / "not-a-directory"
    root.write_text("preserve")
    with pytest.raises(ValueError) as caught:
        helper.run_worker(output, root, arm="static", expected_manifest_sha256=ready["recovery_manifest_sha256"])
    assert caught.value.details["extra"] == ["untrusted.py"]
    assert root.read_text() == "preserve"


def test_v1_bytecode_cache_exclusion_is_preserved(preparation):
    _, helper, _, output, ready, _ = prepared(preparation)
    put(output / "__pycache__/recovery_worker.cpython-310.pyc", "runtime bytecode cache")
    put(output / "base/src/new_interest/experiments/__pycache__/worker.cpython-310.pyc", "cache")
    helper.verify_package(output, ready["recovery_manifest_sha256"], runtime=True)
