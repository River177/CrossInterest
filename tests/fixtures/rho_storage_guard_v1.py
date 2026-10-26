"""Run the frozen RHO worker with one explicit storage-retention override."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any


VERSION = "rho-storage-retention-recovery-v1"
MANIFEST_NAME = "recovery_manifest.json"
LAUNCHER_PATH = "recovery_worker.py"
COMPAT_PATH = "scripts/amlt/sft/run_rho_student.sh"
OLD_VALUE, NEW_VALUE = 3, 25
PLANNED_STEPS, PLANNED_CHECKPOINTS = 2025, 21
SUPPORTED_ARMS = ("static", "dynamic")
IDENTITY_FIELDS = (
    "version", "arm", "stage", "base_runtime_sha256", "base_protocol_sha256",
    "base_source_manifest_sha256", "recovery_manifest_sha256", "launcher_sha256",
)


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if (not isinstance(name, str) or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name or not relative.parts):
        raise ValueError("Unsafe recovery manifest path: " + str(name))
    candidate = root / relative
    if any((root / Path(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Recovery files may not be symlinks: " + name)
    if not candidate.is_file():
        raise ValueError("Recovery file is missing: " + name)
    return candidate


def inventory(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.relative_to(root).parts
            and p.suffix != ".pyc"}


def verify_base(base: Path) -> tuple[dict, dict]:
    base = Path(base).resolve()
    source = read(base / "source_manifest.json")
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen base manifest has no files")
    if any(name.endswith((".safetensors", ".pt", ".bin")) for name in files):
        raise ValueError("Recovery code bundles must not contain model or optimizer weights")
    if inventory(base) != set(files) | {"source_manifest.json"}:
        raise ValueError("Frozen base inventory differs from its manifest")
    for name, expected in files.items():
        if sha(safe_file(base, name)) != expected:
            raise ValueError("Frozen base bytes changed: " + name)
    runtime = {k: v for k, v in files.items()
               if k.startswith(("src/", "scripts/", "configs/")) or k == "requirements-worker.txt"}
    runtime_sha = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    if source.get("runtime_sha256") != runtime_sha:
        raise ValueError("Frozen base runtime hash is inconsistent")
    protocol = read(base / "protocol.json")
    gate = read(base / "gate_acceptance.json")
    if (protocol.get("version") != "mind-rho-student-v1"
            or protocol.get("arms") != ["uniform", "static", "dynamic"]
            or protocol.get("optimizer_steps") != PLANNED_STEPS
            or protocol.get("gpus_per_arm") != 4
            or protocol.get("training_rows") != 21600
            or protocol["sft"].get("persistent_checkpoint_limit") != OLD_VALUE
            or protocol["sft"].get("save_total_limit") != 3
            or protocol["sft"].get("save_strategy") != "steps"
            or protocol["sft"].get("save_steps") != 100
            or math.ceil(protocol["optimizer_steps"] / protocol["sft"]["save_steps"]) != PLANNED_CHECKPOINTS):
        raise ValueError("Unsupported frozen RHO storage recovery plan")
    if (gate.get("passed") is not True
            or gate["identity"]["phase"] != "gate"
            or gate["identity"]["runtime_sha256"] != runtime_sha
            or gate["identity"]["protocol_sha256"] != sha(base / "protocol.json")):
        raise ValueError("Frozen base is not bound to its accepted original gate")
    return source, protocol


def verify_package(package: Path, expected_manifest_sha256: str | None = None) -> dict:
    package = Path(package).resolve()
    manifest_path = package / MANIFEST_NAME
    manifest_sha = sha(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("Recovery manifest differs from the expected package")
    manifest = read(manifest_path)
    if (manifest.get("version") != VERSION or manifest.get("arms") != list(SUPPORTED_ARMS)
            or manifest.get("stage") != "full" or manifest.get("base_directory") != "base"
            or manifest.get("launcher_path") != LAUNCHER_PATH
            or manifest.get("compat_wrapper_path") != COMPAT_PATH
            or manifest.get("changed_fields") != ["persistent_checkpoint_limit"]
            or manifest.get("old_value") != OLD_VALUE or manifest.get("new_value") != NEW_VALUE
            or manifest.get("planned_optimizer_steps") != PLANNED_STEPS
            or manifest.get("planned_max_checkpoint_count") != PLANNED_CHECKPOINTS):
        raise ValueError("Unexpected recovery package scope")
    source, protocol = verify_base(package / "base")
    expected_files = {"base/" + name: digest for name, digest in source["files"].items()}
    expected_files["base/source_manifest.json"] = sha(package / "base/source_manifest.json")
    for name in (LAUNCHER_PATH, COMPAT_PATH):
        expected_files[name] = sha(safe_file(package, name))
    if manifest.get("files") != expected_files or inventory(package) != set(expected_files) | {MANIFEST_NAME}:
        raise ValueError("Recovery package inventory or file hashes differ")
    expected = {"base_runtime_sha256": source["runtime_sha256"],
                "base_protocol_sha256": sha(package / "base/protocol.json"),
                "base_source_manifest_sha256": expected_files["base/source_manifest.json"],
                "launcher_sha256": expected_files[LAUNCHER_PATH]}
    if (any(manifest.get(k) != v for k, v in expected.items())
            or manifest.get("compat_wrapper_sha256") != expected_files[COMPAT_PATH]
            or manifest.get("base_gate_acceptance_sha256") != source["files"]["gate_acceptance.json"]):
        raise ValueError("Recovery package base/launcher identity differs")
    return {"package": package, "base": package / "base", "manifest": manifest,
            "protocol": protocol, "identity": {"version": VERSION, "stage": "full",
                **expected, "recovery_manifest_sha256": manifest_sha}}


def validate_override(value: dict, context: dict) -> None:
    if any(value.get(k) != context["identity"][k] for k in IDENTITY_FIELDS):
        raise ValueError("Storage override belongs to a different recovery package")
    before, after = value["config_before"], value["config_after"]
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if (set(before) != set(after) or changed != ["persistent_checkpoint_limit"]
            or value.get("changed_fields") != changed
            or type(before.get("persistent_checkpoint_limit")) is not int
            or before["persistent_checkpoint_limit"] != OLD_VALUE
            or type(after.get("persistent_checkpoint_limit")) is not int
            or after["persistent_checkpoint_limit"] != NEW_VALUE
            or value.get("old_value") != OLD_VALUE or value.get("new_value") != NEW_VALUE
            or value.get("planned_optimizer_steps") != PLANNED_STEPS
            or value.get("planned_max_checkpoint_count") != PLANNED_CHECKPOINTS
            or value.get("native_training_entrypoint_unchanged") is not True):
        raise ValueError("Storage recovery must change only retention 3 to 25")
    if any(before.get(k) != v for k, v in context["protocol"]["sft"].items()):
        raise ValueError("Original generated config differs from frozen SFT settings")
    if (before.get("rho_arm") != context["identity"]["arm"]
            or before.get("rho_training_numeric_policy") != context["protocol"]["training_numeric_policy"]
            or before.get("resume_from_checkpoint") != "auto"):
        raise ValueError("Recovery changed the native RHO resume contract")


def _record_override(path: Path, value: dict) -> None:
    if path.exists() and read(path) != value:
        previous_hash = sha(path)
        history = path.parent / "storage_retention_override_history" / (previous_hash + ".json")
        if history.exists() and history.read_bytes() != path.read_bytes():
            raise ValueError("Existing retention history has different bytes")
        if not history.exists():
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_bytes(path.read_bytes())
    write(path, value)


def _run_worker_with_override(module: Any, context: dict, root: Path, *, arm: str) -> dict:
    if arm not in SUPPORTED_ARMS:
        raise ValueError("Storage recovery is limited to the explicitly supported arms")
    context = {**context, "identity": {**context["identity"], "arm": arm}}
    expected_arm = arm
    root = Path(root).resolve()
    receipt_path = root / "compact/storage_retention_override.json"
    artifact_reused = (root / "full/artifact_manifest.json").exists()
    receipt = read(receipt_path) if artifact_reused else None
    if receipt is not None:
        validate_override(receipt, context)
        if read(root / "full/resolved_config.json") != receipt["config_after"]:
            raise ValueError("Closed training artifact differs from its original storage override")
    original = module.make_training_config
    original_command = module.command
    calls = 0

    def configuration(project, protocol, model, output_root, local, stage, arm, reference):
        nonlocal receipt, calls
        if stage != "full" or arm != expected_arm or Path(project).resolve() != context["base"]:
            raise ValueError("Storage recovery is restricted to this full frozen arm")
        before, rule = original(project, protocol, model, output_root, local, stage, arm, reference)
        if (rule.get("optimizer_steps") != PLANNED_STEPS or rule.get("world_size") != 4
                or rule.get("rows") != 21600 or rule.get("epochs") != 3):
            raise ValueError("Original worker training plan changed")
        before = copy.deepcopy(before)
        after = copy.deepcopy(before)
        after["persistent_checkpoint_limit"] = NEW_VALUE
        generated = {**context["identity"], "config_before": before, "config_after": after,
                     "changed_fields": ["persistent_checkpoint_limit"], "old_value": OLD_VALUE,
                     "new_value": NEW_VALUE, "planned_optimizer_steps": PLANNED_STEPS,
                     "planned_max_checkpoint_count": PLANNED_CHECKPOINTS,
                     "native_training_entrypoint_unchanged": True}
        validate_override(generated, context)
        calls += 1
        if calls != 1:
            raise ValueError("Recovery unexpectedly generated multiple training configurations")
        if not artifact_reused:
            receipt = generated
            _record_override(receipt_path, receipt)
        print("RHO_STORAGE_RETENTION_OVERRIDE " + json.dumps(receipt, sort_keys=True), flush=True)
        return after, rule

    module.make_training_config = configuration
    try:
        result = module.worker(context["base"], root, "full", expected_arm)
    finally:
        module.make_training_config = original
    if calls != 1 or receipt is None or module.command is not original_command:
        raise ValueError("Recovery did not preserve the original native subprocess launcher")
    validate_override(receipt, context)
    if read(root / "full/resolved_config.json") != receipt["config_after"]:
        raise ValueError("Published resolved config differs from the declared storage override")
    success_path = root / "compact/worker_success.json"
    if result.get("passed") is not True or read(success_path) != result:
        raise ValueError("Original worker did not finish its normal full acceptance")
    result = {**result, "storage_retention_override": receipt}
    _record_override(receipt_path, receipt)
    write(success_path, result)
    return result


def run_worker(package: Path, root: Path, *, arm: str, expected_manifest_sha256: str | None = None) -> dict:
    context = verify_package(package, expected_manifest_sha256)
    if Path(__file__).resolve() != context["package"] / LAUNCHER_PATH:
        raise ValueError("Execute the launcher copied into the verified recovery package")
    sys.path.insert(0, str(context["base"] / "src"))
    module = importlib.import_module("new_interest.experiments.rho_student_experiment")
    expected_module = context["base"] / "src/new_interest/experiments/rho_student_experiment.py"
    if Path(module.__file__).resolve() != expected_module:
        raise ValueError("Worker import did not resolve to the frozen base")
    result = _run_worker_with_override(module, context, root, arm=arm)
    verify_package(package, context["identity"]["recovery_manifest_sha256"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("full",), required=True)
    parser.add_argument("--arm", choices=SUPPORTED_ARMS, required=True)
    parser.add_argument("--recovery-manifest-sha256")
    args = parser.parse_args()
    run_worker(args.package, args.root, arm=args.arm, expected_manifest_sha256=args.recovery_manifest_sha256)


if __name__ == "__main__":
    main()
