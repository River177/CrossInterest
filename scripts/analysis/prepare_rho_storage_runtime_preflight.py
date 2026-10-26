"""Freeze a stdlib-only AMLT old/new storage-guard preflight; never submit it."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import zlib


REPO = Path(__file__).resolve().parents[2]
VERSION = "rho-storage-runtime-preflight-v2"
MAX_COMMAND_UTF16_CODE_UNITS = 20_000
OLD_GUARD_SHA256 = "75c25765d4abf6f11af50641e5d689da336a6f62a585329178e376ac0c22c602"
OLD_REJECTION = "Recovery package inventory or file hashes differ"
OLD_GUARD_FIXTURE = REPO / "tests/fixtures/rho_storage_guard_v1.py"


WORKER_TEMPLATE = r'''"""CPU-only differential check of the actual frozen storage guard modules."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback

PINNED_OLD_GUARD_SHA256 = "@OLD_GUARD_SHA256@"
OLD_REJECTION = "Recovery package inventory or file hashes differ"
FORBIDDEN = ("torch", "transformers", "peft", "fla", "triton", "huggingface_hub", "numpy")
sys.dont_write_bytecode = True


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def within(path, root):
    return path == root or root in path.parents


def kind(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    return "symlink" if stat.S_ISLNK(mode) else "directory" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else "other"


def protected_snapshot(package):
    """Independent traversal: the tested inventory() never supplies this proof."""
    records = {}
    def visit(directory):
        with os.scandir(directory) as entries:
            entries = sorted(entries, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(package).as_posix()
            if relative == "outputs":
                if kind(path) != "directory":
                    raise ValueError("Runtime root outputs must be a real directory")
                continue
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                records[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif stat.S_ISDIR(mode):
                records[relative] = {"kind": "directory"}
                visit(path)
            elif stat.S_ISREG(mode):
                records[relative] = {"kind": "file", "size": entry.stat(follow_symlinks=False).st_size, "sha256": digest(path)}
            else:
                raise ValueError("Unexpected special file outside runtime outputs: " + relative)
    visit(package)
    return records


def wait_for_runtime(package, seconds):
    outputs, logs = package / "outputs", package / "outputs/logs"
    started, deadline = time.monotonic(), time.monotonic() + seconds
    observed = []
    while True:
        output_kind, log_kind = kind(outputs), kind(logs)
        if output_kind not in ("missing", "directory") or log_kind not in ("missing", "directory"):
            raise ValueError("AMLT outputs/logs must be real, non-symlink runtime directories")
        if output_kind == log_kind == "directory":
            with os.scandir(logs) as entries:
                observed = sorted(entry.name for entry in entries if entry.is_file(follow_symlinks=False))
            if observed:
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(.25, max(0., deadline - time.monotonic())))
    if kind(outputs) != "directory" or kind(logs) != "directory":
        raise ValueError("Real AMLT outputs/logs did not appear within the bounded wait; no runtime directories were fabricated")
    probe = None
    if not observed:
        probe = logs / ("rho_storage_guard_preflight_probe_" + str(os.getpid()) + ".json")
        with probe.open("x") as handle:
            json.dump({"source": "CPU preflight generated probe", "purpose": "Make the original inventory failure observable in an existing real runtime logs directory"}, handle)
    return {"outputs_type": kind(outputs), "logs_type": kind(logs), "wait_seconds": time.monotonic() - started,
            "preexisting_regular_log_files": observed[:20], "preexisting_regular_log_file_count": len(observed),
            "failure_witness_source": "preexisting runtime log filename" if observed else "explicit preflight probe",
            "probe_file": None if probe is None else probe.relative_to(package).as_posix(),
            "runtime_directories_created_by_preflight": False, "readiness_enumerated_log_names": True,
            "readiness_read_platform_log_contents": False}


def load_guard(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--old-guard", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=35.)
    args = parser.parse_args()
    package, old_path, output = args.package.resolve(), args.old_guard.resolve(), args.output.resolve()
    if not 0 <= args.wait_seconds <= 35:
        raise ValueError("Runtime wait must be between zero and 35 seconds")
    if any(within(path, package) or within(package, path) for path in (old_path, output, Path(__file__).resolve())):
        raise ValueError("Diagnostic scripts and receipts must live outside the code bundle")
    report = {"version": "rho-storage-runtime-preflight-v1", "passed": False,
              "package": str(package), "old_guard_sha256": PINNED_OLD_GUARD_SHA256,
              "expected_manifest_sha256": args.manifest_sha256, "expected_launcher_sha256": args.launcher_sha256,
              "runtime_script_sha256": digest(__file__), "stdlib_only": True, "guard_entrypoints": [],
              "subprocess_or_network_attempts": [], "forbidden_import_attempts": [],
              "torch_modules_before": [name for name in sys.modules if name == "torch" or name.startswith("torch.")]}
    stage = {"name": "setup"}
    audit = {}
    runtime = package / "outputs"

    def audit_hook(event, values):
        if event == "import" and values and str(values[0]).split(".")[0] in FORBIDDEN:
            report["forbidden_import_attempts"].append(str(values[0]))
            raise RuntimeError("Model/non-stdlib import forbidden in CPU guard preflight")
        if event in ("subprocess.Popen", "os.system", "os.exec", "socket.__new__", "socket.connect"):
            report["subprocess_or_network_attempts"].append(event)
            raise RuntimeError("Subprocess/network operation forbidden in CPU guard preflight")
        if stage["name"] not in ("new_before", "new_after", "old_guard", "old_inventory", "packaging_control"):
            return
        if event not in ("os.scandir", "os.listdir", "open") or not values:
            return
        value = values[0]
        if isinstance(value, int):
            try:
                value = os.readlink("/proc/self/fd/" + str(value))
            except OSError:
                raise RuntimeError("Unresolved descriptor in guard filesystem audit")
        try:
            path = Path(os.path.abspath(os.fsdecode(value)))
        except TypeError:
            raise RuntimeError("Unresolved path in guard filesystem audit")
        if within(path, runtime):
            counters = audit.setdefault(stage["name"], {"runtime_directory_enumerations": 0, "runtime_file_opens": 0})
            key = "runtime_file_opens" if event == "open" else "runtime_directory_enumerations"
            counters[key] += 1
            if stage["name"].startswith("new_"):
                raise RuntimeError("New guard accessed runtime outputs contents: " + event)

    sys.addaudithook(audit_hook)
    try:
        if report["torch_modules_before"]:
            raise ValueError("Torch was already loaded before preflight")
        if digest(old_path) != PINNED_OLD_GUARD_SHA256:
            raise ValueError("Old guard is not the exact original v1 source")
        if digest(package / "recovery_manifest.json") != args.manifest_sha256 or digest(package / "recovery_worker.py") != args.launcher_sha256:
            raise ValueError("Runtime package does not match the externally frozen hashes")
        report["runtime_environment"] = wait_for_runtime(package, args.wait_seconds)
        before = protected_snapshot(package)
        manifest = json.loads((package / "recovery_manifest.json").read_text())
        declared = {name: digest(package / name) for name in manifest["files"]}
        if declared != manifest["files"]:
            raise ValueError("Independent declared-file hashes differ from the package manifest")
        new, old = load_guard(package / "recovery_worker.py", "rho_preflight_new_guard"), load_guard(old_path, "rho_preflight_original_v1_guard")
        report["old_guard_module_file"] = str(old_path)
        for phase in ("new_before",):
            stage["name"] = phase
            context = new.verify_package(package, args.manifest_sha256, runtime=True)
            report["guard_entrypoints"].append("new.verify_package(runtime=True):before")
        stage["name"] = "packaging_control"
        try:
            new.verify_package(package, args.manifest_sha256)
        except ValueError as error:
            details = getattr(error, "details", None)
            if (not isinstance(error, new.PackageIntegrityError) or not isinstance(details, dict)
                    or details.get("extra") != ["outputs"] or details.get("missing") != []
                    or details.get("changed") != []
                    or details.get("invalid_nodes") != {"outputs": "root_outputs_requires_real_directory_and_runtime_mode"}
                    or details.get("counts") != {"extra": 1, "missing": 0, "changed": 0}):
                raise
            report["strict_packaging_rejection"] = details
        else:
            raise ValueError("Packaging incorrectly accepted the runtime outputs directory")
        stage["name"] = "old_guard"
        try:
            old.verify_package(package, args.manifest_sha256)
        except ValueError as error:
            frames = traceback.extract_tb(error.__traceback__)
            target_lines = [i for i, line in enumerate(old_path.read_text().splitlines(), 1) if 'raise ValueError("' + OLD_REJECTION + '")' in line]
            exact_frame = any(Path(frame.filename).resolve() == old_path and frame.name == "verify_package" and frame.lineno in target_lines for frame in frames)
            if str(error) != OLD_REJECTION or not exact_frame:
                raise ValueError("Old guard did not fail at its original inventory rejection") from error
            report["old_guard_rejection"] = {"type": type(error).__name__, "message": str(error), "original_inventory_branch": True,
                                             "function": "verify_package", "source_line": next(frame.lineno for frame in frames if Path(frame.filename).resolve() == old_path and frame.name == "verify_package")}
        else:
            raise ValueError("Original v1 guard unexpectedly accepted this same runtime package")
        report["guard_entrypoints"].append("original_v1.verify_package():unmodified")
        stage["name"] = "old_inventory"
        inventory = old.inventory(package)
        expected = set(manifest["files"]) | {"recovery_manifest.json"}
        missing, extra = sorted(expected - inventory), sorted(inventory - expected)
        if missing or not extra or any(not name.startswith("outputs/") for name in extra):
            raise ValueError("Old failure is not isolated to runtime outputs files")
        report["old_inventory_diff"] = {"missing": missing, "extra_count": len(extra), "extra_paths": extra[:25],
                                         "all_extras_under_root_outputs": True}
        stage["name"] = "new_after"
        repeated = new.verify_package(package, args.manifest_sha256, runtime=True)
        report["guard_entrypoints"].append("new.verify_package(runtime=True):after")
        stage["name"] = "independent_snapshot"
        after = protected_snapshot(package)
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        if changed or context["identity"] != repeated["identity"] or digest(old_path) != PINNED_OLD_GUARD_SHA256:
            raise ValueError("Preflight changed protected package/base/old-source bytes")
        base_before = {name: value for name, value in before.items() if name.startswith("base/")}
        base_after = {name: value for name, value in after.items() if name.startswith("base/")}
        report.update(package_identity=context["identity"], package_manifest_sha256=digest(package / "recovery_manifest.json"),
                      base_runtime_sha256=context["identity"]["base_runtime_sha256"], base_protocol_sha256=context["identity"]["base_protocol_sha256"],
                      package_tree_before_sha256=canonical(before), package_tree_after_sha256=canonical(after),
                      base_tree_before_sha256=canonical(base_before), base_tree_after_sha256=canonical(base_after),
                      protected_changed_paths=changed, declared_file_count=len(declared),
                      base_file_count=sum(value["kind"] == "file" for value in base_before.values()),
                      declared_hashes_sha256=canonical(declared), new_guard_passed_before=True, new_guard_passed_after=True)
        if any(audit.get(phase, {}).get(key, 0) for phase in ("new_before", "new_after") for key in ("runtime_directory_enumerations", "runtime_file_opens")):
            raise ValueError("New guard traversed or read runtime outputs")
        if audit.get("old_guard", {}).get("runtime_directory_enumerations", 0) == 0:
            raise ValueError("Old guard did not exercise its runtime-directory traversal")
        report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "stage": stage["name"]}
    finally:
        stage["name"] = "report"
        report["filesystem_audit"] = {phase: audit.get(phase, {"runtime_directory_enumerations": 0, "runtime_file_opens": 0}) for phase in ("new_before", "new_after", "old_guard", "old_inventory", "packaging_control")}
        report["torch_modules_after"] = [name for name in sys.modules if name == "torch" or name.startswith("torch.")]
        report["forbidden_modules_after"] = [name for name in sys.modules if name.split(".")[0] in FORBIDDEN]
        report["model_entrypoints_called"] = 0
        if report["forbidden_modules_after"] or report["subprocess_or_network_attempts"] or report["forbidden_import_attempts"]:
            report["passed"] = False
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError("Preflight receipt is immutable: " + str(output))
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
        print("RHO_STORAGE_RUNTIME_PREFLIGHT " + json.dumps({"passed": report["passed"], "receipt": str(output), "sha256": digest(output), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_source() -> str:
    return WORKER_TEMPLATE.replace("@OLD_GUARD_SHA256@", OLD_GUARD_SHA256)


def bootstrap_command(manifest: dict, manifest_sha: str, old_source: bytes) -> str:
    worker = worker_source().encode()
    payloads = {"rho_storage_runtime_preflight.py": worker, "original_storage_guard_v1.py": old_source}
    encoded = {name: {"zlib_base64": base64.b64encode(zlib.compress(value, level=9)).decode(),
                      "size": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in payloads.items()}
    lines = ["set -euo pipefail", "python -I -S -B - <<'RHO_PREFLIGHT_BOOTSTRAP'", "import base64, hashlib, json, os, pathlib, sys, zlib",
             "package = pathlib.Path(os.environ['AMLT_CODE_DIR']).resolve()",
             "output = pathlib.Path(os.environ['AMLT_OUTPUT_DIR']).resolve()",
             "if output == package or package in output.parents or output in package.parents: raise ValueError('Preflight assets must be outside code package')",
             "assets = output / 'preflight_assets'", "assets.mkdir(parents=True, exist_ok=True)", "payloads = " + repr(encoded),
             "for name, payload in payloads.items():", "    raw = zlib.decompress(base64.b64decode(payload['zlib_base64'], validate=True))",
             "    if len(raw) != payload['size'] or hashlib.sha256(raw).hexdigest() != payload['sha256']: raise ValueError('Embedded preflight payload hash or size differs')",
             "    path = assets / name", "    if path.exists() and path.read_bytes() != raw: raise ValueError('Existing preflight asset differs')",
             "    path.write_bytes(raw)",
             "args = [sys.executable, '-I', '-S', '-B', str(assets / 'rho_storage_runtime_preflight.py'), '--package', str(package), '--old-guard', str(assets / 'original_storage_guard_v1.py'), '--manifest-sha256', " + repr(manifest_sha) + ", '--launcher-sha256', " + repr(manifest["launcher_sha256"]) + ", '--output', str(output / 'compact/rho_storage_runtime_preflight.json'), '--wait-seconds', '35']",
             "os.execv(sys.executable, args)", "RHO_PREFLIGHT_BOOTSTRAP"]
    return "\n".join(lines)


def prepare(package: Path, manifest_sha256: str, output: Path, *, old_guard: Path = OLD_GUARD_FIXTURE,
            job_name: str = "rho-storage-runtime-guard-preflight") -> dict:
    package, output, old_guard = Path(package).resolve(), Path(output).resolve(), Path(old_guard).resolve()
    if not re.fullmatch(r"[a-f0-9]{64}", manifest_sha256) or sha(package / "recovery_manifest.json") != manifest_sha256:
        raise ValueError("v2 manifest hash differs from its immutable ready receipt")
    if output == package or package in output.parents or output in package.parents:
        raise ValueError("Generated CPU config must not overlap the code bundle")
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", job_name):
        raise ValueError("Invalid CPU job name")
    if sha(old_guard) != OLD_GUARD_SHA256:
        raise ValueError("Old guard does not match the frozen original v1 source")
    manifest = json.loads((package / "recovery_manifest.json").read_text())
    if manifest.get("version") != "rho-storage-retention-recovery-v1" or manifest.get("old_value") != 3 or manifest.get("new_value") != 25:
        raise ValueError("CPU differential preflight requires the compatible frozen storage package")
    if (package / "outputs").exists() or (package / "outputs").is_symlink():
        raise ValueError("Do not freeze an already-mutated runtime code bundle")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or any((package / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)) or sha(package / relative) != expected:
            raise ValueError("Declared recovery package bytes differ: " + name)
    command = bootstrap_command(manifest, manifest_sha256, old_guard.read_bytes())
    # AMLT's JobModel.to_entity() stores the command list as one JSON string
    # property. Azure Table allows at most 32,768 UTF-16 code units per string.
    # Retain extra headroom for submit-time command augmentation.
    table_command_value = json.dumps([command])
    table_command_utf16_bytes = len(table_command_value.encode("utf-16-le"))
    if table_command_utf16_bytes // 2 > MAX_COMMAND_UTF16_CODE_UNITS:
        raise ValueError("CPU preflight command exceeds the conservative Azure Table property size budget")
    output.mkdir(parents=True)
    config_file = output / "preflight.yaml"
    local_dir = "$CONFIG_DIR/" + Path(os.path.relpath(package, config_file.parent)).as_posix()
    config = {"description": "CPU-only real AMLT runtime-directory differential guard preflight; original v1 red and v2 runtime green; no model or dependency installation.",
              "environment": {"image": "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04"},
              "code": {"local_dir": local_dir},
              "target": {"service": "singularity", "name": "Feeds", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"},
              "jobs": [{"name": job_name, "sku": "5C3", "sla_tier": "Basic",
                        "submit_args": {"env": {"AMLT_DOCKERFILE_TEMPLATE": "DEFAULT", "AMLT_NO_RUNTIME_DEPS": True,
                            "_AZUREML_SINGULARITY_JOB_UAI": "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"}},
                        "command": [command]}]}
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    receipt = {"version": VERSION, "prepared": True, "submitted": False, "config": str(config_file),
               "config_sha256": sha(config_file), "package": str(package), "package_manifest_sha256": manifest_sha256,
               "package_launcher_sha256": manifest["launcher_sha256"], "base_runtime_sha256": manifest["base_runtime_sha256"],
               "base_protocol_sha256": manifest["base_protocol_sha256"], "base_source_manifest_sha256": manifest["base_source_manifest_sha256"],
               "old_guard_source": str(old_guard), "old_guard_sha256": OLD_GUARD_SHA256,
               "runtime_worker_sha256": hashlib.sha256(worker_source().encode()).hexdigest(), "job_name": job_name,
               "embedded_payload_encoding": "zlib-level-9+base64",
               "command_table_utf16_code_units": table_command_utf16_bytes // 2,
               "command_table_utf16_bytes": table_command_utf16_bytes,
               "command_table_sha256": hashlib.sha256(table_command_value.encode()).hexdigest(),
               "sku": "5C3", "sla_tier": "Basic", "expected_receipt": "compact/rho_storage_runtime_preflight.json",
               "same_code_directory_as_gpu": True, "probe_assets_written_only_to_amlt_output": True,
               "max_runtime_directory_wait_seconds": 35, "dependency_installation": False, "model_inference": False}
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-guard", type=Path, default=OLD_GUARD_FIXTURE)
    parser.add_argument("--job-name", default="rho-storage-runtime-guard-preflight")
    args = parser.parse_args()
    print(json.dumps(prepare(args.package, args.manifest_sha256, args.output, old_guard=args.old_guard, job_name=args.job_name), indent=2))


if __name__ == "__main__":
    main()
