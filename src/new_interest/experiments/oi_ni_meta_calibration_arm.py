"""Dispatch one remaining calibration arm through the unchanged legacy core.

The mathematical/run identity remains the original serial calibration identity.
This new entry point has a separately hash-bound dispatch identity/provenance;
it never selects an alpha, imports alpha-0 weights, or fabricates a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


ACCEPTED_LEGACY_SOURCE_ID = "38931d42872c03eecae1eca6355c1ca78d60a77f52b4b3793bda96662ce66b02"
DISPATCH_VERSION = "oi-ni-meta-calibration-arm-dispatch-v1"
DISPATCHER_PATH = "src/new_interest/experiments/oi_ni_meta_calibration_arm.py"
LEGACY_FILES = {
    "oi_ni_meta.py": "src/new_interest/experiments/oi_ni_meta.py",
    "../training/oi_ni_meta_backend.py": "src/new_interest/training/oi_ni_meta_backend.py",
    "../training/oi_ni_meta_objective.py": "src/new_interest/training/oi_ni_meta_objective.py",
}
EXPECTED_VERSIONS = {
    "torch": "2.8.0+cu126", "transformers": "5.14.1", "peft": "0.18.0",
    "accelerate": "1.12.0", "pyarrow": "20.0.0", "tokenizers": "0.22.2",
    "safetensors": "0.8.0", "huggingface-hub": "1.14.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes code root: {relative}")
    return path


def calibration_settings(config: Mapping[str, Any], source_identity: str, alpha_index: int) -> dict[str, Any]:
    if type(alpha_index) is not int or alpha_index not in (1, 2):
        raise ValueError("Only remaining alpha indices 1 and 2 may be dispatched")
    if len(config["inner_alpha_candidates"]) != 3 or config["calibration_steps"] != 27:
        raise ValueError("The original three-alpha, 27-step calibration contract must remain unchanged")
    return {"config": dict(config), "mode": "meta", "alpha": config["inner_alpha_candidates"][alpha_index],
            "steps": config["calibration_steps"], "phase": "calibrate", "source_identity": source_identity,
            "eval_dev": False, "selection_sha256": None, "budget_sha256": None}


def calibration_identity(config: Mapping[str, Any], source_identity: str, alpha_index: int) -> str:
    settings = calibration_settings(config, source_identity, alpha_index)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def legacy_source_receipt(code_root: Path, config_path: Path, *,
                          data_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Reproduce the old main's source identity, with mandatory core manifest."""
    config = json.loads(config_path.read_text())
    manifest_path = code_root / "source_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        if sha256(contained(code_root, relative)) != expected:
            raise ValueError(f"Frozen legacy source changed: {relative}")
    if data_hashes is None:
        from new_interest.experiments import oi_ni_meta as core
        data_hashes = core.validate_frozen_data(config, code_root / config["data_dir"])
    files = dict(data_hashes)
    files["protocol"] = sha256(config_path)
    files["worker_bundle"] = sha256(manifest_path)
    files.update({key: sha256(contained(code_root, relative)) for key, relative in LEGACY_FILES.items()})
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"identity": identity, "files": files}


def make_dispatch_manifest(code_root: Path, config_path: Path, *,
                           dispatch_files: Sequence[str] = (DISPATCHER_PATH,),
                           parent_recovery: Path | None = None) -> dict[str, Any]:
    """Local authoring helper; returns data, does not write or launch anything."""
    source = legacy_source_receipt(code_root, config_path)
    if source["identity"] != ACCEPTED_LEGACY_SOURCE_ID:
        raise ValueError("Dispatch must use the accepted legacy numerical source")
    if DISPATCHER_PATH not in dispatch_files:
        raise ValueError("Dispatch file inventory must include this entry point")
    config = json.loads(config_path.read_text())
    return {"version": DISPATCH_VERSION, "legacy_source_identity": source["identity"],
            "legacy_source_manifest_sha256": sha256(code_root / "source_bundle_manifest.json"),
            "protocol_sha256": sha256(config_path), "allowed_alpha_indices": [1, 2],
            "scientific_protocol_unchanged": True,
            "files": {name: sha256(contained(code_root, name)) for name in dispatch_files},
            "expected_run_identities": {str(index): calibration_identity(config, source["identity"], index)
                                        for index in (1, 2)},
            "parent_recovery_sha256": sha256(parent_recovery) if parent_recovery else None}


def validate_dispatch(code_root: Path, config_path: Path, manifest_path: Path, source: Mapping[str, Any],
                       alpha_index: int, parent_recovery: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    expected_run = calibration_identity(config, source["identity"], alpha_index)
    manifest = json.loads(manifest_path.read_text())
    if source["identity"] != ACCEPTED_LEGACY_SOURCE_ID:
        raise ValueError("Current core source is not the independently accepted legacy source")
    if (manifest.get("version") != DISPATCH_VERSION
            or manifest.get("legacy_source_identity") != source["identity"]
            or manifest.get("legacy_source_manifest_sha256") != source["files"]["worker_bundle"]
            or manifest.get("protocol_sha256") != source["files"]["protocol"]
            or manifest.get("allowed_alpha_indices") != [1, 2]
            or manifest.get("scientific_protocol_unchanged") is not True):
        raise ValueError("Dispatcher provenance differs from the accepted calibration source/contract")
    files = manifest.get("files", {})
    if DISPATCHER_PATH not in files:
        raise ValueError("Dispatcher code is not hash-bound")
    for relative, expected in files.items():
        if sha256(contained(code_root, relative)) != expected:
            raise ValueError(f"Dispatcher/launcher bytes changed: {relative}")
    if manifest.get("expected_run_identities") != {
        str(index): calibration_identity(config, source["identity"], index) for index in (1, 2)
    }:
        raise ValueError("Dispatcher alpha run identities differ from the original serial runs")
    recovery_sha = sha256(parent_recovery) if parent_recovery else None
    if recovery_sha != manifest.get("parent_recovery_sha256"):
        raise ValueError("Parent recovery receipt is missing or not hash-bound")
    if parent_recovery:
        recovery = json.loads(parent_recovery.read_text())
        if recovery.get("legacy_source_identity") != source["identity"]:
            raise ValueError("Parent recovery receipt belongs to a different legacy source")
    manifest_sha = sha256(manifest_path)
    dispatch_binding = {"dispatch_manifest_sha256": manifest_sha, "alpha_index": alpha_index,
                        "legacy_run_identity": expected_run, "legacy_source_identity": source["identity"],
                        "parent_recovery_sha256": recovery_sha}
    dispatch_identity = hashlib.sha256(json.dumps(dispatch_binding, sort_keys=True).encode()).hexdigest()
    return {"version": DISPATCH_VERSION, "dispatch_identity": dispatch_identity, **dispatch_binding,
            "entry_point": DISPATCHER_PATH, "legacy_entry_point_reused": False,
            "execution_function": "new_interest.experiments.oi_ni_meta.run_training",
            "selected_alpha": config["inner_alpha_candidates"][alpha_index],
            "phase": "calibrate", "mode": "meta", "steps": 27, "world_size": 1,
            "checkpoint_policy": "only identity-matched committed checkpoints already in this arm output",
            "alpha0_parameters_imported": False, "alpha_selection_performed": False}


def verify_runtime_imports(code_root: Path) -> Any:
    modules = {
        "new_interest.experiments.oi_ni_meta": LEGACY_FILES["oi_ni_meta.py"],
        "new_interest.training.oi_ni_meta_backend": LEGACY_FILES["../training/oi_ni_meta_backend.py"],
        "new_interest.training.oi_ni_meta_objective": LEGACY_FILES["../training/oi_ni_meta_objective.py"],
    }
    loaded = {}
    for name, relative in modules.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != contained(code_root, relative):
            raise ValueError(f"Runtime imported a core module outside the frozen code root: {name}")
        loaded[name] = module
    if Path(__file__).resolve() != contained(code_root, DISPATCHER_PATH):
        raise ValueError("Dispatcher import path differs from its bound source file")
    return loaded["new_interest.experiments.oi_ni_meta"]


def environment_audit(torch_module: Any) -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise ValueError(f"Installed numerical worker versions differ: {versions}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be set before launching the worker")
    return {"packages": versions, "cuda": torch_module.version.cuda,
            "gpu": torch_module.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "world_size": 1, "entry_point": DISPATCHER_PATH}


def execute_arm(core: Any, config: dict[str, Any], snapshot: str, data: Path,
                 output_root: Path, alpha_index: int, source_identity: str) -> dict[str, Any]:
    settings = calibration_settings(config, source_identity, alpha_index)
    if source_identity != ACCEPTED_LEGACY_SOURCE_ID:
        raise ValueError("Refusing a different calibration source")
    result = core.run_training(
        config, snapshot, data, output_root / f"alpha-{alpha_index}",
        mode="meta", alpha=settings["alpha"], steps=settings["steps"],
        phase="calibrate", source_identity=source_identity,
    )
    if (result.get("identity") != calibration_identity(config, source_identity, alpha_index)
            or result.get("mode") != "meta" or result.get("alpha") != settings["alpha"]
            or result.get("steps") != 27 or result.get("dev_evaluated") is not False):
        raise ValueError("Legacy run returned an unexpected calibration identity/scope")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dispatch-manifest", type=Path, default=Path("dispatch_manifest.json"))
    parser.add_argument("--parent-recovery", type=Path)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each recovery arm requires exactly one CUDA GPU")
    code_root = Path.cwd().resolve()
    config = json.loads(args.config.read_text())
    core = verify_runtime_imports(code_root)
    source = legacy_source_receipt(code_root, args.config)
    provenance = validate_dispatch(code_root, args.config, args.dispatch_manifest, source,
                                   args.alpha_index, args.parent_recovery)
    environment = environment_audit(torch)
    args.root.mkdir(parents=True, exist_ok=True)
    source_path = args.root / "source_identity.json"
    if source_path.exists() and json.loads(source_path.read_text())["identity"] != source["identity"]:
        raise ValueError("Output root belongs to another legacy calibration source")
    core.write_json(source_path, source)
    core.write_json(args.root / "environment.json", environment)
    dispatch_root = args.root / "dispatch" / f"alpha-{args.alpha_index}"
    dispatch_root.mkdir(parents=True, exist_ok=True)
    for label, path in {"dispatch_manifest.json": args.dispatch_manifest, "protocol.json": args.config,
                        "dispatcher.py": code_root / DISPATCHER_PATH,
                        **({"parent_recovery.json": args.parent_recovery} if args.parent_recovery else {})}.items():
        target = dispatch_root / label
        if target.exists() and sha256(target) != sha256(path):
            raise ValueError(f"Output dispatch evidence already has different bytes: {label}")
        shutil.copyfile(path, target)
    core.write_json(dispatch_root / "provenance.json", provenance)
    snapshot = core.prepare_snapshot(config, args.root)
    result = execute_arm(core, config, snapshot, code_root / config["data_dir"], args.root,
                         args.alpha_index, source["identity"])
    core.write_json(args.root / f"alpha_{args.alpha_index}_success.json", result)
    core.write_json(dispatch_root / "completion.json", {"dispatch_identity": provenance["dispatch_identity"],
                    "legacy_run_identity": result["identity"], "steps": result["steps"],
                    "alpha_index": args.alpha_index, "alpha_selection_performed": False})


if __name__ == "__main__":
    main()
