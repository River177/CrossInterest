"""Resume the completed single-GPU smoke across a validated metadata fix."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import shutil

from new_interest.experiments.backbone_scaling import (
    CONTRACT, SOURCE_MANIFEST, audit_adapter, audit_gradient, audit_training,
    identity, load_json, require, sha256,
)
from new_interest.training.semantic_id_tokens import write_json


def training_semantics_sha256(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    module.body = [node for node in module.body if not isinstance(node, ast.FunctionDef) or node.name != "_load_config"]
    return hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()


def reuse_completed_single(project: Path, size: str, output: Path, manifest_file: Path) -> None:
    m = load_json(manifest_file)
    require(m["version"] == "backbone-single-smoke-recovery-v1" and m["model_size"] == size,
            "Recovery belongs to another model")
    current = identity(project, size)
    for key in ("model_name", "model_revision", "contract_sha256", "sft_config_sha256", "eval_config_sha256"):
        require(m["source_identity"][key] == current[key], f"Recovery changes frozen condition: {key}")
    require(training_semantics_sha256(project / "src/new_interest/training/train_sft.py") == m["training_semantics_sha256"],
            "Recovery changes training code beyond the configuration reader")
    sources = load_json(project / SOURCE_MANIFEST)["files"]
    for name, digest in m["unchanged_training_sources"].items():
        require(sources[name] == digest == sha256(project / name), f"Recovery changes training dependency: {name}")
    source = Path(m["source_result_root"])
    require(load_json(source / "run_identity.json") == m["source_identity"], "Recovery source identity differs")
    for relative, item in m["files"].items():
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts, "Unsafe recovery path")
        p = source / p
        require(p.is_file() and p.stat().st_size == item["bytes"] and sha256(p) == item["sha256"],
                f"Recovery source bytes differ: {relative}")
    c = load_json(project / CONTRACT)
    require(load_json(source / "environment.json")["packages"] == c["worker_packages"], "Recovery environment differs")
    audit_gradient(load_json(source / "gradient/gradient_audit.json"), c["models"][size])
    training = audit_training(project, size, source / "single", "single")
    adapter = audit_adapter(project, size, source / "single")
    require(adapter["adapter_sha256"] == m["checkpoint_adapter_sha256"], "Completed adapter differs from committed checkpoint")
    for relative, item in m["files"].items():
        dest = output / (relative if relative.startswith(("single/", "gradient/")) else f"recovery_source/{relative}")
        if dest.exists():
            require(sha256(dest) == item["sha256"], f"Existing recovered artifact differs: {relative}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".copying")
        shutil.copyfile(source / relative, partial)
        require(sha256(partial) == item["sha256"], f"Recovery copy differs: {relative}")
        partial.replace(dest)
    write_json(output / "single_recovery_provenance.json", {
        "recovery_manifest_sha256": sha256(manifest_file), "source_identity": m["source_identity"],
        "current_identity": current, "training": training, "adapter": adapter,
        "copied_files": m["files"], "scope": "reuse gradient audit and completed 20-step single only; reload and recipe smoke remain required",
    })
    print("BACKBONE_REUSED_VERIFIED_SINGLE", size, adapter["adapter_sha256"], flush=True)
