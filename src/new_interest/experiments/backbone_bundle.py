"""Freeze and stage small self-contained Backbone bundles, or unpack readback."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import tarfile

from new_interest.experiments.backbone_scaling import (
    CONTRACT, SOURCE_MANIFEST, config_paths, load_json, require, sha256,
    validate_smoke_acceptance,
)
from new_interest.training.semantic_id_tokens import write_json


def runtime_paths(project: Path) -> list[Path]:
    queue = ["new_interest.experiments.backbone_scaling", "new_interest.experiments.backbone_bundle"]
    seen = set()
    paths = set()
    while queue:
        module = queue.pop(0)
        if module in seen:
            continue
        seen.add(module)
        relative = Path("src") / (module.replace(".", "/") + ".py")
        path = project / relative
        require(path.is_file(), f"Missing runtime module: {module}")
        paths.add(relative)
        for parent in relative.parents:
            init = parent / "__init__.py"
            if (project / init).is_file():
                paths.add(init)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("new_interest."):
                queue.append(node.module)
            elif isinstance(node, ast.Import):
                queue.extend(item.name for item in node.names if item.name.startswith("new_interest."))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.fullmatch(r"new_interest(?:\.[a-z_]+)+", node.value):
                    queue.append(node.value)
    paths.update([CONTRACT, Path("requirements-qwen35-worker.txt"), Path("requirements-backbone-qwen35-worker.txt"),
                  Path("scripts/amlt/sft/run_mind_backbone_qwen35.sh")])
    for size in ("2b", "4b"):
        paths.update(config_paths(size))
    paths.update(p.relative_to(project) for p in (project / "manifests").glob("mind_backbone_single_recovery_*_v1.json"))
    return sorted(paths)


def freeze(project: Path) -> dict:
    manifest = {"version": "mind-backbone-runtime-sources-v2",
                "files": {str(p): sha256(project / p) for p in runtime_paths(project)}}
    sealed = project / "artifacts/mind-backbone-qwen35-21600-v1/runtime_sealed_v2.json"
    if sealed.is_file():
        require(load_json(project / SOURCE_MANIFEST) == manifest, "Submitted runtime is sealed; create a separately reviewed revision")
    write_json(project / SOURCE_MANIFEST, manifest)
    return manifest


def stage(project: Path, output: Path, *, size: str | None = None, acceptance: Path | None = None) -> dict:
    sources = load_json(project / SOURCE_MANIFEST)["files"]
    require(sources == {str(p): sha256(project / p) for p in runtime_paths(project)}, "Runtime must be frozen before staging")
    inputs = load_json(project / CONTRACT)["input_files"]
    expected = {**sources, **{i["path"]: i["sha256"] for i in inputs}, str(SOURCE_MANIFEST): sha256(project / SOURCE_MANIFEST)}
    origin = Path("manifests/mind_backbone_qwen35_21600_source_v1.json")
    expected[str(origin)] = sha256(project / origin)
    extra = {}
    if size is not None:
        validate_smoke_acceptance(project, size, acceptance)
        require(acceptance is not None, "Missing smoke receipt")
        name = f"manifests/smoke_acceptances/{size}.json"
        extra[name] = acceptance
        expected[name] = sha256(acceptance)
    elif acceptance is not None:
        raise ValueError("A receipt requires its model size")
    if output.exists():
        inventory = load_json(output / "bundle_inventory.json")
        require(inventory["files"] == expected, "Existing bundle differs; do not overwrite a frozen upload")
        require(all(sha256(output / name) == digest for name, digest in expected.items()), "Existing bundle bytes changed")
        return inventory
    output.mkdir(parents=True)
    for name, digest in expected.items():
        source = extra.get(name, project / name)
        require(sha256(source) == digest, f"Staging source changed: {name}")
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        require(sha256(dest) == digest, f"Staged bytes differ: {name}")
    (output / ".amltignore").write_text("__pycache__/\n*.pyc\n")
    inventory = {"version": "mind-backbone-bundle-v1", "files": expected,
                 "bytes": sum((output / name).stat().st_size for name in expected)}
    write_json(output / "bundle_inventory.json", inventory)
    return inventory


def extract(archive: Path, receipt: Path, output: Path) -> None:
    package = load_json(receipt)
    digest = sha256(archive)
    require(package["sha256"] == digest and package["bytes"] == archive.stat().st_size, "Downloaded package differs")
    marker = output / "extraction_receipt.json"
    if output.exists():
        require(marker.is_file() and load_json(marker)["archive_sha256"] == digest, "Existing extracted results differ")
        return
    output.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            p = Path(member.name)
            require(member.isfile() and not p.is_absolute() and ".." not in p.parts, "Unsafe archive member")
            require(member.name not in names, "Duplicate archive member")
            names.add(member.name)
        for member in members:
            dest = output / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, dest.open("wb") as target:
                shutil.copyfileobj(stream, target)
    manifest = load_json(output / "artifact_manifest.json")
    require(names == set(manifest["files"]) | {"artifact_manifest.json", f"{manifest['phase']}_success.json"}, "Archive inventory differs")
    write_json(marker, {"archive_sha256": digest, "files": len(names)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "stage", "extract"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-size", choices=("2b", "4b"))
    parser.add_argument("--smoke-acceptance", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--package-receipt", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.project.resolve())
        print(json.dumps({"runtime_files": len(result["files"]), "manifest_sha256": sha256(args.project / SOURCE_MANIFEST)}))
    elif args.action == "stage":
        require(args.output is not None, "Missing bundle directory")
        result = stage(args.project.resolve(), args.output.resolve(), size=args.model_size, acceptance=args.smoke_acceptance)
        print(json.dumps({"bundle_files": len(result["files"]), "bytes": result["bytes"]}))
    else:
        require(args.archive is not None and args.package_receipt is not None and args.output is not None, "Missing extraction arguments")
        extract(args.archive, args.package_receipt, args.output)


if __name__ == "__main__":
    main()
