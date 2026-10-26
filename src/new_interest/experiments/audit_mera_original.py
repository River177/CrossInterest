from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from new_interest.experiments.mera_original_core import (
    audit_dataset,
    files_manifest,
    load_json,
    load_resolved_config,
    sha256_file,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_source(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    combined = {
        "audit_version": "mera-original-combined-source-audit-v1",
        "datasets": {},
    }
    for dataset in ("arts", "grocery"):
        config = load_resolved_config(args.config, dataset, "code-faithful")
        result = audit_dataset(config)
        combined["datasets"][dataset] = result
        write_json(output_root / f"{dataset}_source_audit.json", result)
    write_json(output_root / "source_audit.json", combined)
    print("MERA_ORIGINAL_SOURCE_AUDIT_OK", json.dumps({
        key: value["splits"] for key, value in combined["datasets"].items()
    }, sort_keys=True))


def _copy_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)


def _copy_tree_content_only(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith("checkpoint-") for part in relative.parts):
            continue
        if path.is_file():
            _copy_bytes(path, destination / path.relative_to(source))


def _validate_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = load_json(manifest_path)
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Published artifact mismatch: {relative}")
    return manifest


def publish_run(args: argparse.Namespace) -> None:
    source = Path(args.source_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    config = load_resolved_config(args.config, args.dataset, args.contract)
    identity = config.identity()
    success_name = "smoke_success.json" if args.mode == "smoke" else "full_success.json"
    success_path = output_root / success_name
    manifest_path = output_root / "artifact_manifest.json"
    published_root = output_root / "run"

    if success_path.is_file() and manifest_path.is_file():
        success = load_json(success_path)
        if success["identity"] != identity:
            raise RuntimeError("Committed result identity differs from requested run")
        if sha256_file(manifest_path) != success["artifact_manifest_sha256"]:
            raise RuntimeError("Committed result manifest hash mismatch")
        manifest = _validate_manifest(published_root, manifest_path)
        print("MERA_ORIGINAL_PUBLISHED_REUSE_OK", len(manifest["files"]))
        return

    if not source.is_dir():
        raise RuntimeError(f"Missing local run directory: {source}")
    required = (
        "resolved_config.json",
        "environment.json",
        "tokenizer_audit.json",
        "regularizer_audit.json",
        "gradient_audit.json",
        "run_summary.json",
        "reload_reference.json",
        "adapter_state_reference.json",
        "evaluation/metrics.json",
        "evaluation/predictions.jsonl",
        "evaluation/reload_audit.json",
        "final_adapter/adapter_config.json",
    )
    missing = [value for value in required if not (source / value).is_file()]
    if missing:
        raise RuntimeError(f"Cannot publish incomplete run: {missing}")

    staging = output_root / ".run.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _copy_tree_content_only(source, staging)
    staged_files = files_manifest(staging)
    for relative, expected in staged_files.items():
        if sha256_file(staging / relative) != expected:
            raise RuntimeError(f"Staging verification failed: {relative}")
    if published_root.exists():
        shutil.rmtree(published_root)
    os.replace(staging, published_root)
    manifest = {
        "manifest_version": "mera-original-artifact-manifest-v1",
        "identity": identity,
        "files": staged_files,
    }
    write_json(manifest_path, manifest)
    success = {
        "status": "pass",
        "mode": args.mode,
        "dataset": args.dataset,
        "contract": args.contract,
        "identity": identity,
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_files": len(staged_files),
    }
    write_json(success_path, success)
    _validate_manifest(published_root, manifest_path)
    print("MERA_ORIGINAL_PUBLISH_OK", json.dumps(success, sort_keys=True))


def verify_published(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir).resolve()
    success_name = "smoke_success.json" if args.mode == "smoke" else "full_success.json"
    success_path = output_root / success_name
    manifest_path = output_root / "artifact_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Published success/manifest is missing")
    success = load_json(success_path)
    config = load_resolved_config(args.config, args.dataset, args.contract)
    if success["identity"] != config.identity():
        raise RuntimeError("Published identity mismatch")
    if sha256_file(manifest_path) != success["artifact_manifest_sha256"]:
        raise RuntimeError("Published manifest hash mismatch")
    manifest = _validate_manifest(output_root / "run", manifest_path)
    print("MERA_ORIGINAL_PUBLISHED_VERIFY_OK", len(manifest["files"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source")
    source.add_argument("--config", required=True)
    source.add_argument("--output-dir", required=True)
    source.set_defaults(func=audit_source)

    for name, func in (("publish", publish_run), ("verify-published", verify_published)):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--dataset", required=True, choices=("arts", "grocery"))
        command.add_argument(
            "--contract", required=True, choices=("code-faithful", "paper-spec")
        )
        command.add_argument("--mode", required=True, choices=("smoke", "full"))
        command.add_argument("--output-dir", required=True)
        if name == "publish":
            command.add_argument("--source-dir", required=True)
        command.set_defaults(func=func)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
