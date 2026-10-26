"""Build an external storage-only launcher around a byte-identical frozen base."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


REPO = Path(__file__).resolve().parents[2]
LAUNCHER_SOURCE = REPO / "scripts/amlt/sft/rho_storage_recovery_worker.py"


def launcher_module():
    spec = importlib.util.spec_from_file_location("rho_storage_recovery_packaging_helpers", LAUNCHER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compatibility_wrapper(original: str) -> str:
    boundary = 'if [[ "${1:?}" == "diagnose-native" ]]; then\n'
    if original.count(boundary) != 1:
        raise ValueError("Original worker shell entrypoint has an unexpected shape")
    prefix, suffix = original.split(boundary, 1)
    expected_suffix = (
        '  python -m new_interest.experiments.rho_student_experiment diagnose-native \\\n'
        '    --project "$AMLT_CODE_DIR" --root "$AMLT_OUTPUT_DIR"\n'
        'else\n'
        '  python -m new_interest.experiments.rho_student_experiment worker \\\n'
        '    --project "$AMLT_CODE_DIR" --root "$AMLT_OUTPUT_DIR" --phase "$1" --arm "${2:?}"\n'
        'fi\n'
    )
    if suffix != expected_suffix or prefix.count('cd "${AMLT_CODE_DIR:?}"\n') != 1:
        raise ValueError("Original worker install/dispatch boundary changed")
    redirect = (
        'RHO_STORAGE_RECOVERY_PACKAGE_DIR="${AMLT_CODE_DIR:?}"\n'
        'if [[ "$#" != 2 || "${1-}" != full || ( "${2-}" != static && "${2-}" != dynamic ) ]]; then\n'
        '  echo "Storage recovery only supports: full static or full dynamic" >&2\n'
        '  exit 2\n'
        'fi\n'
        'export AMLT_CODE_DIR="$RHO_STORAGE_RECOVERY_PACKAGE_DIR/base"\n'
        'cd "${AMLT_CODE_DIR:?}"\n'
    )
    prefix = prefix.replace('cd "${AMLT_CODE_DIR:?}"\n', redirect)
    return prefix + (
        'python "$RHO_STORAGE_RECOVERY_PACKAGE_DIR/recovery_worker.py" \\\n'
        '  --package "$RHO_STORAGE_RECOVERY_PACKAGE_DIR" --root "$AMLT_OUTPUT_DIR" \\\n'
        '  --phase "$1" --arm "$2"\n'
    )


def prepare_bundle(base_bundle: Path, output: Path) -> dict:
    helper = launcher_module()
    base_bundle, output = Path(base_bundle).resolve(), Path(output).resolve()
    if output == base_bundle or base_bundle in output.parents or output in base_bundle.parents:
        raise ValueError("Recovery output may not overlap the frozen base")
    if output.exists():
        raise FileExistsError("Recovery bundle already exists: " + str(output))
    source, _ = helper.verify_base(base_bundle)
    wrapper = compatibility_wrapper((base_bundle / helper.COMPAT_PATH).read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        for name in sorted(set(source["files"]) | {"source_manifest.json"}):
            destination = staging / "base" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(helper.safe_file(base_bundle, name), destination)
        launcher = staging / helper.LAUNCHER_PATH
        launcher.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LAUNCHER_SOURCE, launcher)
        (staging / helper.COMPAT_PATH).parent.mkdir(parents=True, exist_ok=True)
        (staging / helper.COMPAT_PATH).write_text(wrapper)
        files = {name: helper.sha(staging / name) for name in sorted(helper.inventory(staging))}
        manifest = {
            "version": helper.VERSION, "arms": list(helper.SUPPORTED_ARMS), "stage": "full", "base_directory": "base",
            "base_runtime_sha256": source["runtime_sha256"],
            "base_protocol_sha256": helper.sha(base_bundle / "protocol.json"),
            "base_source_manifest_sha256": helper.sha(base_bundle / "source_manifest.json"),
            "base_gate_acceptance_sha256": helper.sha(base_bundle / "gate_acceptance.json"),
            "launcher_path": helper.LAUNCHER_PATH, "launcher_sha256": files[helper.LAUNCHER_PATH],
            "compat_wrapper_path": helper.COMPAT_PATH, "compat_wrapper_sha256": files[helper.COMPAT_PATH],
            "changed_fields": ["persistent_checkpoint_limit"], "old_value": helper.OLD_VALUE,
            "new_value": helper.NEW_VALUE, "planned_optimizer_steps": helper.PLANNED_STEPS,
            "planned_max_checkpoint_count": helper.PLANNED_CHECKPOINTS, "files": files,
        }
        helper.write(staging / helper.MANIFEST_NAME, manifest)
        helper.verify_package(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    context = helper.verify_package(output)
    return {"ready": True, "bundle": str(output), "arms": list(helper.SUPPORTED_ARMS), **context["identity"],
            "compat_wrapper_sha256": context["manifest"]["compat_wrapper_sha256"],
            "changed_fields": ["persistent_checkpoint_limit"], "old_value": helper.OLD_VALUE,
            "new_value": helper.NEW_VALUE, "planned_optimizer_steps": helper.PLANNED_STEPS,
            "planned_max_checkpoint_count": helper.PLANNED_CHECKPOINTS,
            "base_file_count": len(source["files"]) + 1, "submitted": False}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.base_bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
