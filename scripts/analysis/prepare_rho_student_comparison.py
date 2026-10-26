"""Prepare (never submit) a frozen CPU comparison after all student arms pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from new_interest.infrastructure.paths import portable_path

VERSION = "rho-student-comparison-bundle-v1"
ARMS = ("uniform", "static", "dynamic")
COMPARISON_MODULE = "src/new_interest/experiments/compare_rho_students.py"
PACKAGES = {"numpy": "2.2.6", "pyarrow": "20.0.0", "scipy": "1.15.3"}
PLAN_FIELDS = {"version": "rho-student-comparison-plan-v1", "primary_metric": "macro_ndcg_at_20",
    "primary_contrast": "dynamic_minus_static", "primary_split": "validation", "secondary_split": "dev",
    "bootstrap_draws": 2000, "seed": 42, "unit": "user", "inference_run": False, "source": COMPARISON_MODULE}
UAI = "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_acceptance(accepted, protocol, frozen, protocol_sha):
    if (accepted.get("passed") is not True or set(accepted.get("arms", {})) != set(ARMS)
            or accepted.get("same_initial_state") is not True or accepted.get("same_reference_scores") is not True):
        raise ValueError("Actual passed three-arm acceptance is required before preparing comparison")
    shared = {"runtime_sha256": frozen["runtime_sha256"], "protocol_sha256": protocol_sha,
        "inputs_sha256": canonical({k: v for k, v in frozen["files"].items() if k.startswith("inputs/")}),
        "reference_acceptance_sha256": protocol["reference_acceptance_sha256"]}
    initial_states, reference_scores = set(), set()
    for arm in ARMS:
        worker = accepted["arms"][arm]["worker"]
        if (worker.get("passed") is not True or worker["identity"].get("phase") != "full"
                or worker["identity"].get("arm") != arm or set(worker.get("evaluations", {})) != {"validation", "dev"}
                or any(worker["identity"].get(key) != value for key, value in shared.items())):
            raise ValueError("Accepted student arm differs from the frozen training experiment")
        trained = worker["training"]["full"]
        if (trained.get("passed") is not True
                or (trained["rows"], trained["steps"], trained["epochs"], trained["world_size"]) != (21600, 2025, 3, 4)):
            raise ValueError("Student arm has not completed the frozen full training endpoint")
        initial_states.add(trained["initial_state_sha256"])
        reference_scores.add(trained["student_summary"]["reference_file_sha256"])
        for split in ("validation", "dev"):
            evaluation = worker["evaluations"][split]
            if (evaluation.get("accepted") is not True or evaluation.get("split") != split
                    or evaluation.get("limit") is not None
                    or evaluation.get("rows") != protocol["evaluation"][split + "_rows"]
                    or evaluation.get("source_sha256") != frozen["files"][f"inputs/{split}.parquet"]
                    or evaluation.get("protocol_sha256") != protocol_sha
                    or evaluation.get("kernel_configs_consistent") is not True):
                raise ValueError("Student heldout acceptance is incomplete or belongs to different frozen inputs")
    if len(initial_states) != 1 or len(reference_scores) != 1:
        raise ValueError("Student arms used different initial states or reference scores")


BOOTSTRAP = r'''"""Verify sealed bytes and package versions before importing comparison code."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify(root, expected_manifest_sha, check_runtime=False):
    root=Path(root)
    if sha(root/'source_manifest.json')!=expected_manifest_sha:raise ValueError('Comparison source manifest changed')
    manifest=json.loads((root/'source_manifest.json').read_text())
    if manifest.get('version')!='rho-student-comparison-bundle-v1':raise ValueError('Unknown comparison bundle')
    for name,digest in manifest['files'].items():
        relative=Path(name)
        path=root/relative
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or path.is_symlink() or sha(path)!=digest:
            raise ValueError('Frozen comparison file changed: '+name)
    inputs=json.loads((root/'input_manifest.json').read_text())
    if canonical(inputs['files'])!=manifest['inputs_sha256']:raise ValueError('Comparison input identity changed')
    if any(manifest['files'].get(name)!=digest for name,digest in inputs['files'].items()):
        raise ValueError('Input manifest differs from sealed source')
    acceptance=json.loads((root/'inputs/three_arm_acceptance.json').read_text())
    if (acceptance.get('passed') is not True or set(acceptance.get('arms',{}))!={'uniform','static','dynamic'}
            or acceptance.get('same_initial_state') is not True or acceptance.get('same_reference_scores') is not True):
        raise ValueError('Missing complete three-arm acceptance')
    if sha(root/'inputs/three_arm_acceptance.json')!=manifest['acceptance_sha256']:
        raise ValueError('Accepted result identity changed')
    plan=json.loads((root/'comparison_plan.json').read_text())
    if (plan.get('primary_metric')!='macro_ndcg_at_20' or plan.get('primary_contrast')!='dynamic_minus_static'
            or plan.get('primary_split')!='validation' or plan.get('bootstrap_draws')!=2000
            or plan.get('seed')!=42 or plan.get('unit')!='user' or plan.get('inference_run') is not False
            or sha(root/'src/new_interest/experiments/compare_rho_students.py')!=plan['comparison_code_sha256']
            or sha(root/'protocol.json')!=plan['training_protocol_sha256']):
        raise ValueError('Frozen comparison plan changed')
    if check_runtime:
        actual={name:importlib.metadata.version(name) for name in manifest['packages']}
        if actual!=manifest['packages']:raise ValueError('Comparison numeric package versions differ')
    return {'verified':True,'source_manifest_sha256':expected_manifest_sha,'runtime_checked':check_runtime}

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--check-runtime',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.manifest_sha256,args.check_runtime)))
'''


def prepare_comparison(three_arm_acceptance_file, output_bundle, output_yaml, *, frozen_bundle=None):
    acceptance_file = Path(three_arm_acceptance_file).resolve()
    bundle, yaml_file = Path(output_bundle).resolve(), Path(output_yaml).resolve()
    # Read first: an absent/incomplete training result must not create a bundle.
    accepted = read(acceptance_file)
    root = REPO / "artifacts/rho-student-20260916"
    frozen_dir = Path(frozen_bundle).resolve() if frozen_bundle is not None else root / "bundle-gate-v1"
    plan_path = root / "comparison_plan.json"
    if bundle.exists() or yaml_file.exists():
        raise FileExistsError("Comparison outputs are immutable; choose fresh bundle/config paths")
    if bundle.is_relative_to(frozen_dir) or yaml_file.is_relative_to(frozen_dir) or yaml_file.is_relative_to(bundle):
        raise ValueError("Comparison outputs must not alter frozen training or live inside the upload bundle")
    frozen = read(frozen_dir / "source_manifest.json")
    protocol, plan = read(frozen_dir / "protocol.json"), read(plan_path)
    protocol_sha = sha(frozen_dir / "protocol.json")
    if frozen["files"].get("protocol.json") != protocol_sha or plan.get("training_protocol_sha256") != protocol_sha:
        raise ValueError("Comparison plan is not bound to the frozen training protocol")
    if any(plan.get(key) != value for key, value in PLAN_FIELDS.items()):
        raise ValueError("Comparison plan changes the frozen primary NDCG@20 analysis")
    evaluation = protocol["evaluation"]
    if (evaluation.get("primary") != "macro_ndcg_at_20" or evaluation.get("paired_bootstrap_unit") != "user"
            or evaluation.get("paired_bootstrap_replicates") != 2000 or evaluation.get("seed") != 42
            or evaluation.get("validation_main_min_class_support") != 20):
        raise ValueError("Training protocol and comparison analysis disagree")
    comparison = REPO / COMPARISON_MODULE
    if sha(comparison) != plan.get("comparison_code_sha256"):
        raise ValueError("Completed comparison code differs from comparison_plan.json")
    validate_acceptance(accepted, protocol, frozen, protocol_sha)
    frozen_sources = {name for name in frozen["files"] if name.startswith("src/")}
    actual_sources = {p.relative_to(frozen_dir).as_posix() for p in (frozen_dir/"src").rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if not frozen_sources or actual_sources != frozen_sources:
        raise ValueError("Frozen training source tree differs from its authenticated inventory")
    names = sorted(frozen_sources | {"protocol.json", "inputs/validation.parquet", "inputs/dev.parquet",
                                    "inputs/interest_token_mapping.parquet"})
    for name in names:
        source = frozen_dir / name
        if source.is_symlink() or sha(source) != frozen["files"][name]:
            raise ValueError("Frozen training source/input changed: " + name)
    acceptance_sha, plan_sha = sha(acceptance_file), sha(plan_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="."+bundle.name+"-", dir=bundle.parent))
    try:
        for name in names:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frozen_dir / name, destination)
        (staging / COMPARISON_MODULE).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(comparison, staging / COMPARISON_MODULE)
        shutil.copyfile(acceptance_file, staging / "inputs/three_arm_acceptance.json")
        shutil.copyfile(plan_path, staging / "comparison_plan.json")
        shutil.copyfile(frozen_dir / "source_manifest.json", staging / "training_source_manifest.json")
        (staging / "bootstrap_verify.py").write_text(BOOTSTRAP)
        inputs = {name: sha(staging / name) for name in (
            "inputs/three_arm_acceptance.json", "inputs/validation.parquet", "inputs/dev.parquet", "inputs/interest_token_mapping.parquet")}
        write(staging / "input_manifest.json", {"files": inputs, "acceptance_sha256": acceptance_sha,
              "original_sources": {"acceptance": portable_path(acceptance_file, root=REPO),
                                   "frozen_training_bundle": portable_path(frozen_dir, root=REPO)}})
        inventory = {p.relative_to(staging).as_posix(): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        if inventory[COMPARISON_MODULE] != plan["comparison_code_sha256"] or inventory["inputs/three_arm_acceptance.json"] != acceptance_sha:
            raise ValueError("Comparison overlay/acceptance bytes changed while staging")
        if any(name.endswith((".safetensors", ".pt", ".bin", ".tar", ".tar.gz", ".zip")) for name in inventory):
            raise ValueError("Comparison bundle must not contain models or result archives")
        manifest = {"version": VERSION, "files": inventory, "inputs_sha256": canonical(inputs),
            "acceptance_sha256": acceptance_sha, "comparison_plan_sha256": plan_sha,
            "comparison_code_sha256": plan["comparison_code_sha256"], "packages": PACKAGES,
            "frozen_training_runtime_sha256": frozen["runtime_sha256"],
            "frozen_training_source_manifest_sha256": sha(frozen_dir / "source_manifest.json"),
            "large_artifact_policy": "remote prediction references only; no models or predictions downloaded"}
        write(staging / "source_manifest.json", manifest)
        manifest_sha = sha(staging / "source_manifest.json")
        relative = os.path.relpath(bundle, yaml_file.parent).replace(os.sep, "/")
        command = ["set -euo pipefail", 'cd "$$AMLT_CODE_DIR"',
            'export PYTHONPATH="$$AMLT_CODE_DIR/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
            f"python bootstrap_verify.py --manifest-sha256 {manifest_sha}",
            "python -m pip install --user --no-deps --no-warn-script-location --retries 4 --timeout 90 numpy==2.2.6 pyarrow==20.0.0 scipy==1.15.3",
            f"python bootstrap_verify.py --manifest-sha256 {manifest_sha} --check-runtime",
            "python -m new_interest.experiments.compare_rho_students --acceptance inputs/three_arm_acceptance.json "
            f"--acceptance-sha256 {acceptance_sha} --validation-file inputs/validation.parquet --dev-file inputs/dev.parquet "
            '--mapping-file inputs/interest_token_mapping.parquet --output "$$AMLT_OUTPUT_DIR/comparison.json"']
        config = {"description": "Frozen three-arm RHO comparison: paired user bootstrap, validation macro NDCG@20 primary; CPU analysis only.",
            "environment": {"image": "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04", "setup": []},
            "target": {"service": "sing", "name": "Feeds", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"},
            "code": {"local_dir": "$CONFIG_DIR/"+relative, "ignore": ["__pycache__/", "*.pyc"]},
            "jobs": [{"name": "rho-student-three-arm-comparison", "sku": "5C3", "sla_tier": "Basic",
                "submit_args": {"env": {"AMLT_DOCKERFILE_TEMPLATE": "DEFAULT", "_AZUREML_SINGULARITY_JOB_UAI": UAI}},
                "command": command}]}
        staging.rename(bundle)
        with yaml_file.open("x") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return {"ready": True, "version": VERSION, "bundle": str(bundle), "config": str(yaml_file),
            "acceptance_sha256": acceptance_sha, "source_manifest_sha256": manifest_sha,
            "comparison_code_sha256": plan["comparison_code_sha256"], "comparison_plan_sha256": plan_sha,
            "config_sha256": sha(yaml_file), "submitted": False, "gpu_count": 0}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_comparison(args.acceptance, args.output_bundle, args.output_yaml,
                                       frozen_bundle=args.frozen_bundle), ensure_ascii=False))


if __name__ == "__main__":
    main()
