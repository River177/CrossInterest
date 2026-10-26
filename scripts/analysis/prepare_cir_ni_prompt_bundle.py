"""Freeze the approved prompt-only CIR5 -> NI3 condition on accepted runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.data.build_cir_ni_prompt_dataset import build, VERSION
from new_interest.experiments import cir_warmup as core


def prepare(project: Path, output: Path) -> dict:
    source = project / "artifacts/mind-cir-four-20260918/bundle-v1"
    acceptance_file = project / "artifacts/mind-cir-four-20260918/acceptance/cir5_ni3.json"
    baseline = core.read(acceptance_file)
    old_manifest = core.read(source / "manifests/cir_bundle.json")
    if output.exists():
        raise FileExistsError(output)
    core.require(baseline["status"] == "accepted" and baseline["condition"] == "cir5_ni3", "Baseline not accepted")
    core.require(sha256(source / "manifests/cir_bundle.json") == baseline["bundle_sha256"], "Baseline bundle differs")
    for rel, digest in old_manifest["files"].items():
        core.require(sha256(source / rel) == digest, "Source bytes changed: " + rel)
    # Do not pick up unrelated, currently modified training code from the workspace.
    core.require(sha256(source / "src/new_interest/experiments/cir_warmup.py")
                 == sha256(project / "src/new_interest/experiments/cir_warmup.py"), "Orchestrator differs")
    dataset = project / "data/training/mind_cir_ni_prompt_20260928_v1"
    data_audit = build(source / core.DATA, dataset)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for item in dataset.iterdir():
        if item.is_file():
            shutil.copyfile(item, output / core.DATA / item.name)
    contract = core.read(source / core.CONTRACT)
    for key in ("reuse_verified_gates", "existing_cir1_reference", "existing_cir3_reference", "conditions", "mixed_data"):
        contract.pop(key, None)
    contract.update(version=VERSION,
                    method="Exact NI prompt on unchanged CIR samples: fresh CIR5 then original NI3",
                    training_schedule={"cir_epochs": 5, "ni_epochs": 3},
                    data=core.read(dataset / "statistics.json"),
                    data_metadata_sha256=sha256(dataset / "metadata.json"),
                    prompt_only_baseline={"acceptance_sha256": sha256(acceptance_file),
                                          "bundle_sha256": baseline["bundle_sha256"],
                                          "evaluation": baseline["evaluation"]},
                    comparison_limit="One seed; only CIR prompt and associated provenance change. Gates are rerun for the new prompt; formal runs start from the same frozen initialization.")
    write_json(output / core.CONTRACT, contract)
    refs = output / "manifests/prompt_baseline"
    refs.mkdir()
    shutil.copyfile(acceptance_file, refs / "acceptance.json")
    shutil.copyfile(source / "manifests/cir_bundle.json", refs / "source_bundle.json")
    # Preserve the verified storage-only EIO retry used by the accepted baseline recovery.
    recovery = project / "artifacts/mind-cir-four-20260918/bundle-staged-recovery-0919-v1/recovery"
    overrides = output / "runtime_overrides"
    overrides.mkdir()
    for name in ("sitecustomize.py", "cir_storage_retry.py"):
        shutil.copyfile(recovery / name, overrides / name)
    wrapper = (source / "scripts/amlt/sft/run_cir_warmup.sh").read_text()
    wrapper = wrapper.replace('python -m new_interest.experiments.cir_warmup worker',
        'export CIR_STORAGE_RETRY_ENABLED=1\nexport PYTHONPATH="$AMLT_CODE_DIR/runtime_overrides:$PYTHONPATH"\n'
        'python -m new_interest.experiments.cir_warmup worker')
    wrapper_path = output / "scripts/amlt/sft/run_cir_ni_prompt.sh"
    wrapper_path.write_text(wrapper)
    shutil.copyfile(wrapper_path, project / "scripts/amlt/sft/run_cir_ni_prompt.sh")
    for name in ("src/new_interest/data/build_cir_ni_prompt_dataset.py",):
        shutil.copyfile(project / name, output / name)
    modules = {}
    for name in ("sft_exposure", "old_new_execution"):
        spec = importlib.util.spec_from_file_location("frozen_" + name, output / f"src/new_interest/training/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    stages = {}
    for spec in core.stage_specs(output, output / "formal", gates=False):
        ledger = modules["sft_exposure"].TrainingExposure.from_parquet(
            spec["train_file"], train_sha256=sha256(spec["train_file"]),
            expected_epochs=spec["epochs"], max_samples=None, default_task="new_interest")
        cursor = modules["old_new_execution"]
        cursor.STAGES.add("cir/" + spec["key"])
        checked = cursor.make_cursor_contract(spec["train_file"], stage="cir/" + spec["key"],
            world_size=4, per_device_batch_size=2, epochs=spec["epochs"], global_batch_size=32)
        core.require(checked["total_steps"] == (2700 if spec["task"] == "cir" else 2025), "Wrong training budget")
        stages[spec["key"]] = {"samples": spec["samples"], "epochs": spec["epochs"],
            "steps": checked["total_steps"], "presentations": spec["samples"] * spec["epochs"],
            "parent": spec["parent"], "cursor": checked, "exposure_identity": ledger.identity}
    changed = [rel for rel, digest in old_manifest["files"].items() if sha256(output / rel) != digest]
    expected = {str(core.CONTRACT), *(str(core.DATA / name) for name in ("cir.parquet", "metadata.json", "statistics.json"))}
    core.require(set(changed) == expected, "Unexpected baseline changes: " + repr(changed))
    preflight = {"status": "passed", "source_bundle_sha256": baseline["bundle_sha256"],
        "baseline_acceptance_sha256": sha256(acceptance_file), "data_audit": data_audit,
        "changed_prior_payload_files": changed, "unchanged_prior_payload_files": len(old_manifest["files"])-len(changed),
        "stages": stages, "physical_jobs": 1, "maximum_concurrent_gpus": 4,
        "gates": "Fresh single-CIR/NI and four-GPU CIR/NI, exact reloads and 64 real Dev records; all diagnostic weights excluded from formal training",
        "storage_override": "Byte-identical accepted recovery EIO-only bounded retries"}
    write_json(output / "manifests/prompt_preflight.json", preflight)
    files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".amltignore"
        and p != output / "manifests/cir_bundle.json"}
    write_json(output / "manifests/cir_bundle.json", {"version": VERSION, "files": files})
    write_json(project / "configs/experiments/mind_cir_ni_prompt_0928_v1.json", contract)
    result = {**preflight, "bundle": str(output.relative_to(project)),
              "bundle_sha256": sha256(output / "manifests/cir_bundle.json")}
    write_json(output.parent / "local_preflight.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project.resolve(), args.output.resolve()), indent=2))
