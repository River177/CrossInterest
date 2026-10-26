"""Freeze the four requested CIR5/NI and mixed conditions without altering accepted runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.data.build_cir_mixed_dataset import build
from new_interest.experiments import cir_warmup as core
from new_interest.experiments import cir_schedule as workflow


def prepare(project: Path, output: Path) -> dict:
    source = project / "artifacts/mind-cir3-20260917/bundle-v1"
    acceptance_path = project / "artifacts/mind-cir3-20260917/final_acceptance.json"
    acceptance = core.read(acceptance_path)
    old_manifest = core.read(source / "manifests/cir_bundle.json")
    if output.exists():
        raise FileExistsError(output)
    core.require(acceptance["status"] == "accepted" and sha256(source / "manifests/cir_bundle.json") == acceptance["bundle_sha256"], "Source CIR3 identity differs")
    for name, digest in old_manifest["files"].items():
        core.require(sha256(source / name) == digest, f"Source bytes changed: {name}")
    mixed_dir = project / workflow.MIXED.parent
    if not mixed_dir.exists():
        build(project / core.DATA / "cir.parquet", project / core.DATA / "ni.parquet", mixed_dir)
    metadata = core.read(mixed_dir / "metadata.json")
    core.require(metadata["rows"] == 38871 and metadata["by_task"] == {"cross_interest_reconstruction": 17271, "new_interest": 21600}, "Mixed source counts differ")
    core.require(sha256(mixed_dir / "mixed.parquet") == metadata["output_files"][0]["sha256"], "Mixed data digest differs")
    for file, record in zip(("cir", "ni"), metadata["source_files"]):
        core.require(record["sha256"] == sha256(source / core.DATA / f"{file}.parquet"), "Mixed source is not accepted data")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(mixed_dir, output / workflow.MIXED.parent)
    for name in ("src/new_interest/experiments/cir_schedule.py", "src/new_interest/data/build_cir_mixed_dataset.py",
                 "scripts/amlt/sft/run_cir_schedule.sh"):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
    contract = core.read(source / core.CONTRACT)
    contract.update(version="mind-cir-four-conditions-0918-v1", method="CIR5 -> NI3; CIR5 -> NI5; mixed3; mixed5",
        existing_cir3_reference={"bundle_sha256": acceptance["bundle_sha256"], "acceptance_sha256": sha256(acceptance_path),
            "metrics_sha256": acceptance["evaluation"]["metrics_sha256"], "evaluation": acceptance["evaluation"]},
        conditions={
            "cir5_ni3": {"method": "two_stage", "cir_epochs": 5, "ni_epochs": 3, "steps": [2700, 2025], "presentations": [86355, 64800]},
            "cir5_ni5": {"method": "two_stage", "cir_epochs": 5, "ni_epochs": 5, "steps": [2700, 3375], "presentations": [86355, 108000]},
            "mixed3": {"method": "mixed", "epochs": 3, "steps": [3645], "presentations": [116613], "by_task_presentations": {"cross_interest_reconstruction": 51813, "new_interest": 64800}},
            "mixed5": {"method": "mixed", "epochs": 5, "steps": [6075], "presentations": [194355], "by_task_presentations": {"cross_interest_reconstruction": 86355, "new_interest": 108000}}},
        mixed_data={"path": str(workflow.MIXED), "sha256": sha256(mixed_dir / "mixed.parquet"), "metadata_sha256": sha256(mixed_dir / "metadata.json")},
        comparison_limit="One seed. Epoch-matched staged5/5 and mixed5 have equal exposures and updates; staged resets optimizer/scheduler at handoff, mixed is continuous. NI-only has its historical execution caveat.")
    # Old dataset statistics remain provenance; all new horizons live in conditions.
    contract.pop("training_schedule", None)
    write_json(output / core.CONTRACT, contract)
    refs = output / "manifests/cir3_reference"
    refs.mkdir()
    shutil.copyfile(acceptance_path, refs / "acceptance.json")
    shutil.copyfile(source / "manifests/cir_bundle.json", refs / "source_bundle.json")
    reused = core.reuse_verified_gates(output, Path(contract["reuse_verified_gates"]))
    mixed_readback = workflow.verify_mixed_inputs(output)
    audits = {}
    for module_name in ("sft_exposure", "old_new_execution"):
        spec = importlib.util.spec_from_file_location(f"frozen_{module_name}", output / f"src/new_interest/training/{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        audits[module_name] = module
    stages = {}
    for mode in workflow.MODES:
        for stage in workflow.stage_specs(output, mode):
            key = stage["key"]
            if key in stages:
                continue
            ledger = audits["sft_exposure"].TrainingExposure.from_parquet(stage["train_file"], train_sha256=sha256(stage["train_file"]),
                expected_epochs=stage["epochs"], max_samples=None, default_task="new_interest")
            cursor = audits["old_new_execution"]
            cursor.STAGES.add(f"cir/{key}")
            checked = cursor.make_cursor_contract(stage["train_file"], stage=f"cir/{key}", world_size=4,
                per_device_batch_size=2, epochs=stage["epochs"], global_batch_size=32)
            steps = math.ceil(stage["samples"] / 32) * stage["epochs"]
            core.require(checked["total_steps"] == steps and ledger.identity["samples"] == stage["samples"], "Stage cursor differs")
            stages[key] = {"samples": stage["samples"], "epochs": stage["epochs"], "steps": steps,
                "presentations": stage["samples"] * stage["epochs"], "parent": stage["parent"],
                "train_sha256": sha256(stage["train_file"]), "cursor": checked, "ledger_identity": ledger.identity}
    changed = [name for name, digest in old_manifest["files"].items() if sha256(output / name) != digest]
    core.require(changed == [str(core.CONTRACT)], f"Unexpected changes to accepted runtime: {changed}")
    preflight = {"status": "passed", "source_bundle_sha256": acceptance["bundle_sha256"], "unchanged_prior_payload_files": len(old_manifest["files"])-1,
        "changed_prior_payload_files": changed, "mixed_readback": mixed_readback, "stages": stages,
        "reused_runtime_gates": reused, "new_mixed_gate": "41 examples, 2 four-GPU updates and exact reload plus 64 real Dev records per mixed job",
        "physical_jobs": 3, "conditions": 4, "maximum_concurrent_gpus": 12, "shared_cir5_updates_saved": 2700}
    write_json(output / "manifests/four_condition_preflight.json", preflight)
    payload = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".amltignore"
        and p != output / "manifests/cir_bundle.json"}
    write_json(output / "manifests/cir_bundle.json", {"version": "cir-four-condition-bundle-v1", "files": payload})
    canonical = project / "configs/experiments/mind_cir_four_conditions_0918_v1.json"
    if canonical.exists():
        raise FileExistsError(canonical)
    write_json(canonical, contract)
    result = {"status": "passed", "bundle": str(output.relative_to(project)), "bundle_sha256": sha256(output / "manifests/cir_bundle.json"),
              "preflight_sha256": sha256(output / "manifests/four_condition_preflight.json"), "conditions": contract["conditions"],
              "unchanged_prior_payload_files": preflight["unchanged_prior_payload_files"]}
    write_json(output.parent / "local_preflight.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project.resolve(), args.output.resolve()), indent=2))
