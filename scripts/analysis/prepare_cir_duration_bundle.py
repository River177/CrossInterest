"""Freeze CIR 3 epochs -> NI 3 epochs from the accepted one-epoch CIR runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.experiments.cir_warmup import formal_schedule, reuse_verified_gates, stage_specs


def prepare(project: Path, output: Path) -> dict:
    source = project / "artifacts/mind-cir-20260916/bundle-v3"
    acceptance_path = project / "artifacts/mind-cir-20260916/final_acceptance.json"
    readback = project / "artifacts/mind-cir-20260916/final-readback/mind-cir-108-0916-v3/qwen35-2b-cir-ni-108"
    acceptance = json.loads(acceptance_path.read_text())
    original = json.loads((source / "manifests/cir_bundle.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    if acceptance["status"] != "accepted" or sha256(source / "manifests/cir_bundle.json") != acceptance["bundle_sha256"]:
        raise ValueError("Source runtime is not the accepted CIR condition")
    for name, digest in original["files"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Source payload changed: {name}")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = "src/new_interest/experiments/cir_warmup.py"
    shutil.copyfile(project / runner, output / runner)
    contract_path = output / "configs/experiments/mind_cir_108_v1.json"
    contract = json.loads(contract_path.read_text())
    contract.update(version="mind-cir-108-cir3-ni3-v1",
                    method="CIR 3 epochs then original NI 3 epochs; fresh base initialization; inherit LoRA/SID and reset optimizer/scheduler at handoff",
                    training_schedule={"cir_epochs": 3, "ni_epochs": 3},
                    reuse_verified_gates="manifests/cir1_reference")
    contract["data"].update(cir_epochs=3, cir_optimizer_steps=1620)
    contract["comparison_limit"] = "CIR3 vs accepted CIR1 shares data/model/runtime but has a longer warm-up and scheduler horizon; historical NI-only uses a different execution schedule"
    contract["existing_cir1_reference"] = {
        "experiment": "mind-cir-108-0916-v3", "backend": "dynamic_insect_xyybk1nb24",
        "bundle_sha256": acceptance["bundle_sha256"], "acceptance_sha256": sha256(acceptance_path),
        "metrics_sha256": acceptance["evaluation"]["metrics_sha256"],
        "evaluation": acceptance["evaluation"],
    }
    write_json(contract_path, contract)
    reference = output / contract["reuse_verified_gates"]
    reference.mkdir()
    for old, new in ((acceptance_path, "acceptance.json"),
                     (source / "manifests/cir_bundle.json", "source_bundle.json"),
                     (source / "configs/experiments/mind_cir_108_v1.json", "source_contract.json"),
                     (readback / "complete.json", "complete.json")):
        shutil.copyfile(old, reference / new)
    names = ["gates/evaluation_readback.json"] + [f"gates/{k}/stage_accepted.json" for k in ("single_cir", "single_ni", "ddp_cir", "ddp_ni")]
    for name in names:
        destination = reference / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(readback / name, destination)
    reused = reuse_verified_gates(output, Path(contract["reuse_verified_gates"]))
    schedule = formal_schedule(output)
    if schedule != {"cir_epochs": 3, "ni_epochs": 3, "cir_optimizer_steps": 1620, "ni_optimizer_steps": 2025}:
        raise ValueError("Unexpected new duration")
    spec = importlib.util.spec_from_file_location("cir3_frozen_exposure", output / "src/new_interest/training/sft_exposure.py")
    exposure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exposure)
    spec = importlib.util.spec_from_file_location("cir3_frozen_cursor", output / "src/new_interest/training/old_new_execution.py")
    cursor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cursor)
    stages = {}
    for stage in stage_specs(output, output / "formal", gates=False):
        name = stage["key"]
        ledger = exposure.TrainingExposure.from_parquet(stage["train_file"], train_sha256=sha256(stage["train_file"]),
            expected_epochs=stage["epochs"], max_samples=None, default_task="new_interest")
        cursor.STAGES.add(f"cir/{name}")
        native = cursor.make_cursor_contract(stage["train_file"], stage=f"cir/{name}", world_size=4,
                                            per_device_batch_size=2, epochs=stage["epochs"], global_batch_size=32)
        if native["total_steps"] != schedule[f"{name}_optimizer_steps"] or ledger.identity["samples"] != stage["samples"]:
            raise ValueError(f"Scope mismatch: {name}")
        stages[name] = {"samples": stage["samples"], "epochs": stage["epochs"], "steps": native["total_steps"],
                        "presentations": stage["samples"] * stage["epochs"], "exposure_identity": ledger.identity,
                        "cursor_contract": native}
    changed = [name for name, digest in original["files"].items() if sha256(output / name) != digest]
    if set(changed) != {runner, "configs/experiments/mind_cir_108_v1.json"}:
        raise ValueError(f"Unexpected runtime changes: {changed}")
    evidence = {"status": "passed", "source_bundle_sha256": acceptance["bundle_sha256"],
                "source_acceptance_sha256": sha256(acceptance_path), "changed_existing_files": changed,
                "all_other_source_payload_files_byte_identical": True,
                "unchanged_runtime_file_count": len(original["files"]) - len(changed),
                "reused_gates": reused, "training_schedule": schedule, "stages": stages,
                "data_hashes": {p.name: sha256(p) for p in (output / "data/training/mind_cir_108_v1").glob("*.parquet")},
                "source_dataset_statistics": "Dataset files remain immutable; training_schedule sets the new epoch horizon"}
    write_json(output / "manifests/duration_preflight.json", evidence)
    files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
             and p.name != ".amltignore" and p != output / "manifests/cir_bundle.json"}
    write_json(output / "manifests/cir_bundle.json", {"version": "cir-duration-bundle-v1", "files": files})
    # Save the explicit new experiment config separately; leave the CIR1 config intact.
    config_copy = project / "configs/experiments/mind_cir_108_cir3_ni3_v1.json"
    if config_copy.exists():
        raise FileExistsError(config_copy)
    write_json(config_copy, contract)
    result = {"bundle": str(output.relative_to(project)), "bundle_sha256": sha256(output / "manifests/cir_bundle.json"),
              "preflight_sha256": sha256(output / "manifests/duration_preflight.json"),
              "training_schedule": schedule, "changed_existing_files": changed,
              "runtime_files_unchanged": len(original["files"]) - len(changed)}
    write_json(output.parent / "local_preflight.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project.resolve(), args.output.resolve()), indent=2))
