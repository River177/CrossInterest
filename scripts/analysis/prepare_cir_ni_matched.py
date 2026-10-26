"""Freeze two matched CIR/NI arms using byte-identical accepted SFT runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import yaml

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.data.build_cir_ni_matched import ARMS, VERSION
from new_interest.experiments import cir_warmup as core


def prepare(project: Path, output: Path) -> dict:
    source = project / "artifacts/mind-cir-four-20260918/bundle-v1"
    acceptance_path = project / "artifacts/mind-cir-four-20260918/acceptance/cir5_ni3.json"
    accepted = core.read(acceptance_path)
    old = core.read(source / "manifests/cir_bundle.json")
    core.require(accepted["status"] == "accepted" and accepted["bundle_sha256"] == sha256(source / "manifests/cir_bundle.json"), "Accepted runtime identity differs")
    for rel, digest in old["files"].items():
        core.require(sha256(source / rel) == digest, "Accepted source changed: " + rel)
    core.require(sha256(source / "src/new_interest/experiments/cir_warmup.py") == sha256(project / "src/new_interest/experiments/cir_warmup.py"), "Local orchestrator differs")
    dataset = project / "data/training/mind_cir_ni_matched_20260928_v1"
    data_manifest = core.read(dataset / "metadata.json")
    for rel, digest in data_manifest["files"].items():
        core.require(sha256(dataset / rel) == digest, "Paired dataset changed: " + rel)
    token_audit = core.read(output / "token_audit.json")
    core.require(abs(token_audit["aux_total_token_ratio_cir_to_ni"] - 1) < .03, "Auxiliary total token mismatch exceeds prespecified 3% review threshold")
    configs, reports = {}, {}
    template = yaml.safe_load((project / "configs/amlt/mind_cir_ni_prompt_0928_v1.yaml").read_text())
    modules = {}
    for name in ("sft_exposure", "old_new_execution"):
        spec = importlib.util.spec_from_file_location("matched_frozen_" + name, source / f"src/new_interest/training/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    for arm in ARMS:
        bundle = output / "bundles-v1" / arm
        if bundle.exists():
            raise FileExistsError(bundle)
        shutil.copytree(source, bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in (dataset / arm).iterdir():
            shutil.copyfile(path, bundle / core.DATA / path.name)
        contract = core.read(source / core.CONTRACT)
        for key in ("reuse_verified_gates", "existing_cir1_reference", "existing_cir3_reference", "conditions", "mixed_data"):
            contract.pop(key, None)
        contract.update(version=VERSION + "-" + arm, method=arm + " auxiliary5 -> unchanged original NI3",
                        training_schedule={"cir_epochs": 5, "ni_epochs": 3},
                        data=core.read(dataset / arm / "statistics.json"),
                        data_metadata_sha256=sha256(dataset / arm / "metadata.json"),
                        paired_control={"arm": arm, "rows": 9431, "pair_manifest_sha256": sha256(dataset / "pair_manifest.parquet"),
                            "legacy_auxiliary_stage_key": "cir", "auxiliary_training_task": "new_interest" if arm == "ni_prefix" else "cross_interest_reconstruction",
                            "exact_sample_labels_order_and_behavior_lengths_match": True,
                            "token_audit_sha256": sha256(output / "token_audit.json"),
                            "token_count_ratio_cir_to_ni": token_audit["aux_total_token_ratio_cir_to_ni"]},
                        comparison_limit="Matched two-arm seed42 comparison; behavior lengths and exposures matched, token counts audited but not claimed bitwise equal compute. Old full CIR is background only.")
        write_json(bundle / core.CONTRACT, contract)
        refs = bundle / "manifests/paired_reference"
        refs.mkdir()
        shutil.copyfile(acceptance_path, refs / "accepted_runtime.json")
        shutil.copyfile(source / "manifests/cir_bundle.json", refs / "source_bundle.json")
        shutil.copyfile(output / "token_audit.json", refs / "token_audit.json")
        for rel in ("src/new_interest/data/build_cir_ni_matched.py", "scripts/analysis/prepare_cir_ni_matched.py"):
            target = bundle / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / rel, target)
        recovery = project / "artifacts/mind-cir-four-20260918/bundle-staged-recovery-0919-v1/recovery"
        overrides = bundle / "runtime_overrides"
        overrides.mkdir()
        for name in ("sitecustomize.py", "cir_storage_retry.py"):
            shutil.copyfile(recovery / name, overrides / name)
        wrapper = (source / "scripts/amlt/sft/run_cir_warmup.sh").read_text().replace(
            "python -m new_interest.experiments.cir_warmup worker",
            'export CIR_STORAGE_RETRY_ENABLED=1\nexport PYTHONPATH="$AMLT_CODE_DIR/runtime_overrides:$PYTHONPATH"\n'
            "python -m new_interest.experiments.cir_warmup worker")
        wrapper_rel = Path("scripts/amlt/sft/run_cir_ni_matched.sh")
        (bundle / wrapper_rel).write_text(wrapper)
        local_wrapper = project / wrapper_rel
        if local_wrapper.exists():
            core.require(local_wrapper.read_text() == wrapper, "Shared matched wrapper changed")
        else:
            local_wrapper.write_text(wrapper)
        stages = {}
        for stage in core.stage_specs(bundle, bundle / "formal", gates=False):
            ledger = modules["sft_exposure"].TrainingExposure.from_parquet(stage["train_file"],
                train_sha256=sha256(stage["train_file"]), expected_epochs=stage["epochs"], max_samples=None,
                default_task="new_interest")
            cursor = modules["old_new_execution"]
            cursor.STAGES.add("cir/" + stage["key"])
            checked = cursor.make_cursor_contract(stage["train_file"], stage="cir/" + stage["key"], world_size=4,
                per_device_batch_size=2, epochs=stage["epochs"], global_batch_size=32)
            core.require(checked["total_steps"] == (1475 if stage["task"] == "cir" else 2025), "Training budget differs")
            stages[stage["key"]] = {"samples": stage["samples"], "epochs": stage["epochs"],
                "steps": checked["total_steps"], "presentations": stage["samples"] * stage["epochs"],
                "parent": stage["parent"], "cursor": checked, "exposure_identity": ledger.identity}
        changed = [rel for rel, digest in old["files"].items() if sha256(bundle / rel) != digest]
        expected = {str(core.CONTRACT), *(str(core.DATA / n) for n in ("cir.parquet", "statistics.json", "metadata.json", "selection_manifest.parquet"))}
        core.require(set(changed) == expected, "Unexpected changes to accepted runtime: " + repr(changed))
        report = {"status": "passed", "arm": arm, "stages": stages,
            "accepted_source_bundle_sha256": accepted["bundle_sha256"], "changed_prior_payload_files": changed,
            "unchanged_prior_payload_files": len(old["files"])-len(changed),
            "gates": "Fresh single-GPU and four-GPU auxiliary/NI; exact reloads and 64-record real evaluator. Formal training starts cold.",
            "physical_jobs": 1, "gpus": 4, "large_artifacts": "remote_only"}
        report["builder_sha256"] = sha256(project / "src/new_interest/data/build_cir_ni_matched.py")
        write_json(bundle / "manifests/matched_preflight.json", report)
        payload = {str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob("*"))
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                   and p.name != ".amltignore" and p != bundle / "manifests/cir_bundle.json"}
        write_json(bundle / "manifests/cir_bundle.json", {"version": VERSION, "files": payload})
        canonical = project / f"configs/experiments/mind_cir_ni_matched_{arm}_0928_v1.json"
        if canonical.exists():
            raise FileExistsError(canonical)
        write_json(canonical, contract)
        config = json.loads(json.dumps(template))
        config["description"] = f"Matched CIR versus added NI: {arm}, 9431 paired users/targets, auxiliary5 then NI3."
        config["code"]["local_dir"] = "$CONFIG_DIR/../../" + str(bundle.relative_to(project))
        job = config["jobs"][0]
        job["name"] = "qwen35-2b-" + arm.replace("_", "-") + "5-ni3"
        job["command"][-1] = "bash " + str(wrapper_rel)
        config_path = project / f"configs/amlt/mind_cir_ni_matched_{arm}_0928_v1.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        reports[arm] = {**report, "bundle": str(bundle.relative_to(project)),
                        "bundle_sha256": sha256(bundle / "manifests/cir_bundle.json"),
                        "config": str(config_path.relative_to(project)), "job": job["name"]}
    write_json(output / "local_preflight.json", {"status": "passed", "version": VERSION,
               "physical_jobs": 2, "maximum_concurrent_gpus": 8, "arms": reports})
    return {arm: {k: r[k] for k in ("bundle", "bundle_sha256", "config", "job")} for arm, r in reports.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project.resolve(), args.output.resolve()), indent=2))
