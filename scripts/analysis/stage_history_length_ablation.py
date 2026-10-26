"""Freeze three input-only ablations against the accepted NI-only H50 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import yaml

from new_interest.data.build_history_length_ablation import sha256, TRAIN, DEV, MAPPING


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path.cwd()
    base = args.baseline_project.resolve()
    baseline = json.loads((base / "manifests/mind_shared_2b_ni_only_final_baseline_v1.json").read_text())
    receipt_path = base / baseline["acceptance_receipt"]
    assert baseline["status"] == "independently_accepted"
    assert sha256(receipt_path) == baseline["acceptance_receipt_sha256"]
    accepted = json.loads(receipt_path.read_text())
    donor = base / baseline["full_bundle"]
    donor_manifest = json.loads((donor / "manifests/mind_four_gpu_runtime_sources_v1.json").read_text())
    contract = json.loads((donor / "configs/experiments/mind_backbone_qwen35_21600_v1.json").read_text())
    source_data = Path(contract["input_files"][0]["path"]).parents[2]
    changed_data = Path("data/training/mind_history_length_ablation_20260915_v1")
    metadata = json.loads((project / changed_data / "metadata.json").read_text())
    for record in contract["input_files"][:3]:
        assert sha256(project / record["path"]) == record["sha256"]
        assert metadata["source_files"][str(Path(record["path"]).relative_to(source_data))] == record["sha256"]

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def copy(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def write(relative, value):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + "\n")

    inherited = {}
    for relative, expected in donor_manifest["files"].items():
        if relative.startswith(("src/", "migration_kernel/")) or relative.startswith("requirements-"):
            assert sha256(donor / relative) == expected, relative
            copy(donor / relative, relative)
            inherited[relative] = expected
    additions = ["src/new_interest/data/__init__.py", "src/new_interest/data/build_history_length_ablation.py",
                 "src/new_interest/infrastructure/__init__.py", "src/new_interest/infrastructure/paths.py",
                 "src/new_interest/experiments/history_length_ablation.py",
                 "scripts/amlt/sft/run_history_length_ablation.sh"]
    for relative in additions:
        copy(project / relative, relative)
    for relative in (TRAIN, DEV, MAPPING):
        copy(project / source_data / relative, source_data / relative)
    for path in (project / changed_data).rglob("*"):
        if path.is_file():
            copy(path, changed_data / path.relative_to(project / changed_data))
    write("manifests/h50_baseline.json", baseline)
    copy(receipt_path, "manifests/h50_acceptance.json")
    copy(base / baseline["full_readback_root"] / "snapshot_audit.json", "manifests/h50_snapshot.json")
    copy(base / "artifacts/mind-backbone-qwen35-21600-v1/acceptance/smoke-2b-v2.json", "manifests/h50_single_gpu_smoke_acceptance.json")
    copy(base / "artifacts/mind-qwen35-four-gpu-migration-v1/acceptance/probe-2b-v1.json", "manifests/h50_four_gpu_smoke_acceptance.json")
    # Reuse already-present predictions only for a compact subgroup statistic;
    # never download the H50 adapter or predictions to this Mac.
    prediction_path = base / baseline["predictions_path"]
    cohort = json.loads((project / changed_data / "dev_h50_cohort.json").read_text())
    if prediction_path.is_file():
        import pyarrow.parquet as pq
        from new_interest.experiments.history_length_ablation import subset_metrics
        assert sha256(prediction_path) == baseline["predictions_sha256"]
        stats = subset_metrics(pq.read_table(project / source_data / DEV).to_pylist(),
                               pq.read_table(prediction_path).to_pylist(), set(cohort["impression_ids"]))
        write("manifests/h50_cohort_metrics.json", {**stats, "prediction_sha256": baseline["predictions_sha256"],
                                                 "source": "existing previously accepted local copy"})
    for h in (30, 10, 5):
        data = changed_data / f"h{h}"
        arm = {**contract, "ablation_version": "mind-h50-controlled-length-ablation-20260915-v1",
               "task": "History length ablation", "approved_on": "2026-09-15",
               "models": {"2b": contract["models"]["2b"]}, "max_concurrent_gpus": 12, "gpus_per_model": 4,
               "sku": "80G4-A100", "sla_tier": "Standard",
               "input_files": [{"path": str(data / name), "sha256": sha256(project / data / name),
                                "bytes": (project / data / name).stat().st_size} for name in (TRAIN, DEV, MAPPING)],
               "h50_data": str(source_data), "ablation_data": str(data), "history_limit": h,
               "h50_initial_adapter_sha256": baseline["initial_adapter_state_sha256"],
               "h50_final_sample_prefix_sha256": accepted["training"]["sample_cursor"]["consumed_prefix_sha256"],
               "baseline_acceptance_sha256": baseline["acceptance_receipt_sha256"],
               "execution_schedule": [{"updates": [1, 500], "gpus": 1, "batch": 1, "accumulation": 32, "kernel": "torch-fallback"},
                                      {"updates": [501, 2025], "gpus": 4, "batch": 8, "accumulation": 1, "kernel": "fla-0.5.2-causal-conv1d-1.7.0"}],
               "sample_and_target_selection": "reuse frozen H50 rows and labels exactly"}
        write(f"configs/experiments/history_h{h}.json", arm)
        sft = json.loads((donor / "configs/sft/experiments/mind_backbone_qwen35_2b_21600_v1.json").read_text())
        sft.update(train_file=str(data / TRAIN), semantic_id_token_mapping_file=str(data / MAPPING),
                   run_name=f"qwen35-2b-ni-only-h{h}-21600-seed42-0915-v1",
                   expected_initial_adapter_state_sha256=baseline["initial_adapter_state_sha256"])
        write(f"configs/sft/history_h{h}.json", sft)
        evaluation = json.loads((donor / "configs/id_token/mind_backbone_qwen35_2b_21600_v1.json").read_text())
        evaluation.update(eval_file=str(data / DEV), interest_token_mapping_file=str(data / MAPPING),
                          run_name=f"qwen35-2b-ni-only-h{h}-full-dev-seed42-0915-v1")
        write(f"configs/id_token/history_h{h}.json", evaluation)

    write("manifests/baseline_runtime_reuse.json", {"source_manifest_sha256": sha256(donor / "manifests/mind_four_gpu_runtime_sources_v1.json"),
                                                 "unchanged_inherited_files": inherited,
                                                 "new_files": additions,
                                                 "scope": "Reuse verified training/model/evaluation implementation; add truncation and scheduled orchestration"})
    inventory = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write("manifests/history_ablation_bundle.json", {"files": inventory})
    yaml_file = project / "configs/amlt/mind_history_length_ablation_0915_v1.yaml"
    config = {"description": "H30/H10/H5 input-only NI-only ablation; reuse accepted H50; same samples/labels/seed/2025 steps and historical execution schedule; automatic Full Dev and remote acceptance",
              "environment": {"image": "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04", "setup": ["python --version", "nvidia-smi"]},
              "target": {"service": "singularity", "name": "Recall", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"},
              "code": {"local_dir": "$CONFIG_DIR/../../" + str(out.relative_to(project)), "ignore": ["__pycache__/", "*.pyc"]},
              "jobs": []}
    for h in (30, 10, 5):
        config["jobs"].append({"name": f"qwen35-2b-ni-only-h{h}", "sku": "80G4-A100", "sla_tier": "Standard",
             "submit_args": {"env": {"AMLT_DOCKERFILE_TEMPLATE": "DEFAULT",
             "_AZUREML_SINGULARITY_JOB_UAI": "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"}},
             "command": ["set -euo pipefail", "cd $$AMLT_CODE_DIR", f"bash scripts/amlt/sft/run_history_length_ablation.sh {h}"]})
    yaml_file.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"bundle": str(out), "manifest_sha256": sha256(out / "manifests/history_ablation_bundle.json"),
                      "files": len(inventory), "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()), "yaml": str(yaml_file)}, indent=2))


if __name__ == "__main__":
    main()
