"""Freeze the approved history-order run without changing shared training code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import yaml

from new_interest.data.build_history_order_ablation import DATA, SOURCE, TRAIN, DEV, MAPPING, SOURCE_HASHES, sha256


def stage(project, output):
    donor = project / "artifacts/mind-history-length-ablation-20260915/bundle-v2"
    manifest_path = donor / "manifests/history_ablation_bundle.json"
    inventory = json.loads(manifest_path.read_text())["files"]
    readback = project / "artifacts/mind-history-order-20260923/baseline_readback"
    if output.exists():
        raise FileExistsError(output)
    for rel, digest in SOURCE_HASHES.items():
        if sha256(project / SOURCE / rel) != digest:
            raise ValueError(f"Original data changed: {rel}")
    output.mkdir(parents=True)

    def copy(src, rel):
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    def write(rel, data):
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2) + "\n")

    inherited = {}
    for rel, digest in inventory.items():
        if rel.startswith(("src/", "migration_kernel/", "requirements-")):
            if sha256(donor / rel) != digest:
                raise ValueError(f"Frozen runtime changed: {rel}")
            copy(donor / rel, rel)
            inherited[rel] = digest
    additions = ["src/new_interest/data/build_history_order_ablation.py",
                 "src/new_interest/experiments/history_order_ablation.py",
                 "src/new_interest/experiments/history_order_analysis.py",
                 "scripts/amlt/sft/run_history_order_ablation.sh",
                 "tests/test_history_order_ablation.py", "tests/test_history_order_analysis.py",
                 "tests/test_history_order_worker.py"]
    for rel in additions:
        copy(project / rel, rel)
    for rel in (TRAIN, DEV, MAPPING):
        copy(project / SOURCE / rel, SOURCE / rel)
    for p in sorted((project / DATA).rglob("*")):
        if p.is_file():
            copy(p, DATA / p.relative_to(project / DATA))
    for name in ("h50_baseline.json", "h50_acceptance.json", "h50_snapshot.json",
                 "h50_single_gpu_smoke_acceptance.json", "h50_four_gpu_smoke_acceptance.json"):
        copy(donor / "manifests" / name, "manifests/" + name)
    copy(readback / "readback_receipt.json", "manifests/baseline_remote_readback.json")
    copy(readback / "artifact_manifest.json", "manifests/baseline_artifact_manifest.json")
    baseline = json.loads((donor / "manifests/h50_baseline.json").read_text())
    old = json.loads((donor / "configs/experiments/history_h30.json").read_text())
    contract = {k: old[k] for k in ("models", "recipe", "evaluation", "worker_packages", "h50_initial_adapter_sha256",
                "h50_final_sample_prefix_sha256", "baseline_acceptance_sha256", "execution_schedule")}
    contract.update(version="mind-ni-history-order-20260923-v1", approved_on="2026-09-23",
                    baseline_id=baseline["baseline_id"], baseline_results_prefix="projects/sft4safety/amlt-results/7210843083.71371-a21f4d1c-d098",
                    original_data=str(SOURCE), shuffled_data=str(DATA), max_concurrent_gpus=4, gpus_per_model=4,
                    new_training_runs=1, new_full_dev_evaluations=["B", "C", "D"], full_optimizer_steps=2025,
                    train_rows=21600, dev_impressions=31392, full_selection="final_epoch_3",
                    prompt_policy="unchanged including in chronological order", shuffle_rule="ni-history-order-v1",
                    inherited_smokes="Previously accepted unchanged model/trainer/DDP/evaluator; no new smoke training authorized",
                    input_files=[{"path": str(folder / rel), "sha256": sha256(project / folder / rel)}
                                 for folder in (SOURCE, DATA) for rel in (TRAIN, DEV, MAPPING)])
    write("configs/experiments/history_order.json", contract)
    sft = json.loads((donor / "configs/sft/history_h30.json").read_text())
    sft.update(train_file=str(DATA / TRAIN), semantic_id_token_mapping_file=str(DATA / MAPPING),
               output_dir="artifacts/mind-history-order-20260923/remote/sft",
               run_name="qwen35-2b-ni-history-shuffled-seed42-0923-v1")
    write("configs/sft/history_order.json", sft)
    original_eval = json.loads((donor / "configs/id_token/history_h30.json").read_text())
    for arm in ("B", "C", "D"):
        evaluation = {**original_eval, "eval_file": str((SOURCE if arm == "C" else DATA) / DEV),
                      "interest_token_mapping_file": str(DATA / MAPPING),
                      "adapter_path": "WORKER_MUST_BIND_APPROVED_ADAPTER",
                      "output_dir": f"artifacts/mind-history-order-20260923/remote/{arm}",
                      "run_name": f"qwen35-2b-ni-history-order-{arm}-0923-v1"}
        write(f"configs/id_token/history_order_{arm}.json", evaluation)
    write("manifests/runtime_reuse.json", {"source_manifest_sha256": sha256(manifest_path),
          "unchanged_inherited_files": inherited, "new_files": additions,
          "scope": "Frozen NI-only numerical implementation; add fixed permutations and approved three-cell orchestration"})
    payload = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write("manifests/history_order_bundle.json", {"files": payload})
    config = {"description": "Approved NI-only fixed history-order ablation: reused O/O, evaluate O/S, one S train then S/O and S/S; original prompt and 2025-step runtime",
              "environment": {"image": "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04", "setup": ["python --version", "nvidia-smi"]},
              "target": {"service": "singularity", "name": "Recall", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"},
              "code": {"local_dir": "$CONFIG_DIR/../../" + str(output.relative_to(project)), "ignore": ["__pycache__/", "*.pyc", ".pytest_cache/"]},
              "jobs": [{"name": "qwen35-2b-ni-history-order", "sku": "80G4-A100", "sla_tier": "Standard",
                        "submit_args": {"env": {"AMLT_DOCKERFILE_TEMPLATE": "DEFAULT",
                         "_AZUREML_SINGULARITY_JOB_UAI": "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"}},
                        "command": ["set -euo pipefail", "cd $$AMLT_CODE_DIR", "bash scripts/amlt/sft/run_history_order_ablation.sh"]}]}
    yaml_path = project / "configs/amlt/mind_history_order_0923_v1.yaml"
    if yaml_path.exists():
        raise FileExistsError(yaml_path)
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return {"bundle": str(output), "manifest_sha256": sha256(output / "manifests/history_order_bundle.json"),
            "payload_files": len(payload), "yaml": str(yaml_path), "inherited_numerical_files": len(inherited)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/mind-history-order-20260923/bundle-v1"))
    a = p.parse_args()
    print(json.dumps(stage(Path.cwd(), a.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
