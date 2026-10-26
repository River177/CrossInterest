"""Freeze two paired frequency arms using the accepted CIR numerical runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def audit_bundle(bundle):
    # A fresh process imports only the submitted runtime, including the exposure
    # ledger, cursor builder, tokenizer registration, and tokenization function.
    sys.path.insert(0, str(bundle / "src"))
    import transformers
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    from new_interest.training.sft_data import audit_token_lengths
    from new_interest.training.sft_exposure import TrainingExposure
    from new_interest.experiments.cir_warmup import stage_config, stage_specs
    contract = read(bundle / "configs/experiments/mind_cir_108_v1.json")
    model = contract["model"]
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots" / model["model_revision"]
    expected = read(bundle / "manifests/h50_snapshot.json")["files"]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json"):
        if sha(snapshot / name) != expected[name]:
            raise ValueError(f"Frozen tokenizer file differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    data = bundle / "data/training/mind_cir_108_v1"
    register_semantic_interest_tokens(tokenizer, data / "interest_token_mapping.parquet")
    audits, ledgers = {}, {}
    for task, epochs in (("cir", 1), ("ni", 3)):
        audits[task] = audit_token_lengths(data / f"{task}.parquet", tokenizer,
            max_seq_length=2048, enable_thinking=False,
            output_path=bundle / f"manifests/local_token_audit_{task}.json")
        if (audits[task]["over_max_seq_length"] or audits[task]["min_assistant_tokens"] != 3
                or audits[task]["max_assistant_tokens"] != 3):
            raise ValueError(f"Token contract differs: {task}")
        ledgers[task] = TrainingExposure.from_parquet(data / f"{task}.parquet",
            train_sha256=sha(data / f"{task}.parquet"), expected_epochs=epochs,
            max_samples=None, default_task="new_interest").identity
    # Stage configs/cursors are inspection artifacts, outside the frozen payload.
    check_root = bundle.parent / "stage-config-check"
    stages = {}
    for spec in stage_specs(bundle, check_root, gates=False):
        config = stage_config(bundle, check_root, spec)
        cursor = read(config["old_new_cursor_contract_file"])
        expected_steps = (spec["samples"] + 31) // 32 * spec["epochs"]
        if cursor["total_steps"] != expected_steps or ledgers[spec["task"]]["samples"] != spec["samples"]:
            raise ValueError("Stage exposure or cursor differs")
        if spec["parent"] is None and config.get("adapter_path"):
            raise ValueError("CIR unexpectedly warm-started")
        if spec["parent"] and config["adapter_path"] != str(check_root / "cir/final_adapter"):
            raise ValueError("NI handoff points to another arm")
        write(check_root / f"{spec['key']}/launch_config.json", config)
        stages[spec["key"]] = {"samples": spec["samples"], "epochs": spec["epochs"],
            "optimizer_steps": expected_steps, "presentations": spec["samples"] * spec["epochs"],
            "input_adapter": config.get("adapter_path"), "cursor_contract_sha256": sha(config["old_new_cursor_contract_file"])}
    write(bundle / "manifests/local_exposure_audit.json", ledgers)
    result = {"status": "passed", "local_transformers_version": transformers.__version__,
              "worker_transformers_version": contract["worker_packages"]["transformers"],
              "worker_repeats_full_token_audit": True, "token_audits": audits, "stages": stages}
    write(bundle / "manifests/frequency_preflight.json", result)
    return result


def prepare(project, output):
    source = project / "artifacts/mind-cir-20260916/bundle-v3"
    acceptance_path = project / "artifacts/mind-cir-20260916/final_acceptance.json"
    acceptance = read(acceptance_path)
    original = read(source / "manifests/cir_bundle.json")
    paired = project / "data/training/mind_cir_frequency_paired_v1"
    pair_audit = read(paired / "paired_audit.json")
    if output.exists():
        raise FileExistsError(output)
    if acceptance["status"] != "accepted" or sha(source / "manifests/cir_bundle.json") != acceptance["bundle_sha256"]:
        raise ValueError("Source CIR numerical runtime lacks acceptance")
    for name, digest in original["files"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Source CIR payload changed: {name}")
    output.mkdir(parents=True)
    result = {"status": "prepared", "paired_audit_sha256": sha(paired / "paired_audit.json"), "arms": {}}
    for arm in ("singleton", "repeated"):
        bundle = output / arm / "bundle-v1"
        shutil.copytree(source, bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        data = bundle / "data/training/mind_cir_108_v1"
        for p in (paired / arm).iterdir():
            if p.is_file():
                shutil.copyfile(p, data / p.name)
        for name in ("ni.parquet", "dev_full_grouped.parquet", "interest_token_mapping.parquet"):
            if sha(data / name) != sha(source / "data/training/mind_cir_108_v1" / name):
                raise ValueError("Original NI/Dev/mapping changed")
        config_path = bundle / "configs/experiments/mind_cir_108_v1.json"
        contract = read(config_path)
        contract.update(version=f"mind-cir-frequency-{arm}-v1", frequency_arm=arm,
                        method=f"Paired H50 {arm} CIR 1 epoch then original NI 3 epochs",
                        data=read(data / "statistics.json"), data_metadata_sha256=sha(data / "metadata.json"),
                        training_schedule={"cir_epochs": 1, "ni_epochs": 3},
                        paired_users_sha256=pair_audit["paired_users_sha256"],
                        comparison_limit="Primary comparison uses the same paired users and budget; target distribution and deletion length differ. Original random CIR has more samples and is contextual only.")
        write(config_path, contract)
        provenance = {"source_bundle_sha256": acceptance["bundle_sha256"],
                      "source_acceptance_sha256": sha(acceptance_path),
                      "runtime_policy": "All original Python runtime files, worker requirements and shell wrapper byte-identical; new input gates execute before full training",
                      "pair_audit_sha256": sha(paired / "paired_audit.json"),
                      "construction_source_sha256": sha(project / "src/new_interest/data/build_cir_frequency_dataset.py")}
        write(bundle / "manifests/frequency_provenance.json", provenance)
        shutil.copyfile(paired / "paired_audit.json", bundle / "manifests/frequency_paired_audit.json")
        shutil.copyfile(acceptance_path, bundle / "manifests/original_cir1_acceptance.json")
        # Include new construction code as provenance; it is not on the train path.
        for name in ("src/new_interest/data/build_cir_frequency_dataset.py", "scripts/analysis/prepare_cir_frequency_bundles.py"):
            (bundle / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, bundle / name)
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--audit-bundle", str(bundle)], check=True, env=env)
        # Do not let an incidental local source edit modify the accepted Trainer.
        runtime_names = [n for n in original["files"] if n.startswith("src/") or n.startswith("scripts/") or n.startswith("requirements")]
        for name in runtime_names:
            if sha(bundle / name) != original["files"][name]:
                raise ValueError(f"Numerical runtime changed: {name}")
        payload = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob("*"))
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                   and p.name != ".amltignore" and p != bundle / "manifests/cir_bundle.json"}
        write(bundle / "manifests/cir_bundle.json", {"version": "cir-frequency-bundle-v1", "files": payload})
        write(project / f"configs/experiments/mind_cir_frequency_{arm}_v1.json", contract)
        result["arms"][arm] = {"bundle": str(bundle.relative_to(project)),
            "bundle_sha256": sha(bundle / "manifests/cir_bundle.json"),
            "preflight": read(bundle / "manifests/frequency_preflight.json"),
            "byte_identical_runtime_files": len(runtime_names)}
    write(output / "local_preflight.json", result)
    return {"status": result["status"], "arms": {a: {k: v for k, v in r.items() if k != "preflight"} for a, r in result["arms"].items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/mind-cir-frequency-20260917"))
    parser.add_argument("--audit-bundle", type=Path)
    args = parser.parse_args()
    result = audit_bundle(args.audit_bundle.resolve()) if args.audit_bundle else prepare(args.project.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
