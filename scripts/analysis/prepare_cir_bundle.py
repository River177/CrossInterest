"""Freeze the CIR arm using accepted OI/NI runtime bytes and compact H50 evidence."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

from new_interest.data.build_cir_dataset import sha256, write_json


def bundle_payload(output: Path) -> dict[str, str]:
    """Manifest worker payload, excluding AMLT's local-only upload control file."""
    return {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            and p.name != ".amltignore" and p != output / "manifests/cir_bundle.json"}


def prepare(project: Path, runtime: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    source_contract = json.loads((runtime / "configs/experiments/mind_old_new_interest_v1.json").read_text())
    data = project / "data/training/mind_cir_108_v1"
    for item in source_contract["new_interest"]["input_files"][:3]:
        name = "ni.parquet" if "21600.parquet" in item["path"] else ("dev_full_grouped.parquet" if "dev_full" in item["path"] else "interest_token_mapping.parquet")
        if sha256(data / name) != item["sha256"]:
            raise ValueError(f"CIR changed the accepted NI input: {name}")
    output.mkdir(parents=True)
    copied = {}
    for source in sorted((runtime / "src").rglob("*.py")):
        relative = source.relative_to(runtime)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[str(relative)] = sha256(source)
    for name in ("src/new_interest/data/build_cir_dataset.py", "src/new_interest/experiments/cir_warmup.py",
                 "src/new_interest/training/sft_exposure.py"):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
    shutil.copytree(data, output / "data/training/mind_cir_108_v1")
    shutil.copyfile(runtime / "requirements-old-new-qwen35-worker.txt", output / "requirements-cir-worker.txt")
    evidence = project / "artifacts/mind-history-length-ablation-20260915/bundle-v2/manifests"
    (output / "manifests").mkdir()
    for name in ("h50_baseline.json", "h50_snapshot.json", "h50_acceptance.json",
                 "h50_single_gpu_smoke_acceptance.json", "h50_four_gpu_smoke_acceptance.json"):
        shutil.copyfile(evidence / name, output / "manifests" / name)
    baseline = json.loads((output / "manifests/h50_baseline.json").read_text())
    contract = {"version": "mind-cir-108-v1", "model": source_contract["model"],
                "models": {"2b": source_contract["model"]}, "recipe": source_contract["recipe"],
                "worker_packages": source_contract["worker_packages"],
                "data_metadata_sha256": sha256(data / "metadata.json"),
                "method": "CIR 1 epoch then original NI 3 epochs; inherit LoRA/SID; reset optimizer/scheduler",
                "kernel_profile": "torch-fallback-explicit",
                "execution": {"world_size": 4, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "global_batch_size": 32},
                "comparison_limit": "Existing NI-only used a historical single-to-four-GPU/FLA schedule; compare observed utility, not isolate a causal mechanism",
                "existing_baseline": baseline["baseline_id"], "submit_baseline": False,
                "data": json.loads((data / "statistics.json").read_text())}
    config = json.loads((runtime / "configs/sft/experiments/mind_old_new_interest_new_v1.json").read_text())
    config.pop("adapter_path", None)
    config.update(semantic_id_token_mapping_file="data/training/mind_cir_108_v1/interest_token_mapping.parquet",
                  gradient_checkpointing_enabled=True, expected_initial_adapter_state_sha256=baseline["initial_adapter_state_sha256"],
                  exposure_default_task="new_interest", audit_training_exposure=True, record_initial_adapter_state=True)
    evaluation = json.loads((runtime / "configs/id_token/mind_old_new_interest_two_stage_v1.json").read_text())
    evaluation.update(eval_file="data/training/mind_cir_108_v1/dev_full_grouped.parquet",
                      interest_token_mapping_file="data/training/mind_cir_108_v1/interest_token_mapping.parquet", batch_size=2,
                      run_name="mind-cir-108-v1-dev")
    for name, content in (("configs/experiments/mind_cir_108_v1.json", contract),
                          ("configs/sft/mind_cir_base_v1.json", config), ("configs/id_token/mind_cir_v1.json", evaluation)):
        write_json(project / name, content)
        write_json(output / name, content)
    write_json(output / "manifests/runtime_reuse.json", {
        "source_bundle": str(runtime), "copied_files": copied,
        "workflow_overrides": {"src/new_interest/training/sft_exposure.py": sha256(output / "src/new_interest/training/sft_exposure.py")},
        "source_contract_sha256": sha256(runtime / "configs/experiments/mind_old_new_interest_v1.json"),
        "new_scope": "CIR construction and stage orchestration; original numerical and acceptance code retained"})
    wrapper = Path("scripts/amlt/sft/run_cir_warmup.sh")
    (output / wrapper).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(project / wrapper, output / wrapper)
    # Exercise the exact bundled telemetry constructor before allocating a GPU.
    spec = importlib.util.spec_from_file_location("cir_bundle_exposure", output / "src/new_interest/training/sft_exposure.py")
    exposure_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exposure_module)
    exposure_audits = {}
    for task, epochs in (("cir", 1), ("ni", 3)):
        source = data / f"{task}.parquet"
        ledger = exposure_module.TrainingExposure.from_parquet(source, train_sha256=sha256(source),
            expected_epochs=epochs, max_samples=None, default_task="new_interest")
        exposure_audits[task] = ledger.identity
    write_json(output / "manifests/local_exposure_audit.json", exposure_audits)
    # Full local token audit uses only tokenizer files, never downloads model weights.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    from new_interest.training.sft_data import audit_token_lengths
    model = contract["model"]
    snapshot = Path(snapshot_download(model["model_name"], revision=model["model_revision"],
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "config.json"]))
    expected = json.loads((output / "manifests/h50_snapshot.json").read_text())["files"]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json"):
        if sha256(snapshot / name) != expected[name]:
            raise ValueError(f"Tokenizer identity mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    register_semantic_interest_tokens(tokenizer, data / "interest_token_mapping.parquet")
    audits = {}
    for task in ("cir", "ni"):
        audits[task] = audit_token_lengths(data / f"{task}.parquet", tokenizer, max_seq_length=2048,
            enable_thinking=False, output_path=output / "manifests" / f"local_token_audit_{task}.json")
        if audits[task]["over_max_seq_length"] or audits[task]["min_assistant_tokens"] != 3 or audits[task]["max_assistant_tokens"] != 3:
            raise ValueError(f"CIR token audit failed: {task}")
    (output / ".amltignore").write_text("__pycache__/\n*.pyc\n/outputs/\n")
    write_json(output / "manifests/cir_bundle.json", {"version": "cir-bundle-v2", "files": bundle_payload(output)})
    result = {"bundle": str(output), "bundle_sha256": sha256(output / "manifests/cir_bundle.json"),
              "cir_samples": contract["data"]["cir_rows"], "cir_steps": contract["data"]["cir_optimizer_steps"],
              "ni_steps": 2025, "token_audits": audits}
    write_json(project / "artifacts/mind-cir-20260916/local_preflight.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.project.resolve(), a.runtime.resolve(), a.output.resolve())
