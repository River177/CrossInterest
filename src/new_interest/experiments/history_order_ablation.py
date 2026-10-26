"""Approved three-cell history-order ablation on the frozen NI-only runtime."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq

from new_interest.data.build_history_order_ablation import audit_pair, sha256, TRAIN, DEV, MAPPING
from new_interest.experiments.history_length_ablation import read, write, require, single_train

CONTRACT = Path("configs/experiments/history_order.json")


def bind(project, arm="D"):
    require(arm in {"A", "B", "C", "D"}, "Unknown evaluation arm")
    from new_interest.experiments import backbone_scaling as baseline
    contract = read(project / CONTRACT)
    baseline.CONTRACT = CONTRACT
    baseline.DATA = Path(contract["shuffled_data"])
    baseline.TRAIN = baseline.DATA / TRAIN
    baseline.MAPPING = baseline.DATA / MAPPING
    baseline.CANDIDATE_MAPPING = baseline.MAPPING
    baseline.DEV = Path(contract["original_data"] if arm in {"A", "C"} else contract["shuffled_data"]) / DEV
    return baseline, contract


def verify_bundle(project):
    manifest = read(project / "manifests/history_order_bundle.json")
    for relative, expected in manifest["files"].items():
        require(sha256(project / relative) == expected, f"Frozen bundle changed: {relative}")
    return sha256(project / "manifests/history_order_bundle.json")


def stage_config(project, root, stage):
    baseline, _ = bind(project)
    config = read(project / "configs/sft/history_order.json")
    config.update(train_file=str(project / baseline.TRAIN),
                  semantic_id_token_mapping_file=str(project / baseline.MAPPING),
                  output_dir=str(root / stage / "sft"),
                  checkpoint_dir=f"/scratch/history-order-{root.name}-{stage}",
                  persistent_checkpoint_dir=str(root / stage / "checkpoints"),
                  resume_from_checkpoint="auto")
    if stage == "single":
        config.update(per_device_train_batch_size=1, gradient_accumulation_steps=32)
    elif stage == "final":
        config.update(per_device_train_batch_size=8, gradient_accumulation_steps=1,
                      gradient_checkpointing_enabled=True,
                      execution_migration={"source_checkpoint": str(root / "single/checkpoints/checkpoint-500"),
                          "source_stage_artifacts": str(root / "single/source-stage"),
                          "cursor_contract_file": str(root / "single/cursor_contract.json")})
    else:
        raise ValueError("Unknown training stage")
    return config


def run_factory(project, root):
    from new_interest.experiments import backbone_four_gpu as execution
    def run(command, log, devices=None):
        env = dict(os.environ)
        if devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = devices
        local = Path("/scratch") / f"history-order-{root.name}-logs" / log
        try:
            require(execution.run(command, project, local, env=env) == 0, f"Worker command failed: {log}")
        finally:
            if local.exists():
                dest = root / "logs" / log
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, dest)
    return run


def install_fast(project, root, run):
    run([sys.executable, str(project / "migration_kernel/install_fast_kernels.py"),
         "--manifest", str(project / "migration_kernel/install_manifest.json"),
         "--wheel-cache", "/scratch/history-order-fast-wheels", "--output", str(root / "fast_kernel_environment.json")],
        "install-fast.log", "0")


def audit_baseline(project, root):
    baseline, contract = bind(project, "A")
    source = Path("/mnt/default") / contract["baseline_results_prefix"]
    frozen = read(project / "manifests/baseline_remote_readback.json")
    for name, item in frozen["files"].items():
        require(sha256(source / name) == item["sha256"], f"Baseline compact evidence changed: {name}")
    manifest = read(source / "artifact_manifest.json")
    selected = {name: item for name, item in manifest["files"].items()
                if name.startswith(("sft/final_adapter/", "real_eval/"))
                or name in {"sft/run_summary.json", "sft/semantic_reload_reference.json", "sft/semantic_adapter_reload_audit.json"}}
    for name, item in selected.items():
        fp = source / name
        require(fp.stat().st_size == item["bytes"] and sha256(fp) == item["sha256"], f"Baseline payload changed: {name}")
    accepted = read(project / "manifests/h50_baseline.json")
    require(sha256(source / "sft/final_adapter/adapter_model.safetensors") == accepted["adapter_sha256"], "Wrong NI-only model")
    require(sha256(source / "real_eval/merged/id_token_predictions.parquet") == accepted["predictions_sha256"], "Wrong baseline predictions")
    metrics = baseline.audit_evaluation(project, "2b", source, "full")
    require(metrics["metrics_sha256"] == accepted["metrics_sha256"], "Wrong baseline metrics")
    receipt = {"source": str(source), "verified_files": selected, "evaluation": metrics,
               "adapter": baseline.audit_adapter(project, "2b", source / "sft"),
               "historical_reload": baseline.audit_reload(project, "2b", source / "sft")}
    write(root / "baseline_readback.json", receipt)
    print("HISTORY_ORDER_BASELINE_VERIFIED", json.dumps(metrics), flush=True)
    return source


def verify_reload(project, adapter, reference, output, run, log):
    _, contract = bind(project)
    run([sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
         "--model-name", contract["models"]["2b"]["model_name"],
         "--model-revision", contract["models"]["2b"]["model_revision"],
         "--adapter-path", str(adapter), "--mapping-file", str(project / Path(contract["original_data"]) / MAPPING),
         "--reference-file", str(reference), "--output", str(output), "--atol", "0", "--rtol", "0"], log, "0")
    result = read(output)
    require(result["all_close"] and result["adapter_state_integrity"]["all_tensors_exact"], "Fresh reload failed")


def evaluate(project, root, arm, adapter, run):
    baseline, contract = bind(project, arm)
    arm_root = root / "evaluations" / arm
    identity = {"arm": arm, "adapter_file_sha256": sha256(adapter / "adapter_model.safetensors"),
                "eval_sha256": sha256(project / baseline.DEV), "bundle_sha256": sha256(project / "manifests/history_order_bundle.json")}
    identity_file = arm_root / "identity.json"
    if identity_file.exists():
        require(read(identity_file) == identity, "Evaluation arm identity changed")
    else:
        write(identity_file, identity)
    if not (arm_root / "accepted.json").exists():
        def shard(s):
            run([sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking",
                 "--config", str(project / f"configs/id_token/history_order_{arm}.json"),
                 "--adapter-path", str(adapter), "--output-dir", str(arm_root / f"real_eval/shard-{s:02d}"),
                 "--num-shards", "4", "--shard-index", str(s), "--batch-size", "8"], f"eval-{arm}-{s}.log", str(s))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(shard, range(4)))
        from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
        merge_id_token_shards([arm_root / f"real_eval/shard-{i:02d}" for i in range(4)], project / baseline.DEV,
                             project / baseline.MAPPING, arm_root / "real_eval/merged", expected_records=31392)
    receipt = baseline.audit_evaluation(project, "2b", arm_root, "full")
    require(receipt["target_occurrences"] == 37728, "Target occurrence count differs")
    write(arm_root / "accepted.json", {**identity, "evaluation": receipt})
    print("HISTORY_ORDER_EVALUATION_ACCEPTED", arm, json.dumps(receipt), flush=True)
    return receipt


def train(project, root, run):
    from new_interest.training.execution_migration import checkpoint_state_audit, validate_execution_transition
    from new_interest.training.global_sample_cursor import build_hf_cursor_contract
    baseline, contract = bind(project)
    single_config = stage_config(project, root, "single")
    write(root / "single/training_config.json", single_config)
    if not (root / "single/accepted.json").is_file():
        run([sys.executable, "-m", "new_interest.experiments.history_order_ablation", "single", "--config", str(root / "single/training_config.json")], "train-single.log", "0")
    source = root / "single/checkpoints/checkpoint-500"
    source_marker = read(source / "checkpoint_complete.json")
    state = checkpoint_state_audit(source, source_marker["identity"])
    require(state["global_step"] == 500 and state["max_steps"] == 2025, "Wrong first-stage checkpoint")
    if (root / "single/accepted.json").is_file():
        require(read(root / "single/accepted.json")["checkpoint_marker_sha256"] == sha256(source / "checkpoint_complete.json"),
                "Committed first-stage checkpoint changed")
    cursor = build_hf_cursor_contract(sha256(project / baseline.TRAIN), 21600, 42, 3, read(source / "trainer_state.json"))
    write(root / "single/cursor_contract.json", cursor)
    require(read(root / "single/sft/initial_adapter_state.json")["state_sha256"] == contract["h50_initial_adapter_sha256"], "Initialization differs from H50")
    stage = root / "single/source-stage"
    stage.mkdir(exist_ok=True)
    for name in ("initial_adapter_state.json", "training_start_audit.json", "training_resume_audits.jsonl"):
        original_evidence = root / "single/sft" / name
        if original_evidence.exists():
            shutil.copyfile(original_evidence, stage / name)
    write(root / "single/accepted.json", {"checkpoint_marker_sha256": sha256(source / "checkpoint_complete.json"), "state": state})

    # Fast kernels were installed before B. The single-stage child explicitly
    # disables them; that process-local override cannot leak into four-rank SFT.
    config = stage_config(project, root, "final")
    write(root / "final/training_config.json", config)
    if not (root / "training_accepted.json").is_file():
        run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4",
             "-m", "new_interest.training.train_sft", "--config", str(root / "final/training_config.json")], "train-final.log")
        run([sys.executable, "-m", "new_interest.training.verify_semantic_id_token_adapter",
             "--model-name", contract["models"]["2b"]["model_name"], "--model-revision", contract["models"]["2b"]["model_revision"],
             "--adapter-path", str(root / "final/sft/final_adapter"), "--mapping-file", str(project / baseline.MAPPING),
             "--reference-file", str(root / "final/sft/semantic_reload_reference.json"),
             "--output", str(root / "final/sft/semantic_adapter_reload_audit.json"), "--atol", "0", "--rtol", "0"], "reload.log", "0")
    sft = root / "final/sft"
    summary = read(sft / "run_summary.json")
    for key, expected in {"train_samples": 21600, "actual_optimizer_steps": 2025, "expected_optimizer_steps": 2025,
                          "configured_epochs": 3, "world_size": 4, "effective_global_batch_size": 32,
                          "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary", "seed": 42}.items():
        require(summary[key] == expected, f"Final training contract differs: {key}")
    require(math.isfinite(summary["train_metrics"]["train_loss"]), "Nonfinite final loss")
    require(summary["train_file_sha256"] == sha256(project / baseline.TRAIN), "Wrong final training input")
    expected_recipe = {**contract["recipe"], "per_device_train_batch_size": 8, "gradient_accumulation_steps": 1}
    baseline.audit_resolved_recipe(read(sft / "resolved_config.json"), summary["checkpoint_identity"], expected_recipe)
    for key in ("model_name", "model_revision"):
        require(summary[key] == contract["models"]["2b"][key], "Wrong final model identity")
    require(summary["token_audit"]["over_max_seq_length"] == 0, "Overlength training input")
    require(summary["semantic_token_drift"]["all_input_rows_changed"] and summary["semantic_token_drift"]["all_output_rows_changed"],
            "Some SID rows did not train")
    transition = validate_execution_transition(source_marker["identity"], summary["checkpoint_identity"])
    migration = read(sft / "execution_migration_start.json")
    require(migration["status"] == "passed" and migration["global_step"] == 500 and len(migration["rank_restores"]) == 4, "State migration failed")
    final_cursor = read(sft / "training_sample_cursor.json")
    require(final_cursor["global_step"] == 2025 and final_cursor["consumed_global_samples"] == 64800
            and final_cursor["all_rank_real_sample_order_exact"], "Final sample cursor failed")
    require(final_cursor["consumed_prefix_sha256"] == contract["h50_final_sample_prefix_sha256"], "Sample order differs from H50")
    for attempt in (sft / "execution_attempts").glob("attempt-*"):
        for rank in range(4):
            rng = read(attempt / f"rng_rank_{rank}.json")
            require(rng["status"] == "passed" and rng["actual_rng_sha256"] == rng["expected_rng_sha256"], "Migration RNG mismatch")
            require(read(attempt / f"kernels_rank_{rank}.json")["is_fast_path_available"], "Final kernel binding differs from H50")
    acceptance = {"status": "passed", "adapter": baseline.audit_adapter(project, "2b", sft),
                  "reload": baseline.audit_reload(project, "2b", sft), "cursor": final_cursor,
                  "transition": transition, "steps": 2025, "samples": 21600, "epochs": 3}
    write(root / "training_accepted.json", acceptance)


def worker(project, root):
    from new_interest.experiments import backbone_four_gpu as execution
    project, root = project.resolve(), root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle_hash = verify_bundle(project)
    baseline, contract = bind(project)
    identity = {"protocol": contract["version"], "bundle_sha256": bundle_hash}
    if (root / "identity.json").exists():
        require(read(root / "identity.json") == identity, "Output belongs to another run")
    else:
        write(root / "identity.json", identity)
    inputs = {}
    for split, rel in (("train", TRAIN), ("dev", DEV)):
        inputs[split] = audit_pair(pq.read_table(project / contract["original_data"] / rel).to_pylist(),
                                  pq.read_table(project / contract["shuffled_data"] / rel).to_pylist(), split)
    write(root / "input_audit.json", inputs)
    execution.environment_audit(project, root / "environment.json")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    baseline.prepare_snapshot(project, "2b", root / "snapshot_audit.json")
    snapshot = read(root / "snapshot_audit.json")
    require(snapshot["files"] == read(project / "manifests/h50_snapshot.json")["files"], "Base snapshot differs")
    from transformers import AutoTokenizer
    from new_interest.training.sft_data import audit_token_lengths
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot_path"], local_files_only=True)
    register_semantic_interest_tokens(tokenizer, project / baseline.MAPPING)
    audit = audit_token_lengths(project / baseline.TRAIN, tokenizer, max_seq_length=2048,
                               enable_thinking=False, output_path=root / "token_audit.json")
    require(audit["samples"] == 21600 and audit["over_max_seq_length"] == 0
            and audit["min_assistant_tokens"] == audit["max_assistant_tokens"] == 3, "Token contract differs")
    source = audit_baseline(project, root)
    run = run_factory(project, root)
    install_fast(project, root, run)
    if not (root / "baseline_fresh_reload.json").exists():
        verify_reload(project, source / "sft/final_adapter", source / "sft/semantic_reload_reference.json",
                      root / "baseline_fresh_reload.json", run, "baseline-reload.log")
    require(read(root / "baseline_fresh_reload.json")["all_close"], "Baseline reload not accepted")
    write(root / "startup_ready.json", {**identity, "status": "source_input_snapshot_token_baseline_checks_passed"})
    evaluate(project, root, "B", source / "sft/final_adapter", run)
    print("HISTORY_ORDER_TRAIN_START", flush=True)
    train(project, root, run)
    for arm in ("C", "D"):
        evaluate(project, root, arm, root / "final/sft/final_adapter", run)
    run([sys.executable, "-m", "new_interest.experiments.history_order_analysis", "--project", str(project),
         "--root", str(root), "--baseline-root", str(source)], "analysis.log")
    files = {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file() and "checkpoints" not in p.parts
             and "logs" not in p.parts and p.name not in {"artifact_manifest.json", "complete.json"}
             and not p.name.endswith(".partial")}
    write(root / "artifact_manifest.json", {"identity": identity, "files": files})
    for relative, item in files.items():
        require(sha256(root / relative) == item["sha256"], "Persisted artifact hash differs")
    write(root / "complete.json", {**identity, "status": "remote_training_reload_full_dev_and_comparison_verified",
                                  "artifact_manifest_sha256": sha256(root / "artifact_manifest.json"),
                                  "comparison_sha256": sha256(root / "comparison.json"),
                                  "training": read(root / "training_accepted.json"),
                                  "evaluation": {a: read(root / "evaluations" / a / "accepted.json") for a in ("B", "C", "D")}})
    print("HISTORY_ORDER_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "single"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.mode == "single":
        single_train(args.config)
    else:
        worker(args.project, args.root)


if __name__ == "__main__":
    main()
