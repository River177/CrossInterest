"""Run input-only H30/H10/H5 ablations with the accepted H50 execution schedule.

This orchestrator runs against the frozen Backbone runtime in the submitted bundle.
The first 500 updates use its original single-GPU implementation; the remaining
updates use its accepted four-GPU migration, retaining optimizer/scheduler/RNG state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq

from new_interest.data.build_history_length_ablation import audit_pair, sha256, TRAIN, DEV, MAPPING


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_bundle(project):
    manifest = read(project / "manifests/history_ablation_bundle.json")
    for relative, expected in manifest["files"].items():
        require(sha256(project / relative) == expected, f"Frozen bundle changed: {relative}")
    return sha256(project / "manifests/history_ablation_bundle.json")


def bind(project, limit):
    from new_interest.experiments import backbone_scaling as baseline
    contract_file = Path(f"configs/experiments/history_h{limit}.json")
    contract = read(project / contract_file)
    dataset = Path(contract["ablation_data"])
    # These path-only bindings let the original independent adapter and metric
    # validators inspect the new arm without modifying their numerical logic.
    baseline.CONTRACT = contract_file
    baseline.DATA = dataset
    baseline.TRAIN, baseline.DEV, baseline.MAPPING = dataset / TRAIN, dataset / DEV, dataset / MAPPING
    baseline.CANDIDATE_MAPPING = baseline.MAPPING
    return baseline, contract


def stage_config(project, limit, root, stage):
    baseline, contract = bind(project, limit)
    config = read(project / f"configs/sft/history_h{limit}.json")
    config.update(train_file=str(project / baseline.TRAIN),
                  semantic_id_token_mapping_file=str(project / baseline.MAPPING),
                  output_dir=str(root / stage / "sft"),
                  checkpoint_dir=f"/scratch/history-h{limit}-{root.name}-{stage}",
                  persistent_checkpoint_dir=str(root / stage / "checkpoints"),
                  resume_from_checkpoint="auto")
    if stage == "single":
        config.update(per_device_train_batch_size=1, gradient_accumulation_steps=32)
    else:
        source = root / "single/checkpoints/checkpoint-500"
        config.update(per_device_train_batch_size=8, gradient_accumulation_steps=1,
                      gradient_checkpointing_enabled=True,
                      execution_migration={"source_checkpoint": str(source),
                          "source_stage_artifacts": str(root / "single/source-stage"),
                          "cursor_contract_file": str(root / "single/cursor_contract.json")})
    return config


def stop_at_500(base):
    from transformers import TrainerCallback

    class StopAt500(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= 500:
                require(state.global_step == 500, "Single-GPU stage crossed its frozen stop")
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            require(state.max_steps == 2025, "Stopping at 500 changed the LR horizon")
            require(state.global_step <= 500, "First-stage resume is beyond its endpoint")
            if state.global_step == 500:
                control.should_training_stop = True
            return control

    class StoppedTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(StopAt500())

    return StoppedTrainer


def single_train(config_file):
    from new_interest.training.train_sft import train
    from transformers.models.qwen3_5 import modeling_qwen3_5
    # Preserve the historical first-500 Torch fallback even if a retry node has
    # the later-stage optional kernels installed in its user site.
    modeling_qwen3_5.is_fast_path_available = False
    modeling_qwen3_5.causal_conv1d_fn = None
    modeling_qwen3_5.causal_conv1d_update = None
    modeling_qwen3_5.chunk_gated_delta_rule = None
    modeling_qwen3_5.fused_recurrent_gated_delta_rule = None
    modeling_qwen3_5.FusedRMSNormGated = None
    result = train(read(config_file), trainer_class_transform=stop_at_500)
    require(result["actual_optimizer_steps"] == 500, "First stage did not stop at 500")


def subset_metrics(records, predictions, selected_ids):
    support, hits, gains = Counter(), {k: Counter() for k in (1, 5, 10, 20)}, {k: defaultdict(list) for k in (1, 5, 10, 20)}
    count = 0
    for record, prediction in zip(records, predictions):
        require(str(record["impression_id"]) == str(prediction["impression_id"]), "Cohort prediction order differs")
        if record["impression_id"] not in selected_ids:
            continue
        count += 1
        ranks = {label: rank for rank, label in enumerate(prediction["ranked_labels"], 1)}
        for target in record["acceptable_subcategories"]:
            support[target] += 1
            for k in hits:
                if ranks.get(target, 21) <= k:
                    hits[k][target] += 1
                    gains[k][target].append(1 / math.log2(ranks[target] + 1))
    require(count == len(selected_ids) and support, "Cohort coverage differs")
    metrics = {}
    for k in hits:
        metrics[f"interest_macro_recall_at_{k}"] = math.fsum(hits[k][label] / n for label, n in support.items()) / len(support)
        metrics[f"interest_macro_ndcg_at_{k}"] = math.fsum(math.fsum(gains[k][label]) / n for label, n in support.items()) / len(support)
    return {"records": count, "supported_classes": len(support), "target_occurrences": sum(support.values()), "metrics": metrics}


def worker(project, limit, root):
    from new_interest.experiments import backbone_four_gpu as execution
    from new_interest.training.execution_migration import checkpoint_state_audit, validate_execution_transition
    from new_interest.training.global_sample_cursor import build_hf_cursor_contract

    project, root = project.resolve(), root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle_hash = verify_bundle(project)
    baseline, contract = bind(project, limit)
    identity = {"history_limit": limit, "bundle_sha256": bundle_hash, "protocol": contract["ablation_version"]}
    if (root / "identity.json").exists():
        require(read(root / "identity.json") == identity, "Output belongs to another frozen run")
    else:
        write(root / "identity.json", identity)

    original = project / contract["h50_data"]
    audit = {}
    for split, name in (("train", TRAIN), ("dev", DEV)):
        audit[split] = audit_pair(pq.read_table(original / name).to_pylist(),
                                 pq.read_table(project / contract["ablation_data"] / name).to_pylist(), limit)
    write(root / "input_audit.json", audit)
    execution.environment_audit(project, root / "environment.json")
    # All three arms are offline readers of the already accepted shared snapshot.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    baseline.prepare_snapshot(project, "2b", root / "snapshot_audit.json")
    actual_snapshot = read(root / "snapshot_audit.json")
    require(actual_snapshot["files"] == read(project / "manifests/h50_snapshot.json")["files"], "Base snapshot bytes differ from H50")

    def run(command, log, devices=None):
        env = dict(os.environ)
        if devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = devices
        log_path = Path("/scratch") / f"history-h{limit}-{root.name}-logs" / log
        require(execution.run(command, project, log_path, env=env) == 0, f"Worker command failed: {log}")
        destination = root / "logs" / log
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log_path, destination)

    # Re-audit the full shortened input on the exact worker tokenizer/runtime.
    from transformers import AutoTokenizer
    from new_interest.training.sft_data import audit_token_lengths
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    tokenizer = AutoTokenizer.from_pretrained(actual_snapshot["snapshot_path"], local_files_only=True)
    register_semantic_interest_tokens(tokenizer, project / baseline.MAPPING)
    token_audit = audit_token_lengths(project / baseline.TRAIN, tokenizer, max_seq_length=2048,
                                     enable_thinking=False, output_path=root / "token_audit.json")
    require(token_audit["samples"] == 21600 and token_audit["over_max_seq_length"] == 0
            and token_audit["min_assistant_tokens"] == token_audit["max_assistant_tokens"] == 3,
            "Token audit differs from the H50 completion contract")
    write(root / "startup_ready.json", {**identity, "status": "input_environment_snapshot_token_audits_passed",
                                       "training_schedule": "1 GPU updates 1-500; 4 GPUs updates 501-2025; global batch32"})
    single_config = stage_config(project, limit, root, "single")
    write(root / "single/training_config.json", single_config)
    if not (root / "single/accepted.json").is_file():
        run([sys.executable, "-m", "new_interest.experiments.history_length_ablation", "single", "--config", str(root / "single/training_config.json")], "train-single.log", "0")
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

    run([sys.executable, str(project / "migration_kernel/install_fast_kernels.py"),
         "--manifest", str(project / "migration_kernel/install_manifest.json"),
         "--wheel-cache", f"/scratch/history-fast-wheels-h{limit}",
         "--output", str(root / "fast_kernel_environment.json")], "install-fast.log", "0")
    config = stage_config(project, limit, root, "final")
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

    def evaluate(shard):
        run([sys.executable, "-m", "new_interest.evaluation.evaluate_id_token_ranking",
             "--config", str(project / f"configs/id_token/history_h{limit}.json"),
             "--adapter-path", str(sft / "final_adapter"), "--output-dir", str(root / f"real_eval/shard-{shard:02d}"),
             "--num-shards", "4", "--shard-index", str(shard), "--batch-size", "8"], f"eval-{shard}.log", str(shard))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate, range(4)))
    from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
    merge_id_token_shards([root / f"real_eval/shard-{i:02d}" for i in range(4)], project / baseline.DEV,
                         project / baseline.MAPPING, root / "real_eval/merged", expected_records=31392)
    evaluation = baseline.audit_evaluation(project, "2b", root, "full")
    cohort = set(read(project / "data/training/mind_history_length_ablation_20260915_v1/dev_h50_cohort.json")["impression_ids"])
    cohort_result = subset_metrics(pq.read_table(project / baseline.DEV).to_pylist(),
                                  pq.read_table(root / "real_eval/merged/id_token_predictions.parquet").to_pylist(), cohort)
    write(root / "evaluation_readback.json", evaluation)
    write(root / "cohort_metrics.json", cohort_result)
    manifest = {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in root.rglob("*") if p.is_file() and p.name not in {"artifact_manifest.json", "complete.json"}
                and "checkpoints" not in p.parts and not p.name.endswith(".partial")}
    write(root / "artifact_manifest.json", {"files": manifest})
    for relative, item in manifest.items():
        require(sha256(root / relative) == item["sha256"], "Persisted artifact failed readback")
    write(root / "complete.json", {**identity, "status": "remote_training_reload_prediction_and_metric_checks_passed",
                                  "artifact_manifest_sha256": sha256(root / "artifact_manifest.json"),
                                  "training": acceptance, "evaluation": evaluation, "h50_cohort": cohort_result})
    print("HISTORY_ABLATION_COMPLETE", limit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("single", "worker"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--history-limit", type=int, choices=(5, 10, 30))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.mode == "single":
        single_train(args.config)
    else:
        worker(args.project, args.history_limit, args.root)


if __name__ == "__main__":
    main()
