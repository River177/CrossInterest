"""Independently accept one requested CIR condition using compact remote receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def accept(project: Path, manifest_path: Path, condition: str) -> dict:
    tracking = read(project / manifest_path)
    spec = tracking["conditions"][condition]
    job = tracking["jobs"][spec["job_mode"]]
    bundle = project / tracking["bundle"]
    require(sha(bundle / "manifests/cir_bundle.json") == tracking["bundle_sha256"], "Bundle identity")
    contract = read(bundle / "configs/experiments/mind_cir_108_v1.json")
    preflight = read(bundle / "manifests/four_condition_preflight.json")
    base = project / tracking["artifact_root"]
    jobroot = base / "final-readback" / job.get("experiment", tracking["experiment"]) / job["job"]
    root = jobroot / spec["output_subdir"]
    complete = read(root / "complete.json")
    require(complete["status"] == "accepted" and complete["condition"] == condition
            and complete["mode"] == spec["job_mode"] and complete["bundle_sha256"] == tracking["bundle_sha256"], "Condition identity")
    require(complete["stage_keys"] == spec["stage_keys"] and complete["protocol"] == contract["conditions"][condition], "Condition protocol")
    require(len(complete["stages"]) == len(spec["stage_keys"]), "Incomplete stage list")
    stage_reports = {}
    for key, accepted in zip(spec["stage_keys"], complete["stages"]):
        expected = preflight["stages"][key]
        folder = jobroot / "formal" / key
        stage, summary = read(folder / "stage_accepted.json"), read(folder / "run_summary.json")
        reload, initial, start = [read(folder / name) for name in ("semantic_adapter_reload_audit.json", "initial_adapter_state.json", "training_start_audit.json")]
        require(stage == accepted and stage["optimizer_steps"] == summary["actual_optimizer_steps"] == expected["steps"], "Stage updates")
        require(stage["epochs"] == summary["configured_epochs"] == expected["epochs"] and stage["samples"] == expected["samples"], "Stage duration")
        data = bundle / ("data/training/mind_cir_schedule_108_v1/mixed.parquet" if key.startswith("mixed") else
                         f"data/training/mind_cir_108_v1/{'cir' if key == 'cir5' else 'ni'}.parquet")
        require(stage["train_file_sha256"] == summary["train_file_sha256"] == sha(data) == expected["train_sha256"], "Training input")
        observed = summary["observed_training_exposure"]
        require(observed["presentations"] == expected["presentations"] and
                observed["min_presentations_per_sample"] == observed["max_presentations_per_sample"] == expected["epochs"], "Exact exposure")
        rows = pq.read_table(data).to_pylist()
        classes = Counter((row.get("training_task", "new_interest"), row["target_subcategory"]) for row in rows)
        for (task, label), count in classes.items():
            require(observed["by_task_and_class_presentations"][task][label] == count * expected["epochs"], "Task/class exposure")
        require(sum(x["assistant_tokens"] for x in observed["by_task"].values()) == expected["presentations"] * 3, "Supervised tokens")
        require(stage["stage_parent"] == expected["parent"], "Stage parent")
        digest = hashlib.sha256()
        for tensor, info in sorted(initial["tensors"].items()):
            digest.update(json.dumps([tensor, info], sort_keys=True, separators=(",", ":")).encode())
        require(digest.hexdigest() == initial["state_sha256"] == start["source_adapter_state_sha256"] == start["actual_adapter_state_sha256"], "Initial tensor manifest")
        require(start["fresh_optimization_state"] and start["global_step"] == start["optimizer_state_entries"] == start["scheduler_last_epoch"] == 0, "Fresh optimizer/scheduler")
        parent_hash = None
        if expected["parent"]:
            parent_hash = stage_reports[expected["parent"]]["adapter_sha256"]
            require(start["checkpoint_identity"]["input_adapter_sha256"] == parent_hash and
                    initial["state_sha256"] == initial["input_adapter_state_sha256"], "Parameter handoff")
        else:
            baseline = read(bundle / "manifests/h50_baseline.json")
            require(start["checkpoint_identity"]["input_adapter_sha256"] is None and
                    initial["state_sha256"] == baseline["initial_adapter_state_sha256"], "Cold initialization")
        integrity = reload["adapter_state_integrity"]
        require(reload["all_close"] and reload["atol"] == reload["rtol"] == 0 and integrity["all_tensors_exact"]
                and integrity["tensor_count"] == 373 and integrity["saved_state_sha256"] == integrity["loaded_state_sha256"], "Exact reload")
        require(all(s["max_absolute_difference"] == 0 and s["top20_exact_match"] for s in reload["samples"]), "Exact reload logits")
        stage_reports[key] = {"steps": expected["steps"], "epochs": expected["epochs"], "samples": expected["samples"],
            "presentations": expected["presentations"], "parent_adapter_sha256": parent_hash,
            "adapter_sha256": stage["adapter"]["adapter_sha256"], "initial_state_sha256": initial["state_sha256"], "exact_reload": True}
    metrics = read(root / "real_eval/merged/metrics.json")
    merge = read(root / "real_eval/merged/merge_manifest.json")
    remote = read(root / "evaluation_readback.json")
    require(remote == complete["evaluation"] and remote["records"] == metrics["samples"] == merge["record_count"] == merge["expected_record_count"] == 31392, "Evaluation coverage")
    require(remote["metrics_sha256"] == merge["metrics_sha256"] == sha(root / "real_eval/merged/metrics.json"), "Metrics digest")
    require(remote["prediction_sha256"] == merge["merged_prediction_sha256"] and remote["evaluation_identity"] == metrics["evaluation_identity"] == merge["evaluation_identity"], "Prediction identity")
    data_root = bundle / "data/training/mind_cir_108_v1"
    require(metrics["eval_sha256"] == sha(data_root / "dev_full_grouped.parquet") and metrics["mapping_sha256"] == sha(data_root / "interest_token_mapping.parquet"), "Evaluation inputs")
    require(metrics["label_vocabulary_size"] == 108 and metrics["top_k"] == 20 and not metrics["history_prediction_filtering"], "Ranking protocol")
    for k in ("model_name", "model_revision"):
        require(metrics[k] == contract["model"][k], "Evaluation model")
    dev = pq.read_table(data_root / "dev_full_grouped.parquet").to_pylist()
    support = Counter(c for row in dev for c in row["acceptable_subcategories"])
    require(dict(support) == metrics["per_class_support"] and len(support) == remote["supported_classes"] == 104
            and sum(support.values()) == remote["target_occurrences"] == 37728, "Class supports")
    shards = []
    for i in range(4):
        folder = root / f"real_eval/shard-{i:02d}"
        s, identity = read(folder / "metrics.json"), read(folder / "shard_manifest.json")
        require(s["samples"] == identity["shard_record_count"] == 7848 and s["shard_index"] == identity["shard_index"] == i, "Shard coverage")
        require(identity["first_record_index"] == i and identity["last_record_index"] == 31388 + i, "Shard topology")
        for field in ("model_name", "model_revision", "adapter_sha256", "eval_sha256", "mapping_sha256", "evaluation_identity"):
            require(s[field] == identity[field] == metrics[field], "Shard identity")
        require(identity["prediction_sha256"] == merge["shard_prediction_sha256"][str(i)], "Shard digest")
        expected = Counter(c for row in dev[i::4] for c in row["acceptable_subcategories"])
        require(dict(expected) == s["per_class_support"], "Modulo shard supports")
        shards.append(s)
    values = {}
    for k in (1, 5, 10, 20):
        for measure in ("recall", "ndcg"):
            per_class = []
            for label, n in support.items():
                total = math.fsum(s["per_class_support"].get(label, 0) * s[f"per_class_{measure}"][str(k)].get(label, 0) for s in shards)
                if measure == "recall":
                    require(abs(total - round(total)) < 1e-9, "Noninteger recall numerator")
                value = total / n
                require(abs(value - metrics[f"per_class_{measure}"][str(k)][label]) < 1e-12, "Per-class metric")
                per_class.append(value)
            key = f"interest_macro_{measure}_at_{k}"
            values[key] = math.fsum(per_class) / 104
            require(abs(values[key] - metrics[key]) < 1e-12 and abs(values[key] - remote["metrics"][key]) < 1e-12, "Macro metric")
    result = {"status": "accepted", "condition": condition, "bundle_sha256": tracking["bundle_sha256"],
        "backend": job["backend_id"], "job": job["job"], "stages": stage_reports,
        "evaluation": remote, "independently_recomputed_metrics": values, "full_dev_records": 31392,
        "files": {str(p.relative_to(jobroot)): sha(p) for p in sorted(jobroot.rglob("*.json")) if p.is_file()},
        "large_artifact_policy": "remote_only"}
    output = base / "acceptance" / f"{condition}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {"status": "accepted", "condition": condition, "metrics": values, "acceptance_sha256": sha(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/mind_cir_four_108_20260918_v1.json"))
    parser.add_argument("--condition", choices=("cir5_ni3", "cir5_ni5", "mixed3", "mixed5"), required=True)
    args = parser.parse_args()
    print(json.dumps(accept(Path.cwd(), args.manifest, args.condition), indent=2))
