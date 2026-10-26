"""Cross-check final CIR receipts and recompute metrics from compact shard statistics."""
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def acceptance_scope(project, manifest_file=None):
    manifest_file = manifest_file or Path("manifests/mind_cir_108_20260916_v1.json")
    manifest = read(project / manifest_file)
    active = manifest["active_run"]
    bundle = project / active["bundle"]
    check(sha(bundle / "manifests/cir_bundle.json") == active["bundle_sha256"], "Tracked bundle changed")
    contract = read(bundle / "configs/experiments/mind_cir_108_v1.json")
    schedule = contract.get("training_schedule", {"cir_epochs": 1, "ni_epochs": 3})
    check(all(type(v) is int and v > 0 for v in schedule.values()), "Invalid epoch schedule")
    base = project / manifest.get("artifact_root", "artifacts/mind-cir-20260916")
    root = base / "final-readback" / active["experiment"] / active["job"]
    report = project / manifest.get("report", "docs/experiments/2026-09-16/mind_cir_warmup.md")
    return bundle, root, base, report.with_name(report.stem + "_results.csv"), contract, schedule


def accept(project, manifest_file=None, *, write_outputs=True):
    bundle, root, artifact_root, csv_path, contract, schedule = acceptance_scope(project, manifest_file)
    cir_epochs, ni_epochs = schedule["cir_epochs"], schedule["ni_epochs"]
    cir_samples = contract["data"]["cir_rows"]
    cir_steps, ni_steps = math.ceil(cir_samples / 32) * cir_epochs, 675 * ni_epochs
    complete = read(root / "complete.json")
    metrics = read(root / "formal/real_eval/merged/metrics.json")
    merge = read(root / "formal/real_eval/merged/merge_manifest.json")
    remote = read(root / "formal/evaluation_readback.json")
    baseline = read(bundle / "manifests/h50_baseline.json")
    check(complete["status"] == "accepted" and complete["bundle_sha256"] == sha(bundle / "manifests/cir_bundle.json"), "Completion identity")
    check(complete["evaluation"] == remote and complete["baseline_reference"] == baseline, "Completion receipt links")
    check(remote["records"] == metrics["samples"] == merge["record_count"] == merge["expected_record_count"] == 31392, "Full coverage")
    check(remote["target_occurrences"] == 37728 and remote["supported_classes"] == metrics["supported_interest_count"] == 104, "Target denominator")
    check(remote["metrics_sha256"] == merge["metrics_sha256"] == sha(root / "formal/real_eval/merged/metrics.json"), "Metrics hash")
    check(remote["prediction_sha256"] == merge["merged_prediction_sha256"], "Prediction hash link")
    check(remote["evaluation_identity"] == metrics["evaluation_identity"] == merge["evaluation_identity"], "Evaluation identity")
    check(metrics["eval_sha256"] == sha(bundle / "data/training/mind_cir_108_v1/dev_full_grouped.parquet"), "Dev identity")
    check(metrics["mapping_sha256"] == sha(bundle / "data/training/mind_cir_108_v1/interest_token_mapping.parquet"), "SID mapping")
    check(metrics["label_vocabulary_size"] == 108 and not metrics["history_prediction_filtering"] and metrics["top_k"] == 20, "Ranking protocol")
    check(metrics["model_name"] == "Qwen/Qwen3.5-2B" and metrics["model_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc", "Model identity")
    dev = pq.read_table(bundle / "data/training/mind_cir_108_v1/dev_full_grouped.parquet").to_pylist()
    support = Counter(c for r in dev for c in r["acceptable_subcategories"])
    check(dict(support) == metrics["per_class_support"] and sum(support.values()) == 37728, "Ground-truth support")
    shards = []
    for index in range(4):
        folder = root / f"formal/real_eval/shard-{index:02d}"
        m, manifest = read(folder / "metrics.json"), read(folder / "shard_manifest.json")
        check(m["samples"] == manifest["shard_record_count"] == 7848 and m["shard_index"] == manifest["shard_index"] == index, "Shard scope")
        check(manifest["first_record_index"] == index and manifest["last_record_index"] == 31388 + index, "Shard endpoints")
        for key in ("model_name", "model_revision", "adapter_sha256", "eval_sha256", "mapping_sha256", "evaluation_identity"):
            check(m[key] == manifest[key] == metrics[key], f"Shard identity: {key}")
        check(manifest["prediction_sha256"] == merge["shard_prediction_sha256"][str(index)], "Shard prediction link")
        expected = Counter(c for r in dev[index::4] for c in r["acceptable_subcategories"])
        check(dict(expected) == m["per_class_support"], "Actual modulo-shard label support")
        shards.append(m)
    recomputed = {}
    for k in (1, 5, 10, 20):
        for measure in ("recall", "ndcg"):
            values = {}
            for label, n in support.items():
                total = math.fsum(m["per_class_support"].get(label, 0) * m[f"per_class_{measure}"][str(k)].get(label, 0) for m in shards)
                if measure == "recall":
                    check(abs(total - round(total)) < 1e-9, "Recall numerator is not an integer")
                values[label] = total / n
                check(abs(values[label] - metrics[f"per_class_{measure}"][str(k)][label]) < 1e-12, "Per-class aggregation")
            key = f"interest_macro_{measure}_at_{k}"
            recomputed[key] = math.fsum(values.values()) / 104
            check(abs(recomputed[key] - metrics[key]) < 1e-12 and abs(recomputed[key] - remote["metrics"][key]) < 1e-12, "Macro aggregation")
    for i, (name, step, n, epochs) in enumerate((("cir", cir_steps, cir_samples, cir_epochs), ("ni", ni_steps, 21600, ni_epochs))):
        stage = read(root / f"formal/{name}/stage_accepted.json")
        summary = read(root / f"formal/{name}/run_summary.json")
        reload = read(root / f"formal/{name}/semantic_adapter_reload_audit.json")
        check(stage == complete["stages"][i] and stage["optimizer_steps"] == step and stage["samples"] == n and stage["epochs"] == epochs, "Stage completion")
        check(summary["actual_optimizer_steps"] == step and summary["observed_training_exposure"]["presentations"] == n * epochs, "Stage exposure")
        check(reload["all_close"] and reload["atol"] == reload["rtol"] == 0 and reload["adapter_state_integrity"]["all_tensors_exact"], "Exact stage reload")
        check(reload["adapter_state_integrity"]["saved_state_sha256"] == reload["adapter_state_integrity"]["loaded_state_sha256"], "Reload tensor identity")
    baseline_path = Path("/Users/river/.codex/worktrees/3357/CrossInterest") / baseline["metrics_path"]
    check(sha(baseline_path) == baseline["metrics_sha256"], "Baseline artifact hash")
    baseline_metrics = read(baseline_path)
    # Saved metrics and their independent fsum readback differ by at most an ULP.
    baseline_max_error = max(abs(baseline_metrics[k] - v) for k, v in baseline["evaluation"]["metrics"].items())
    check(baseline_max_error < 1e-12, "Baseline readback agreement")
    for key in ("eval_sha256", "mapping_sha256", "samples", "model_name", "model_revision", "history_prediction_filtering", "top_k"):
        check(baseline_metrics[key] == metrics[key], f"Baseline comparison scope: {key}")
    comparison = []
    cir1 = None
    if contract.get("existing_cir1_reference"):
        cir1_path = bundle / contract["reuse_verified_gates"] / "acceptance.json"
        check(sha(cir1_path) == contract["existing_cir1_reference"]["acceptance_sha256"], "CIR1 acceptance identity")
        cir1 = read(cir1_path)
        check(cir1["status"] == "accepted" and cir1["training_steps"]["cir"] == 540, "CIR1 comparison condition")
        check(cir1["evaluation"] == contract["existing_cir1_reference"]["evaluation"], "CIR1 metric identity")
    for measure, k in (("recall", 10), ("recall", 20), ("ndcg", 10), ("ndcg", 20)):
        key = f"interest_macro_{measure}_at_{k}"
        before, after = baseline_metrics[key], recomputed[key]
        comparison.append({"metric": key, "NI_only": before, "CIR_to_NI": after,
                           "delta_percentage_points": 100 * (after - before), "relative_change_percent": 100 * (after / before - 1)})
        if cir1:
            prior = cir1["evaluation"]["metrics"][key]
            comparison[-1].update(CIR1_to_NI=prior, delta_vs_CIR1_percentage_points=100 * (after - prior))
    result = {"status": "accepted", "backend_native_status": "Completed", "bundle_sha256": complete["bundle_sha256"],
        "complete_json_sha256": sha(root / "complete.json"), "training_steps": {"cir": cir_steps, "ni": ni_steps},
        "training_presentations": {"cir": cir_samples * cir_epochs, "ni": 21600 * ni_epochs},
        "full_dev": {"records": 31392, "target_occurrences": 37728, "supported_classes": 104, "ranking_classes": 108, "logical_shards": 4, "records_per_shard": 7848},
        "validation": {"remote_full_prediction_identity_topology_and_metric_checks": True,
            "local_frozen_ground_truth_and_modulo_shard_support_checks": True,
            "local_all_class_recall_and_ndcg_recomputed_from_shard_statistics": True,
            "all_reported_metrics_match_within_1e_12": True, "stage_lineage_and_exact_reload_receipts_verified": True,
            "baseline_metrics_file_hash_verified": True, "baseline_original_vs_fsum_max_abs_error": baseline_max_error},
        "evaluation": remote, "comparison": comparison, "baseline_metrics_sha256": sha(baseline_path),
        "NI_adapter_file_sha256": complete["stages"][1]["adapter"]["adapter_sha256"],
        "evaluation_adapter_directory_sha256": metrics["adapter_sha256"],
        "large_artifact_policy": "Weights/checkpoints/full predictions remain remote; local compact statistics and receipts",
        "files": {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}}
    output = artifact_root / "final_acceptance.json"
    if write_outputs:
        output.write_text(json.dumps(result, indent=2) + "\n")
        with csv_path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    return {"status": result["status"], "comparison": comparison, "full_dev": result["full_dev"],
            "training_steps": result["training_steps"], "written": write_outputs,
            "acceptance_sha256": sha(output) if write_outputs else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(accept(Path.cwd(), args.manifest, write_outputs=not args.check_only), indent=2))
