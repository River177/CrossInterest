"""Independently verify reference scores on the remote persistent filesystem.

No model tensors or prediction files are transferred by this reader. Its output
is a compact acceptance report with metrics, hashes, and restoration pointers.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import _atomic_write_json, canonical_sha256, sha256_file
from new_interest.infrastructure.paths import portable_path
from new_interest.training.semantic_id_tokens import load_interest_token_mapping

EXPECTED_VERSION = "rho-reference-first-sid-full-effective-vocab-fp32-v1"
EXPECTED_LOSS = "first-supervised-SID; native-effective-head-cast; FP32-projection-and-logsumexp; no-prior-correction"
SOURCE_FIELDS = ("source_sample_id", "user_id", "target_subcategory", "messages")
RUNTIME_FIELDS = {"evaluation_identity", "shard_index", "shard_record_count", "full_vocabulary_size",
    "scores_sha256", "metrics_sha256", "loading_audit_sha256", "checkpoint_files", "elapsed_seconds", "restoration",
    "kernel_selection_sha256"}
STABLE_KERNEL_PROTOCOL = "fla-valid-candidates-stable-sha256-v1"


def verify_kernel_selection(directory, manifest, common_configs=None):
    """Verify kernel evidence without importing Triton or evaluating a model."""
    selection = manifest.get("kernel_selection", "native")
    if selection == "native":
        if (manifest.get("kernel_selection_protocol", "native-fla-autotune-v1") != "native-fla-autotune-v1"
                or manifest.get("kernel_selector_sha256") is not None
                or "kernel_selection_sha256" in manifest):
            raise ValueError("Native kernel-selection identity is inconsistent")
        return None
    if selection != "stable_fla_v1" or manifest.get("kernel_selection_protocol") != STABLE_KERNEL_PROTOCOL:
        raise ValueError("Unsupported stable kernel-selection protocol")
    from new_interest.training import probe_kernel_diagnostic as selector
    if manifest.get("kernel_selector_sha256") != sha256_file(Path(selector.__file__)):
        raise ValueError("Stable kernel selector source identity mismatch")
    path = Path(directory) / "kernel_selection.json"
    if sha256_file(path) != manifest.get("kernel_selection_sha256"):
        raise ValueError("Kernel selection receipt hash mismatch")
    receipt = json.loads(path.read_text())
    for key in ("kernel_selection", "kernel_selection_protocol", "kernel_selector_sha256", "evaluation_identity"):
        if receipt.get(key) != manifest.get(key):
            raise ValueError("Kernel receipt/evaluation identity mismatch")
    snapshots = receipt.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots or canonical_sha256(snapshots) != receipt.get("snapshots_sha256"):
        raise ValueError("Kernel snapshot content hash mismatch")
    seen = {} if common_configs is None else common_configs
    stages = set()
    for snapshot in snapshots:
        if not isinstance(snapshot.get("stage"), str) or not isinstance(snapshot.get("kernels"), dict) or not snapshot["kernels"]:
            raise ValueError("Stable kernel snapshot has no selected configurations")
        stages.add(snapshot["stage"])
        for name, kernel in snapshot["kernels"].items():
            if not isinstance(kernel.get("cache"), dict) or not kernel["cache"] or kernel.get("cache_results") is not False:
                raise ValueError("Stable kernel snapshot has an invalid cache/disk-write policy")
            for key, config in kernel["cache"].items():
                if not isinstance(key, str) or not isinstance(config, str) or not config:
                    raise ValueError("Invalid selected kernel configuration")
                pair = (name, key)
                if pair in seen and seen[pair] != config:
                    raise ValueError("Stable kernel configuration differs for an identical cache key")
                seen[pair] = config
    expected_stages = {f"checkpoint:{start}:{min(start+manifest['checkpoint_size'], manifest['shard_record_count'])}"
        for start in range(0, manifest["shard_record_count"], manifest["checkpoint_size"])}
    if "native_head_audit" not in stages or not expected_stages <= stages:
        raise ValueError("Kernel receipt does not cover head audit and all score checkpoints")
    return {"sha256": manifest["kernel_selection_sha256"], "snapshots_sha256": receipt["snapshots_sha256"],
            "snapshot_count": len(snapshots)}


def read_source(path):
    rows = pq.read_table(path, columns=list(SOURCE_FIELDS)).to_pylist()
    if not rows or any(r[k] is None for r in rows for k in SOURCE_FIELDS):
        raise ValueError("Empty or null reference source")
    for key in ("source_sample_id", "user_id"):
        if len({str(r[key]) for r in rows}) != len(rows):
            raise ValueError(f"Duplicate source {key}")
    return rows


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value-maximum) for value in values))


def close(actual, expected, name, *, tolerance=2e-5):
    if not math.isfinite(float(actual)) or abs(float(actual)-float(expected)) > tolerance:
        raise ValueError(f"Independent {name} recomputation mismatch: {actual} vs {expected}")


def verify_row(row, record, index, identity):
    if (row["record_index"] != index or row["source_sample_id"] != str(record["source_sample_id"])
            or row["user_id"] != str(record["user_id"])
            or row["target_subcategory"] != str(record["target_subcategory"])
            or row["input_sha256"] != canonical_sha256({k: record[k] for k in SOURCE_FIELDS})
            or row["evaluation_identity"] != identity["evaluation_identity"]):
        raise ValueError("Independent input/order/identity mismatch")
    target = identity["labels"].index(str(record["target_subcategory"]))
    if row["target_interest_index"] != target or row["target_token_id"] != identity["token_ids"][target]:
        raise ValueError("Independent target SID identity mismatch")
    scores = [float(value) for value in row["sid_scores_fp32"]]
    if len(scores) != len(identity["labels"]) or not all(math.isfinite(value) for value in scores):
        raise ValueError("Invalid SID score vector")
    if row["full_vocabulary_size"] != identity["full_vocabulary_size"] or row["full_vocabulary_size"] <= max(identity["token_ids"]):
        raise ValueError("Invalid full-vocabulary denominator size")
    if not 1 <= row["prefix_tokens"] <= identity["max_seq_length"]:
        raise ValueError("Invalid first-SID prefix length")
    close(row["target_logit_fp32"], scores[target], "target logit", tolerance=0)
    conditional_lse = logsumexp(scores)
    close(row["sid_logsumexp_fp32"], conditional_lse, "conditional logsumexp")
    close(row["sid_conditional_nll"], conditional_lse-scores[target], "conditional NLL")
    # Full logits need not be stored. The full denominator and target scalar
    # independently determine NLL; this checks serialization/arithmetic, not a
    # second full model inference. The loading/head audit supplies that evidence.
    full_lse = float(row["full_vocab_logsumexp_fp32"])
    if not math.isfinite(full_lse) or full_lse + 2e-5 < conditional_lse:
        raise ValueError("Full denominator is smaller than its SID subset")
    close(row["full_vocab_first_sid_nll"], full_lse-scores[target], "full-vocabulary NLL")
    if row["full_vocab_first_sid_nll"] < -2e-5:
        raise ValueError("Negative full-vocabulary loss")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rank = order.index(target)+1
    if row["ranked_interest_indices"] != order[:min(20, len(scores))] or row["target_sid_rank"] != rank:
        raise ValueError("Independent SID rank/top20 mismatch")


def independent_metrics(rows):
    n = len(rows)
    labels = sorted({r["target_subcategory"] for r in rows})
    def stats(key):
        values = sorted(float(r[key]) for r in rows)
        return {"mean": math.fsum(values)/n, "min": values[0], "max": values[-1],
            "p10": values[round((n-1)*.1)], "median": values[round((n-1)*.5)],
            "p90": values[round((n-1)*.9)]}
    return {"rows": n, "users": len({r["user_id"] for r in rows}),
        "class_support": dict(sorted(Counter(r["target_subcategory"] for r in rows).items())),
        "full_vocab_first_sid_nll": stats("full_vocab_first_sid_nll"),
        "sid_conditional_nll": stats("sid_conditional_nll"),
        "sid_recall": {str(k): sum(r["target_sid_rank"] <= k for r in rows)/n for k in (1, 5, 10, 20)},
        "per_class_mean_full_vocab_first_sid_nll": {label: math.fsum(r["full_vocab_first_sid_nll"]
            for r in rows if r["target_subcategory"] == label)/sum(r["target_subcategory"] == label for r in rows)
            for label in labels}}


def verify_reference(score_dirs, teacher_train_file, score_file, mapping_file, output=None,
                     adapter_path=None, teacher_run_summary=None):
    dirs = [Path(p) for p in score_dirs]
    if not dirs or len(set(p.resolve() for p in dirs)) != len(dirs):
        raise ValueError("Missing or duplicate score directories")
    manifests = [json.loads((d/"manifest.json").read_text()) for d in dirs]
    reference = manifests[0]
    if reference["identity_version"] != EXPECTED_VERSION or reference["loss_definition"] != EXPECTED_LOSS:
        raise ValueError("Unsupported scoring definition")
    if len({m["evaluation_identity"] for m in manifests}) != 1:
        raise ValueError("Mixed teacher scoring identities")
    if len(dirs) != reference["num_shards"] or sorted(m["shard_index"] for m in manifests) != list(range(len(dirs))):
        raise ValueError("Missing or duplicate logical shards")
    teacher = read_source(teacher_train_file)
    records = read_source(score_file)
    if len(teacher) != reference["teacher_rows"] or len(records) != reference["full_record_count"]:
        raise ValueError("Teacher/score source cardinality changed")
    for key in ("source_sample_id", "user_id"):
        if {r[key] for r in teacher} & {r[key] for r in records}:
            raise ValueError(f"Teacher/score {key} overlap")
    for path, field in ((teacher_train_file, "teacher_train_sha256"), (score_file, "score_file_sha256"), (mapping_file, "mapping_sha256")):
        if sha256_file(Path(path)) != reference[field]:
            raise ValueError(f"Source hash mismatch: {field}")
    mapping = load_interest_token_mapping(mapping_file)
    if ([r["target_subcategory"] for r in mapping] != reference["labels"]
            or [r["interest_token"] for r in mapping] != reference["tokens"]):
        raise ValueError("Mapping order differs")
    adapter = Path(adapter_path or reference["restoration"]["adapter_path"])
    adapter_files = {str(p.relative_to(adapter)): sha256_file(p) for p in sorted(adapter.rglob("*"))
                     if p.is_file() and p.name not in {"README.md", "training_args.bin"}}
    if adapter_files != reference["adapter_files"] or canonical_sha256(adapter_files) != reference["adapter_sha256"]:
        raise ValueError("Saved teacher adapter artifact mismatch")
    summary_path = Path(teacher_run_summary or reference["restoration"]["teacher_summary"])
    if sha256_file(summary_path) != reference["teacher_summary_sha256"]:
        raise ValueError("Teacher provenance summary hash changed")
    summary = json.loads(summary_path.read_text())
    if (summary["train_file_sha256"] != reference["teacher_train_sha256"] or summary.get("input_adapter") is not None
            or summary["model_revision"] != reference["model_revision"]
            or summary.get("semantic_id_loss_scope", "full_vocabulary") != "full_vocabulary"):
        raise ValueError("Teacher fresh-base/disjoint provenance invalid")
    rows, artifact_receipts, common_kernel_configs = [], [], {}
    for directory, manifest in zip(dirs, manifests):
        identity = {k: v for k, v in manifest.items() if k not in RUNTIME_FIELDS}
        if canonical_sha256(identity) != manifest["evaluation_identity"]:
            raise ValueError("Manifest canonical identity mismatch")
        if manifest["full_vocabulary_size"] != reference["full_vocabulary_size"]:
            raise ValueError("Inconsistent full-vocabulary size")
        kernel_receipt = verify_kernel_selection(directory, manifest, common_kernel_configs)
        for filename, key in (("scores.parquet", "scores_sha256"), ("metrics.json", "metrics_sha256"),
                              ("loading_audit.json", "loading_audit_sha256")):
            if sha256_file(directory/filename) != manifest[key]:
                raise ValueError(f"Artifact hash mismatch: {filename}")
        audit = json.loads((directory/"loading_audit.json").read_text())
        if (not audit["pretrained_text_loaded"] or audit["missing_text_keys"] or audit["unexpected_text_keys"]
                or not audit["adapter_tensor_reload"]["accepted"]
                or audit["head_projection"]["native_head_max_abs_error"] != 0
                or audit["head_projection"]["full_vocabulary_size"] != reference["full_vocabulary_size"]):
            raise ValueError("Fresh reload/effective head audit did not pass")
        indices = list(range(manifest["shard_index"], len(records), manifest["num_shards"]))
        part = pq.read_table(directory/"scores.parquet").to_pylist()
        if len(part) != len(indices) or len(part) != manifest["shard_record_count"]:
            raise ValueError("Logical shard row count mismatch")
        for row, index in zip(part, indices):
            verify_row(row, records[index], index, manifest)
        cached_rows = []
        expected_checkpoint_names = {f"shards/{start:09d}-{min(start+manifest['checkpoint_size'], len(part)):09d}.parquet"
            for start in range(0, len(part), manifest["checkpoint_size"])}
        if set(manifest["checkpoint_files"]) != expected_checkpoint_names:
            raise ValueError("Checkpoint topology differs")
        actual_checkpoint_names = {str(p.relative_to(directory)) for p in (directory/"shards").glob("*.parquet")}
        if actual_checkpoint_names != expected_checkpoint_names:
            raise ValueError("Extra/missing checkpoint files")
        for filename, digest in sorted(manifest["checkpoint_files"].items()):
            if sha256_file(directory/filename) != digest:
                raise ValueError("Checkpoint hash mismatch")
            cached_rows.extend(pq.read_table(directory/filename).to_pylist())
        if cached_rows != part:
            raise ValueError("Final scores differ from committed shards")
        expected = independent_metrics(part)
        cached = json.loads((directory/"metrics.json").read_text())
        if cached["rows"] != len(part) or cached["class_support"] != expected["class_support"]:
            raise ValueError("Cached metric support mismatch")
        close(cached["mean_full_vocab_first_sid_nll"], expected["full_vocab_first_sid_nll"]["mean"], "mean full NLL", tolerance=1e-10)
        close(cached["mean_sid_conditional_nll"], expected["sid_conditional_nll"]["mean"], "mean conditional NLL", tolerance=1e-10)
        for k, value in expected["sid_recall"].items():
            close(cached["sid_recall"][k], value, "recall", tolerance=0)
        rows.extend(part)
        artifact_receipts.append({"directory": portable_path(directory), "manifest_sha256": sha256_file(directory/"manifest.json"),
                                 "scores_sha256": manifest["scores_sha256"],
                                 **({"kernel_selection_receipt": kernel_receipt} if kernel_receipt else {})})
    rows.sort(key=lambda r: r["record_index"])
    if [r["record_index"] for r in rows] != list(range(len(records))):
        raise ValueError("Complete opposite-fold union has missing/duplicate row indices")
    if len({r["source_sample_id"] for r in rows}) != len(records) or len({r["user_id"] for r in rows}) != len(records):
        raise ValueError("Duplicate sample/user in complete reference output")
    report = {"accepted": True, "reader_version": EXPECTED_VERSION+"-independent-reader-v1",
        "reader_sha256": sha256_file(Path(__file__)), "evaluation_identity": reference["evaluation_identity"],
        "loss_definition": EXPECTED_LOSS, "teacher_rows": len(teacher), "teacher_score_user_overlap": 0,
        "full_opposite_fold_coverage": True, "metrics": independent_metrics(rows),
        "teacher_train_sha256": reference["teacher_train_sha256"], "score_file_sha256": reference["score_file_sha256"],
        "mapping_sha256": reference["mapping_sha256"], "adapter_sha256": reference["adapter_sha256"],
        "kernel_selection": reference.get("kernel_selection", "native"),
        "kernel_selection_protocol": reference.get("kernel_selection_protocol", "native-fla-autotune-v1"),
        "kernel_selector_sha256": reference.get("kernel_selector_sha256"),
        "scored_source_ids_sha256": canonical_sha256(sorted(r["source_sample_id"] for r in rows)),
        "artifacts": artifact_receipts,
        "verification_scope": "all artifact hashes, every source identity, saved tensor reload receipt, native effective head parity receipt, independent NLL algebra/conditional108 softmax/ranks/support; no repeated model inference",
        "large_artifact_policy": "predictions and model artifacts remain on persistent remote storage"}
    if reference.get("kernel_selection", "native") == "stable_fla_v1":
        selections = [{"kernel": kernel, "cache_key": key, "config": config}
                      for (kernel, key), config in sorted(common_kernel_configs.items())]
        report.update(selected_kernel_configs=selections,
                      selected_kernel_configs_sha256=canonical_sha256(selections))
    if output is not None:
        _atomic_write_json(Path(output), report)
    return report


def verify_twofold(config):
    folds = config["folds"]
    if len(folds) != 2:
        raise ValueError("Expected exactly two reference teachers")
    source = read_source(config["full_train_file"])
    reports, fold_sources = [], []
    for fold in folds:
        reports.append(verify_reference(**{**fold, "mapping_file": config["mapping_file"]}))
        fold_sources.append(read_source(fold["score_file"]))
    if (reports[0]["teacher_train_sha256"] != reports[1]["score_file_sha256"]
            or reports[1]["teacher_train_sha256"] != reports[0]["score_file_sha256"]):
        raise ValueError("Two teachers did not score each other's training halves")
    combined = fold_sources[0]+fold_sources[1]
    if len(combined) != len(source):
        raise ValueError("Twofold union size differs from full NI source")
    for key in ("source_sample_id", "user_id"):
        if len({r[key] for r in combined}) != len(combined) or {r[key] for r in combined} != {r[key] for r in source}:
            raise ValueError("Twofold union has missing/duplicate source users/samples")
    expected = {r["source_sample_id"]: canonical_sha256(r) for r in source}
    if any(expected[r["source_sample_id"]] != canonical_sha256(r) for r in combined):
        raise ValueError("A fold row changed from the original full NI source")
    report = {"accepted": True, "twofold_union_exact": True, "rows": len(source),
        "full_train_sha256": sha256_file(Path(config["full_train_file"])),
        "mapping_sha256": sha256_file(Path(config["mapping_file"])), "folds": reports,
        "large_artifact_policy": "all scores and teacher weights verified remotely; compact report only"}
    _atomic_write_json(Path(config["output"]), report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--twofold-config", type=Path)
    parser.add_argument("--score-dirs", nargs="+")
    for name in ("teacher-train-file", "score-file", "mapping-file", "output", "adapter-path", "teacher-run-summary"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    if args.twofold_config:
        report = verify_twofold(json.loads(args.twofold_config.read_text()))
    else:
        values = {k: v for k, v in vars(args).items() if k != "twofold_config" and v is not None}
        required = {"score_dirs", "teacher_train_file", "score_file", "mapping_file", "output"}
        if not required <= set(values):
            parser.error("Provide --twofold-config, or score directories and all source/output paths")
        report = verify_reference(**values)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
