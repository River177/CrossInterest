"""Paired user-cluster uncertainty from the three accepted remote student arms.

No model is loaded or inference launched. Predictions stay on the worker's
remote filesystem; the sole output is a compact metrics/provenance report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np
import pyarrow.parquet as pq
import scipy
from scipy import sparse

from new_interest.experiments.sid_relation_bias import metrics as repository_metrics
from new_interest.experiments.rho_reference_recovery import _overlap


VERSION = "rho-three-student-paired-user-bootstrap-v1"
ARMS = ("uniform", "static", "dynamic")
METRICS = ("macro_recall_at_10", "macro_recall_at_20", "macro_ndcg_at_10", "macro_ndcg_at_20")
CONTRASTS = (("static", "uniform"), ("dynamic", "uniform"), ("dynamic", "static"))
PAIRED_FIELDS = ("source_sample_id", "user_id", "impression_id", "input_sha256", "targets", "history_subcategories")
RARE_CLASS = "Artificial Intelligence & Robotics"
BOOTSTRAP_RULE = {
    "unit": "user", "sampling": "sample N user IDs with replacement from the N observed users in each split",
    "cluster_contents": "all impressions and target occurrences of each selected user share its multiplicity",
    "class_universe": "fixed before bootstrap: validation support>=20; Dev observed supported classes",
    "missing_class_rule": "in each draw, average paired class differences over originally eligible classes with positive drawn support; identical class set for all arms",
    "empty_draw_rule": "omit draws with zero eligible classes from percentile intervals; count them; do not replace draws",
    "draw_sharing": "all three contrasts use exactly the same draws within a split",
    "interval": "two-sided 95% percentile interval, numpy quantile linear interpolation",
    "seed_per_split": 42, "draws": 2000,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def _source(path, names):
    fields = pq.ParquetFile(path).schema_arrow.names
    if set(PAIRED_FIELDS).difference(fields):
        raise ValueError("Heldout source is missing paired identity fields")
    columns = [*PAIRED_FIELDS, *(["prompt"] if "prompt" in fields else [])]
    rows = pq.read_table(path, columns=columns).to_pylist()
    if not rows:
        raise ValueError("Empty heldout source")
    samples, impressions = set(), set()
    for row in rows:
        for field in ("source_sample_id", "user_id", "impression_id", "input_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("Missing heldout source identity: " + field)
        if row["source_sample_id"] in samples or row["impression_id"] in impressions:
            raise ValueError("Heldout source repeats a source ID or impression")
        samples.add(row["source_sample_id"])
        impressions.add(row["impression_id"])
        targets = row.get("targets")
        if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets) or not set(targets) <= set(names):
            raise ValueError("Heldout targets must be unique nonempty known mapped classes")
        if not isinstance(row.get("history_subcategories"), list):
            raise ValueError("Heldout historical category identity is missing")
        if "prompt" in row and hashlib.sha256(row["prompt"].encode()).hexdigest() != row["input_sha256"]:
            raise ValueError("Heldout prompt differs from its input SHA-256")
    return rows


def _validate_predictions(rows, source, names):
    if len(rows) != len(source):
        raise ValueError("Prediction coverage differs from original heldout source")
    if [row.get("record_index") for row in rows] != list(range(len(source))):
        raise ValueError("Prediction indices must be the exact complete source order")
    for prediction, original in zip(rows, source):
        if any(prediction.get(field) != original[field] for field in PAIRED_FIELDS):
            raise ValueError("Prediction source/user/target/input identity mismatch")
        values = prediction.get("scores")
        if (not isinstance(values, list) or len(values) != len(names)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
            raise ValueError("Prediction score vector cardinality or finiteness differs")
        ranking = prediction.get("ranking")
        if not isinstance(ranking, list) or any(type(index) is not int for index in ranking):
            raise ValueError("Prediction ranking must contain integer interest indices")
        if ranking != sorted(range(len(names)), key=lambda index: (-values[index], index)):
            raise ValueError("Prediction ranking differs from exact stable score ordering")


def _compare_metrics(actual, expected):
    for key in ("records", "users", "target_occurrences", "supported_classes"):
        if actual.get(key) != expected[key]:
            raise ValueError("Accepted metric coverage/macro denominator mismatch: " + key)
    for key in METRICS:
        if not isinstance(actual.get(key), (int, float)) or not math.isclose(actual[key], expected[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Accepted macro metric recomputation differs: " + key)
    if set(actual.get("per_class", {})) != set(expected["per_class"]):
        raise ValueError("Accepted per-class metric coverage differs")
    for name, values in expected["per_class"].items():
        for key, value in values.items():
            got = actual["per_class"][name].get(key)
            if key == "support":
                if got != value:
                    raise ValueError("Accepted class support differs")
            elif not isinstance(got, (int, float)) or not math.isclose(got, value, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Accepted class metric differs")


def _verify_kernel_receipt(directory, done, identity, common_configs):
    path = directory / "kernel_selection.json"
    if sha(path) != done.get("kernel_selection_sha256"):
        raise ValueError("Heldout kernel receipt hash mismatch")
    receipt = read(path)
    if (identity.get("kernel_selection_protocol") != "fla-valid-candidates-stable-sha256-v1"
            or receipt.get("evaluation_identity") != canonical(identity)
            or any(receipt.get(key) != identity.get(key) for key in
                   ("kernel_selection", "kernel_selection_protocol", "kernel_selector_sha256"))):
        raise ValueError("Heldout kernel protocol/evaluation identity mismatch")
    snapshots = receipt.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots or canonical(snapshots) != receipt.get("snapshots_sha256"):
        raise ValueError("Heldout kernel snapshot content hash mismatch")
    stages = set()
    for snapshot in snapshots:
        if not isinstance(snapshot.get("stage"), str) or not isinstance(snapshot.get("kernels"), dict) or not snapshot["kernels"]:
            raise ValueError("Heldout kernel snapshot has no selected configurations")
        stages.add(snapshot["stage"])
        for name, kernel in snapshot["kernels"].items():
            if (not isinstance(name, str) or not name or not isinstance(kernel.get("cache"), dict)
                    or not kernel["cache"] or kernel.get("cache_results") is not False):
                raise ValueError("Heldout kernel cache or timing-write policy is invalid")
            for key, config in kernel["cache"].items():
                if not isinstance(key, str) or not key or not isinstance(config, str) or not config.strip():
                    raise ValueError("Heldout selected kernel configuration must be a nonempty string")
                pair = (name, key)
                if pair in common_configs and common_configs[pair] != config:
                    raise ValueError("Cross-arm heldout kernel configuration differs for identical cache key: " + repr(pair))
                common_configs[pair] = config
    expected = {f"rows:{start}:{min(start + 256, done['rows'])}" for start in range(0, done["rows"], 256)}
    if not expected <= stages:
        raise ValueError("Heldout kernel snapshots do not cover all evaluation chunks")
    return {"directory": str(directory), "kernel_selection_sha256": done["kernel_selection_sha256"],
            "snapshots_sha256": receipt["snapshots_sha256"], "snapshot_count": len(snapshots)}


def _read_arm_split(worker, arm, split, source_path, source, names, mapping_sha, validation_min_support, common_kernel_configs):
    report = worker["evaluations"][split]
    if (report.get("accepted") is not True or report.get("split") != split or report.get("limit") is not None
            or report.get("rows") != len(source) or report.get("source_sha256") != sha(source_path)
            or report.get("protocol_sha256") != worker["identity"]["protocol_sha256"]
            or report.get("kernel_configs_consistent") is not True):
        raise ValueError("Heldout acceptance source/protocol/coverage mismatch")
    prediction_file = Path(report["prediction_file"])
    if sha(prediction_file) != report["prediction_sha256"]:
        raise ValueError("Accepted prediction file hash mismatch")
    if read(prediction_file.parent / "evaluation_acceptance.json") != report:
        raise ValueError("Remote heldout acceptance differs from the final three-arm report")
    metrics_file = prediction_file.parent / "metrics.json"
    if sha(metrics_file) != report["metrics_sha256"] or read(metrics_file) != report["metrics"]:
        raise ValueError("Accepted metrics file hash/content mismatch")
    artifacts = report.get("artifacts", [])
    if len(artifacts) != 4 or len({item["directory"] for item in artifacts}) != 4:
        raise ValueError("Heldout acceptance lacks four distinct source shards")
    identities, kernel_receipts = [], []
    for shard, artifact in enumerate(artifacts):
        directory = Path(artifact["directory"])
        if sha(directory / "complete.json") != artifact["complete_sha256"]:
            raise ValueError("Heldout shard completion hash mismatch")
        done = read(directory / "complete.json")
        identity = done["identity"]
        expected = {"role": "heldout_interest_ranking", "split": split, "input_sha256": sha(source_path),
            "protocol_sha256": worker["identity"]["protocol_sha256"], "runtime_sha256": worker["identity"]["runtime_sha256"],
            "mapping_sha256": mapping_sha, "adapter_sha256": report["adapter_sha256"],
            "kernel_selection": "stable_fla_v1", "projection": "native-effective-head-FP32",
            "shard": shard, "shards": 4, "limit": None, "batch_size": 4}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Heldout shard source/model/mapping/kernel identity mismatch")
        if (canonical(identity["adapter_files"]) != identity["adapter_sha256"]
                or identity["adapter_files"].get("adapter_model.safetensors") != worker["training"]["full"]["adapter_sha256"]):
            raise ValueError("Evaluated adapter does not match the accepted trained weights")
        if done["rows"] != len(range(shard, len(source), 4)):
            raise ValueError("Heldout shard row coverage differs")
        kernel_receipts.append(_verify_kernel_receipt(directory, done, identity, common_kernel_configs))
        identities.append({key: value for key, value in identity.items() if key not in ("shard", "adapter_files", "adapter_sha256")})
    if any(value != identities[0] for value in identities):
        raise ValueError("Evaluation shards used inconsistent source or inference identities")
    rows = pq.read_table(prediction_file).to_pylist()
    _validate_predictions(rows, source, names)
    recomputed = repository_metrics(rows, names)
    _compare_metrics(report["metrics"], recomputed)
    main = repository_metrics(rows, names, min_support=validation_min_support) if split == "validation" else recomputed
    if split == "validation":
        _compare_metrics(report["metrics"].get("main_107", {}), main)
    return rows, {"all_supported": recomputed, "main": main}, {"arm": arm, "prediction_file": str(prediction_file),
        "prediction_sha256": report["prediction_sha256"], "metrics_sha256": report["metrics_sha256"],
        "source_sha256": report["source_sha256"], "adapter_sha256": report["adapter_sha256"],
        "kernel_receipts": kernel_receipts,
        "inference_identity_without_adapter": identities[0]}


def paired_user_bootstrap(rows_by_arm, names, eligible_classes, *, draws=2000, seed=42, batch_size=64):
    """Use sparse class-by-user contributions and shared user multiplicities.

    This kernel returns uncertainty only for paired differences. Per-arm point
    means are recomputed by the repository metrics function in the caller.
    """
    if set(rows_by_arm) != set(ARMS) or type(draws) is not int or draws <= 0 or type(seed) is not int:
        raise ValueError("Bootstrap requires all three arms and positive draw count")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Bootstrap batch size must be positive")
    base = rows_by_arm["uniform"]
    if not base:
        raise ValueError("Bootstrap source is empty")
    for arm in ARMS[1:]:
        if len(rows_by_arm[arm]) != len(base) or any(
            any(left.get(field) != right.get(field) for field in ("record_index", *PAIRED_FIELDS))
            for left, right in zip(base, rows_by_arm[arm])):
            raise ValueError("Bootstrap arms do not contain exactly paired user/source/target rows")
    classes = list(eligible_classes)
    if not classes or len(set(classes)) != len(classes) or not set(classes) <= set(names):
        raise ValueError("Bootstrap eligible class universe is invalid")
    support = Counter(target for row in base for target in row["targets"])
    if any(support[name] <= 0 for name in classes):
        raise ValueError("Bootstrap cannot include originally unsupported classes")
    users = sorted({row["user_id"] for row in base})
    user_index, class_index, name_index = ({name: i for i, name in enumerate(values)} for values in (users, classes, names))
    coordinates, who, values = [], [], {arm: [] for arm in ARMS}
    for row_index, row in enumerate(base):
        positions = {arm: {index: rank + 1 for rank, index in enumerate(rows_by_arm[arm][row_index]["ranking"])} for arm in ARMS}
        for target in row["targets"]:
            if target not in class_index:
                continue
            coordinates.append(class_index[target])
            who.append(user_index[row["user_id"]])
            for arm in ARMS:
                rank = positions[arm][name_index[target]]
                discount = 1.0 / math.log2(rank + 1)
                values[arm].append([float(rank <= 10), float(rank <= 20), discount if rank <= 10 else 0., discount if rank <= 20 else 0.])
    coordinates, who = np.asarray(coordinates), np.asarray(who)
    shape = (len(classes), len(users))
    supports = sparse.coo_matrix((np.ones(len(who)), (coordinates, who)), shape=shape).tocsr()
    values = {arm: np.asarray(array, dtype=np.float64) for arm, array in values.items()}
    matrices = {}
    for arm in ("static", "dynamic"):
        delta = values[arm] - values["uniform"]
        matrices[arm] = []
        for metric in range(len(METRICS)):
            matrix = sparse.coo_matrix((delta[:, metric], (coordinates, who)), shape=shape).tocsr()
            matrix.eliminate_zeros()
            matrices[arm].append(matrix)
    denominator = np.asarray(supports.sum(axis=1)).ravel()
    point = {arm: np.array([np.mean(np.asarray(matrix.sum(axis=1)).ravel() / denominator)
                           for matrix in matrices[arm]]) for arm in matrices}
    point["dynamic-static"] = point["dynamic"] - point["static"]
    generated = {arm: [] for arm in ("static", "dynamic", "dynamic-static")}
    rng, draw_digest = np.random.default_rng(seed), hashlib.sha256()
    class_counts = []
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws - start)
        sampled = rng.integers(0, len(users), size=(count, len(users)), dtype=np.int64)
        draw_digest.update(sampled.astype("<i8", copy=False).tobytes())
        weights = np.stack([np.bincount(indices, minlength=len(users)) for indices in sampled], axis=1).astype(np.float64)
        den = supports @ weights
        valid = den > 0
        valid_counts = valid.sum(axis=0)
        class_counts.extend(valid_counts.tolist())
        usable = valid_counts > 0
        means = {}
        for arm, matrices_for_arm in matrices.items():
            scores = []
            for matrix in matrices_for_arm:
                numerator = matrix @ weights
                class_means = np.divide(numerator, den, out=np.zeros_like(numerator), where=valid)
                scores.append(class_means[:, usable].sum(axis=0) / valid_counts[usable])
            means[arm] = np.stack(scores, axis=1)
            generated[arm].append(means[arm])
        generated["dynamic-static"].append(means["dynamic"] - means["static"])
    usable_draws = sum(count > 0 for count in class_counts)
    if usable_draws == 0:
        raise ValueError("Every bootstrap draw lost all eligible classes")
    comparisons = {}
    for newer, older in CONTRASTS:
        key = newer if older == "uniform" else "dynamic-static"
        samples = np.concatenate(generated[key], axis=0)
        intervals = np.quantile(samples, [.025, .975], axis=0, method="linear")
        comparisons[newer + "_minus_" + older] = {
            "delta": dict(zip(METRICS, point[key].tolist())),
            "delta_percentage_points": dict(zip(METRICS, (point[key] * 100).tolist())),
            "ci95": {metric: [float(intervals[0, i]), float(intervals[1, i])] for i, metric in enumerate(METRICS)},
            "ci95_percentage_points": {metric: [float(intervals[0, i] * 100), float(intervals[1, i] * 100)]
                                        for i, metric in enumerate(METRICS)}}
    return {"comparisons": comparisons, "bootstrap": {**BOOTSTRAP_RULE, "draws": draws, "seed_per_split": seed,
        "eligible_classes": classes, "eligible_class_count": len(classes), "observed_users": len(users),
        "sampled_users_per_draw": len(users), "usable_draws": usable_draws,
        "draws_missing_any_eligible_class": sum(count < len(classes) for count in class_counts),
        "zero_supported_class_draws": draws - usable_draws,
        "draw_supported_class_count_range": [min(class_counts), max(class_counts)],
        "draw_indices_sha256": draw_digest.hexdigest(), "user_order_sha256": canonical(users),
        "generator": "numpy.default_rng(PCG64); integers over sorted observed users; seed reset per split"}}


def compare_students(acceptance_path, *, expected_acceptance_sha256, validation_file, dev_file, mapping_file,
                     output_file, bootstrap_draws=2000, seed=42, production=True, validation_min_support=20):
    acceptance_path, output_file, mapping_file = Path(acceptance_path), Path(output_file), Path(mapping_file)
    source_files = {"validation": Path(validation_file), "dev": Path(dev_file)}
    if sha(acceptance_path) != expected_acceptance_sha256:
        raise ValueError("Final three-arm acceptance hash mismatch")
    if production and (bootstrap_draws, seed, validation_min_support) != (2000, 42, 20):
        raise ValueError("Frozen production comparison requires 2000 draws, seed42, validation support>=20")
    if type(validation_min_support) is not int or validation_min_support < 1:
        raise ValueError("Validation minimum class support must be positive")
    if output_file.exists():
        raise FileExistsError("Comparison output is immutable; use a fresh output path")
    accepted = read(acceptance_path)
    if (accepted.get("passed") is not True or set(accepted.get("arms", {})) != set(ARMS)
            or accepted.get("same_initial_state") is not True or accepted.get("same_reference_scores") is not True):
        raise ValueError("A complete three-arm, same-initialization/reference acceptance is required")
    workers = {arm: accepted["arms"][arm]["worker"] for arm in ARMS}
    for arm, worker in workers.items():
        if (worker.get("passed") is not True or worker["identity"].get("arm") != arm
                or worker["identity"].get("phase") != "full" or set(worker["evaluations"]) != set(source_files)):
            raise ValueError("Wrong student arm, phase or evaluation family")
    shared_fields = ("runtime_sha256", "protocol_sha256", "inputs_sha256", "reference_acceptance_sha256")
    if any(len({worker["identity"][field] for worker in workers.values()}) != 1 for field in shared_fields):
        raise ValueError("Student experiment identities differ across arms")
    if (len({worker["training"]["full"]["initial_state_sha256"] for worker in workers.values()}) != 1
            or len({worker["training"]["full"]["student_summary"]["reference_file_sha256"] for worker in workers.values()}) != 1):
        raise ValueError("Actual initial model or reference-score identities differ")
    if production and any((worker["training"]["full"]["rows"], worker["training"]["full"]["steps"],
                           worker["training"]["full"]["epochs"]) != (21600, 2025, 3) for worker in workers.values()):
        raise ValueError("Student training endpoint differs from the frozen three-arm plan")
    mapping = sorted(pq.read_table(mapping_file).to_pylist(), key=lambda row: row["interest_index"])
    names = [row["target_subcategory"] for row in mapping]
    if (not names or len(set(names)) != len(names) or [row["interest_index"] for row in mapping] != list(range(len(names)))
            or any(row["interest_token"] != f'<interest_{row["interest_index"]:03d}>' for row in mapping)
            or production and len(names) != 108):
        raise ValueError("Mapping vocabulary/order differs")
    protected = [acceptance_path, mapping_file, *source_files.values()]
    protected.extend(Path(worker["evaluations"][split]["prediction_file"]).parent for worker in workers.values() for split in source_files)
    if any(_overlap(output_file, path) for path in protected):
        raise ValueError("Comparison output overlaps immutable source/evaluation artifacts")
    results, provenance = {}, {}
    for split, source_file in source_files.items():
        source = _source(source_file, names)
        support = Counter(target for row in source for target in row["targets"])
        threshold = validation_min_support if split == "validation" else 1
        eligible = [name for name in names if support[name] >= threshold]
        if not eligible:
            raise ValueError("No eligible classes for the fixed macro denominator")
        if production:
            expected = (5354, 5354, 5354, 107) if split == "validation" else (31392, 27082, 37728, 104)
            observed = (len(source), len({row["user_id"] for row in source}), sum(support.values()), len(eligible))
            if observed != expected:
                raise ValueError("Heldout source size or fixed macro denominator differs")
            if split == "validation" and (support[RARE_CLASS] != 4 or any(support[name] != 50 for name in eligible)
                                          or set(support) - set(eligible) != {RARE_CLASS}):
                raise ValueError("Validation must preserve 107 classes by 50 plus four rare AI examples")
        predictions, means, receipts, common_kernel_configs = {}, {}, [], {}
        for arm in ARMS:
            rows, recomputed, receipt = _read_arm_split(workers[arm], arm, split, source_file, source, names,
                sha(mapping_file), validation_min_support, common_kernel_configs)
            predictions[arm], means[arm] = rows, recomputed
            receipts.append(receipt)
        if any(receipt["inference_identity_without_adapter"] != receipts[0]["inference_identity_without_adapter"] for receipt in receipts):
            raise ValueError("Student arms did not share the same heldout inference/source protocol")
        paired = paired_user_bootstrap(predictions, names, eligible, draws=bootstrap_draws, seed=seed)
        for newer, older in CONTRASTS:
            contrast = paired["comparisons"][newer + "_minus_" + older]
            for metric in METRICS:
                delta = means[newer]["main"][metric] - means[older]["main"][metric]
                if not math.isclose(delta, contrast["delta"][metric], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Paired contribution denominator differs from repository metrics")
        rare = None
        if split == "validation" and support[RARE_CLASS]:
            rare = {"class": RARE_CLASS, "support": support[RARE_CLASS],
                    "users": len({row["user_id"] for row in source if RARE_CLASS in row["targets"]}),
                    "included_in_primary_macro": RARE_CLASS in eligible,
                    "point_metrics": {arm: means[arm]["all_supported"]["per_class"][RARE_CLASS] for arm in ARMS},
                    "intervals_computed": False}
        results[split] = {"records": len(source), "users": len({row["user_id"] for row in source}),
            "target_occurrences": sum(support.values()), "main_macro_classes": len(eligible),
            "main_class_names": eligible, "main_support_threshold": threshold,
            "excluded_low_support_classes": {name: count for name, count in sorted(support.items()) if name not in eligible},
            "means": {arm: {metric: means[arm]["main"][metric] for metric in METRICS} for arm in ARMS},
            "all_supported_means": {arm: {metric: means[arm]["all_supported"][metric] for metric in METRICS} for arm in ARMS},
            "rare_validation_class": rare, "cross_arm_kernel_configs_consistent": True,
            "kernel_cache_keys_checked": len(common_kernel_configs),
            "selected_kernel_configs_sha256": canonical([
                {"kernel": name, "cache_key": key, "config": value} for (name, key), value in sorted(common_kernel_configs.items())]),
            **paired}
        provenance[split] = {"source_file": str(source_file), "source_sha256": sha(source_file),
                             "arms": receipts, "class_support": dict(sorted(support.items()))}
    report = {"passed": True, "version": VERSION, "acceptance_file": str(acceptance_path),
        "acceptance_sha256": expected_acceptance_sha256, "mapping_file": str(mapping_file), "mapping_sha256": sha(mapping_file),
        "comparison_code_sha256": sha(Path(__file__)),
        "metric_implementation_sha256": sha(Path(repository_metrics.__code__.co_filename)),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "metric_definition": "repository sid_relation_bias.metrics: per-target recall/reciprocal-log2 rank discount, then unweighted mean over eligible classes",
        "bootstrap_selection_rules": {**BOOTSTRAP_RULE, "draws": bootstrap_draws, "seed_per_split": seed},
        "primary_comparison": {"split": "validation", "contrast": "dynamic_minus_static", "metric": "macro_ndcg_at_20",
                               "dev_role": "consistency check under the same fixed contrast and metric"},
        "family": {"contrasts": [newer + "_minus_" + older for newer, older in CONTRASTS],
                   "metrics": list(METRICS), "splits": list(source_files), "intervals_are_marginal_not_multiplicity_adjusted": True,
                   "interpretation": "user-sampling uncertainty conditional on one trained seed; no causal noise-truth claim or training-seed uncertainty"},
        "results": results, "provenance": provenance,
        "verification_scope": "accepted prediction/source/metric/completion/kernel hashes; cross-arm identical-cache-key kernel consistency; every paired row, target and stable rank; repository macro recomputation; no model inference",
        "large_artifact_policy": "remote prediction reads only; compact comparison output; no downloads"}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(output_file.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if output_file.exists():
        temporary.unlink()
        raise FileExistsError(output_file)
    temporary.replace(output_file)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_students(args.acceptance, expected_acceptance_sha256=args.acceptance_sha256,
        validation_file=args.validation_file, dev_file=args.dev_file, mapping_file=args.mapping_file, output_file=args.output)
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "primary_comparison": result["primary_comparison"]}))


if __name__ == "__main__":
    main()
