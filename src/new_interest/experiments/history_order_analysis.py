"""Independent paired analysis of the four fixed NI history-order conditions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from new_interest.data.build_history_order_ablation import DEV, sha256

ARMS = ("A", "B", "C", "D")
CUTOFFS = (1, 5, 10, 20)
COMPARISONS = (("B", "A"), ("D", "A"), ("C", "A"), ("D", "B"), ("D", "C"))
PRIMARY_COLUMNS = (("recall", 10), ("ndcg", 10), ("recall", 20), ("ndcg", 20))


def metric_key(kind, cutoff):
    return f"interest_macro_{kind}_at_{cutoff}"


def outcomes(records, predictions):
    if set(predictions) != set(ARMS):
        raise ValueError("Exactly A/B/C/D predictions required")
    if not records or len({str(r["impression_id"]) for r in records}) != len(records):
        raise ValueError("Empty or duplicated impressions")
    for arm, rows in predictions.items():
        if len(rows) != len(records):
            raise ValueError(f"Wrong row coverage: {arm}")
        for record, prediction in zip(records, rows):
            if str(record["impression_id"]) != str(prediction["impression_id"]):
                raise ValueError("Misaligned paired predictions")
            labels = prediction["ranked_labels"]
            if len(labels) != len(set(labels)):
                raise ValueError("Duplicate ranking label")
    users = sorted({str(r["user_id"]) for r in records})
    labels = sorted({str(t) for r in records for t in r["acceptable_subcategories"]})
    ui, ci = {u: i for i, u in enumerate(users)}, {c: i for i, c in enumerate(labels)}
    user_index, class_index, record_index = [], [], []
    ranks = {arm: [] for arm in ARMS}
    for i, row in enumerate(records):
        maps = {arm: {c: k for k, c in enumerate(predictions[arm][i]["ranked_labels"], 1)} for arm in ARMS}
        for label in row["acceptable_subcategories"]:
            user_index.append(ui[str(row["user_id"])])
            class_index.append(ci[str(label)])
            record_index.append(i)
            for arm in ARMS:
                ranks[arm].append(maps[arm].get(label, 21))
    values = {}
    for arm, rank in ranks.items():
        rank = np.asarray(rank)
        values[arm] = {}
        for k in CUTOFFS:
            hits = (rank <= k).astype(np.float64)
            values[arm][metric_key("recall", k)] = hits
            values[arm][metric_key("ndcg", k)] = hits / np.log2(rank + 1)
    return {"users": users, "labels": labels, "user_index": np.array(user_index),
            "class_index": np.array(class_index), "record_index": np.array(record_index), "values": values}


def aggregate(encoded, selected=None):
    ci = encoded["class_index"]
    use = np.ones(len(ci), dtype=bool) if selected is None else np.asarray(selected)[encoded["record_index"]]
    support = np.bincount(ci[use], minlength=len(encoded["labels"]))
    supported = support > 0
    if not supported.any():
        raise ValueError("No supported classes")
    result = {}
    for arm in ARMS:
        result[arm] = {key: float(np.mean(np.bincount(ci[use], weights=v[use], minlength=len(support))[supported] / support[supported]))
                       for key, v in encoded["values"][arm].items()}
    return {"supported_classes": int(supported.sum()), "target_occurrences": int(support.sum()),
            "support": {c: int(n) for c, n in zip(encoded["labels"], support) if n}, "metrics": result}


def paired_bootstrap(encoded, draws=2000, seed=42):
    rng = np.random.default_rng(seed)
    ci, ui = encoded["class_index"], encoded["user_index"]
    columns = [metric_key(*column) for column in PRIMARY_COLUMNS]
    differences = [(f"{a}-{b}", key, encoded["values"][a][key] - encoded["values"][b][key])
                   for a, b in COMPARISONS for key in columns]
    samples = np.empty((draws, len(differences)))
    for draw in range(draws):
        user_weights = rng.exponential(size=len(encoded["users"]))
        while (user_weights <= 0).any():
            bad = user_weights <= 0
            user_weights[bad] = rng.exponential(size=int(bad.sum()))
        weight = user_weights[ui]
        denominator = np.bincount(ci, weights=weight, minlength=len(encoded["labels"]))
        for j, (_, _, delta) in enumerate(differences):
            samples[draw, j] = np.mean(np.bincount(ci, weights=weight * delta, minlength=len(denominator)) / denominator)
    point = aggregate(encoded)["metrics"]
    report = {}
    for j, (name, key, _) in enumerate(differences):
        a, b = name.split("-")
        report.setdefault(name, {})[key] = {
            "difference_pp": 100 * (point[a][key] - point[b][key]),
            "bayesian_bootstrap_95_interval_pp": (100 * np.quantile(samples[:, j], [0.025, 0.975])).tolist(),
        }
    return {"method": "paired-user-positive-Exp(1)-Bayesian-bootstrap-fixed-class-support", "draws": draws,
            "seed": seed, "training_and_permutation_uncertainty_included": False, "comparisons": report}


def ranking_changes(predictions):
    report = {}
    for a, b in (("A", "B"), ("C", "D")):
        pairs = list(zip(predictions[a], predictions[b]))
        if not pairs:
            raise ValueError("No paired predictions")
        report[f"{a}/{b}"] = {
            "top1_changed_fraction": sum(x["ranked_labels"][0] != y["ranked_labels"][0] for x, y in pairs) / len(pairs),
            "mean_top10_overlap_fraction": math.fsum(len(set(x["ranked_labels"][:10]) & set(y["ranked_labels"][:10])) / 10 for x, y in pairs) / len(pairs),
        }
    return report


def analyze(project, root, baseline_root):
    contract = json.loads((project / "configs/experiments/history_order.json").read_text())
    records = pq.read_table(project / contract["original_data"] / DEV).to_pylist()
    predictions, metrics, sources = {}, {}, {}
    for arm in ARMS:
        folder = baseline_root if arm == "A" else root / "evaluations" / arm
        folder = folder / "real_eval/merged"
        predictions[arm] = pq.read_table(folder / "id_token_predictions.parquet").to_pylist()
        metrics[arm] = json.loads((folder / "metrics.json").read_text())
        sources[arm] = {"predictions_sha256": sha256(folder / "id_token_predictions.parquet"),
                        "metrics_sha256": sha256(folder / "metrics.json"), "evaluation_identity": metrics[arm]["evaluation_identity"]}
    enc = outcomes(records, predictions)
    full = aggregate(enc)
    if (len(records), len(enc["users"]), full["supported_classes"], full["target_occurrences"]) != (31392, 27082, 104, 37728):
        raise ValueError("Full Dev coverage differs")
    for arm in ARMS:
        for key, value in full["metrics"][arm].items():
            if abs(value - metrics[arm][key]) > 1e-12:
                raise ValueError(f"Independent macro recomputation differs: {arm}/{key}")
    long_history = [r["history_length"] >= 2 for r in records]
    if sum(long_history) != 30538:
        raise ValueError("Frozen history-length subset differs")
    result = {"status": "passed", "baseline_id": contract["baseline_id"], "source_identity": sources,
              "full_dev": {"records": len(records), "users": len(enc["users"]), **full},
              "history_length_ge_2": {"records": sum(long_history), **aggregate(enc, long_history)},
              "ranking_changes": ranking_changes(predictions), "uncertainty": paired_bootstrap(enc)}
    from new_interest.experiments.history_length_ablation import write
    write(root / "comparison.json", result)
    print("HISTORY_ORDER_ANALYSIS_VERIFIED", json.dumps(full["metrics"]), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    a = p.parse_args()
    analyze(a.project, a.root, a.baseline_root)


if __name__ == "__main__":
    main()
