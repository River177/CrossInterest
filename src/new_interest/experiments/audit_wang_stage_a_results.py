from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq


CORE_GENERATION_KEYS = (
    "samples",
    "match_rate",
    "valid_label_rate",
    "prefix_compliance_rate",
    "under_30_words_rate",
    "set_hit_at_1",
    "recall_at_1",
    "interest_macro_recall_at_1",
    "novelty_compliance_rate",
    "supported_interest_count",
    "per_class_recall_at_1",
    "per_class_support",
    "prediction_counts",
    "unique_generated_labels",
    "generation_entropy",
    "normalized_generation_entropy",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or actual.keys() != expected.keys():
            raise ValueError(f"{name}: mapping keys differ")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{name}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{name}: list lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_close(left, right, f"{name}[{index}]")
        return
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{name}: {actual!r} != {expected!r}")


def _ordered_rows(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    indices = [int(row["record_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"{path}: record indices are not contiguous and ordered")
    impression_ids = [str(row["impression_id"]) for row in rows]
    if len(set(impression_ids)) != len(impression_ids):
        raise ValueError(f"{path}: duplicate impression IDs")
    return rows


def recompute_generation(
    rows: Iterable[dict[str, Any]],
    *,
    labels: set[str],
    response_prefix: str,
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("Generation predictions are empty")
    target_support: Counter[str] = Counter()
    target_hits: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    prefix_hits = match_hits = set_hits = novelty_hits = length_hits = 0
    recall_values: list[float] = []
    for row in rows:
        output = str(row["output"])
        stripped = output.strip()
        has_prefix = stripped.startswith(response_prefix)
        label = stripped[len(response_prefix) :].strip() if has_prefix else None
        valid_label = label in labels if label is not None else False
        targets = {str(value) for value in row["acceptable_subcategories"]}
        history = {str(value) for value in row["history_subcategories"]}
        if not targets:
            raise ValueError("Generation row has no acceptable target")
        hit = label in targets if label is not None else False
        recall = 1.0 / len(targets) if hit else 0.0
        novelty = bool(valid_label and label not in history)
        prefix_hits += int(has_prefix)
        match_hits += int(valid_label)
        set_hits += int(hit)
        novelty_hits += int(novelty)
        length_hits += int(len(stripped.split()) < 30)
        recall_values.append(recall)
        if label is not None:
            prediction_counts[label] += 1
        for target in targets:
            target_support[target] += 1
            if label == target:
                target_hits[target] += 1
        expected_row = {
            "stripped_output": stripped,
            "has_prefix": has_prefix,
            "parsed_label": label,
            "valid_label": valid_label,
            "under_30_words": len(stripped.split()) < 30,
            "set_hit_at_1": hit,
            "recall_at_1": recall,
            "novelty_compliant": novelty,
        }
        for key, expected in expected_row.items():
            _assert_close(row[key], expected, f"generation row {row['record_index']}.{key}")
    total = len(rows)
    per_class_recall = {
        target: target_hits[target] / support
        for target, support in sorted(target_support.items())
    }
    probabilities = [count / total for count in prediction_counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalizer = math.log(len(labels)) if len(labels) > 1 else 1.0
    return {
        "samples": total,
        "match_rate": match_hits / total,
        "valid_label_rate": match_hits / total,
        "prefix_compliance_rate": prefix_hits / total,
        "under_30_words_rate": length_hits / total,
        "set_hit_at_1": set_hits / total,
        "recall_at_1": mean(recall_values),
        "interest_macro_recall_at_1": mean(per_class_recall.values()),
        "novelty_compliance_rate": novelty_hits / total,
        "supported_interest_count": len(target_support),
        "per_class_recall_at_1": per_class_recall,
        "per_class_support": dict(sorted(target_support.items())),
        "prediction_counts": dict(
            sorted(prediction_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "unique_generated_labels": sum(label in labels for label in prediction_counts),
        "generation_entropy": entropy,
        "normalized_generation_entropy": entropy / normalizer,
    }


def recompute_ranking(
    rows: Iterable[dict[str, Any]],
    *,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("Ranking predictions are empty")
    metric_values: dict[str, list[float]] = defaultdict(list)
    class_support: Counter[str] = Counter()
    class_retrieved = {cutoff: Counter() for cutoff in cutoffs}
    class_gain = {cutoff: Counter() for cutoff in cutoffs}
    top1_counts: Counter[str] = Counter()
    for row in rows:
        labels = [str(value) for value in row["ranked_labels"]]
        scores = [float(value) for value in row["ranked_scores"]]
        targets = {str(value) for value in row["acceptable_subcategories"]}
        if len(labels) != len(set(labels)) or len(labels) != len(scores):
            raise ValueError("Invalid ranking shape or duplicate labels")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Non-finite ranking score")
        if labels:
            top1_counts[labels[0]] += 1
        for target in targets:
            class_support[target] += 1
        first_rank = next(
            (index for index, label in enumerate(labels, start=1) if label in targets),
            None,
        )
        row_metrics: dict[str, float] = {}
        for cutoff in cutoffs:
            top_k = labels[:cutoff]
            retrieved = targets.intersection(top_k)
            recall = len(retrieved) / len(targets)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, label in enumerate(top_k, start=1)
                if label in targets
            )
            ideal_hits = min(cutoff, len(targets))
            idcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )
            row_metrics[f"recall_at_{cutoff}"] = recall
            row_metrics[f"ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0
            row_metrics[f"hit_at_{cutoff}"] = float(bool(retrieved))
            for target in targets:
                if target in top_k:
                    class_retrieved[cutoff][target] += 1
                    rank = labels.index(target) + 1
                    class_gain[cutoff][target] += 1.0 / math.log2(rank + 1)
        row_metrics[f"mrr_at_{cutoffs[-1]}"] = (
            1.0 / first_rank
            if first_rank is not None and first_rank <= cutoffs[-1]
            else 0.0
        )
        row_metrics["ranking_length"] = float(len(labels))
        for key, value in row_metrics.items():
            _assert_close(row[key], value, f"ranking row {row['record_index']}.{key}")
            metric_values[key].append(value)
    per_class_recall = {
        str(cutoff): {
            target: class_retrieved[cutoff][target] / support
            for target, support in sorted(class_support.items())
        }
        for cutoff in cutoffs
    }
    per_class_ndcg = {
        str(cutoff): {
            target: class_gain[cutoff][target] / support
            for target, support in sorted(class_support.items())
        }
        for cutoff in cutoffs
    }
    total = len(rows)
    top1_entropy = -sum(
        (count / total) * math.log(count / total)
        for count in top1_counts.values()
    )
    label_space_size = len(class_support)
    metrics: dict[str, Any] = {
        "samples": total,
        **{key: mean(values) for key, values in sorted(metric_values.items())},
        "supported_interest_count": label_space_size,
        "per_class_recall": per_class_recall,
        "per_class_ndcg": per_class_ndcg,
        "per_class_support": dict(sorted(class_support.items())),
        "unique_top1_labels": len(top1_counts),
        "max_top1_label_count": max(top1_counts.values(), default=0),
        "max_top1_label_share": max(top1_counts.values(), default=0) / total,
        "top1_label_entropy": top1_entropy,
        "normalized_top1_label_entropy": (
            top1_entropy / math.log(label_space_size) if label_space_size > 1 else 0.0
        ),
        "top1_label_counts": dict(
            sorted(top1_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    for cutoff in cutoffs:
        metrics[f"interest_macro_recall_at_{cutoff}"] = mean(
            per_class_recall[str(cutoff)].values()
        )
        metrics[f"interest_macro_ndcg_at_{cutoff}"] = mean(
            per_class_ndcg[str(cutoff)].values()
        )
    return metrics


def _audit_job(
    root: Path,
    *,
    label_registry: Mapping[str, set[str]],
) -> dict[str, Any]:
    marker = _load_json(root / "full-validation_success.json")
    contract = _load_json(root / "run_contract.json")
    _assert_close(marker["contract"], contract, f"{root.name}.contract")
    if contract["condition"] != "balanced" or contract["mode"] != "full-validation":
        raise ValueError(f"{root}: unexpected experiment condition")
    for relative, expected_hash in marker["artifact_sha256"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{root}: missing artifact {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{root}: hash mismatch for {relative}")
    reload_audit = _load_json(root / "text_reload_verification.json")
    if not reload_audit["passed"] or reload_audit["max_absolute_error"] > reload_audit["tolerance"]:
        raise ValueError(f"{root}: reload verification failed")
    state = _load_json(root / "adapter_state_verification.json")["adapter_state_integrity"]
    if not state["all_tensors_exact"] or state["saved_state_sha256"] != state["loaded_state_sha256"]:
        raise ValueError(f"{root}: adapter state verification failed")
    if state["missing_keys"] or state["unexpected_keys"] or state["mismatched_keys"]:
        raise ValueError(f"{root}: adapter state keys differ")
    summary = _load_json(root / "sft/run_summary.json")
    if int(summary["train_samples"]) != int(contract["train_rows"]):
        raise ValueError(f"{root}: training row count mismatch")
    losses = [float(value["loss"]) for value in summary["loss_history"]]
    window = min(5, len(losses) // 2)
    if window == 0 or mean(losses[-window:]) >= mean(losses[:window]):
        raise ValueError(f"{root}: training loss did not decrease")

    views: dict[str, Any] = {}
    for generation_dir in sorted((root / "eval").glob("generation-*")):
        prediction_path = generation_dir / "generation_predictions.parquet"
        metrics_path = generation_dir / "metrics.json"
        if not prediction_path.is_file() or not metrics_path.is_file():
            continue
        published = _load_json(metrics_path)
        rows = _ordered_rows(prediction_path)
        labels_sha256 = str(published["labels_sha256"])
        if labels_sha256 not in label_registry:
            raise ValueError(
                f"{generation_dir}: unknown labels hash {labels_sha256}"
            )
        labels = label_registry[labels_sha256]
        if "labels" in published:
            _assert_close(
                sorted(str(value) for value in published["labels"]),
                sorted(labels),
                f"{generation_dir}.labels",
            )
        recomputed = recompute_generation(
            rows,
            labels=labels,
            response_prefix=str(published["response_prefix"]),
        )
        for key in CORE_GENERATION_KEYS:
            _assert_close(published[key], recomputed[key], f"{generation_dir}.{key}")
        views[generation_dir.name] = {
            key: recomputed[key]
            for key in (
                "samples",
                "match_rate",
                "set_hit_at_1",
                "recall_at_1",
                "interest_macro_recall_at_1",
                "novelty_compliance_rate",
                "unique_generated_labels",
                "normalized_generation_entropy",
            )
        }
    for ranking_dir in sorted((root / "eval").glob("ranking-*")):
        prediction_path = ranking_dir / "text_label_predictions.parquet"
        metrics_path = ranking_dir / "metrics.json"
        if not prediction_path.is_file() or not metrics_path.is_file():
            continue
        published = _load_json(metrics_path)
        rows = _ordered_rows(prediction_path)
        cutoffs = tuple(int(value) for value in published["cutoffs"])
        recomputed = recompute_ranking(rows, cutoffs=cutoffs)
        for key, value in recomputed.items():
            _assert_close(published[key], value, f"{ranking_dir}.{key}")
        views[ranking_dir.name] = {
            key: recomputed[key]
            for key in (
                "samples",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "ndcg_at_5",
                "ndcg_at_10",
                "interest_macro_recall_at_10",
                "interest_macro_ndcg_at_10",
                "unique_top1_labels",
                "normalized_top1_label_entropy",
            )
        }
    return {
        "model_name": contract["model_name"],
        "model_revision": contract["model_revision"],
        "dataset": contract["dataset"],
        "condition": contract["condition"],
        "train_rows": contract["train_rows"],
        "artifact_hash_count": len(marker["artifact_sha256"]),
        "reload_max_absolute_error": reload_audit["max_absolute_error"],
        "adapter_tensor_count": state["tensor_count"],
        "adapter_total_numel": state["total_numel"],
        "loss_first_window": mean(losses[:window]),
        "loss_last_window": mean(losses[-window:]),
        "views": views,
    }


def _load_label_registry(data_root: Path) -> dict[str, set[str]]:
    registry: dict[str, set[str]] = {}
    for path in sorted(data_root.rglob("labels_*.parquet")):
        digest = _sha256(path)
        labels = {
            str(value)
            for value in pq.read_table(
                path,
                columns=["target_subcategory"],
            )["target_subcategory"].to_pylist()
        }
        if digest in registry and registry[digest] != labels:
            raise ValueError(f"Conflicting label files for hash {digest}")
        registry[digest] = labels
    if not registry:
        raise ValueError(f"No labels_*.parquet files under {data_root}")
    return registry


def audit_readback(
    readback_root: Path | str,
    data_root: Path | str,
) -> dict[str, Any]:
    readback_root = Path(readback_root)
    label_registry = _load_label_registry(Path(data_root))
    markers = sorted(readback_root.rglob("full-validation_success.json"))
    if len(markers) != 6:
        raise ValueError(f"Expected six success markers, found {len(markers)}")
    jobs = [
        _audit_job(marker.parent, label_registry=label_registry)
        for marker in markers
    ]
    identities = {(job["model_name"], job["dataset"]) for job in jobs}
    if len(identities) != 6:
        raise ValueError("Model/dataset result identities are not unique")
    return {
        "audit_version": "wang-stage-a-local-readback-v1",
        "passed": True,
        "job_count": len(jobs),
        "model_dataset_identities": [list(value) for value in sorted(identities)],
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Wang Stage-A full readback")
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_readback(args.readback_root, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
