from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_future_interest import TRACKS
from new_interest.evaluation.ranking_metrics import (
    paired_bootstrap_metric_delta,
)
from new_interest.infrastructure.paths import portable_path


METRICS_JSON = "future_novel_interest_baselines_metrics.json"
METRICS_CSV = "future_novel_interest_baselines_metrics.csv"
REPORT_MARKDOWN = "future_novel_interest_baselines_generated.md"
REPORT_FILES = (METRICS_JSON, METRICS_CSV, REPORT_MARKDOWN)
BOOTSTRAP_METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_20",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
    "mrr_at_20",
)
CUTOFFS = (1, 5, 10, 20)
MACRO_RECALL_METRICS = tuple(f"macro_recall_at_k_{cutoff}" for cutoff in CUTOFFS)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _resolve_path(raw: object, *, manifest_path: Path) -> Path:
    if raw is None or raw == "":
        raise ValueError("Artifact path must be non-empty")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _portable(path: Path, *, root: Path | None) -> str:
    return portable_path(path, root=root)


def _manifest_path_value(
    value: object,
    *,
    manifest_path: Path,
    root: Path | None,
) -> object:
    if value is None:
        return None
    return _portable(
        _resolve_path(value, manifest_path=manifest_path),
        root=root,
    )


def _bootstrap_config(
    manifest: Mapping[str, object],
    manifest_path: Path,
) -> Mapping[str, object]:
    for value in (manifest.get("bootstrap"), manifest.get("config")):
        if isinstance(value, Mapping):
            nested = value.get("bootstrap", value)
            if isinstance(nested, Mapping):
                return nested
    resolved = manifest_path.parent / "resolved_config.json"
    if resolved.is_file():
        config = _load_json(resolved)
        value = config.get("bootstrap")
        if isinstance(value, Mapping):
            return value
    return {}


def _metric_value(metrics: Mapping[str, object], name: str, path: Path) -> float:
    if name not in metrics:
        raise ValueError(f"{path}: missing required metric {name!r}")
    try:
        return float(metrics[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: metric {name!r} is not numeric") from error


def _macro_recall(metrics: Mapping[str, object], cutoff: int, path: Path) -> float:
    values = metrics.get("macro_recall_at_k")
    if not isinstance(values, Mapping) or str(cutoff) not in values:
        raise ValueError(
            f"{path}: missing required macro_recall_at_k[{cutoff!r}]"
        )
    try:
        return float(values[str(cutoff)])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: macro recall at {cutoff} is not numeric"
        ) from error


def _read_details(
    path: Path,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(
        path,
        columns=["example_id", "positive_labels", *BOOTSTRAP_METRICS],
    ).to_pylist()
    if not rows:
        raise ValueError(f"{path}: details are empty")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or not raw.get("example_id"):
            raise ValueError(f"{path}: detail row {index} has no example_id")
        row = dict(raw)
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"{path}: duplicate example_id {example_id!r}")
        seen.add(example_id)
        for metric in BOOTSTRAP_METRICS:
            _metric_value(row, metric, path)
        result.append(row)
    try:
        sample_count = int(_metric_value(metrics, "samples", metrics_path))
    except ValueError:
        raise
    if sample_count != len(result):
        raise ValueError(
            f"{metrics_path}: samples={sample_count} but details has "
            f"{len(result)} rows"
        )
    return result


def _load_vocabulary(
    metrics_by_track: Mapping[str, Mapping[str, object]],
    *,
    manifest_path: Path,
) -> tuple[str, ...]:
    raw_paths = {
        metrics.get("vocabulary_path")
        for metrics in metrics_by_track.values()
        if metrics.get("vocabulary_path") is not None
    }
    if len(raw_paths) > 1:
        raise ValueError("Tracks refer to different vocabulary paths")
    if raw_paths:
        path = _resolve_path(next(iter(raw_paths)), manifest_path=manifest_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(path)
        if "subcategory" not in table.column_names:
            raise ValueError(f"{path}: missing 'subcategory' column")
        labels = tuple(str(value) for value in table["subcategory"].to_pylist())
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(f"{path}: vocabulary labels must be unique and non-empty")
        return labels
    labels = {
        str(label)
        for metrics in metrics_by_track.values()
        for label in dict(metrics.get("per_label_support", {}))
    }
    return tuple(sorted(labels))


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> list[dict[str, object]]:
    raw_conditions = manifest.get("conditions")
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ) or not raw_conditions:
        raise ValueError("Manifest must contain a non-empty 'conditions' list")
    conditions: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_conditions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Condition {index} must be an object")
        condition = dict(raw)
        condition_id = condition.get("baseline_id")
        if condition_id is None or str(condition_id) == "":
            raise ValueError(f"Condition {index} is missing baseline_id")
        condition_id = str(condition_id)
        if condition_id in seen:
            raise ValueError(f"Duplicate condition {condition_id!r}")
        seen.add(condition_id)
        tracks = condition.get("tracks")
        if not isinstance(tracks, Mapping):
            raise ValueError(f"{condition_id}: missing tracks object")
        missing_tracks = set(TRACKS) - set(tracks)
        extra_tracks = set(tracks) - set(TRACKS)
        if missing_tracks or extra_tracks:
            raise ValueError(
                f"{condition_id}: tracks must be exactly {TRACKS}; "
                f"missing={sorted(missing_tracks)}, extra={sorted(extra_tracks)}"
            )
        normalized_tracks: dict[str, object] = {}
        for track in TRACKS:
            raw_track = tracks[track]
            if not isinstance(raw_track, Mapping):
                raise ValueError(f"{condition_id}/{track}: track must be an object")
            metrics_raw = raw_track.get("metrics_path")
            details_raw = raw_track.get("details_path")
            if metrics_raw is None or details_raw is None:
                raise ValueError(
                    f"{condition_id}/{track}: metrics_path and details_path "
                    "are required"
                )
            metrics_path = _resolve_path(metrics_raw, manifest_path=manifest_path)
            details_path = _resolve_path(details_raw, manifest_path=manifest_path)
            metrics = _load_json(metrics_path)
            if metrics.get("track") not in (None, track):
                raise ValueError(
                    f"{metrics_path}: track is {metrics.get('track')!r}, "
                    f"expected {track!r}"
                )
            for metric in BOOTSTRAP_METRICS:
                _metric_value(metrics, metric, metrics_path)
            for cutoff in CUTOFFS:
                _macro_recall(metrics, cutoff, metrics_path)
            normalized_tracks[track] = {
                "metrics_path": metrics_path,
                "details_path": details_path,
                "metrics": metrics,
                "predictions_path": raw_track.get("predictions_path"),
            }
        conditions.append(
            {
                "baseline_id": condition_id,
                "predictions_path": condition.get("predictions_path"),
                "tracks": normalized_tracks,
            }
        )
    return conditions


def _label_support(
    metrics: Mapping[str, object],
    details: Sequence[Mapping[str, object]],
    labels: Sequence[str],
) -> dict[str, int]:
    raw_support = metrics.get("per_label_support")
    if isinstance(raw_support, Mapping):
        support = {str(label): int(value) for label, value in raw_support.items()}
    else:
        support = {}
        for row in details:
            for label in row.get("positive_labels", []):
                support[str(label)] = support.get(str(label), 0) + 1
    return {label: int(support.get(label, 0)) for label in labels}


def _label_tiers(
    manifest: Mapping[str, object],
    manifest_path: Path,
    labels: Sequence[str],
    fallback_support: Mapping[str, int],
) -> dict[str, object]:
    support = dict(fallback_support)
    benchmark_value = manifest.get("benchmark_dir")
    if benchmark_value is not None:
        benchmark_dir = _resolve_path(
            benchmark_value,
            manifest_path=manifest_path,
        )
        statistics_path = benchmark_dir / "statistics.json"
        if statistics_path.is_file():
            statistics = _load_json(statistics_path)
            audit = statistics.get("train_only_audit")
            if isinstance(audit, Mapping):
                raw_support = audit.get(
                    "per_label_distinct_user_positive_support"
                )
                if isinstance(raw_support, Mapping):
                    support = {
                        str(label): int(value)
                        for label, value in raw_support.items()
                    }
    ordered = sorted(
        labels,
        key=lambda label: (-int(support.get(label, 0)), label),
    )
    base, remainder = divmod(len(ordered), 3)
    sizes = [base + (index < remainder) for index in range(3)]
    names = ("head", "torso", "tail")
    tiers: dict[str, object] = {}
    start = 0
    for name, size in zip(names, sizes):
        tier_labels = ordered[start : start + size]
        start += size
        tier_support = [int(support.get(label, 0)) for label in tier_labels]
        tiers[name] = {
            "labels": tier_labels,
            "label_count": len(tier_labels),
            "train_support_min": min(tier_support, default=0),
            "train_support_max": max(tier_support, default=0),
        }
    return {
        "definition": (
            "Labels sorted by Train distinct-user positive support, then "
            "split into three near-equal label-count groups."
        ),
        "tiers": tiers,
    }


def _label_tier_rows(
    conditions: Sequence[Mapping[str, object]],
    label_tiers: Mapping[str, object],
) -> list[dict[str, object]]:
    tiers = label_tiers["tiers"]
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for track in TRACKS:
            metrics = condition["tracks"][track]["metrics"]
            support = {
                str(label): int(value)
                for label, value in dict(
                    metrics.get("per_label_support", {})
                ).items()
            }
            recalls = metrics.get("per_label_recall")
            if not isinstance(recalls, Mapping):
                raise ValueError(
                    f"{condition['baseline_id']}/{track}: missing "
                    "per_label_recall"
                )
            for tier_name, tier in tiers.items():
                row: dict[str, object] = {
                    "condition_id": condition["baseline_id"],
                    "track": track,
                    "tier": tier_name,
                    "label_count": tier["label_count"],
                }
                supported_labels = [
                    label
                    for label in tier["labels"]
                    if support.get(label, 0) > 0
                ]
                row["test_supported_labels"] = len(supported_labels)
                for cutoff in CUTOFFS:
                    cutoff_values = recalls.get(str(cutoff))
                    if not isinstance(cutoff_values, Mapping):
                        raise ValueError(
                            f"{condition['baseline_id']}/{track}: missing "
                            f"per_label_recall[{cutoff}]"
                        )
                    values = [
                        float(cutoff_values[label])
                        for label in supported_labels
                    ]
                    row[f"macro_recall_at_{cutoff}"] = (
                        sum(values) / len(values) if values else 0.0
                    )
                rows.append(row)
    return rows


def _format_delta(
    bootstrap: Mapping[str, object] | None,
    metric: str,
) -> str:
    if not bootstrap:
        return "—"
    entry = bootstrap["metrics"][metric]
    return (
        f"{float(entry['observed_delta']):+.4f} "
        f"[{float(entry['lower']):+.4f}, {float(entry['upper']):+.4f}]"
    )


def _markdown(
    *,
    manifest_path: Path,
    output_dir: Path,
    root: Path | None,
    manifest: Mapping[str, object],
    conditions: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    bootstraps: Sequence[Mapping[str, object]],
    label_tables: Mapping[str, Mapping[str, int]],
    unsupported: Mapping[str, Sequence[str]],
    reference_baseline: str,
) -> str:
    bootstrap_by_key = {
        (str(item["condition_id"]), str(item["track"])): item["result"]
        for item in bootstraps
    }
    lines = [
        "# Future Novel-Interest Baseline Results (Draft)",
        "",
        "This draft is generated directly from the baseline manifest. It "
        "contains no invented examples, translations, or numerical claims.",
        "",
        f"- Reference baseline: `{reference_baseline}`",
        f"- Rule version: `{manifest.get('rule_version', 'not recorded')}`",
        f"- Manifest: `{_portable(manifest_path, root=root)}`",
        f"- Output directory: `{_portable(output_dir, root=root)}`",
        "",
        "## Track results",
    ]
    for track in TRACKS:
        lines.extend(
            [
                "",
                f"### `{track}`",
                "",
                "| Condition | N | Recall@1 | Recall@5 | Recall@10 | "
                "Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | "
                "Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 | "
                "Macro Recall@1 | Macro Recall@5 | Macro Recall@10 | "
                "Macro Recall@20 |",
                "|---|---:|" + "---:|" * 17,
            ]
        )
        for row in rows:
            if row["track"] != track:
                continue
            values = [
                row["condition_id"],
                f"{int(row['samples']):,}",
                *[f"{float(row[metric]):.4f}" for metric in BOOTSTRAP_METRICS],
                *[
                    f"{float(row[f'macro_recall_at_k_{cutoff}']):.4f}"
                    for cutoff in CUTOFFS
                ],
            ]
            lines.append("| " + " | ".join(map(str, values)) + " |")
        lines.extend(
            [
                "",
                "Bootstrap deltas versus the reference (observed delta "
                "with percentile interval):",
                "",
                "| Condition | Recall@1 | Recall@5 | Recall@10 | Recall@20 | "
                "NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@1 | Hit@5 | "
                "Hit@10 | Hit@20 | MRR@20 |",
                "|---|" + "---:|" * len(BOOTSTRAP_METRICS),
            ]
        )
        for condition in conditions:
            condition_id = str(condition["baseline_id"])
            if condition_id == reference_baseline:
                continue
            result = bootstrap_by_key[(condition_id, track)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        condition_id,
                        *[
                            _format_delta(result, metric)
                            for metric in BOOTSTRAP_METRICS
                        ],
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Label support"])
    for track in TRACKS:
        support = label_tables[track]
        supported = [(label, count) for label, count in support.items() if count]
        lines.extend(
            [
                "",
                f"### `{track}`",
                "",
                f"Supported labels: {len(supported)}; unsupported labels: "
                f"{len(unsupported[track])}.",
                "",
                "| Label | Positive-target support |",
                "|---|---:|",
                *[f"| {label} | {count} |" for label, count in supported],
                "",
                "Unsupported labels: "
                + (", ".join(f"`{label}`" for label in unsupported[track]) or "none"),
            ]
        )
    lines.extend(
        [
            "",
            "## Data and run paths",
            "",
            f"- Benchmark directory: `{_manifest_path_value(manifest.get('benchmark_dir'), manifest_path=manifest_path, root=root) or 'not recorded'}`",
            f"- Artifact directory: `{_manifest_path_value(manifest.get('artifact_dir'), manifest_path=manifest_path, root=root) or 'not recorded'}`",
        ]
    )
    for condition in conditions:
        for track in TRACKS:
            item = condition["tracks"][track]
            lines.extend(
                [
                    f"- `{condition['baseline_id']}/{track}` metrics: "
                    f"`{_portable(item['metrics_path'], root=root)}`; details: "
                    f"`{_portable(item['details_path'], root=root)}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Never-exposed unknown: labels that were never exposed or "
            "otherwise absent from the evaluated candidate set are not "
            "evidence of recoverability.",
            "- Future positives are observed benchmark events, not randomized "
            "outcomes; this report makes no causal claim.",
            "- This report uses deterministic ranking metrics and paired "
            "user bootstrap intervals; it uses no LLM judge.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_future_interest_report(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
    reference_baseline: str | None = None,
    replicates: int | None = None,
    seed: int | None = None,
    confidence: float | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Build flattened comparisons and a human-editable future-interest draft."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = _load_json(manifest_path)
    conditions = _validate_manifest(manifest, manifest_path=manifest_path)
    config = _bootstrap_config(manifest, manifest_path)
    configured_reference = str(
        config.get("reference_baseline", "global_popularity")
    )
    reference = reference_baseline or configured_reference
    by_id = {str(item["baseline_id"]): item for item in conditions}
    if reference not in by_id:
        raise ValueError(
            f"Configured reference baseline {reference!r} is not present"
        )
    bootstrap_replicates = int(
        replicates if replicates is not None else config.get("replicates", 2_000)
    )
    bootstrap_seed = int(seed if seed is not None else config.get("seed", 20260812))
    bootstrap_confidence = float(
        confidence if confidence is not None else config.get("confidence", 0.95)
    )
    if bootstrap_replicates < 2_000:
        raise ValueError("bootstrap replicates must be at least 2000")
    if not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    reference_item = by_id[reference]
    label_tables: dict[str, dict[str, int]] = {}
    unsupported: dict[str, list[str]] = {}
    reference_details: dict[str, list[dict[str, object]]] = {}
    for track in TRACKS:
        metric_items = {
            str(item["baseline_id"]): item["tracks"][track]["metrics"]
            for item in conditions
        }
        reference_track = reference_item["tracks"][track]
        details = _read_details(
            reference_track["details_path"],
            reference_track["metrics"],
            reference_track["metrics_path"],
        )
        reference_details[track] = details
        labels = _load_vocabulary(metric_items, manifest_path=manifest_path)
        label_tables[track] = _label_support(
            reference_track["metrics"],
            details,
            labels,
        )
        unsupported[track] = [
            label for label, count in label_tables[track].items() if count == 0
        ]
    label_tiers = _label_tiers(
        manifest,
        manifest_path,
        tuple(label_tables[TRACKS[0]]),
        label_tables[TRACKS[0]],
    )
    label_tier_results = _label_tier_rows(conditions, label_tiers)

    rows: list[dict[str, object]] = []
    bootstraps: list[dict[str, object]] = []
    for condition in conditions:
        condition_id = str(condition["baseline_id"])
        for track in TRACKS:
            item = condition["tracks"][track]
            prediction_value = item["predictions_path"]
            if prediction_value is None:
                prediction_value = condition.get("predictions_path")
            metrics = item["metrics"]
            row: dict[str, object] = {
                "condition_id": condition_id,
                "track": track,
                "reference_baseline": reference,
                "samples": int(metrics["samples"]),
                "metrics_path": _portable(item["metrics_path"], root=root),
                "details_path": _portable(item["details_path"], root=root),
                "predictions_path": (
                    _portable(
                        _resolve_path(
                            prediction_value,
                            manifest_path=manifest_path,
                        ),
                        root=root,
                    )
                    if prediction_value is not None
                    else None
                ),
            }
            for metric in BOOTSTRAP_METRICS:
                row[metric] = _metric_value(metrics, metric, item["metrics_path"])
            for cutoff in CUTOFFS:
                row[f"macro_recall_at_k_{cutoff}"] = _macro_recall(
                    metrics, cutoff, item["metrics_path"]
                )
            if condition_id != reference:
                baseline_details = reference_details[track]
                comparison_details = _read_details(
                    item["details_path"],
                    metrics,
                    item["metrics_path"],
                )
                baseline_ids = {
                    str(detail["example_id"]) for detail in baseline_details
                }
                comparison_ids = {
                    str(detail["example_id"]) for detail in comparison_details
                }
                if baseline_ids != comparison_ids:
                    raise ValueError(
                        f"Paired comparison requires identical evaluated user "
                        f"IDs for {condition_id}/{track}; "
                        f"reference_only={sorted(baseline_ids - comparison_ids)[:5]}, "
                        f"comparison_only={sorted(comparison_ids - baseline_ids)[:5]}"
                    )
                result = paired_bootstrap_metric_delta(
                    baseline_details,
                    comparison_details,
                    metric_names=BOOTSTRAP_METRICS,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                    confidence=bootstrap_confidence,
                )
                bootstraps.append(
                    {
                        "condition_id": condition_id,
                        "track": track,
                        "reference_baseline": reference,
                        "result": result,
                    }
                )
                for metric in BOOTSTRAP_METRICS:
                    entry = result["metrics"][metric]
                    row[f"delta_{metric}"] = float(entry["observed_delta"])
                    row[f"delta_{metric}_lower"] = float(entry["lower"])
                    row[f"delta_{metric}_upper"] = float(entry["upper"])
            else:
                for metric in BOOTSTRAP_METRICS:
                    row[f"delta_{metric}"] = None
                    row[f"delta_{metric}_lower"] = None
                    row[f"delta_{metric}_upper"] = None
            rows.append(row)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Report output directory already exists: {output_dir}. "
            "Use overwrite=True."
        )
    backup = output_dir.with_name(output_dir.name + ".old")
    if output_dir.exists() and overwrite and backup.exists():
        raise FileExistsError(
            f"Cannot replace {output_dir}: backup exists at {backup}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_path": _portable(manifest_path, root=root),
        "output_paths": {
            name: _portable(output_dir / name, root=root)
            for name in REPORT_FILES
        },
        "reference_baseline": reference,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence": bootstrap_confidence,
            "metric_names": list(BOOTSTRAP_METRICS),
        },
        "rows": rows,
        "bootstrap_deltas": bootstraps,
        "label_support": label_tables,
        "unsupported_labels": unsupported,
        "label_tiers": label_tiers,
        "label_tier_results": label_tier_results,
        "paths": {
            "benchmark_dir": _manifest_path_value(
                manifest.get("benchmark_dir"),
                manifest_path=manifest_path,
                root=root,
            ),
            "artifact_dir": _manifest_path_value(
                manifest.get("artifact_dir"),
                manifest_path=manifest_path,
                root=root,
            ),
        },
    }
    with tempfile.TemporaryDirectory(
        prefix=".future-interest-report-",
        dir=str(output_dir.parent),
    ) as staging_name:
        staging = Path(staging_name)
        (staging / METRICS_JSON).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fieldnames = list(rows[0])
        with (staging / METRICS_CSV).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (staging / REPORT_MARKDOWN).write_text(
            _markdown(
                manifest_path=manifest_path,
                output_dir=output_dir,
                root=root,
                manifest=manifest,
                conditions=conditions,
                rows=rows,
                bootstraps=bootstraps,
                label_tables=label_tables,
                unsupported=unsupported,
                reference_baseline=reference,
            ),
            encoding="utf-8",
        )
        if output_dir.exists():
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the future novel-interest baseline report."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_future_interest_report(
                args.manifest,
                args.output_dir,
                overwrite=args.overwrite,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
