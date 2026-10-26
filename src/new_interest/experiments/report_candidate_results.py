from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_RESULTS = {
    "qwen_base_behavior": (
        "artifacts/candidate-selection-k10/amlt/"
        "candidate-k10-qwen-base-v2/candidate-base-behavior/"
        "evaluation/metrics.json"
    ),
    "qwen_base_interest": (
        "artifacts/candidate-selection-k10/amlt/"
        "candidate-k10-qwen-base-v2/candidate-base-interest/"
        "evaluation/metrics.json"
    ),
    "qwen35_base_behavior": (
        "artifacts/candidate-selection-k10/amlt/"
        "candidate-qwen35-base-v1/qwen35-base-behavior/"
        "evaluation/metrics.json"
    ),
    "qwen35_base_interest": (
        "artifacts/candidate-selection-k10/amlt/"
        "candidate-qwen35-base-v1/qwen35-base-interest/"
        "evaluation/metrics.json"
    ),
    "qwen_sft_behavior": (
        "artifacts/candidate-selection-k10/"
        "qwen_sft_behavior/metrics.json"
    ),
    "qwen_sft_interest": (
        "artifacts/candidate-selection-k10/"
        "qwen_sft_interest/metrics.json"
    ),
    "gpt56_behavior": (
        "artifacts/candidate-selection-k10/"
        "gpt56_behavior/full/metrics.json"
    ),
    "gpt56_interest": (
        "artifacts/candidate-selection-k10/"
        "gpt56_interest/full/metrics.json"
    ),
    "gpt5_behavior": (
        "artifacts/candidate-selection-k10/"
        "gpt5_behavior/full/metrics.json"
    ),
    "gpt5_interest": (
        "artifacts/candidate-selection-k10/"
        "gpt5_interest/full/metrics.json"
    ),
}

LEGACY_METADATA = "Not recorded"

ROW_FIELDS = [
    "condition",
    "condition_id",
    "model",
    "input",
    "thinking_reasoning",
    "decoding",
    "samples",
    "category_accuracy",
    "macro_category_accuracy",
    "macro_category_count",
    "strict_accuracy",
    "candidate_valid_rate",
    "prefix_compliance_rate",
    "empty_output_rate",
    "multiple_candidates_rate",
    "request_failures",
    "truncation_count",
    "truncation_rate",
    "missing_final_answer_count",
    "missing_final_answer_rate",
    "enable_thinking",
    "reasoning_effort",
    "requested_reasoning_effort",
    "total_reasoning_tokens",
    "think_block_count",
    "think_block_rate",
    "closed_think_block_count",
    "closed_think_block_rate",
    "source_experiment",
    "source_job",
    "metrics_path",
]


def _condition_metadata(name: str) -> tuple[str, str]:
    input_type = "behavior_only" if name.endswith("behavior") else "interest_only"
    if name.startswith("qwen35_base"):
        model = "Qwen3.5-9B"
    elif name.startswith("qwen_base"):
        model = "Qwen3-8B Base"
    elif name.startswith("qwen_sft"):
        model = "Qwen3-8B 10K SFT"
    elif name.startswith("gpt56"):
        model = "GPT-5.6 Sol"
    elif name.startswith("gpt5"):
        model = "GPT-5"
    else:
        raise ValueError(name)
    return model, input_type


def _legacy_registry() -> list[dict[str, object]]:
    registry = []
    for condition_id, metrics_path in DEFAULT_RESULTS.items():
        model, input_type = _condition_metadata(condition_id)
        registry.append(
            {
                "condition_id": condition_id,
                "model": model,
                "input": input_type,
                "thinking_reasoning": LEGACY_METADATA,
                "decoding": LEGACY_METADATA,
                "metrics_path": metrics_path,
            }
        )
    return registry


def _load_manifest(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_registry(
    registry: object,
) -> list[dict[str, object]]:
    if isinstance(registry, Mapping) and "conditions" in registry:
        registry = registry["conditions"]
    elif isinstance(registry, Mapping):
        registry = [
            {"condition_id": condition_id, **condition}
            for condition_id, condition in registry.items()
            if isinstance(condition, Mapping)
        ]

    if not isinstance(registry, Sequence) or isinstance(registry, str):
        raise ValueError(
            "Condition registry must be a list, an ID mapping, or an object "
            "with a 'conditions' list."
        )

    normalized = []
    for index, raw_condition in enumerate(registry):
        if not isinstance(raw_condition, Mapping):
            raise ValueError(f"Condition {index} must be an object.")
        condition = dict(raw_condition)
        aliases = {
            "condition_id": ("condition_id", "condition", "id"),
            "model": ("model", "model_display_name"),
            "input": ("input",),
            "thinking_reasoning": (
                "thinking_reasoning",
                "reasoning_mode",
                "thinking_mode",
            ),
            "decoding": ("decoding", "decoding_label"),
            "metrics_path": ("metrics_path", "path"),
        }
        result: dict[str, object] = {}
        for field, field_aliases in aliases.items():
            value = next(
                (
                    condition[alias]
                    for alias in field_aliases
                    if alias in condition
                ),
                None,
            )
            if value is None or value == "":
                raise ValueError(
                    f"Condition {index} is missing required field {field!r}."
                )
            result[field] = value
        result["source_experiment"] = condition.get("source_experiment")
        result["source_job"] = condition.get("source_job")
        normalized.append(result)
    return normalized


def _metric_or_generation_stat(
    metrics: Mapping[str, object],
    *names: str,
) -> object:
    for name in names:
        if name in metrics:
            return metrics[name]
    generation_stats = metrics.get("generation_stats")
    if isinstance(generation_stats, Mapping):
        for name in names:
            if name in generation_stats:
                return generation_stats[name]
    return None


def _failure_summary(row: Mapping[str, object]) -> str:
    parts = []
    request_failures = row["request_failures"]
    if request_failures is not None:
        parts.append(f"failures: {request_failures}")

    for label, count_key, rate_key in (
        ("truncated", "truncation_count", "truncation_rate"),
        (
            "missing final",
            "missing_final_answer_count",
            "missing_final_answer_rate",
        ),
    ):
        count = row[count_key]
        rate = row[rate_key]
        if count is not None and rate is not None:
            parts.append(f"{label}: {count} ({float(rate):.2%})")
        elif count is not None:
            parts.append(f"{label}: {count}")
        elif rate is not None:
            parts.append(f"{label}: {float(rate):.2%}")
    return "; ".join(parts) if parts else "—"


def build_report(
    output_dir: Path,
    *,
    require_complete: bool = True,
    condition_registry: object | None = None,
    manifest_path: Path | None = None,
) -> list[dict[str, object]]:
    if condition_registry is not None and manifest_path is not None:
        raise ValueError(
            "Pass either condition_registry or manifest_path, not both."
        )
    if manifest_path is not None:
        condition_registry = _load_manifest(manifest_path)
    registry = (
        _legacy_registry()
        if condition_registry is None
        else _normalize_registry(condition_registry)
    )

    rows = []
    missing = []
    for condition in registry:
        condition_id = str(condition["condition_id"])
        raw_path = str(condition["metrics_path"])
        path = Path(raw_path)
        if not path.is_file():
            missing.append(str(path))
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        per_category_accuracy = metrics["per_category_accuracy"]
        macro_category_accuracy = metrics.get(
            "macro_category_accuracy"
        )
        if macro_category_accuracy is None:
            macro_category_accuracy = sum(
                per_category_accuracy.values()
            ) / len(per_category_accuracy)
        rows.append(
            {
                "condition": condition_id,
                "condition_id": condition_id,
                "model": condition["model"],
                "input": condition["input"],
                "thinking_reasoning": condition["thinking_reasoning"],
                "decoding": condition["decoding"],
                "samples": metrics["samples"],
                "category_accuracy": metrics["category_accuracy"],
                "macro_category_accuracy": macro_category_accuracy,
                "macro_category_count": metrics.get(
                    "macro_category_count",
                    len(per_category_accuracy),
                ),
                "strict_accuracy": metrics["strict_accuracy"],
                "candidate_valid_rate": metrics[
                    "candidate_valid_rate"
                ],
                "prefix_compliance_rate": metrics[
                    "prefix_compliance_rate"
                ],
                "empty_output_rate": metrics["empty_output_rate"],
                "multiple_candidates_rate": metrics[
                    "multiple_candidates_rate"
                ],
                "request_failures": metrics.get("request_failures"),
                "truncation_count": _metric_or_generation_stat(
                    metrics,
                    "truncation_count",
                    "truncated_count",
                    "reached_max_new_tokens_count",
                ),
                "truncation_rate": _metric_or_generation_stat(
                    metrics,
                    "truncation_rate",
                    "truncated_rate",
                    "reached_max_new_tokens_rate",
                ),
                "missing_final_answer_count": (
                    _metric_or_generation_stat(
                        metrics,
                        "missing_final_answer_count",
                    )
                ),
                "missing_final_answer_rate": (
                    _metric_or_generation_stat(
                        metrics,
                        "missing_final_answer_rate",
                    )
                ),
                "enable_thinking": metrics.get("enable_thinking"),
                "reasoning_effort": metrics.get("reasoning_effort"),
                "requested_reasoning_effort": metrics.get(
                    "requested_reasoning_effort"
                ),
                "total_reasoning_tokens": metrics.get(
                    "total_reasoning_tokens"
                ),
                "think_block_count": _metric_or_generation_stat(
                    metrics,
                    "think_block_count",
                ),
                "think_block_rate": _metric_or_generation_stat(
                    metrics,
                    "think_block_rate",
                ),
                "closed_think_block_count": (
                    _metric_or_generation_stat(
                        metrics,
                        "closed_think_block_count",
                    )
                ),
                "closed_think_block_rate": (
                    _metric_or_generation_stat(
                        metrics,
                        "closed_think_block_rate",
                    )
                ),
                "source_experiment": condition.get("source_experiment"),
                "source_job": condition.get("source_job"),
                "metrics_path": str(path.resolve()),
            }
        )
    if require_complete and missing:
        raise FileNotFoundError(
            "Missing candidate results:\n" + "\n".join(missing)
        )
    rows.sort(
        key=lambda row: (
            row["input"],
            row["model"],
            row["thinking_reasoning"],
            row["decoding"],
            row["condition_id"],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# K=10 Candidate Selection Results",
        "",
        "| Model | Input | Thinking/Reasoning | Decoding | N | Micro | "
        "Macro | Strict | Candidate-valid | Failure/Truncation |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            "| {model} | {input} | {thinking_reasoning} | {decoding} | "
            "{samples:,} | "
            "{category_accuracy:.2%} | {macro_category_accuracy:.2%} | "
            "{strict_accuracy:.2%} | "
            "{candidate_valid_rate:.2%} | "
            "{failure_summary} |".format(
                **row,
                failure_summary=_failure_summary(row),
            )
        )
    if missing:
        markdown.extend(
            ["", "## Missing", "", *[f"- `{path}`" for path in missing]]
        )
    (output_dir / "report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the candidate selection comparison."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/candidate-selection-k10/comparison"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON condition registry for an explicit comparison.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        json.dumps(
            build_report(
                args.output_dir,
                require_complete=not args.allow_missing,
                manifest_path=args.manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
