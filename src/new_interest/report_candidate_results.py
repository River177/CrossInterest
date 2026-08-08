from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def build_report(
    output_dir: Path,
    *,
    require_complete: bool = True,
) -> list[dict[str, object]]:
    rows = []
    missing = []
    for condition, raw_path in DEFAULT_RESULTS.items():
        path = Path(raw_path)
        if not path.is_file():
            missing.append(str(path))
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        model, input_type = _condition_metadata(condition)
        rows.append(
            {
                "condition": condition,
                "model": model,
                "input": input_type,
                "samples": metrics["samples"],
                "category_accuracy": metrics["category_accuracy"],
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
                "metrics_path": str(path.resolve()),
            }
        )
    if require_complete and missing:
        raise FileNotFoundError(
            "Missing candidate results:\n" + "\n".join(missing)
        )
    rows.sort(key=lambda row: (row["input"], row["model"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "comparison.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    markdown = [
        "# K=10 Candidate Selection Results",
        "",
        "| Model | Input | N | Selection Accuracy | Strict Format Accuracy | "
        "Candidate-valid | Prefix Compliance | Empty |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {model} | {input} | {samples:,} | "
            "{category_accuracy:.2%} | {strict_accuracy:.2%} | "
            "{candidate_valid_rate:.2%} | "
            "{prefix_compliance_rate:.2%} | "
            "{empty_output_rate:.2%} |".format(**row)
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
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        json.dumps(
            build_report(
                args.output_dir,
                require_complete=not args.allow_missing,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
