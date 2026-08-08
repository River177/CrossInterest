from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_report(
    state_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = []
    for key, entry in state["experiments"].items():
        metrics = entry.get("metrics_10k")
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "experiment": key,
                "task_type": entry["task_type"],
                "size": entry["size"],
                "train_loss": (
                    entry.get("train_summary", {})
                    .get("train_metrics", {})
                    .get("train_loss")
                ),
                **{
                    metric: metrics.get(metric)
                    for metric in (
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "recall_at_20",
                        "ndcg_at_1",
                        "ndcg_at_5",
                        "ndcg_at_10",
                        "ndcg_at_20",
                        "hit_at_20",
                        "mrr_at_20",
                        "runtime_seconds",
                        "samples_per_second",
                    )
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["task_type"],
            int(str(row["size"]).removesuffix("k")),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dev10k_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "dev10k_metrics.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    best_models = state.get("best_models", {})
    summary = {
        "experiments": rows,
        "best_models": best_models,
        "completed_at": state.get("completed_at"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# Qwen3-8B SFT Scaling Results",
        "",
        "| Input | Size | Recall@1 | Recall@5 | Recall@10 | Recall@20 | "
        "NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {task_type} | {size} | {recall_at_1:.4f} | "
            "{recall_at_5:.4f} | {recall_at_10:.4f} | "
            "{recall_at_20:.4f} | {ndcg_at_1:.4f} | "
            "{ndcg_at_5:.4f} | {ndcg_at_10:.4f} | "
            "{ndcg_at_20:.4f} |".format(**row)
        )
    if best_models:
        markdown.extend(
            [
                "",
                "## Best models",
                "",
                *[
                    f"- **{task_type}:** `{experiment}`"
                    for task_type, experiment in sorted(
                        best_models.items()
                    )
                ],
            ]
        )
    (output_dir / "report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Qwen serial experiment comparison."
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("artifacts/gcr/qwen3/serial_state.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gcr/qwen3/comparisons"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        json.dumps(
            build_report(args.state_path, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
