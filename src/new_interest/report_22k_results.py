from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONDITIONS = (
    ("qwen3", "interest_only", "qwen3-interest-22k"),
    ("qwen3", "behavior_only", "qwen3-behavior-22k"),
    ("llama31", "interest_only", "llama31-interest-22k"),
    ("llama31", "behavior_only", "llama31-behavior-22k"),
)


def collect_results(
    artifacts_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    rows = []
    missing = []
    for model, task_type, job_name in CONDITIONS:
        job_root = artifacts_root / job_name
        train_path = job_root / "sft/run_summary.json"
        metrics_path = job_root / "beam/metrics.json"
        if not train_path.is_file() or not metrics_path.is_file():
            missing.append(job_name)
            continue
        train = json.loads(train_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model,
                "task_type": task_type,
                "train_samples": train["train_samples"],
                "train_loss": train["train_metrics"]["train_loss"],
                "train_runtime": train["train_metrics"]["train_runtime"],
                "max_tokens": train["token_audit"]["max_tokens"],
                "p99_tokens": train["token_audit"]["p99_tokens"],
                **{
                    key: metrics[key]
                    for key in (
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
                        "ranking_length",
                        "runtime_seconds",
                        "samples_per_second",
                    )
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "conditions": rows,
        "missing_jobs": missing,
        "complete": not missing,
    }
    (output_dir / "qwen_llama_22k_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "qwen_llama_22k_results.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Qwen/Llama 22K SFT and Beam results."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(
            "artifacts/amlt/crossinterest-qwen-llama-22k-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/amlt/crossinterest-qwen-llama-22k-v1/comparison"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        json.dumps(
            collect_results(args.artifacts_root, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
