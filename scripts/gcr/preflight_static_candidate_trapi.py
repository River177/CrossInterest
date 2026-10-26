from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_interest.evaluation.evaluate_candidate_trapi import evaluate_trapi
from new_interest.experiments.run_static_candidate_condition import (
    resolve_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real TRAPI request for selected full conditions."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--condition", action="append", dest="conditions", required=True
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {
        str(condition["id"]): condition
        for condition in manifest["conditions"]
    }
    for condition_id in args.conditions:
        if condition_id not in by_id:
            raise KeyError(f"Unknown condition: {condition_id}")
        _, condition, data_file, _ = resolve_condition(
            args.manifest, condition_id, "full"
        )
        config = dict(condition["parameters"])
        config.update(
            {
                "data_file": str(data_file),
                "output_dir": str(args.output_root / condition_id),
                "pilot_size": 1,
                "checkpoint_size": 1,
                "max_attempts": 5,
                "minimum_success_rate": 1.0,
                "seed": int(manifest["seed"]),
            }
        )
        metrics = evaluate_trapi(config)
        summary = {
            "condition": condition_id,
            "request_success_rate": metrics["request_success_rate"],
            "candidate_valid_rate": metrics["candidate_valid_rate"],
            "prefix_compliance_rate": metrics["prefix_compliance_rate"],
            "truncated_output_count": metrics["truncated_output_count"],
            "total_reasoning_tokens": metrics["total_reasoning_tokens"],
        }
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
