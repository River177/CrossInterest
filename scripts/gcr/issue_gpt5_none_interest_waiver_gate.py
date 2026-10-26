from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.experiments.finalize_static_candidate_pilots import (
    _sha256,
    finalize_static_candidate_pilots,
)
from new_interest.experiments.run_static_candidate_condition import (
    resolve_condition,
)
from new_interest.infrastructure.paths import portable_path


PASSING_CONDITION = "gpt5_none_behavior_uniform"
WAIVED_CONDITION = "gpt5_none_interest_uniform"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue the explicit user-waived GPT-5 None single-view gate."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    pilot_root = Path(
        json.loads(args.manifest.read_text(encoding="utf-8"))["artifact_root"]
    ) / "pilot"
    base_gate_path = args.output.with_name(f".{args.output.name}.passing-base.json")
    if base_gate_path.exists():
        raise FileExistsError(base_gate_path)
    try:
        finalize_static_candidate_pilots(
            args.manifest,
            pilot_root,
            base_gate_path,
            condition_ids=[PASSING_CONDITION],
        )
        gate = json.loads(base_gate_path.read_text(encoding="utf-8"))

        _, condition, data_path, expected_samples = resolve_condition(
            args.manifest, WAIVED_CONDITION, "pilot"
        )
        root = pilot_root / WAIVED_CONDITION
        required = {
            name: root / name
            for name in (
                "acceptance.json",
                "experiment_identity.json",
                "metrics.json",
                "predictions.parquet",
                "resolved_config.json",
            )
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        acceptance = json.loads(required["acceptance.json"].read_text())
        metrics = json.loads(required["metrics.json"].read_text())
        identity = json.loads(required["experiment_identity.json"].read_text())
        checks = acceptance["checks"]
        if acceptance.get("passed") is not False:
            raise ValueError("Waived condition must retain a failing acceptance")
        if checks["candidate_valid_rate"]["passed"] is not False:
            raise ValueError("Expected candidate-validity to be the failed check")
        for key in (
            "request_success_rate",
            "prefix_compliance_rate",
            "no_truncation",
            "zero_reasoning_tokens",
        ):
            if checks[key]["passed"] is not True:
                raise ValueError(f"Unexpected additional failed check: {key}")
        if float(metrics["candidate_valid_rate"]) != 0.97:
            raise ValueError("Waiver is scoped to the observed 0.97 valid rate")
        if int(metrics["samples"]) != expected_samples != 0:
            raise ValueError("Pilot sample count mismatch")
        expected_identity = {
            "condition_id": WAIVED_CONDITION,
            "phase": "pilot",
            "evaluator": str(condition["evaluator"]),
            "strategy": str(condition["strategy"]),
            "task_type": str(condition["task_type"]),
            "manifest_sha256": _sha256(args.manifest),
            "data_sha256": _sha256(data_path),
            "expected_samples": expected_samples,
        }
        mismatches = {
            key: {"expected": value, "actual": identity.get(key)}
            for key, value in expected_identity.items()
            if identity.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Identity mismatch: {mismatches}")
        expected_ids = pq.read_table(data_path, columns=["selection_id"])[
            "selection_id"
        ].to_pylist()
        actual_ids = pq.read_table(
            required["predictions.parquet"], columns=["selection_id"]
        )["selection_id"].to_pylist()
        if actual_ids != expected_ids:
            raise ValueError("Waived prediction IDs do not match pilot data")

        waiver = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "User explicitly authorized full launch after being informed "
                "that 30/1000 GPT-5 None Interest outputs were outside the "
                "candidate set (candidate_valid_rate=0.97)."
            ),
            "failed_acceptance_preserved": True,
            "candidate_valid_rate": metrics["candidate_valid_rate"],
            "candidate_invalid_count": 30,
        }
        gate["scope"] = "condition_subset_with_user_waiver"
        gate["selected_condition_ids"] = [
            PASSING_CONDITION,
            WAIVED_CONDITION,
        ]
        gate["condition_count"] = 2
        gate["user_waivers"] = {WAIVED_CONDITION: waiver}
        gate["conditions"][WAIVED_CONDITION] = {
            "passed": True,
            "acceptance_passed": False,
            "user_waiver": waiver,
            "evaluator": condition["evaluator"],
            "strategy": condition["strategy"],
            "task_type": condition["task_type"],
            "samples": expected_samples,
            "macro_category_accuracy": metrics["macro_category_accuracy"],
            "candidate_valid_rate": metrics["candidate_valid_rate"],
            "prefix_compliance_rate": metrics["prefix_compliance_rate"],
            "artifacts": {
                name: {
                    "path": portable_path(path),
                    "sha256": _sha256(path),
                }
                for name, path in required.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{args.output.name}.",
            dir=args.output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(gate, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        try:
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        base_gate_path.unlink(missing_ok=True)
    print(args.output)


if __name__ == "__main__":
    main()
