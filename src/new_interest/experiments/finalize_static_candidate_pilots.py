from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from new_interest.experiments.run_static_candidate_condition import (
    DEFAULT_MANIFEST,
    DEFAULT_PILOT_GATE,
    _load_manifest,
    resolve_condition,
)
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-candidate-pilot-gate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_static_candidate_pilots(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    pilot_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_PILOT_GATE,
    *,
    overwrite: bool = False,
    condition_ids: list[str] | tuple[str, ...] | None = None,
) -> Path:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    pilot_root = (
        Path(pilot_root)
        if pilot_root is not None
        else Path(str(manifest["artifact_root"])) / "pilot"
    )
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Pilot gate already exists: {output_path}. Use --overwrite."
        )

    available = {
        str(condition["id"]): condition
        for condition in manifest["conditions"]
    }
    if condition_ids is None:
        selected_conditions = list(manifest["conditions"])
    else:
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("Duplicate condition IDs in requested pilot scope")
        unknown = sorted(set(condition_ids).difference(available))
        if unknown:
            raise KeyError(f"Unknown pilot conditions: {unknown}")
        selected_conditions = [available[value] for value in condition_ids]
        if not selected_conditions:
            raise ValueError("Pilot condition scope must not be empty")

    condition_results: dict[str, object] = {}
    strategy_selection_ids: dict[str, list[str]] = {}
    manifest_hash = _sha256(manifest_path)
    for condition in selected_conditions:
        condition_id = str(condition["id"])
        _, resolved, data_path, expected_samples = resolve_condition(
            manifest_path, condition_id, "pilot"
        )
        root = pilot_root / condition_id
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
            raise FileNotFoundError(
                f"{condition_id}: incomplete pilot artifacts: {missing}"
            )
        acceptance = json.loads(required["acceptance.json"].read_text())
        identity = json.loads(required["experiment_identity.json"].read_text())
        metrics = json.loads(required["metrics.json"].read_text())
        if acceptance.get("passed") is not True:
            raise ValueError(f"{condition_id}: pilot acceptance did not pass")
        expected_identity = {
            "condition_id": condition_id,
            "phase": "pilot",
            "evaluator": str(resolved["evaluator"]),
            "strategy": str(resolved["strategy"]),
            "task_type": str(resolved["task_type"]),
            "manifest_sha256": manifest_hash,
            "data_sha256": _sha256(data_path),
            "expected_samples": expected_samples,
        }
        mismatches = {
            key: {"expected": value, "actual": identity.get(key)}
            for key, value in expected_identity.items()
            if identity.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{condition_id}: identity mismatch: {mismatches}")
        if int(metrics.get("samples", -1)) != expected_samples:
            raise ValueError(f"{condition_id}: metrics sample count mismatch")
        expected_ids = [
            str(value)
            for value in pq.read_table(
                data_path, columns=["selection_id"]
            )["selection_id"].to_pylist()
        ]
        strategy = str(resolved["strategy"])
        reference_ids = strategy_selection_ids.setdefault(strategy, expected_ids)
        if expected_ids != reference_ids:
            raise ValueError(
                f"{condition_id}: pilot data identities/order do not match "
                f"other {strategy} Behavior/Interest conditions"
            )
        actual_ids = [
            str(value)
            for value in pq.read_table(
                required["predictions.parquet"], columns=["selection_id"]
            )["selection_id"].to_pylist()
        ]
        if actual_ids != expected_ids:
            raise ValueError(
                f"{condition_id}: prediction identities/order do not match pilot data"
            )
        condition_results[condition_id] = {
            "passed": True,
            "evaluator": resolved["evaluator"],
            "strategy": resolved["strategy"],
            "task_type": resolved["task_type"],
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

    gate = {
        "rule_version": RULE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "scope": (
            "all_conditions"
            if condition_ids is None
            else "condition_subset"
        ),
        "selected_condition_ids": [
            str(condition["id"]) for condition in selected_conditions
        ],
        "manifest_path": portable_path(manifest_path),
        "manifest_sha256": manifest_hash,
        "condition_count": len(condition_results),
        "conditions": condition_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(gate, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify all static candidate pilots and issue the full-run gate."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_PILOT_GATE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--condition",
        dest="conditions",
        action="append",
        help="Certify one condition; repeat for an explicit subset gate.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = finalize_static_candidate_pilots(
        args.manifest,
        args.pilot_root,
        args.output,
        overwrite=args.overwrite,
        condition_ids=args.conditions,
    )
    print(output)


if __name__ == "__main__":
    main()
