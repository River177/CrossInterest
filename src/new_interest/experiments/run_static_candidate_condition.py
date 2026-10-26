from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_candidate_openai import evaluate_openai
from new_interest.evaluation.evaluate_candidate_trapi import evaluate_trapi
from new_interest.infrastructure.paths import portable_path


DEFAULT_MANIFEST = Path("configs/candidate/static_user_disjoint_v1.json")
DEFAULT_PILOT_GATE = Path(
    "configs/candidate/static_user_disjoint_v1_pilot_gate.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("conditions"), list):
        raise ValueError(f"{path}: malformed experiment manifest")
    identifiers = [str(row["id"]) for row in value["conditions"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: duplicate condition IDs")
    return value


def validate_pilot_gate(
    manifest_path: Path | str,
    gate_path: Path | str = DEFAULT_PILOT_GATE,
    *,
    condition_id: str | None = None,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    gate_path = Path(gate_path)
    if not gate_path.is_file():
        raise FileNotFoundError(
            f"Full evaluation is locked until all pilots pass: {gate_path}"
        )
    manifest = _load_manifest(manifest_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError(f"{gate_path}: pilot gate is not passing")
    if gate.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"{gate_path}: manifest SHA-256 does not match")
    expected = {str(row["id"]) for row in manifest["conditions"]}
    conditions = gate.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(f"{gate_path}: malformed condition results")
    actual = set(conditions)
    if not actual.issubset(expected):
        raise ValueError(f"{gate_path}: condition set contains unknown conditions")
    if condition_id is None and actual != expected:
        raise ValueError(f"{gate_path}: condition set does not match manifest")
    if condition_id is not None and condition_id not in actual:
        raise ValueError(
            f"{gate_path}: condition {condition_id!r} is not certified"
        )
    failed = [
        condition_id
        for condition_id, value in conditions.items()
        if not isinstance(value, dict) or value.get("passed") is not True
    ]
    if failed:
        raise ValueError(f"{gate_path}: non-passing conditions: {failed}")
    return gate


def resolve_condition(
    manifest_path: Path | str,
    condition_id: str,
    phase: str,
    *,
    output_dir: Path | str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tuple[dict[str, object], dict[str, object], Path, int]:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    if phase not in {"pilot", "full"}:
        raise ValueError("phase must be pilot or full")
    matches = [
        row for row in manifest["conditions"] if str(row["id"]) == condition_id
    ]
    if not matches:
        raise KeyError(f"Unknown condition {condition_id!r}")
    condition = dict(matches[0])
    data = dict(manifest["data"])
    root = Path(str(data[f"{phase}_root"]))
    filename = str(data[f"{phase}_filename"])
    expected_samples = int(data[f"{phase}_samples"])
    data_file = (
        root
        / str(condition["strategy"])
        / str(condition["task_type"])
        / filename
    )
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    actual_samples = pq.ParquetFile(data_file).metadata.num_rows
    if actual_samples != expected_samples:
        raise ValueError(
            f"{data_file}: expected {expected_samples} rows, found {actual_samples}"
        )
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else Path(str(manifest["artifact_root"])) / phase / condition_id
    )
    config = dict(condition["parameters"])
    config.update(
        {
            "data_file": str(data_file),
            "output_dir": str(resolved_output),
            "seed": int(config.get("seed", manifest["seed"])),
        }
    )
    if condition["evaluator"] == "openai":
        config["expected_samples"] = expected_samples
    if condition["evaluator"] == "hf":
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError("Invalid shard settings")
        config["num_shards"] = num_shards
        config["shard_index"] = shard_index
    return manifest, condition, data_file, expected_samples | 0


def evaluate_pilot_acceptance(
    metrics: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    evaluator: str,
    expected_zero_reasoning_tokens: bool,
) -> dict[str, object]:
    success_rate = float(metrics.get("request_success_rate", 1.0))
    candidate_valid = float(metrics["candidate_valid_rate"])
    prefix = float(metrics["prefix_compliance_rate"])
    if evaluator == "hf":
        generation = dict(metrics.get("generation_stats", {}))
        truncation_count = int(generation.get("reached_max_new_tokens_count", 0))
    else:
        truncation_count = int(metrics.get("truncated_output_count", 0))
    reasoning_tokens = int(metrics.get("total_reasoning_tokens", 0))
    checks = {
        "request_success_rate": {
            "value": success_rate,
            "minimum": float(policy["minimum_request_success_rate"]),
            "passed": success_rate >= float(policy["minimum_request_success_rate"]),
        },
        "candidate_valid_rate": {
            "value": candidate_valid,
            "minimum": float(policy["minimum_candidate_valid_rate"]),
            "passed": candidate_valid >= float(policy["minimum_candidate_valid_rate"]),
        },
        "prefix_compliance_rate": {
            "value": prefix,
            "minimum": float(policy["minimum_prefix_compliance_rate"]),
            "passed": prefix >= float(policy["minimum_prefix_compliance_rate"]),
        },
        "no_truncation": {
            "value": truncation_count,
            "required": bool(policy["require_no_truncation"]),
            "passed": (
                truncation_count == 0
                if bool(policy["require_no_truncation"])
                else True
            ),
        },
        "zero_reasoning_tokens": {
            "value": reasoning_tokens,
            "required": (
                expected_zero_reasoning_tokens
                and bool(policy["require_zero_reasoning_tokens_when_expected"])
            ),
            "passed": (
                reasoning_tokens == 0
                if expected_zero_reasoning_tokens
                and bool(policy["require_zero_reasoning_tokens_when_expected"])
                else True
            ),
        },
    }
    return {
        "passed": all(bool(value["passed"]) for value in checks.values()),
        "checks": checks,
    }


def run_condition(
    manifest_path: Path | str,
    condition_id: str,
    phase: str,
    *,
    output_dir: Path | str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    pilot_gate: Path | str = DEFAULT_PILOT_GATE,
    concurrency: int | None = None,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    if phase == "full":
        validate_pilot_gate(
            manifest_path,
            pilot_gate,
            condition_id=condition_id,
        )
    manifest, condition, data_file, expected_samples = resolve_condition(
        manifest_path,
        condition_id,
        phase,
        output_dir=output_dir,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else Path(str(manifest["artifact_root"])) / phase / condition_id
    )
    config = dict(condition["parameters"])
    config.update(
        {
            "data_file": str(data_file),
            "output_dir": str(resolved_output),
            "seed": int(config.get("seed", manifest["seed"])),
        }
    )
    runtime_overrides: dict[str, object] = {}
    if concurrency is not None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        config["concurrency"] = concurrency
        runtime_overrides["concurrency"] = concurrency
    if _sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("Experiment manifest changed while resolving condition")
    data_sha256 = _sha256(data_file)
    evaluator = str(condition["evaluator"])
    if evaluator == "trapi":
        metrics = evaluate_trapi(config)
    elif evaluator == "openai":
        config["expected_samples"] = expected_samples
        metrics = evaluate_openai(config)
    elif evaluator == "hf":
        config["num_shards"] = num_shards
        config["shard_index"] = shard_index
        from new_interest.evaluation.evaluate_candidate_qwen import evaluate_qwen

        metrics = evaluate_qwen(config)
    else:
        raise ValueError(f"Unknown evaluator {evaluator!r}")

    if _sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("Experiment manifest changed during evaluation")
    if _sha256(data_file) != data_sha256:
        raise RuntimeError("Candidate data changed during evaluation")

    resolved_output.mkdir(parents=True, exist_ok=True)
    identity = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "condition_id": condition_id,
        "phase": phase,
        "evaluator": evaluator,
        "strategy": condition["strategy"],
        "task_type": condition["task_type"],
        "manifest_path": portable_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "data_file": portable_path(data_file),
        "data_sha256": data_sha256,
        "expected_samples": expected_samples,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "runtime_overrides": runtime_overrides,
    }
    (resolved_output / "experiment_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if phase == "pilot":
        acceptance = evaluate_pilot_acceptance(
            metrics,
            dict(manifest["pilot_acceptance"]),
            evaluator=evaluator,
            expected_zero_reasoning_tokens=bool(
                config.get("expected_zero_reasoning_tokens", False)
            ),
        )
        (resolved_output / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not acceptance["passed"]:
            raise RuntimeError(
                f"Pilot acceptance failed for {condition_id}: {acceptance['checks']}"
            )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one static candidate-selection condition."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--pilot-gate", type=Path, default=DEFAULT_PILOT_GATE)
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override evaluator concurrency without changing the frozen manifest.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = run_condition(
        args.manifest,
        args.condition,
        args.phase,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        pilot_gate=args.pilot_gate,
        concurrency=args.concurrency,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
