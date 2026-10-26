from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from new_interest.experiments.run_static_candidate_condition import (
    evaluate_pilot_acceptance,
    resolve_condition,
    run_condition,
    validate_pilot_gate,
)
from new_interest.experiments.finalize_static_candidate_pilots import (
    finalize_static_candidate_pilots,
)
import pytest


MANIFEST = Path("configs/candidate/static_user_disjoint_v1.json")


def test_manifest_has_expected_condition_matrix() -> None:
    manifest = json.loads(MANIFEST.read_text())
    conditions = manifest["conditions"]
    assert len(conditions) == 18
    assert len({row["id"] for row in conditions}) == 18
    assert sum(row["strategy"] == "uniform" for row in conditions) == 14
    assert sum(
        row["strategy"] == "train_frequency_matched" for row in conditions
    ) == 4
    assert sum(row["evaluator"] == "trapi" for row in conditions) == 10
    assert sum(row["evaluator"] == "openai" for row in conditions) == 2
    assert sum(row["evaluator"] == "hf" for row in conditions) == 6
    for row in conditions:
        if row["evaluator"] == "hf":
            assert row["parameters"]["do_sample"] is False
            assert row["parameters"].get("presence_penalty") is None
            assert row["parameters"]["max_new_tokens"] == 64


def test_resolves_shared_pilot_and_full_data(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    for phase, strategy, task in [
        ("pilot", "uniform", "behavior_only"),
        ("full", "train_frequency_matched", "interest_only"),
    ]:
        root = tmp_path / phase
        manifest["data"][f"{phase}_root"] = str(root)
        data_file = root / strategy / task / manifest["data"][f"{phase}_filename"]
        data_file.parent.mkdir(parents=True)
        pq.write_table(pa.table({"row": range(manifest["data"][f"{phase}_samples"])}), data_file)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    _, condition, pilot, pilot_samples = resolve_condition(
        manifest_path, "gpt56_medium_behavior_uniform", "pilot"
    )
    assert condition["task_type"] == "behavior_only"
    assert pilot.name == "dev_pilot_1000.parquet"
    assert pilot_samples == 1000

    _, condition, full, full_samples = resolve_condition(
        manifest_path, "qwen35_interest_frequency_matched", "full"
    )
    assert condition["strategy"] == "train_frequency_matched"
    assert full.name == "dev_expanded.parquet"
    assert full_samples == 37817


def test_pilot_acceptance_enforces_all_gates() -> None:
    policy = {
        "minimum_request_success_rate": 0.99,
        "minimum_candidate_valid_rate": 0.99,
        "minimum_prefix_compliance_rate": 0.99,
        "require_no_truncation": True,
        "require_zero_reasoning_tokens_when_expected": True,
    }
    metrics = {
        "request_success_rate": 1.0,
        "candidate_valid_rate": 0.995,
        "prefix_compliance_rate": 0.999,
        "total_reasoning_tokens": 0,
        "generation_stats": {"reached_max_new_tokens_count": 0},
    }
    accepted = evaluate_pilot_acceptance(
        metrics,
        policy,
        evaluator="hf",
        expected_zero_reasoning_tokens=True,
    )
    assert accepted["passed"]

    metrics["candidate_valid_rate"] = 0.98
    rejected = evaluate_pilot_acceptance(
        metrics,
        policy,
        evaluator="hf",
        expected_zero_reasoning_tokens=True,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["candidate_valid_rate"]["passed"]


def test_full_gate_requires_exact_manifest_and_condition_set(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    import hashlib

    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "conditions": {
                    row["id"]: {"passed": True} for row in manifest["conditions"]
                },
            }
        )
    )
    assert validate_pilot_gate(MANIFEST, gate)["passed"] is True
    value = json.loads(gate.read_text())
    value["conditions"].pop(next(iter(value["conditions"])))
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="condition set"):
        validate_pilot_gate(MANIFEST, gate)


def test_condition_scoped_gate_certifies_only_listed_condition(
    tmp_path: Path,
) -> None:
    import hashlib

    manifest = json.loads(MANIFEST.read_text())
    first = str(manifest["conditions"][0]["id"])
    second = str(manifest["conditions"][1]["id"])
    gate = tmp_path / "subset-gate.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "scope": "condition_subset",
                "manifest_sha256": hashlib.sha256(
                    MANIFEST.read_bytes()
                ).hexdigest(),
                "conditions": {first: {"passed": True}},
            }
        )
    )

    assert validate_pilot_gate(
        MANIFEST,
        gate,
        condition_id=first,
    )["passed"] is True
    with pytest.raises(ValueError, match="is not certified"):
        validate_pilot_gate(MANIFEST, gate, condition_id=second)
    with pytest.raises(ValueError, match="condition set"):
        validate_pilot_gate(MANIFEST, gate)


def test_run_rejects_data_changed_during_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_file = data_root / "uniform/behavior_only/dev_pilot_1000.parquet"
    data_file.parent.mkdir(parents=True)
    pq.write_table(pa.table({"selection_id": ["before"]}), data_file)
    manifest = {
        "experiment_id": "fixture",
        "seed": 17,
        "data": {
            "pilot_root": str(data_root),
            "pilot_filename": "dev_pilot_1000.parquet",
            "pilot_samples": 1,
            "full_root": str(data_root),
            "full_filename": "dev_pilot_1000.parquet",
            "full_samples": 1,
        },
        "artifact_root": str(tmp_path / "artifacts"),
        "pilot_acceptance": {
            "minimum_request_success_rate": 0.99,
            "minimum_candidate_valid_rate": 0.99,
            "minimum_prefix_compliance_rate": 0.99,
            "require_no_truncation": True,
            "require_zero_reasoning_tokens_when_expected": True,
        },
        "conditions": [
            {
                "id": "fixture_condition",
                "evaluator": "trapi",
                "strategy": "uniform",
                "task_type": "behavior_only",
                "parameters": {},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def mutate_data(_config: dict[str, object]) -> dict[str, object]:
        pq.write_table(pa.table({"selection_id": ["after"]}), data_file)
        return {
            "samples": 1,
            "request_success_rate": 1.0,
            "candidate_valid_rate": 1.0,
            "prefix_compliance_rate": 1.0,
            "total_reasoning_tokens": 0,
        }

    monkeypatch.setattr(
        "new_interest.experiments.run_static_candidate_condition.evaluate_trapi",
        mutate_data,
    )
    with pytest.raises(RuntimeError, match="Candidate data changed"):
        run_condition(manifest_path, "fixture_condition", "pilot")


def test_finalizes_pilot_artifacts_into_gate_certificate(tmp_path: Path) -> None:
    data_root = tmp_path / "pilot-data"
    data_file = data_root / "uniform/behavior_only/dev_pilot_1000.parquet"
    data_file.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"selection_id": ["s1"], "answer_category": ["A"]}),
        data_file,
    )
    artifact_root = tmp_path / "models"
    manifest = {
        "experiment_id": "fixture",
        "seed": 17,
        "data": {
            "pilot_root": str(data_root),
            "pilot_filename": "dev_pilot_1000.parquet",
            "pilot_samples": 1,
            "full_root": str(data_root),
            "full_filename": "dev_pilot_1000.parquet",
            "full_samples": 1,
        },
        "artifact_root": str(artifact_root),
        "pilot_acceptance": {},
        "conditions": [
            {
                "id": "fixture_condition",
                "evaluator": "trapi",
                "strategy": "uniform",
                "task_type": "behavior_only",
                "parameters": {},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    import hashlib

    root = artifact_root / "pilot/fixture_condition"
    root.mkdir(parents=True)
    (root / "acceptance.json").write_text(json.dumps({"passed": True}))
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "samples": 1,
                "macro_category_accuracy": 1.0,
                "candidate_valid_rate": 1.0,
                "prefix_compliance_rate": 1.0,
            }
        )
    )
    (root / "resolved_config.json").write_text("{}")
    pq.write_table(pa.table({"selection_id": ["s1"]}), root / "predictions.parquet")
    (root / "experiment_identity.json").write_text(
        json.dumps(
            {
                "condition_id": "fixture_condition",
                "phase": "pilot",
                "evaluator": "trapi",
                "strategy": "uniform",
                "task_type": "behavior_only",
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
                "expected_samples": 1,
            }
        )
    )
    gate_path = tmp_path / "gate.json"
    finalize_static_candidate_pilots(manifest_path, None, gate_path)
    assert validate_pilot_gate(manifest_path, gate_path)["condition_count"] == 1


def test_finalizer_rejects_unpaired_behavior_interest_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import importlib

    manifest = {
        "experiment_id": "fixture",
        "conditions": [{"id": "behavior"}, {"id": "interest"}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pilot_root = tmp_path / "pilots"
    data_paths: dict[str, Path] = {}

    for condition_id, task_type, selection_id in (
        ("behavior", "behavior_only", "behavior-row"),
        ("interest", "interest_only", "interest-row"),
    ):
        data_path = tmp_path / f"{condition_id}.parquet"
        pq.write_table(pa.table({"selection_id": [selection_id]}), data_path)
        data_paths[condition_id] = data_path
        root = pilot_root / condition_id
        root.mkdir(parents=True)
        (root / "acceptance.json").write_text(json.dumps({"passed": True}))
        (root / "metrics.json").write_text(
            json.dumps(
                {
                    "samples": 1,
                    "macro_category_accuracy": 1.0,
                    "candidate_valid_rate": 1.0,
                    "prefix_compliance_rate": 1.0,
                }
            )
        )
        (root / "resolved_config.json").write_text("{}")
        pq.write_table(
            pa.table({"selection_id": [selection_id]}),
            root / "predictions.parquet",
        )
        (root / "experiment_identity.json").write_text(
            json.dumps(
                {
                    "condition_id": condition_id,
                    "phase": "pilot",
                    "evaluator": "fixture",
                    "strategy": "uniform",
                    "task_type": task_type,
                    "manifest_sha256": manifest_hash,
                    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "expected_samples": 1,
                }
            )
        )

    def fake_resolve(
        _manifest_path: Path, condition_id: str, _phase: str
    ) -> tuple[dict[str, object], dict[str, object], Path, int]:
        task_type = "behavior_only" if condition_id == "behavior" else "interest_only"
        condition = {
            "id": condition_id,
            "evaluator": "fixture",
            "strategy": "uniform",
            "task_type": task_type,
        }
        return manifest, condition, data_paths[condition_id], 1

    module = importlib.import_module(
        "new_interest.experiments.finalize_static_candidate_pilots"
    )
    monkeypatch.setattr(module, "resolve_condition", fake_resolve)
    with pytest.raises(ValueError, match="Behavior/Interest"):
        finalize_static_candidate_pilots(
            manifest_path, pilot_root, tmp_path / "gate.json"
        )
