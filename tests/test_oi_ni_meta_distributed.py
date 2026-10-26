from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from new_interest.experiments.oi_ni_meta_distributed import (
    EXECUTION_SPEC, ENGINE_FILES, LEGACY_FILES, GateTolerances, compare_tensors,
    digest_json, owned_evaluation_shards, owned_positions, source_receipt,
    tensor_digest, timing_totals, validate_compute_budget, validate_prediction_rows, validate_selection_lineage,
    DistributedContext, primary_phase, validate_gate_evidence,
)
from new_interest.experiments import oi_ni_meta as core


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fake_source(tmp_path):
    config = {
        "inner_alpha_candidates": [0.0001, 0.001, 0.01], "calibration_steps": 27,
        "gate_steps": 2, "data_metadata_sha256": "d" * 64,
        "pilot_steps": 540, "max_compute_multiplier": 5,
    }
    config_path = tmp_path / "configs/protocol.json"
    write_json(config_path, config)
    manifest_files = {"configs/protocol.json": core.sha256_file(config_path)}
    for relative in LEGACY_FILES.values():
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"frozen {relative}")
        manifest_files[relative] = core.sha256_file(file)
    for relative in ENGINE_FILES:
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"new {relative}")
    write_json(tmp_path / "source_bundle_manifest.json", {"files": manifest_files})
    source = source_receipt(tmp_path, config_path, {"metadata.json": "d" * 64})
    return config_path, config, source


def fake_selection(tmp_path, config, source):
    results = [{"mode": "meta", "steps": 27, "alpha": alpha,
                "validation_metrics": {"interest_macro_recall_at_10": recall,
                                       "interest_macro_ndcg_at_10": 0.1}}
               for alpha, recall in zip(config["inner_alpha_candidates"], [0.20, 0.23, 0.21])]
    selection = {"source_identity": source["legacy_source_identity"], "validation_only": True,
                 "chosen_alpha": 0.001, "results": results}
    gate = {"accepted": True, "source_identity": source["source_identity"],
            "legacy_source_identity": source["legacy_source_identity"], "phase": "gate", "mode": "meta",
            "steps": 2, "world_size": 4, "execution_spec": source["execution_spec"],
            "gate_tolerances": source["gate_tolerances"]}
    selection_path, gate_path = tmp_path / "selection.json", tmp_path / "gate.json"
    write_json(selection_path, selection); write_json(gate_path, gate)
    lineage = {"accepted": True, "selection_sha256": core.sha256_file(selection_path),
               "gate_receipt_sha256": core.sha256_file(gate_path),
               "parent_source_identity": source["legacy_source_identity"],
               "execution_source_identity": source["source_identity"],
               "protocol_sha256": source["protocol_sha256"],
               "data_metadata_sha256": config["data_metadata_sha256"]}
    lineage_path = tmp_path / "lineage.json"
    write_json(lineage_path, lineage)
    return selection_path, lineage_path, gate_path


def test_role_partition_keeps_one_original_global_episode():
    ownership = [owned_positions(16, rank) for rank in range(4)]
    assert all(len(values) == 4 for values in ownership)
    assert sorted(sum(ownership, [])) == list(range(16))
    assert len(set(sum(ownership, []))) == 16
    assert ownership[2] == [2, 6, 10, 14]
    with pytest.raises(ValueError):
        owned_positions(16, 4)


def test_evaluation_partition_preserves_original_global_shards_and_batches():
    shards = [owned_evaluation_shards(31392, rank) for rank in range(4)]
    assert list(map(len, shards)) == [31, 31, 31, 30]
    ordered = sorted(sum(shards, []))
    assert ordered[0] == (0, 256) and ordered[-1] == (31232, 31392)
    assert [i for start, stop in ordered for i in range(start, stop)] == list(range(31392))
    assert all(start % 256 == 0 and start % 8 == 0 for start, _ in ordered)
    assert owned_evaluation_shards(64, 1) == []


def test_source_preserves_legacy_identity_but_binds_distinct_engine(tmp_path):
    config_path, _, source = fake_source(tmp_path)
    expected_legacy = hashlib.sha256(json.dumps(source["legacy_files"], sort_keys=True).encode()).hexdigest()
    assert source["legacy_source_identity"] == expected_legacy
    assert source["source_identity"] != expected_legacy
    assert source["execution_spec"]["rng_planning_mode"] == "real_preview"
    assert "ordered" in source["execution_spec"]["gradient_reduction"]
    p = tmp_path / ENGINE_FILES[0]
    p.write_text(p.read_text() + " changed engine")
    changed = source_receipt(tmp_path, config_path, {"metadata.json": "d" * 64})
    assert changed["legacy_source_identity"] == expected_legacy
    assert changed["source_identity"] != source["source_identity"]


def test_source_refuses_a_changed_legacy_core(tmp_path):
    config_path, _, _ = fake_source(tmp_path)
    (tmp_path / next(iter(LEGACY_FILES.values()))).write_text("changed")
    with pytest.raises(ValueError, match="Frozen legacy source changed"):
        source_receipt(tmp_path, config_path, {"metadata.json": "d" * 64})


def test_legacy_selection_requires_explicit_lineage_and_keeps_bytes(tmp_path):
    _, config, source = fake_source(tmp_path)
    selection, lineage, gate = fake_selection(tmp_path, config, source)
    before = selection.read_bytes()
    alpha, binding = validate_selection_lineage(selection, lineage, gate, source, config)
    assert alpha == 0.001 and binding["selection_sha256"] == core.sha256_file(selection)
    assert selection.read_bytes() == before
    changed = json.loads(lineage.read_text()); changed["accepted"] = False; write_json(lineage, changed)
    with pytest.raises(ValueError, match="explicitly accepted"):
        validate_selection_lineage(selection, lineage, gate, source, config)


def test_relabelled_selection_or_changed_gate_cannot_bypass_lineage(tmp_path):
    _, config, source = fake_source(tmp_path)
    selection, lineage, gate = fake_selection(tmp_path, config, source)
    changed = json.loads(selection.read_text()); changed["source_identity"] = source["source_identity"]
    write_json(selection, changed)
    binding = json.loads(lineage.read_text()); binding["selection_sha256"] = core.sha256_file(selection)
    write_json(lineage, binding)
    with pytest.raises(ValueError, match="legacy validation-only"):
        validate_selection_lineage(selection, lineage, gate, source, config)


def test_gate_must_exercise_hvp_under_the_exact_frozen_tolerances(tmp_path):
    _, config, source = fake_source(tmp_path)
    selection, lineage, gate = fake_selection(tmp_path, config, source)
    changed = json.loads(gate.read_text()); changed["mode"] = "mixed"; write_json(gate, changed)
    binding = json.loads(lineage.read_text()); binding["gate_receipt_sha256"] = core.sha256_file(gate)
    write_json(lineage, binding)
    with pytest.raises(ValueError, match="compatibility gate"):
        validate_selection_lineage(selection, lineage, gate, source, config)


def test_selection_recomputes_frozen_metric_tie_break(tmp_path):
    _, config, source = fake_source(tmp_path)
    selection, lineage, gate = fake_selection(tmp_path, config, source)
    changed = json.loads(selection.read_text()); changed["chosen_alpha"] = 0.01
    write_json(selection, changed)
    binding = json.loads(lineage.read_text()); binding["selection_sha256"] = core.sha256_file(selection)
    write_json(lineage, binding)
    with pytest.raises(ValueError, match="winner"):
        validate_selection_lineage(selection, lineage, gate, source, config)


def test_prediction_validation_uses_global_indices_and_fixed_tie_break():
    labels = [f"L{i:03d}" for i in range(108)]
    records = [{"impression_id": str(i), "user_id": f"u{i}", "acceptable_subcategories": [labels[1]]}
               for i in range(3)]
    tokens = [[1,2,3], [4,5,6], [7,8,9]]
    score = np.full(108, -math_log(108))
    row = {"row_index": 2, "impression_id": "2", "acceptable_subcategories": [labels[1]],
           "input_sha256": core.input_digest(records[2], tokens[2]), "ranked_indices": list(range(20)),
           "ranked_labels": labels[:20], "ranked_scores": score[:20].tolist(),
           "all_sid_log_probabilities": score.tolist()}
    validate_prediction_rows([row], [2], records, tokens, labels)
    row["ranked_indices"] = list(reversed(range(20)))
    with pytest.raises(ValueError, match="tie rule"):
        validate_prediction_rows([row], [2], records, tokens, labels)


def math_log(value):
    import math
    return math.log(value)


def test_comparison_reports_global_error_and_digest_tracks_dtype():
    torch = pytest.importorskip("torch")
    expected = {"a": torch.tensor([1., 2.]), "b": torch.tensor(3.)}
    actual = {"a": torch.tensor([1.00001, 2.]), "b": torch.tensor(3.)}
    result = compare_tensors(actual, expected, atol=1e-4, rtol=0., relative_l2_limit=1e-5)
    assert result["accepted"] and result["relative_l2"] > 0
    assert not compare_tensors(actual, expected, atol=1e-4, rtol=0., relative_l2_limit=1e-8)["accepted"]
    assert tensor_digest(expected) != tensor_digest({name: value.double() for name,value in expected.items()})
    with pytest.raises(FloatingPointError):
        compare_tensors({"a": torch.tensor(float("nan"))}, {"a": torch.tensor(0.)}, atol=0, rtol=0)


def test_engine_protocol_has_fixed_global_batch_and_predeclared_numerical_gate():
    assert EXECUTION_SPEC["global_support"] + EXECUTION_SPEC["global_query"] == 32
    assert EXECUTION_SPEC["microbatch_size"] == 1 and EXECUTION_SPEC["world_size"] == 4
    t = GateTolerances()
    assert (t.gradient_atol, t.gradient_rtol, t.gradient_relative_l2) == (2e-6, 2e-5, 2e-5)
    assert (t.update_atol, t.update_rtol) == (2e-8, 2e-6)
    assert digest_json({"b": 1, "a": 2}) == digest_json({"a": 2, "b": 1})


def test_compute_budget_is_new_engine_pilot_time_bound_and_step_bounded(tmp_path):
    _, config, source = fake_source(tmp_path)
    path = tmp_path / "budget.json"
    value = {"source_identity": source["source_identity"], "world_size": 4,
             "timing_basis": "four_rank_pilot_training_seconds", "steps": 2700}
    write_json(path, value)
    assert validate_compute_budget(path, source, config)[0] == 2700
    for field, invalid in [("steps", 2701), ("steps", 539), ("steps", True),
                           ("source_identity", source["legacy_source_identity"]),
                           ("world_size", 1), ("timing_basis", "engineering_gate_including_serial_reference")]:
        write_json(path, {**value, field: invalid})
        with pytest.raises(ValueError):
            validate_compute_budget(path, source, config)


def test_timing_retains_committed_update_cost_and_separate_retry_overhead(tmp_path):
    path = tmp_path / "training.jsonl"
    rows = [{"step": 1, "seconds": 10., "timing_includes_serial_gate": False},
            {"step": 2, "seconds": 20., "timing_includes_serial_gate": False},
            {"step": 2, "seconds": 21., "timing_includes_serial_gate": False}]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = timing_totals(path, 2, 4, phase="pilot")
    assert result["training_wall_seconds"] == 31
    assert result["training_gpu_seconds"] == 124
    assert result["recorded_step_gpu_seconds_including_retries"] == 204
    with pytest.raises(ValueError, match="cover every"):
        timing_totals(path, 3, 4, phase="pilot")
    with pytest.raises(ValueError, match="mixed-scope"):
        timing_totals(path, 2, 4, phase="gate")


def test_rank_zero_failure_is_broadcast_before_peers_enter_next_collective(monkeypatch):
    from new_interest.experiments import oi_ni_meta_distributed as engine
    delivered = []
    def broadcast(value, context):
        if context.primary:
            delivered.append(value)
        return delivered[-1]
    monkeypatch.setattr(engine, "_broadcast", broadcast)
    with pytest.raises(RuntimeError, match="Rank-zero cache failed: OSError: disk"):
        with primary_phase(DistributedContext(0,4,0), "cache"):
            raise OSError("disk")
    with pytest.raises(RuntimeError, match="Rank-zero cache failed: OSError: disk"):
        with primary_phase(DistributedContext(1,4,1), "cache"):
            pass
    assert delivered == [{"ok": False, "error": {"type": "OSError", "message": "disk", "phase": "cache"}}]


def test_final_cursor_does_not_replace_missing_gate_proof(tmp_path):
    # Mimic a retry after the final optimizer checkpoint; its existence is not
    # evidence that every serial/four-rank comparison was persisted.
    write_json(tmp_path / "checkpoints/checkpoint-000002/manifest.json", {"identity": "run", "files": {}})
    write_json(tmp_path / "success.json", {"identity": "run", "steps": 2, "accepted": True})
    write_json(tmp_path / "model_audit.json", {"memory_bytes": 100, "trainable_parameters": {"p": [1]},
                                               "initial_trainable_sha256": "initial"})
    write_json(tmp_path / "token_audit.json", {"target_in_input": False, "over_length": 0, "oi_max": 10, "ni_max": 12})
    write_json(tmp_path / "long_sequence_hvp_audit.json", {"identity": "run", "parameter_update_performed": False,
        "scope": "gradient-only engineering gate; not a method score", "world_size": 4, "owner_rank": 0,
        "global_support": 1, "global_query": 1, "empty_ranks": [1,2,3], "oi_tokens": 10, "ni_tokens": 12,
        "hvp_norm": 1., "seconds": 1., "peak_gpu_bytes": 50})
    write_json(tmp_path / "execution_audits/initialization-000000.json", {"identity": "run", "event": "initialization",
        "step": 0, "world_size": 4, "accepted": True,
        "ranks": [{"rank": rank, "parameter_sha256": "initial", "tensor_sha256": "initial-typed"} for rank in range(4)]})
    from dataclasses import asdict
    with pytest.raises(FileNotFoundError, match="before-update-000001"):
        validate_gate_evidence(tmp_path, "run", {"gate_tolerances": asdict(GateTolerances()), "source_identity": "source"},
                               {"gate_steps": 2})
