from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments import read_oi_ni_meta_distributed as reader


def h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value, compact=False):
    options = {"sort_keys": True}
    if compact:
        options["separators"] = (",", ":")
    return h(json.dumps(value, **options))


def comparison(shapes: dict[str, list[int]], kind: str, values=None):
    prefix = "loss" if kind == "loss" else "update" if kind == "update" else "gradient"
    rows = []
    for name in sorted(shapes):
        shape, scalar = shapes[name], (values or {}).get(name, 1.0)
        n = math.prod(shape)
        rows.append({"name": name, "shape": shape, "numel": n, "accepted": True,
                     "difference_squared_l2": 0.0, "reference_squared_l2": n * scalar ** 2,
                     "max_tolerance_ratio": 0.0, "max_abs_error": 0.0, "relative_l2": 0.0,
                     **({"actual_value": scalar, "reference_value": scalar} if n == 1 else {})})
    return {"accepted": True, "atol": reader.TOLERANCES[prefix + "_atol"],
            "rtol": reader.TOLERANCES[prefix + "_rtol"], "relative_l2": 0.0,
            "relative_l2_limit": reader.TOLERANCES["gradient_relative_l2"] if kind == "gradients" else None,
            "tensors": rows}


def rank_event(root, identity, event, step, count=2):
    ranks = []
    for rank in range(4):
        row = {"rank": rank, "local_rank": rank, "parameter_sha256": h(f"parameters-{step}"),
               "tensor_sha256": h(f"tensors-{step}")}
        if event != "fresh_reload":
            row.update(optimizer_step_min=step, optimizer_step_max=step,
                       optimizer_state_count=count if step else 0)
        else:
            row["reload_max_abs_error"] = 0.0
        ranks.append(row)
    path = root / "execution_audits" / f"{event}-{step:06d}.json"
    dump(path, {"identity": identity, "event": event, "step": step, "world_size": 4,
                "accepted": True, "ranks": ranks})
    return path


@pytest.fixture
def gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    root, data = tmp_path / "run", tmp_path / "data"
    root.mkdir(); data.mkdir()
    (data / "frozen.bin").write_bytes(b"frozen data")
    metadata = {"artifacts": {"frozen.bin": {"sha256": file_hash(data / "frozen.bin")}}}
    dump(data / "metadata.json", metadata)
    config = {"world_size": 1, "seed": 42, "gate_steps": 2, "pilot_steps": 540,
              "calibration_steps": 27, "max_compute_multiplier": 5, "inner_alpha_candidates": [0.0001, 0.001, 0.01],
              "data_metadata_sha256": file_hash(data / "metadata.json")}
    dump(root / "lineage" / "protocol.json", config)
    legacy_files = {"metadata.json": file_hash(data / "metadata.json"),
                    "frozen.bin": file_hash(data / "frozen.bin"),
                    "protocol": file_hash(root / "lineage" / "protocol.json"), "worker_bundle": h("core manifest")}
    source = {"version": "oi-ni-meta-four-rank-v1", "legacy_files": legacy_files,
              "engine_files": {name: h(name) for name in reader.ENGINE_FILES},
              "execution_spec": copy.deepcopy(reader.EXECUTION), "gate_tolerances": dict(reader.TOLERANCES),
              "protocol_sha256": legacy_files["protocol"], "core_manifest_sha256": legacy_files["worker_bundle"]}
    source["legacy_source_identity"] = digest(legacy_files)
    source["source_identity"] = digest({key: source[key] for key in
        ("legacy_source_identity", "engine_files", "execution_spec", "gate_tolerances")}, compact=True)
    dump(root / "source_identity.json", source)
    resolved = {"config": config, "steps": 2, "phase": "gate", "mode": "meta", "alpha": 0.001,
                "source_identity": source["source_identity"], "legacy_source_identity": source["legacy_source_identity"],
                "execution_spec": copy.deepcopy(reader.EXECUTION), "eval_dev": False, "selection_binding": {}}
    identity = digest(resolved)
    resolved["identity"] = identity
    dump(root / "resolved_config.json", resolved)
    shapes = {"lora_A": [2], "sid": [2]}
    dump(root / "model_audit.json", {"world_size": 4, "source_identity": source["source_identity"], "trainable_parameters": shapes,
          "initial_trainable_sha256": h("parameters-0")})
    rank_event(root, identity, "initialization", 0)
    rank_event(root, identity, "long_sequence_no_update", 0)
    dump(root / "long_sequence_hvp_audit.json", {"world_size": 4, "global_support": 1,
          "global_query": 1, "owner_rank": 0, "empty_ranks": [1, 2, 3]})
    dump(root / "token_audit.json", {"fixture": True})
    logs = []
    for step in (1, 2):
        rank_event(root, identity, "after_update", step)
        logs.append({"step": step, "mode": "meta", "seconds": 1.0, "global_support": 16, "global_query": 16,
                     "timing_includes_serial_gate": True,
                     "gradient_sha256": h(f"gradient-{step}"), "parameter_sha256": h(f"parameters-{step}"),
                     "rng_plan": {"version": "serial-real-forward-rng-v1", "planning_mode": "real_preview",
                         "episode_seed": 42 + 104729 * (step - 1), "caller_rng_preserved": True, "device_type": "cuda",
                         "role_chunk_counts": {"oi": 16, "ni": 16}, "preview_forward_count": 32,
                         "state_sha256": h(f"rng-{step}")}})
        dump(root / "equivalence" / f"before-update-{step:06d}.json", {
            "identity": identity, "step": step, "kind": "before_update",
            "source_identity": source["source_identity"],
            "loss": comparison({"loss": []}, "loss"),
            "norms": comparison({"inner": [], "hvp": []}, "norms", {"inner": 3.0, "hvp": 2.0}),
            "gradients": comparison(shapes, "gradients")})
        dump(root / "equivalence" / f"after-update-{step:06d}.json", {
            **comparison(shapes, "update"), "identity": identity, "step": step, "kind": "after_update",
            "source_identity": source["source_identity"]})
    (root / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs))
    checkpoint = root / "checkpoints" / "checkpoint-000002"
    checkpoint.mkdir(parents=True)
    dump(checkpoint / "state.json", {"identity": identity, "world_size": 4, "step": 2,
          "trainable_sha256": h("parameters-2")})
    torch.save({"world_size": 4, "legacy_rng_is_rank0_only": True,
                "rng_by_rank": [{"python": None, "numpy": None, "torch": None, "cuda": None} for _ in range(4)]},
               checkpoint / "training_state.pt")
    dump(checkpoint / "manifest.json", {"identity": identity, "files": {
        p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()}})
    rank_event(root, identity, "checkpoint", 2)
    rank_event(root, identity, "final_deployed", 2)
    rank_event(root, identity, "fresh_reload", 2)
    dump(root / "fresh_reload_audit.json", {"world_size": 4, "trainable_sha256": h("parameters-2"),
          "max_abs_error_by_rank": [0.0] * 4})
    directory = root / "validation-000002"
    directory.mkdir()
    for start in range(0, 1024, 256):
        path = directory / f"predictions-{start:06d}.parquet"
        pq.write_table(pa.table({"row_index": range(start, start + 256)}), path)
        dump(path.with_suffix(".json"), {"identity": identity, "world_size": 4,
             "rank": start // 256, "rows": 256, "sha256": file_hash(path)})
    dump(directory / "metrics.json", {"world_size": 4, "global_shard_size": 256, "batch_size": 8})
    success = {"identity": identity, "steps": 2, "world_size": 4, "phase": "gate", "mode": "meta", "alpha": 0.001,
               "source_identity": source["source_identity"], "legacy_source_identity": source["legacy_source_identity"],
               "checkpoint_relative": "checkpoints/checkpoint-000002",
               "training_wall_seconds": 2.0, "training_gpu_seconds": 8.0,
               "recorded_step_wall_seconds_including_retries": 2.0,
               "recorded_step_gpu_seconds_including_retries": 8.0,
               "median_step_seconds": 1.0, "timing_scope": "engineering_gate_including_serial_reference",
               "timing_includes_planning_and_communication": True,
               "timing_excludes_checkpoint_eval_setup_and_unfinished_steps": True}
    dump(root / "success.json", success)
    dump(root / "compatibility_gate.json", {**success, "accepted": True,
          "execution_spec": copy.deepcopy(reader.EXECUTION), "gate_tolerances": dict(reader.TOLERANCES),
          "evidence_sha256": {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*") if p.is_file()}})
    base_calls = []

    def base_read(run, dataset, **kwargs):
        base_calls.append(kwargs)
        assert run == root and dataset == data and kwargs["scope"] == "gate"
        assert kwargs["expected_topology"] == {"validation": (1024, 1024, None)}
        assert json.loads((root / "lineage" / "protocol.json").read_text())["world_size"] == 1
        return {"identity": identity, "steps": 2, "checkpoint": {"trainable_sha256": h("parameters-2")},
                "validation": {"samples": 1024}, "status": "PASS"}

    monkeypatch.setattr(reader.legacy, "read_run", base_read)
    monkeypatch.setattr(reader, "saved_replica_digest", lambda *args: h("tensors-2"))
    return root, data, source, resolved, shapes, base_calls


def test_gate_reuses_legacy_reader_and_checks_four_rank_execution(gate) -> None:
    root, data, _, _, _, calls = gate
    receipt = reader.read_distributed_run(root, data, scope="gate")
    assert len(calls) == 1
    assert receipt["status"] == "PASS" and receipt["world_size"] == 4
    assert receipt["evaluation_ownership"]["validation"]["rows_by_rank"] == {str(i): 256 for i in range(4)}
    assert len(receipt["step_evidence"]["gate_comparisons"]) == 2


@pytest.mark.parametrize("fault", ["ratio", "global_l2", "threshold", "missing_tensor", "scalar", "shape"])
def test_numeric_comparison_never_trusts_accepted_flag(fault: str) -> None:
    shapes = {"x": [2]}
    report = comparison(shapes, "gradients")
    if fault == "ratio":
        report["tensors"][0]["max_tolerance_ratio"] = 1.01
    elif fault == "global_l2":
        report["relative_l2"] = 0.1
    elif fault == "threshold":
        report["atol"] = 1.0
    elif fault == "missing_tensor":
        report["tensors"] = []
    elif fault == "scalar":
        shapes = {"loss": []}
        report = comparison(shapes, "loss")
        report["tensors"][0]["actual_value"] = 2.0
    else:
        report["tensors"][0]["shape"] = [3]
    with pytest.raises(ValueError):
        reader.inspect_comparison(report, shapes, kind="loss" if fault == "scalar" else "gradients")


def test_per_tensor_relative_norm_may_exceed_global_limit_when_absolute_bound_passes() -> None:
    report = comparison({"large": [2], "tiny": [2]}, "gradients")
    tiny = report["tensors"][1]
    tiny.update(difference_squared_l2=2e-14, reference_squared_l2=0.0, max_abs_error=1e-7,
                max_tolerance_ratio=0.05, relative_l2=math.sqrt(2e-14) / 1e-30)
    report["relative_l2"] = 1e-7
    result = reader.inspect_comparison(report, {"large": [2], "tiny": [2]}, kind="gradients")
    assert result["relative_l2"] == pytest.approx(1e-7)


@pytest.mark.parametrize("fault", ["rank_missing", "parameter_digest", "optimizer_step", "optimizer_coverage", "reload_error"])
def test_rank_acceptance_requires_numeric_and_digest_agreement(gate, fault: str) -> None:
    root, _, _, resolved, _, _ = gate
    event = "fresh_reload" if fault == "reload_error" else "after_update"
    path = root / "execution_audits" / f"{event}-000002.json"
    audit = json.loads(path.read_text())
    if fault == "rank_missing":
        audit["ranks"].pop()
    elif fault == "parameter_digest":
        audit["ranks"][1]["parameter_sha256"] = h("different")
    elif fault == "optimizer_step":
        audit["ranks"][1]["optimizer_step_min"] = 1
    elif fault == "optimizer_coverage":
        audit["ranks"][1]["optimizer_state_count"] = 1
    else:
        audit["ranks"][1]["reload_max_abs_error"] = 1e-5
    dump(path, audit)
    with pytest.raises(ValueError):
        reader.inspect_rank_event(path, resolved["identity"], event, 2, 2)


@pytest.mark.parametrize("fault", ["world", "nodes", "reduction", "rng_mode", "source_hash", "legacy_hash", "tolerance", "literal_protocol"])
def test_execution_and_source_lineage_are_independently_bound(gate, fault: str) -> None:
    root, data, source, resolved, _, _ = gate
    if fault == "world":
        source["execution_spec"]["world_size"] = 1
    elif fault == "nodes":
        source["execution_spec"]["nodes"] = 2
    elif fault == "reduction":
        source["execution_spec"]["gradient_reduction"] = "rank_local_meta_average"
    elif fault == "rng_mode":
        source["execution_spec"]["rng_planning_mode"] = "per_chunk_seed"
    elif fault == "source_hash":
        source["engine_files"][next(iter(source["engine_files"]))] = h("changed")
    elif fault == "legacy_hash":
        source["legacy_files"]["worker_bundle"] = h("changed")
    elif fault == "tolerance":
        source["gate_tolerances"]["gradient_rtol"] = 1.0
    else:
        (root / "lineage" / "protocol.json").write_text("{}")
    dump(root / "source_identity.json", source)
    with pytest.raises(ValueError):
        reader.inspect_source(root, data, resolved)


@pytest.mark.parametrize("fault", ["wrong_owner", "missing_shard", "overlapping_rows", "wrong_world"])
def test_evaluation_requires_all_four_unique_global_shard_owners(gate, fault: str) -> None:
    root, _, _, resolved, _, _ = gate
    directory = root / "validation-000002"
    path = directory / "predictions-000256.parquet"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if fault == "wrong_owner":
        receipt["rank"] = 0
    elif fault == "missing_shard":
        path.unlink()
    elif fault == "wrong_world":
        receipt["world_size"] = 1
    else:
        pq.write_table(pa.table({"row_index": range(256)}), path)
        receipt["sha256"] = file_hash(path)
    dump(path.with_suffix(".json"), receipt)
    with pytest.raises(ValueError):
        reader.inspect_shard_owners(directory, 1024, resolved["identity"])


@pytest.mark.parametrize("fault", ["local_batch", "new_seed", "dummy_preview", "missing_step", "comparison_identity"])
def test_step_evidence_preserves_global_batch_and_original_random_stream(gate, fault: str) -> None:
    root, _, _, resolved, shapes, _ = gate
    path = root / "training.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "local_batch":
        rows[0]["global_support"] = 4
    elif fault == "new_seed":
        rows[0]["rng_plan"]["episode_seed"] += 1
    elif fault == "dummy_preview":
        rows[0]["rng_plan"]["planning_mode"] = "dummy"
    elif fault == "missing_step":
        rows.pop()
    else:
        before = root / "equivalence" / "before-update-000001.json"
        report = json.loads(before.read_text())
        report["identity"] = h("wrong run")
        dump(before, report)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        reader.inspect_steps(root, resolved, shapes)


def test_pilot_selection_recomputes_winner_and_binds_verified_gate(gate, tmp_path: Path) -> None:
    gate_root, _, source, resolved, _, _ = gate
    root = tmp_path / "pilot"
    folder = root / "lineage"
    folder.mkdir(parents=True)
    selection = {"source_identity": source["legacy_source_identity"], "validation_only": True,
                 "chosen_alpha": 0.001, "results": [
        {"alpha": alpha, "mode": "meta", "steps": 27, "validation_metrics": {
            "samples": 2160, "supported_interest_count": 108,
            "interest_macro_recall_at_10": 0.1 if alpha == 0.0001 else 0.2,
            "interest_macro_ndcg_at_10": 0.1}} for alpha in (0.0001, 0.001, 0.01)]}
    dump(folder / "selection.json", selection)
    (folder / "compatibility_gate.json").write_bytes((gate_root / "compatibility_gate.json").read_bytes())
    expected = {"selection_sha256": file_hash(folder / "selection.json"),
                "gate_receipt_sha256": file_hash(gate_root / "compatibility_gate.json"),
                "parent_source_identity": source["legacy_source_identity"],
                "execution_source_identity": source["source_identity"], "protocol_sha256": source["protocol_sha256"],
                "data_metadata_sha256": resolved["config"]["data_metadata_sha256"]}
    dump(folder / "selection_lineage.json", {"accepted": True, **expected})
    settings = {**resolved, "alpha": 0.001,
                "selection_binding": {**expected, "lineage_sha256": file_hash(folder / "selection_lineage.json")}}
    verified_gate = {"source_identity": source["source_identity"]}
    reader.inspect_selection(root, settings, source, gate_root, verified_gate)
    # Keep all hashes consistent with a manually changed winner; the actual
    # validation argmax, not any acceptance/hash flag, still has to reject it.
    selection["chosen_alpha"] = 0.01
    dump(folder / "selection.json", selection)
    expected["selection_sha256"] = file_hash(folder / "selection.json")
    dump(folder / "selection_lineage.json", {"accepted": True, **expected})
    settings["alpha"] = 0.01
    settings["selection_binding"] = {**expected, "lineage_sha256": file_hash(folder / "selection_lineage.json")}
    with pytest.raises(ValueError, match="winner"):
        reader.inspect_selection(root, settings, source, gate_root, verified_gate)


@pytest.mark.parametrize("fault", ["empty_rank_path", "long_update", "final_deployed", "reload_summary"])
def test_final_acceptance_checks_the_extra_four_rank_gate_paths(gate, fault: str) -> None:
    root, data, _, _, _, _ = gate
    if fault == "empty_rank_path":
        path = root / "long_sequence_hvp_audit.json"
        value = json.loads(path.read_text())
        value["empty_ranks"] = []
    elif fault == "reload_summary":
        path = root / "fresh_reload_audit.json"
        value = json.loads(path.read_text())
        value["max_abs_error_by_rank"][3] = 0.001
    else:
        event, step = ("long_sequence_no_update", 0) if fault == "long_update" else ("final_deployed", 2)
        path = root / "execution_audits" / f"{event}-{step:06d}.json"
        value = json.loads(path.read_text())
        for row in value["ranks"]:
            row["parameter_sha256"] = h("all ranks agree on wrong state")
    dump(path, value)
    with pytest.raises(ValueError):
        reader.read_distributed_run(root, data, scope="gate")


def test_compute_budget_requires_four_rank_timing_lineage(gate) -> None:
    root, _, source, resolved, _, _ = gate
    path = root / "lineage" / "compute_budget.json"
    budget = {"world_size": 4, "source_identity": source["source_identity"],
              "timing_basis": "four_rank_pilot_training_seconds", "steps": 1080}
    dump(path, budget)
    config = {**resolved, "mode": "mixed", "alpha": 0.0, "steps": 1080,
              "budget_binding": {"steps": 1080, "budget_sha256": file_hash(path)}}
    reader.inspect_compute_budget(root, config, source)
    budget["timing_basis"] = "single_gpu_gate_seconds"
    dump(path, budget)
    config["budget_binding"]["budget_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="four-rank"):
        reader.inspect_compute_budget(root, config, source)


@pytest.mark.parametrize("kind", ["loss", "norms", "gradients", "update"])
def test_independent_reader_accepts_real_fp32_comparison_summaries(kind: str) -> None:
    torch = pytest.importorskip("torch")
    # This import is a test oracle only. The production reader never imports
    # the worker's comparison/reduction code.
    from new_interest.experiments.oi_ni_meta_distributed import compare_tensors

    if kind == "loss":
        expected = {"loss": torch.tensor(2.0)}
    elif kind == "norms":
        expected = {"hvp": torch.tensor(110736.411, dtype=torch.float64), "inner": torch.tensor(3.0)}
    else:
        expected = {"large": torch.linspace(-2.0, 2.0, 256), "zero": torch.zeros(4)}
    actual = {key: value + 1e-8 for key, value in expected.items()}
    prefix = "loss" if kind == "loss" else "update" if kind == "update" else "gradient"
    report = compare_tensors(actual, expected, atol=reader.TOLERANCES[prefix + "_atol"],
                             rtol=reader.TOLERANCES[prefix + "_rtol"],
                             relative_l2_limit=reader.TOLERANCES["gradient_relative_l2"] if kind == "gradients" else None)
    result = reader.inspect_comparison(report, {name: list(value.shape) for name, value in expected.items()}, kind=kind)
    assert result["max_tolerance_ratio"] <= 1.0


def test_saved_replica_digest_binds_native_dtype_and_peft_canonical_names(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    checkpoint = "checkpoints/checkpoint-000002"
    directory = tmp_path / checkpoint / "adapter"
    directory.mkdir(parents=True)
    runtime_names = {
        "causal_lm.base_model.model.q_proj.lora_A.default.weight": [2, 2],
        "causal_lm.base_model.model.embed_tokens.token_adapter.trainable_tokens_delta.default": [108, 4],
    }
    tensors = {"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2),
               "base_model.model.embed_tokens.token_adapter.trainable_tokens_delta": torch.ones(108, 4, dtype=torch.bfloat16)}
    file = directory / "adapter_model.safetensors"
    save_file(tensors, str(file))
    first = reader.saved_replica_digest(tmp_path, checkpoint, runtime_names)
    tensors["base_model.model.embed_tokens.token_adapter.trainable_tokens_delta"] = tensors[
        "base_model.model.embed_tokens.token_adapter.trainable_tokens_delta"].float()
    save_file(tensors, str(file))
    second = reader.saved_replica_digest(tmp_path, checkpoint, runtime_names)
    # Float-cast values are identical; the native-dtype replica fingerprint
    # must nevertheless distinguish this unsupported serialization change.
    assert first != second


@pytest.mark.parametrize("fault", ["one_gpu_accounting", "wrong_scope", "retry_double_count"])
def test_timing_is_recomputed_with_four_gpu_cost_and_retry_scope(gate, fault: str) -> None:
    root, _, _, resolved, _, _ = gate
    success = json.loads((root / "success.json").read_text())
    if fault == "one_gpu_accounting":
        success["training_gpu_seconds"] = 2.0
    elif fault == "wrong_scope":
        success["timing_scope"] = "four_rank_pilot_training_seconds"
    else:
        success["training_wall_seconds"] = 4.0
    with pytest.raises(ValueError):
        reader.inspect_timing(root, resolved, success)


def test_gate_acceptance_binds_the_concrete_evidence_files(gate) -> None:
    root, _, _, _, _, _ = gate
    gate_receipt = json.loads((root / "compatibility_gate.json").read_text())
    assert reader.inspect_gate_evidence(root, gate_receipt, 2) > 20
    gate_receipt["evidence_sha256"]["equivalence/before-update-000001.json"] = h("not the audited comparison")
    with pytest.raises(ValueError, match="evidence hash"):
        reader.inspect_gate_evidence(root, gate_receipt, 2)
