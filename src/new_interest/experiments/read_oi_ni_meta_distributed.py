"""Independent readback for four-rank execution of the frozen OI/NI recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq

from new_interest.experiments import read_oi_ni_meta as legacy


EXECUTION = {
    "engine": "oi-ni-meta-four-rank-v1", "world_size": 4, "nodes": 1,
    "global_support": 16, "global_query": 16, "microbatch_size": 1,
    "partition": "global_role_position_mod_world_size", "rng_planning_mode": "real_preview",
    "rng_seed_rule": "seed + 104729 * episode_index", "evaluation_shard_size": 256,
    "evaluation_batch_size": 8, "gate_validation_rows": 1024,
    "gradient_sync": "manual_global_inner_query_hvp_reduction_no_ddp",
    "gradient_reduction": "ordered_global_microchunk_broadcast_accumulation",
}
TOLERANCES = {"loss_atol": 1e-6, "loss_rtol": 1e-6, "gradient_atol": 2e-6,
              "gradient_rtol": 2e-5, "gradient_relative_l2": 2e-5,
              "update_atol": 2e-8, "update_rtol": 2e-6}
ENGINE_FILES = {"src/new_interest/experiments/oi_ni_meta_distributed.py",
                "src/new_interest/training/oi_ni_meta_distributed_objective.py",
                "src/new_interest/training/oi_ni_meta_rng.py"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    def invalid(value: str):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def json_digest(value: Any, *, compact: bool = False) -> str:
    options = {"sort_keys": True, "allow_nan": False}
    if compact:
        options["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def number(value: Any, label: str, *, nonnegative: bool = True) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Invalid finite number: {label}")
    require(not nonnegative or value >= 0, f"Negative value: {label}")
    return float(value)


def integer(value: Any, label: str) -> int:
    require(type(value) is int and value >= 0, f"Invalid integer: {label}")
    return value


def digest_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"Invalid digest: {label}")
    return value


def near(actual: float, expected: float, label: str, *, rtol: float = 1e-9) -> None:
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=1e-30), f"Inconsistent numeric evidence: {label}")


def inspect_source(root: Path, data: Path, resolved: Mapping[str, Any]) -> dict[str, Any]:
    source = read_json(root / "source_identity.json")
    require(source.get("version") == EXECUTION["engine"], "Unknown execution source version")
    require(source.get("execution_spec") == EXECUTION == resolved.get("execution_spec"), "Execution specification changed")
    require(source.get("gate_tolerances") == TOLERANCES, "Pre-frozen gate tolerances changed")
    old_files, new_files = source["legacy_files"], source["engine_files"]
    require(set(new_files) == ENGINE_FILES, "New engine source file set differs")
    for name, digest in {**old_files, **new_files}.items():
        digest_string(digest, name)
    old_identity = json_digest(old_files)
    require(old_identity == source["legacy_source_identity"] == resolved["legacy_source_identity"],
            "Legacy source identity does not recompute")
    binding = {"legacy_source_identity": old_identity, "engine_files": new_files,
               "execution_spec": EXECUTION, "gate_tolerances": TOLERANCES}
    require(json_digest(binding, compact=True) == source["source_identity"] == resolved["source_identity"],
            "Distributed source identity does not recompute")
    require(source["protocol_sha256"] == old_files["protocol"]
            and source["core_manifest_sha256"] == old_files["worker_bundle"], "Source lineage header mismatch")
    protocol = root / "lineage" / "protocol.json"
    require(legacy.sha256(protocol) == source["protocol_sha256"]
            and read_json(protocol) == resolved["config"], "Literal legacy protocol was changed")
    metadata = read_json(data / "metadata.json")
    require(old_files["metadata.json"] == legacy.sha256(data / "metadata.json")
            == resolved["config"]["data_metadata_sha256"], "Frozen metadata source lineage differs")
    for name, item in metadata["artifacts"].items():
        require(old_files.get(name) == item["sha256"] == legacy.sha256(legacy._inside(data, name)),
                f"Frozen data lineage differs: {name}")
    return source


def inspect_comparison(report: Mapping[str, Any], shapes: Mapping[str, list[int]],
                       *, kind: str) -> dict[str, Any]:
    """Recompute norm summaries and check numeric tolerance ratios, not flags."""
    prefix = "loss" if kind == "loss" else "update" if kind == "update" else "gradient"
    atol, rtol = TOLERANCES[f"{prefix}_atol"], TOLERANCES[f"{prefix}_rtol"]
    limit = TOLERANCES["gradient_relative_l2"] if kind == "gradients" else None
    require(report.get("atol") == atol and report.get("rtol") == rtol
            and report.get("relative_l2_limit") == limit, f"Comparison thresholds changed: {kind}")
    rows = report.get("tensors", [])
    require([row["name"] for row in rows] == sorted(shapes), f"Comparison tensor name coverage differs: {kind}")
    difference_sum, reference_sum, maximum_ratio = 0.0, 0.0, 0.0
    for row in rows:
        name, shape = row["name"], row["shape"]
        require(shape == shapes[name] and all(type(x) is int and x >= 0 for x in shape), "Comparison tensor shape differs")
        count = math.prod(shape)
        require(integer(row["numel"], "numel") == count and count > 0, "Comparison tensor size differs")
        diff_sq = number(row["difference_squared_l2"], "difference squared L2")
        ref_sq = number(row["reference_squared_l2"], "reference squared L2")
        absolute = number(row["max_abs_error"], "maximum absolute error")
        ratio = number(row["max_tolerance_ratio"], "maximum tolerance ratio")
        relative = number(row["relative_l2"], "tensor relative L2")
        near(relative, math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-30), name)
        require(absolute * absolute <= diff_sq * (1 + 1e-9) + 1e-60
                and diff_sq <= count * absolute * absolute * (1 + 1e-9) + 1e-60,
                "Absolute error and squared error contradict each other")
        lower_bound = absolute / (atol + rtol * math.sqrt(ref_sq))
        require(ratio * (1 + 3e-7) + 1e-12 >= lower_bound and ratio <= 1.0,
                f"Elementwise tolerance violated or inconsistent: {name}")
        if count == 1:
            actual = number(row["actual_value"], "actual scalar", nonnegative=False)
            reference = number(row["reference_value"], "reference scalar", nonnegative=False)
            # Worker comparisons subtract FP32 values before accumulating the
            # squares in FP64; reproduce that declared arithmetic independently.
            difference = float(np.float32(actual) - np.float32(reference))
            near(absolute, abs(difference), "scalar absolute error")
            near(diff_sq, difference * difference, "scalar squared error")
            near(ref_sq, reference * reference, "scalar reference squared")
            denominator = np.float32(atol) + np.float32(rtol) * abs(np.float32(reference))
            near(ratio, float(np.float32(abs(difference)) / denominator), "scalar tolerance ratio", rtol=2e-7)
        require(row.get("accepted") is True, "Comparison flag disagrees with complete numeric acceptance")
        difference_sum += diff_sq
        reference_sum += ref_sq
        maximum_ratio = max(maximum_ratio, ratio)
    global_relative = math.sqrt(difference_sum / max(reference_sum, 1e-60))
    near(number(report["relative_l2"], "global relative L2"), global_relative, "global relative L2")
    require(limit is None or global_relative <= limit, "Global gradient relative L2 limit exceeded")
    require(report.get("accepted") is True, "Comparison aggregate is not accepted")
    return {"tensors": len(rows), "max_tolerance_ratio": maximum_ratio, "relative_l2": global_relative}


def inspect_rank_event(path: Path, identity: str, event: str, step: int,
                       parameter_count: int) -> dict[str, str]:
    audit = read_json(path)
    require(audit.get("identity") == identity and audit.get("event") == event
            and audit.get("step") == step and audit.get("world_size") == 4, "Rank event identity/topology differs")
    ranks = audit.get("ranks", [])
    require([row.get("rank") for row in ranks] == list(range(4)), "Rank event must cover all four ranks exactly")
    require({integer(row["local_rank"], "local rank") for row in ranks} == set(range(4)), "Local device ownership differs")
    for key in ("parameter_sha256", "tensor_sha256"):
        values = {digest_string(row[key], key) for row in ranks}
        require(len(values) == 1, f"Rank {key} divergence")
    if event in {"initialization", "restore", "after_update", "checkpoint", "long_sequence_no_update", "final_deployed"}:
        for row in ranks:
            require(integer(row["optimizer_step_min"], "optimizer min step") == step
                    == integer(row["optimizer_step_max"], "optimizer max step"), "Rank optimizer cursor differs")
            expected_states = 0 if step == 0 else parameter_count
            require(integer(row["optimizer_state_count"], "optimizer state count") == expected_states,
                    "Rank optimizer state coverage differs")
    if event == "fresh_reload":
        require(all(number(row["reload_max_abs_error"], "rank reload error") == 0 for row in ranks),
                "A rank failed exact reload")
    require(audit.get("accepted") is True, "Rank event reports failed acceptance")
    return {key: ranks[0][key] for key in ("parameter_sha256", "tensor_sha256")}


def saved_replica_digest(root: Path, checkpoint_relative: str, shapes: Mapping[str, Any]) -> str:
    import torch
    from safetensors.torch import load_file
    saved = {}
    for path in sorted((root / checkpoint_relative / "adapter").glob("*.safetensors")):
        for name, value in load_file(str(path), device="cpu").items():
            canonical = legacy._canonical_name(name)
            require(canonical not in saved, "Duplicate saved canonical tensors")
            saved[canonical] = value
    digest = hashlib.sha256()
    for name in sorted(shapes):
        value = saved[legacy._canonical_name(name)].contiguous()
        require(list(value.shape) == shapes[name], "Replica digest tensor shape differs")
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inspect_shard_owners(directory: Path, count: int, identity: str) -> dict[str, Any]:
    metrics = read_json(directory / "metrics.json")
    require(metrics.get("world_size") == 4 and metrics.get("global_shard_size") == 256
            and metrics.get("batch_size") == 8, "Evaluation execution topology differs")
    expected = [f"predictions-{start:06d}.parquet" for start in range(0, count, 256)]
    require(sorted(p.name for p in directory.glob("predictions-*.parquet")) == expected, "Global evaluation shard coverage differs")
    owners = {rank: 0 for rank in range(4)}
    for start in range(0, count, 256):
        shard = directory / f"predictions-{start:06d}.parquet"
        receipt = read_json(shard.with_suffix(".json"))
        owner, stop = (start // 256) % 4, min(start + 256, count)
        require(receipt.get("identity") == identity and receipt.get("world_size") == 4
                and integer(receipt["rank"], "evaluation owner") == owner
                and receipt["rows"] == stop - start and receipt["sha256"] == legacy.sha256(shard),
                "Evaluation shard ownership/identity mismatch")
        rows = pq.read_table(shard, columns=["row_index"])["row_index"].to_pylist()
        require(rows == list(range(start, stop)), "Evaluation global row indices differ")
        owners[owner] += len(rows)
    require(count < 1024 or set(rank for rank, rows in owners.items() if rows) == set(range(4)),
            "Evaluation did not exercise all four rank owners")
    return {"global_shards": len(expected), "rows_by_rank": {str(k): v for k, v in owners.items()}}


def inspect_steps(root: Path, resolved: Mapping[str, Any], shapes: Mapping[str, list[int]]) -> dict[str, Any]:
    identity, steps, mode = resolved["identity"], integer(resolved["steps"], "steps"), resolved["mode"]
    logs = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines() if line.strip()]
    latest: dict[int, dict[str, Any]] = {}
    for row in logs:
        step = integer(row["step"], "logged step")
        require(1 <= step <= steps and row["mode"] == mode, "Training log step/mode differs")
        latest[step] = row
    require(set(latest) == set(range(1, steps + 1)), "Training log lacks committed step coverage")
    diagnostics = []
    for step, row in sorted(latest.items()):
        require(row["global_support"] == row["global_query"] == 16, "Global episode size changed")
        number(row["seconds"], "step seconds")
        plan = row["rng_plan"]
        expected_roles = {"oi": 16, "ni": 16, **({"no_history": 16} if mode == "meta_no_history" else {})}
        require(plan.get("version") == "serial-real-forward-rng-v1" and plan.get("planning_mode") == "real_preview"
                and plan.get("episode_seed") == resolved["config"]["seed"] + 104729 * (step - 1)
                and plan.get("caller_rng_preserved") is True and plan.get("device_type") == "cuda"
                and plan.get("role_chunk_counts") == expected_roles
                and plan.get("preview_forward_count") == sum(expected_roles.values()), "Serial dropout planning contract changed")
        digest_string(plan["state_sha256"], "RNG plan")
        rank_digest = inspect_rank_event(root / "execution_audits" / f"after_update-{step:06d}.json",
                                         identity, "after_update", step, len(shapes))
        require(row["parameter_sha256"] == rank_digest["parameter_sha256"], "Log/update parameter digest differs")
        digest_string(row["gradient_sha256"], "global gradient")
        if resolved["phase"] == "gate":
            before = read_json(root / "equivalence" / f"before-update-{step:06d}.json")
            after = read_json(root / "equivalence" / f"after-update-{step:06d}.json")
            for report, kind in ((before, "before_update"), (after, "after_update")):
                require(report.get("identity") == identity and report.get("step") == step
                        and report.get("kind") == kind and report.get("source_identity") == resolved["source_identity"],
                        "Equivalence report identity/cursor/source differs")
            diagnostics.append({"step": step,
                "loss": inspect_comparison(before["loss"], {"loss": []}, kind="loss"),
                "norms": inspect_comparison(before["norms"], {"inner": [], "hvp": []}, kind="norms"),
                "gradients": inspect_comparison(before["gradients"], shapes, kind="gradients"),
                "adam_parameters": inspect_comparison(after, shapes, kind="update")})
    return {"committed_steps": steps, "log_rows": len(logs), "gate_comparisons": diagnostics}


def inspect_timing(root: Path, resolved: Mapping[str, Any], success: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[int, float] = {}
    all_times = []
    gate = resolved["phase"] == "gate"
    for line in (root / "training.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step, elapsed = integer(row["step"], "timing step"), number(row["seconds"], "step duration")
        require(1 <= step <= resolved["steps"] and elapsed > 0
                and row.get("timing_includes_serial_gate") is gate, "Timing cursor/scope differs")
        canonical[step] = elapsed
        all_times.append(elapsed)
    require(set(canonical) == set(range(1, resolved["steps"] + 1)), "Timing misses committed updates")
    wall, recorded = math.fsum(canonical.values()), math.fsum(all_times)
    computed = {"training_wall_seconds": wall, "training_gpu_seconds": 4 * wall,
                "recorded_step_wall_seconds_including_retries": recorded,
                "recorded_step_gpu_seconds_including_retries": 4 * recorded,
                "median_step_seconds": float(np.median(list(canonical.values())))}
    for key, value in computed.items():
        near(number(success[key], key), value, key)
    scope = "engineering_gate_including_serial_reference" if gate else "four_rank_pilot_training_seconds"
    require(success.get("timing_scope") == scope
            and success.get("timing_includes_planning_and_communication") is True
            and success.get("timing_excludes_checkpoint_eval_setup_and_unfinished_steps") is True,
            "Timing accounting scope changed")
    return {**computed, "timing_scope": scope}


def inspect_gate_evidence(root: Path, gate: Mapping[str, Any], steps: int) -> int:
    required = {"model_audit.json", "token_audit.json", "long_sequence_hvp_audit.json",
                "source_identity.json", "lineage/protocol.json", "fresh_reload_audit.json",
                "execution_audits/initialization-000000.json",
                f"execution_audits/checkpoint-{steps:06d}.json",
                f"execution_audits/fresh_reload-{steps:06d}.json",
                f"checkpoints/checkpoint-{steps:06d}/manifest.json",
                f"validation-{steps:06d}/metrics.json"}
    for step in range(1, steps + 1):
        required.update({f"equivalence/before-update-{step:06d}.json",
                         f"equivalence/after-update-{step:06d}.json",
                         f"execution_audits/after_update-{step:06d}.json"})
    for start in range(0, 1024, 256):
        required.update({f"validation-{steps:06d}/predictions-{start:06d}.parquet",
                         f"validation-{steps:06d}/predictions-{start:06d}.json"})
    evidence = gate.get("evidence_sha256", {})
    require(required <= set(evidence), "Compatibility gate evidence inventory is incomplete")
    for relative, value in evidence.items():
        require(digest_string(value, "gate evidence") == legacy.sha256(legacy._inside(root, relative)),
                "Compatibility gate evidence hash differs")
    return len(evidence)


def read_distributed_run(run_root: Path | str, data_dir: Path | str, *, scope: str,
                         gate_run_root: Path | str | None = None,
                         tokenizer_cache_dir: Path | str | None = None) -> dict[str, Any]:
    root, data = Path(run_root).resolve(), Path(data_dir).resolve()
    require(scope in {"gate", "pilot"}, "Unknown distributed readback scope")
    base = legacy.read_run(root, data, scope=scope, tokenizer_cache_dir=tokenizer_cache_dir,
                           expected_topology={"validation": (1024, 1024, None)} if scope == "gate" else None)
    resolved = read_json(root / "resolved_config.json")
    identity, steps = base["identity"], base["steps"]
    source = inspect_source(root, data, resolved)
    success = read_json(root / "success.json")
    require(success.get("world_size") == 4 and success.get("source_identity") == source["source_identity"]
            and success.get("legacy_source_identity") == source["legacy_source_identity"], "Run execution/source header differs")
    model = read_json(root / "model_audit.json")
    require(model.get("world_size") == 4 and model.get("source_identity") == source["source_identity"],
            "Model audit does not attest four ranks and the execution source")
    shapes = model["trainable_parameters"]
    initial = inspect_rank_event(root / "execution_audits" / "initialization-000000.json",
                                 identity, "initialization", 0, len(shapes))
    require(initial["parameter_sha256"] == model["initial_trainable_sha256"], "Initial rank/model digest differs")
    if scope == "gate":
        longest = read_json(root / "long_sequence_hvp_audit.json")
        require(longest.get("world_size") == 4 and longest.get("global_support") == 1
                and longest.get("global_query") == 1 and longest.get("owner_rank") == 0
                and longest.get("empty_ranks") == [1, 2, 3], "Longest-input gate did not exercise the empty-rank path")
        unchanged = inspect_rank_event(root / "execution_audits" / "long_sequence_no_update-000000.json",
                                        identity, "long_sequence_no_update", 0, len(shapes))
        require(unchanged == initial, "Gradient-only longest-input gate changed replica parameters")
    step_evidence = inspect_steps(root, resolved, shapes)
    timing = inspect_timing(root, resolved, success)
    last = inspect_rank_event(root / "execution_audits" / f"after_update-{steps:06d}.json",
                              identity, "after_update", steps, len(shapes))
    for checkpoint in sorted((root / "checkpoints").glob("checkpoint-*")):
        state = read_json(checkpoint / "state.json")
        step = integer(state["step"], "checkpoint cursor")
        event = inspect_rank_event(root / "execution_audits" / f"checkpoint-{step:06d}.json",
                                   identity, "checkpoint", step, len(shapes))
        updated = inspect_rank_event(root / "execution_audits" / f"after_update-{step:06d}.json",
                                     identity, "after_update", step, len(shapes))
        require(event == updated and state.get("world_size") == 4
                and state.get("trainable_sha256") == event["parameter_sha256"], "Checkpoint/rank update digests differ")
    for path in sorted((root / "execution_audits").glob("restore-*.json")):
        state = read_json(path)
        step = integer(state["step"], "restore cursor")
        restored = inspect_rank_event(path, identity, "restore", step, len(shapes))
        original = inspect_rank_event(root / "execution_audits" / f"checkpoint-{step:06d}.json",
                                      identity, "checkpoint", step, len(shapes))
        require(restored == original, "Restored rank states differ from committed checkpoint")
    reloaded = inspect_rank_event(root / "execution_audits" / f"fresh_reload-{steps:06d}.json",
                                  identity, "fresh_reload", steps, len(shapes))
    deployed = inspect_rank_event(root / "execution_audits" / f"final_deployed-{steps:06d}.json",
                                  identity, "final_deployed", steps, len(shapes))
    require(last == deployed == reloaded and last["parameter_sha256"] == base["checkpoint"]["trainable_sha256"],
            "Final update, fresh reload, or independently read tensor digests differ")
    reload_audit = read_json(root / "fresh_reload_audit.json")
    require(reload_audit.get("world_size") == 4 and reload_audit.get("trainable_sha256") == last["parameter_sha256"]
            and reload_audit.get("max_abs_error_by_rank") == [0.0] * 4,
            "Four-rank fresh-reload summary differs from replica evidence")
    require(last["tensor_sha256"] == saved_replica_digest(root, success["checkpoint_relative"], shapes),
            "Saved tensor/dtype replica digest differs")
    import torch
    state = torch.load(root / success["checkpoint_relative"] / "training_state.pt", map_location="cpu", weights_only=False)
    require(state.get("world_size") == 4 and len(state.get("rng_by_rank", [])) == 4
            and state.get("legacy_rng_is_rank0_only") is True, "Distributed restore RNG topology differs")
    for rank_rng in state["rng_by_rank"]:
        require(set(rank_rng) == {"python", "numpy", "torch", "cuda"}, "Missing per-rank RNG state")
    owners = {"validation": inspect_shard_owners(root / f"validation-{steps:06d}", base["validation"]["samples"], identity)}
    if scope == "gate":
        gate = read_json(root / "compatibility_gate.json")
        require(gate.get("identity") == identity and gate.get("world_size") == 4 and gate.get("steps") == steps
                and gate.get("execution_spec") == EXECUTION and gate.get("gate_tolerances") == TOLERANCES
                and gate.get("source_identity") == source["source_identity"]
                and gate.get("legacy_source_identity") == source["legacy_source_identity"]
                and gate.get("phase") == "gate" and gate.get("mode") == "meta" and gate.get("alpha") == resolved["alpha"],
                "Compatibility gate header differs")
        require(gate.get("accepted") is True, "Compatibility gate reports failure")
        gate_evidence_files = inspect_gate_evidence(root, gate, steps)
    else:
        require(gate_run_root is not None, "Pilot requires complete downloaded gate evidence, not an accepted flag")
        gate_receipt = read_distributed_run(gate_run_root, data, scope="gate", tokenizer_cache_dir=tokenizer_cache_dir)
        inspect_selection(root, resolved, source, Path(gate_run_root).resolve(), gate_receipt)
        if resolved["phase"] == "compute_matched":
            inspect_compute_budget(root, resolved, source)
        owners["dev"] = inspect_shard_owners(root / "dev", base["dev"]["samples"], identity)
    return {"status": "PASS", "scope": scope, "identity": identity, "world_size": 4,
            "source_identity": source["source_identity"], "legacy_source_identity": source["legacy_source_identity"],
            "execution_spec": EXECUTION, "gate_tolerances": TOLERANCES,
            "step_evidence": step_evidence, "evaluation_ownership": owners,
            "timing": timing, **({"gate_evidence_files": gate_evidence_files} if scope == "gate" else {}),
            "replicas_and_checkpoint_agree": True, "legacy_readback": base,
            "reader_sha256": legacy.sha256(Path(__file__))}


def inspect_selection(root: Path, resolved: Mapping[str, Any], source: Mapping[str, Any],
                      gate_root: Path, gate_receipt: Mapping[str, Any]) -> None:
    folder = root / "lineage"
    selection_path, lineage_path, copied_gate = (folder / name for name in
        ("selection.json", "selection_lineage.json", "compatibility_gate.json"))
    selection, lineage = read_json(selection_path), read_json(lineage_path)
    config = resolved["config"]
    expected = {"selection_sha256": legacy.sha256(selection_path),
                "gate_receipt_sha256": legacy.sha256(gate_root / "compatibility_gate.json"),
                "parent_source_identity": source["legacy_source_identity"],
                "execution_source_identity": source["source_identity"],
                "protocol_sha256": source["protocol_sha256"],
                "data_metadata_sha256": config["data_metadata_sha256"]}
    require(legacy.sha256(copied_gate) == expected["gate_receipt_sha256"]
            and gate_receipt["source_identity"] == source["source_identity"], "Pilot gate lineage differs from verified full gate")
    require(all(lineage.get(key) == value for key, value in expected.items()) and lineage.get("accepted") is True,
            "Explicit legacy-to-distributed selection lineage mismatch")
    expected["lineage_sha256"] = legacy.sha256(lineage_path)
    require(all(resolved["selection_binding"].get(key) == value for key, value in expected.items()),
            "Resolved selection file hashes differ")
    require(selection.get("validation_only") is True and selection.get("source_identity") == source["legacy_source_identity"],
            "Selection was not made with frozen legacy validation")
    results, candidates = selection.get("results", []), config["inner_alpha_candidates"]
    require(len(results) == len(candidates) and sorted(row["alpha"] for row in results) == sorted(candidates),
            "Selection lacks the complete frozen alpha candidates")
    for row in results:
        require(row["mode"] == "meta" and row["steps"] == config["calibration_steps"], "Calibration budget/mode changed")
        metrics = row["validation_metrics"]
        require(metrics["samples"] == 2160 and metrics["supported_interest_count"] == 108,
                "Calibration validation topology differs")
        for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
            require(number(metrics[key], key) <= 1, "Invalid calibration metric")
    winner = max(results, key=lambda row: (row["validation_metrics"]["interest_macro_recall_at_10"],
                 row["validation_metrics"]["interest_macro_ndcg_at_10"], -row["alpha"]))
    require(selection["chosen_alpha"] == winner["alpha"] and resolved["alpha"] ==
            (0.0 if resolved["mode"] == "mixed" else winner["alpha"]), "Alpha selection winner differs")


def inspect_compute_budget(root: Path, resolved: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    path = root / "lineage" / "compute_budget.json"
    budget = read_json(path)
    steps = integer(budget["steps"], "compute budget steps")
    config = resolved["config"]
    require(resolved["mode"] == "mixed" and resolved["alpha"] == 0
            and budget.get("world_size") == 4 and budget.get("source_identity") == source["source_identity"]
            and budget.get("timing_basis") == "four_rank_pilot_training_seconds",
            "Compute control is not based on this four-rank pilot")
    require(config["pilot_steps"] <= steps <= config["max_compute_multiplier"] * config["pilot_steps"]
            and steps == resolved["steps"] == resolved["budget_binding"]["steps"]
            and legacy.sha256(path) == resolved["budget_binding"]["budget_sha256"],
            "Compute budget cursor or frozen receipt differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("gate", "pilot"), required=True)
    parser.add_argument("--gate-run-root", type=Path)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.run_root.resolve())
            and not (args.gate_run_root and args.output.resolve().is_relative_to(args.gate_run_root.resolve())),
            "Receipt must be outside committed runs")
    require(not args.output.exists(), "Refusing to overwrite distributed readback")
    receipt = read_distributed_run(args.run_root, args.data_dir, scope=args.scope,
                gate_run_root=args.gate_run_root, tokenizer_cache_dir=args.tokenizer_cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
