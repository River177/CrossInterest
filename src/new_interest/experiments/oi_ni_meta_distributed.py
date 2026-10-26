"""Four-rank execution engine for the frozen OI/NI exact-meta protocol.

The single-GPU engine remains immutable. This engine has its own source/run
identity and accepts a legacy alpha selection only through an explicit,
hash-bound compatibility receipt. Replicas are not wrapped in DDP: the
distributed objective synchronizes the global virtual gradient and HVP.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments import oi_ni_meta as core


ENGINE_VERSION = "oi-ni-meta-four-rank-v1"
ENGINE_FILES = (
    "src/new_interest/experiments/oi_ni_meta_distributed.py",
    "src/new_interest/training/oi_ni_meta_distributed_objective.py",
    "src/new_interest/training/oi_ni_meta_rng.py",
)
LEGACY_FILES = {
    "oi_ni_meta.py": "src/new_interest/experiments/oi_ni_meta.py",
    "../training/oi_ni_meta_backend.py": "src/new_interest/training/oi_ni_meta_backend.py",
    "../training/oi_ni_meta_objective.py": "src/new_interest/training/oi_ni_meta_objective.py",
}
EXECUTION_SPEC = {
    "engine": ENGINE_VERSION,
    "world_size": 4,
    "nodes": 1,
    "global_support": 16,
    "global_query": 16,
    "microbatch_size": 1,
    "partition": "global_role_position_mod_world_size",
    "rng_planning_mode": "real_preview",
    "rng_seed_rule": "seed + 104729 * episode_index",
    "evaluation_shard_size": 256,
    "evaluation_batch_size": 8,
    "gate_validation_rows": 1024,
    "gradient_sync": "manual_global_inner_query_hvp_reduction_no_ddp",
    "gradient_reduction": "ordered_global_microchunk_broadcast_accumulation",
}


@dataclass(frozen=True)
class GateTolerances:
    loss_atol: float = 1e-6
    loss_rtol: float = 1e-6
    gradient_atol: float = 2e-6
    gradient_rtol: float = 2e-5
    gradient_relative_l2: float = 2e-5
    update_atol: float = 2e-8
    update_rtol: float = 2e-6


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def owned_positions(count: int, rank: int, world_size: int = 4) -> list[int]:
    if count < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid count/rank/world size")
    return list(range(rank, count, world_size))


def owned_evaluation_shards(count: int, rank: int, world_size: int = 4,
                            shard_size: int = 256) -> list[tuple[int, int]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return [(index * shard_size, min((index + 1) * shard_size, count))
            for index in owned_positions(math.ceil(count / shard_size), rank, world_size)]


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes root: {relative}")
    return path


def source_receipt(code_root: Path, config_path: Path, data_hashes: Mapping[str, str],
                   *, engine_files: Sequence[str] = ENGINE_FILES,
                   tolerances: GateTolerances = GateTolerances()) -> dict[str, Any]:
    """Reconstruct the old identity exactly, then bind the distinct new engine."""
    manifest_path = code_root / "source_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        if core.sha256_file(_contained(code_root, relative)) != expected:
            raise ValueError(f"Frozen legacy source changed: {relative}")
    legacy = dict(data_hashes)
    legacy["protocol"] = core.sha256_file(config_path)
    legacy["worker_bundle"] = core.sha256_file(manifest_path)
    legacy.update({key: core.sha256_file(code_root / relative) for key, relative in LEGACY_FILES.items()})
    # Deliberately retain the legacy engine's JSON serialization convention.
    legacy_identity = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    new_files = {relative: core.sha256_file(_contained(code_root, relative)) for relative in engine_files}
    binding = {"legacy_source_identity": legacy_identity, "engine_files": new_files,
               "execution_spec": EXECUTION_SPEC, "gate_tolerances": asdict(tolerances)}
    return {"version": ENGINE_VERSION, "source_identity": digest_json(binding),
            "legacy_source_identity": legacy_identity, "legacy_files": legacy,
            "engine_files": new_files, "execution_spec": dict(EXECUTION_SPEC),
            "gate_tolerances": asdict(tolerances), "protocol_sha256": legacy["protocol"],
            "core_manifest_sha256": legacy["worker_bundle"]}


def validate_selection_lineage(selection_path: Path, lineage_path: Path, gate_path: Path,
                               source: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, dict[str, str]]:
    """Do not relabel a legacy selection as if it came from the new engine."""
    selection = json.loads(selection_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    gate = json.loads(gate_path.read_text())
    expected = {
        "selection_sha256": core.sha256_file(selection_path),
        "gate_receipt_sha256": core.sha256_file(gate_path),
        "parent_source_identity": source["legacy_source_identity"],
        "execution_source_identity": source["source_identity"],
        "protocol_sha256": source["protocol_sha256"],
        "data_metadata_sha256": config["data_metadata_sha256"],
    }
    if lineage.get("accepted") is not True or any(lineage.get(k) != v for k, v in expected.items()):
        raise ValueError("Legacy alpha-selection lineage is not explicitly accepted and hash-bound")
    if (gate.get("accepted") is not True or gate.get("source_identity") != source["source_identity"]
            or gate.get("legacy_source_identity") != source["legacy_source_identity"]
            or gate.get("phase") != "gate" or gate.get("mode") != "meta"
            or gate.get("steps") != config["gate_steps"] or gate.get("world_size") != 4
            or gate.get("execution_spec") != source["execution_spec"]
            or gate.get("gate_tolerances") != source["gate_tolerances"]):
        raise ValueError("Four-rank compatibility gate has the wrong source identity")
    if (selection.get("validation_only") is not True
            or selection.get("source_identity") != source["legacy_source_identity"]):
        raise ValueError("Selection is not the frozen legacy validation-only result")
    candidates = [float(value) for value in config["inner_alpha_candidates"]]
    results = selection.get("results", [])
    if (len(results) != len(candidates) or sorted(float(row["alpha"]) for row in results) != sorted(candidates)
            or any(row["mode"] != "meta" or row["steps"] != config["calibration_steps"] for row in results)):
        raise ValueError("Selection does not contain every frozen calibration condition")
    chosen = max(results, key=lambda row: (
        row["validation_metrics"]["interest_macro_recall_at_10"],
        row["validation_metrics"]["interest_macro_ndcg_at_10"], -float(row["alpha"])))
    alpha = float(selection["chosen_alpha"])
    if alpha != float(chosen["alpha"]) or alpha not in candidates:
        raise ValueError("Selection winner does not follow the frozen validation rule")
    return alpha, {**expected, "lineage_sha256": core.sha256_file(lineage_path)}


def validate_compute_budget(path: Path, source: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[int, str]:
    budget = json.loads(path.read_text())
    steps = budget.get("steps")
    if (budget.get("source_identity") != source["source_identity"] or budget.get("world_size") != 4
            or budget.get("timing_basis") != "four_rank_pilot_training_seconds"):
        raise ValueError("Compute control must use this four-rank engine's pilot training timings")
    if (not isinstance(steps, int) or isinstance(steps, bool)
            or not config["pilot_steps"] <= steps <= config["max_compute_multiplier"] * config["pilot_steps"]):
        raise ValueError("Compute control exceeds the frozen step budget")
    return steps, core.sha256_file(path)


def timing_totals(path: Path, steps: int, world_size: int, *, phase: str) -> dict[str, Any]:
    canonical: dict[int, float] = {}
    recorded = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step, elapsed = row["step"], row["seconds"]
        if (not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= steps
                or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0
                or row["timing_includes_serial_gate"] is not (phase == "gate")):
            raise ValueError("Invalid or mixed-scope step timing record")
        canonical[step] = float(elapsed)
        recorded.append(float(elapsed))
    if sorted(canonical) != list(range(1, steps + 1)):
        raise ValueError("Training timing does not cover every committed update")
    seconds = math.fsum(canonical.values())
    return {"training_wall_seconds": seconds, "training_gpu_seconds": world_size * seconds,
            "recorded_step_wall_seconds_including_retries": math.fsum(recorded),
            "recorded_step_gpu_seconds_including_retries": world_size * math.fsum(recorded),
            "median_step_seconds": float(np.median(list(canonical.values()))),
            "timing_scope": "engineering_gate_including_serial_reference" if phase == "gate" else "four_rank_pilot_training_seconds",
            "timing_includes_planning_and_communication": True,
            "timing_excludes_checkpoint_eval_setup_and_unfinished_steps": True}


def compare_tensors(actual: Mapping[str, Any], expected: Mapping[str, Any], *,
                    atol: float, rtol: float, relative_l2_limit: float | None = None) -> dict[str, Any]:
    import torch
    if set(actual) != set(expected):
        raise ValueError("Comparison tensor names differ")
    rows = []
    difference_squared = 0.
    reference_squared = 0.
    for name in sorted(actual):
        a, b = actual[name].detach().float().cpu(), expected[name].detach().float().cpu()
        if a.shape != b.shape:
            raise ValueError(f"Comparison shape differs: {name}")
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        if not finite:
            raise FloatingPointError(f"Nonfinite compatibility comparison: {name}")
        difference = a - b
        difference_squared += float(difference.double().square().sum())
        reference_squared += float(b.double().square().sum())
        denominator = atol + rtol * b.abs()
        ratio = torch.where(denominator > 0, difference.abs() / denominator,
                            torch.where(difference == 0, 0., float("inf")))
        rows.append({"name": name, "accepted": finite and torch.allclose(a, b, atol=atol, rtol=rtol),
                     "shape": list(a.shape), "numel": a.numel(),
                     "difference_squared_l2": float(difference.double().square().sum()),
                     "reference_squared_l2": float(b.double().square().sum()),
                     "max_tolerance_ratio": float(ratio.max()) if a.numel() else 0.,
                     "max_abs_error": float(difference.abs().max()) if a.numel() else 0.,
                     "relative_l2": float(difference.double().norm() / b.double().norm().clamp_min(1e-30)),
                     **({"actual_value": float(a), "reference_value": float(b)} if a.numel() == 1 else {})})
    relative_l2 = math.sqrt(difference_squared / max(reference_squared, 1e-60))
    return {"accepted": all(row["accepted"] for row in rows)
                        and (relative_l2_limit is None or relative_l2 <= relative_l2_limit),
            "atol": atol, "rtol": rtol, "relative_l2": relative_l2,
            "relative_l2_limit": relative_l2_limit, "tensors": rows}


def tensor_digest(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_gate_evidence(root: Path, identity: str, source: Mapping[str, Any],
                           config: Mapping[str, Any]) -> dict[str, str]:
    """Recheck gate evidence on finalization/resume, before publishing acceptance."""
    paths: set[str] = set()
    def read(relative):
        paths.add(relative)
        return json.loads((root / relative).read_text())
    def require(condition, message):
        if not condition:
            raise ValueError(f"Incomplete distributed gate evidence: {message}")
    model = read("model_audit.json")
    tokens = read("token_audit.json")
    longest = read("long_sequence_hvp_audit.json")
    require(longest.get("identity") == identity and longest.get("parameter_update_performed") is False,
            "longest-input identity/update flag")
    require(longest.get("scope") == "gradient-only engineering gate; not a method score"
            and longest.get("world_size") == 4 and longest.get("owner_rank") == 0
            and longest.get("global_support") == longest.get("global_query") == 1
            and longest.get("empty_ranks") == [1,2,3], "longest-input empty-rank topology")
    require(tokens.get("target_in_input") is False and tokens.get("over_length") == 0
            and longest.get("oi_tokens") == tokens.get("oi_max")
            and longest.get("ni_tokens") == tokens.get("ni_max"), "longest-input token coverage")
    require(all(isinstance(longest.get(key), (int,float)) and math.isfinite(longest[key])
                and longest[key] > 0 for key in ("hvp_norm", "seconds", "peak_gpu_bytes"))
            and longest["peak_gpu_bytes"] <= model["memory_bytes"], "longest-input finite HVP/time/memory")
    limits = GateTolerances(**source["gate_tolerances"])
    names = set(model["trainable_parameters"])
    def comparison(value, expected_names, atol, rtol, relative_limit=None):
        require(value.get("accepted") is True and value.get("atol") == atol and value.get("rtol") == rtol,
                "comparison tolerance/acceptance")
        rows = value.get("tensors", [])
        require(len(rows) == len(expected_names) and {row["name"] for row in rows} == expected_names,
                "comparison parameter coverage")
        for row in rows:
            require(row.get("accepted") is True and math.isfinite(row["max_tolerance_ratio"])
                    and 0 <= row["max_tolerance_ratio"] <= 1
                    and math.isfinite(row["difference_squared_l2"]) and row["difference_squared_l2"] >= 0
                    and math.isfinite(row["reference_squared_l2"]) and row["reference_squared_l2"] >= 0,
                    "comparison tensor error bound")
        relative = math.sqrt(math.fsum(row["difference_squared_l2"] for row in rows)
                             / max(math.fsum(row["reference_squared_l2"] for row in rows), 1e-60))
        require(math.isclose(relative, value["relative_l2"], rel_tol=1e-10, abs_tol=1e-12)
                and (relative_limit is None or relative <= relative_limit), "comparison global relative L2")
    def replicas(event, step, expected_hash=None):
        receipt = read(f"execution_audits/{event}-{step:06d}.json")
        ranks = receipt.get("ranks", [])
        require(receipt.get("accepted") is True and receipt.get("identity") == identity
                and receipt.get("event") == event and receipt.get("step") == step
                and receipt.get("world_size") == 4 and len(ranks) == 4
                and {row["rank"] for row in ranks} == {0,1,2,3}, f"{event} rank coverage")
        require(len({row["parameter_sha256"] for row in ranks}) == 1
                and len({row["tensor_sha256"] for row in ranks}) == 1
                and (expected_hash is None or ranks[0]["parameter_sha256"] == expected_hash), f"{event} replica parameters")
        if event in {"after_update", "checkpoint"}:
            require(all(row["optimizer_step_min"] == row["optimizer_step_max"] == step
                        and row["optimizer_state_count"] == len(names) for row in ranks), f"{event} optimizer cursor")
        if event == "fresh_reload":
            require(all(row["reload_max_abs_error"] == 0 for row in ranks), "fresh reload values")
        return ranks[0]["parameter_sha256"]
    replicas("initialization", 0, model["initial_trainable_sha256"])
    steps = int(config["gate_steps"])
    final_hash = None
    for step in range(1, steps + 1):
        before = read(f"equivalence/before-update-{step:06d}.json")
        after = read(f"equivalence/after-update-{step:06d}.json")
        for value, kind in ((before, "before_update"), (after, "after_update")):
            require(value.get("identity") == identity and value.get("step") == step
                    and value.get("kind") == kind and value.get("source_identity") == source["source_identity"],
                    "comparison identity/step")
        comparison(before["loss"], {"loss"}, limits.loss_atol, limits.loss_rtol)
        comparison(before["norms"], {"inner", "hvp"}, limits.gradient_atol, limits.gradient_rtol)
        comparison(before["gradients"], names, limits.gradient_atol, limits.gradient_rtol, limits.gradient_relative_l2)
        comparison(after, names, limits.update_atol, limits.update_rtol)
        final_hash = replicas("after_update", step)
    replicas("checkpoint", steps, final_hash)
    replicas("fresh_reload", steps, final_hash)
    checkpoint = root / "checkpoints" / f"checkpoint-{steps:06d}"
    core.verify_manifest(checkpoint, identity)
    paths.add(str((checkpoint / "manifest.json").relative_to(root)))
    count = int(EXECUTION_SPEC["gate_validation_rows"])
    owners = []
    directory = f"validation-{steps:06d}"
    metrics = read(f"{directory}/metrics.json")
    require(metrics.get("identity") == identity and metrics.get("samples") == count, "gate validation count")
    for start in range(0, count, 256):
        relative = f"{directory}/predictions-{start:06d}.parquet"
        receipt = read(relative.removesuffix(".parquet") + ".json")
        paths.add(relative)
        require(receipt.get("identity") == identity and receipt.get("world_size") == 4
                and receipt.get("rank") == (start // 256) % 4
                and receipt.get("sha256") == core.sha256_file(root / relative), "validation shard ownership/hash")
        indices = pq.read_table(root / relative, columns=["row_index"])["row_index"].to_pylist()
        require(indices == list(range(start, min(start + 256, count))), "validation shard row coverage")
        owners.append(receipt["rank"])
    require(set(owners) == {0,1,2,3}, "all four evaluation owners")
    for relative in ("source_identity.json", "lineage/protocol.json", "fresh_reload_audit.json"):
        paths.add(relative)
    return {relative: core.sha256_file(root / relative) for relative in sorted(paths)}


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 4 or local_world != 4 or not torch.cuda.is_available() or not 0 <= local < torch.cuda.device_count():
        raise RuntimeError("Launch this engine with torchrun and exactly four CUDA ranks")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    return DistributedContext(dist.get_rank(), dist.get_world_size(), local)


def _broadcast(value: Any, context: DistributedContext) -> Any:
    import torch.distributed as dist
    values = [value if context.primary else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


@contextmanager
def primary_phase(context: DistributedContext, label: str):
    """All ranks enter; rank-zero exceptions reach peers before the next collective."""
    failure = None
    try:
        yield
    except Exception as error:
        if not context.primary:
            raise
        failure = {"type": type(error).__name__, "message": str(error), "phase": label}
    envelope = _broadcast({"ok": failure is None, "error": failure}, context)
    if not envelope["ok"]:
        error = envelope["error"]
        raise RuntimeError(f"Rank-zero {error['phase']} failed: {error['type']}: {error['message']}")


def _barrier() -> None:
    import torch.distributed as dist
    dist.barrier()


def _rank_max_seconds(elapsed: float) -> float:
    import torch
    import torch.distributed as dist
    value = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _replica_digest(values: Mapping[str, Any], context: DistributedContext) -> str:
    import torch.distributed as dist
    local = tensor_digest(values)
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Rank tensors diverged: {gathered}")
    return local


def execution_audit(context: DistributedContext, model: Any, root: Path, identity: str,
                     event: str, step: int, *, optimizer: Any | None = None,
                     reload_error: float | None = None) -> str:
    import torch.distributed as dist
    parameters = {name: value for name,value in model.named_parameters() if value.requires_grad}
    local = {"rank": context.rank, "local_rank": context.local_rank,
             "parameter_sha256": core.parameter_digest(model), "tensor_sha256": tensor_digest(parameters)}
    if optimizer is not None:
        cursors = [int(float(value["step"])) for value in optimizer.state.values() if "step" in value]
        local.update(optimizer_step_min=min(cursors, default=0), optimizer_step_max=max(cursors, default=0),
                     optimizer_state_count=len(cursors))
    if reload_error is not None:
        local["reload_max_abs_error"] = reload_error
    ranks = [None] * context.world_size
    dist.all_gather_object(ranks, local)
    accepted = (len({row["parameter_sha256"] for row in ranks}) == 1
                and len({row["tensor_sha256"] for row in ranks}) == 1
                and (optimizer is None or all(row["optimizer_step_min"] == row["optimizer_step_max"] == step for row in ranks))
                and (reload_error is None or all(row["reload_max_abs_error"] == 0. for row in ranks)))
    with primary_phase(context, "execution_audit primary action"):
        if context.primary:
            core.write_json(root / "execution_audits" / f"{event}-{step:06d}.json",
                            {"identity": identity, "event": event, "step": step, "world_size": context.world_size,
                             "accepted": accepted, "ranks": ranks})
    if not accepted:
        raise RuntimeError(f"Rank consistency failed at {event}, step {step}")
    return local["parameter_sha256"]


def _rank_rng() -> dict[str, Any]:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(torch.cuda.current_device())}


def _restore_rank_rng(state: Mapping[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], torch.cuda.current_device())


def save_checkpoint(context: DistributedContext, model: Any, tokenizer: Any, optimizer: Any,
                    scheduler: Any, step: int, root: Path, identity: str,
                    probe_tokens: list[list[int]], *, mode: str, alpha: float) -> Path:
    import torch
    import torch.distributed as dist
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    parameter_hash = execution_audit(context, model, root, identity, "checkpoint", step, optimizer=optimizer)
    states = [None] * context.world_size
    dist.all_gather_object(states, _rank_rng())
    destination = root / "checkpoints" / f"checkpoint-{step:06d}"
    with primary_phase(context, "save_checkpoint primary action"):
        if context.primary:
            scratch = Path(os.environ.get("META_LOCAL_SCRATCH", "/tmp/oi-ni-meta")) / identity[:20]
            local = scratch / f"checkpoint-{step:06d}"
            local.mkdir(parents=True, exist_ok=True)
            model.causal_lm.save_pretrained(local / "adapter", safe_serialization=True, save_embedding_layers=False)
            tokenizer.save_pretrained(local / "adapter")
            probe = core.capture_probe(model, tokenizer, probe_tokens)
            torch.save({"identity": identity, "step": step, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "parameter_names": list(trainable),
                        "rng_by_rank": states, "world_size": context.world_size,
                        "rng": {**states[0], "cuda": [states[0]["cuda"]]},
                        "legacy_rng_is_rank0_only": True}, local / "training_state.pt")
            core.write_json(local / "state.json", {"identity": identity, "step": step,
                            "world_size": context.world_size, "trainable_sha256": parameter_hash, "mode": mode, "alpha": alpha})
            core.write_json(local / "reload_probe.json", {"input_ids": probe_tokens, "scores": probe,
                            "batch_size": 1, "backend": model.backend_audit})
            core.publish(local, destination, identity)
            core.log("checkpoint_committed", step=step, world_size=context.world_size)
    _barrier()
    return destination


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]], indices: Sequence[int],
                             records: Sequence[Mapping[str, Any]], tokens: Sequence[Sequence[int]],
                             labels: Sequence[str]) -> None:
    if [row["row_index"] for row in rows] != list(indices):
        raise ValueError("Prediction global row coverage differs")
    for row, index in zip(rows, indices):
        record = records[index]
        if (row["impression_id"] != str(record["impression_id"])
                or row["acceptable_subcategories"] != list(record["acceptable_subcategories"])
                or row["input_sha256"] != core.input_digest(record, list(tokens[index]))):
            raise ValueError("Prediction source identity differs")
        scores = np.asarray(row["all_sid_log_probabilities"])
        if scores.shape != (len(labels),) or not np.isfinite(scores).all():
            raise ValueError("Invalid full SID score vector")
        order = np.lexsort((np.arange(len(labels)), -scores))[:20]
        if (row["ranked_indices"] != order.tolist()
                or row["ranked_labels"] != [labels[i] for i in order]
                or row["ranked_scores"] != [float(scores[i]) for i in order]):
            raise ValueError("Prediction ranking/tie rule differs")


def parallel_predict(context: DistributedContext, model: Any, tokens: list[list[int]],
                     records: list[dict[str, Any]], tokenizer: Any, labels: list[str],
                     output: Path, identity: str) -> dict[str, Any]:
    import torch
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    for start, stop in owned_evaluation_shards(len(records), context.rank, context.world_size):
        shard = output / f"predictions-{start:06d}.parquet"
        receipt = shard.with_suffix(".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if (saved["identity"] != identity or saved["rank"] != context.rank
                    or saved["sha256"] != core.sha256_file(shard)):
                raise ValueError("Distributed prediction shard identity differs")
            rows = pq.read_table(shard).to_pylist()
        else:
            rows = []
            for batch_start in range(start, stop, 8):
                batch_stop = min(batch_start + 8, stop)
                with torch.no_grad():
                    scores = model(**core.batch_inputs(tokens[batch_start:batch_stop], tokenizer.pad_token_id))
                    scores = torch.log_softmax(scores.float(), -1).cpu().numpy()
                for index, score in zip(range(batch_start, batch_stop), scores):
                    order = np.lexsort((np.arange(len(labels)), -score))[:20]
                    record = records[index]
                    rows.append({"row_index": index, "impression_id": str(record["impression_id"]),
                                 "user_id": str(record.get("user_id", "")),
                                 "input_sha256": core.input_digest(record, tokens[index]),
                                 "acceptable_subcategories": list(record["acceptable_subcategories"]),
                                 "ranked_labels": [labels[i] for i in order], "ranked_indices": order.tolist(),
                                 "ranked_scores": [float(score[i]) for i in order],
                                 "all_sid_log_probabilities": score.tolist()})
            validate_prediction_rows(rows, range(start, stop), records, tokens, labels)
            temporary = shard.with_name(shard.name + f".tmp-{os.getpid()}")
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, shard)
            core.write_json(receipt, {"identity": identity, "rank": context.rank,
                            "world_size": context.world_size, "rows": len(rows),
                            "sha256": core.sha256_file(shard)})
        validate_prediction_rows(rows, range(start, stop), records, tokens, labels)
    _barrier()
    metrics = None
    with primary_phase(context, "parallel_predict primary action"):
        if context.primary:
            merged = []
            for start in range(0, len(records), 256):
                shard = output / f"predictions-{start:06d}.parquet"
                receipt = json.loads(shard.with_suffix(".json").read_text())
                if (receipt["identity"] != identity or receipt["rank"] != (start // 256) % context.world_size
                        or receipt["sha256"] != core.sha256_file(shard)):
                    raise ValueError("Merged evaluation ownership/hash differs")
                merged.extend(pq.read_table(shard).to_pylist())
            validate_prediction_rows(merged, range(len(records)), records, tokens, labels)
            ranked = [[{"label": label, "score": score} for label, score in zip(row["ranked_labels"], row["ranked_scores"])]
                      for row in merged]
            metrics, _ = core.evaluate_ranked_outputs(records, ranked)
            metrics.update(identity=identity, no_test_time_adaptation=True, history_filtering=False,
                           projection_dtype="float32", tie_break="interest_index_ascending",
                           world_size=context.world_size, global_shard_size=256, batch_size=8)
            core.write_json(output / "metrics.json", metrics)
            core.log("evaluation_completed", samples=len(records), world_size=context.world_size)
    metrics = _broadcast(metrics, context)
    model.train()
    return metrics


def planned_local_chunks(context: DistributedContext, model: Any, episode_seed: int,
                          oi_chunks: Sequence[Any], ni_chunks: Sequence[Any], empty_chunks: Sequence[Any],
                          mode: str, cache_keys: Mapping[str, Sequence[Any]]) -> tuple[list[Any], list[Any], list[Any], dict]:
    from new_interest.training.oi_ni_meta_objective import LossChunk
    from new_interest.training.oi_ni_meta_rng import bind_planned_rng, plan_episode_rng
    plan = None
    with primary_phase(context, "planned_local_chunks primary action"):
        if context.primary:
            plan = plan_episode_rng(model, episode_seed, oi_chunks, ni_chunks,
                                    empty_chunks if mode == "meta_no_history" else None,
                                    mode=mode, cache_keys=cache_keys, planning_mode="real_preview")
    plan = _broadcast(plan, context)
    def select(chunks, role):
        return [LossChunk(bind_planned_rng(chunks[i].loss_fn, plan[role][i]), weight=chunks[i].weight)
                for i in owned_positions(len(chunks), context.rank, context.world_size)]
    return (select(oi_chunks, "oi"), select(ni_chunks, "ni"),
            select(empty_chunks, "no_history") if mode == "meta_no_history" else [], plan["audit"])


def _cpu_tree(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _serial_adam_reference(model: Any, optimizer: Any, gradients: Mapping[str, Any],
                            config: Mapping[str, Any]) -> dict[str, Any]:
    """Engineering-gate update only; restore the deployed state before return."""
    import torch
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    original = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    state = _cpu_tree(optimizer.state_dict())
    try:
        for name, parameter in parameters.items():
            parameter.grad = gradients[name].to(parameter)
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        return {name: value.detach().cpu().clone() for name, value in parameters.items()}
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(original[name].to(parameter))
        optimizer.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)


def run_training(context: DistributedContext, config: dict[str, Any], source: dict[str, Any],
                 snapshot: str, data: Path, root: Path, *, phase: str, mode: str, alpha: float,
                 selection_binding: Mapping[str, str] | None = None,
                 budget_binding: Mapping[str, Any] | None = None,
                 lineage_files: Mapping[str, Path] | None = None) -> dict[str, Any]:
    import torch
    from new_interest.training.oi_ni_meta_objective import compute_meta_gradients
    from new_interest.training.oi_ni_meta_distributed_objective import compute_distributed_meta_gradients
    tolerances = GateTolerances(**source["gate_tolerances"])
    steps = int(config["gate_steps"] if phase == "gate" else
                budget_binding["steps"] if phase == "compute_matched" else config["pilot_steps"])
    settings = {"config": config, "source_identity": source["source_identity"],
                "legacy_source_identity": source["legacy_source_identity"], "execution_spec": EXECUTION_SPEC,
                "phase": phase, "mode": mode, "alpha": alpha, "steps": steps,
                "eval_dev": phase in {"pilot", "compute_matched"},
                "selection_binding": dict(selection_binding or {}), "budget_binding": dict(budget_binding or {})}
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    completed = None
    checkpoint = None
    with primary_phase(context, "run_training primary action"):
        if context.primary:
            root.mkdir(parents=True, exist_ok=True)
            if (root / "resolved_config.json").exists() and json.loads((root / "resolved_config.json").read_text())["identity"] != identity:
                raise ValueError("Different execution identity cannot reuse this run directory")
            if (root / "success.json").exists() and (root / "manifest.json").exists():
                core.verify_manifest(root, identity)
                completed = json.loads((root / "success.json").read_text())
                if phase == "gate":
                    validate_gate_evidence(root, identity, source, config)
            if completed is None:
                core.write_json(root / "resolved_config.json", {"identity": identity, **settings})
                core.write_json(root / "source_identity.json", source)
                for label, path in (lineage_files or {}).items():
                    destination = root / "lineage" / f"{label}.json"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and core.sha256_file(destination) != core.sha256_file(path):
                        raise ValueError(f"Existing lineage artifact differs: {label}")
                    shutil.copyfile(path, destination)
                checkpoint = core.latest_checkpoint(root, identity)
    completed = _broadcast(completed, context)
    if completed is not None:
        return completed
    checkpoint_string = _broadcast(str(checkpoint) if checkpoint else None, context)
    checkpoint = Path(checkpoint_string) if checkpoint_string else None
    model, tokenizer, _, initialization = core.make_model(config, snapshot, data / "interest_token_mapping.parquet",
                                                          checkpoint / "adapter" if checkpoint else None)
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    _replica_digest(parameters, context)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=config["outer_learning_rate"], weight_decay=config["weight_decay"])
    warmup = max(1, round(steps * config["warmup_ratio"]))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        fraction = min(1., (step - warmup) / max(1, steps - warmup))
        return .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    start_step = 0
    if checkpoint:
        core.verify_manifest(checkpoint, identity)
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if (state["identity"] != identity or state["world_size"] != context.world_size
                or state["parameter_names"] != list(parameters) or len(state["rng_by_rank"]) != context.world_size):
            raise ValueError("Checkpoint execution topology/parameter identity differs")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rank_rng(state["rng_by_rank"][context.rank])
        start_step = int(state["step"])
    initial_hash = execution_audit(context, model, root, identity,
                                   "restore" if checkpoint else "initialization", start_step, optimizer=optimizer)
    with primary_phase(context, "run_training primary action"):
        if context.primary:
            audit = {"backend": model.backend_audit, "initialization": initialization,
                     "initial_trainable_sha256": initial_hash, "source_identity": source["source_identity"],
                     "world_size": context.world_size, "resumed_from_step": start_step,
                     "trainable_parameters": {name: list(value.shape) for name, value in parameters.items()},
                     "trainable_dtypes": {name: str(value.dtype) for name, value in parameters.items()},
                     "device": torch.cuda.get_device_name(context.local_rank),
                     "memory_bytes": torch.cuda.get_device_properties(context.local_rank).total_memory}
            if not (root / "model_audit.json").exists():
                if checkpoint:
                    raise ValueError("Resume requires the original initialization audit")
                core.write_json(root / "model_audit.json", audit)
            else:
                core.write_json(root / "attempts" / f"{uuid.uuid4().hex}.json", audit)
    oi = pq.read_table(data / "fit_oi.parquet").to_pylist()
    ni = pq.read_table(data / "fit_ni.parquet").to_pylist()
    validation = [core.as_eval_record(row) for row in pq.read_table(data / "validation_ni.parquet").to_pylist()]
    episodes = pq.read_table(data / "episodes.parquet").to_pylist()
    oi_tokens = core.encode_records(tokenizer, oi, max_length=config["max_seq_length"])
    ni_tokens = core.encode_records(tokenizer, ni, max_length=config["max_seq_length"])
    empty_tokens = core.encode_records(tokenizer, oi, max_length=config["max_seq_length"], field="no_history_prompt")
    validation_tokens = core.encode_records(tokenizer, validation, max_length=config["max_seq_length"])
    labels = [str(row["target_subcategory"]) for row in core.load_interest_token_mapping(data / "interest_token_mapping.parquet")]
    with primary_phase(context, "run_training primary action"):
        if context.primary:
            core.write_json(root / "token_audit.json", {"oi": len(oi), "ni": len(ni), "validation": len(validation),
                            "oi_max": max(map(len, oi_tokens)), "ni_max": max(map(len, ni_tokens)),
                            "validation_max": max(map(len, validation_tokens)), "target_in_input": False,
                            "over_length": 0, "microbatch_size": 1, "global_support": 16, "global_query": 16})
    durations = []
    model.train()
    if phase == "gate" and checkpoint is None:
        import torch.distributed as dist
        support_index = max(range(len(oi_tokens)), key=lambda i: len(oi_tokens[i]))
        query_index = max(range(len(ni_tokens)), key=lambda i: len(ni_tokens[i]))
        longest_f = core.loss_chunks([support_index], oi, oi_tokens, tokenizer.pad_token_id, 1)
        longest_g = core.loss_chunks([query_index], ni, ni_tokens, tokenizer.pad_token_id, 1)
        torch.cuda.reset_peak_memory_stats(); _barrier(); torch.cuda.synchronize()
        long_begin = time.monotonic()
        local_f, local_g, _, long_plan = planned_local_chunks(
            context, model, int(config["seed"]), longest_f, longest_g, [], "meta",
            {"oi": [len(oi_tokens[support_index])], "ni": [len(ni_tokens[query_index])]})
        worst = compute_distributed_meta_gradients(model, local_f, local_g, alpha=alpha, mode="meta",
                    reduction="ordered", global_positions={
                        "oi": owned_positions(1, context.rank, context.world_size),
                        "ni": owned_positions(1, context.rank, context.world_size)})
        torch.cuda.synchronize()
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        unchanged = execution_audit(context, model, root, identity, "long_sequence_no_update", 0, optimizer=optimizer)
        if unchanged != initial_hash:
            raise RuntimeError("Gradient-only longest-input gate changed deployed parameters")
        with primary_phase(context, "run_training primary action"):
            if context.primary:
                core.write_json(root / "long_sequence_hvp_audit.json", {
                    "identity": identity, "scope": "gradient-only engineering gate; not a method score",
                    "oi_tokens": len(oi_tokens[support_index]), "ni_tokens": len(ni_tokens[query_index]),
                    "loss": float(worst.loss), "hvp_norm": float(worst.hvp_norm),
                    "seconds": time.monotonic() - long_begin, "peak_gpu_bytes": int(peak.item()),
                    "parameter_update_performed": False, "world_size": context.world_size,
                    "global_support": 1, "global_query": 1, "owner_rank": 0, "empty_ranks": [1,2,3], "rng_plan": long_plan})
        del worst; torch.cuda.empty_cache()
    for index in range(start_step, steps):
        episode = episodes[index % len(episodes)]
        support, query = episode["support_indices"], episode["query_indices"]
        if len(support) != 16 or len(query) != 16:
            raise ValueError("Original global episode must contain 16 support and 16 query rows")
        seed = int(config["seed"]) + 104729 * index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        f = core.loss_chunks(support, oi, oi_tokens, tokenizer.pad_token_id, 1)
        g = core.loss_chunks(query, ni, ni_tokens, tokenizer.pad_token_id, 1)
        empty = core.loss_chunks(support, oi, empty_tokens, tokenizer.pad_token_id, 1)
        _barrier(); torch.cuda.synchronize()
        begin = time.monotonic()
        stage_begin = time.monotonic()
        local_f, local_g, local_empty, plan_audit = planned_local_chunks(
            context, model, seed, f, g, empty, mode,
            {"oi": [len(oi_tokens[i]) for i in support], "ni": [len(ni_tokens[i]) for i in query],
             "no_history": [len(empty_tokens[i]) for i in support]})
        torch.cuda.synchronize()
        planning_seconds = _rank_max_seconds(time.monotonic() - stage_begin)
        reference = None
        serial_seconds = 0.
        with primary_phase(context, "run_training primary action"):
            if phase == "gate" and context.primary:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize(); serial_begin = time.monotonic()
                serial = compute_meta_gradients(model, f, g, alpha=alpha, mode=mode,
                                                no_history_oi=empty if mode == "meta_no_history" else None)
                reference = {"gradients": _cpu_tree(serial.gradients),
                             "loss": serial.loss.detach().cpu(), "inner_grad_norm": serial.inner_grad_norm.detach().cpu(),
                             "hvp_norm": serial.hvp_norm.detach().cpu()}
                del serial
                torch.cuda.synchronize(); serial_seconds = time.monotonic() - serial_begin
        if phase == "gate":
            serial_seconds = _rank_max_seconds(serial_seconds)
        _barrier()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); stage_begin = time.monotonic()
        result = compute_distributed_meta_gradients(model, local_f, local_g, alpha=alpha, mode=mode,
                            local_no_history_oi=local_empty if mode == "meta_no_history" else None,
                            reduction="ordered", global_positions={
                                "oi": owned_positions(16, context.rank, context.world_size),
                                "ni": owned_positions(16, context.rank, context.world_size),
                                "no_history_oi": owned_positions(16, context.rank, context.world_size)
                                                 if mode == "meta_no_history" else [],
                            })
        gradient_hash = _replica_digest(result.gradients, context)
        torch.cuda.synchronize()
        objective_seconds = _rank_max_seconds(time.monotonic() - stage_begin)
        gate_comparison = None
        serial_update = None
        comparison_begin = time.monotonic()
        with primary_phase(context, "run_training primary action"):
            if phase == "gate" and context.primary:
                gate_comparison = {
                    "identity": identity, "step": index + 1, "kind": "before_update",
                    "source_identity": source["source_identity"],
                    "loss": compare_tensors({"loss": result.loss}, {"loss": reference["loss"]},
                                             atol=tolerances.loss_atol, rtol=tolerances.loss_rtol),
                    "norms": compare_tensors({"inner": result.inner_grad_norm, "hvp": result.hvp_norm},
                                              {"inner": reference["inner_grad_norm"], "hvp": reference["hvp_norm"]},
                                              atol=tolerances.gradient_atol, rtol=tolerances.gradient_rtol),
                    "gradients": compare_tensors(result.gradients, reference["gradients"],
                                                  atol=tolerances.gradient_atol, rtol=tolerances.gradient_rtol,
                                                  relative_l2_limit=tolerances.gradient_relative_l2)}
                if all(gate_comparison[key]["accepted"] for key in ("loss", "norms", "gradients")):
                    serial_update = _serial_adam_reference(model, optimizer, reference["gradients"], config)
                core.write_json(root / "equivalence" / f"before-update-{index+1:06d}.json", gate_comparison)
        gate_comparison = _broadcast(gate_comparison, context)
        if gate_comparison is not None and not all(gate_comparison[key]["accepted"] for key in ("loss", "norms", "gradients")):
            raise RuntimeError("Four-rank versus serial gradient compatibility gate failed")
        torch.cuda.synchronize()
        comparison_seconds = _rank_max_seconds(time.monotonic() - comparison_begin) if phase == "gate" else 0.
        torch.cuda.synchronize(); update_begin = time.monotonic()
        for name, parameter in parameters.items():
            parameter.grad = result.gradients[name].to(parameter)
        if index == 0:
            sid_audit = {}
            for name, parameter in parameters.items():
                if "trainable_tokens_delta" in name:
                    norms = parameter.grad.float().reshape(108, -1).norm(dim=1)
                    if not bool(torch.isfinite(norms).all() and (norms > 0).all()):
                        raise ValueError("Missing/nonfinite SID row gradients")
                    sid_audit[name] = {"finite_rows": 108, "nonzero_rows": 108}
            if not sid_audit:
                raise ValueError("No trainable SID rows found")
            with primary_phase(context, "run_training primary action"):
                if context.primary:
                    core.write_json(root / "sid_gradient_audit.json", sid_audit)
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config["max_grad_norm"], error_if_nonfinite=True)
        applied_lr = optimizer.param_groups[0]["lr"]
        optimizer.step(); scheduler.step()
        parameter_hash = execution_audit(context, model, root, identity, "after_update", index + 1, optimizer=optimizer)
        torch.cuda.synchronize()
        update_seconds = _rank_max_seconds(time.monotonic() - update_begin)
        after_comparison = None
        comparison_begin = time.monotonic()
        with primary_phase(context, "run_training primary action"):
            if phase == "gate" and context.primary:
                after_comparison = compare_tensors(parameters, serial_update,
                                                    atol=tolerances.update_atol, rtol=tolerances.update_rtol)
                after_comparison.update(identity=identity, step=index + 1, kind="after_update",
                                        source_identity=source["source_identity"])
                core.write_json(root / "equivalence" / f"after-update-{index+1:06d}.json", after_comparison)
        after_comparison = _broadcast(after_comparison, context)
        if after_comparison is not None and not after_comparison["accepted"]:
            raise RuntimeError("Four-rank versus serial Adam update compatibility gate failed")
        torch.cuda.synchronize()
        if phase == "gate":
            comparison_seconds += _rank_max_seconds(time.monotonic() - comparison_begin)
        torch.cuda.synchronize(); elapsed = time.monotonic() - begin
        durations.append(elapsed)
        with primary_phase(context, "run_training primary action"):
            if context.primary:
                row = {"step": index+1, "mode": mode, "seconds": elapsed, "loss": float(result.loss),
                       "oi_loss": float(result.oi_loss), "ni_loss": float(result.ni_loss),
                       "ni_fast_loss": float(result.adapted_ni_loss), "hvp_norm": float(result.hvp_norm),
                       "inner_grad_norm": float(result.inner_grad_norm), "gradient_norm": float(norm),
                       "applied_lr": applied_lr, "next_lr": optimizer.param_groups[0]["lr"],
                       "gradient_sha256": gradient_hash, "parameter_sha256": parameter_hash,
                       "rng_planning_seconds": planning_seconds,
                       "serial_reference_objective_seconds": serial_seconds,
                       "distributed_objective_seconds": objective_seconds,
                       "distributed_outer_update_seconds": update_seconds,
                       "gate_comparison_seconds": comparison_seconds,
                       "equivalent_distributed_step_seconds": planning_seconds + objective_seconds + update_seconds,
                       "equivalent_distributed_step_seconds_is_diagnostic": phase == "gate",
                       "stage_timing_aggregation": "maximum_across_four_ranks",
                       "rng_plan": plan_audit, "global_support": 16, "global_query": 16,
                       "resumed_from_step": start_step, "timing_includes_serial_gate": phase == "gate"}
                with (root / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                core.log("training_step", **row)
        del result, reference, serial_update
        if (index+1) % config["checkpoint_every"] == 0 or index+1 == steps:
            checkpoint = save_checkpoint(context, model, tokenizer, optimizer, scheduler, index+1, root,
                                         identity, validation_tokens[:4], mode=mode, alpha=alpha)
        if (index+1) % config["validation_every"] == 0 or index+1 == steps:
            subset = EXECUTION_SPEC["gate_validation_rows"] if phase == "gate" else len(validation)
            parallel_predict(context, model, validation_tokens[:subset], validation[:subset], tokenizer, labels,
                             root / f"validation-{index+1:06d}", identity)
    if checkpoint is None:
        raise ValueError("No committed distributed checkpoint")
    subset = EXECUTION_SPEC["gate_validation_rows"] if phase == "gate" else len(validation)
    validation_metrics = parallel_predict(context, model, validation_tokens[:subset], validation[:subset], tokenizer,
                                           labels, root / f"validation-{steps:06d}", identity)
    final_hash = execution_audit(context, model, root, identity, "final_deployed", steps, optimizer=optimizer)
    del model, optimizer, scheduler, parameters
    gc.collect(); torch.cuda.empty_cache(); _barrier()
    model, tokenizer, _, _ = core.make_model(config, snapshot, data / "interest_token_mapping.parquet", checkpoint / "adapter")
    probe = json.loads((checkpoint / "reload_probe.json").read_text())
    actual = core.capture_probe(model, tokenizer, probe["input_ids"])
    reload_error = float(np.max(np.abs(np.asarray(actual) - np.asarray(probe["scores"]))))
    import torch.distributed as dist
    errors = [None] * context.world_size
    dist.all_gather_object(errors, reload_error)
    reloaded_hash = execution_audit(context, model, root, identity, "fresh_reload", steps, reload_error=reload_error)
    if reloaded_hash != final_hash or any(error != 0. for error in errors):
        raise RuntimeError("Fresh distributed adapter reload changed parameters or probe logits")
    with primary_phase(context, "run_training primary action"):
        if context.primary:
            core.write_json(root / "fresh_reload_audit.json", {"max_abs_error_by_rank": errors,
                            "max_abs_error": 0., "deployed_outer_parameters": True,
                            "trainable_sha256": final_hash, "trainable_tensors_exact": True, "world_size": context.world_size})
    if phase in {"pilot", "compute_matched"}:
        dev = pq.read_table(data / "dev_full_grouped.parquet").to_pylist()
        dev_tokens = core.encode_records(tokenizer, dev, max_length=config["max_seq_length"])
        metrics = parallel_predict(context, model, dev_tokens, dev, tokenizer, labels, root / "dev", identity)
        if metrics["samples"] != 31392 or metrics["supported_interest_count"] != 104:
            raise ValueError("Full Dev topology differs")
    timing = timing_totals(root / "training.jsonl", steps, context.world_size, phase=phase) if context.primary else None
    timing = _broadcast(timing, context)
    success = {"identity": identity, "source_identity": source["source_identity"],
               "legacy_source_identity": source["legacy_source_identity"], "phase": phase, "mode": mode,
               "alpha": alpha, "steps": steps, "world_size": context.world_size,
               "checkpoint_relative": str(checkpoint.relative_to(root)), "validation_metrics": validation_metrics,
               "deployed_trainable_sha256": final_hash, "fresh_reload_max_abs_error": 0.,
               **timing,
               "dev_evaluated": phase in {"pilot", "compute_matched"}}
    with primary_phase(context, "run_training primary action"):
        if context.primary:
            gate_evidence = validate_gate_evidence(root, identity, source, config) if phase == "gate" else None
            core.write_json(root / "success.json", success)
            if phase == "gate":
                core.write_json(root / "compatibility_gate.json", {"accepted": True, **success,
                                "gate_tolerances": asdict(tolerances), "execution_spec": EXECUTION_SPEC,
                                "evidence_sha256": gate_evidence})
            core.write_json(root / "manifest.json", {"identity": identity, "files": core.inventory(root)})
    _barrier()
    del model; gc.collect(); torch.cuda.empty_cache()
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["gate", "pilot", "compute_matched"], required=True)
    parser.add_argument("--mode", choices=["mixed", "meta", "meta_no_history"], default="meta")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--selection-lineage-file", type=Path)
    parser.add_argument("--compatibility-gate-file", type=Path)
    parser.add_argument("--budget-file", type=Path)
    args = parser.parse_args()
    if args.phase == "gate" and args.mode != "meta":
        raise ValueError("The compatibility gate must exercise the exact-meta HVP path")
    if args.phase == "compute_matched" and (args.mode != "mixed" or args.budget_file is None):
        raise ValueError("Compute-matched control requires Mixed and an explicit timing-budget receipt")
    context = initialize_distributed()
    try:
        config = json.loads(args.config.read_text())
        if (config["microbatch_size"] != 1 or config["episode_classes"] != 8
                or config["support_per_class"] != 2 or config["query_per_class"] != 2
                or config["pilot_steps"] != 540 or config["evaluation_batch_size"] != 8):
            raise ValueError("The distributed engine cannot change the frozen episode/evaluation protocol")
        data = Path(config["data_dir"])
        source = None
        snapshot = None
        with primary_phase(context, "main primary action"):
            if context.primary:
                data_hashes = core.validate_frozen_data(config, data)
                source = source_receipt(Path.cwd(), args.config, data_hashes)
                args.root.mkdir(parents=True, exist_ok=True)
                source_path = args.root / "source_identity.json"
                if source_path.exists() and json.loads(source_path.read_text())["source_identity"] != source["source_identity"]:
                    raise ValueError("Output root belongs to a different execution source")
                core.write_json(source_path, source)
                snapshot = core.prepare_snapshot(config, args.root)
                import torch
                core.write_json(args.root / "environment.json", {"packages": {name: importlib.metadata.version(name)
                                for name in ("torch", "transformers", "peft", "accelerate", "pyarrow", "tokenizers",
                                             "safetensors", "huggingface-hub")}, "world_size": context.world_size,
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(context.local_rank),
                                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
        source = _broadcast(source, context)
        snapshot = _broadcast(snapshot, context)
        selection_binding = None
        budget_binding = None
        lineage_files = {"protocol": args.config}
        alpha = float(config["gate_alpha"])
        if args.phase in {"pilot", "compute_matched"}:
            if any(value is None for value in (args.selection_file, args.selection_lineage_file, args.compatibility_gate_file)):
                raise ValueError("Pilot requires legacy selection, accepted lineage, and four-rank compatibility gate")
            with primary_phase(context, "main primary action"):
                if context.primary:
                    alpha, selection_binding = validate_selection_lineage(
                        args.selection_file, args.selection_lineage_file, args.compatibility_gate_file, source, config)
            alpha, selection_binding = _broadcast((alpha, selection_binding), context)
            lineage_files.update(selection=args.selection_file, selection_lineage=args.selection_lineage_file,
                                 compatibility_gate=args.compatibility_gate_file)
        if args.phase == "compute_matched":
            with primary_phase(context, "main primary action"):
                if context.primary:
                    budget_steps, budget_sha = validate_compute_budget(args.budget_file, source, config)
                    budget_binding = {"steps": budget_steps, "budget_sha256": budget_sha}
            budget_binding = _broadcast(budget_binding, context)
            lineage_files["compute_budget"] = args.budget_file
        if args.mode == "mixed":
            alpha = 0.
        output_name = "mixed_compute" if args.phase == "compute_matched" else args.mode
        result = run_training(context, config, source, snapshot, data, args.root / output_name,
                               phase=args.phase, mode=args.mode, alpha=alpha, selection_binding=selection_binding,
                               budget_binding=budget_binding, lineage_files=lineage_files)
        with primary_phase(context, "main primary action"):
            if context.primary:
                core.write_json(args.root / f"{output_name}_success.json", result)
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
