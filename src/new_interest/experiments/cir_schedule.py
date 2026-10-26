"""Four approved CIR duration/mixing conditions on the accepted numerical runtime."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import math
from pathlib import Path
import platform

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments import cir_warmup as core

MIXED = Path("data/training/mind_cir_schedule_108_v1/mixed.parquet")
MODES = ("staged", "mixed3", "mixed5")


def stage_specs(project: Path, mode: str) -> list[dict]:
    core.require(mode in MODES, "Unknown CIR schedule mode")
    if mode == "staged":
        sequence = [("cir5", "cir", 17271, 5, None),
                    ("ni3", "ni", 21600, 3, "cir5"),
                    ("ni5", "ni", 21600, 5, "cir5")]
    else:
        epochs = 3 if mode == "mixed3" else 5
        sequence = [(mode, "mixed", 38871, epochs, None)]
    return [{"key": key, "task": task, "samples": n, "epochs": epochs, "parent": parent,
             "world_size": 4, "per_device_train_batch_size": 2,
             "global_batch_size": 32, "accumulation": 32,
             "train_file": project / (MIXED if task == "mixed" else core.DATA / f"{task}.parquet")}
            for key, task, n, epochs, parent in sequence]


def conditions(mode: str) -> list[tuple[str, list[str]]]:
    core.require(mode in MODES, "Unknown CIR schedule mode")
    return [("cir5_ni3", ["cir5", "ni3"]), ("cir5_ni5", ["cir5", "ni5"])] if mode == "staged" else [(mode, [mode])]


def verify_mixed_inputs(project: Path) -> dict:
    """Check the actual writer output against every original field and row."""
    cir = pq.read_table(project / core.DATA / "cir.parquet")
    ni = pq.read_table(project / core.DATA / "ni.parquet")
    mixed = pq.read_table(project / MIXED)
    expected = {"cross_interest_reconstruction": cir, "new_interest": ni}
    actual = mixed.to_pylist()
    core.require(len(actual) == len(cir) + len(ni) == 38871, "Mixed row count changed")
    core.require(len({row["sample_id"] for row in actual}) == len(actual), "Mixed sample IDs collide")
    offset = 0
    for task, source in expected.items():
        for row, original in zip(actual[offset:offset + len(source)], source.to_pylist()):
            core.require(row["training_task"] == task, "Mixed task label changed")
            core.require(all(row[k] == v for k, v in original.items()), "Mixed modified a source field")
        offset += len(source)
    return {"status": "passed", "rows": len(actual),
            "by_task": dict(Counter(row["training_task"] for row in actual)),
            "sampling": "one uniform global permutation per epoch; no task or class resampling",
            "source_rows_and_fields_preserved": True, "mixed_sha256": core.sha256(project / MIXED)}


def initialize(project: Path, root: Path, mode: str) -> dict:
    import torch
    contract = core.read(project / core.CONTRACT)
    manifest = project / "manifests/cir_bundle.json"
    for name, digest in core.read(manifest)["files"].items():
        core.require(core.sha256(project / name) == digest, f"Frozen payload changed: {name}")
    identity = {"version": contract["version"], "mode": mode, "bundle_sha256": core.sha256(manifest)}
    if (root / "identity.json").exists():
        core.require(core.read(root / "identity.json") == identity, "Output has another condition identity")
    else:
        core.write(root / "identity.json", identity)
    packages = {name: importlib.metadata.version(name) for name in contract["worker_packages"]}
    core.require(packages == contract["worker_packages"], "Worker package versions changed")
    core.require(torch.cuda.device_count() == 4, "Four GPUs required")
    devices = [{"name": torch.cuda.get_device_properties(i).name,
                "bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(4)]
    core.require(all("A100" in x["name"] and x["bytes"] >= 79 * 1024**3 for x in devices), "Unexpected GPU allocation")
    core.write(root / "environment.json", {"packages": packages, "devices": devices,
               "python": platform.python_version(), "cuda": torch.version.cuda, "kernel_profile": "torch-fallback-explicit"})
    _, shared = core.bind()
    snapshot = shared.prepare_snapshot(project, "2b", root / "snapshot_audit.json")
    core.require(core.read(root / "snapshot_audit.json")["files"] == core.read(project / "manifests/h50_snapshot.json")["files"], "Snapshot changed")
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    from new_interest.training.sft_data import audit_token_lengths
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    register_semantic_interest_tokens(tokenizer, project / shared.MAPPING)
    tasks = {s["task"]: s for s in stage_specs(project, mode)}
    for task, spec in tasks.items():
        audit = audit_token_lengths(spec["train_file"], tokenizer, max_seq_length=2048,
                    enable_thinking=False, output_path=root / f"token_audit_{task}.json")
        core.require(audit["samples"] == spec["samples"] and audit["over_max_seq_length"] == 0
                     and audit["min_assistant_tokens"] == audit["max_assistant_tokens"] == 3, "Token audit failed")
    reused = core.reuse_verified_gates(project, Path(contract["reuse_verified_gates"]))
    core.write(root / "reused_runtime_gates.json", {**identity, **reused})
    if mode.startswith("mixed"):
        core.write(root / "mixed_input_readback.json", verify_mixed_inputs(project))
        # New mixed-data boundary: exercise both task labels and a padded tail
        # on four GPUs, then reuse the same real evaluator before formal SFT.
        table = pq.read_table(project / MIXED)
        task_values = table["training_task"].to_pylist()
        positions = []
        for task, n in (("cross_interest_reconstruction", 20), ("new_interest", 21)):
            eligible = [i for i, value in enumerate(task_values) if value == task]
            longest = max(eligible, key=lambda i: len(table["prompt"][i].as_py()))
            positions.extend([longest] + [i for i in eligible if i != longest][:n-1])
        subset = table.take(pa.array(positions, type=pa.int64()))
        path = root / "gates/mixed41.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            core.require(pq.read_table(path).equals(subset), "Mixed gate input changed")
        else:
            pq.write_table(subset, path, compression="zstd")
        spec = {**stage_specs(project, mode)[0], "key": "ddp_mixed", "train_file": path, "samples": 41, "epochs": 1}
        stage = core.train_stage(project, root / "gates", spec)
        evaluation = core.evaluate(project, root / "gates", "ddp_mixed", gates=True)
        core.write(root / "gates/accepted.json", {**identity, "stage": stage, "evaluation": evaluation})
    return identity


def worker(project: Path, root: Path, mode: str):
    project, root = project.resolve(), root.resolve()
    identity = initialize(project, root, mode)
    contract = core.read(project / core.CONTRACT)
    formal = root / "formal"
    stages = {}
    results = {}
    print("CIR_SCHEDULE_FORMAL_START", mode, flush=True)
    for spec in stage_specs(project, mode):
        stages[spec["key"]] = core.train_stage(project, formal, spec)
        for name, keys in conditions(mode):
            if keys[-1] != spec["key"]:
                continue
            folder = root / "conditions" / name
            result = core.evaluate(project, folder, formal / spec["key"], gates=False)
            record = {**identity, "status": "accepted", "condition": name,
                      "stages": [stages[k] for k in keys], "stage_keys": keys,
                      "evaluation": result, "protocol": contract["conditions"][name],
                      "baseline_reference": core.read(project / "manifests/h50_baseline.json"),
                      "prior_cir3_reference": contract["existing_cir3_reference"],
                      "large_artifacts_policy": "remote_only"}
            core.write(folder / "complete.json", record)
            results[name] = record
            print("CIR_CONDITION_ACCEPTED", name, result, flush=True)
    core.write(root / "complete.json", {**identity, "status": "accepted", "conditions": results})
    print("CIR_SCHEDULE_JOB_ACCEPTED", mode, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    worker(args.project, args.root, args.mode)
