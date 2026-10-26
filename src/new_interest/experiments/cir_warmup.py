"""CIR warm-up followed by unchanged NI SFT, on a frozen accepted SFT runtime.

The submitted bundle supplies the previously accepted Trainer, global sampler,
checkpoint publisher, and independent adapter/evaluation readers. This module
owns only CIR stage orchestration and its new input/lineage contract.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256

CONTRACT = Path("configs/experiments/mind_cir_108_v1.json")
DATA = Path("data/training/mind_cir_108_v1")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(project=None):
    from new_interest.experiments import old_new_interest as original
    from new_interest.experiments import backbone_scaling as shared
    original.CONTRACT = shared.CONTRACT = CONTRACT
    shared.TRAIN, shared.DEV, shared.MAPPING = DATA / "ni.parquet", DATA / "dev_full_grouped.parquet", DATA / "interest_token_mapping.parquet"
    shared.CANDIDATE_MAPPING = DATA / "interest_token_mapping.parquet"
    if project is not None:
        config = read(Path(project) / "configs/sft/mind_cir_base_v1.json")
        shared.MAPPING = Path(config["semantic_id_token_mapping_file"])
    return original, shared


def reference_kernels():
    """Use one explicit numerical path in training, reload, and evaluation."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model
    model.is_fast_path_available = False
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        setattr(model, name, None)


def formal_schedule(project: Path) -> dict:
    """Resolve the epoch horizon independently of immutable dataset statistics."""
    requested = read(project / CONTRACT).get("training_schedule", {"cir_epochs": 1, "ni_epochs": 3})
    require(set(requested) == {"cir_epochs", "ni_epochs"}, "Unknown CIR schedule field")
    require(all(type(v) is int and v > 0 for v in requested.values()), "Epochs must be positive integers")
    require(requested["ni_epochs"] == 3, "The original NI stage must remain three epochs")
    n = read(project / DATA / "statistics.json")["cir_rows"]
    return {**requested, "cir_optimizer_steps": math.ceil(n / 32) * requested["cir_epochs"],
            "ni_optimizer_steps": 675 * requested["ni_epochs"]}


def stage_specs(project: Path, root: Path, *, gates: bool) -> list[dict]:
    if gates:
        sequence = [("single_cir", "cir", None, 32, 2, 1, 1, 8),
                    ("single_ni", "ni", "single_cir", 32, 2, 1, 1, 8),
                    ("ddp_cir", "cir", None, 41, 1, 4, 2, 32),
                    ("ddp_ni", "ni", "ddp_cir", 41, 1, 4, 2, 32)]
    else:
        n = read(project / DATA / "statistics.json")["cir_rows"]
        schedule = formal_schedule(project)
        sequence = [("cir", "cir", None, n, schedule["cir_epochs"], 4, 2, 32),
                    ("ni", "ni", "cir", 21600, schedule["ni_epochs"], 4, 2, 32)]
    return [{"key": key, "task": task, "parent": parent, "samples": n, "epochs": epochs,
             "world_size": world, "per_device_train_batch_size": micro,
             "global_batch_size": batch, "accumulation": batch,
             "train_file": root / "gate_data" / f"{task}-{n}.parquet" if gates else project / DATA / f"{task}.parquet"}
            for key, task, parent, n, epochs, world, micro, batch in sequence]


def make_gate_data(project, root):
    for task in ("cir", "ni"):
        table = pq.read_table(project / DATA / f"{task}.parquet")
        ids = table["sample_id"].to_pylist()
        order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"cir-gate:42:{ids[i]}".encode()).digest())
        # Exercise the longest real CIR/NI prompt in the gate as well.
        prompts = table["prompt"].to_pylist()
        longest = max(range(len(prompts)), key=lambda i: len(prompts[i]))
        order = [longest] + [i for i in order if i != longest]
        for n in (32, 41):
            target = root / "gate_data" / f"{task}-{n}.parquet"
            subset = table.take(pa.array(order[:n], type=pa.int64()))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(pq.read_table(target).equals(subset), "Existing gate data changed")
            else:
                pq.write_table(subset, target, compression="zstd")


def stage_config(project, root, spec):
    from new_interest.training import old_new_execution as cursor
    label = f"cir/{spec['key']}"
    cursor.STAGES.add(label)
    contract = cursor.make_cursor_contract(spec["train_file"], stage=label,
        world_size=spec["world_size"], per_device_batch_size=spec["per_device_train_batch_size"],
        epochs=spec["epochs"], global_batch_size=spec["global_batch_size"])
    cursor_file = root / spec["key"] / "cursor_contract.json"
    if cursor_file.exists():
        require(read(cursor_file) == contract, "Stage cursor identity changed")
    else:
        write(cursor_file, contract)
    config = read(project / "configs/sft/mind_cir_base_v1.json")
    config.update(train_file=str(spec["train_file"]), epochs=spec["epochs"],
                  output_dir=str(root / spec["key"]),
                  checkpoint_dir=f"/scratch/cir-{hashlib.sha256(str(root).encode()).hexdigest()[:12]}-{spec['key']}",
                  persistent_checkpoint_dir=str(root / f"checkpoints-{spec['key']}"),
                  resume_from_checkpoint="auto", old_new_cursor_contract_file=str(cursor_file),
                  ignore_data_skip=False, per_device_train_batch_size=spec["per_device_train_batch_size"],
                  gradient_accumulation_steps=spec["global_batch_size"] // (spec["world_size"] * spec["per_device_train_batch_size"]),
                  run_name=f"{read(project / CONTRACT)['version']}-{root.name}-{spec['key']}-seed42",
                  logging_steps=1 if spec["samples"] < 100 else 5)
    if spec["parent"]:
        config["adapter_path"] = str(root / spec["parent"] / "final_adapter")
        config.pop("expected_initial_adapter_state_sha256", None)
    else:
        config.pop("adapter_path", None)
    return config


def run(command, project, log, devices=None):
    env = dict(os.environ)
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    log.parent.mkdir(parents=True, exist_ok=True)
    # Active logs remain node-local. Only closed logs are published.
    local = Path("/scratch") / ("cir-log-" + hashlib.sha256(str(log).encode()).hexdigest()[:16] + ".txt")
    try:
        with local.open("w") as stream:
            process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                stream.write(line)
                if any(x in line for x in ("loss", "Error", "Traceback", "CIR_", "CHECKPOINT_", "VERIFIED", "RESUME", "%|")):
                    print(line.rstrip()[:1800], flush=True)
            code = process.wait()
        require(code == 0, f"CIR subprocess failed ({code}): {log.name}")
    finally:
        if local.exists():
            shutil.copyfile(local, log)


def invocation(command, *arguments):
    return [sys.executable, "-m", "new_interest.experiments.cir_warmup", command, *map(str, arguments)]


def train_stage(project, root, spec):
    original, shared = bind(project)
    folder = root / spec["key"]
    config = stage_config(project, root, spec)
    config_file = folder / "launch_config.json"
    write(config_file, config)
    if not (folder / "stage_accepted.json").exists():
        run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(spec["world_size"]),
             "-m", "new_interest.experiments.cir_warmup", "train", "--config", str(config_file)],
            project, root / "logs" / f"train-{spec['key']}.log", "0" if spec["world_size"] == 1 else None)
        source = Path(config["checkpoint_dir"]) / "trainer_state.json"
        require(source.exists(), "Missing final native Trainer state")
        shutil.copyfile(source, folder / "trainer_state.json")
        run(invocation("verify", "--", "--model-name", config["model_name"], "--model-revision", config["model_revision"],
                       "--adapter-path", folder / "final_adapter", "--mapping-file", project / shared.MAPPING,
                       "--reference-file", folder / "semantic_reload_reference.json", "--output", folder / "semantic_adapter_reload_audit.json",
                       "--atol", "0", "--rtol", "0"), project, root / "logs" / f"reload-{spec['key']}.log", "0")
    stage = original.audit_stage(project, root, spec)
    stage["adapter"] = shared.audit_adapter(project, "2b", folder)
    stage["reload"] = shared.audit_reload(project, "2b", folder)
    write(folder / "stage_accepted.json", stage)
    print("CIR_STAGE_ACCEPTED", spec["key"], json.dumps(stage), flush=True)
    return stage


def evaluate(project, root, adapter_stage, *, gates):
    from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
    _, shared = bind(project)
    def one(index):
        command = invocation("infer", "--", "--config", project / "configs/id_token/mind_cir_v1.json",
            "--adapter-path", root / adapter_stage / "final_adapter", "--output-dir", root / f"real_eval/shard-{index:02d}",
            "--num-shards", "4", "--shard-index", index, "--batch-size", "2")
        if gates:
            command.extend(["--max-examples", "64"])
        run(command, project, root / "logs" / f"eval-{index}.log", str(index))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in list(executor.map(one, range(4))):
            pass
    merge_id_token_shards([root / f"real_eval/shard-{i:02d}" for i in range(4)],
        project / shared.DEV, project / shared.CANDIDATE_MAPPING, root / "real_eval/merged", expected_records=64 if gates else 31392)
    # Independent reader recomputes every metric and verifies all shard identities.
    result = shared.audit_evaluation(project, "2b", root, "smoke" if gates else "full", logical_shards=4)
    from new_interest.evaluation.evaluate_beam import sha256_directory
    require(read(root / "real_eval/merged/metrics.json")["adapter_sha256"] == sha256_directory(root / adapter_stage / "final_adapter"),
            "Evaluation used another adapter")
    write(root / "evaluation_readback.json", result)
    return result


def reuse_verified_gates(project: Path, reference: Path) -> dict:
    """Reuse hash-bound GPU gates only for unchanged training/evaluation runtime."""
    source = project / reference
    acceptance = read(source / "acceptance.json")
    previous = read(source / "source_bundle.json")
    require(acceptance["status"] == "accepted", "Source CIR run is not accepted")
    require(sha256(source / "source_bundle.json") == acceptance["bundle_sha256"], "Source bundle identity differs")
    old_contract = read(source / "source_contract.json")
    require(sha256(source / "source_contract.json") == previous["files"][str(CONTRACT)], "Source contract differs")
    current = read(project / CONTRACT)
    for key in ("model", "models", "recipe", "worker_packages", "kernel_profile", "execution", "data_metadata_sha256"):
        require(current[key] == old_contract[key], f"Gate reuse changes {key}")
    # Only orchestration and its duration contract may differ. Trainer, sampler,
    # loss, data, tokenizer, launch wrapper and evaluators remain byte-identical.
    allowed = {str(CONTRACT), "src/new_interest/experiments/cir_warmup.py"}
    for name, digest in previous["files"].items():
        if name not in allowed:
            require(sha256(project / name) == digest, f"Gate reuse changes runtime: {name}")
    require(sha256(source / "complete.json") == acceptance["complete_json_sha256"], "Source completion differs")
    complete = read(source / "complete.json")
    require(complete["status"] == "accepted" and complete["bundle_sha256"] == acceptance["bundle_sha256"], "Source completion identity")
    require(complete["stages"][1]["epochs"] == 3, "Source runtime lacks accepted three-epoch training")
    stages = []
    for key, steps in (("single_cir", 8), ("single_ni", 8), ("ddp_cir", 2), ("ddp_ni", 2)):
        name = f"gates/{key}/stage_accepted.json"
        require(sha256(source / name) == acceptance["files"][name], "Source gate receipt differs")
        stage = read(source / name)
        require(stage["optimizer_steps"] == steps and stage["reload"]["all_tensors_exact"]
                and stage["reload"]["logits_exact"] and stage["reload"]["tensor_count"] == 373, "Source gate did not pass")
        stages.append(stage)
    name = "gates/evaluation_readback.json"
    require(sha256(source / name) == acceptance["files"][name], "Source evaluation gate differs")
    evaluation = read(source / name)
    require(evaluation["records"] == 64, "Source evaluation scope differs")
    return {"status": "reused_verified", "source_bundle_sha256": acceptance["bundle_sha256"],
            "source_acceptance_sha256": sha256(source / "acceptance.json"),
            "scope": "Unchanged model/data/numerical runtime; CIR epoch horizon is configured separately",
            "stages": stages, "evaluation": evaluation}


def worker(project, root):
    import torch
    project, root = project.resolve(), root.resolve()
    manifest_file = project / "manifests/cir_bundle.json"
    for relative, digest in read(manifest_file)["files"].items():
        require(sha256(project / relative) == digest, f"Frozen bundle changed: {relative}")
    contract = read(project / CONTRACT)
    identity = {"version": contract["version"], "bundle_sha256": sha256(manifest_file)}
    if (root / "identity.json").exists():
        require(read(root / "identity.json") == identity, "Output belongs to another CIR run")
    else:
        write(root / "identity.json", identity)
    versions = {name: importlib.metadata.version(name) for name in contract["worker_packages"]}
    require(versions == contract["worker_packages"], "Worker package versions changed")
    require(torch.cuda.device_count() == 4, "CIR requires four GPUs")
    devices = [{"name": torch.cuda.get_device_properties(i).name, "bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(4)]
    require(all("A100" in x["name"] and x["bytes"] >= 79 * 1024**3 for x in devices), "Unexpected GPU allocation")
    write(root / "environment.json", {"packages": versions, "devices": devices, "python": platform.python_version(), "cuda": torch.version.cuda,
                                    "kernel_profile": "torch-fallback-explicit"})
    original, shared = bind(project)
    snapshot = shared.prepare_snapshot(project, "2b", root / "snapshot_audit.json")
    require(read(root / "snapshot_audit.json")["files"] == read(project / "manifests/h50_snapshot.json")["files"], "Base model snapshot differs")
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    from new_interest.training.sft_data import audit_token_lengths
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    training_config = read(project / "configs/sft/mind_cir_base_v1.json")
    token_count = int(training_config.get("semantic_id_token_count", 108))
    register_semantic_interest_tokens(tokenizer, project / shared.MAPPING, expected_count=token_count)
    audits = {}
    for task in ("cir", "ni"):
        audits[task] = audit_token_lengths(project / DATA / f"{task}.parquet", tokenizer, max_seq_length=2048,
            enable_thinking=False, output_path=root / f"token_audit_{task}.json")
        joint = "joint_alignment" in contract
        valid_completions = (audits[task]["min_assistant_tokens"] >= 3 if joint else
                             audits[task]["min_assistant_tokens"] == audits[task]["max_assistant_tokens"] == 3)
        require(audits[task]["over_max_seq_length"] == 0 and valid_completions,
                "Input token or native completion contract changed")
    if "history_token" in contract or "joint_alignment" in contract:
        from new_interest.training.sft_data import apply_chat_template
        dev_rows = pq.read_table(project / shared.DEV, columns=["prompt"]).to_pylist()
        lengths = [len(tokenizer(apply_chat_template(tokenizer,
            [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True,
            enable_thinking=False), add_special_tokens=False)["input_ids"]) for row in dev_rows]
        require(bool(lengths) and max(lengths) <= 2048, "Dev prompt exceeds context boundary")
        write(root / "dev_token_audit.json", {"records": len(lengths), "max_tokens": max(lengths),
              "over_max_seq_length": sum(n > 2048 for n in lengths)})
        module = ("token_joint_alignment" if "joint_alignment" in contract else "history_token")
        if module == "token_joint_alignment":
            from new_interest.experiments.token_joint_alignment import validate_completion_lengths
            write(root / "joint_length_audit.json", validate_completion_lengths(project, tokenizer))
        gradient_file = root / "gradient_audit.json"
        if not gradient_file.exists():
            run([sys.executable, "-m", f"new_interest.experiments.{module}", "gradient",
                 "--project", str(project), "--output", str(gradient_file)], project, root / "logs/gradient.log", "0")
        gradient = read(gradient_file)
        require(gradient["interest_token_count"] == token_count and gradient["all_token_rows_finite_and_nonzero"],
                "Expanded-vocabulary gradient gate failed")
    gate = root / "gates"
    if contract.get("reuse_verified_gates"):
        reused = reuse_verified_gates(project, Path(contract["reuse_verified_gates"]))
        write(gate / "accepted.json", {**identity, **reused})
        print("CIR_GATES_REUSED", json.dumps(reused), flush=True)
    else:
        make_gate_data(project, gate)
        gate_stages = [train_stage(project, gate, spec) for spec in stage_specs(project, gate, gates=True)]
        gate_eval = evaluate(project, gate, "ddp_ni", gates=True)
        write(gate / "accepted.json", {**identity, "stages": gate_stages, "evaluation": gate_eval})
    require(read(gate / "accepted.json")["evaluation"]["records"] == 64, "Gate publication readback failed")
    print("CIR_GATES_ACCEPTED_STARTING_FORMAL", flush=True)
    formal = root / "formal"
    stages = [train_stage(project, formal, spec) for spec in stage_specs(project, formal, gates=False)]
    result = evaluate(project, formal, "ni", gates=False)
    baseline = read(project / "manifests/h50_baseline.json")
    write(root / "complete.json", {**identity, "status": "accepted", "data": {**read(project / DATA / "statistics.json"), **formal_schedule(project)},
          "stages": stages, "evaluation": result, "baseline_reference": baseline,
          "large_artifacts_policy": "Remote retention only; local compact reports and hashes"})
    print("CIR_FULL_ACCEPTED", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", type=Path, required=True)
    for command in ("verify", "infer"):
        p = sub.add_parser(command)
        p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.project, args.root)
    else:
        reference_kernels()
        if args.command == "train":
            from new_interest.training.train_sft import train
            train(read(args.config))
        else:
            module = "new_interest.training.verify_semantic_id_token_adapter" if args.command == "verify" else "new_interest.evaluation.evaluate_id_token_ranking"
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            sys.argv = [module, *arguments]
            runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
