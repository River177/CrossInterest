"""Rescore the two immutable reference checkpoints with one pinned kernel mode.

This is an inference-only worker. It never constructs an optimizer, resumes
Trainer, or claims that a new forward is an observation of the historical
training process. All large files and verification stay on remote storage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import pyarrow.parquet as pq

from new_interest.experiments.rho_reference_recovery import _overlap
from new_interest.training.checkpoint_persistence import CHECKPOINT_MARKER, validate_persistent_checkpoint


VERSION = "rho-twofold-stable-rescore-v1"
INITIAL_STATE = "b2a8ade503f989994e444cd9c9d7c6e96ad98a51b540d08da4e7c62118ecbe4a"
LIMITATION = ("Original fold1 training-to-fresh native forward comparison failed; original raw probe vectors "
              "were not persisted. New independent fresh probes establish only the pinned checkpoint's "
              "current inference reproducibility, not equivalence to that lost historical forward.")


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def persist_file(source, destination, expected_sha=None):
    """Publish exact known bytes; existing committed artifacts are immutable."""
    source, destination = Path(source), Path(destination)
    digest = sha(source)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Source bytes differ from bound digest: " + str(source))
    if destination.exists():
        if sha(destination) != digest:
            raise ValueError("Refusing to replace different persisted bytes: " + str(destination))
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        if sha(temporary) != digest:
            raise ValueError("Published byte stream differs")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def verify_parent(contract, fold, train_file, output_root):
    """Read and authenticate all parent files, including original optimizer state."""
    import torch
    if (contract.get("version") != VERSION or contract.get("kernel_selection") != "stable_fla_v1"
            or contract.get("model_updates") != 0 or contract.get("folds") != 2
            or contract.get("expected_initial_state_sha256") != INITIAL_STATE):
        raise ValueError("Unsupported immutable-checkpoint rescore contract")
    if fold not in (0, 1) or set(contract["parents"]) != {"0", "1"}:
        raise ValueError("Expected exactly two bound parent checkpoints")
    if len({binding["checkpoint_identity"]["model_revision"] for binding in contract["parents"].values()}) != 1:
        raise ValueError("The two parent model revisions differ")
    if not contract.get("historical_reload_issue"):
        raise ValueError("Historical reload failure must remain in the rescore contract")
    for binding in contract["parents"].values():
        if _overlap(Path(output_root), Path(binding["checkpoint_path"]).parent):
            raise ValueError("Rescore outputs must not overlap either immutable parent")
    binding = contract["parents"][str(fold)]
    parent = Path(binding["checkpoint_path"])
    if not parent.is_absolute() or parent.name != "checkpoint-1014":
        raise ValueError("Parent must be an absolute final checkpoint-1014")
    if sha(parent / CHECKPOINT_MARKER) != binding["checkpoint_marker_sha256"]:
        raise ValueError("Parent marker differs from frozen rescore contract")
    identity = binding["checkpoint_identity"]
    required_identity = {"world_size": 4, "epochs": 3.0, "seed": 42,
        "effective_global_batch_size": 32, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8, "input_adapter_sha256": None,
        "sample_weight_file_sha256": None, "loss_reduction": "token_mean",
        "semantic_id_loss_scope": "full_vocabulary", "enable_thinking": False,
        "run_name": f"rho-reference-fold{fold}-full-seed42", "train_file_sha256": sha(train_file)}
    if any(identity.get(key) != value for key, value in required_identity.items()):
        raise ValueError("Parent identity differs from original unweighted cross-fitted training")
    rows = pq.read_table(train_file, columns=["user_id", "target_subcategory"]).to_pylist()
    if (len(rows) != 10800 or len({row["user_id"] for row in rows}) != 10800
            or len(Counter(row["target_subcategory"] for row in rows)) != 108
            or set(Counter(row["target_subcategory"] for row in rows).values()) != {100}):
        raise ValueError("Parent training fold is not 108 classes by 100 unique users")
    marker = validate_persistent_checkpoint(parent, identity)
    names = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
             "trainer_state.json", "training_args.bin", *(f"rng_state_{rank}.pth" for rank in range(4))}
    if names.difference(marker["files"]) or marker["global_step"] != 1014:
        raise ValueError("Parent manifest omits final four-rank checkpoint evidence")
    if marker["files"]["adapter_model.safetensors"]["sha256"] != binding["adapter_sha256"]:
        raise ValueError("Parent adapter differs from contracted completed weights")
    state = read(parent / "trainer_state.json")
    if state.get("global_step") != 1014 or state.get("epoch") != 3.0:
        raise ValueError("Parent Trainer endpoint differs")
    losses = [{"step": int(row["step"]), "loss": float(row["loss"])}
              for row in state["log_history"] if "step" in row and "loss" in row]
    if [row["step"] for row in losses] != list(range(1, 1015)) or not all(math.isfinite(row["loss"]) for row in losses):
        raise ValueError("Original 1014-step loss history is incomplete")
    optimizer = torch.load(parent / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(parent / "scheduler.pt", map_location="cpu", weights_only=True)
    entries = optimizer.get("state", {})
    if len(entries) != 373 or any(float(entry["step"]) != 1014 for entry in entries.values()):
        raise ValueError("Original saved optimizer endpoint differs from 1014")
    if scheduler.get("last_epoch") != 1014:
        raise ValueError("Original saved scheduler endpoint differs from 1014")
    return {"parent": str(parent), "marker": marker, "identity": identity, "loss_history": losses,
            "optimizer_state_entries": len(entries), "optimizer_step_min": 1014,
            "optimizer_step_max": 1014, "scheduler_last_epoch": 1014,
            "parent_marker_sha256": binding["checkpoint_marker_sha256"],
            "adapter_sha256": binding["adapter_sha256"], "parent_files_mutated": False}


def build_lineage_summary(contract, fold, verified, train_file):
    """Describe original training provenance; do not invent a new training run."""
    identity = verified["identity"]
    return {"artifact_role": "checkpoint_lineage_reconstructed_for_inference_only",
        "model_name": "Qwen/Qwen3.5-2B", "model_revision": identity["model_revision"],
        "input_adapter": None, "input_adapter_field_scope": "original training initialization, not current inference loading",
        "sample_weight_file": None, "train_file": str(train_file), "train_file_sha256": sha(train_file),
        "train_samples": 10800, "configured_epochs": 3.0, "world_size": 4, "seed": 42,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "expected_optimizer_steps": 1014, "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary",
        "loss_history": verified["loss_history"], "checkpoint_identity": identity,
        "parent_checkpoint": verified["parent"], "parent_marker_sha256": verified["parent_marker_sha256"],
        "original_training_runtime_sha256": contract["original_training_runtime_sha256"],
        "original_observed_initial_state_sha256": contract["expected_initial_state_sha256"],
        "initial_state_evidence": "frozen original native runtime log observation, supplied in rescore contract",
        "new_training_updates": 0, "training_repeated": False,
        "original_training_forward_reference_reconstructed": False,
        "historical_reload_issue": contract["historical_reload_issue"], "historical_parity_limitation": LIMITATION,
        "fold": fold}


def verify_teacher(directory, expected_identity):
    directory = Path(directory)
    manifest = read(directory / "artifact_manifest.json")
    if manifest.get("identity") != expected_identity or not manifest.get("files"):
        raise ValueError("Persisted scoring teacher identity differs")
    required = {"run_summary.json", "final_adapter/adapter_model.safetensors", "final_adapter/adapter_config.json",
                "final_adapter/tokenizer.json", "final_adapter/tokenizer_config.json",
                "final_adapter/interest_token_mapping.parquet"}
    if required.difference(manifest["files"]):
        raise ValueError("Persisted scoring teacher omits authenticated required files")
    for name, item in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Unsafe scoring teacher manifest path")
        path = directory / relative
        if not path.is_file() or path.is_symlink() or sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("Persisted scoring teacher bytes differ: " + name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"artifact_manifest.json"}:
        raise ValueError("Unauthenticated extra files in scoring teacher")
    return manifest


def stage_teacher(project, model_path, root, fold, contract, verified, identity, train_file):
    from safetensors.torch import load_file
    import torch
    from transformers import AutoTokenizer
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    directory = root / "teacher"
    if (directory / "artifact_manifest.json").exists():
        return verify_teacher(directory, identity)
    parent = Path(verified["parent"])
    staging = Path("/scratch") / f"rho-rescore-teacher-{fold}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=True)
    adapter = staging / "final_adapter"
    adapter.mkdir(exist_ok=True)
    persist_file(parent / "adapter_model.safetensors", adapter / "adapter_model.safetensors", verified["adapter_sha256"])
    persist_file(parent / "adapter_config.json", adapter / "adapter_config.json",
                 verified["marker"]["files"]["adapter_config.json"]["sha256"])
    state = load_file(str(adapter / "adapter_model.safetensors"))
    sid = [value for key, value in state.items() if "trainable_tokens_delta" in key]
    if len(state) != 373 or len(sid) != 1 or list(sid[0].shape) != [108, 2048]:
        raise ValueError("Parent is not the compact 373-tensor Qwen3.5-2B SID adapter")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Nonfinite parent adapter tensor")
    mapping = root / "inputs/interest_token_mapping.parquet"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    register_semantic_interest_tokens(tokenizer, mapping, expected_count=108)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if (parent / "tokenizer.json").exists():
        original = AutoTokenizer.from_pretrained(parent, local_files_only=True, trust_remote_code=False)
        if original.pad_token_id is None:
            original.pad_token = original.eos_token
        if (tokenizer.get_vocab(), tokenizer.chat_template, tokenizer.pad_token_id) != (
                original.get_vocab(), original.chat_template, original.pad_token_id):
            raise ValueError("Reconstructed original-base SID tokenizer differs from saved checkpoint")
        tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                           "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model"}
        for name in sorted(tokenizer_names.intersection(verified["marker"]["files"])):
            persist_file(parent / name, adapter / name, verified["marker"]["files"][name]["sha256"])
    else:
        tokenizer.save_pretrained(adapter)
    persist_file(mapping, adapter / "interest_token_mapping.parquet")
    write(staging / "run_summary.json", build_lineage_summary(contract, fold, verified, train_file))
    files = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            name = path.relative_to(staging).as_posix()
            persist_file(path, directory / name)
            files[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write(directory / "artifact_manifest.json", {"identity": identity, "files": files,
          "original_training_forward_reference_reconstructed": False})
    return verify_teacher(directory, identity)


def run_probe(config):
    """One process, one independently loaded GPU model, matched native/FP32 inputs."""
    import torch
    from transformers import AutoTokenizer
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from new_interest.evaluation import score_rho_reference as scoring
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    from new_interest.training.verify_semantic_id_token_adapter import compare_adapter_state_dicts
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], local_files_only=True, trust_remote_code=False)
    spec = register_semantic_interest_tokens(tokenizer, config["interest_token_mapping_file"], expected_count=108)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    records = pq.read_table(config["probe_file"]).slice(0, 4).to_pylist()
    if len(records) != 4:
        raise ValueError("Expected four fixed native/FP32 probe inputs")
    model, loading = scoring.load_model(config, spec, tokenizer)
    saved = load_file(str(Path(config["adapter_path"]) / "adapter_model.safetensors"))
    before = compare_adapter_state_dicts(saved, get_peft_model_state_dict(model, save_embedding_layers=False))
    if not before["all_tensors_exact"]:
        raise ValueError("Probe load is not tensor/dtype exact to parent checkpoint")
    rows, prefixes = [], []
    with torch.inference_mode(), scoring.kernel_selection_context("stable_fla_v1") as kernels:
        weight = scoring.effective_output_weight(model).detach().float()
        for record in records:
            prefix = scoring.encode_prefix(tokenizer, record, spec, max_seq_length=2048, enable_thinking=False)
            prefixes.append(prefix)
            ids = torch.tensor([prefix["input_ids"]], device=model.device)
            native = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=1).logits[:, -1].float()
            fp32 = scoring.score_batch(model, [prefix], tokenizer.pad_token_id, weight)
            sid_ids = torch.tensor(spec.token_ids, device=native.device)
            native_sid = native.index_select(1, sid_ids)[0].tolist()
            values = scoring.loss_rows(fp32, [prefix["target_interest_index"]], spec.token_ids)[0]
            rows.append({"source_sample_id": record["source_sample_id"],
                "source_identity_sha256": scoring.source_identity(record), "prefix_ids_sha256": canonical(prefix["input_ids"]),
                "native_sid_logits": native_sid, "native_full_logsumexp": float(native.logsumexp(1)[0]),
                "fp32": values})
            write(config["probe_output"], {"completed": False, "probe_identity": config["probe_identity"],
                  "rows": rows, "model_updates": 0, "historical_training_parity_claimed": False})
            rows[-1]["native_head_audit"] = scoring.audit_native_head(model, prefix, weight)
            kernels.capture(str(record["source_sample_id"]))
        kernels.capture("native_head_audit")
        grouped = scoring.score_batch(model, prefixes, tokenizer.pad_token_id, weight)
        batch4 = scoring.loss_rows(grouped, [prefix["target_interest_index"] for prefix in prefixes], spec.token_ids)
        write(config["probe_output"], {"completed": False, "probe_identity": config["probe_identity"],
              "rows": rows, "batch4_fp32": batch4, "model_updates": 0, "historical_training_parity_claimed": False})
        kernels.capture("checkpoint:0:4")
        kernel_receipt = kernels.receipt(evaluation_identity=canonical(config["probe_identity"]))
    after = compare_adapter_state_dicts(saved, get_peft_model_state_dict(model, save_embedding_layers=False))
    kernel_directory = Path(config["probe_output"]).with_suffix("")
    write(kernel_directory / "kernel_selection.json", kernel_receipt)
    kernel_manifest = {**scoring.kernel_selection_identity("stable_fla_v1"),
        "evaluation_identity": canonical(config["probe_identity"]),
        "kernel_selection_sha256": sha(kernel_directory / "kernel_selection.json"),
        "checkpoint_size": 4, "shard_record_count": 4}
    report = {"version": VERSION, "completed": True, "probe_identity": config["probe_identity"],
              "rows": rows, "batch4_fp32": batch4, "loading": loading, "adapter_before": before, "adapter_after": after,
              "kernel_receipt": kernel_receipt, "kernel_directory": str(kernel_directory),
              "kernel_manifest": kernel_manifest, "model_updates": 0,
              "historical_training_parity_claimed": False}
    # Preserve observed vectors even if subsequent equality/immutability assertions fail.
    write(config["probe_output"], report)
    if not after["all_tensors_exact"]:
        raise ValueError("Probe changed parent adapter tensors")
    return report


def compare_probes(first, second):
    from new_interest.experiments.read_rho_reference import verify_kernel_selection
    if first.get("completed") is not True or second.get("completed") is not True:
        raise ValueError("Independent probe process did not complete all observations")
    if first["probe_identity"] != second["probe_identity"] or first["model_updates"] != 0 or second["model_updates"] != 0:
        raise ValueError("Fresh probe identities or update counts differ")
    if len(first["rows"]) != 4 or len(second["rows"]) != 4:
        raise ValueError("Incomplete independent fresh probe coverage")
    common_configs = {}
    for report in (first, second):
        if not report["adapter_before"]["all_tensors_exact"] or not report["adapter_after"]["all_tensors_exact"]:
            raise ValueError("Fresh probe adapter immutability failed")
        if report["kernel_receipt"]["kernel_selection"] != "stable_fla_v1":
            raise ValueError("Fresh probe kernel mode differs")
        if report["kernel_manifest"]["evaluation_identity"] != canonical(report["probe_identity"]):
            raise ValueError("Probe kernel evidence belongs to a different probe identity")
        if read(Path(report["kernel_directory"]) / "kernel_selection.json") != report["kernel_receipt"]:
            raise ValueError("Probe kernel receipt differs from persisted evidence")
        verify_kernel_selection(report["kernel_directory"], report["kernel_manifest"], common_configs)
    if len(first["batch4_fp32"]) != 4 or first["batch4_fp32"] != second["batch4_fp32"]:
        raise ValueError("Independent fresh production-batch4 FP32 probe mismatch")
    for left, right in zip(first["rows"], second["rows"]):
        for key in ("source_sample_id", "source_identity_sha256", "prefix_ids_sha256", "native_sid_logits",
                    "native_full_logsumexp", "fp32"):
            if left[key] != right[key]:
                raise ValueError("Independent fresh native/FP32 probe mismatch: " + key)
        if left["native_head_audit"]["native_head_max_abs_error"] != 0 or right["native_head_audit"]["native_head_max_abs_error"] != 0:
            raise ValueError("Probe native effective-head audit failed")
    return {"accepted": True, "native_and_fp32_exact": True, "production_batch4_fp32_exact": True,
            "independent_processes": 2, "probe_rows": 4, "overlapping_kernel_cache_keys_consistent": True,
            "source_sample_ids": [row["source_sample_id"] for row in first["rows"]],
            "model_updates": 0, "historical_training_parity_claimed": False, "limitation": LIMITATION}


def processes(project, jobs, log_root):
    """Run independent GPU processes, retaining remote logs and propagating failures."""
    log_root.mkdir(parents=True, exist_ok=True)
    active, streams = [], []
    try:
        for index, (device, arguments) in enumerate(jobs):
            stream = (log_root / f"process-{index}.log").open("w")
            streams.append(stream)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device), "PYTHONUNBUFFERED": "1"}
            active.append(subprocess.Popen([sys.executable, *arguments], cwd=project, env=env,
                                            stdout=stream, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in active):
            if any(process.poll() not in (None, 0) for process in active):
                raise RuntimeError("Rescore subprocess failed; inspect persisted process logs")
            print("RHO_RESCORE_PROCESSES_ACTIVE " + json.dumps({"total": len(active),
                  "complete": sum(process.poll() == 0 for process in active)}), flush=True)
            time.sleep(30)
        if any(process.returncode != 0 for process in active):
            raise RuntimeError("Rescore subprocess failed; inspect persisted process logs")
    finally:
        for process in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()


def worker(project, root, fold):
    import torch
    from huggingface_hub import snapshot_download
    from new_interest.evaluation.score_rho_reference import kernel_selection_identity
    project, root = Path(project), Path(root)
    contract, protocol = read(project / "rescore_contract.json"), read(project / "protocol.json")
    if any(parent["checkpoint_identity"]["model_revision"] != protocol["model_revision"]
           for parent in contract["parents"].values()):
        raise ValueError("Both parents must match the frozen model revision")
    source = read(project / "source_manifest.json")
    if "rescore_contract.json" not in source["files"]:
        raise ValueError("Rescore contract is not in the frozen source inventory")
    for name, digest in source["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(project / relative) != digest:
            raise ValueError("Frozen rescore source differs: " + name)
    for parent in contract["parents"].values():
        if _overlap(root, Path(parent["checkpoint_path"]).parent):
            raise ValueError("Refusing to write into original parent checkpoint storage")
    required = {"torch": "2.8.0+cu126", "transformers": "5.14.1", "peft": "0.18.0",
                "accelerate": "1.12.0", "tokenizers": "0.22.2", "safetensors": "0.8.0"}
    packages = {name: importlib.metadata.version(name) for name in required}
    if packages != required or torch.cuda.device_count() != 4:
        raise ValueError("Rescore requires the frozen four-GPU inference environment")
    identity = {"version": VERSION, "fold": fold, "contract_sha256": sha(project / "rescore_contract.json"),
                "protocol_sha256": sha(project / "protocol.json"), "runtime_sha256": source["runtime_sha256"],
                "kernel": kernel_selection_identity("stable_fla_v1")}
    if (root / "run_identity.json").exists() and read(root / "run_identity.json") != identity:
        raise ValueError("Rescore output identity differs; use a new output directory")
    write(root / "run_identity.json", identity)
    write(root / "compact/environment.json", {"packages": packages, "gpus": [torch.cuda.get_device_name(i) for i in range(4)]})
    for name in ("fold0.parquet", "fold1.parquet", "interest_token_mapping.parquet"):
        persist_file(project / "inputs" / name, root / "inputs" / name, source["files"]["inputs/" + name])
    train, opposite = root / "inputs" / f"fold{fold}.parquet", root / "inputs" / f"fold{1-fold}.parquet"
    mapping = root / "inputs/interest_token_mapping.parquet"
    verified = verify_parent(contract, fold, train, root)
    if verified["identity"]["model_revision"] != protocol["model_revision"]:
        raise ValueError("Parent and frozen model revision differ")
    snapshot_binding = read(project / "source_snapshot.json")
    if (snapshot_binding["model_name"], snapshot_binding["model_revision"]) != (protocol["model_name"], protocol["model_revision"]):
        raise ValueError("Frozen pretrained snapshot identity differs")
    snapshot = Path(snapshot_download(protocol["model_name"], revision=protocol["model_revision"], local_files_only=True))
    model = Path("/scratch/rho-reference-rescore-model")
    for name, digest in snapshot_binding["files"].items():
        persist_file(snapshot / name, model / name, digest)
    teacher_manifest = stage_teacher(project, model, root, fold, contract, verified, identity, train)
    parent_receipt = {key: value for key, value in verified.items() if key not in ("loss_history", "marker")}
    write(root / "compact/checkpoint_verification.json", {**parent_receipt,
          "parent_file_count": len(verified["marker"]["files"]), "all_parent_file_hashes_verified": True,
          "new_training_updates": 0, "lineage_summary_sha256": sha(root / "teacher/run_summary.json"),
          "historical_reload_issue": contract["historical_reload_issue"], "limitation": LIMITATION})
    base_config = {"model_name": str(model), "model_revision": protocol["model_revision"],
        "adapter_path": str(root / "teacher/final_adapter"), "teacher_run_summary": str(root / "teacher/run_summary.json"),
        "teacher_train_file": str(train), "score_file": str(opposite), "interest_token_mapping_file": str(mapping),
        "batch_size": 4, "checkpoint_size": 256, "num_shards": 4, "max_seq_length": 2048,
        "attn_implementation": "sdpa", "enable_thinking": False, "kernel_selection": "stable_fla_v1"}
    probe_identity = {**identity, "parent_adapter_sha256": verified["adapter_sha256"], "probe_file_sha256": sha(train)}
    probe_jobs = []
    for replica in range(2):
        path = root / "configs" / f"probe-{replica}.json"
        write(path, {**base_config, "probe_identity": probe_identity, "probe_file": str(train),
                     "probe_output": str(root / "probes" / f"replica-{replica}.json")})
        probe_jobs.append((replica, ["-m", "new_interest.experiments.rho_reference_rescore", "--probe-config", str(path)]))
    processes(project, probe_jobs, root / "logs/probes")
    first, second = [read(root / "probes" / f"replica-{i}.json") for i in range(2)]
    probe_report = compare_probes(first, second)
    write(root / "compact/fresh_probe_acceptance.json", {**probe_report,
        "probe_artifacts": [{"path": str(root / "probes" / f"replica-{i}.json"),
                            "sha256": sha(root / "probes" / f"replica-{i}.json")} for i in range(2)]})
    score_jobs, directories = [], []
    for shard in range(4):
        directory = root / "oof_stable_fla_v1" / f"shard-{shard}"
        directories.append(directory)
        path = root / "configs" / f"score-{shard}.json"
        write(path, {**base_config, "output_dir": str(directory), "shard_index": shard})
        score_jobs.append((shard, ["-m", "new_interest.evaluation.score_rho_reference", "--config", str(path)]))
    processes(project, score_jobs, root / "logs/scoring")
    acceptance = root / "compact/reference_acceptance.json"
    processes(project, [(0, ["-m", "new_interest.experiments.read_rho_reference", "--score-dirs", *map(str, directories),
        "--teacher-train-file", str(train), "--score-file", str(opposite), "--mapping-file", str(mapping),
        "--adapter-path", str(root / "teacher/final_adapter"), "--teacher-run-summary", str(root / "teacher/run_summary.json"),
        "--output", str(acceptance)])], root / "logs/readback")
    result = read(acceptance)
    if (not result["accepted"] or result["metrics"]["rows"] != 10800 or result["teacher_score_user_overlap"] != 0
            or any(result.get(key) != value for key, value in identity["kernel"].items())):
        raise ValueError("Independent remote OOF readback did not accept the complete opposite fold")
    verify_teacher(root / "teacher", identity)
    if sha(Path(verified["parent"]) / CHECKPOINT_MARKER) != verified["parent_marker_sha256"]:
        raise ValueError("Immutable parent marker changed during inference")
    success = {"passed": True, "version": VERSION, "identity": identity, "fold": fold,
        "kernel_selection": "stable_fla_v1", "scored_rows": 10800, "new_training_updates": 0,
        "parent_checkpoint": verified["parent"], "parent_marker_sha256": verified["parent_marker_sha256"],
        "parent_adapter_sha256": verified["adapter_sha256"], "parent_files_mutated": False,
        "original_training_runtime_sha256": contract["original_training_runtime_sha256"],
        "original_observed_initial_state_sha256": contract["expected_initial_state_sha256"],
        "historical_training_reload_parity_claimed": False, "historical_reload_issue": contract["historical_reload_issue"],
        "historical_parity_limitation": LIMITATION, "reference_acceptance_sha256": sha(acceptance),
        "teacher_manifest_sha256": sha(root / "teacher/artifact_manifest.json"),
        "fresh_probe_acceptance_sha256": sha(root / "compact/fresh_probe_acceptance.json"),
        "restoration": {"teacher": str(root / "teacher"), "train_file": str(train), "score_file": str(opposite),
                        "mapping_file": str(mapping), "score_dirs": list(map(str, directories))},
        "large_artifact_policy": "remote_only", "student_training_started": False}
    write(root / "compact/worker_success.json", success)
    print("RHO_RESCORE_WORKER_SUCCESS " + json.dumps(success), flush=True)
    return success


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--probe-config", type=Path)
    args = parser.parse_args()
    if args.probe_config is not None:
        run_probe(read(args.probe_config))
    else:
        if args.project is None or args.root is None or args.fold is None:
            parser.error("Require --project, --root and --fold")
        try:
            worker(args.project, args.root, args.fold)
        except Exception as error:
            # Parent-path safety is checked again before emitting any failure file.
            contract = read(args.project / "rescore_contract.json")
            if not any(_overlap(args.root, Path(p["checkpoint_path"]).parent) for p in contract["parents"].values()):
                write(args.root / "compact/worker_failure.json", {"accepted": False,
                      "error": f"{type(error).__name__}: {error}", "new_training_updates": 0,
                      "historical_reload_issue": contract["historical_reload_issue"]})
            raise


if __name__ == "__main__":
    main()
