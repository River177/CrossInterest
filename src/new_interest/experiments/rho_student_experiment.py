"""Frozen three-arm RHO student experiment; all large artifacts stay remote."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.rho_twofold_reference import (
    accept_training, command, publish_training, read, sha, verify_manifest, write,
)
from new_interest.evaluation.evaluate_beam import canonical_sha256


VERSION = "mind-rho-student-v1"
ARMS = ("uniform", "static", "dynamic")


def verify_sources(project):
    project = Path(project)
    manifest = read(project / "source_manifest.json")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(project / relative) != digest:
            raise ValueError("Frozen student source changed: " + name)
    protocol = read(project / "protocol.json")
    if protocol.get("version") != VERSION or protocol.get("arms") != list(ARMS):
        raise ValueError("Student experiment scope differs")
    if protocol.get("training_numeric_policy") != "torch-deterministic-algorithms-v1":
        raise ValueError("Student worker requires the shared deterministic training policy")
    if sha(project / "inputs/reference_acceptance.json") != protocol["reference_acceptance_sha256"]:
        raise ValueError("Reference acceptance differs from the frozen protocol")
    return protocol, manifest


def prepare_model(project, root):
    import torch
    from huggingface_hub import snapshot_download
    protocol, source = verify_sources(project)
    required = {"torch": "2.8.0+cu126", "transformers": "5.14.1", "peft": "0.18.0",
                "accelerate": "1.12.0", "tokenizers": "0.22.2", "safetensors": "0.8.0"}
    actual = {name: importlib.metadata.version(name) for name in required}
    if actual != required or torch.cuda.device_count() != 4:
        raise ValueError("Student worker requires the frozen four A100 environment")
    if any(torch.cuda.get_device_capability(i) != (8, 0) for i in range(4)):
        raise ValueError("Student worker GPU architecture differs")
    write(root / "compact/environment.json", {"packages": actual,
          "gpus": [torch.cuda.get_device_name(i) for i in range(4)]})
    expected = read(project / "source_snapshot.json")
    if (expected["model_name"], expected["model_revision"]) != (protocol["model_name"], protocol["model_revision"]):
        raise ValueError("Pretrained model snapshot identity differs")
    snapshot = Path(snapshot_download(protocol["model_name"], revision=protocol["model_revision"], local_files_only=True))
    model = Path("/scratch/rho-student-model")
    for name, digest in expected["files"].items():
        destination = model / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha(destination) != digest:
            if sha(snapshot / name) != digest:
                raise ValueError("Pretrained snapshot bytes differ: " + name)
            shutil.copyfile(snapshot / name, destination)
        if sha(destination) != digest:
            raise ValueError("Model staging failed: " + name)
    write(root / "compact/snapshot_audit.json", expected)
    return protocol, source, model


def verify_reload(model, adapter, mapping, reference, output):
    """Compare the durable current-model native probes with a fresh process."""
    from new_interest.evaluation.score_rho_reference import kernel_selection_context
    from new_interest.training import verify_semantic_id_token_adapter as verifier
    original = verifier.deterministic_sdpa_math_context
    receipts = []

    @contextmanager
    def stable_math(torch):
        # The verifier enters this after constructing the model and importing FLA.
        with original(torch), kernel_selection_context("stable_fla_v1") as audit:
            yield
            audit.capture("fresh_native_reload")
            receipts.append(audit.receipt())

    verifier.deterministic_sdpa_math_context = stable_math
    try:
        result = verifier.verify_semantic_id_token_adapter(
            str(model), adapter, mapping, reference, output, atol=0, rtol=0)
    finally:
        verifier.deterministic_sdpa_math_context = original
    if len(receipts) != 1:
        raise ValueError("Missing stable fresh native reload evidence")
    result["kernel_selection"] = "stable_fla_v1"
    result["kernel_receipt"] = receipts[0]
    write(output, result)
    return result


def evaluation_identity(project, adapter, split, shard, limit):
    from new_interest.evaluation.score_rho_reference import kernel_selection_identity
    from new_interest.evaluation import score_rho_reference
    if split not in ("validation", "dev") or not 0 <= shard < 4:
        raise ValueError("Unsupported evaluation split or shard")
    files = {str(p.relative_to(adapter)): sha(p) for p in sorted(adapter.rglob("*"))
             if p.is_file() and p.name not in ("README.md", "training_args.bin")}
    source = read(project / "source_manifest.json")
    protocol = read(project / "protocol.json")
    observed = {"src/new_interest/experiments/rho_student_experiment.py": sha(Path(__file__)),
                "src/new_interest/evaluation/score_rho_reference.py": sha(Path(score_rho_reference.__file__))}
    if any(source["files"].get(name) != digest for name, digest in observed.items()):
        raise ValueError("Heldout evaluator differs from its frozen runtime")
    return {"version": VERSION, "role": "heldout_interest_ranking", "split": split,
            "input_sha256": sha(project / f"inputs/{split}.parquet"),
            "protocol_sha256": sha(project / "protocol.json"),
            "runtime_sha256": source["runtime_sha256"], "source_snapshot_sha256": sha(project / "source_snapshot.json"),
            "model_revision": protocol["model_revision"], "evaluation_sources": observed,
            "adapter_files": files, "adapter_sha256": canonical_sha256(files),
            "mapping_sha256": sha(project / "inputs/interest_token_mapping.parquet"),
            **kernel_selection_identity("stable_fla_v1"), "projection": "native-effective-head-FP32",
            "shard": shard, "shards": 4, "limit": limit, "batch_size": 4}


def evaluate(project, root, model_path, adapter, split, shard, limit=None):
    import torch
    from transformers import AutoTokenizer
    from new_interest.evaluation.score_rho_reference import (
        effective_output_weight, kernel_selection_context, load_model, score_batch, restore_kernel_receipt,
    )
    from new_interest.training.semantic_id_tokens import validate_saved_interest_tokens
    from new_interest.training.sft_data import apply_chat_template
    project, root, adapter = Path(project), Path(root), Path(adapter)
    root.mkdir(parents=True, exist_ok=True)
    protocol = read(project / "protocol.json")
    identity = evaluation_identity(project, adapter, split, shard, limit)
    if (root / "identity.json").exists() and read(root / "identity.json") != identity:
        raise ValueError("Evaluation output identity differs")
    write(root / "identity.json", identity)
    rows = pq.read_table(project / f"inputs/{split}.parquet").to_pylist()
    if limit is not None:
        rows = rows[:limit]
    indices = list(range(shard, len(rows), 4))
    tok = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    spec = validate_saved_interest_tokens(tok, project / "inputs/interest_token_mapping.parquet", expected_count=108)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model, loading = load_model({"model_name": str(model_path), "model_revision": protocol["model_revision"],
                                 "adapter_path": str(adapter)}, spec, tok)
    write(root / "loading_audit.json", loading)
    with torch.inference_mode(), kernel_selection_context("stable_fla_v1") as audit:
        restore_kernel_receipt(audit, root / "kernel_selection.json", canonical_sha256(identity))
        weight = effective_output_weight(model).detach().float()
        token_ids = torch.tensor(spec.token_ids, device=weight.device)
        for start in range(0, len(indices), 256):
            selected = indices[start:start + 256]
            output = root / f"part-{start:08d}.parquet"
            marker = output.with_suffix(".json")
            if marker.exists():
                receipt = read(marker)
                if receipt["identity"] != identity or receipt["sha256"] != sha(output) or receipt["indices"] != selected:
                    raise ValueError("Evaluation checkpoint identity/hash differs")
                continue
            predictions = []
            for offset in range(0, len(selected), 4):
                batch_indices = selected[offset:offset + 4]
                prefixes = []
                for i in batch_indices:
                    row = rows[i]
                    text = apply_chat_template(tok, [{"role": "user", "content": row["prompt"]}],
                                               add_generation_prompt=True, enable_thinking=False)
                    encoded = tok(text, add_special_tokens=False)["input_ids"]
                    if encoded != row["input_ids"][:row["prefix_length"]] or len(encoded) > 2048:
                        raise ValueError("Heldout prompt/tokenizer prefix mismatch")
                    prefixes.append({"input_ids": encoded})
                full = score_batch(model, prefixes, tok.pad_token_id, weight)
                scores = full.index_select(1, token_ids).cpu()
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite heldout scores")
                rankings = scores.argsort(dim=-1, descending=True, stable=True)
                for i, score, ranking in zip(batch_indices, scores.tolist(), rankings.tolist()):
                    row = rows[i]
                    predictions.append({"record_index": i, **{k: row[k] for k in (
                        "source_sample_id", "user_id", "impression_id", "input_sha256", "targets", "history_subcategories")},
                        "scores": score, "ranking": ranking})
            local = Path("/scratch") / f"rho-eval-{uuid.uuid4().hex}.parquet"
            pq.write_table(pa.Table.from_pylist(predictions), local, compression="zstd")
            temporary = output.with_suffix(".partial")
            shutil.copyfile(local, temporary)
            if sha(temporary) != sha(local):
                raise ValueError("Evaluation publication hash mismatch")
            os.replace(temporary, output)
            local.unlink()
            audit.capture(f"rows:{start}:{start + len(selected)}")
            write(root / "kernel_selection.json", audit.receipt(canonical_sha256(identity)))
            write(marker, {"identity": identity, "sha256": sha(output), "indices": selected})
            print(f"RHO_STUDENT_EVAL {split} shard={shard} {start + len(selected)}/{len(indices)}", flush=True)
        write(root / "kernel_selection.json", audit.receipt(canonical_sha256(identity)))
    write(root / "complete.json", {"identity": identity, "rows": len(indices),
          "kernel_selection_sha256": sha(root / "kernel_selection.json"),
          "loading_audit_sha256": sha(root / "loading_audit.json")})


def read_evaluation(project, roots, output, adapter, split, limit=None):
    """Independent process verifies every persisted prediction and recomputes metrics."""
    from new_interest.experiments.sid_relation_bias import metrics
    project, output, adapter = Path(project), Path(output), Path(adapter)
    original = pq.read_table(project / f"inputs/{split}.parquet").to_pylist()
    if limit is not None:
        original = original[:limit]
    names = [r["target_subcategory"] for r in sorted(pq.read_table(
        project / "inputs/interest_token_mapping.parquet").to_pylist(), key=lambda r: r["interest_index"])]
    if len(roots) != 4 or len(set(map(str, roots))) != 4:
        raise ValueError("Expected four unique evaluation shards")
    combined, inventories, common = [], [], {}
    for shard, directory in enumerate(map(Path, roots)):
        identity = evaluation_identity(project, adapter, split, shard, limit)
        done = read(directory / "complete.json")
        if done["identity"] != identity or done["kernel_selection_sha256"] != sha(directory / "kernel_selection.json"):
            raise ValueError("Evaluation completion identity/hash mismatch")
        if done["loading_audit_sha256"] != sha(directory / "loading_audit.json"):
            raise ValueError("Evaluation loading receipt changed")
        loading = read(directory / "loading_audit.json")
        if (loading.get("adapter_tensor_reload", {}).get("accepted") is not True
                or loading.get("pretrained_text_loaded") is not True
                or loading.get("missing_text_keys") or loading.get("unexpected_text_keys")):
            raise ValueError("Evaluation model reload did not pass")
        kernels = read(directory / "kernel_selection.json")
        if (kernels["evaluation_identity"] != canonical_sha256(identity)
                or any(kernels[k] != identity[k] for k in ("kernel_selection", "kernel_selection_protocol", "kernel_selector_sha256"))):
            raise ValueError("Evaluation kernel identity mismatch")
        if canonical_sha256(kernels["snapshots"]) != kernels["snapshots_sha256"]:
            raise ValueError("Evaluation kernel snapshot digest mismatch")
        for snapshot in kernels["snapshots"]:
            if not isinstance(snapshot.get("stage"), str) or not isinstance(snapshot.get("kernels"), dict) or not snapshot["kernels"]:
                raise ValueError("Evaluation kernel snapshot has no selected kernels")
            for name, kernel in snapshot["kernels"].items():
                if not isinstance(kernel.get("cache"), dict) or not kernel["cache"] or kernel.get("cache_results") is not False:
                    raise ValueError("Invalid evaluation kernel cache or timing-cache policy")
                for key, config in kernel["cache"].items():
                    if not isinstance(key, str) or not isinstance(config, str) or not config:
                        raise ValueError("Malformed evaluation kernel selection")
                    pair = (name, key)
                    if pair in common and common[pair] != config:
                        raise ValueError("Conflicting evaluation kernel selections")
                    common[pair] = config
        records, files = [], []
        for p in sorted(directory.glob("part-*.parquet")):
            receipt = read(p.with_suffix(".json"))
            if receipt["identity"] != identity or receipt["sha256"] != sha(p):
                raise ValueError("Evaluation shard file changed")
            part = pq.read_table(p).to_pylist()
            if [row["record_index"] for row in part] != receipt["indices"]:
                raise ValueError("Evaluation shard index receipt differs")
            records.extend(part)
            files.append({"file": str(p), "sha256": sha(p)})
        if [r["record_index"] for r in records] != list(range(shard, len(original), 4)) or len(records) != done["rows"]:
            raise ValueError("Evaluation shard coverage differs")
        expected_stages = {f"rows:{i}:{min(i + 256, len(records))}" for i in range(0, len(records), 256)}
        if not expected_stages.issubset({snapshot["stage"] for snapshot in kernels["snapshots"]}):
            raise ValueError("Kernel evidence does not cover all evaluation chunks")
        combined.extend(records)
        inventories.append({"directory": str(directory), "complete_sha256": sha(directory / "complete.json"), "files": files})
    combined.sort(key=lambda row: row["record_index"])
    if [r["record_index"] for r in combined] != list(range(len(original))):
        raise ValueError("Heldout prediction coverage differs")
    for prediction, row in zip(combined, original):
        if any(prediction[k] != row[k] for k in ("source_sample_id", "user_id", "impression_id", "input_sha256", "targets", "history_subcategories")):
            raise ValueError("Heldout prediction input identity mismatch")
        values = prediction["scores"]
        if len(values) != 108 or not all(math.isfinite(x) for x in values):
            raise ValueError("Invalid saved SID scores")
        if sorted(range(108), key=lambda i: (-values[i], i)) != prediction["ranking"]:
            raise ValueError("Stored ranking differs from scores")
    report = metrics(combined, names)
    if split == "validation":
        report["main_107"] = metrics(combined, names, min_support=20)
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(combined), output / "predictions.parquet", compression="zstd")
    write(output / "metrics.json", report)
    acceptance = {"accepted": True, "split": split, "rows": len(combined), "metrics": report,
                  "source_sha256": sha(project / f"inputs/{split}.parquet"),
                  "protocol_sha256": sha(project / "protocol.json"), "limit": limit,
                  "adapter_sha256": evaluation_identity(project, adapter, split, 0, limit)["adapter_sha256"],
                  "prediction_sha256": sha(output / "predictions.parquet"), "metrics_sha256": sha(output / "metrics.json"),
                  "kernel_configs_consistent": True, "kernel_cache_keys_checked": len(common),
                  "reader_sha256": sha(Path(__file__)), "artifacts": inventories,
                  "prediction_file": str(output / "predictions.parquet"), "large_artifact_policy": "remote_only"}
    write(output / "evaluation_acceptance.json", acceptance)
    print("RHO_STUDENT_EVALUATION_ACCEPTED " + json.dumps({"split": split, "rows": len(combined)}), flush=True)
    return acceptance


def run_parallel(project, jobs, logs):
    logs.mkdir(parents=True, exist_ok=True)
    active, handles = [], []
    try:
        for device, args in jobs:
            handle = (logs / f"process-{device}.log").open("w")
            handles.append(handle)
            active.append(subprocess.Popen([sys.executable, *args], cwd=project,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}, stdout=handle, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in active):
            if any(p.poll() not in (None, 0) for p in active):
                raise RuntimeError("Student subprocess failed; inspect phase logs")
            print("RHO_STUDENT_PROCESSES " + json.dumps({"complete": sum(p.poll() == 0 for p in active), "total": len(active)}), flush=True)
            time.sleep(30)
        if any(p.returncode != 0 for p in active):
            raise RuntimeError("Student subprocess failed")
    finally:
        for p in active:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for handle in handles:
            handle.close()


def run_evaluation(project, root, model, adapter, split, limit=None):
    roots = [root / f"shard-{i}" for i in range(4)]
    jobs = []
    for i, directory in enumerate(roots):
        args = ["-m", "new_interest.experiments.rho_student_experiment", "evaluate", "--project", str(project),
                "--root", str(directory), "--model", str(model), "--adapter", str(adapter), "--split", split, "--shard", str(i)]
        if limit is not None:
            args.extend(["--limit", str(limit)])
        jobs.append((i, args))
    run_parallel(project, jobs, root / "logs")
    args = [sys.executable, "-m", "new_interest.experiments.rho_student_experiment", "read-evaluation", "--project", str(project),
            "--root", str(root / "readback"), "--adapter", str(adapter), "--split", split,
            "--score-dirs", *map(str, roots)]
    if limit is not None:
        args.extend(["--limit", str(limit)])
    command(args, project, root / "readback.log")
    return read(root / "readback/evaluation_acceptance.json")


def make_training_config(project, protocol, model, root, local, stage, arm, reference):
    full = stage == "full"
    rule = ({"rows": 21600, "classes": 108, "rows_per_class": 200, "epochs": 3, "world_size": 4,
             "optimizer_steps": 2025} if full else protocol["gate"][stage])
    train = project / "inputs" / ("train.parquet" if full else f"{stage}_train.parquet")
    cfg = dict(protocol["sft"])
    cfg.update(model_name=str(model), model_revision=protocol["model_revision"], train_file=str(train),
               semantic_id_token_mapping_file=str(project / "inputs/interest_token_mapping.parquet"),
               output_dir=str(local), checkpoint_dir=str(local / "checkpoints"),
               persistent_checkpoint_dir=str(root / f"checkpoints-{stage}"), resume_from_checkpoint="auto",
               run_name=f"rho-student-{arm}-{stage}-seed42", epochs=rule["epochs"],
               sample_weight_file=None, rho_arm=arm, rho_protocol_file=str(project / "protocol.json"),
               rho_reference_file=str(reference["file"]), rho_reference_file_sha256=reference["sha256"],
               rho_epoch_weight_dir=str(root / f"weights-{stage}"), rho_expected_classes=rule["classes"],
               rho_expected_rows_per_class=rule["rows_per_class"], rho_score_batch_size=4,
               rho_stable_training=True, rho_require_native_parity=not full,
               rho_training_numeric_policy=protocol["training_numeric_policy"],
               rho_diagnostic_output=str(root / "compact" / f"{stage}_native_diagnostic.json"))
    if not full:
        cfg["save_steps"] = 2
    return cfg, rule


def final_fp32_probes(project, root, model, adapter, train, steps, epochs):
    from new_interest.experiments.rho_student_probe import compare_probes
    protocol = read(project / "protocol.json")
    jobs, outputs = [], []
    for i in range(2):
        output = root / f"replica-{i}.json"
        outputs.append(output)
        jobs.append((i, ["-m", "new_interest.experiments.rho_student_probe", "--model-name", str(model),
            "--model-revision", protocol["model_revision"], "--adapter", str(adapter),
            "--mapping", str(project / "inputs/interest_token_mapping.parquet"), "--train-file", str(train),
            "--step", str(steps), "--epochs", str(epochs), "--output", str(output)]))
    run_parallel(project, jobs, root / "logs")
    result = compare_probes(outputs)
    write(root / "acceptance.json", result)
    return result


def worker(project, root, phase, arm):
    from new_interest.experiments.prepare_rho_student_reference import prepare_reference
    project, root = Path(project), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if phase not in ("gate", "full") or arm not in ARMS or (phase == "gate" and arm != "dynamic"):
        raise ValueError("Unconfigured student phase/arm")
    protocol, source, model = prepare_model(project, root)
    identity = {"version": VERSION, "phase": phase, "arm": arm,
                "runtime_sha256": source["runtime_sha256"], "protocol_sha256": sha(project / "protocol.json"),
                "reference_acceptance_sha256": protocol["reference_acceptance_sha256"],
                "inputs_sha256": canonical_sha256({k: v for k, v in source["files"].items() if k.startswith("inputs/")})}
    if (root / "run_identity.json").exists() and read(root / "run_identity.json") != identity:
        raise ValueError("Student worker output identity differs")
    write(root / "run_identity.json", identity)
    if phase == "full":
        gate = read(project / "gate_acceptance.json")
        for field in ("runtime_sha256", "protocol_sha256", "inputs_sha256", "reference_acceptance_sha256"):
            if gate["identity"][field] != identity[field]:
                raise ValueError("Gate identity differs from formal student: " + field)
        if not gate["passed"] or gate["identity"]["phase"] != "gate":
            raise ValueError("Formal student requires matching accepted gate")
    scratch = Path("/scratch") / f"rho-student-{arm}-{phase}"
    scratch.mkdir(parents=True, exist_ok=True)
    stages = ("single", "ddp") if phase == "gate" else ("full",)
    training = {}
    for stage in stages:
        train = project / "inputs" / ("train.parquet" if stage == "full" else f"{stage}_train.parquet")
        reference = prepare_reference(project / "inputs/reference_acceptance.json",
                                      protocol["reference_acceptance_sha256"], train, root / f"reference-{stage}")
        local, target = scratch / stage, root / stage
        cfg, rule = make_training_config(project, protocol, model, root, local, stage, arm, reference)
        stage_identity = {**identity, "stage": stage, "train_sha256": sha(train), "reference_sha256": reference["sha256"]}
        if not (target / "artifact_manifest.json").exists():
            cfg_path = scratch / f"{stage}-config.json"
            write(cfg_path, cfg)
            env = {**os.environ, "RHO_SYNC_DIR": str(scratch / f"sync-{stage}-{uuid.uuid4().hex}")}
            command([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={rule['world_size']}",
                     "-m", "new_interest.training.rho_student_sft", "--config", str(cfg_path)],
                    project, scratch / f"train-{stage}.log", env)
            command([sys.executable, "-m", "new_interest.experiments.rho_student_experiment", "verify-reload",
                     "--model", str(model), "--adapter", str(local / "final_adapter"),
                     "--mapping", str(project / "inputs/interest_token_mapping.parquet"),
                     "--reference", str(local / "semantic_reload_reference.json"),
                     "--root", str(local / "semantic_adapter_reload_audit.json")],
                    project, scratch / f"reload-{stage}.log")
            accept_training(local, rule["rows"], rule["optimizer_steps"], rule["world_size"], rule["epochs"],
                            require_decrease=stage == "single", expected_train_sha=sha(train))
            publish_training(local, target, stage_identity)
        verify_manifest(target, stage_identity)
        accepted = accept_training(target, rule["rows"], rule["optimizer_steps"], rule["world_size"], rule["epochs"],
                                   require_decrease=stage == "single", expected_train_sha=sha(train))
        student = read(target / "rho_student_summary.json")
        accepted["student_summary"] = student
        accepted["student_summary_sha256"] = sha(target / "rho_student_summary.json")
        probes = final_fp32_probes(project, root / f"probes-{stage}", model, target / "final_adapter", train,
                                   rule["optimizer_steps"], rule["epochs"])
        if probes.get("accepted") is not True:
            raise ValueError("Fresh student batch4 FP32 probes failed")
        accepted["fresh_fp32_probes"] = probes
        training[stage] = accepted
        print("RHO_STUDENT_TRAINING_ACCEPTED " + json.dumps({"arm": arm, "stage": stage,
              "steps": rule["optimizer_steps"], "rows": rule["rows"]}), flush=True)
    write(root / "compact/training_acceptance.json", training)
    stage = stages[-1]
    evaluations = {}
    for split in (("dev",) if phase == "gate" else ("validation", "dev")):
        result = run_evaluation(project, root / f"evaluation-{split}", model, root / stage / "final_adapter", split,
                                limit=64 if phase == "gate" else None)
        expected = 64 if phase == "gate" else protocol["evaluation"][f"{split}_rows"]
        if not result["accepted"] or result["rows"] != expected:
            raise ValueError("Independent heldout readback incomplete")
        evaluations[split] = result
        write(root / f"compact/{split}_acceptance.json", result)
    success = {"passed": True, "identity": identity, "training": training, "evaluations": evaluations,
               "remote_output": str(root), "large_artifact_policy": "remote_only",
               "report_scope": "Three-arm RHO-inspired soft weighting; single-seed evidence, no noise-truth claim"}
    write(root / "compact/worker_success.json", success)
    if phase == "gate":
        write(root / "compact/gate_acceptance.json", success)
    print("RHO_STUDENT_WORKER_SUCCESS " + json.dumps({"phase": phase, "arm": arm, "passed": True}), flush=True)
    return success


def diagnose_native_gradients(project, root):
    """Drive the original failing HF callback, and stop before optimizer update 1."""
    from new_interest.experiments.prepare_rho_student_reference import prepare_reference
    project, root = Path(project), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    protocol, source, model = prepare_model(project, root)
    contract = read(project / "diagnostic_contract.json")
    if (contract.get("version") != "rho-native-gradient-zero-update-diagnostic-v1"
            or contract.get("optimizer_updates") != 0
            or contract.get("training_protocol_sha256") != sha(project / "protocol.json")):
        raise ValueError("Native diagnostic contract differs")
    train = project / "inputs/single_train.parquet"
    if sha(train) != contract["train_file_sha256"]:
        raise ValueError("Diagnostic changed the failed gate's source examples")
    reference = prepare_reference(project / "inputs/reference_acceptance.json",
                                  protocol["reference_acceptance_sha256"], train, root / "reference")
    local = Path("/scratch") / f"rho-native-gradient-diagnostic-{uuid.uuid4().hex}"
    cfg, _ = make_training_config(project, protocol, model, root, local, "single", "dynamic", reference)
    cfg.update(rho_native_diagnostic_only=True, rho_require_native_parity=True,
               resume_from_checkpoint=None, persistent_checkpoint_dir=None,
               rho_diagnostic_output=str(root / "compact/native_gradient_diagnostic.json"))
    config = local / "diagnostic-config.json"
    write(config, cfg)
    write(root / "compact/diagnostic_identity.json", {"source_runtime_sha256": source["runtime_sha256"],
          "contract_sha256": sha(project / "diagnostic_contract.json"), "protocol_sha256": sha(project / "protocol.json"),
          "train_file_sha256": sha(train), "optimizer_updates": 0})
    env = {**os.environ, "RHO_SYNC_DIR": str(local / "sync")}
    command([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "-m",
             "new_interest.training.rho_student_sft", "--config", str(config)], project, local / "diagnostic.log", env)
    result = read(root / "compact/native_gradient_diagnostic.json")
    if result.get("optimizer_updates") != 0:
        raise ValueError("The native diagnostic must not update parameters")
    print("RHO_NATIVE_DIAGNOSTIC_FINISHED " + json.dumps({"optimizer_updates": 0,
          "report": str(root / "compact/native_gradient_diagnostic.json")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("worker", "evaluate", "read-evaluation", "verify-reload", "diagnose-native"))
    parser.add_argument("--project", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("gate", "full"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--split", choices=("validation", "dev"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--score-dirs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.project, args.root, args.phase, args.arm)
    elif args.action == "evaluate":
        evaluate(args.project, args.root, args.model, args.adapter, args.split, args.shard, args.limit)
    elif args.action == "read-evaluation":
        read_evaluation(args.project, args.score_dirs, args.root, args.adapter, args.split, args.limit)
    elif args.action == "verify-reload":
        verify_reload(args.model, args.adapter, args.mapping, args.reference, args.root)
    else:
        diagnose_native_gradients(args.project, args.root)


if __name__ == "__main__":
    main()
