"""Fresh checkpoint probes of the actual four-example student scoring batch.

The worker launches two separate processes. This module only loads a checkpoint,
scores four source-bound examples, and compares their FP32 scoring receipts. It
does not repeat the worker's training-held versus fresh native reload verifier.
All raw probe rows and model files stay on the remote worker filesystem.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket

import pyarrow.parquet as pq

from new_interest.evaluation import score_rho_reference as reference
from new_interest.evaluation import score_rho_student as student
from new_interest.evaluation.evaluate_beam import _atomic_write_json, canonical_sha256, sha256_file
from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens, validate_saved_interest_tokens

VERSION = "rho-student-fresh-checkpoint-batch4-fp32-v1"
FP32_FIELDS = ("full_vocab_first_sid_nll", "target_logit_fp32", "full_vocab_logsumexp_fp32",
               "sid_conditional_nll", "sid_logsumexp_fp32")


def adapter_inventory(adapter):
    adapter = Path(adapter)
    if not (adapter/"adapter_model.safetensors").is_file() or not (adapter/"adapter_config.json").is_file():
        raise ValueError("A complete saved student adapter is required")
    files = {str(path.relative_to(adapter)): sha256_file(path) for path in sorted(adapter.rglob("*"))
             if path.is_file() and path.name not in {"README.md", "training_args.bin"}}
    return files


def run_probe(model_name, model_revision, adapter, mapping, train_file, step, epochs, output):
    from transformers import AutoTokenizer

    if type(step) is not int or step < 0 or not math.isfinite(float(epochs)) or float(epochs) < 0:
        raise ValueError("Probe requires a valid explicit training endpoint")
    adapter, mapping, train_file, output = map(Path, (adapter, mapping, train_file, output))
    train_sha, mapping_sha = sha256_file(train_file), sha256_file(mapping)
    inventory = adapter_inventory(adapter)
    batches = pq.ParquetFile(train_file).iter_batches(batch_size=4, columns=list(reference.RECORD_FIELDS))
    first = next(batches, None)
    records = [] if first is None else first.to_pylist()
    if len(records) != 4:
        raise ValueError("Actual batch4 probe requires at least four training records")
    saved = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    saved_spec = validate_saved_interest_tokens(saved, mapping, expected_count=108)
    local = Path(model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=None if local else model_revision,
        local_files_only=local, trust_remote_code=False)
    spec = register_semantic_interest_tokens(tokenizer, mapping, expected_count=108)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (tokenizer.get_vocab() != saved.get_vocab() or tokenizer.chat_template != saved.chat_template
            or tokenizer.pad_token_id != saved.pad_token_id or spec.token_ids != saved_spec.token_ids):
        raise ValueError("Original registration and saved student tokenizer differ")
    tokenizer.padding_side = "right"
    identity = {"version": VERSION, "model_name": str(model_name), "model_revision": model_revision,
        "global_step": step, "completed_epochs": float(epochs), "training_file_sha256": train_sha,
        "training_file_rows": pq.read_metadata(train_file).num_rows, "mapping_sha256": mapping_sha,
        "adapter_files": inventory, "adapter_files_sha256": canonical_sha256(inventory),
        "student_helper_sha256": sha256_file(Path(student.__file__)),
        "reference_helper_sha256": sha256_file(Path(reference.__file__)),
        "probe_source_sha256": sha256_file(Path(__file__)), "batch_size": 4, "probe_rows": 4,
        "loss_definition": reference.LOSS_DEFINITION,
        **reference.kernel_selection_identity("stable_fla_v1")}
    identity_sha = canonical_sha256(identity)
    endpoint = {"global_step": step, "completed_epochs": float(epochs), "probe_identity_sha256": identity_sha}
    config = {"model_name": str(model_name), "model_revision": model_revision, "adapter_path": str(adapter),
              "attn_implementation": "sdpa", "device": "cuda:0"}
    model, loading = reference.load_model(config, spec, tokenizer)
    result = student.score_current_student(model, tokenizer, records, spec, student_state=endpoint,
                                          num_shards=1, shard_index=0, batch_size=4, max_seq_length=2048)
    if (adapter_inventory(adapter) != inventory or sha256_file(train_file) != train_sha
            or sha256_file(mapping) != mapping_sha):
        raise ValueError("Probe changed its checkpoint or source artifacts")
    report = {"completed": True, "probe_identity": identity, "probe_identity_sha256": identity_sha,
        "process": {"pid": os.getpid(), "host": socket.gethostname()},
        "probe_inputs": [{"source_sample_id": str(r["source_sample_id"]), "user_id": str(r["user_id"]),
                          "target_subcategory": str(r["target_subcategory"]), "input_sha256": reference.source_identity(r)}
                         for r in records],
        "result": result, "loading_audit": loading, "model_updates": 0, "optimizer_constructed": False,
        "checkpoint_artifacts_unchanged": True,
        "historical_training_native_parity_checked_here": False,
        "restoration": {"adapter": str(adapter), "mapping": str(mapping), "train_file": str(train_file)}}
    report["rows_sha256"] = canonical_sha256(result["rows"])
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output, report)
    return report


def _validate_report(report, common_configs):
    if (report.get("completed") is not True or report.get("model_updates") != 0
            or report.get("optimizer_constructed") is not False
            or report.get("checkpoint_artifacts_unchanged") is not True):
        raise ValueError("Fresh student probe did not complete without updates")
    identity, result = report["probe_identity"], report["result"]
    if (identity.get("version") != VERSION or identity.get("batch_size") != 4 or identity.get("probe_rows") != 4
            or identity.get("loss_definition") != reference.LOSS_DEFINITION
            or canonical_sha256(identity) != report["probe_identity_sha256"]
            or canonical_sha256(identity["adapter_files"]) != identity["adapter_files_sha256"]):
        raise ValueError("Fresh student probe identity/hash differs")
    if (identity["student_helper_sha256"] != sha256_file(Path(student.__file__))
            or identity["reference_helper_sha256"] != sha256_file(Path(reference.__file__))
            or identity["probe_source_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("Fresh student probe helper source differs")
    kernel_identity = reference.kernel_selection_identity("stable_fla_v1")
    if any(identity.get(key) != value for key, value in kernel_identity.items()):
        raise ValueError("Fresh student probe kernel identity differs")
    expected_endpoint = {"global_step": identity["global_step"], "completed_epochs": identity["completed_epochs"],
                         "probe_identity_sha256": report["probe_identity_sha256"]}
    score_identity = result["identity"]
    if (score_identity["role"] != student.ROLE or score_identity["student_state"] != expected_endpoint
            or score_identity["full_record_count"] != 4 or score_identity["num_shards"] != 1
            or score_identity["batch_size"] != 4
            or score_identity["mapping_sha256"] != identity["mapping_sha256"]
            or score_identity["loss_definition"] != identity["loss_definition"]
            or any(score_identity.get(key) != value for key, value in kernel_identity.items())
            or canonical_sha256({k: v for k, v in score_identity.items() if k != "evaluation_identity"}) != score_identity["evaluation_identity"]):
        raise ValueError("Actual batch4 scorer identity differs")
    rows, inputs = result["rows"], report["probe_inputs"]
    if len(rows) != 4 or len(inputs) != 4 or [r["record_index"] for r in rows] != list(range(4)):
        raise ValueError("Incomplete actual batch4 probe coverage")
    if score_identity["ordered_source_sha256"] != canonical_sha256([row["input_sha256"] for row in inputs]):
        raise ValueError("Fresh student first-four source hash differs")
    coverage = result["coverage"]
    if (coverage["shard_index"] != 0 or coverage["num_shards"] != 1 or coverage["scored_rows"] != 4
            or coverage["full_record_count"] != 4 or coverage["exact_modulo_coverage"] is not True
            or coverage["record_indices_sha256"] != canonical_sha256(list(range(4)))):
        raise ValueError("Fresh student scoring coverage receipt differs")
    if len({r["source_sample_id"] for r in rows}) != 4 or len({r["user_id"] for r in rows}) != 4:
        raise ValueError("Duplicate probe source identity")
    if canonical_sha256(rows) != report["rows_sha256"]:
        raise ValueError("Fresh student probe row hash differs")
    for row, expected in zip(rows, inputs):
        if any(row[key] != value for key, value in expected.items()) or row["evaluation_identity"] != score_identity["evaluation_identity"]:
            raise ValueError("Fresh student probe input/score identity differs")
        if any(not math.isfinite(float(row[key])) for key in FP32_FIELDS) or not all(math.isfinite(float(v)) for v in row["sid_scores_fp32"]):
            raise ValueError("Nonfinite fresh student FP32 scores")
    audit = result["state_audit"]
    if (any(audit.get(key) is not True for key in ("parameters_unchanged", "gradients_unchanged", "training_flags_restored", "rng_restored"))
            or audit["model_updates"] != 0 or audit["adapter_state_before"] != audit["adapter_state_after"]
            or audit["adapter_state_before"]["sha256"] != score_identity["adapter_state_sha256"]):
        raise ValueError("Fresh student adapter/gradient state changed")
    loading = report["loading_audit"]
    if (not loading["adapter_tensor_reload"]["accepted"] or not loading["pretrained_text_loaded"]
            or loading["missing_text_keys"] or loading["unexpected_text_keys"]):
        raise ValueError("Fresh student checkpoint tensor load failed")
    receipt = result["kernel_receipt"]
    if (any(receipt.get(key) != value for key, value in kernel_identity.items())
            or receipt["evaluation_identity"] != score_identity["evaluation_identity"]
            or canonical_sha256(receipt["snapshots"]) != receipt["snapshots_sha256"]
            or len(receipt["snapshots"]) != 1 or receipt["snapshots"][0]["stage"] != "student_shard_complete"):
        raise ValueError("Fresh student kernel receipt identity/hash differs")
    kernels = receipt["snapshots"][0]["kernels"]
    if not kernels:
        raise ValueError("Fresh student probe records no controlled kernel selections")
    for name, kernel in kernels.items():
        if not kernel.get("cache") or kernel.get("cache_results") is not False:
            raise ValueError("Fresh student kernel cache evidence is invalid")
        for cache_key, config in kernel["cache"].items():
            key = (name, cache_key)
            if key in common_configs and common_configs[key] != config:
                raise ValueError("Fresh student probes selected different kernels for a shared key")
            common_configs[key] = config


def compare_probes(paths):
    paths = [Path(path) for path in paths]
    if len(paths) != 2 or paths[0].resolve() == paths[1].resolve():
        raise ValueError("Expected two independent fresh-process probe files")
    reports = [json.loads(path.read_text()) for path in paths]
    common = {}
    for report in reports:
        _validate_report(report, common)
    first, second = reports
    process_ids = [(r["process"]["host"], r["process"]["pid"]) for r in reports]
    if process_ids[0] == process_ids[1]:
        raise ValueError("Fresh student probes were not produced by two distinct processes")
    if first["probe_identity"] != second["probe_identity"] or first["result"]["identity"] != second["result"]["identity"]:
        raise ValueError("Independent fresh student probe identities differ")
    if first["probe_inputs"] != second["probe_inputs"]:
        raise ValueError("Independent fresh student probe inputs differ")
    # Canonical JSON comparison also distinguishes -0.0 from +0.0. No score
    # tolerance or rounding is introduced for the serialized FP32 fields.
    if canonical_sha256(first["result"]["rows"]) != canonical_sha256(second["result"]["rows"]):
        raise ValueError("Independent fresh student FP32 probe rows differ exactly")
    selections = [{"kernel": name, "cache_key": key, "config": config} for (name, key), config in sorted(common.items())]
    return {"accepted": True, "version": VERSION, "independent_processes": 2, "probe_rows": 4,
        "production_batch4_fp32_exact": True, "adapter_hash_invariance": True,
        "overlapping_kernel_cache_keys_consistent": True, "model_updates": 0,
        "probe_identity_sha256": first["probe_identity_sha256"],
        "global_step": first["probe_identity"]["global_step"], "completed_epochs": first["probe_identity"]["completed_epochs"],
        "training_file_sha256": first["probe_identity"]["training_file_sha256"],
        "adapter_files_sha256": first["probe_identity"]["adapter_files_sha256"],
        "selected_kernel_configs": selections, "selected_kernel_configs_sha256": canonical_sha256(selections),
        "probe_artifacts": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "historical_training_native_parity_checked_here": False}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for flag in ("model-name", "model-revision", "adapter", "mapping", "train-file", "output"):
        parser.add_argument("--"+flag, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    args = parser.parse_args()
    report = run_probe(**vars(args))
    print(json.dumps({"completed": report["completed"], "probe_identity_sha256": report["probe_identity_sha256"],
                      "probe_rows": 4, "model_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
