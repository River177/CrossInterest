import copy
import json
from pathlib import Path

import pytest

from new_interest.evaluation import score_rho_reference as reference
from new_interest.evaluation import score_rho_student as student
from new_interest.evaluation.evaluate_beam import canonical_sha256, sha256_file
from new_interest.experiments import rho_student_probe as probe


def write(path, report):
    path.write_text(json.dumps(report, sort_keys=True)+"\n")


def report(pid, *, step=675):
    kernel = reference.kernel_selection_identity("stable_fla_v1")
    inventory = {"adapter_model.safetensors": "saved-adapter-hash", "adapter_config.json": "config-hash"}
    identity = {"version": probe.VERSION, "model_name": "frozen/Qwen3.5-2B", "model_revision": "pinned",
        "global_step": step, "completed_epochs": 1., "training_file_sha256": "full-train-hash",
        "training_file_rows": 21600, "mapping_sha256": "mapping", "adapter_files": inventory,
        "adapter_files_sha256": canonical_sha256(inventory), "student_helper_sha256": sha256_file(Path(student.__file__)),
        "reference_helper_sha256": sha256_file(Path(reference.__file__)), "probe_source_sha256": sha256_file(Path(probe.__file__)),
        "batch_size": 4, "probe_rows": 4, "loss_definition": reference.LOSS_DEFINITION, **kernel}
    identity_sha = canonical_sha256(identity)
    inputs = [{"source_sample_id": f"source-{i}", "user_id": f"user-{i}", "target_subcategory": f"class-{i}",
               "input_sha256": canonical_sha256({"input": i})} for i in range(4)]
    score_identity = {"identity_version": student.VERSION, "role": student.ROLE,
        "student_state": {"global_step": step, "completed_epochs": 1., "probe_identity_sha256": identity_sha},
        "adapter_state_sha256": "live-state", "full_record_count": 4, "num_shards": 1, "batch_size": 4,
        "ordered_source_sha256": canonical_sha256([r["input_sha256"] for r in inputs]),
        "mapping_sha256": "mapping", "loss_definition": reference.LOSS_DEFINITION,
        "labels": [f"class-{i}" for i in range(4)], "token_ids": [60, 61, 62, 63], **kernel}
    score_identity["evaluation_identity"] = canonical_sha256(score_identity)
    rows = [{**item, "record_index": i, "evaluation_identity": score_identity["evaluation_identity"],
        "score_role": student.ROLE, "full_vocab_first_sid_nll": 2., "target_logit_fp32": 0.,
        "full_vocab_logsumexp_fp32": 2., "sid_conditional_nll": 1.3862943649291992,
        "sid_logsumexp_fp32": 1.3862943649291992, "sid_scores_fp32": [0., 0., 0., 0.],
        "target_interest_index": i, "target_token_id": 60+i, "target_sid_rank": i+1,
        "ranked_interest_indices": [0, 1, 2, 3], "prefix_tokens": 5+i, "full_vocabulary_size": 64}
        for i, item in enumerate(inputs)]
    snapshots = [{"stage": "student_shard_complete", "kernels": {"fla.kernel": {
        "cache": {"shape4": "BLOCK=32"}, "cache_results": False, "best_config": "BLOCK=32"}}}]
    state = {"sha256": "live-state", "tensor_count": 373}
    result = {"identity": score_identity, "rows": rows,
        "kernel_receipt": {**kernel, "evaluation_identity": score_identity["evaluation_identity"],
                           "snapshots": snapshots, "snapshots_sha256": canonical_sha256(snapshots)},
        "coverage": {"shard_index": 0, "num_shards": 1, "scored_rows": 4, "full_record_count": 4,
                     "record_indices_sha256": canonical_sha256(list(range(4))), "exact_modulo_coverage": True},
        "state_audit": {"adapter_state_before": state, "adapter_state_after": copy.deepcopy(state), "model_updates": 0,
            "parameters_unchanged": True, "gradients_unchanged": True, "training_flags_restored": True, "rng_restored": True}}
    return {"completed": True, "probe_identity": identity, "probe_identity_sha256": identity_sha,
        "process": {"host": "same-node", "pid": pid}, "probe_inputs": inputs, "result": result,
        "loading_audit": {"pretrained_text_loaded": True, "missing_text_keys": [], "unexpected_text_keys": [],
                          "adapter_tensor_reload": {"accepted": True}},
        "model_updates": 0, "optimizer_constructed": False, "checkpoint_artifacts_unchanged": True,
        "rows_sha256": canonical_sha256(rows)}


def pair(tmp_path):
    paths = [tmp_path/"first.json", tmp_path/"second.json"]
    values = [report(100), report(101)]
    for path, value in zip(paths, values):
        write(path, value)
    return paths, values


def test_actual_batch4_compact_probe_acceptance(tmp_path):
    paths, _ = pair(tmp_path)
    accepted = probe.compare_probes(paths)
    assert accepted["accepted"] and accepted["production_batch4_fp32_exact"]
    assert accepted["adapter_hash_invariance"] and accepted["probe_rows"] == 4
    assert accepted["independent_processes"] == 2 and accepted["model_updates"] == 0
    assert accepted["selected_kernel_configs_sha256"] == canonical_sha256(accepted["selected_kernel_configs"])
    assert "rows" not in accepted and "state_audit" not in accepted


@pytest.mark.parametrize("case", ["target_nll", "input", "endpoint", "kernel", "adapter", "coverage", "same_process", "row_hash"])
def test_fresh_probe_compare_rejects_wrong_scores_inputs_or_identity(tmp_path, case):
    paths, values = pair(tmp_path)
    changed = values[1]
    if case == "target_nll":
        changed["result"]["rows"][0]["full_vocab_first_sid_nll"] += .125
    elif case == "input":
        changed["result"]["rows"][0]["user_id"] = "foreign-user"
        changed["probe_inputs"][0]["user_id"] = "foreign-user"
    elif case == "endpoint":
        changed = report(101, step=676)
    elif case == "kernel":
        receipt = changed["result"]["kernel_receipt"]
        receipt["snapshots"][0]["kernels"]["fla.kernel"]["cache"]["shape4"] = "BLOCK=64"
        receipt["snapshots_sha256"] = canonical_sha256(receipt["snapshots"])
    elif case == "adapter":
        changed["result"]["state_audit"]["adapter_state_after"]["sha256"] = "changed"
    elif case == "coverage":
        changed["result"]["coverage"]["scored_rows"] = 3
    elif case == "same_process":
        changed["process"]["pid"] = 100
    else:
        changed["rows_sha256"] = "wrong"
    if case != "row_hash":
        changed["rows_sha256"] = canonical_sha256(changed["result"]["rows"])
    write(paths[1], changed)
    with pytest.raises(ValueError):
        probe.compare_probes(paths)


def test_fp32_signed_zero_is_not_silently_treated_as_bit_exact(tmp_path):
    paths, values = pair(tmp_path)
    values[1]["result"]["rows"][0]["target_logit_fp32"] = -0.0
    values[1]["rows_sha256"] = canonical_sha256(values[1]["result"]["rows"])
    write(paths[1], values[1])
    with pytest.raises(ValueError, match="FP32 probe rows differ exactly"):
        probe.compare_probes(paths)


def test_probe_requires_two_distinct_files(tmp_path):
    paths, _ = pair(tmp_path)
    with pytest.raises(ValueError, match="two independent"):
        probe.compare_probes([paths[0], paths[0]])


def test_saved_adapter_inventory_binds_tokenizer_and_weights(tmp_path):
    adapter = tmp_path/"adapter"
    adapter.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "README.md"):
        (adapter/name).write_text(name)
    original = probe.adapter_inventory(adapter)
    assert "README.md" not in original and "tokenizer.json" in original
    (adapter/"tokenizer.json").write_text("different tokenizer")
    assert probe.adapter_inventory(adapter) != original
