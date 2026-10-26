from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from new_interest.experiments import rho_reference_rescore as rescore
from new_interest.training.checkpoint_persistence import CheckpointValidationError, publish_checkpoint


@pytest.fixture
def parent_case(tmp_path):
    train = tmp_path / "fold1.parquet"
    pq.write_table(pa.Table.from_pylist([{"user_id": f"user-{i}", "target_subcategory": f"target-{i // 100}"}
                                       for i in range(10800)]), train)
    identity = {"world_size": 4, "epochs": 3.0, "seed": 42, "effective_global_batch_size": 32,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "input_adapter_sha256": None, "sample_weight_file_sha256": None,
        "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary",
        "enable_thinking": False, "model_revision": "original-revision",
        "run_name": "rho-reference-fold1-full-seed42", "train_file_sha256": rescore.sha(train)}
    source = tmp_path / "original-scratch/checkpoint-1014"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"authenticated adapter bytes")
    (source / "adapter_config.json").write_text('{}')
    (source / "training_args.bin").write_bytes(b"original training arguments")
    for rank in range(4):
        (source / f"rng_state_{rank}.pth").write_bytes(f"original RNG {rank}".encode())
    torch.save({"state": {i: {"step": torch.tensor(1014.)} for i in range(373)}}, source / "optimizer.pt")
    torch.save({"last_epoch": 1014}, source / "scheduler.pt")
    rescore.write(source / "trainer_state.json", {"global_step": 1014, "epoch": 3.0,
        "log_history": [{"step": i, "loss": 2.0 / i} for i in range(1, 1015)]})
    parent = publish_checkpoint(source, tmp_path / "old-remote/checkpoints-full", identity)
    binding = {"checkpoint_path": str(parent), "checkpoint_marker_sha256": rescore.sha(parent / "checkpoint_complete.json"),
               "checkpoint_identity": identity, "adapter_sha256": rescore.sha(parent / "adapter_model.safetensors")}
    contract = {"version": rescore.VERSION, "kernel_selection": "stable_fla_v1", "model_updates": 0, "folds": 2,
        "expected_initial_state_sha256": rescore.INITIAL_STATE, "original_training_runtime_sha256": "a" * 64,
        "historical_reload_issue": {"native_max_abs_difference": [0.125]*4, "original_raw_probe_logits_available": False},
        "parents": {"0": {**binding, "checkpoint_path": str(tmp_path / "other-old/checkpoints-full/checkpoint-1014")}, "1": binding}}
    return {"contract": contract, "fold": 1, "train_file": train, "output_root": tmp_path / "new-remote",
            "parent": parent}


def verify(case):
    return rescore.verify_parent(**{key: case[key] for key in ("contract", "fold", "train_file", "output_root")})


def parent_hashes(parent):
    return {path.name: rescore.sha(path) for path in parent.iterdir() if path.is_file()}


def reseal(case, name):
    parent = case["parent"]
    path = parent / "checkpoint_complete.json"
    marker = rescore.read(path)
    marker["files"][name] = {"sha256": rescore.sha(parent / name), "size": (parent / name).stat().st_size}
    rescore.write(path, marker)
    case["contract"]["parents"]["1"]["checkpoint_marker_sha256"] = rescore.sha(path)


def test_parent_endpoint_verified_without_mutating_or_resuming_training(parent_case):
    before = parent_hashes(parent_case["parent"])
    result = verify(parent_case)
    assert result["optimizer_step_min"] == result["optimizer_step_max"] == result["scheduler_last_epoch"] == 1014
    assert len(result["loss_history"]) == 1014
    assert result["parent_files_mutated"] is False
    assert parent_hashes(parent_case["parent"]) == before
    summary = rescore.build_lineage_summary(parent_case["contract"], 1, result, parent_case["train_file"])
    assert summary["input_adapter"] is None
    assert summary["input_adapter_field_scope"].startswith("original training initialization")
    assert summary["artifact_role"] == "checkpoint_lineage_reconstructed_for_inference_only"
    assert summary["new_training_updates"] == 0 and not summary["training_repeated"]
    assert not summary["original_training_forward_reference_reconstructed"]
    assert summary["historical_reload_issue"]["native_max_abs_difference"] == [.125] * 4
    assert "not equivalence" in summary["historical_parity_limitation"]


@pytest.mark.parametrize("mutation", ["marker", "weights", "train_identity", "no_failure", "native_mode", "updates", "output_parent", "other_parent"])
def test_rescore_rejects_changed_parent_scope_or_protocol(parent_case, mutation):
    case = parent_case
    if mutation == "marker":
        case["contract"]["parents"]["1"]["checkpoint_marker_sha256"] = "0" * 64
    elif mutation == "weights":
        (case["parent"] / "adapter_model.safetensors").write_bytes(b"tampered bytes")
    elif mutation == "train_identity":
        case["contract"]["parents"]["1"]["checkpoint_identity"]["input_adapter_sha256"] = "a" * 64
    elif mutation == "no_failure":
        case["contract"]["historical_reload_issue"] = {}
    elif mutation == "native_mode":
        case["contract"]["kernel_selection"] = "native"
    elif mutation == "updates":
        case["contract"]["model_updates"] = 1
    elif mutation == "output_parent":
        case["output_root"] = case["parent"].parent.parent
    elif mutation == "other_parent":
        case["output_root"] = Path(case["contract"]["parents"]["0"]["checkpoint_path"]).parent
    with pytest.raises((ValueError, CheckpointValidationError)):
        verify(case)


@pytest.mark.parametrize("name", ["optimizer.pt", "scheduler.pt", "trainer_state.json"])
def test_authenticated_but_wrong_training_endpoint_rejected(parent_case, name):
    parent = parent_case["parent"]
    if name == "optimizer.pt":
        torch.save({"state": {0: {"step": torch.tensor(1013.)}}}, parent / name)
    elif name == "scheduler.pt":
        torch.save({"last_epoch": 1013}, parent / name)
    else:
        state = rescore.read(parent / name)
        state["epoch"] = 2.99
        rescore.write(parent / name, state)
    reseal(parent_case, name)
    with pytest.raises(ValueError, match="endpoint differs"):
        verify(parent_case)


def test_file_publication_is_byte_exact_idempotent_and_refuses_overwrite(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "persisted/file"
    source.write_bytes(b"original closed stream")
    expected = rescore.sha(source)
    assert rescore.persist_file(source, destination, expected) == expected
    assert rescore.persist_file(source, destination, expected) == expected
    source.write_bytes(b"new source must not overwrite existing result")
    with pytest.raises(ValueError, match="Refusing to replace"):
        rescore.persist_file(source, destination)
    assert destination.read_bytes() == b"original closed stream"


def test_scoring_teacher_manifest_requires_all_consumed_files(tmp_path):
    names = ["run_summary.json", "final_adapter/adapter_model.safetensors", "final_adapter/adapter_config.json",
             "final_adapter/tokenizer.json", "final_adapter/tokenizer_config.json", "final_adapter/interest_token_mapping.parquet"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    identity = {"contract": "frozen"}
    files = {name: {"sha256": rescore.sha(tmp_path / name), "bytes": (tmp_path / name).stat().st_size} for name in names}
    rescore.write(tmp_path / "artifact_manifest.json", {"identity": identity, "files": files})
    assert rescore.verify_teacher(tmp_path, identity)["files"] == files
    (tmp_path / "final_adapter/added_tokens.json").write_text('{}')
    with pytest.raises(ValueError, match="Unauthenticated extra"):
        rescore.verify_teacher(tmp_path, identity)


def probe_report(directory):
    from new_interest.evaluation.score_rho_reference import kernel_selection_identity
    identity = {"contract": "same"}
    selection = kernel_selection_identity("stable_fla_v1")
    snapshots = [{"stage": stage, "kernels": {"fla.kernel": {"cache": {"key": "config"}, "cache_results": False}}}
                 for stage in ("native_head_audit", "checkpoint:0:4")]
    receipt = {**selection, "evaluation_identity": rescore.canonical(identity),
               "snapshots": snapshots, "snapshots_sha256": rescore.canonical(snapshots)}
    rescore.write(directory / "kernel_selection.json", receipt)
    return {"completed": True, "probe_identity": identity, "model_updates": 0,
        "adapter_before": {"all_tensors_exact": True}, "adapter_after": {"all_tensors_exact": True},
        "kernel_receipt": receipt, "kernel_directory": str(directory),
        "kernel_manifest": {**selection, "evaluation_identity": rescore.canonical(identity),
            "kernel_selection_sha256": rescore.sha(directory / "kernel_selection.json"), "checkpoint_size": 4, "shard_record_count": 4},
        "batch4_fp32": [{"full_vocab_first_sid_nll": float(i)} for i in range(4)],
        "rows": [{"source_sample_id": f"sample{i}", "source_identity_sha256": str(i), "prefix_ids_sha256": str(i),
                  "native_sid_logits": [1., 2.], "native_full_logsumexp": 4.,
                  "fp32": {"sid_scores_fp32": [1.0001, 2.0001], "full_vocab_first_sid_nll": 3.},
                  "native_head_audit": {"native_head_max_abs_error": 0}} for i in range(4)]}


def test_fresh_probe_acceptance_is_exact_and_never_claims_historical_parity(tmp_path):
    first, second = probe_report(tmp_path / "replica0"), probe_report(tmp_path / "replica1")
    result = rescore.compare_probes(first, second)
    assert result["accepted"] and result["native_and_fp32_exact"]
    assert result["historical_training_parity_claimed"] is False


@pytest.mark.parametrize("mutation", ["native", "fp32", "batch4", "prefix", "weights", "kernel", "head", "identity", "cache_config", "incomplete"])
def test_probe_comparison_cannot_hide_native_or_fp32_difference(tmp_path, mutation):
    first, second = probe_report(tmp_path / "replica0"), probe_report(tmp_path / "replica1")
    if mutation == "native":
        second["rows"][0]["native_sid_logits"][0] += .125
    elif mutation == "fp32":
        second["rows"][0]["fp32"]["sid_scores_fp32"][0] += .000001
    elif mutation == "batch4":
        second["batch4_fp32"][0]["full_vocab_first_sid_nll"] += .000001
    elif mutation == "prefix":
        second["rows"][0]["prefix_ids_sha256"] = "changed"
    elif mutation == "weights":
        second["adapter_after"]["all_tensors_exact"] = False
    elif mutation == "kernel":
        second["kernel_receipt"]["kernel_selection"] = "native"
    elif mutation == "head":
        second["rows"][0]["native_head_audit"]["native_head_max_abs_error"] = .125
    elif mutation == "identity":
        second["probe_identity"]["contract"] = "other"
    elif mutation == "incomplete":
        second["completed"] = False
    elif mutation == "cache_config":
        receipt = second["kernel_receipt"]
        for snapshot in receipt["snapshots"]:
            snapshot["kernels"]["fla.kernel"]["cache"]["key"] = "different configuration"
        receipt["snapshots_sha256"] = rescore.canonical(receipt["snapshots"])
        directory = Path(second["kernel_directory"])
        rescore.write(directory / "kernel_selection.json", receipt)
        second["kernel_manifest"]["kernel_selection_sha256"] = rescore.sha(directory / "kernel_selection.json")
    with pytest.raises(ValueError):
        rescore.compare_probes(first, second)
