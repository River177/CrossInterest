from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from new_interest.experiments import rho_reference_recovery as recovery
from new_interest.training.checkpoint_persistence import CheckpointValidationError, publish_checkpoint
from new_interest.training.train_sft import _training_checkpoint_identity


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _tree_hashes(root):
    return {str(path.relative_to(root)): recovery._sha(path)
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    project = tmp_path / "project"
    project.mkdir()
    train = project / "single-train.parquet"
    pq.write_table(pa.Table.from_pylist([{"source_sample_id": str(i)} for i in range(32)]), train)
    config = {"model_name": "/scratch/rho-reference-model", "model_revision": "original-model-revision",
              "run_name": "rho-reference-fold0-single-seed42", "train_file": str(train),
              "epochs": 5, "seed": 42, "per_device_train_batch_size": 1,
              "gradient_accumulation_steps": 8, "learning_rate": 2e-5,
              "gradient_checkpointing_use_reentrant": False, "loss_reduction": "token_mean",
              "semantic_id_loss_scope": "full_vocabulary", "enable_thinking": False,
              "persistent_checkpoint_dir": str(tmp_path / "new-remote/checkpoints-single"),
              "resume_from_checkpoint": "auto"}
    identity = _training_checkpoint_identity(config, train_file=train, sample_weight_file=None, adapter_path=None)
    source = tmp_path / "original-local/checkpoint-20"
    source.mkdir(parents=True)
    _write(source / "trainer_state.json", {"global_step": 20, "epoch": 5.0,
        "log_history": [{"step": step, "loss": 5.0 / step} for step in range(1, 21)]})
    _write(source / "adapter_config.json", {"peft_type": "LORA"})
    (source / "training_args.bin").write_bytes(b"original training arguments")
    (source / "rng_state.pth").write_bytes(b"original RNG state")
    torch.save({"state": {0: {"step": torch.tensor(20.), "exp_avg": torch.ones(2)},
                           1: {"step": torch.tensor(20.), "exp_avg": torch.zeros(2)}},
                "param_groups": [{"params": [0, 1]}]}, source / "optimizer.pt")
    torch.save({"last_epoch": 20, "_step_count": 21, "base_lrs": [2e-5]}, source / "scheduler.pt")
    safetensors.save_file({"lora_A": torch.ones(2, 2), "lora_B": torch.full((2, 2), .7)},
                         source / "adapter_model.safetensors")
    parent = publish_checkpoint(source, tmp_path / "old-remote/checkpoints-single", identity)
    _write(project / "recovery_parent.json", {"checkpoint_path": str(parent),
        "checkpoint_marker_sha256": recovery._sha(parent / "checkpoint_complete.json"),
        "runtime_sha256": "a" * 64})
    return {"project": project, "config": config, "parent": parent,
            "local_root": tmp_path / "new-scratch/checkpoints", "output": tmp_path / "new-scratch/single",
            "receipt": tmp_path / "new-remote/compact/single-recovery.json"}


def _prepare(case):
    return recovery.prepare_single_recovery(case["project"], case["config"], case["local_root"])


def _completed_recovery(case, prepared):
    output = case["output"]
    output.mkdir(parents=True)
    (output / "final_adapter").mkdir()
    state = safetensors.load_file(str(Path(prepared["resume_from_checkpoint"]) / "adapter_model.safetensors"))
    safetensors.save_file(state, output / "final_adapter/adapter_model.safetensors")
    _write(output / "run_summary.json", {"resumed_from_checkpoint": prepared["resume_from_checkpoint"],
        "train_file_sha256": prepared["checkpoint_identity"]["train_file_sha256"],
        "checkpoint_identity": prepared["checkpoint_identity"], "input_adapter": None,
        "sample_weight_file": None, "loss_history": prepared["parent_loss_history"]})
    endpoint = {"global_step": 20, "epoch": 5.0, **prepared["parent_optimizer_endpoint"]}
    _write(output / "rank-0-training-runtime.json", {"rank": 0, "world_size": 1,
        "train_samples": 32, "expected_optimizer_steps": 20, "completed_as_configured": True,
        "initial_state": {"state_sha256": "b" * 64, "lora_b_all_zero": True},
        "train_begin": {**endpoint, "resumed_from_checkpoint": True, "fresh_optimization_state": False},
        "train_end": endpoint})
    _write(output / "semantic_adapter_reload_audit.json", {"all_close": True,
        "adapter_state_integrity": {"all_tensors_exact": True}})


def _reseal_test_parent(case, changed_file):
    """Bind a deliberately malformed fixture, so semantic checks are exercised."""
    parent = case["parent"]
    marker_path = parent / "checkpoint_complete.json"
    marker = json.loads(marker_path.read_text())
    file = parent / changed_file
    marker["files"][changed_file] = {"size": file.stat().st_size, "sha256": recovery._sha(file)}
    _write(marker_path, marker)
    binding_path = case["project"] / "recovery_parent.json"
    binding = json.loads(binding_path.read_text())
    binding["checkpoint_marker_sha256"] = recovery._sha(marker_path)
    _write(binding_path, binding)


def test_hydrate_and_accept_zero_update_recovery_preserves_parent_and_discloses_missing_original_audit(case):
    before = _tree_hashes(case["parent"])
    config_before = copy.deepcopy(case["config"])
    prepared = _prepare(case)
    assert prepared["resume_from_checkpoint"] == str(case["local_root"] / "checkpoint-20")
    assert case["config"] == config_before  # caller must explicitly select this resume path
    assert prepared["parent_optimizer_endpoint"]["optimizer_state_entries"] == 2
    _completed_recovery(case, prepared)
    result = recovery.validate_single_recovery(prepared, case["output"], case["receipt"])
    assert result["accepted"] and result["new_optimizer_updates"] == 0
    assert result["original_initial_runtime_audit_missing"] is True
    assert result["initial_state_provenance"] == "reconstructed_before_resume_not_original_start_observation"
    assert result["reconstructed_initial_state_sha256"] == "b" * 64
    assert result["parent_adapter_tensors_exact"] and result["fresh_gpu_reload_exact"]
    assert result["parent_loss_record_count"] == 20
    assert _tree_hashes(case["parent"]) == before
    assert json.loads(case["receipt"].read_text()) == result


def test_hydration_retries_replace_only_local_copy(case):
    before = _tree_hashes(case["parent"])
    first = _prepare(case)
    local = Path(first["resume_from_checkpoint"])
    (local / "optimizer.pt").write_bytes(b"interrupted local hydration")
    second = _prepare(case)
    assert second == first
    assert _tree_hashes(case["parent"]) == before


def test_parent_marker_and_checkpoint_bytes_are_both_verified(case):
    binding_path = case["project"] / "recovery_parent.json"
    binding = json.loads(binding_path.read_text())
    _write(binding_path, {**binding, "checkpoint_marker_sha256": "0" * 64})
    with pytest.raises(ValueError, match="marker differs"):
        _prepare(case)
    _write(binding_path, binding)
    (case["parent"] / "optimizer.pt").write_bytes(b"bad parent bytes")
    with pytest.raises(CheckpointValidationError, match="size mismatch|hash mismatch"):
        _prepare(case)


def test_fresh_derived_identity_must_equal_the_bound_parent(case):
    case["config"]["learning_rate"] = .01
    with pytest.raises(CheckpointValidationError, match="identity mismatch"):
        _prepare(case)


@pytest.mark.parametrize("mutation", ["source_root", "persistent_parent", "adapter", "world", "epochs"])
def test_parent_write_risks_and_changed_recipe_rejected(case, monkeypatch, mutation):
    if mutation == "source_root":
        case["local_root"] = case["parent"].parent
    elif mutation == "persistent_parent":
        case["config"]["persistent_checkpoint_dir"] = str(case["parent"].parent)
    elif mutation == "adapter":
        case["config"]["adapter_path"] = "/a/supervised/adapter"
    elif mutation == "world":
        monkeypatch.setenv("WORLD_SIZE", "4")
    elif mutation == "epochs":
        case["config"]["epochs"] = 6
    before = _tree_hashes(case["parent"])
    with pytest.raises(ValueError, match="separate|forbids|unchanged"):
        _prepare(case)
    assert _tree_hashes(case["parent"]) == before


@pytest.mark.parametrize("file", ["optimizer.pt", "scheduler.pt", "trainer_state.json"])
def test_committed_endpoint_must_really_be_finished_twenty_updates(case, file):
    parent = case["parent"]
    if file == "optimizer.pt":
        optimizer = torch.load(parent / file, weights_only=True)
        optimizer["state"][0]["step"] = torch.tensor(19.)
        torch.save(optimizer, parent / file)
    elif file == "scheduler.pt":
        torch.save({"last_epoch": 19}, parent / file)
    else:
        state = json.loads((parent / file).read_text())
        state["epoch"] = 4.75
        _write(parent / file, state)
    _reseal_test_parent(case, file)
    with pytest.raises(ValueError, match="twenty updates|completed single gate"):
        _prepare(case)


@pytest.mark.parametrize("mutation", ["begin", "end", "new_loss", "wrong_resume", "weights", "reload", "fresh_begin"])
def test_recovery_rejects_new_updates_or_incomplete_proof(case, mutation):
    prepared = _prepare(case)
    _completed_recovery(case, prepared)
    output = case["output"]
    if mutation in ("begin", "end", "fresh_begin"):
        path = output / "rank-0-training-runtime.json"
        audit = json.loads(path.read_text())
        if mutation == "fresh_begin":
            audit["train_begin"]["fresh_optimization_state"] = True
        else:
            audit[f"train_{mutation}"]["global_step"] = 21
        _write(path, audit)
    elif mutation in ("new_loss", "wrong_resume"):
        path = output / "run_summary.json"
        summary = json.loads(path.read_text())
        if mutation == "new_loss":
            summary["loss_history"].append({"step": 21, "loss": .1})
        else:
            summary["resumed_from_checkpoint"] = "/wrong/checkpoint-20"
        _write(path, summary)
    elif mutation == "weights":
        path = output / "final_adapter/adapter_model.safetensors"
        state = safetensors.load_file(str(path))
        state["lora_B"] += .0001
        safetensors.save_file(state, path)
    elif mutation == "reload":
        _write(output / "semantic_adapter_reload_audit.json", {"all_close": False})
    with pytest.raises(ValueError):
        recovery.validate_single_recovery(prepared, output, case["receipt"])
    assert not case["receipt"].exists()


def test_local_checkpoint_may_not_be_silently_modified_by_resumed_run(case):
    prepared = _prepare(case)
    _completed_recovery(case, prepared)
    local = Path(prepared["resume_from_checkpoint"])
    (local / "optimizer.pt").write_bytes(b"new optimizer state")
    with pytest.raises(ValueError, match="Hydrated parent checkpoint bytes changed"):
        recovery.validate_single_recovery(prepared, case["output"], case["receipt"])


def test_receipt_cannot_modify_parent_and_storage_mount_aliases_overlap(case):
    prepared = _prepare(case)
    _completed_recovery(case, prepared)
    before = _tree_hashes(case["parent"])
    with pytest.raises(ValueError, match="must not modify"):
        recovery.validate_single_recovery(prepared, case["output"], case["parent"] / "trainer_state.json")
    assert _tree_hashes(case["parent"]) == before
    assert recovery._overlap(Path('/mnt/default/projects/sft4safety/old/checkpoints'),
                             Path('/mnt/output/projects/sft4safety/old/checkpoints/checkpoint-20'))
    assert not recovery._overlap(Path('/mnt/default/projects/sft4safety/old/checkpoints'),
                                 Path('/mnt/output/projects/sft4safety/new/checkpoints'))


def test_real_hf_completed_checkpoint_resume_calls_audits_without_new_optimizer_updates(tmp_path):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.14.1":
        pytest.skip("Verify the exact worker Trainer resume semantics")
    pytest.importorskip("accelerate")
    from new_interest.training.rho_reference_sft import _optimizer_snapshot

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)

        def forward(self, features, labels):
            logits = self.linear(features)
            return {"loss": torch.nn.functional.cross_entropy(logits, labels), "logits": logits}

    class EndpointCallback(transformers.TrainerCallback):
        def __init__(self):
            self.optimizer_steps = 0
            self.begin = None
            self.end = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.begin = _optimizer_snapshot(kwargs["optimizer"], kwargs["lr_scheduler"], state)

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.optimizer_steps += 1

        def on_train_end(self, args, state, control, **kwargs):
            self.end = _optimizer_snapshot(kwargs["optimizer"], kwargs["lr_scheduler"], state)

    rows = [{"features": torch.tensor([i / 32., 1.]), "labels": i % 3} for i in range(32)]
    def arguments(output):
        return transformers.TrainingArguments(output_dir=str(output), num_train_epochs=5,
            per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-5,
            optim="adamw_torch", use_cpu=True, report_to=[], save_strategy="steps", save_steps=20,
            logging_steps=1, disable_tqdm=True, remove_unused_columns=False, seed=42,
            dataloader_num_workers=0)
    original_callback = EndpointCallback()
    original = transformers.Trainer(model=TinyClassifier(), args=arguments(tmp_path / "original"),
        train_dataset=rows, callbacks=[original_callback])
    original.train()
    assert original_callback.optimizer_steps == 20
    checkpoint = tmp_path / "original/checkpoint-20"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    parent_losses = recovery._loss_history(state)
    parent_tensors = {key: tensor.detach().clone() for key, tensor in original.model.state_dict().items()}
    resumed_callback = EndpointCallback()
    resumed = transformers.Trainer(model=TinyClassifier(), args=arguments(tmp_path / "recovered"),
        train_dataset=rows, callbacks=[resumed_callback])
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed_callback.optimizer_steps == 0
    assert resumed_callback.begin == resumed_callback.end
    assert resumed_callback.begin == {"global_step": 20, "epoch": 5.0, "optimizer_state_entries": 2,
                                      "optimizer_step_min": 20, "optimizer_step_max": 20,
                                      "scheduler_last_epoch": 20}
    assert recovery._loss_history({"log_history": resumed.state.log_history}) == parent_losses
    for key, tensor in resumed.model.state_dict().items():
        assert torch.equal(tensor, parent_tensors[key])
