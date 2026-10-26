from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from new_interest.training.rho_reference_sft import (
    LocalCheckpointCoordinator,
    TrainingRuntimeAudit,
    validate_reference_config,
)


class FakeDistributed:
    def __init__(self, rank, barrier, operation_active):
        self.rank = rank
        self.shared_barrier = barrier
        self.operation_active = operation_active
        self.collectives = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 2

    def get_rank(self):
        return self.rank

    def barrier(self):
        assert not self.operation_active.is_set(), "Collective entered during network I/O"
        self.collectives += 1
        self.shared_barrier.wait(timeout=2)

    def broadcast_object_list(self, *args, **kwargs):
        raise AssertionError("Must not broadcast while rank zero does network I/O")


@pytest.mark.parametrize("failure", [False, True])
def test_peers_poll_local_outcome_without_collective_during_network_io(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    barrier = threading.Barrier(2)
    active = threading.Event()
    backends = [FakeDistributed(rank, barrier, active) for rank in range(2)]
    coordinators = [
        LocalCheckpointCoordinator(tmp_path, distributed=backend, timeout_seconds=2, poll_seconds=.001)
        for backend in backends
    ]
    calls = []

    def network_operation():
        active.set()
        time.sleep(.05)
        calls.append(1)
        active.clear()
        if failure:
            raise OSError("simulated mounted-storage failure")
        return tmp_path / "committed-checkpoint"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(coordinator, network_operation) for coordinator in coordinators]
        for future in futures:
            if failure:
                with pytest.raises(RuntimeError, match="OSError: simulated mounted-storage failure"):
                    future.result(timeout=3)
            else:
                assert future.result(timeout=3) == str(tmp_path / "committed-checkpoint")
    assert calls == [1]
    assert [backend.collectives for backend in backends] == [1, 1]


def test_rendezvous_reuse_and_missing_rank_zero_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    backend = FakeDistributed(1, threading.Barrier(1), threading.Event())
    coordinator = LocalCheckpointCoordinator(tmp_path, distributed=backend, timeout_seconds=.01, poll_seconds=.001)
    with pytest.raises(RuntimeError, match="timed out"):
        coordinator(lambda: pytest.fail("Nonzero rank must not publish"))
    (tmp_path / "operation-000000.json").write_text(json.dumps({"sequence": 0, "value": "stale", "error": None}))
    fresh = LocalCheckpointCoordinator(tmp_path, distributed=backend)
    with pytest.raises(RuntimeError, match="already exists"):
        fresh(lambda: None)


@pytest.mark.parametrize("extra", [
    {"adapter_path": "/mixed/adapter"},
    {"sample_weight_file": "/old/weights.parquet"},
    {"mera_relations_file": "/old/relations.json"},
    {"semantic_id_loss_scope": "taxonomy"},
    {"loss_reduction": "per_example_mean"},
    {"enable_thinking": True},
    {"gradient_checkpointing_use_reentrant": True},
])
def test_reference_config_rejects_non_native_or_leaking_initialization(extra):
    with pytest.raises(ValueError):
        validate_reference_config({
            "semantic_id_token_mapping_file": "/frozen/mapping.parquet",
            "gradient_checkpointing_use_reentrant": False,
            **extra,
        })


def test_native_reference_config_requires_explicit_non_reentrant_override():
    config = {"semantic_id_token_mapping_file": "/frozen/mapping.parquet"}
    with pytest.raises(ValueError, match="non-reentrant"):
        validate_reference_config(config)
    validate_reference_config({**config, "gradient_checkpointing_use_reentrant": False})


def runtime(tmp_path, *, epochs=3, samples=10800, world_size=4):
    config = {
        "output_dir": str(tmp_path), "epochs": epochs,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    }
    return TrainingRuntimeAudit(
        config, {"state_sha256": "initial", "tensor_count": 1, "tensors": {}},
        train_samples=samples, rank=0, world_size=world_size,
    )


def test_runtime_records_actual_fresh_and_final_optimizer_state(tmp_path):
    audit = runtime(tmp_path)
    optimizer = SimpleNamespace(state={})
    scheduler = SimpleNamespace(last_epoch=0)
    state = SimpleNamespace(global_step=0, epoch=0)
    audit.begin(state, optimizer, scheduler, resumed=False)
    assert audit.expected_steps == 1014
    begin = json.loads(audit.output.read_text())["train_begin"]
    assert begin["fresh_optimization_state"] is True
    state.global_step = 1014
    state.epoch = 3.0
    optimizer.state = {"parameter": {"step": 1014}}
    scheduler.last_epoch = 1014
    audit.end(state, optimizer, scheduler)
    saved = json.loads(audit.output.read_text())
    assert saved["completed_as_configured"] is True
    assert saved["train_end"]["optimizer_step_min"] == 1014


def test_runtime_rejects_stale_optimizer_and_short_training(tmp_path):
    audit = runtime(tmp_path)
    state = SimpleNamespace(global_step=0, epoch=0)
    with pytest.raises(ValueError, match="fresh optimization"):
        audit.begin(state, SimpleNamespace(state={"p": {"step": 1}}), SimpleNamespace(last_epoch=0), resumed=False)
    state.global_step, state.epoch = 1000, 1000 / 338
    with pytest.raises(ValueError, match="endpoint"):
        audit.end(state, SimpleNamespace(state={"p": {"step": 1000}}), SimpleNamespace(last_epoch=1000))
    assert json.loads(audit.output.read_text())["completed_as_configured"] is False


def test_runtime_resume_cursor_alignment(tmp_path):
    audit = runtime(tmp_path)
    state = SimpleNamespace(global_step=400, epoch=400 / 338)
    optimizer = SimpleNamespace(state={"p": {"step": 400}})
    audit.begin(state, optimizer, SimpleNamespace(last_epoch=400), resumed=True)
    assert audit.payload["train_begin"]["fresh_optimization_state"] is False
    with pytest.raises(ValueError, match="cursors differ"):
        audit.begin(state, optimizer, SimpleNamespace(last_epoch=399), resumed=True)


def test_adapter_initial_manifest_detects_changed_or_pretrained_lora_b():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from new_interest.training.rho_reference_sft import adapter_initial_state

    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(3, 2)), LoraConfig(r=2, target_modules=["0"]))
    initial = adapter_initial_state(model)
    assert initial["tensor_count"] == 2
    assert initial["lora_b_all_zero"] is True
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if "lora_A" in n).add_(1)
    assert adapter_initial_state(model)["state_sha256"] != initial["state_sha256"]
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if "lora_B" in n).add_(1)
    with pytest.raises(ValueError, match="LoRA B is not fresh"):
        adapter_initial_state(model)


def test_scoped_patches_restore_after_training_failure(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from new_interest.training import train_sft
    from new_interest.training.rho_reference_sft import train_reference

    monkeypatch.setenv("RHO_SYNC_DIR", str(tmp_path))
    original_trainer = transformers.Trainer
    original_operation = train_sft._run_rank_zero_checkpoint_operation

    def fail(config, *, trainer_base):
        assert transformers.Trainer is original_trainer
        assert issubclass(trainer_base, original_trainer)
        assert trainer_base is not original_trainer
        assert train_sft._run_rank_zero_checkpoint_operation is not original_operation
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(train_sft, "train", fail)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        train_reference({
            "semantic_id_token_mapping_file": "/frozen/mapping.parquet",
            "gradient_checkpointing_use_reentrant": False,
        })
    assert transformers.Trainer is original_trainer
    assert train_sft._run_rank_zero_checkpoint_operation is original_operation


def test_actual_cpu_trainer_callbacks_preserve_four_optimizer_updates(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from new_interest.training.rho_reference_sft import _instrumented_trainer

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 2)

        def forward(self, input_ids, labels=None, **kwargs):
            logits = self.projection(input_ids.float())
            return {"loss": torch.nn.functional.cross_entropy(logits, labels), "logits": logits}

    torch.manual_seed(42)
    model = get_peft_model(TinyClassifier(), LoraConfig(r=2, target_modules=["projection"]))
    config = {
        "output_dir": str(tmp_path), "epochs": 2,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    }
    trainer_class = _instrumented_trainer(Trainer, TrainerCallback, config)
    trainer = trainer_class(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "checkpoints"), use_cpu=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=2, report_to=[], disable_tqdm=True,
            save_strategy="no", remove_unused_columns=False,
            dataloader_pin_memory=False,
        ),
        train_dataset=[
            {"input_ids": torch.tensor([1., float(i), -1.]), "labels": i % 2}
            for i in range(16)
        ],
    )
    trainer.train()
    record = json.loads((tmp_path / "rank-0-training-runtime.json").read_text())
    assert record["train_begin"]["fresh_optimization_state"] is True
    assert record["completed_as_configured"] is True
    assert record["train_end"]["global_step"] == 4
    assert record["train_end"]["optimizer_state_entries"] == 2


COLD_ENTRY_REPRO = r'''
import inspect
from new_interest.training import train_sft
from new_interest.training.rho_reference_sft import train_reference

class ImportBoundaryVerified(BaseException):
    pass

def inspect_actual_import(seed):
    actual = inspect.currentframe().f_back.f_locals["Trainer"]
    if actual.__name__ != "ReferenceTrainer":
        raise AssertionError("Real train_reference -> train_sft.train resolved uninstrumented " + str(actual))
    print("ACTUAL_ENTRY_REFERENCE_TRAINER", flush=True)
    raise ImportBoundaryVerified()

# Stop at the first line after train_sft's actual local imports. Do not import
# Trainer in advance: that warming masked the production LazyModule bug.
train_sft._set_seed = inspect_actual_import
try:
    train_reference({"semantic_id_token_mapping_file": "frozen-mapping.parquet",
                     "gradient_checkpointing_use_reentrant": False})
except ImportBoundaryVerified:
    pass
else:
    raise AssertionError("Actual train_sft import boundary was not reached")
'''


def test_cold_real_reference_entry_selects_instrumented_trainer(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["RHO_SYNC_DIR"] = str(tmp_path / "cold-entry-sync")
    result = subprocess.run(
        [sys.executable, "-c", COLD_ENTRY_REPRO], env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTUAL_ENTRY_REFERENCE_TRAINER" in result.stdout


def test_train_sft_default_without_injection_keeps_native_trainer(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    import inspect
    from transformers.trainer import Trainer
    from new_interest.training import train_sft

    class BoundaryVerified(BaseException):
        pass

    def inspect_actual_import(seed):
        assert inspect.currentframe().f_back.f_locals["Trainer"] is Trainer
        raise BoundaryVerified()

    monkeypatch.setattr(train_sft, "_set_seed", inspect_actual_import)
    with pytest.raises(BoundaryVerified):
        train_sft.train({})


def test_real_reference_entry_trains_and_writes_runtime_receipt(tmp_path, monkeypatch):
    """Exercise wrapper -> real train_sft -> actual Trainer -> SID adapter save."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    import importlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from new_interest.training.rho_reference_sft import train_reference

    base = tmp_path / "tiny-base"
    torch.manual_seed(42)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=6, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=["full_attention"], tie_word_embeddings=True,
        attention_dropout=0, use_cache=False,
    ))
    model.save_pretrained(base)
    backend = Tokenizer(models.WordLevel({name: i for i, name in enumerate([
        "<unk>", "<pad>", "<eos>", "hello", "movie", "sport",
    ])}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = '{{ messages[0]["content"] }}{% if messages|length > 1 %} {{ messages[1]["content"] }} <eos>{% endif %}'
    tokenizer.save_pretrained(base)
    mapping = tmp_path / "mapping.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"interest_index": i, "interest_token": f"<interest_{i:03d}>", "target_subcategory": name}
        for i, name in enumerate(["movie", "sport"])
    ]), mapping)
    rows = []
    for i in range(4):
        prompt = "hello " + ("movie" if i % 2 == 0 else "sport")
        response = f"<interest_{i%2:03d}>"
        rows.append({"source_sample_id": str(i), "prompt": prompt, "response": response,
                     "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]})
    train_file = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train_file)

    # The production entry intentionally targets CUDA. Only device arguments are
    # adapted for this tiny CPU fixture; the actual data/model/trainer path runs.
    transformers = importlib.import_module("transformers")
    original_arguments = transformers.TrainingArguments

    def cpu_arguments(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, tf32=False, optim="adamw_torch",
                      dataloader_pin_memory=False, disable_tqdm=True)
        return original_arguments(**kwargs)

    monkeypatch.setattr(transformers, "TrainingArguments", cpu_arguments)
    monkeypatch.setenv("RHO_SYNC_DIR", str(tmp_path / "sync"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = tmp_path / "sft"
    config = {
        "model_name": str(base), "train_file": str(train_file), "output_dir": str(output),
        "semantic_id_token_mapping_file": str(mapping), "semantic_id_token_count": 2,
        "run_name": "cpu-real-reference-entry", "epochs": 2, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2, "gradient_checkpointing_use_reentrant": False,
        "lora_r": 2, "lora_alpha": 4, "lora_dropout": 0, "target_modules": "all-linear",
        "learning_rate": .001, "warmup_ratio": 0, "wandb_mode": "disabled",
        "logging_steps": 1, "save_strategy": "no", "dataloader_num_workers": 0,
        "max_seq_length": 64, "seed": 42,
    }
    summary = train_reference(config)
    receipt = json.loads((output / "rank-0-training-runtime.json").read_text())
    assert summary["train_samples"] == 4
    assert receipt["train_begin"]["fresh_optimization_state"] is True
    assert receipt["train_end"]["global_step"] == 4
    assert receipt["train_end"]["scheduler_last_epoch"] == 4
    assert receipt["completed_as_configured"] is True
    assert (output / "final_adapter/adapter_model.safetensors").is_file()
