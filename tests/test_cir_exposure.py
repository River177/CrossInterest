import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.training.sft_exposure import (
    TrainingExposure, ExposureIndexedDataset, ExposureCollator, make_exposure_trainer,
)


def rows():
    return [{"sample_id": f"cir:{i}", "target_subcategory": "A",
             "training_task": "cross_interest_reconstruction"} for i in range(13)]


def test_cir_parquet_exposure_checkpoint_and_exact_counts(tmp_path):
    path = tmp_path / "cir.parquet"
    pq.write_table(pa.Table.from_pylist(rows()), path)
    def load():
        return TrainingExposure.from_parquet(path, train_sha256="a"*64, expected_epochs=1,
                                             max_samples=None, default_task="new_interest")
    original = load()
    original.record(list(range(8)), [3]*8, [10]*8)
    original.save_checkpoint(tmp_path / "checkpoint", 1)
    (tmp_path / "checkpoint/trainer_state.json").write_text(json.dumps({"global_step": 1}))
    restored = load()
    restored.restore_checkpoint(tmp_path / "checkpoint")
    restored.record(list(range(8,13)) + [-1], [3]*5+[0], [10]*6)
    result = restored.finish(tmp_path / "final", optimizer_steps=2)
    assert result["presentations"] == 13 and result["padding_slots_ignored"] == 1
    assert result["by_task"]["cross_interest_reconstruction"]["assistant_tokens"] == 39
    assert restored.counts == [1]*13


def test_unknown_task_and_wrong_exposure_still_fail(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        TrainingExposure([{**rows()[0], "training_task":"typo"}], train_sha256="x", expected_epochs=1)
    ledger = TrainingExposure(rows(), train_sha256="x", expected_epochs=1)
    ledger.record([0]*13, [3]*13, [10]*13)
    with pytest.raises(ValueError, match="Actual exposure"):
        ledger.finish(tmp_path, optimizer_steps=2)


def test_actual_native_trainer_accepts_cir_and_counts_tail(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from new_interest.training.sft_data import CompletionOnlyCollator
    torch.manual_seed(42)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=1, pad_token_id=0))
    examples = [{"input_ids":[1,3,4,2], "attention_mask":[1]*4, "labels":[-100,-100,4,2]} for _ in rows()]
    ledger = TrainingExposure(rows(), train_sha256="x", expected_epochs=1)
    args = transformers.TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
        num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=4,
        optim="adamw_torch", save_strategy="no", logging_strategy="no", disable_tqdm=True,
        dataloader_num_workers=0, remove_unused_columns=False, seed=42)
    trainer = make_exposure_trainer(transformers.Trainer, ledger)(model=model, args=args,
        train_dataset=ExposureIndexedDataset(examples),
        data_collator=ExposureCollator(CompletionOnlyCollator(SimpleNamespace(pad_token_id=0))))
    trainer.train()
    assert trainer.state.global_step == 4
    assert ledger.finish(tmp_path / "accepted", optimizer_steps=4)["presentations"] == 13
