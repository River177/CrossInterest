from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest

from new_interest.evaluation import score_rho_student as student
from new_interest.evaluation.evaluate_beam import canonical_sha256
from new_interest.evaluation.score_rho_reference import encode_prefix, project_full_vocab, source_identity
from new_interest.training.semantic_id_tokens import InterestTokenEntry, capture_torch_numeric_backend


class Tokenizer:
    pad_token_id = 0

    def __len__(self):
        return 64

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        text = " ".join(m["role"]+" "+m["content"] for m in messages)
        return text + (" assistant" if add_generation_prompt else " end")

    def __call__(self, text, **kwargs):
        words = {"user": 2, "assistant": 3, "end": 4, "history": 5, "news": 6}
        words.update({str(i): i+7 for i in range(20)})
        words.update({f"<interest_{i:03d}>": 60+i for i in range(4)})
        ids = [words[word] for word in text.split()]
        return {"input_ids": ids, "attention_mask": [1]*len(ids)}


def cohort(n=9):
    return [{"source_sample_id": f"sample-{i}", "user_id": f"user-{i}", "target_subcategory": f"class-{i%4}",
        "messages": [{"role": "user", "content": "history "+str(i)+" news"*(i%3)},
                     {"role": "assistant", "content": f"<interest_{i%4:03d}>"}]} for i in range(n)]


def spec():
    return SimpleNamespace(mapping_sha256="fixed-mapping", token_ids=[60, 61, 62, 63],
        entries=tuple(InterestTokenEntry(i, f"<interest_{i:03d}>", f"class-{i}", (5,), 60+i) for i in range(4)))


class Config:
    def all_kwargs(self):
        return {"BLOCK": 32, "num_warps": 4}

    def __str__(self):
        return "BLOCK=32,num_warps=4"


class Tuner:
    def __init__(self):
        self.cache = {"old-training-key": Config()}
        self.cache_results = True
        self._bench = lambda *args, config, **kwargs: [1., 1., 1.]

    def run(self, key):
        if key not in self.cache:
            config = Config()
            self._bench(config=config)
            self.cache[key] = config
        self.best_config = self.cache[key]


def tiny():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):
        pytest.skip("Qwen3.5 fixture dependencies unavailable")
    torch.manual_seed(73)
    cfg = transformers.Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["full_attention"]*2, tie_word_embeddings=True, attention_dropout=0, use_cache=False)
    cfg._attn_implementation = "eager"
    model = peft.get_peft_model(transformers.Qwen3_5ForCausalLM(cfg), peft.LoraConfig(r=2, lora_alpha=4,
        lora_dropout=.2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", trainable_token_indices=[60, 61, 62, 63]))
    model.train()
    next(m for m in model.modules() if isinstance(m, torch.nn.Dropout)).eval()
    first = next(p for p in model.parameters() if p.requires_grad)
    first.grad = torch.ones_like(first)
    model.register_buffer("runtime_probe", torch.tensor([2.]))
    model.register_buffer("lazy_probe", None)
    return model


def rng_state(torch):
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def assert_rng_equal(torch, left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0] and np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def install_probe_hook(model, tuner, *, fail=False):
    import torch
    calls = []

    def hook(module, args, kwargs):
        assert all(not m.training for m in model.modules())
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        random.random(); np.random.rand(); torch.rand(3)
        model.runtime_probe.add_(1)
        model.lazy_probe = torch.tensor([3.])
        tuner.run(tuple(kwargs["input_ids"].shape))
        calls.append(kwargs["input_ids"].clone())
        if fail:
            raise RuntimeError("intentional forward error")

    handle = model.get_base_model().model.register_forward_pre_hook(hook, with_kwargs=True)
    return handle, calls


@pytest.mark.parametrize("fail", [False, True])
def test_live_scoring_restores_flags_rng_buffers_gradients_and_tuners_on_every_exit(fail):
    torch = pytest.importorskip("torch")
    model, tuner = tiny(), Tuner()
    original_flags = [(m, m.training) for m in model.modules()]
    original_values = {n: p.detach().clone() for n, p in model.named_parameters()}
    original_gradients = {n: (p.grad, None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    original_buffer = model.runtime_probe
    original_cache, original_bench = tuner.cache, tuner._bench
    original_rng, original_backend = rng_state(torch), capture_torch_numeric_backend(torch)
    handle, calls = install_probe_hook(model, tuner, fail=fail)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            if fail:
                with pytest.raises(RuntimeError, match="intentional forward error"):
                    student.score_current_student(model, Tokenizer(), cohort(), spec(), student_state={"epoch": 1},
                                                  kernel_tuners={"fla.test": tuner})
            else:
                result = student.score_current_student(model, Tokenizer(), cohort(), spec(), student_state={"epoch": 1},
                                                       kernel_tuners={"fla.test": tuner})
                assert result["state_audit"]["model_updates"] == 0
            assert torch.is_autocast_enabled("cpu")
    finally:
        handle.remove()
    assert calls and all(m.training == flag for m, flag in original_flags)
    assert_rng_equal(torch, original_rng, rng_state(torch))
    assert capture_torch_numeric_backend(torch) == original_backend
    assert model.runtime_probe is original_buffer and model.runtime_probe.tolist() == [2.]
    assert model.lazy_probe is None
    assert tuner.cache is original_cache and tuner._bench is original_bench and tuner.cache_results is True
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original_values[name])
        grad, value = original_gradients[name]
        assert parameter.grad is grad
        if value is not None:
            assert torch.equal(parameter.grad, value)


def test_modulo_shards_have_exact_source_identity_and_one_head_copy_per_call(monkeypatch):
    torch = pytest.importorskip("torch")
    model, records, mapping = tiny(), cohort(), spec()
    original = student.effective_output_weight
    copies = []

    def observed(model):
        copies.append(1)
        return original(model)

    monkeypatch.setattr(student, "effective_output_weight", observed)
    results = []
    for shard in range(4):
        tuner = Tuner()
        handle, _ = install_probe_hook(model, tuner)
        try:
            results.append(student.score_current_student(model, Tokenizer(), records, mapping,
                student_state={"run": "test", "global_step": 0, "epoch": 0},
                shard_index=shard, batch_size=2, kernel_tuners={"fla.test": tuner}))
        finally:
            handle.remove()
    assert len(copies) == 4
    assert len({r["identity"]["evaluation_identity"] for r in results}) == 1
    all_rows = [row for result in results for row in result["rows"]]
    assert sorted(r["record_index"] for r in all_rows) == list(range(len(records)))
    for shard, result in enumerate(results):
        assert [r["record_index"] for r in result["rows"]] == list(range(shard, len(records), 4))
        assert result["coverage"]["exact_modulo_coverage"]
        receipt = result["kernel_receipt"]
        assert receipt["snapshots_sha256"] == canonical_sha256(receipt["snapshots"])
        assert len(receipt["snapshots"]) == 1
        assert receipt["snapshots"][0]["stage"] == "student_shard_complete"
        for row in result["rows"]:
            original_row = records[row["record_index"]]
            assert row["source_sample_id"] == original_row["source_sample_id"]
            assert row["user_id"] == original_row["user_id"]
            assert row["target_subcategory"] == original_row["target_subcategory"]
            assert row["input_sha256"] == source_identity(original_row)
            assert row["score_role"] == "current_student_in_sample"


def test_live_scores_match_last_valid_hidden_projection_and_change_with_state():
    torch = pytest.importorskip("torch")
    model, records, mapping, tokenizer = tiny(), cohort(4), spec(), Tokenizer()
    state = {"epoch": 0, "global_step": 0}
    tuner = Tuner()
    handle, _ = install_probe_hook(model, tuner)
    try:
        first = student.score_current_student(model, tokenizer, records, mapping, student_state=state,
                                             num_shards=1, kernel_tuners={"fla.test": tuner})
    finally:
        handle.remove()
    state["epoch"] = 99
    assert first["identity"]["student_state"]["epoch"] == 0
    model.eval()
    with torch.inference_mode():
        weight = student.effective_output_weight(model).float()
        for row, record in zip(first["rows"], records):
            prefix = encode_prefix(tokenizer, record, mapping, max_seq_length=2048, enable_thinking=False)
            ids = torch.tensor([prefix["input_ids"]])
            hidden = model.get_base_model().model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[:, -1]
            logits = project_full_vocab(hidden, weight)
            expected = torch.nn.functional.cross_entropy(logits, torch.tensor([prefix["target_token_id"]]))
            assert row["full_vocab_first_sid_nll"] == pytest.approx(float(expected), abs=1e-5)
    tuner = Tuner()
    handle, _ = install_probe_hook(model, tuner)
    try:
        second = student.score_current_student(model, tokenizer, records, mapping, student_state={"epoch": 1, "global_step": 675},
                                              num_shards=1, kernel_tuners={"fla.test": tuner})
    finally:
        handle.remove()
    assert first["identity"]["evaluation_identity"] != second["identity"]["evaluation_identity"]
    assert first["identity"]["adapter_state_sha256"] == second["identity"]["adapter_state_sha256"]


@pytest.mark.parametrize("case", ["duplicate_user", "duplicate_sample", "empty_shard", "wrapped", "missing_state", "overlong"])
def test_bad_live_scoring_inputs_fail_without_damaging_training_state(case):
    torch = pytest.importorskip("torch")
    model, records, mapping, tuner = tiny(), cohort(4), spec(), Tuner()
    kwargs = {"student_state": {"epoch": 0}, "num_shards": 1, "kernel_tuners": {"fla.test": tuner}}
    if case == "duplicate_user": records[1]["user_id"] = records[0]["user_id"]
    elif case == "duplicate_sample": records[1]["source_sample_id"] = records[0]["source_sample_id"]
    elif case == "empty_shard": kwargs.update(num_shards=5, shard_index=4)
    elif case == "wrapped": model._original_forward = model.forward
    elif case == "missing_state": kwargs["student_state"] = {}
    elif case == "overlong": kwargs["max_seq_length"] = 2
    flags, rng = [(m, m.training) for m in model.modules()], rng_state(torch)
    with pytest.raises(ValueError):
        student.score_current_student(model, Tokenizer(), records, mapping, **kwargs)
    assert all(m.training == flag for m, flag in flags)
    assert_rng_equal(torch, rng, rng_state(torch))
