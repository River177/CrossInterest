import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.evaluation.evaluate_beam import canonical_sha256, sha256_file
from new_interest.evaluation.score_rho_reference import (
    LOSS_DEFINITION, VERSION, effective_output_weight, encode_prefix, load_cohorts,
    loss_rows, project_full_vocab, score_batch, source_identity, validate_rows,
    verify_saved_adapter_tensors, audit_native_head,
    kernel_selection_context, kernel_selection_identity, restore_kernel_receipt,
)
from new_interest.experiments.read_rho_reference import verify_reference, verify_twofold, RUNTIME_FIELDS


def record(index, label="a"):
    target = 0 if label == "a" else 1
    return {"source_sample_id": f"s{index}", "user_id": f"u{index}", "target_subcategory": label,
        "messages": [{"role": "user", "content": f"history {index}"},
                     {"role": "assistant", "content": f"<interest_{target:03d}>"}]}


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def tiny_model(dtype=None, hybrid=False):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):
        pytest.skip("Qwen3.5 worker dependencies unavailable")
    torch.manual_seed(173)
    layers = ["full_attention", "linear_attention", "full_attention"] if hybrid else ["full_attention"]*2
    config = transformers.Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=len(layers), num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=layers, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = "eager"
    base = transformers.Qwen3_5ForCausalLM(config).to(dtype=dtype or torch.float32)
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", trainable_token_indices=[60, 61, 62, 63]))
    with torch.no_grad():
        rows = model.get_input_embeddings().token_adapter.trainable_tokens_delta["default"]
        rows.copy_(torch.linspace(-1.231, .9789, rows.numel()).reshape_as(rows))
    return model.eval()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_full_projection_includes_native_cast_tied_sid_rows(dtype_name):
    torch = pytest.importorskip("torch")
    model = tiny_model(getattr(torch, dtype_name))
    output = model.get_output_embeddings()
    with torch.inference_mode():
        hidden = torch.randn(2, 32).to(getattr(torch, dtype_name))
        weight = effective_output_weight(model)
        native = output(hidden)
        reconstructed = torch.nn.functional.linear(hidden, weight)
        assert torch.equal(native, reconstructed)
        actual = project_full_vocab(hidden, weight)
        expected = torch.nn.functional.linear(hidden.float(), weight.float())
        assert torch.equal(actual, expected)
        wrong = torch.nn.functional.linear(hidden.float(), output.token_adapter.base_layer.weight.float())
        assert not torch.equal(actual[:, 60:], wrong[:, 60:])
        deltas = output.token_adapter.trainable_tokens_delta["default"]
        assert torch.equal(weight[60:], deltas.to(weight))
        if dtype_name == "bfloat16":
            assert not torch.equal(weight[60:].float(), deltas.float())
        audit = audit_native_head(model, {"input_ids": [2, 3, 7]}, weight.float())
        assert audit["native_head_max_abs_error"] == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_prefill_selects_last_real_hidden_state_with_right_padding(hybrid):
    torch = pytest.importorskip("torch")
    model = tiny_model(hybrid=hybrid)
    prefixes = [{"input_ids": [2, 3, 4]}, {"input_ids": [2, 9, 7, 6, 5]}]
    with torch.inference_mode():
        weight = effective_output_weight(model).float()
        together = score_batch(model, prefixes, 0, weight)
        separate = torch.cat([score_batch(model, [p], 0, weight) for p in prefixes])
    torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-5)


def test_native_adapter_reload_matches_all_saved_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    model = tiny_model()
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    assert verify_saved_adapter_tensors(model, tmp_path)["accepted"]
    with torch.no_grad():
        model.get_input_embeddings().token_adapter.trainable_tokens_delta["default"][0, 0] += .25
    with pytest.raises(ValueError, match="tensor differs"):
        verify_saved_adapter_tensors(model, tmp_path)


def test_loss_statistics_full_denominator_and_ties():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[8., 5., 1., 2., 2.], [-1., 0., 2., 4., 0.]])
    rows = loss_rows(logits, [1, 0], [2, 3, 4])
    assert rows[0]["ranked_interest_indices"] == [1, 2, 0]
    assert rows[0]["target_sid_rank"] == 1
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([3, 2]), reduction="none")
    assert [r["full_vocab_first_sid_nll"] for r in rows] == pytest.approx(expected.tolist())
    for row in rows:
        assert row["full_vocab_first_sid_nll"] > row["sid_conditional_nll"]
    with pytest.raises(ValueError, match="finite FP32"):
        loss_rows(logits.double(), [1, 0], [2, 3, 4])


def test_prefix_uses_shared_sft_boundary_and_rejects_gold_explanation(tmp_path):
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
        "user": 3, "assistant": 4, "history": 5, "0": 6, "a": 7, "b": 8}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% if m['role']=='assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    mapping = tmp_path / "mapping.parquet"
    write_parquet(mapping, [{"interest_index": i, "interest_token": f"<interest_{i:03d}>", "target_subcategory": label} for i, label in enumerate(["a", "b"])])
    spec = register_semantic_interest_tokens(tok, mapping, expected_count=2)
    prefix = encode_prefix(tok, record(0), spec, max_seq_length=32, enable_thinking=False)
    assert prefix["input_ids"] == [3, 5, 6, 4]
    assert prefix["target_token_id"] == spec.token_ids[0]
    bad = record(0)
    bad["messages"][1]["content"] = "because history <interest_000>"
    with pytest.raises(ValueError, match="only the target SID"):
        encode_prefix(tok, bad, spec, max_seq_length=32, enable_thinking=False)
    with pytest.raises(ValueError, match="exceeding"):
        encode_prefix(tok, record(0), spec, max_seq_length=3, enable_thinking=False)


def test_cohorts_reject_user_overlap_even_if_sample_ids_differ(tmp_path):
    train, score = tmp_path/"train.parquet", tmp_path/"score.parquet"
    write_parquet(train, [record(0), record(1)])
    second = record(2)
    second["user_id"] = "u0"
    write_parquet(score, [second])
    with pytest.raises(ValueError, match="user_id overlap"):
        load_cohorts(train, score)
    write_parquet(score, [record(2), record(2)])
    with pytest.raises(ValueError, match="duplicate source_sample_id"):
        load_cohorts(train, score)


def make_scored_fixture(tmp_path, *, train_rows=None, score_rows=None, num_shards=2,
                        train_path=None, score_path=None, mapping_path=None):
    train_rows = train_rows or [record(0), record(1, "b")]
    score_rows = score_rows or [record(2), record(3, "b"), record(4), record(5, "b")]
    train = train_path or tmp_path/"train.parquet"
    source = score_path or tmp_path/"source.parquet"
    mapping = mapping_path or tmp_path/"mapping.parquet"
    write_parquet(train, train_rows)
    write_parquet(source, score_rows)
    if not mapping.exists():
        write_parquet(mapping, [{"interest_index": i, "interest_token": f"<interest_{i:03d}>", "target_subcategory": label} for i, label in enumerate(["a", "b"])])
    adapter = tmp_path/"teacher"/"final_adapter"
    adapter.mkdir(parents=True)
    (adapter/"adapter_model.safetensors").write_bytes(b"fake tensor payload for model-free reader test")
    summary = tmp_path/"teacher"/"run_summary.json"
    write_json(summary, {"input_adapter": None, "model_revision": "fixed", "train_file_sha256": sha256_file(train), "semantic_id_loss_scope": "full_vocabulary"})
    adapter_files = {"adapter_model.safetensors": sha256_file(adapter/"adapter_model.safetensors")}
    identity = {"identity_version": VERSION, "loss_definition": LOSS_DEFINITION,
        "model_name": "Qwen/Qwen3.5-2B", "model_revision": "fixed", "teacher_train_sha256": sha256_file(train),
        "score_file_sha256": sha256_file(source), "mapping_sha256": sha256_file(mapping),
        "adapter_files": adapter_files, "adapter_sha256": canonical_sha256(adapter_files),
        "teacher_summary_sha256": sha256_file(summary), "teacher_rows": len(train_rows), "full_record_count": len(score_rows),
        "labels": ["a", "b"], "tokens": ["<interest_000>", "<interest_001>"], "token_ids": [3, 4],
        "num_shards": num_shards, "checkpoint_size": 2, "max_seq_length": 32}
    identity["evaluation_identity"] = canonical_sha256(identity)
    score_dirs = []
    for shard in range(num_shards):
        directory = tmp_path/f"scores{shard}"
        directory.mkdir()
        score_dirs.append(directory)
        rows = []
        for index in range(shard, len(score_rows), num_shards):
            r = score_rows[index]
            target = ["a", "b"].index(r["target_subcategory"])
            rows.append({"record_index": index, "source_sample_id": r["source_sample_id"], "user_id": r["user_id"],
                "target_subcategory": r["target_subcategory"], "input_sha256": source_identity(r),
                "evaluation_identity": identity["evaluation_identity"], "target_interest_index": target,
                "target_token_id": 3+target, "sid_scores_fp32": [0., 0.], "target_logit_fp32": 0.,
                "full_vocab_logsumexp_fp32": 2., "full_vocab_first_sid_nll": 2., "sid_logsumexp_fp32": 0.6931471805599453,
                "sid_conditional_nll": 0.6931471805599453, "ranked_interest_indices": [0, 1],
                "target_sid_rank": target+1, "full_vocabulary_size": 6, "prefix_tokens": 10})
        checkpoints = {}
        for start in range(0, len(rows), 2):
            name = f"shards/{start:09d}-{min(start+2, len(rows)):09d}.parquet"
            write_parquet(directory/name, rows[start:start+2])
            checkpoints[name] = sha256_file(directory/name)
        write_parquet(directory/"scores.parquet", rows)
        metrics = {"rows": len(rows), "mean_full_vocab_first_sid_nll": 2., "mean_sid_conditional_nll": 0.6931471805599453,
            "class_support": {label: sum(r["target_subcategory"] == label for r in rows) for label in {r["target_subcategory"] for r in rows}},
            "sid_recall": {str(k): sum(r["target_sid_rank"] <= k for r in rows)/len(rows) for k in (1, 5, 10, 20)}}
        write_json(directory/"metrics.json", metrics)
        write_json(directory/"loading_audit.json", {"pretrained_text_loaded": True, "missing_text_keys": [], "unexpected_text_keys": [],
            "adapter_tensor_reload": {"accepted": True}, "head_projection": {"native_head_max_abs_error": 0, "full_vocabulary_size": 6}})
        write_json(directory/"manifest.json", {**identity, "shard_index": shard, "shard_record_count": len(rows),
            "full_vocabulary_size": 6, "scores_sha256": sha256_file(directory/"scores.parquet"),
            "metrics_sha256": sha256_file(directory/"metrics.json"), "loading_audit_sha256": sha256_file(directory/"loading_audit.json"),
            "checkpoint_files": checkpoints, "elapsed_seconds": 1., "restoration": {"adapter_path": str(adapter), "teacher_summary": str(summary)}})
    return {"score_dirs": score_dirs, "teacher_train_file": train, "score_file": source, "mapping_file": mapping}


def test_independent_reader_complete_sharded_union_and_resume_validation(tmp_path):
    args = make_scored_fixture(tmp_path)
    report = verify_reference(**args, output=tmp_path/"accepted.json")
    assert report["accepted"] and report["metrics"]["rows"] == 4
    assert report["metrics"]["sid_recall"]["1"] == .5
    assert report["teacher_score_user_overlap"] == 0
    assert "selected_kernel_configs" not in report
    assert "selected_kernel_configs_sha256" not in report
    manifest = json.loads((args["score_dirs"][0]/"manifest.json").read_text())
    rows = pq.read_table(args["score_dirs"][0]/"scores.parquet").to_pylist()
    records = pq.read_table(args["score_file"]).to_pylist()[::2]
    validate_rows(rows, records, [0, 2], manifest)
    wrong = copy.deepcopy(manifest)
    wrong["evaluation_identity"] = "different teacher"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_rows(rows, records, [0, 2], wrong)
    with pytest.raises(ValueError, match="Missing or duplicate logical shards"):
        verify_reference(**{**args, "score_dirs": args["score_dirs"][:1]})


@pytest.mark.parametrize("mutation", ["nll", "conditional", "rank", "wrong_user", "artifact"])
def test_independent_reader_detects_corrupted_cached_scores(tmp_path, mutation):
    args = make_scored_fixture(tmp_path)
    directory = args["score_dirs"][0]
    rows = pq.read_table(directory/"scores.parquet").to_pylist()
    if mutation == "nll":
        rows[0]["full_vocab_first_sid_nll"] = 1.5
    elif mutation == "conditional":
        rows[0]["sid_conditional_nll"] = .5
    elif mutation == "rank":
        rows[0]["ranked_interest_indices"] = [1, 0]
    elif mutation == "wrong_user":
        rows[0]["user_id"] = "wrong"
    else:
        rows[0]["prefix_tokens"] += 1
    write_parquet(directory/"scores.parquet", rows)
    if mutation != "artifact":
        # Even if an upstream producer republishes internally inconsistent rows
        # with a new hash, the independent arithmetic/source checks reject it.
        manifest = json.loads((directory/"manifest.json").read_text())
        manifest["scores_sha256"] = sha256_file(directory/"scores.parquet")
        write_json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_reference(**args)


def test_twofold_acceptance_covers_original_users_exactly_once(tmp_path):
    first, second = [record(0), record(1, "b")], [record(2), record(3, "b")]
    full = tmp_path/"full.parquet"
    write_parquet(full, first+second)
    a_path, b_path, mapping = tmp_path/"a.parquet", tmp_path/"b.parquet", tmp_path/"mapping.parquet"
    a = make_scored_fixture(tmp_path/"A", train_rows=first, score_rows=second,
        train_path=a_path, score_path=b_path, mapping_path=mapping, num_shards=1)
    b = make_scored_fixture(tmp_path/"B", train_rows=second, score_rows=first,
        train_path=b_path, score_path=a_path, mapping_path=mapping, num_shards=1)
    config = {"full_train_file": full, "mapping_file": mapping, "output": tmp_path/"accepted.json",
              "folds": [{k: v for k, v in fold.items() if k != "mapping_file"} for fold in [a, b]]}
    report = verify_twofold(config)
    assert report["accepted"] and report["twofold_union_exact"] and report["rows"] == 4
    broken = first+second
    broken[0] = {**broken[0], "messages": [{"role": "user", "content": "changed"}, broken[0]["messages"][1]]}
    write_parquet(full, broken)
    with pytest.raises(ValueError, match="changed from the original"):
        verify_twofold(config)


class Candidate:
    def __init__(self, block, timing):
        self.block, self.timing = block, timing

    def all_kwargs(self):
        return {"BLOCK": self.block, "num_warps": 4}

    def __str__(self):
        return str(self.all_kwargs())


class FakeTuner:
    def __init__(self, reverse=False):
        self.configs = [Candidate(1, 5 if reverse else 1), Candidate(2, 1 if reverse else 5)]
        self.cache = {"old": self.configs[0]}
        self.cache_results = True
        self._bench = lambda *args, config, **kwargs: [config.timing]*3

    def run(self):
        if "probe" not in self.cache:
            self.cache["probe"] = min(self.configs, key=lambda c: self._bench(config=c))
        self.best_config = self.cache["probe"]
        return self.best_config


def test_native_kernel_default_never_discovers_or_mutates_tuners(monkeypatch):
    from new_interest.training import probe_kernel_diagnostic as selector
    monkeypatch.setattr(selector, "discover_tuners", lambda: pytest.fail("Native mode must not discover FLA"))
    assert kernel_selection_identity() == kernel_selection_identity("native")
    with kernel_selection_context() as audit:
        audit.capture("native")
        assert audit.receipt()["snapshots"] == []
    with pytest.raises(ValueError, match="Unsupported kernel selection"):
        kernel_selection_identity("unknown")


@pytest.mark.parametrize("failure", [False, True])
def test_stable_kernel_context_restores_all_state_on_exit_and_error(failure):
    tuner = FakeTuner()
    original_cache, original_bench = tuner.cache, tuner._bench
    try:
        with kernel_selection_context("stable_fla_v1", tuners={"fla.test": tuner}) as audit:
            assert tuner.cache == {} and tuner.cache_results is False
            tuner.run()
            audit.capture("native_head_audit")
            receipt = audit.receipt("test-identity")
            assert receipt["snapshots"][0]["kernels"]["fla.test"]["cache"]
            assert receipt["snapshots_sha256"] == canonical_sha256(receipt["snapshots"])
            if failure:
                raise RuntimeError("probe failed")
    except RuntimeError:
        assert failure
    assert tuner.cache is original_cache and tuner._bench is original_bench
    assert tuner.cache_results is True and not hasattr(tuner, "best_config")


def test_stable_selection_controls_tiny_fp32_probe_despite_opposite_timings():
    torch = pytest.importorskip("torch")
    outputs = []
    for reverse in (False, True):
        tuner = FakeTuner(reverse)
        with kernel_selection_context("stable_fla_v1", tuners={"fla.test": tuner}) as audit:
            choice = tuner.run()
            hidden = torch.tensor([[.5, 1., -1.]])
            weight = torch.tensor([[1., 2., 3.], [3., 2., 1.]]) * choice.block
            scores = project_full_vocab(hidden, weight)
            outputs.append(scores)
            audit.capture("controlled_tiny_probe")
    assert torch.equal(outputs[0], outputs[1])


def stable_fixture(args):
    for directory in args["score_dirs"]:
        manifest = json.loads((directory/"manifest.json").read_text())
        manifest.update(kernel_selection_identity("stable_fla_v1"))
        manifest["evaluation_identity"] = canonical_sha256({k: v for k, v in manifest.items() if k not in RUNTIME_FIELDS})
        rows = pq.read_table(directory/"scores.parquet").to_pylist()
        for row in rows:
            row["evaluation_identity"] = manifest["evaluation_identity"]
        write_parquet(directory/"scores.parquet", rows)
        manifest["scores_sha256"] = sha256_file(directory/"scores.parquet")
        stages = ["native_head_audit"]
        for start in range(0, len(rows), manifest["checkpoint_size"]):
            stop = min(start+manifest["checkpoint_size"], len(rows))
            name = f"shards/{start:09d}-{stop:09d}.parquet"
            write_parquet(directory/name, rows[start:stop])
            manifest["checkpoint_files"][name] = sha256_file(directory/name)
            stages.append(f"checkpoint:{start}:{stop}")
        # Different logical shards can visit distinct cache keys; no whole-
        # snapshot equality is required, only consistency of keys they share.
        cache = {"common": "BLOCK=32", f"shard-{manifest['shard_index']}": "BLOCK=64"}
        snapshots = [{"stage": stage, "kernels": {"fla.test": {"cache": cache,
            "cache_results": False, "best_config": "BLOCK=64"}}} for stage in stages]
        receipt = {**kernel_selection_identity("stable_fla_v1"), "evaluation_identity": manifest["evaluation_identity"],
                   "snapshots": snapshots, "snapshots_sha256": canonical_sha256(snapshots)}
        write_json(directory/"kernel_selection.json", receipt)
        manifest["kernel_selection_sha256"] = sha256_file(directory/"kernel_selection.json")
        write_json(directory/"manifest.json", manifest)
    return args


def test_reader_accepts_stable_selector_and_different_shard_cache_keys(tmp_path):
    args = stable_fixture(make_scored_fixture(tmp_path))
    report = verify_reference(**args)
    assert report["accepted"] and report["kernel_selection"] == "stable_fla_v1"
    assert all(a["kernel_selection_receipt"]["snapshot_count"] == 2 for a in report["artifacts"])
    assert report["selected_kernel_configs"] == [
        {"kernel": "fla.test", "cache_key": "common", "config": "BLOCK=32"},
        {"kernel": "fla.test", "cache_key": "shard-0", "config": "BLOCK=64"},
        {"kernel": "fla.test", "cache_key": "shard-1", "config": "BLOCK=64"},
    ]
    assert report["selected_kernel_configs_sha256"] == canonical_sha256(report["selected_kernel_configs"])


@pytest.mark.parametrize("mutation", ["hash", "identity", "selector", "same_key_different_config", "missing_checkpoint"])
def test_reader_rejects_stable_kernel_receipt_corruption(tmp_path, mutation):
    args = stable_fixture(make_scored_fixture(tmp_path))
    directory = args["score_dirs"][-1]
    manifest = json.loads((directory/"manifest.json").read_text())
    receipt = json.loads((directory/"kernel_selection.json").read_text())
    if mutation == "identity":
        receipt["evaluation_identity"] = "wrong"
    elif mutation == "selector":
        receipt["kernel_selector_sha256"] = "0"*64
    elif mutation == "same_key_different_config":
        receipt["snapshots"][-1]["kernels"]["fla.test"]["cache"]["common"] = "BLOCK=128"
    elif mutation == "missing_checkpoint":
        receipt["snapshots"] = receipt["snapshots"][:1]
    else:
        receipt["snapshots"][0]["stage"] = "changed"
    receipt["snapshots_sha256"] = canonical_sha256(receipt["snapshots"])
    write_json(directory/"kernel_selection.json", receipt)
    if mutation != "hash":
        manifest["kernel_selection_sha256"] = sha256_file(directory/"kernel_selection.json")
        write_json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError, match="Kernel|kernel"):
        verify_reference(**args)


def test_kernel_resume_rejects_different_evaluation_identity(tmp_path):
    tuner = FakeTuner()
    path = tmp_path/"kernel_selection.json"
    with kernel_selection_context("stable_fla_v1", tuners={"fla.test": tuner}) as audit:
        tuner.run()
        audit.capture("checkpoint:0:1")
        write_json(path, audit.receipt("original"))
        with pytest.raises(ValueError, match="resume receipt identity"):
            restore_kernel_receipt(audit, path, "different")


def test_stable_capture_rejects_lazily_discovered_uncontrolled_tuner_and_restores(monkeypatch):
    from new_interest.training import probe_kernel_diagnostic as selector
    original, late = FakeTuner(), FakeTuner()
    discovered = {"fla.original": original}
    monkeypatch.setattr(selector, "discover_tuners", lambda: discovered)
    original_cache, original_bench = original.cache, original._bench
    with pytest.raises(ValueError, match="uncontrolled newly discovered.*fla.late"):
        with kernel_selection_context("stable_fla_v1") as audit:
            original.run()
            discovered["fla.late"] = late
            audit.capture("first_forward")
    assert original.cache is original_cache and original._bench is original_bench
    assert original.cache_results is True


def test_stable_coverage_compares_object_ids_not_discovery_aliases(monkeypatch):
    from new_interest.training import probe_kernel_diagnostic as selector
    tuner = FakeTuner()
    discovered = {"fla.original": tuner}
    monkeypatch.setattr(selector, "discover_tuners", lambda: dict(discovered))
    with kernel_selection_context("stable_fla_v1") as audit:
        tuner.run()
        discovered.clear()
        discovered["fla.new_alias"] = tuner
        audit.capture("same_object_new_alias")
        assert len(audit.snapshots) == 1


def test_stable_receipt_rechecks_coverage_after_last_capture(monkeypatch):
    from new_interest.training import probe_kernel_diagnostic as selector
    tuner = FakeTuner()
    discovered = {"fla.original": tuner}
    monkeypatch.setattr(selector, "discover_tuners", lambda: dict(discovered))
    with kernel_selection_context("stable_fla_v1") as audit:
        tuner.run()
        audit.capture("native_head_audit")
        discovered["fla.late"] = FakeTuner()
        with pytest.raises(ValueError, match="uncontrolled newly discovered"):
            audit.receipt()
