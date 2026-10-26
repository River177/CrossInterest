from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments import read_oi_ni_meta as reader


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, identity: str) -> None:
    _dump(root / "manifest.json", {"identity": identity, "files": {
        p.relative_to(root).as_posix(): _hash(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest.json"}})


def _backend_audit() -> dict:
    return {"version": "qwen35-exact-meta-torch-v1", "transformers_version": "5.14.1",
            "torch_version": "2.8.0+cu126", "attention": "eager",
            "gated_delta": "transformers.torch_chunk_gated_delta_rule",
            "convolution": "torch.nn.Conv1d + torch.nn.functional.silu",
            "gated_norm": "transformers.Qwen3_5RMSNormGated",
            "linear_attention_modules": ["base_model.model.layers.0.linear_attn"],
            "full_attention_modules": ["base_model.model.layers.1.self_attn"],
            "replaced_norm_modules": [],
            "unregistered_tied_alias_modules": ["base_model.model.lm_head.token_adapter.trainable_tokens_delta",
                                                "base_model.model.lm_head.token_adapter.trainable_tokens_original"],
            "tied_state_alias_policy": "one registered input owner; tied output holds the same objects by reference",
            "checkpointing_was_enabled": False, "internal_gradient_checkpointing": False,
            "use_cache": False, "projection_dtype": "float32",
            "dropout_policy": "unchanged; caller must control stochasticity for derivative comparisons"}


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 1000 + int(re.fullmatch(r"<interest_(\d+)>", token).group(1))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + messages[0]["content"] + ":ASSISTANT:"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if isinstance(text, list):
            return {"input_ids": [self(item, add_special_tokens=False)["input_ids"] for item in text]}
        if text.startswith("<interest_"):
            return {"input_ids": [self.convert_tokens_to_ids(text)]}
        return {"input_ids": list(text.encode())}


def _known_metrics(kind: str, identity: str) -> dict:
    """Analytic answers for ascending-index rankings, not production metric code."""
    d2, d4 = 1 / math.log2(3), 1 / math.log2(5)
    if kind == "validation":
        n, support = 2, {"C0": 1, "C1": 1}
        discounts = {"C0": 1.0, "C1": d2}
        recall1, hit1, ndcg1 = 0.5, 0.5, 0.5
        ndcg_all, mrr = (1 + d2) / 2, 0.75
    else:
        n, support = 3, {"C0": 1, "C1": 1, "C2": 2, "C3": 1}
        discounts = {"C0": 1.0, "C1": d2, "C2": 0.5, "C3": d4}
        recall1, hit1, ndcg1 = 1 / 6, 1 / 3, 1 / 3
        ndcg_all = ((1.5 / (1 + d2)) + d2 + ((0.5 + d4) / (1 + d2))) / 3
        mrr = (1 + 0.5 + 1 / 3) / 3
    result = {"identity": identity, "samples": n, "supported_interest_count": len(support),
              "per_class_support": support, "per_class_recall": {}, "per_class_ndcg": {},
              "top1_label_counts": {"C0": n}, "mrr_at_20": mrr,
              "no_test_time_adaptation": True, "history_filtering": False,
              "projection_dtype": "float32", "tie_break": "interest_index_ascending"}
    for k in (1, 5, 10, 20):
        recalls = {c: float(c == "C0") if k == 1 else 1.0 for c in support}
        gains = {c: float(c == "C0") if k == 1 else discounts[c] for c in support}
        result["per_class_recall"][str(k)] = recalls
        result["per_class_ndcg"][str(k)] = gains
        result[f"interest_macro_recall_at_{k}"] = sum(recalls.values()) / len(support)
        result[f"interest_macro_ndcg_at_{k}"] = sum(gains.values()) / len(support)
        result[f"recall_at_{k}"] = recall1 if k == 1 else 1.0
        result[f"hit_at_{k}"] = hit1 if k == 1 else 1.0
        result[f"ndcg_at_{k}"] = ndcg1 if k == 1 else ndcg_all
    return result


@pytest.fixture
def synthetic_run(tmp_path: Path, request):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    root, data = tmp_path / "downloaded", tmp_path / "data"
    checkpoint = root / "checkpoints" / "checkpoint-000002"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    data.mkdir()
    scope = getattr(request, "param", "pilot")
    tokenizer = Tokenizer()
    mapping = [{"interest_index": i, "interest_token": f"<interest_{i:03d}>", "target_subcategory": f"C{i}"}
               for i in range(108)]
    pq.write_table(pa.Table.from_pylist(mapping), data / "interest_token_mapping.parquet")
    validation = [{"user_id": f"V{i}", "impression_id": f"v{i}", "prompt": f"VH{i}",
                   "target_subcategory": f"C{i}"} for i in range(2)]
    dev = [{"user_id": f"D{i}", "impression_id": f"d{i}", "prompt": f"DH{i}",
            "acceptable_subcategories": targets} for i, targets in enumerate([["C0", "C2"], ["C1"], ["C3", "C2"]])]
    pq.write_table(pa.Table.from_pylist(validation), data / "validation_ni.parquet")
    pq.write_table(pa.Table.from_pylist(dev), data / "dev_full_grouped.parquet")
    for name in ("fit_oi", "fit_ni"):
        pq.write_table(pa.Table.from_pylist([{"user_id": name, "prompt": f"{name} history"}]), data / f"{name}.parquet")
    _dump(data / "metadata.json", {"artifacts": {p.name: {"sha256": _hash(p)} for p in data.glob("*.parquet")}})
    config = {"semantic_id_token_count": 108, "sid_projection_dtype": "float32",
              "history_prediction_filtering": False, "enable_thinking": False,
              "ranking_tie_break": "ascending_frozen_interest_index", "fresh_initialization": True,
              "pilot_steps": 2, "gate_steps": 2, "calibration_steps": 2,
              "gate_alpha": 0.001, "inner_parameters": "lora_A_and_lora_B_only",
              "outer_parameters": "lora_and_trainable_sid_rows", "world_size": 1,
              "meta_gradient": "exact_recomputed_hessian_vector_product", "model_dtype": "bfloat16",
              "dropout_seed_rule": "seed + 104729 * episode_index",
              "max_compute_multiplier": 5, "lora_r": 2, "max_seq_length": 256,
              "data_metadata_sha256": _hash(data / "metadata.json")}
    settings = {"config": config, "phase": scope, "steps": 2, "mode": "meta", "alpha": 0.001,
                "source_identity": "frozen-source", "eval_dev": scope == "pilot", "selection_sha256": "selection"}
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    _dump(root / "resolved_config.json", {"identity": identity, **settings})
    runtime = {
        "causal_lm.base_model.model.layers.0.q_proj.lora_A.default.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "causal_lm.base_model.model.layers.0.q_proj.lora_B.default.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "causal_lm.base_model.model.embed_tokens.token_adapter.trainable_tokens_delta.default": torch.ones(108, 4),
    }
    saved = {}
    for name, tensor in runtime.items():
        canonical = name.removeprefix("causal_lm.").replace(".default.", ".")
        if canonical.endswith(".default"):
            canonical = canonical[:-8]
        saved[canonical] = tensor
    save_file(saved, str(adapter / "adapter_model.safetensors"))
    _dump(adapter / "adapter_config.json", {"peft_type": "LORA", "r": 2})
    _dump(adapter / "tokenizer_config.json", {"fixture": True})
    digest = hashlib.sha256()
    for name, tensor in runtime.items():
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    deployed = digest.hexdigest()
    _dump(root / "model_audit.json", {"backend": _backend_audit(), "memory_bytes": 1_000_000_000,
          "trainable_parameters": {name: list(value.shape) for name, value in runtime.items()},
          "trainable_dtypes": {name: str(value.dtype) for name, value in runtime.items()},
          "initial_trainable_sha256": "different-initial-parameters"})
    sid_name = next(name for name in runtime if "trainable_tokens_delta" in name)
    _dump(root / "sid_gradient_audit.json", {sid_name: {"finite_rows": 108, "nonzero_rows": 108}})
    _dump(root / "fresh_reload_audit.json", {"trainable_sha256": deployed, "max_abs_error": 0.0,
          "trainable_tensors_exact": True, "deployed_outer_parameters": True})
    _dump(checkpoint / "state.json", {"identity": identity, "step": 2, "mode": "meta", "alpha": 0.001})
    state = {"identity": identity, "step": 2, "parameter_names": list(runtime),
             "scheduler": {"last_epoch": 2}, "rng": {"python": None, "numpy": None, "torch": None, "cuda": []},
             "optimizer": {"param_groups": [{"params": list(range(3))}], "state": {
                 i: {"step": torch.tensor(2.0), "exp_avg": torch.zeros_like(t), "exp_avg_sq": torch.ones_like(t)}
                 for i, t in enumerate(runtime.values())}}}
    torch.save(state, checkpoint / "training_state.pt")
    probe_tokens = [list(("USER:" + row["prompt"] + ":ASSISTANT:").encode()) for row in validation]
    _dump(checkpoint / "reload_probe.json", {"input_ids": probe_tokens, "scores": [[0.0] * 108 for _ in validation],
          "batch_size": 1, "backend": _backend_audit()})
    oi_max = len(b"USER:fit_oi history:ASSISTANT:")
    ni_max = len(b"USER:fit_ni history:ASSISTANT:")
    _dump(root / "token_audit.json", {"oi": 1, "ni": 1, "validation": 2,
          "oi_max": oi_max, "ni_max": ni_max, "validation_max": max(map(len, probe_tokens)),
          "over_length": 0, "target_in_input": False})
    if scope == "gate":
        _dump(root / "long_sequence_hvp_audit.json", {"identity": identity,
              "scope": "gradient-only engineering gate; not a method score",
              "oi_tokens": oi_max, "ni_tokens": ni_max, "loss": 1.0, "hvp_norm": 2.0,
              "seconds": 1.0, "peak_gpu_bytes": 100_000_000, "parameter_update_performed": False})
    for kind, records in (("validation", validation), ("dev", dev)):
        if kind == "dev" and scope != "pilot":
            continue
        directory = root / ("validation-000002" if kind == "validation" else "dev")
        directory.mkdir()
        predictions = []
        for i, row in enumerate(records):
            targets = row.get("acceptable_subcategories", [row.get("target_subcategory")])
            ids = list(("USER:" + row["prompt"] + ":ASSISTANT:").encode())
            payload = {"input_ids": ids, "impression_id": row["impression_id"], "user_id": row["user_id"], "targets": targets}
            input_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            predictions.append({"row_index": i, "impression_id": row["impression_id"], "user_id": row["user_id"],
                "acceptable_subcategories": targets, "input_sha256": input_hash,
                "ranked_indices": list(range(20)), "ranked_labels": [f"C{j}" for j in range(20)],
                "ranked_scores": [-math.log(108)] * 20, "all_sid_log_probabilities": [-math.log(108)] * 108})
        shard = directory / "predictions-000000.parquet"
        pq.write_table(pa.Table.from_pylist(predictions), shard)
        _dump(shard.with_suffix(".json"), {"identity": identity, "sha256": _hash(shard), "rows": len(records)})
        _dump(directory / "metrics.json", _known_metrics(kind, identity))
    _dump(root / "success.json", {"identity": identity, "steps": 2, "mode": "meta", "alpha": 0.001,
          "checkpoint": "/remote/machine/path/must/not/be/used", "checkpoint_relative": "checkpoints/checkpoint-000002",
          "deployed_trainable_sha256": deployed, "fresh_reload_max_abs_error": 0.0, "dev_evaluated": scope == "pilot",
          "validation_metrics": _known_metrics("validation", identity)})

    def rebind():
        _manifest(checkpoint, identity)
        _manifest(root, identity)

    def read(**kwargs):
        return reader.read_run(root, data, scope=scope, tokenizer=tokenizer, expected_hidden_size=4,
                              expected_topology={"validation": (2, 2, 2), "dev": (3, 5, 4)}, **kwargs)

    rebind()
    return root, data, checkpoint, identity, tokenizer, read, rebind


def test_independent_full_readback_reconstructs_tensors_and_all_macro_metrics(synthetic_run) -> None:
    root, _, _, _, _, read, _ = synthetic_run
    before = _hash(root / "manifest.json")
    receipt = read()
    assert receipt["status"] == "PASS"
    assert receipt["checkpoint_relative"] == "checkpoints/checkpoint-000002"
    assert receipt["checkpoint"]["sid_shape"] == [108, 4]
    assert receipt["checkpoint"]["tensor_count"] == 3
    assert receipt["checkpoint"]["optimizer_step_min"] == receipt["checkpoint"]["optimizer_step_max"] == 2
    assert receipt["dev"]["target_occurrences"] == 5
    assert receipt["dev"]["supported_interest_count"] == 4
    expected = (1 + 1 / math.log2(3) + 0.5 + 1 / math.log2(5)) / 4
    assert receipt["dev"]["interest_macro_ndcg_at_10"] == pytest.approx(expected)
    assert receipt["new_model_forward_performed"] is False
    assert _hash(root / "manifest.json") == before


def test_per_interest_ndcg_is_not_impression_set_ndcg() -> None:
    records = [{"user_id": "u1", "acceptable_subcategories": ["A", "B"]},
               {"user_id": "u2", "acceptable_subcategories": ["B"]}]
    result = reader.recompute_metrics(records, [[1, 0, 2, 3], [2, 3, 1, 0]], ["A", "B", "C", "D"])
    assert result["per_class_support"] == {"A": 1, "B": 2}
    assert result["ndcg_at_10"] == pytest.approx(0.75)
    assert result["interest_macro_ndcg_at_10"] == pytest.approx((1 / math.log2(3) + 0.75) / 2)


def test_rejects_payload_hash_tampering(synthetic_run) -> None:
    root, _, _, _, _, read, _ = synthetic_run
    (root / "fresh_reload_audit.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        read()


@pytest.mark.parametrize("fault", ["input_hash", "user", "ranking", "nonfinite", "unnormalized", "missing_row"])
def test_independent_prediction_validation_rejects_self_consistent_bad_shards(synthetic_run, fault: str) -> None:
    root, _, _, identity, _, read, rebind = synthetic_run
    path = root / "dev" / "predictions-000000.parquet"
    rows = pq.read_table(path).to_pylist()
    if fault == "input_hash":
        rows[0]["input_sha256"] = "wrong"
    elif fault == "user":
        rows[0]["user_id"] = "wrong"
    elif fault == "ranking":
        rows[0]["ranked_indices"][0:2] = [1, 0]
    elif fault == "nonfinite":
        rows[0]["all_sid_log_probabilities"][0] = float("nan")
    elif fault == "unnormalized":
        rows[0]["all_sid_log_probabilities"] = [-1.0] * 108
    else:
        rows.pop()
    pq.write_table(pa.Table.from_pylist(rows), path)
    _dump(path.with_suffix(".json"), {"identity": identity, "sha256": _hash(path), "rows": len(rows)})
    rebind()
    with pytest.raises(ValueError):
        read()


def test_rejects_published_metric_even_when_artifact_hashes_match(synthetic_run) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    path = root / "dev" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["interest_macro_recall_at_10"] = 0.95
    _dump(path, metrics)
    rebind()
    with pytest.raises(ValueError, match="Metric recomputation"):
        read()


@pytest.mark.parametrize("fault", ["optimizer_step", "optimizer_name_order", "scheduler_step", "moment_shape"])
def test_rejects_training_state_cursor_or_binding_errors(synthetic_run, fault: str) -> None:
    import torch
    _, _, checkpoint, _, _, read, rebind = synthetic_run
    path = checkpoint / "training_state.pt"
    state = torch.load(path, weights_only=False)
    if fault == "optimizer_step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(1.0)
    elif fault == "optimizer_name_order":
        state["parameter_names"].reverse()
    elif fault == "scheduler_step":
        state["scheduler"]["last_epoch"] = 1
    else:
        state["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    torch.save(state, path)
    rebind()
    with pytest.raises(ValueError):
        read()


def test_rejects_saved_tensor_content_changed_after_reload_claim(synthetic_run) -> None:
    from safetensors.torch import load_file, save_file
    _, _, checkpoint, _, _, read, rebind = synthetic_run
    path = checkpoint / "adapter" / "adapter_model.safetensors"
    saved = load_file(str(path))
    saved[next(iter(saved))] += 1
    save_file(saved, str(path))
    rebind()
    with pytest.raises(ValueError, match="deployment tensor digest"):
        read()


def test_rejects_full_embedding_or_extra_sid_rows(synthetic_run) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    _, _, checkpoint, _, _, read, rebind = synthetic_run
    path = checkpoint / "adapter" / "adapter_model.safetensors"
    saved = load_file(str(path))
    saved["base_model.model.lm_head.weight"] = torch.zeros(1000, 4)
    save_file(saved, str(path))
    rebind()
    with pytest.raises(ValueError, match="bind exactly"):
        read()


def test_rejects_checkpoint_remote_path_substitution(synthetic_run) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    path = root / "success.json"
    success = json.loads(path.read_text())
    success["checkpoint_relative"] = "../../remote-checkpoint"
    _dump(path, success)
    rebind()
    with pytest.raises(ValueError, match="escapes"):
        read()


def test_cli_never_writes_a_receipt_into_committed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(sys, "argv", ["read_oi_ni_meta", "--run-root", str(root), "--data-dir", str(tmp_path),
                        "--output", str(root / "receipt.json"), "--scope", "gate"])
    with pytest.raises(ValueError, match="outside committed run"):
        reader.main()
    assert not (root / "receipt.json").exists()


@pytest.mark.parametrize("synthetic_run", ["gate", "calibrate", "pilot"], indirect=True)
def test_scope_uses_the_correct_phase_budget_and_evaluation_topology(synthetic_run) -> None:
    root, _, _, _, _, read, _ = synthetic_run
    phase = json.loads((root / "resolved_config.json").read_text())["phase"]
    receipt = read()
    assert receipt["scope"] == phase
    assert ("dev" in receipt) is (phase == "pilot")
    assert receipt["validation"]["samples"] == 2


def test_rejects_wrong_sid_row_count_before_accepting_the_adapter(synthetic_run) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    _, _, checkpoint, _, _, read, rebind = synthetic_run
    path = checkpoint / "adapter" / "adapter_model.safetensors"
    saved = load_file(str(path))
    name = next(key for key in saved if "trainable_tokens_delta" in key)
    saved[name] = torch.ones(109, 4)
    save_file(saved, str(path))
    rebind()
    with pytest.raises(ValueError, match="SID replacement shape"):
        read()


def test_rejects_a_success_pointing_to_an_older_than_latest_checkpoint(synthetic_run) -> None:
    root, _, _, identity, _, read, rebind = synthetic_run
    newer = root / "checkpoints" / "checkpoint-000003"
    newer.mkdir()
    _dump(newer / "state.json", {"identity": identity, "step": 3})
    _manifest(newer, identity)
    rebind()
    with pytest.raises(ValueError, match="Latest checkpoint"):
        read()


def test_default_tokenizer_loader_is_local_only(synthetic_run, monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    root, data, checkpoint, _, tokenizer, _, _ = synthetic_run
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    receipt = reader.read_run(root, data, scope="pilot", expected_hidden_size=4,
                              expected_topology={"validation": (2, 2, 2), "dev": (3, 5, 4)})
    assert receipt["status"] == "PASS"
    assert calls == [(str(checkpoint / "adapter"),
                      {"local_files_only": True, "trust_remote_code": False, "cache_dir": None})]


@pytest.mark.parametrize("synthetic_run", ["gate"], indirect=True)
def test_gate_receipt_binds_longest_inputs_sid_gradients_and_exact_backend(synthetic_run) -> None:
    _, _, _, _, _, read, _ = synthetic_run
    receipt = read()
    gate = receipt["long_sequence_hvp_gate"]
    assert gate["oi_tokens"] == len(b"USER:fit_oi history:ASSISTANT:")
    assert gate["ni_tokens"] == len(b"USER:fit_ni history:ASSISTANT:")
    assert gate["independent_token_counts"] == {"oi": 1, "ni": 1, "validation": 2}
    assert gate["hvp_norm"] > 0
    assert gate["parameter_update_performed"] is False
    assert gate["hvp_recomputed_locally"] is False
    assert receipt["sid_gradient_audit"]["finite_rows"] == 108
    assert receipt["sid_gradient_audit"]["nonzero_rows"] == 108
    assert receipt["exact_backend"]["torch_version"] == "2.8.0+cu126"
    assert receipt["exact_backend"]["audit_copies_verified"] == 2


@pytest.mark.parametrize("synthetic_run", ["gate"], indirect=True)
@pytest.mark.parametrize("filename", ["long_sequence_hvp_audit.json", "sid_gradient_audit.json", "token_audit.json"])
def test_gate_rejects_missing_audits_even_if_manifest_is_self_consistent(synthetic_run, filename: str) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    (root / filename).unlink()
    rebind()
    with pytest.raises(ValueError, match="Missing required"):
        read()


@pytest.mark.parametrize("synthetic_run", ["gate"], indirect=True)
@pytest.mark.parametrize("field,value", [
    ("identity", "another-run"), ("scope", "method score"), ("parameter_update_performed", True),
    ("loss", float("nan")), ("hvp_norm", 0.0), ("hvp_norm", -1.0), ("hvp_norm", float("inf")),
    ("seconds", 0.0), ("peak_gpu_bytes", 0), ("peak_gpu_bytes", 2_000_000_000), ("oi_tokens", 1),
])
def test_gate_rejects_invalid_long_sequence_claims(synthetic_run, field: str, value) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    path = root / "long_sequence_hvp_audit.json"
    audit = json.loads(path.read_text())
    audit[field] = value
    _dump(path, audit)
    rebind()
    with pytest.raises(ValueError):
        read()


@pytest.mark.parametrize("synthetic_run", ["gate"], indirect=True)
def test_gate_recomputes_longest_inputs_instead_of_trusting_two_matching_audits(synthetic_run) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    token_path, long_path = root / "token_audit.json", root / "long_sequence_hvp_audit.json"
    token_audit, long_audit = json.loads(token_path.read_text()), json.loads(long_path.read_text())
    token_audit["oi_max"] = long_audit["oi_tokens"] = 1
    _dump(token_path, token_audit)
    _dump(long_path, long_audit)
    rebind()
    with pytest.raises(ValueError, match="differs from frozen oi"):
        read()


@pytest.mark.parametrize("synthetic_run", ["gate"], indirect=True)
@pytest.mark.parametrize("field,value", [("validation", 1), ("validation_max", 1), ("over_length", 1), ("target_in_input", True)])
def test_gate_rejects_incomplete_or_unsafe_token_audit(synthetic_run, field: str, value) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    path = root / "token_audit.json"
    audit = json.loads(path.read_text())
    audit[field] = value
    _dump(path, audit)
    rebind()
    with pytest.raises(ValueError):
        read()


@pytest.mark.parametrize("fault", ["finite_rows", "nonzero_rows", "wrong_owner", "extra_owner"])
def test_sid_gradient_audit_must_bind_exactly_all_canonical_rows(synthetic_run, fault: str) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    path = root / "sid_gradient_audit.json"
    audit = json.loads(path.read_text())
    name = next(iter(audit))
    if fault in {"finite_rows", "nonzero_rows"}:
        audit[name][fault] = 107
    elif fault == "wrong_owner":
        audit[name.replace("embed_tokens", "lm_head")] = audit.pop(name)
    else:
        audit[name.replace("embed_tokens", "lm_head")] = dict(audit[name])
    _dump(path, audit)
    rebind()
    with pytest.raises(ValueError, match="SID gradient audit"):
        read()


@pytest.mark.parametrize("field,value", [
    ("version", "unaudited"), ("transformers_version", "5.15.0"), ("torch_version", "2.8.0+cu121"),
    ("attention", "sdpa"), ("gated_delta", "fused"), ("convolution", "fused_causal_conv1d"),
    ("gated_norm", "FusedRMSNormGated"), ("internal_gradient_checkpointing", True), ("use_cache", True),
    ("tied_state_alias_policy", "registered output owner"), ("dropout_policy", "disabled"),
    ("full_attention_modules", []), ("linear_attention_modules", ["base_model.model.layers.0.linear_attn"] * 2),
    ("full_attention_modules", ["base_model.model.layers.3.self_attn"]),
])
def test_exact_backend_rejects_each_unaudited_execution_setting(field: str, value) -> None:
    audit = _backend_audit()
    audit[field] = value
    with pytest.raises(ValueError, match="(?i)exact backend"):
        reader.inspect_exact_backend(audit, "test")


def test_exact_backend_is_checked_in_checkpoint_probe_not_only_initial_model(synthetic_run) -> None:
    _, _, checkpoint, _, _, read, rebind = synthetic_run
    path = checkpoint / "reload_probe.json"
    probe = json.loads(path.read_text())
    probe["backend"]["attention"] = "sdpa"
    _dump(path, probe)
    rebind()
    with pytest.raises(ValueError, match="Exact backend attention"):
        read()


def test_resume_backend_may_repair_startup_state_but_must_preserve_exact_contract(synthetic_run) -> None:
    root, _, _, _, _, read, rebind = synthetic_run
    attempt = json.loads((root / "model_audit.json").read_text())
    attempt["backend"]["checkpointing_was_enabled"] = True
    attempt["backend"]["unregistered_tied_alias_modules"] = []
    attempt["backend"]["replaced_norm_modules"] = ["base_model.model.layers.0.linear_attn.norm"]
    path = root / "attempts" / "resume.json"
    _dump(path, attempt)
    rebind()
    assert read()["exact_backend"]["audit_copies_verified"] == 3
    attempt["backend"]["gated_delta"] = "fused"
    _dump(path, attempt)
    rebind()
    with pytest.raises(ValueError, match="Exact backend gated_delta"):
        read()


@pytest.mark.parametrize("field,value", [
    ("inner_parameters", "lora_and_sid"), ("outer_parameters", "lora_only"),
    ("meta_gradient", "first_order"), ("model_dtype", "float16"), ("world_size", 2),
    ("world_size", True), ("dropout_seed_rule", "global_rng_only"),
])
def test_exact_meta_config_rejects_derivative_or_parameter_scope_changes(synthetic_run, field: str, value) -> None:
    root = synthetic_run[0]
    config = json.loads((root / "resolved_config.json").read_text())["config"]
    config[field] = value
    with pytest.raises(ValueError, match="exact-meta"):
        reader.inspect_exact_config(config)
