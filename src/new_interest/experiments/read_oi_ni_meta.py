"""Independent, CPU-only readback of downloaded OI/NI meta pilot artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


CUTOFFS = (1, 5, 10, 20)
EXACT_BACKEND_CONTRACT = {
    "version": "qwen35-exact-meta-torch-v1",
    "transformers_version": "5.14.1",
    "torch_version": "2.8.0+cu126",
    "attention": "eager",
    "gated_delta": "transformers.torch_chunk_gated_delta_rule",
    "convolution": "torch.nn.Conv1d + torch.nn.functional.silu",
    "gated_norm": "transformers.Qwen3_5RMSNormGated",
    "tied_state_alias_policy": "one registered input owner; tied output holds the same objects by reference",
    "internal_gradient_checkpointing": False,
    "use_cache": False,
    "projection_dtype": "float32",
    "dropout_policy": "unchanged; caller must control stochasticity for derivative comparisons",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def invalid(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _inside(root: Path, relative: str) -> Path:
    _require(not Path(relative).is_absolute(), "Artifact path must be relative")
    result = (root / relative).resolve()
    _require(result.is_relative_to(root.resolve()), "Artifact path escapes root")
    return result


def verify_directory(root: Path, identity: str | None = None) -> dict[str, Any]:
    """Rebuild a complete inventory independently of the worker implementation."""
    manifest = _json(root / "manifest.json")
    if identity is not None:
        _require(manifest["identity"] == identity, "Manifest identity mismatch")
    expected = manifest["files"]
    actual: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        # Worker manifests intentionally omit nested manifests. Verify each of
        # those separately; reject unfinished temporary files in a final run.
        _require(item.resolve().is_relative_to(root.resolve()), "Artifact symlink escapes root")
        _require(".tmp" not in item.name, "Unfinished temporary artifact in committed run")
        if item.name == "manifest.json":
            continue
        actual[item.relative_to(root).as_posix()] = sha256(item)
    _require(set(actual) == set(expected), "Manifest inventory mismatch")
    for relative, digest in expected.items():
        _inside(root, relative)
        _require(actual[relative] == digest, f"Artifact hash mismatch: {relative}")
    return manifest


def _integer(value: Any, name: str) -> int:
    number = float(value)
    _require(not isinstance(value, bool) and math.isfinite(number) and number.is_integer(),
             f"Invalid integer {name}")
    return int(number)


def _canonical_name(name: str) -> str:
    if name.startswith("causal_lm."):
        name = name[len("causal_lm."):]
    name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
    if name.endswith(".trainable_tokens_delta.default"):
        name = name[:-len(".default")]
    return name


def inspect_exact_backend(audit: dict[str, Any], context: str) -> dict[str, Any]:
    """Check the recorded exact-autograd configuration, not only its head dtype."""
    _require(isinstance(audit, dict), f"Missing exact backend audit: {context}")
    for key, expected in EXACT_BACKEND_CONTRACT.items():
        actual = audit.get(key)
        _require(actual == expected and (not isinstance(expected, bool) or actual is expected),
                 f"Exact backend {key} mismatch: {context}")
    lists: dict[str, list[str]] = {}
    for key in ("linear_attention_modules", "full_attention_modules", "replaced_norm_modules",
                "unregistered_tied_alias_modules"):
        value = audit.get(key)
        _require(isinstance(value, list) and all(isinstance(name, str) and name for name in value)
                 and len(value) == len(set(value)), f"Invalid exact backend {key}: {context}")
        lists[key] = value
    linear, full = lists["linear_attention_modules"], lists["full_attention_modules"]
    _require(bool(linear) and bool(full) and set(linear).isdisjoint(full),
             f"Incomplete exact backend attention topology: {context}")
    layer_indices: list[int] = []
    prefixes = set()
    for names, kind in ((linear, "linear_attn"), (full, "self_attn")):
        for name in names:
            match = re.fullmatch(r"(.+\.layers\.)(\d+)\." + kind, name)
            _require(match is not None, f"Unexpected exact backend module name: {context}")
            prefixes.add(match.group(1))
            layer_indices.append(int(match.group(2)))
    _require(len(prefixes) == 1 and sorted(layer_indices) == list(range(len(layer_indices))),
             f"Exact backend has missing/duplicate layer indices: {context}")
    _require(set(lists["replaced_norm_modules"]) <= {f"{name}.norm" for name in linear},
             f"Exact backend norm replacement is outside a gated-delta layer: {context}")
    aliases = lists["unregistered_tied_alias_modules"]
    if aliases:
        alias_parts = [name.rsplit(".", 1) for name in aliases]
        _require(len(alias_parts) == 2 and len({part[0] for part in alias_parts}) == 1
                 and "lm_head" in alias_parts[0][0]
                 and {part[1] for part in alias_parts} == {"trainable_tokens_delta", "trainable_tokens_original"},
                 f"Invalid exact backend tied-state alias audit: {context}")
    _require(type(audit.get("checkpointing_was_enabled")) is bool,
             f"Missing exact backend checkpointing startup audit: {context}")
    # Startup repair actions may differ after resume; the resulting kernels,
    # topology, alias policy and stochasticity contract must remain identical.
    return {**{key: audit[key] for key in EXACT_BACKEND_CONTRACT},
            "linear_attention_modules": linear, "full_attention_modules": full}


def inspect_exact_config(config: dict[str, Any]) -> None:
    expected = {"inner_parameters": "lora_A_and_lora_B_only",
                "outer_parameters": "lora_and_trainable_sid_rows",
                "meta_gradient": "exact_recomputed_hessian_vector_product",
                "model_dtype": "bfloat16", "world_size": 1,
                "dropout_seed_rule": "seed + 104729 * episode_index"}
    _require(all(config.get(key) == value and type(config.get(key)) is type(value)
                 for key, value in expected.items()),
             "Configuration violates the exact-meta derivative/backend contract")


def inspect_sid_gradients(root: Path, model_audit: dict[str, Any], sid_count: int = 108) -> dict[str, Any]:
    path = root / "sid_gradient_audit.json"
    _require(path.is_file(), "Missing required SID gradient audit")
    audit = _json(path)
    expected = [name for name in model_audit["trainable_parameters"] if "trainable_tokens_delta" in name]
    _require(len(expected) == 1 and set(audit) == set(expected), "SID gradient audit parameter binding mismatch")
    row = audit[expected[0]]
    _require(_integer(row.get("finite_rows", -1), "finite SID gradient rows") == sid_count
             and _integer(row.get("nonzero_rows", -1), "nonzero SID gradient rows") == sid_count,
             "SID gradient audit does not cover all 108 finite/nonzero rows")
    return {"parameter_name": expected[0], "finite_rows": sid_count, "nonzero_rows": sid_count,
            "audit_sha256": sha256(path), "gradient_recomputed_locally": False}


def inspect_long_sequence_gate(root: Path, data: Path, config: dict[str, Any], identity: str,
                               model_audit: dict[str, Any], tokenizer: Any,
                               sid_ids: set[int]) -> dict[str, Any]:
    """Bind the recorded HVP gate to independently re-tokenized longest inputs."""
    long_path, token_path = root / "long_sequence_hvp_audit.json", root / "token_audit.json"
    _require(long_path.is_file() and token_path.is_file(), "Missing required long-sequence/token gate audit")
    audit, tokens = _json(long_path), _json(token_path)
    _require(audit.get("identity") == identity, "Long-sequence HVP identity mismatch")
    _require(audit.get("scope") == "gradient-only engineering gate; not a method score"
             and audit.get("parameter_update_performed") is False,
             "Long-sequence HVP must be a gradient-only gate without a parameter update")
    _require(tokens.get("over_length") == 0 and tokens.get("target_in_input") is False,
             "Token gate reports overlength input or target leakage")
    for key in ("loss", "hvp_norm", "seconds"):
        value = audit.get(key)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and (value >= 0 if key == "loss" else value > 0),
                 f"Invalid long-sequence HVP {key}")
    peak = _integer(audit.get("peak_gpu_bytes", -1), "long-sequence peak GPU bytes")
    memory = _integer(model_audit.get("memory_bytes", -1), "recorded GPU memory bytes")
    _require(0 < peak <= memory, "Invalid long-sequence HVP peak memory")
    maxima: dict[str, int] = {}
    counts: dict[str, int] = {}
    for role, filename in (("oi", "fit_oi.parquet"), ("ni", "fit_ni.parquet"),
                           ("validation", "validation_ni.parquet")):
        maximum, count = 0, 0
        for batch in pq.ParquetFile(data / filename).iter_batches(batch_size=256, columns=["prompt"]):
            prompts = batch.column(0).to_pylist()
            rendered = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False) for prompt in prompts]
            encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            _require(len(encoded) == len(prompts), "Gate tokenizer batch coverage mismatch")
            for ids in encoded:
                _require(0 < len(ids) <= config["max_seq_length"] and not sid_ids.intersection(ids),
                         "Independent gate tokenization found overlength/empty input or a SID token")
                maximum = max(maximum, len(ids))
                count += 1
        _require(count > 0 and _integer(tokens.get(role, -1), f"{role} token count") == count
                 and _integer(tokens.get(f"{role}_max", -1), f"{role} max tokens") == maximum,
                 f"Token gate count/maximum differs from frozen {role} inputs")
        maxima[role], counts[role] = maximum, count
    _require(_integer(audit.get("oi_tokens", -1), "HVP OI tokens") == maxima["oi"]
             and _integer(audit.get("ni_tokens", -1), "HVP NI tokens") == maxima["ni"],
             "Long-sequence HVP did not use the longest frozen OI/NI inputs")
    return {"hvp_norm": audit["hvp_norm"], "loss": audit["loss"], "seconds": audit["seconds"],
            "peak_gpu_bytes": peak, "oi_tokens": maxima["oi"], "ni_tokens": maxima["ni"],
            "independent_token_counts": counts, "independent_token_maxima": maxima,
            "audit_sha256": sha256(long_path), "token_audit_sha256": sha256(token_path),
            "parameter_update_performed": False, "hvp_recomputed_locally": False}


def inspect_checkpoint(checkpoint: Path, identity: str, steps: int,
                       model_audit: dict[str, Any], success: dict[str, Any],
                       reload_audit: dict[str, Any], *, hidden_size: int = 2048,
                       sid_count: int = 108, lora_rank: int = 16) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    verify_directory(checkpoint, identity)
    summary = _json(checkpoint / "state.json")
    _require(summary["identity"] == identity and _integer(summary["step"], "checkpoint step") == steps,
             "Checkpoint state identity/step mismatch")
    _require(summary["mode"] == success["mode"] and summary["alpha"] == success["alpha"],
             "Checkpoint mode/alpha mismatch")
    # This is a user-authorized, local, hash-verified training checkpoint. The
    # optimizer/RNG payload contains trusted pickle types beyond weights_only.
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    _require(state["identity"] == identity and _integer(state["step"], "training step") == steps,
             "Training state identity/step mismatch")
    names = state["parameter_names"]
    shapes = model_audit["trainable_parameters"]
    _require(names == list(shapes) and len(names) == len(set(names)), "Optimizer parameter name order mismatch")
    _require(set(state["rng"]) >= {"python", "numpy", "torch", "cuda"}, "Missing resume RNG states")
    optimizer = state["optimizer"]
    parameter_ids = [key for group in optimizer["param_groups"] for key in group["params"]]
    _require(len(parameter_ids) == len(names) == len(set(parameter_ids)), "Optimizer parameter ID binding mismatch")
    _require(set(optimizer["state"]) == set(parameter_ids), "Missing or extra optimizer parameter states")
    _require(_integer(state["scheduler"]["last_epoch"], "scheduler cursor") == steps,
             "Scheduler cursor differs from completed steps")
    tensors: dict[str, Any] = {}
    adapter = checkpoint / "adapter"
    _require(not list(adapter.glob("*.bin")) and not list(adapter.glob("*.pt")), "Unexpected adapter pickle/full weight file")
    safe_files = sorted(adapter.glob("*.safetensors"))
    _require(bool(safe_files), "Missing adapter safetensors")
    for path in safe_files:
        for name, tensor in load_file(str(path), device="cpu").items():
            _require(name not in tensors, "Duplicate safetensor parameter")
            _require(tensor.is_floating_point() and bool(torch.isfinite(tensor).all()), "Nonfinite/nonfloating saved parameter")
            tensors[name] = tensor
    saved = {_canonical_name(name): value for name, value in tensors.items()}
    _require(len(saved) == len(tensors), "Duplicate canonical parameter names")
    canonical_names = [_canonical_name(name) for name in names]
    _require(len(set(canonical_names)) == len(names) and set(canonical_names) == set(saved),
             "Saved tensors do not bind exactly to optimizer/model names")
    sid_names = [name for name in saved if ".trainable_tokens_delta" in name]
    _require(len(sid_names) == 1 and "embed_tokens" in sid_names[0] and "lm_head" not in sid_names[0]
             and sid_names[0].endswith(".trainable_tokens_delta"),
             "Expected one canonical input-embedding SID replacement tensor")
    _require(tuple(saved[sid_names[0]].shape) == (sid_count, hidden_size), "SID replacement shape mismatch")
    for name, tensor in saved.items():
        if name in sid_names:
            continue
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", name)
        _require(match is not None and tensor.ndim == 2, f"Unexpected non-LoRA/full embedding tensor: {name}")
        side = match.group(2)
        _require(tensor.shape[0 if side == "A" else 1] == lora_rank, "LoRA rank differs from config")
        peer = f"{match.group(1)}.lora_{'B' if side == 'A' else 'A'}.weight"
        _require(peer in saved, "Unpaired LoRA factors")
    digest = hashlib.sha256()
    observed_steps = []
    for name, canonical, key in zip(names, canonical_names, parameter_ids):
        tensor = saved[canonical]
        _require(list(tensor.shape) == shapes[name], "Saved/model tensor shape mismatch")
        if "trainable_dtypes" in model_audit:
            _require(set(model_audit["trainable_dtypes"]) == set(names)
                     and model_audit["trainable_dtypes"][name] == str(tensor.dtype),
                     "Saved/model trainable dtype audit mismatch")
        moment = optimizer["state"][key]
        step = _integer(moment["step"], "optimizer per-parameter step")
        _require(step == steps, "Optimizer per-parameter step differs from outer cursor")
        observed_steps.append(step)
        for field in ("exp_avg", "exp_avg_sq"):
            value = moment[field]
            _require(list(value.shape) == list(tensor.shape) and bool(torch.isfinite(value).all()),
                     f"Invalid optimizer {field} shape/values")
        _require(bool((moment["exp_avg_sq"] >= 0).all()), "Negative optimizer second moment")
        digest.update(name.encode())
        digest.update(tensor.float().numpy().tobytes())
    deployed = digest.hexdigest()
    _require(deployed == success["deployed_trainable_sha256"] == reload_audit["trainable_sha256"],
             "Saved deployment tensor digest differs from success/reload audit")
    _require(deployed != model_audit["initial_trainable_sha256"], "No parameter change from initialization")
    _require(reload_audit.get("trainable_tensors_exact") is True
             and reload_audit.get("deployed_outer_parameters") is True
             and reload_audit.get("max_abs_error") == 0.0
             and success.get("fresh_reload_max_abs_error") == 0.0,
             "Fresh reload audit does not certify the deployed outer parameters")
    return {"tensor_count": len(saved), "sid_tensor": sid_names[0],
            "sid_shape": list(saved[sid_names[0]].shape), "optimizer_step_min": min(observed_steps),
            "optimizer_step_max": max(observed_steps), "trainable_sha256": deployed,
            "training_state_sha256": sha256(checkpoint / "training_state.pt")}


def _record(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.setdefault("acceptable_subcategories", [row.get("target_subcategory")])
    if "impression_id" not in result:
        result["impression_id"] = str(result["source_sample_id"])
    return result


def encode_input(tokenizer: Any, row: dict[str, Any], max_length: int,
                 sid_ids: set[int]) -> list[int]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    _require(0 < len(ids) <= max_length, "Empty/overlength evaluation input")
    _require(not sid_ids.intersection(ids), "SID answer token in evaluation input")
    return ids


def input_sha256(row: dict[str, Any], ids: list[int]) -> str:
    payload = {"input_ids": ids, "impression_id": str(row["impression_id"]),
               "user_id": str(row.get("user_id", "")), "targets": list(row["acceptable_subcategories"])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def recompute_metrics(records: list[dict[str, Any]], ranked_indices: list[list[int]],
                      labels: list[str]) -> dict[str, Any]:
    """Count every target occurrence; class NDCG uses one-positive ideal gain."""
    _require(len(records) == len(ranked_indices) and bool(records), "Metric record coverage mismatch")
    support: Counter[str] = Counter()
    hits = {k: Counter() for k in CUTOFFS}
    gains: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in CUTOFFS}
    micro: dict[str, list[float]] = defaultdict(list)
    top1: Counter[str] = Counter()
    for record, indices in zip(records, ranked_indices):
        targets = record["acceptable_subcategories"]
        _require(bool(targets) and len(targets) == len(set(targets)) and set(targets) <= set(labels),
                 "Invalid acceptable-target set")
        support.update(targets)
        ranks = {labels[c]: index + 1 for index, c in enumerate(indices)}
        top1[labels[indices[0]]] += 1
        relevant_ranks = [ranks[c] for c in targets if c in ranks]
        micro["mrr_at_20"].append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)
        for k in CUTOFFS:
            retrieved = [c for c in targets if ranks.get(c, 21) <= k]
            gain = [1.0 / math.log2(ranks[c] + 1) for c in retrieved]
            ideal = math.fsum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(targets)) + 1))
            micro[f"recall_at_{k}"].append(len(retrieved) / len(targets))
            micro[f"hit_at_{k}"].append(float(bool(retrieved)))
            micro[f"ndcg_at_{k}"].append(math.fsum(gain) / ideal)
            hits[k].update(retrieved)
            for c, value in zip(retrieved, gain):
                gains[k][c].append(value)
    per_recall = {str(k): {c: hits[k][c] / n for c, n in sorted(support.items())} for k in CUTOFFS}
    per_ndcg = {str(k): {c: math.fsum(gains[k][c]) / n for c, n in sorted(support.items())} for k in CUTOFFS}
    return {"samples": len(records), "target_occurrences": sum(support.values()),
            "users": len({str(r["user_id"]) for r in records}), "supported_interest_count": len(support),
            "per_class_support": dict(sorted(support.items())), "per_class_recall": per_recall,
            "per_class_ndcg": per_ndcg, "top1_label_counts": dict(top1),
            **{key: math.fsum(values) / len(records) for key, values in micro.items()},
            **{f"interest_macro_recall_at_{k}": math.fsum(per_recall[str(k)].values()) / len(support) for k in CUTOFFS},
            **{f"interest_macro_ndcg_at_{k}": math.fsum(per_ndcg[str(k)].values()) / len(support) for k in CUTOFFS}}


def _compare(published: Any, computed: Any, context: str) -> None:
    if isinstance(computed, dict):
        _require(isinstance(published, dict) and set(published) == set(computed), f"Metric keys mismatch: {context}")
        for key, value in computed.items():
            _compare(published[key], value, f"{context}.{key}")
    elif isinstance(computed, int):
        _require(published == computed, f"Metric count mismatch: {context}")
    else:
        _require(math.isfinite(float(published)) and math.isclose(float(published), computed, rel_tol=1e-10, abs_tol=1e-12),
                 f"Metric recomputation mismatch: {context}")


def inspect_evaluation(directory: Path, source: list[dict[str, Any]], labels: list[str],
                       tokenizer: Any, sid_ids: set[int], max_length: int, identity: str,
                       expected: tuple[int, int, int | None]) -> dict[str, Any]:
    records = [_record(row) for row in source]
    _require(len(records) == expected[0], "Source evaluation sample count mismatch")
    _require(len({str(row["impression_id"]) for row in records}) == len(records), "Duplicate source impression")
    published = _json(directory / "metrics.json")
    _require(published["identity"] == identity and published.get("no_test_time_adaptation") is True
             and published.get("history_filtering") is False and published.get("projection_dtype") == "float32"
             and published.get("tie_break") == "interest_index_ascending", "Evaluation protocol/identity mismatch")
    rank_lists: list[list[int]] = []
    shards = sorted(directory.glob("predictions-*.parquet"))
    _require(bool(shards), "Missing prediction shards")
    offset = 0
    for shard in shards:
        receipt = _json(shard.with_suffix(".json"))
        _require(receipt["identity"] == identity and receipt["sha256"] == sha256(shard), "Prediction receipt mismatch")
        rows = pq.read_table(shard).to_pylist()
        _require(bool(rows) and receipt["rows"] == len(rows), "Prediction shard row count mismatch")
        _require(shard.name == f"predictions-{offset:06d}.parquet", "Prediction shard starts at wrong row")
        for row in rows:
            _require(offset < len(records) and row["row_index"] == offset, "Prediction row coverage mismatch")
            original = records[offset]
            _require(row["user_id"] == str(original["user_id"])
                     and row["impression_id"] == str(original["impression_id"])
                     and row["acceptable_subcategories"] == original["acceptable_subcategories"],
                     "Prediction source user/impression/targets mismatch")
            ids = encode_input(tokenizer, original, max_length, sid_ids)
            _require(row["input_sha256"] == input_sha256(original, ids), "Prediction input token hash mismatch")
            scores = np.asarray(row["all_sid_log_probabilities"], dtype=np.float64)
            _require(scores.shape == (len(labels),) and bool(np.isfinite(scores).all()), "Invalid all-SID score vector")
            _require(np.all(scores <= 1e-6) and math.isclose(float(np.exp(scores).sum()), 1.0, rel_tol=1e-5, abs_tol=1e-6),
                     "All-SID scores are not normalized log probabilities")
            order = np.lexsort((np.arange(len(labels)), -scores))[:20].tolist()
            _require(row["ranked_indices"] == order and row["ranked_labels"] == [labels[i] for i in order]
                     and row["ranked_scores"] == [float(scores[i]) for i in order], "Independent Top20 reconstruction mismatch")
            rank_lists.append(order)
            offset += 1
    _require(offset == len(records), "Incomplete prediction coverage")
    computed = recompute_metrics(records, rank_lists, labels)
    _require(computed["target_occurrences"] == expected[1]
             and (expected[2] is None or computed["supported_interest_count"] == expected[2]), "Evaluation target topology mismatch")
    for key, value in computed.items():
        if key not in {"target_occurrences", "users"}:
            _require(key in published, f"Missing published metric: {key}")
            _compare(published[key], value, key)
    return {"samples": computed["samples"], "target_occurrences": computed["target_occurrences"],
            "users": computed["users"], "supported_interest_count": computed["supported_interest_count"],
            "shards": len(shards), "metrics_sha256": sha256(directory / "metrics.json"),
            **{key: value for key, value in computed.items() if key.startswith("interest_macro_")}}


def read_run(run_root: Path | str, data_dir: Path | str, *, scope: str,
             tokenizer_cache_dir: Path | str | None = None, tokenizer: Any | None = None,
             expected_topology: dict[str, tuple[int, int, int | None]] | None = None,
             expected_hidden_size: int = 2048) -> dict[str, Any]:
    root, data = Path(run_root).resolve(), Path(data_dir).resolve()
    _require(scope in {"gate", "calibrate", "pilot"}, "Unknown readback scope")
    resolved = _json(root / "resolved_config.json")
    identity = resolved.pop("identity")
    _require(hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest() == identity,
             "Resolved configuration identity does not recompute")
    verify_directory(root, identity)
    config = resolved["config"]
    _require(config["semantic_id_token_count"] == 108 and config["sid_projection_dtype"] == "float32"
             and config["history_prediction_filtering"] is False and config["enable_thinking"] is False
             and config["ranking_tie_break"] == "ascending_frozen_interest_index"
             and config["fresh_initialization"] is True, "Configuration violates frozen SID inference protocol")
    inspect_exact_config(config)
    phase = resolved["phase"]
    _require(phase in ({"pilot", "compute_matched"} if scope == "pilot" else {scope}), "Scope/phase mismatch")
    if scope == "gate":
        _require(resolved["mode"] == "meta" and resolved["alpha"] == config.get("gate_alpha"),
                 "Gate must exercise the configured nonzero meta inner step")
        _require(isinstance(resolved["alpha"], (int, float)) and resolved["alpha"] > 0,
                 "Gate requires a positive meta inner step")
    steps = _integer(resolved["steps"], "planned steps")
    planned = config[{"pilot": "pilot_steps", "gate": "gate_steps", "calibrate": "calibration_steps"}[scope]]
    _require(steps > 0 and (steps == planned or (phase == "compute_matched" and resolved["mode"] == "mixed"
             and planned <= steps <= planned * config["max_compute_multiplier"])), "Step budget violates frozen configuration")
    success = _json(root / "success.json")
    _require(success["identity"] == identity and success["steps"] == steps
             and success["mode"] == resolved["mode"] and success["alpha"] == resolved["alpha"], "Success identity/configuration mismatch")
    _require(success.get("dev_evaluated") is (scope == "pilot") and resolved.get("eval_dev") is (scope == "pilot"),
             "Scope/Dev completion flag mismatch")
    checkpoint = _inside(root, success["checkpoint_relative"])
    _require(checkpoint == root / "checkpoints" / f"checkpoint-{steps:06d}", "Final checkpoint relative path/cursor mismatch")
    checkpoint_dirs = sorted((root / "checkpoints").glob("checkpoint-*"))
    _require(bool(checkpoint_dirs) and checkpoint_dirs[-1] == checkpoint, "Latest checkpoint is not the exact completed cursor")
    for folder in checkpoint_dirs:
        verify_directory(folder, identity)
    model_audit, reload_audit = _json(root / "model_audit.json"), _json(root / "fresh_reload_audit.json")
    backend = inspect_exact_backend(model_audit.get("backend"), "initial model")
    backend_copies = 1
    for folder in checkpoint_dirs:
        recorded = _json(folder / "reload_probe.json")
        _require(inspect_exact_backend(recorded.get("backend"), f"checkpoint {folder.name}") == backend,
                 "Checkpoint exact backend differs from initial model")
        backend_copies += 1
    for path in sorted((root / "attempts").glob("*.json")):
        attempt = _json(path)
        _require(inspect_exact_backend(attempt.get("backend"), f"attempt {path.name}") == backend
                 and attempt.get("trainable_parameters") == model_audit["trainable_parameters"],
                 "Resume attempt changed exact backend or trainable parameter topology")
        backend_copies += 1
    checkpoint_receipt = inspect_checkpoint(checkpoint, identity, steps, model_audit, success, reload_audit,
                                            hidden_size=expected_hidden_size, lora_rank=config["lora_r"])
    sid_gradient_receipt = inspect_sid_gradients(root, model_audit)
    metadata = _json(data / "metadata.json")
    _require(sha256(data / "metadata.json") == config["data_metadata_sha256"], "Frozen data metadata identity mismatch")
    for name, item in metadata["artifacts"].items():
        _require(sha256(_inside(data, name)) == item["sha256"], f"Frozen source artifact mismatch: {name}")
    for name in ("validation_ni.parquet", "dev_full_grouped.parquet", "interest_token_mapping.parquet"):
        _require(name in metadata["artifacts"], "Evaluation data missing from frozen manifest")
    mapping = pq.read_table(data / "interest_token_mapping.parquet").to_pylist()
    _require([row["interest_index"] for row in mapping] == list(range(108)), "Mapping is not in frozen SID index order")
    labels = [row["target_subcategory"] for row in mapping]
    _require(len(set(labels)) == 108, "Duplicate target labels")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "adapter"), local_files_only=True,
                    trust_remote_code=False, cache_dir=str(tokenizer_cache_dir) if tokenizer_cache_dir else None)
    sid_ids: set[int] = set()
    for index, row in enumerate(mapping):
        token = row["interest_token"]
        _require(token == f"<interest_{index:03d}>", "SID token spelling differs")
        token_id = tokenizer.convert_tokens_to_ids(token)
        _require(isinstance(token_id, int) and tokenizer(token, add_special_tokens=False)["input_ids"] == [token_id],
                 "SID is not one registered tokenizer token")
        sid_ids.add(token_id)
    _require(len(sid_ids) == 108, "Tokenizer SID IDs are not distinct")
    validation = pq.read_table(data / "validation_ni.parquet").to_pylist()
    val_users = {str(row["user_id"]) for row in validation}
    _require(len(val_users) == len(validation), "Validation is not one target per user")
    fit_users: set[str] = set()
    for name in ("fit_oi.parquet", "fit_ni.parquet"):
        _require(name in metadata["artifacts"], "Fit data missing from frozen identity")
        fit_users.update(pq.read_table(data / name, columns=["user_id"])["user_id"].to_pylist())
    _require(not fit_users.intersection(val_users), "Validation users leaked into fitting data")
    gate_receipt = (inspect_long_sequence_gate(root, data, config, identity, model_audit, tokenizer, sid_ids)
                    if scope == "gate" else None)
    topologies = {"validation": (64, 64, None) if scope == "gate" else (2160, 2160, 108),
                  "dev": (31392, 37728, 104)}
    if expected_topology is not None:
        topologies.update(expected_topology)
    if scope == "gate":
        validation = validation[:topologies["validation"][0]]
    probe = _json(checkpoint / "reload_probe.json")
    expected_probe = [encode_input(tokenizer, _record(row), config["max_seq_length"], sid_ids) for row in validation[:4]]
    _require(probe["input_ids"] == expected_probe and probe["batch_size"] == 1, "Reload probe input provenance mismatch")
    probe_scores = np.asarray(probe["scores"])
    _require(probe_scores.shape == (len(expected_probe), 108) and bool(np.isfinite(probe_scores).all()), "Invalid saved reload probe scores")
    val_directory = root / f"validation-{steps:06d}"
    validation_receipt = inspect_evaluation(val_directory, validation, labels, tokenizer, sid_ids,
                                           config["max_seq_length"], identity, topologies["validation"])
    _require(success["validation_metrics"] == _json(val_directory / "metrics.json"), "Success contains stale validation metrics")
    receipt = {"status": "PASS", "scope": scope, "phase": phase, "identity": identity,
               "mode": success["mode"], "alpha": success["alpha"], "steps": steps,
               "checkpoint_relative": success["checkpoint_relative"],
               "run_manifest_sha256": sha256(root / "manifest.json"),
               "data_metadata_sha256": sha256(data / "metadata.json"),
               "checkpoint": checkpoint_receipt, "validation": validation_receipt,
               "exact_backend": {"version": backend["version"], "torch_version": backend["torch_version"],
                   "transformers_version": backend["transformers_version"], "audit_copies_verified": backend_copies,
                   "linear_attention_layers": len(backend["linear_attention_modules"]),
                   "full_attention_layers": len(backend["full_attention_modules"]),
                   "native_torch_kernels": True, "internal_gradient_checkpointing": False, "use_cache": False},
               "sid_gradient_audit": sid_gradient_receipt,
               "independent_metrics_and_tensor_readback": True, "new_model_forward_performed": False,
               "tokenizer_source": "downloaded adapter; local_files_only",
               "reader_sha256": sha256(Path(__file__))}
    if gate_receipt is not None:
        receipt["long_sequence_hvp_gate"] = gate_receipt
    if scope == "pilot":
        dev = pq.read_table(data / "dev_full_grouped.parquet").to_pylist()
        _require(not {str(row["user_id"]) for row in dev}.intersection(fit_users | val_users), "Dev user leakage")
        receipt["dev"] = inspect_evaluation(root / "dev", dev, labels, tokenizer, sid_ids,
                                            config["max_seq_length"], identity, topologies["dev"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("gate", "calibrate", "pilot"), required=True)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    args = parser.parse_args()
    _require(not args.output.resolve().is_relative_to(args.run_root.resolve()), "Receipt output must be outside committed run root")
    _require(not args.output.exists(), "Refusing to overwrite readback receipt")
    receipt = read_run(args.run_root, args.data_dir, scope=args.scope, tokenizer_cache_dir=args.tokenizer_cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
