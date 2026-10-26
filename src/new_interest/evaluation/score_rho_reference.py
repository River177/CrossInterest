"""Cache cross-fitted first-SID losses without loading evaluation targets into prompts.

The loss is over the complete effective output vocabulary, with the final
projection and logsumexp in FP32. PEFT's replacement SID rows are included after
the same cast used by its native head. The conditional SID loss is diagnostic.
Run this on remote persistent storage; its Parquet output contains predictions.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    _atomic_write_json, _atomic_write_parquet, canonical_sha256, sha256_file,
)
from new_interest.infrastructure.paths import portable_path
from new_interest.training.semantic_id_tokens import (
    load_interest_token_mapping, prepare_semantic_adapter_base,
    register_semantic_interest_tokens,
)
from new_interest.training.sft_data import tokenize_training_example

VERSION = "rho-reference-first-sid-full-effective-vocab-fp32-v1"
LOSS_DEFINITION = "first-supervised-SID; native-effective-head-cast; FP32-projection-and-logsumexp; no-prior-correction"
RECORD_FIELDS = ("source_sample_id", "user_id", "target_subcategory", "messages")
STABLE_KERNEL_PROTOCOL = "fla-valid-candidates-stable-sha256-v1"


def kernel_selection_identity(selection="native"):
    if selection == "native":
        return {"kernel_selection": "native", "kernel_selection_protocol": "native-fla-autotune-v1",
                "kernel_selector_sha256": None}
    if selection != "stable_fla_v1":
        raise ValueError(f"Unsupported kernel selection: {selection}")
    from new_interest.training import probe_kernel_diagnostic as selector
    return {"kernel_selection": selection, "kernel_selection_protocol": STABLE_KERNEL_PROTOCOL,
            "kernel_selector_sha256": sha256_file(Path(selector.__file__))}


class KernelSelectionAudit:
    def __init__(self, identity, tuners, *, check_discovery=False):
        self.identity, self.tuners, self.snapshots = identity, tuners, []
        self.check_discovery = check_discovery
        self.controlled_tuner_ids = frozenset(id(tuner) for tuner in tuners.values())

    def verify_coverage(self):
        """Fail if a forward lazily imported an uncontrolled FLA autotuner.

        Discovery aliases can change after import, so compare object identities,
        not the module/symbol names used to describe them in a snapshot.
        Explicit injected tuner mappings skip discovery for CPU mechanism tests.
        """
        if not self.check_discovery:
            return
        from new_interest.training.probe_kernel_diagnostic import discover_tuners
        missing = sorted(name for name, tuner in discover_tuners().items() if id(tuner) not in self.controlled_tuner_ids)
        if missing:
            raise ValueError("Stable kernel selection encountered uncontrolled newly discovered FLA autotuners: " + ", ".join(missing))

    def capture(self, stage):
        if self.identity["kernel_selection"] == "native":
            return
        self.verify_coverage()
        from new_interest.training.probe_kernel_diagnostic import snapshot
        self.snapshots.append({"stage": str(stage), "kernels": snapshot(self.tuners)})

    def receipt(self, evaluation_identity=None):
        self.verify_coverage()
        return {**self.identity, "evaluation_identity": evaluation_identity,
                "snapshots": self.snapshots, "snapshots_sha256": canonical_sha256(self.snapshots)}


@contextmanager
def kernel_selection_context(selection="native", *, tuners=None):
    """Enter after importing/loading Qwen; keep every compared forward inside.

    ``fixed_selection`` restores all autotuner mutations even on an exception.
    Native mode does not discover/import Triton or mutate any tuner state.
    ``tuners`` is an explicit dependency seam for small CPU tests.
    """
    identity = kernel_selection_identity(selection)
    if selection == "native":
        yield KernelSelectionAudit(identity, {})
        return
    from new_interest.training.probe_kernel_diagnostic import fixed_selection
    with fixed_selection(tuners) as controlled:
        yield KernelSelectionAudit(identity, controlled, check_discovery=tuners is None)


def restore_kernel_receipt(audit, path, evaluation_identity):
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    if (any(previous.get(k) != v for k, v in audit.identity.items())
            or previous.get("evaluation_identity") != evaluation_identity
            or canonical_sha256(previous["snapshots"]) != previous["snapshots_sha256"]):
        raise ValueError("Kernel selection resume receipt identity/hash mismatch")
    audit.snapshots = previous["snapshots"]


def source_identity(row: dict[str, Any]) -> str:
    return canonical_sha256({key: row[key] for key in RECORD_FIELDS})


def load_cohorts(teacher_train_file, score_file):
    """Validate the whole cohorts, before any logical sharding or model load."""
    cohorts = []
    for name, path in (("teacher", teacher_train_file), ("score", score_file)):
        missing = set(RECORD_FIELDS) - set(pq.ParquetFile(path).schema_arrow.names)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        rows = pq.read_table(path, columns=list(RECORD_FIELDS)).to_pylist()
        if not rows or any(row[k] is None for row in rows for k in RECORD_FIELDS):
            raise ValueError(f"{name}: empty cohort or null identity")
        for key in ("source_sample_id", "user_id"):
            if len({str(row[key]) for row in rows}) != len(rows):
                raise ValueError(f"{name}: duplicate {key}; expected one example per user")
        cohorts.append(rows)
    train, score = cohorts
    for key in ("source_sample_id", "user_id"):
        if {str(r[key]) for r in train} & {str(r[key]) for r in score}:
            raise ValueError(f"Teacher/score {key} overlap")
    return train, score


def encode_prefix(tokenizer, row, spec, *, max_seq_length, enable_thinking):
    entries = {e.target_subcategory: e for e in spec.entries}
    entry = entries[str(row["target_subcategory"])]
    messages = row["messages"]
    # Only direct SID supervision is authorized. A gold explanation would leak
    # target information into the reference prefix even if SID itself is sliced.
    if len(messages) != 2 or messages[1]["content"] != entry.interest_token:
        raise ValueError("Reference scoring requires an assistant containing only the target SID")
    encoded = tokenize_training_example(tokenizer, messages,
        max_seq_length=max_seq_length, enable_thinking=enable_thinking)
    sid_ids = set(spec.token_ids)
    positions = [i for i, value in enumerate(encoded["labels"]) if value in sid_ids]
    if len(positions) != 1 or encoded["labels"][positions[0]] != entry.token_id:
        raise ValueError("Missing, duplicate, or mismatched supervised SID")
    position = positions[0]
    if position < 1:
        raise ValueError("SID has no causal prefix")
    if any(value != -100 for value in encoded["labels"][:position]):
        raise ValueError("Unexpected supervised content before target SID")
    return {"input_ids": encoded["input_ids"][:position],
        "target_interest_index": entry.interest_index, "target_token_id": entry.token_id}


def effective_output_weight(model):
    """Read the exact native effective head, including tied PEFT token adapters.

    Do not read ``token_adapter.base_layer.weight`` alone, and do not substitute
    uncased FP32 trainable rows: native PEFT casts its replacement rows to the
    base weight dtype before index_copy. An FP32 projection then lifts these
    effective rows, not a numerically different set of parameters.
    """
    output = model.get_output_embeddings()
    adapter = getattr(output, "token_adapter", None)
    if adapter is not None and (adapter.disable_adapters or adapter.merged
                               or len(adapter.active_adapters) != 1):
        raise ValueError("Expected one active unmerged trainable-token adapter")
    if getattr(output, "bias", None) is not None:
        raise ValueError("Reference projection currently requires a bias-free head")
    if hasattr(output, "lora_A"):
        raise ValueError("Unexpected LoRA output head; effective weight must include its update")
    return output.weight


def project_full_vocab(last_hidden, effective_weight):
    import torch
    from torch.nn import functional as F
    with torch.autocast(device_type=last_hidden.device.type, enabled=False):
        return F.linear(last_hidden.float(), effective_weight.float())


def loss_rows(logits, target_indices, token_ids):
    """Keep sufficient statistics for an independent, model-free readback."""
    import torch
    if logits.dtype != torch.float32 or logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite FP32 full-vocabulary logits")
    ids = torch.tensor(token_ids, device=logits.device)
    scores = logits.index_select(1, ids)
    target = torch.tensor(target_indices, device=logits.device)
    target_logits = scores.gather(1, target[:, None]).squeeze(1)
    full_lse, sid_lse = logits.logsumexp(1), scores.logsumexp(1)
    order = scores.argsort(dim=1, descending=True, stable=True)
    ranks = (order == target[:, None]).nonzero()[:, 1] + 1
    rows = []
    for i in range(len(logits)):
        rows.append({"full_vocab_first_sid_nll": float(full_lse[i] - target_logits[i]),
            "target_logit_fp32": float(target_logits[i]),
            "full_vocab_logsumexp_fp32": float(full_lse[i]),
            "sid_conditional_nll": float(sid_lse[i] - target_logits[i]),
            "sid_logsumexp_fp32": float(sid_lse[i]),
            "sid_scores_fp32": scores[i].tolist(), "target_sid_rank": int(ranks[i]),
            "ranked_interest_indices": order[i, :min(20, len(token_ids))].tolist()})
    return rows


def validate_rows(rows, records, indices, identity):
    if len(rows) != len(records):
        raise ValueError("Checkpoint row count mismatch")
    for row, record, index in zip(rows, records, indices):
        if (row["record_index"] != index or row["source_sample_id"] != str(record["source_sample_id"])
                or row["user_id"] != str(record["user_id"])
                or row["input_sha256"] != source_identity(record)
                or row["evaluation_identity"] != identity["evaluation_identity"]):
            raise ValueError("Checkpoint input/order/evaluation identity mismatch")
        target = identity["labels"].index(str(record["target_subcategory"]))
        if row["target_interest_index"] != target or row["target_token_id"] != identity["token_ids"][target]:
            raise ValueError("Checkpoint target mapping mismatch")
        scores = row["sid_scores_fp32"]
        if len(scores) != len(identity["labels"]) or not all(math.isfinite(float(x)) for x in scores):
            raise ValueError("Checkpoint SID score cardinality/finiteness mismatch")
        for key in ("target_logit_fp32", "full_vocab_logsumexp_fp32", "full_vocab_first_sid_nll",
                    "sid_logsumexp_fp32", "sid_conditional_nll"):
            if not math.isfinite(float(row[key])):
                raise ValueError("Non-finite cached loss")
        if abs(scores[target] - row["target_logit_fp32"]) > 1e-6:
            raise ValueError("Cached target logit differs from SID scores")
        if abs(row["full_vocab_first_sid_nll"] -
               (row["full_vocab_logsumexp_fp32"] - row["target_logit_fp32"])) > 1e-5:
            raise ValueError("Cached full-vocabulary NLL algebra mismatch")
        if row["full_vocab_first_sid_nll"] < -1e-5 or row["full_vocab_first_sid_nll"] + 1e-5 < row["sid_conditional_nll"]:
            raise ValueError("Invalid full/conditional loss ordering")
        order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
        if row["ranked_interest_indices"] != order[:min(20, len(scores))] or row["target_sid_rank"] != order.index(target) + 1:
            raise ValueError("Cached SID ranking mismatch")


def score_batch(model, prefixes, pad_token_id, fp32_weight):
    import torch
    n = max(len(row["input_ids"]) for row in prefixes)
    # Right padding matches SFT. Select the last nonpadding hidden position;
    # taking [:, -1] here silently scores padding for variable-length batches.
    ids = [r["input_ids"] + [pad_token_id] * (n-len(r["input_ids"])) for r in prefixes]
    masks = [[1] * len(r["input_ids"]) + [0] * (n-len(r["input_ids"])) for r in prefixes]
    device = next(model.parameters()).device
    batch = {"input_ids": torch.tensor(ids, device=device), "attention_mask": torch.tensor(masks, device=device)}
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(f"Expected Qwen3_5ForCausalLM, got {type(base).__name__}")
    with torch.inference_mode():
        hidden = base.model(**batch, use_cache=False, return_dict=True).last_hidden_state
        last = torch.tensor([len(r["input_ids"])-1 for r in prefixes], device=device)
        selected = hidden[torch.arange(len(prefixes), device=device), last]
        return project_full_vocab(selected, fp32_weight)


def verify_saved_adapter_tensors(model, adapter):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    saved = load_file(str(Path(adapter) / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(saved) != set(loaded):
        raise ValueError("Fresh adapter reload tensor keys differ")
    for key, expected in saved.items():
        actual = loaded[key].detach().cpu()
        if actual.shape != expected.shape or not torch.equal(actual.float(), expected.float()):
            raise ValueError(f"Fresh adapter reload tensor differs: {key}")
    return {"accepted": True, "tensor_count": len(saved), "comparison": "exact FP32 values per saved tensor"}


def load_model(config, spec, tokenizer):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model_path = str(config.get("model_path", config["model_name"]))
    local = Path(model_path).is_dir()
    model, loading = AutoModelForCausalLM.from_pretrained(model_path,
        revision=None if local else config["model_revision"], dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"), trust_remote_code=False,
        local_files_only=local, output_loading_info=True)
    missing = list(loading.get("missing_keys", []))
    if model.config.tie_word_embeddings and model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
        missing = [key for key in missing if key != "lm_head.weight"]
    unexpected = [key for key in loading.get("unexpected_keys", [])
                  if not key.startswith(("model.visual.", "visual.", "mtp."))]
    if missing or unexpected or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Pretrained text loading mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    if type(model).__name__ != "Qwen3_5ForCausalLM" or not model.config.tie_word_embeddings:
        raise ValueError("Expected tied Qwen3.5 text causal LM")
    if model.config.hidden_size != 2048:
        raise ValueError("Reference teacher is not the expected Qwen3.5-2B configuration")
    if config.get("expected_num_hidden_layers") is not None and model.config.num_hidden_layers != int(config["expected_num_hidden_layers"]):
        raise ValueError("Loaded layer count differs from frozen base configuration")
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision is not None and loaded_revision != config["model_revision"]:
        raise ValueError("Loaded model revision differs from frozen teacher revision")
    initialization = prepare_semantic_adapter_base(model, spec,
        initialization_mode=config.get("semantic_id_initialization_mode", "mean_norm_matched"))
    model = PeftModel.from_pretrained(model, config["adapter_path"], is_trainable=False)
    reload = verify_saved_adapter_tensors(model, config["adapter_path"])
    model.config.use_cache = False
    model = model.to(config.get("device", "cuda:0")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.get_output_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError("Tokenizer and effective output vocabulary differ")
    return model, {"pretrained_text_loaded": True, "missing_text_keys": missing,
        "unexpected_text_keys": unexpected, "adapter_tensor_reload": reload,
        "semantic_initialization": initialization,
        "model_configuration": {"hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "loaded_revision": loaded_revision, "configured_revision": config["model_revision"],
            "attn_implementation": config.get("attn_implementation", "sdpa")}}


def audit_native_head(model, prefix, fp32_weight):
    """Verify native projection includes the same rows; quantify FP32 rounding."""
    import torch
    from torch.nn import functional as F
    captured = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach()))
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            ids = torch.tensor([prefix["input_ids"]], device=device)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False, logits_to_keep=1).logits[:, -1]
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one native output-head call")
    hidden = captured[0][:, -1]
    with torch.inference_mode():
        reconstructed = F.linear(hidden, effective_output_weight(model))
        fp32 = project_full_vocab(hidden, fp32_weight)
    native_error = float((logits.float() - reconstructed.float()).abs().max())
    if native_error != 0:
        raise ValueError(f"Native effective output-head parity failed: {native_error}")
    return {"native_head_max_abs_error": native_error,
        "fp32_vs_native_projection_max_abs_difference": float((logits.float()-fp32).abs().max()),
        "full_vocabulary_size": int(fp32_weight.shape[0]),
        "head_weight_dtype_before_fp32": str(effective_output_weight(model).dtype),
        "fp32_projection_is_scoring_definition": True}


def score(config):
    import torch
    from transformers import AutoTokenizer
    config = dict(config)
    batch_size = int(config.get("batch_size", 1))
    checkpoint_size = int(config.get("checkpoint_size", 256))
    num_shards, shard_index = int(config.get("num_shards", 1)), int(config.get("shard_index", 0))
    if min(batch_size, checkpoint_size, num_shards) < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid batch/checkpoint/logical shard configuration")
    train_file, score_file, mapping, adapter, out = [Path(config[key]) for key in
        ("teacher_train_file", "score_file", "interest_token_mapping_file", "adapter_path", "output_dir")]
    train, records = load_cohorts(train_file, score_file)
    mapping_rows = load_interest_token_mapping(mapping)
    if len(mapping_rows) != int(config.get("semantic_id_token_count", 108)):
        raise ValueError("Unexpected SID vocabulary size")
    model_path = str(config.get("model_path", config["model_name"]))
    local = Path(model_path).is_dir()
    tok = AutoTokenizer.from_pretrained(model_path, revision=None if local else config["model_revision"],
        trust_remote_code=False, local_files_only=local)
    spec = register_semantic_interest_tokens(tok, mapping, expected_count=len(mapping_rows))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    saved_tok = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tok.get_vocab() != saved_tok.get_vocab() or tok.chat_template != saved_tok.chat_template or tok.pad_token_id != saved_tok.pad_token_id:
        raise ValueError("Original-tokenizer registration differs from saved training tokenizer")
    tok.padding_side = "right"
    teacher_summary = config.get("teacher_run_summary")
    if teacher_summary is None and (adapter.parent / "run_summary.json").exists():
        teacher_summary = str(adapter.parent / "run_summary.json")
    if teacher_summary is not None:
        summary = json.loads(Path(teacher_summary).read_text())
        if (summary["train_file_sha256"] != sha256_file(train_file)
                or summary.get("input_adapter") is not None
                or summary["model_revision"] != config["model_revision"]
                or summary.get("semantic_id_loss_scope", "full_vocabulary") != "full_vocabulary"):
            raise ValueError("Teacher training provenance is not fresh-base cross-fitted full-vocabulary SFT")
    else:
        raise ValueError("A completed teacher run_summary.json is required")
    artifact_files = {str(p.relative_to(adapter)): sha256_file(p) for p in sorted(adapter.rglob("*"))
                      if p.is_file() and p.name not in {"README.md", "training_args.bin"}}
    identity = {"identity_version": VERSION, "loss_definition": LOSS_DEFINITION,
        "model_name": config["model_name"], "model_revision": config["model_revision"],
        "teacher_train_sha256": sha256_file(train_file), "score_file_sha256": sha256_file(score_file),
        "mapping_sha256": sha256_file(mapping), "adapter_files": artifact_files,
        "adapter_sha256": canonical_sha256(artifact_files), "teacher_summary_sha256": sha256_file(Path(teacher_summary)),
        "teacher_rows": len(train), "full_record_count": len(records),
        "teacher_user_overlap": 0, "labels": [e.target_subcategory for e in spec.entries],
        "token_ids": spec.token_ids, "tokens": [e.interest_token for e in spec.entries],
        "batch_size": batch_size, "padding_side": "right", "checkpoint_size": checkpoint_size,
        "num_shards": num_shards, "max_seq_length": int(config.get("max_seq_length", 2048)),
        "enable_thinking": bool(config.get("enable_thinking", False)),
        "attn_implementation": config.get("attn_implementation", "sdpa"),
        "model_dtype": "bfloat16", "projection_dtype": "float32", "tf32": False,
        "allow_bf16_reduced_precision_reduction": True,
        "software_versions": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "tokenizers", "safetensors")},
        "cuda_runtime": torch.version.cuda,
        "base_config_sha256": sha256_file(Path(model_path)/"config.json") if local else None,
        "scorer_sha256": sha256_file(Path(__file__)),
        **kernel_selection_identity(config.get("kernel_selection", "native"))}
    identity["evaluation_identity"] = canonical_sha256(identity)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "identity.json"
    shard_identity = {**identity, "shard_index": shard_index}
    if existing.exists() and json.loads(existing.read_text()) != shard_identity:
        raise ValueError("Refusing to resume a different reference-scoring identity")
    _atomic_write_json(existing, shard_identity)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    indices = list(range(shard_index, len(records), num_shards))
    if not indices:
        raise ValueError("Empty logical shard")
    selected = [records[i] for i in indices]
    prefixes = [encode_prefix(tok, row, spec, max_seq_length=identity["max_seq_length"],
                             enable_thinking=identity["enable_thinking"]) for row in selected]
    model, loading = load_model(config, spec, tok)
    with kernel_selection_context(config.get("kernel_selection", "native")) as kernel_audit:
        stable = identity["kernel_selection"] == "stable_fla_v1"
        kernel_path = out / "kernel_selection.json"
        if stable:
            restore_kernel_receipt(kernel_audit, kernel_path, identity["evaluation_identity"])
        with torch.inference_mode():
            weight = effective_output_weight(model).detach().float()
        head_audit = audit_native_head(model, prefixes[0], weight)
        kernel_audit.capture("native_head_audit")
        _atomic_write_json(out / "loading_audit.json", {**loading, "head_projection": head_audit})
        (out / "shards").mkdir(exist_ok=True)
        all_rows, started = [], time.monotonic()
        checkpoint_files = {}
        for start in range(0, len(selected), checkpoint_size):
            stop = min(start + checkpoint_size, len(selected))
            path = out / "shards" / f"{start:09d}-{stop:09d}.parquet"
            kernel_stage = f"checkpoint:{start}:{stop}"
            if path.exists():
                if stable and not any(s["stage"] == kernel_stage for s in kernel_audit.snapshots):
                    raise ValueError("Cached scores have no committed kernel-selection snapshot")
                rows = pq.read_table(path).to_pylist()
            else:
                rows = []
                for offset in range(start, stop, batch_size):
                    batch = prefixes[offset:min(offset+batch_size, stop)]
                    logits = score_batch(model, batch, tok.pad_token_id, weight)
                    rows.extend(loss_rows(logits, [p["target_interest_index"] for p in batch], spec.token_ids))
                if stable:
                    kernel_audit.capture(kernel_stage)
                    _atomic_write_json(kernel_path, kernel_audit.receipt(identity["evaluation_identity"]))
                for row, record, prefix, index in zip(rows, selected[start:stop], prefixes[start:stop], indices[start:stop]):
                    row.update(record_index=index, source_sample_id=str(record["source_sample_id"]),
                        user_id=str(record["user_id"]), target_subcategory=str(record["target_subcategory"]),
                        input_sha256=source_identity(record), evaluation_identity=identity["evaluation_identity"],
                        target_interest_index=prefix["target_interest_index"], target_token_id=prefix["target_token_id"],
                        prefix_tokens=len(prefix["input_ids"]), full_vocabulary_size=head_audit["full_vocabulary_size"])
            validate_rows(rows, selected[start:stop], indices[start:stop], identity)
            if not path.exists():
                _atomic_write_parquet(path, rows)
            checkpoint_files[str(path.relative_to(out))] = sha256_file(path)
            all_rows.extend(rows)
            print("RHO_REFERENCE_SCORE " + json.dumps({"rows": len(all_rows), "total": len(selected),
                "shard_index": shard_index, "elapsed_seconds": time.monotonic()-started}), flush=True)
        expected_paths = {out / p for p in checkpoint_files}
        if set((out / "shards").glob("*.parquet")) != expected_paths:
            raise ValueError("Unrecognized resume shard files")
        metrics = {"rows": len(all_rows), "mean_full_vocab_first_sid_nll": sum(r["full_vocab_first_sid_nll"] for r in all_rows)/len(all_rows),
            "mean_sid_conditional_nll": sum(r["sid_conditional_nll"] for r in all_rows)/len(all_rows),
            "class_support": dict(sorted(Counter(r["target_subcategory"] for r in all_rows).items())),
            "sid_recall": {str(k): sum(r["target_sid_rank"] <= k for r in all_rows)/len(all_rows) for k in (1, 5, 10, 20)}}
        _atomic_write_parquet(out / "scores.parquet", all_rows)
        _atomic_write_json(out / "metrics.json", metrics)
        _atomic_write_json(out / "resolved_config.json", config)
        if stable:
            _atomic_write_json(kernel_path, kernel_audit.receipt(identity["evaluation_identity"]))
        _atomic_write_json(out / "manifest.json", {**shard_identity, "shard_record_count": len(all_rows),
            "full_vocabulary_size": head_audit["full_vocabulary_size"],
            "scores_sha256": sha256_file(out / "scores.parquet"), "metrics_sha256": sha256_file(out / "metrics.json"),
            "loading_audit_sha256": sha256_file(out / "loading_audit.json"), "checkpoint_files": checkpoint_files,
            **({"kernel_selection_sha256": sha256_file(kernel_path)} if stable else {}),
            "elapsed_seconds": time.monotonic()-started,
            "restoration": {"adapter_path": portable_path(adapter), "teacher_summary": portable_path(teacher_summary),
                            "teacher_train_file": portable_path(train_file), "score_file": portable_path(score_file),
                            "mapping_file": portable_path(mapping)}})
        return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    for flag in ("adapter-path", "teacher-train-file", "score-file", "interest-token-mapping-file",
                 "output-dir", "model-name", "model-path", "model-revision", "teacher-run-summary", "device"):
        parser.add_argument("--" + flag)
    for flag in ("batch-size", "checkpoint-size", "num-shards", "shard-index", "max-seq-length"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"))
    parser.add_argument("--kernel-selection", choices=("native", "stable_fla_v1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    print(json.dumps(score(config), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
