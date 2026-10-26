"""Frozen, resumable single-GPU OI/NI exact-meta pilot and readback.

Only a functional LoRA step is virtual. Durable checkpoints always contain the
outer/deployed model, optimizer, schedule, and RNG cursor. Evaluation has no
test-time adaptation. All arms share the FP32 108-SID projection backend.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.training.semantic_id_tokens import (
    initialize_semantic_interest_token_rows, load_interest_token_mapping,
    prepare_semantic_adapter_base, register_semantic_interest_tokens,
    sha256_file, trainable_token_indices, validate_saved_interest_tokens,
)
from new_interest.training.sft_data import apply_chat_template


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def log(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def inventory(directory: Path, *, exclude: tuple[str, ...] = ("manifest.json",)) -> dict[str, str]:
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in exclude and ".tmp-" not in p.name}


def verify_manifest(directory: Path, identity: str | None = None) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if identity is not None and manifest["identity"] != identity:
        raise ValueError(f"Committed identity differs: {directory}")
    for relative, digest in manifest["files"].items():
        if not (directory / relative).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Manifest path escapes artifact")
        if sha256_file(directory / relative) != digest:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    if inventory(directory) != manifest["files"]:
        raise ValueError("Artifact inventory has unbound or missing files")
    return manifest


def publish(source: Path, destination: Path, identity: str) -> None:
    if (destination / "manifest.json").exists():
        committed = verify_manifest(destination, identity)
        if committed["files"] != inventory(source):
            raise ValueError(f"Refusing replacement of committed artifact: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    files = inventory(source)
    extras = set(inventory(destination)).difference(files)
    if extras:
        raise ValueError(f"Unbound files in incomplete destination: {sorted(extras)}")
    for relative, digest in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        shutil.copyfile(source / relative, temporary)
        if sha256_file(temporary) != digest:
            raise ValueError("Copy readback failed")
        os.replace(temporary, target)
    write_json(destination / "manifest.json", {"identity": identity, "files": files})


def seed_everything(seed: int) -> None:
    import torch
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting CUDA")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state() -> dict[str, Any]:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def prepare_snapshot(config: dict[str, Any], output: Path) -> str:
    from huggingface_hub import snapshot_download
    try:
        folder = snapshot_download(config["model_name"], revision=config["model_revision"],
                                   cache_dir=config["hf_cache_dir"], local_files_only=True)
    except Exception as error:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise
        log("snapshot_cache_miss", error_type=type(error).__name__)
        folder = snapshot_download(config["model_name"], revision=config["model_revision"],
                                   cache_dir=config["hf_cache_dir"],
                                   allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
    path = Path(folder)
    if sha256_file(path / "config.json") != config["model_config_sha256"]:
        raise ValueError("Pinned model config does not match")
    required = [p for p in path.glob("*.safetensors")]
    if not required:
        raise ValueError("Missing pretrained weight shards")
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        names = set(json.loads(index_file.read_text())["weight_map"].values())
        if not all((path / name).is_file() for name in names):
            raise ValueError("Incomplete model weight index")
    receipt = {"model_name": config["model_name"], "revision": config["model_revision"],
               "files": {p.name: sha256_file(p) for p in sorted(path.iterdir())
                         if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".jinja", ".model"}}}
    write_json(output / "snapshot_audit.json", receipt)
    return folder


def make_model(config: dict[str, Any], snapshot: str, mapping: Path,
               adapter: Path | None = None, *, wrapper_class=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from new_interest.training.oi_ni_meta_backend import RestrictedSIDModel
    from new_interest.training.train_sft import configure_row_only_adapter_saves
    seed_everything(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter) if adapter else snapshot, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    spec = (validate_saved_interest_tokens(tokenizer, mapping, expected_count=108) if adapter else
            register_semantic_interest_tokens(tokenizer, mapping, expected_count=108))
    base = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.bfloat16,
                                               attn_implementation="eager", local_files_only=True)
    base.config.use_cache = False
    if adapter:
        initialization = prepare_semantic_adapter_base(base, spec, initialization_mode="mean_norm_matched")
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    else:
        base.resize_token_embeddings(spec.resized_vocabulary_size)
        initialization, _ = initialize_semantic_interest_token_rows(base, spec, initialization_mode="mean_norm_matched")
        model = get_peft_model(base, LoraConfig(
            r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules="all-linear", task_type="CAUSAL_LM", bias="none",
            trainable_token_indices=trainable_token_indices(base, spec),
        ))
    configure_row_only_adapter_saves(model)
    wrapped = (wrapper_class or RestrictedSIDModel)(model, spec.token_ids).to("cuda")
    trainable = {name: p for name, p in wrapped.named_parameters() if p.requires_grad}
    if not trainable or not any("lora_A" in name for name in trainable):
        raise ValueError("Missing LoRA parameters")
    if any(not any(part in name for part in ("lora_A", "lora_B", "trainable_tokens_delta"))
           for name in trainable):
        raise ValueError("Unexpected trainable full module")
    return wrapped, tokenizer, spec, initialization


def encode_records(tokenizer: Any, records: list[dict], *, max_length: int,
                   field: str = "prompt") -> list[list[int]]:
    encoded = []
    for record in records:
        rendered = apply_chat_template(tokenizer, [{"role": "user", "content": record[field]}],
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if not ids or len(ids) > max_length:
            raise ValueError(f"Overlength/empty prompt: {record.get('sample_id')}: {len(ids)}")
        if any(token >= tokenizer.vocab_size for token in ids):
            # Added SID tokens never belong in the history-only prompt.
            added = set(tokenizer.convert_tokens_to_ids(f"<interest_{i:03d}>") for i in range(108))
            if added.intersection(ids):
                raise ValueError("Target SID token leaked into input")
        encoded.append(ids)
    return encoded


def batch_inputs(sequences: list[list[int]], pad_token: int) -> dict[str, Any]:
    import torch
    width = max(map(len, sequences))
    return {"input_ids": torch.tensor([[pad_token] * (width-len(row)) + row for row in sequences], device="cuda"),
            "attention_mask": torch.tensor([[0] * (width-len(row)) + [1] * len(row) for row in sequences], device="cuda")}


def loss_chunks(indices: list[int], records: list[dict], tokens: list[list[int]],
                pad_token: int, microbatch: int):
    import torch
    from new_interest.training.oi_ni_meta_objective import LossChunk
    chunks = []
    for start in range(0, len(indices), microbatch):
        selected = indices[start:start+microbatch]
        sequences = [tokens[index] for index in selected]
        labels = [int(records[index]["interest_index"]) for index in selected]
        def loss_fn(view, sequences=sequences, labels=labels):
            output = view(**batch_inputs(sequences, pad_token))
            return torch.nn.functional.cross_entropy(output, torch.tensor(labels, device="cuda"))
        chunks.append(LossChunk(loss_fn, weight=len(selected)))
    return chunks


def as_eval_record(row: dict) -> dict:
    result = dict(row)
    if "acceptable_subcategories" not in result:
        result["acceptable_subcategories"] = [str(row["target_subcategory"])]
    if "impression_id" not in result:
        result["impression_id"] = str(row["source_sample_id"])
    return result


def input_digest(row: dict, token_ids: list[int]) -> str:
    value = {"input_ids": token_ids, "impression_id": str(row["impression_id"]),
             "user_id": str(row.get("user_id", "")), "targets": list(row["acceptable_subcategories"])}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_frozen_data(config: dict[str, Any], data: Path) -> dict[str, str]:
    metadata_path = data / "metadata.json"
    if sha256_file(metadata_path) != config["data_metadata_sha256"]:
        raise ValueError("Frozen data metadata hash differs")
    metadata = json.loads(metadata_path.read_text())
    hashes = {"metadata.json": sha256_file(metadata_path)}
    for name, item in metadata["artifacts"].items():
        if not (data / name).resolve().is_relative_to(data.resolve()):
            raise ValueError("Data artifact escapes root")
        digest = sha256_file(data / name)
        if digest != item["sha256"]:
            raise ValueError(f"Frozen data hash differs: {name}")
        hashes[name] = digest
    oi = pq.read_table(data / "fit_oi.parquet").to_pylist()
    ni = pq.read_table(data / "fit_ni.parquet").to_pylist()
    val = pq.read_table(data / "validation_ni.parquet").to_pylist()
    dev = pq.read_table(data / "dev_full_grouped.parquet", columns=["user_id"]).to_pylist()
    for rows in [oi, ni, val]:
        if [r["row_index"] for r in rows] != list(range(len(rows))):
            raise ValueError("Data row indices differ")
        if len({r["user_id"] for r in rows}) != len(rows):
            raise ValueError("More than one target per role/user")
    fitting = {r["user_id"] for r in oi+ni}
    valid = {r["user_id"] for r in val}
    dev_users = {r["user_id"] for r in dev}
    if fitting & valid or (fitting | valid) & dev_users:
        raise ValueError("User leakage between fit, validation, or Dev")
    if (len(oi), len(ni), len(val)) != (17037,19440,2160):
        raise ValueError("Frozen pilot row counts differ")
    if Counter(r["interest_index"] for r in val) != Counter({i:20 for i in range(108)}):
        raise ValueError("Validation is not 20 per class")
    episodes = pq.read_table(data / "episodes.parquet").to_pylist()
    if len(episodes) != config["pilot_steps"]:
        raise ValueError("Episode horizon differs")
    for index, episode in enumerate(episodes):
        classes = episode["class_indices"]
        support, query = episode["support_indices"], episode["query_indices"]
        if episode["episode_index"] != index or len(classes) != 8 or len(set(classes)) != 8:
            raise ValueError("Invalid episode class topology")
        if len(support) != 16 or len(set(support)) != 16 or len(query) != 16 or len(set(query)) != 16:
            raise ValueError("Invalid episode sample topology")
        expected = Counter({c:2 for c in classes})
        if (Counter(oi[i]["interest_index"] for i in support) != expected or
                Counter(ni[i]["interest_index"] for i in query) != expected):
            raise ValueError("Episode class marginals differ")
        if {oi[i]["user_id"] for i in support} & {ni[i]["user_id"] for i in query}:
            raise ValueError("Episode support/query user overlap")
    return hashes


def predict(model: Any, tokens: list[list[int]], records: list[dict], tokenizer: Any,
            labels: list[str], output: Path, identity: str, *, batch_size: int = 8,
            shard_size: int = 256) -> dict[str, Any]:
    import torch
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    all_predictions = []
    for shard_start in range(0, len(records), shard_size):
        shard = output / f"predictions-{shard_start:06d}.parquet"
        receipt = shard.with_suffix(".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["identity"] != identity or sha256_file(shard) != saved["sha256"]:
                raise ValueError("Evaluation shard identity/hash mismatch")
            rows = pq.read_table(shard).to_pylist()
        else:
            rows = []
            for start in range(shard_start, min(shard_start+shard_size, len(records)), batch_size):
                stop = min(start+batch_size, shard_start+shard_size, len(records))
                with torch.no_grad():
                    scores = model(**batch_inputs(tokens[start:stop], tokenizer.pad_token_id))
                    scores = torch.log_softmax(scores.float(), dim=-1).cpu().numpy()
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite evaluation scores")
                for offset, score in enumerate(scores):
                    index = start+offset
                    order = np.lexsort((np.arange(len(labels)), -score))
                    row = records[index]
                    rows.append({"row_index": index, "impression_id": str(row["impression_id"]),
                                 "user_id": str(row.get("user_id", "")),
                                 "input_sha256": input_digest(row, tokens[index]),
                                 "acceptable_subcategories": list(row["acceptable_subcategories"]),
                                 "ranked_labels": [labels[i] for i in order[:20]],
                                 "ranked_indices": [int(i) for i in order[:20]],
                                 "ranked_scores": [float(score[i]) for i in order[:20]],
                                 "all_sid_log_probabilities": [float(x) for x in score]})
            temporary = shard.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, shard)
            write_json(receipt, {"identity": identity, "sha256": sha256_file(shard), "rows": len(rows)})
        expected = list(range(shard_start, min(shard_start+shard_size, len(records))))
        if [row["row_index"] for row in rows] != expected:
            raise ValueError("Evaluation row coverage differs")
        for row, index in zip(rows, expected):
            if (row["impression_id"] != str(records[index]["impression_id"]) or
                    row["acceptable_subcategories"] != list(records[index]["acceptable_subcategories"]) or
                    row["input_sha256"] != input_digest(records[index], tokens[index])):
                raise ValueError("Evaluation source record mismatch")
            scores = np.asarray(row["all_sid_log_probabilities"])
            order = np.lexsort((np.arange(len(labels)), -scores))[:20]
            if (scores.shape != (len(labels),) or not np.isfinite(scores).all() or
                    row["ranked_indices"] != [int(i) for i in order] or
                    row["ranked_labels"] != [labels[i] for i in order] or
                    row["ranked_scores"] != [float(scores[i]) for i in order]):
                raise ValueError("Invalid full-candidate ranking")
        all_predictions.extend(rows)
        log("evaluation_progress", rows=len(all_predictions), total=len(records))
    ranked = [[{"label": label, "score": score} for label, score in zip(row["ranked_labels"], row["ranked_scores"])]
              for row in all_predictions]
    metrics, _ = evaluate_ranked_outputs(records, ranked)
    metrics.update(identity=identity, no_test_time_adaptation=True, history_filtering=False,
                   projection_dtype="float32", tie_break="interest_index_ascending")
    write_json(output / "metrics.json", metrics)
    model.train()
    return metrics


def capture_probe(model: Any, tokenizer: Any, tokens: list[list[int]]) -> list[list[float]]:
    import torch
    model.eval()
    with torch.no_grad():
        scores = [model(**batch_inputs([row], tokenizer.pad_token_id)).cpu().float().tolist()[0]
                  for row in tokens]
    model.train()
    return scores


def save_checkpoint(model: Any, tokenizer: Any, optimizer: Any, scheduler: Any,
                    step: int, root: Path, scratch: Path, identity: str,
                    probe_tokens: list[list[int]], extra: dict[str, Any]) -> Path:
    import torch
    local = scratch / f"checkpoint-{step:06d}"
    local.mkdir(parents=True, exist_ok=True)
    model.causal_lm.save_pretrained(local / "adapter", safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(local / "adapter")
    probe = capture_probe(model, tokenizer, probe_tokens)
    torch.save({"step": step, "identity": identity, "optimizer": optimizer.state_dict(),
                "parameter_names": [name for name,p in model.named_parameters() if p.requires_grad],
                "scheduler": scheduler.state_dict(), "rng": rng_state()}, local / "training_state.pt")
    write_json(local / "state.json", {"identity": identity, "step": step, **extra})
    write_json(local / "reload_probe.json", {"input_ids": probe_tokens, "scores": probe,
                                             "batch_size": 1, "backend": model.backend_audit})
    destination = root / "checkpoints" / local.name
    publish(local, destination, identity)
    log("checkpoint_committed", step=step)
    return destination


def latest_checkpoint(root: Path, identity: str) -> Path | None:
    candidates = sorted((root / "checkpoints").glob("checkpoint-[0-9]*"), reverse=True)
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            verify_manifest(candidate, identity)
            return candidate
    return None


def parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_training(config: dict[str, Any], snapshot: str, data: Path, root: Path,
                 *, mode: str, alpha: float, steps: int, phase: str,
                 source_identity: str, eval_dev: bool = False,
                 selection_sha256: str | None = None,
                 budget_sha256: str | None = None) -> dict[str, Any]:
    import torch
    from new_interest.training.oi_ni_meta_objective import compute_meta_gradients
    settings = {"config": config, "mode": mode, "alpha": alpha, "steps": steps,
                "phase": phase, "source_identity": source_identity,
                "eval_dev": eval_dev, "selection_sha256": selection_sha256,
                "budget_sha256": budget_sha256}
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "resolved_config.json").exists():
        if json.loads((root / "resolved_config.json").read_text())["identity"] != identity:
            raise ValueError("Refusing to reuse a run directory with a different identity")
    if (root / "success.json").exists() and (root / "manifest.json").exists():
        success = json.loads((root / "success.json").read_text())
        if success["identity"] != identity:
            raise ValueError("Completed run identity differs")
        verify_manifest(root, identity)
        verify_manifest(root / success["checkpoint_relative"], identity)
        return success
    write_json(root / "resolved_config.json", {"identity": identity, **settings})
    mapping_path = data / "interest_token_mapping.parquet"
    labels = [str(row["target_subcategory"]) for row in load_interest_token_mapping(mapping_path)]
    checkpoint = latest_checkpoint(root, identity)
    model, tokenizer, spec, initialization = make_model(config, snapshot, mapping_path,
                                                       checkpoint / "adapter" if checkpoint else None)
    attempt = uuid.uuid4().hex[:12]
    model_audit = {"backend": model.backend_audit,
                "initialization": initialization, "initial_trainable_sha256": parameter_digest(model),
                "trainable_parameters": {n: list(p.shape) for n,p in model.named_parameters() if p.requires_grad},
                "trainable_dtypes": {n: str(p.dtype) for n,p in model.named_parameters() if p.requires_grad},
                "device": torch.cuda.get_device_name(0), "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    if not (root / "model_audit.json").exists():
        if checkpoint:
            raise ValueError("Resume is missing original initialization audit")
        write_json(root / "model_audit.json", model_audit)
    else:
        write_json(root / "attempts" / f"{attempt}.json", model_audit)
    oi = pq.read_table(data / "fit_oi.parquet").to_pylist()
    ni = pq.read_table(data / "fit_ni.parquet").to_pylist()
    validation = [as_eval_record(row) for row in pq.read_table(data / "validation_ni.parquet").to_pylist()]
    episodes = pq.read_table(data / "episodes.parquet").to_pylist()
    if steps > len(episodes) and not (phase == "compute_matched" and mode == "mixed"
                                     and steps <= config["max_compute_multiplier"] * len(episodes)):
        raise ValueError("Frozen episode budget exhausted")
    oi_tokens = encode_records(tokenizer, oi, max_length=config["max_seq_length"])
    no_history_tokens = encode_records(tokenizer, oi, max_length=config["max_seq_length"], field="no_history_prompt")
    ni_tokens = encode_records(tokenizer, ni, max_length=config["max_seq_length"])
    val_tokens = encode_records(tokenizer, validation, max_length=config["max_seq_length"])
    write_json(root / "token_audit.json", {"oi": len(oi_tokens), "ni": len(ni_tokens), "validation": len(val_tokens),
               "oi_max": max(map(len, oi_tokens)), "ni_max": max(map(len, ni_tokens)),
               "validation_max": max(map(len, val_tokens)), "over_length": 0, "target_in_input": False})
    trainable = {name: p for name,p in model.named_parameters() if p.requires_grad}
    if phase == "gate" and not checkpoint:
        support_index = max(range(len(oi_tokens)), key=lambda i: len(oi_tokens[i]))
        query_index = max(range(len(ni_tokens)), key=lambda i: len(ni_tokens[i]))
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        worst = compute_meta_gradients(model,
                    loss_chunks([support_index], oi, oi_tokens, tokenizer.pad_token_id, 1),
                    loss_chunks([query_index], ni, ni_tokens, tokenizer.pad_token_id, 1),
                    alpha=alpha, mode="meta")
        write_json(root / "long_sequence_hvp_audit.json", {
            "identity": identity, "scope": "gradient-only engineering gate; not a method score",
            "oi_tokens": len(oi_tokens[support_index]), "ni_tokens": len(ni_tokens[query_index]),
            "loss": float(worst.loss), "hvp_norm": float(worst.hvp_norm),
            "seconds": time.monotonic()-begin, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "parameter_update_performed": False})
        del worst
        torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=config["outer_learning_rate"],
                                 weight_decay=config["weight_decay"])
    warmup = max(1, round(steps*config["warmup_ratio"]))
    def multiplier(step: int) -> float:
        if step < warmup:
            return (step+1)/max(1,warmup)
        fraction = min(1.0, (step-warmup)/max(1,steps-warmup))
        return 0.5*(1+math.cos(math.pi*fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    start_step = 0
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume state identity mismatch")
        if state["parameter_names"] != list(trainable):
            raise ValueError("Optimizer parameter name order differs")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        start_step = int(state["step"])
        log("checkpoint_resumed", step=start_step)
    scratch = Path(os.environ.get("META_LOCAL_SCRATCH", "/tmp/oi-ni-meta")) / identity[:20]
    scratch.mkdir(parents=True, exist_ok=True)
    model.train()
    durations = []
    last_metrics = None
    for index in range(start_step, steps):
        # Fixed per-episode dropout stream keeps real F/G stochasticity aligned
        # across arms despite the no-history arm performing an extra inner F.
        torch.manual_seed(int(config["seed"]) + 104729 * index)
        torch.cuda.manual_seed_all(int(config["seed"]) + 104729 * index)
        episode = episodes[index % len(episodes)]
        support = [int(i) for i in episode["support_indices"]]
        query = [int(i) for i in episode["query_indices"]]
        if set(oi[i]["user_id"] for i in support).intersection(ni[i]["user_id"] for i in query):
            raise ValueError("Episode support/query user overlap")
        f = loss_chunks(support, oi, oi_tokens, tokenizer.pad_token_id, config["microbatch_size"])
        g = loss_chunks(query, ni, ni_tokens, tokenizer.pad_token_id, config["microbatch_size"])
        f_empty = loss_chunks(support, oi, no_history_tokens, tokenizer.pad_token_id, config["microbatch_size"])
        torch.cuda.synchronize()
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        result = compute_meta_gradients(model, f, g, alpha=alpha, mode=mode,
                        no_history_oi=f_empty if mode == "meta_no_history" else None)
        for name, parameter in trainable.items():
            gradient = result.gradients[name]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite outer gradient: {name}")
            parameter.grad = gradient.to(parameter)
        sid_audit = {}
        if index == 0:
            for name,p in trainable.items():
                if "trainable_tokens_delta" in name:
                    norms = p.grad.float().reshape(108,-1).norm(dim=1)
                    if not (norms > 0).all():
                        raise ValueError("A SID row has zero gradient")
                    sid_audit[name] = {"finite_rows": 108, "nonzero_rows": 108}
            if not sid_audit:
                raise ValueError("Missing trainable SID row audit")
            write_json(root / "sid_gradient_audit.json", sid_audit)
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), config["max_grad_norm"], error_if_nonfinite=True)
        applied_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-begin
        durations.append(elapsed)
        report = {"step": index+1, "loss": float(result.loss), "oi_loss": float(result.oi_loss),
                  "ni_loss": float(result.ni_loss), "ni_fast_loss": float(result.adapted_ni_loss),
                  "gradient_norm": float(norm), "seconds": elapsed,
                  "applied_lr": applied_lr, "next_lr": optimizer.param_groups[0]["lr"],
                  "attempt": attempt, "resumed_from_step": start_step,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        if hasattr(result, "inner_grad_norm"):
            report.update(inner_grad_norm=float(result.inner_grad_norm), hvp_norm=float(result.hvp_norm))
        with (root / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(report, allow_nan=False)+"\n")
        log("training_step", mode=mode, **report)
        if (index+1) % config["checkpoint_every"] == 0 or index+1 == steps:
            checkpoint = save_checkpoint(model, tokenizer, optimizer, scheduler, index+1, root, scratch,
                                         identity, val_tokens[:4], {"mode": mode, "alpha": alpha})
        if (index+1) % config["validation_every"] == 0 or index+1 == steps:
            eval_records, eval_tokens = validation, val_tokens
            if phase == "gate":
                eval_records, eval_tokens = validation[:64], val_tokens[:64]
            last_metrics = predict(model, eval_tokens, eval_records, tokenizer, labels,
                                   root / f"validation-{index+1:06d}", identity,
                                   batch_size=config["evaluation_batch_size"])
    if checkpoint is None:
        raise ValueError("No completed training checkpoint")
    # A retry can begin after the final checkpoint but before validation/commit.
    final_records = validation[:64] if phase == "gate" else validation
    final_tokens = val_tokens[:64] if phase == "gate" else val_tokens
    last_metrics = predict(model, final_tokens, final_records, tokenizer, labels,
                           root / f"validation-{steps:06d}", identity,
                           batch_size=config["evaluation_batch_size"])
    final_digest = parameter_digest(model)
    initial_digest = json.loads((root / "model_audit.json").read_text())["initial_trainable_sha256"]
    if final_digest == initial_digest:
        raise ValueError("Training did not change any trainable parameter")
    del model, optimizer, scheduler, trainable
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer, _, _ = make_model(config, snapshot, mapping_path, checkpoint / "adapter")
    reference = json.loads((checkpoint / "reload_probe.json").read_text())
    reloaded = capture_probe(model, tokenizer, reference["input_ids"])
    error = float(np.max(np.abs(np.asarray(reloaded)-np.asarray(reference["scores"]))))
    if error != 0.0 or parameter_digest(model) != final_digest:
        raise ValueError(f"Fresh reload mismatch: logits={error}")
    write_json(root / "fresh_reload_audit.json", {"max_abs_error": error, "trainable_tensors_exact": True,
                                                "trainable_sha256": final_digest, "deployed_outer_parameters": True})
    if eval_dev:
        records = pq.read_table(data / "dev_full_grouped.parquet").to_pylist()
        dev_tokens = encode_records(tokenizer, records, max_length=config["max_seq_length"])
        metrics = predict(model, dev_tokens, records, tokenizer, labels, root / "dev", identity,
                          batch_size=config["evaluation_batch_size"])
        if metrics["samples"] != 31392 or metrics["supported_interest_count"] != 104:
            raise ValueError("Full Dev topology mismatch")
    success = {"identity": identity, "mode": mode, "alpha": alpha, "steps": steps,
               "checkpoint": str(checkpoint), "fresh_reload_max_abs_error": error,
               "checkpoint_relative": str(checkpoint.relative_to(root)),
               "validation_metrics": last_metrics, "median_step_seconds": float(np.median(durations)) if durations else None,
               "deployed_trainable_sha256": final_digest, "dev_evaluated": eval_dev}
    write_json(root / "success.json", success)
    write_json(root / "manifest.json", {"identity": identity, "files": inventory(root)})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["gate", "calibrate", "pilot", "compute_matched"], required=True)
    parser.add_argument("--mode", choices=["mixed", "meta", "meta_no_history"], default="meta")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--budget-file", type=Path)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This frozen pilot requires exactly one CUDA GPU")
    config = json.loads(args.config.read_text())
    data = Path(config["data_dir"])
    source_files = validate_frozen_data(config, data)
    source_files["protocol"] = sha256_file(args.config)
    if Path("source_bundle_manifest.json").exists():
        bundle = json.loads(Path("source_bundle_manifest.json").read_text())
        for name, digest in bundle["files"].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Frozen worker source changed: {name}")
        source_files["worker_bundle"] = sha256_file(Path("source_bundle_manifest.json"))
    for name in ["oi_ni_meta.py", "../training/oi_ni_meta_backend.py", "../training/oi_ni_meta_objective.py"]:
        p = Path(__file__).parent / name
        source_files[name] = sha256_file(p)
    source_identity = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    if (args.root / "source_identity.json").exists():
        if json.loads((args.root / "source_identity.json").read_text())["identity"] != source_identity:
            raise ValueError("Worker output belongs to a different frozen source")
    write_json(args.root / "source_identity.json", {"identity": source_identity, "files": source_files})
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "transformers", "peft", "accelerate", "pyarrow", "tokenizers", "safetensors", "huggingface-hub"]}
    write_json(args.root / "environment.json", {"packages": versions, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
    snapshot = prepare_snapshot(config, args.root)
    if args.phase == "gate":
        result = run_training(config, snapshot, data, args.root / "gate", mode="meta",
                              alpha=config["gate_alpha"], steps=config["gate_steps"], phase="gate",
                              source_identity=source_identity)
        write_json(args.root / "gate_success.json", result)
    elif args.phase == "calibrate":
        results = []
        for index, alpha in enumerate(config["inner_alpha_candidates"]):
            results.append(run_training(config, snapshot, data, args.root / f"alpha-{index}", mode="meta",
                         alpha=alpha, steps=config["calibration_steps"], phase="calibrate", source_identity=source_identity))
        chosen = max(results, key=lambda r: (r["validation_metrics"]["interest_macro_recall_at_10"],
                     r["validation_metrics"]["interest_macro_ndcg_at_10"], -r["alpha"]))
        write_json(args.root / "alpha_selection.json", {"source_identity": source_identity,
                    "chosen_alpha": chosen["alpha"], "validation_only": True, "results": results})
    else:
        if args.selection_file is None:
            raise ValueError("Pilot requires a frozen validation-only alpha selection receipt")
        selection = json.loads(args.selection_file.read_text())
        if selection["source_identity"] != source_identity or selection.get("validation_only") is not True:
            raise ValueError("Alpha selection identity differs")
        selected_alpha = float(selection["chosen_alpha"])
        if selected_alpha not in config["inner_alpha_candidates"] or (args.alpha is not None and args.alpha != selected_alpha):
            raise ValueError("Pilot alpha differs from frozen selection")
        budget_sha = None
        steps = args.steps or config["pilot_steps"]
        output_name = args.mode
        if args.phase == "compute_matched":
            if args.mode != "mixed" or args.budget_file is None:
                raise ValueError("Compute control requires mixed mode and a frozen timing-budget receipt")
            budget = json.loads(args.budget_file.read_text())
            if budget["source_identity"] != source_identity:
                raise ValueError("Compute budget source differs")
            steps = int(budget["steps"])
            if not config["pilot_steps"] <= steps <= config["max_compute_multiplier"]*config["pilot_steps"]:
                raise ValueError("Compute control exceeds bounded pilot budget")
            budget_sha = sha256_file(args.budget_file)
            output_name = "mixed_compute"
        result = run_training(config, snapshot, data, args.root / output_name, mode=args.mode,
                  alpha=0.0 if args.mode == "mixed" else selected_alpha,
                  steps=steps, phase=args.phase, source_identity=source_identity,
                  eval_dev=True, selection_sha256=sha256_file(args.selection_file), budget_sha256=budget_sha)
        write_json(args.root / f"{output_name}_success.json", result)


if __name__ == "__main__":
    main()
