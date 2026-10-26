from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.constrained_beam import (
    build_response_trie,
    decode_ranked_sequences,
    fill_unique_predictions,
    make_prefix_allowed_tokens_fn,
)
from new_interest.evaluation.metrics import evaluate_ranked_outputs
from new_interest.training.sft_data import apply_chat_template


EVALUATION_IDENTITY_VERSION = (
    "constrained-beam-sharded-v3-semantic-adapter-staged-parquet"
)
ADAPTER_SEMANTIC_HASH_VERSION = "peft-evaluation-adapter-v1"
ADAPTER_SEMANTIC_HASH_EXCLUDED_PATHS = frozenset(
    {
        "README.md",
        "training_args.bin",
    }
)
ADAPTER_CONFIG_UNORDERED_LIST_FIELDS = frozenset(
    {
        "exclude_modules",
        "modules_to_save",
        "target_modules",
    }
)
PARQUET_PUBLICATION_VERSION = "local-stage-verified-mounted-rename-v1"


class ParquetPublicationError(RuntimeError):
    """Raised when a staged Parquet file cannot be committed safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _canonical_adapter_config(path: Path) -> bytes:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse adapter config: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter config must be a JSON object: {path}")
    normalized = dict(payload)
    for field in ADAPTER_CONFIG_UNORDERED_LIST_FIELDS:
        value = normalized.get(field)
        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            normalized[field] = sorted(set(value))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_adapter_semantics(path: Path) -> str:
    """Hash only evaluation-relevant adapter payloads with stable config order."""

    if not path.is_dir():
        raise FileNotFoundError(path)
    relative_files = {
        item.relative_to(path).as_posix(): item
        for item in path.rglob("*")
        if item.is_file()
    }
    missing = [
        required
        for required in ("adapter_config.json",)
        if required not in relative_files
    ]
    weight_names = {
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
    if not weight_names.intersection(relative_files):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if missing:
        raise ValueError(
            f"Adapter is missing required evaluation files: {', '.join(missing)}"
        )
    included = sorted(
        (relative, item)
        for relative, item in relative_files.items()
        if relative not in ADAPTER_SEMANTIC_HASH_EXCLUDED_PATHS
    )
    if not included:
        raise ValueError(f"Cannot hash empty evaluation adapter: {path}")
    digest = hashlib.sha256()
    digest.update(ADAPTER_SEMANTIC_HASH_VERSION.encode("utf-8"))
    for relative, item in included:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        if relative == "adapter_config.json":
            content_digest = hashlib.sha256(
                _canonical_adapter_config(item)
            ).digest()
        else:
            content_digest = bytes.fromhex(sha256_file(item))
        digest.update(content_digest)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_input_sha256(record: dict[str, object]) -> str:
    return canonical_sha256(record)


def select_logical_shard(
    records: list[dict[str, object]],
    *,
    num_shards: int,
    shard_index: int,
) -> tuple[list[int], list[dict[str, object]]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    indices = [
        index
        for index in range(len(records))
        if index % num_shards == shard_index
    ]
    return indices, [records[index] for index in indices]


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _default_parquet_staging_dir() -> Path:
    configured = os.environ.get("NEW_INTEREST_PARQUET_STAGE_DIR")
    if configured:
        return Path(configured)
    scratch = Path("/scratch")
    if scratch.is_dir() and os.access(scratch, os.W_OK):
        return scratch / "new-interest-parquet-staging"
    return Path(tempfile.gettempdir()) / "new-interest-parquet-staging"


def _validate_parquet_publication(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_schema: pa.Schema,
) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ParquetPublicationError(f"Parquet file is missing or empty: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise ParquetPublicationError(
            f"Parquet SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_hash}"
        )
    table = pq.read_table(path)
    if table.num_rows != expected_rows:
        raise ParquetPublicationError(
            f"Parquet row count mismatch for {path}: "
            f"expected {expected_rows}, got {table.num_rows}"
        )
    if not table.schema.equals(expected_schema, check_metadata=False):
        raise ParquetPublicationError(f"Parquet schema mismatch for {path}")


def _publish_parquet_once(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_schema: pa.Schema,
) -> None:
    temporary = destination.with_name(
        f".{destination.name}.partial-{uuid.uuid4().hex}"
    )
    try:
        with source.open("rb") as input_handle, temporary.open(
            "wb"
        ) as output_handle:
            shutil.copyfileobj(
                input_handle,
                output_handle,
                length=1024 * 1024,
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        _validate_parquet_publication(
            temporary,
            expected_sha256=expected_sha256,
            expected_rows=expected_rows,
            expected_schema=expected_schema,
        )
        os.replace(temporary, destination)
        _validate_parquet_publication(
            destination,
            expected_sha256=expected_sha256,
            expected_rows=expected_rows,
            expected_schema=expected_schema,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(
    path: Path,
    rows: list[dict[str, object]],
    *,
    staging_dir: Path | None = None,
    publish_attempts: int = 3,
    retry_seconds: float = 2.0,
) -> None:
    if publish_attempts <= 0:
        raise ValueError("publish_attempts must be positive")
    if retry_seconds < 0:
        raise ValueError("retry_seconds must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    local_staging_dir = staging_dir or _default_parquet_staging_dir()
    local_staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f".{path.name}.",
        dir=local_staging_dir,
        delete=False,
    ) as handle:
        local_path = Path(handle.name)
    try:
        table = pa.Table.from_pylist(rows)
        pq.write_table(
            table,
            local_path,
            compression="zstd",
            use_dictionary=True,
        )
        expected_hash = sha256_file(local_path)
        _validate_parquet_publication(
            local_path,
            expected_sha256=expected_hash,
            expected_rows=table.num_rows,
            expected_schema=table.schema,
        )
        last_error: BaseException | None = None
        for attempt in range(1, publish_attempts + 1):
            try:
                _publish_parquet_once(
                    local_path,
                    path,
                    expected_sha256=expected_hash,
                    expected_rows=table.num_rows,
                    expected_schema=table.schema,
                )
                return
            except (OSError, pa.ArrowException, ParquetPublicationError) as error:
                last_error = error
                if attempt < publish_attempts and retry_seconds:
                    time.sleep(retry_seconds)
        raise ParquetPublicationError(
            f"Failed to publish {path} after {publish_attempts} attempts"
        ) from last_error
    finally:
        local_path.unlink(missing_ok=True)


def _validate_ranked_output(
    ranked: list[dict[str, object]],
    *,
    labels: set[str],
    top_k: int,
    context: str,
) -> None:
    ranked_labels = [str(item["label"]) for item in ranked]
    if len(ranked_labels) != top_k:
        raise ValueError(
            f"{context}: expected {top_k} ranked labels, "
            f"got {len(ranked_labels)}"
        )
    if len(set(ranked_labels)) != top_k:
        raise ValueError(f"{context}: ranked labels are not unique")
    unknown = sorted(set(ranked_labels).difference(labels))
    if unknown:
        raise ValueError(f"{context}: unknown ranked labels: {unknown}")


def _load_labels(path: Path) -> list[str]:
    return sorted(
        {
            str(row["target_subcategory"])
            for row in pq.read_table(
                path,
                columns=["target_subcategory"],
            ).to_pylist()
        }
    )


def _generate_constrained_candidates(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, Any],
    labels: list[str],
    *,
    num_beams: int,
    max_new_tokens: int,
):
    import torch

    trie, _ = build_response_trie(tokenizer, labels)
    initial_input_length = encoded["input_ids"].shape[1]
    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
        trie,
        initial_input_length=initial_input_length,
    )
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return decode_ranked_sequences(
        tokenizer,
        trie,
        generated.sequences,
        generated.sequences_scores,
        initial_input_length=initial_input_length,
        num_return_sequences=num_beams,
    )


def generate_ranked_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, object]],
    labels: list[str],
    *,
    batch_size: int,
    num_beams: int,
    top_k: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> tuple[
    list[list[dict[str, object]]],
    list[dict[str, int]],
]:
    if num_beams > len(labels):
        raise ValueError(
            f"num_beams={num_beams} exceeds {len(labels)} labels"
        )
    if top_k <= 0 or top_k > num_beams:
        raise ValueError("top_k must be positive and no greater than num_beams")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    ranked_outputs: list[list[dict[str, object]]] = []
    generation_diagnostics: list[dict[str, int]] = []

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [
            apply_chat_template(
                tokenizer,
                [{"role": "user", "content": str(row["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for row in batch
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
        }
        ranked_batches = _generate_constrained_candidates(
            model,
            tokenizer,
            encoded,
            labels,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        for batch_offset, initial_predictions in enumerate(ranked_batches):

            def generate_remaining(remaining_labels: list[str]):
                retry_encoded = tokenizer(
                    [texts[batch_offset]],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                retry_encoded = {
                    key: value.to(model.device)
                    for key, value in retry_encoded.items()
                }
                retry_beams = min(num_beams, len(remaining_labels))
                return _generate_constrained_candidates(
                    model,
                    tokenizer,
                    retry_encoded,
                    remaining_labels,
                    num_beams=retry_beams,
                    max_new_tokens=max_new_tokens,
                )[0]

            filled = fill_unique_predictions(
                initial_predictions,
                labels,
                top_k=top_k,
                generate_remaining=generate_remaining,
            )
            ranked_outputs.append(
                [
                    {
                        "label": prediction.label,
                        "response": prediction.response,
                        "score": prediction.score,
                    }
                    for prediction in filled.predictions
                ]
            )
            generation_diagnostics.append(
                {
                    "dedup_retry_rounds": filled.retry_rounds,
                    "initial_unique_label_count": (
                        filled.initial_unique_count
                    ),
                }
            )
    return ranked_outputs, generation_diagnostics


def evaluate_beam(config: dict[str, Any]) -> dict[str, object]:
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    adapter_path = Path(config["adapter_path"])
    eval_file = Path(config["eval_file"])
    labels_file = Path(config["retained_interests_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_beams = int(config.get("num_beams", 40))
    top_k = int(config.get("top_k", 20))
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    cutoffs = tuple(int(value) for value in config.get(
        "cutoffs",
        (1, 5, 10, 20),
    ))
    if max(cutoffs) > top_k:
        raise ValueError("Metric cutoff cannot exceed top_k")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    labels = _load_labels(labels_file)
    label_set = set(labels)
    all_records = pq.read_table(eval_file).to_pylist()
    max_examples = config.get("max_examples")
    if max_examples is not None:
        all_records = all_records[: int(max_examples)]
    global_indices, records = select_logical_shard(
        all_records,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if not records:
        raise ValueError(
            f"Logical shard {shard_index}/{num_shards} has no records"
        )

    eval_sha256 = sha256_file(eval_file)
    labels_sha256 = sha256_file(labels_file)
    adapter_sha256 = sha256_adapter_semantics(adapter_path)
    identity_payload = {
        "identity_version": EVALUATION_IDENTITY_VERSION,
        "model_name": model_name,
        "adapter_sha256": adapter_sha256,
        "adapter_hash_version": ADAPTER_SEMANTIC_HASH_VERSION,
        "eval_sha256": eval_sha256,
        "labels_sha256": labels_sha256,
        "full_record_count": len(all_records),
        "label_count": len(labels),
        "labels": labels,
        "max_examples": (
            int(max_examples) if max_examples is not None else None
        ),
        "num_shards": num_shards,
        "num_beams": num_beams,
        "top_k": top_k,
        "cutoffs": list(cutoffs),
        "max_new_tokens": int(config.get("max_new_tokens", 32)),
        "enable_thinking": bool(config.get("enable_thinking", False)),
    }
    evaluation_identity = canonical_sha256(identity_payload)
    input_hashes = [record_input_sha256(record) for record in records]

    batch_size = int(config.get("batch_size", 4))
    checkpoint_size = int(config.get("checkpoint_size", 64))
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    ranked_outputs: list[list[dict[str, object]]] = []
    generation_diagnostics: list[dict[str, int]] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()
    for local_start in range(0, len(records), checkpoint_size):
        local_stop = min(local_start + checkpoint_size, len(records))
        checkpoint_path = (
            shards_dir / f"{local_start:09d}-{local_stop:09d}.parquet"
        )
        if checkpoint_path.is_file():
            checkpoint_rows = pq.read_table(checkpoint_path).to_pylist()
            if len(checkpoint_rows) != local_stop - local_start:
                raise ValueError(
                    f"{checkpoint_path}: expected "
                    f"{local_stop - local_start} rows, "
                    f"got {len(checkpoint_rows)}"
                )
            expected_indices = global_indices[local_start:local_stop]
            actual_indices = [
                int(row["record_index"]) for row in checkpoint_rows
            ]
            if actual_indices != expected_indices:
                raise ValueError(
                    f"{checkpoint_path}: global record indices do not "
                    "match eval data"
                )
            expected_impressions = [
                str(row["impression_id"])
                for row in records[local_start:local_stop]
            ]
            actual_impressions = [
                str(row["impression_id"]) for row in checkpoint_rows
            ]
            if actual_impressions != expected_impressions:
                raise ValueError(
                    f"{checkpoint_path}: impression IDs do not match "
                    "eval data"
                )
            expected_hashes = input_hashes[local_start:local_stop]
            actual_hashes = [
                str(row["input_sha256"]) for row in checkpoint_rows
            ]
            if actual_hashes != expected_hashes:
                raise ValueError(
                    f"{checkpoint_path}: input hashes do not match eval data"
                )
            checkpoint_identities = {
                str(row["evaluation_identity"]) for row in checkpoint_rows
            }
            if checkpoint_identities != {evaluation_identity}:
                raise ValueError(
                    f"{checkpoint_path}: evaluation identity mismatch"
                )
            checkpoint_outputs = []
            checkpoint_diagnostics = []
            for row in checkpoint_rows:
                ranked = [
                    {
                        "label": label,
                        "response": response,
                        "score": score,
                    }
                    for label, response, score in zip(
                        row["ranked_labels"],
                        row["ranked_responses"],
                        row["ranked_scores"],
                    )
                ]
                _validate_ranked_output(
                    ranked,
                    labels=label_set,
                    top_k=top_k,
                    context=(
                        f"{checkpoint_path}:record_index="
                        f"{row['record_index']}"
                    ),
                )
                checkpoint_outputs.append(ranked)
                checkpoint_diagnostics.append(
                    {
                        "dedup_retry_rounds": int(
                            row["dedup_retry_rounds"]
                        ),
                        "initial_unique_label_count": int(
                            row["initial_unique_label_count"]
                        ),
                    }
                )
            resumed_samples += len(checkpoint_rows)
            ranked_outputs.extend(checkpoint_outputs)
            generation_diagnostics.extend(checkpoint_diagnostics)
            continue

        batch_outputs, batch_diagnostics = generate_ranked_outputs(
            model,
            tokenizer,
            records[local_start:local_stop],
            labels,
            batch_size=batch_size,
            num_beams=num_beams,
            top_k=top_k,
            max_new_tokens=int(config.get("max_new_tokens", 32)),
            enable_thinking=bool(config.get("enable_thinking", False)),
        )
        checkpoint_rows = [
            {
                "record_index": global_indices[local_start + offset],
                "input_sha256": input_hashes[local_start + offset],
                "evaluation_identity": evaluation_identity,
                "impression_id": str(
                    records[local_start + offset]["impression_id"]
                ),
                "ranked_labels": [
                    item["label"] for item in ranked
                ],
                "ranked_responses": [
                    item["response"] for item in ranked
                ],
                "ranked_scores": [
                    item["score"] for item in ranked
                ],
                **batch_diagnostics[offset],
            }
            for offset, ranked in enumerate(batch_outputs)
        ]
        for row, ranked in zip(checkpoint_rows, batch_outputs):
            _validate_ranked_output(
                ranked,
                labels=label_set,
                top_k=top_k,
                context=f"record_index={row['record_index']}",
            )
        _atomic_write_parquet(checkpoint_path, checkpoint_rows)
        generated_samples += len(batch_outputs)
        ranked_outputs.extend(batch_outputs)
        generation_diagnostics.extend(batch_diagnostics)

    metrics, details = evaluate_ranked_outputs(
        records,
        ranked_outputs,
        cutoffs=cutoffs,
    )
    runtime_seconds = time.time() - started_at
    metrics["runtime_seconds"] = runtime_seconds
    metrics["samples_per_second"] = (
        generated_samples / runtime_seconds
        if generated_samples
        else None
    )
    metrics["generated_samples_this_run"] = generated_samples
    metrics["resumed_samples"] = resumed_samples
    metrics["checkpoint_shard_count"] = (
        len(records) + checkpoint_size - 1
    ) // checkpoint_size
    metrics["checkpoint_size"] = checkpoint_size
    metrics["num_beams"] = num_beams
    metrics["top_k"] = top_k
    metrics["cutoffs"] = list(cutoffs)
    metrics["model_name"] = model_name
    metrics["adapter_path"] = str(adapter_path.resolve())
    metrics["eval_file"] = str(eval_file.resolve())
    metrics["eval_sha256"] = eval_sha256
    metrics["labels_sha256"] = labels_sha256
    metrics["adapter_sha256"] = adapter_sha256
    metrics["adapter_hash_version"] = ADAPTER_SEMANTIC_HASH_VERSION
    metrics["parquet_publication_version"] = PARQUET_PUBLICATION_VERSION
    metrics["evaluation_identity"] = evaluation_identity
    metrics["logical_num_shards"] = num_shards
    metrics["logical_shard_index"] = shard_index
    metrics["full_record_count"] = len(all_records)
    metrics["logical_shard_record_count"] = len(records)
    retry_rounds = [
        item["dedup_retry_rounds"] for item in generation_diagnostics
    ]
    initial_unique_counts = [
        item["initial_unique_label_count"]
        for item in generation_diagnostics
    ]
    metrics["dedup_retry_example_count"] = sum(
        int(value > 0) for value in retry_rounds
    )
    metrics["dedup_retry_round_count"] = sum(retry_rounds)
    metrics["minimum_initial_unique_label_count"] = min(
        initial_unique_counts
    )

    prediction_rows = [
        {
            "record_index": global_index,
            "input_sha256": input_hash,
            "evaluation_identity": evaluation_identity,
            **detail,
            **diagnostic,
            "ranked_responses": [
                item["response"] for item in ranked
            ],
        }
        for global_index, input_hash, detail, ranked, diagnostic in zip(
            global_indices,
            input_hashes,
            details,
            ranked_outputs,
            generation_diagnostics,
        )
    ]
    _atomic_write_parquet(
        output_dir / "beam_predictions.parquet",
        prediction_rows,
    )
    _atomic_write_json(output_dir / "metrics.json", metrics)
    _atomic_write_json(output_dir / "resolved_config.json", config)
    _atomic_write_json(
        output_dir / "shard_manifest.json",
        {
            **identity_payload,
            "evaluation_identity": evaluation_identity,
            "shard_index": shard_index,
            "shard_record_count": len(records),
            "first_record_index": global_indices[0],
            "last_record_index": global_indices[-1],
            "prediction_sha256": sha256_file(
                output_dir / "beam_predictions.parquet"
            ),
        },
    )

    wandb_mode = str(config.get("wandb_mode", "offline"))
    if wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = wandb_mode
        run = wandb.init(
            project=str(
                config.get(
                    "wandb_project",
                    "cross-interest-generative-sft",
                )
            ),
            name=str(config["run_name"]),
            job_type="beam-evaluation",
            config=config,
        )
        run.log(
            {
                f"beam/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        artifact = wandb.Artifact(
            f"{config['run_name']}-beam-predictions",
            type="evaluation",
        )
        artifact.add_file(
            str(output_dir / "beam_predictions.parquet")
        )
        artifact.add_file(str(output_dir / "metrics.json"))
        run.log_artifact(artifact)
        run.finish()
    return metrics


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run constrained Beam ranking evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_json(args.config)
    for key in (
        "model_name",
        "adapter_path",
        "eval_file",
        "output_dir",
        "run_name",
        "max_examples",
        "num_shards",
        "shard_index",
        "wandb_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    metrics = evaluate_beam(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
