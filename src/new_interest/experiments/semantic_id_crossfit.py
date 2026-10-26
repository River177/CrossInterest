from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.evaluate_beam import (
    canonical_sha256,
    sha256_directory,
    sha256_file,
)
from new_interest.training.semantic_id_tokens import (
    SEMANTIC_INITIALIZATION_VERSION,
    load_interest_token_mapping,
    prepare_semantic_adapter_base,
    validate_saved_interest_tokens,
)
from new_interest.training.sft_data import apply_chat_template


SCORER_VERSION = "semantic-id-prior-adjusted-oof-v1"
WEIGHT_VERSION = "semantic-id-fold-global-quantile-v1"
CLASS_WEIGHT_VERSION = "semantic-id-within-class-quantile-v1"
FRONTIER_WEIGHT_VERSION = "semantic-id-within-class-frontier-v1"

SCORE_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("target_token", pa.string(), nullable=False),
        pa.field("target_raw_logprob", pa.float64(), nullable=False),
        pa.field("fold_target_prior", pa.float64(), nullable=False),
        pa.field(
            "prior_adjusted_target_logprob", pa.float64(), nullable=False
        ),
        pa.field("prior_adjusted_nll", pa.float64(), nullable=False),
        pa.field("prior_adjusted_margin", pa.float64(), nullable=False),
        pa.field("teacher_identity", pa.string(), nullable=False),
    ]
)

WEIGHT_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("prior_adjusted_nll", pa.float64(), nullable=False),
        pa.field("prior_adjusted_margin", pa.float64(), nullable=False),
        pa.field("fold_global_rank", pa.int32(), nullable=False),
        pa.field("fold_global_difficulty_quantile", pa.float64(), nullable=False),
        pa.field("raw_weight", pa.float64(), nullable=False),
        pa.field("sample_weight", pa.float64(), nullable=False),
    ]
)

CLASS_WEIGHT_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("prior_adjusted_nll", pa.float64(), nullable=False),
        pa.field("prior_adjusted_margin", pa.float64(), nullable=False),
        pa.field("within_class_rank", pa.int32(), nullable=False),
        pa.field("within_class_difficulty_quantile", pa.float64(), nullable=False),
        pa.field("raw_weight", pa.float64(), nullable=False),
        pa.field("sample_weight", pa.float64(), nullable=False),
    ]
)


def _atomic_write_table(path: Path, rows: Sequence[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(list(rows), schema=schema),
            temporary,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_per_class(
    rows: Iterable[dict[str, object]], max_per_class: int | None
) -> list[dict[str, object]]:
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["target_subcategory"])].append(row)
    selected: list[dict[str, object]] = []
    for label, values in sorted(by_class.items()):
        ordered = sorted(values, key=lambda row: str(row["source_sample_id"]))
        if max_per_class is not None:
            if max_per_class <= 0:
                raise ValueError("max_per_class must be positive")
            ordered = ordered[:max_per_class]
        if not ordered:
            raise ValueError(f"No rows selected for {label}")
        selected.extend(ordered)
    return sorted(selected, key=lambda row: str(row["source_sample_id"]))


def prepare_semantic_id_fold(
    semantic_train_file: Path | str,
    fold_manifest: Path | str,
    output_dir: Path | str,
    *,
    fold: int,
    max_per_class: int | None = None,
) -> dict[str, object]:
    """Materialize one Semantic-ID teacher split from the frozen fold manifest."""

    semantic_train_file = Path(semantic_train_file)
    fold_manifest = Path(fold_manifest)
    output_dir = Path(output_dir)
    rows = pq.read_table(semantic_train_file).to_pylist()
    manifest_rows = pq.read_table(fold_manifest).to_pylist()
    manifest = {str(row["source_sample_id"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Fold manifest has duplicate source_sample_id values")
    row_ids = [str(row["source_sample_id"]) for row in rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(manifest):
        raise ValueError("Semantic-ID Train and fold manifest identities differ")
    for row in rows:
        expected = manifest[str(row["source_sample_id"])]
        for column in ("user_id", "target_subcategory"):
            if row[column] != expected[column]:
                raise ValueError(
                    f"{row['source_sample_id']}: fold mismatch in {column}"
                )
        messages = row["messages"]
        if len(messages) != 2 or not str(messages[1]["content"]).startswith(
            "<interest_"
        ):
            raise ValueError(f"{row['source_sample_id']}: not Semantic-ID Train")

    heldout = _select_per_class(
        (
            row
            for row in rows
            if int(manifest[str(row["source_sample_id"])]["fold"]) == fold
        ),
        max_per_class,
    )
    train = _select_per_class(
        (
            row
            for row in rows
            if int(manifest[str(row["source_sample_id"])]["fold"]) != fold
        ),
        max_per_class,
    )
    labels = {str(row["target_subcategory"]) for row in rows}
    if {str(row["target_subcategory"]) for row in train} != labels:
        raise ValueError("Teacher Train does not cover every class")
    if {str(row["target_subcategory"]) for row in heldout} != labels:
        raise ValueError("Held-out split does not cover every class")
    train_path = output_dir / "train.parquet"
    heldout_path = output_dir / "heldout.parquet"
    _atomic_write_table(train_path, train, pq.ParquetFile(semantic_train_file).schema_arrow)
    _atomic_write_table(
        heldout_path, heldout, pq.ParquetFile(semantic_train_file).schema_arrow
    )
    summary = {
        "fold": fold,
        "classes": len(labels),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_users": len({str(row["user_id"]) for row in train}),
        "heldout_users": len({str(row["user_id"]) for row in heldout}),
        "overlapping_source_ids": len(
            {str(row["source_sample_id"]) for row in train}
            & {str(row["source_sample_id"]) for row in heldout}
        ),
        "max_per_class": max_per_class,
        "source_sha256": {
            "semantic_train": sha256_file(semantic_train_file),
            "fold_manifest": sha256_file(fold_manifest),
        },
        "output_sha256": {
            "train": sha256_file(train_path),
            "heldout": sha256_file(heldout_path),
        },
    }
    _atomic_write_json(output_dir / "split_summary.json", summary)
    return summary


def prior_adjusted_score_rows(
    restricted_logits: np.ndarray,
    target_offsets: Sequence[int],
    identities: Sequence[dict[str, object]],
    *,
    labels: Sequence[str],
    tokens: Sequence[str],
    fold: int,
    teacher_identity: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Remove a fold teacher's marginal class preference from 108-way logits."""

    logits = np.asarray(restricted_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape != (len(identities), len(labels)):
        raise ValueError("restricted_logits shape does not match rows/classes")
    if len(tokens) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("labels/tokens must be unique and aligned")
    if len(target_offsets) != len(identities):
        raise ValueError("target offsets do not align with identities")
    if not np.isfinite(logits).all():
        raise ValueError("restricted_logits contains a non-finite value")
    shifted = logits - logits.max(axis=1, keepdims=True)
    raw_exp = np.exp(shifted)
    log_probabilities = shifted - np.log(
        raw_exp.sum(axis=1, keepdims=True)
    )
    probabilities = np.exp(log_probabilities)
    prior = probabilities.mean(axis=0)
    if not np.isfinite(prior).all() or np.any(prior <= 0.0):
        raise ValueError("Fold output prior must be finite and positive")
    adjusted = log_probabilities - np.log(prior)[None, :]
    adjusted -= adjusted.max(axis=1, keepdims=True)
    adjusted_exp = np.exp(adjusted)
    adjusted_logprobabilities = adjusted - np.log(
        adjusted_exp.sum(axis=1, keepdims=True)
    )

    output: list[dict[str, object]] = []
    for index, (row, offset) in enumerate(zip(identities, target_offsets)):
        if not 0 <= int(offset) < len(labels):
            raise ValueError(f"Invalid target offset: {offset}")
        offset = int(offset)
        alternatives = np.delete(adjusted_logprobabilities[index], offset)
        output.append(
            {
                "source_sample_id": str(row["source_sample_id"]),
                "user_id": str(row["user_id"]),
                "target_subcategory": str(row["target_subcategory"]),
                "fold": fold,
                "target_token": str(tokens[offset]),
                "target_raw_logprob": float(log_probabilities[index, offset]),
                "fold_target_prior": float(prior[offset]),
                "prior_adjusted_target_logprob": float(
                    adjusted_logprobabilities[index, offset]
                ),
                "prior_adjusted_nll": float(
                    -adjusted_logprobabilities[index, offset]
                ),
                "prior_adjusted_margin": float(
                    alternatives.max() - adjusted_logprobabilities[index, offset]
                ),
                "teacher_identity": teacher_identity,
            }
        )
    summary = {
        "rows": len(output),
        "classes": len(labels),
        "fold": fold,
        "fold_prior_sum": float(prior.sum()),
        "fold_prior_min": float(prior.min()),
        "fold_prior_max": float(prior.max()),
        "prior_adjusted_nll_min": min(float(row["prior_adjusted_nll"]) for row in output),
        "prior_adjusted_nll_mean": sum(float(row["prior_adjusted_nll"]) for row in output) / len(output),
        "prior_adjusted_nll_max": max(float(row["prior_adjusted_nll"]) for row in output),
    }
    return output, summary


def score_semantic_id_oof(
    *,
    model_name: str,
    adapter_path: Path | str,
    heldout_file: Path | str,
    mapping_file: Path | str,
    output_dir: Path | str,
    fold: int,
    batch_size: int = 8,
    expected_token_count: int = 108,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    heldout_file = Path(heldout_file)
    mapping_file = Path(mapping_file)
    output_dir = Path(output_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    spec = validate_saved_interest_tokens(
        tokenizer, mapping_file, expected_count=expected_token_count
    )
    mapping = load_interest_token_mapping(mapping_file)
    labels = [str(row["target_subcategory"]) for row in mapping]
    tokens = [str(row["interest_token"]) for row in mapping]
    label_offset = {label: index for index, label in enumerate(labels)}
    rows = pq.read_table(heldout_file).to_pylist()
    source_ids = [str(row["source_sample_id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)) or not rows:
        raise ValueError("Held-out rows must have unique non-empty identities")
    target_offsets = [label_offset[str(row["target_subcategory"])] for row in rows]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    semantic_base_audit = prepare_semantic_adapter_base(model, spec)
    model = PeftModel.from_pretrained(model, adapter_path)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    batches: list[np.ndarray] = []
    token_index = torch.tensor(spec.token_ids, dtype=torch.long, device=device)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            apply_chat_template(
                tokenizer,
                [{"role": "user", "content": str(row["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, -1, :]
            restricted = logits.index_select(1, token_index).float().cpu().numpy()
        batches.append(restricted)
    restricted_logits = np.concatenate(batches, axis=0)
    identity_payload = {
        "scorer_version": SCORER_VERSION,
        "model_name": model_name,
        "adapter_sha256": sha256_directory(adapter_path),
        "heldout_sha256": sha256_file(heldout_file),
        "mapping_sha256": sha256_file(mapping_file),
        "fold": fold,
        "rows": len(rows),
        "labels": labels,
        "tokens": tokens,
        "semantic_initialization_version": SEMANTIC_INITIALIZATION_VERSION,
        "semantic_base_tied_embeddings": bool(
            semantic_base_audit["tie_word_embeddings"]
        ),
    }
    teacher_identity = canonical_sha256(identity_payload)
    scored, summary = prior_adjusted_score_rows(
        restricted_logits,
        target_offsets,
        rows,
        labels=labels,
        tokens=tokens,
        fold=fold,
        teacher_identity=teacher_identity,
    )
    score_path = output_dir / "oof_scores.parquet"
    _atomic_write_table(score_path, scored, SCORE_SCHEMA)
    result = {
        **identity_payload,
        **summary,
        "teacher_identity": teacher_identity,
        "oof_scores_sha256": sha256_file(score_path),
    }
    _atomic_write_json(output_dir / "score_summary.json", result)
    return result


def compute_fold_global_weights(
    scores: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_fold: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in scores:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"Duplicate OOF score: {source_id}")
        seen.add(source_id)
        difficulty = float(row["prior_adjusted_nll"])
        margin = float(row["prior_adjusted_margin"])
        if not math.isfinite(difficulty) or not math.isfinite(margin):
            raise ValueError(f"Non-finite OOF score: {source_id}")
        by_fold[int(row["fold"])].append(row)
    staged: list[tuple[dict[str, object], int, float, float]] = []
    for fold, fold_rows in sorted(by_fold.items()):
        ordered = sorted(
            fold_rows,
            key=lambda row: (
                float(row["prior_adjusted_nll"]),
                str(row["source_sample_id"]),
            ),
        )
        if len(ordered) < 2:
            raise ValueError(f"Fold {fold} needs at least two scored rows")
        denominator = len(ordered) - 1
        for rank, row in enumerate(ordered):
            quantile = rank / denominator
            staged.append((row, rank, quantile, 1.0 / (1.0 + quantile)))
    # Keep the normalization bit-stable between the local Python used to
    # freeze weights and the Python 3.10 AMLT worker that audits them.
    global_mean = math.fsum(item[3] for item in staged) / len(staged)
    output = []
    for row, rank, quantile, raw_weight in staged:
        output.append(
            {
                "source_sample_id": str(row["source_sample_id"]),
                "user_id": str(row["user_id"]),
                "target_subcategory": str(row["target_subcategory"]),
                "fold": int(row["fold"]),
                "prior_adjusted_nll": float(row["prior_adjusted_nll"]),
                "prior_adjusted_margin": float(row["prior_adjusted_margin"]),
                "fold_global_rank": rank,
                "fold_global_difficulty_quantile": quantile,
                "raw_weight": raw_weight,
                "sample_weight": raw_weight / global_mean,
            }
        )
    return sorted(output, key=lambda row: str(row["source_sample_id"]))


def compute_within_class_weights(
    scores: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Apply the original Cross-Fitted within-class rank recipe.

    Each target class is ranked over all of its OOF rows, irrespective of the
    teacher fold. The rank quantile is converted to ``1/(1+q)`` and then
    normalized to mean one within that class.
    """

    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in scores:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"Duplicate OOF score: {source_id}")
        seen.add(source_id)
        difficulty = float(row["prior_adjusted_nll"])
        margin = float(row["prior_adjusted_margin"])
        if not math.isfinite(difficulty) or not math.isfinite(margin):
            raise ValueError(f"Non-finite OOF score: {source_id}")
        by_class[str(row["target_subcategory"])].append(row)

    output: list[dict[str, object]] = []
    for label, class_rows in sorted(by_class.items()):
        ordered = sorted(
            class_rows,
            key=lambda row: (
                float(row["prior_adjusted_nll"]),
                str(row["source_sample_id"]),
            ),
        )
        if len(ordered) < 2:
            raise ValueError(f"Class {label} needs at least two scored rows")
        denominator = len(ordered) - 1
        staged: list[tuple[dict[str, object], int, float, float]] = []
        for rank, row in enumerate(ordered):
            quantile = rank / denominator
            staged.append((row, rank, quantile, 1.0 / (1.0 + quantile)))
        # ``sum`` changed its floating-point reduction algorithm in Python
        # 3.12. Use ``fsum`` so weights generated by the local Python 3.13
        # environment are bit-stable with the Python 3.10 AMLT worker audit.
        class_mean = math.fsum(item[3] for item in staged) / len(staged)
        for row, rank, quantile, raw_weight in staged:
            output.append(
                {
                    "source_sample_id": str(row["source_sample_id"]),
                    "user_id": str(row["user_id"]),
                    "target_subcategory": label,
                    "fold": int(row["fold"]),
                    "prior_adjusted_nll": float(row["prior_adjusted_nll"]),
                    "prior_adjusted_margin": float(row["prior_adjusted_margin"]),
                    "within_class_rank": rank,
                    "within_class_difficulty_quantile": quantile,
                    "raw_weight": raw_weight,
                    "sample_weight": raw_weight / class_mean,
                }
            )
    return sorted(output, key=lambda row: str(row["source_sample_id"]))


def compute_frontier_weights(
    scores: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Emphasize the within-class learnability frontier.

    The easiest and most difficult OOF examples receive the same floor weight,
    while the middle difficulty quantile receives twice that raw weight. The
    intervention is symmetric and each target class is normalized to mean one.
    """

    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in scores:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"Duplicate OOF score: {source_id}")
        seen.add(source_id)
        difficulty = float(row["prior_adjusted_nll"])
        margin = float(row["prior_adjusted_margin"])
        if not math.isfinite(difficulty) or not math.isfinite(margin):
            raise ValueError(f"Non-finite OOF score: {source_id}")
        by_class[str(row["target_subcategory"])].append(row)

    output: list[dict[str, object]] = []
    for label, class_rows in sorted(by_class.items()):
        ordered = sorted(
            class_rows,
            key=lambda row: (
                float(row["prior_adjusted_nll"]),
                str(row["source_sample_id"]),
            ),
        )
        if len(ordered) < 3:
            raise ValueError(f"Class {label} needs at least three scored rows")
        denominator = len(ordered) - 1
        staged: list[tuple[dict[str, object], int, float, float]] = []
        for rank, row in enumerate(ordered):
            quantile = rank / denominator
            raw_weight = 0.5 + min(quantile, 1.0 - quantile)
            staged.append((row, rank, quantile, raw_weight))
        class_mean = math.fsum(item[3] for item in staged) / len(staged)
        for row, rank, quantile, raw_weight in staged:
            output.append(
                {
                    "source_sample_id": str(row["source_sample_id"]),
                    "user_id": str(row["user_id"]),
                    "target_subcategory": label,
                    "fold": int(row["fold"]),
                    "prior_adjusted_nll": float(row["prior_adjusted_nll"]),
                    "prior_adjusted_margin": float(row["prior_adjusted_margin"]),
                    "within_class_rank": rank,
                    "within_class_difficulty_quantile": quantile,
                    "raw_weight": raw_weight,
                    "sample_weight": raw_weight / class_mean,
                }
            )
    return sorted(output, key=lambda row: str(row["source_sample_id"]))


def build_semantic_id_global_weights(
    score_files: Sequence[Path | str],
    train_file: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    score_files = [Path(path) for path in score_files]
    train_file = Path(train_file)
    output_dir = Path(output_dir)
    scores = [
        row
        for path in score_files
        for row in pq.read_table(path).to_pylist()
    ]
    train_rows = pq.read_table(
        train_file, columns=["source_sample_id", "user_id", "target_subcategory"]
    ).to_pylist()
    train_by_id = {str(row["source_sample_id"]): row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError("Train file contains duplicate source identities")
    score_ids = {str(row["source_sample_id"]) for row in scores}
    if score_ids != set(train_by_id) or len(scores) != len(score_ids):
        raise ValueError("OOF scores do not exactly cover the weighted Train file")
    for row in scores:
        expected = train_by_id[str(row["source_sample_id"])]
        for column in ("user_id", "target_subcategory"):
            if row[column] != expected[column]:
                raise ValueError(f"{row['source_sample_id']}: mismatch in {column}")
    weights = compute_fold_global_weights(scores)
    weight_path = output_dir / "sample_weights.parquet"
    _atomic_write_table(weight_path, weights, WEIGHT_SCHEMA)
    values = [float(row["sample_weight"]) for row in weights]
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in weights:
        by_class[str(row["target_subcategory"])].append(float(row["sample_weight"]))
    class_means = {
        label: math.fsum(v) / len(v) for label, v in sorted(by_class.items())
    }
    summary = {
        "weight_version": WEIGHT_VERSION,
        "rows": len(weights),
        "classes": len(by_class),
        "fold_counts": dict(sorted(Counter(str(row["fold"]) for row in weights).items())),
        "global_weight_mean": math.fsum(values) / len(values),
        "global_weight_min": min(values),
        "global_weight_max": max(values),
        "global_weight_max_min_ratio": max(values) / min(values),
        "class_weight_mean_min": min(class_means.values()),
        "class_weight_mean_max": max(class_means.values()),
        "class_weight_means_all_one": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in class_means.values()
        ),
        "formula": {
            "quantile": "fold-global rank/(fold_rows-1)",
            "raw_weight": "1/(1+quantile)",
            "sample_weight": "raw_weight/global mean(raw_weight)",
            "class_normalization": False,
        },
        "input_sha256": {
            "train_file": sha256_file(train_file),
            **{str(path): sha256_file(path) for path in score_files},
        },
        "output_sha256": {"sample_weights": sha256_file(weight_path)},
    }
    _atomic_write_json(output_dir / "weight_summary.json", summary)
    return summary


def build_semantic_id_class_weights(
    score_files: Sequence[Path | str],
    train_file: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    score_files = [Path(path) for path in score_files]
    train_file = Path(train_file)
    output_dir = Path(output_dir)
    scores = [
        row
        for path in score_files
        for row in pq.read_table(path).to_pylist()
    ]
    train_rows = pq.read_table(
        train_file, columns=["source_sample_id", "user_id", "target_subcategory"]
    ).to_pylist()
    train_by_id = {str(row["source_sample_id"]): row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError("Train file contains duplicate source identities")
    score_ids = {str(row["source_sample_id"]) for row in scores}
    if score_ids != set(train_by_id) or len(scores) != len(score_ids):
        raise ValueError("OOF scores do not exactly cover the weighted Train file")
    for row in scores:
        expected = train_by_id[str(row["source_sample_id"])]
        for column in ("user_id", "target_subcategory"):
            if row[column] != expected[column]:
                raise ValueError(f"{row['source_sample_id']}: mismatch in {column}")

    weights = compute_within_class_weights(scores)
    weight_path = output_dir / "sample_weights.parquet"
    _atomic_write_table(weight_path, weights, CLASS_WEIGHT_SCHEMA)
    values = [float(row["sample_weight"]) for row in weights]
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in weights:
        by_class[str(row["target_subcategory"])].append(row)
    class_means = {
        label: math.fsum(float(row["sample_weight"]) for row in class_rows)
        / len(class_rows)
        for label, class_rows in sorted(by_class.items())
    }
    class_counts = {
        label: len(class_rows) for label, class_rows in sorted(by_class.items())
    }
    summary = {
        "weight_version": CLASS_WEIGHT_VERSION,
        "rows": len(weights),
        "classes": len(by_class),
        "class_counts": class_counts,
        "fold_counts": dict(
            sorted(Counter(str(row["fold"]) for row in weights).items())
        ),
        "global_weight_mean": math.fsum(values) / len(values),
        "global_weight_min": min(values),
        "global_weight_max": max(values),
        "global_weight_max_min_ratio": max(values) / min(values),
        "class_weight_mean_min": min(class_means.values()),
        "class_weight_mean_max": max(class_means.values()),
        "class_weight_means_all_one": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in class_means.values()
        ),
        "formula": {
            "quantile": "within-class rank/(class_rows-1) over all OOF rows",
            "raw_weight": "1/(1+quantile)",
            "sample_weight": "raw_weight/within-class mean(raw_weight)",
            "rank_scope": "target_subcategory",
            "class_normalization": True,
        },
        "input_sha256": {
            "train_file": sha256_file(train_file),
            **{str(path): sha256_file(path) for path in score_files},
        },
        "output_sha256": {"sample_weights": sha256_file(weight_path)},
    }
    _atomic_write_json(output_dir / "weight_summary.json", summary)
    return summary


def build_semantic_id_frontier_weights(
    score_files: Sequence[Path | str],
    train_file: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    """Build frontier weights for an exact Train subset from full OOF scores."""

    score_files = [Path(path) for path in score_files]
    train_file = Path(train_file)
    output_dir = Path(output_dir)
    scores = [
        row
        for path in score_files
        for row in pq.read_table(path).to_pylist()
    ]
    all_score_ids = [str(row["source_sample_id"]) for row in scores]
    if len(all_score_ids) != len(set(all_score_ids)):
        raise ValueError("OOF scores contain duplicate source identities")
    score_by_id = {
        str(row["source_sample_id"]): row for row in scores
    }
    train_rows = pq.read_table(
        train_file,
        columns=["source_sample_id", "user_id", "target_subcategory"],
    ).to_pylist()
    train_by_id = {str(row["source_sample_id"]): row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError("Train file contains duplicate source identities")
    missing = set(train_by_id).difference(score_by_id)
    if missing:
        raise ValueError(
            f"OOF scores do not cover {len(missing)} weighted Train rows"
        )
    selected_scores = [score_by_id[source_id] for source_id in train_by_id]
    for row in selected_scores:
        expected = train_by_id[str(row["source_sample_id"])]
        for column in ("user_id", "target_subcategory"):
            if row[column] != expected[column]:
                raise ValueError(
                    f"{row['source_sample_id']}: mismatch in {column}"
                )

    weights = compute_frontier_weights(selected_scores)
    weight_path = output_dir / "sample_weights.parquet"
    _atomic_write_table(weight_path, weights, CLASS_WEIGHT_SCHEMA)
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in weights:
        by_class[str(row["target_subcategory"])].append(row)
    values = [float(row["sample_weight"]) for row in weights]
    class_means = {
        label: math.fsum(float(row["sample_weight"]) for row in class_rows)
        / len(class_rows)
        for label, class_rows in sorted(by_class.items())
    }
    summary = {
        "weight_version": FRONTIER_WEIGHT_VERSION,
        "rows": len(weights),
        "classes": len(by_class),
        "class_counts": {
            label: len(class_rows)
            for label, class_rows in sorted(by_class.items())
        },
        "fold_counts": dict(
            sorted(Counter(str(row["fold"]) for row in weights).items())
        ),
        "global_weight_mean": math.fsum(values) / len(values),
        "global_weight_min": min(values),
        "global_weight_max": max(values),
        "global_weight_max_min_ratio": max(values) / min(values),
        "class_weight_mean_min": min(class_means.values()),
        "class_weight_mean_max": max(class_means.values()),
        "class_weight_means_all_one": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in class_means.values()
        ),
        "formula": {
            "quantile": "within-class rank/(class_rows-1)",
            "raw_weight": "0.5 + min(quantile, 1-quantile)",
            "sample_weight": "raw_weight/within-class mean(raw_weight)",
            "rank_scope": "target_subcategory",
            "class_normalization": True,
            "shape": "symmetric triangular learnability frontier",
        },
        "input_sha256": {
            "train_file": sha256_file(train_file),
            **{str(path): sha256_file(path) for path in score_files},
        },
        "output_sha256": {"sample_weights": sha256_file(weight_path)},
    }
    _atomic_write_json(output_dir / "weight_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--semantic-train-file", type=Path, required=True)
    prepare.add_argument("--fold-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--fold", type=int, required=True)
    prepare.add_argument("--max-per-class", type=int)
    score = subparsers.add_parser("score")
    score.add_argument("--model-name", required=True)
    score.add_argument("--adapter-path", type=Path, required=True)
    score.add_argument("--heldout-file", type=Path, required=True)
    score.add_argument("--mapping-file", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--fold", type=int, required=True)
    score.add_argument("--batch-size", type=int, default=8)
    weights = subparsers.add_parser("weights")
    weights.add_argument("--score-files", type=Path, nargs="+", required=True)
    weights.add_argument("--train-file", type=Path, required=True)
    weights.add_argument("--output-dir", type=Path, required=True)
    class_weights = subparsers.add_parser("class-weights")
    class_weights.add_argument("--score-files", type=Path, nargs="+", required=True)
    class_weights.add_argument("--train-file", type=Path, required=True)
    class_weights.add_argument("--output-dir", type=Path, required=True)
    frontier_weights = subparsers.add_parser("frontier-weights")
    frontier_weights.add_argument(
        "--score-files", type=Path, nargs="+", required=True
    )
    frontier_weights.add_argument("--train-file", type=Path, required=True)
    frontier_weights.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_semantic_id_fold(
            args.semantic_train_file,
            args.fold_manifest,
            args.output_dir,
            fold=args.fold,
            max_per_class=args.max_per_class,
        )
    elif args.command == "score":
        result = score_semantic_id_oof(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            heldout_file=args.heldout_file,
            mapping_file=args.mapping_file,
            output_dir=args.output_dir,
            fold=args.fold,
            batch_size=args.batch_size,
        )
    elif args.command == "weights":
        result = build_semantic_id_global_weights(
            args.score_files, args.train_file, args.output_dir
        )
    elif args.command == "class-weights":
        result = build_semantic_id_class_weights(
            args.score_files, args.train_file, args.output_dir
        )
    else:
        result = build_semantic_id_frontier_weights(
            args.score_files, args.train_file, args.output_dir
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
