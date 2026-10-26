from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "solar-gnip-sasrec-full-softmax-v1"
DEFAULT_CONFIG = Path("configs/solar_gnip/sasrec_gnip_v1.json")
_HISTORY_LINE = re.compile(r"^\s*\d+\.\s+\[([^\]]+)\]\s+", re.MULTILINE)


@dataclass(frozen=True)
class Architecture:
    hidden_dimension: int
    blocks: int
    attention_heads: int
    dropout: float
    maximum_history_length: int


@dataclass(frozen=True)
class GNIPExample:
    example_id: str
    user_id: str
    prompt: str
    history: tuple[int, ...]
    positives: tuple[int, ...]


@dataclass(frozen=True)
class GNIPCorpus:
    input_labels: tuple[str, ...]
    labels: tuple[str, ...]
    train: tuple[GNIPExample, ...]
    dev_1k: tuple[GNIPExample, ...]
    dev_full: tuple[GNIPExample, ...]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("SASRec-GNIP requires PyTorch") from error
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> str:
    torch = _torch()
    if requested != "auto":
        return str(torch.device(requested))
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_config(
    path_or_config: Path | str | Mapping[str, object],
) -> tuple[dict[str, object], Path | None]:
    if isinstance(path_or_config, Mapping):
        return dict(path_or_config), None
    path = Path(path_or_config)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SASRec-GNIP config must be a JSON object")
    return value, path


def _parse_prompt_history(prompt: str) -> tuple[str, ...]:
    values = tuple(match.group(1).strip() for match in _HISTORY_LINE.finditer(prompt))
    if not values:
        raise ValueError("prompt contains no numbered [Interest] History rows")
    return values


def _labels_from_train(path: Path) -> tuple[str, ...]:
    values = pq.read_table(path, columns=["target_subcategory"])[
        "target_subcategory"
    ].to_pylist()
    labels = tuple(sorted({str(value) for value in values}))
    if not labels:
        raise ValueError(f"{path}: no target labels")
    return labels


def _load_train(
    path: Path,
    label_to_id: Mapping[str, int],
    input_label_to_id: Mapping[str, int],
) -> tuple[GNIPExample, ...]:
    output: list[GNIPExample] = []
    seen_examples: set[str] = set()
    seen_users: set[str] = set()
    columns = [
        "sample_id",
        "user_id",
        "history_length",
        "prompt",
        "target_subcategory",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            example_id = str(row["sample_id"])
            user_id = str(row["user_id"])
            if example_id in seen_examples:
                raise ValueError(f"duplicate Train example {example_id}")
            if user_id in seen_users:
                raise ValueError(f"duplicate Train user {user_id}")
            seen_examples.add(example_id)
            seen_users.add(user_id)
            raw_history = _parse_prompt_history(str(row["prompt"]))
            if len(raw_history) != int(row["history_length"]):
                raise ValueError(f"{example_id}: History length mismatch")
            unknown = sorted(set(raw_history) - set(input_label_to_id))
            if unknown:
                raise ValueError(f"{example_id}: unknown History labels {unknown}")
            target_name = str(row["target_subcategory"])
            if target_name not in label_to_id:
                raise ValueError(f"{example_id}: unknown target {target_name!r}")
            if target_name in raw_history:
                raise ValueError(f"{example_id}: target occurs in History")
            output.append(
                GNIPExample(
                    example_id=example_id,
                    user_id=user_id,
                    prompt=str(row["prompt"]),
                    history=tuple(input_label_to_id[value] for value in raw_history),
                    positives=(label_to_id[target_name],),
                )
            )
    if not output:
        raise ValueError(f"{path}: no Train examples")
    return tuple(output)


def _load_dev(
    path: Path,
    label_to_id: Mapping[str, int],
    input_label_to_id: Mapping[str, int],
) -> tuple[GNIPExample, ...]:
    output: list[GNIPExample] = []
    seen_examples: set[str] = set()
    columns = [
        "impression_id",
        "user_id",
        "history_length",
        "prompt",
        "acceptable_subcategories",
        "history_subcategories",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            example_id = str(row["impression_id"])
            if example_id in seen_examples:
                raise ValueError(f"duplicate Dev impression {example_id}")
            seen_examples.add(example_id)
            raw_history = tuple(str(value) for value in row["history_subcategories"])
            parsed_history = _parse_prompt_history(str(row["prompt"]))
            if raw_history != parsed_history:
                raise ValueError(f"{example_id}: prompt/column History mismatch")
            if len(raw_history) != int(row["history_length"]):
                raise ValueError(f"{example_id}: History length mismatch")
            positives = tuple(dict.fromkeys(str(value) for value in row["acceptable_subcategories"]))
            if not positives:
                raise ValueError(f"{example_id}: empty acceptable target set")
            unknown_history = sorted(set(raw_history) - set(input_label_to_id))
            if unknown_history:
                raise ValueError(f"{example_id}: History labels outside input taxonomy {unknown_history}")
            unknown_targets = sorted(set(positives) - set(label_to_id))
            if unknown_targets:
                raise ValueError(f"{example_id}: targets outside Train output taxonomy {unknown_targets}")
            overlap = sorted(set(raw_history) & set(positives))
            if overlap:
                raise ValueError(f"{example_id}: positives occur in History {overlap}")
            output.append(
                GNIPExample(
                    example_id=example_id,
                    user_id=str(row["user_id"]),
                    prompt=str(row["prompt"]),
                    history=tuple(input_label_to_id[value] for value in raw_history),
                    positives=tuple(label_to_id[value] for value in positives),
                )
            )
    if not output:
        raise ValueError(f"{path}: no Dev examples")
    return tuple(output)


def load_corpus(dataset_dir: Path) -> GNIPCorpus:
    train_path = dataset_dir / "train/behavior_interest/21600.parquet"
    dev_1k_path = dataset_dir / "eval/behavior_interest/dev_1k_grouped.parquet"
    dev_full_path = dataset_dir / "eval/behavior_interest/dev_full_grouped.parquet"
    for path in (train_path, dev_1k_path, dev_full_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = _labels_from_train(train_path)
    label_to_id = {label: index for index, label in enumerate(labels)}
    input_label_values = set(labels)
    for batch in pq.ParquetFile(train_path).iter_batches(columns=["prompt"], batch_size=4096):
        for prompt in batch.column(0).to_pylist():
            input_label_values.update(_parse_prompt_history(str(prompt)))
    for path in (dev_1k_path, dev_full_path):
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["history_subcategories"], batch_size=4096
        ):
            for values in batch.column(0).to_pylist():
                input_label_values.update(str(value) for value in values)
    input_labels = tuple(sorted(input_label_values))
    input_label_to_id = {label: index for index, label in enumerate(input_labels)}
    return GNIPCorpus(
        input_labels=input_labels,
        labels=labels,
        train=_load_train(train_path, label_to_id, input_label_to_id),
        dev_1k=_load_dev(dev_1k_path, label_to_id, input_label_to_id),
        dev_full=_load_dev(dev_full_path, label_to_id, input_label_to_id),
    )


def audit_corpus(
    corpus: GNIPCorpus,
    *,
    dataset_dir: Path,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = dict(expected or {})
    target_counts = Counter(corpus.labels[row.positives[0]] for row in corpus.train)
    train_users = {row.user_id for row in corpus.train}
    dev_users = {row.user_id for row in corpus.dev_full}
    history_lengths = [len(row.history) for row in corpus.train]
    dev_target_occurrences = sum(len(row.positives) for row in corpus.dev_full)
    supported_dev_labels = {
        corpus.labels[target]
        for row in corpus.dev_full
        for target in row.positives
    }
    audit: dict[str, object] = {
        "rule_version": RULE_VERSION,
        "dataset_dir": portable_path(dataset_dir),
        "train_rows": len(corpus.train),
        "train_users": len(train_users),
        "labels": len(corpus.labels),
        "input_labels": len(corpus.input_labels),
        "target_count_min": min(target_counts.values()),
        "target_count_max": max(target_counts.values()),
        "history_length_min": min(history_lengths),
        "history_length_max": max(history_lengths),
        "dev_1k_impressions": len(corpus.dev_1k),
        "dev_full_impressions": len(corpus.dev_full),
        "dev_full_users": len(dev_users),
        "dev_target_occurrences": dev_target_occurrences,
        "dev_supported_interests": len(supported_dev_labels),
        "train_dev_user_overlap": len(train_users & dev_users),
        "source_sha256": {
            "train": _sha256(dataset_dir / "train/behavior_interest/21600.parquet"),
            "dev_1k": _sha256(dataset_dir / "eval/behavior_interest/dev_1k_grouped.parquet"),
            "dev_full": _sha256(dataset_dir / "eval/behavior_interest/dev_full_grouped.parquet"),
        },
    }
    checks = {
        "train_rows": len(corpus.train),
        "train_users": len(train_users),
        "labels": len(corpus.labels),
        "target_count_min": min(target_counts.values()),
        "target_count_max": max(target_counts.values()),
        "maximum_history_length": max(history_lengths),
        "dev_1k_impressions": len(corpus.dev_1k),
        "dev_full_impressions": len(corpus.dev_full),
        "dev_full_users": len(dev_users),
        "dev_target_occurrences": dev_target_occurrences,
        "dev_supported_interests": len(supported_dev_labels),
        "train_dev_user_overlap": len(train_users & dev_users),
    }
    mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in checks.items()
        if key in expected and expected[key] != value
    }
    audit["expected_checks"] = checks
    audit["mismatches"] = mismatches
    audit["passed"] = not mismatches
    if mismatches:
        raise ValueError(f"SASRec-GNIP data gate failed: {mismatches}")
    return audit


def _architecture(value: Mapping[str, object]) -> Architecture:
    result = Architecture(
        hidden_dimension=int(value["hidden_dimension"]),
        blocks=int(value["blocks"]),
        attention_heads=int(value["attention_heads"]),
        dropout=float(value["dropout"]),
        maximum_history_length=int(value["maximum_history_length"]),
    )
    if result.hidden_dimension <= 0 or result.blocks <= 0:
        raise ValueError("hidden_dimension and blocks must be positive")
    if result.attention_heads <= 0 or result.hidden_dimension % result.attention_heads:
        raise ValueError("attention_heads must divide hidden_dimension")
    if not 0.0 <= result.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if result.maximum_history_length <= 0:
        raise ValueError("maximum_history_length must be positive")
    return result


def build_model(
    *, input_label_count: int, label_count: int, architecture: Architecture
) -> Any:
    torch = _torch()
    nn = torch.nn

    class SASRecGNIP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_embeddings = nn.Embedding(
                input_label_count + 1,
                architecture.hidden_dimension,
                padding_idx=0,
            )
            self.position_embeddings = nn.Embedding(architecture.maximum_history_length, architecture.hidden_dimension)
            self.input_dropout = nn.Dropout(architecture.dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=architecture.hidden_dimension,
                nhead=architecture.attention_heads,
                dim_feedforward=architecture.hidden_dimension * 4,
                dropout=architecture.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=architecture.blocks, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(architecture.hidden_dimension)
            self.output = nn.Linear(architecture.hidden_dimension, label_count)
            self.apply(self._initialise)

        @staticmethod
        def _initialise(module: Any) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, input_ids: Any, lengths: Any) -> Any:
            if input_ids.ndim != 2 or lengths.ndim != 1:
                raise ValueError("input_ids must be 2D and lengths must be 1D")
            sequence_length = input_ids.shape[1]
            positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
            hidden = self.item_embeddings(input_ids) + self.position_embeddings(positions)
            hidden = self.input_dropout(hidden)
            padding_mask = input_ids.eq(0)
            causal_mask = torch.triu(
                torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=input_ids.device),
                diagonal=1,
            )
            hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
            last_index = lengths.to(dtype=torch.long) - 1
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), last_index]
            return self.output(self.final_norm(pooled))

    return SASRecGNIP()


def _batch_tensors(
    examples: Sequence[GNIPExample],
    indices: Sequence[int],
    *,
    maximum_history_length: int,
    device: str,
    include_targets: bool,
) -> tuple[Any, Any, Any | None]:
    torch = _torch()
    selected = [examples[int(index)] for index in indices]
    lengths = [len(row.history) for row in selected]
    if not lengths or max(lengths) > maximum_history_length:
        raise ValueError("invalid History length in batch")
    width = max(lengths)
    values = np.zeros((len(selected), width), dtype=np.int64)
    for index, row in enumerate(selected):
        values[index, : len(row.history)] = np.asarray(row.history, dtype=np.int64) + 1
    input_ids = torch.as_tensor(values, dtype=torch.long, device=device)
    length_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
    targets = None
    if include_targets:
        if any(len(row.positives) != 1 for row in selected):
            raise ValueError("training batches require exactly one target")
        targets = torch.as_tensor(
            [row.positives[0] for row in selected], dtype=torch.long, device=device
        )
    return input_ids, length_tensor, targets


def _index_batches(
    count: int,
    *,
    batch_size: int,
    shuffle_seed: int | None,
) -> Iterable[np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


def _train_epoch(
    model: Any,
    examples: Sequence[GNIPExample],
    *,
    optimizer: Any,
    batch_size: int,
    maximum_history_length: int,
    device: str,
    shuffle_seed: int,
) -> float:
    torch = _torch()
    model.train()
    total_loss = 0.0
    total_rows = 0
    for indices in _index_batches(len(examples), batch_size=batch_size, shuffle_seed=shuffle_seed):
        input_ids, lengths, targets = _batch_tensors(
            examples,
            indices,
            maximum_history_length=maximum_history_length,
            device=device,
            include_targets=True,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("non-finite SASRec-GNIP training loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(indices)
        total_rows += len(indices)
    return total_loss / total_rows


def score_examples(
    model: Any,
    examples: Sequence[GNIPExample],
    *,
    batch_size: int,
    maximum_history_length: int,
    device: str,
) -> np.ndarray:
    torch = _torch()
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for indices in _index_batches(len(examples), batch_size=batch_size, shuffle_seed=None):
            input_ids, lengths, _ = _batch_tensors(
                examples,
                indices,
                maximum_history_length=maximum_history_length,
                device=device,
                include_targets=False,
            )
            chunks.append(model(input_ids, lengths).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _ranking(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")


def _single_target_selection_metrics(
    examples: Sequence[GNIPExample], scores: np.ndarray
) -> dict[str, float]:
    ranked = _ranking(scores)
    by_label_recall: dict[int, list[float]] = defaultdict(list)
    by_label_ndcg: dict[int, list[float]] = defaultdict(list)
    for row, order in zip(examples, ranked, strict=True):
        target = row.positives[0]
        positions = np.flatnonzero(order == target)
        if len(positions) != 1:
            raise ValueError("target rank is not unique")
        rank = int(positions[0]) + 1
        by_label_recall[target].append(float(rank <= 20))
        by_label_ndcg[target].append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
    return {
        "macro_recall_at_20": float(np.mean([np.mean(values) for values in by_label_recall.values()])),
        "macro_ndcg_at_10": float(np.mean([np.mean(values) for values in by_label_ndcg.values()])),
    }


def stratified_split(
    examples: Sequence[GNIPExample], *, seed: int, validation_fraction: float
) -> tuple[tuple[GNIPExample, ...], tuple[GNIPExample, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    by_target: dict[int, list[GNIPExample]] = defaultdict(list)
    for row in examples:
        if len(row.positives) != 1:
            raise ValueError("stratified Train split requires single targets")
        by_target[row.positives[0]].append(row)
    fit: list[GNIPExample] = []
    validation: list[GNIPExample] = []
    for target, rows in sorted(by_target.items()):
        ordered = sorted(rows, key=lambda row: row.example_id)
        np.random.default_rng(seed + target).shuffle(ordered)
        validation_count = max(1, int(round(len(ordered) * validation_fraction)))
        if validation_count >= len(ordered):
            raise ValueError(f"target {target} has too few rows for split")
        validation.extend(ordered[:validation_count])
        fit.extend(ordered[validation_count:])
    return tuple(sorted(fit, key=lambda row: row.example_id)), tuple(
        sorted(validation, key=lambda row: row.example_id)
    )


def _is_better(candidate: Mapping[str, float], best: Mapping[str, float] | None) -> bool:
    if best is None:
        return True
    return (
        float(candidate["macro_recall_at_20"]),
        float(candidate["macro_ndcg_at_10"]),
    ) > (
        float(best["macro_recall_at_20"]),
        float(best["macro_ndcg_at_10"]),
    )


def _fit_select_epoch(
    corpus: GNIPCorpus,
    *,
    architecture: Architecture,
    training: Mapping[str, object],
    device: str,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    torch = _torch()
    seed = int(training["seed"])
    fit, validation = stratified_split(
        corpus.train,
        seed=seed,
        validation_fraction=float(training["validation_fraction"]),
    )
    _seed_everything(seed)
    model = build_model(
        input_label_count=len(corpus.input_labels),
        label_count=len(corpus.labels),
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    maximum_epochs = int(training["maximum_epochs"])
    validation_interval = int(training["validation_interval"])
    patience = int(training["patience_validations"])
    batch_size = int(training["batch_size"])
    evaluation_batch_size = int(training["evaluation_batch_size"])
    history: list[dict[str, object]] = []
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        loss = _train_epoch(
            model,
            fit,
            optimizer=optimizer,
            batch_size=batch_size,
            maximum_history_length=architecture.maximum_history_length,
            device=device,
            shuffle_seed=seed + epoch,
        )
        if epoch % validation_interval:
            continue
        scores = score_examples(
            model,
            validation,
            batch_size=evaluation_batch_size,
            maximum_history_length=architecture.maximum_history_length,
            device=device,
        )
        metrics = _single_target_selection_metrics(validation, scores)
        row: dict[str, object] = {"epoch": epoch, "train_loss": loss, **metrics}
        history.append(row)
        print("SASREC_SELECTION " + json.dumps(row, sort_keys=True), flush=True)
        if _is_better(metrics, best_metrics):
            best_epoch = epoch
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if not best_epoch or best_metrics is None:
        raise ValueError("SASRec-GNIP epoch selection produced no checkpoint")
    split_audit = {
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "fit_target_counts": dict(sorted(Counter(corpus.labels[row.positives[0]] for row in fit).items())),
        "validation_target_counts": dict(
            sorted(Counter(corpus.labels[row.positives[0]] for row in validation).items())
        ),
    }
    return best_epoch, history, {"metrics": best_metrics, "split": split_audit}


def _fit_full(
    corpus: GNIPCorpus,
    *,
    architecture: Architecture,
    training: Mapping[str, object],
    selected_epoch: int,
    device: str,
) -> tuple[Any, list[dict[str, object]]]:
    torch = _torch()
    seed = int(training["seed"])
    _seed_everything(seed)
    model = build_model(
        input_label_count=len(corpus.input_labels),
        label_count=len(corpus.labels),
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    losses: list[dict[str, object]] = []
    for epoch in range(1, selected_epoch + 1):
        loss = _train_epoch(
            model,
            corpus.train,
            optimizer=optimizer,
            batch_size=int(training["batch_size"]),
            maximum_history_length=architecture.maximum_history_length,
            device=device,
            shuffle_seed=seed + epoch,
        )
        losses.append({"epoch": epoch, "train_loss": loss})
        if epoch == selected_epoch or epoch % int(training["validation_interval"]) == 0:
            print(
                "SASREC_FULL_RETRAIN "
                + json.dumps(losses[-1], sort_keys=True),
                flush=True,
            )
    return model, losses


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("invalid candidate k")
    order = _ranking(scores)[:, :k]
    values = np.take_along_axis(scores, order, axis=1)
    return order, values


def _candidate_rows(
    examples: Sequence[GNIPExample],
    *,
    labels: Sequence[str],
    input_labels: Sequence[str],
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    split: str,
    in_sample: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    retrieved = 0
    total_targets = 0
    any_hit = 0
    history_overlap_predictions = 0
    for example, indices, scores in zip(examples, top_indices, top_scores, strict=True):
        candidate_set = {int(value) for value in indices.tolist()}
        history_names = {input_labels[value] for value in example.history}
        target_hits = sum(target in candidate_set for target in example.positives)
        retrieved += target_hits
        total_targets += len(example.positives)
        any_hit += int(target_hits > 0)
        history_overlap = sum(
            labels[int(value)] in history_names for value in indices.tolist()
        )
        history_overlap_predictions += history_overlap
        rows.append(
            {
                "example_id": example.example_id,
                "user_id": example.user_id,
                "split": split,
                "prompt": example.prompt,
                "history_interest_ids": [f"H{value:03d}" for value in example.history],
                "history_interests": [input_labels[value] for value in example.history],
                "positive_interest_ids": [f"C{value:03d}" for value in example.positives],
                "positive_interests": [labels[value] for value in example.positives],
                "candidate_interest_ids": [f"C{int(value):03d}" for value in indices],
                "candidate_interests": [labels[int(value)] for value in indices],
                "candidate_scores": [float(value) for value in scores],
                "target_in_candidates": bool(target_hits),
                "candidate_recall_at_20": target_hits / len(example.positives),
                "history_overlap_count": history_overlap,
                "in_sample_candidates": in_sample,
            }
        )
    metrics = {
        "examples": len(rows),
        "target_occurrences": total_targets,
        "candidate_recall_at_20": retrieved / total_targets,
        "example_hit_at_20": any_hit / len(rows),
        "history_overlap_predictions": history_overlap_predictions,
        "in_sample_candidates": in_sample,
    }
    return rows, metrics


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows)), path, compression="zstd", row_group_size=4096)


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    backup = output.with_name(output.name + ".old")
    if backup.exists():
        raise FileExistsError(f"stale backup blocks publication: {backup}")
    if output.exists():
        os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output)


def run_smoke(
    path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    output_dir: Path,
    requested_device: str = "auto",
    overwrite: bool = False,
) -> Path:
    config, config_path = _load_config(path_or_config)
    dataset_dir = Path(str(config["dataset_dir"]))
    corpus = load_corpus(dataset_dir)
    audit = audit_corpus(corpus, dataset_dir=dataset_dir, expected=dict(config["expected_counts"]))
    architecture = _architecture(dict(config["architecture"]))
    device = _resolve_device(requested_device)
    torch = _torch()
    seed = int(dict(config["training"])["seed"])
    _seed_everything(seed)
    model = build_model(
        input_label_count=len(corpus.input_labels),
        label_count=len(corpus.labels),
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(dict(config["training"])["learning_rate"]))
    examples = corpus.train[:32]
    input_ids, lengths, targets = _batch_tensors(
        examples,
        np.arange(len(examples)),
        maximum_history_length=architecture.maximum_history_length,
        device=device,
        include_targets=True,
    )
    model.train()
    before = model(input_ids, lengths)
    loss = torch.nn.functional.cross_entropy(before, targets)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_logits = model(input_ids, lengths).float().cpu()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        checkpoint = staging / "checkpoint.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": asdict(architecture),
                "input_labels": corpus.input_labels,
                "labels": corpus.labels,
            },
            checkpoint,
        )
        reloaded = build_model(
            input_label_count=len(corpus.input_labels),
            label_count=len(corpus.labels),
            architecture=architecture,
        ).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        reloaded.load_state_dict(payload["state_dict"], strict=True)
        reloaded.eval()
        with torch.no_grad():
            actual_logits = reloaded(input_ids, lengths).float().cpu()
        reload_max_abs_error = float((expected_logits - actual_logits).abs().max())
        if reload_max_abs_error != 0.0:
            raise ValueError(f"checkpoint reload mismatch: {reload_max_abs_error}")
        _write_json(staging / "data_audit.json", audit)
        _write_json(
            staging / "smoke.json",
            {
                "rule_version": RULE_VERSION,
                "config_path": portable_path(config_path) if config_path else None,
                "device": device,
                "examples": len(examples),
                "loss": float(loss.detach().cpu()),
                "finite_loss": bool(torch.isfinite(loss)),
                "logits_shape": list(expected_logits.shape),
                "reload_max_abs_error": reload_max_abs_error,
                "passed": True,
            },
        )
        _publish(staging, output_dir, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def run_experiment(
    path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    output_dir: Path,
    requested_device: str = "auto",
    overwrite: bool = False,
) -> Path:
    config, config_path = _load_config(path_or_config)
    dataset_dir = Path(str(config["dataset_dir"]))
    corpus = load_corpus(dataset_dir)
    audit = audit_corpus(corpus, dataset_dir=dataset_dir, expected=dict(config["expected_counts"]))
    architecture = _architecture(dict(config["architecture"]))
    training = dict(config["training"])
    device = _resolve_device(requested_device)
    selected_epoch, selection_history, selection = _fit_select_epoch(
        corpus, architecture=architecture, training=training, device=device
    )
    model, full_losses = _fit_full(
        corpus,
        architecture=architecture,
        training=training,
        selected_epoch=selected_epoch,
        device=device,
    )
    split_examples = {
        "train": corpus.train,
        "dev_1k": corpus.dev_1k,
        "dev_full": corpus.dev_full,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        _write_json(staging / "data_audit.json", audit)
        _write_json(staging / "resolved_config.json", config)
        _write_json(staging / "selection.json", {"selected_epoch": selected_epoch, **selection})
        _write_parquet(staging / "selection_history.parquet", selection_history)
        _write_parquet(staging / "full_training_history.parquet", full_losses)
        vocabulary_rows = [
            {"interest_id": f"C{index:03d}", "class_index": index, "interest": label}
            for index, label in enumerate(corpus.labels)
        ]
        _write_parquet(staging / "interest_vocabulary.parquet", vocabulary_rows)
        input_vocabulary_rows = [
            {"history_interest_id": f"H{index:03d}", "input_index": index, "interest": label}
            for index, label in enumerate(corpus.input_labels)
        ]
        _write_parquet(
            staging / "history_interest_vocabulary.parquet",
            input_vocabulary_rows,
        )
        torch = _torch()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": asdict(architecture),
                "input_labels": corpus.input_labels,
                "labels": corpus.labels,
                "selected_epoch": selected_epoch,
                "rule_version": RULE_VERSION,
            },
            staging / "model.pt",
        )
        candidate_metrics: dict[str, object] = {}
        for split, examples in split_examples.items():
            scores = score_examples(
                model,
                examples,
                batch_size=int(training["evaluation_batch_size"]),
                maximum_history_length=architecture.maximum_history_length,
                device=device,
            )
            indices, values = _top_k(scores, int(config["candidate_k"]))
            rows, metrics = _candidate_rows(
                examples,
                labels=corpus.labels,
                input_labels=corpus.input_labels,
                top_indices=indices,
                top_scores=values,
                split=split,
                in_sample=split == "train",
            )
            _write_parquet(staging / f"candidates/{split}.parquet", rows)
            _write_json(staging / f"metrics/{split}.json", metrics)
            candidate_metrics[split] = metrics
        manifest_paths = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "rule_version": RULE_VERSION,
            "config_path": portable_path(config_path) if config_path else None,
            "device": device,
            "selected_epoch": selected_epoch,
            "candidate_metrics": candidate_metrics,
            "artifact_sha256": {
                str(path.relative_to(staging)): _sha256(path) for path in manifest_paths
            },
        }
        _write_json(staging / "manifest.json", manifest)
        _publish(staging, output_dir, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export SASRec-GNIP candidates")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    runner = run_smoke if args.smoke else run_experiment
    output = runner(
        args.config,
        output_dir=args.output_dir,
        requested_device=args.device,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
