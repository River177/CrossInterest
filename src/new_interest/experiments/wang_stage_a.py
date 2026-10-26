from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from new_interest.data.build_wang_stage_a_dataset import RESPONSE_PREFIX


CONFIG_VERSION = "wang-stage-a-matrix-v1"
MODEL_KEYS = ("qwen35", "qwen2")
DATASET_KEYS = ("arts", "grocery", "mind")
TRAINING_CONDITIONS = ("balanced", "random")
EVALUATION_VIEWS = ("supported", "full")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: Path | str) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("config_version") != CONFIG_VERSION:
        raise ValueError(f"Unsupported Wang Stage-A matrix: {path}")
    project_root = path.parents[2]
    return project_root, raw


def _validate_choice(value: str, choices: tuple[str, ...], *, name: str) -> None:
    if value not in choices:
        raise ValueError(f"Unsupported {name}: {value}; expected one of {choices}")


def _data_path(
    project_root: Path,
    matrix: dict[str, Any],
    relative: str,
) -> Path:
    return (project_root / str(matrix["data_root"]) / relative).resolve()


def audit_matrix(path: Path | str) -> dict[str, Any]:
    project_root, matrix = load_matrix(path)
    method = matrix["method"]
    if method["scope"] != "high-level language policy only":
        raise ValueError("Wang matrix unexpectedly includes Stage B")
    if method["response_prefix"] != RESPONSE_PREFIX:
        raise ValueError("Wang response prefix differs from the dataset builder")
    if int(method["context_clusters"]) != 2:
        raise ValueError("Wang Stage A must use K=2")
    pairs_per_label = int(method["pairs_per_label"])
    training = matrix["training"]
    expected_batch = (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
    )
    if expected_batch != int(training["effective_global_batch_size"]):
        raise ValueError("Configured single-GPU effective batch is inconsistent")
    if bool(training["enable_thinking"]):
        raise ValueError("Wang controlled generation must disable thinking output")
    if bool(training["gradient_checkpointing_use_reentrant"]):
        raise ValueError("Reentrant gradient checkpointing is forbidden")

    payload: dict[str, Any] = {
        "config_version": CONFIG_VERSION,
        "models": {},
        "datasets": {},
    }
    for model_key in MODEL_KEYS:
        model = matrix["models"][model_key]
        for required in ("name", "revision", "requirements", "learning_rate"):
            if not model.get(required):
                raise ValueError(f"{model_key}: missing {required}")
        payload["models"][model_key] = {
            "name": model["name"],
            "revision": model["revision"],
            "requirements": model["requirements"],
        }

    for dataset_key in DATASET_KEYS:
        dataset = matrix["datasets"][dataset_key]
        supported_labels = int(dataset["supported_labels"])
        expected_rows = supported_labels * pairs_per_label
        if int(dataset["train_rows"]) != expected_rows:
            raise ValueError(f"{dataset_key}: train row count is not labels*10")
        dataset_payload: dict[str, Any] = {"files": {}}
        for condition in TRAINING_CONDITIONS:
            spec = dataset["train"][condition]
            file_path = _data_path(project_root, matrix, spec["path"])
            if sha256_file(file_path) != str(spec["sha256"]):
                raise ValueError(f"{dataset_key}/{condition}: frozen hash mismatch")
            rows = pq.read_table(file_path).to_pylist()
            if len(rows) != expected_rows:
                raise ValueError(f"{dataset_key}/{condition}: row count mismatch")
            for row in rows:
                if len(row["context_clusters"]) != 2:
                    raise ValueError(f"{dataset_key}/{condition}: non-K=2 prompt")
                expected_response = f"{RESPONSE_PREFIX}{row['target_subcategory']}"
                if row["response"] != expected_response:
                    raise ValueError(f"{dataset_key}/{condition}: response mismatch")
                if row["messages"][1]["content"] != expected_response:
                    raise ValueError(f"{dataset_key}/{condition}: message mismatch")
            counts = Counter(str(row["target_subcategory"]) for row in rows)
            if condition == "balanced":
                if len(counts) != supported_labels or set(counts.values()) != {
                    pairs_per_label
                }:
                    raise ValueError(f"{dataset_key}: balanced data are not 10/label")
            dataset_payload["files"][f"train_{condition}"] = {
                "path": str(file_path),
                "rows": len(rows),
                "sha256": spec["sha256"],
                "observed_labels": len(counts),
            }
        for key, spec in dataset.items():
            if key in {"train", "train_rows", "supported_labels", "full_labels"}:
                continue
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            file_path = _data_path(project_root, matrix, spec["path"])
            if sha256_file(file_path) != str(spec["sha256"]):
                raise ValueError(f"{dataset_key}/{key}: frozen hash mismatch")
            rows = pq.ParquetFile(file_path).metadata.num_rows
            if "rows" in spec and rows != int(spec["rows"]):
                raise ValueError(f"{dataset_key}/{key}: row count mismatch")
            dataset_payload["files"][key] = {
                "path": str(file_path),
                "rows": rows,
                "sha256": spec["sha256"],
            }
        payload["datasets"][dataset_key] = dataset_payload
    return payload


def training_config(
    matrix_path: Path | str,
    *,
    model_key: str,
    dataset_key: str,
    condition: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    _validate_choice(model_key, MODEL_KEYS, name="model")
    _validate_choice(dataset_key, DATASET_KEYS, name="dataset")
    _validate_choice(condition, TRAINING_CONDITIONS, name="condition")
    project_root, matrix = load_matrix(matrix_path)
    model = matrix["models"][model_key]
    training = dict(matrix["training"])
    train_spec = matrix["datasets"][dataset_key]["train"][condition]
    train_path = _data_path(project_root, matrix, train_spec["path"])
    training.update(
        {
            "model_name": model["name"],
            "model_revision": model["revision"],
            "train_file": str(train_path),
            "output_dir": str(Path(output_dir)),
            "run_name": (
                f"wang-stage-a-{dataset_key}-{model_key}-{condition}-seed42"
            ),
            "learning_rate": float(model["learning_rate"]),
            "forbidden_trainable_module_substrings": ["vision", "visual"],
        }
    )
    training.pop("effective_global_batch_size", None)
    return training


def evaluation_config(
    matrix_path: Path | str,
    *,
    model_key: str,
    dataset_key: str,
    split: str,
    view: str,
    output_dir: Path | str,
    adapter_path: Path | str | None,
    kind: str,
    top_k: int | None = None,
    cutoffs: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    _validate_choice(model_key, MODEL_KEYS, name="model")
    _validate_choice(dataset_key, DATASET_KEYS, name="dataset")
    _validate_choice(view, EVALUATION_VIEWS, name="evaluation view")
    if kind not in {"generation", "ranking"}:
        raise ValueError(f"Unsupported evaluation kind: {kind}")
    project_root, matrix = load_matrix(matrix_path)
    model = matrix["models"][model_key]
    dataset = matrix["datasets"][dataset_key]
    eval_key = f"{split}_{view}"
    labels_key = f"labels_{view}"
    if eval_key not in dataset:
        raise ValueError(f"{dataset_key} has no {eval_key} split")
    evaluation = dict(matrix["evaluation"])
    config: dict[str, Any] = {
        "model_name": model["name"],
        "model_revision": model["revision"],
        "eval_file": str(
            _data_path(project_root, matrix, dataset[eval_key]["path"])
        ),
        "output_dir": str(Path(output_dir)),
        "response_prefix": matrix["method"]["response_prefix"],
        "enable_thinking": bool(evaluation["enable_thinking"]),
        "max_seq_length": int(evaluation["max_seq_length"]),
        "checkpoint_size": int(evaluation["checkpoint_size"]),
    }
    if adapter_path is not None:
        config["adapter_path"] = str(Path(adapter_path))
    if kind == "generation":
        config.update(
            {
                "labels_file": str(
                    _data_path(project_root, matrix, dataset[labels_key]["path"])
                ),
                "batch_size": int(model["generation_batch_size"]),
                "max_new_tokens": int(evaluation["max_new_tokens"]),
            }
        )
    else:
        resolved_top_k = int(evaluation["top_k"] if top_k is None else top_k)
        resolved_cutoffs = tuple(
            int(value)
            for value in (
                evaluation["cutoffs"] if cutoffs is None else cutoffs
            )
        )
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")
        if (
            not resolved_cutoffs
            or any(value <= 0 for value in resolved_cutoffs)
            or max(resolved_cutoffs) > resolved_top_k
        ):
            raise ValueError("cutoffs must be positive and no larger than top_k")
        config.update(
            {
                "retained_interests_file": str(
                    _data_path(project_root, matrix, dataset[labels_key]["path"])
                ),
                "batch_size": int(model["label_batch_size"]),
                "scoring_backend": evaluation["scoring_backend"],
                "top_k": resolved_top_k,
                "cutoffs": list(resolved_cutoffs),
            }
        )
    return config


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and resolve Wang Stage-A experiment configs."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/wang_stage_a/experiment_matrix_v1.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")

    train_parser = subparsers.add_parser("write-training")
    train_parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    train_parser.add_argument("--dataset", choices=DATASET_KEYS, required=True)
    train_parser.add_argument(
        "--condition", choices=TRAINING_CONDITIONS, required=True
    )
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--destination", type=Path, required=True)

    for kind in ("generation", "ranking"):
        eval_parser = subparsers.add_parser(f"write-{kind}")
        eval_parser.add_argument("--model", choices=MODEL_KEYS, required=True)
        eval_parser.add_argument("--dataset", choices=DATASET_KEYS, required=True)
        eval_parser.add_argument("--split", required=True)
        eval_parser.add_argument("--view", choices=EVALUATION_VIEWS, required=True)
        eval_parser.add_argument("--adapter-path", type=Path)
        eval_parser.add_argument("--output-dir", type=Path, required=True)
        eval_parser.add_argument("--destination", type=Path, required=True)
        if kind == "ranking":
            eval_parser.add_argument("--top-k", type=int)
            eval_parser.add_argument("--cutoffs", type=int, nargs="+")

    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit_matrix(args.matrix), indent=2))
        return
    if args.command == "write-training":
        payload = training_config(
            args.matrix,
            model_key=args.model,
            dataset_key=args.dataset,
            condition=args.condition,
            output_dir=args.output_dir,
        )
    else:
        kind = args.command.removeprefix("write-")
        payload = evaluation_config(
            args.matrix,
            model_key=args.model,
            dataset_key=args.dataset,
            split=args.split,
            view=args.view,
            output_dir=args.output_dir,
            adapter_path=args.adapter_path,
            kind=kind,
            top_k=getattr(args, "top_k", None),
            cutoffs=(
                tuple(args.cutoffs)
                if getattr(args, "cutoffs", None) is not None
                else None
            ),
        )
    _write_config(args.destination, payload)
    print(args.destination)


if __name__ == "__main__":
    main()
