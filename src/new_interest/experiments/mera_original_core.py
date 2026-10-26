from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


CONFIG_VERSION = "mera-original-matrix-v1"
SUPPORTED_DATASETS = ("arts", "grocery")
SUPPORTED_CONTRACTS = ("code-faithful", "paper-spec")
QUOTED_VALUE_RE = re.compile(r'"([^"]+)"')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ResolvedMeRAOriginalConfig:
    config_path: Path
    project_root: Path
    dataset: str
    contract: str
    payload: dict[str, Any]
    source_root: Path
    dataset_root: Path
    source_manifest_path: Path

    @property
    def dataset_config(self) -> dict[str, Any]:
        return dict(self.payload["dataset"])

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.payload["training"])

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.payload["model"])

    @property
    def regularizer(self) -> dict[str, Any]:
        return dict(self.payload["regularizer"])

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.payload["evaluation"])

    def data_path(self, key: str) -> Path:
        if key == "author_predictions":
            return self.source_root / str(self.dataset_config[key])
        return self.dataset_root / str(self.dataset_config[key])

    def identity(self) -> dict[str, Any]:
        manifest = load_json(self.source_manifest_path)
        dataset_manifest = manifest["datasets"][self.dataset]
        implementation_paths = (
            "src/new_interest/experiments/mera_original_core.py",
            "src/new_interest/experiments/audit_mera_original.py",
            "src/new_interest/training/train_mera_original.py",
            "src/new_interest/evaluation/evaluate_mera_original.py",
            "scripts/amlt/sft/run_mera_original.sh",
            "scripts/amlt/sft/install_mera_original_worker_deps.sh",
            "requirements-mera-original-worker.txt",
        )
        implementation = {
            relative: sha256_file(self.project_root / relative)
            for relative in implementation_paths
        }
        return {
            "identity_version": "mera-original-run-identity-v1",
            "upstream_commit": manifest["upstream"]["commit"],
            "model_name": self.model["name"],
            "model_revision": self.model["revision"],
            "dataset": self.dataset,
            "contract": self.contract,
            "source_manifest_sha256": sha256_file(
                self.source_manifest_path
            ),
            "config_sha256": sha256_file(self.config_path),
            "dataset_files": dict(dataset_manifest["files"]),
            "author_predictions_sha256": dataset_manifest[
                "author_predictions"
            ]["sha256"],
            "implementation_sha256": implementation,
            "resolved_config_sha256": canonical_json_sha256(self.payload),
        }


def _resolve_project_path(config_path: Path, value: str) -> tuple[Path, Path]:
    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / value
        if candidate.exists():
            return parent, candidate.resolve()
    project_root = config_path.parents[2]
    return project_root, (project_root / value).resolve()


def load_resolved_config(
    config_path: str | Path,
    dataset: str,
    contract: str,
) -> ResolvedMeRAOriginalConfig:
    config_path = Path(config_path).resolve()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if contract not in SUPPORTED_CONTRACTS:
        raise ValueError(f"Unsupported contract: {contract}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw.get("config_version") != CONFIG_VERSION:
        raise ValueError(f"Unsupported config version: {raw.get('config_version')}")
    project_root, source_root = _resolve_project_path(
        config_path, str(raw["source_root"])
    )
    source_manifest_path = (project_root / raw["source_manifest"]).resolve()
    dataset_config = dict(raw["datasets"][dataset])
    contract_config = dict(raw["contracts"][contract])
    contract_dataset = dict(contract_config.pop("datasets", {}).get(dataset, {}))
    regularizer = {**contract_config, **contract_dataset}
    required_regularizer = {
        "regularizer_semantics",
        "structure_weight",
        "collaborative_weight",
        "negative_samples",
        "cooccurrence_threshold",
    }
    missing = required_regularizer - regularizer.keys()
    if missing:
        raise ValueError(f"Regularizer config missing: {sorted(missing)}")
    payload = {
        "config_version": raw["config_version"],
        "dataset": dataset_config,
        "dataset_name": dataset,
        "contract_name": contract,
        "model": dict(raw["model"]),
        "training": dict(raw["training"]),
        "evaluation": dict(raw["evaluation"]),
        "regularizer": regularizer,
        "source_root": str(raw["source_root"]),
        "source_manifest": str(raw["source_manifest"]),
    }
    return ResolvedMeRAOriginalConfig(
        config_path=config_path,
        project_root=project_root.resolve(),
        dataset=dataset,
        contract=contract,
        payload=payload,
        source_root=source_root,
        dataset_root=(source_root / dataset_config["directory"]).resolve(),
        source_manifest_path=source_manifest_path,
    )


def load_categories(path: Path) -> tuple[str, ...]:
    categories: list[str] = []
    ids: list[int] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected label<TAB>id")
        label, raw_id = parts
        categories.append(label)
        ids.append(int(raw_id))
    if len(categories) != len(set(categories)):
        raise ValueError(f"{path}: category labels are not unique")
    if ids != list(range(len(ids))):
        raise ValueError(f"{path}: category ids are not contiguous from zero")
    return tuple(categories)


def quote_category(category: str) -> str:
    return f'"{category}"'


def parse_quoted_values(text: str) -> tuple[str, ...]:
    return tuple(QUOTED_VALUE_RE.findall(text))


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list")
    required = {"instruction", "input", "output"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"{path}: invalid record at row {index}")
    return rows


def training_prompt(row: Mapping[str, Any], *, include_output: bool = True) -> str:
    output = str(row["output"]) if include_output else ""
    if row["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request. \n\n"
            f"### Instruction:\n{row['instruction']}\n\n"
            f"### Input:\n{row['input']}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.  \n\n"
        f"### Instruction:\n{row['instruction']}\n\n"
        f"### Response:\n{output}"
    )


def inference_text(row: Mapping[str, Any]) -> str:
    history = parse_quoted_values(str(row["input"]))
    if not history:
        raise ValueError("MeRA inference row has no quoted history categories")
    return "".join(quote_category(value) for value in history)


def precision_recall_at_k(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
    *,
    max_k: int = 4,
) -> dict[str, Any]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Predictions and targets must have the same nonzero length")
    precision = [0.0] * max_k
    recall = [0.0] * max_k
    rows: list[dict[str, Any]] = []
    for row_index, (predicted, actual) in enumerate(zip(predictions, targets)):
        predicted = tuple(predicted)
        actual_set = set(actual)
        if len(predicted) < max_k or len(set(predicted[:max_k])) != max_k:
            raise ValueError(f"Invalid Top-{max_k} prediction at row {row_index}")
        if not actual_set:
            raise ValueError(f"Empty target set at row {row_index}")
        row_precision: list[float] = []
        row_recall: list[float] = []
        for offset in range(max_k):
            hits = len(set(predicted[: offset + 1]) & actual_set)
            p_value = hits / (offset + 1)
            r_value = hits / len(actual_set)
            precision[offset] += p_value
            recall[offset] += r_value
            row_precision.append(p_value)
            row_recall.append(r_value)
        rows.append(
            {
                "row_index": row_index,
                "precision_at_k": row_precision,
                "recall_at_k": row_recall,
            }
        )
    count = len(predictions)
    return {
        "examples": count,
        "precision_at_k": [value / count for value in precision],
        "recall_at_k": [value / count for value in recall],
        "per_example": rows,
    }


def evaluate_author_predictions(config: ResolvedMeRAOriginalConfig) -> dict[str, Any]:
    records = load_records(config.data_path("author_predictions"))
    predictions = [
        tuple(
            value if value.startswith('"') else f'"{value.strip(chr(34))}"'
            for value in row["predict"]
        )
        for row in records
    ]
    targets = [
        tuple(quote_category(value) for value in parse_quoted_values(row["output"]))
        for row in records
    ]
    result = precision_recall_at_k(predictions, targets)
    result["precision_at_k_rounded"] = [
        round(value, 4) for value in result["precision_at_k"]
    ]
    result["recall_at_k_rounded"] = [
        round(value, 4) for value in result["recall_at_k"]
    ]
    return result


def audit_dataset(config: ResolvedMeRAOriginalConfig) -> dict[str, Any]:
    manifest = load_json(config.source_manifest_path)
    dataset_manifest = manifest["datasets"][config.dataset]
    for relative, expected_hash in dataset_manifest["files"].items():
        path = config.dataset_root / relative
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    author_path = config.data_path("author_predictions")
    if sha256_file(author_path) != dataset_manifest["author_predictions"]["sha256"]:
        raise ValueError(f"Frozen author prediction hash mismatch: {author_path}")

    categories = load_categories(config.data_path("category_map"))
    category_set = set(categories)
    if len(categories) != int(config.dataset_config["expected_categories"]):
        raise ValueError("Category count mismatch")
    tree = load_json(config.data_path("category_tree"))
    if len(tree) != int(config.dataset_config["expected_parents"]):
        raise ValueError("Parent count mismatch")
    tree_children = [child for children in tree.values() for child in children]
    if len(tree_children) != len(set(tree_children)) or set(tree_children) != category_set:
        raise ValueError("Category tree does not cover every category exactly once")
    matrix = np.load(config.data_path("cooccurrence_matrix"))
    if matrix.shape != (len(categories), len(categories)):
        raise ValueError("Co-occurrence matrix shape mismatch")

    split_payload: dict[str, Any] = {}
    for split, key in (("train", "train_file"), ("valid", "valid_file"), ("test", "test_file")):
        rows = load_records(config.data_path(key))
        expected_rows = int(config.dataset_config["expected_rows"][split])
        if len(rows) != expected_rows:
            raise ValueError(f"{split} row count mismatch")
        histories: list[int] = []
        targets_per_row: list[int] = []
        overlap_rows = 0
        for row_index, row in enumerate(rows):
            history = parse_quoted_values(str(row["input"]))
            targets = parse_quoted_values(str(row["output"]))
            if not history or not targets:
                raise ValueError(f"{split} row {row_index} has empty history/target")
            unknown = (set(history) | set(targets)) - category_set
            if unknown:
                raise ValueError(f"{split} row {row_index} has unknown categories")
            overlap_rows += int(bool(set(history) & set(targets)))
            histories.append(len(history))
            targets_per_row.append(len(set(targets)))
        expected_target_range = (1, 1) if split != "test" else None
        observed_range = (min(targets_per_row), max(targets_per_row))
        if expected_target_range is not None and observed_range != expected_target_range:
            raise ValueError(f"{split} is not single-target")
        if overlap_rows:
            raise ValueError(f"{split} has history-target overlap")
        split_payload[split] = {
            "rows": len(rows),
            "history_length_min": min(histories),
            "history_length_max": max(histories),
            "history_length_mean": sum(histories) / len(histories),
            "target_count_min": observed_range[0],
            "target_count_max": observed_range[1],
            "target_count_mean": sum(targets_per_row) / len(targets_per_row),
            "history_target_overlap_rows": overlap_rows,
        }

    probabilities = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    paper_positive = probabilities.copy()
    paper_positive[paper_positive < 0.05] = 0
    np.fill_diagonal(paper_positive, 0)
    code_anchors = int((matrix.var(axis=0) > 1).sum())
    paper_collaborative_anchors = int((paper_positive.sum(axis=1) > 0).sum())
    paper_structure_anchors = sum(
        len(children) for children in tree.values() if len(children) >= 2
    )
    expected = config.dataset_config
    if code_anchors != int(expected["expected_code_collaborative_anchors"]):
        raise ValueError("Code-faithful collaborative anchor count mismatch")
    if paper_collaborative_anchors != int(expected["expected_paper_collaborative_anchors"]):
        raise ValueError("Paper collaborative anchor count mismatch")
    if paper_structure_anchors != int(expected["expected_paper_structure_anchors"]):
        raise ValueError("Paper structure anchor count mismatch")

    author = evaluate_author_predictions(config)
    expected_author = dataset_manifest["author_predictions"]
    if author["precision_at_k_rounded"] != expected_author["precision_at_k"]:
        raise ValueError("Author precision metrics do not reproduce")
    if author["recall_at_k_rounded"] != expected_author["recall_at_k"]:
        raise ValueError("Author recall metrics do not reproduce")
    return {
        "audit_version": "mera-original-dataset-audit-v1",
        "dataset": config.dataset,
        "identity": config.identity(),
        "categories": len(categories),
        "parents": len(tree),
        "matrix_shape": list(matrix.shape),
        "splits": split_payload,
        "regularizer_coverage": {
            "code_collaborative_anchors": code_anchors,
            "paper_structure_anchors": paper_structure_anchors,
            "paper_collaborative_anchors": paper_collaborative_anchors,
            "paper_positive_edges": int((paper_positive > 0).sum()),
        },
        "author_metrics": {
            key: value for key, value in author.items() if key != "per_example"
        },
    }


def resolve_trainable_weight(module: Any) -> Any:
    if hasattr(module, "modules_to_save"):
        active = getattr(module, "active_adapter", None)
        modules = module.modules_to_save
        if active in modules:
            return modules[active].weight
        if len(modules) == 1:
            return next(iter(modules.values())).weight
    if hasattr(module, "weight"):
        return module.weight
    raise TypeError(f"Cannot resolve trainable weight from {type(module)!r}")


def model_category_rows(model: Any, token_ids: Sequence[int]) -> tuple[Any, Any]:
    import torch

    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    input_weight = resolve_trainable_weight(input_module)
    output_weight = resolve_trainable_weight(output_module)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=input_weight.device)
    output_ids = torch.tensor(token_ids, dtype=torch.long, device=output_weight.device)
    return (
        input_weight.index_select(0, input_ids),
        output_weight.index_select(0, output_ids),
    )


def tensor_mapping_manifest(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    tensors: dict[str, Any] = {}
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor PEFT state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "manifest_version": "mera-original-peft-state-v1",
        "tensor_count": len(tensors),
        "value_count": sum(
            math.prod(entry["shape"]) for entry in tensors.values()
        ),
        "tensors": tensors,
    }


def peft_state_manifest(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return tensor_mapping_manifest(get_peft_model_state_dict(model))


class MeRAOriginalRegularizer:
    def __init__(
        self,
        *,
        token_ids: Sequence[int],
        categories: Sequence[str],
        category_tree: Mapping[str, Sequence[str]],
        cooccurrence_matrix: np.ndarray,
        contract: Mapping[str, Any],
    ) -> None:
        import torch

        self.token_ids = tuple(int(value) for value in token_ids)
        self.categories = tuple(categories)
        self.index = {value: offset for offset, value in enumerate(categories)}
        self.contract = dict(contract)
        self.semantics = str(contract["regularizer_semantics"])
        self.structure_weight = float(contract["structure_weight"])
        self.collaborative_weight = float(contract["collaborative_weight"])
        self.negative_samples = int(contract["negative_samples"])
        self.temperature = 1.0
        self.parent_groups = tuple(
            tuple(self.index[value] for value in values)
            for values in category_tree.values()
        )
        self.parent_for_index = {
            child: parent
            for parent, children in enumerate(self.parent_groups)
            for child in children
        }
        matrix = torch.tensor(cooccurrence_matrix, dtype=torch.float64)
        if self.semantics == "released-code":
            valid = torch.where(matrix.var(dim=0) > 1)[0]
            positive = matrix / matrix.sum(dim=1, keepdim=True)
            positive[torch.isnan(positive)] = 0.5
            valid_mask = torch.zeros(matrix.shape[0], dtype=torch.bool)
            valid_mask[valid] = True
            positive[:, ~valid_mask] = 0
            negative = 1.0 - positive
            negative[torch.isnan(negative)] = 0.5
            negative[:, ~valid_mask] = 0
            positive.fill_diagonal_(0)
            negative.fill_diagonal_(0)
            self.collaborative_anchors = tuple(int(value) for value in valid.tolist())
        elif self.semantics == "paper-equation":
            threshold = float(contract["cooccurrence_threshold"])
            positive = matrix / matrix.sum(dim=1, keepdim=True)
            positive[torch.isnan(positive)] = 0
            positive[positive < threshold] = 0
            positive.fill_diagonal_(0)
            negative = 1.0 - positive
            negative.fill_diagonal_(0)
            self.collaborative_anchors = tuple(
                int(value)
                for value in torch.where(positive.sum(dim=1) > 0)[0].tolist()
            )
        else:
            raise ValueError(f"Unknown regularizer semantics: {self.semantics}")
        self.positive_weights = positive
        self.negative_weights = negative
        self.structure_anchors = tuple(
            child
            for children in self.parent_groups
            if len(children) >= 2
            for child in children
        )
        if not self.collaborative_anchors:
            raise ValueError("No collaborative anchors are eligible")
        self.calls = 0
        self.structure_loss_sum = 0.0
        self.collaborative_loss_sum = 0.0

    def audit(self) -> dict[str, Any]:
        return {
            "regularizer_version": "mera-original-regularizer-v1",
            "semantics": self.semantics,
            "structure_weight": self.structure_weight,
            "collaborative_weight": self.collaborative_weight,
            "negative_samples": self.negative_samples,
            "cooccurrence_threshold": self.contract["cooccurrence_threshold"],
            "structure_parent_groups": len(self.parent_groups),
            "structure_anchors": (
                len(self.structure_anchors)
                if self.semantics == "paper-equation"
                else sum(1 for group in self.parent_groups if len(group) >= 2)
            ),
            "collaborative_anchors": len(self.collaborative_anchors),
            "regularized_modules": ["input_embeddings", "lm_head"],
            "similarity": "unnormalized_dot_product",
            "calls": self.calls,
            "mean_structure_loss": self.structure_loss_sum / max(self.calls, 1),
            "mean_collaborative_loss": self.collaborative_loss_sum / max(self.calls, 1),
        }

    def _infonce(self, rows: Any, samples: Sequence[Sequence[int]]) -> Any:
        import torch
        import torch.nn.functional as functional

        data = torch.tensor(samples, dtype=torch.long, device=rows.device)
        selected = rows[data]
        anchors = selected[:, 0]
        candidates = selected[:, 1:]
        logits = (candidates * anchors.unsqueeze(1)).sum(dim=-1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=rows.device)
        return functional.cross_entropy(logits.float(), labels)

    def _code_structure_samples(self) -> list[list[int]]:
        all_parents = set(range(len(self.parent_groups)))
        samples: list[list[int]] = []
        for parent, children in enumerate(self.parent_groups):
            if len(children) < 2:
                continue
            anchor, positive = random.sample(list(children), 2)
            negatives = [
                child
                for other in sorted(all_parents - {parent})
                for child in self.parent_groups[other]
            ]
            sampled_negatives = random.sample(negatives, self.negative_samples)
            samples.append([anchor, positive, *sampled_negatives])
        return samples

    def _paper_structure_samples(self) -> list[list[int]]:
        samples: list[list[int]] = []
        for anchor in self.structure_anchors:
            parent = self.parent_for_index[anchor]
            siblings = [
                value for value in self.parent_groups[parent] if value != anchor
            ]
            positive = random.choice(siblings)
            negatives = [
                value
                for other_parent, children in enumerate(self.parent_groups)
                if other_parent != parent
                for value in children
            ]
            samples.append(
                [anchor, positive, *random.sample(negatives, self.negative_samples)]
            )
        return samples

    def _collaborative_samples(self, device: Any) -> list[list[int]]:
        import torch

        samples: list[list[int]] = []
        for anchor in self.collaborative_anchors:
            positive_weights = self.positive_weights[anchor].to(device)
            positive = int(
                torch.multinomial(
                    positive_weights, 1, replacement=False
                ).item()
            )
            negative_weights = self.negative_weights[anchor].to(device).clone()
            if self.semantics == "paper-equation":
                negative_weights[positive] = 0
            negatives = torch.multinomial(
                negative_weights, self.negative_samples, replacement=False
            ).tolist()
            samples.append([anchor, positive, *negatives])
        return samples

    def compute(self, model: Any) -> dict[str, Any]:
        input_rows, output_rows = model_category_rows(model, self.token_ids)
        zero = input_rows.sum() * 0.0
        structure_loss = zero
        collaborative_loss = zero
        if self.structure_weight > 0:
            samples = (
                self._code_structure_samples()
                if self.semantics == "released-code"
                else self._paper_structure_samples()
            )
            structure_loss = self._infonce(input_rows, samples)
            structure_loss = structure_loss + self._infonce(output_rows, samples)
        if self.collaborative_weight > 0:
            samples = self._collaborative_samples(input_rows.device)
            collaborative_loss = self._infonce(input_rows, samples)
            collaborative_loss = collaborative_loss + self._infonce(
                output_rows, samples
            )
        weighted_loss = (
            self.structure_weight * structure_loss
            + self.collaborative_weight * collaborative_loss
        )
        self.calls += 1
        self.structure_loss_sum += float(structure_loss.detach().cpu())
        self.collaborative_loss_sum += float(collaborative_loss.detach().cpu())
        return {
            "structure_loss": structure_loss,
            "collaborative_loss": collaborative_loss,
            "weighted_loss": weighted_loss,
        }


def jsonable_versions() -> dict[str, Any]:
    import platform

    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
    ):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            versions[name] = f"unavailable:{type(error).__name__}"
    return versions


def files_manifest(root: Path, *, exclude_names: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude_names)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def expected_optimizer_steps(rows: int, training: Mapping[str, Any]) -> int:
    batches = math.ceil(rows / int(training["per_device_train_batch_size"]))
    updates = max(batches // int(training["gradient_accumulation_steps"]), 1)
    return math.ceil(float(training["epochs"]) * updates)
