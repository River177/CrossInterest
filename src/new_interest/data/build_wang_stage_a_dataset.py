from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "wang-stage-a-balanced-transition-v1"
RESPONSE_PREFIX = "New cluster: "
DEFAULT_PAIRS_PER_LABEL = 10
DEFAULT_RANDOM_SEED = 2024
DEFAULT_AMAZON_ROOT = Path(
    "data/external/mera_original/"
    "c07bdbe57bf82e5c938b478035baee78d630b17c/data"
)
DEFAULT_MIND_ROOT = Path("data/processed/new_interest_static_user_disjoint_v1")
DEFAULT_MIND_TARGETS = Path(
    "data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/"
    "target_interest_vocabulary.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/training/wang_stage_a_balanced_transition_v1"
)
QUOTED_VALUE_RE = re.compile(r'"([^"]+)"')


@dataclass(frozen=True)
class Transition:
    source_id: str
    user_id: str
    history: tuple[str, ...]
    context: tuple[str, str]
    targets: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSource:
    name: str
    domain: str
    train: tuple[Transition, ...]
    evaluations: Mapping[str, tuple[Transition, ...]]
    labels: tuple[str, ...]
    source_files: tuple[Path, ...]


@dataclass(frozen=True)
class DatasetBuildResult:
    name: str
    balanced_rows: int
    random_rows: int
    supported_labels: tuple[str, ...]
    artifact_hashes: Mapping[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def parse_quoted_values(text: str) -> tuple[str, ...]:
    return tuple(QUOTED_VALUE_RE.findall(text))


def most_recent_unique_clusters(
    history: Sequence[str],
    *,
    k: int = 2,
) -> tuple[str, ...]:
    if k <= 0:
        raise ValueError("k must be positive")
    selected_reversed: list[str] = []
    seen: set[str] = set()
    for raw_label in reversed(history):
        label = str(raw_label)
        if label in seen:
            continue
        seen.add(label)
        selected_reversed.append(label)
        if len(selected_reversed) == k:
            break
    return tuple(reversed(selected_reversed))


def _prompt(domain: str, context: Sequence[str]) -> str:
    if len(context) != 2:
        raise ValueError("Wang Stage A requires exactly two context clusters")
    return (
        f"The {domain} I interacted with most recently are in the following "
        "clusters:\n"
        f"Cluster 1: {{{context[0]}}}\n"
        f"Cluster 2: {{{context[1]}}}.\n"
        "Each cluster is described with a predefined category name. With less "
        f"than 30 words, generate a new and different {domain} cluster I will "
        "interact with next. The new cluster must differ from both clusters "
        f"above. Copy exactly one predefined cluster name and use the prefix "
        f'"{RESPONSE_PREFIX}".'
    )


def _response(target: str) -> str:
    return f"{RESPONSE_PREFIX}{target}"


def _load_category_map(path: Path) -> tuple[str, ...]:
    rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected label<TAB>id")
        label, raw_id = parts
        rows.append((int(raw_id), label))
    rows.sort()
    ids = [row[0] for row in rows]
    labels = tuple(row[1] for row in rows)
    if ids != list(range(len(rows))):
        raise ValueError(f"{path}: category ids are not contiguous")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{path}: duplicate category label")
    return labels


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list")
    return [dict(row) for row in value]


def _amazon_transitions(
    path: Path,
    *,
    dataset_name: str,
    split: str,
    labels: set[str],
) -> tuple[Transition, ...]:
    transitions: list[Transition] = []
    for row_index, row in enumerate(_load_json_rows(path)):
        history = parse_quoted_values(str(row.get("input", "")))
        targets = tuple(dict.fromkeys(parse_quoted_values(str(row.get("output", "")))))
        if not history or not targets:
            raise ValueError(f"{path}: row {row_index} has empty history or target")
        unknown = (set(history) | set(targets)).difference(labels)
        if unknown:
            raise ValueError(f"{path}: row {row_index} has unknown labels {sorted(unknown)}")
        if set(history).intersection(targets):
            raise ValueError(f"{path}: row {row_index} target occurs in history")
        context = most_recent_unique_clusters(history)
        if len(context) != 2:
            continue
        transitions.append(
            Transition(
                source_id=f"{dataset_name}:{split}:{row_index}",
                user_id=f"{dataset_name}:{split}:{row_index}",
                history=tuple(history),
                context=(context[0], context[1]),
                targets=targets,
            )
        )
    return tuple(transitions)


def load_amazon_source(
    root: Path | str,
    *,
    name: str,
) -> DatasetSource:
    root = Path(root)
    specifications = {
        "arts": (
            root / "Arts",
            "art or craft categories",
        ),
        "grocery": (
            root / "Grocery_and_Gourmet_Food",
            "grocery categories",
        ),
    }
    if name not in specifications:
        raise ValueError(f"Unsupported Amazon dataset: {name}")
    dataset_root, domain = specifications[name]
    category_map = dataset_root / "layer2_map.txt"
    split_paths = {
        "train": dataset_root / "train.json",
        "validation": dataset_root / "valid.json",
        "test": dataset_root / "test.json",
    }
    for path in (category_map, *split_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = _load_category_map(category_map)
    label_set = set(labels)
    transitions = {
        split: _amazon_transitions(
            path,
            dataset_name=name,
            split=split,
            labels=label_set,
        )
        for split, path in split_paths.items()
    }
    return DatasetSource(
        name=name,
        domain=domain,
        train=transitions["train"],
        evaluations={
            "validation": transitions["validation"],
            "test": transitions["test"],
        },
        labels=labels,
        source_files=(category_map, *split_paths.values()),
    )


def _mind_label(row: Mapping[str, Any]) -> str:
    target = row["target_interest"]
    if not isinstance(target, Mapping):
        raise ValueError("MIND target_interest must be a mapping")
    return str(target["subcategory"])


def _mind_history(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["subcategory"]) for item in row["history"])


def _mind_train_transitions(
    path: Path,
    *,
    output_labels: set[str],
) -> tuple[Transition, ...]:
    transitions: list[Transition] = []
    seen_users: set[str] = set()
    for row in pq.read_table(
        path,
        columns=["sample_id", "user_id", "history", "target_interest"],
    ).to_pylist():
        user_id = str(row["user_id"])
        if user_id in seen_users:
            raise ValueError("MIND Train is not one row per user")
        seen_users.add(user_id)
        target = _mind_label(row)
        if target not in output_labels:
            continue
        history = _mind_history(row)
        if target in set(history):
            raise ValueError(f"{row['sample_id']}: target occurs in full history")
        context = most_recent_unique_clusters(history)
        if len(context) != 2:
            continue
        transitions.append(
            Transition(
                source_id=str(row["sample_id"]),
                user_id=user_id,
                history=history,
                context=(context[0], context[1]),
                targets=(target,),
            )
        )
    return tuple(transitions)


def _mind_dev_transitions(
    path: Path,
    *,
    output_labels: set[str],
) -> tuple[Transition, ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(
        path,
        columns=[
            "sample_id",
            "impression_id",
            "user_id",
            "history",
            "target_interest",
        ],
    ).to_pylist():
        target = _mind_label(row)
        if target not in output_labels:
            continue
        impression_id = str(row["impression_id"])
        user_id = str(row["user_id"])
        history = _mind_history(row)
        if target in set(history):
            raise ValueError(f"{row['sample_id']}: target occurs in full history")
        value = grouped.setdefault(
            impression_id,
            {
                "user_id": user_id,
                "history": history,
                "targets": [],
            },
        )
        if value["user_id"] != user_id or value["history"] != history:
            raise ValueError(f"MIND impression {impression_id} has inconsistent context")
        if target not in value["targets"]:
            value["targets"].append(target)

    transitions: list[Transition] = []
    for impression_id, value in grouped.items():
        history = tuple(value["history"])
        context = most_recent_unique_clusters(history)
        if len(context) != 2:
            continue
        transitions.append(
            Transition(
                source_id=f"mind:dev:{impression_id}",
                user_id=str(value["user_id"]),
                history=history,
                context=(context[0], context[1]),
                targets=tuple(value["targets"]),
            )
        )
    return tuple(transitions)


def load_mind_source(
    root: Path | str,
    target_vocabulary: Path | str,
) -> DatasetSource:
    root = Path(root)
    target_vocabulary = Path(target_vocabulary)
    train_path = root / "train.parquet"
    dev_path = root / "dev.parquet"
    interest_vocabulary = root / "interest_vocabulary.parquet"
    for path in (train_path, dev_path, interest_vocabulary, target_vocabulary):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_table = pq.read_table(target_vocabulary)
    target_column = (
        "subcategory"
        if "subcategory" in target_table.column_names
        else "target_subcategory"
    )
    labels = tuple(
        str(value)
        for value in target_table[target_column].to_pylist()
    )
    if len(labels) != len(set(labels)):
        raise ValueError("MIND output vocabulary contains duplicates")
    output_labels = set(labels)
    return DatasetSource(
        name="mind",
        domain="news topic categories",
        train=_mind_train_transitions(
            train_path,
            output_labels=output_labels,
        ),
        evaluations={
            "dev": _mind_dev_transitions(
                dev_path,
                output_labels=output_labels,
            )
        },
        labels=labels,
        source_files=(
            train_path,
            dev_path,
            interest_vocabulary,
            target_vocabulary,
        ),
    )


def _pair_frequencies(
    transitions: Sequence[Transition],
) -> dict[str, Counter[tuple[str, str]]]:
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for transition in transitions:
        if len(transition.targets) != 1:
            raise ValueError("Training transitions must have one target")
        counts[transition.targets[0]][transition.context] += 1
    return dict(counts)


def select_balanced_transitions(
    transitions: Sequence[Transition],
    *,
    pairs_per_label: int,
) -> tuple[list[tuple[Transition, int]], tuple[str, ...]]:
    if pairs_per_label <= 0:
        raise ValueError("pairs_per_label must be positive")
    frequencies = _pair_frequencies(transitions)
    representatives: dict[tuple[str, tuple[str, str]], Transition] = {}
    for transition in transitions:
        target = transition.targets[0]
        representatives.setdefault((target, transition.context), transition)
    supported = tuple(
        sorted(
            target
            for target, pair_counts in frequencies.items()
            if len(pair_counts) >= pairs_per_label
        )
    )
    selected_by_target: dict[str, list[tuple[Transition, int]]] = {}
    for target in supported:
        ordered_pairs = sorted(
            frequencies[target].items(),
            key=lambda item: (-item[1], item[0]),
        )[:pairs_per_label]
        selected_by_target[target] = [
            (representatives[(target, pair)], frequency)
            for pair, frequency in ordered_pairs
        ]
    # Round-robin labels so smoke prefixes retain class coverage instead of
    # containing only the alphabetically first labels.
    selected = [
        selected_by_target[target][pair_rank]
        for pair_rank in range(pairs_per_label)
        for target in supported
    ]
    return selected, supported


def select_random_transitions(
    transitions: Sequence[Transition],
    *,
    supported_labels: set[str],
    sample_count: int,
    seed: int,
) -> list[tuple[Transition, int]]:
    eligible = [
        transition
        for transition in transitions
        if transition.targets[0] in supported_labels
    ]
    if sample_count > len(eligible):
        raise ValueError("Random baseline sample count exceeds eligible transitions")
    pair_counts = _pair_frequencies(eligible)
    ordered = sorted(
        eligible,
        key=lambda transition: (
            canonical_sha256(
                {
                    "seed": seed,
                    "source_id": transition.source_id,
                }
            ),
            transition.source_id,
        ),
    )[:sample_count]
    return [
        (
            transition,
            pair_counts[transition.targets[0]][transition.context],
        )
        for transition in ordered
    ]


def _training_rows(
    dataset: DatasetSource,
    selected: Iterable[tuple[Transition, int]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (transition, frequency) in enumerate(selected):
        target = transition.targets[0]
        prompt = _prompt(dataset.domain, transition.context)
        response = _response(target)
        rows.append(
            {
                "sample_id": f"{dataset.name}:{condition}:{rank:06d}",
                "source_sample_id": transition.source_id,
                "source_user_id": transition.user_id,
                "dataset": dataset.name,
                "condition": condition,
                "context_clusters": list(transition.context),
                "pair_frequency": frequency,
                "target_subcategory": target,
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return rows


def _evaluation_rows(
    dataset: DatasetSource,
    transitions: Sequence[Transition],
    *,
    supported_labels: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in transitions:
        targets = list(transition.targets)
        if supported_labels is not None:
            targets = [target for target in targets if target in supported_labels]
        if not targets:
            continue
        prompt = _prompt(dataset.domain, transition.context)
        rows.append(
            {
                "impression_id": transition.source_id,
                "user_id": transition.user_id,
                "dataset": dataset.name,
                "prompt": prompt,
                "context_clusters": list(transition.context),
                "history_subcategories": list(transition.history),
                "acceptable_subcategories": targets,
                "acceptable_responses": [_response(target) for target in targets],
            }
        )
    return rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty Parquet file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def build_dataset(
    dataset: DatasetSource,
    output_dir: Path | str,
    *,
    pairs_per_label: int = DEFAULT_PAIRS_PER_LABEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> DatasetBuildResult:
    output_dir = Path(output_dir)
    balanced, supported = select_balanced_transitions(
        dataset.train,
        pairs_per_label=pairs_per_label,
    )
    if not supported:
        raise ValueError(f"{dataset.name}: no label has sufficient context pairs")
    expected_balanced = len(supported) * pairs_per_label
    if len(balanced) != expected_balanced:
        raise ValueError(f"{dataset.name}: balanced selection count mismatch")
    random_rows = select_random_transitions(
        dataset.train,
        supported_labels=set(supported),
        sample_count=expected_balanced,
        seed=random_seed,
    )

    _write_parquet(
        output_dir / "train" / "balanced.parquet",
        _training_rows(dataset, balanced, condition="balanced"),
    )
    _write_parquet(
        output_dir / "train" / "random.parquet",
        _training_rows(dataset, random_rows, condition="random"),
    )
    _write_parquet(
        output_dir / "labels_full.parquet",
        [
            {"label_id": index, "target_subcategory": label}
            for index, label in enumerate(dataset.labels)
        ],
    )
    _write_parquet(
        output_dir / "labels_supported.parquet",
        [
            {"label_id": index, "target_subcategory": label}
            for index, label in enumerate(supported)
        ],
    )

    evaluation_statistics: dict[str, Any] = {}
    for split, transitions in dataset.evaluations.items():
        full_rows = _evaluation_rows(
            dataset,
            transitions,
            supported_labels=None,
        )
        supported_rows = _evaluation_rows(
            dataset,
            transitions,
            supported_labels=set(supported),
        )
        _write_parquet(
            output_dir / "eval" / f"{split}_full.parquet",
            full_rows,
        )
        _write_parquet(
            output_dir / "eval" / f"{split}_supported.parquet",
            supported_rows,
        )
        evaluation_statistics[split] = {
            "eligible_context_rows": len(transitions),
            "full_rows": len(full_rows),
            "supported_rows": len(supported_rows),
            "full_target_occurrences": sum(
                len(row["acceptable_subcategories"]) for row in full_rows
            ),
            "supported_target_occurrences": sum(
                len(row["acceptable_subcategories"]) for row in supported_rows
            ),
        }

    balanced_counts = Counter(
        transition.targets[0] for transition, _ in balanced
    )
    random_counts = Counter(
        transition.targets[0] for transition, _ in random_rows
    )
    pair_counts = _pair_frequencies(dataset.train)
    statistics = {
        "rule_version": RULE_VERSION,
        "dataset": dataset.name,
        "response_prefix": RESPONSE_PREFIX,
        "context_clusters": 2,
        "pairs_per_label": pairs_per_label,
        "source_train_eligible_context_rows": len(dataset.train),
        "full_label_count": len(dataset.labels),
        "train_observed_label_count": len(pair_counts),
        "supported_label_count": len(supported),
        "supported_labels": list(supported),
        "underfilled_labels": {
            label: len(pair_counts.get(label, {}))
            for label in dataset.labels
            if len(pair_counts.get(label, {})) < pairs_per_label
        },
        "balanced": {
            "rows": len(balanced),
            "min_per_label": min(balanced_counts.values()),
            "max_per_label": max(balanced_counts.values()),
            "target_counts": dict(sorted(balanced_counts.items())),
        },
        "random": {
            "seed": random_seed,
            "rows": len(random_rows),
            "observed_labels": len(random_counts),
            "min_per_observed_label": min(random_counts.values()),
            "max_per_observed_label": max(random_counts.values()),
            "target_counts": dict(sorted(random_counts.items())),
        },
        "evaluation": evaluation_statistics,
    }
    _write_json(output_dir / "statistics.json", statistics)

    artifact_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule_version": RULE_VERSION,
        "dataset": dataset.name,
        "domain": dataset.domain,
        "method_source": {
            "title": "LLMs for User Interest Exploration in Large-scale Recommendation Systems",
            "doi": "10.1145/3640457.3688161",
            "arxiv": "2405.16363",
        },
        "adaptation": (
            "Dataset taxonomies are used as predefined interest clusters; "
            "the original industrial traffic-balanced multimodal clustering "
            "and online evaluation data are unavailable."
        ),
        "output_dir": portable_path(output_dir),
        "source_sha256": {
            portable_path(path): sha256_file(path)
            for path in dataset.source_files
        },
        "artifact_sha256": artifact_hashes,
        "selection_identity": canonical_sha256(
            {
                "rule_version": RULE_VERSION,
                "dataset": dataset.name,
                "pairs_per_label": pairs_per_label,
                "random_seed": random_seed,
                "supported_labels": supported,
                "balanced_source_ids": [
                    transition.source_id for transition, _ in balanced
                ],
                "random_source_ids": [
                    transition.source_id for transition, _ in random_rows
                ],
            }
        ),
    }
    _write_json(output_dir / "metadata.json", metadata)
    artifact_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    return DatasetBuildResult(
        name=dataset.name,
        balanced_rows=len(balanced),
        random_rows=len(random_rows),
        supported_labels=supported,
        artifact_hashes=artifact_hashes,
    )


def _publish(staging: Path, output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Use --overwrite explicitly."
        )
    backup: Path | None = None
    if output_root.exists():
        backup = output_root.with_name(f".{output_root.name}.backup-{os.getpid()}")
        if backup.exists():
            raise FileExistsError(backup)
        os.replace(output_root, backup)
    try:
        os.replace(staging, output_root)
    except BaseException:
        if backup is not None and backup.exists() and not output_root.exists():
            os.replace(backup, output_root)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_wang_stage_a_datasets(
    *,
    amazon_root: Path | str = DEFAULT_AMAZON_ROOT,
    mind_root: Path | str = DEFAULT_MIND_ROOT,
    mind_target_vocabulary: Path | str = DEFAULT_MIND_TARGETS,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    pairs_per_label: int = DEFAULT_PAIRS_PER_LABEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> tuple[DatasetBuildResult, ...]:
    output_root = Path(output_root)
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Use --overwrite explicitly."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    )
    try:
        sources = (
            load_amazon_source(amazon_root, name="arts"),
            load_amazon_source(amazon_root, name="grocery"),
            load_mind_source(mind_root, mind_target_vocabulary),
        )
        results = tuple(
            build_dataset(
                source,
                staging / source.name,
                pairs_per_label=pairs_per_label,
                random_seed=random_seed,
            )
            for source in sources
        )
        root_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "pairs_per_label": pairs_per_label,
            "random_seed": random_seed,
            "datasets": {
                result.name: {
                    "balanced_rows": result.balanced_rows,
                    "random_rows": result.random_rows,
                    "supported_label_count": len(result.supported_labels),
                    "supported_labels": list(result.supported_labels),
                    "metadata_sha256": result.artifact_hashes["metadata.json"],
                    "statistics_sha256": result.artifact_hashes["statistics.json"],
                }
                for result in results
            },
        }
        _write_json(staging / "manifest.json", root_manifest)
        _publish(staging, output_root, overwrite=overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Wang et al. Stage-A balanced-transition and random-control "
            "datasets for Amazon Arts, Amazon Grocery, and MIND."
        )
    )
    parser.add_argument("--amazon-root", type=Path, default=DEFAULT_AMAZON_ROOT)
    parser.add_argument("--mind-root", type=Path, default=DEFAULT_MIND_ROOT)
    parser.add_argument(
        "--mind-target-vocabulary",
        type=Path,
        default=DEFAULT_MIND_TARGETS,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--pairs-per-label",
        type=int,
        default=DEFAULT_PAIRS_PER_LABEL,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = build_wang_stage_a_datasets(
        amazon_root=args.amazon_root,
        mind_root=args.mind_root,
        mind_target_vocabulary=args.mind_target_vocabulary,
        output_root=args.output_root,
        pairs_per_label=args.pairs_per_label,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(
        "WANG_STAGE_A_DATASET_BUILD_OK",
        json.dumps(
            {
                result.name: {
                    "balanced_rows": result.balanced_rows,
                    "random_rows": result.random_rows,
                    "supported_labels": len(result.supported_labels),
                }
                for result in results
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
