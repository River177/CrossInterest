"""Freeze user-held-out OI/NI data and class-marginal matched episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "mind-oi-ni-meta-pilot-v1"
DEFAULT_OUTPUT_DIR = Path("data/training/mind_oi_ni_meta_pilot_v1")
HISTORY_HEADER = (
    "The headlines and corresponding topic categories of the news items I "
    "recently viewed, in chronological order, are listed below. Each item is "
    'formatted as "[Topic category] Headline":'
)
OI_INSTRUCTION = (
    "Based on both the headlines and their topic categories, predict a "
    "previously observed topic category that I am likely to click again. "
    "The predicted category must appear among the topic categories in the "
    "history above. Respond with only the learned interest token assigned "
    "to the predicted category, in the form <interest_NNN>."
)
NO_HISTORY_PROMPT = f"{HISTORY_HEADER}\n\n[Empty history]\n\n{OI_INSTRUCTION}"
REQUIRED_COLUMNS = {
    "sample_id", "source_sample_id", "split", "task_type", "impression_id",
    "user_id", "timestamp", "history_length", "prompt", "response",
    "messages", "target_subcategory",
}
EPISODE_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("class_indices", pa.list_(pa.int16()), nullable=False),
    pa.field("support_indices", pa.list_(pa.int64()), nullable=False),
    pa.field("query_indices", pa.list_(pa.int64()), nullable=False),
])
SLOT_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("class_slot", pa.int16(), nullable=False),
    pa.field("interest_index", pa.int16(), nullable=False),
    pa.field("target_subcategory", pa.string(), nullable=False),
    pa.field("support_indices", pa.list_(pa.int64()), nullable=False),
    pa.field("query_indices", pa.list_(pa.int64()), nullable=False),
])


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    fit_oi_rows: int
    fit_ni_rows: int
    validation_ni_rows: int
    classes: int
    episodes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def _rng(seed: int, namespace: str) -> random.Random:
    return random.Random(int(_priority(seed, "episode-rng", namespace), 16))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path, *, root: Path | None = None) -> dict[str, object]:
    serialized = portable_path(path, root=root)
    # Worktrees commonly symlink frozen data back into the current checkout.
    # Preserve a portable repository path when that resolved target is local.
    if Path(serialized).is_absolute():
        serialized = portable_path(path)
    return {"path": serialized, "sha256": _sha256(path),
            "bytes": path.stat().st_size}


def no_history_prompt(prompt: str) -> str:
    """Fail closed if a source task uses another prompt template."""
    prefix, suffix = HISTORY_HEADER + "\n\n", "\n\n" + OI_INSTRUCTION
    if not prompt.startswith(prefix) or not prompt.endswith(suffix):
        raise ValueError("OI prompt does not match the frozen history/task template")
    if not prompt[len(prefix):-len(suffix)].startswith("1. ["):
        raise ValueError("OI prompt has no numbered history to ablate")
    return NO_HISTORY_PROMPT


def _read_training(path: Path, task: str, mapping: dict[str, tuple[int, str]]) -> pa.Table:
    table = pq.read_table(path)
    missing = REQUIRED_COLUMNS.difference(table.column_names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    reserved = {"row_index", "source_row_index", "interest_index", "no_history_prompt"}
    if reserved.intersection(table.column_names):
        raise ValueError(f"{path}: already has builder-owned columns")
    seen: dict[str, set[str]] = {name: set() for name in ("sample_id", "source_sample_id", "user_id")}
    for row in table.to_pylist():
        for key, values in seen.items():
            value = row[key]
            if not isinstance(value, str) or not value or value in values:
                raise ValueError(f"{path}: missing or duplicate {key}: {value}")
            values.add(value)
        label = row["target_subcategory"]
        if label not in mapping or row["response"] != mapping[label][1]:
            raise ValueError(f"{path}: target/SID mismatch for {row['source_sample_id']}")
        if row["split"] != "train" or row["task_type"] != "behavior_interest":
            raise ValueError(f"{path}: unexpected source split/task_type")
        if row.get("training_task", task) != task:
            raise ValueError(f"{path}: inconsistent training_task")
        expected = [{"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]}]
        if row["messages"] != expected:
            raise ValueError(f"{path}: prompt/response/messages differ")
        if task == "old_interest":
            no_history_prompt(row["prompt"])
    return table


def _enrich(table: pa.Table, indices: list[int], task: str,
            mapping: dict[str, tuple[int, str]]) -> pa.Table:
    selected = table.take(pa.array(indices, type=pa.int64()))
    if "training_task" not in selected.column_names:
        selected = selected.append_column("training_task", pa.array([task] * len(indices)))
    selected = selected.append_column("row_index", pa.array(range(len(indices)), type=pa.int64()))
    selected = selected.append_column("source_row_index", pa.array(indices, type=pa.int64()))
    selected = selected.append_column("interest_index", pa.array(
        [mapping[label][0] for label in selected["target_subcategory"].to_pylist()], type=pa.int16()))
    if task == "old_interest":
        selected = selected.append_column("no_history_prompt", pa.array(
            [NO_HISTORY_PROMPT] * len(indices), type=pa.string()))
    return selected


class _Cycle:
    """Shuffle each complete cycle; defer forbidden entries without dropping them."""

    def __init__(self, values: list[int], rng: random.Random):
        self.values = sorted(values)
        self.rng = rng
        self.pending: deque[int] = deque()

    def take(self, count: int, forbidden: set[int]) -> list[int]:
        if len(set(self.values).difference(forbidden)) < count:
            raise ValueError("Not enough distinct eligible entries for the episode")
        picked: list[int] = []
        # A cycle can end with only forbidden entries. Keep those deferred while
        # starting the next cycle; do not consume their future training exposure.
        deferred: list[int] = []
        while len(picked) < count:
            if not self.pending:
                cycle = self.values.copy()
                self.rng.shuffle(cycle)
                self.pending.extend(cycle)
            value = self.pending.popleft()
            if value in forbidden or value in picked:
                deferred.append(value)
            else:
                picked.append(value)
        self.pending.extendleft(reversed(deferred))
        return picked


def build_episodes(
    fit_oi: pa.Table, fit_ni: pa.Table, *, labels: dict[int, str], seed: int,
    steps: int, episode_classes: int, support_per_class: int, query_per_class: int,
) -> tuple[pa.Table, pa.Table, dict[str, object]]:
    """Pair only class marginals, never individuals or support/query positions."""
    if not 1 <= episode_classes <= len(labels):
        raise ValueError("episode_classes must be in [1, number of classes]")
    if min(steps, support_per_class, query_per_class) < 1:
        raise ValueError("steps and per-class counts must be positive")
    oi, ni = fit_oi.to_pylist(), fit_ni.to_pylist()
    pools: dict[str, dict[int, list[int]]] = {"support": defaultdict(list), "query": defaultdict(list)}
    for role, rows in (("support", oi), ("query", ni)):
        for row in rows:
            pools[role][int(row["interest_index"])].append(int(row["row_index"]))
    for c in labels:
        for role, count in (("support", support_per_class), ("query", query_per_class)):
            if len(pools[role][c]) < count:
                raise ValueError(f"{labels[c]}: fit {role} has fewer than {count} independent users")
    class_rng = _rng(seed, "classes")
    # One permutation is completed before the next, with no replacement within
    # each episode even when an episode crosses a permutation boundary.
    class_pending: list[int] = []
    streams = {role: {c: _Cycle(pool, _rng(seed, f"{role}:{c}"))
                      for c, pool in by_class.items()} for role, by_class in pools.items()}
    class_counts: Counter[int] = Counter({c: 0 for c in labels})
    exposures = {"support": Counter(), "query": Counter()}
    episodes, slots = [], []
    for step in range(steps):
        classes: list[int] = []
        while len(classes) < episode_classes:
            if not class_pending:
                class_pending = sorted(labels)
                class_rng.shuffle(class_pending)
            eligible = next((i for i, c in enumerate(class_pending) if c not in classes), None)
            if eligible is None:
                raise ValueError("Cannot preserve class-cycle balance and episode uniqueness")
            classes.append(class_pending.pop(eligible))
        support_by_class = {c: streams["support"][c].take(support_per_class, set()) for c in classes}
        support_indices = [i for c in classes for i in support_by_class[c]]
        support_users = {str(oi[i]["user_id"]) for i in support_indices}
        if len(support_users) != len(support_indices):
            raise ValueError("Duplicate support users within episode")
        query_by_class: dict[int, list[int]] = {}
        for slot, c in enumerate(classes):
            forbidden = {i for i in pools["query"][c] if ni[i]["user_id"] in support_users}
            query_by_class[c] = streams["query"][c].take(query_per_class, forbidden)
            slots.append({"episode_index": step, "class_slot": slot,
                          "interest_index": c, "target_subcategory": labels[c],
                          "support_indices": support_by_class[c], "query_indices": query_by_class[c]})
        query_indices = [i for c in classes for i in query_by_class[c]]
        query_users = {str(ni[i]["user_id"]) for i in query_indices}
        if len(query_users) != len(query_indices) or support_users.intersection(query_users):
            raise ValueError("Support/query user collision within episode")
        class_counts.update(classes)
        exposures["support"].update(support_indices)
        exposures["query"].update(query_indices)
        episodes.append({"episode_index": step, "class_indices": classes,
                         "support_indices": support_indices, "query_indices": query_indices})
    if max(class_counts.values()) - min(class_counts.values()) > 1:
        raise ValueError("Class episode exposure differs by more than one")
    stats = {"class_appearances": {str(c): class_counts[c] for c in sorted(labels)},
             "support_total": sum(exposures["support"].values()),
             "query_total": sum(exposures["query"].values()),
             "support_unique_rows": len(exposures["support"]),
             "query_unique_rows": len(exposures["query"]),
             "cross_role_user_collisions": 0}
    return pa.Table.from_pylist(episodes, EPISODE_SCHEMA), pa.Table.from_pylist(slots, SLOT_SCHEMA), stats


def build_oi_ni_meta_dataset(
    ni_path: Path | str, oi_path: Path | str, mapping_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR, *, dev_path: Path | str,
    seed: int = 42, validation_per_class: int = 20, expected_classes: int = 108,
    expected_ni_per_class: int = 200, steps: int = 540, episode_classes: int = 8,
    support_per_class: int = 2, query_per_class: int = 2,
    source_config_path: Path | str | None = None,
    source_root: Path | str | None = None,
) -> BuildResult:
    ni_path, oi_path, mapping_path, dev_path, output_dir = map(
        Path, (ni_path, oi_path, mapping_path, dev_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output_dir}")
    if not 0 < validation_per_class < expected_ni_per_class:
        raise ValueError("validation_per_class must leave a nonempty NI fit split")
    source_paths = {"ni": ni_path, "oi": oi_path, "mapping": mapping_path, "dev": dev_path}
    if source_config_path is not None:
        source_paths["source_config"] = Path(source_config_path)
    root = Path(source_root) if source_root is not None else None
    sources = {key: _identity(path, root=root) for key, path in source_paths.items()}
    if source_config_path is not None:
        config = json.loads(Path(source_config_path).read_text(encoding="utf-8"))
        frozen = {Path(item["path"]).as_posix(): item["sha256"]
                  for item in config["new_interest"]["input_files"]}
        for key in ("ni", "mapping", "dev"):
            identity = sources[key]
            # Compare resolved file identities, not a symlink's serialized
            # checkout-relative spelling, while still requiring the exact hash.
            recorded = {digest for relative, digest in frozen.items()
                        if ((root or Path.cwd()) / relative).resolve() == source_paths[key].resolve()}
            if recorded != {identity["sha256"]}:
                raise ValueError(f"{key}: source differs from frozen config identity")
    mapping_table = pq.read_table(mapping_path)
    mapping_rows = mapping_table.to_pylist()
    labels = {int(row["interest_index"]): str(row["target_subcategory"]) for row in mapping_rows}
    mapping = {str(row["target_subcategory"]): (int(row["interest_index"]), str(row["interest_token"]))
               for row in mapping_rows}
    if (len(mapping_rows) != expected_classes or len(mapping) != expected_classes
            or set(labels) != set(range(expected_classes))
            or len({item[1] for item in mapping.values()}) != expected_classes):
        raise ValueError("SID mapping must have unique contiguous classes, labels, and tokens")
    ni = _read_training(ni_path, "new_interest", mapping)
    oi = _read_training(oi_path, "old_interest", mapping)
    ni_rows, oi_rows = ni.to_pylist(), oi.to_pylist()
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ni_rows):
        by_class[row["target_subcategory"]].append(i)
    if set(by_class) != set(mapping) or any(len(rows) != expected_ni_per_class for rows in by_class.values()):
        raise ValueError(f"NI must have exactly {expected_ni_per_class} users in each mapped class")
    validation_indices: set[int] = set()
    split_manifest = []
    for c in sorted(by_class):
        ordered = sorted(by_class[c], key=lambda i: (
            _priority(seed, "oi-ni-meta-validation", f"{c}:{ni_rows[i]['user_id']}"),
            ni_rows[i]["user_id"]))
        validation_indices.update(ordered[:validation_per_class])
        for rank, i in enumerate(ordered):
            row = ni_rows[i]
            split_manifest.append({"source_row_index": i, "source_sample_id": row["source_sample_id"],
                                   "user_id": row["user_id"], "target_subcategory": c,
                                   "interest_index": mapping[c][0], "within_class_hash_rank": rank,
                                   "selection_hash": _priority(seed, "oi-ni-meta-validation", f"{c}:{row['user_id']}"),
                                   "meta_split": "validation" if i in validation_indices else "fit"})
    val_users = {ni_rows[i]["user_id"] for i in validation_indices}
    fit_ni = _enrich(ni, [i for i in range(len(ni)) if i not in validation_indices], "new_interest", mapping)
    val_ni = _enrich(ni, sorted(validation_indices), "new_interest", mapping)
    fit_oi = _enrich(oi, [i for i, row in enumerate(oi_rows) if row["user_id"] not in val_users], "old_interest", mapping)
    # Source assertions are independent of episode construction and fail closed.
    fit_oi_counts = Counter(fit_oi["target_subcategory"].to_pylist())
    for c in mapping:
        if fit_oi_counts[c] < max(2, support_per_class):
            raise ValueError(f"{c}: fit OI has {fit_oi_counts[c]} independent users; need at least {max(2, support_per_class)}")
    dev_users = set(pq.read_table(dev_path, columns=["user_id"])["user_id"].to_pylist())
    source_users = {row["user_id"] for row in ni_rows + oi_rows}
    if dev_users.intersection(source_users):
        raise ValueError("Original Train/Dev user overlap")
    episodes, slots, episode_stats = build_episodes(
        fit_oi, fit_ni, labels=labels, seed=seed, steps=steps,
        episode_classes=episode_classes, support_per_class=support_per_class,
        query_per_class=query_per_class)
    statistics = {"source_ni_rows": len(ni), "source_oi_rows": len(oi),
                  "fit_ni_rows": len(fit_ni), "fit_oi_rows": len(fit_oi),
                  "validation_ni_rows": len(val_ni), "validation_users": len(val_users),
                  "oi_removed_for_validation_users": len(oi) - len(fit_oi),
                  "fit_oi_class_counts": dict(sorted(fit_oi_counts.items())),
                  "fit_ni_class_counts": dict(sorted(Counter(fit_ni["target_subcategory"].to_pylist()).items())),
                  "validation_class_counts": dict(sorted(Counter(val_ni["target_subcategory"].to_pylist()).items())),
                  "source_oi_class_counts": dict(sorted(Counter(oi["target_subcategory"].to_pylist()).items())),
                  "dev_rows": pq.ParquetFile(dev_path).metadata.num_rows,
                  "dev_users": len(dev_users), "train_dev_user_overlap": 0,
                  "episodes": episode_stats}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oi-ni-meta-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        for name, table in (("fit_oi", fit_oi), ("fit_ni", fit_ni), ("validation_ni", val_ni),
                            ("episodes", episodes), ("episode_class_slots", slots),
                            ("validation_manifest", pa.Table.from_pylist(split_manifest))):
            pq.write_table(table, staging / f"{name}.parquet", compression="zstd")
        shutil.copyfile(mapping_path, staging / "interest_token_mapping.parquet")
        shutil.copyfile(dev_path, staging / "dev_full_grouped.parquet")
        _write_json(staging / "statistics.json", statistics)
        artifacts = {path.name: _identity(path, root=staging) for path in sorted(staging.iterdir())}
        # Ensure publication never binds changing input files to old identities.
        for key, path in source_paths.items():
            if _sha256(path) != sources[key]["sha256"]:
                raise ValueError(f"Source changed during build: {key}")
        for key, filename in (("mapping", "interest_token_mapping.parquet"), ("dev", "dev_full_grouped.parquet")):
            if artifacts[filename]["sha256"] != sources[key]["sha256"]:
                raise ValueError(f"Copied {key} bytes differ")
        metadata = {"rule_version": RULE_VERSION, "seed": seed,
                    "validation_per_class": validation_per_class, "expected_classes": expected_classes,
                    "expected_ni_per_class": expected_ni_per_class, "steps": steps,
                    "episode_classes": episode_classes, "support_per_class": support_per_class,
                    "query_per_class": query_per_class, "sources": sources, "artifacts": artifacts,
                    "implementation": _identity(Path(__file__)),
                    "output_dir": portable_path(output_dir),
                    "split_rule": 'SHA256("<seed>:oi-ni-meta-validation:<target>:<user_id>"); first K within class',
                    "episode_matching": "class_marginal_only; no individual pair assignment",
                    "episode_indexing": {"base": 0, "support_indices": "fit_oi.parquet:row_index",
                                         "query_indices": "fit_ni.parquet:row_index",
                                         "flattening": "class_indices order, then within-class draw order"},
                    "episode_sampling": "seeded class permutations; per-class cyclic sample shuffles; forbidden users deferred",
                    "no_history_prompt": NO_HISTORY_PROMPT,
                    "no_history_scope": "inner support only; original OI prompt/messages retained for outer loss",
                    "validation_scope": "Train user holdout; never used in support/query episodes",
                    "dev_scope": "byte-identical evaluation copy; only user IDs read to audit disjointness"}
        _write_json(staging / "metadata.json", metadata)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during build: {output_dir}")
        os.rename(staging, output_dir)
    return BuildResult(output_dir, len(fit_oi), len(fit_ni), len(val_ni), len(labels), steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ni", "oi", "mapping", "dev"):
        parser.add_argument(f"--{name}-path", f"--{name}-file", dest=f"{name}_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-config-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    for name, default in (("seed", 42), ("validation-per-class", 20), ("expected-classes", 108),
                          ("expected-ni-per-class", 200), ("steps", 540), ("episode-classes", 8),
                          ("support-per-class", 2), ("query-per-class", 2)):
        parser.add_argument(f"--{name}", type=int, default=default)
    result = build_oi_ni_meta_dataset(**vars(parser.parse_args()))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
