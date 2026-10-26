from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_generative_dataset import (
    GENERATIVE_SCHEMA,
    RESPONSE_PREFIX,
    _format_prompt,
)
from new_interest.data.build_sft_subsets import GROUPED_EVAL_SCHEMA
from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-sud-crossfit-reweighting-qwen35-v1"
DEFAULT_SEED = 20260818
DEFAULT_FOLDS = 5
DEFAULT_SFT_DIR = Path(
    "data/training/sft_static_user_disjoint_balanced_200_h50_v1"
)
DEFAULT_RAW_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_sft21600_h50_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/training/"
    "sft_static_user_disjoint_crossfit_reweighting_qwen35_v1"
)

TASK_TYPES = ("behavior_only", "interest_only")
SHUFFLE_SEEDS = (20260818, 20260819, 20260820)

FOLD_SCHEMA = pa.schema(
    [
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("target_subcategory", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("within_class_hash_rank", pa.int16(), nullable=False),
        pa.field("fold_priority", pa.string(), nullable=False),
    ]
)

SHUFFLE_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("recipient_user_id", pa.string(), nullable=False),
        pa.field("donor_user_id", pa.string(), nullable=False),
        pa.field("recipient_history_length", pa.int32(), nullable=False),
        pa.field("donor_history_length", pa.int32(), nullable=False),
        pa.field(
            "recipient_distinct_interest_count", pa.int32(), nullable=False
        ),
        pa.field("donor_distinct_interest_count", pa.int32(), nullable=False),
        pa.field("match_level", pa.string(), nullable=False),
        pa.field("selection_priority", pa.string(), nullable=False),
        pa.field("recipient_target_count", pa.int16(), nullable=False),
        pa.field("target_history_overlap", pa.int16(), nullable=False),
        pa.field("is_fixed_point", pa.bool_(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    train_rows: int
    classes: int
    folds: int
    eval_impressions: int
    eval_users: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _priority(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{seed}:{namespace}:{value}".encode("utf-8")
    ).hexdigest()


def _require_columns(path: Path, columns: Iterable[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    missing = set(columns).difference(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def assign_balanced_hash_folds(
    rows: list[dict[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    num_folds: int = DEFAULT_FOLDS,
) -> list[dict[str, object]]:
    """Assign exact class-balanced folds using a stable hash ordering."""
    if num_folds <= 1:
        raise ValueError("num_folds must exceed one")
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        source_id = str(row["source_sample_id"])
        if source_id in seen:
            raise ValueError(f"duplicate source_sample_id {source_id}")
        seen.add(source_id)
        by_target[str(row["target_subcategory"])].append(row)

    manifest: list[dict[str, object]] = []
    for target, class_rows in sorted(by_target.items()):
        if len(class_rows) % num_folds:
            raise ValueError(
                f"{target}: class size {len(class_rows)} is not divisible "
                f"by {num_folds}"
            )
        ordered = sorted(
            class_rows,
            key=lambda row: (
                _priority(
                    seed,
                    "fold",
                    str(row["source_sample_id"]),
                ),
                str(row["source_sample_id"]),
            ),
        )
        for rank, row in enumerate(ordered):
            source_id = str(row["source_sample_id"])
            manifest.append(
                {
                    "source_sample_id": source_id,
                    "user_id": str(row["user_id"]),
                    "target_subcategory": target,
                    "fold": rank % num_folds,
                    "within_class_hash_rank": rank,
                    "fold_priority": _priority(seed, "fold", source_id),
                }
            )
    return sorted(manifest, key=lambda row: str(row["source_sample_id"]))


def _bucket(value: int) -> str:
    if value <= 0:
        raise ValueError("History bucket values must be positive")
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 35:
        return "21-35"
    return "36+"


def _scan_donor_pool(
    pool: list[str],
    *,
    recipient: str,
    forbidden_targets: set[str],
    histories: dict[str, dict[str, object]],
    seed: int,
) -> str | None:
    if not pool:
        return None
    start = int(_priority(seed, "shuffle-start", recipient), 16) % len(pool)
    for offset in range(len(pool)):
        donor = pool[(start + offset) % len(pool)]
        donor_categories = set(histories[donor]["categories"])
        if donor != recipient and donor_categories.isdisjoint(
            forbidden_targets
        ):
            return donor
    return None


def build_matched_donor_mapping(
    histories: dict[str, dict[str, object]],
    targets_by_user: dict[str, set[str]],
    *,
    seed: int,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Build one deterministic donor per user, shared by both input views."""
    users = sorted(histories)
    if set(users) != set(targets_by_user):
        raise ValueError("History and target user identities differ")
    exact: dict[tuple[int, str], list[str]] = defaultdict(list)
    same_length: dict[int, list[str]] = defaultdict(list)
    bucketed: dict[tuple[str, str], list[str]] = defaultdict(list)
    length_bucketed: dict[str, list[str]] = defaultdict(list)
    for user in users:
        categories = [str(value) for value in histories[user]["categories"]]
        length = len(categories)
        distinct = len(set(categories))
        exact[(length, _bucket(distinct))].append(user)
        same_length[length].append(user)
        bucketed[(_bucket(length), _bucket(distinct))].append(user)
        length_bucketed[_bucket(length)].append(user)

    mapping: dict[str, str] = {}
    audit_rows: list[dict[str, object]] = []
    for recipient in users:
        recipient_categories = [
            str(value) for value in histories[recipient]["categories"]
        ]
        recipient_length = len(recipient_categories)
        recipient_distinct = len(set(recipient_categories))
        forbidden = targets_by_user[recipient]
        pools = (
            (
                "exact_length_and_distinct_bucket",
                exact[(recipient_length, _bucket(recipient_distinct))],
            ),
            ("exact_length", same_length[recipient_length]),
            (
                "length_and_distinct_buckets",
                bucketed[
                    (_bucket(recipient_length), _bucket(recipient_distinct))
                ],
            ),
            (
                "length_bucket",
                length_bucketed[_bucket(recipient_length)],
            ),
            ("unmatched", users),
        )
        donor = None
        match_level = ""
        for level, pool in pools:
            donor = _scan_donor_pool(
                pool,
                recipient=recipient,
                forbidden_targets=forbidden,
                histories=histories,
                seed=seed,
            )
            if donor is not None:
                match_level = level
                break
        if donor is None:
            raise ValueError(f"No valid shuffled-History donor for {recipient}")
        mapping[recipient] = donor
        donor_categories = [
            str(value) for value in histories[donor]["categories"]
        ]
        overlap = forbidden.intersection(donor_categories)
        audit_rows.append(
            {
                "seed": seed,
                "recipient_user_id": recipient,
                "donor_user_id": donor,
                "recipient_history_length": recipient_length,
                "donor_history_length": len(donor_categories),
                "recipient_distinct_interest_count": recipient_distinct,
                "donor_distinct_interest_count": len(set(donor_categories)),
                "match_level": match_level,
                "selection_priority": _priority(
                    seed,
                    f"shuffle-donor:{recipient}",
                    donor,
                ),
                "recipient_target_count": len(forbidden),
                "target_history_overlap": len(overlap),
                "is_fixed_point": recipient == donor,
            }
        )
    return mapping, audit_rows


def _load_paired_train(
    sft_dir: Path,
) -> tuple[
    dict[str, list[dict[str, object]]],
    list[dict[str, object]],
]:
    task_rows: dict[str, list[dict[str, object]]] = {}
    indexed: dict[str, dict[str, dict[str, object]]] = {}
    for task_type in TASK_TYPES:
        path = sft_dir / "train" / task_type / "21600.parquet"
        _require_columns(
            path,
            (
                "source_sample_id",
                "user_id",
                "impression_id",
                "target_subcategory",
                "messages",
            ),
        )
        rows = pq.read_table(path).to_pylist()
        by_id = {str(row["source_sample_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{path}: duplicate source_sample_id")
        task_rows[task_type] = rows
        indexed[task_type] = by_id
    if indexed[TASK_TYPES[0]].keys() != indexed[TASK_TYPES[1]].keys():
        raise ValueError("Behavior and Interest Train identities differ")

    identity_rows = []
    for source_id in sorted(indexed[TASK_TYPES[0]]):
        left = indexed[TASK_TYPES[0]][source_id]
        right = indexed[TASK_TYPES[1]][source_id]
        for column in (
            "source_sample_id",
            "user_id",
            "impression_id",
            "target_subcategory",
        ):
            if left[column] != right[column]:
                raise ValueError(
                    f"{source_id}: Behavior/Interest mismatch in {column}"
                )
        identity_rows.append(left)
    return task_rows, identity_rows


def _load_grouped_raw_dev(
    raw_dir: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, set[str]],
]:
    path = raw_dir / "dev.parquet"
    _require_columns(
        path,
        (
            "impression_id",
            "user_id",
            "timestamp",
            "history",
            "target_interest",
        ),
    )
    groups: dict[str, list[dict[str, object]]] = {}
    order: list[str] = []
    for row in pq.read_table(path).to_pylist():
        impression = str(row["impression_id"])
        if impression not in groups:
            groups[impression] = []
            order.append(impression)
        groups[impression].append(row)

    behavior_rows: list[dict[str, object]] = []
    interest_rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, object]] = {}
    targets_by_user: dict[str, set[str]] = defaultdict(set)
    for impression in order:
        rows = groups[impression]
        first = rows[0]
        user = str(first["user_id"])
        history = first["history"]
        if not isinstance(history, list) or not history:
            raise ValueError(f"{impression}: invalid History")
        for row in rows[1:]:
            if row["history"] != history or str(row["user_id"]) != user:
                raise ValueError(f"{impression}: inconsistent grouped identity")
        titles = [str(item["title"]) for item in history]
        categories = [str(item["subcategory"]) for item in history]
        signature = {"titles": titles, "categories": categories}
        previous = histories.setdefault(user, signature)
        if previous != signature:
            raise ValueError(f"{user}: inconsistent static History")
        targets: list[str] = []
        seen_targets: set[str] = set()
        for row in rows:
            target = str(row["target_interest"]["subcategory"])
            targets_by_user[user].add(target)
            if target not in seen_targets:
                targets.append(target)
                seen_targets.add(target)
        common = {
            "impression_id": impression,
            "user_id": user,
            "timestamp": first["timestamp"],
            "history_length": len(history),
            "acceptable_responses": [
                f"{RESPONSE_PREFIX}{target}" for target in targets
            ],
            "acceptable_subcategories": targets,
            "history_subcategories": categories,
        }
        behavior_rows.append(
            {**common, "prompt": _format_prompt("behavior_only", titles)}
        )
        interest_rows.append(
            {
                **common,
                "prompt": _format_prompt("interest_only", categories),
            }
        )
    return behavior_rows, interest_rows, histories, dict(targets_by_user)


def _verify_frozen_real_eval(
    generated: dict[str, list[dict[str, object]]],
    sft_dir: Path,
) -> None:
    for task_type in TASK_TYPES:
        path = sft_dir / "eval" / task_type / "dev_full_grouped.parquet"
        frozen = pq.read_table(path).to_pylist()
        current = generated[task_type]
        if len(frozen) != len(current):
            raise ValueError(f"{task_type}: grouped Dev row count changed")
        for index, (left, right) in enumerate(zip(frozen, current)):
            if left != right:
                raise ValueError(
                    f"{task_type}: grouped Dev differs at row {index}"
                )


def _write_table(
    path: Path,
    rows: list[dict[str, object]],
    schema: pa.Schema,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=2_000,
    )


def _render_shuffled(
    real_rows: list[dict[str, object]],
    histories: dict[str, dict[str, object]],
    mapping: dict[str, str],
    *,
    task_type: str,
) -> list[dict[str, object]]:
    output = []
    for row in real_rows:
        donor = mapping[str(row["user_id"])]
        history = histories[donor]
        categories = [str(value) for value in history["categories"]]
        values = (
            [str(value) for value in history["titles"]]
            if task_type == "behavior_only"
            else categories
        )
        output.append(
            {
                **row,
                "history_length": len(categories),
                "prompt": _format_prompt(task_type, values),
                "history_subcategories": categories,
            }
        )
    return output


def _publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    backup = output_dir.with_name(output_dir.name + ".old")
    if overwrite and output_dir.exists() and backup.exists():
        raise FileExistsError(f"Cannot overwrite while backup exists: {backup}")
    if output_dir.exists():
        os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            os.replace(backup, output_dir)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output_dir)


def build_crossfit_reweighting_dataset(
    sft_dir: Path | str = DEFAULT_SFT_DIR,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = DEFAULT_SEED,
    num_folds: int = DEFAULT_FOLDS,
    shuffle_seeds: tuple[int, ...] = SHUFFLE_SEEDS,
    expected_train_rows: int | None = 21_600,
    expected_classes: int | None = 108,
    expected_eval_rows: int | None = 31_392,
    overwrite: bool = False,
) -> BuildResult:
    sft_dir = Path(sft_dir)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    task_rows, identity_rows = _load_paired_train(sft_dir)
    manifest = assign_balanced_hash_folds(
        identity_rows,
        seed=seed,
        num_folds=num_folds,
    )
    targets = Counter(str(row["target_subcategory"]) for row in manifest)
    if expected_train_rows is not None and len(manifest) != expected_train_rows:
        raise ValueError(
            f"Expected {expected_train_rows} Train rows, got {len(manifest)}"
        )
    if expected_classes is not None and len(targets) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, got {len(targets)}")
    fold_by_id = {
        str(row["source_sample_id"]): int(row["fold"]) for row in manifest
    }

    behavior, interest, histories, targets_by_user = _load_grouped_raw_dev(
        raw_dir
    )
    generated = {"behavior_only": behavior, "interest_only": interest}
    _verify_frozen_real_eval(generated, sft_dir)
    if expected_eval_rows is not None and len(behavior) != expected_eval_rows:
        raise ValueError(
            f"Expected {expected_eval_rows} Dev impressions, got {len(behavior)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-sud-crossfit-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        _write_table(staging / "fold_manifest.parquet", manifest, FOLD_SCHEMA)
        fold_stats: dict[str, object] = {}
        for task_type in TASK_TYPES:
            rows = task_rows[task_type]
            for fold in range(num_folds):
                heldout = [
                    row
                    for row in rows
                    if fold_by_id[str(row["source_sample_id"])] == fold
                ]
                train = [
                    row
                    for row in rows
                    if fold_by_id[str(row["source_sample_id"])] != fold
                ]
                root = staging / "teacher" / task_type / f"fold_{fold}"
                _write_table(root / "train.parquet", train, GENERATIVE_SCHEMA)
                _write_table(
                    root / "heldout.parquet", heldout, GENERATIVE_SCHEMA
                )
                fold_stats[f"{task_type}/fold_{fold}"] = {
                    "train_rows": len(train),
                    "heldout_rows": len(heldout),
                    "heldout_classes": len(
                        {str(row["target_subcategory"]) for row in heldout}
                    ),
                }

        for task_type in TASK_TYPES:
            _write_table(
                staging / "eval" / task_type / "real.parquet",
                generated[task_type],
                GROUPED_EVAL_SCHEMA,
            )
        shuffle_rows: list[dict[str, object]] = []
        shuffle_stats: dict[str, object] = {}
        for shuffle_seed in shuffle_seeds:
            mapping, audit = build_matched_donor_mapping(
                histories,
                targets_by_user,
                seed=shuffle_seed,
            )
            shuffle_rows.extend(audit)
            for task_type in TASK_TYPES:
                shuffled = _render_shuffled(
                    generated[task_type],
                    histories,
                    mapping,
                    task_type=task_type,
                )
                _write_table(
                    staging
                    / "eval"
                    / task_type
                    / f"shuffled_{shuffle_seed}.parquet",
                    shuffled,
                    GROUPED_EVAL_SCHEMA,
                )
            shuffle_stats[str(shuffle_seed)] = {
                "users": len(mapping),
                "fixed_points": sum(bool(row["is_fixed_point"]) for row in audit),
                "target_history_overlaps": sum(
                    int(row["target_history_overlap"]) for row in audit
                ),
                "match_levels": dict(
                    sorted(Counter(str(row["match_level"]) for row in audit).items())
                ),
            }
        _write_table(
            staging / "shuffle_manifest.parquet",
            shuffle_rows,
            SHUFFLE_MANIFEST_SCHEMA,
        )

        per_fold_class = Counter(
            (str(row["target_subcategory"]), int(row["fold"]))
            for row in manifest
        )
        statistics = {
            "rule_version": RULE_VERSION,
            "seed": seed,
            "train": {
                "rows": len(manifest),
                "users": len({str(row["user_id"]) for row in manifest}),
                "classes": len(targets),
                "min_per_class": min(targets.values()),
                "max_per_class": max(targets.values()),
                "folds": num_folds,
                "rows_per_fold": dict(
                    sorted(
                        Counter(str(row["fold"]) for row in manifest).items()
                    )
                ),
                "min_per_class_per_fold": min(per_fold_class.values()),
                "max_per_class_per_fold": max(per_fold_class.values()),
                "teacher_splits": fold_stats,
            },
            "eval": {
                "impressions": len(behavior),
                "users": len(histories),
                "target_occurrences": sum(
                    len(row["acceptable_subcategories"]) for row in behavior
                ),
                "supported_target_interests": len(
                    {
                        str(target)
                        for row in behavior
                        for target in row["acceptable_subcategories"]
                    }
                ),
                "shuffle": shuffle_stats,
            },
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_paths = sorted(
            path for path in staging.rglob("*") if path.is_file()
        )
        source_paths = [
            sft_dir / "train" / task / "21600.parquet"
            for task in TASK_TYPES
        ] + [
            sft_dir / "eval" / task / "dev_full_grouped.parquet"
            for task in TASK_TYPES
        ] + [raw_dir / "dev.parquet"]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rule_version": RULE_VERSION,
            "seed": seed,
            "fold_hash": f'SHA256("{seed}:fold:<source_sample_id>")',
            "fold_assignment": (
                "within each target class, sort by fold hash and source ID; "
                "assign rank modulo five"
            ),
            "sft_source_dir": portable_path(sft_dir),
            "raw_source_dir": portable_path(raw_dir),
            "output_dir": portable_path(output_dir),
            "shuffle_seeds": list(shuffle_seeds),
            "shuffle_contract": (
                "one donor per user and seed, shared by Behavior/Interest; "
                "no self donor and donor History excludes every retained "
                "target of the recipient user"
            ),
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "implementation_sha256": {
                portable_path(Path(__file__)): _sha256(Path(__file__))
            },
            "artifact_sha256": {
                path.relative_to(staging).as_posix(): _sha256(path)
                for path in artifact_paths
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir, overwrite=overwrite)

    return BuildResult(
        output_dir=output_dir,
        train_rows=len(manifest),
        classes=len(targets),
        folds=num_folds,
        eval_impressions=len(behavior),
        eval_users=len(histories),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_crossfit_reweighting_dataset(
        args.sft_dir,
        args.raw_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
