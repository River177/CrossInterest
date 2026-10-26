from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "static-user-disjoint-semantic-users-v1"
DEFAULT_DATASET_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_v1"
)
DEFAULT_CANDIDATE_FILE = Path("data/processed/new_interest/train.parquet")
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/new_interest_static_user_disjoint_semantic_users_v1"
)
DEFAULT_SEED = 20260815
DEFAULT_VALIDATION_FRACTION = 0.1


USER_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field("history_unique_subcategory_count", pa.int16(), nullable=False),
        pa.field("history_titles", pa.list_(pa.string()), nullable=False),
        pa.field("history_subcategories", pa.list_(pa.string()), nullable=False),
        pa.field(
            "history_unique_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("target_count", pa.int16(), nullable=False),
        pa.field(
            "future_novel_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_exposed_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "future_exposed_unclicked_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(path: Path, columns: Iterable[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    missing = set(columns) - set(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _split_for_user(
    user_id: str,
    *,
    seed: int,
    validation_fraction: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:semantic-user:{user_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "validation" if bucket < validation_fraction else "fit"


def _history_values(history: object, *, context: str) -> tuple[list[str], list[str]]:
    if not isinstance(history, list) or not history:
        raise ValueError(f"{context}: History must be a non-empty list")
    titles: list[str] = []
    subcategories: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            raise ValueError(f"{context}: malformed History item")
        title = item.get("title")
        subcategory = item.get("subcategory")
        if not isinstance(title, str) or not title:
            raise ValueError(f"{context}: History title must be non-empty")
        if not isinstance(subcategory, str) or not subcategory:
            raise ValueError(
                f"{context}: History subcategory must be non-empty"
            )
        titles.append(title)
        subcategories.append(subcategory)
    return titles, subcategories


def _user_row(
    *,
    user_id: str,
    split: str,
    history: object,
    positives: set[str],
    vocabulary: set[str],
) -> dict[str, object]:
    titles, history_labels = _history_values(
        history, context=f"{split} user {user_id!r}"
    )
    history_unique = sorted(set(history_labels))
    invalid_history = set(history_unique) - vocabulary
    invalid_positives = positives - vocabulary
    if invalid_history:
        raise ValueError(
            f"{split} user {user_id!r}: invalid History labels "
            f"{sorted(invalid_history)}"
        )
    if invalid_positives:
        raise ValueError(
            f"{split} user {user_id!r}: invalid positives "
            f"{sorted(invalid_positives)}"
        )
    if not positives:
        raise ValueError(f"{split} user {user_id!r}: no positive labels")
    overlap = set(history_unique) & positives
    if overlap:
        raise ValueError(
            f"{split} user {user_id!r}: targets occur in raw History "
            f"{sorted(overlap)}"
        )
    ordered_positives = sorted(positives)
    return {
        "user_id": user_id,
        "split": split,
        "history_length": len(titles),
        "history_unique_subcategory_count": len(history_unique),
        "history_titles": titles,
        "history_subcategories": history_labels,
        "history_unique_subcategories": history_unique,
        "target_count": len(ordered_positives),
        "future_novel_subcategories": ordered_positives,
        # The semantic experiment evaluates only the full-taxonomy track.  The
        # observed positives are retained here for evaluator schema compatibility;
        # this field must not be interpreted as a complete exposure set.
        "future_exposed_subcategories": ordered_positives,
        "future_exposed_unclicked_subcategories": [],
    }


def _write_rows(
    path: Path,
    rows: Iterable[dict[str, object]],
    *,
    batch_size: int,
) -> int:
    writer = pq.ParquetWriter(
        path,
        USER_SCHEMA,
        compression="zstd",
        use_dictionary=True,
    )
    pending: list[dict[str, object]] = []
    count = 0
    try:
        for row in rows:
            pending.append(row)
            count += 1
            if len(pending) >= batch_size:
                writer.write_table(
                    pa.Table.from_pylist(pending, schema=USER_SCHEMA),
                    row_group_size=batch_size,
                )
                pending.clear()
        if pending:
            writer.write_table(
                pa.Table.from_pylist(pending, schema=USER_SCHEMA),
                row_group_size=batch_size,
            )
    finally:
        writer.close()
    return count


def _recover_train_positives(
    candidate_path: Path,
    *,
    vocabulary: set[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    _require_columns(candidate_path, ("user_id", "target_interest"))
    positives: dict[str, set[str]] = defaultdict(set)
    candidate_rows = 0
    for batch in pq.ParquetFile(candidate_path).iter_batches(
        columns=["user_id", "target_interest"],
        batch_size=16_384,
    ):
        for row in batch.to_pylist():
            user_id = str(row["user_id"])
            target = row["target_interest"]
            if not user_id or not isinstance(target, dict):
                raise ValueError(f"{candidate_path}: malformed candidate row")
            label = target.get("subcategory")
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"{candidate_path}: candidate target is missing subcategory"
                )
            if label not in vocabulary:
                raise ValueError(
                    f"{candidate_path}: unknown target subcategory {label!r}"
                )
            positives[user_id].add(label)
            candidate_rows += 1
    if not positives:
        raise ValueError(f"{candidate_path}: no candidate users")
    return dict(positives), {
        "candidate_rows": candidate_rows,
        "candidate_users": len(positives),
        "distinct_user_interest_pairs": sum(map(len, positives.values())),
    }


def build_static_user_semantic_benchmark(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    candidate_file: Path | str = DEFAULT_CANDIDATE_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    batch_size: int = 8_192,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build user-level fit/validation/dev data without changing the base corpus."""
    dataset_dir = Path(dataset_dir)
    candidate_file = Path(candidate_file)
    output_dir = Path(output_dir)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite."
        )
    backup = output_dir.with_name(output_dir.name + ".old")
    if output_dir.exists() and overwrite and backup.exists():
        raise FileExistsError(f"Cannot overwrite while backup exists: {backup}")

    train_path = dataset_dir / "train.parquet"
    dev_path = dataset_dir / "dev.parquet"
    vocabulary_path = dataset_dir / "interest_vocabulary.parquet"
    _require_columns(train_path, ("user_id", "history", "target_interest"))
    _require_columns(dev_path, ("user_id", "history", "target_interest"))
    _require_columns(vocabulary_path, ("subcategory",))
    vocabulary_table = pq.read_table(vocabulary_path)
    vocabulary_labels = [
        str(value) for value in vocabulary_table["subcategory"].to_pylist()
    ]
    vocabulary = set(vocabulary_labels)
    if not vocabulary or len(vocabulary) != len(vocabulary_labels):
        raise ValueError(f"{vocabulary_path}: vocabulary must be unique")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".static-user-semantic-benchmark-",
        dir=str(output_dir.parent),
    ) as temporary_name:
        staging = Path(temporary_name) / "publish"
        staging.mkdir()
        train_positives, recovery_stats = _recover_train_positives(
            candidate_file,
            vocabulary=vocabulary,
        )

        writers = {
            split: pq.ParquetWriter(
                staging / f"{split}.parquet",
                USER_SCHEMA,
                compression="zstd",
                use_dictionary=True,
            )
            for split in ("fit", "validation")
        }
        pending: dict[str, list[dict[str, object]]] = {
            "fit": [],
            "validation": [],
        }
        split_counts: Counter[str] = Counter()
        target_count_distribution: Counter[int] = Counter()
        seen_train_users: set[str] = set()
        try:
            for batch in pq.ParquetFile(train_path).iter_batches(
                columns=["user_id", "history", "target_interest"],
                batch_size=batch_size,
            ):
                for source_row in batch.to_pylist():
                    user_id = str(source_row["user_id"])
                    if user_id in seen_train_users:
                        raise ValueError(
                            f"{train_path}: duplicate selected user {user_id!r}"
                        )
                    seen_train_users.add(user_id)
                    positives = train_positives.pop(user_id, None)
                    if positives is None:
                        raise ValueError(
                            f"{candidate_file}: no candidates for selected user "
                            f"{user_id!r}"
                        )
                    selected = source_row["target_interest"]
                    selected_label = (
                        selected.get("subcategory")
                        if isinstance(selected, dict)
                        else None
                    )
                    if selected_label not in positives:
                        raise ValueError(
                            f"{train_path}: selected target is absent from "
                            f"recovered positives for {user_id!r}"
                        )
                    split = _split_for_user(
                        user_id,
                        seed=seed,
                        validation_fraction=validation_fraction,
                    )
                    row = _user_row(
                        user_id=user_id,
                        split=split,
                        history=source_row["history"],
                        positives=positives,
                        vocabulary=vocabulary,
                    )
                    pending[split].append(row)
                    split_counts[split] += 1
                    target_count_distribution[int(row["target_count"])] += 1
                    if len(pending[split]) >= batch_size:
                        writers[split].write_table(
                            pa.Table.from_pylist(
                                pending[split], schema=USER_SCHEMA
                            ),
                            row_group_size=batch_size,
                        )
                        pending[split].clear()
            for split in writers:
                if pending[split]:
                    writers[split].write_table(
                        pa.Table.from_pylist(
                            pending[split], schema=USER_SCHEMA
                        ),
                        row_group_size=batch_size,
                    )
        finally:
            for writer in writers.values():
                writer.close()
        if train_positives:
            raise ValueError(
                f"{candidate_file}: {len(train_positives)} candidate users are "
                "absent from the selected Train corpus"
            )
        if sum(split_counts.values()) != recovery_stats["candidate_users"]:
            raise ValueError("selected Train users and candidate users disagree")

        dev_users: dict[str, dict[str, object]] = {}
        for batch in pq.ParquetFile(dev_path).iter_batches(
            columns=["user_id", "history", "target_interest"],
            batch_size=batch_size,
        ):
            for source_row in batch.to_pylist():
                user_id = str(source_row["user_id"])
                history = source_row["history"]
                target = source_row["target_interest"]
                label = (
                    target.get("subcategory")
                    if isinstance(target, dict)
                    else None
                )
                if not isinstance(label, str) or not label:
                    raise ValueError(f"{dev_path}: malformed target for {user_id!r}")
                titles, subcategories = _history_values(
                    history, context=f"dev user {user_id!r}"
                )
                signature = (tuple(titles), tuple(subcategories))
                state = dev_users.get(user_id)
                if state is None:
                    dev_users[user_id] = {
                        "history": history,
                        "signature": signature,
                        "positives": {label},
                    }
                else:
                    if state["signature"] != signature:
                        raise ValueError(
                            f"{dev_path}: inconsistent raw History for {user_id!r}"
                        )
                    positives = state["positives"]
                    assert isinstance(positives, set)
                    positives.add(label)

        def dev_rows() -> Iterable[dict[str, object]]:
            for user_id in sorted(dev_users):
                state = dev_users[user_id]
                positives = state["positives"]
                assert isinstance(positives, set)
                yield _user_row(
                    user_id=user_id,
                    split="dev",
                    history=state["history"],
                    positives=positives,
                    vocabulary=vocabulary,
                )

        dev_count = _write_rows(
            staging / "dev.parquet",
            dev_rows(),
            batch_size=batch_size,
        )
        pq.write_table(
            vocabulary_table,
            staging / "interest_vocabulary.parquet",
            compression="zstd",
            use_dictionary=True,
        )

        statistics = {
            "rule_version": RULE_VERSION,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "fit_users": split_counts["fit"],
            "validation_users": split_counts["validation"],
            "train_users": sum(split_counts.values()),
            "dev_users": dev_count,
            "dev_source_rows": pq.ParquetFile(dev_path).metadata.num_rows,
            "train_recovery": recovery_stats,
            "train_target_count_distribution": {
                str(count): frequency
                for count, frequency in sorted(
                    target_count_distribution.items()
                )
            },
            "dev_target_count_distribution": dict(
                sorted(
                    Counter(
                        len(state["positives"])
                        for state in dev_users.values()
                    ).items()
                )
            ),
        }
        source_paths = [
            train_path,
            dev_path,
            vocabulary_path,
            candidate_file,
        ]
        metadata = {
            "rule_version": RULE_VERSION,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "implementation_sha256": {
                portable_path(Path(__file__)): _sha256(Path(__file__))
            },
            "source_sha256": {
                portable_path(path): _sha256(path) for path in source_paths
            },
            "semantics": {
                "unit": "user",
                "train_positives": (
                    "distinct union of target rows in candidate_file by "
                    "selected Train user"
                ),
                "train_user_weight": "one total unit per user",
                "dev_positives": "distinct union of all retained Dev targets",
                "history": (
                    "exact History stored in dataset_dir; no impression replay"
                ),
                "dev_exposure_field": (
                    "observed positives only; exhaustive ranking over the "
                    "Train-supported label space"
                ),
            },
        }
        (staging / "statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
    return {
        "output_dir": str(output_dir),
        "statistics": statistics,
        "metadata": metadata,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--candidate-file", type=Path, default=DEFAULT_CANDIDATE_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_static_user_semantic_benchmark(
        args.dataset_dir,
        args.candidate_file,
        args.output_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
