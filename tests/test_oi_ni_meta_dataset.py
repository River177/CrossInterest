from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_generative_dataset import GENERATIVE_SCHEMA
from new_interest.data.build_oi_ni_meta_dataset import (
    HISTORY_HEADER,
    NO_HISTORY_PROMPT,
    OI_INSTRUCTION,
    build_episodes,
    build_oi_ni_meta_dataset,
    no_history_prompt,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(c: int, user: str, role: str) -> dict[str, object]:
    source_id = f"train:{user}:{c}:{role}"
    instruction = OI_INSTRUCTION if role == "old_interest" else "Predict an unseen topic."
    prompt = f"{HISTORY_HEADER}\n\n1. [Target {c}] Private headline for {user}\n\n{instruction}"
    response = f"<interest_{c:03d}>"
    return {
        "sample_id": source_id + ":behavior_interest", "source_sample_id": source_id,
        "split": "train", "task_type": "behavior_interest", "user_id": user,
        "impression_id": user, "timestamp": datetime(2019, 11, 14),
        "history_length": 1, "prompt": prompt, "response": response,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        "target_subcategory": f"Target {c}",
        **({"training_task": role} if role == "old_interest" else {}),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "sources"
    root.mkdir()
    ni = [_row(c, f"N{c}-{j}", "new_interest") for c in range(4) for j in range(6)]
    validation, nonvalidation = {}, {}
    for c in range(4):
        users = sorted((f"N{c}-{j}" for j in range(6)), key=lambda u: (
            hashlib.sha256(f"42:oi-ni-meta-validation:Target {c}:{u}".encode()).hexdigest(), u))
        validation[c], nonvalidation[c] = users[0], users[1]
    oi = []
    for c in range(4):
        oi.extend(_row(c, f"O{c}-{j}", "old_interest") for j in range(4))
        # Cross-task users can have different OI/NI labels. Removal must be by
        # user, not by label or source ID, and episode exclusions span classes.
        oi.append(_row(c, validation[(c + 1) % 4], "old_interest"))
        oi.append(_row(c, nonvalidation[(c + 1) % 4], "old_interest"))
    paths = {name: root / f"{name}.parquet" for name in ("ni", "oi", "mapping", "dev")}
    pq.write_table(pa.Table.from_pylist(ni, schema=GENERATIVE_SCHEMA), paths["ni"])
    oi_schema = GENERATIVE_SCHEMA.append(pa.field("training_task", pa.string()))
    pq.write_table(pa.Table.from_pylist(oi, schema=oi_schema), paths["oi"])
    pq.write_table(pa.Table.from_pylist([
        {"interest_index": c, "interest_token": f"<interest_{c:03d}>", "target_subcategory": f"Target {c}"}
        for c in range(4)]), paths["mapping"])
    # No Dev labels are needed to construct training data or episodes.
    pq.write_table(pa.Table.from_pylist([{"user_id": "DEV-only", "unused": "not for selection"}]), paths["dev"])
    return paths


def _build(paths: dict[str, Path], output: Path, **kwargs: object):
    options = {"validation_per_class": 1, "expected_classes": 4,
               "expected_ni_per_class": 6, "steps": 43, "episode_classes": 3}
    options.update(kwargs)
    return build_oi_ni_meta_dataset(paths["ni"], paths["oi"], paths["mapping"], output,
                                    dev_path=paths["dev"], **options)


def test_preserves_frozen_rows_and_excludes_validation_users_from_both_roles(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    hashes = {name: _hash(path) for name, path in paths.items()}
    output = tmp_path / "built"
    result = _build(paths, output)
    assert (result.fit_ni_rows, result.validation_ni_rows, result.fit_oi_rows) == (20, 4, 20)
    val = pq.read_table(output / "validation_ni.parquet")
    val_users = set(val["user_id"].to_pylist())
    for filename, original in (("fit_ni", paths["ni"]), ("validation_ni", paths["ni"]), ("fit_oi", paths["oi"])):
        table = pq.read_table(output / f"{filename}.parquet")
        source = pq.read_table(original)
        assert table.select(source.column_names).equals(source.take(table["source_row_index"]))
        assert table["row_index"].to_pylist() == list(range(len(table)))
        if filename.startswith("fit"):
            assert not val_users.intersection(table["user_id"].to_pylist())
    oi = pq.read_table(output / "fit_oi.parquet")
    assert set(oi["no_history_prompt"].to_pylist()) == {NO_HISTORY_PROMPT}
    assert "Private headline" not in NO_HISTORY_PROMPT
    assert all(f"<interest_{c:03d}>" not in NO_HISTORY_PROMPT for c in range(4))
    assert NO_HISTORY_PROMPT.endswith(OI_INSTRUCTION)
    assert "no_history_prompt" not in pq.read_table(output / "fit_ni.parquet").column_names
    assert _hash(output / "interest_token_mapping.parquet") == hashes["mapping"]
    assert _hash(output / "dev_full_grouped.parquet") == hashes["dev"]
    assert {name: _hash(path) for name, path in paths.items()} == hashes
    metadata = json.loads((output / "metadata.json").read_text())
    for filename, identity in metadata["artifacts"].items():
        assert _hash(output / filename) == identity["sha256"]
    assert metadata["episode_matching"].startswith("class_marginal_only")


def test_episodes_are_deterministic_balanced_and_user_disjoint(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    _build(paths, first)
    _build(paths, second)
    assert _hash(first / "episodes.parquet") == _hash(second / "episodes.parquet")
    assert _hash(first / "validation_manifest.parquet") == _hash(second / "validation_manifest.parquet")
    oi = pq.read_table(first / "fit_oi.parquet").to_pylist()
    ni = pq.read_table(first / "fit_ni.parquet").to_pylist()
    counts: Counter[int] = Counter()
    for episode in pq.read_table(first / "episodes.parquet").to_pylist():
        classes = episode["class_indices"]
        assert len(classes) == len(set(classes)) == 3
        support = [oi[i] for i in episode["support_indices"]]
        query = [ni[i] for i in episode["query_indices"]]
        assert len({row["user_id"] for row in support}) == 6
        assert len({row["user_id"] for row in query}) == 6
        assert not {row["user_id"] for row in support}.intersection(row["user_id"] for row in query)
        expected = Counter({c: 2 for c in classes})
        assert Counter(row["interest_index"] for row in support) == expected
        assert Counter(row["interest_index"] for row in query) == expected
        counts.update(classes)
    assert max(counts.values()) - min(counts.values()) == 1


def test_108_classes_cross_round_boundaries_without_duplicate_classes() -> None:
    labels = {c: f"Target {c}" for c in range(108)}
    support, query = [], []
    for c in labels:
        for j in range(3):
            support.append({"row_index": len(support), "interest_index": c, "user_id": f"S{c}-{j}"})
        for j in range(5):
            query.append({"row_index": len(query), "interest_index": c,
                          "user_id": f"S{(c + 1) % 108}-{j}" if j < 3 else f"Q{c}-{j}"})
    episodes, slots, stats = build_episodes(
        pa.Table.from_pylist(support), pa.Table.from_pylist(query), labels=labels,
        seed=42, steps=540, episode_classes=8, support_per_class=2, query_per_class=2)
    assert len(episodes) == 540
    assert len(slots) == 4320
    assert set(stats["class_appearances"].values()) == {40}
    for episode in episodes.to_pylist():
        assert len(set(episode["class_indices"])) == 8
        su = {support[i]["user_id"] for i in episode["support_indices"]}
        qu = {query[i]["user_id"] for i in episode["query_indices"]}
        assert len(su) == len(qu) == 16
        assert su.isdisjoint(qu)


def test_validation_membership_is_stable_under_source_row_reordering(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    _build(paths, first, steps=1)
    ni = pq.read_table(paths["ni"])
    pq.write_table(ni.take(pa.array(list(reversed(range(len(ni)))))), paths["ni"])
    _build(paths, second, steps=1)
    assert set(pq.read_table(first / "validation_ni.parquet")["user_id"].to_pylist()) == set(
        pq.read_table(second / "validation_ni.parquet")["user_id"].to_pylist())


def test_rejects_sparse_oi_class_after_validation_removal(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    oi = pq.read_table(paths["oi"])
    retained = [i for i, row in enumerate(oi.to_pylist())
                if row["target_subcategory"] != "Target 0" or row["user_id"] == "O0-0"]
    pq.write_table(oi.take(pa.array(retained)), paths["oi"])
    with pytest.raises(ValueError, match="fit OI has 1 independent users"):
        _build(paths, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("fault", ["duplicate_user", "wrong_token", "unknown_oi_template", "dev_overlap"])
def test_rejects_identity_or_information_boundary_violations(tmp_path: Path, fault: str) -> None:
    paths = _fixture(tmp_path)
    source = "oi" if fault == "unknown_oi_template" else "ni"
    table = pq.read_table(paths[source])
    rows = table.to_pylist()
    if fault == "duplicate_user":
        rows[1]["user_id"] = rows[0]["user_id"]
    elif fault == "wrong_token":
        rows[0]["response"] = "<interest_999>"
    elif fault == "unknown_oi_template":
        rows[0]["prompt"] = "Target 0 is the answer"
        rows[0]["messages"][0]["content"] = rows[0]["prompt"]
    else:
        pq.write_table(pa.Table.from_pylist([{"user_id": rows[0]["user_id"]}]), paths["dev"])
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), paths[source])
    with pytest.raises(ValueError):
        _build(paths, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_never_replaces_existing_directory(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        _build(paths, output)
    assert sentinel.read_text() == "unchanged"


def test_verifies_config_frozen_input_hashes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_root = paths["ni"].parent
    config = source_root / "source_config.json"
    config.write_text(json.dumps({"new_interest": {"input_files": [
        {"path": paths[k].name, "sha256": _hash(paths[k])} for k in ("ni", "mapping", "dev")
    ]}}))
    _build(paths, tmp_path / "good", steps=1, source_config_path=config, source_root=source_root)
    dev = pq.read_table(paths["dev"])
    pq.write_table(dev.append_column("new_column", pa.array([1])), paths["dev"])
    with pytest.raises(ValueError, match="dev: source differs"):
        _build(paths, tmp_path / "bad", source_config_path=config, source_root=source_root)


def test_no_history_ablation_rejects_target_bearing_task_suffix() -> None:
    with pytest.raises(ValueError, match="template"):
        no_history_prompt(f"{HISTORY_HEADER}\n\n1. [T] Title\n\nAnswer Target 0.")


def test_config_identity_handles_worktree_data_symlink(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "data").symlink_to(paths["ni"].parent, target_is_directory=True)
    linked = {name: checkout / "data" / path.name for name, path in paths.items()}
    config = checkout / "frozen.json"
    config.write_text(json.dumps({"new_interest": {"input_files": [
        {"path": f"data/{paths[k].name}", "sha256": _hash(paths[k])} for k in ("ni", "mapping", "dev")
    ]}}))
    result = _build(linked, tmp_path / "built", steps=1, source_config_path=config, source_root=checkout)
    assert result.fit_ni_rows == 20
