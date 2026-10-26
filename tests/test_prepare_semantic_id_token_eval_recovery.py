from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.prepare_semantic_id_token_eval_recovery import (
    clean_stale_eval_caches,
)


RESULT_ID = "7212919146.88353-1d349268-8286"
STALE_IDENTITY = "a" * 64


def _build_source(tmp_path: Path) -> Path:
    root = tmp_path / RESULT_ID
    adapter = root / "sft" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (root / "sft" / "run_summary.json").write_text(
        json.dumps({"expected_optimizer_steps": 2025}),
        encoding="utf-8",
    )
    for index in range(4):
        shard = root / "eval" / f"shard-{index:02d}"
        cache = shard / "shards"
        cache.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [{"evaluation_identity": STALE_IDENTITY, "record_index": index}]
            ),
            cache / "000000000-000000001.parquet",
        )
        (shard / "keep.log").write_text("diagnostic evidence", encoding="utf-8")
    return root


def test_cleanup_removes_only_stale_shard_caches(tmp_path: Path) -> None:
    root = _build_source(tmp_path)
    manifest_path = tmp_path / "cleanup.json"

    manifest = clean_stale_eval_caches(
        root,
        manifest_path,
        expected_source_result_id=RESULT_ID,
        expected_stale_identity=STALE_IDENTITY,
    )

    assert len(manifest["removed_cache_directories"]) == 4
    assert all(row["status"] == "removed" for row in manifest["shards"])
    for index in range(4):
        shard = root / "eval" / f"shard-{index:02d}"
        assert not (shard / "shards").exists()
        assert (shard / "keep.log").read_text() == "diagnostic evidence"
    assert (root / "sft" / "final_adapter" / "adapter_model.safetensors").is_file()
    assert json.loads(manifest_path.read_text()) == manifest


def test_cleanup_is_idempotent_after_success(tmp_path: Path) -> None:
    root = _build_source(tmp_path)
    clean_stale_eval_caches(
        root,
        tmp_path / "first.json",
        expected_source_result_id=RESULT_ID,
        expected_stale_identity=STALE_IDENTITY,
    )
    manifest = clean_stale_eval_caches(
        root,
        tmp_path / "second.json",
        expected_source_result_id=RESULT_ID,
        expected_stale_identity=STALE_IDENTITY,
    )
    assert manifest["removed_cache_directories"] == []
    assert all(row["status"] == "already_absent" for row in manifest["shards"])


def test_cleanup_rejects_unexpected_identity(tmp_path: Path) -> None:
    root = _build_source(tmp_path)
    with pytest.raises(ValueError, match="Unexpected evaluation identities"):
        clean_stale_eval_caches(
            root,
            tmp_path / "cleanup.json",
            expected_source_result_id=RESULT_ID,
            expected_stale_identity="b" * 64,
        )
    assert (root / "eval" / "shard-00" / "shards").is_dir()


def test_cleanup_rejects_wrong_source_result(tmp_path: Path) -> None:
    root = _build_source(tmp_path)
    with pytest.raises(ValueError, match="Source result identity mismatch"):
        clean_stale_eval_caches(
            root,
            tmp_path / "cleanup.json",
            expected_source_result_id="wrong-result",
            expected_stale_identity=STALE_IDENTITY,
        )
