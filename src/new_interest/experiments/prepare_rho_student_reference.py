"""Prepare source-ordered scalar OOF losses from the accepted remote artifacts.

Run on the remote student worker. This module has no transfer/download client:
it rereads the eight existing score artifacts where they live, verifies them,
and writes only the requested scalar-reference sidecar to a new output path.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.experiments.read_rho_reference import (
    EXPECTED_LOSS, verify_kernel_selection, verify_reference,
)
from new_interest.experiments.rho_reference_recovery import _overlap
from new_interest.training.rho_dynamic_weights import canonical_sha256, source_identity


VERSION = "rho-student-accepted-oof-reference-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def _source_rows(path):
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("Student source cohort is empty")
    ids, users = set(), set()
    for row in rows:
        digest = source_identity(row)
        sample, user = row["source_sample_id"], row["user_id"]
        if sample in ids or user in users:
            raise ValueError("Student source repeats a sample or user")
        if "messages" not in row:
            raise ValueError("Student source must include the original messages for identity verification")
        if row.get("training_task", "new_interest") != "new_interest":
            raise ValueError("Only new-interest student source rows are supported")
        row["input_sha256"] = digest
        ids.add(sample)
        users.add(user)
    return rows


def _source_digest(rows):
    return canonical_sha256([{key: row[key] for key in
        ("source_sample_id", "user_id", "target_subcategory", "input_sha256")} for row in rows])


def _verify_output(table, source, joint_sha):
    if len(table) != len(source):
        raise ValueError("Prepared reference row count differs from requested source")
    for index, (row, original) in enumerate(zip(table.to_pylist(), source)):
        if row.get("source_row_index") != index or row.get("joint_acceptance_sha256") != joint_sha:
            raise ValueError("Prepared reference order or original acceptance identity differs")
        for field in ("source_sample_id", "user_id", "target_subcategory", "input_sha256"):
            if row.get(field) != original[field]:
                raise ValueError("Prepared reference source identity mismatch: " + field)
        if not math.isfinite(row["full_vocab_first_sid_nll"]):
            raise ValueError("Prepared reference has nonfinite loss")


def prepare_reference(acceptance_path, expected_sha, train_file, output_dir, *, expected_total_rows=21600):
    """Reaudit both complete folds once, then select exact real source rows.

    ``expected_total_rows`` is an explicit small-fixture seam; production uses
    the default 21600. Committed output can be reused by its bound acceptance,
    requested-source hash and Parquet hash without rereading remote teachers.
    """
    acceptance_path, train_file, output_dir = Path(acceptance_path), Path(train_file), Path(output_dir)
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)):
        raise ValueError("Expected acceptance SHA-256 must be explicitly supplied")
    if sha(acceptance_path) != expected_sha:
        raise ValueError("Joint acceptance differs from its frozen SHA-256")
    if type(expected_total_rows) is not int or expected_total_rows <= 0 or expected_total_rows % 2:
        raise ValueError("Expected complete reference size must be a positive even integer")
    acceptance = read(acceptance_path)
    if (acceptance.get("passed") is not True or acceptance.get("rows") != expected_total_rows
            or acceptance.get("users") != expected_total_rows
            or acceptance.get("twofold_union_exact_from_verified_complementary_inputs") is not True
            or acceptance.get("kernel_selection") != "stable_fla_v1"
            or acceptance.get("common_kernel_configs_consistent") is not True):
        raise ValueError("Joint report does not accept the complete stable twofold reference")
    folds = acceptance.get("folds", [])
    if len(folds) != 2 or sorted(fold["worker"]["fold"] for fold in folds) != [0, 1]:
        raise ValueError("Joint acceptance must contain each teacher fold once")
    protected = [acceptance_path, train_file]
    for fold in folds:
        restoration = fold["worker"]["restoration"]
        protected.extend(Path(restoration[name]) for name in ("teacher", "train_file", "score_file", "mapping_file"))
        protected.extend(Path(directory) for directory in restoration["score_dirs"])
        if "parent_checkpoint" in fold["worker"]:
            protected.append(Path(fold["worker"]["parent_checkpoint"]).parent)
    if any(_overlap(output_dir, path) for path in protected):
        raise ValueError("Reference output would overlap immutable source or teacher artifacts")
    source = _source_rows(train_file)
    identity = {"version": VERSION, "joint_acceptance_sha256": expected_sha,
                "source_train_file_sha256": sha(train_file), "source_identity_sha256": _source_digest(source),
                "expected_total_rows": expected_total_rows}
    receipt_path, target_file = output_dir / "reference_preparation.json", output_dir / "reference_scores.parquet"
    if output_dir.exists():
        if not receipt_path.is_file():
            raise FileExistsError("Uncommitted reference output already exists: " + str(output_dir))
        receipt = read(receipt_path)
        if receipt.get("identity") != identity or receipt.get("passed") is not True or sha(target_file) != receipt["file_sha256"]:
            raise ValueError("Existing prepared reference identity or bytes differ")
        _verify_output(pq.read_table(target_file), source, expected_sha)
        return {"file": str(target_file), "sha256": receipt["file_sha256"], "rows": len(source),
                "receipt": receipt, "receipt_file": str(receipt_path)}

    complete, score_users, fold_reports, artifact_receipts, common_kernels = {}, set(), [], [], {}
    for fold in sorted(folds, key=lambda item: item["worker"]["fold"]):
        worker, accepted = fold["worker"], fold["reference"]
        fold_index, restoration = worker["fold"], worker["restoration"]
        if (worker.get("passed") is not True or worker.get("new_training_updates") != 0
                or accepted.get("accepted") is not True or accepted.get("loss_definition") != EXPECTED_LOSS
                or accepted.get("kernel_selection") != "stable_fla_v1"
                or accepted.get("teacher_score_user_overlap") != 0
                or accepted["metrics"]["rows"] != expected_total_rows // 2):
            raise ValueError("A fold lacks accepted disjoint stable first-SID scoring")
        teacher = Path(restoration["teacher"])
        original_report = teacher.parent / "compact/reference_acceptance.json"
        if sha(original_report) != worker["reference_acceptance_sha256"] or read(original_report) != accepted:
            raise ValueError("Remote per-fold acceptance differs from the bound joint report")
        if sha(teacher / "final_adapter/adapter_model.safetensors") != worker["parent_adapter_sha256"]:
            raise ValueError("Scoring adapter differs from its accepted original training checkpoint")
        directories = [Path(directory) for directory in restoration["score_dirs"]]
        bound_artifacts = {item["directory"]: item for item in accepted["artifacts"]}
        if len(directories) != 4 or len(bound_artifacts) != 4 or set(map(str, directories)) != set(bound_artifacts):
            raise ValueError("Each accepted teacher must supply its exact four score artifacts")
        verified = verify_reference(score_dirs=directories, teacher_train_file=restoration["train_file"],
            score_file=restoration["score_file"], mapping_file=restoration["mapping_file"],
            adapter_path=teacher / "final_adapter", teacher_run_summary=teacher / "run_summary.json")
        for field in ("evaluation_identity", "loss_definition", "teacher_train_sha256", "score_file_sha256",
                      "mapping_sha256", "adapter_sha256", "scored_source_ids_sha256", "metrics"):
            if verified[field] != accepted[field]:
                raise ValueError("Fresh remote readback differs from accepted reference: " + field)
        fold_reports.append(verified)
        for directory in directories:
            bound = bound_artifacts[str(directory)]
            if (sha(directory / "manifest.json") != bound["manifest_sha256"]
                    or sha(directory / "scores.parquet") != bound["scores_sha256"]):
                raise ValueError("Accepted score artifact hash differs")
            manifest = read(directory / "manifest.json")
            verify_kernel_selection(directory, manifest, common_kernels)
            for record in pq.read_table(directory / "scores.parquet").to_pylist():
                sample, user = record["source_sample_id"], record["user_id"]
                if sample in complete or user in score_users:
                    raise ValueError("Complete OOF reference repeats a source sample or user across folds")
                complete[sample] = {**{key: record[key] for key in ("source_sample_id", "user_id", "target_subcategory",
                    "input_sha256", "full_vocab_first_sid_nll", "evaluation_identity")},
                    "teacher_fold": fold_index, "teacher_train_sha256": accepted["teacher_train_sha256"],
                    "parent_adapter_sha256": worker["parent_adapter_sha256"], "joint_acceptance_sha256": expected_sha}
                score_users.add(user)
            artifact_receipts.append({"directory": str(directory), "manifest_sha256": bound["manifest_sha256"],
                                      "scores_sha256": bound["scores_sha256"]})
    if (fold_reports[0]["teacher_train_sha256"] != fold_reports[1]["score_file_sha256"]
            or fold_reports[1]["teacher_train_sha256"] != fold_reports[0]["score_file_sha256"]
            or fold_reports[0]["mapping_sha256"] != fold_reports[1]["mapping_sha256"]):
        raise ValueError("The teachers did not score their exact complementary training halves")
    if len(complete) != expected_total_rows or len(score_users) != expected_total_rows:
        raise ValueError("Actual full reference union size differs from accepted twofold coverage")
    full_counts = Counter(record["target_subcategory"] for record in complete.values())
    if expected_total_rows == 21600 and (len(full_counts) != 108 or set(full_counts.values()) != {200}):
        raise ValueError("Full reference class support differs from 108 classes by 200 examples")
    selected = []
    for index, row in enumerate(source):
        if row["source_sample_id"] not in complete:
            raise ValueError("Student source is not a real subset of the accepted OOF cohort")
        reference = complete[row["source_sample_id"]]
        for field in ("user_id", "target_subcategory", "input_sha256"):
            if reference[field] != row[field]:
                raise ValueError("Student source differs from its scored OOF input: " + field)
        selected.append({"source_row_index": index, **reference})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".staging-", dir=output_dir.parent))
    try:
        pq.write_table(pa.Table.from_pylist(selected), staging / target_file.name, compression="zstd")
        _verify_output(pq.read_table(staging / target_file.name), source, expected_sha)
        receipt = {"passed": True, "identity": identity, "joint_acceptance_path": str(acceptance_path),
            "joint_acceptance_sha256": expected_sha, "file_sha256": sha(staging / target_file.name),
            "source_rows": len(source), "verified_full_reference_rows": len(complete), "score_artifacts_verified": 8,
            "subset": len(source) < len(complete), "source_order_preserved": True,
            "source_inputs_exact": True, "teacher_score_user_overlap": 0, "twofold_union_unique": True,
            "full_class_support": dict(sorted(full_counts.items())),
            "selected_class_support": dict(sorted(Counter(row["target_subcategory"] for row in selected).items())),
            "kernel_selection": "stable_fla_v1", "kernel_cache_keys_checked": len(common_kernels),
            "common_kernel_configs_consistent": True, "artifacts": artifact_receipts,
            "reference_evaluation_identities": [report["evaluation_identity"] for report in fold_reports],
            "original_fold1_training_reference_equivalence_established": acceptance.get(
                "original_fold1_training_reference_equivalence_established", False),
            "verification_scope": "fresh remote independent reader for all eight score artifacts, kernels, source/teacher identity and NLL arithmetic; no second model inference",
            "large_artifact_policy": "remote-only reads and scalar-reference output; no download client"}
        (staging / receipt_path.name).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"file": str(target_file), "sha256": receipt["file_sha256"], "rows": len(source),
            "receipt": receipt, "receipt_file": str(receipt_path)}
