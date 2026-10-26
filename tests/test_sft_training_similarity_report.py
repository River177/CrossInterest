from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from new_interest.experiments.report_sft_training_similarity import (
    write_sft_training_similarity_report,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifacts(root: Path) -> tuple[Path, Path]:
    artifact = root / "artifact"
    artifact.mkdir()
    _write_json(
        artifact / "resolved_config.json",
        {
            "rule_version": "sft-training-similarity-v1",
            "history_limit": 50,
            "row_weighting": "one equal weight per SFT row",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "revision-test",
                "dimensions": 384,
            },
            "bootstrap": {
                "replicates": 2000,
                "seed": 17,
                "confidence": 0.95,
            },
            "permutation": {
                "replicates": 2000,
                "seed": 17,
                "confidence": 0.95,
            },
        },
    )
    _write_json(
        artifact / "summary.json",
        {
            "rows": {"train": 22000, "dev": 10000},
            "primary_rows": {"train": 21743, "dev": 9862},
            "rows_without_novel_controls": {"train": 257, "dev": 138},
            "counts": {
                "train": {
                    "positive_titles": 22151,
                    "controls": 1047208,
                    "novel_controls": 688268,
                    "duplicate_control_tokens": 3,
                },
                "dev": {
                    "positive_titles": 12346,
                    "controls": 375636,
                    "novel_controls": 263798,
                    "duplicate_control_tokens": 1,
                },
            },
        },
    )
    similarity_rows = []
    for split, rows, positive, control, delta in (
        ("train", 21743, 0.61, 0.48, 0.13),
        ("dev", 9862, 0.59, 0.47, 0.12),
    ):
        for score_name in (
            "nearest_50",
            "top3_mean",
            "centroid",
            "nearest_5",
            "last",
        ):
            similarity_rows.append(
                {
                    "split": split,
                    "score_name": score_name,
                    "control_scope": "novel",
                    "rows": rows,
                    "excluded_rows": (
                        22000 - rows if split == "train" else 10000 - rows
                    ),
                    "positive_mean": positive,
                    "positive_ci_low": positive - 0.01,
                    "positive_ci_high": positive + 0.01,
                    "positive_p10": 0.2,
                    "positive_p25": 0.4,
                    "positive_p50": 0.6,
                    "positive_p75": 0.7,
                    "positive_p90": 0.8,
                    "control_mean": control,
                    "control_ci_low": control - 0.01,
                    "control_ci_high": control + 0.01,
                    "delta_mean": delta,
                    "delta_ci_low": delta - 0.01,
                    "delta_ci_high": delta + 0.01,
                    "auc_mean": 0.68,
                    "auc_ci_low": 0.67,
                    "auc_ci_high": 0.69,
                    "hit_at_1_mean": 0.41,
                    "hit_at_1_ci_low": 0.40,
                    "hit_at_1_ci_high": 0.42,
                    "positive_le_control_mean_rate": 0.30,
                    "positive_le_control_mean_ci_low": 0.29,
                    "positive_le_control_mean_ci_high": 0.31,
                    "positive_le_control_max_rate": 0.70,
                    "positive_le_control_max_ci_low": 0.69,
                    "positive_le_control_max_ci_high": 0.71,
                }
            )
            if score_name == "nearest_50":
                all_row = dict(similarity_rows[-1])
                all_row.update(
                    {
                        "control_scope": "all",
                        "rows": 22000 if split == "train" else 10000,
                        "excluded_rows": 0,
                        "control_mean": control + 0.02,
                        "delta_mean": delta - 0.02,
                        "auc_mean": 0.52,
                        "hit_at_1_mean": 0.08,
                    }
                )
                similarity_rows.append(all_row)
    _write_csv(artifact / "similarity_metrics.csv", similarity_rows)
    category_similarity_rows = []
    for split, rows in (("train", 21743), ("dev", 9862)):
        for name_level, positive, control in (
            ("subcategory", 0.72, 0.61),
            ("parent", 0.81, 0.73),
        ):
            for scope in ("novel", "all"):
                category_similarity_rows.append(
                    {
                        "split": split,
                        "name_level": name_level,
                        "score_name": "nearest_50",
                        "control_scope": scope,
                        "rows": rows if scope == "novel" else (
                            22000 if split == "train" else 10000
                        ),
                        "excluded_rows": (
                            257 if split == "train" else 138
                        ) if scope == "novel" else 0,
                        "positive_mean": positive,
                        "positive_ci_low": positive - 0.01,
                        "positive_ci_high": positive + 0.01,
                        "positive_p10": 0.4,
                        "positive_p25": 0.6,
                        "positive_p50": positive,
                        "positive_p75": 0.85,
                        "positive_p90": 0.9,
                        "control_mean": control,
                        "control_ci_low": control - 0.01,
                        "control_ci_high": control + 0.01,
                        "delta_mean": positive - control,
                        "delta_ci_low": positive - control - 0.01,
                        "delta_ci_high": positive - control + 0.01,
                        "auc_mean": 0.66,
                        "auc_ci_low": 0.65,
                        "auc_ci_high": 0.67,
                        "hit_at_1_mean": 0.31,
                        "hit_at_1_ci_low": 0.30,
                        "hit_at_1_ci_high": 0.32,
                        "positive_le_control_mean_rate": 0.32,
                        "positive_le_control_mean_ci_low": 0.31,
                        "positive_le_control_mean_ci_high": 0.33,
                        "positive_le_control_max_rate": 0.69,
                        "positive_le_control_max_ci_low": 0.68,
                        "positive_le_control_max_ci_high": 0.70,
                    }
                )
    _write_csv(
        artifact / "category_name_similarity_metrics.csv",
        category_similarity_rows,
    )
    taxonomy_rows = []
    for split in ("train", "dev"):
        for scope, observed in (
            ("full", 0.72),
            ("recent5", 0.48),
            ("last", 0.25),
        ):
            taxonomy_rows.append(
                {
                    "split": split,
                    "history_scope": scope,
                    "metric": "same_parent_fraction",
                    "rows": 22000 if split == "train" else 10000,
                    "observed": observed,
                    "observed_ci_low": observed - 0.01,
                    "observed_ci_high": observed + 0.01,
                    "permutation_expected": observed - 0.10,
                    "permutation_null_ci_low": observed - 0.11,
                    "permutation_null_ci_high": observed - 0.09,
                    "observed_minus_expected": 0.10,
                    "lift": 1.16,
                }
            )
    _write_csv(artifact / "taxonomy_metrics.csv", taxonomy_rows)
    _write_csv(
        artifact / "slice_metrics.csv",
        [
            {
                "split": "train",
                "slice_type": "taxonomy_semantic_quadrant",
                "slice_value": "cross_parent__higher_than_control",
                "membership_policy": "single",
                "rows": 100,
                "primary_rows": 100,
                "same_parent_fraction": 0,
                "positive_similarity": 0.6,
                "all_control_delta": 0.1,
                "novel_control_delta": 0.12,
                "all_control_auc": 0.65,
                "novel_control_auc": 0.68,
                "all_control_hit_at_1": 0.4,
                "novel_control_hit_at_1": 0.42,
            }
        ],
    )
    _write_csv(
        artifact / "parent_transition_matrix.csv",
        [
            {
                "split": "train",
                "last_history_parent": "news",
                "target_parent": "sports",
                "target_occurrences": 10,
                "row_weight": 10.0,
                "same_parent": 0,
            }
        ],
    )
    _write_json(
        artifact / "prompt_overlap.json",
        {
            "user_overlap": {
                "train_users": 20000,
                "dev_users": 9500,
                "users": 339,
                "dev_rows": 339,
            },
            "tasks": {
                "behavior": {
                    "train_duplicate_row_excess": 10,
                    "train_conflicting_prompt_count": 2,
                    "dev_exact_prompt_overlap_rows": 407,
                    "dev_any_target_seen_for_prompt_rows": 15,
                    "dev_all_targets_seen_for_prompt_rows": 10,
                },
                "interest": {
                    "train_duplicate_row_excess": 20,
                    "train_conflicting_prompt_count": 3,
                    "dev_exact_prompt_overlap_rows": 733,
                    "dev_any_target_seen_for_prompt_rows": 95,
                    "dev_all_targets_seen_for_prompt_rows": 80,
                },
            },
        },
    )
    _write_json(
        artifact / "cases.json",
        {
            "cases": [
                {
                    "case_id": "train:cross:1",
                    "row_id": "train:1:1",
                    "split": "train",
                    "quadrant": "cross_parent__higher_than_control",
                    "source_sample_ids": ["train:1:1"],
                    "impression_id": "1",
                    "user_id": "U1",
                    "behavior_prompt": "Behavior prompt",
                    "interest_prompt": "Interest prompt",
                    "history": [
                        {
                            "news_id": "H1",
                            "category": "news",
                            "subcategory": "local",
                            "title": "Recent history title",
                        }
                    ],
                    "target_categories": ["sports"],
                    "target_subcategories": ["soccer"],
                    "positive_news": [
                        {
                            "news_id": "P1",
                            "category": "sports",
                            "subcategory": "soccer",
                            "title": "Positive title",
                            "nearest_50": 0.7,
                            "nearest_history": {
                                "news_id": "H1",
                                "category": "news",
                                "subcategory": "local",
                                "title": "Recent history title",
                                "cosine": 0.7,
                            },
                            "rank_among_all_controls": 1,
                            "rank_among_novel_controls": 1,
                        }
                    ],
                    "top_all_controls": [],
                    "top_novel_controls": [
                        {
                            "news_id": "C1",
                            "category": "finance",
                            "subcategory": "markets",
                            "title": "Control title",
                            "nearest_50": 0.4,
                            "candidate_index": 0,
                        }
                    ],
                    "metrics": {
                        "same_parent_fraction": 0,
                        "nearest_50_positive_mean": 0.7,
                        "nearest_50_novel_control_mean": 0.4,
                        "nearest_50_novel_delta": 0.3,
                        "nearest_50_novel_auc": 1.0,
                        "nearest_50_novel_hit_at_1": 1.0,
                    },
                }
            ]
        },
    )
    (artifact / "analysis_rows.parquet").write_bytes(b"analysis")
    (artifact / "title_embeddings.parquet").write_bytes(b"embeddings")
    (artifact / "category_name_embeddings.parquet").write_bytes(
        b"category embeddings"
    )
    output_files = [
        path
        for path in artifact.iterdir()
        if path.name != "manifest.json"
    ]
    _write_json(
        artifact / "manifest.json",
        {
            "rule_version": "sft-training-similarity-v1",
            "artifact_dir": str(artifact),
            "inputs": {
                "sft_train_behavior": "data/train.parquet",
                "source_dev": "data/dev.parquet",
            },
            "output_sha256": {
                path.name: _sha256(path) for path in output_files
            },
        },
    )
    translations = root / "translations.json"
    _write_json(
        translations,
        {
            "translations": {
                "Behavior prompt": "行为任务提示",
                "New topic: soccer": "新主题：足球",
                "Recent history title": "近期历史标题",
                "Positive title": "正样本标题",
                "Control title": "对照标题",
            }
        },
    )
    return artifact, translations


def test_writes_five_part_bilingual_report(tmp_path: Path) -> None:
    artifact, translations = _write_artifacts(tmp_path)
    output = tmp_path / "report.md"

    result = write_sft_training_similarity_report(
        artifact,
        output,
        translations_path=translations,
    )

    report = output.read_text(encoding="utf-8")
    headings = [
        "## 1. 研究问题（Research Questions）",
        "## 2. 训练数据构造（Training Data Construction）",
        "## 3. 测试设计（Testing）",
        "## 4. 实验结果（Results）",
        "## 5. 数据与产物路径（Artifact Paths）",
    ]
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Behavior prompt" in report
    assert "行为任务提示" in report
    assert "Positive title" in report
    assert "正样本标题" in report
    assert "类别名称语义相似度" in report
    assert "subcategory" in report
    assert "21,743" in report
    assert "0.1300" in report
    assert "LLM judge" in report
    assert str((artifact / "analysis_rows.parquet").absolute()) in report
    assert "## 局限" not in report
    assert "## 结论" not in report
    assert Path(result["key_results_path"]).is_file()


def test_report_requires_hashes_for_every_consumed_artifact(
    tmp_path: Path,
) -> None:
    artifact, translations = _write_artifacts(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["output_sha256"]["summary.json"]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="missing hashes"):
        write_sft_training_similarity_report(
            artifact,
            tmp_path / "report.md",
            translations_path=translations,
        )
