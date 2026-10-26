from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_outputs(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    expected = manifest.get("output_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("manifest.json is missing output_sha256")
    required = {
        "resolved_config.json",
        "summary.json",
        "analysis_rows.parquet",
        "title_embeddings.parquet",
        "category_name_embeddings.parquet",
        "similarity_metrics.csv",
        "category_name_similarity_metrics.csv",
        "taxonomy_metrics.csv",
        "slice_metrics.csv",
        "parent_transition_matrix.csv",
        "prompt_overlap.json",
        "cases.json",
    }
    missing_hashes = required - {str(name) for name in expected}
    if missing_hashes:
        raise ValueError(
            "manifest.json is missing hashes for consumed artifacts: "
            f"{sorted(missing_hashes)}"
        )
    for name, digest in expected.items():
        path = artifact_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"Artifact hash mismatch for {path}: {actual} != {digest}"
            )


def _translation_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = _read_json(path).get("translations")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: translations must be an object")
    return {str(key): str(translation) for key, translation in value.items()}


def _translate(text: str, translations: Mapping[str, str]) -> str:
    try:
        translation = translations[text]
    except KeyError as error:
        raise ValueError(
            f"Missing faithful Chinese translation for {text!r}"
        ) from error
    if not translation.strip():
        raise ValueError(f"Empty Chinese translation for {text!r}")
    return translation


def _int(value: Any) -> str:
    return f"{int(value):,}"


def _float(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _ci(row: Mapping[str, Any], prefix: str) -> str:
    return (
        f"[{_float(row[f'{prefix}_ci_low'])}, "
        f"{_float(row[f'{prefix}_ci_high'])}]"
    )


def _find_row(
    rows: Sequence[Mapping[str, str]],
    **conditions: str,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(name) == value for name, value in conditions.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one CSV row for {conditions}, found {len(matches)}"
        )
    return matches[0]


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _markdown_path(path: Path) -> str:
    absolute = _absolute_without_resolving(path)
    return f"[`{absolute}`]({absolute})"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{uuid4().hex}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _training_example(
    case: Mapping[str, Any],
    translations: Mapping[str, str],
) -> list[str]:
    prompt = str(case["behavior_prompt"])
    target = str(case["target_subcategories"][0])
    response = f"New topic: {target}"
    return [
        "### 2.2 真实训练样本",
        "",
        f"- **Provenance:** `{case['row_id']}`；impression "
        f"`{case['impression_id']}`；source "
        f"`{case['source_sample_ids'][0]}`；label `{target}`。",
        "- **Record type:** behavior-only target record（不是 control 或 "
        "held-out response）。",
        "",
        "**Original prompt**",
        "",
        "```text",
        prompt,
        "```",
        "",
        "**中文翻译**",
        "",
        "```text",
        _translate(prompt, translations),
        "```",
        "",
        "**Target response**",
        "",
        "```text",
        response,
        "```",
        "",
        "**中文翻译**",
        "",
        "```text",
        _translate(response, translations),
        "```",
        "",
    ]


def _case_block(
    case: Mapping[str, Any],
    translations: Mapping[str, str],
) -> list[str]:
    metrics = case["metrics"]
    lines = [
        f"#### {case['quadrant']} — `{case['row_id']}`",
        "",
        f"- **Split / provenance:** `{case['split']}`；impression "
        f"`{case['impression_id']}`；source "
        f"`{', '.join(case['source_sample_ids'])}`。",
        f"- **Target taxonomy:** parent "
        f"`{', '.join(case['target_categories'])}`；subcategory "
        f"`{', '.join(case['target_subcategories'])}`。",
        f"- **Metrics:** positive={_float(metrics['nearest_50_positive_mean'])}, "
        f"novel-control={_float(metrics['nearest_50_novel_control_mean'])}, "
        f"delta={_float(metrics['nearest_50_novel_delta'])}, "
        f"AUC={_float(metrics['nearest_50_novel_auc'])}, "
        f"Hit@1={_float(metrics['nearest_50_novel_hit_at_1'])}。",
        "",
        "**最近 history 标题（英文 / 中文）**",
        "",
    ]
    for item in case["history"][-5:]:
        title = str(item["title"])
        lines.append(
            f"- `{item['news_id']}` {title}  \n"
            f"  中文：{_translate(title, translations)}"
        )
    lines.extend(["", "**正样本（英文 / 中文）**", ""])
    for item in case["positive_news"]:
        title = str(item["title"])
        nearest = item["nearest_history"]
        lines.append(
            f"- `{item['news_id']}` {title}  \n"
            f"  中文：{_translate(title, translations)}  \n"
            f"  nearest history=`{nearest['news_id']}`, "
            f"cosine={_float(item['nearest_50'])}, "
            f"novel-control rank={item['rank_among_novel_controls']}"
        )
    lines.extend(["", "**最相似 novel exposed-unclicked controls**", ""])
    for item in case["top_novel_controls"]:
        title = str(item["title"])
        lines.append(
            f"- `{item['news_id']}` {title}  \n"
            f"  中文：{_translate(title, translations)}；"
            f"cosine={_float(item['nearest_50'])}"
        )
    lines.append("")
    return lines


def _artifact_index(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    output_path: Path,
    key_results_path: Path,
    translations_path: Path | None,
) -> list[str]:
    lines = ["## 5. 数据与产物路径（Artifact Paths）", ""]
    inputs = manifest.get("inputs", {})
    if isinstance(inputs, Mapping):
        lines.append("**输入数据**")
        lines.append("")
        for label, value in sorted(inputs.items()):
            path = Path(str(value))
            if not path.is_absolute():
                path = Path.cwd() / path
            lines.append(f"- `{label}`: {_markdown_path(path)}")
        lines.append("")
    lines.append("**分析机器产物**")
    lines.append("")
    for name in (
        "manifest.json",
        "resolved_config.json",
        "summary.json",
        "analysis_rows.parquet",
        "title_embeddings.parquet",
        "category_name_embeddings.parquet",
        "similarity_metrics.csv",
        "category_name_similarity_metrics.csv",
        "taxonomy_metrics.csv",
        "slice_metrics.csv",
        "parent_transition_matrix.csv",
        "prompt_overlap.json",
        "cases.json",
    ):
        lines.append(f"- `{name}`: {_markdown_path(artifact_dir / name)}")
    lines.extend(
        [
            "",
            "**报告产物**",
            "",
            f"- Markdown report: {_markdown_path(output_path)}",
            f"- Key-results JSON: {_markdown_path(key_results_path)}",
        ]
    )
    if translations_path is not None:
        lines.append(
            f"- Verified Chinese translations: "
            f"{_markdown_path(translations_path)}"
        )
    lines.append("")
    return lines


def write_sft_training_similarity_report(
    artifact_dir: Path,
    output_path: Path,
    *,
    translations_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render the verified five-part SFT data-analysis report."""

    artifact_dir = Path(artifact_dir)
    output_path = Path(output_path)
    translations_path = (
        None if translations_path is None else Path(translations_path)
    )
    key_results_path = output_path.with_name(
        f"{output_path.stem}_key_results.json"
    )
    if (output_path.exists() or key_results_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Report output already exists: {output_path}"
        )

    manifest = _read_json(artifact_dir / "manifest.json")
    _verify_outputs(artifact_dir, manifest)
    config = _read_json(artifact_dir / "resolved_config.json")
    summary = _read_json(artifact_dir / "summary.json")
    prompt_overlap = _read_json(artifact_dir / "prompt_overlap.json")
    cases_payload = _read_json(artifact_dir / "cases.json")
    similarity = _read_csv(artifact_dir / "similarity_metrics.csv")
    category_similarity = _read_csv(
        artifact_dir / "category_name_similarity_metrics.csv"
    )
    taxonomy = _read_csv(artifact_dir / "taxonomy_metrics.csv")
    slices = _read_csv(artifact_dir / "slice_metrics.csv")
    transitions = _read_csv(
        artifact_dir / "parent_transition_matrix.csv"
    )
    translations = _translation_map(translations_path)
    cases = list(cases_payload.get("cases", []))
    train_cases = [case for case in cases if case["split"] == "train"]
    if not train_cases:
        raise ValueError("cases.json contains no Train case")
    training_case = train_cases[0]

    lines = [
        "# SFT 训练数据：标题、类别名称语义与一级类别关系分析",
        "",
        "## 1. 研究问题（Research Questions）",
        "",
        "本报告只分析当前实际使用的 balanced SFT22K Train 与 natural "
        "grouped Dev10K，分别回答：",
        "",
        "1. 正样本标题与模型可见 H50 history 的语义相似度有多高；",
        "2. 正样本的二级类别名称与一级类别名称是否在语义上接近 history "
        "中的类别名称；",
        "3. 正样本是否比同一 impression 的 exposed-unclicked 新闻更接近 "
        "history；",
        "4. exact novel subcategory 不在 history 时，target parent 是否仍已"
        "出现在 full H50、recent-5 或 last click；",
        "5. history breadth、exposure difficulty、prompt overlap 与"
        " taxonomy × semantic quadrant 如何分布。",
        "",
        "主比较使用 history-novel `-0` controls；secondary 比较使用全部 "
        "`-0` controls。主指标为 nearest-H50 cosine delta、within-"
        "impression AUC、Hit@1 与 same-parent fraction。",
        "",
        "## 2. 训练数据构造（Training Data Construction）",
        "",
        "### 2.1 数据来源与最终规模",
        "",
        f"Train 使用 balanced SFT22K；Dev 使用 natural grouped Dev10K。"
        f"每行按 SFT row 等权，history 使用最多最近 "
        f"{config['history_limit']} 条并保留顺序与重复。",
        "",
        "| split | SFT rows | positive titles | all `-0` controls | "
        "novel `-0` controls | primary rows | excluded (no novel control) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev"):
        lines.append(
            f"| {split} | {_int(summary['rows'][split])} | "
            f"{_int(summary['counts'][split]['positive_titles'])} | "
            f"{_int(summary['counts'][split]['controls'])} | "
            f"{_int(summary['counts'][split]['novel_controls'])} | "
            f"{_int(summary['primary_rows'][split])} | "
            f"{_int(summary['rows_without_novel_controls'][split])} |"
        )
    lines.extend(
        [
            "",
            "Primary control 的 subcategory 必须不在该 row 的 H50 中；"
            "没有 novel control 的 row 显式排除，不回退到其他负例。"
            "同一 raw candidate token 的重复会保留并计入 denominator。",
            "",
            *_training_example(training_case, translations),
            "## 3. 测试设计（Testing）",
            "",
            "### 3.1 Test model/location",
            "",
            "本分析不运行 SFT 生成模型，也没有远程推理 job。唯一模型是冻结的"
            f"标题 encoder `{config['embedding']['model_name']}`，revision "
            f"`{config['embedding']['revision']}`，"
            f"{config['embedding']['dimensions']} 维；每条标题 L2 normalize "
            "后只用于离线 cosine 特征。",
            "",
            "### 3.2 Test-data provenance",
            "",
            "Train row 由 `source_sample_id` 回连 processed Train；Dev row "
            "由 `impression_id + acceptable_subcategories` 回连 processed "
            "Dev。两者再回连 raw MIND impression，以 label `0` 构造唯一"
            "对照集合。Train 与 Dev 始终分别统计，不合并成总体比例。",
            "",
            "### 3.3 Test examples",
            "",
            f"代表性真实 record 为 `{training_case['row_id']}`：history "
            f"末条标题 “{training_case['history'][-1]['title']}”"
            f"（{_translate(str(training_case['history'][-1]['title']), translations)}）；"
            f"正样本 “{training_case['positive_news'][0]['title']}”"
            f"（{_translate(str(training_case['positive_news'][0]['title']), translations)}）。"
            "完整 provenance 与 control 排名见 Results case study。",
            "",
            "### 3.4 Judge procedure",
            "",
            "本分析没有使用 **LLM judge**，因此不存在 judge system prompt、"
            "rubric template 或 JSON judge schema。全部字段由确定性公式产生：",
            "",
            "- `nearest_50`: target/control 与 H50 标题 cosine 的最大值；",
            "- `top3_mean`: 最大的 `min(3, |H|)` 个 cosine 的均值；",
            "- `centroid`: 与 normalized history centroid 的 cosine；",
            "- `AUC`: 随机 positive 分数高于随机 control 的概率，tie 计 "
            "`0.5`；",
            "- `Hit@1`: positive 最大值高于 control 最大值计 `1`，tie 计 "
            "`0.5`；",
            "- `same_parent_fraction`: row 内 target parent 已在对应 history "
            "scope 出现的 target 比例。",
            "- 类别名称相似度使用同一个冻结 encoder：二级类别比较 "
            "`target_subcategories` 与 history/control subcategory 名称；"
            "一级类别比较 `target_categories` 与 history/control parent "
            "名称。history 重复点击与 control token 重复都保留。",
            "",
            f"均值与比例的 95% CI 使用 "
            f"{_int(config['bootstrap']['replicates'])} 次 row bootstrap"
            f"（seed `{config['bootstrap']['seed']}`）；taxonomy null 使用 "
            f"{_int(config['permutation']['replicates'])} 次 split 内 target-set "
            f"permutation（seed `{config['permutation']['seed']}`），保留每行"
            " target-set size 与 target marginal。",
            "",
            "## 4. 实验结果（Results）",
            "",
            "### 4.1 标题语义相似度",
            "",
            "| split | score | n | positive | novel control | delta "
            "(95% CI) | AUC (95% CI) | Hit@1 (95% CI) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "dev"):
        for score_name in (
            "nearest_50",
            "top3_mean",
            "centroid",
            "nearest_5",
            "last",
        ):
            row = _find_row(
                similarity,
                split=split,
                score_name=score_name,
                control_scope="novel",
            )
            lines.append(
                f"| {split} | {score_name} | {_int(row['rows'])} | "
                f"{_float(row['positive_mean'])} | "
                f"{_float(row['control_mean'])} | "
                f"{_float(row['delta_mean'])} {_ci(row, 'delta')} | "
                f"{_float(row['auc_mean'])} {_ci(row, 'auc')} | "
                f"{_float(row['hit_at_1_mean'])} {_ci(row, 'hit_at_1')} |"
            )
    all_scope_rows = [
        row
        for row in similarity
        if row["score_name"] == "nearest_50"
        and row["control_scope"] == "all"
    ]
    if len(all_scope_rows) == 2:
        lines.extend(
            [
                "",
                "**Nearest-H50：primary 与 secondary control 口径对照**",
                "",
                "| split | scope | n | delta | AUC | Hit@1 | "
                "positive ≤ control mean | positive ≤ control max |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in ("train", "dev"):
            for scope in ("novel", "all"):
                row = _find_row(
                    similarity,
                    split=split,
                    score_name="nearest_50",
                    control_scope=scope,
                )
                lines.append(
                    f"| {split} | {scope} | {_int(row['rows'])} | "
                    f"{_float(row['delta_mean'])} | "
                    f"{_float(row['auc_mean'])} | "
                    f"{_float(row['hit_at_1_mean'])} | "
                    f"{_float(row['positive_le_control_mean_rate'])} | "
                    f"{_float(row['positive_le_control_max_rate'])} |"
                )
        train_primary = _find_row(
            similarity,
            split="train",
            score_name="nearest_50",
            control_scope="novel",
        )
        dev_primary = _find_row(
            similarity,
            split="dev",
            score_name="nearest_50",
            control_scope="novel",
        )
        lines.extend(
            [
                "",
                "Observed result：正样本相对 novel controls 有稳定但不大的"
                f"平均优势（Train delta={_float(train_primary['delta_mean'])}，"
                f"Dev delta={_float(dev_primary['delta_mean'])}），AUC 分别为 "
                f"{_float(train_primary['auc_mean'])} / "
                f"{_float(dev_primary['auc_mean'])}。但 "
                f"{_percent(train_primary['positive_le_control_max_rate'])} "
                f"的 Train rows 与 "
                f"{_percent(dev_primary['positive_le_control_max_rate'])} "
                "的 Dev rows 中，positive row mean 仍不高于该 impression "
                "最相似的 novel control。加入所有 `-0` 后，AUC 更接近 "
                "0.5，说明“标题接近 history”只提供弱排序信号，而不是"
                "充分判据。",
            ]
        )
    lines.extend(
        [
            "",
            "### 4.2 类别名称语义相似度",
            "",
            "这里把类别名称本身送入同一 MiniLM encoder。`subcategory` "
            "衡量二级类别名称，`parent` 衡量一级类别名称；与标题指标一样，"
            "每个 SFT row 等权。",
            "",
            "| split | name level | scope | n | positive | control | "
            "delta (95% CI) | AUC | Hit@1 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "dev"):
        for name_level in ("subcategory", "parent"):
            for scope in ("novel", "all"):
                row = _find_row(
                    category_similarity,
                    split=split,
                    name_level=name_level,
                    score_name="nearest_50",
                    control_scope=scope,
                )
                lines.append(
                    f"| {split} | {name_level} | {scope} | "
                    f"{_int(row['rows'])} | "
                    f"{_float(row['positive_mean'])} | "
                    f"{_float(row['control_mean'])} | "
                    f"{_float(row['delta_mean'])} {_ci(row, 'delta')} | "
                    f"{_float(row['auc_mean'])} | "
                    f"{_float(row['hit_at_1_mean'])} |"
                )
    train_subcategory = _find_row(
        category_similarity,
        split="train",
        name_level="subcategory",
        score_name="nearest_50",
        control_scope="novel",
    )
    dev_subcategory = _find_row(
        category_similarity,
        split="dev",
        name_level="subcategory",
        score_name="nearest_50",
        control_scope="novel",
    )
    train_parent = _find_row(
        category_similarity,
        split="train",
        name_level="parent",
        score_name="nearest_50",
        control_scope="novel",
    )
    dev_parent = _find_row(
        category_similarity,
        split="dev",
        name_level="parent",
        score_name="nearest_50",
        control_scope="novel",
    )
    lines.extend(
        [
            "",
            "Observed result：二级类别名称相对 novel controls 的 nearest-H50 "
            f"delta 为 Train {_float(train_subcategory['delta_mean'])}、Dev "
            f"{_float(dev_subcategory['delta_mean'])}；一级类别名称为 Train "
            f"{_float(train_parent['delta_mean'])}、Dev "
            f"{_float(dev_parent['delta_mean'])}。一级类别名称指标会同时包含"
            " exact same-parent（同名向量 cosine=1）与不同 parent 名称之间的"
            "语义接近，因此必须与下一节的 exact same-parent rate 一起解释。"
            "二级类别的 all-control delta 显著为负，是因为 all controls "
            "允许其 subcategory 已在 H50 中出现，此时 nearest cosine 可直接"
            "达到 1；这不是与 novel-control 口径可互换的结果。当前 primary "
            "control 只保证 subcategory novel，并不保证 parent novel，所以"
            "parent-name AUC 接近 0.5 仍是合理的独立观察。",
        ]
    )
    lines.extend(
        [
            "",
            "### 4.3 一级类别关系与 permutation baseline",
            "",
            "| split | history scope | n | observed same-parent | 95% CI | "
            "permutation expected | observed − expected | lift |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "dev"):
        for scope in ("full", "recent5", "last"):
            row = _find_row(
                taxonomy,
                split=split,
                history_scope=scope,
                metric="same_parent_fraction",
            )
            lift = (
                "NA" if row["lift"] in {"", "None"} else _float(row["lift"])
            )
            lines.append(
                f"| {split} | {scope} | {_int(row['rows'])} | "
                f"{_float(row['observed'])} | "
                f"[{_float(row['observed_ci_low'])}, "
                f"{_float(row['observed_ci_high'])}] | "
                f"{_float(row['permutation_expected'])} | "
                f"{_float(row['observed_minus_expected'])} | {lift} |"
            )
    train_full = _find_row(
        taxonomy,
        split="train",
        history_scope="full",
        metric="same_parent_fraction",
    )
    dev_full = _find_row(
        taxonomy,
        split="dev",
        history_scope="full",
        metric="same_parent_fraction",
    )
    lines.extend(
        [
            "",
            "Raw same-parent rate 在 Train/Dev 都约为一半，但 permutation "
            "改变了其含义：balanced Train 高于保持 target marginal 的 null"
            f"（+{_float(train_full['observed_minus_expected'])}），natural "
            f"Dev 反而低于 null（{_float(dev_full['observed_minus_expected'])}）。"
            "因此“约 50% 与历史同一级类别”主要受 history coverage 与 target "
            "分布共同影响，不能单独作为强兴趣延展证据。recent-5 约 0.277，"
            "last click 仅约 0.083，说明一级类别关系更多来自较宽的累计"
            "history，而不是紧邻的最后一次点击。",
        ]
    )
    lines.extend(
        [
            "",
            "### 4.4 Prompt ambiguity 与 Train/Dev overlap",
            "",
            "| task | Train duplicate excess | Train conflicting prompts | "
            "Dev exact-prompt overlap rows | any target seen | all targets seen |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in ("behavior", "interest"):
        value = prompt_overlap["tasks"][task]
        lines.append(
            f"| {task} | {_int(value['train_duplicate_row_excess'])} | "
            f"{_int(value['train_conflicting_prompt_count'])} | "
            f"{_int(value['dev_exact_prompt_overlap_rows'])} | "
            f"{_int(value['dev_any_target_seen_for_prompt_rows'])} | "
            f"{_int(value['dev_all_targets_seen_for_prompt_rows'])} |"
        )
    user_overlap = prompt_overlap["user_overlap"]
    behavior_overlap = prompt_overlap["tasks"]["behavior"]
    interest_overlap = prompt_overlap["tasks"]["interest"]
    lines.extend(
        [
            "",
            f"Train/Dev 共有 {_int(user_overlap['users'])} 个 user，覆盖 "
            f"{_int(user_overlap['dev_rows'])} 个 Dev rows"
            f"（{_percent(user_overlap['dev_rows'] / summary['rows']['dev'])}）。"
            " exact prompt overlap 并不等于答案泄漏：Behavior 中仅 "
            f"{_int(behavior_overlap['dev_any_target_seen_for_prompt_rows'])}"
            f"/{_int(behavior_overlap['dev_exact_prompt_overlap_rows'])} 个"
            " overlap rows 至少有一个 Dev target 已在 Train 同 prompt "
            "出现；Interest 为 "
            f"{_int(interest_overlap['dev_any_target_seen_for_prompt_rows'])}"
            f"/{_int(interest_overlap['dev_exact_prompt_overlap_rows'])}。"
            "另一方面，Train 内存在数百个 exact-prompt conflicts，说明相同"
            "模型输入仍可能对应多个 future-interest labels；这是监督信号"
            "本身的 one-to-many ambiguity，而不是简单的 user overlap。",
            "",
            "### 4.5 Taxonomy × semantic evidence",
            "",
            "| split | quadrant | rows | primary delta | primary AUC | "
            "primary Hit@1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    quadrant_rows = [
        row
        for row in slices
        if row["slice_type"] == "taxonomy_semantic_quadrant"
    ]
    for row in quadrant_rows:
        lines.append(
            f"| {row['split']} | {row['slice_value']} | "
            f"{_int(row['rows'])} | {_float(row['novel_control_delta'])} | "
            f"{_float(row['novel_control_auc'])} | "
            f"{_float(row['novel_control_hit_at_1'])} |"
        )
    for split in ("train", "dev"):
        split_quadrants = [
            row for row in quadrant_rows if row["split"] == split
        ]
        if not split_quadrants:
            continue
        total = sum(int(row["rows"]) for row in split_quadrants)
        higher = sum(
            int(row["rows"])
            for row in split_quadrants
            if row["slice_value"].endswith("__higher_than_control")
        )
        lines.append(
            f"{split}: {_int(higher)}/{_int(total)} "
            f"({_percent(higher / total)}) primary rows 的 positive similarity "
            "高于 novel-control mean；其余 rows 即使 taxonomy 同 parent，"
            "标题证据也可能较弱。"
        )

    history_slice_rows = [
        row for row in slices if row["slice_type"] == "history_length"
    ]
    control_slice_rows = [
        row for row in slices if row["slice_type"] == "control_count"
    ]
    if history_slice_rows and control_slice_rows:
        lines.extend(
            [
                "",
                "### 4.6 History breadth 与 exposure difficulty",
                "",
                "**History length**",
                "",
                "| split | H length | total n | primary n | "
                "same-parent (all rows) | primary delta | primary AUC | "
                "primary Hit@1 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in history_slice_rows:
            lines.append(
                f"| {row['split']} | {row['slice_value']} | "
                f"{_int(row['rows'])} | "
                f"{_int(row['primary_rows'])} | "
                f"{_float(row['same_parent_fraction'])} | "
                f"{_float(row['novel_control_delta'])} | "
                f"{_float(row['novel_control_auc'])} | "
                f"{_float(row['novel_control_hit_at_1'])} |"
            )
        lines.extend(
            [
                "",
                "Same-parent rate 随 history 变长明显上升，但 semantic delta "
                "在中等长度后并不继续单调增加；这支持把 taxonomy overlap "
                "与真正的标题证据分开衡量。",
                "",
                "**Exposure candidate count**",
                "",
                "| split | all controls | total n | primary n | "
                "primary delta | primary AUC | primary Hit@1 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in control_slice_rows:
            lines.append(
                f"| {row['split']} | {row['slice_value']} | "
                f"{_int(row['rows'])} | "
                f"{_int(row['primary_rows'])} | "
                f"{_float(row['novel_control_delta'])} | "
                f"{_float(row['novel_control_auc'])} | "
                f"{_float(row['novel_control_hit_at_1'])} |"
            )
        lines.extend(
            [
                "",
                "Hit@1 对候选数极其敏感：候选越多，positive 成为唯一最相似"
                "标题的概率快速下降；AUC 与 mean delta 相对稳定，因此不同"
                " exposure size 之间不应只比较 Hit@1。",
            ]
        )

    all_target_slice_rows = [
        row
        for row in slices
        if row["slice_type"] == "target_subcategory"
    ]
    target_slice_rows = [
        row
        for row in all_target_slice_rows
        if int(row["rows"]) >= 50
        and row["novel_control_delta"] not in {"", None}
    ]
    if target_slice_rows:
        target_counts = {
            split: len(
                {
                    row["slice_value"]
                    for row in all_target_slice_rows
                    if row["split"] == split
                }
            )
            for split in ("train", "dev")
        }
        lines.extend(
            [
                "",
                "### 4.7 Target heterogeneity",
                "",
                "以下仅展示 support ≥ 50 的 target subcategory 两端样本；"
                f"完整 observed slices（Train {target_counts['train']} 类，"
                f"Dev {target_counts['dev']} 类）见 `slice_metrics.csv`。",
                "",
                "| split | end | target subcategory | rows | same-parent | "
                "primary delta | primary AUC |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for split in ("train", "dev"):
            split_targets = sorted(
                (
                    row
                    for row in target_slice_rows
                    if row["split"] == split
                ),
                key=lambda row: float(row["novel_control_delta"]),
            )
            selected_targets = [
                ("lowest", row) for row in split_targets[:3]
            ] + [
                ("highest", row) for row in split_targets[-3:][::-1]
            ]
            for end, row in selected_targets:
                lines.append(
                    f"| {split} | {end} | {row['slice_value']} | "
                    f"{_int(row['rows'])} | "
                    f"{_float(row['same_parent_fraction'])} | "
                    f"{_float(row['novel_control_delta'])} | "
                    f"{_float(row['novel_control_auc'])} |"
                )
        lines.extend(
            [
                "",
                "不同 target 的可预测性差异很大：例如 Dev 的 `Air & Rail "
                "Travel` 与 `Archaeology & Human Origins` 呈负 delta，"
                "而 `Journalism & News Media`、`Public Services & Social "
                "Welfare` 呈较强正 delta。该差异是按 target 切片的观察值，"
                "不代表类别本身造成模型难度。",
            ]
        )

    unique_transition_pairs = {
        (row["last_history_parent"], row["target_parent"])
        for row in transitions
    }
    lines.extend(
        [
            "",
            "### 4.8 Parent transition 与真实 case studies",
            "",
            "Parent transition matrix 共包含 "
            f"{_int(len(transitions))} 个 split-specific 非零 "
            "`(split, last parent, target parent)` cells，对应合并 split "
            f"标签后的 {_int(len(unique_transition_pairs))} 个不同 "
            "`(last parent, target parent)` pairs；完整计数见 CSV。",
            "",
        ]
    )
    selected_cases: list[Mapping[str, Any]] = []
    seen_quadrants: set[str] = set()
    for case in sorted(
        cases,
        key=lambda item: (
            item["quadrant"],
            item["split"] != "dev",
            item["row_id"],
        ),
    ):
        if case["quadrant"] in seen_quadrants:
            continue
        seen_quadrants.add(case["quadrant"])
        selected_cases.append(case)
    for case in selected_cases:
        lines.extend(_case_block(case, translations))

    lines.extend(
        _artifact_index(
            artifact_dir,
            manifest,
            output_path,
            key_results_path,
            translations_path,
        )
    )
    report = "\n".join(lines).rstrip() + "\n"
    key_results = {
        "artifact_type": "sft_training_similarity_key_results",
        "artifact_version": 1,
        "source_manifest": str(
            _absolute_without_resolving(artifact_dir / "manifest.json")
        ),
        "report_path": str(_absolute_without_resolving(output_path)),
        "rows": summary["rows"],
        "primary_rows": summary["primary_rows"],
        "nearest_50_novel": {
            split: dict(
                _find_row(
                    similarity,
                    split=split,
                    score_name="nearest_50",
                    control_scope="novel",
                )
            )
            for split in ("train", "dev")
        },
        "category_name_nearest_50_novel": {
            name_level: {
                split: dict(
                    _find_row(
                        category_similarity,
                        split=split,
                        name_level=name_level,
                        score_name="nearest_50",
                        control_scope="novel",
                    )
                )
                for split in ("train", "dev")
            }
            for name_level in ("subcategory", "parent")
        },
        "same_parent_full": {
            split: dict(
                _find_row(
                    taxonomy,
                    split=split,
                    history_scope="full",
                    metric="same_parent_fraction",
                )
            )
            for split in ("train", "dev")
        },
        "case_ids": [case["case_id"] for case in selected_cases],
    }
    _atomic_write_text(output_path, report)
    _atomic_write_json(key_results_path, key_results)
    return {
        "report_path": str(output_path),
        "key_results_path": str(key_results_path),
        "cases": len(selected_cases),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the SFT training-data similarity report."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/sft-training-data-analysis/v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/2026-08-13/"
            "sft_training_data_similarity_analysis.md"
        ),
    )
    parser.add_argument(
        "--translations",
        type=Path,
        default=Path(
            "configs/analysis/"
            "sft_training_similarity_case_translations_v1.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = write_sft_training_similarity_report(
        args.artifact_dir,
        args.output,
        translations_path=args.translations,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
