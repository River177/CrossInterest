# 静态用户隔离 K=10 Candidate Select（GPU-only interim）

> 状态：`exploratory_dev_gpu_only_interim`。本报告只汇总 6 个已经完整生成并核验的 GPU 条件；12 个 API full 条件因 Gemini 调用触发 Copilot 限流而停止，其 partial shards 不进入任何正式指标。该 Dev 已用于此前分析，不能作为全新的 final test。

## 1. 研究问题（Research Questions）

本实验研究静态、Train/Dev 用户隔离的新兴趣预测：给定用户完整原始 History 和 10 个候选兴趣，zero-shot 模型必须选择 1 个下一步新兴趣。没有进行 SFT、GRPO、adapter 训练或候选 ranking。

具体问题为：

1. `Behavior`（完整历史标题）与 `Interest`（完整历史兴趣序列）哪一种输入更适合 zero-shot closed-set 选择？
2. 模型是否超过只使用 Static Train 的 Global Popularity、Last-interest Conditional 和 Semantic History baseline？
3. 将均匀负例替换为 Train 频率匹配负例后，Qwen3.5 的结果是否稳定？
4. 结果在 answer position、History 长度和 Train 兴趣频率 head/torso/tail 上如何变化？

主指标是 **Interest Macro Accuracy**，不是 User-Macro：

\[
\mathrm{InterestMacroAccuracy}
=\frac{1}{111}\sum_{y=1}^{111}
\frac{\#\{i:\hat y_i=y_i=y\}}{\#\{i:y_i=y\}}.
\]

即先对每个 Dev 正例兴趣分别计算 Accuracy，再让 111 个兴趣等权平均。次指标包括 Micro Accuracy、strict Accuracy、Candidate-valid rate，以及 2,000 次 user-cluster paired bootstrap 95% CI。

## 2. 训练数据构造（Training Data Construction）

### 2.1 本实验没有训练 LLM

Qwen3.5-9B 与 Llama-3.1-8B-Instruct 都是未接 adapter 的 base instruct checkpoint。本实验没有“训练 prompt/response 示例”。Static Train 只用于冻结候选兴趣空间、估计三个统计 baseline，以及在 Train 内拟合和选择 Semantic baseline；Dev label 未用于拟合或选参。

### 2.2 Static Train/Dev

源数据为 `data/raw/MINDlarge_reclassified`。静态数据规则版本是 `static-user-disjoint-single-train-target-v1`：Train 每个 eligible 用户通过 seed `20260806` 的 SHA-256 priority 只保留一个 target；Dev 删除所有原始 Train 用户，并保留剩余用户的全部 eligible target。History 完整取 behavior 行原始 History，不追加 impression 点击，也不截断。

| split/用途 | rows | users | impressions | 正例兴趣数 |
|---|---:|---:|---:|---:|
| Static Train | 587,187 | 587,187 | — | 124 个在 Train 出现的兴趣 |
| Static Dev / Candidate Select | 37,817 | 27,125 | 31,451 | 111 |

### 2.3 K=10 候选构造

每条 Static Dev target row 生成一个选择样本，共 37,817 条，不对 `user-interest` 去重。候选含 1 个正确兴趣和 9 个负例；负例排除同一 impression 的全部正确兴趣、原始 History 中出现的全部兴趣及当前正例。`Behavior` 与 `Interest` 共享完全相同的 `selection_id`、候选、答案和正确位置。

| 负例策略 | 规则 | seed | rows | 正确位置分布 |
|---|---|---:|---:|---|
| Uniform | SHA-256 priority，无放回 | 20260806 | 37,817 | 每个位置 3,781–3,782 |
| Train-frequency-matched | 按 Static Train 兴趣频率做确定性 exponential-race，无放回 | 20260806 | 37,817 | 每个位置 3,781–3,782 |

候选构造规则版本为 `static-user-disjoint-candidate-selection-k10-v1.2`。History 长度最小 1、中位数 7、P90 19、最大 356。

### 2.4 Baseline 冻结规则

本节记录 GPU-only interim 汇总时使用的原始 baseline。2026-08-16 的扩展超参数搜索作为独立 follow-up 记录在第 6 节，不回写或替换这里的原始结果。

| baseline | 仅使用的信号 | 冻结方式 |
|---|---|---|
| Uniform Random | 候选位置 | 理论 10%；用 hash seeds `20260815/16/17` 验证 |
| Global Popularity | Static Train 的用户计数 | 在 10 个候选中选全局计数最高者 |
| Last-interest Conditional | `count(target \mid History最后兴趣)` | 无支持时回退 Global |
| Semantic History | 标题 embedding + Ridge residual | `all-MiniLM-L6-v2`，只在 Static Train 内 fit/tune/holdout；选中 ridge alpha `1.0`、semantic weight `48.0` |

Semantic baseline 的 Train 内用户划分为 469,643 fit、58,518 tune、59,026 holdout。Last-interest 与 Semantic baseline 还分别使用 seeds `20260815/16/17` 构造 matched shuffled-History 控制；只替换 History，用户、target、候选和正确位置保持不变，且不允许映射到自身。

## 3. 测试设计（Testing）

### 3.1 测试模型与运行位置

| 模型 | full 条件 | 输入 | 负例 | 推理配置 |
|---|---:|---|---|---|
| `Qwen/Qwen3.5-9B` | 4 | Behavior / Interest | Uniform / Frequency-matched | `do_sample=false`，thinking=false，`max_new_tokens=64`，batch=4 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 2 | Behavior / Interest | Uniform | `do_sample=false`，`max_new_tokens=64`，batch=4 |

六个 full 条件均在 AMLT Singularity `Feeds`、`80G1-A100` 上远程运行，seed 为 `20260815`，无 adapter，W&B disabled。配置只记录 Hugging Face model name，没有另行记录 model revision commit，因此这里不声称 checkpoint revision 已 pin。远程实验 `static-candidate-full-v1-0815b` 的六个 job 均为 `pass`：

- `qwen35-behavior-uniform`
- `qwen35-interest-uniform`
- `qwen35-behavior-frequency-matched`
- `qwen35-interest-frequency-matched`
- `llama31-behavior-uniform`
- `llama31-interest-uniform`

### 3.2 测试数据来源与 pilot gate

Full 测试使用第 2 节所述 Static Dev，37,817 rows、27,125 users、31,451 impressions、111 个正例兴趣。先从同一数据确定性选择覆盖全部 111 类的 1,000-row pilot；canonical manifest SHA-256 为 `4d1da9fb51fd0bbd934ecb3c92c401b0f840fc9425ee67abc2579e000a3d3579`。18/18 pilot 条件均通过成功率、Candidate-valid、格式、截断/上下文、`reasoning=none` 和 Behavior/Interest 身份 gate，之后才启动 full。

六个 GPU full 结果下载后逐条件核验：37,817 rows、27,125 users、111 interests；`selection_id` 唯一且与输入身份集合一致；answer 与 source row 一致；manifest/data SHA-256 与 experiment identity 一致。非法输出保留在分母。

### 3.3 真实测试样例

样例 `selection_id=static-dev:dev:100011:7`，用户 `U743273`，History 长度 9，正确答案是 `Television & Streaming`。

Interest 输入的英文原文：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Civil Rights & Social Justice
2. Film
3. Celebrity & Creator Culture
4. Historical Events & Eras
5. Celebrity & Creator Culture
6. Labor & Employment
7. Charity & Volunteering
8. Pets & Animal Companions
9. Theater & Performing Arts

Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should not appear in the history sequence above.

Candidates:
1. Ice Hockey
2. Cars & Driving
3. Population & Demographic Change
4. Architecture & Residential Design
5. Elections & Political Campaigns
6. Travel Planning & Deals
7. Destinations & Trip Ideas
8. Home Appliances & Technology
9. Astrology & Divination
10. Television & Streaming

Allowed responses (copy exactly one line):
New topic: Ice Hockey
New topic: Cars & Driving
New topic: Population & Demographic Change
New topic: Architecture & Residential Design
New topic: Elections & Political Campaigns
New topic: Travel Planning & Deals
New topic: Destinations & Trip Ideas
New topic: Home Appliances & Technology
New topic: Astrology & Divination
New topic: Television & Streaming

This is a closed-set choice. Copy exactly one complete line from the Allowed responses list above, even if none is an ideal match. Never invent, rename, or substitute another category. Do not include any explanation or other text.
```

中文翻译：

```text
下面按时间顺序列出我最近浏览的新闻对应的主题类别：

1. 公民权利与社会正义
2. 电影
3. 名人与创作者文化
4. 历史事件与时代
5. 名人与创作者文化
6. 劳工与就业
7. 慈善与志愿服务
8. 宠物与动物伴侣
9. 戏剧与表演艺术

请从下列候选中选择我下一步最可能感兴趣的一个主题类别。所选类别不应出现在上面的历史序列中。

候选：冰球；汽车与驾驶；人口与人口结构变化；建筑与住宅设计；选举与政治竞选；旅行规划与优惠；目的地与旅行灵感；家用电器与技术；占星与占卜；电视与流媒体。

允许的回答：必须逐字复制且只复制一行 `New topic: <候选类别>`。
这是封闭集合选择。即使没有理想选项，也必须精确复制一条允许的回答；不得创造、改名或替换类别，不得添加解释或其他文字。
```

Uniform 条件的真实输出：

| 条件 | 原始输出 | 中文 | 是否正确 |
|---|---|---|---:|
| Qwen3.5 Behavior | `New topic: Television & Streaming` | 新主题：电视与流媒体 | 是 |
| Qwen3.5 Interest | `New topic: Television & Streaming` | 新主题：电视与流媒体 | 是 |
| Llama Behavior | `New topic: Ice Hockey` | 新主题：冰球 | 否 |
| Llama Interest | `New topic: Television & Streaming` | 新主题：电视与流媒体 | 是 |

Frequency-matched 候选不同，但正确答案仍在第 10 位：Qwen3.5 Behavior 输出 `New topic: Fashion & Personal Style`（新主题：时尚与个人风格，错误）；Qwen3.5 Interest 输出 `New topic: Television & Streaming`（正确）。该单例仅展示可审计的输入输出，不代表整体因果解释。

### 3.4 解析、指标和置信区间

本实验没有使用 LLM judge，因此不存在 judge prompt、judge schema 或 judge-error rate。输出由确定性 parser 评估：

- 合法格式为 `New topic: <exact candidate category>`，且只能有这一行。
- `Candidate-valid` 表示解析出的兴趣属于当前 10 个候选。
- `strict Accuracy` 同时要求 exact format 和正确类别；不是单独的“格式合规率”。
- 非法、空输出、多候选或额外文本一律记错，不从 37,817 或每兴趣分母删除。
- `prefix_compliance_rate` 检查 `New topic:` 前缀；六个 full 条件均为 100%。

95% CI 使用 2,000 次 bootstrap，以用户为聚类重采样单位；每次在重采样数据上重新计算 Interest Macro Accuracy。配对比较在同一批 bootstrap replicate 上相减。

## 4. 实验结果（Results）

### 4.1 完成边界

| 阶段 | 条件数 | 每条件 rows | 状态 | 是否进入 full 指标 |
|---|---:|---:|---|---:|
| Pilot v1.2 | 18 | 1,000 | 18/18 passed | 否，仅作启动 gate |
| GPU full | 6 | 37,817 | 6/6 passed 并完成身份/哈希/格式核验 | 是 |
| API full | 12 | 256–3,968 partial | 因 Copilot rate limiting 停止 | 否 |

API partial rows 分别为：Gemini Behavior 3,840、Gemini Interest 3,968；GPT-5.6 medium Behavior/Interest Uniform 256/320、Frequency-matched 256/320；GPT-5.6 none Behavior/Interest 448/448；GPT-5 medium Behavior/Interest 320/512；GPT-5 none Behavior/Interest 2,432/2,432。这些数字只说明已生成量，不能用于模型指标。

### 4.2 Full 主结果

| 候选策略 | 条件 | Micro（k/n） | Interest Macro (95% CI) | Candidate-valid |
|---|---|---:|---:|---:|
| Uniform | Global Popularity | 14,905/37,817 (39.41%) | 13.22% [13.04, 14.01] | 100.000% |
| Uniform | Last-interest | 15,376/37,817 (40.66%) | 14.20% [13.91, 15.13] | 100.000% |
| Uniform | Semantic History | 15,892/37,817 (42.02%) | **15.43%** [15.12, 16.43] | 100.000% |
| Uniform | Qwen3.5 Behavior | 6,831/37,817 (18.06%) | 14.50% [13.19, 15.79] | 99.997%（1 invalid） |
| Uniform | Qwen3.5 Interest | 4,975/37,817 (13.16%) | 12.71% [11.97, 14.11] | 99.974%（10 invalid） |
| Uniform | Llama Behavior | 2,882/37,817 (7.62%) | 10.39% [8.68, 11.84] | 100.000% |
| Uniform | Llama Interest | 5,965/37,817 (15.77%) | 13.70% [12.04, 15.04] | 99.889%（42 invalid） |
| Frequency-matched | Global Popularity | 3,520/37,817 (9.31%) | 2.79% [2.74, 2.99] | 100.000% |
| Frequency-matched | Last-interest | 4,914/37,817 (12.99%) | 3.24% [2.97, 3.64] | 100.000% |
| Frequency-matched | Semantic History | 6,615/37,817 (17.49%) | 4.52% [4.24, 5.03] | 100.000% |
| Frequency-matched | Qwen3.5 Behavior | 5,747/37,817 (15.20%) | 11.56% [10.75, 12.88] | 99.995%（2 invalid） |
| Frequency-matched | Qwen3.5 Interest | 5,350/37,817 (14.15%) | **14.39%** [12.78, 15.80] | 99.981%（7 invalid） |

Uniform 下没有一个 GPU LLM 明确超过 Semantic History baseline。Qwen3.5 Behavior 的点估计最接近 Semantic（−0.92 pp），但 paired CI 跨 0。Micro 与 Interest Macro 的排序明显不同，例如 Uniform Global 的 Micro 39.41% 但 Macro 只有 13.22%，说明 head 兴趣支配 Micro，不能用 Micro 替代主指标。

### 4.3 随机与 shuffled-History 控制

三个固定 hash seed 的随机 Micro 分别为 10.12%、10.21%、10.09%；Interest Macro 分别为 8.96%、11.41%、9.79%，围绕理论 10% 波动。

| 候选策略 | baseline | 真实 History Macro | 三个 shuffled-History Macro |
|---|---|---:|---:|
| Uniform | Last-interest | 14.20% | 12.99%, 13.18%, 12.98% |
| Uniform | Semantic | 15.43% | 13.20%, 13.13%, 13.25% |
| Frequency-matched | Last-interest | 3.24% | 3.17%, 3.11%, 2.94% |
| Frequency-matched | Semantic | 4.52% | 3.50%, 3.86%, 3.66% |

真实 History 在两个 baseline 上都高于 matched shuffled History，但增益相对有限；这支持“存在弱的个性化残差信号”，不支持把全部成绩解释为强个性化能力。

### 4.4 与 baseline 的配对差值

表中单位为百分点，括号为 user-cluster paired bootstrap 95% CI。

| 模型条件 | Δ Global | Δ Last-interest | Δ Semantic |
|---|---:|---:|---:|
| Uniform Qwen Behavior | +1.29 [−0.42, +2.36] | +0.30 [−1.49, +1.38] | −0.92 [−2.78, +0.19] |
| Uniform Qwen Interest | −0.50 [−1.56, +0.55] | −1.49 [−2.59, −0.42] | −2.72 [−3.87, −1.59] |
| Uniform Llama Behavior | −2.82 [−4.95, −1.64] | −3.81 [−5.96, −2.57] | −5.04 [−7.23, −3.83] |
| Uniform Llama Interest | +0.48 [−1.60, +1.66] | −0.51 [−2.61, +0.69] | −1.73 [−3.93, −0.54] |
| Frequency Qwen Behavior | +8.77 [+7.90, +9.98] | +8.32 [+7.41, +9.62] | +7.04 [+6.08, +8.28] |
| Frequency Qwen Interest | +11.59 [+9.91, +12.94] | +11.15 [+9.42, +12.54] | +9.86 [+8.15, +11.20] |

Frequency-matched 候选会把高频兴趣作为更强负例，因此 Global/Conditional/Semantic baseline 的 Macro 点估计显著下降；上表的跨策略差值不能理解成模型绝对能力提升。

### 4.5 Behavior/Interest 与负例策略的配对差值

| 配对比较 | 点估计差值 | 95% CI |
|---|---:|---:|
| Llama Interest − Behavior（Uniform） | +3.31 pp | [+1.28, +5.44] |
| Qwen Interest − Behavior（Uniform） | −1.79 pp | [−3.15, +0.17] |
| Qwen Interest − Behavior（Frequency） | +2.83 pp | [+1.01, +3.90] |
| Qwen Behavior Frequency − Uniform | −2.94 pp | [−4.26, −1.06] |
| Qwen Interest Frequency − Uniform | +1.67 pp | [−0.52, +3.03] |

输入表示的效果并不一致：Llama 在 Uniform 下 Interest 显著更好；Qwen 在 Uniform 下 Behavior 点估计更好但 CI 跨 0，在 Frequency-matched 下则是 Interest 显著更好。结果不支持一个与模型和候选构造无关的“标题一定优于兴趣序列”或相反结论。

### 4.6 分层结果

Head/torso/tail 按 Static Train 的 124 个已出现兴趣按频率排序：前 20% 为 head，接下来的 40% 为 torso，其余为 tail。表内均为各 slice 自身支持兴趣上的 Interest Macro Accuracy。

| 条件 | Head | Torso | Tail | History-bin Macro 最小–最大 | Position-bin Macro 最小–最大 |
|---|---:|---:|---:|---|---|
| Uniform Qwen Behavior | 18.24% | 13.78% | 12.91% | `1-5` 12.84% – `36+` 16.43% | pos 8 10.53% – pos 2 23.17% |
| Uniform Qwen Interest | 12.86% | 12.20% | 13.32% | `21-35` 12.71% – `36+` 17.12% | pos 9 8.54% – pos 2 23.13% |
| Uniform Llama Behavior | 7.35% | 10.47% | 12.39% | `11-20` 9.11% – `6-10` 10.51% | pos 6 1.28% – pos 1 41.18% |
| Uniform Llama Interest | 13.94% | 13.80% | 13.39% | `36+` 12.10% – `1-5` 14.09% | pos 7 6.17% – pos 2 32.92% |
| Frequency Qwen Behavior | 12.96% | 12.63% | 9.10% | `6-10` 11.74% – `36+` 14.85% | pos 8 8.14% – pos 2 19.25% |
| Frequency Qwen Interest | 13.51% | 13.58% | 16.11% | `1-5` 13.26% – `36+` 15.56% | pos 8 7.98% – pos 2 24.74% |

Answer-position slice 每个位置只覆盖 92–98 个兴趣，而不是完整 111 个，因此这些极值主要用于发现位置敏感性，不能直接当作全局位置效应。逐兴趣 support/Accuracy、所有 History/position slice 的完整数值保存在 Parquet 产物中。

## 5. 数据与产物路径（Artifact Paths）

### 数据与构建元数据

- 原始数据：`data/raw/MINDlarge_reclassified/`
- Static 数据目录：`data/processed/new_interest_static_user_disjoint_v1/`
- Static Train/Dev：`data/processed/new_interest_static_user_disjoint_v1/train.parquet`、`data/processed/new_interest_static_user_disjoint_v1/dev.parquet`
- Train 选择清单：`data/processed/new_interest_static_user_disjoint_v1/train_selection_manifest.parquet`
- Static 统计与元数据：`data/processed/new_interest_static_user_disjoint_v1/statistics.json`、`data/processed/new_interest_static_user_disjoint_v1/metadata.json`
- Candidate 数据目录：`data/training/candidate_selection_static_user_disjoint_k10_v1/`
- Uniform Behavior/Interest：`data/training/candidate_selection_static_user_disjoint_k10_v1/uniform/behavior_only/dev_expanded.parquet`、`data/training/candidate_selection_static_user_disjoint_k10_v1/uniform/interest_only/dev_expanded.parquet`
- Frequency Behavior/Interest：`data/training/candidate_selection_static_user_disjoint_k10_v1/train_frequency_matched/behavior_only/dev_expanded.parquet`、`data/training/candidate_selection_static_user_disjoint_k10_v1/train_frequency_matched/interest_only/dev_expanded.parquet`
- Candidate manifests：`data/training/candidate_selection_static_user_disjoint_k10_v1/uniform/candidate_manifest.parquet`、`data/training/candidate_selection_static_user_disjoint_k10_v1/train_frequency_matched/candidate_manifest.parquet`
- Candidate 统计与元数据：`data/training/candidate_selection_static_user_disjoint_k10_v1/statistics.json`、`data/training/candidate_selection_static_user_disjoint_k10_v1/metadata.json`

### 配置、pilot 与 baseline

- Canonical 18-condition manifest：`configs/candidate/static_user_disjoint_v1.json`
- GPU-only interim manifest：`configs/candidate/static_user_disjoint_v1_gpu_only_interim.json`
- Pilot gate：`configs/candidate/static_user_disjoint_v1_pilot_gate.json`
- AMLT full config：`configs/amlt/static_candidate_full.yaml`
- Pilot 输出：`artifacts/candidate-selection-static-user-disjoint-k10-v1/models/pilot_v1_2/`
- Baseline 根目录：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines/`
- Baseline summary/manifest：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines/summary.json`、`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines/manifest.json`
- Semantic 模型与选参：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines/semantic_model.npz`、`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines/semantic_tuning.json`
- Shuffled-History 映射：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines/shuffle_manifest.parquet`

### Full 模型输出与 partial API 输出

- 完整 GPU 结果根目录：`artifacts/candidate-selection-static-user-disjoint-k10-v1/models/full/`
- 完整条件目录：`qwen35_behavior_uniform/`、`qwen35_interest_uniform/`、`qwen35_behavior_frequency_matched/`、`qwen35_interest_frequency_matched/`、`llama31_behavior_uniform/`、`llama31_interest_uniform/`
- 每个完整目录包含 `predictions.parquet`、`metrics.json`、`resolved_config.json`、`experiment_identity.json` 和 resumable `shards/`。
- 被排除的 API partial 条件也位于同一 full 根目录，仅保留 `run.log` 与 partial `shards/`；没有把它们聚合为正式 `predictions.parquet` 或指标。

### 汇总产物与远程记录

- GPU-only interim 汇总：`artifacts/candidate-selection-static-user-disjoint-k10-v1/analysis_gpu_only_interim/`
- 总表：`condition_metrics.parquet`
- 逐兴趣结果：`per_interest_metrics.parquet`
- 分层结果：`slice_metrics.parquet`
- Bootstrap CI：`bootstrap_intervals.json`
- 配对比较：`paired_comparisons.json`
- 汇总状态：`summary.json`
- Source/output SHA-256：`manifest.json`
- AMLT experiment：`static-candidate-full-v1-0815b`，六个 job 名见 3.1；仅下载约 3.2–3.5 MB/condition 的 evaluation 输出，没有下载或训练 adapter/checkpoint。
- 本报告：`docs/experiments/2026-08-15/static_user_disjoint_candidate_selection.md`

## 6. 2026-08-16 Semantic 超参数扩展（Follow-up）

这是在看到原始搜索最优点落于网格边界后进行的探索性 follow-up。搜索仍严格只使用 Static Train：469,643 个用户用于拟合，58,518 个用户用于按 Interest Macro Accuracy 选参，59,026 个 selection-excluded holdout 用户只用于选参后的检查。Dev label 未参与拟合或参数选择；但由于该 Dev 已被前序实验反复分析，下面的 Dev 数值仍只能解释为 exploratory Dev，不能作为新的无偏 final-test 结果。

扩展网格预先固定为 9 个 Ridge penalty 与 12 个 semantic weight，共 108 个组合：

- `ridge_alpha ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`
- `semantic_weight ∈ {0, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256}`

仍以 tune Interest Macro 为首要选择指标，Micro 只作 tie-break。最终选中 `ridge_alpha=3.0`、`semantic_weight=256.0`。

| 数据/候选 | 原组合 `α=1, λ=48` | 扩展组合 `α=3, λ=256` | Macro 差值 | Micro 变化 |
|---|---:|---:|---:|---:|
| Train internal tune | 15.48% | **18.60%** | +3.12 pp | 43.64% → 41.07% |
| Train selection-excluded holdout | 15.20% | **18.21%** | +3.01 pp | 43.61% → 40.83% |
| Uniform exploratory Dev | 15.43% | **17.31%** | +1.89 pp | 42.02% → 38.68% |
| Frequency-matched exploratory Dev | 4.52% | **6.09%** | +1.56 pp | 17.49% → 17.94% |

Uniform Dev 上，扩展 Semantic 相对 Global 的 Macro 差值为 +4.10 pp，user-cluster paired bootstrap 95% CI 为 `[+3.20, +5.16]`；相对 Last-interest 为 +3.11 pp，95% CI 为 `[+2.19, +4.18]`。真实 History 的 Macro 为 17.31%，三个 matched shuffled-History 分别为 13.17%、12.75%、12.52%，三个真实减 shuffled 的 95% CI 均不跨 0。

Frequency-matched Dev 上，扩展 Semantic 相对 Global/Last-interest 的 Macro 差值分别为 +3.29 pp `[+2.92, +3.85]` 和 +2.85 pp `[+2.33, +3.49]`。真实 History 为 6.09%，三个 shuffled-History 为 4.28%、4.54%、4.34%。

这次优化直接针对 Interest Macro，因此 Macro 提高而 Uniform Micro 下降是预期的 head/tail 权衡，不能称为所有指标都改善。由于 `semantic_weight=256` 仍位于本轮网格上边界，随后按用户要求继续进行了第 7 节的 Train-only 宽范围搜索；17.31% 仅是 v1 网格结果，不再是最终调参结果。

扩展实验产物位于：

- 根目录：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines_semantic_expanded_v1/`
- 网格、选中参数与 internal holdout：`semantic_tuning.json`
- 冻结模型：`semantic_model.npz`
- 两种候选策略的完整指标与预测：`uniform/`、`train_frequency_matched/`
- 配置与 SHA-256：`resolved_config.json`、`manifest.json`

## 7. 2026-08-16 Semantic 宽范围继续搜索

第二轮先固定 `ridge_alpha ∈ {1, 3, 10}`、`semantic_weight ∈ {256, 384, 512, 768, 1024, 1536, 2048}`。Train tune 与 selection-excluded holdout 在 `λ=2048` 仍继续提高，因此第三轮使用几何网格一直扩展到 `λ=1,048,576`，以覆盖接近纯 semantic residual、全局先验权重趋近于零的极限。每轮参数选择仍只读取 Static Train tune；Dev label 不参与参数选择。

在每个 λ 下选择 Train tune Macro 最高的 alpha 后，宽范围曲线为：

| λ | 最佳 α | Train tune Macro | Train tune Micro |
|---:|---:|---:|---:|
| 2,048 | 10 | 20.80% | 38.48% |
| 4,096 | 10 | 20.90% | 38.11% |
| 6,144 | 10 | 21.01% | 38.01% |
| 8,192 | 10 | 21.03% | 37.94% |
| **12,288** | **10** | **21.04%** | 37.88% |
| 16,384 | 10 | 21.02% | 37.83% |
| 32,768 | 10 | 20.99% | 37.75% |
| 131,072 | 10 | 21.03% | 37.71% |
| 1,048,576 | 10 | 21.01% | 37.69% |

最终选择 `ridge_alpha=10`、`semantic_weight=12,288`。它是搜索网格内部峰值；更大的 λ 进入约 20.99%–21.03% 的平台，没有继续提高。selection-excluded holdout Macro 为 20.23%，相对 v2 的 19.99% 再提高 0.24 pp，且与 tune 的差距为 0.80 pp。

| 版本 | 参数 `(α, λ)` | Uniform Dev Macro / Micro | Frequency Dev Macro / Micro |
|---|---|---:|---:|
| 原始 | `(1, 48)` | 15.43% / 42.02% | 4.52% / 17.49% |
| v1 | `(3, 256)` | 17.31% / 38.68% | 6.09% / 17.94% |
| v2 | `(10, 2,048)` | 17.88% / 35.95% | 7.43% / 18.03% |
| **v3** | **`(10, 12,288)`** | **18.19% / 35.40%** | **7.61% / 18.06%** |

相对原始 `(1,48)`，v3 的 Uniform Macro 提高 +2.76 pp，user-cluster paired bootstrap 95% CI 为 `[+1.54,+4.08]`；Frequency-matched 提高 +3.09 pp，95% CI 为 `[+2.33,+3.91]`。v3 相对 v2 的最后一小步在 Uniform 上为 +0.31 pp、CI `[−0.16,+0.90]`，不能排除零；在 Frequency-matched 上为 +0.18 pp、CI `[+0.09,+0.31]`。

Uniform 的分层变化进一步说明这是 Macro 导向的 head/tail 交换：原始到 v3，head 从 53.46% 降到 37.00%，torso 从 7.50% 升到 19.00%，tail 从 0.03% 升到 4.00%。因此 v3 是当前 Interest Macro 目标下的最终冻结参数，但不是 Micro 最优参数。由于 Train tune 已出现内部峰值且极大 λ 平台已经覆盖，不再需要继续扩大 λ。

继续搜索产物：

- v2：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines_semantic_expanded_v2/`
- v3 最终冻结结果：`artifacts/candidate-selection-static-user-disjoint-k10-v1/baselines_semantic_expanded_v3/`
- v3 网格与 holdout：`baselines_semantic_expanded_v3/semantic_tuning.json`
- v3 配置、输入和产物 SHA-256：`baselines_semantic_expanded_v3/resolved_config.json`、`baselines_semantic_expanded_v3/manifest.json`
