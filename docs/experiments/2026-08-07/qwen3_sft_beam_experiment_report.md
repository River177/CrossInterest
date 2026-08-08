# Qwen3-8B 生成式新主题预测：SFT 与 Beam Search 实验报告

> **报告状态：**截至 2026-08-07 13:48（UTC+8）。六个 SFT 训练和六个
> Dev10K Beam 评估均已完成。Interest-only 10K full Dev 已按用户要求暂停，
> 不作为本轮完成条件。

## 1. 研究问题（Research Questions）

本实验研究以下问题：

1. 给定用户最近浏览的新闻标题，Qwen3-8B 能否直接生成一个历史中未出现的二级主题类别？
2. 将标题替换成对应的二级主题序列后，是否能提升新主题预测效果？
3. 将每类近似均衡的训练规模从 1K 增加到 5K、10K，Recall 与 NDCG 如何变化？
4. 在同一 impression 可能有多个正确新主题时，受限 Beam Search 的 Top-K 排名效果如何？

对比条件：

| 维度 | 条件 |
|---|---|
| 基座模型 | `Qwen/Qwen3-8B` |
| 输入表示 | Behavior-only、Interest-only |
| 训练规模 | 1K、5K、10K |
| 训练方法 | BF16 LoRA SFT |
| 主要指标 | Recall@1/5/10/20、NDCG@1/5/10/20 |
| 辅助指标 | Hit@K、MRR@20、平均排名长度 |

主比较使用固定的 grouped Dev 10K。每种输入的最佳规模再运行完整 grouped Dev。

## 2. 训练数据构造（Training Data Construction）

### 2.1 原始来源与新主题定义

原始数据为重分类后的 MINDlarge：

```text
data/raw/MINDlarge_reclassified/
```

兴趣标签只使用二级类别名称。一个当前点击主题如果没有出现在该 behavior 的历史
点击主题集合中，就被定义为新主题：

```text
target ∈ current_clicked_subcategories
target ∉ history_clicked_subcategories
```

该定义不要求目标主题以前曝光过，也不使用未点击候选新闻。

### 2.2 One-target Train

一次 Train behavior 可能包含多个不同的新主题。为避免相同 Prompt 对应多个训练答案，
Train 对每个 impression 只选择一个目标：

```text
target_priority =
SHA256("20260806:target:<source_sample_id>")
```

选择 priority 最小的目标。该步骤使用固定 seed `20260806`，结果可复现。

Dev 不执行单目标选择，而是保留过滤后所有正确主题，用于多答案集合评估。

### 2.3 类别平衡

唯一目标选择后，只保留 Train 中至少有 200 个样本的主题：

| 项目 | 数量 |
|---|---:|
| 原始 Train 新主题样本 | 1,489,348 |
| 每 impression 一个目标后 | 1,219,553 |
| 保留主题数 | 110 |
| 每个主题选择数 | 200 |
| 平衡 Train 总数 | 22,000 |

类别内采样使用独立 priority：

```text
balance_priority =
SHA256("20260806:balance:<source_sample_id>")
```

不足 200 条的类别被删除，不进行重复采样。

### 2.4 历史截断与顺序

输入只保留 history 最后 50 条：

```python
recent_history = history[-50:]
```

历史顺序沿用 MIND `behaviors.tsv` 的原始排列，视为从较早到较新。Interest-only
逐新闻映射为二级类别，保留顺序与重复次数，不去重、不合并连续重复类别。

### 2.5 两种训练表示

#### Behavior-only

输入只包含新闻标题，因此模板使用 `headlines`，而不是将标题称为完整 article：

```text
The headlines of the news items I recently viewed are listed below:

{numbered_headlines}

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

#### Interest-only

输入为历史新闻对应的二级主题类别：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

{numbered_subcategories}

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

两种表示的目标格式相同：

```text
New topic: <canonical subcategory>
```

不包含一级类别、候选列表、解释或 CoT。

### 2.6 嵌套 1K/5K/10K 子集

110 类按稳定 class priority 排序，类内按 balance priority 排序，再执行
round-robin。1K、5K、10K 是严格前缀，因此：

```text
1K ⊂ 5K ⊂ 10K
```

| 子集 | 总数 | 类别数 | 每类最少 | 每类最多 |
|---|---:|---:|---:|---:|
| 1K | 1,000 | 110 | 9 | 10 |
| 5K | 5,000 | 110 | 45 | 46 |
| 10K | 10,000 | 110 | 90 | 91 |

Behavior-only 与 Interest-only 在每个规模使用完全相同的 `source_sample_id` 集合。

### 2.7 真实训练样例

来源：

```text
source_sample_id: train:1024358:116
record type: SFT target record
target_subcategory: Vehicle Buying & Ownership
```

#### Behavior-only 原文

```text
The headlines of the news items I recently viewed are listed below:

1. China's Xi warns attempts to divide China will end in 'shuttered bones'
2. 'We're standing by it': Sheriff doubles down on 'In God We Trust' decals after atheist group's complaint
3. Nike will look into runner Mary Cain's allegations of abuse

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

目标回答：

```text
New topic: Vehicle Buying & Ownership
```

中文翻译：

```text
下面列出我最近浏览的新闻标题：

1. 中国国家主席习近平警告，任何分裂中国的企图都将导致“粉身碎骨”
2. “我们坚持这么做”：在无神论团体投诉后，警长仍坚持使用“我们信仰上帝”贴花
3. 耐克将调查跑步运动员 Mary Cain 提出的虐待指控

根据这些标题，预测一个我接下来可能感兴趣的新主题类别。预测类别应不同于浏览历史
所反映的主题。回答必须以“New topic: ”开头，之后只输出类别名称。
```

目标回答翻译：`新的主题：车辆购买与持有`。

#### Interest-only 原文

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Diplomacy & Geopolitics
2. Religion in Public Life
3. Olympic & Athletics Sports

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

目标回答：

```text
New topic: Vehicle Buying & Ownership
```

中文翻译：

```text
下面按时间顺序列出我最近浏览的新闻所对应的主题类别：

1. 外交与地缘政治
2. 公共生活中的宗教
3. 奥运与田径运动

根据这一主题类别序列，预测一个我接下来可能感兴趣的新主题类别。预测类别不能出现
在上面的序列中。回答必须以“New topic: ”开头，之后只输出类别名称。
```

## 3. 测试设计（Testing）

### 3.1 测试模型与运行位置

| 项目 | 设置 |
|---|---|
| 基座模型 | `Qwen/Qwen3-8B` |
| Thinking | `enable_thinking=false` |
| Adapter | 每个输入×规模对应独立 LoRA adapter |
| 训练/推理平台 | Microsoft Research AMLT，Feeds Singularity |
| GPU | 每作业 1×A100 80GB，`80G1-A100` |
| SLA | Standard |
| AMLT 主实验 | `crossinterest-qwen3-scaling-v1` |
| Beam recovery 实验 | `beam-eval-v2-crossinterest-qwen3-scaling-v1` |
| Interest full Dev | `beam-full-crossinterest-qwen3-scaling-v1` |
| W&B project | `cross-interest-generative-sft` |

模型与 adapter 均在远程 A100 上加载；metrics 和 prediction Parquet 下载到本机。

### 3.2 SFT 设置

| 参数 | 值 |
|---|---:|
| Precision | BF16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | all linear |
| Epochs | 3 |
| Micro batch | 4 |
| Gradient accumulation | 8 |
| Effective batch | 32 |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Warmup ratio | 0.03 |
| Scheduler | cosine |
| Optimizer | `adamw_torch_fused` |
| Gradient checkpointing | enabled |
| Max sequence length | 2,048 |
| Seed | 42 |

只对 assistant completion 计算 causal LM loss；user Prompt token 和 padding token
的 label 均为 `-100`。

### 3.3 测试数据来源

Dev 先过滤到 Train 保留的 110 类，但不删除同一 impression 的多个正确目标：

| 项目 | 数量 |
|---|---:|
| 过滤后的 target rows | 242,839 |
| grouped impressions | 199,471 |
| 单答案 impressions | 167,375 |
| 多答案 impressions | 32,096 |
| 最大正确答案数 | 13 |

固定 Dev10K 从 199,471 个 impression 中按
`SHA256("20260806:eval-impression:<impression_id>")` 选择 10,000 个。

### 3.4 Beam Search 设置

输出空间被限制为 110 个 canonical response：

```text
New topic: <canonical subcategory>
```

110 条 response 经 tokenizer 编码后构造成 token trie，通过
`prefix_allowed_tokens_fn` 限制每一步可生成 token。

| 参数 | 值 |
|---|---:|
| Candidate beam width | 32 |
| 最终排名 | 去重后 Top-20 |
| Sampling | disabled |
| `max_new_tokens` | 32 |
| Batch size | 2 |
| Checkpoint size | 每 64 impressions 写一个 Parquet shard |
| 排序依据 | `sequences_scores` 降序 |
| 指标截断点 | 1、5、10、20 |

使用 32 个候选 beams 是因为标准 20-beam 会产生重复终止路径，不能始终提供 20 个
不同类别。若去重后不足 20 类，缺失位置按未召回处理。

### 3.5 评估程序与指标含义

本实验**不使用 LLM Judge**，因此没有 Judge system prompt 或主观 rubric。
所有结果由确定性 Python 评估器计算：

```text
src/new_interest/metrics.py
src/new_interest/constrained_beam.py
src/new_interest/evaluate_beam.py
```

对 impression `u`，正确集合为 `G_u`，Top-K 预测为 `P_u@K`：

```text
Recall@K(u) = |P_u@K ∩ G_u| / |G_u|
```

二元相关性：

```text
rel_i = 1, prediction_i ∈ G_u
rel_i = 0, otherwise

DCG@K = Σ rel_i / log2(i + 1)
IDCG@K = 前 min(K, |G_u|) 位均相关时的 DCG
NDCG@K = DCG@K / IDCG@K
```

最终指标是所有 impression 指标的宏平均。

辅助指标：

- `Hit@K`：Top-K 是否至少命中一个正确主题；
- `MRR@20`：第一个正确主题名次倒数；
- `ranking_length`：去重后的实际排名长度；
- `per_class_recall`：各目标类别的召回率；
- `by_answer_count`：按正确答案数量分组的指标。

`Recall@K` 不要求必须命中全部正确主题。假设某个 impression 有三个正确主题，
Top-K 命中其中一个，则 `Recall@K=1/3`，而 `Hit@K=1`；命中其中两个时
`Recall@K=2/3`，`Hit@K` 仍为 1。只有命中全部三个主题时 `Recall@K=1`。

### 3.6 真实测试样例

#### 命中样例

```text
impression_id: 178
model: Qwen3-8B + Interest-only 10K LoRA
correct target: Cars & Driving
```

测试 Prompt 原文：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. General & Multi-Sport Coverage
2. Medicine & Medical Research

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
下面按时间顺序列出最近浏览新闻对应的主题类别：

1. 综合及多项体育报道
2. 医学与医疗研究

根据这一主题序列预测一个新的后续兴趣类别。预测类别不能出现在上面的历史中。
```

实际 Top-20：

```text
1. Baseball & Softball
2. Government & Public Policy
3. Severe Storms
4. Crime & Public Safety
5. Celebrity & Creator Culture
6. Air & Rail Travel
7. Courts & Legal Cases
8. Nutrition & Healthy Eating
9. Companies & Corporate Strategy
10. Vehicle Buying & Ownership
11. War & Armed Conflict
12. Infrastructure & Transportation Policy
13. Cooking & Recipes
14. Mental Health
15. Television & Streaming
16. Cars & Driving
17. Film
18. Biology & Genetics
19. Wildlife & Conservation
20. Public Health
```

中文翻译：目标“汽车与驾驶”出现在第 16 位，因此
`Recall@20=1`，`NDCG@20=0.2447`。

#### 未命中样例

```text
impression_id: 597
model: Qwen3-8B + Interest-only 10K LoRA
correct target: Elections & Political Campaigns
```

测试 Prompt 原文：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Television & Streaming
2. Fashion & Personal Style
3. Celebrity & Creator Culture

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
最近浏览新闻对应的主题依次为：

1. 电视与流媒体
2. 时尚与个人风格
3. 名人与创作者文化

根据该序列预测一个新的后续兴趣类别，且不能与历史类别重复。
```

实际 Top-20 为：

```text
Government & Public Policy; Companies & Corporate Strategy;
Nutrition & Healthy Eating; Careers & Professional Development;
Hotels & Accommodation; Infrastructure & Transportation Policy;
Theater & Performing Arts; Courts & Legal Cases;
Vehicle Buying & Ownership; Air & Rail Travel;
Household Cleaning & Laundry; Crime & Public Safety;
Architecture & Residential Design; Medicine & Medical Research;
Culture & Heritage Travel; Cooking & Recipes; Fitness & Exercise;
Restaurants & Dining; Cars & Driving; Books & Literature
```

中文说明：正确目标“选举与政治竞选”未进入 Top-20，因此
`Recall@20=0`、`NDCG@20=0`。

## 4. 实验结果（Results）

### 4.1 SFT 训练结果

六个训练均完成，且没有样本超过 2,048 tokens。

| 输入 | 规模 | Train loss | 训练时间 | P99 tokens | Max tokens |
|---|---:|---:|---:|---:|---:|
| Interest-only | 1K | 0.8755 | 9.9 min | 502 | 521 |
| Interest-only | 5K | 0.6257 | 47.1 min | 500 | 523 |
| Interest-only | 10K | 0.5797 | 92.2 min | 501 | 554 |
| Behavior-only | 1K | 0.9799 | 19.1 min | 1,100 | 1,142 |
| Behavior-only | 5K | 0.6667 | 94.4 min | 1,109 | 1,168 |
| Behavior-only | 10K | 0.6084 | 187.6 min | 1,111 | 1,179 |

### 4.2 已完成的 Dev10K Beam 结果

所有已完成结果均有 `10,000/10,000` impression prediction，评估异常数为 0。
置信区间未计算。

| Input Setting | Scale | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interest-only | 1K | 1.17% | 9.22% | 17.99% | 28.31% | 1.41% | 5.21% | 8.16% | 10.90% | 3,227/10,000 (32.27%) |
| Interest-only | 5K | 1.31% | 8.68% | 18.60% | 34.09% | 1.61% | 5.06% | 8.32% | 12.43% | 3,842/10,000 (38.42%) |
| Interest-only | 10K | **1.60%** | **9.29%** | **20.81%** | **36.42%** | **1.91%** | **5.57%** | **9.37%** | **13.52%** | **4,083/10,000 (40.83%)** |
| Interest-only | 22K | **3.29%** | **14.36%** | **24.90%** | **39.90%** | **3.98%** | **9.12%** | **12.63%** | **16.53%** | **4,430/10,000 (44.30%)** |
| Behavior-only | 1K | 0.53% | 6.23% | 13.54% | 23.27% | 0.66% | 3.38% | 5.81% | 8.40% | 2,681/10,000 (26.81%) |
| Behavior-only | 5K | 1.01% | 6.41% | 15.42% | 26.40% | 1.30% | 3.77% | 6.73% | 9.65% | 3,000/10,000 (30.00%) |
| Behavior-only | 10K | **2.29%** | **10.80%** | **19.20%** | **29.05%** | **2.76%** | **6.72%** | **9.51%** | **12.11%** | **3,293/10,000 (32.93%)** |
| Behavior-only | 22K | **2.28%** | **10.27%** | **19.05%** | **30.33%** | **2.82%** | **6.45%** | **9.35%** | **12.31%** | **3,410/10,000 (34.10%)** |

当前已观察到：

- Interest-only 的 Recall@20 从 1K 的 28.31% 提升到 10K 的 36.42%；
- Interest-only 的 NDCG@20 从 10.90% 提升到 13.52%；
- Interest-only 22K 进一步达到 Recall@20 39.90%、NDCG@20 16.53%；
- 在 1K 条件下，Interest-only 的 Recall@20 和 NDCG@20 都高于 Behavior-only；
- 当前 Interest-only 最佳规模为 10K，已提交完整 grouped Dev 评估。
- Behavior-only 5K 的 Recall@20 和 NDCG@20 均高于 Behavior-only 1K；
- Behavior-only 10K 进一步提升到 Recall@20 29.05%、NDCG@20 12.11%，
  因而是 Behavior-only 最佳规模；
- Behavior-only 22K 的 Recall@20 提升到 30.33%、NDCG@20 提升到 12.31%，
  但提升幅度小于 Interest-only 的 10K→22K 增益；

### 4.3 排名完整性

32 个候选 beams 经类别去重后，少数 impression 不足 20 个唯一类别。缺失名次按
未召回处理：

| 输入 | 规模 | 少于 20 个排名的 impressions | 平均排名长度 |
|---|---:|---:|---:|
| Interest-only | 1K | 288/10,000 | 19.9633 |
| Interest-only | 5K | 184/10,000 | 19.9764 |
| Interest-only | 10K | 187/10,000 | 19.9762 |
| Behavior-only | 1K | 38/10,000 | 19.9951 |
| Behavior-only | 5K | 115/10,000 | 19.9853 |
| Behavior-only | 10K | 84/10,000 | 19.9892 |

### 4.4 当前远程运行状态

截至报告时间：

| 作业 | 状态 |
|---|---|
| Interest-only 10K full Dev Beam | 已暂停（129/3,117 shards） |

Behavior-only 按以下顺序选择最佳规模：

1. NDCG@20；
2. Recall@20；
3. NDCG@10；
4. 若仍相同，选择更小数据集。

最终选择结果：

```text
Interest-only best: 10K
Behavior-only best: 10K
```

## 5. 数据与产物路径（Artifact Paths）

K=10 候选选择扩展实验见：

```text
docs/candidate_selection_experiment_report.md
```

### 5.1 数据构造

- [原始数据](/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified)
- [新主题 Train](/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest/train.parquet)
- [新主题 Dev](/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest/dev.parquet)
- [Balanced metadata](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/generative_new_interest_balanced_200_h50/metadata.json)
- [保留类别](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/generative_new_interest_balanced_200_h50/retained_interests.parquet)
- [Train 采样清单](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/generative_new_interest_balanced_200_h50/selected_train_samples.parquet)

### 5.2 SFT 数据

- [SFT metadata](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/metadata.json)
- [SFT statistics](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/statistics.json)
- [子集清单](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/subset_manifest.parquet)
- [Behavior Train](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/train/behavior_only)
- [Interest Train](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/train/interest_only)
- [Behavior Dev](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/eval/behavior_only)
- [Interest Dev](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/eval/interest_only)

### 5.3 代码与配置

- [训练代码](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/train_sft.py)
- [Completion-only 数据处理](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/sft_data.py)
- [Beam 评估](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/evaluate_beam.py)
- [Token trie](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/constrained_beam.py)
- [指标](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/metrics.py)
- [SFT 配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/sft/qwen3_8b_lora.yaml)
- [Beam 配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/beam)
- [AMLT 六训练配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/amlt/qwen3_8b_scaling.yaml)
- [Beam recovery 配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/amlt/qwen3_beam_map.yaml)
- [Full Dev 配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/amlt/qwen3_full_beam_map.yaml)

### 5.4 本机结果

训练 adapter 与 checkpoint 只保存在 AMLT Blob/GCR，不保存在本机。

- [Beam 结果根目录](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1)
- [Interest 1K metrics](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/beam-eval-v2-qwen3-interest-1k/beam/metrics.json)
- [Interest 5K metrics](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/beam-eval-v2-qwen3-interest-5k/beam/metrics.json)
- [Interest 10K metrics](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/beam-eval-v2-qwen3-interest-10k/beam/metrics.json)
- [Behavior 1K metrics](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/beam-eval-v2-qwen3-behavior-1k/beam/metrics.json)
- [Behavior 5K metrics](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/beam-eval-v2-qwen3-behavior-5k/beam/metrics.json)
- [Behavior 10K metrics](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/beam-eval-v2-qwen3-behavior-10k/beam/metrics.json)
- [六实验对比 CSV](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/qwen3_dev10k_comparison.csv)
- [六实验对比 JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/amlt/beam-eval-v2-crossinterest-qwen3-scaling-v1/qwen3_dev10k_comparison.json)

### 5.5 远程实验

```text
AMLT training experiment:
crossinterest-qwen3-scaling-v1

AMLT Dev10K Beam experiment:
beam-eval-v2-crossinterest-qwen3-scaling-v1

AMLT Interest full Dev experiment:
beam-full-crossinterest-qwen3-scaling-v1

W&B:
https://wandb.ai/river177-southeast-university/cross-interest-generative-sft
```

训练作业：

```text
qwen3-interest-1k
qwen3-interest-5k
qwen3-interest-10k
qwen3-behavior-1k
qwen3-behavior-5k
qwen3-behavior-10k
```
