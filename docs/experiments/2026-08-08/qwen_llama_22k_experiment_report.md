# Qwen3-8B 与 Llama-3.1-8B：22K SFT 跨模型实验报告

> **状态：已完成。**四个 22K SFT 与 grouped Dev10K Beam 评估均已完成。
> Qwen3 Behavior/Interest 与 Llama Interest 在 Feeds A100 运行；Llama Behavior 因 Feeds 节点反复
> 重调度，转至 GCR A100 从远端 checkpoint 恢复并完成。

## 1. 研究问题（Research Questions）

1. Qwen3-8B 从 10K 扩展到严格平衡 22K 后，Behavior-only 与 Interest-only 的
   Recall/NDCG 是否继续提升？
2. 在相同 22K 数据、LoRA 参数与 Dev10K 上，Qwen3-8B 与
   Llama-3.1-8B-Instruct 的效果有何差异？
3. 两个模型中，Interest-only 是否仍优于 Behavior-only？

对比矩阵：

| 模型 | Behavior 22K | Interest 22K |
|---|---:|---:|
| Qwen3-8B | 完成 | 完成 |
| Llama-3.1-8B-Instruct | 完成 | 完成 |

主要指标：Recall@1/5/10/20 与 NDCG@1/5/10/20。

## 2. 训练数据构造（Training Data Construction）

### 2.1 来源与新主题定义

原始来源为重分类后的 MINDlarge：

```text
data/raw/MINDlarge_reclassified/
```

一个当前点击的二级主题如果没有出现在该 behavior 的历史点击主题中，就定义为
新主题：

```text
target ∈ current_clicked_subcategories
target ∉ history_clicked_subcategories
```

Train 对同一 impression 的多个新主题使用固定 seed `20260806` 选择一个目标，
避免相同 Prompt 对应多个训练答案；Dev 保留全部正确新主题，用于集合评估。

平衡训练源：

```text
data/training/generative_new_interest_balanced_200_h50/
```

每种输入：

```text
22,000 samples
110 second-level topic categories
200 samples per category
one target per Train impression
most recent 50 history items
```

SFT 文件：

```text
data/training/sft_experiments/train/
├── behavior_only/22k.parquet
└── interest_only/22k.parquet
```

22K 是完整平衡序列，且保持：

```text
1K ⊂ 5K ⊂ 10K ⊂ 22K
```

两种输入使用相同 source IDs、答案和顺序。

| 条件 | 类别数 | 每类样本 | 总样本 | held-out Train |
|---|---:|---:|---:|---:|
| Behavior-only 22K | 110 | 200 | 22,000 | 0 |
| Interest-only 22K | 110 | 200 | 22,000 | 0 |

历史保留 MIND 原始顺序，仅使用最后 50 条；Interest-only 保留重复主题，不去重。

### 2.2 Prompt 与目标

Behavior-only 输入历史标题；Interest-only 输入对应的二级主题序列。标准回答：

```text
New topic: <canonical subcategory>
```

Prompt 不包含候选项，不输出解释或 CoT。

### 2.3 真实训练样例

来源：

```text
source_sample_id: train:1024358:116
impression_id: 1024358
record type: SFT target
target: Vehicle Buying & Ownership
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

目标翻译：`新的主题：车辆购买与持有`。

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
下面按时间顺序列出最近浏览新闻对应的主题：

1. 外交与地缘政治
2. 公共生活中的宗教
3. 奥运与田径运动

根据该主题序列预测一个新的后续兴趣类别，且不能与历史主题重复。回答必须以
“New topic: ”开头，之后只输出类别名称。
```

### 2.4 Token audit

| 模型 | 输入 | P50 | P99 | Max | >2048 |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Interest | 223 | 501 | 554 | 0 |
| Qwen3-8B | Behavior | 414 | 1,109 | 1,191 | 0 |
| Llama-3.1-8B | Interest | 236 | 482 | 535 | 0 |
| Llama-3.1-8B | Behavior | 421 | 1,070 | 1,147 | 0 |

四个条件均可使用 `max_seq_length=2048`，没有静默截断。

## 3. 测试设计（Testing）

### 3.1 模型与平台

```text
Qwen/Qwen3-8B
meta-llama/Llama-3.1-8B-Instruct
```

平台：

```text
AMLT experiment: crossinterest-qwen-llama-22k-v1
target: Feeds Singularity
SKU: 80G1-A100
SLA: Standard
jobs: 4
W&B project: cross-interest-generative-sft
```

实际运行位置：

| 条件 | 运行位置 | 远程状态 |
|---|---|---|
| Qwen Behavior | Feeds A100 | pass |
| Qwen Interest | Feeds A100 | pass |
| Llama Interest | Feeds A100 | pass |
| Llama Behavior | GCR A100 | completed；Feeds 副本保持 paused |

### 3.2 SFT 参数

| 参数 | 值 |
|---|---:|
| Precision | BF16 |
| Epochs | 3 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | all linear |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Warmup | 0.03 |
| Scheduler | cosine |
| Effective batch | 32 |
| Seed | 42 |

只对 assistant completion 计算 loss。

### 3.3 Dev10K 与 Beam

使用与现有 scaling 相同、冻结的 grouped Dev10K：

```text
10,000 unique impressions
12,164 acceptable target rows
110 retained target categories
```

同一 impression 可包含多个正确新主题；模型只生成一次排序列表。

```text
candidate beams: 32
deduplicated Top-20
do_sample: false
batch size: 2
checkpoint shard: 64 impressions
max_new_tokens: 32
```

只允许生成 110 个 canonical `New topic:` response。

指标由确定性 Python 代码计算，不使用 LLM Judge：

```text
src/new_interest/evaluation/metrics.py
src/new_interest/evaluation/constrained_beam.py
src/new_interest/evaluation/evaluate_beam.py
```

### 3.4 评估程序（无 LLM Judge）

本实验不使用 LLM Judge、主观 rubric 或拒答判定。所有结果由确定性 Python
程序计算，因此 judge error 为 0，10,000/10,000 条预测均完成。

对 impression `u`，正确集合为 `G_u`，Top-K 预测为 `P_u@K`：

```text
Recall@K(u) = |P_u@K ∩ G_u| / |G_u|
Hit@K(u) = 1, if |P_u@K ∩ G_u| > 0; otherwise 0
```

二元相关性：

```text
rel_i = 1, prediction_i ∈ G_u
DCG@K = Σ rel_i / log2(i + 1)
IDCG@K = 前 min(K, |G_u|) 位均相关时的 DCG
NDCG@K = DCG@K / IDCG@K
```

每个 impression 先单独计算，再进行宏平均。

### 3.5 真实测试样例

#### Interest-only：Llama 命中、Qwen 未命中

```text
impression_id: 2395
correct target: American Football
```

Prompt 原文：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Accidents & Emergency Response
2. Government & Public Policy
3. Government & Public Policy

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
最近浏览新闻的主题依次是：事故与应急响应、政府与公共政策、政府与公共政策。
根据该序列预测一个历史中未出现的新主题类别。
```

Qwen Top-20 不包含 `American Football`：

```text
Journalism & News Media; Crime & Public Safety; Celebrity & Creator Culture;
Theater & Performing Arts; Severe Storms; Courts & Legal Cases;
Companies & Corporate Strategy; Home Improvement & Repair;
Constitutional & Civil Law; Baseball & Softball; Olympic & Athletics Sports;
Cars & Driving; Cooking & Recipes; Telecommunications; Immigration & Citizenship;
Books & Literature; Space & Astronomy; Biology & Genetics;
Wildlife & Conservation; Personal Finance
```

Qwen：`Recall@20=0`，`NDCG@20=0`。

Llama 将 `American Football` 排在第 15 位：

```text
Constitutional & Civil Law; Crime & Public Safety; Celebrity & Creator Culture;
Journalism & News Media; Courts & Legal Cases; Companies & Corporate Strategy;
Economy & Economic Policy; War & Armed Conflict; Cars & Driving;
Books & Literature; Theater & Performing Arts; Baseball & Softball;
Adventure & Nature Travel; Cooking & Recipes; American Football; ...
```

Llama：`Recall@20=1`，`NDCG@20=0.25`。该判断完全由正确主题在排名中的位置决定。

#### Behavior-only：Llama 命中、Qwen 未命中

```text
impression_id: 178
correct target: Cars & Driving
```

Prompt 原文：

```text
The headlines of the news items I recently viewed are listed below:

1. Look who's watching
2. America's Unsafe Drugs and Medical Products

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
最近浏览的标题是“看看谁在观看”和“美国不安全的药品与医疗产品”。
根据这些标题预测一个历史主题中未出现的新类别。
```

Qwen Top-20 未包含 `Cars & Driving`，因此 `Recall@20=0`。
Llama 将 `Cars & Driving` 排在第 14 位，因此
`Recall@20=1`、`NDCG@20=0.2560`。

## 4. 实验结果（Results）

### 4.1 完成状态

| 模型 | 输入 | 状态 |
|---|---|---|
| Qwen3-8B | Interest | SFT 与 Dev10K Beam 完成 |
| Qwen3-8B | Behavior | SFT 与 Dev10K Beam 完成 |
| Llama-3.1-8B | Interest | SFT 与 Dev10K Beam 完成 |
| Llama-3.1-8B | Behavior | SFT 与 Dev10K Beam 完成 |

### 4.2 最终结果表

| 模型 | 输入 | Train loss | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Behavior | 0.5638 | 2.28% | 10.27% | 19.05% | 30.33% | 12.31% |
| Qwen3-8B | Interest | 0.5476 | 3.29% | 14.36% | 24.90% | 39.90% | 16.53% |
| Llama-3.1-8B | Behavior | **0.4133** | 1.81% | 9.20% | 18.25% | 34.03% | 12.77% |
| Llama-3.1-8B | Interest | 0.5816 | 3.60% | 14.90% | 25.38% | 43.29% | 17.81% |

所有 Recall/NDCG 均为 `n=10,000` impression 的宏平均；置信区间未计算。

#### Qwen3-8B 完整指标

| Input Setting | Scale | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Behavior-only | 22K | 2.28% | 10.27% | 19.05% | 30.33% | 2.82% | 6.45% | 9.35% | 12.31% | 34.10% |
| Interest-only | 22K | 3.29% | 14.36% | 24.90% | 39.90% | 3.98% | 9.12% | 12.63% | 16.53% | 44.30% |

#### Llama-3.1-8B-Instruct 完整指标

| Input Setting | Scale | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Behavior-only | 22K | 1.81% | 9.20% | 18.25% | 34.03% | 2.18% | 5.65% | 8.66% | 12.77% | 38.07% |
| Interest-only | 22K | 3.60% | 14.90% | 25.38% | 43.29% | 4.43% | 9.63% | 13.13% | 17.81% | 47.86% |

当前 Interest-only 22K 对比：

| 模型 | Recall@20 | NDCG@20 | Hit@20 |
|---|---:|---:|---:|
| Qwen3-8B | 39.90% | 16.53% | 44.30% |
| Llama-3.1-8B-Instruct | **43.29%** | **17.81%** | **47.86%** |

Behavior-only 22K 对比：

| 模型 | Recall@20 | NDCG@20 | Hit@20 |
|---|---:|---:|---:|
| Qwen3-8B | 30.33% | 12.31% | 34.10% |
| Llama-3.1-8B-Instruct | **34.03%** | **12.77%** | **38.07%** |

四个条件中，Llama Interest 22K 的 Recall@20 与 NDCG@20 最高。两种模型内，
Interest-only 均明显优于 Behavior-only。

### 4.3 完整排名指标

| 模型 | 输入 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 | MRR@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Behavior | 2.82% | 6.45% | 9.35% | 12.31% | 3,410/10,000 | 7.88% |
| Qwen3-8B | Interest | 3.98% | 9.12% | 12.63% | 16.53% | 4,430/10,000 | 10.78% |
| Llama-3.1-8B | Behavior | 2.18% | 5.65% | 8.66% | 12.77% | 3,807/10,000 | 7.47% |
| Llama-3.1-8B | Interest | **4.43%** | **9.63%** | **13.13%** | **17.81%** | **4,786/10,000** | **11.56%** |

### 4.4 训练与推理效率

| 模型 | 输入 | 训练时间 | Train samples/s | Beam 时间 | Beam samples/s |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Behavior | 6.93 h | 2.64 | 7.96 h | 0.349 |
| Qwen3-8B | Interest | 3.47 h | 5.29 | 4.40 h | 0.631 |
| Llama-3.1-8B | Behavior | 4.85 h | 3.78 | 7.98 h | 0.348 |
| Llama-3.1-8B | Interest | 3.00 h | 6.10 | 4.11 h | 0.675 |

## 5. 数据与产物路径（Artifact Paths）

### 本机数据与代码

- [22K Behavior](../../../data/training/sft_experiments/train/behavior_only/22k.parquet)
- [22K Interest](../../../data/training/sft_experiments/train/interest_only/22k.parquet)
- [SFT metadata](../../../data/training/sft_experiments/metadata.json)
- [Qwen Interest config](../../../configs/sft/experiments/qwen3_interest_22k.yaml)
- [Qwen Behavior config](../../../configs/sft/experiments/qwen3_behavior_22k.yaml)
- [Llama Interest config](../../../configs/sft/experiments/llama31_interest_22k.yaml)
- [Llama Behavior config](../../../configs/sft/experiments/llama31_behavior_22k.yaml)
- [AMLT config](../../../configs/amlt/archive/sft-beam/qwen_llama_22k.yaml)
- [Token audits](../../../artifacts/gcr/token-audit-22k)
- [四条件 CSV](../../../artifacts/22k-results/comparison/qwen_llama_22k_results.csv)
- [四条件 JSON](../../../artifacts/22k-results/comparison/qwen_llama_22k_results.json)
- [Qwen/Llama Feeds metrics root](../../../artifacts/amlt/22k-results/crossinterest-qwen-llama-22k-v1)
- [Llama Behavior metrics](../../../artifacts/gcr/22k/llama31-behavior/beam/metrics.json)
- [结果收集器](../../../src/new_interest/experiments/report_22k_results.py)

本机没有保存 checkpoint 或 adapter 权重。

### 远程模型权重

```text
AMLT Blob:
projects/sft4safety/amlt-results/<job-id>/sft/final_adapter

GCR:
/home/v-wuhao1/CrossInterest/artifacts/sft/llama31-8b-behavior-22k/final_adapter
```

### 远程实验与 W&B

```text
AMLT experiment: crossinterest-qwen-llama-22k-v1
Jobs:
- qwen3-interest-22k
- qwen3-behavior-22k
- llama31-interest-22k
- llama31-behavior-22k (Feeds paused; final run completed on GCR)
```

W&B：

- [Qwen Interest combined SFT/Beam](https://wandb.ai/river177-southeast-university/cross-interest-generative-sft/runs/7213945379.48621-59a93e0f-da77)
- [Qwen Behavior combined SFT/Beam](https://wandb.ai/river177-southeast-university/cross-interest-generative-sft/runs/7213945379.48504-c0a50911-2b28)
- [Llama Interest combined SFT/Beam](https://wandb.ai/river177-southeast-university/cross-interest-generative-sft/runs/7213944011.29668-537c7df6-f32b)
- [Llama Behavior resumed SFT](https://wandb.ai/river177-southeast-university/cross-interest-generative-sft/runs/58o1vv17)
- [Llama Behavior Beam](https://wandb.ai/river177-southeast-university/cross-interest-generative-sft/runs/a76ww77j)
