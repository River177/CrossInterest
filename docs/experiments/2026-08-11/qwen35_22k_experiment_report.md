# Qwen3.5-9B 22K 新兴趣预测实验报告

> **状态：已完成。** Behavior-only 与 Interest-only 的 22K LoRA SFT、冻结
> grouped Dev10K constrained Beam Top-20 评估均已完成。Checkpoint、adapter
> 和基础模型权重仅保留在 AMLT Blob；本机只保存训练摘要、指标、必要预测与配置。

## 1. 研究问题（Research Questions）

本实验测试：

1. `Qwen/Qwen3.5-9B` 能否在当前平衡 22K 数据上稳定学习新兴趣预测，而不复现旧
   Qwen3.5 候选字母任务中的严重过拟合和类别先验漂移？
2. 在相同训练数据、直接主题输出和 frozen Dev10K Beam 协议下，
   Qwen3.5-9B 的 Behavior-only 与 Interest-only 表现如何？
3. Qwen3.5-9B 相比已完成的 Qwen3-8B 和 Llama-3.1-8B 22K 结果是否提升？

主指标为 Recall@1/5/10/20、NDCG@1/5/10/20、Hit@20 和 MRR@20。

## 2. 训练数据构造（Training Data Construction）

### 2.1 数据来源与选择规则

训练数据来自重分类后的 MINDlarge：

```text
data/raw/MINDlarge_reclassified/
```

若当前点击的二级主题未出现在该用户历史点击主题中，则定义为新兴趣：

```text
target ∈ current_clicked_subcategories
target ∉ history_clicked_subcategories
```

Train 对每个 impression 使用 seed `20260806` 固定选择一个 target，避免同一 Prompt
对应多个训练答案。历史最多保留最近 50 条。最终保留 110 个二级主题，每类 200 条：

| 输入 | 类别数 | 每类样本 | 总样本 | held-out Train |
|---|---:|---:|---:|---:|
| Behavior-only | 110 | 200 | 22,000 | 0 |
| Interest-only | 110 | 200 | 22,000 | 0 |

两种输入使用相同 source sample、target 与顺序：

```text
data/training/sft_experiments/train/
├── behavior_only/22k.parquet
└── interest_only/22k.parquet
```

回答始终是：

```text
New topic: <canonical second-level topic>
```

不使用 A-J 字母、候选主题、解释或 CoT。训练 loss 仅覆盖 assistant completion。

### 2.2 Token 与模型审计

| 输入 | Mean | P50 | P90 | P99 | Max | >2048 |
|---|---:|---:|---:|---:|---:|---:|
| Behavior | 519.18 | 416 | 1,033 | 1,114 | 1,195 | 0 |
| Interest | 265.43 | 223 | 475 | 497 | 553 | 0 |

实际模型审计：

```text
base model class: Qwen3_5ForCausalLM
model_type: qwen3_5_text
wrapped class: PeftModelForCausalLM
total parameters: 8,997,081,600
trainable parameters: 43,278,336
trainable ratio: 0.4810%
forbidden visual/vision trainable parameters: 0
tokenizer: Qwen2Tokenizer
```

因此本实验使用纯文本 CausalLM 路径，没有训练 Qwen3.5 的视觉模块。

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

目标翻译：`新的主题：车辆购买与持有`。

## 3. 测试设计（Testing）

### 3.1 模型、训练位置与参数

```text
model: Qwen/Qwen3.5-9B
environment: Transformers 5.14.1 / PEFT 0.18.0 / Accelerate 1.12.0
platform: Feeds Singularity
training experiment: qwen35-22k-16gpu-20260810-v2
training jobs:
  qwen35-interest-22k
  qwen35-behavior-22k
hardware: 8 × A100 80GB per input, 16 GPUs total
precision: BF16
attention: SDPA
thinking: disabled
epochs: 3
learning rate: 2e-5
warmup ratio: 0.03
scheduler: cosine
weight decay: 0.01
LoRA rank / alpha / dropout: 16 / 32 / 0.05
LoRA targets: all-linear in the text-only CausalLM
per-device batch: 1
gradient accumulation: 4
global effective batch: 32
seed: 42
```

训练结果：

| 输入 | Steps | Runtime | Samples/s | Train loss |
|---|---:|---:|---:|---:|
| Behavior | 2,064 | 7h 17m | 2.514 | 0.5635 |
| Interest | 2,064 | 5h 13m | 3.518 | 0.5485 |

初始 16 卡 v1 在训练开始前失败：8 个 rank 同时从 Hugging Face 下载四个权重
分片，部分 rank 观察到不同的缺失 shard。v2 改为每个节点先由单进程完成
`snapshot_download`，随后启用 Hugging Face/Transformers offline mode，再启动
`torchrun`。v2 的 8 个 rank 均成功加载并完成训练。

### 3.2 Smoke 与 Pilot

32-example smoke 验证了纯文本模型加载、LoRA、loss 下降、adapter 保存/重载和
constrained Beam。正式 22K 前还训练了严格嵌套的 1K pilot：

| 输入 | Pilot Recall@20 | Pilot NDCG@20 | Top-1 类别数 | 最大 Top-1 占比 | 归一化熵 |
|---|---:|---:|---:|---:|---:|
| Behavior | 23.03% | 8.55% | 22 | 31.8% | 0.451 |
| Interest | 25.70% | 10.30% | 27 | 20.7% | 0.530 |

两者 NDCG@20 均高于 110 类随机基线 6.69%，且没有单类别完全塌缩。

### 3.3 测试数据与解码

测试集为与 Qwen3/Llama 22K 完全相同的 frozen grouped Dev10K：

```text
10,000 unique impressions
12,164 acceptable target rows
110 retained categories
```

同一 impression 可有多个正确主题。模型仅生成一次排序：

```text
evaluation experiment: qwen35-22k-beam-20260811-v1
jobs:
  qwen35-beam-dev10k-qwen35-interest-22k
  qwen35-beam-dev10k-qwen35-behavior-22k
num_beams: 32
deduplicated top_k: 20
do_sample: false
batch_size: 2
checkpoint shard size: 64
max_new_tokens: 32
thinking: disabled
```

评估在 AMLT 远程执行；本机只下载了 metrics、prediction Parquet 和 resolved config。

### 3.4 真实测试样例

#### Interest Top-1 命中

```text
impression_id: 6839

The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Journalism & News Media
2. Television & Streaming

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

模型 Top-5：

```text
1. Celebrity & Creator Culture
2. Crime & Public Safety
3. Government & Public Policy
4. Courts & Legal Cases
5. Companies & Corporate Strategy
```

正确集合：`Celebrity & Creator Culture`，因此 Hit@1=1。

中文翻译：

```text
历史主题依次为“新闻与新闻媒体”“电视与流媒体”。模型需要预测一个未出现在历史中的
新主题。模型首选“名人与创作者文化”，其后为“犯罪与公共安全”“政府与公共政策”
“法院与法律案件”“公司与企业战略”。正确主题为“名人与创作者文化”。
```

#### Interest Top-20 未命中

```text
impression_id: 596

The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Crime & Public Safety
2. Government & Public Policy
3. Elections & Political Campaigns
4. Television & Streaming

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

模型 Top-5：

```text
1. Courts & Legal Cases
2. Celebrity & Creator Culture
3. Journalism & News Media
4. Companies & Corporate Strategy
5. Economy & Economic Policy
```

正确集合：`Shopping & Gift Ideas`，该主题未进入 Top-20。

中文翻译：

```text
历史主题依次为“犯罪与公共安全”“政府与公共政策”“选举与政治竞选”“电视与流媒体”。
模型 Top-5 为“法院与法律案件”“名人与创作者文化”“新闻与新闻媒体”“公司与企业战略”
“经济与经济政策”。真实新主题是“购物与礼物创意”，因此 Top-20 未命中。
```

#### Behavior Top-1 命中

```text
impression_id: 334

The headlines of the news items I recently viewed are listed below:

1. In-N-Out heiress Lynsi Snyder opens up in rare interview
2. The 2020 Ram 1500 EcoDiesel's Official MPG Numbers Are Impressive
3. Felicity Huffman Is Scheduled to Be Released from Prison on October 27 After Serving 13 Days
4. Meghan McCain confronts Trump Jr.: 'You and your family have hurt a lot of people'

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

模型 Top-5：

```text
1. Celebrity & Creator Culture
2. Shopping & Gift Ideas
3. Cars & Driving
4. Vehicle Buying & Ownership
5. Electric & Alternative Vehicles
```

正确集合：`Celebrity & Creator Culture`，因此 Hit@1=1。

中文翻译：

```text
历史包括 In-N-Out 女继承人访谈、Ram 1500 柴油车油耗、Felicity Huffman 服刑释放，
以及 Meghan McCain 与 Trump Jr. 的冲突。模型首选“名人与创作者文化”，随后为
“购物与礼物创意”“汽车与驾驶”“车辆购买与持有”“电动与替代能源车辆”。
正确主题为“名人与创作者文化”。
```

#### Behavior Top-20 未命中

```text
impression_id: 566

The headlines of the news items I recently viewed are listed below:

1. A Louisiana man decided to steal an electric powered shopping cart from Walmart, police say
2. Keanu Reeves holds hands with Alexandra Grant as the Internet swoons, plus more news
3. Jamie Lee Curtis Opens Up About Being 20 Years Sober, Going Public With Her Addiction

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

模型 Top-5：

```text
1. Crime & Public Safety
2. Shopping & Gift Ideas
3. Celebrity & Creator Culture
4. Fashion & Personal Style
5. Courts & Legal Cases
```

正确集合：`Nutrition & Healthy Eating`，该主题未进入 Top-20。

中文翻译：

```text
历史包括沃尔玛电动购物车盗窃、Keanu Reeves 恋情新闻、Jamie Lee Curtis 谈戒除
成瘾。模型 Top-5 为“犯罪与公共安全”“购物与礼物创意”“名人与创作者文化”
“时尚与个人风格”“法院与法律案件”。真实新主题是“营养与健康饮食”，因此
Top-20 未命中。
```

### 3.5 指标程序

本实验不使用 LLM Judge。排序被限制为 110 个 canonical topic，指标由确定性代码
计算：

```text
src/new_interest/evaluation/constrained_beam.py
src/new_interest/evaluation/metrics.py
src/new_interest/evaluation/evaluate_beam.py
```

对 impression \(i\)，正确集合为 \(G_i\)，前 K 个预测为 \(P_{i,K}\)：

```text
Recall@K_i = |P_i,K ∩ G_i| / |G_i|
Hit@K_i = 1[P_i,K ∩ G_i ≠ ∅]
DCG@K_i = Σ rel(r) / log2(r + 1)
NDCG@K_i = DCG@K_i / IDCG@K_i
MRR@20_i = 1 / first_relevant_rank, if rank ≤ 20, else 0
```

最终指标为 10,000 个 impression 的 macro mean。详细定义见
`docs/metrics/recall_ndcg_hit.md`。

## 4. 实验结果（Results）

### 4.1 Qwen3.5-9B Dev10K

| Input Setting | Scale | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Behavior-only | 22K | 1.95% | 10.07% | 17.76% | **32.64%** | 2.37% | 6.18% | 8.73% | **12.62%** | 36.48% |
| Interest-only | 22K | 4.12% | 16.60% | 26.95% | **45.27%** | 5.06% | 10.78% | 14.23% | **18.99%** | 49.68% |

其他指标：

| 输入 | MRR@20 | Top-1 类别数 | 最大 Top-1 占比 | 归一化 Top-1 熵 | Beam runtime |
|---|---:|---:|---:|---:|---:|
| Behavior | 7.63% | 84 | 9.69% | 0.771 | 10h 39m |
| Interest | 12.60% | 89 | 11.35% | 0.749 | 3h 43m |

最终预测分布覆盖 84/89 个 Top-1 类别，最大单类占比均低于 12%，没有复现旧实验的
单一输出先验漂移。

### 4.2 Macro Class Recall/NDCG@10

前表的 Recall/NDCG 是对 10,000 个 impressions 等权平均。类别宏平均采用不同
口径：

```text
Class Recall@10(c)
  = 含真实类别 c 的 impressions 中，c 进入 Top-10 的比例

Class NDCG@10(c)
  = 含真实类别 c 的 impressions 中，
    若 c 位于 rank r≤10 则记 1/log2(r+1)，否则记 0

Macro Class metric
  = 对所有有正样本支持的类别等权平均
```

110 个 retained labels 中，冻结 Dev10K 的 acceptable targets 实际覆盖 101 类；
另外 9 类 support=0，指标无定义，故不纳入宏平均。

| Input Setting | supported classes | Macro Class Recall@10 | Macro Class NDCG@10 |
|---|---:|---:|---:|
| Behavior-only | 101 | **14.29%** | **7.31%** |
| Interest-only | 101 | **17.74%** | **8.89%** |

类别 support 高度不均衡，范围为 1–1,361；因此该指标与 impression-level
Recall@10/NDCG@10 差异明显。

相同类别宏平均口径下的理论随机基线：

| Random baseline | Macro Class Recall@10 | Macro Class NDCG@10 |
|---|---:|---:|
| 从全部 110 类随机 | 9.09% | 4.13% |
| 每条 impression 排除历史主题后随机 | 10.37% | 4.71% |

这里的排除历史随机值不同于 impression-level 的 10.18% / 4.84%，因为类别宏平均
先对每个真实类别的出现样本求平均，再对 101 个有 support 的类别等权平均；且
Class NDCG 将每个类别单独视为唯一相关项。

完整跨模型 @10 对比：

| Model | Recall@10 B | Recall@10 I | NDCG@10 B | NDCG@10 I | Macro Recall@10 B | Macro Recall@10 I | Macro NDCG@10 B | Macro NDCG@10 I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | **19.05%** | 24.90% | **9.35%** | 12.63% | 13.71% | 17.37% | **7.36%** | 8.81% |
| Qwen3.5-9B | 17.76% | **26.95%** | 8.73% | **14.23%** | 14.29% | **17.74%** | 7.31% | 8.89% |
| Llama3.1-8B | 18.25% | 25.38% | 8.66% | 13.13% | **15.04%** | 17.71% | 7.34% | **9.38%** |
| Random Guess | 10.18% | 10.18% | 4.84% | 4.84% | 10.37% | 10.37% | 4.71% | 4.71% |

### 4.3 跨模型 22K 对比

| 模型 | 输入 | Recall@20 | NDCG@20 | Hit@20 | MRR@20 |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Behavior | 30.33% | 12.31% | 34.10% | 7.88% |
| Llama-3.1-8B | Behavior | **34.03%** | **12.77%** | **38.07%** | 7.47% |
| Qwen3.5-9B | Behavior | 32.64% | 12.62% | 36.48% | **7.63%** |
| Qwen3-8B | Interest | 39.90% | 16.53% | 44.30% | 10.78% |
| Llama-3.1-8B | Interest | 43.29% | 17.81% | 47.86% | 11.56% |
| Qwen3.5-9B | Interest | **45.27%** | **18.99%** | **49.68%** | **12.60%** |

观察值：

- Qwen3.5 Interest 相比 Qwen3-8B Interest：
  Recall@20 +5.36 个百分点，NDCG@20 +2.46 个百分点。
- Qwen3.5 Interest 相比 Llama Interest：
  Recall@20 +1.98 个百分点，NDCG@20 +1.19 个百分点。
- Qwen3.5 Behavior 高于 Qwen3-8B Behavior，但略低于 Llama Behavior：
  Recall@20 低 1.39 个百分点，NDCG@20 低 0.15 个百分点。
- 在 Qwen3.5 内部，Interest 相比 Behavior：
  Recall@20 +12.62 个百分点，NDCG@20 +6.37 个百分点。

完整小数结果见
[`qwen35_22k_results.csv`](qwen35_22k_results.csv)。

## 5. 数据与产物路径（Artifact Paths）

### 5.1 代码与配置

```text
requirements-qwen35-worker.txt
src/new_interest/training/train_sft.py
src/new_interest/training/sft_data.py
src/new_interest/evaluation/evaluate_beam.py
src/new_interest/evaluation/constrained_beam.py
src/new_interest/evaluation/metrics.py
configs/sft/experiments/qwen35_behavior_22k.yaml
configs/sft/experiments/qwen35_interest_22k.yaml
configs/beam/qwen35_behavior_22k.json
configs/beam/qwen35_interest_22k.json
configs/amlt/qwen35_22k_smoke.yaml
configs/amlt/qwen35_22k_pilot.yaml
configs/amlt/qwen35_22k_train.yaml
configs/amlt/qwen35_22k_beam_map.yaml
scripts/amlt/sft/run_qwen35_22k.sh
scripts/amlt/evaluation/run_qwen35_22k_beam_map.sh
scripts/amlt/evaluation/install_qwen35_worker_deps.sh
```

### 5.2 数据

```text
data/training/sft_experiments/train/behavior_only/22k.parquet
data/training/sft_experiments/train/interest_only/22k.parquet
data/training/sft_experiments/eval/behavior_only/dev_10k_grouped.parquet
data/training/sft_experiments/eval/interest_only/dev_10k_grouped.parquet
data/training/generative_new_interest_balanced_200_h50/retained_interests.parquet
```

### 5.3 AMLT 远程实验

```text
smoke:
  qwen35-22k-smoke-20260810-v1
pilot:
  qwen35-22k-pilot-20260810-v1
successful training:
  qwen35-22k-16gpu-20260810-v2
  jobs: qwen35-behavior-22k, qwen35-interest-22k
final evaluation:
  qwen35-22k-beam-20260811-v1
  jobs:
    qwen35-beam-dev10k-qwen35-behavior-22k
    qwen35-beam-dev10k-qwen35-interest-22k
```

`qwen35-22k-16gpu-20260810-v1` 是缓存竞争诊断运行，在训练前失败，没有有效
checkpoint。

### 5.4 本地小型产物

```text
artifacts/sft/qwen35-9b-22k-summaries/
artifacts/qwen35-22k-preflight/
artifacts/beam/qwen35-9b-behavior-22k/
artifacts/beam/qwen35-9b-interest-22k/
docs/experiments/2026-08-11/qwen35_22k_results.csv
docs/experiments/2026-08-11/qwen35_22k_macro_class_metrics.{csv,json}
docs/experiments/2026-08-11/qwen35_22k_per_class_metrics.csv
docs/experiments/2026-08-11/qwen35_22k_macro_class_random_baselines.json
docs/experiments/2026-08-11/model_comparison_recall_ndcg_macro10.{csv,json}
docs/experiments/2026-08-11/qwen35_22k_experiment_report.md
```

本地目录不包含 `adapter_model.safetensors`、基础模型权重或训练 checkpoint。
