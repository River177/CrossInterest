# Qwen3-8B SFT22K vs GRPO1K：CoT 条件标签排序评测

> **状态：已完成。**冻结 Dev10K 上的 16 个 AMLT shard jobs 全部通过；四个条件均聚合为
> 10,000 条结果，每条 repaired ranking 包含 110 个唯一主题，所有分数均为有限值。

## 1. 研究问题（Research Questions）

本实验回答：

1. 从 Qwen3-8B SFT22K 继续进行独立 GRPO1K 后，Behavior-only 与
   Interest-only 的新兴趣排名是否改善？
2. GRPO1K 的收益来自更好的主题排序，还是主要来自更高的严格 CoT 格式合规率？
3. 在统一的 “greedy CoT + 110 个合法主题条件概率打分” 口径下，
   Recall/NDCG@10、@20、@50 如何变化？

比较条件：

| 条件 | Base | 初始化 adapter | 后续训练 |
|---|---|---|---|
| SFT22K | Qwen/Qwen3-8B | 无 | 22K completion-only LoRA SFT |
| GRPO1K | Qwen/Qwen3-8B | 对应 SFT22K adapter | 独立 1K GRPO |

Behavior-only 与 Interest-only 分开训练、分开评测。主指标为严格端到端
Recall/NDCG@10、@20、@50；诊断指标为修复 malformed CoT 后的同组指标。

## 2. 训练数据构造（Training Data Construction）

### 2.1 SFT22K

来源为重分类 MINDlarge 的严格新主题训练池。一个当前点击主题未出现在历史点击主题中时，
记为新主题。Train 每个 impression 使用一个固定目标；历史最多保留最近 50 条。

SFT 子集使用 seed `20260806` 的稳定 class-round-robin prefix：

```text
1K ⊂ 5K ⊂ 10K ⊂ 22K
```

| 条件 | 类别数 | 每类样本 | 总样本 | held-out Train |
|---|---:|---:|---:|---:|
| SFT22K Behavior | 110 | 200 | 22,000 | 0 |
| SFT22K Interest | 110 | 200 | 22,000 | 0 |

SFT 参数：3 epochs、BF16、LoRA rank/alpha `16/32`、dropout `0.05`、
effective batch 32、learning rate `2e-4`、cosine scheduler、seed 42。
训练关闭 Qwen thinking，只训练 assistant completion。

#### Behavior SFT 真实样例

来源：`source_sample_id=train:648369:15`，记录类型：SFT target。

英文原文：

```text
The headlines of the news items I recently viewed are listed below:

1. The News In Cartoons

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

目标：

```text
New topic: Labor & Employment
```

中文翻译：

```text
下面列出我最近浏览的新闻标题：

1. 漫画中的新闻

根据这些标题，预测一个我接下来可能感兴趣的新主题类别。预测类别应不同于浏览历史所反映的
主题。回答必须以 “New topic: ” 开头，之后只给出类别名称。
```

目标翻译：`新的主题：劳动与就业`。

#### Interest SFT 真实样例

来源：`source_sample_id=train:1965955:74`，记录类型：SFT target。

英文原文：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Golf

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

目标：

```text
New topic: Archaeology & Human Origins
```

中文翻译：

```text
下面按时间顺序列出我最近浏览新闻对应的主题类别：

1. 高尔夫

根据该主题序列，预测一个我接下来可能感兴趣的新主题类别。预测类别不得出现在上述序列中。
回答必须以 “New topic: ” 开头，之后只给出类别名称。
```

目标翻译：`新的主题：考古与人类起源`。

### 2.2 GRPO1K

GRPO1K 是 SFT 1K prefix 的格式转换，不重新采样；Behavior/Interest source IDs 完全配对。
每类 9–10 条，共 110 类、1,000 条。GRPO5K 是独立预备分支，本报告未训练或评测。

Prompt 追加的实际指令：

```text
Reason about the user's history using exactly one short sentence of at most 30 words inside <think> and </think>. Do not compare multiple candidate topics. You must close </think> before answering. Then, on a new line, provide the final answer in exactly this format:
New topic: <category name>

Output only the think block and the final answer. Do not include any other text.
```

中文含义：在 `<think>` 标签内用不超过 30 个英文词的一句话简短推理，必须闭合标签；
下一行严格输出 `New topic: <类别>`，不得追加其他文本。

GRPO 设置：1 epoch、4×A100 80GB、rollout `n=16`、temperature `1.0`、
top-p `0.95`、max response 768、batch 32、actor LR `1e-5`、KL coefficient
`0.01`、LoRA `16/32`。两个 GRPO1K 分支都从原始对应 SFT22K adapter 独立开始。

Reward：

```text
exact final answer: 0.8
strict non-empty <think>...</think> format: 0.2
maximum: 1.0
```

| 条件 | 类别数 | 每类样本 | 总样本 | 控制样本 |
|---|---:|---:|---:|---:|
| GRPO1K Behavior | 110 | 9–10 | 1,000 | 0 |
| GRPO1K Interest | 110 | 9–10 | 1,000 | 0 |

## 3. 测试设计（Testing）

### 3.1 测试模型与运行位置

四个条件均使用 Qwen3-8B base 加远端 LoRA adapter。推理在 Feeds Singularity
`80G1-A100` 上运行；每个条件分成 4 个互斥的 2,500-row shards，共 16 个 jobs。
所有 jobs 状态为 `pass`。adapter/checkpoint 未下载到本机；本机只保存必要的
metrics、prediction 与配置。

生成参数：

```text
enable_thinking: true
do_sample: false
max_new_tokens: 768
batch_size: 4
candidate_batch_size: 22
```

### 3.2 测试数据来源

冻结 grouped Dev10K：

```text
data/training/sft_experiments/eval/
├── behavior_only/dev_10k_grouped.parquet
└── interest_only/dev_10k_grouped.parquet
```

选择规则为 impression ID 上的 SHA-256 priority；两种输入包含相同 10,000 个
impressions 和相同 acceptable target sets。Dev 保留每个 impression 的全部正确新主题。
标签词表固定为 Train 保留的 110 个二级主题。

### 3.3 测试样例

#### 样例 A：SFT Behavior 格式失败，但修复后 Top-10 命中

`impression_id=346995`。

英文 Prompt：

```text
The headlines of the news items I recently viewed are listed below:

1. Celebs celebrate Halloween 2019

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
下面列出我最近浏览的新闻标题：

1. 明星庆祝 2019 年万圣节

根据这些标题，预测一个我接下来可能感兴趣的新主题类别。预测类别应不同于浏览历史所反映的
主题。回答必须以 “New topic: ” 开头，之后只给出类别名称。
```

模型原始输出：

```text
<think>
Okay, the user has viewed a news item about Celebs celebrating Halloween 2019. That's fashion, culture, or personal style. I need to pick a different category. Beauty & Grooming is a good fit since it's related to personal style and celebrity culture. Let me make sure there's no overlap. Yeah, Halloween fashion could tie into beauty trends. So, Beauty & Grooming makes sense as the next topic.
New topic: Beauty & Grooming
```

中文翻译：模型认为万圣节明星新闻与时尚、美容相关，最终预测“美容与仪容”；但没有闭合
`</think>`。

正确主题：`Shopping & Gift Ideas`。严格格式为 false，因此 strict Recall@10=0；
修复未闭合标签后，该主题排名第 7，因此 repaired Recall@10=1、NDCG@10=0.3333。

#### 样例 B：GRPO Behavior 格式正确，Top-10 部分命中

`impression_id=301912`。

英文 Prompt：

```text
The headlines of the news items I recently viewed are listed below:

1. Bud Light Ads Honor Nats Fan

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
下面列出我最近浏览的新闻标题：

1. Bud Light 广告向华盛顿国民队球迷致敬

根据这些标题，预测一个我接下来可能感兴趣的新主题类别。预测类别应不同于浏览历史所反映的
主题。回答必须以 “New topic: ” 开头，之后只给出类别名称。
```

模型完整输出：

```text
<think>
Okay, the user's recent viewing history includes a headline about Bud Light Ads honoring a Nats fan. The Nats are the Washington Nationals, a baseball team. So the topic here is sports, specifically baseball. The user is likely interested in sports-related content. The task is to predict a new topic different from sports. Possible categories could be food, music, travel, etc. Since the user is a fan of a baseball team, maybe they also enjoy food, like beer (Bud Light is a beer). But the category should be different from sports. Food & Dining is a good fit here. Alternatively, Television & Streaming or Music. But considering the beer connection, Food & Dining makes sense. Let me check the guidelines again. The predicted category must be different from the current topics. Sports is the current topic, so Food & Dining is a valid choice. I'll go with Food & Dining.
</think>

New topic: Food & Dining
```

中文翻译：模型识别到标题与华盛顿国民队、棒球和 Bud Light 啤酒有关；由于要求预测不同于
当前体育主题的新类别，它选择了“食品与餐饮”。

完整输出含非空 `<think>...</think>`，严格格式为 true。正确主题集合为
`American Football` 与 `Pets & Animal Companions`；后者排名第 7，因此
Recall@10=0.5、NDCG@10=0.2044。greedy final answer 本身没有命中。

#### 样例 C：GRPO Interest greedy 与排名均命中

`impression_id=189360`。

英文 Prompt：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Economy & Economic Policy

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

中文翻译：

```text
下面按时间顺序列出我最近浏览新闻对应的主题类别：

1. 经济与经济政策

根据该主题序列，预测一个我接下来可能感兴趣的新主题类别。预测类别不得出现在上述序列中。
回答必须以 “New topic: ” 开头，之后只给出类别名称。
```

模型输出：

```text
<think>
Okay, the user has only viewed news items under Economy & Economic Policy. They might be interested in related areas like Public Health, since economic policies often impact healthcare systems and government spending. Public Health isn't in their history, so it's a good fit.
</think>

New topic: Public Health
```

中文翻译：模型判断经济政策经常影响医疗体系和政府支出，因此预测此前未出现的“公共卫生”。

唯一正确主题为 `Public Health`；greedy 命中，且条件概率排名第 1，因此
Recall/NDCG@10 均为 1。

### 3.4 确定性 judge 与指标

本实验没有使用 LLM-as-a-judge，也没有 judge system/user prompt 或 JSON schema。
Judge 是代码中的确定性规则。

#### (a) 严格格式规则

实际正则：

```regex
\A\s*<think>\s*(?P<think>\S[\s\S]*?)\s*</think>\s*(?P<answer>New topic: [^\r\n]+)\s*\Z
```

含义：

- `<think>` 内容必须非空；
- 必须闭合 `</think>`；
- 闭合后只能有一行 `New topic: <主题>`；
- 不能有尾随解释。

#### (b) 110-label 排名

先 greedy 生成一条 reasoning；固定该 reasoning 后，对 110 个 canonical topic 逐个计算：

```text
score(label) =
  mean log P(label tokens + EOS | user prompt, generated reasoning, "New topic:")
```

分数含 EOS，并按 label token 数量归一化。按分数降序得到完整 110-label ranking。

严格指标：

```text
strict ranking = ranking, if original output is strictly formatted
strict ranking = empty, otherwise
```

因此格式失败样本的 strict Recall/NDCG 全部为 0。

修复指标只用于诊断：确定性地闭合或重建 malformed think block，再进行相同 label
scoring；它不改变 strict format flag。

对每个 impression：

```text
Recall@K = |TopK ∩ acceptable targets| / |acceptable targets|
NDCG@K = DCG@K / IDCG@K
```

表中结果是 10,000 个 impression 的均值，不是单一二项比例，因此不提供 `k/n`
或二项置信区间。格式率与 greedy hit 则提供精确计数。

## 4. 实验结果（Results）

### 4.1 严格端到端主指标

| 条件 | 输入 | n | 格式合规 | Greedy strict Hit@1 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT22K | Behavior | 10,000 | 974/10,000 (9.74%) | 12/10,000 (0.12%) | 0.86% | 0.42% | 1.68% | 0.64% | 4.76% | 1.27% |
| GRPO1K | Behavior | 10,000 | 8,482/10,000 (84.82%) | 99/10,000 (0.99%) | 7.38% | 3.66% | 15.05% | 5.66% | 39.20% | 10.59% |
| SFT22K | Interest | 10,000 | 8,061/10,000 (80.61%) | 117/10,000 (1.17%) | 7.38% | 3.75% | 14.19% | 5.53% | 36.40% | 10.04% |
| GRPO1K | Interest | 10,000 | 9,957/10,000 (99.57%) | 141/10,000 (1.41%) | 9.71% | 5.01% | 19.23% | 7.49% | 47.15% | 13.17% |

严格指标中，GRPO1K 相对 SFT22K：

| 输入 | Δ Recall@10 | Δ NDCG@10 | Δ Recall@20 | Δ NDCG@20 | Δ Recall@50 | Δ NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| Behavior | +6.52 pp | +3.24 pp | +13.37 pp | +5.02 pp | +34.44 pp | +9.32 pp |
| Interest | +2.33 pp | +1.26 pp | +5.04 pp | +1.96 pp | +10.75 pp | +3.13 pp |

### 4.2 修复格式后的排序诊断

| 条件 | 输入 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT22K | Behavior | 8.16% | 4.04% | 16.50% | 6.22% | **46.02%** | **12.23%** |
| GRPO1K | Behavior | **8.28%** | **4.08%** | **16.74%** | **6.29%** | 44.46% | 11.95% |
| SFT22K | Interest | 9.28% | 4.74% | 17.47% | 6.88% | 45.27% | 12.53% |
| GRPO1K | Interest | **9.76%** | **5.04%** | **19.28%** | **7.52%** | **47.32%** | **13.23%** |

测量定义层面的解释：

- Behavior 的严格大幅提升主要来自格式率从 9.74% 提升至 84.82%；
- 修复格式后，Behavior@10/@20 仅小幅提升，而 @50 略降
  （Recall -1.55 pp，NDCG -0.28 pp）；
- Interest 在修复口径下 @10/@20/@50 均提升，说明除格式收益外还有小幅排序收益；
- greedy strict Hit@1 很低，因为训练 reward 使用单个固定 SFT target，而 Dev
  是多答案集合；本报告的主要任务仍是完整主题排名，不以 greedy 单答案替代 Recall/NDCG。

完整 CSV：
[qwen3_sft22k_grpo1k_cot_label_ranking_metrics.csv](qwen3_sft22k_grpo1k_cot_label_ranking_metrics.csv)。

完整性检查：

```text
AMLT jobs: 16/16 pass
rows per condition: 10,000
ranking length per row: 110
duplicate labels: 0
non-finite scores: 0
missing evaluation indices: 0
judge errors: 0 (deterministic parser/scorer)
```

基于该诊断，不自动启动 GRPO5K：Interest 显示一致但较小的排序提升，Behavior 的主要收益是
格式学习，且 repaired Recall/NDCG@50 略低。下一阶段应先明确目标是继续优化严格格式，还是优化
主题排序，再决定 GRPO5K 的 reward 与评测门槛。

## 5. 数据与产物路径（Artifact Paths）

### 本地训练与数据

- SFT Behavior 22K：
  `data/training/sft_experiments/train/behavior_only/22k.parquet`
- SFT Interest 22K：
  `data/training/sft_experiments/train/interest_only/22k.parquet`
- GRPO Behavior 1K：
  `data/training/grpo_experiments/behavior_only/1k.parquet`
- GRPO Interest 1K：
  `data/training/grpo_experiments/interest_only/1k.parquet`
- SFT manifest：
  `data/training/sft_experiments/metadata.json`
- GRPO manifest/statistics：
  `data/training/grpo_experiments/{metadata.json,statistics.json}`
- Dev10K：
  `data/training/sft_experiments/eval/{behavior_only,interest_only}/dev_10k_grouped.parquet`
- 110-label vocabulary：
  `data/training/generative_new_interest_balanced_200_h50/retained_interests.parquet`

### 代码与配置

- CoT dataset/prompt：
  `src/new_interest/data/build_grpo_dataset.py`
- GRPO reward：
  `src/new_interest/training/grpo_reward.py`
- CoT label-ranking evaluator：
  `src/new_interest/evaluation/evaluate_cot_label_ranking.py`
- shard aggregator：
  `src/new_interest/evaluation/aggregate_cot_label_results.py`
- AMLT config：
  `configs/amlt/qwen_cot_label_eval_map.yaml`
- map wrapper：
  `scripts/amlt/evaluation/run_qwen_cot_label_eval_map.sh`
- tests：
  `tests/test_cot_label_ranking.py`

### 远端 AMLT

- SFT22K adapters：
  `crossinterest-qwen-llama-22k-v1`
  - `qwen3-interest-22k`
  - `qwen3-behavior-22k`
- GRPO1K adapters：
  `qwen-grpo1k-partial-v1`
  - `grpo-1k-qwen3-interest-22k`
  - `grpo-1k-qwen3-behavior-22k`
- 完整 Dev10K 评测：
  `qwen-cot-label-dev10k-v1`
  - `cot-label-eval-s{0,1,2,3}-qwen3-{interest,behavior}-22k`
  - `cot-label-eval-s{0,1,2,3}-grpo-1k-qwen3-{interest,behavior}-22k`
- 每个评测 job 的远端输出：
  `cot_label_eval/{metrics.json,predictions.parquet,resolved_config.json,shards/}`

### 本地聚合结果

- 聚合 metrics/predictions：
  `/Users/river/.copilot/session-state/04516570-14ad-4f3f-8396-53b54986ebe2/files/cot-label-dev10k-aggregated/{sft22k,grpo1k}/{behavior_only,interest_only}/`
- 下载的远端 shard 结果：
  `/Users/river/.copilot/session-state/04516570-14ad-4f3f-8396-53b54986ebe2/files/cot-label-dev10k-v1/`
- 本报告：
  `/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-09/qwen3_sft22k_grpo1k_cot_label_ranking_report.md`
- 指标 CSV：
  `/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-09/qwen3_sft22k_grpo1k_cot_label_ranking_metrics.csv`
