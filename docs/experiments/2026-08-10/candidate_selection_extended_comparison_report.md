# K=10 候选选择扩展模型与 Reasoning 对比报告

## 1. 研究问题（Research Questions）

本实验在同一份 K=10 新兴趣候选数据上回答以下问题：

1. Gemini 3.6 Flash 与 Llama-3.1-8B-Instruct 的候选选择能力如何？
2. Qwen3-8B、Qwen3.5-9B 使用 greedy 与 non-thinking sampling 时是否有差异？
3. GPT-5 与 GPT-5.6 Sol 在 `reasoning_effort=none` 和 `medium` 下如何变化？
4. Behavior-only 标题历史与 Interest-only 主题历史分别适合哪些模型？
5. Micro Accuracy 与对 101 个实际目标类别等权计算的 Macro Accuracy 是否给出不同结论？

每条记录包含一个正确主题和九个负主题，因此随机选择基线为 10%。

## 2. 训练数据构造（Training Data Construction）

本轮是纯评估实验，没有新增训练或微调。所有模型均使用官方 post-trained /
instruct checkpoint 或远程部署：

| 模型 | 本轮训练记录 | Adapter | 评估输入 |
|---|---:|---|---|
| Gemini 3.6 Flash | 0 | 无 | Behavior、Interest |
| Llama-3.1-8B-Instruct | 0 | 无 | Behavior、Interest |
| Qwen3-8B | 0 | 无 | Behavior、Interest |
| Qwen3.5-9B | 0 | 无 | Behavior、Interest |
| GPT-5 | 0 | 无 | Behavior、Interest |
| GPT-5.6 Sol | 0 | 无 | Behavior、Interest |

测试数据固定来自：

```text
data/training/candidate_selection_k10/
```

构造规则没有变化：

- grouped Dev10K 展开为每种输入 12,164 条；
- 每条有一个正确答案和九个负样本；
- 负样本不属于该 impression 的任何正确主题；
- 负样本不出现在历史主题中；
- Behavior 与 Interest 共享相同 `selection_id`、候选、答案和正确位置；
- 正确位置在十个位置间近似严格均衡；
- 目标来自 101 个实际出现的正答案类别，候选词表共 110 类。

标准监督标签不是训练 response，而是确定性的 held-out 测试答案：

```text
New topic: <exact candidate category>
```

## 3. 测试设计（Testing）

### 3.1 测试模型与运行位置

| 模型 | checkpoint / deployment | 运行位置 |
|---|---|---|
| Gemini 3.6 Flash | `gemini-3.6-flash` | 本机 `copilot-api` OpenAI-compatible proxy |
| Llama-3.1-8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Feeds A100 80GB |
| Qwen3-8B | `Qwen/Qwen3-8B` | Feeds A100 80GB |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | Feeds A100 80GB |
| GPT-5 | `gpt-5_2025-08-07` | TRAPI `redmond/interactive` |
| GPT-5.6 Sol | `gpt-5.6-sol_2026-07-09` | TRAPI `redmond/interactive` |

主要推理设置：

| 条件 | 参数 |
|---|---|
| Gemini | `reasoning_effort=minimal`, T=0.7, top-p=0.8, max tokens=128 |
| Llama | non-thinking sampling, T=0.7, top-p=0.8, top-k=20 |
| Qwen greedy | `enable_thinking=false`, `do_sample=false` |
| Qwen3 sampling | `enable_thinking=false`, T=0.7, top-p=0.8, top-k=20 |
| Qwen3.5 official non-thinking | T=0.7, top-p=0.8, top-k=20, presence penalty=1.5 |
| GPT none | `reasoning_effort=none`, max completion tokens=512 |
| GPT medium | `reasoning_effort=medium`，复用已验证的完整结果 |

Gemini 首轮 pilot 使用默认 reasoning 时，可见答案被 reasoning 挤占；改为
`minimal` 后，32 条 Behavior 和 Interest pilot 均达到 100% Candidate-valid 与
Prefix Compliance，随后才运行全量。

Qwen thinking smoke 发现平均约 1,400+ generated tokens，并有样本达到 2,048-token
上限仍无最终答案。用户随后要求取消 thinking 测试；对应 32 个 AMLT full-shard jobs
已全部释放，因此 thinking 结果不进入本报告。

### 3.2 测试数据来源与完整性

```text
Behavior:
data/training/candidate_selection_k10/behavior_only/dev_10k_expanded.parquet

Interest:
data/training/candidate_selection_k10/interest_only/dev_10k_expanded.parquet
```

每个正式条件均验证：

- 12,164/12,164 条预测；
- 12,164 个唯一 `selection_id`；
- Behavior 与 Interest 数据身份一致；
- 远程请求失败数为 0；
- TRAPI `none` 条件总 reasoning tokens 为 0；
- 输出使用同一个确定性 parser 重新计算指标。

### 3.3 真实测试样例

#### 样例 A：短历史、模型结果分化

```text
selection_id: dev:4220:19
answer: New topic: Celebrity & Creator Culture
```

Behavior 原始 Prompt：

```text
The headlines of the news items I recently viewed are listed below:

1. Bahareh Zare Bahari: Beauty queen says she will be killed if she is deported to Iran

Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should represent a new topic that is not reflected in my viewing history.

Candidates:
1. Infrastructure & Transportation Policy
2. Policing & Corrections
3. Celebrity & Creator Culture
4. Fitness & Exercise
5. Destinations & Trip Ideas
6. Television & Streaming
7. Population & Demographic Change
8. Government & Public Policy
9. Motorsports
10. Restaurants & Dining

Respond with exactly "New topic: <category>", replacing <category> with one exact category name from the candidate list. Do not include any other text.
```

中文翻译：

```text
我最近查看的新闻标题如下：

1. Bahareh Zare Bahari：选美冠军称，如果被遣返回伊朗，她将被杀害

请从下面候选中选择一个我下一步最可能感兴趣的主题。所选类别应是浏览历史中尚未
体现的新主题。候选依次为：基础设施与交通政策、警务与惩教、名人与创作者文化、
健身与锻炼、目的地与旅行灵感、电视与流媒体、人口与人口结构变化、政府与公共
政策、赛车运动、餐厅与餐饮。

只输出 `New topic: <category>`，其中类别名必须与候选完全一致，不要输出其他内容。
```

| 条件 | 实际输出 | 正确 |
|---|---|---|
| Gemini Behavior | `New topic: Government & Public Policy` | 否 |
| Llama Behavior | `New topic: Destinations & Trip Ideas` | 否 |
| GPT-5 none Behavior | `New topic: Celebrity & Creator Culture` | 是 |
| GPT-5.6 none Behavior | `New topic: Celebrity & Creator Culture` | 是 |
| Gemini Interest | `New topic: Population & Demographic Change` | 否 |
| Llama Interest | `New topic: Motorsports` | 否 |

所有输出格式都可确定解析；错误来自类别选择，而不是格式 judge。

#### 样例 B：多个模型共同命中

```text
selection_id: dev:1027:93
answer: New topic: Shopping & Gift Ideas
```

Interest 原始 Prompt：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Crime & Public Safety
2. Weather Forecasts & Seasonal Conditions
3. Journalism & News Media
4. Civil Rights & Social Justice
5. Severe Storms
6. Cooking & Recipes
7. Cooking & Recipes
8. Celebrity & Creator Culture
9. Celebrity & Creator Culture
10. Crime & Public Safety
11. Celebrity & Creator Culture
12. Cooking & Recipes
13. Cooking & Recipes
14. American Football
15. Cooking & Recipes
16. Cooking & Recipes
17. Cooking & Recipes
18. Cooking & Recipes
19. Celebrity & Creator Culture
20. Television & Streaming

Candidates:
1. Civic Life & Public Commemorations
2. Electric & Alternative Vehicles
3. Shopping & Gift Ideas
4. Home Appliances & Technology
5. Faith Communities & Practice
6. Cars & Driving
7. Household Cleaning & Laundry
8. Golf
9. Religion in Public Life
10. Elections & Political Campaigns
```

中文翻译：历史主题依次涉及犯罪与公共安全、天气、新闻媒体、公民权利、强风暴、
烹饪、名人文化、美式足球和电视流媒体；模型需要从十个未出现候选中预测下一兴趣。
标准答案是“购物与礼物灵感”。

Gemini、Llama 和 GPT-5.6 Sol 在 Interest 条件下均输出：

```text
New topic: Shopping & Gift Ideas
```

### 3.4 Judge 与指标程序

本实验不使用 LLM Judge，因此不存在 judge system prompt 或主观 rubric。唯一 judge
是确定性解析代码：

```text
src/new_interest/evaluation/candidate_metrics.py
```

解析依次移除：

- 可选 `New topic:`；
- `7.`、`B.`、`Option B:` 等编号；
- 外层引号；
- 末尾句号。

之后必须与一个候选类别完全一致。指标定义：

```text
Micro Accuracy =
  全部正确预测数 / 12,164

Macro Accuracy =
  mean(
    某正答案类别的正确数 / 该类别样本数
    for 测试中有正样本的 101 个类别
  )
```

Macro 不把九个仅存在于候选词表、但从未成为正答案的类别按零分计入。Strict Format
要求原字符串严格等于 `New topic: <answer_category>`；Candidate-valid 检查归一化
类别是否属于该条的十个候选。

## 4. 实验结果（Results）

Micro 的区间为 Wilson 95% CI。Macro 是 101 个类别 accuracy 的等权平均。

| 模型 | 输入 | Reasoning | Decoding | Correct | Micro Accuracy (95% CI) | Macro | Strict | Candidate-valid |
|---|---|---|---|---:|---:|---:|---:|---:|
| GPT-5 | Behavior | medium | single generation | 1,395/12,164 | 11.47% [10.91%, 12.05%] | 13.63% | 11.47% | 100.00% |
| GPT-5 | Behavior | none | single generation | 815/12,164 | 6.70% [6.27%, 7.16%] | 7.13% | 6.70% | 99.86% |
| GPT-5.6 Sol | Behavior | medium | single generation | 1,903/12,164 | 15.64% [15.01%, 16.30%] | 14.98% | 15.64% | 99.80% |
| GPT-5.6 Sol | Behavior | none | single generation | 2,138/12,164 | 17.58% [16.91%, 18.26%] | 15.91% | 17.58% | 100.00% |
| Gemini 3.6 Flash | Behavior | minimal | T=0.7, top-p=0.8 | 2,332/12,164 | **19.17% [18.48%, 19.88%]** | **16.04%** | 19.17% | 100.00% |
| Llama-3.1-8B | Behavior | none | T=0.7, top-p=0.8, top-k=20 | 1,440/12,164 | 11.84% [11.28%, 12.42%] | 12.33% | 10.87% | 99.97% |
| Qwen3-8B | Behavior | disabled | greedy | 1,834/12,164 | 15.08% [14.45%, 15.72%] | 15.81% | 11.26% | 99.90% |
| Qwen3-8B | Behavior | disabled | sampling | 1,818/12,164 | 14.95% [14.32%, 15.59%] | 14.53% | 11.32% | 99.90% |
| Qwen3.5-9B | Behavior | disabled | greedy | 2,098/12,164 | 17.25% [16.59%, 17.93%] | 13.45% | 17.25% | 99.99% |
| Qwen3.5-9B | Behavior | disabled | official non-thinking sampling | 1,926/12,164 | 15.83% [15.20%, 16.49%] | 12.27% | 15.83% | 98.80% |
| GPT-5 | Interest | medium | single generation | 2,306/12,164 | 18.96% [18.27%, 19.66%] | 16.18% | 18.96% | 99.99% |
| GPT-5 | Interest | none | single generation | 2,284/12,164 | 18.78% [18.09%, 19.48%] | 16.62% | 18.78% | 98.27% |
| GPT-5.6 Sol | Interest | medium | single generation | 2,440/12,164 | 20.06% [19.36%, 20.78%] | 17.16% | 20.06% | 99.88% |
| GPT-5.6 Sol | Interest | none | single generation | 2,558/12,164 | **21.03% [20.31%, 21.76%]** | **18.04%** | 21.03% | 100.00% |
| Gemini 3.6 Flash | Interest | minimal | T=0.7, top-p=0.8 | 2,381/12,164 | 19.57% [18.88%, 20.29%] | 16.11% | 19.57% | 99.97% |
| Llama-3.1-8B | Interest | none | T=0.7, top-p=0.8, top-k=20 | 1,676/12,164 | 13.78% [13.18%, 14.40%] | 12.66% | 13.11% | 98.36% |
| Qwen3-8B | Interest | disabled | greedy | 1,012/12,164 | 8.32% [7.84%, 8.82%] | 10.55% | 8.32% | 99.98% |
| Qwen3-8B | Interest | disabled | sampling | 1,001/12,164 | 8.23% [7.75%, 8.73%] | 10.58% | 8.23% | 99.98% |
| Qwen3.5-9B | Interest | disabled | greedy | 1,507/12,164 | 12.39% [11.82%, 12.99%] | 11.58% | 12.39% | 99.98% |
| Qwen3.5-9B | Interest | disabled | official non-thinking sampling | 1,468/12,164 | 12.07% [11.50%, 12.66%] | 11.24% | 12.07% | 98.50% |

主要观测：

- **Behavior 最佳**是 Gemini 3.6 Flash：Micro 19.17%，Macro 16.04%。
- **Interest 最佳**是 GPT-5.6 Sol reasoning none：Micro 21.03%，Macro 18.04%。
- GPT-5 Behavior 对 reasoning 高度敏感：medium 到 none 从 11.47% 降至 6.70%；
  Interest 则基本不变，18.96% 对 18.78%。
- GPT-5.6 Sol 关闭 reasoning 后反而提高：Behavior +1.93pp，Interest +0.97pp；
  同时 reasoning tokens 从 medium 条件的非零值变为严格 0。
- Gemini 的 Behavior 与 Interest 相近，Interest 高 0.40pp；两路 Macro 几乎相同。
- Llama 使用 Interest 比 Behavior 高 1.94pp，但整体低于 Gemini、GPT-5.6 和最佳
  Qwen Behavior。
- Qwen3 sampling 与 greedy 基本相同，无法修复 Interest 低于随机的问题。
- Qwen3.5 official non-thinking sampling 低于 greedy：Behavior -1.42pp，
  Interest -0.32pp；该差异同时包含 sampling 与 presence penalty 的影响。
- Micro 与 Macro 排名不完全一致。例如 Qwen3 Behavior greedy 的 Micro 为
  15.08%，Macro 为 15.81%；Qwen3.5 Behavior greedy 的 Micro 更高（17.25%），
  但 Macro 仅 13.45%，说明其收益更集中于高支持类别。

完整机器可读表见：

```text
artifacts/candidate-selection-k10/extended-comparison/comparison.csv
artifacts/candidate-selection-k10/extended-comparison/comparison.json
```

## 5. 数据与产物路径（Artifact Paths）

### 数据与规则

- `data/training/candidate_selection_k10/behavior_only/dev_10k_expanded.parquet`
- `data/training/candidate_selection_k10/interest_only/dev_10k_expanded.parquet`
- `data/training/candidate_selection_k10/metadata.json`
- `data/training/candidate_selection_k10/statistics.json`
- `src/new_interest/evaluation/candidate_metrics.py`

### 配置与评估代码

- `configs/candidate/extended_comparison_manifest.json`
- `configs/candidate/gemini36_flash_behavior.json`
- `configs/candidate/gemini36_flash_interest.json`
- `configs/candidate/llama31_sampling_behavior.json`
- `configs/candidate/llama31_sampling_interest.json`
- `configs/candidate/gpt5_none_behavior.json`
- `configs/candidate/gpt5_none_interest.json`
- `configs/candidate/gpt56_none_behavior.json`
- `configs/candidate/gpt56_none_interest.json`
- `src/new_interest/evaluation/evaluate_candidate_openai.py`
- `src/new_interest/evaluation/evaluate_candidate_trapi.py`
- `src/new_interest/evaluation/evaluate_candidate_qwen.py`
- `src/new_interest/experiments/report_candidate_results.py`

### Gemini

- `artifacts/candidate-selection-k10/gemini36_flash_behavior/`
- `artifacts/candidate-selection-k10/gemini36_flash_interest/`
- Proxy：`http://127.0.0.1:4141/v1`
- Model：`gemini-3.6-flash`

### Llama

- AMLT experiment：`candidate-llama-sampling-v1`
- Jobs：`llama31-sampling-behavior`、`llama31-sampling-interest`
- `artifacts/candidate-selection-k10/amlt/candidate-llama-sampling-v1/`

### Qwen

- Qwen3 greedy：`artifacts/candidate-selection-k10/amlt/candidate-k10-qwen-base-v2/`
- Qwen3 sampling：
  `artifacts/candidate-selection-k10/amlt/candidate-qwen3-sampling-behavior-v2/`
  与 `candidate-qwen-sampling-v1/`
- Qwen3.5 greedy：`artifacts/candidate-selection-k10/amlt/candidate-qwen35-base-v1/`
- Qwen3.5 official non-thinking：
  `artifacts/candidate-selection-k10/amlt/candidate-qwen35-nonthinking-v1/`
- 已取消的 thinking experiment：`candidate-qwen-thinking-shards-v1`
  （32/32 jobs killed，不纳入指标）

### TRAPI

- `artifacts/candidate-selection-k10/gpt5_behavior/full/`
- `artifacts/candidate-selection-k10/gpt5_interest/full/`
- `artifacts/candidate-selection-k10/gpt5_none_behavior/`
- `artifacts/candidate-selection-k10/gpt5_none_interest/`
- `artifacts/candidate-selection-k10/gpt56_behavior/full/`
- `artifacts/candidate-selection-k10/gpt56_interest/full/`
- `artifacts/candidate-selection-k10/gpt56_none_behavior/`
- `artifacts/candidate-selection-k10/gpt56_none_interest/`
- TRAPI instance：`redmond/interactive`
- API version：`2024-10-21`

### 最终汇总

- `artifacts/candidate-selection-k10/extended-comparison/comparison.csv`
- `artifacts/candidate-selection-k10/extended-comparison/comparison.json`
- `artifacts/candidate-selection-k10/extended-comparison/report.md`
