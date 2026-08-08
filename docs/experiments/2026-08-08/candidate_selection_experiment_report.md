# K=10 候选主题选择实验报告

## 1. 研究问题（Research Questions）

本实验比较在相同 K=10 候选集合下，不同模型和历史表示的单次选择能力：

1. Qwen3-8B Base 与 10K SFT adapter 哪个更适合候选选择？
2. Behavior-only 标题历史与 Interest-only 主题历史哪个效果更好？
3. GPT-5.6 Sol、GPT-5、Qwen3-8B 与 Qwen3.5-9B 的候选准确率有何差异？
4. 开放式新主题 SFT 是否能迁移到显式候选选择任务？

每条记录有 10 个候选和唯一正确答案，随机基线为 10%。

## 2. 训练数据构造（Training Data Construction）

候选测试不新增训练数据，使用已经完成训练的 Qwen3-8B 10K LoRA adapter。

| SFT 条件 | 训练数据 | 训练数 | 主题数 |
|---|---|---:|---:|
| Behavior-only | 最近 50 条新闻标题 | 10,000 | 110 |
| Interest-only | 最近 50 条二级主题 | 10,000 | 110 |

训练目标为开放式生成：

```text
New topic: <category>
```

训练时没有候选列表。该设计用于测试开放式 SFT 能否迁移到候选选择。

候选测试数据来自固定 grouped Dev10K：

```text
data/training/candidate_selection_k10/
```

多答案 impression 按每个正确主题展开，最终每种输入有 12,164 条。

每条数据：

- 1 个正确主题；
- 9 个负主题；
- 负主题不是任何正确答案；
- 负主题未出现在历史中；
- 正确位置在 10 个位置间均衡随机分布；
- Behavior 与 Interest 共享完全相同的候选、答案和位置。

标准答案：

```text
New topic: <exact candidate category>
```

详细数据样例见：

```text
docs/candidate_selection_data_examples.md
```

## 3. 测试设计（Testing）

### 3.1 模型与运行位置

| 条件 | 模型/Adapter | 运行位置 |
|---|---|---|
| Qwen Base | `Qwen/Qwen3-8B` | Feeds A100 80GB |
| Qwen3.5 | `Qwen/Qwen3.5-9B` | Feeds A100 80GB |
| Qwen SFT Behavior | Behavior 10K LoRA | Feeds A100 80GB |
| Qwen SFT Interest | Interest 10K LoRA | Feeds A100 80GB |
| GPT-5.6 Sol | `gpt-5.6-sol_2026-07-09` | TRAPI `redmond/interactive` |
| GPT-5 | `gpt-5_2025-08-07` | TRAPI `redmond/interactive` |

Qwen thinking 被显式关闭。Qwen3.5 使用官方 post-trained
`Qwen/Qwen3.5-9B`，不是额外 SFT adapter。测试均为普通单次生成，不使用动态
token trie、候选约束或多次采样。

### 3.2 推理参数

#### Qwen

```text
do_sample: false
max_new_tokens: 32
enable_thinking: false
```

为检验 greedy decoding 是否导致 Qwen 表现偏低，另对 Qwen3-8B 和
Qwen3.5-9B 完成一轮全量 sampling 重试。两代模型统一使用相同运行环境、
Transformers 5.14.1、batch size 和随机种子：

```text
do_sample: true
temperature: 0.7
top_p: 0.8
top_k: 20
seed: 20260808
max_new_tokens: 32
enable_thinking: false
Behavior batch size: 4
Interest batch size: 16
```

随机种子按 batch 的全局起始记录编号确定，因此 shard 恢复不会改变后续样本的
采样结果。

#### GPT-5.6 Sol

```text
reasoning_effort: medium
max_completion_tokens: 512
concurrency: 4
```

#### GPT-5

```text
reasoning_effort: medium
max_completion_tokens: 4096
concurrency: 4
```

GPT-5 在 512 和 1,024 token 预算下大量 reasoning token 挤占可见输出，因此最终使用
4,096。最终全量条件均无请求失败。

### 3.3 输出解析

主指标 Selection Accuracy（文件中字段名为 `category_accuracy`）先进行无损格式归一化：

- 移除可选 `New topic: `；
- 移除候选编号或字母，例如 `7.`、`Option B:`；
- 移除包裹引号和末尾句号；
- 归一化后必须与唯一正确类别完全一致。

因此以下输出会被判为正确：

```text
New topic: 7. Music
Option B: "Music".
```

Strict Format Accuracy 仍要求：

```text
prediction.strip() == "New topic: <answer_category>"
```

同时记录：

- Selection Accuracy：忽略无害格式后类别是否正确；
- Strict Format Accuracy：完整字符串是否严格符合要求；
- Candidate-valid Rate：输出类别是否属于该条候选；
- Prefix Compliance：是否使用 `New topic: `；
- Empty Output Rate；
- 多候选输出率。

本实验不使用 LLM Judge，所有指标由确定性代码计算：

```text
src/new_interest/candidate_metrics.py
```

### 3.4 真实测试样例

```text
selection_id: dev:4220:19
answer: New topic: Celebrity & Creator Culture
```

Behavior-only Prompt：

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

中文翻译：用户最近看过一条关于可能被遣返回伊朗的选美冠军新闻，需要从 10 个候选
中选择一个新的后续兴趣，标准答案为“名人与创作者文化”。

十条件实际输出：

| 条件 | 输出 | 正确 |
|---|---|---|
| Qwen Base Behavior | `New topic: Celebrity & Creator Culture` | 是 |
| Qwen Base Interest | `New topic: Infrastructure & Transportation Policy` | 否 |
| Qwen3.5 Behavior | `New topic: Government & Public Policy` | 否 |
| Qwen3.5 Interest | `New topic: Government & Public Policy` | 否 |
| Qwen SFT Behavior | `New topic: Population & Demographic Change` | 否 |
| Qwen SFT Interest | `New topic: Government & Public Policy` | 否 |
| GPT-5.6 Behavior | `New topic: Celebrity & Creator Culture` | 是 |
| GPT-5.6 Interest | `New topic: Government & Public Policy` | 否 |
| GPT-5 Behavior | `New topic: Population & Demographic Change` | 否 |
| GPT-5 Interest | `New topic: Government & Public Policy` | 否 |

## 4. 实验结果（Results）

每个条件均完成 12,164/12,164 条预测，请求失败数为 0。置信区间为二项比例
Wilson 95% CI。

| 模型 | 输入 | Correct | Selection Accuracy (95% CI) | Strict Format | Candidate-valid | Prefix |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | Behavior | 1,903/12,164 | 15.64% [15.01%, 16.30%] | 15.64% | 99.80% | 99.80% |
| GPT-5.6 Sol | Interest | **2,440/12,164** | **20.06% [19.36%, 20.78%]** | **20.06%** | 99.88% | 99.88% |
| GPT-5 | Behavior | 1,395/12,164 | 11.47% [10.91%, 12.05%] | 11.47% | 100.00% | 100.00% |
| GPT-5 | Interest | 2,306/12,164 | 18.96% [18.27%, 19.66%] | 18.96% | 99.99% | 99.99% |
| Qwen3.5-9B | Behavior | **2,098/12,164** | **17.25% [16.59%, 17.93%]** | **17.25%** | 99.99% | 100.00% |
| Qwen3.5-9B | Interest | 1,507/12,164 | 12.39% [11.82%, 12.99%] | 12.39% | 99.98% | 100.00% |
| Qwen3-8B Base | Behavior | 1,834/12,164 | 15.08% [14.45%, 15.72%] | 11.26% | 99.90% | 100.00% |
| Qwen3-8B Base | Interest | 1,012/12,164 | 8.32% [7.84%, 8.82%] | 8.32% | 99.98% | 100.00% |
| Qwen3-8B 10K SFT | Behavior | 808/12,164 | 6.64% [6.21%, 7.10%] | 6.64% | 98.96% | 100.00% |
| Qwen3-8B 10K SFT | Interest | 431/12,164 | 3.54% [3.23%, 3.89%] | 3.54% | 33.21% | 100.00% |

结果显示：

- GPT-5.6 Interest 表现最高，Selection Accuracy 为 20.06%，约为随机基线的 2 倍；
- Qwen3.5 Behavior 为 17.25%，比 Qwen3-8B Base Behavior 高 2.17 个百分点，
  也是所有 Behavior-only 条件中的最高结果；
- Qwen3.5 Interest 为 12.39%，比 Qwen3-8B Base Interest 高 4.07 个百分点，
  从低于随机提升为显著高于 10% 随机基线；
- GPT 模型使用 Interest-only 时均优于 Behavior-only；
- Qwen3-8B Base 与 Qwen3.5-9B 均使用 Behavior-only 时优于 Interest-only；
- 两个 Qwen SFT adapter 在候选任务上均低于对应 Base；
- Qwen Base Behavior 有 19.3% 输出携带候选编号；格式归一化后 Candidate-valid
  从 80.58% 修正为 99.90%，Selection Accuracy 从 11.26% 修正为 15.08%；
- Behavior SFT 的格式合法率较高，但 Selection Accuracy 下降到 6.64%；
- Interest SFT 仍倾向生成开放式新主题，只有 33.21% 输出属于给定候选。

这些结果描述的是任务迁移现象：SFT 针对开放式生成优化，并未使用候选选择数据训练。

Qwen3.5 两路输出均未出现 `<think>` 内容。Behavior 与 Interest 的
Candidate-valid 分别为 99.99% 和 99.98%，严格格式与 Selection Accuracy 完全一致。
模型仍有位置偏好，尤其倾向前两个候选：

| 输入 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 | Pos 9 | Pos 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Behavior | 15.00% | 14.67% | 10.27% | 9.60% | 8.03% | 7.91% | 7.76% | 7.48% | 8.02% | 11.25% |
| Interest | 16.01% | 15.02% | 12.90% | 12.50% | 8.95% | 7.30% | 7.33% | 6.43% | 5.75% | 7.78% |

### 4.1 Sampling 重试

四个 sampling 条件均完成 12,164/12,164 条预测。Selection Accuracy 仍使用相同的
确定性格式归一化规则。

| 模型 | 输入 | Decoding | Correct | Selection Accuracy (95% CI) | Greedy | 差值 | Candidate-valid |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | Behavior | Sampling | 1,818/12,164 | 14.95% [14.32%, 15.59%] | 15.08% | -0.13pp | 99.90% |
| Qwen3-8B | Interest | Sampling | 1,001/12,164 | 8.23% [7.75%, 8.73%] | 8.32% | -0.09pp | 99.98% |
| Qwen3.5-9B | Behavior | Sampling | 1,962/12,164 | 16.13% [15.49%, 16.79%] | 17.25% | -1.12pp | 99.97% |
| Qwen3.5-9B | Interest | Sampling | 1,481/12,164 | 12.18% [11.61%, 12.77%] | 12.39% | -0.21pp | 99.88% |

Sampling 没有改善任何一个条件：

- Qwen3-8B 的两路变化都小于 0.14 个百分点，结果与 greedy 基本相同；
- Qwen3.5-9B Behavior 下降 1.12 个百分点，Interest 下降 0.21 个百分点；
- Qwen3-8B Interest 在 sampling 下仍为 8.23%，因此低于随机的问题不能归因于
  greedy decoding；
- Qwen3-8B 的位置偏好也基本不变。Behavior 仍有 23.37% 输出选择第 10 项，
  Interest 仍有 24.81% 输出选择第 1 项；
- 四个 sampling 条件均无 `<think>` 输出，Candidate-valid 均高于 99.8%。

Qwen3-8B Behavior sampling 的 Strict Format Accuracy 为 11.32%，低于 14.95%
Selection Accuracy，原因仍是模型偶尔在类别前添加候选编号；这些输出经过格式归一化
后仍可确定地映射到唯一候选。

Qwen3-8B Base Interest 低于随机的专项诊断见：

```text
docs/experiments/2026-08-08/qwen_base_interest_bias_diagnosis.md
```

## 5. 数据与产物路径（Artifact Paths）

### 数据

- [candidate data](../../../data/training/candidate_selection_k10)
- [candidate metadata](../../../data/training/candidate_selection_k10/metadata.json)
- [candidate statistics](../../../data/training/candidate_selection_k10/statistics.json)
- [candidate examples](candidate_selection_data_examples.md)

### 代码与配置

- [candidate builder](../../../src/new_interest/build_candidate_selection.py)
- [candidate metrics](../../../src/new_interest/candidate_metrics.py)
- [Qwen evaluator](../../../src/new_interest/evaluate_candidate_qwen.py)
- [TRAPI evaluator](../../../src/new_interest/evaluate_candidate_trapi.py)
- [candidate configs](../../../configs/candidate)

### 结果

- [final comparison CSV](../../../artifacts/candidate-selection-k10/comparison/comparison.csv)
- [final comparison JSON](../../../artifacts/candidate-selection-k10/comparison/comparison.json)
- [generated report](../../../artifacts/candidate-selection-k10/comparison/report.md)
- [Qwen Base AMLT results](../../../artifacts/candidate-selection-k10/amlt/candidate-k10-qwen-base-v2)
- [Qwen3.5 AMLT results](../../../artifacts/candidate-selection-k10/amlt/candidate-qwen35-base-v1)
- [Qwen sampling AMLT results](../../../artifacts/candidate-selection-k10/amlt/candidate-qwen-sampling-v1)
- [Qwen3 Behavior sampling retry](../../../artifacts/candidate-selection-k10/amlt/candidate-qwen3-sampling-behavior-v2)
- [Qwen SFT results](../../../artifacts/candidate-selection-k10/qwen_sft_behavior)
- [GPT-5.6 Behavior](../../../artifacts/candidate-selection-k10/gpt56_behavior/full)
- [GPT-5.6 Interest](../../../artifacts/candidate-selection-k10/gpt56_interest/full)
- [GPT-5 Behavior](../../../artifacts/candidate-selection-k10/gpt5_behavior/full)
- [GPT-5 Interest](../../../artifacts/candidate-selection-k10/gpt5_interest/full)

Qwen3.5 远程 AMLT 记录：

- Experiment：`candidate-qwen35-base-v1`
- Jobs：`qwen35-base-behavior`、`qwen35-base-interest`
- Config：`configs/amlt/candidate_qwen35_base.yaml`
- Behavior metrics：`artifacts/candidate-selection-k10/amlt/candidate-qwen35-base-v1/qwen35-base-behavior/evaluation/metrics.json`
- Interest metrics：`artifacts/candidate-selection-k10/amlt/candidate-qwen35-base-v1/qwen35-base-interest/evaluation/metrics.json`

Sampling 远程 AMLT 记录：

- 主 Experiment：`candidate-qwen-sampling-v1`
- 主 Jobs：`qwen3-sampling-interest`、`qwen35-sampling-behavior`、
  `qwen35-sampling-interest`
- Qwen3 Behavior 首次 Job：`qwen3-sampling-behavior`，节点异常卡住后取消
- Qwen3 Behavior 重试 Experiment：`candidate-qwen3-sampling-behavior-v2`
- Qwen3 Behavior 重试 Job：`qwen3-sampling-behavior`
- Config：`configs/amlt/candidate_qwen_sampling.yaml`
- Qwen3 Behavior metrics：`artifacts/candidate-selection-k10/amlt/candidate-qwen3-sampling-behavior-v2/qwen3-sampling-behavior/evaluation/metrics.json`
- Qwen3 Interest metrics：`artifacts/candidate-selection-k10/amlt/candidate-qwen-sampling-v1/qwen3-sampling-interest/evaluation/metrics.json`
- Qwen3.5 Behavior metrics：`artifacts/candidate-selection-k10/amlt/candidate-qwen-sampling-v1/qwen35-sampling-behavior/evaluation/metrics.json`
- Qwen3.5 Interest metrics：`artifacts/candidate-selection-k10/amlt/candidate-qwen-sampling-v1/qwen35-sampling-interest/evaluation/metrics.json`
