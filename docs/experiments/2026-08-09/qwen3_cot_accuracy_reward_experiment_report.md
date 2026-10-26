# Qwen3-8B CoT Accuracy 与 Correctness-Gated Reward 实验

> **状态：在预注册 gate 处停止。**acceptable-target 数据、GPT-5.6 Sol teacher、
> 三种 CoT-SFT warm-up、统一 Accuracy evaluator 和 correctness-gated reward 均已完成；
> 但没有一个 CoT-SFT1K 条件同时满足格式、合法标签和 no-CoT retention 门槛，因此
> correctness GRPO1K/5K 未启动。

## 1. 研究问题（Research Questions）

1. CoT 单点预测能否用 Set Accuracy@1 替代不合适的 CoT Recall/NDCG？
2. old partial-reward GRPO1K 是否提升 no-CoT 直接预测 Accuracy？
3. GPT-5.6 Sol teacher CoT-SFT warm-up 能否同时达到：
   - strict CoT format；
   - canonical 110-label 输出；
   - 不损伤 no-CoT Accuracy？
4. correctness-gated reward 是否具有足够的 reward variance 进入 GRPO？

主指标：

```text
Set Accuracy@1 = predicted label ∈ acceptable targets
Strict Set Accuracy@1 = Set Accuracy@1 AND strict CoT format
```

## 2. 训练数据构造（Training Data Construction）

### 2.1 Acceptable-target manifest

从：

```text
data/processed/new_interest/train.parquet
```

按 `impression_id` 聚合 retained 110-label 中的全部正确新主题，再与嵌套
SFT 1K/5K source IDs join。

5K manifest：

| acceptable targets | impressions |
|---:|---:|
| 1 | 4,030 |
| 2 | 707 |
| 3 | 193 |
| 4 | 52 |
| 5 | 11 |
| 6 | 5 |
| 7 | 1 |
| 8 | 1 |

其中前 1K 有 184 条 multi-target impressions。Behavior/Interest 的 IDs、
acceptable sets 和顺序完全相同。

### 2.2 GPT-5.6 Sol teacher

已验证并调用：

```text
TRAPI instance: redmond/interactive
deployment: gpt-5.6-sol_2026-07-09
API version: 2024-10-21
```

20-call live smoke 首次受 429 concurrency 限制；并发从 8 降至 4、重试等待改为
5/10 秒后恢复至 20/20。完整 1K×2 输入最终为 2,000/2,000 success。

教师输入包含原始 Prompt 与 selected target，只返回：

```json
{"reasoning": "..."}
```

student 输入不暴露 target；构造器确定性生成：

```text
<think>
{teacher reasoning}
</think>
New topic: {selected target}
```

#### Behavior 真实 teacher 样例

目标：`Relationships & Dating`。

英文 reasoning：

```text
Several viewed headlines focus on celebrity romance and personal relationships, including Eva Mendes and Ryan Gosling, Miley Cyrus and Cody Simpson, Kristen Stewart’s proposal plans, celebrity couples with age gaps, and J.D. Scott’s wedding. This recurring attention to courtship, commitment, and couple dynamics suggests likely interest in the broader Relationships & Dating category.
```

中文翻译：

```text
多条浏览标题都集中在明星恋情和个人关系，包括 Eva Mendes 与 Ryan Gosling、
Miley Cyrus 与 Cody Simpson、Kristen Stewart 的求婚计划、年龄差明星伴侣以及
J.D. Scott 的婚礼。对恋爱、承诺和伴侣互动的持续关注，说明用户可能对更广义的
“关系与约会”主题感兴趣。
```

#### Interest 真实 teacher 样例

目标：`Genealogy & Family History`。

英文 reasoning：

```text
The combination of interest in Historical Events & Eras, Parenting & Family Life, and Faith Communities & Practice suggests curiosity about how family identity, traditions, and personal stories connect across generations, making genealogy and family-history content a plausible next interest.
```

中文翻译：

```text
对历史时期、育儿与家庭生活、信仰社群与实践的组合兴趣，表明用户可能好奇家庭身份、
传统和个人故事如何跨代连接，因此家谱与家庭史是合理的下一兴趣。
```

### 2.3 CoT-SFT variants

均从原始 SFT22K adapter 独立开始：

| variant | data | epochs | LR |
|---|---|---:|---:|
| pure-v1 | 1K teacher CoT | 2 | 5e-5 |
| pure-v2 | 1K teacher CoT | 3 | 2e-5 |
| rehearsal | 每 source 3×CoT + 1×direct，共 4K | 1 | 3e-5 |

Token audit：

| input | max tokens | over 2048 |
|---|---:|---:|
| Behavior CoT1K | 1,204 | 0 |
| Interest CoT1K | 592 | 0 |

## 3. 测试设计（Testing）

### 3.1 模型与位置

Base：`Qwen/Qwen3-8B`。SFT22K、old GRPO1K 和各 CoT-SFT adapter 均在 Feeds
A100 上远端加载；adapter/checkpoint 未下载本机。

### 3.2 冻结测试集

```text
data/training/sft_experiments/eval/
├── behavior_only/dev_10k_grouped.parquet
└── interest_only/dev_10k_grouped.parquet
```

C0/C1 no-CoT 完整 Dev10K 已运行；C1 faithful CoT Accuracy 直接复用此前同一
legacy Prompt 下已完成的 10K raw outputs。C2 variants 只运行 128-sample gate，
未通过时不扩大到 Dev10K。

### 3.3 测试 Prompt

no-CoT：

```text
enable_thinking=false
do_sample=false
New topic: <label>
```

新 CoT：

```text
Reason about the user's history inside <think> and </think>.
The reasoning must be non-empty, and you must close </think> before answering.
```

old GRPO1K 使用其训练时 legacy Prompt，避免 Prompt mismatch。

### 3.4 Judge 与 reward

没有 LLM judge。Accuracy、格式、合法标签与新颖性均由确定性 parser 判断。

新 reward：

```python
if not (valid and novel and final_label in acceptable_targets):
    reward = 0.0
else:
    reward = 0.8 + 0.2 * strict_format
```

因此错误答案无法单独获得格式奖励；任一 acceptable target 均算正确。

## 4. 实验结果（Results）

### 4.1 C0/C1 完整 no-CoT Dev10K

| input | C0 SFT22K | C1 old GRPO1K | delta | paired 95% CI | McNemar p |
|---|---:|---:|---:|---|---:|
| Behavior | 244/10,000 (2.44%) | 259/10,000 (2.59%) | +0.15 pp | [+0.01, +0.31] pp | 0.0722 |
| Interest | 437/10,000 (4.37%) | 487/10,000 (4.87%) | +0.50 pp | [+0.25, +0.75] pp | 0.000077 |

C1 faithful CoT full 10K：

| input | Set Accuracy | Strict Accuracy | strict format | valid label | truncation |
|---|---:|---:|---:|---:|---:|
| Behavior | 99/10,000 (0.99%) | 0.99% | 84.82% | 76.78% | 13.34% |
| Interest | 141/10,000 (1.41%) | 1.41% | 99.57% | 95.30% | 0.43% |

### 4.2 CoT-SFT1K gate（n=128）

| variant | input | CoT Set Acc | strict format | valid label | no-CoT Set Acc |
|---|---|---:|---:|---:|---:|
| pure-v1 | Behavior | 3.91% | 97.66% | 78.91% | 2.34% |
| pure-v1 | Interest | 3.12% | 0.00% | 48.44% | 3.12% |
| pure-v2 | Behavior | 0.78% | 45.31% | 54.69% | 2.34% |
| pure-v2 | Interest | 0.00% | 14.06% | 87.50% | 3.91% |
| rehearsal 3:1 | Behavior | 1.56% | 80.47% | 71.09% | 2.34% |
| rehearsal 3:1 | Interest | 0.00% | 0.78% | 8.59% | 4.69% |

预注册 gate：

```text
strict format ≥95%
valid label ≥98%
truncation ≤5%
no-CoT Set Accuracy drop ≤0.5pp
```

没有任何 variant 在 Behavior 与 Interest 上同时满足 gate。

### 4.3 Reward preflight

pure-v1：

| input | nonzero | mixed groups | all-zero groups | strict format | truncation |
|---|---:|---:|---:|---:|---:|
| Behavior | 1.81% | 17.97% | 82.03% | 90.28% | 0.05% |
| Interest | 0.39% | 6.25% | 93.75% | 0.05% | 0.15% |

rehearsal Behavior：

```text
nonzero completion rate: 1.46%
mixed-reward groups: 19.53%
all-zero groups: 80.47%
strict format: 65.04%
truncation: 0%
correctness reward mass: 24.0
format bonus mass: 5.4
```

正确性 reward 已不再被格式 bonus 主导，但 reward density 仍低于预注册
`nonzero ≥5%` 门槛。

### 4.4 Gate 决策

correctness GRPO1K 未启动，原因：

1. 三种 CoT-SFT warm-up 都未满足联合 gate；
2. Interest 对自由长度 CoT 的格式学习极不稳定；
3. canonical label rate 明显下降；
4. Behavior reward mixed rate接近门槛，但 nonzero reward 仅 1–2%；
5. 扩大到 5K 或直接 GRPO 会违反“不得用规模掩盖 gate failure”的预注册规则。

因此 GRPO5K 同样未启动。

## 5. 数据与产物路径（Artifact Paths）

### 本地数据与代码

- acceptable targets：
  `data/training/grpo_acceptable_target_sets/`
- multi-target GRPO：
  `data/training/grpo_experiments/`
- GPT-5.6 teacher：
  `artifacts/cot_teacher/gpt56_sol/`
- CoT-SFT1K：
  `data/training/cot_sft_experiments/`
- 3:1 rehearsal：
  `data/training/cot_sft_rehearsal_experiments/`
- reward：
  `src/new_interest/training/grpo_reward.py`
- Accuracy evaluator：
  `src/new_interest/evaluation/evaluate_cot_accuracy.py`
- paired comparator：
  `src/new_interest/evaluation/compare_paired_accuracy.py`
- teacher generator：
  `src/new_interest/data/generate_cot_teacher.py`
- report CSV：
  `docs/experiments/2026-08-09/qwen3_cot_accuracy_reward_metrics.csv`

### 远端实验

- C0/C1 no-CoT full：
  `qwen-cot-accuracy-controls-dev10k-v1`
- C1 faithful CoT source outputs：
  `qwen-cot-label-dev10k-v1`
- pure CoT-SFT：
  `qwen-cot-sft1k-v1`, `qwen-cot-sft1k-v2`
- rehearsal：
  `qwen-cot-rehearsal1k-v1`
- Accuracy gates：
  `qwen-cot-sft1k-accuracy-smoke-v1`,
  `qwen-cot-sft1k-v2-accuracy-smoke`,
  `qwen-cot-rehearsal1k-accuracy-smoke-v1`
- reward preflights：
  `qwen-correctness-grpo-preflight-v1`,
  `qwen-correctness-grpo-preflight-v2`,
  `qwen-cot-rehearsal-grpo-preflight-v1`

模型权重与 checkpoints 仅保存在 AMLT Blob；本机未下载 adapter/checkpoint。
