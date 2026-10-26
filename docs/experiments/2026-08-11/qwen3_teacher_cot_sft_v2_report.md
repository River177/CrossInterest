# Qwen3-8B Teacher CoT-SFT1K v2 实验报告

> **状态：在 CoT-SFT gate 处停止。** GPT-5.6 Sol Teacher 数据、Behavior/Interest
> CoT-SFT1K、Dev128 双模式评测和 Interest Dev10K 均已完成。两个分支未同时满足
> canonical-label 与 no-CoT retention 门槛，因此预注册流程在 gate 处停止。随后
> 用户将 no-CoT 下降改为监控项，并批准单独提交 fixed-target correctness
> preflight；optimizer GRPO 仍未启动。

## 1. 研究问题（Research Questions）

1. 在 Student Prompt 与原始 SFT22K Prompt 逐字一致时，GPT-5.6 Sol
   Teacher CoT-SFT 能否激活稳定的 Qwen3 native thinking？
2. Behavior 与 Interest 两种输入能否同时满足：
   - strict `<think>...</think>\nNew topic:` 格式；
   - canonical 110-label 输出；
   - 低截断率；
   - no-CoT 能力保留？
3. 满足上述 gate 后，是否值得进入 fixed-single-target correctness GRPO
   preflight？

主指标：

```text
Set Accuracy@1 = predicted label ∈ acceptable targets
Strict Set Accuracy@1 = Set Accuracy@1 AND strict output format
```

Gate：

```text
Dev128:
  strict format ≥ 90%
  valid canonical label ≥ 95%
  truncation ≤ 5%
  no-CoT Accuracy drop ≤ 0.5pp

Dev10K:
  strict format ≥ 95%
  valid canonical label ≥ 98%
  truncation ≤ 5%
  no-CoT Accuracy drop ≤ 0.5pp
```

## 2. 训练数据构造（Training Data Construction）

### 2.1 来源与选择

训练源为严格平衡、Behavior/Interest 配对的 SFT 1K prefix：

```text
data/training/sft_experiments/train/
├── behavior_only/1k.parquet
└── interest_only/1k.parquet
```

两种输入共享 1,000 个 `source_sample_id`、impression、fixed target 与顺序；
覆盖 110 个二级主题，每类 9 或 10 条。Teacher selection 使用 source-order
prefix，target 不得出现在 Interest history。

| condition | Behavior | Interest | paired source IDs | targets |
|---|---:|---:|---:|---:|
| Teacher CoT-SFT1K v2 | 1,000 | 1,000 | 1,000 | 110 |

### 2.2 GPT-5.6 Sol Teacher

| field | value |
|---|---|
| instance | `redmond/interactive` |
| deployment | `gpt-5.6-sol_2026-07-09` |
| API version | `2024-10-21` |
| max completion tokens | 256 |
| temperature | omitted |
| concurrency | 4 |
| attempts | 3 |

Teacher 可见历史和 private fixed target，但只返回 JSON `reasoning`。自动过滤：

- canonical target 名称；
- `<think>` / `</think>`；
- `New topic`；
- 空 reasoning；
- `private target`、`reference target`、`ground truth` 等元话语。

最终 2,000/2,000 success。原 preview 的 40 条 reasoning 中 38 条复用；2 条因
meta-target language 被重新生成。reasoning 平均 49.47 words，p95 64，最大 95。

### 2.3 Student 数据

Student Prompt 保留原 SFT22K 文本，不追加 CoT 指令。Assistant target：

```text
<think>
{teacher reasoning}
</think>
New topic: {fixed target}
```

Token audit：

| input | samples | mean tokens | p95 | max | over 2048 |
|---|---:|---:|---:|---:|---:|
| Behavior | 1,000 | 583.33 | 1,130 | 1,222 | 0 |
| Interest | 1,000 | 328.73 | 554 | 599 | 0 |

### 2.4 真实训练样例

#### Behavior

**Original prompt**

```text
The headlines of the news items I recently viewed are listed below:

1. James Comey Would Like to Help

Based on these headlines, predict a new topic category that I am likely to be
interested in next. The predicted category should be different from the topics
reflected in my viewing history. Begin your response with "New topic: " and
provide only the category name after it.
```

**中文翻译**

```text
下面列出了我最近浏览的新闻标题：

1. James Comey 希望提供帮助

根据这些标题，预测我接下来可能感兴趣的一个新主题类别。预测类别应不同于浏览
历史中体现的主题。回答必须以“New topic: ”开头，之后只提供类别名称。
```

**Target response**

```text
<think>
The viewed headline concerns a prominent public figure and appears framed as a
reported or analytical current-affairs piece, making a broader interest in
coverage of public events and influential people a plausible next step.
</think>
New topic: Journalism & News Media
```

**中文翻译**

```text
<think>
该标题涉及一位知名公众人物，并采用新闻报道或时事分析的形式，因此下一步对公共
事件及有影响力人物的新闻报道产生更广泛兴趣是合理的。
</think>
新主题：新闻与新闻媒体
```

#### Interest

**Original prompt**

```text
The topic categories associated with the news items I recently viewed, in
chronological order, are listed below:

1. Music

Based on this sequence of topic categories, predict a new topic category that I
am likely to be interested in next. The predicted category should not appear in
the sequence above. Begin your response with "New topic: " and provide only the
category name after it.
```

**中文翻译**

```text
下面按时间顺序列出了我最近浏览新闻对应的主题类别：

1. 音乐

根据该主题序列，预测我接下来可能感兴趣的新主题类别。预测类别不得出现在上述
序列中。回答必须以“New topic: ”开头，之后只提供类别名称。
```

**Target response**

```text
<think>
A recent interest in music can plausibly lead to another closely related
entertainment category because artists often appear in documentaries, concert
productions, interviews, award shows, and fictional works built around
performers or soundtracks. These overlapping formats make the transition a
natural extension of the same interest.
</think>
New topic: Television & Streaming
```

**中文翻译**

```text
<think>
近期对音乐的兴趣可以合理延伸到另一个紧密相关的娱乐类别，因为艺人经常出现在
纪录片、演唱会制作、访谈、颁奖典礼以及围绕表演者或原声音乐构建的虚构作品中。
这些重叠形式使这种转移成为同一兴趣的自然延伸。
</think>
新主题：电视与流媒体
```

## 3. 测试设计（Testing）

### 3.1 模型与位置

| field | Behavior | Interest |
|---|---|---|
| base | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| initialization | Behavior SFT22K LoRA | Interest SFT22K LoRA |
| data | Behavior Teacher CoT 1K | Interest Teacher CoT 1K |
| epochs / steps | 2 / 64 | 2 / 64 |
| LR | `2e-5` | `2e-5` |
| effective batch | 32 | 32 |
| final train loss | 2.5159 | 2.2325 |
| runtime | 851.7 s | 436.4 s |
| location | GCR A100 80GB | GCR A100 80GB |

Adapters/checkpoints 保留于 GCR；本机只保存 metrics、必要 predictions、配置与报告。

### 3.2 测试数据来源

冻结 grouped Dev：

```text
data/training/sft_experiments/eval/
├── behavior_only/dev_10k_grouped.parquet
└── interest_only/dev_10k_grouped.parquet
```

Dev128 使用每个文件的前 128 条进行快速 gate。只有 Interest 通过 Dev128，故仅
Interest 扩大至完整 Dev10K。Behavior 的 epoch-1 与 epoch-2 checkpoints 均评测，
两者均未通过。

主结果使用 greedy decoding，保证与现有 Accuracy 实验口径一致：

- native-CoT：原 Prompt，`enable_thinking=True`，max new tokens 768；
- no-CoT：原 Prompt，`enable_thinking=False`，max new tokens 64。

### 3.3 真实测试样例

#### Interest：正确且严格

**Prompt**

```text
1. Crime & Public Safety
2. Space & Astronomy
3. Earthquakes & Geological Hazards
4. Accidents & Emergency Response
5. American Football
6. Crime & Public Safety
7. Air & Rail Travel
8. Accidents & Emergency Response
9. Companies & Corporate Strategy
10. Civil Rights & Social Justice
11. Courts & Legal Cases
12. Companies & Corporate Strategy
13. Celebrity & Creator Culture
14. Historical Events & Eras
15. Food Culture & Trends
16. Courts & Legal Cases
```

**中文翻译**

```text
1. 犯罪与公共安全
2. 太空与天文学
3. 地震与地质灾害
4. 事故与应急响应
5. 美式橄榄球
6. 犯罪与公共安全
7. 航空与铁路旅行
8. 事故与应急响应
9. 公司与企业战略
10. 公民权利与社会正义
11. 法院与法律案件
12. 公司与企业战略
13. 名人与创作者文化
14. 历史事件与时代
15. 食品文化与趋势
16. 法院与法律案件
```

**Model response**

```text
<think>
The recent viewing history includes a mix of public safety, emergency response,
legal cases, corporate strategy, and historical events, suggesting an interest
in broader societal issues, institutional structures, and the interplay between
public policy and everyday life. A category that aligns with this pattern would
be one that explores how institutions, laws, and historical developments shape
modern society and its challenges.
</think>

New topic: Government & Public Policy
```

**中文翻译**

```text
<think>
近期浏览历史包含公共安全、应急响应、法律案件、企业战略和历史事件，表明用户对
更广泛的社会问题、制度结构以及公共政策与日常生活之间的互动感兴趣。符合这一
模式的类别应关注制度、法律和历史发展如何塑造现代社会及其挑战。
</think>

新主题：政府与公共政策
```

Acceptable target 为 `Government & Public Policy`；strict、valid、set hit 均为
true。

#### Behavior：格式正确但类别不在 canonical label set

**Prompt**

```text
1. Look who's watching
2. America's Unsafe Drugs and Medical Products
```

**中文翻译**

```text
1. 看看是谁在监视
2. 美国不安全的药品和医疗产品
```

**Model response**

```text
<think>
The viewing history includes articles about consumer privacy and the safety of
medical products. These topics are related to public health, consumer rights,
and regulatory oversight. A related but distinct topic would be the regulation
of consumer products, which involves government oversight, safety standards,
and enforcement mechanisms.
</think>

New topic: Consumer Protection
```

**中文翻译**

```text
<think>
浏览历史包含消费者隐私和医疗产品安全文章。这些主题涉及公共健康、消费者权利和
监管监督。一个相关但不同的主题是消费品监管，其中包括政府监督、安全标准和执法
机制。
</think>

新主题：消费者保护
```

输出格式严格，但 `Consumer Protection` 不属于保留的 110 个 canonical labels，
因此 `valid_label=false`，且 acceptable target `Cars & Driving` 未命中。

### 3.4 判定程序

本实验没有使用 LLM judge。格式与指标由规则解析器确定。

Strict CoT parser（`src/new_interest/training/grpo_reward.py`）：

```python
r"\A<think>(?P<think>[\s\S]*?)</think>\s*"
r"(?P<answer>New topic: [^\r\n]*\S)\Z"
```

规则含义：

- `<think>` 内容必须非空；
- 必须闭合 `</think>`；
- 最后必须且只能有一行 `New topic: <label>`；
- final answer 后不得有尾随文本。

Accuracy parser 位于
`src/new_interest/evaluation/evaluate_cot_accuracy.py`：

- `strict_format`：满足上述完整格式；
- `valid_label`：预测 label 属于保留的 110 labels；
- `novelty_compliant`：预测 label 不在该用户 history；
- `set_hit_at_1`：预测 label 属于该 Dev impression 的任一 acceptable target；
- `strict_set_hit_at_1`：同时满足 set hit 与 strict format；
- `truncation_rate`：生成未遇到 EOS 且达到 max new tokens 的比例。

无 judge error 字段；全部 128/10,000 records 均完成规则判定。

## 4. 实验结果（Results）

### 4.1 Dev128

| input | mode | Set Acc | strict format | valid label | novelty | truncation | mean tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Behavior | native-CoT | 2/128 (1.56%) | 127/128 (99.22%) | 110/128 (85.94%) | 90/128 (70.31%) | 1/128 (0.78%) | 74.04 |
| Behavior | no-CoT | 3/128 (2.34%) | 128/128 (100%) | 128/128 (100%) | 100/128 (78.13%) | 0/128 | 6.98 |
| Interest | native-CoT | 1/128 (0.78%) | 128/128 (100%) | 124/128 (96.88%) | 122/128 (95.31%) | 0/128 | 87.59 |
| Interest | no-CoT | 6/128 (4.69%) | 128/128 (100%) | 128/128 (100%) | 128/128 (100%) | 0/128 | 6.97 |

Behavior epoch-1 checkpoint：

| mode | Set Acc | strict format | valid label | truncation |
|---|---:|---:|---:|---:|
| native-CoT | 4/128 (3.13%) | 126/128 (98.44%) | 114/128 (89.06%) | 2/128 (1.56%) |
| no-CoT | 4/128 (3.13%) | 128/128 (100%) | 128/128 (100%) | 0/128 |

Behavior epoch-1/epoch-2 的 valid label 均低于 95%，且 no-CoT 相对冻结的
SFT22K Dev128 基线 6/128 (4.69%) 分别下降 1.56pp / 2.34pp，因此停止。

### 4.2 Interest Dev10K

| mode | Set Acc | strict format | valid label | novelty | truncation | mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| native-CoT | 206/10,000 (2.06%) | 9,981/10,000 (99.81%) | 9,711/10,000 (97.11%) | 9,497/10,000 (94.97%) | 14/10,000 (0.14%) | 88.65 |
| no-CoT | 339/10,000 (3.39%) | 10,000/10,000 (100%) | 9,996/10,000 (99.96%) | 9,992/10,000 (99.92%) | 0/10,000 | 6.86 |

冻结的 Interest SFT22K no-CoT 基线为 437/10,000 (4.37%)；CoT-SFT v2 为
339/10,000 (3.39%)，下降 98 hits / 0.98pp。native-CoT valid label 97.11%
也低于 98% gate。

### 4.3 Gate

| input | format gate | valid-label gate | truncation gate | no-CoT retention | final |
|---|---|---|---|---|---|
| Behavior | pass | fail | pass | fail | **stop** |
| Interest | pass | fail (97.11% < 98%) | pass | fail (-0.98pp) | **stop** |

Teacher CoT-SFT 成功解决了此前最严重的格式与截断问题，但没有同时满足 canonical
label 与 no-CoT retention gate。按预注册停止条件未直接启动 GRPO。用户随后批准
只读 fixed-target correctness `128×16` preflight；该 preflight 已提交 AMLT，
不包含 optimizer update。

置信区间未计算；本报告呈现完整计数与比例。详细表见
[metrics CSV](qwen3_teacher_cot_sft_v2_metrics.csv)。

## 5. 数据与产物路径（Artifact Paths）

### 本地仓库

- Teacher selection：
  `artifacts/cot_teacher/gpt56_sol_v2/selection.parquet`
- Teacher records：
  `artifacts/cot_teacher/gpt56_sol_v2/teacher_records.parquet`
- Teacher metadata：
  `artifacts/cot_teacher/gpt56_sol_v2/teacher_metadata.json`
- Teacher audit sample：
  `artifacts/cot_teacher/gpt56_sol_v2/audit_sample_40.json`
- CoT-SFT data：
  `data/training/cot_sft_v2_experiments/{behavior_only,interest_only}/1k.parquet`
- CoT-SFT metadata：
  `data/training/cot_sft_v2_experiments/metadata.json`
- Token audits：
  `data/training/cot_sft_v2_experiments/{behavior_only,interest_only}_token_audit.json`
- Training configs：
  `configs/sft/experiments/qwen3_{behavior,interest}_cot_v2_1k.yaml`
- Teacher config：
  `configs/cot_teacher/gpt56_sol_v2_1k.yaml`
- Metrics：
  `docs/experiments/2026-08-11/qwen3_teacher_cot_sft_v2_metrics.{csv,json}`

### Session artifacts

- Dev128 metrics/predictions：
  `/Users/river/.copilot/session-state/04516570-14ad-4f3f-8396-53b54986ebe2/files/cot-sft-v2-dev128/`
- Interest Dev10K metrics/predictions：
  `/Users/river/.copilot/session-state/04516570-14ad-4f3f-8396-53b54986ebe2/files/cot-sft-v2-dev10k/`
- Training summaries/configs/audits：
  `/Users/river/.copilot/session-state/04516570-14ad-4f3f-8396-53b54986ebe2/files/cot-sft-v2-training/`

### GCR A100

- Behavior adapter/checkpoints：
  `/home/v-wuhao1/CrossInterest/artifacts/sft/qwen3-behavior-cot-v2-full/`
- Interest adapter/checkpoints：
  `/home/v-wuhao1/CrossInterest/artifacts/sft/qwen3-interest-cot-v2-full/`
- Dev128 raw outputs：
  `/home/v-wuhao1/CrossInterest/artifacts/cot-v2-gates/dev128/`
- Interest Dev10K raw outputs：
  `/home/v-wuhao1/CrossInterest/artifacts/cot-v2-gates/dev10k/interest/`

模型权重未下载到本机。

### AMLT correctness preflight

- Experiment：
  `crossinterest-qwen3-cot-v2-correctness-preflight-v1`
- Jobs：
  - `behavior-correctness-preflight`
  - `interest-correctness-preflight`
- 提交状态：queued
- Adapter Blob：
  `projects/sft4safety/manual-artifacts/cot-v2-20260811/`
