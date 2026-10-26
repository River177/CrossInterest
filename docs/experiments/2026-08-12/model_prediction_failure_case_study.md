# 新兴趣预测失败诊断与案例研究

> **状态：诊断完成，建议暂不进入 GRPO。** 本报告使用现有数据、Qwen3.5-9B
> Beam 预测、五模型 K=10 候选预测和无模型统计基线，区分标签语义噪声、任务定义
> 问题、部分可观测性、输入表示损失与模型误差。除 120 条标题标签质检外，所有数字
> 都是对固定数据文件的全量确定性统计。

## 1. 研究问题（Research Questions）

本轮回答：

1. 模型低准确率主要来自主题标签标错，还是历史本身无法唯一决定下一次点击？
2. 同一个输入是否在 Train/Dev 中对应互相冲突的 target？
3. 静态 MIND history 是否把用户已经点击过的主题再次定义为“新兴趣”？
4. Behavior 标题与 Interest 主题序列分别保留、丢失了什么预测信息？
5. 类别频率、日期漂移、负样本采样和多答案数量如何影响当前指标？
6. 在进入 SFT/GRPO 优化前，哪些数据和 benchmark 问题必须先解决？

主要比较条件：

- 原始自然分布 Train/Dev；
- 类别均衡 SFT22K；
- frozen grouped Dev10K；
- Qwen3.5-9B Behavior/Interest 22K Beam；
- Qwen3、Qwen3.5、Gemini 3.6 Flash、GPT-5.6 Sol、GPT-5 的 K=10 预测；
- Train popularity、last-interest conditional 等无模型基线。

主要指标为 exact-input target 冲突率、经验众数 Top-1、Recall/Hit/NDCG@20、
K=10 Accuracy、跨模型 oracle、目标频率分层结果和标题标签质检类别。

## 2. 训练数据构造（Training Data Construction）

### 2.1 数据来源与“新兴趣”定义

原始来源为 GPT-5.5 重分类后的 MINDlarge：

```text
data/raw/MINDlarge_reclassified/
```

主题标签生成记录：

```text
model: gpt-5.5_2026-04-24
prompt_version: mind-large-fixed-taxonomy-token-efficient-v2
taxonomy: taxonomy/reader_interest_taxonomy_final.json
unique_news: 130,379
```

当前数据把一次 impression 中被点击、且未出现在该 behavior 静态点击历史中的二级
主题定义为新兴趣：

```text
target in current_clicked_subcategories
target not in behavior_history_subcategories
```

它不要求 target 相对该用户更早的 Train impression 也未出现。一个 impression
可以有多个 target，初始 processed 数据会为每个 target 各写一行。

| 数据 | impressions/rows | target classes | history |
|---|---:|---:|---|
| Raw Train | 1,219,553 impressions / 1,489,348 rows | 124 observed | 完整静态 history |
| Raw Dev | 199,644 impressions / 243,118 rows | 119 observed | 完整静态 history |
| SFT22K Behavior | 22,000 rows | 110，每类 200 | 最近 50 条标题 |
| SFT22K Interest | 22,000 rows | 110，每类 200 | 最近 50 个主题 |
| Grouped Dev10K | 10,000 impressions / 12,164 target memberships | 109 observed | 最近 50 条 |
| K=10 expanded | 12,164 rows | 1 正例 + 9 负例 | 与 Dev10K 配对 |

SFT Train 对每个 impression 固定选一个 target，再按类别均衡到每类 200。Dev
保留一个 impression 的全部 acceptable targets。因此 Train 学习的是单一 fixed
target，Beam 测试使用 target set。

### 2.2 真实训练冲突样例 A：静态 Behavior history

**Original prompt**

```text
The headlines of the news items I recently viewed are listed below:

1. Jamie Foxx Sparks Romance Rumors With Dana Caprio After Katie Split

Based on these headlines, predict a new topic category that I am likely to be
interested in next. The predicted category should be different from the topics
reflected in my viewing history. Begin your response with "New topic: " and
provide only the category name after it.
```

**中文翻译**

```text
我最近浏览的新闻标题如下：

1. Jamie Foxx 与 Katie 分手后，被传与 Dana Caprio 相恋

根据这些标题预测一个历史中尚未出现、但我接下来可能感兴趣的新主题。回答只能是
“New topic: <主题>”。
```

同一用户 `U515908`、同一完整输入对应两个监督 target：

| source | timestamp | target response | target news |
|---|---|---|---|
| `train:40:20` | 2019-11-13 12:34:08 | `New topic: Courts & Legal Cases` | `Kodak Black Sentenced to Over 3 Years in Prison in Weapons Case` |
| `train:1750912:106` | 2019-11-11 07:23:03 | `New topic: Cruises` | `7 Women Share What It's Really Like To Work On A Cruise Ship` |

目标新闻中文：

- Kodak Black 因武器案被判三年多监禁；
- 7 位女性讲述在邮轮上工作的真实体验。

这不是两个标签描述同一点击，而是相同静态 history 在不同时间产生了不同点击。

### 2.3 真实训练冲突样例 B：Interest 表示碰撞

**Original prompt**

```text
The topic categories associated with the news items I recently viewed, in
chronological order, are listed below:

1. Golf

Based on this sequence of topic categories, predict a new topic category that I
am likely to be interested in next. The predicted category should not appear in
the sequence above. Begin your response with "New topic: " and provide only the
category name after it.
```

**中文翻译**

```text
最近浏览主题只有“高尔夫”。请预测一个历史中未出现的后续新主题，只输出
“New topic: <主题>”。
```

SFT22K 中同一 prompt 至少对应：

| source | user | target response |
|---|---|---|
| `train:1965955:74` | `U663667` | `New topic: Archaeology & Human Origins` |
| `train:1200255:119` | `U41206` | `New topic: Severe Storms` |

这类冲突还包含不同用户被压缩成相同主题序列的情况。SFT22K 最大 Interest prompt
组包含 44 行、44 个用户和 34 个不同 target。

## 3. 测试设计（Testing）

### 3.1 测试模型与运行位置

主要开放式排序模型：

```text
base: Qwen/Qwen3.5-9B
train: Behavior/Interest SFT22K LoRA
platform: Feeds Singularity, 8 × A100 80GB per input
evaluation: frozen grouped Dev10K
num_beams: 32
deduplicated top_k: 20
do_sample: false
thinking: disabled
```

K=10 五模型诊断面板：

| model | decoding/reasoning | location |
|---|---|---|
| Qwen3-8B | greedy, thinking disabled | Feeds A100 |
| Qwen3.5-9B | greedy, thinking disabled | Feeds A100 |
| Gemini 3.6 Flash | T=0.7, top-p=0.8, minimal reasoning | local Copilot API proxy |
| GPT-5.6 Sol | reasoning none | TRAPI `redmond/interactive` |
| GPT-5 | reasoning medium | TRAPI `redmond/interactive` |

所有正式 candidate 条件均有 12,164/12,164 条输出；本报告没有重新运行远程模型。

### 3.2 测试数据来源

```text
data/training/sft_experiments/eval/
├── behavior_only/dev_10k_grouped.parquet
└── interest_only/dev_10k_grouped.parquet

data/training/candidate_selection_k10/
├── behavior_only/dev_10k_expanded.parquet
└── interest_only/dev_10k_expanded.parquet
```

Dev10K 是确定性 SHA-256 priority 子集。84.10% impression 只有一个 acceptable
target，15.90% 有两个或更多 target。K=10 为每个 target 展开一行，排除该
impression 的其他 acceptable targets 和历史主题后采九个负例。

### 3.3 诊断切片

本轮额外计算：

- 完整 prompt SHA-256 相同但 target 不同的组；
- 同一用户静态 history、跨 timestamp target 变化；
- seen/unseen user 和 Train 中已出现 target；
- 单 target 的 Train support 四分位；
- acceptable target 数量；
- Behavior/Interest 同模型配对；
- 五模型和十个“模型 × 输入”条件的 oracle；
- 两路 Qwen3.5 Beam 都未命中的真实 case。

### 3.4 判定程序

正式指标不使用 LLM Judge。

- Candidate：`src/new_interest/evaluation/candidate_metrics.py`
- Beam：`src/new_interest/evaluation/metrics.py`
- 无模型关系分析：
  `src/new_interest/experiments/analyze_interest_relationships.py`

定义：

```text
exact-input conflict =
  identical full prompt has at least two distinct targets

prompt-mode accuracy =
  for each identical prompt, always predict its most frequent target

candidate oracle =
  at least one included model predicts the correct candidate

both-beam-miss =
  neither Behavior nor Interest Qwen3.5 ranking contains a target in Top-20
```

`prompt-mode accuracy` 只是当前重复输入上的经验 Top-1 上限，不是未知 Dev 的理论
Bayes 上限。

另对固定 Dev10K 做了 seed `20260812` 的 120 条标题标签探索性质检，覆盖 99 个
标签。质检只根据标题分为：

```text
clear_match
boundary_ambiguous
likely_mismatch
insufficient_title
```

该质检不是正式 judge 指标，没有 JSON schema，不参与模型分数，也不能外推成总体
标签准确率。

## 4. 实验结果（Results）

### 4.1 核心诊断

当前证据支持以下优先级：

| factor | evidence | diagnosis |
|---|---|---|
| 历史条件信号弱/部分可观测 | last-interest 修正 MI 仅 0.0613 bit，解释约 1.00% target entropy | **主要原因** |
| 静态 history 与动态“新兴趣”定义不一致 | Train/Dev 内同一用户 history 100% 静态；13.30% seen-user Dev target 已在 Train 出现 | **主要数据定义问题** |
| 自然数据相同输入多 target | Raw Train 83.61%/85.02% 行处于冲突 prompt 组 | **主要监督歧义** |
| 类别先验和日期漂移 | Train/Dev target TV distance 约 0.428；均衡 Train 与长尾 Dev 不匹配 | **主要评测/校准因素** |
| 输入表示损失 | Behavior 与 Interest 错误互补；Interest 短序列产生大量跨用户碰撞 | **重要但非单向因素** |
| 标题主题标签语义错误 | 120 条探索样本中 5 条 likely mismatch、20 条边界歧义 | **存在，但未证明是主因** |
| 输出格式/解码 | Candidate-valid 约 98%–100%；sampling/greedy 基本不修复 | **不是主要原因** |

因此“数据噪声大”需要拆开：真正占主导的不是随机把新闻标成错误主题，而是输入没有
包含决定下一次点击所需的 session/exposure 信息，并且静态 history 被重复用于多个
时间点和多个 target。

### 4.2 相同输入对应不同 target

#### 自然 Raw 数据

| data | conflict rows | rate | empirical prompt-mode Top-1 |
|---|---:|---:|---:|
| Train Behavior | 1,245,277 / 1,489,348 | 83.612% | 44.053% |
| Train Interest | 1,266,196 / 1,489,348 | 85.017% | 41.878% |
| Dev Behavior | 127,405 / 243,118 | 52.405% | 68.791% |
| Dev Interest | 134,304 / 243,118 | 55.242% | 66.251% |

Raw 数据的一行代表一个 target membership，所以一部分冲突来自同一 impression 的
多点击 target；更严重的是，同一用户跨 impression 仍使用同一静态 history。

```text
Train users: 587,187
users with multiple different targets under one static input: 348,441

Dev users: 163,917
users with multiple different targets under one static input: 48,387
```

#### 经过强筛选的 SFT22K

| data | conflict groups | conflict rows | rate | prompt-mode Top-1 |
|---|---:|---:|---:|---:|
| Behavior 22K | 525 | 1,074 / 22,000 | 4.882% | 97.514% |
| Interest 22K | 641 | 1,590 / 22,000 | 7.227% | 95.782% |

22K 的 one-target-per-impression 和类别采样显著降低了显式冲突，但没有修复底层
history 静态问题。Interest 还额外把不同标题历史压缩成相同主题序列。

Grouped Dev10K 大部分 prompt 唯一，因此全体 prompt-mode oracle 仍为 Behavior
98.82%、Interest 96.92%；但只看重复 prompt 行，oracle 降至：

```text
Behavior: 144/262 = 54.96%
Interest: 248/556 = 44.60%
```

这说明 exact duplicate 冲突是真实问题，但它单独不能解释模型 Top-1 只有几个百分点；
剩余差距主要来自未观测因素和弱统计关系。

### 4.3 无模型规则比 SFT Beam 更强

固定 Dev10K：

| method | input | Recall@20 | NDCG@20 | Hit@20 |
|---|---|---:|---:|---:|
| Qwen3.5-9B SFT22K | Interest | 45.27% | 18.99% | 49.68% |
| Qwen3.5-9B SFT22K | Behavior | 32.64% | 12.62% | 36.48% |
| Train global popularity | 无历史模型 | 56.25% | 22.35% | 61.02% |
| Last-interest conditional frequency | Interest 统计 | **59.28%** | **24.42%** | **63.90%** |
| Recent-5 conditional average | Interest 统计 | 59.12% | 23.44% | 63.76% |

模型超过了 110 类随机 Recall@20 18.18%，说明不是完全没有学习；但它尚未超过简单
Train 统计规则。此时直接做 GRPO，reward 很可能主要强化当前模型先验和 benchmark
捷径，而不是恢复缺失的用户状态。

### 4.4 类别频率、日期漂移与 benchmark

Train22K 每类固定 200 条，但 candidate Dev 正例高度长尾：

```text
American Football: 1,361
Shopping & Gift Ideas: 1,011
Policing & Corrections: 850
...
Astrology & Divination: 1
Boating & Watercraft: 1
Earthquakes & Geological Hazards: 1
```

自然 Train/Dev target distribution：

```text
total variation distance ≈ 0.428
Jensen-Shannon divergence ≈ 0.204 bit
Train/Dev Top-10 overlap = 5/10
```

单 target impression 按 Train support 四分：

| target support | n | Interest Hit@1 | Interest Hit@20 | Behavior Hit@20 |
|---|---:|---:|---:|---:|
| Q1 rare, 346–9,879 | 2,272 | 2.07% | 29.97% | 26.50% |
| Q2, 10,796–19,212 | 2,158 | 3.06% | 45.97% | 33.69% |
| Q3, 19,736–31,110 | 2,054 | 5.06% | **58.18%** | **42.02%** |
| Q4 frequent, 33,826–61,462 | 1,926 | 6.28% | 52.44% | 31.46% |

低频 target 明显更难；最高频组没有继续单调上升，符合强日期漂移和类别先验失配。

原 K=10 benchmark 的负样本近似均匀，而正样本长尾：

| baseline/data | Accuracy |
|---|---:|
| uniform random | 10.00% |
| Train popularity，原候选集 | 31.99% |
| last-interest conditional，原候选集 | 33.37% |
| Train popularity，频率匹配 1K | 8.80% |
| last-interest conditional，频率匹配 1K | 10.00% |
| last-interest lift，频率匹配 1K | 13.60% |

所以原 candidate 分数混入了大量“候选中谁更流行”，不能只用 10% 随机线解释。

### 4.5 多答案数量会显著改变难度

Qwen3.5 Beam：

| acceptable targets | impressions | Interest Hit@20 | Behavior Hit@20 | 两路都 miss |
|---:|---:|---:|---:|---:|
| 1 | 8,410 | 46.11% | 33.27% | 48.55% |
| 2 | 1,195 | 65.02% | 50.21% | 29.12% |
| 3 | 283 | 75.97% | 59.01% | 17.67% |
| 4 | 73 | 84.93% | 68.49% | 9.59% |

多答案 impression 给 Top-20 更多命中机会。因此总体 Hit@20 不能直接代表“唯一下一
兴趣”的可预测率；单 target 切片更接近严格诊断。

### 4.6 seen user、重复 target 与静态 history

按 retained-110 Train 定义 seen user：

| slice | n | Interest Hit@20 | Behavior Hit@20 |
|---|---:|---:|---:|
| unseen user | 2,555 | 54.21% | 42.23% |
| seen user | 7,445 | 48.13% | 34.51% |
| target 未在该用户 Train targets 出现 | 8,841 | 48.40% | 35.93% |
| 至少一个 target 已在 Train 出现 | 1,159 | 59.45% | 40.64% |

seen user 并未因身份重叠而更容易，因为模型输入没有 user ID，且重复使用静态 history。
相反，“已经在 Train 点击过、但静态 history 未记录”的 target 更容易，说明当前
benchmark 把部分兴趣复现当成了兴趣发现。

### 4.7 多模型共同失败与表示互补

K=10 五模型面板：

| input | best single model | 5-model oracle | 五个全错 | 全错且 ≥3 模型同选一个错误答案 |
|---|---:|---:|---:|---:|
| Behavior | Gemini 19.17% | 46.45% | 6,514/12,164 (53.55%) | 3,308/12,164 (27.20%) |
| Interest | GPT-5.6 21.03% | 38.98% | 7,422/12,164 (61.02%) | 5,073/12,164 (41.71%) |

把五个模型的 Behavior 和 Interest 共十个条件合并：

```text
at least one correct: 7,142/12,164 = 58.71%
all ten wrong:        5,022/12,164 = 41.29%
all ten correct:      5/12,164
```

同一模型的两种输入也明显互补：

| model | both correct | Behavior only | Interest only | either correct |
|---|---:|---:|---:|---:|
| Qwen3 | 344 | 1,490 | 668 | 20.57% |
| Qwen3.5 | 619 | 1,479 | 888 | 24.55% |
| Gemini | 1,002 | 1,330 | 1,379 | 30.51% |
| GPT-5.6 | 913 | 1,225 | 1,645 | 31.10% |
| GPT-5 | 409 | 986 | 1,897 | 27.06% |

因此不能简单断言 Behavior 或 Interest 总是更好。标题提供实体和事件语义；主题序列
更短、更易学习，但会丢失跨主题桥接信息并制造 prompt collision。

### 4.8 Case study A：历史无法解释真实点击

```text
impression_id: 41551
history length: 1
```

**History**

```text
Simone Biles Shares Sweet Birthday Tribute for "Man of My Dreams"
Stacey Ervin Jr.
```

中文：Simone Biles 为“梦中情人”Stacey Ervin Jr. 送上甜蜜生日祝福。

**Observed target**

```text
Motorcycles
Top 5 Craziest Motorcycles From EICMA 2019
```

中文：摩托车；2019 EICMA 最疯狂的五款摩托车。

Interest K=10 候选包括 `Motorcycles`、`Consumer Technology`、`Fitness &
Exercise` 等。五个 Interest 模型全部输出：

```text
New topic: Consumer Technology
```

Behavior 五模型则分散到 `Parenting & Family Life` 和 `Fitness & Exercise`。
Qwen3.5 Behavior/Interest 两路 Top-20 都没有 `Motorcycles`。

这条 target 标签与新闻标题匹配，失败原因不是明显错标，而是当前唯一历史点击几乎不
包含预测摩托车点击的证据。模型只能依赖全局先验或候选间的常识联想。

### 4.9 Case study B：被错误定义为“新”的重复兴趣

```text
impression_id: 284807
user_id: U562515
timestamp: 2019-11-15 11:31:04
```

**Static history**

```text
Why President Trump's kids are probably celebrating his move to Florida
```

中文：为什么特朗普总统的孩子们可能在庆祝他搬到佛罗里达。

**Current target**

```text
Public Services & Social Welfare
Upstate county sees increase in homeless children
```

中文：公共服务与社会福利；某上州县无家可归儿童数量增加。

但同一用户前一天已经有：

```text
2019-11-14 10:25:15
Public Services & Social Welfare
One in 5 children live below the poverty line...
```

中文：五分之一儿童生活在贫困线以下……

Interest K=10 五模型全部预测 `Economy & Economic Policy`；两路 Beam 都未把
`Public Services & Social Welfare` 放入 Top-20。

这不是严格意义的“第一次新兴趣”。它只相对 MIND behavior 中未更新的静态 history
为新，属于数据定义污染。

### 4.10 Case study C：Behavior 标题保留了跨主题桥梁

```text
impression_id: 338845
target: Courts & Legal Cases
target news: Myles Garrett is getting suspended. Could he also be charged
with a crime?
```

中文：Myles Garrett 将被禁赛；他是否还可能受到刑事指控？

历史七条新闻包括五条 NFL 内容，以及宗教和政府新闻。Interest 表示只剩：

```text
American Football
Religion in Public Life
Government & Public Policy
American Football × 4
```

Behavior 标题保留了具体 NFL 事件和人物。K=10 中五个 Behavior 模型全部正确输出：

```text
New topic: Courts & Legal Cases
```

五个 Interest 模型分别输出外交、AI、基础设施和高等教育，没有一个正确。

这里标题中的“体育事件可能转入刑事指控”是预测桥梁，而纯主题序列把它压缩掉了。

### 4.11 Case study D：Interest 抽象有时反而去除标题噪声

```text
impression_id: 177268
target: Economy & Economic Policy
target news: Warren Tax Plan Would Bring Rates Over 100% for Some
```

中文：沃伦税收计划可能使部分人的综合税率超过 100%。

历史：

```text
1. Nationals beat Astros in Game 7 to win World Series
2. Clinton-Obama emails sought by Sen. Ron Johnson amid Dems' impeachment inquiry
3. Meghan McCain confronts Trump Jr.: "You and your family have hurt a lot of people"
```

中文：

1. 国民队在世界大赛第七场击败太空人夺冠；
2. 民主党弹劾调查期间，参议员 Ron Johnson 要求获取 Clinton-Obama 邮件；
3. Meghan McCain 质问 Trump Jr.：“你和你的家人伤害了很多人。”

Interest 序列为：

```text
Baseball & Softball
Government & Public Policy
Government & Public Policy
```

五个 Interest 模型全部正确输出 `Economy & Economic Policy`；五个 Behavior
模型全部错误。这里主题抽象削弱了人物和媒体叙事噪声，突出连续公共政策兴趣。

### 4.12 标题标签语义质检

探索性质检结果：

| result | count |
|---|---:|
| clear match | 92/120 |
| boundary ambiguous | 20/120 |
| likely mismatch | 5/120 |
| insufficient title | 3/120 |

真实 likely-mismatch 例子：

| title | assigned | plausible |
|---|---|---|
| `Illegal Botox manifested as 'leggings' seized at DHL` | Cybersecurity & Digital Privacy | Beauty & Grooming / Medicine & Medical Research |
| `Someone Finally Invented a Jar Lid That Won't Get Stuck` | Home Appliances & Technology | Cooking & Recipes |
| `Those 99-can Pabst 'family packs'?...` | Gambling & Lotteries | Wine, Beer & Spirits |

中文：

- DHL 查获伪装成“紧身裤”的非法肉毒杆菌产品；
- 终于有人发明了不会卡住的罐盖；
- Pabst 的 99 罐“家庭装”啤酒将来到 Milwaukee。

边界歧义例子包括 `Weather Forecasts & Seasonal Conditions` vs `Climate Change`、
`Real Estate` vs `Hotels & Accommodation`。因此 taxonomy 噪声需要治理，但当前
小样本不支持“多数错误都是标签标错”的判断。

### 4.13 进入 GRPO 前的建议实验顺序

动态自然分布基础数据已经生成：

```text
data/processed/new_interest_dynamic_v1/
Train: 2,186,664 impressions, 1,157,913 with new interest
Dev:     365,199 impressions,   179,555 with new interest
```

此前 case `U562515 / impression 284807` 中的
`Public Services & Social Welfare` 已被正确标为 repeated，而不是 novel。
动态平衡 generative 数据也已生成：

```text
data/training/generative_new_interest_dynamic_balanced_200_h50/
110 classes × 200 Train impressions = 22,000
online Dev retained target rows = 214,355
```

当前 SFT subsets、candidate 和 GRPO 下游尚未从该新 generative 数据重新生成。

1. **固定动态 base 作为唯一 source（已完成）。** 后续数据必须从
   `new_interest_dynamic_v1` 派生，不再从静态 `new_interest` 派生。
2. **拆分 cohort。** 分开报告 unseen user、seen-but-truly-new、seen-repeated-target，
   以及 single/multi-target。
3. **改为时间滚动验证。** 至少使用连续日期 Train/validation/test，禁止在同一天
   Dev 上反复选择规则。
4. **保留自然先验与均衡条件两套 Train。** 自然 Train 衡量真实预测，均衡 Train
   衡量历史条件关系；不能混用解释。
5. **重做 candidate benchmark。** 同时运行均匀负例、Train-frequency-matched、
   语义 hard negatives，并对多个候选排列 seed 求平均。
6. **先跑简单可解释模型。** global/recent popularity、user target frequency、
   last-interest transition、线性/embedding classifier 必须成为正式基线。
7. **再判断是否需要 GRPO。** 只有当 SFT 在动态、时间外推、频率匹配 benchmark 上
   稳定超过无模型基线，且剩余错误主要是排序/推理而不是 target 歧义时，GRPO 才有
   清晰优化对象。

当前更合理的下一步不是设计更复杂 reward，而是建立一个模型确实有机会从输入中恢复
target 的数据集。

## 5. 数据与产物路径（Artifact Paths）

### 数据与构造

- `data/processed/new_interest_dynamic_v1/`
- `data/training/generative_new_interest_dynamic_balanced_200_h50/`
- `data/processed/new_interest/train.parquet`
- `data/processed/new_interest/dev.parquet`
- `data/processed/new_interest/metadata.json`
- `data/processed/new_interest/statistics.json`
- `data/training/sft_experiments/train/behavior_only/22k.parquet`
- `data/training/sft_experiments/train/interest_only/22k.parquet`
- `data/training/sft_experiments/eval/behavior_only/dev_10k_grouped.parquet`
- `data/training/sft_experiments/eval/interest_only/dev_10k_grouped.parquet`
- `data/training/candidate_selection_k10/`
- `src/new_interest/data/build_dataset.py`
- `src/new_interest/data/build_generative_dataset.py`
- `src/new_interest/data/build_sft_subsets.py`
- `src/new_interest/data/build_candidate_selection.py`

### 既有统计与报告

- `artifacts/interest-relationship-analysis/sft-dev10k-h50-retained110/`
- `docs/experiments/2026-08-09/history_new_interest_relationship_analysis.md`
- `docs/experiments/2026-08-08/qwen_base_interest_bias_diagnosis.md`
- `docs/experiments/2026-08-10/candidate_selection_extended_comparison_report.md`
- `docs/experiments/2026-08-11/qwen35_22k_experiment_report.md`

### Qwen3.5 Beam

- `artifacts/beam/qwen35-9b-behavior-22k/qwen35-22k-beam-20260811-v1/qwen35-beam-dev10k-qwen35-behavior-22k/beam/`
- `artifacts/beam/qwen35-9b-interest-22k/qwen35-22k-beam-20260811-v1/qwen35-beam-dev10k-qwen35-interest-22k/beam/`
- AMLT experiment：`qwen35-22k-beam-20260811-v1`

### Candidate predictions

- `artifacts/candidate-selection-k10/amlt/candidate-k10-qwen-base-v2/`
- `artifacts/candidate-selection-k10/amlt/candidate-qwen35-base-v1/`
- `artifacts/candidate-selection-k10/gemini36_flash_{behavior,interest}/`
- `artifacts/candidate-selection-k10/gpt56_none_{behavior,interest}/`
- `artifacts/candidate-selection-k10/gpt5_{behavior,interest}/full/`
- `artifacts/candidate-selection-k10/extended-comparison/comparison.csv`

### 本轮诊断中间产物

- `/Users/river/.copilot/session-state/103356e0-ba4d-4d59-8fe4-008b7dbc64f1/files/error-analysis/summary.json`
- `/Users/river/.copilot/session-state/103356e0-ba4d-4d59-8fe4-008b7dbc64f1/files/error-analysis/cases.json`
- `/Users/river/.copilot/session-state/103356e0-ba4d-4d59-8fe4-008b7dbc64f1/files/error-analysis/prompt_collisions.json`
- `/Users/river/.copilot/session-state/103356e0-ba4d-4d59-8fe4-008b7dbc64f1/files/error-analysis/beam_support_quartiles.csv`
- `/Users/river/.copilot/session-state/103356e0-ba4d-4d59-8fe4-008b7dbc64f1/files/error-analysis/beam_slice_outcomes.csv`

### 报告

- `docs/experiments/2026-08-12/model_prediction_failure_case_study.md`
