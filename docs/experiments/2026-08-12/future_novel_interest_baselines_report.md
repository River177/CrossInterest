# Future Observed Novel-Interest Baselines：双 Track 实验报告

日期：2026-08-12

## 1. 结论摘要

本实验将任务重新定义为：

```text
固定 MIND 用户画像
→ 未来有标签日志中实际观测到、且历史中未出现的兴趣集合
```

全量结果表明：

1. **任务不是纯随机噪声。** 在 Exposure-restricted Track 上，简单
   `global_popularity` 已达到 Recall@20 `0.7957`；使用 history embedding 的
   masked multi-label 模型达到 `0.7997`。固定 history 对未来兴趣存在可学习信号。
2. **现有信号主要由曝光范围和全局流行度解释。** Exposure-restricted 中，
   embedding multi-label 相对 popularity 的 Recall@20 仅提高
   `+0.0041`，而 NDCG@20 反而降低 `-0.0038`。
3. **Full-taxonomy 仍未解决。** 在 125 类中直接检索时，global popularity 的
   Recall@20 为 `0.6358`，明显高于最好的个性化模型 `0.4307`。当前个性化方法
   还不能在没有未来曝光集合帮助时超过流行度先验。
4. **个性化模型的价值主要体现在长尾覆盖，而不是微平均命中。**
   Full-taxonomy 的 Macro Recall@20：popularity `0.2310`，
   embedding multi-label `0.3736`；后者还把 top-1 预测覆盖到 124 个类别，
   而 popularity 只覆盖 18 个类别。
5. **conditional lift 证明“去流行度”不能直接等同于“个性化”。**
   它在 Exposure-restricted 的 Macro Recall@20 高达 `0.7029`，但微平均
   Recall@20 只有 `0.5889`；在 Full-taxonomy 更下降到 `0.0311`，因为极稀有或
   无训练正例标签被系统性排到前面。

因此，当前最重要的下一步不是 GRPO，而是构建并验证一个
**流行度校准的个性化混合模型**，同时优化微平均有效性与长尾覆盖。

## 2. 研究问题

给定用户在 MIND `behaviors.tsv` 中携带的固定历史点击画像 \(H(u)\)，预测合并
Train/Dev 有标签未来窗口中，用户实际点击但历史中未出现的主题：

\[
Y_{\text{new}}(u)=F_{\text{clicked}}(u)\setminus H(u)
\]

本轮回答三个问题：

1. 固定 history 是否包含可预测未来新兴趣的信号？
2. 差结果主要来自随机噪声，还是流行度、曝光偏差、长尾和多标签不确定性？
3. 在进一步使用 LLM、SFT 或 GRPO 前，什么可解释 baseline 必须先被超过？

## 3. 数据构造

### 3.1 样本单位与划分

- 一行对应一个 raw-MIND user，而不是一个 impression-target；
- 同一用户的 MIND history 必须完全一致；
- Train 与 Dev 的有标签 behaviors 合并为完整未来观察窗口；
- 点击顺序和重复点击原样保留，不做 news ID 或时间窗口去重；
- 多标签 target set 内主题自然唯一，这不等于删除点击事件；
- 使用 ordered history news IDs 的 SHA-256 `profile_hash` 分组；
- 完全相同的输入画像不能跨 Train/Validation/Test。

| 项目 | 数值 |
|---|---:|
| Users | 736,340 |
| Exact profile groups | 726,561 |
| Train / Validation / Test | 589,588 / 73,588 / 73,164 |
| Taxonomy | 125 |
| Test users with positive novel targets | 62,687 / 73,164 (85.68%) |
| Mean fixed-history length | 18.64 |
| Mean novel targets，all users | 2.194 |
| Mean novel targets，positive Test users | 2.547 |
| Mean Exposure-restricted candidates，positive Test users | 32.91 |
| Mean Full-taxonomy candidates，positive Test users | 114.60 |

Test 中没有未来 novel target 的 10,477 个用户不进入排名指标，但其数量和比例保留
在 metrics 中。

### 3.2 Positive、weak negative 与 unknown

- **Positive**：未来点击、且不在固定 history 中的主题；
- **Weak negative**：未来实际曝光、历史中未出现、并且在完整未来窗口中从未点击的主题；
- **Unknown**：未来从未曝光的主题。

如果同一主题下某篇新闻被点击、另一篇未点击，该主题只标为 positive，不能同时作为
negative。

### 3.3 两个评测 Track

**Exposure-restricted**

```text
candidates = future exposed topics - history topics
positives  = future clicked topics - history topics
```

这是条件于真实未来曝光集合的诊断 Track。它回答“在日志实际给用户看过的兴趣中，
history 能否帮助排序”，但不能被解释成一个可直接部署的开放候选生成器。

**Full-taxonomy**

```text
candidates = all 125 topics - history topics
positives  = future clicked topics - history topics
```

未曝光主题在此仍是 unknown，不是真负例。该 Track 更接近“从整个主题空间提出未来可能
兴趣”，但指标只验证日志中观测到的 positives，并不覆盖用户全部真实偏好。

## 4. Baselines

| Baseline | 输入与打分 |
|---|---|
| `global_popularity` | Train distinct-user positive support，经 Laplace smoothing |
| `conditional_frequency` | 聚合 \(P(\text{future topic}\mid\text{history topic})\) |
| `conditional_lift` | conditional frequency 除以 target prior |
| `embedding_cosine` | history title 均值向量与 topic 文本向量 cosine |
| `multilabel_topic_logistic` | 125 维 history topic frequency + masked weighted BCE |
| `multilabel_embedding_logistic` | 384 维 history title embedding + masked weighted BCE |

Embedding 模型固定为：

```text
sentence-transformers/all-MiniLM-L6-v2
revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

两个 multi-label 模型只把 positive 和 exposed-but-unclicked weak negatives 纳入
loss；never-exposed labels 被 mask。topic 模型训练 9 epochs，embedding 模型训练
8 epochs，均由 Validation loss early stopping。

`Global Institutions & Cooperation` 在 Train 中没有 positive support，两个
multi-label 模型对此使用显式平滑常数 prior，不伪造可训练 classifier。

## 5. 主结果

指标在 62,687 个 positive Test users 上计算。完整浮点结果、全部 Hit 指标和每用户
bootstrap 明细见配套 CSV/JSON。

### 5.1 Exposure-restricted

| Model | R@1 | R@5 | R@10 | R@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global popularity | 0.1197 | 0.4003 | 0.5883 | 0.7957 | 0.2281 | 0.3201 | 0.3920 | 0.4604 |
| conditional frequency | 0.1202 | 0.3879 | 0.5838 | 0.7986 | 0.2271 | 0.3081 | 0.3835 | 0.4544 |
| conditional lift | 0.0605 | 0.2289 | 0.3738 | 0.5889 | 0.0929 | 0.1663 | 0.2210 | 0.2890 |
| embedding cosine | 0.0987 | 0.3350 | 0.5123 | 0.7344 | 0.1753 | 0.2589 | 0.3266 | 0.3992 |
| topic multi-label | 0.1048 | 0.3515 | 0.5300 | 0.7504 | 0.1872 | 0.2735 | 0.3416 | 0.4139 |
| embedding multi-label | **0.1211** | 0.3935 | 0.5846 | **0.7997** | 0.2220 | 0.3118 | 0.3850 | 0.4566 |

| Model | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 | Macro R@1 | Macro R@5 | Macro R@10 | Macro R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global popularity | 0.2281 | 0.6304 | 0.8098 | 0.9360 | 0.4082 | 0.0201 | 0.0994 | 0.1901 | 0.3627 |
| conditional frequency | 0.2271 | 0.6131 | 0.8060 | 0.9378 | 0.3973 | 0.0205 | 0.0992 | 0.1934 | 0.3694 |
| conditional lift | 0.0929 | 0.3516 | 0.5499 | 0.7769 | 0.2236 | 0.1403 | 0.3558 | 0.5049 | **0.7029** |
| embedding cosine | 0.1753 | 0.5300 | 0.7354 | 0.9055 | 0.3395 | 0.0545 | 0.2205 | 0.3688 | 0.5984 |
| topic multi-label | 0.1872 | 0.5534 | 0.7499 | 0.9137 | 0.3548 | 0.0734 | 0.2766 | 0.4483 | 0.6834 |
| embedding multi-label | 0.2220 | 0.6142 | 0.8027 | **0.9397** | 0.3973 | 0.0696 | 0.2562 | 0.4043 | 0.6312 |

Exposure mask 将平均候选空间从 114.60 降至 32.91，因此所有模型都明显受益。最好的
微平均 Recall@20 只比 popularity 高约 0.4 个百分点，说明该 Track 的大部分可预测性
来自“未来实际曝光了什么”，而不是 history 本身提供的强个性化信号。

### 5.2 Full-taxonomy

| Model | R@1 | R@5 | R@10 | R@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global popularity | **0.0668** | **0.2727** | **0.4377** | **0.6358** | **0.1486** | **0.2155** | **0.2785** | **0.3426** |
| conditional frequency | 0.0432 | 0.2363 | 0.4118 | 0.6197 | 0.0989 | 0.1804 | 0.2471 | 0.3143 |
| conditional lift | 0.0000 | 0.0012 | 0.0070 | 0.0311 | 0.0000 | 0.0007 | 0.0030 | 0.0109 |
| embedding cosine | 0.0266 | 0.1020 | 0.1746 | 0.3026 | 0.0625 | 0.0830 | 0.1107 | 0.1516 |
| topic multi-label | 0.0250 | 0.1012 | 0.1708 | 0.2813 | 0.0594 | 0.0815 | 0.1079 | 0.1432 |
| embedding multi-label | 0.0392 | 0.1503 | 0.2585 | 0.4307 | 0.0964 | 0.1236 | 0.1645 | 0.2196 |

| Model | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 | Macro R@1 | Macro R@5 | Macro R@10 | Macro R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global popularity | **0.1486** | **0.4933** | **0.6778** | **0.8311** | **0.3043** | 0.0116 | 0.0636 | 0.1246 | 0.2310 |
| conditional frequency | 0.0989 | 0.4438 | 0.6520 | 0.8198 | 0.2619 | 0.0082 | 0.0583 | 0.1212 | 0.2314 |
| conditional lift | 0.0000 | 0.0032 | 0.0185 | 0.0806 | 0.0067 | 0.0000 | 0.0334 | 0.0910 | 0.2080 |
| embedding cosine | 0.0625 | 0.2222 | 0.3514 | 0.5293 | 0.1456 | 0.0239 | 0.0995 | 0.1748 | 0.3035 |
| topic multi-label | 0.0594 | 0.2194 | 0.3399 | 0.4960 | 0.1404 | 0.0322 | 0.1170 | 0.2011 | 0.3301 |
| embedding multi-label | 0.0964 | 0.3100 | 0.4671 | 0.6539 | 0.2034 | **0.0358** | **0.1404** | **0.2312** | **0.3736** |

Full-taxonomy 暴露了当前方法的核心缺口：个性化模型分散了预测质量，却没有足够准确地
把用户级信号集中到真实 positives 上。global popularity 因为未来点击分布高度长尾，
在 micro metrics 上仍是最强 baseline。

## 6. Paired bootstrap

以 global popularity 为 reference，按同一批用户进行 2,000 次 paired bootstrap，
95% percentile interval：

| Track | Model | Δ Recall@20 | Δ NDCG@20 |
|---|---|---:|---:|
| Exposure | conditional frequency | +0.0030 [+0.0021, +0.0039] | -0.0060 [-0.0069, -0.0052] |
| Exposure | conditional lift | -0.2068 [-0.2106, -0.2033] | -0.1714 [-0.1741, -0.1687] |
| Exposure | embedding cosine | -0.0613 [-0.0640, -0.0584] | -0.0612 [-0.0637, -0.0587] |
| Exposure | topic multi-label | -0.0452 [-0.0482, -0.0424] | -0.0466 [-0.0490, -0.0441] |
| Exposure | embedding multi-label | **+0.0041 [+0.0015, +0.0066]** | -0.0038 [-0.0062, -0.0014] |
| Full | conditional frequency | -0.0161 [-0.0179, -0.0143] | -0.0283 [-0.0293, -0.0273] |
| Full | conditional lift | -0.6047 [-0.6079, -0.6016] | -0.3316 [-0.3337, -0.3297] |
| Full | embedding cosine | -0.3332 [-0.3371, -0.3292] | -0.1910 [-0.1933, -0.1885] |
| Full | topic multi-label | -0.3545 [-0.3585, -0.3507] | -0.1994 [-0.2018, -0.1972] |
| Full | embedding multi-label | -0.2051 [-0.2088, -0.2010] | -0.1230 [-0.1255, -0.1205] |

Exposure 中 embedding multi-label 的 Recall@20 增益统计上稳定，但数值很小，并且没有
转化为更好的 NDCG。它更像是“在 top-20 边缘多覆盖少量 positives”，而不是整体排序
改善。

## 7. Head / torso / tail

按 **Train distinct-user positive support** 降序排列 125 个标签，再按标签数量近似三等分：

- head：42 类，Train support 10,984–54,594；
- torso：42 类，Train support 2,653–10,325；
- tail：41 类，Train support 0–2,363。

下表是各 tier 的 per-label Macro Recall@20：

| Model | Exposure head | Exposure torso | Exposure tail | Full head | Full torso | Full tail |
|---|---:|---:|---:|---:|---:|---:|
| global popularity | **0.8323** | 0.1522 | 0.0838 | **0.6645** | 0.0120 | 0.0001 |
| conditional frequency | **0.8383** | 0.1658 | 0.0836 | 0.6596 | 0.0178 | 0.0002 |
| conditional lift | 0.4076 | **0.8087** | **0.9069** | 0.0044 | 0.1326 | **0.5086** |
| embedding cosine | 0.6465 | 0.5821 | 0.5641 | 0.2931 | 0.2971 | 0.3217 |
| topic multi-label | 0.6417 | 0.6831 | 0.7284 | 0.2418 | **0.3646** | 0.3881 |
| embedding multi-label | 0.7533 | 0.5946 | 0.5390 | 0.4403 | 0.3538 | 0.3232 |

这说明：

- popularity 几乎只解决 head；
- lift 几乎只解决 tail，并牺牲绝大多数用户的常见 positives；
- embedding multi-label 是当前最均衡的模型，但仍未找到比 popularity 更好的整体
  calibration。

## 8. Target count 与 history length 切片

下表比较 global popularity 与当前最强个性化模型的 Recall@20：

| Track | Slice | N | Popularity | Embedding multi-label |
|---|---|---:|---:|---:|
| Exposure | 1 target | 23,439 | 0.8825 | **0.8955** |
| Exposure | 2 targets | 15,800 | 0.8137 | **0.8217** |
| Exposure | 3–5 targets | 18,567 | **0.7198** | 0.7148 |
| Exposure | 6+ targets | 4,881 | **0.6088** | 0.5918 |
| Full | 1 target | 23,439 | **0.6688** | 0.4386 |
| Full | 2 targets | 15,800 | **0.6393** | 0.4302 |
| Full | 3–5 targets | 18,567 | **0.6103** | 0.4248 |
| Full | 6+ targets | 4,881 | **0.5630** | 0.4171 |

| Track | History length | N | Popularity | Embedding multi-label |
|---|---|---:|---:|---:|
| Exposure | 1–10 | 30,720 | **0.8395** | 0.8376 |
| Exposure | 11–25 | 18,117 | 0.7823 | **0.7925** |
| Exposure | 26–50 | 8,898 | 0.7244 | **0.7322** |
| Exposure | 51–100 | 3,900 | 0.6992 | **0.7071** |
| Exposure | 101+ | 1,052 | 0.7081 | **0.7324** |
| Full | 1–10 | 30,720 | **0.6534** | 0.3936 |
| Full | 11–25 | 18,117 | **0.6206** | 0.4475 |
| Full | 26–50 | 8,898 | **0.6159** | 0.4784 |
| Full | 51–100 | 3,900 | **0.6140** | 0.5042 |
| Full | 101+ | 1,052 | **0.6327** | 0.5506 |

多 target 用户明显更难；固定 history 并不能唯一确定未来会点击哪一种新兴趣。
另一方面，Full-taxonomy 中 embedding multi-label 随 history 变长持续改善，说明更丰富
的历史确实提供了个性化信号，只是当前模型的校准仍弱于 popularity。

## 9. 预测多样性

| Model | Exposure unique top-1 / max share | Full unique top-1 / max share |
|---|---:|---:|
| global popularity | 46 / 51.36% | 18 / 60.55% |
| conditional frequency | 47 / 35.28% | 15 / 99.45% |
| conditional lift | 124 / 6.85% | 3 / 99.67% |
| embedding cosine | 120 / 6.95% | 120 / 6.19% |
| topic multi-label | 120 / 9.28% | 118 / 11.06% |
| embedding multi-label | 121 / 5.89% | **124 / 4.99%** |

多样性本身不等于正确性，但它揭示了 micro metrics 的代价：popularity 的强结果建立在
高度集中的少量 head labels 上。未来方法需要明确优化 accuracy–coverage trade-off，
不能只看一个总体 Recall。

## 10. Case studies

以下案例均来自 Test benchmark；英文标题为 raw MIND 标题，中文为忠实翻译。

### Case A：跨同一语义域的新兴趣是可学的

用户 `U100171`，固定 history：

| English | 中文 |
|---|---|
| NFL suspends ex-Ravens RB Collins 3 games for spring arrest | NFL 因春季被捕事件将前乌鸦队跑卫 Collins 禁赛三场 |
| Mary Cain details alleged physical and emotional abuse at the hands of Nike, Alberto Salazar | Mary Cain 讲述其声称遭 Nike 与 Alberto Salazar 身体和情感虐待的经历 |

History topics：`American Football`、`Olympic & Athletics Sports`。  
未来点击：`Report: Carmelo Anthony returning to NBA with Trail Blazers`
（报道称 Carmelo Anthony 将加盟开拓者重返 NBA），target 为 `Basketball`。

| Model | Exposure target rank | Full target rank |
|---|---:|---:|
| global popularity | 5 | 17 |
| conditional lift | 1 | 44 |
| topic multi-label | 1 | **1** |
| embedding multi-label | 2 | 3 |

这是相对“干净”的 interest transition：不同运动项目之间存在稳定关系。Full-taxonomy
中 topic multi-label 能把 Basketball 排到第 1，证明 history signal 并非不存在。

### Case B：同一固定 history 对应多个不同语义模式

用户 `U100006` 的三条 history 同时涉及 celebrity/hair 与 American football：

| English | 中文 |
|---|---|
| Back to Blonde! Britney Spears Celebrates New Golden Locks... | 重回金发：Britney Spears 展示新染的金色头发 |
| Freshman Georgia Southern offensive lineman Jordan Wiggins dies at 18 | Georgia Southern 大一进攻锋线球员 Jordan Wiggins 18 岁去世 |
| AP Top 25: LSU No. 1 ahead of 'Bama... | 美联社 Top 25：LSU 超过 Alabama 位列第一 |

未来 novel targets 有五个：

```text
Basketball
Beauty & Grooming
Charity & Volunteering
Crime & Public Safety
Medicine & Medical Research
```

Exposure-restricted 中：

- embedding cosine：Basketball rank 1、Beauty rank 5，但 Crime rank 62；
- embedding multi-label：Crime rank 3、Basketball rank 4、Medicine rank 13，
  但 Beauty rank 54、Charity rank 37；
- popularity：Crime rank 7、Medicine rank 10、Basketball rank 16。

不同模型分别抓住体育、外貌和社会新闻模式，没有一个单向量排序能同时覆盖所有 future
interests。这里的主要问题不是标签错误，而是 **multi-modal user intent 被压缩成一个
向量和一个排序**。

### Case C：文章级桥接关系会被主题级模型丢失

用户 `U100094` 的 history 包含汽车、退休理财、American football、野生动物和犯罪：

| English | 中文 |
|---|---|
| Man Reveals Big American Muscle Car Barn Find | 男子展示在谷仓中发现的大批美式肌肉车 |
| 9 Types of Retirement Income That Are Tax-Free | 九类免税退休收入 |
| One Man Has Owned This 1964 Shelby Cobra for 50 Years | 一名男子拥有这辆 1964 Shelby Cobra 已 50 年 |

四个未来 novel targets：

- `Film`：*Ford v Ferrari: the real story*（《极速车王》背后的真实故事）；
- `Shopping & Gift Ideas`：低于 25 美元的热门礼物；
- `Civic Life & Public Commemorations`：97 岁二战老兵乘坐 B-17；
- `Elections & Political Campaigns`：Tim Ryan 支持 Biden。

Exposure-restricted 中，popularity 只把 Shopping 排到第 3，其余 targets 位于
15–26；embedding multi-label 的四个 ranks 为 24、26、53、54。汽车 history 与
*Ford v Ferrari* 在文章内容层面高度相关，但 target taxonomy 是 `Film`。单纯把
history 压成 topic counts 或把 topic 名称作为检索目标，会丢失这种
**real-content bridge**。

这支持后续采用“生成/检索真实内容，再映射到兴趣 taxonomy”的两阶段方法，而不是直接
在 125 个抽象标签上做一次性分类。

## 11. 对“数据噪声”的判断

当前差结果不能归因于一个单独的“脏数据比例”。更准确的分解是：

1. **一对多不确定性**：同一固定 history 合理地对应多个未来兴趣；
2. **曝光选择偏差**：只有被平台展示的内容才有机会成为 observed positive；
3. **unknown 大量存在**：Full-taxonomy 中未曝光主题不能当作负例；
4. **极端长尾**：micro 指标偏好 head，macro 指标又可能奖励过度 tail ranking；
5. **标签抽象损失**：文章级语义桥接不一定能从 topic name 直接恢复；
6. **固定画像陈旧性**：MIND history 没有覆盖完整、动态的近期行为；
7. **弱负例不等于真实不喜欢**：未点击还受位置、时机、标题和竞争候选影响。

因此，本 benchmark 更适合研究“在有偏离线观察中恢复未来可能兴趣”，而不是宣称模型
预测了完整真实偏好，更不能据此作“创造兴趣”或“减少 filter bubble”的因果结论。

## 12. 下一步

### 12.1 首要实验：popularity-calibrated personalization

在 Validation 上分别为两个 Track 调整：

\[
s(u,t)=\alpha\log P_{\text{train}}(t)
 +(1-\alpha)s_{\text{personal}}(u,t)
\]

其中 `s_personal` 优先使用 embedding multi-label。需要同时报告：

- micro Recall/NDCG；
- Macro Recall；
- head/torso/tail；
- top-1 coverage 与最大标签占比；
- calibration 曲线。

这是当前最小、最关键的实验：验证能否保留 popularity 的 head accuracy，同时获得
个性化模型的 torso/tail coverage。

### 12.2 用户表示：从单向量改为多兴趣 mixture

将 history 分成多个语义簇或多个 interest vectors，对每个 future topic 使用
max/mixture scoring，避免 Case B 中体育、外貌和社会新闻信号互相平均。

### 12.3 内容 grounding

研究路线应转为：

```text
Novel-interest/content generator
→ user alignment ranker
→ real-news grounding
→ taxonomy mapping for evaluation
```

Case C 说明直接预测抽象标签会损失文章级桥接关系。

### 12.4 暂缓 GRPO

在以下条件满足前，不应把主要资源投入 GRPO：

1. 个性化混合模型在 Full-taxonomy 至少超过 popularity；
2. 改善同时存在于 micro 与 macro，而不是只把质量移到 tail；
3. 真实内容 grounding 比直接标签分类提供稳定增益；
4. reward 能区分 observed positive、weak negative 和 unknown。

## 13. 复现与产物路径

### 数据

- Benchmark：
  `data/processed/future_novel_interest_users_v1/`
- Config：
  `configs/future_interest/benchmark_v1.json`
- Builder：
  `src/new_interest/data/build_future_interest_benchmark.py`

### Baselines

- Config：
  `configs/future_interest/baselines_v1.json`
- Manifest：
  `artifacts/future-novel-interest/baselines-v1/manifest.json`
- 每个 baseline 的 predictions、两个 Track 的 metrics/details：
  `artifacts/future-novel-interest/baselines-v1/<baseline>/`

### 报告

- 本报告：
  `docs/experiments/2026-08-12/future_novel_interest_baselines_report.md`
- 完整比较 CSV：
  `docs/experiments/2026-08-12/future_novel_interest_baselines_artifacts/future_novel_interest_baselines_metrics.csv`
- 完整比较、paired bootstrap、label support 与 label tiers JSON：
  `docs/experiments/2026-08-12/future_novel_interest_baselines_artifacts/future_novel_interest_baselines_metrics.json`
- Manifest 自动生成的机器报告草稿：
  `docs/experiments/2026-08-12/future_novel_interest_baselines_artifacts/future_novel_interest_baselines_generated.md`

### 命令

```bash
PYTHONPATH=src python -m \
  new_interest.experiments.run_future_interest_baselines \
  configs/future_interest/baselines_v1.json

PYTHONPATH=src python -m \
  new_interest.experiments.report_future_interest_baselines \
  artifacts/future-novel-interest/baselines-v1/manifest.json \
  docs/experiments/2026-08-12/future_novel_interest_baselines_artifacts
```

本实验不使用 LLM judge。所有结论来自确定性排名指标、真实 Test rows 和 paired
user bootstrap。
