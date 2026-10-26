# 历史兴趣与新兴趣关系：无模型数据分析

> **状态：已完成第一轮。**本报告不调用预测模型，只使用 Train 中冻结的频率、共现和
> 用户历史计数，在 Dev 与固定 Dev10K 上评估。分析对象是已有 GPT-5.5 重分类标签；
> “无模型”仅指本轮统计与预测过程。

## 1. 研究问题（Research Questions）

1. 数据中的“新兴趣”在一级类别上是延展，还是跨到新的一级类别？
2. 最后一个、最近五个或完整历史兴趣，对新兴趣有多强的统计关系？
3. 不使用模型，仅靠 Train 流行度、用户既往目标和历史条件共现，能在 Dev 上达到什么效果？
4. Train 与 Dev 的新闻周期漂移、用户重叠和静态 history 是否影响任务解释？
5. K=10 候选任务中的高分有多少来自正负样本频率差异，而不是历史条件信号？

## 2. 数据与定义

自然分布分析直接使用：

```text
data/processed/new_interest/train.parquet
data/processed/new_interest/dev.parquet
data/processed/new_interest/interest_vocabulary.parquet
```

每个 behavior 的已点击历史保留顺序和重复。当前 impression 中被点击、且精确二级兴趣
未出现在静态 history 中的主题被定义为新兴趣。一个 impression 可以有多个新兴趣；本分析
先按 impression 聚合，保留全部 target set，不使用 22K 类别均衡 Train。

同时运行实验对齐口径：最近 50 条 history、110 个保留标签、固定 grouped Dev10K。

| 口径 | Train impressions | Dev impressions | Dev target memberships |
|---|---:|---:|---:|
| 自然全历史、125 词表 | 1,219,553 | 199,644 | 243,118 |
| H50、110 标签、full Dev | 1,218,856 | 199,471 | 242,839 |
| H50、110 标签、固定 Dev10K | 1,218,856 | 10,000 | 12,164 |

## 3. 描述性结果

### 3.1 历史结构

| 指标 | Train | Dev |
|---|---:|---:|
| History 长度 mean / p50 / p90 | 24.79 / 14 / 59 | 24.32 / 14 / 58 |
| 唯一二级兴趣 mean / p50 / p90 | 12.38 / 9 / 26 | 12.19 / 9 / 26 |
| 唯一一级类别 mean / p50 / p90 | 7.31 / 7 / 13 | 7.23 / 7 / 13 |
| 含重复二级兴趣的 histories | 80.42% | 80.20% |
| 相邻重复兴趣的 histories | 60.65% | 60.50% |
| 平均重复项比例 | 31.77% | 31.70% |
| 多目标 impressions | 16.21% | 16.11% |

历史很长且重复较多，因此“完整历史中出现过某一级类别”本身是一个高覆盖事件，不能直接
作为关系强的证据。

### 3.2 二级兴趣新，但一级类别不一定新

精确二级 target 在 history 中的重叠为 0，符合构建定义。但 target 一级类别与 history 的
关系如下：

| 条件 | Train observed | 独立期望 | Dev observed | 独立期望 |
|---|---:|---:|---:|---:|
| Target parent 在完整历史中 | 51.12% | 52.39% | 50.43% | 53.07% |
| 等于最后一个 history parent | 8.69% | 8.86% | 8.47% | 9.39% |
| 出现在最近 5 条 parent 中 | 28.37% | 29.09% | 27.53% | 30.29% |
| 出现在最近 10 条 parent 中 | 38.53% | 39.57% | 37.24% | 40.75% |

约一半 target 属于 history 已覆盖的一级类别，但该比例没有高于由 history 覆盖率和 target
类别分布产生的独立期望，反而略低。因此不能把“50% 同一级类别”解释为强烈的同类延展；
更准确的描述是：**约一半是同大类的新子兴趣、约一半是跨大类，但一级类别关系整体很弱。**

### 3.3 历史到新兴趣的统计依赖很弱

互信息使用 Miller-Madow 一阶偏差修正。Train 样本量更大，作为主判断依据：

| 历史上下文 | Train MI | 修正后 MI | 解释 target entropy | 修正后解释比例 |
|---|---:|---:|---:|---:|
| 最后一个二级兴趣 | 0.0687 bit | 0.0613 bit | 1.13% | **1.00%** |
| 最近 5 条中的唯一兴趣 | 0.0534 bit | 0.0515 bit | 0.87% | **0.84%** |
| 完整历史中的唯一兴趣 | 0.0349 bit | 0.0343 bit | 0.56% | **0.55%** |
| 最后一级类别 → target 一级类别 | 0.0108 bit | 0.0107 bit | 0.28% | **0.28%** |

最近兴趣比完整历史更有信息，但仍只解释约 1% 的 target uncertainty。完整历史平均共现反而
稀释信号。

高支持的条件转移也不集中。Train 中最后兴趣对应的最高条件概率通常只有 5%–9%，而且许多
目标只是当日高频主题。例如：

| 最后兴趣 | 新兴趣 | P(target\|last) | Lift | 同 parent |
|---|---|---:|---:|---|
| Charity & Volunteering | Theater & Performing Arts | 8.53% | 1.88 | 否 |
| Baseball & Softball | American Football | 5.99% | 1.77 | 是 |
| Beauty & Grooming | Fashion & Personal Style | 5.53% | 2.17 | 是 |
| Celebrity & Creator Culture | Theater & Performing Arts | 5.40% | 1.19 | 是 |

这些是描述性关联，不应解释为因果转移。

### 3.4 新闻周期漂移很强

Train 覆盖 2019-11-09 至 2019-11-14，Dev 只有 2019-11-15。Target 分布变化：

| 指标 | 值 |
|---|---:|
| Train/Dev total variation distance | **0.4273** |
| Jensen-Shannon divergence | 0.2007 bit |
| Train/Dev Top-10 重叠 | 5/10 |
| Dev 质量落在 Train Top-1 | 0.043% |
| Dev 质量落在 Train Top-10 | 34.72% |

Train 总体第一名 `Theater & Performing Arts` 占 4.54%，到 Dev 只剩 0.043%。Dev 第一名
变为 `American Football`，占 11.57%。即使只用 Train 最后一天，第一名也变成
`Wildlife & Conservation`，仍不能稳定预测下一天。

### 3.5 用户与静态 history 风险

| 审计项 | 自然 full Dev | 固定 Dev10K |
|---|---:|---:|
| Train/Dev 重叠用户 | 120,330 | 7,360 |
| Dev impressions 来自重叠用户 | **74.63%** | **74.45%** |
| 重叠用户静态 history 完全相同 | **100%** | **100%** |
| 已见用户 Dev targets 曾在 Train 对该用户出现 | 13.33% | 13.30% |

此外，Train 内 5.93% target memberships 是同一用户相同 target 的重复记录；同一用户的
history 始终不更新。这意味着当前“新兴趣”是**相对静态 history 新**，并不保证相对用户
前面几天已经发生的点击新。后续数据构造应优先研究动态累积 history。

## 4. 无模型基线

所有规则只在 Train 拟合，并排除当前 history 中已有的精确兴趣。下表使用与模型完全相同的
H50、110 标签、固定 Dev10K，按 impression 宏平均：

| 无模型规则 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| Train 全局流行度 | 0.05% | 24.39% | 40.96% | 56.25% | 22.35% |
| Train 最后一天流行度 | 2.12% | 20.49% | 27.13% | 47.57% | 18.52% |
| 同用户 Train target 频率 | **5.06%** | 23.18% | **41.84%** | 57.02% | **24.51%** |
| 最后兴趣条件频率 | 2.96% | **25.45%** | 41.54% | **59.28%** | 24.42% |
| 最后一级类别条件频率 | 1.99% | 25.22% | 40.15% | 58.79% | 23.90% |
| 最近 5 条条件概率平均 | 0.63% | 25.21% | 40.94% | 59.12% | 23.44% |
| 完整历史条件概率平均 | 0.03% | 20.42% | 36.67% | 54.52% | 19.84% |

这些 Top-K 指标受到 Dev 流行度高度集中的影响。最值得关注的是：最后兴趣条件规则相对全局
流行度在 Recall@20 只提高约 3.03 个百分点；完整历史规则反而更差。

### K=10 候选集

| 数据 | 无模型规则 | Accuracy |
|---|---|---:|
| 原始均匀负样本 12,164 | Train 全局流行度 | 31.99% |
| 原始均匀负样本 12,164 | 同用户 Train target | 32.50% |
| 原始均匀负样本 12,164 | 最后兴趣条件频率 | **33.37%** |
| 原始均匀负样本 12,164 | 去流行度后的最后兴趣 Lift | 13.61% |
| 频率匹配诊断 1,000 | Train 全局流行度 | 8.80% |
| 频率匹配诊断 1,000 | 同用户 Train target | 11.30% |
| 频率匹配诊断 1,000 | 最后兴趣条件频率 | 10.00% |
| 频率匹配诊断 1,000 | 去流行度后的最后兴趣 Lift | **13.50%** |

原始候选集的 32%–33% 主要来自正样本长尾分布和近均匀负采样。频率匹配后，普通历史条件
频率回到随机水平，去流行度 Lift 只有 13.5% 且仅 `n=1,000`，说明存在少量关系信号，
但远弱于原始 benchmark 展示的流行度捷径。频率匹配结果使用完整 125 词表拟合；该诊断
文件含 6 个不在当前 retained-110 中的负候选，不过所有正答案都属于 retained-110。

## 5. 当前判断与下一步

当前数据支持以下判断：

1. 新二级兴趣与静态历史存在少量最近兴趣关联，但整体可预测性弱。
2. 一级类别延展没有超过独立期望，不能把同 parent 比例直接解释为偏好延展。
3. 新闻日期/主题流行度是强干扰因素；K=10 原始负采样尤其放大了该捷径。
4. 同一用户的 history 在 Train/Dev 不更新，使一部分“新兴趣”实际上是此前已经发生过的主题。
5. 后续应先重建动态 history，再讨论模型是否学到了真正的跨兴趣迁移。

建议下一轮按以下顺序推进：

1. 从原始 behaviors 按用户和时间累积所有先前点击，重算动态新兴趣。
2. 分开评估 unseen user、seen user 但 target 从未出现、seen user repeated target。
3. 构造按日期滚动的 Train/validation/test，避免在同一 Dev 上选择统计规则。
4. K=10 同时报告均匀负样本、频率匹配负样本和多个排列 seed。
5. 在动态数据上再比较 global popularity、recent popularity、last-interest lift 和模型。

## 6. 数据与产物路径

- 分析代码：`src/new_interest/experiments/analyze_interest_relationships.py`
- 测试：`tests/test_interest_relationship_analysis.py`
- 自然全历史结果：`artifacts/interest-relationship-analysis/natural-full-history/`
- H50/110 full Dev：`artifacts/interest-relationship-analysis/sft-h50-retained110/`
- H50/110 固定 Dev10K：`artifacts/interest-relationship-analysis/sft-dev10k-h50-retained110/`
- K=10 完整 125 词表诊断：`artifacts/interest-relationship-analysis/candidate-dev10k-h50-full125/`
- 主汇总：各目录中的 `summary.json`
- 无模型 Top-K：各目录中的 `dev_baseline_metrics.csv`
- K=10 无模型结果：`candidate-dev10k-h50-full125/candidate_baseline_metrics.csv`
- Train 关系明细：各目录中的 `train_relationships.csv`
- Train/Dev 目标分布：各目录中的 `target_distribution.csv`
