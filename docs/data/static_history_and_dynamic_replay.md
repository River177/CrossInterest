# MIND 多个 Impression 共享静态 History：旧数据处理、遗漏原因与动态修复

## 1. 文档目的

本文说明 CrossInterest 早期数据集中的一个关键方法问题：

> 原始 MINDlarge 中，同一用户的多个 impression 各自包含一个 `history` 字段，
> 但这些字段不是随 impression 更新的滚动历史；它们是完全相同的用户级固定快照。

旧 pipeline 直接使用每行携带的 `history` 判断当前点击是否为“新兴趣”，没有把该
用户此前 impression 中的正点击加入后续历史。因此，某个主题即使已经在较早
impression 中被点击，只要它不在固定快照里，之后再次点击时仍可能被标为
`novel`。

本文重点回答：

1. 我们之前具体如何构建数据；
2. 为什么之前没有考虑同一用户在更早 impression 中点击过的新闻；
3. 该问题产生了什么影响；
4. 当前动态版本如何修复。

## 2. 原始 Behavior 行表示什么

原始文件：

```text
data/raw/MINDlarge_reclassified/train/behaviors.tsv
data/raw/MINDlarge_reclassified/dev/behaviors.tsv
```

每行包含五个字段：

```text
impression_id
user_id
timestamp
history
impressions
```

示意：

```text
123    U1    11/15/2019 12:00:00 PM    N10 N20 N30    N40-0 N50-1 N60-0
```

字段含义：

- `history`：一串此前点击新闻的 news ID；
- `impressions`：本次曝光的新闻；
- `N50-1`：本次曝光中点击了 N50；
- `N40-0`、`N60-0`：新闻被曝光但没有点击。

### 2.1 MIND 官方如何定义 History

MIND 官方数据格式文档对 `history` 的原文定义是：

> “The news click history (ID list of clicked news) of this user before this
> impression. The clicked news articles are ordered by time.”

即：

> 该用户在当前 impression 之前的新闻点击历史，点击新闻按时间排序。

来源：

- [MIND official introduction and data format](https://msnews.github.io/assets/doc/introduction.md)
- [Microsoft Learn: Microsoft News Recommendation Dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)

如果只看这句话，很容易把每一行的 `history` 理解为截至该 impression 已经更新完成
的滚动历史。这也是旧 pipeline 最初采用的解释。

### 2.2 官方数据构造窗口给出的更具体含义

官方说明和 MIND 论文同时描述了时间窗口：

- 使用前四周的点击行为构建用户新闻点击历史；
- 第五周的行为用于训练；
- 第五周最后一天用于验证；
- 第六周用于测试。

来源：

- [MIND official introduction](https://msnews.github.io/assets/doc/introduction.md)
- [MIND paper, ACL 2020](https://aclanthology.org/2020.acl-main.331.pdf)

因此，在发布的数据协议中，更稳妥的解释是：

```text
history
= 训练/评测行为窗口开始前，由前四周点击构建的有序用户画像
```

官方文档没有明确承诺：

```text
第五周内较早 impression 的点击
会被追加到第五周内较晚 impression 的 history 字段
```

Microsoft Recommenders 的官方生态 baseline loader 也逐行读取文件中已有的
`history`，进行长度截断后送入模型，不会自行把当前 impression 点击追加到后续
行。这说明逐行使用固定 history 是标准 MIND baseline 的常见协议，但 loader 本身
不是上游生成实现的证明。

- [Microsoft Recommenders MINDIterator](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/recommenders/models/newsrec/io/mind_iterator.py)

综上：

> 旧 pipeline 不是错误读取了 MIND 文件格式，而是遵循了标准的 row-provided
> history 协议；问题在于我们把该协议下的“历史中未出现”进一步解释成了用户完整
> 时间线上的“真正新兴趣”。

本文说的“此前 impression 看过的新闻”，在动态点击历史中严格指此前
impression 的 **正点击新闻**。未点击的 `-0` 新闻属于曝光信息，不应被加入点击
history。

## 3. 全量证据：同一用户的 History 完全不变

我们独立扫描了原始 Train 和 Dev `behaviors.tsv`，对每个用户每行的完整 history
news ID 字符串计算 SHA-256；比较的是完整顺序和全部 ID，不只是长度、集合或主题。

### 3.1 分 split 结果

| split | behavior rows | users | 多-impression users | 这些用户的 rows | history 变化 users | 单用户最多 impressions |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,186,664 | 698,357 | 467,986 | 1,956,293 | **0** | 106 |
| Dev | 365,199 | 248,971 | 73,859 | 190,087 | **0** | 27 |

### 3.2 Train 与 Dev 合并结果

```text
behavior rows:                     2,551,863
unique users:                        736,340
Train/Dev overlapping users:         210,988
users with multiple impressions:     521,158
rows from multi-impression users:  2,336,681
users with any history mismatch:            0
maximum impressions for one user:         107
```

跨 Train/Dev 重叠用户的 history 也没有变化。

因此，更准确的描述不是“多个 impression 没有 history”，而是：

> 每个 impression 行都带有 history，但同一用户的所有行重复携带同一份 history
> 快照。

### 3.3 真实用户例子

用户 `U536528` 有 107 个 impression。其固定 history 包含 730 条新闻。

这里的 107 指原始 `behaviors.tsv` 中的 107 条推荐曝光日志，不应解释为 107 次
相互独立的兴趣选择或 107 个独立 session。该用户的日志细分为：

```text
time range: 2019-11-09 05:00:10 -- 2019-11-15 05:59:23
recommendation impression rows: 107
total exposed candidate entries: 1,500
positive click labels: 107
unique clicked news IDs: 48
```

18 个 news ID 被重复点击标记。部分记录在很短时间内连续指向同一点击新闻，例如
`N118589` 在 53 秒内出现在 5 条不同 impression 中；全体 107 行中有 29 对相邻
同-news 点击发生在 60 秒以内。原始数据本身无法区分这些记录究竟来自真实的重复
曝光/重复点击、页面刷新，还是日志层面的重复。因此，107 是日志事件数，不是 107
个独立监督信号。动态 novelty 判断使用已见主题集合，所以同一主题第一次出现后，
这些重复记录会被判为 repeated；但构建 prompt 时是否还要对重复 news event 去重，
应作为独立的数据清洗问题处理。

```text
Train impression 3030
timestamp: 2019-11-12 10:02:05
history prefix:
N59450 N90402 N27587 N109736 N43373 N125282 N94295 N120773 ...

Train impression 24808
timestamp: 2019-11-10 06:47:03
history prefix:
N59450 N90402 N27587 N109736 N43373 N125282 N94295 N120773 ...

Train impression 33987
timestamp: 2019-11-09 05:00:10
history prefix:
N59450 N90402 N27587 N109736 N43373 N125282 N94295 N120773 ...
```

不仅前缀相同，完整 730 个 news ID 及其顺序都完全相同。后续 impression 的正点击
不会自动出现在另一行的原始 `history` 字段中。

## 4. 我们之前是怎么做的

旧基础 builder：

```text
src/new_interest/data/build_dataset.py
```

旧输出：

```text
data/processed/new_interest/
```

### 4.1 逐行独立读取

旧代码顺序读取每个 behavior，但每一行彼此独立：

```python
for row in behaviors:
    impression_id, user_id, time, raw_history, raw_impressions = row
    history = parse(raw_history)
    positive_news = parse_clicked_news(raw_impressions)
```

`user_id` 和 `timestamp` 被写入输出，但没有用来：

- 对同一用户的事件排序；
- 建立用户状态；
- 把较早 impression 的点击追加到后续 history；
- 把 Train 结束时的状态传入 Dev。

### 4.2 只相对当前行的 raw history 判断 novelty

旧规则为：

```python
history_interests = interests(raw_history)
current_positive_interests = interests(current_clicked_news)

new_targets = [
    target
    for target in current_positive_interests
    if target not in history_interests
]
```

数学上等价于：

```text
old_novel(u, impression)
  = current_clicked_interests
    - interests(raw_history_field)
```

它没有计算：

```text
current_clicked_interests
- interests(all earlier positive clicks from this user)
```

### 4.3 一个 impression 被展开成多行

如果一次 impression 点击了多个 raw-history 中未出现的主题，旧基础数据会按 target
展开：

```text
one impression
  -> target A row
  -> target B row
  -> target C row
```

因此旧 `new_interest/train.parquet` 和 `dev.parquet` 是 target-membership
粒度，不是 impression 粒度。

### 4.4 后续平衡进一步选择一个 target

旧生成式平衡数据：

```text
data/training/generative_new_interest_balanced_200_h50/
```

处理步骤：

1. 对 Train 中每个 impression 确定性选择一个 target；
2. 保留至少有 200 条 Train 样本的主题；
3. 每类采样 200 条；
4. Prompt 只保留 `history[-50:]`；
5. 得到 110 类、22,000 条 Train。

需要强调：

> `history[-50:]` 只是从固定 raw history 中取最后 50 条，并没有加入此前
> impression 的正点击。

### 4.5 旧数据实际测量的任务

旧任务不是严格的“预测用户第一次出现的新兴趣”，而是：

> 给定 MIND 提供的固定历史快照，预测一个当前点击、且未出现在该固定快照中的主题。

这个定义在旧 metadata 中也被记录为：

```text
A current clicked category-subcategory pair absent from the behavior
record's clicked-history interest set.
```

旧构建过程本身与这个局部定义一致；问题在于我们后来把它解释成了更强的、时间动态的
“用户新兴趣”。

## 5. 为什么之前没有考虑更早 Impression 的点击

这不是因为原始数据没有 `user_id`、`timestamp` 或点击标签；这些字段都存在。主要是
以下方法和工程假设共同造成的。

### 5.1 把 raw history 当成了当前时刻的完整历史

最初采用了自然但未经验证的假设：

```text
每个 behavior 行中的 history
= 该 impression 发生前的用户点击历史
```

如果该假设成立，逐行处理就是正确的，因为上游已经替我们维护了滚动 history。

但全量审计显示，实际数据更接近：

```text
每个 behavior 行中的 history
= 该用户的一份固定种子历史/画像快照
```

我们在最初构建前没有先检查：

```text
count_distinct(history) grouped by user_id
```

这是最直接的遗漏。

### 5.2 初始目标被限定为“相对 behavior history 新”

第一版 builder 的规则名是：

```text
clicked-interest-absent-from-history-v1
```

实现目标是构建一个简化数据集，而不是重放完整用户事件流。代码因此以 behavior
为处理单位，而不是以 user timeline 为处理单位。

换句话说，旧代码准确实现了一个较弱的问题定义，但当实验问题逐渐变成“预测用户真正
的新兴趣”时，没有同步提升数据定义。

### 5.3 Pipeline 没有用户状态

旧 builder 没有类似：

```python
state_by_user[user_id]
```

也没有：

```python
events.sort(key=(user_id, timestamp))
```

因此它从结构上就不可能知道：

- 用户上一条 impression 点击了什么；
- 某个主题是否已在几小时前或几天前出现；
- Train 中出现过的主题是否在 Dev 中再次出现。

### 5.4 旧基础输出丢弃了重建动态时间线所需的信息

旧基础数据只输出满足局部 novelty 的 target rows。以下信息没有完整保留：

- 没有 novel target 的 impression；
- 当前 impression 中属于已有兴趣的正点击；
- 当前 impression 的完整正点击集合；
- impression-level novel/repeated target set。

一旦只保留 `new_interest/*.parquet`，就无法可靠地从该文件反向重建动态历史。因此
修复必须重新读取原始 `behaviors.tsv`，不能只修改旧 processed Parquet。

### 5.5 测试只验证了行内 novelty

旧测试重点验证：

- target 不在该行 `history` 中；
- 一个 impression 的相同主题点击会合并；
- 多个 target 会展开成多行；
- unknown news ID 显式报错；
- 构建过程确定性、可复现。

没有测试：

- 同一用户的 history 是否随 timestamp 变化；
- history 长度是否随用户事件单调增长；
- 已在 earlier impression 点击的主题是否再次被标 novel；
- Train 用户状态是否传入 Dev。

测试覆盖了实现，却没有覆盖更强的时间语义。

### 5.6 类别平衡掩盖了问题

每类 200 条平衡会使最终 Train 看起来结构非常整齐：

```text
110 classes
200 samples per class
22,000 total
```

但类别均衡只控制 target frequency，不验证 target 是否真的是该用户第一次出现。
同时，one-target-per-impression 会减少显式多答案冲突，使问题更不容易从训练文件
表面被发现。

### 5.7 早期分析关注模型与候选偏差，而不是时间一致性

此前主要审计：

- 类别平衡；
- Prompt/response 格式；
- token 长度；
- Candidate 正确位置；
- 模型位置偏好；
- Beam Recall/NDCG；
- Train/Dev 类别分布。

这些检查都可以在静态数据上通过。直到进一步分析 same-user target repetition 和
exact-input conflict，才暴露“新兴趣”定义与用户时间线不一致。

## 6. 旧定义造成的具体问题

### 6.1 已点击主题会再次被标为新兴趣

真实案例：

```text
user_id: U562515

2019-11-14 Train impression 802530
clicked topic:
Public Services & Social Welfare

2019-11-15 Dev impression 284807
clicked topic:
Public Services & Social Welfare
```

两条 behavior 的原始固定 history 都没有该主题。

旧规则：

```text
Dev target = novel
```

动态规则：

```text
Dev target = repeated
```

因为 11 月 14 日的正点击应当在 11 月 15 日预测之前加入用户历史。

### 6.2 相同输入对应不同监督答案

同一用户的固定 history 可以在多个 timestamp 上产生完全不同的点击：

```text
same user
same raw history
different impression
different target
```

Interest-only 还会把不同用户的标题历史压缩为相同主题序列，进一步增加输入碰撞。

这不是普通的分类标签错误，而是输入中缺失决定 target 的时间和 session 状态。

### 6.3 “新兴趣”数量被高估

静态与动态自然分布对比：

| split | 静态新兴趣 impressions | 动态新兴趣 impressions | 被移除的伪新 impressions | 降幅 |
|---|---:|---:|---:|---:|
| Train | 1,219,553 | 1,157,913 | 61,640 | 5.05% |
| Dev | 199,644 | 179,555 | 20,089 | 10.06% |

Target membership：

| split | 静态 novel memberships | 动态 novel memberships | 被重新判为 repeated | 降幅 |
|---|---:|---:|---:|---:|
| Train | 1,489,348 | 1,401,017 | 88,331 | 5.93% |
| Dev | 243,118 | 214,627 | 28,491 | 11.72% |

Dev 降幅更大，因为在线 Dev history 已经包含：

- 固定初始 history；
- 同用户全部更早 Train 正点击；
- 同一天更早 Dev 正点击。

## 7. 当前动态修复如何处理

新基础 builder：

```text
src/new_interest/data/build_dynamic_dataset.py
```

输出：

```text
data/processed/new_interest_dynamic_v1/
```

动态版本是 CrossInterest 为新的在线序列研究问题增加的重构，不是 MIND 官方
baseline 协议的原样复现。它主动使用第五周 Train 和第五周最后一天的已标注 Dev
中可观察到的更早正点击，建立在线更新状态。官方第六周 Test 没有公开点击标签，
不参与动态重放。

### 7.1 新定义

对于用户 \(u\) 在时间 \(t\) 的 impression：

```text
dynamic_history(u, t)
  = initial_raw_history(u)
    + all positive clicks of u strictly before t

novel_targets(u, t)
  = current_clicked_interests
    - interests(dynamic_history(u, t))

repeated_targets(u, t)
  = current_clicked_interests
    ∩ interests(dynamic_history(u, t))
```

### 7.2 时间重放

处理顺序：

```text
group by user_id
sort by timestamp
replay all Train events
carry final Train state into Dev
replay Dev online
```

同一时间戳的事件采用：

```text
label all impressions first
then append all positive clicks
```

避免同一时刻的点击互相泄漏。

### 7.3 Impression-level 基础 schema

新基础数据一次 impression 一行，并保留：

```text
history
clicked_news
clicked_interests
novel_interests
repeated_interests
has_new_interest
```

即使 impression 没有 novel target，也保留该行。这样后续可以构造：

- 是否出现新兴趣；
- 新兴趣集合排序；
- 单目标生成；
- 重复兴趣预测；
- exposure-aware 任务。

### 7.4 动态平衡数据

动态生成式平衡数据：

```text
data/training/generative_new_interest_dynamic_balanced_200_h50/
```

规则：

1. 每个 Train impression 从全部 novel targets 中确定性选择一个；
2. 选择后统计各类别支持量；
3. 保留至少 200 条的 110 类；
4. 每类采样 200 个唯一 impression；
5. Train 共 22,000 条；
6. Dev 保留 retained-110 中的全部 novel targets。

Prompt 最多使用最近 50 条动态点击，但 novelty 始终使用完整动态历史判断；截断只影响
模型输入长度，不会把更早出现过的主题重新判为 novel。

## 8. 旧实验应如何解释

旧静态数据和旧实验仍可复现，但其结论必须限定为：

> 相对 MIND 固定历史快照的新主题预测。

不能直接解释为：

> 相对用户完整时间线的第一次新兴趣预测。

旧结果仍然能用于研究：

- 模型对标题/主题序列的条件排序；
- 类别先验和日期漂移；
- Candidate 位置偏好；
- 静态快照任务下的模型比较。

但后续 SFT、Candidate、Beam 和 GRPO 主实验应从动态数据重新生成，不能把旧静态
benchmark 继续当作最终任务。

## 9. 防止再次发生的校验

动态 builder 和测试现在要求：

```text
同一用户的 raw seed history 必须一致，否则显式报错
Train 时间必须整体早于 Dev
用户 dynamic history 只能追加，不能回退
novel targets 必须不在完整 history_before 中
repeated targets 必须已在完整 history_before 中
同一时间戳 label 后再统一更新
Train 状态必须传入在线 Dev
一个主题在某个更早 timestamp 出现后，之后不能再次标为 novel
```

此外，未来任何序列数据构建都应在训练前先报告：

```text
distinct histories per user
events per user
history update rate
same-user repeated target rate
exact-input target conflict rate
time range and split chronology
```

## 10. 相关路径

### 原始数据

- `data/raw/MINDlarge_reclassified/train/behaviors.tsv`
- `data/raw/MINDlarge_reclassified/dev/behaviors.tsv`
- `data/raw/MINDlarge_reclassified/{train,dev}/news.tsv`

### 旧静态实现与数据

- `src/new_interest/data/build_dataset.py`
- `src/new_interest/data/build_generative_dataset.py`
- `data/processed/new_interest/`
- `data/training/generative_new_interest_balanced_200_h50/`

### 新动态实现与数据

- `src/new_interest/data/build_dynamic_dataset.py`
- `src/new_interest/data/build_dynamic_generative_dataset.py`
- `tests/test_build_dynamic_dataset.py`
- `tests/test_build_dynamic_generative_dataset.py`
- `data/processed/new_interest_dynamic_v1/`
- `data/training/generative_new_interest_dynamic_balanced_200_h50/`

### 统计与元数据

- `data/processed/new_interest_dynamic_v1/statistics.json`
- `data/processed/new_interest_dynamic_v1/metadata.json`
- `data/training/generative_new_interest_dynamic_balanced_200_h50/statistics.json`
- `data/training/generative_new_interest_dynamic_balanced_200_h50/metadata.json`
