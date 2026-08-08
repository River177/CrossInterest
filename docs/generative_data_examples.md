# 生成式新兴趣推荐：真实数据样例

本文档展示 `generative_new_interest` 数据集中的真实记录。所有例子都直接来自：

```text
data/training/generative_new_interest/
├── behavior_only/
│   ├── train.parquet
│   └── dev.parquet
└── interest_only/
    ├── train.parquet
    └── dev.parquet
```

任务是根据用户完整历史，以 `New topic: <二级兴趣名称>` 的格式直接生成一个
新兴趣。模型不会看到候选列表，因此这不是排序任务或多项选择题。

## 两种输入形式

| 类型 | 输入内容 | 输出内容 |
|---|---|---|
| Behavior-only | 用户历史点击的新闻标题 | 一个二级兴趣名称 |
| Interest-only | 历史新闻对应的二级兴趣名称 | 一个二级兴趣名称 |

Interest-only 中没有一级兴趣。历史兴趣保持原始顺序，重复出现的兴趣不会被去重。

## 样例一：只有一条历史

源样本：

```text
source_sample_id: train:40:20
history_length: 1
```

### Behavior-only

**User**

```text
The headlines of the news items I recently viewed are listed below:

1. Jamie Foxx Sparks Romance Rumors With Dana Caprio After Katie Split

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

**Assistant**

```text
New topic: Courts & Legal Cases
```

### Interest-only

**User**

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Celebrity & Creator Culture

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

**Assistant**

```text
New topic: Courts & Legal Cases
```

这两条训练记录来自同一个源样本，因此答案完全相同。区别只在于：

- Behavior-only 只展示新闻标题；
- Interest-only 只展示二级兴趣名称。

## 样例二：Interest-only 保留重复兴趣

源样本：

```text
source_sample_id: train:29:65
history_length: 5
```

### Behavior-only

**User**

```text
The headlines of the news items I recently viewed are listed below:

1. Mom 'Grieving Really Hard' After Daughter, 10, Dies Being Ejected from Amusement Park Ride
2. Michigan Judge Has Blocked Ban of Flavored Vapes, Bucking Nationwide Trend
3. Federal judge overturns ObamaCare transgender protections
4. CDC Is Still Warning People to Stop Vaping Completely But Why?
5. 10 hospitalized after insulin administered instead of flu shots

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

**Assistant**

```text
New topic: Pets & Animal Companions
```

### Interest-only

**User**

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Accidents & Emergency Response
2. Courts & Legal Cases
3. Constitutional & Civil Law
4. Public Health
5. Accidents & Emergency Response

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

**Assistant**

```text
New topic: Pets & Animal Companions
```

`Accidents & Emergency Response` 在第 1 条和第 5 条各出现一次。数据保留这两次
出现，因为重复次数也可以反映用户的兴趣强度。

## 样例三：同一历史可以对应多个新兴趣

MIND 的一次 behavior 可能同时点击多个属于不同新兴趣的新闻。当前数据定义会为每个
不同的新兴趣生成一条记录，因此可能出现相同 Prompt、不同答案。

该现象仍存在于完整版本 `generative_new_interest/` 和 Balanced 版本的 Dev 中。
当前 Balanced 版本的 **Train** 会先从同一 impression 中确定性随机选择一个目标，
因此训练集中不会再出现相同 impression 的多个答案。

下面两条记录来自同一个 impression：

```text
impression_id: 68
user_id: U158238
history_length: 3
```

### 共同的 Behavior-only Prompt

```text
The headlines of the news items I recently viewed are listed below:

1. Jess Lost 75 Pounds Without Counting Calories and Did This 1 Inspiring Thing Before Every Meal
2. Video captures terrifying moment woman slips at Grand Canyon
3. Maryland man, 46, gets probation after he killed a father and his four daughters when his SUV crossed a median and slammed into their minivan

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

对应两条训练记录：

| `source_sample_id` | Assistant 答案 |
|---|---|
| `train:68:66` | `New topic: Relationships & Dating` |
| `train:68:114` | `New topic: Motorcycles` |

### 共同的 Interest-only Prompt

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Fitness & Exercise
2. Accidents & Emergency Response
3. Courts & Legal Cases

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

对应的答案同样是：

```text
New topic: Relationships & Dating
```

和：

```text
New topic: Motorcycles
```

这表示该 behavior 中观察到了两个不同的新兴趣，而不是其中一条数据错误。对于单答案
SFT，这会形成“同一输入存在多个有效答案”的训练现象；进行实验时应把它作为数据设计
的一部分考虑。

## 完整 Parquet 记录

下面是样例一的 Interest-only 记录。实际 Parquet 中的 `messages` 可以直接交给
Chat SFT 数据加载器：

```json
{
  "sample_id": "train:40:20:interest_only",
  "source_sample_id": "train:40:20",
  "split": "train",
  "task_type": "interest_only",
  "impression_id": "40",
  "user_id": "U515908",
  "timestamp": "2019-11-13 12:34:08",
  "history_length": 1,
  "prompt": "The topic categories associated with the news items I recently viewed, in chronological order, are listed below:\n\n1. Celebrity & Creator Culture\n\nBased on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with \"New topic: \" and provide only the category name after it.",
  "response": "New topic: Courts & Legal Cases",
  "messages": [
    {
      "role": "user",
      "content": "The topic categories associated with the news items I recently viewed, in chronological order, are listed below:\n\n1. Celebrity & Creator Culture\n\nBased on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with \"New topic: \" and provide only the category name after it."
    },
    {
      "role": "assistant",
      "content": "New topic: Courts & Legal Cases"
    }
  ],
  "target_subcategory": "Courts & Legal Cases"
}
```

训练记录中不包含 `target_category`，Interest-only 也不会保存或展示任何一级兴趣。

## 字段说明

| 字段 | 示例 | 用途 |
|---|---|---|
| `sample_id` | `train:40:20:interest_only` | 当前训练记录的唯一 ID |
| `source_sample_id` | `train:40:20` | 回查原始 new-interest 样本 |
| `task_type` | `interest_only` | 区分两种输入形式 |
| `history_length` | `1` | 完整历史条目数量 |
| `prompt` | 自然语言输入 | 非 Chat 训练或检查数据 |
| `response` | `New topic: Courts & Legal Cases` | 固定前缀加二级兴趣标签 |
| `messages` | user/assistant 两条消息 | Chat SFT |
| `target_subcategory` | `Courts & Legal Cases` | 显式标签字段 |

## 用 PyArrow 查看样本

```python
import pyarrow.parquet as pq

table = pq.read_table(
    "data/training/generative_new_interest/interest_only/train.parquet",
    columns=[
        "sample_id",
        "history_length",
        "prompt",
        "response",
        "messages",
    ],
)

for row in table.slice(0, 3).to_pylist():
    print(row["prompt"])
    print("Answer:", row["response"])
```

对于完整数据，建议使用 batch，避免一次载入全部记录：

```python
parquet = pq.ParquetFile(
    "data/training/generative_new_interest/behavior_only/train.parquet"
)

for batch in parquet.iter_batches(batch_size=1024):
    rows = batch.to_pylist()
```

---

## Balanced-200 + Recent-50 样例

平衡截断版本位于：

```text
data/training/generative_new_interest_balanced_200_h50/
```

Train 先从每个 impression 的多个新兴趣中确定性随机选择一个，再保留样本数不少于
200 的 110 个二级兴趣，每类选择 200 条。Dev 不执行单目标选择，保持多目标自然
分布，但只保留这 110 个 Train 兴趣。

### 真实截断样本

下面的真实 Dev 样本原本有 57 条历史：

```text
source_sample_id: dev:2:86
source_history_length: 57
generated_history_length: 50
response: New topic: Golf
```

截断方式是保留最后 50 条：

```python
recent_history = history[-50:]
```

原始历史第一条被删除：

```text
Stars who've been on one show for a long time
```

第 7 条是截断后进入 Prompt 的第一条：

```text
Rihanna Slays in Sexy Black Bikini in Slow-Motion Video -- Watch!
```

最后一条历史保持不变：

```text
A Military Camera Said 'Made in U.S.A.' The Screen Was in Chinese.
```

生成记录中的 `history_length` 为 50，编号重新从 `1` 到 `50`。Behavior-only
和 Interest-only 使用同一段最近历史，因此两种输入可以严格配对比较。

### 平衡结果

| 数据 | 类别数 | 每类数量 | 总样本数 |
|---|---:|---:|---:|
| Train | 110 | 200 | 22,000 |
| Dev | 109 个实际出现 | 自然分布 | 242,839 |

Dev 中最少的保留类别有 1 条样本，最多的类别有 28,136 条。Dev 不进行平衡，
也不随机删除同一 impression 的其他有效目标，避免改变真实评估分布。

### 查看采样清单

```python
import pyarrow.parquet as pq

selected = pq.read_table(
    "data/training/generative_new_interest_balanced_200_h50/"
    "selected_train_samples.parquet"
)

retained = pq.read_table(
    "data/training/generative_new_interest_balanced_200_h50/"
    "retained_interests.parquet"
)

print(selected.slice(0, 5).to_pylist())
print(retained.slice(0, 5).to_pylist())
```

`selected_train_samples.parquet` 中每个 `source_sample_id` 和
`impression_id` 都唯一。每条记录保存固定的 `sampling_seed=20260806`、
`target_selection_priority` 和 `balance_sampling_priority`。
