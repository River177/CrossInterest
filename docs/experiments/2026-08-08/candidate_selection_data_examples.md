# K=10 候选主题选择数据：构造规则与真实样例

本文档只展示候选数据。**尚未运行 Base 或 SFT 模型测试。**

## 数据位置

```text
data/training/candidate_selection_k10/
├── behavior_only/
│   └── dev_10k_expanded.parquet
├── interest_only/
│   └── dev_10k_expanded.parquet
├── candidate_manifest.parquet
├── statistics.json
└── metadata.json
```

两种输入各包含 **12,164 条**记录，来自固定 grouped Dev10K 的 10,000 个
impression。

## 构造规则

每条记录有 10 个候选：

- 1 个正样本；
- 9 个负样本；
- 正样本是该 impression 的一个真实新主题；
- 负样本不属于该 impression 的任何正确主题；
- 负样本不属于最近 50 条历史中的任何主题；
- 候选不重复；
- 答案输出为 `New topic: <完整类别名称>`，不使用字母或编号。

Dev 中一个 impression 可能有多个正确主题。此时每个正确主题各生成一条候选记录，
而其他正确主题也会从负样本池排除。

## 数据规模与位置分布

| 指标 | 数量 |
|---|---:|
| grouped impressions | 10,000 |
| 展开记录 | 12,164 |
| 候选数 | 10 |
| 合法负样本池最小值 | 73 |
| 合法负样本池中位数 | 100 |
| 合法负样本池最大值 | 109 |

正确答案位置：

| 位置（从 1 开始） | 样本数 |
|---:|---:|
| 1 | 1,217 |
| 2 | 1,217 |
| 3 | 1,217 |
| 4 | 1,217 |
| 5 | 1,216 |
| 6 | 1,216 |
| 7 | 1,216 |
| 8 | 1,216 |
| 9 | 1,216 |
| 10 | 1,216 |

位置通过稳定 SHA-256 随机分配，同时保持总体均衡。

## 样例一：Behavior-only 与 Interest-only 配对

```text
selection_id: dev:178:18
impression_id: 178
answer_category: Cars & Driving
answer: New topic: Cars & Driving
correct_position: 10
```

### Behavior-only

```text
The headlines of the news items I recently viewed are listed below:

1. Look who's watching
2. America's Unsafe Drugs and Medical Products

Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should represent a new topic that is not reflected in my viewing history.

Candidates:
1. Pets & Animal Companions
2. Music
3. Markets & Investing
4. Personal Finance
5. Careers & Professional Development
6. Crime & Public Safety
7. Wildlife & Conservation
8. Sports Business & Administration
9. Severe Storms
10. Cars & Driving

Respond with exactly "New topic: <category>", replacing <category> with one exact category name from the candidate list. Do not include any other text.
```

标准答案：

```text
New topic: Cars & Driving
```

### Interest-only

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. General & Multi-Sport Coverage
2. Medicine & Medical Research

Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should not appear in the history sequence above.

Candidates:
1. Pets & Animal Companions
2. Music
3. Markets & Investing
4. Personal Finance
5. Careers & Professional Development
6. Crime & Public Safety
7. Wildlife & Conservation
8. Sports Business & Administration
9. Severe Storms
10. Cars & Driving

Respond with exactly "New topic: <category>", replacing <category> with one exact category name from the candidate list. Do not include any other text.
```

标准答案同样是：

```text
New topic: Cars & Driving
```

两条记录的 candidates、答案和正确位置完全一致，只改变历史表示。

## 样例二：同一 impression 的多个正确主题分别展开

```text
impression_id: 3204
all acceptable targets:
- Courts & Legal Cases
- American Football
```

原始历史对应主题：

```text
Economy & Economic Policy
Religion in Public Life
Economy & Economic Policy
```

该 impression 生成两条选择记录。

### 记录 A

```text
selection_id: dev:3204:30
answer_category: Courts & Legal Cases
answer: New topic: Courts & Legal Cases
```

候选：

```text
1. Infrastructure & Transportation Policy
2. Crafts & Collecting
3. Civic Life & Public Commemorations
4. Immigration & Citizenship
5. Companies & Corporate Strategy
6. Archaeology & Human Origins
7. Theater & Performing Arts
8. Policing & Corrections
9. Personal Growth
10. Courts & Legal Cases
```

### 记录 B

```text
selection_id: dev:3204:5
answer_category: American Football
answer: New topic: American Football
```

候选：

```text
1. Parenting & Family Life
2. Energy & Sustainability
3. Floods, Droughts & Wildfires
4. Basketball
5. Agriculture & Food Production
6. Sexual & Reproductive Health
7. Elections & Political Campaigns
8. Sleep Health
9. Celebrity & Creator Culture
10. American Football
```

构造记录 A 时，`American Football` 不会成为负样本；构造记录 B 时，
`Courts & Legal Cases` 也不会成为负样本。

## 样例三：正确答案位于第一个位置

```text
selection_id: dev:8960:78
answer_category: Parenting & Family Life
answer: New topic: Parenting & Family Life
correct_position: 1
```

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Fitness & Exercise
2. Severe Storms
3. Pets & Animal Companions

Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should not appear in the history sequence above.

Candidates:
1. Parenting & Family Life
2. Telecommunications
3. Genealogy & Family History
4. Entrepreneurship & Small Business
5. Restaurants & Dining
6. Cruises
7. Culture & Heritage Travel
8. Wildlife & Conservation
9. Household Cleaning & Laundry
10. Home Appliances & Technology

Respond with exactly "New topic: <category>", replacing <category> with one exact category name from the candidate list. Do not include any other text.
```

该样例说明正确答案并不固定在第 10 位。

## Parquet 字段

| 字段 | 含义 |
|---|---|
| `selection_id` | 稳定选择记录 ID |
| `impression_id` | 原始 impression |
| `user_id` | 用户 ID |
| `timestamp` | behavior 时间 |
| `task_type` | `behavior_only` / `interest_only` |
| `history_length` | 最近历史数量 |
| `prompt` | 完整候选选择 Prompt |
| `candidates` | 有序 10 个候选类别 |
| `answer_category` | 唯一正确候选类别 |
| `answer` | `New topic: <answer_category>` |
| `correct_position` | 0–9 的零基位置 |
| `all_acceptable_subcategories` | 原 impression 的全部正确新主题 |
| `history_subcategories` | 最近历史主题序列 |

## 读取示例

```python
import pyarrow.parquet as pq

table = pq.read_table(
    "data/training/candidate_selection_k10/"
    "interest_only/dev_10k_expanded.parquet"
)

row = table.slice(0, 1).to_pylist()[0]
print(row["prompt"])
print("Candidates:", row["candidates"])
print("Answer:", row["answer"])
```

候选数据已完成构造与验证，可用于后续 Base、SFT 和 GPT 模型测试。
