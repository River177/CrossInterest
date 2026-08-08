# New Interest Dataset

当前 Qwen3-8B SFT 与 Beam Search 实验报告：

```text
docs/experiments/2026-08-07/qwen3_sft_beam_experiment_report.md
```

K=10 候选主题选择数据说明：

```text
docs/experiments/2026-08-08/candidate_selection_data_examples.md
```

K=10 候选选择模型实验报告：

```text
docs/experiments/2026-08-08/candidate_selection_experiment_report.md
```

Qwen3-8B Base Interest 低于随机诊断：

```text
docs/experiments/2026-08-08/qwen_base_interest_bias_diagnosis.md
```

Qwen3/Llama 22K 跨模型实验报告：

```text
docs/experiments/2026-08-08/qwen_llama_22k_experiment_report.md
```

全部实验报告索引：

```text
docs/experiments/README.md
```

Recall, NDCG, and Hit metric definitions:

```text
docs/metrics/recall_ndcg_hit.md
```

本项目从重分类后的 MINDlarge 数据构造一个简化的新兴趣预测数据集。

## 新兴趣定义

一个当前点击兴趣如果没有出现在该行为记录的历史点击兴趣中，就被定义为新兴趣：

```text
target_interest ∈ current_clicked_interests
target_interest ∉ history_clicked_interests
```

兴趣使用一级类别和二级类别组合表示，例如：

```text
Crime, Law & Justice / Courts & Legal Cases
```

本数据集：

- 不要求该兴趣以前曝光过；
- 不使用未点击的 `-0` 新闻；
- 不构造负样本或候选列表；
- 不进行类别平衡；
- 一次行为包含多个不同新兴趣时，每个新兴趣生成一条样本；
- 同一新兴趣对应多篇当前点击新闻时，只生成一条样本，并保存全部目标新闻。

## 原始数据

唯一保留的原始数据位于：

```text
data/raw/MINDlarge_reclassified/
```

Train 和 Dev 包含带 `-1/-0` 标签的 behavior，可以构造监督样本。MIND Test
没有点击标签，因此不生成 Test。

## 构造数据

```bash
PYTHONPATH=src python -m new_interest.build_dataset \
  --input-dir data/raw/MINDlarge_reclassified \
  --output-dir data/processed/new_interest
```

已有输出需要显式覆盖：

```bash
PYTHONPATH=src python -m new_interest.build_dataset --overwrite
```

## 输出

```text
data/processed/new_interest/
├── train.parquet
├── dev.parquet
├── interest_vocabulary.parquet
├── statistics.json
└── metadata.json
```

### 数据规模

| Split | 原始行为数 | 有新兴趣的行为数 | 样本数 | 出现的目标兴趣类别 |
|---|---:|---:|---:|---:|
| Train | 2,186,664 | 1,219,553 | **1,489,348** | 124 |
| Dev | 365,199 | 199,644 | **243,118** | 119 |

完整兴趣词表包含 125 个一级-二级类别组合。

## Parquet Schema

| 字段 | 含义 |
|---|---|
| `sample_id` | 稳定样本 ID：split、impression ID、兴趣 ID |
| `split` | `train` 或 `dev` |
| `impression_id` | 原始 MIND impression ID |
| `user_id` | 用户 ID |
| `timestamp` | 当前行为时间 |
| `history` | 完整历史点击新闻，保持原始顺序 |
| `target_interest` | 当前新兴趣的一级类别和二级类别 |
| `target_news` | 当前点击且属于目标兴趣的全部新闻 |

`history` 每项包含：

```json
{
  "news_id": "N127817",
  "category": "Arts & Entertainment",
  "subcategory": "Celebrity & Creator Culture",
  "title": "Jamie Foxx Sparks Romance Rumors With Dana Caprio After Katie Split"
}
```

`target_news` 每项包含新闻 ID 和标题。

## 真实样例

```json
{
  "sample_id": "train:40:20",
  "split": "train",
  "impression_id": "40",
  "user_id": "U515908",
  "timestamp": "2019-11-13 12:34:08",
  "history": [
    {
      "news_id": "N127817",
      "category": "Arts & Entertainment",
      "subcategory": "Celebrity & Creator Culture",
      "title": "Jamie Foxx Sparks Romance Rumors With Dana Caprio After Katie Split"
    }
  ],
  "target_interest": {
    "category": "Crime, Law & Justice",
    "subcategory": "Courts & Legal Cases"
  },
  "target_news": [
    {
      "news_id": "N24701",
      "title": "Kodak Black Sentenced to Over 3 Years in Prison in Weapons Case"
    }
  ]
}
```

该用户历史中只有 `Celebrity & Creator Culture`，当前点击新闻属于
`Courts & Legal Cases`，因此该兴趣被标记为新兴趣。

## 读取示例

```python
import pyarrow.parquet as pq

table = pq.read_table("data/processed/new_interest/train.parquet")
first_row = table.slice(0, 1).to_pylist()[0]
print(first_row["history"])
print(first_row["target_interest"])
```

对于完整数据，建议按 batch 读取：

```python
parquet = pq.ParquetFile("data/processed/new_interest/train.parquet")

for batch in parquet.iter_batches(batch_size=8192):
    rows = batch.to_pylist()
```

## 统计与可复现性

- `statistics.json` 保存目标兴趣频次、历史长度和目标新闻数分布；
- `metadata.json` 保存规则版本、raw manifest、数据规模和输出 SHA-256；
- `interest_vocabulary.parquet` 保存稳定的 125 类兴趣 ID；
- 构造器使用临时 staging 目录，只有全部成功后才移动到正式输出路径。

## 测试

```bash
pytest -q
```

## 已知限制

- “新兴趣”只根据 MIND behavior 中提供的点击历史判断；
- 如果更早的点击没有被包含在该 history 字段中，数据集无法获知；
- 同一用户可以在不同 behavior 中多次产生相同的新兴趣样本，只要该 behavior
  的 history 中仍未出现该兴趣；
- 数据集不表示用户是否曾经看到过但未点击该兴趣。

---

## 生成式新兴趣推荐数据

在上述完整新兴趣样本之上，项目进一步构造了两套开放式生成数据。模型不接收候选
类别，也不执行排序或多项选择，而是以
`New topic: <二级兴趣名称>` 的格式直接生成答案。

更完整的真实数据展示见：

```text
docs/generative_data_examples.md
```

| 数据类型 | 历史输入 | 模型输出 |
|---|---|---|
| Behavior-only | 历史点击新闻标题 | 一个二级兴趣名称 |
| Interest-only | 历史新闻对应的二级兴趣序列 | 一个二级兴趣名称 |

Interest-only 的输入、输出、messages 和标签字段都只有二级兴趣，不包含一级兴趣。
历史兴趣保持原始顺序和重复次数。

### 构造命令

```bash
PYTHONPATH=src python -m new_interest.build_generative_dataset
```

已有输出需要显式覆盖：

```bash
PYTHONPATH=src python -m new_interest.build_generative_dataset --overwrite
```

### 输出文件

```text
data/training/generative_new_interest/
├── behavior_only/
│   ├── train.parquet
│   └── dev.parquet
├── interest_only/
│   ├── train.parquet
│   └── dev.parquet
├── statistics.json
└── metadata.json
```

| 数据类型 | Train | Dev | 文件大小 |
|---|---:|---:|---:|
| Behavior-only | 1,489,348 | 243,118 | 约 1.9 GB |
| Interest-only | 1,489,348 | 243,118 | 约 296 MB |

两套数据包含相同的源样本和答案，只改变历史表示。

### Behavior-only Prompt

```text
The headlines of the news items I recently viewed are listed below:

1. {title_1}
2. {title_2}
...

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

真实样例：

```text
The headlines of the news items I recently viewed are listed below:

1. Jamie Foxx Sparks Romance Rumors With Dana Caprio After Katie Split

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

Assistant：

```text
New topic: Courts & Legal Cases
```

### Interest-only Prompt

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. {subcategory_1}
2. {subcategory_2}
...

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

真实样例：

```text
The topic categories associated with the news items I recently viewed, in chronological order, are listed below:

1. Celebrity & Creator Culture

Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
```

Assistant：

```text
New topic: Courts & Legal Cases
```

### 训练 Schema

| 字段 | 含义 |
|---|---|
| `sample_id` | 带数据类型后缀的训练样本 ID |
| `source_sample_id` | 对应的原始 new-interest 样本 ID |
| `split` | `train` 或 `dev` |
| `task_type` | `behavior_only` 或 `interest_only` |
| `impression_id` | 原始 impression ID |
| `user_id` | 原始用户 ID |
| `timestamp` | 原始行为时间 |
| `history_length` | 完整历史条目数 |
| `prompt` | 固定模板生成的完整输入 |
| `response` | 目标二级兴趣名称 |
| `messages` | user/assistant Chat messages |
| `target_subcategory` | 与 `response` 相同的显式标签字段 |

训练 Schema 不包含 `target_category` 或其他一级兴趣字段。

### Chat 格式

```json
[
  {
    "role": "user",
    "content": "<prompt>"
  },
  {
    "role": "assistant",
    "content": "New topic: Courts & Legal Cases"
  }
]
```

回答只包含固定前缀 `New topic: ` 和二级兴趣名称，不包含解释、CoT 或选项字母。

### 完整历史与上下文长度

数据构造阶段不截断历史：

| 数据类型 | Split | Prompt 字符数中位数 | P90 | P99 | 最大值 |
|---|---|---:|---:|---:|---:|
| Behavior-only | Train | 1,184 | 4,531 | 11,593 | 62,439 |
| Behavior-only | Dev | 1,175 | 4,494 | 11,411 | 47,671 |
| Interest-only | Train | 595 | 1,837 | 4,473 | 23,063 |
| Interest-only | Dev | 592 | 1,823 | 4,413 | 18,559 |

最大历史分别为 Train 800 条、Dev 628 条。后续训练时应根据具体 Qwen/Llama
tokenizer 和上下文窗口决定截断策略；当前 Parquet 始终保留完整原始历史。

---

## Balanced-200 + Recent-50 版本

用于正式训练时，项目另外提供了类别平衡、历史截断版本：

```text
data/training/generative_new_interest_balanced_200_h50/
├── behavior_only/
│   ├── train.parquet
│   └── dev.parquet
├── interest_only/
│   ├── train.parquet
│   └── dev.parquet
├── selected_train_samples.parquet
├── retained_interests.parquet
├── statistics.json
└── metadata.json
```

该版本不会覆盖前面的完整数据。

### 平衡规则

- Train 中同一 impression 出现多个新兴趣时，先使用固定 seed 选择一个目标；
- 唯一目标选择后，1,489,348 条源样本变为 1,219,553 条；
- 再保留数量至少为 200 的二级兴趣；
- 共保留 110 个二级兴趣；
- 每个兴趣确定性选择 200 条 Train 样本；
- 不足 200 条的 14 个兴趣直接删除，不进行重复采样；
- 每种任务的 Train 均为 **22,000 条**，且 `impression_id` 唯一；
- Dev 不随机选择单一目标，也不做平衡或重复采样；
- Dev 只过滤 Train 未保留的兴趣，同一 impression 可以继续对应多个目标；
- Dev 删除 279 条样本，最终每种任务为 **242,839 条**；
- Dev 实际出现 109 类；`Artificial Intelligence & Robotics` 在 Train 中保留，
  但过滤后的 Dev 中没有对应样本。

Train 的 impression 内目标选择使用：

```text
target_priority = SHA256("20260806:target:<source_sample_id>")
```

类别内 200 条采样使用独立的 priority：

```text
balance_priority = SHA256("20260806:balance:<source_sample_id>")
```

因此目标选择和类别平衡都不会依赖候选项或文件的遍历顺序。

### 最近 50 条历史

两种任务都使用：

```python
recent_history = history[-50:]
```

- Behavior-only 提取最近 50 条新闻标题；
- Interest-only 提取同一批新闻对应的二级兴趣；
- 历史不足 50 条时全部保留；
- `history_length` 表示实际进入 Prompt 的历史数，最大为 50；
- 原始完整历史仍可通过 `source_sample_id` 回查。

| Split | 样本数 | 被截断样本 | 截断后历史中位数 | 最大历史 |
|---|---:|---:|---:|---:|
| Train | 22,000 | 4,030 | 18 | 50 |
| Dev | 242,839 | 31,047 | 14 | 50 |

### 构造命令

```bash
PYTHONPATH=src python -m new_interest.build_generative_dataset \
  --output-dir data/training/generative_new_interest_balanced_200_h50 \
  --one-target-per-train-impression \
  --history-limit 50 \
  --train-samples-per-class 200 \
  --train-min-class-count 200 \
  --filter-dev-to-train-classes \
  --sampling-seed 20260806
```

已有输出需要增加 `--overwrite`。

### 文件规模

| 文件 | 大小 |
|---|---:|
| Behavior-only Train | 约 28 MB |
| Behavior-only Dev | 约 206 MB |
| Interest-only Train | 约 4.0 MB |
| Interest-only Dev | 约 33 MB |
| 完整目录 | 约 275 MB |

`selected_train_samples.parquet` 记录全部 22,000 个采样 ID、impression ID、
目标选择 priority、类别平衡 priority 和 seed；`retained_interests.parquet`
记录 110 类在唯一目标选择后的 Train 数、最终选择数及 Dev 数。
