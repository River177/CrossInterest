# Static User-Disjoint 21,600 Balanced SFT

> 状态：四个条件均已完成 3-epoch 训练和完整 Dev constrained-Beam 评测，本地合并预测已通过独立完整性与 SHA-256 验收。Llama Interest 的 AMLT 终态被平台标为 `failed`，但用户进程为 `exit_code=0`，训练、四个 Beam shard、merged 结果及全部哈希门禁均完整；因此接受产物，不重跑该条件。

## 1. 研究问题（Research Questions）

本实验检验：在 Train/Dev 用户完全隔离、Train 每个兴趣等量 200 个不同用户的条件下，LoRA SFT 能否利用静态原始 History 预测用户下一次出现的新兴趣。比较四个条件：

- Qwen3.5-9B × Behavior History（新闻标题）；
- Qwen3.5-9B × Interest History（兴趣类别序列）；
- Llama-3.1-8B-Instruct × Behavior History；
- Llama-3.1-8B-Instruct × Interest History。

主指标是 Dev 中 104 个有支持兴趣类别等权平均的 Interest-Macro Recall@10 与 Interest-Macro NDCG@10。辅助报告 @1/@5/@20、impression-level Recall/NDCG/Hit/MRR，以及 Top-1 类别覆盖与分布。本 Dev 已用于此前探索，结果只能定位为探索性 Dev，不能冒充全新 final test。

## 2. 训练数据构造（Training Data Construction）

数据来自 `data/processed/new_interest_static_user_disjoint_v1`，经 `generative_new_interest_static_user_disjoint_balanced_200_h50_v1` 构造平衡训练集，再冻结为本实验目录。采样 seed 为 `20260806`。训练集保留 108 个兴趣，每个兴趣恰好选择 200 个不同用户；每位用户只出现一次，因此每路共有 21,600 条样本和 21,600 个用户。Behavior 与 Interest 两路使用完全相同的用户、impression 和 target，仅输入表示不同。

History 只使用源 behavior 的原始 History 最近最多 50 条，不追加 impression 点击。已验证 Train 最大 History 长度为 50；Dev 同样为 50。Train 与 Dev 用户交集为 0。

| 输入条件 | 样本 | 用户 | 兴趣类别 | 每类用户 | History 上限 |
|---|---:|---:|---:|---:|---:|
| Behavior | 21,600 | 21,600 | 108 | 200 | 50 |
| Interest | 21,600 | 21,600 | 108 | 200 | 50 |

两路共享 target response 格式：`New topic: <exact category>`。训练采用 completion-only loss。

### 2.1 配对训练样例

来源：`source_sample_id=train:1820310:100`，用户 `U152344`，target 为 `Internet & Social Platforms`。

Behavior 原始 prompt：

> The headlines of the news items I recently viewed are listed below:
>
> 1. The News In Cartoons
>
> Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.

中文翻译：

> 下面列出了我最近浏览的新闻标题：
>
> 1. 漫画新闻
>
> 根据这些标题，预测我接下来可能感兴趣的一个新主题类别。预测类别应不同于浏览历史中体现的主题。回答必须以“New topic: ”开头，其后只给出类别名称。

Interest 原始 prompt：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Trivia & Puzzles
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.

中文翻译：

> 下面按时间顺序列出了我最近浏览新闻对应的主题类别：
>
> 1. 冷知识与谜题
>
> 根据该主题类别序列，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在上面的序列中。回答必须以“New topic: ”开头，其后只给出类别名称。

两路 target response 均为：

> New topic: Internet & Social Platforms

中文含义：新主题为“互联网与社交平台”。

## 3. 测试设计（Testing）

### 3.1 测试模型与位置

四个条件均在 AMLT/Feeds 远程运行。Qwen3.5 两路训练产物来自 `static-sft-full-0816-v1` 的 `80G8-A100` 训练阶段；训练完成并通过 21,600 samples、3 epochs、2,025 optimizer steps、global batch 32、token/model audit 等门禁。原任务未完成完整 Beam，因此通过 `qwen35-beam-recovery-v3-g4-static-sft-full-0816-v1` 在 `80G4-A100` 上复用已验证 adapter，并以 4 个逻辑 shard 断点续跑完整 Dev；Behavior 与 Interest job 最终均为 `pass`。

Llama 两路先在 `static-sft-llama-ddp-smoke-0817-v3-g4` 完成 `80G4-A100` DDP smoke，再由 `static-sft-llama-full-recovery-0817-v3-g4` 承担正式训练与 4-shard Beam。Behavior job 最终为 `pass`。Interest job 的训练实际完成 2,025/2,025 steps，训练 gate、四个 Beam shard和 merge gate 均完成，用户进程退出码为 0；平台日志中观察到的非零退出来自预运行的交互能力检查，而非训练或评测进程，因此 AMLT 的 `failed` 标记与完整用户产物不一致。训练公共设置为 3 epochs、seed 42、BF16、global batch 32、LoRA `r=16/alpha=32/dropout=0.05`、all-linear、最大序列长度 2048、W&B disabled；4-GPU 训练通过提高 gradient accumulation 保持 global batch 不变。Qwen3.5 学习率为 `2e-5`，Llama 学习率为 `2e-4`。

完整评测均使用与分片数无关的同一解码协议：32 beams、Top-20、`max_new_tokens=32`、无采样、无 thinking。每个 shard 保存全局 `record_index`、输入 SHA-256 与评测身份；合并器拒绝缺失、重复、顺序错误、身份冲突、未知类别、重复 Top-20 或 response 文本不一致。

### 3.2 测试数据来源

完整 grouped Dev 含 31,392 个 impressions、27,082 个用户、37,728 个 target occurrences 和 104 个有 support 的兴趣。4,643 个 impressions 含多个正确 target，单个 impression 最多 12 个。训练词表共有 108 类；Dev 未出现的 4 类不进入 Interest-Macro 分母。

### 3.3 配对测试样例

来源：Dev impression `362173`，用户 `U737543`，正确兴趣为 `Celebrity & Creator Culture`。

Behavior 原始 prompt：

> The headlines of the news items I recently viewed are listed below:
>
> 1. The most powerful LGBTQ movies
>
> Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.

中文翻译：

> 下面列出了我最近浏览的新闻标题：
>
> 1. 最具影响力的 LGBTQ 电影
>
> 根据这些标题，预测我接下来可能感兴趣的一个新主题类别。预测类别应不同于浏览历史中体现的主题。回答必须以“New topic: ”开头，其后只给出类别名称。

Interest 原始 prompt：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Film
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.

中文翻译：

> 下面按时间顺序列出了我最近浏览新闻对应的主题类别：
>
> 1. 电影
>
> 根据该主题类别序列，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在上面的序列中。回答必须以“New topic: ”开头，其后只给出类别名称。

该样例的正确 response 为 `New topic: Celebrity & Creator Culture`。完整 Full 预测中的实际结果如下；表中保留英文类别原文，中文含义均为对应兴趣类别名称。

| 模型 | 输入 | Top-5 类别 | 正确类别 rank |
|---|---|---|---:|
| Qwen3.5-9B | Behavior | Crime & Public Safety; Visual Arts & Design; Fashion & Personal Style; **Celebrity & Creator Culture**; Theater & Performing Arts | 4 |
| Qwen3.5-9B | Interest | **Celebrity & Creator Culture**; Crime & Public Safety; Shopping & Gift Ideas; Government & Public Policy; Medicine & Medical Research | 1 |
| Llama-3.1-8B-Instruct | Behavior | Nutrition & Healthy Eating; **Celebrity & Creator Culture**; Visual Arts & Design; Mental Health; Crime & Public Safety | 2 |
| Llama-3.1-8B-Instruct | Interest | **Celebrity & Creator Culture**; Crime & Public Safety; Crafts & Collecting; Artificial Intelligence & Robotics; Shopping & Gift Ideas | 1 |

### 3.4 指标计算

本实验不使用 LLM judge。对于兴趣类别 `y` 和 cutoff `K`，先在所有 target 为 `y` 的 Dev occurrence 上计算 Recall@K 与按命中 rank 折扣的 NDCG@K，再对 104 个有 support 类别等权平均：

\[
\mathrm{InterestMacroRecall@K}=\frac{1}{104}\sum_y\frac{\#\{y\text{ appears in Top-K}\}}{\#\{\text{Dev targets equal }y\}},
\]

\[
\mathrm{InterestMacroNDCG@K}=\frac{1}{104}\sum_y\frac{\sum \mathbf{1}[r_y\le K]/\log_2(r_y+1)}{\#\{\text{Dev targets equal }y\}}.
\]

同一 impression 的多个正确 target 各自贡献一次类别级 occurrence。缺失或非法预测不从分母删除。

## 4. 实验结果（Results）

所有数值均由每个条件 31,392 条 merged predictions 全局重算，不使用 shard 局部均值或 Smoke 指标。主指标如下（百分比）：

| 模型 | 输入 | Interest-Macro Recall@10 | Interest-Macro NDCG@10 | merged predictions | Top-20 合法且不重复 |
|---|---|---:|---:|---:|---:|
| Qwen3.5-9B | Behavior | 14.5035% | 7.2633% | 31,392 | 31,392 / 31,392 |
| Qwen3.5-9B | Interest | **16.0568%** | **8.1487%** | 31,392 | 31,392 / 31,392 |
| Llama-3.1-8B-Instruct | Behavior | 15.4729% | 7.2166% | 31,392 | 31,392 / 31,392 |
| Llama-3.1-8B-Instruct | Interest | 15.9476% | 7.9154% | 31,392 | 31,392 / 31,392 |

### 4.1 完整 Interest-Macro 指标

| 模型 | 输入 | R@1 | NDCG@1 | R@5 | NDCG@5 | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | Behavior | 2.3784% | 2.3784% | 8.1059% | 5.2046% | 14.5035% | 7.2633% | 27.7357% | 10.5685% |
| Qwen3.5-9B | Interest | 2.5740% | 2.5740% | 9.4793% | 6.0483% | 16.0568% | 8.1487% | 29.4989% | 11.5062% |
| Llama-3.1-8B-Instruct | Behavior | 1.8881% | 1.8881% | 7.8785% | 4.7772% | 15.4729% | 7.2166% | 27.1840% | 10.1306% |
| Llama-3.1-8B-Instruct | Interest | 2.4129% | 2.4129% | 9.2457% | 5.7806% | 15.9476% | 7.9154% | 27.2418% | 10.7418% |

### 4.2 Impression-level 辅助指标

| 模型 | 输入 | Recall@10 | NDCG@10 | Hit@10 | MRR@20 | Top-1 覆盖类别 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | Behavior | 20.6811% | 10.0639% | 23.4423% | 8.7014% | 78 / 108 |
| Qwen3.5-9B | Interest | 29.0168% | 15.8194% | 32.5529% | 14.1814% | 82 / 108 |
| Llama-3.1-8B-Instruct | Behavior | 21.9908% | 10.5222% | 24.8471% | 8.9549% | 98 / 108 |
| Llama-3.1-8B-Instruct | Interest | 28.6727% | 15.2014% | 32.0687% | 13.3150% | 93 / 108 |

### 4.3 训练与结果验收

| 条件 | Train samples | epochs / steps | global batch | train loss | token 超限 | 禁止视觉参数 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 Behavior | 21,600 | 3 / 2,025 | 32 | 0.1700 | 0 | 0 |
| Qwen3.5 Interest | 21,600 | 3 / 2,025 | 32 | 0.3366 | 0 | 0 |
| Llama Behavior | 21,600 | 3 / 2,025 | 32 | 0.5815 | 0 | 0 |
| Llama Interest | 21,600 | 3 / 2,025 | 32 | 0.5838 | 0 | 0 |

本地独立验收重新计算并确认：每路 4 个 shard 各 7,848 条；merged `record_index` 恰好为 `0..31391`；31,392 个 impression、input SHA-256 均唯一；104 个 per-interest support 行合计 37,728 个 target occurrences；108 类词表内每条 Top-20 均为不同合法类别；Behavior/Interest 的 record、impression 和正确 target 严格配对；merged prediction、per-interest、metrics、四个 shard prediction 及 adapter 目录哈希均与 manifest/metrics 一致。

观察到的直接差值是：同一模型下 Interest 输入均高于 Behavior 输入；Qwen3.5 的差值为 Recall@10 `+1.5533` 个百分点、NDCG@10 `+0.8854` 个百分点，Llama 的差值为 `+0.4747` 和 `+0.6988` 个百分点。这是探索性 Dev 上的描述性比较，不解释为全新 final-test 结论。

## 5. 数据与产物路径（Artifact Paths）

本地数据与配置：

- 数据根目录：`data/training/sft_static_user_disjoint_balanced_200_h50_v1`
- 数据元数据：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/metadata.json`
- 数据统计：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/statistics.json`
- Behavior Train：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/train/behavior_only/21600.parquet`
- Interest Train：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/train/interest_only/21600.parquet`
- Behavior Dev：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/eval/behavior_only/dev_full_grouped.parquet`
- Interest Dev：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/eval/interest_only/dev_full_grouped.parquet`
- SFT configs：`configs/sft/experiments/static_sud_{qwen35,llama31}_{behavior,interest}_21600.yaml`
- Beam configs：`configs/beam/static_sud_{qwen35,llama31}_{behavior,interest}_full.json`
- AMLT 原始 Full config：`configs/amlt/static_sft_full_0816_v1.yaml`
- Llama 4-GPU smoke config：`configs/amlt/static_sft_llama_ddp_smoke_0817_v3_g4.yaml`
- Qwen 4-GPU Beam recovery config：`configs/amlt/static_sft_qwen_beam_recovery_0817_v3_g4.yaml`
- Llama 4-GPU Full recovery config：`configs/amlt/static_sft_llama_full_recovery_0817_v3_g4.yaml`
- AMLT wrapper：`scripts/amlt/sft/run_static_user_disjoint_balanced_sft.sh`

远程 AMLT：

- Qwen 原始训练实验：`static-sft-full-0816-v1`
- Llama 4-GPU smoke：`static-sft-llama-ddp-smoke-0817-v3-g4`
- Qwen 4-GPU Beam recovery：`qwen35-beam-recovery-v3-g4-static-sft-full-0816-v1`
- Llama 4-GPU Full recovery：`static-sft-llama-full-recovery-0817-v3-g4`

已下载并验收的本地产物：

- SFT：`artifacts/static-sud-sft/{qwen35,llama31}-{behavior,interest}`
- Beam：`artifacts/static-sud-beam/{qwen35,llama31}-{behavior,interest}`
- 每个 Beam 条件的完整预测：`artifacts/static-sud-beam/<condition>/merged/beam_predictions.parquet`
- 每个 Beam 条件的指标：`artifacts/static-sud-beam/<condition>/merged/metrics.json`
- 每个 Beam 条件的类别级指标：`artifacts/static-sud-beam/<condition>/merged/per_interest_metrics.parquet`
- 每个 Beam 条件的合并清单：`artifacts/static-sud-beam/<condition>/merged/merge_manifest.json`
- 本报告：`docs/experiments/2026-08-16/static_user_disjoint_balanced_sft.md`

远端结果目录：

- Qwen Behavior：`7213064229.42653-9c29a9c5-e9c7`（AMLT `pass`）
- Qwen Interest：`7213064229.42619-a2c67492-a056`（AMLT `pass`）
- Llama Behavior：`7213060831.77311-20b9b738-2331`（AMLT `pass`）
- Llama Interest：`7213060831.77241-7ee859e5-afa1`（AMLT 标记 `failed`；用户进程 exit 0，完整产物通过独立验收）
