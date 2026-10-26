# Static User-Disjoint Semantic-initialized ID Token SFT

> 状态：四个条件均已完成3 epochs/2,025 optimizer steps、完整 Dev 评测和本地独立审计。Qwen Interest 原任务只在评测阶段因旧 shard identity 冲突失败；冻结 adapter 未受影响，清理四个旧 shard cache 后的独立恢复评测已通过。本文指标均来自未屏蔽 History 的108类完整排序。

## 1. 研究问题（Research Questions）

本实验检验：将108个兴趣名称替换为单 token ID，并用原始类别名称 token embedding 的均值进行语义初始化后，LoRA SFT 是否能在 Static User-Disjoint 完整 Dev 上优于当前文本标签生成方案。四个条件为：

- Qwen3.5-9B × Behavior History；
- Qwen3.5-9B × Interest History；
- Llama-3.1-8B-Instruct × Behavior History；
- Llama-3.1-8B-Instruct × Interest History。

主指标为104个 Dev 支持兴趣等权平均的 Interest-Macro Recall@10 与 Interest-Macro NDCG@10。直接对照是同一21,600条训练集上的 text-label SFT；NRMS 只作为非 SFT 参考。所有比较均使用完整108类预测空间，不屏蔽 History 中已经出现的类别。本 Dev 已用于此前分析，因此结果只定位为探索性 Dev，不能冒充全新 final test。

由于本实验没有 Random-init ID Token 条件，它只能回答“Semantic-initialized ID Token 整体方案是否优于文本标签生成”，不能单独归因语义初始化本身的效果。

## 2. 训练数据构造（Training Data Construction）

数据目录为 `data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1`。它逐条复用现有 H50 平衡 SFT 数据：108类、每类200个不同用户、每路21,600条，Behavior 与 Interest 的用户、impression、target 和顺序严格配对。History 等于原始 behavior History 最近最多50条，不追加 impression 点击；Train 与 Dev 用户交集为0。

兴趣名称按字典序稳定映射为 `<interest_000>` 至 `<interest_107>`。Prompt 不提供名称与 ID 的映射表，response 和 assistant message 只保留一个 ID Token。

| 输入条件 | 样本 | 用户 | 兴趣类别 | 每类用户 | History 上限 |
|---|---:|---:|---:|---:|---:|
| Behavior | 21,600 | 21,600 | 108 | 200 | 50 |
| Interest | 21,600 | 21,600 | 108 | 200 | 50 |

### 2.1 配对训练样例

来源：`source_sample_id=train:1820310:100`，用户 `U152344`，target 为 `Internet & Social Platforms`，稳定映射为 `<interest_065>`。

Behavior 原始 prompt：

> The headlines of the news items I recently viewed are listed below:
>
> 1. The News In Cartoons
>
> Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面列出了我最近浏览的新闻标题：
>
> 1. 漫画新闻
>
> 根据这些标题，预测我接下来可能感兴趣的一个新主题类别。预测类别应不同于浏览历史中体现的主题。只输出分配给该类别的已学习兴趣 token，格式为 `<interest_NNN>`。

Interest 原始 prompt：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Trivia & Puzzles
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出了我最近浏览新闻对应的主题类别：
>
> 1. 冷知识与谜题
>
> 根据该主题类别序列，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在上面的序列中。只输出分配给该类别的已学习兴趣 token，格式为 `<interest_NNN>`。

两路 target response 均为：

> `<interest_065>`

它通过冻结词表映射回 `Internet & Social Platforms`。

## 3. 测试设计（Testing）

### 3.1 测试模型与位置

四路正式实验在 AMLT/Feeds 的 `80G4-A100` 上运行，实验名为 `static-semantic-id-token-recovery-full-0819-v1`。固定训练设置为3 epochs、2,025 optimizer steps、seed 42、BF16、global batch 32、LoRA `r=16/alpha=32/dropout=0.05`、all-linear、completion-only loss、最大长度2048、W&B disabled。Qwen3.5 学习率为 `2e-5`，Llama 学习率为 `2e-4`。

每个模型先注册108个 additional special tokens，并逐个验证为单 token。input embedding 与 untied `lm_head` 分别按类别名称原始 token 向量均值初始化，并恢复到组成 token 的平均范数；只训练108个新增 token 行，同时训练 all-linear LoRA。最终 adapter 必须只包含 LoRA 与紧凑的108行 token delta，禁止保存完整 embedding 矩阵。

评测不使用 Beam：每条 Dev prompt 只做一次 forward，读取最后有效位置的 next-token logits，从108个 ID Token logits 中直接排序并输出 Top-20。四张 GPU 分别计算一个逻辑 shard，合并器拒绝缺失、重复、顺序错误、身份哈希冲突、未知 ID 或重复 Top-20。

### 3.2 测试数据来源

grouped Dev 来自同一 Semantic-ID 数据目录，含31,392个 impressions、27,082个用户、37,728个 target occurrences；104个兴趣在 Dev 中有 support，另4个训练兴趣不进入 Interest-Macro 分母。Behavior 与 Interest 的 impression、用户、target 和 record 顺序严格配对。

评测始终在完整108类上排序，不根据 History 屏蔽任何类别。非法或缺失预测不得从分母删除。

### 3.3 配对测试样例

来源：Dev impression `362173`，用户 `U737543`，正确兴趣为 `Celebrity & Creator Culture`，对应 `<interest_019>`。

Behavior 原始 prompt：

> The headlines of the news items I recently viewed are listed below:
>
> 1. The most powerful LGBTQ movies
>
> Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面列出了我最近浏览的新闻标题：
>
> 1. 最具影响力的 LGBTQ 电影
>
> 根据这些标题，预测我接下来可能感兴趣的一个新主题类别。预测类别应不同于浏览历史中体现的主题。只输出分配给该类别的已学习兴趣 token，格式为 `<interest_NNN>`。

Interest 原始 prompt：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Film
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出了我最近浏览新闻对应的主题类别：
>
> 1. 电影
>
> 根据该主题类别序列，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在上面的序列中。只输出分配给该类别的已学习兴趣 token，格式为 `<interest_NNN>`。

在四路已验收 prediction 中，该样例呈现出 Behavior 与 Interest 输入的真实差异：

| 条件 | Top-5 ID Token（映射后的兴趣） | 正确类别 rank | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Qwen Behavior | `<interest_025>`（Comics & Graphic Narratives）；`<interest_045>`（Film）；`<interest_032>`（Crime & Public Safety）；`<interest_019>`（Celebrity & Creator Culture）；`<interest_073>`（Music） | 4 | 1 | 0.4307 |
| Qwen Interest | `<interest_019>`（Celebrity & Creator Culture）；`<interest_032>`（Crime & Public Safety）；`<interest_025>`（Comics & Graphic Narratives）；`<interest_005>`（American Football）；`<interest_097>`（Television & Streaming） | 1 | 1 | 1.0000 |
| Llama Behavior | `<interest_102>`（Visual Arts & Design）；`<interest_031>`（Crafts & Collecting）；`<interest_079>`（Pets & Animal Companions）；`<interest_019>`（Celebrity & Creator Culture）；`<interest_001>`（Adventure & Nature Travel） | 4 | 1 | 0.4307 |
| Llama Interest | `<interest_019>`（Celebrity & Creator Culture）；`<interest_005>`（American Football）；`<interest_032>`（Crime & Public Safety）；`<interest_031>`（Crafts & Collecting）；`<interest_028>`（Consumer Technology） | 1 | 1 | 1.0000 |

两行都来自 `record_index=30123` 的 merged prediction；Behavior 与 Interest 的输入 SHA-256 分别为 `a5f9546652b6d044492f8aad445ee22451f21575ae9d36910d50a7039656516b` 与 `ae73bca1c49e4f9da5ea67c599739c02275e32cdc8ac354a789dc86e0e8368f4`。这里仅报告模型实际排序，不把单个成功案例解释为总体效果。

### 3.4 指标计算

本实验不使用 LLM judge。对于兴趣类别 `y` 和 cutoff `K`，先在 target 为 `y` 的所有 Dev occurrence 上计算命中率与 rank 折扣收益，再对104个有 support 类别等权平均：

\[
\mathrm{InterestMacroRecall@K}=\frac{1}{104}\sum_y
\frac{\#\{y\text{ appears in Top-K}\}}
{\#\{\text{Dev targets equal }y\}},
\]

\[
\mathrm{InterestMacroNDCG@K}=\frac{1}{104}\sum_y
\frac{\sum \mathbf{1}[r_y\le K]/\log_2(r_y+1)}
{\#\{\text{Dev targets equal }y\}}.
\]

同时报告 @1/@5/@10/@20、impression-level Recall/NDCG/Hit/MRR、Top-1 类别覆盖、预测熵与类别分布。每条 prediction 的 Top-20 必须是108类词表内20个不同 ID。

## 4. 实验结果（Results）

四路 Semantic-ID 均已满足训练、评测和本地独立审计门。冻结的未屏蔽 History 对照与最终结果如下。

| 方法 | 模型 | 输入 | Interest-Macro Recall@10 | Interest-Macro NDCG@10 | History 屏蔽 |
|---|---|---|---:|---:|---|
| Semantic-ID SFT | Qwen3.5-9B | Behavior | 17.2047% | 8.0849% | 否 |
| Semantic-ID SFT | Qwen3.5-9B | Interest | **18.8753%** | **9.4266%** | 否 |
| Semantic-ID SFT | Llama-3.1-8B-Instruct | Behavior | 17.6902% | 8.3849% | 否 |
| Semantic-ID SFT | Llama-3.1-8B-Instruct | Interest | 16.8207% | 8.4664% | 否 |
| Text-label SFT | Qwen3.5-9B | Behavior | 14.5035% | 7.2633% | 否 |
| Text-label SFT | Qwen3.5-9B | Interest | 16.0568% | 8.1487% | 否 |
| Text-label SFT | Llama-3.1-8B-Instruct | Behavior | 15.4729% | 7.2166% | 否 |
| Text-label SFT | Llama-3.1-8B-Instruct | Interest | 15.9476% | 7.9154% | 否 |
| NRMS（非 SFT 参考） | MiniLM-NRMS | Behavior | 15.9787% | 7.7508% | 否 |
| NRMS（非 SFT 参考） | MiniLM-NRMS | Interest | 17.5138% | **9.7166%** | 否 |

NRMS 行特意使用 `*-nrms-unfiltered-eval-v1` 结果；旧报告中19.2179%/9.5065%和20.4840%/11.1818%的 NRMS 分数进行了 History 类别屏蔽，不进入本实验对照。

Semantic-ID 相对同模型 text-label SFT 在四个条件的两个主指标上全部提升。最佳 Semantic-ID 条件是 Qwen Interest：Macro Recall@10 为18.8753%，Macro NDCG@10 为9.4266%。它相对 Qwen text-label Interest 分别提高 `+2.8186/+1.2779` 个百分点；相对未屏蔽 NRMS Interest，Recall 高 `+1.3615` 个百分点，但 NDCG 低 `-0.2900` 个百分点。Behavior 输入下 Llama Semantic-ID 最强，为17.6902%/8.3849%。这说明单 token 方案不仅消除了长标签累积概率问题，也形成了更好的总体排序；但没有 Random-init ID Token 对照，不能把全部提升单独归因于语义初始化。

四路本地独立审计均读取31,392条 Dev，验证训练状态确实达到 `global_step=max_steps=2,025` 和3 epochs；逐条验证 `record_index`、impression、输入 SHA-256、20个不同合法 ID、token/label 映射与有限分数；再从37,728个 target occurrences 独立重算104类 Interest-Macro 指标。远端与本地数值逐值一致，且 `history_prediction_filtering=false`。

### 4.1 完整 cutoff、覆盖与排序质量

| 模型 | 输入 | Macro R@1 | Macro R@5 | Macro R@10 | Macro R@20 | Macro N@1 | Macro N@5 | Macro N@10 | Macro N@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 | Behavior | 1.8827% | 8.9561% | 17.2047% | 29.8658% | 1.8827% | 5.4682% | 8.0849% | 11.2705% |
| Qwen3.5 | Interest | 2.8918% | 10.9000% | 18.8753% | 31.2085% | 2.8918% | 6.8786% | 9.4266% | 12.5068% |
| Llama | Behavior | 1.9529% | 9.3522% | 17.6902% | 29.4916% | 1.9529% | 5.7269% | 8.3849% | 11.3310% |
| Llama | Interest | 2.7234% | 9.6660% | 16.8207% | 28.8212% | 2.7234% | 6.1751% | 8.4664% | 11.4753% |

| 模型 | 输入 | Impression Recall@10 | Impression NDCG@10 | Hit@10 | MRR@20 | Top-1 覆盖 | 最大 Top-1 占比 | 归一化 Top-1 熵 | restricted entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 | Behavior | 20.8281% | 10.3237% | 23.4582% | 8.8687% | 106/108 | 7.7791% | 0.8484 | 4.5105 |
| Qwen3.5 | Interest | 31.6147% | 17.4867% | 35.1969% | 15.2994% | 108/108 | 12.8886% | 0.7885 | 4.4407 |
| Llama | Behavior | 19.4343% | 9.5835% | 22.0725% | 8.2377% | 108/108 | 3.7430% | 0.9754 | 3.7891 |
| Llama | Interest | 25.9574% | 13.6190% | 29.2176% | 11.7458% | 108/108 | 6.1130% | 0.9262 | 4.0197 |

Qwen Interest 的主指标和 impression-level 指标最高，但 Top-1 分布也最集中：最大类别占12.8886%，归一化熵0.7885。Llama 两路 Top-1 更均匀，尤其 Behavior 覆盖108类且最大占比仅3.7430%。因此 Qwen Interest 的更高效果伴随更强的类别偏好，后续应继续检查 per-interest 尾部表现，而不能只看总体 Recall。

### 4.2 相对 Text-label 与未屏蔽 NRMS 的差值

下表单位均为百分点，正数表示 Semantic-ID 更高。

| 模型 | 输入 | vs Text R@10 | vs Text N@10 | vs NRMS R@10 | vs NRMS N@10 |
|---|---|---:|---:|---:|---:|
| Qwen3.5 | Behavior | +2.7013 | +0.8215 | +1.2260 | +0.3341 |
| Qwen3.5 | Interest | +2.8186 | +1.2779 | +1.3615 | -0.2900 |
| Llama | Behavior | +2.2173 | +1.1683 | +1.7115 | +0.6342 |
| Llama | Interest | +0.8731 | +0.5510 | -0.6931 | -1.2502 |

最稳定的结论是 Semantic-ID 全面优于对应 text-label SFT。与 NRMS 的关系则取决于输入和指标：两个 Behavior 条件都超过 NRMS；Qwen Interest 的 Recall 超过 NRMS、NDCG 略低；Llama Interest 两项均低于 NRMS。换言之，Semantic-ID 解决了生成标签表示上的一部分问题，但高 rank 位置的个性化排序仍有提升空间。

### 4.3 单步 logits 吞吐与 32-Beam 对照

| 模型 | 输入 | Semantic-ID 4卡 wall time | 吞吐 | Text-label 32-Beam 最大 shard runtime | 直接 wall-time 比 |
|---|---|---:|---:|---:|---:|
| Qwen3.5 | Behavior | 915.24s | 34.30 impressions/s | 6,241.38s | 6.8× |
| Qwen3.5 | Interest | 619.78s | 50.65 impressions/s | 251.31s | 不可比 |
| Llama | Behavior | 578.65s | 54.25 impressions/s | 11,109.56s | 19.2× |
| Llama | Interest | 337.17s | 93.10 impressions/s | 10,559.69s | 31.3× |

四路 Semantic-ID 都是 clean evaluation，`generated_samples=31,392`、`resumed_samples=0`。Text-label Beam 的 Qwen Behavior 每个 shard 已复用5,056–5,120/7,848条，Llama Behavior 各复用1,216条，所以6.8×和19.2×是混有断点续跑的保守观测值，不是严格 clean-run benchmark。Qwen Interest Beam 每个 shard 已复用7,744–7,848条，剩余运行时间不足以代表完整 Beam 成本，因此不计算加速比。Llama Interest Beam 无复用，31.3×是四组中最可直接解释的观测比较。

### 4.4 原标签 token 长度相关性

Llama 原始 tokenizer 的108个标签 token 数分布为1/2/3/4/5/6/7/8/9/10 token 分别2/16/10/28/27/15/3/1/1/1类；Qwen 为1/2/3/4/5/6/7/8/10 token 分别2/16/10/30/30/13/4/2/1类。以下在104个 Dev 支持类别上计算 Spearman 相关。

| 模型 | 输入 | 指标 | Text-label 性能与长度 ρ（p） | Semantic-ID 性能与长度 ρ（p） | Semantic-ID−Text 增益与长度 ρ（p） |
|---|---|---|---:|---:|---:|
| Qwen3.5 | Behavior | Recall@10 | +0.0416（0.6748） | -0.0411（0.6789） | -0.1592（0.1066） |
| Qwen3.5 | Behavior | NDCG@10 | +0.0638（0.5202） | -0.0208（0.8342） | -0.1871（0.0572） |
| Qwen3.5 | Interest | Recall@10 | +0.0655（0.5088） | -0.0343（0.7295） | -0.1484（0.1326） |
| Qwen3.5 | Interest | NDCG@10 | +0.0718（0.4691） | +0.0229（0.8179） | -0.1124（0.2557） |
| Llama | Behavior | Recall@10 | -0.2365（0.0157） | -0.1517（0.1243） | +0.0039（0.9690） |
| Llama | Behavior | NDCG@10 | -0.2306（0.0185） | -0.1243（0.2088） | +0.0051（0.9588） |
| Llama | Interest | Recall@10 | -0.1482（0.1333） | -0.0522（0.5987） | -0.0223（0.8226） |
| Llama | Interest | NDCG@10 | -0.1382（0.1619） | -0.0234（0.8139） | -0.0528（0.5944） |

Llama text-label Behavior 中，标签越长、per-interest Recall/NDCG 越低的负相关在 Semantic-ID 后减弱；但增益本身与长度几乎无关。Qwen 的 text-label 长度相关本来就接近0，Semantic-ID 增益与长度只有弱负相关。综合来看，现有结果不支持“提升主要由长标签类别贡献”的强解释；更可能是单步、同长度类别打分同时消除了序列概率与生成路径偏差。这里是未经多重检验校正的描述性相关，不作为独立显著性结论。

最终四路均满足：3 epochs/2,025 completed steps；非空紧凑 adapter；31,392条 merged predictions；37,728个 targets；104个 Macro 类别；每条20个不同合法 ID；无缺失、重复、顺序或身份哈希冲突；`history_prediction_filtering=false`；本地独立重算与远端 metrics 一致。

## 5. 数据与产物路径（Artifact Paths）

数据与实现：

- 数据根目录：`data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1`
- 映射表：`data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1/interest_token_mapping.parquet`
- 数据构建器：`src/new_interest/data/build_semantic_id_token_sft_dataset.py`
- Semantic-ID 实现：`src/new_interest/training/semantic_id_tokens.py`
- 训练入口：`src/new_interest/training/train_sft.py`
- 单步 logits 评测：`src/new_interest/evaluation/evaluate_id_token_ranking.py`
- shard 合并：`src/new_interest/evaluation/merge_id_token_shards.py`
- 本地独立审计：`src/new_interest/experiments/audit_semantic_id_token_results.py`
- 受限旧缓存清理：`src/new_interest/experiments/prepare_semantic_id_token_eval_recovery.py`
- AMLT wrapper：`scripts/amlt/sft/run_static_user_disjoint_semantic_id_token_sft.sh`
- AMLT full 配置：`configs/amlt/static_semantic_id_token_recovery_full_0819_v1.yaml`
- Qwen Interest 恢复 wrapper：`scripts/amlt/evaluation/run_semantic_id_token_eval_recovery.sh`
- Qwen Interest 恢复配置：`configs/amlt/static_semantic_id_token_qwen_interest_eval_recovery_0820_v1.yaml`

远端 AMLT：

- 实验：`static-semantic-id-token-recovery-full-0819-v1`
- Qwen Behavior：`qwen35-behavior-semantic-id-recovery-full`
- Qwen Interest：`qwen35-interest-semantic-id-recovery-full`
- Llama Behavior：`llama31-behavior-semantic-id-recovery-full`
- Llama Interest：`llama31-interest-semantic-id-recovery-full`
- Qwen Interest 仅评测恢复实验：`static-semantic-id-token-qwen-interest-eval-recovery-0820-v1`
- Qwen Interest 仅评测恢复 job：`qwen35-interest-semantic-id-eval-recovery`（`pass`）

原 Qwen Interest full job 已完成 checkpoint-2025 与 final adapter，但在评测阶段因旧 shard 的 evaluation identity `29835edb...` 与终态 adapter identity `2b815886...` 不一致而失败。恢复任务只删除 `eval/shard-00..03/shards` 四个旧缓存目录，保留 checkpoints、final adapter 和失败日志；清理证据保存在该条件的 `stale_cache_cleanup_manifest.json`。恢复评测的新 identity 为 `2b8158862db692544e5dc985b10673364b7c99a318f44559afc722406cd06233`。

冻结对照：

- Text-label SFT：`artifacts/static-sud-beam/{qwen35,llama31}-{behavior,interest}/merged/metrics.json`
- Behavior NRMS 未屏蔽：`artifacts/future-novel-interest/static-user-disjoint-sft21600-h50-nrms-unfiltered-eval-v1/conditions/dev_exploratory/selected_nrms_unfiltered/metrics.json`
- Interest NRMS 未屏蔽：`artifacts/future-novel-interest/static-user-disjoint-sft21600-h50-interest-nrms-unfiltered-eval-v1/conditions/dev_exploratory/selected_nrms_unfiltered/metrics.json`

当前报告：`docs/experiments/2026-08-18/static_user_disjoint_semantic_id_token_sft.md`。四路 recovery full 的统一本地根目录为 `artifacts/static-sud-semantic-id-token/static-semantic-id-token-recovery-full-0819-v1/`。

已验收的 Qwen Behavior：

- condition：`artifacts/static-sud-semantic-id-token/static-semantic-id-token-recovery-full-0819-v1/qwen35-behavior-semantic-id-recovery-full`
- adapter：`sft/final_adapter/adapter_model.safetensors`（176,720,344 bytes）
- predictions：`eval/merged/id_token_predictions.parquet`，SHA-256 `1a24eb7967cd117086f088896b5172e6436e24ac173bb419cf5553ffee41d7f8`
- metrics：`eval/merged/metrics.json`，SHA-256 `c080061333f995ab0694eb9ac7a9a91fef7e5c51e2b64043d5db69c6c0df54b5`
- per-interest：`eval/merged/per_interest_metrics.parquet`，SHA-256 `50548e4c3463b09a947b1ac289fb1423c1993154a6f7e29eaced0b7ace8806f8`
- merge manifest：`eval/merged/merge_manifest.json`，SHA-256 `9099cc74798f1aedeb72cdf263f374416bb3f0fc27338be1917589b2c4e7178b`
- local audit：`local_audit.json`

已验收的 Qwen Interest：

- condition：`artifacts/static-sud-semantic-id-token/static-semantic-id-token-recovery-full-0819-v1/qwen35-interest-semantic-id-recovery-full`
- adapter：`sft/final_adapter/adapter_model.safetensors`（176,720,344 bytes）
- predictions：`eval/merged/id_token_predictions.parquet`，SHA-256 `fb6d1282f971357bdd349eed4478fb09ddcabf30b82b74295c696bf2f5eff69d`
- metrics：`eval/merged/metrics.json`，SHA-256 `12d65a6d56d9df6d5b9733934f840d24ecbe988e48b23ef357bcd32bebb226c0`
- per-interest：`eval/merged/per_interest_metrics.parquet`，SHA-256 `f48e5d18dc6283bb6f13c22c131ca2223850b2b491801fccffa74a6048132b15`
- merge manifest：`eval/merged/merge_manifest.json`，SHA-256 `2f65bff288e434d0455f20e139b914d0200d259caf6731f2a74f4d836fef7a56`
- cleanup manifest：`stale_cache_cleanup_manifest.json`
- local audit：`local_audit.json`

已验收的 Llama Behavior：

- condition：`artifacts/static-sud-semantic-id-token/static-semantic-id-token-recovery-full-0819-v1/llama31-behavior-semantic-id-recovery-full`
- adapter：`sft/final_adapter/adapter_model.safetensors`（171,371,504 bytes）
- predictions：`eval/merged/id_token_predictions.parquet`，SHA-256 `428b976113a7d606a75f75d13c40cc90c72a068a3d4b3f31906831a87c9844b1`
- metrics：`eval/merged/metrics.json`，SHA-256 `8ba0439183f8b2009aba6f0b4ec07cf2a063dca1b63ea5d36e315867bce9d73b`
- per-interest：`eval/merged/per_interest_metrics.parquet`，SHA-256 `cfbad261b9f11b0bb02c8e0caee0422e3cd12b1f1a231b33ed69462e6b049ff8`
- merge manifest：`eval/merged/merge_manifest.json`，SHA-256 `b26a5d59d69d9513fb83f41d60c4ebd1f842695b97052fc039ddb21a2fbb2577`
- local audit：`local_audit.json`

已验收的 Llama Interest：

- condition：`artifacts/static-sud-semantic-id-token/static-semantic-id-token-recovery-full-0819-v1/llama31-interest-semantic-id-recovery-full`
- adapter：`sft/final_adapter/adapter_model.safetensors`（171,371,504 bytes）
- predictions：`eval/merged/id_token_predictions.parquet`，SHA-256 `06992edeb41a906e93f4902a6ae774c2e2d2c02a724e016106addb11ca8e7e9a`
- metrics：`eval/merged/metrics.json`，SHA-256 `a1ea9dea25c6e965e1f96c4a0f1cb0855d5419e0d3005255a2a978fea6dfb252`
- per-interest：`eval/merged/per_interest_metrics.parquet`，SHA-256 `c55d5f3c1a50e0426b88ca3113323ba56a141a72a94ceb0356c4a2addec97744`
- merge manifest：`eval/merged/merge_manifest.json`，SHA-256 `a933506c5b065ee749ec400130b722f2ea2efcd425b4b1662a7d1fad7c00e7d2`
- local audit：`local_audit.json`

四个 `local_audit.json` 均为 `status=passed`，是本文最终验收的机器可读入口。
