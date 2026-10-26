# Text-label T0-E / T1 消融：Semantic-ID 到底优化了什么

> 状态：T0-E 与 T1 四个模型×输入条件均已完成并通过本地独立审计。本文所有结果均为未屏蔽 History 的完整 108 类 Dev 排序；报告同时包含 @1/@5/@10/@20，正文重点展示 @10 与 @20。

## 1. 研究问题（Research Questions）

本实验拆分 Semantic-initialized ID Token 相对 Text-label SFT 的收益来源：

1. **推理长度/搜索路径效应**：冻结原 Text-label adapter，将 32-Beam（T0 Beam）替换为对全部 108 个标签计算类别名称 token 平均 log-probability 的 exhaustive 排序（T0-E）。
2. **训练长度加权效应**：保持 Text-label 输出，改用每个样本先对 completion token 求均值、再在 batch 内平均的 T1 loss；分别用 exhaustive（T1-E）和原 32-Beam（T1 Beam）评测。
3. **单 token 表示的剩余收益**：比较 Semantic-ID 与已经做过训练长度归一化的 T1，判断收益能否被“长标签在训练和推理中吃亏”解释。

四个配对条件为 Qwen3.5-9B / Llama-3.1-8B-Instruct × Behavior / Interest。主要指标为 104 个有 Dev support 兴趣等权平均的 Interest-Macro Recall 与 NDCG。

## 2. 训练数据构造（Training Data Construction）

T0-E 不重新训练，直接冻结原 Text-label SFT adapter。T1 逐条复用 `sft_static_user_disjoint_balanced_200_h50_v1`：Behavior 与 Interest 各 21,600 条、21,600 个不同用户、108 类×每类 200 用户；两路用户、impression、target、时间和样本顺序严格配对。History 最长 50，Train/Dev 用户交集为 0。

T0 的 completion-only token loss 为整个 batch 的 token 平均；T1 改为严格的 per-example mean：

\[
L_{T1}=\frac{1}{B}\sum_{i=1}^{B}\frac{1}{T_i}\sum_{t=1}^{T_i}\ell_{it}.
\]

因此 T1 只消除“completion 较长的样本在训练目标中权重更大”这一项，不改变标签文字、tokenizer、数据、LoRA、训练步数或解码候选空间。

| 模型 | 输入 | 样本 | 训练完成 | T1 train loss | adapter bytes |
|---|---|---:|---|---:|---:|
| Qwen3.5-9B | Behavior | 21,600 | 3 epochs / 2,025 steps | 0.5730 | 173,181,072 |
| Qwen3.5-9B | Interest | 21,600 | 3 epochs / 2,025 steps | 0.5598 | 173,181,072 |
| Llama-3.1-8B-Instruct | Behavior | 21,600 | 3 epochs / 2,025 steps | 0.5955 | 167,832,240 |
| Llama-3.1-8B-Instruct | Interest | 21,600 | 3 epochs / 2,025 steps | 0.5997 | 167,832,240 |

四路均为 4-GPU、global batch 32、seed 42、BF16、completion-only loss、max length 2048、LoRA `r=16/alpha=32/dropout=0.05`、all-linear；Qwen LR `2e-5`，Llama LR `2e-4`。本地审计读取 checkpoint-2025 的 `trainer_state.json`，确认 `global_step=max_steps=2,025`、epoch=3.0、loss 有限、无超长样本、adapter 非空。

### 2.1 配对训练样例

来源：`source_sample_id=train:1820310:100`，target 为 `Internet & Social Platforms`。

Behavior 原始输入：

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
> 根据这些标题，预测我接下来可能感兴趣的一个新主题类别。预测类别应不同于浏览历史中体现的主题。请以“New topic: ”开头，并且之后只输出类别名称。

Interest 原始输入中的唯一历史类别为 `Trivia & Puzzles`，其余任务要求相同。两路训练 response 均为：

> New topic: Internet & Social Platforms

中文翻译：

> 新主题：互联网与社交平台

## 3. 测试设计（Testing）

### 3.1 模型、位置与评测条件

正式任务运行于 AMLT/Feeds `80G4-A100`：

- T0-E Qwen：`static-text-label-t0e-qwen-full-0824-v1`；
- T0-E Llama：`static-text-label-t0e-llama-full-0824-v1`；
- T1 训练及 Llama 评测：`static-text-label-t1-full-0824-v1`；
- 已完成 Qwen T1 adapter 的仅评测恢复：`static-text-label-t1-qwen-eval-recovery-0825-v1`。

四种 Text-label 评测定义如下：

| 条件 | adapter | 排序方式 |
|---|---|---|
| T0 Beam | 原 Text-label SFT | 受限 32-Beam，Top-20 |
| T0-E | 原 Text-label SFT | exhaustive 108 类；类别名称 token mean log-probability；不计固定前缀与 EOS |
| T1 Beam | per-example-mean Text-label SFT | 与 T0 相同的受限 32-Beam |
| T1-E | per-example-mean Text-label SFT | 与 T0-E 相同的 exhaustive 评分 |

Semantic-ID 使用语义初始化的单 ID Token，在一个 next-token 位置上对 108 个 ID logits 做 restricted ranking。T0-E 与 Semantic-ID 都是 exhaustive，但前者仍将一个类别表示为多个原始语言 token 的条件概率链，后者把每类表示为一个独立可训练输出行，因此二者不等价。

### 3.2 测试数据来源

Behavior/Interest grouped Dev 各含 31,392 impressions、27,082 users、37,728 target occurrences；104 个兴趣有 support，候选词表固定为 108 类。两路 impression、user、timestamp、history length、targets 和顺序经本地逐条核验完全一致。

所有评测均保留 History 中出现过的类别，不做候选屏蔽或预测后处理。每条记录必须输出 20 个不同且合法的类别；缺失、重复、顺序错误、input hash 或 evaluation identity 冲突均视为失败。

### 3.3 测试样例

Dev `record_index=30123`，impression `362173`，Interest History 为 `Film`，target 为 `Celebrity & Creator Culture`。

原始 prompt：

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
> 根据该序列预测我接下来可能感兴趣的新主题。预测类别不应出现在上述序列中。请以“New topic: ”开头，并且之后只输出类别名称。

Qwen Interest 的真实排序为：

| 方法 | Top-5 | target rank |
|---|---|---:|
| T0 Beam | Celebrity & Creator Culture; Crime & Public Safety; Shopping & Gift Ideas; Government & Public Policy; Medicine & Medical Research | 1 |
| T0-E | Floods, Droughts & Wildfires; Gardening & Landscaping; Weather Forecasts & Seasonal Conditions; Civic Life & Public Commemorations; Comics & Graphic Narratives | 6 |
| T1 Beam | Celebrity & Creator Culture; Crime & Public Safety; Government & Public Policy; Medicine & Medical Research; Shopping & Gift Ideas | 1 |
| T1-E | Gardening & Landscaping; Floods, Droughts & Wildfires; Weather Forecasts & Seasonal Conditions; Civic Life & Public Commemorations; Comics & Graphic Narratives | 6 |
| Semantic-ID | Celebrity & Creator Culture; Crime & Public Safety; Comics & Graphic Narratives; American Football; Television & Streaming | 1 |

该样例只用于展示不同评分路径的实际输出，不能替代总体指标。

### 3.4 指标与审计

不使用 LLM judge。对每个有 support 的兴趣分别计算 target occurrence 的 Recall@K 和折扣增益，再对 104 类等权平均得到 Interest-Macro Recall/NDCG@K。完整 cutoff 为 1/5/10/20。

本地审计从原始 Dev 与 merged predictions 重新计算指标，并验证：31,392 条记录、37,728 targets、104 个 Macro 类别、108 类词表、20 个不同合法 Top-20、record order、impression、input SHA-256、evaluation identity、adapter identity、merged prediction/metrics/per-interest 文件 SHA-256。16 组 Text-label 评测均通过，远端与本地重算值在 `1e-12` 容差内一致。

## 4. 实验结果（Results）

### 4.1 完整主结果

| 模型×输入 | 方法 | Macro R@10 | Macro N@10 | Macro R@20 | Macro N@20 |
|---|---|---:|---:|---:|---:|
| Qwen Behavior | T0 Beam | 14.5035% | 7.2633% | 27.7357% | 10.5685% |
|  | T0-E | 12.0196% | 5.6414% | 23.1813% | 8.4487% |
|  | T1 Beam | 15.2146% | 7.6891% | 28.3197% | 10.9574% |
|  | T1-E | 12.6343% | 5.8968% | 23.8996% | 8.7299% |
|  | Semantic-ID | **17.2047%** | **8.0849%** | **29.8658%** | **11.2705%** |
| Qwen Interest | T0 Beam | 16.0568% | 8.1487% | 29.4989% | 11.5062% |
|  | T0-E | 13.2078% | 6.3331% | 24.3555% | 9.1117% |
|  | T1 Beam | 16.2660% | 8.2951% | 29.3603% | 11.5885% |
|  | T1-E | 13.0988% | 6.1825% | 24.1233% | 8.9479% |
|  | Semantic-ID | **18.8753%** | **9.4266%** | **31.2085%** | **12.5068%** |
| Llama Behavior | T0 Beam | 15.4729% | 7.2166% | 27.1840% | 10.1306% |
|  | T0-E | 13.5280% | 6.5624% | 24.6414% | 9.3326% |
|  | T1 Beam | 14.8098% | 7.1412% | 27.0908% | 10.2270% |
|  | T1-E | 13.3038% | 6.3886% | 24.1859% | 9.1081% |
|  | Semantic-ID | **17.6902%** | **8.3849%** | **29.4916%** | **11.3310%** |
| Llama Interest | T0 Beam | 15.9476% | 7.9154% | 27.2418% | 10.7418% |
|  | T0-E | 14.1158% | 6.7657% | 26.5044% | 9.8379% |
|  | T1 Beam | 16.0209% | 8.0994% | 27.5165% | 10.9808% |
|  | T1-E | 14.6440% | 7.0226% | 26.0952% | 9.9010% |
|  | Semantic-ID | **16.8207%** | **8.4664%** | **28.8212%** | **11.4753%** |

@1/@5 的完整数值保存在 `comparison.csv` 与 `final_verification.json`。

### 4.2 机制消融

下表为四个模型×输入条件的平均百分点差值；“正向条件”表示差值大于 0 的条件数。

| 对比 | 解释对象 | ΔR@10 | ΔN@10 | ΔR@20 | ΔN@20 | 正向条件（R10/N10/R20/N20） |
|---|---|---:|---:|---:|---:|---|
| T0-E − T0 Beam | 只改推理为 exhaustive label-token mean | -2.2774 | -1.3104 | -3.2444 | -1.5541 | 0/4 · 0/4 · 0/4 · 0/4 |
| T1-E − T0-E | 在相同 exhaustive 评测下只改训练 loss | +0.2024 | +0.0470 | -0.0947 | -0.0109 | 2/4 · 2/4 · 1/4 · 2/4 |
| T1 Beam − T0 Beam | 在相同 Beam 评测下只改训练 loss | +0.0826 | +0.1702 | +0.1567 | +0.2016 | 3/4 · 3/4 · 2/4 · 4/4 |
| Semantic-ID − T1 Beam | 单 ID Token 相对已做训练长度归一化的最佳匹配对照 | +2.0699 | +0.7845 | +1.7750 | +0.7075 | 4/4 · 4/4 · 4/4 · 4/4 |

Semantic-ID 相对每个 T1 Beam 条件的差值如下：

| 模型×输入 | ΔR@10 | ΔN@10 | ΔR@20 | ΔN@20 |
|---|---:|---:|---:|---:|
| Qwen Behavior | +1.9901 | +0.3958 | +1.5461 | +0.3131 |
| Qwen Interest | +2.6094 | +1.1315 | +1.8482 | +0.9184 |
| Llama Behavior | +2.8804 | +1.2437 | +2.4008 | +1.1041 |
| Llama Interest | +0.7998 | +0.3670 | +1.3048 | +0.4945 |

### 4.3 对“Semantic-ID 优化了什么”的判断

**训练样本长度权重不是主要来源。** T1-E 相对 T0-E 只在两个条件的 @10 指标上提高，四条件平均仅 `+0.2024/+0.0470` 个百分点；到 @20 平均变为 `-0.0947/-0.0109`。相同 Beam 评测下，T1 相对 T0 的平均变化也只有 `+0.0826/+0.1702`（@10）和 `+0.1567/+0.2016`（@20），且 Recall 并非四路一致提升。因此，给每个正样本的 completion loss 等权不能复现 Semantic-ID 增益。

**单纯消除推理时的标签长度累积概率也不能解释收益。** T0-E 已对标签 token 求均值并穷举 108 类，但四路均低于 T0 Beam：平均 @10 下降 `2.2774/1.3104` 个百分点，@20 下降 `3.2444/1.5541`。这不表示 exhaustive 原理上更差，而是说明“标签 token 均值”并不等价于干净的 108 类分类：每类仍由不同的自然语言子词链组成，保留了共享子词、条件分解、结束位置与语言先验；同时它与 Beam 的完整序列分数、EOS 和候选搜索路径并非同一评分函数。

**现有证据最支持输出空间重参数化。** Semantic-ID 把“生成一个多 token 类别名称”改成“在同一个时间步、同一个 108 维候选集合中选择一个类别”，每类拥有独立可训练 token 行。这样同时去掉了：

- 类别名称的多步自回归分解和 teacher-forcing 链；
- 类别间共享子词造成的参数耦合；
- EOS/终止概率与不同序列长度的校准差异；
- 32-Beam 截断、重复标签去重和补齐路径；
- 预训练词频对自然语言类别名称的直接偏置。

关键证据是：即使 Text-label 已使用 per-example-mean 训练，并选择四路都明显强于 T1-E 的 T1 Beam 作为更强对照，Semantic-ID 在四个条件、@10/@20 的 Recall 与 NDCG 上仍全部提高，平均增益为 `+2.0699/+0.7845`（@10）与 `+1.7750/+0.7075`（@20）。因此 Semantic-ID 的主要价值更像是把生成问题改造成共享表示上的单步多类决策，而不是只修复“长标签 loss 更大”。

本实验仍不能把剩余收益单独归因于 semantic initialization：没有 Random-init ID Token 条件，因而“单 token 重参数化”和“语义初值”仍绑定在一起。

## 5. 数据与产物路径（Artifact Paths）

实现与配置：

- T1 loss：`src/new_interest/training/train_sft.py`
- T0-E evaluator：`src/new_interest/evaluation/evaluate_text_label_ranking.py`
- T0-E merge：`src/new_interest/evaluation/merge_text_label_ranking_shards.py`
- 本地独立审计：`src/new_interest/experiments/audit_text_label_ablation_results.py`
- AMLT wrapper：`scripts/amlt/sft/run_static_user_disjoint_text_label_ablation.sh`
- Qwen T1 评测恢复配置：`configs/amlt/static_text_label_t1_qwen_eval_recovery_0825_v1.yaml`

本地结果：

- 汇总审计：`artifacts/static-sud-text-label-ablation/final_verification.json`
- 完整 @1/@5/@10/@20 表：`artifacts/static-sud-text-label-ablation/comparison.csv`
- T0-E：`artifacts/static-sud-text-label-ablation/t0e/downloads/`
- T1：`artifacts/static-sud-text-label-ablation/t1/downloads/`
- 冻结 T0 Beam：`artifacts/static-sud-beam/{qwen35,llama31}-{behavior,interest}/merged/`
- Semantic-ID：`artifacts/static-sud-semantic-id-token/static-semantic-id-token-recovery-full-0819-v1/`

远端 AMLT：

- `static-text-label-t0e-qwen-full-0824-v1`：2/2 pass；
- `static-text-label-t0e-llama-full-0824-v1`：2/2 pass；
- `static-text-label-t1-full-0824-v1`：四个 T1 adapter 均完成；Llama 两路完整评测 pass；
- `static-text-label-t1-qwen-eval-recovery-0825-v1`：Qwen Behavior/Interest 仅评测恢复 2/2 pass。
