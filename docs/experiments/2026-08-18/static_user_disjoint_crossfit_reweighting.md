# Static user-disjoint cross-fitted sample reweighting — Exploratory Dev

## 1. 研究问题（Research Questions）

本实验在 `Qwen/Qwen3.5-9B` 上比较四个配对条件：Behavior Vanilla、Behavior Weighted、Interest Vanilla、Interest Weighted。核心问题是：在训练样本与超参数不变时，基于五折 out-of-fold（OOF）类别 token NLL 的类内重加权，是否提高真实 History Dev 上的 Interest-Macro Recall@10 与 Interest-Macro NDCG@10；以及 Behavior/Interest 两种 History 表示下的变化是否一致。

本文只报告 **exploratory Dev**。用户于 2026-08-21 明确停止 shuffled-History 评测；shuffled 对照不属于本报告证据，因此不能把观察到的变化归因于“模型更好地利用了 History”。输出先验、标签覆盖或排序分布变化仍是可能解释。

## 2. 训练数据构造（Training Data Construction）

- Train：21,600 个互异用户，108 个兴趣类别，每类 200 个用户；每个用户只保留一个 Train target。Behavior 与 Interest 条件共享同一批 `source_sample_id`。
- Behavior 输入为按时间排序的原始新闻标题；Interest 输入为同一 History 映射出的兴趣类别序列。
- 五折划分：类别内按 `SHA256("20260818:fold:<source_sample_id>")` 排序后轮转分折。每折 4,320 个 held-out 用户、17,280 个 teacher Train 用户，每类每折 40 个 held-out 用户。
- OOF score：epoch-3 teacher forcing，只计算类别名称 token 的平均 NLL，不包含 `New topic:` 与 EOS。
- Weighted 条件把每类 200 个样本的 NLL rank 转为 `q=rank/199`、`raw=1/(1+q)`，再归一到类内均值 1；Vanilla 条件不使用 sample weight。

| condition | History view | rows/users | classes | sample weights |
|---|---|---:|---:|---|
| Behavior Vanilla | raw headlines | 21,600 | 108 | none |
| Behavior Weighted | raw headlines | 21,600 | 108 | `sample_weights/behavior.parquet` |
| Interest Vanilla | interest sequence | 21,600 | 108 | none |
| Interest Weighted | interest sequence | 21,600 | 108 | `sample_weights/interest.parquet` |

### 真实训练样例

Behavior，`source_sample_id=train:386196:81`：

> **English prompt:** The headlines of the news items I recently viewed are listed below:
>
> 1. Harden-Westbrook argument goes viral
>
> Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
>
> **Target response:** `New topic: American Football`

中文：最近浏览的新闻标题为“哈登与威斯布鲁克的争论走红”。根据这些标题预测下一项可能感兴趣、且不同于既有 History 主题的新类别；回答必须以 `New topic:` 开头且只给类别名。目标回答为“新主题：美式橄榄球”。

Interest，`source_sample_id=train:1835405:95`：

> **English prompt:** The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Music
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Begin your response with "New topic: " and provide only the category name after it.
>
> **Target response:** `New topic: Sports Business & Administration`

中文：最近浏览内容的兴趣类别序列只有“音乐”。根据该序列预测下一项可能感兴趣、且未出现在序列中的新类别；回答必须以 `New topic:` 开头且只给类别名。目标回答为“新主题：体育商业与管理”。

## 3. 测试设计（Testing）

### 3.1 模型与运行位置

- Base model：`Qwen/Qwen3.5-9B`。
- LoRA：`r=16`、`alpha=32`、dropout `0.05`，训练 3 epochs；21,600 rows、2,025 optimizer steps、8-rank DDP、effective global batch 32。
- 推理：AMLT / Feeds，32 beams、Top-20、无采样、无 thinking，并过滤 History 中已有兴趣。
- 本次四条件最终归档不保存 checkpoint 文件；按用户要求只保留远端 checkpoint marker/hash 证据、训练摘要与 final adapter。四份 final adapter 均已本地 SHA-256 校验。

原始 Interest 两个任务均为 `passed`。原始 Behavior 两个任务完成训练后只在评测阶段失败；两个 eval-only recovery 均先完成 `eval/real/merged`，随后因用户停止 shuffled 评测而被精确取消，最终状态是 `killed`，不是 `passed`。

### 3.2 Dev 来源与完整性

Real Dev 含 31,392 个 impression、27,082 个用户、37,728 个 target occurrence、104 个有支持的兴趣类别。Behavior/Interest 的 `(impression_id, user_id, acceptable_subcategories)` 完全配对。

本地独立审计对四个条件分别确认：31,392 个唯一 record identity；每条恰好 20 个互异且位于 108-label vocabulary 内的预测；8 个逻辑分片；104 个有支持兴趣；prediction、metrics、per-interest 与 merge manifest 哈希一致；所有数值有限；并从预测重新计算全部 ranking metrics，与远端 metrics 一致。

### 3.3 真实 Dev 例子

Behavior，`impression_id=184523`，真实 target 为 `American Football`：

> **English prompt:** The headlines of the news items I recently viewed are listed below: 1. MLB umpire Eric Cooper dead at 52. Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history.

中文：最近浏览的标题是“MLB 裁判 Eric Cooper 52 岁去世”。根据该标题预测下一项可能感兴趣、且不同于既有 History 主题的新类别。

- Vanilla Top-10：Baseball & Softball；Crime & Public Safety；Celebrity & Creator Culture；Government & Public Policy；Courts & Legal Cases；Economy & Economic Policy；Companies & Corporate Strategy；War & Armed Conflict；Historical Events & Eras；Medicine & Medical Research。
- Weighted Top-10：Crime & Public Safety；Government & Public Policy；Baseball & Softball；Courts & Legal Cases；Celebrity & Creator Culture；Companies & Corporate Strategy；Economy & Economic Policy；Medicine & Medical Research；**American Football**；Severe Storms。
- 中文概述：Vanilla 的前十名没有真实 target；Weighted 将“美式橄榄球”排到第 9。这个单例只展示度量如何变化，不构成机制证据。

### 3.4 评分过程与含义

本实验没有 LLM judge；评分完全由确定性 label ranking 计算。

- `Recall@K`：对每个 impression，Top-K 命中的可接受 target 数除以该 impression 的可接受 target 数，再对 impression 求平均。
- `NDCG@K`：对命中 target 按排名使用 `1/log2(rank+1)` 折损，并除以该 impression 的理想 DCG，再对 impression 求平均。
- `Interest-Macro Recall/NDCG@K`：先对每个 target interest 的所有 occurrence 求均值，再对 104 个有支持 interest 等权平均；这是主要指标。
- 置信区间：2,000 次 paired user-clustered bootstrap；每次以用户为重采样单位，并在 replicate 内重新计算 interest macro。

## 4. 实验结果（Results）

### 4.1 四条件 real Dev

| condition | Macro R@10 (95% CI) | Macro NDCG@10 (95% CI) | R@20 | Macro R@20 | Hit@1 | unique Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Behavior Vanilla | 0.14788 [0.13831, 0.16025] | 0.07485 [0.06876, 0.08322] | 0.39570 | 0.27451 | 0.02654 | 75 |
| Behavior Weighted | 0.15726 [0.14673, 0.17151] | 0.07765 [0.07132, 0.08641] | 0.41934 | 0.28210 | 0.02609 | 90 |
| Interest Vanilla | 0.16116 [0.15336, 0.17356] | 0.08206 [0.07735, 0.08954] | 0.49321 | 0.29352 | 0.06473 | 83 |
| Interest Weighted | 0.17093 [0.16043, 0.18558] | 0.08541 [0.07994, 0.09319] | 0.49651 | 0.29323 | 0.05944 | 93 |

### 4.2 Weighted − Vanilla 配对差值

| view | Δ Macro R@10 (95% CI) | Δ Macro NDCG@10 (95% CI) | Δ R@20 | Δ Macro R@20 | Δ Hit@1 | Δ unique Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Behavior | +0.00938 [+0.00346, +0.01661] | +0.00280 [-0.00015, +0.00581] | +0.02364 | +0.00759 | -0.00045 | +15 |
| Interest | +0.00977 [+0.00034, +0.01870] | +0.00335 [-0.00065, +0.00703] | +0.00331 | -0.00029 | -0.00529 | +10 |

两种 view 的 Macro Recall@10 点估计都提高约 0.94–0.98 个百分点，paired bootstrap 95% CI 均高于 0。Macro NDCG@10 点估计提高，但 CI 都跨过 0，因此证据较弱。Behavior 的 @20 改善较明显；Interest 的 Macro Recall@20 基本不变。两个 Weighted 条件的 Hit@1 都下降，同时 unique Top-1 增加，说明重加权改变了排序/覆盖分布，而不是在所有 cutoff 上一致改善。

Behavior 与 Interest 的 OOF category-token NLL Pearson 相关为 0.97965；最终 sample weight Pearson 相关为 0.65245，最低权重十分位的 Jaccard 为 0.31547。两种 teacher 的不确定性高度相关，但实际降权集合并不相同。

保守结论：在本次 real exploratory Dev 上，OOF 类内重加权对 Macro Recall@10 有正向信号，并扩大 Top-1 标签覆盖；它没有稳定提高 Top-1 或 Macro Recall@20。由于 shuffled 对照被用户停止且未纳入证据，不能确认收益来自更好利用 History，也不能排除输出先验与覆盖变化。

## 5. 数据与产物路径（Artifact Paths）

- Frozen cross-fit data：`data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1/`
- Behavior/Interest Train：`data/training/sft_static_user_disjoint_balanced_200_h50_v1/train/{behavior_only,interest_only}/21600.parquet`
- Label vocabulary：`data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet`
- Four condition roots：`artifacts/static-sud-crossfit-reweighting/qwen35-v1/final/{behavior-vanilla,behavior-weighted,interest-vanilla,interest-weighted}/`
- Per-condition exact local SHA-256/provenance：`final/<condition>/local_provenance_manifest.json`
- Independent real audit：`artifacts/static-sud-crossfit-reweighting/qwen35-v1/verification/real_results_audit.json`
- Real-only metrics and slices：`artifacts/static-sud-crossfit-reweighting/qwen35-v1/analysis/condition_metrics.parquet`、`per_interest_metrics.parquet`、`slice_metrics.parquet`
- 2,000-replicate bootstrap：`artifacts/static-sud-crossfit-reweighting/qwen35-v1/analysis/bootstrap_intervals.json`
- Analysis manifest：`artifacts/static-sud-crossfit-reweighting/qwen35-v1/analysis/manifest.json`

AMLT provenance：

- Original experiment：`static-sud-crossfit-final-v1-0818`。
- Interest Vanilla：`honest_atemoya_r2htmht5gw`，result `7212788530.09200-da8d2f7e-5551`，passed。
- Interest Weighted：`loving_cheese_9q0h80dssm`，result `7212788530.09170-1390b448-a950`，passed。
- Behavior Vanilla source：result `7212788530.09261-77e8fede-8ba2`；训练完成、原始任务在评测阶段失败。
- Behavior Weighted source：result `7212788530.09227-18411bb2-ac34`；训练完成、原始任务在评测阶段失败。
- Behavior recovery experiment：`static-sud-crossfit-behavior-eval-recovery-v1-0821`。
- Behavior Vanilla recovery：`good_cheetah_vr4plgsswq`，result `7212738953.66495-4a25a0ef-9ccc`；real 合并后被用户取消，最终 killed。
- Behavior Weighted recovery：`willing_quince_dbs2c8smwr`，result `7212738953.66418-d30ebdc1-c487`；real 合并后被用户取消，最终 killed。

Interest 原始 artifact manifest 的所有本地保留文件均无哈希不匹配。按用户要求，57 个 checkpoint 文件/条件不在本地；Interest Weighted 另有一个已排除的 shuffled merged prediction 未保留。上述缺失均不影响四个 real 结果的独立审计。

## 6. Llama-3.1-8B-Instruct real-only 探索性复现（2026-08-23）

### 6.1 范围与完整性

使用同一 frozen Train 与真实 History Dev，在 `meta-llama/Llama-3.1-8B-Instruct` 上复现四个条件。每个条件均训练 21,600 rows、3 epochs、2,025 optimizer steps，使用 8-rank DDP 与 effective global batch 32；loss 全部有限。两个 Vanilla 条件的 `sample_weight_file` 与哈希均为 `null`；Behavior Weighted 仅使用经审计的 Llama Behavior 权重（SHA-256 `3dfd3f68494d9832f86844096a80e29294fb624013c95bdc55f8ea9021c1c330`），Interest Weighted 仅使用 Llama Interest 权重（SHA-256 `6dc7df5ea7fe4f13409a6abf6d9501caeed05dd4a589eea1bc9f933f361e4182`）。

四个最终任务均为 `passed`。每个条件的独立本地审计均重新计算 @1/@5/@10/@20 指标，并确认 31,392 个唯一 Top-20 prediction rows、8 个逻辑分片、104 个有支持兴趣、Behavior/Interest 输入 identity 完全配对、所有指标与 score 有限，且重算结果与远端 metrics 精确一致。四份 manifest 各有 614 个条目；本地逐文件验证 554 个允许保留的 payload，0 个哈希不匹配，另外 60 个 checkpoint 文件按策略刻意不下载。最终本地 checkpoint 目录与文件数均为 0。

本次 Llama 复现严格为 **real-only exploratory Dev**：没有运行、下载或分析任何 shuffled-History evaluation。因而以下差值不能识别 History 利用机制，也不能排除输出先验、类别覆盖或随机训练波动。

### 6.2 Llama 四条件 real Dev

| condition | Macro R@10 (95% CI) | Macro NDCG@10 (95% CI) | R@20 | Macro R@20 | Hit@1 | unique Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Behavior Vanilla | 0.15487 [0.14002, 0.16566] | 0.07489 [0.06773, 0.08157] | 0.40018 | 0.26222 | 0.02784 | 98 |
| Behavior Cross-fitted | 0.16029 [0.14375, 0.17228] | 0.07537 [0.06859, 0.08225] | 0.39478 | 0.28034 | 0.03093 | 98 |
| Interest Vanilla | 0.15922 [0.14767, 0.17497] | 0.07831 [0.07261, 0.08641] | 0.47724 | 0.27649 | 0.05565 | 93 |
| Interest Cross-fitted | 0.15887 [0.14825, 0.17306] | 0.07982 [0.07357, 0.08809] | 0.46274 | 0.27866 | 0.05422 | 96 |

### 6.3 Llama Cross-fitted − Vanilla 配对差值

| view | Δ Macro R@10 (95% CI) | Δ Macro NDCG@10 (95% CI) | Δ R@20 | Δ Macro R@20 | Δ Hit@1 | Δ unique Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Behavior | +0.00541 [-0.00506, +0.01537] | +0.00048 [-0.00404, +0.00545] | -0.00539 | +0.01812 | +0.00309 | 0 |
| Interest | -0.00035 [-0.01000, +0.00776] | +0.00151 [-0.00207, +0.00475] | -0.01450 | +0.00217 | -0.00143 | +3 |

置信区间来自 2,000 次 paired user-clustered bootstrap，并在每个 replicate 内重新计算 interest macro。两个 view 的主要 @10 差值区间都跨过 0：Behavior 的 Macro Recall@10 点估计为 +0.54 个百分点，Interest 为 -0.04 个百分点；Macro NDCG@10 分别为 +0.05 与 +0.15 个百分点。Behavior 的 Macro R@20 点估计增加 1.81 个百分点，但整体 R@20 下降 0.54 个百分点；Interest 的整体 R@20 下降 1.45 个百分点。保守结论是：这次 Llama real-only 复现没有提供 cross-fitted reweighting 在主要指标上稳定优于 Vanilla 的证据。

### 6.4 Llama 产物与 AMLT provenance

- AMLT experiment：`static-sud-crossfit-llama31-final-real-v1-0822`（只提交一次最终成功尝试，未重跑）。
- Behavior Vanilla：`clever_atemoya_7518zc2640`，result `7212558312.48642-b835bc47-80cd`，passed。
- Behavior Cross-fitted：`quiet_panda_6phb6hyrdp`，result `7212558312.48560-2ca0ea30-4954`，passed。
- Interest Vanilla：`polite_sugar_nzngkc97z1`，result `7212558312.48482-c45a4b5d-6ad7`，passed。
- Interest Cross-fitted：`gentle_beach_9npcgxqx18`，result `7212558312.48424-738fe1ac-9dcf`，passed。
- 下载根目录：`artifacts/static-sud-crossfit-reweighting/llama31-v1/final/static-sud-crossfit-llama31-final-real-v1-0822/`。
- 下载与训练语义核验：`artifacts/static-sud-crossfit-reweighting/llama31-v1/verification/download_and_training_verification.json`。
- 独立 real-only 审计：`artifacts/static-sud-crossfit-reweighting/llama31-v1/verification/real_results_audit.json`。
- 2,000-replicate bootstrap 与分析：`artifacts/static-sud-crossfit-reweighting/llama31-v1/analysis/`。

## 7. Behavior+Interest → Semantic-ID：类内与全局 Cross-Fitted 权重对照（2026-08-29）

### 7.1 研究问题（Research Questions）

本实验只改变 Cross-Fitted 权重的 rank/归一化范围，比较以下两个条件：

- **Within-class Weighted**：在每个 target class 的 200 条 OOF 样本内排序并归一化。
- **Fold-global Weighted**：在每个 held-out fold 的 4,320 条样本上跨 target class 排序，最后只做全局均值归一化。

两个条件均使用 `Qwen/Qwen3.5-9B`、`Behavior+Interest → Semantic-ID`、同一批五折 OOF score、同一 Train/real-Dev、同一训练超参数和同一确定性 Top-20 评测。研究问题是：将 rank 范围由类内改为 fold-global 后，总体 Recall/NDCG 和 Interest-Macro Recall/NDCG 如何变化。

### 7.2 训练数据构造（Training Data Construction）

- Train：21,600 个互异用户/样本，108 个 target class，每类 200 条；每个用户只保留一个 Train target。
- 输入：按时间顺序拼接每条新闻的 topic category 与 headline，即 Behavior+Interest。
- 输出：单个已学习 Semantic-ID token（`<interest_NNN>`）。
- 五折 OOF：每折用 17,280 条训练 teacher，并对 4,320 条 held-out 样本打分；五折合并后覆盖全部 21,600 条且无重复。
- OOF score：`prior_adjusted_nll`，两个最终条件复用完全相同的 score 与 fold assignment。
- Within-class：`q=rank/199`、`raw=1/(1+q)`，每类均值归一为 1。
- Fold-global：每折 rank 恰为 `0..4319`，`q=rank/4319`、`raw=1/(1+q)`，只将 21,600 条权重的全局均值归一为 1；各类均值范围为 `0.9045828–1.2189342`，确认没有类内归一化。

| condition | weight scope | normalization | weight SHA-256 |
|---|---|---|---|
| Within-class Weighted | 每类 200 条 | per-class mean = 1 | `6588ebf20afc875a3f517cfb1336e5536e20906f6ca7acc0767dc222c5761e99` |
| Fold-global Weighted | 每折 4,320 条，跨类别 | global mean = 1 | `4d41c6daae4f03a968e5e05effa232dda593a0ba8055523df9bed78c148c33e0` |

真实训练样例 `source_sample_id=train:1447803:66`：

> **English prompt:** The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":
>
> 1. [Celebrity & Creator Culture] Kylie Jenner and her bestie get surgery together, plus more news
> 2. [Celebrity & Creator Culture] 'Yabba Dabba Doo!' Kimye and Kids Rock Flintstone Halloween Costumes
> 3. [Crime & Public Safety] Ambushed in Mexico, Mormon Children Tried to Save One Another
>
> Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.
>
> **Target response:** `<interest_085>`（`Relationships & Dating`）

中文：输入同时给出按时间排列的新闻标题及其 topic category；模型需要预测 History 中未出现的新兴趣类别，并且只输出对应 Semantic-ID。该样本的目标为“情感关系与约会”，映射到 `<interest_085>`。

### 7.3 测试设计（Testing）

- Student：`Qwen/Qwen3.5-9B`，LoRA `r=16`、`alpha=32`、dropout `0.05`。
- 训练：4×A100、3 epochs、per-device batch 1、gradient accumulation 8、effective global batch 32、seed 42，最终 optimizer step 恰为 2,025。
- real-Dev：31,392 个唯一 impression/prediction，104 个有支持兴趣；每条输出 20 个互异且位于 108-label vocabulary 内的合法排序。
- 评分：确定性计算 Recall/NDCG 与 Interest-Macro Recall/NDCG@1/@5/@10/@20；没有 LLM judge，也没有 shuffled-History 对照。
- 完整性：两个条件共享相同 Train、Dev 与 mapping SHA-256；最终 adapter、prediction、metrics、per-interest metrics、merge manifest 与 artifact manifest 均完成本地读回核验。

冻结输入 SHA-256：

- Train：`dc8f04c172bf2481f4414d2c7a0ca510ae2bc0e125c2a03d984072b72b4a8140`
- real-Dev：`c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd`
- Semantic-ID mapping：`45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a`
- fold manifest：`3d25e36b4c34ae2516ac6fccee3e2c4d6596ac610076c6ef0f8b8505cd3a4b3a`

### 7.4 实验结果（Results）

下表数值均为百分比；`Δ` 为 Fold-global − Within-class，单位为百分点（pp）。

| metric | cutoff | Within-class | Fold-global | Δ pp |
|---|---:|---:|---:|---:|
| Recall | @1 | 5.3000 | **5.8507** | **+0.5508** |
| Recall | @5 | 18.9864 | **20.5905** | **+1.6041** |
| Recall | @10 | 30.3886 | **32.0186** | **+1.6301** |
| Recall | @20 | 46.5810 | **48.3432** | **+1.7622** |
| NDCG | @1 | 6.1958 | **6.8330** | **+0.6371** |
| NDCG | @5 | 12.5843 | **13.7077** | **+1.1234** |
| NDCG | @10 | 16.3898 | **17.5169** | **+1.1271** |
| NDCG | @20 | 20.6182 | **21.7903** | **+1.1722** |
| Interest-Macro Recall | @1 | **3.4724** | 3.3412 | −0.1312 |
| Interest-Macro Recall | @5 | **12.6973** | 12.5365 | −0.1608 |
| Interest-Macro Recall | @10 | **20.8722** | 20.5591 | −0.3131 |
| Interest-Macro Recall | @20 | 33.6840 | **33.8615** | **+0.1776** |
| Interest-Macro NDCG | @1 | **3.4724** | 3.3412 | −0.1312 |
| Interest-Macro NDCG | @5 | **8.0284** | 7.9714 | −0.0570 |
| Interest-Macro NDCG | @10 | **10.6361** | 10.5378 | −0.0983 |
| Interest-Macro NDCG | @20 | 13.8469 | **13.8924** | **+0.0456** |

观察结果：Fold-global 在所有总体 Recall/NDCG cutoff 上都更高，其中 Recall@10 与 NDCG@10 分别提高 `+1.6301 pp` 和 `+1.1271 pp`；但 Interest-Macro@1/@5/@10 略低，仅在 @20 略高。因此，如果主指标是 impression-level Recall/NDCG，应优先 Fold-global；如果主指标是 Interest-Macro@10，本次点估计仍是 Within-class 略好。当前未运行 paired bootstrap，不能把这些差值表述为统计显著。

独立审计确认：

- 21,600 条权重、五折各 4,320 条、每折 rank 完整且随 `prior_adjusted_nll` 单调；全局权重均值精确为 1。
- Fold-global 训练最终 loss history 到 step 2,025；因集群系统维护从持久化的 checkpoint-1100 恢复，不存在重复提交。
- final adapter 保存/重载均为 498 个 tensor，saved/loaded state SHA-256 完全相同，四个样例 logits 与 Top-20 精确一致。
- 31,392 条 prediction 的 `record_index`、`impression_id` 与 input SHA-256 均唯一；Top-20 标签/token 合法且无重复。
- 从 prediction 独立重算全部 @1/@5/@10/@20 指标，与归档 metrics 的最大绝对误差为 `7.41e-15`；104 行 per-interest metrics 精确一致。
- 已下载的 artifact manifest 文件无哈希不匹配；merged prediction、metrics、per-interest metrics 与四个 shard prediction 的 merge SHA-256 全部通过。

### 7.5 数据与产物路径（Artifact Paths）

- Behavior+Interest Semantic-ID Train/Dev/mapping：`data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1/`
- 五折 fold manifest：`data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1/fold_manifest.parquet`
- Within-class weights：`data/training/sft_static_user_disjoint_semantic_id_class_crossfit_qwen35_behavior_interest_v1/`
- Fold-global weights：`data/training/sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_behavior_interest_v1/`
- Within-class final readback：`artifacts/static-sud-semantic-id-class-crossfit/qwen35-behavior-interest/final/static-sud-semantic-id-class-crossfit-qwen35-behavior-interest-final-real-v1-0827/`
- Fold-global final readback：`artifacts/static-sud-semantic-id-global-crossfit/qwen35-behavior-interest/final-readback/static-sud-semantic-id-global-crossfit-qwen35-behavior-interest-final-real-v1-0828/`

AMLT provenance：

- Within-class experiment：`static-sud-semantic-id-class-crossfit-qwen35-behavior-interest-final-real-v1-0827`；job `qwen35-behavior-interest-semantic-id-class-crossfit-final`；passed。
- Fold-global smoke：`static-sud-semantic-id-global-crossfit-qwen35-behavior-interest-smoke-v1-0828`；job `qwen35-behavior-interest-semantic-id-global-crossfit-smoke`；passed。
- Fold-global full：`static-sud-semantic-id-global-crossfit-qwen35-behavior-interest-final-real-v1-0828`；job `qwen35-behavior-interest-semantic-id-global-crossfit-final`（AMLT job ID `affable_squash_kd2z3t6sb1`）；passed。
- Fold-global success marker：`full_success.json`，artifact manifest SHA-256 `9fdf7136a431c74da74b4e70381a061d6f1da375dea87a1de06fb74391050cd2`。
