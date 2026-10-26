# 冻结 MiniLM 标题表示的 NRMS 新兴趣预测

## 结论

在完整原始 History 上训练 NRMS 后，最终选择：

- `hidden_dimension = 128`
- `attention_heads = 4`
- `dropout = 0.1`
- `class_balance_beta = 0`
- `learning_rate = 1e-3`
- `weight_decay = 0`
- `epoch = 4`

NRMS 在探索性 Dev 上得到 **Interest-Macro Recall@10 15.9866%**、**Interest-Macro NDCG@10 7.8284%**。它高于 Global Popularity 的 11.5264% / 5.2247%，说明标题级 History 确实含有可学习的新兴趣信号；但明显低于 Balanced Ridge 的 19.8275% / 9.4655% 和 Linear Ranker 的 22.1331% / 11.7596%。

因此，当前实验不支持“NRMS 比简单语义模型更好”。在单标签且长尾的新兴趣监督下，冻结 MiniLM 后再学习标题级 self-attention，没有胜过完整 History mean embedding 上的线性排序器。

## 模型

每条 History 标题先使用固定 revision 的 `sentence-transformers/all-MiniLM-L6-v2` 编码成384维向量。NRMS 对完整标题序列执行：

1. 可学习的线性投影和 GELU；
2. multi-head self-attention；
3. residual connection 与 layer normalization；
4. additive attention pooling，得到一个用户向量；
5. 无 bias 的124类线性分类器，输出完整兴趣词表分数。

标题 encoder 完全冻结，缓存包含74,553个唯一标题，revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。模型没有显式 Global Popularity 特征。训练与推理均屏蔽用户原始 History 已出现的兴趣；History 不追加 impression、不截断，Train 最长序列为800。

## 数据与选择协议

- Static Train：587,187用户，每用户一个固定选择的新兴趣标签。
- 内部分割：469,643 model-fit、58,518 tune、59,026 frozen holdout。
- Dev：27,125个 Train-unseen 用户、37,055个 distinct 用户–新兴趣正例、111个有支持兴趣。
- 兴趣空间：Static Train 实际支持的124类。
- 主指标：Interest-Macro Recall@10 与 Interest-Macro NDCG@10。
- Dev 只在 Train-only 选参和全 Train 重训结束后评估一次。

架构阶段搜索6种 `hidden × heads × dropout` 组合，在 epoch 2/4 共评估12个 Tune checkpoint。优化阶段固定最佳架构，搜索：

- `beta ∈ {0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1}`；
- `learning_rate ∈ {0.001, 0.003}`；
- `weight_decay ∈ {0, 1e-5}`；
- epoch `{2, 4, 6, 8}`。

优化阶段共36个训练条件、144个 Tune checkpoint。为避免只靠过度类别平衡提高 Interest-Macro，合格条件必须满足 Tune User-Macro Recall@10 不低于 Linear Ranker 的 **35.5925%**。再从合格 Pareto 点中保留距最佳 Interest-Macro Recall 不超过0.2个百分点的点，并最大化 NDCG。

最终 Tune 点为 Interest-Macro Recall@10 **15.4473%**、NDCG@10 **7.3768%**、User-Macro Recall@10 **35.9069%**。`weight_decay=0` 与 `1e-5` 的该点数值相同，以较小 weight decay 打破并列。

## 冻结结果

### Interest-Macro 主指标

| 评估集 | 方法 | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| Internal holdout | Global Popularity | 11.7883% | 5.3119% |
| Internal holdout | Balanced Ridge | 24.9566% | 12.8303% |
| Internal holdout | Linear Ranker | **26.4864%** | **14.1742%** |
| Internal holdout | NRMS | 15.4077% | 7.4531% |
| Dev（探索性） | Global Popularity | 11.5264% | 5.2247% |
| Dev（探索性） | Balanced Ridge | 19.8275% | 9.4655% |
| Dev（探索性） | Linear Ranker | **22.1331%** | **11.7596%** |
| Dev（探索性） | NRMS | 15.9866% | 7.8284% |

Dev 上 NRMS 相对 Global 的 Recall/NDCG 提高 **4.4602 / 2.6037 pp**，但相对 Linear Ranker 低 **6.1464 / 3.9312 pp**。Internal holdout 上也呈现相同结论，因此不是只在反复查看过的 Dev 上发生的排序。

### User-Macro 诊断

| 评估集 | 方法 | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| Internal holdout | Global Popularity | **43.9417%** | **21.9923%** |
| Internal holdout | Balanced Ridge | 34.4628% | 19.0707% |
| Internal holdout | Linear Ranker | 35.5064% | 18.9352% |
| Internal holdout | NRMS | 36.0011% | 17.9925% |
| Dev（探索性） | Global Popularity | **46.3481%** | **20.9077%** |
| Dev（探索性） | Balanced Ridge | 33.4923% | 19.1395% |
| Dev（探索性） | Linear Ranker | 30.6968% | 17.1993% |
| Dev（探索性） | NRMS | 27.1401% | 13.6681% |

NRMS 在 holdout 上满足预设 User-Recall floor，但在 Dev 上该指标明显下降。选参门槛只约束 Train internal tune，不能保证跨 split 的 User-Macro 泛化。

## 解释与边界

- NRMS 高于 Global，支持“History 有预测信号”，但不能据此证明标题级 attention 是最合适的提取方式。
- 当前 Train 每用户只有一个观测新兴趣，类别分布长尾；提高 `beta` 虽能改善部分 Interest-Macro 点，却会大幅降低 User-Macro，最后只有 `beta=0` 的较早 epoch 进入合格区。
- Linear Ranker 使用预先聚合的完整 History embedding，参数更少、优化目标直接对应全词表排序；当前结果表明这种强归纳偏置更适合现有监督规模与噪声。
- Dev 已用于此前分析，只能称为 exploratory Dev，不能作为全新 final test。
- 本实验在固定124类词表上输出 Top-10，不是任意自然语言开放生成。
- 当前为点估计，尚未补用户聚类 bootstrap 置信区间。

## 产物与验证

- 标题缓存：`artifacts/future-novel-interest/static-user-disjoint-nrms-title-embeddings-v1/`
- NRMS 产物：`artifacts/future-novel-interest/static-user-disjoint-nrms-v1/`
- 完整搜索：`tuning.json`
- 最终模型：`selected_model.npz`
- 汇总：`summary.json`
- 逐兴趣指标和 Top-10：`conditions/<split>/<condition>/`
- 输入输出哈希：`manifest.json`
- 实现：`src/new_interest/experiments/tune_semantic_history_nrms.py`
- 配置：`configs/future_interest/static_user_disjoint_nrms_v1.json`

独立下载后验证了17个输出 SHA-256 和12个输入 SHA-256；holdout 59,026用户、Dev 27,125用户/37,055个 distinct 正例均与冻结协议一致；Top-10 无重复，预测与原始 History 兴趣交集为0。远端作业在全部结果写出后因最终 `cp` 尝试保留挂载权限而被 AMLT 标记 failed，训练、选参、holdout 和 Dev 输出本身均已完整下载并通过哈希验证，因此没有重跑 GPU 作业。
