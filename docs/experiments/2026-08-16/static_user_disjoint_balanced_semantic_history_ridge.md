# 类别平衡 Semantic History Ridge 的开放式 Top-10 Pareto 调参

## 结论

固定现有 `all-MiniLM-L6-v2` History embedding，只搜索 Ridge penalty `α` 与类别平衡强度 `β` 后，Train-only Pareto 规则选择：

- `ridge_alpha = 10`
- `class_balance_beta = 0.5`

在探索性 Dev 上，Interest-Macro Recall@10 从当前未平衡 Ridge 的 **17.0544%** 提高到 **19.8275%**，Interest-Macro NDCG@10 从 **8.0450%** 提高到 **9.4655%**。对应提升为 +2.7731 和 +1.4205 个百分点。

同方向提升也出现在完全未参与选参的 Static Train internal holdout：Recall@10 从 20.6288% 提高到 24.9566%，NDCG@10 从 9.8201% 提高到 12.8303%。因此提升不是只存在于已反复查看的 Dev。

代价是 User-Macro 指标下降。该参数是为“每个兴趣类别等权”目标选择的，不是平均用户指标最优点。

## 协议

- 监督数据：`new_interest_static_user_disjoint_v1/train.parquet`，587,187 个用户，每用户一个 SHA-256 选定的新兴趣标签。
- History：完整原始 Behavior History，不追加 impression，不截断。
- Embedding：冻结的 `sentence-transformers/all-MiniLM-L6-v2` 384 维标题 mean-pooling embedding；本实验没有重新编码。
- 兴趣空间：Static Train 支持的124个兴趣，预测时删除该用户 History 已出现的兴趣。
- 内部分割：469,643 model-fit、58,518 tune、59,026 selection-excluded holdout。
- Dev：27,125个 Train-unseen 用户、37,055个 distinct 用户–新兴趣正例、111个有正例支持的兴趣。
- 每个用户输出 Top-10；没有使用 Global Popularity 融合项。

每个目标兴趣为 `y` 的 Train 用户获得样本权重：

\[
w_y=n_y^{-\beta}.
\]

所有权重再整体缩放，使观测样本的平均权重为1，避免不同 `β` 改变 Ridge penalty 的量纲。搜索网格为：

- `α ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000}`；
- `β ∈ {0, 0.25, 0.5, 0.75, 1}`；
- 共55个配置。

`β=0` 是未平衡 Ridge；`β=1` 使每个类别在拟合中的总权重相同；最终 `β=0.5` 使单样本权重与 `1/√n_y` 成正比，类别总权重与 `√n_y` 成正比。

## Pareto 选择规则

只在 Train tune 上执行：

1. 删除同时不优于其他配置的 dominated 点；
2. Recall@10 必须不低于 Global Popularity；
3. Recall@10 必须位于全网格最佳值的0.2个百分点以内；
4. 在合格 Pareto 点中最大化 Interest-Macro NDCG@10；
5. 依次以更高 Recall、更小 `β`、更小 `α` 打破完全并列。

Tune Pareto 前沿如下：

| α | β | Interest-Macro Recall@10 | Interest-Macro NDCG@10 | 状态 |
|---:|---:|---:|---:|---|
| 1000 | 0.75 | 25.6555% | **13.7542%** | NDCG最高，但Recall低于允许下限 |
| **10** | **0.50** | 26.0071% | **13.3673%** | **最终选择** |
| 30 | 0.50 | **26.0310%** | 13.3413% | Recall最高 |

Tune Global Popularity 为 Recall@10 11.9697%、NDCG@10 5.3896%。

## 冻结后结果

### Interest-Macro 主指标

| 评估集 | 方法 | Interest-Macro Recall@10 | Interest-Macro NDCG@10 |
|---|---|---:|---:|
| Internal holdout | Global Popularity | 11.7883% | 5.3119% |
| Internal holdout | 当前 Ridge `(α=10, β=0)` | 20.6288% | 9.8201% |
| Internal holdout | **Balanced Ridge `(α=10, β=0.5)`** | **24.9566%** | **12.8303%** |
| Dev（探索性） | Global Popularity | 11.5264% | 5.2247% |
| Dev（探索性） | 当前 Ridge `(α=10, β=0)` | 17.0544% | 8.0450% |
| Dev（探索性） | **Balanced Ridge `(α=10, β=0.5)`** | **19.8275%** | **9.4655%** |

### User-Macro 次要诊断

| 评估集 | 方法 | User-Macro Recall@10 | User-Macro NDCG@10 |
|---|---|---:|---:|
| Internal holdout | 当前 Ridge | **41.2954%** | **22.8124%** |
| Internal holdout | Balanced Ridge | 34.4628% | 19.0707% |
| Dev（探索性） | 当前 Ridge | **38.9352%** | **22.4025%** |
| Dev（探索性） | Balanced Ridge | 33.4923% | 19.1395% |

这说明类别平衡确实把模型从高频兴趣和高支持类别移向更均衡的兴趣覆盖，但会牺牲平均用户命中率。部署时不能把 Balanced Ridge 称为“所有指标更好”；它只是在当前主目标 Interest-Macro Recall/NDCG 上更好。

Dev 111个有支持兴趣中，相对当前 Ridge：

- Recall@10：51个提高、18个不变、42个下降；
- NDCG@10：60个提高、12个不变、39个下降。

## 边界

- Dev 已用于前序分析，本结果只能称为探索性 Dev，不能作为全新 final-test 结果。
- Internal tune/holdout 使用每个 Static Train 用户唯一选定的标签；Dev 评估使用每个用户的全部 distinct 新兴趣。这一监督–评估差异被保留，以与当前 Candidate Ridge 完全一致。
- 本实验是固定兴趣词表上的开放式 Top-10 推荐，不是生成任意自然语言类别。
- 当前结果是点估计，尚未对选定模型运行用户聚类 bootstrap 置信区间。

## 产物与验证

- 实验入口：`src/new_interest/experiments/tune_semantic_history_ridge.py`
- 配置：`configs/future_interest/static_user_disjoint_balanced_semantic_history_ridge_v1.json`
- 完整产物：`artifacts/future-novel-interest/static-user-disjoint-balanced-semantic-history-ridge-v1/`
- 全部55个配置与 Pareto 前沿：`tuning.json`
- 最终模型：`selected_model.npz`
- 汇总：`summary.json`
- 逐兴趣指标与 Top-10：`conditions/<split>/<condition>/`
- 输入与输出 SHA-256：`manifest.json`

验证结果：7个输入与22个产物 SHA-256 全部匹配；6个输出条件行数正确、Top-10 无重复；Dev Selected Top-10 与原始 History 兴趣交集为0；完整测试 `pytest -q` 为 **265 passed**。
