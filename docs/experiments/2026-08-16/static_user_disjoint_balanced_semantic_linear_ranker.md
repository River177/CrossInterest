# 类别平衡 Semantic History Linear Ranker

## 结论

固定 `all-MiniLM-L6-v2` History embedding 后，以全兴趣词表 softmax 排序损失替代逐类别 Ridge 回归，Train-only Pareto 搜索选择：

- `class_balance_beta = 0.90`
- `l2 = 1e-5`
- `temperature = 1.0`
- `learning_rate = 0.01`
- `epochs = 12`

在完全未参与选参的 Static Train internal holdout 上，Linear Ranker 相对 Balanced Ridge 将 Interest-Macro Recall@10 从 **24.9566%** 提高到 **26.4864%**，Interest-Macro NDCG@10 从 **12.8304%** 提高到 **14.1742%**。

在探索性 Dev 上，对应指标从 **19.8286% / 9.4658%** 提高到 **22.1331% / 11.7596%**。这说明直接优化全词表相对排序，比对每个类别独立做平方误差回归更符合开放式 Top-10 任务。

## 模型与损失

每个用户使用冻结的 384 维完整 History 标题 embedding `x_u`，每个兴趣有一列可学习权重 `W_y`：

\[
s(u,y)=\frac{x_u^\top W_y}{T}.
\]

模型没有 bias，也没有加入 Global Popularity。训练时把原始 History 中出现的兴趣从 softmax 分母删除：

\[
\mathcal L_u=-w_{y_u}\log
\frac{\exp s(u,y_u)}
{\sum_{y\notin H_u}\exp s(u,y)}
+\frac{\lambda}{2}\lVert W\rVert_F^2.
\]

类别权重为 `w_y ∝ n_y^{-β}`，并缩放到观测样本平均权重为1。`β=0` 不平衡，`β=1` 使各类别总训练权重相同；最终 `β=0.9` 接近类别等权，但仍为高频类别保留少量支持度优势。

## 数据与选择协议

- Static Train：587,187用户，每用户一个 SHA-256 选定的新兴趣标签。
- 内部分割：469,643 model-fit、58,518 tune、59,026 holdout。
- Dev：27,125个 Train-unseen 用户，37,055个 distinct 用户–新兴趣正例，111个有正例支持的兴趣。
- 兴趣空间：Static Train 实际支持的124类；原始词表另有1个 Train 无支持类别，不进入候选空间。
- History：完整原始 Behavior History，不追加 impression，不截断。
- 预测：删除用户 History 已出现类别后，对124类完整打分并输出Top-10。

最终网格包含：

- `β ∈ {0, 0.25, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95, 1}`；
- `L2 ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}`；
- `temperature ∈ {0.5, 1, 2}`；
- `learning rate ∈ {0.003, 0.01, 0.03}`；
- 在 epoch `{1, 2, 4, 6, 8, 10, 12}` 评估。

共405个训练条件、2,835个 Tune checkpoint。选择时先要求 Tune User-Macro Recall@10 不低于 Balanced Ridge，再在 Interest-Macro Recall/NDCG Pareto 前沿中要求 Recall 不低于 Global 且距合格点最佳 Recall 不超过0.2个百分点，最后最大化 Interest-Macro NDCG@10。

最终 Tune 点为 Recall@10 **27.1146%**、NDCG@10 **14.4379%**、User-Macro Recall@10 **35.5925%**。合格点最佳 Recall 为27.2054%，最终点以0.0908个百分点 Recall 换取更高 NDCG。

## 冻结结果

### Interest-Macro 主指标

| 评估集 | 方法 | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| Internal holdout | Global Popularity | 11.7883% | 5.3119% |
| Internal holdout | Current Ridge `(α=10, β=0)` | 20.6288% | 9.8201% |
| Internal holdout | Balanced Ridge `(α=10, β=0.5)` | 24.9566% | 12.8304% |
| Internal holdout | **Linear Ranker** | **26.4864%** | **14.1742%** |
| Dev（探索性） | Global Popularity | 11.5264% | 5.2247% |
| Dev（探索性） | Current Ridge `(α=10, β=0)` | 17.0544% | 8.0450% |
| Dev（探索性） | Balanced Ridge `(α=10, β=0.5)` | 19.8286% | 9.4658% |
| Dev（探索性） | **Linear Ranker** | **22.1331%** | **11.7596%** |

相对 Balanced Ridge：

- Internal holdout：Recall +1.5298 pp，NDCG +1.3438 pp；
- exploratory Dev：Recall +2.3045 pp，NDCG +2.2938 pp。

### User-Macro 诊断

| 评估集 | 方法 | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| Internal holdout | Current Ridge | 41.2954% | 22.8124% |
| Internal holdout | Balanced Ridge | 34.4628% | 19.0709% |
| Internal holdout | Linear Ranker | **35.5064%** | 18.9352% |
| Dev（探索性） | Current Ridge | **38.9352%** | **22.4025%** |
| Dev（探索性） | Balanced Ridge | 33.4960% | 19.1406% |
| Dev（探索性） | Linear Ranker | 30.6968% | 17.1993% |

Linear Ranker 针对 Interest-Macro 目标更好，但不是平均用户指标最优模型。尤其在 Dev 上，Interest-Macro 提升伴随 User-Macro 下降，说明模型把更多Top-10容量分配给低频兴趣；不能把结果描述为所有用户都获得了更好推荐。

## 边界与产物

- Dev 已用于前序探索，只能称为 exploratory Dev，不能冒充全新 final test。
- Train 每用户只有一个观测新兴趣标签，而 Dev 使用每用户全部 distinct 正例；观测标签仍是不完整的未来兴趣样本。
- 当前报告是点估计，尚未补用户聚类 bootstrap 置信区间。
- 本实验是固定124类兴趣词表上的开放式Top-10，不是任意自然语言生成。

实现与产物：

- 入口：`src/new_interest/experiments/tune_semantic_history_linear_ranker.py`
- 配置：`configs/future_interest/static_user_disjoint_balanced_semantic_linear_ranker_v1.json`
- 产物：`artifacts/future-novel-interest/static-user-disjoint-balanced-semantic-linear-ranker-v1/`
- 完整网格：`tuning.json`
- 最终模型：`selected_model.npz`
- 汇总：`summary.json`
- 逐兴趣指标和Top-10：`conditions/<split>/<condition>/`
- 输入输出哈希：`manifest.json`
