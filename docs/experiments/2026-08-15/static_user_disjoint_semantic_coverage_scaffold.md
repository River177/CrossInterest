# Global Coverage Scaffold 与 History 语义重排实验

> **已被新口径取代：** 本报告按 User-Macro 选参与验收；用户随后明确要求最终主指标
> 为 Interest-Macro。请以
> [Interest-Macro 重跑报告](static_user_disjoint_semantic_coverage_interest_macro.md)
> 为准。本报告仅保留为旧口径对照。

## 结论

最终主指标应当是 **User-Macro Recall@K** 和 **User-Macro NDCG@K**：先对每个用户的全部 distinct 新兴趣计算指标，再让每个用户等权。现有代码中的裸字段 `recall_at_k` / `ndcg_at_k` 本来就是这个口径；本实验新增明确字段 `user_macro_recall_at_k` / `user_macro_ndcg_at_k`。旧字段 `macro_recall_at_k` 实际是兴趣类别等权的 Label-Macro Recall，本实验将其明确别名为 `label_macro_recall_at_k`，只作为长尾诊断。

实验支持一个清楚但有限的故事：Global 提供 Top-10 覆盖骨架，真实 History 语义负责把骨架中的正确兴趣提前。Recall-safe 版本逐用户保持 Global Top-10 集合不变，因此 Dev User-Macro Recall@10 精确保持 46.4364%；同时把 User-Macro NDCG@10 从 20.9370% 提高到 25.9674%（+5.0303 个百分点，95% paired-bootstrap CI [4.8736, 5.2016]）。这说明 History 能稳定改善“命中的兴趣排在哪里”，且不是 History 长度或主题数造成的伪信号。

允许 10 个语义挑战者进入候选池后，Train-internal holdout 的 Recall@10 / NDCG@10 都显著提升，但 Dev Recall@10 只有 +0.0273 个百分点，CI [−0.3809, 0.4260]，不显著；NDCG@10 仍显著提升 +4.7938，[4.5712, 5.0003]。因此扩展版没有通过 Dev 的“双指标均提升”门槛。当前最稳妥的模型是 Recall-safe 重排器，不是开放式语义扩展器。

## 指标定义

设用户 $u$ 的 distinct 新兴趣集合为 $P_u$，模型前 $K$ 个预测为 $R_{u,K}$：

\[
\mathrm{UserMacroRecall@K}
=\frac{1}{|U|}\sum_{u\in U}
\frac{|P_u\cap R_{u,K}|}{|P_u|}.
\]

每个用户的二元相关性 DCG 以该用户的 $\min(K, |P_u|)$ 个理想命中计算 IDCG：

\[
\mathrm{UserMacroNDCG@K}
=\frac{1}{|U|}\sum_{u\in U}
\frac{\mathrm{DCG}_u@K}{\mathrm{IDCG}_u@K}.
\]

两者都以用户为统计单位。一个拥有 10 个 target 的用户与一个只有 1 个 target 的用户权重相同。Label-Macro Recall 则先按兴趣类别计算 Recall，再对有支持的类别等权；它回答类别公平/长尾问题，不能替代用户级主指标。NDCG 没有在本实验中构造“按类别 Macro”的版本。

## 方法

语义模型沿用冻结的 `sentence-transformers/all-MiniLM-L6-v2` 标题 History embedding。线性 Ridge 从 History embedding 预测用户等权多正例分布相对 Global 先验的残差。对每个用户：

1. 用 Global 先验产生 Top-$M$ 锚点；
2. 用 History 残差产生 Top-$E$ 语义挑战者；
3. 在两者并集内按 `Global + λ × semantic residual` 重排；
4. 未进入并集的兴趣按 Global 顺序回退。

两个冻结版本为：

- Recall-safe：Ridge alpha=10，$M=10, E=0, λ=40$。它只重排 Global Top-10，逐用户保证 Recall@10 不变。
- Coverage expansion：Ridge alpha=10，$M=10, E=10, λ=48$。它允许语义兴趣替换 Global Top-10 中的项目，可能提高或降低 Recall。

旧 Fit 的 528,401 个用户用固定 SHA-256 划为 model-fit 422,984、scaffold-tune 52,710、internal-holdout 52,707。Ridge alpha、$M$、$E$、$λ$ 只在 tune 上联合选择。冻结超参数后，用全部 587,187 个 Train 用户重拟合线性模型，再在 27,125 个 Train-unseen Dev 用户上作探索性评估。Dev 已在前一轮被查看，因此不能再作为全新的最终测试集。

## 主结果

单位均为百分比。

| 评估集 | 模型 | User-Macro Recall@10 | User-Macro NDCG@10 | Label-Macro Recall@10 |
|---|---|---:|---:|---:|
| Internal holdout | Global | 43.1173 | 26.4391 | 12.2118 |
| Internal holdout | Recall-safe reranker | 43.1173 | 29.0832 | 12.2118 |
| Internal holdout | Coverage expansion | **46.8291** | **30.7751** | **15.4296** |
| Dev（探索性） | Global | 46.4364 | 20.9370 | 11.5232 |
| Dev（探索性） | Recall-safe reranker | 46.4364 | **25.9674** | 11.5232 |
| Dev（探索性） | Coverage expansion | **46.4636** | 25.7308 | **14.0417** |

相对 Global 的 paired user bootstrap：

| 评估集 | 模型 | Recall@10 delta [95% CI] | NDCG@10 delta [95% CI] |
|---|---|---:|---:|
| Internal holdout | Recall-safe | 0.0000 [0.0000, 0.0000] | +2.6442 [2.5230, 2.7649] |
| Internal holdout | Expansion | +3.7117 [3.4569, 3.9681] | +4.3360 [4.1676, 4.5049] |
| Dev（探索性） | Recall-safe | 0.0000 [0.0000, 0.0000] | +5.0303 [4.8736, 5.2016] |
| Dev（探索性） | Expansion | +0.0273 [−0.3809, 0.4260] | +4.7938 [4.5712, 5.0003] |

## Shuffled-History 对照

matched shuffle 只替换 History embedding donor；用户、target 和真实 History candidate mask 不变。优先严格匹配 History 长度桶与 distinct topic 数桶。Internal holdout 有两个 exact cell 各只有一个用户，三个 seed 中共 6 条映射采用确定性的 singleton relaxed derangement；全部 239,496 条映射 fixed point 为 0。

在 Dev 上，Coverage expansion 的真实 History 相对三组 shuffled History：

- User-Macro Recall@10 高 +6.07 至 +6.32 个百分点；
- User-Macro NDCG@10 高 +3.59 至 +3.82 个百分点；
- 所有 paired-bootstrap 95% CI 下界均大于 0。

Recall-safe 的 Top-10 集合对 actual/shuffled 都固定为 Global，因此 Recall 必然相同；真实 History 的 NDCG@10 比三组 shuffled 高 +1.38 至 +1.55 个百分点，所有 CI 均排除 0。这是最直接的“对齐 History 改善排序，而不是统计桶本身起作用”的证据。

## 为什么 Internal holdout 成功而 Dev Recall 没有显著提升

两个集合的用户活跃度与 target 数分布明显不同。Internal holdout 只有 21,337 / 52,707（40.5%）用户是单 target；Dev 有 20,451 / 27,125（75.4%）是单 target。Coverage expansion 在 Dev 单 target 用户的 Recall@10 从 48.6724% 降至 48.4231%（−0.2493 个百分点），但在 2 target、3–5 target、6+ target 用户分别提升约 +0.856、+0.959、+0.338 个百分点，最终互相抵消为总体 +0.0273。

这说明当前 Train-internal target 结构不能直接代表 Train-unseen Dev 用户。模型学到的 History 排序信号真实存在，但“应该让多少语义挑战者替换热门候选”的校准没有跨分布泛化。

## 验收与下一步

- Coverage expansion：Internal holdout PASS；Dev 双指标门槛 FAIL，因为 Recall@10 CI 包含 0。
- Recall-safe：Internal holdout 与 Dev 都满足逐用户 Recall@10 精确不变，同时 NDCG@10 显著超过 Global 和全部 shuffled History。
- Tail 50 个标签在 Dev Top-10 仍为 0 Recall；本实验没有解决极端长尾。
- 若业务目标是“绝不牺牲 Top-10 覆盖，同时改善排序”，应采用 Recall-safe 版本。
- 若目标是让 Recall@10 也超过 Global，下一步应学习受约束的 challenger admission/calibration，并对单 target 用户与多 target 用户分别校准；需要新的未见时间切片或用户 holdout 作最终检验。
- 按原先“Recall@10 和 NDCG@10 都显著超过 Global 与 shuffled”才启动昂贵 LLM 训练的门槛，目前仍不应启动 Qwen/LoRA 全量训练。

## 产物与验证

- 实验入口：`src/new_interest/experiments/semantic_coverage_scaffold.py`
- 配置：`configs/future_interest/static_user_disjoint_semantic_coverage_v1.json`
- 完整产物：`artifacts/future-novel-interest/static-user-disjoint-semantic-coverage-v1/`
- 指标与逐用户结果：`conditions/{internal_holdout,dev_exploratory}/<condition>/`
- 选参：`tuning.json`
- paired bootstrap：`comparisons.json`
- 验收：`acceptance.json`
- 内部分割清单：`internal_split_manifest.parquet`
- shuffle donor 清单：`shuffle_manifest.parquet`
- 全部 SHA-256：`manifest.json`

独立核验了 74 个 source、implementation 和 artifact 哈希，均匹配；Recall-safe Top-10 集合在两个评估集均为 0 个不一致；无残留 `.old` 目录。最终完整测试：`pytest -q`，243 passed。
