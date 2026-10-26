# Interest-Macro Global Coverage Scaffold 与 History 语义重排实验

## 结论

本实验以 **Interest-Macro Recall@10** 和 **Interest-Macro NDCG@10** 为主指标，
不是 User-Macro。模型先在每个有 Dev 正例支持的兴趣上计算指标，再让兴趣等权。
User-Macro 仅保留为次要诊断。

按 Interest-Macro 重新联合选择 Ridge alpha、Global pool、语义扩展数和语义权重后，
Coverage expansion 在 27,125 个 Train-unseen Dev 用户上将：

- Interest-Macro Recall@10 从 11.5232% 提高到 14.0333%，提升 2.5101 个百分点，
  95% 用户聚类 bootstrap CI [2.0725, 3.0112]；
- Interest-Macro NDCG@10 从 5.2238% 提高到 6.6546%，提升 1.4307 个百分点，
  95% CI [1.1925, 1.7330]。

两项指标也都显著超过三组 matched shuffled-History。Internal holdout 与 Dev 的
Interest-Macro 验收均通过。真实 History 因此包含可跨用户泛化的兴趣级排序信号，
而不只是全局热门兴趣或 History 长度/主题数桶的作用。

代价是次要的 User-Macro Recall@10 从 46.4364% 小幅降到 46.2653%。因此该模型
适合“让每种兴趣等权”的研究目标；若线上目标是绝不降低平均用户 Top-10 覆盖，
则应使用 Recall-safe 版本。

## 指标定义

令 $P_u$ 是用户 $u$ 的 distinct 新兴趣集合，$r_u(y)$ 是兴趣 $y$ 的预测名次，
$U_y=\{u:y\in P_u\}$，只令 Dev 中 $|U_y|>0$ 的兴趣进入宏平均：

\[
\mathrm{Recall}_y@K
=\frac{1}{|U_y|}\sum_{u\in U_y}\mathbf{1}[r_u(y)\le K],
\]

\[
\mathrm{NDCG}_y@K
=\frac{1}{|U_y|}\sum_{u\in U_y}
\frac{\mathbf{1}[r_u(y)\le K]}{\log_2(r_u(y)+1)}.
\]

每个兴趣在自己的任务中是唯一相关项，因此 IDCG 为 1。最终对有 support 的兴趣
等权平均：

\[
\mathrm{InterestMacroMetric}@K
=\frac{1}{|L_+|}\sum_{y\in L_+}\mathrm{Metric}_y@K.
\]

Dev 有 111 个 supported interests；Internal holdout 有 124 个。高频兴趣不会因为
正例多而获得更大宏平均权重。

## 模型与选参

模型沿用冻结的 `sentence-transformers/all-MiniLM-L6-v2` 标题 History embedding。
线性 Ridge 学习用户等权多正例分布相对 Global 先验的残差。Coverage scaffold：

1. Global 先验给出 Top-$M$ 锚点；
2. History 残差给出 Top-$E$ 语义挑战者；
3. 并集内用 `Global + λ × semantic residual` 排序；
4. 其他兴趣按 Global 顺序回退。

528,401 个旧 Fit 用户按固定 SHA-256 划分为 model-fit 422,984、tune 52,710、
internal-holdout 52,707。选参规则为：最大化 tune Interest-Macro NDCG@10，约束
Interest-Macro Recall@10 不低于 Global。最终参数为：

- Coverage expansion：Ridge alpha=1，$M=10$，$E=10$，$λ=48$；
- Recall-safe：Ridge alpha=1，$M=10$，$E=0$，$λ=48$。

参数冻结后，在全部 587,187 个 Train 用户上重拟合线性模型。Dev 已经在此前实验中
被查看，因此这里仍标为探索性 Dev，而不是全新 final test。

## 主结果

数值均为百分比。

| 评估集 | 模型 | Interest-Macro Recall@10 | Interest-Macro NDCG@10 | User-Macro Recall@10（次要） | User-Macro NDCG@10（次要） |
|---|---|---:|---:|---:|---:|
| Internal holdout | Global | 12.2118 | 5.4871 | 43.1173 | 26.4391 |
| Internal holdout | Recall-safe | 12.2118 | 6.2478 | 43.1173 | 29.0874 |
| Internal holdout | Coverage expansion | **15.4448** | **7.4472** | **46.6947** | **30.7424** |
| Dev（探索性） | Global | 11.5232 | 5.2238 | **46.4364** | 20.9370 |
| Dev（探索性） | Recall-safe | 11.5232 | 5.6931 | **46.4364** | **25.9703** |
| Dev（探索性） | Coverage expansion | **14.0333** | **6.6546** | 46.2653 | 25.6602 |

Coverage expansion 相对 Global 的 Interest-Macro paired bootstrap：

| 评估集 | Recall@10 delta [95% CI] | NDCG@10 delta [95% CI] |
|---|---:|---:|
| Internal holdout | +3.2330 [3.1520, 3.3506] | +1.9600 [1.9248, 2.0180] |
| Dev（探索性） | +2.5101 [2.0725, 3.0112] | +1.4307 [1.1925, 1.7330] |

Bootstrap 以用户为抽样簇：每次有放回抽取用户，然后重新计算每个兴趣的 support、
Recall 和 NDCG，最后才对该次抽样中有 support 的兴趣宏平均。它不是对 per-user
指标取平均。

## Shuffled-History 对照

三组 matched shuffle 只替换 History embedding donor，保留用户、target 和真实
History candidate mask。239,496 条映射没有 fixed point；其中 239,490 条严格匹配
History 长度桶与主题数桶，6 条 singleton 使用确定性 relaxed derangement。

Dev Coverage expansion 的真实 History 相对三组 shuffled History：

- Interest-Macro Recall@10 高 2.4961 至 2.6544 个百分点；三组 CI 下界为
  1.8789、1.9613、2.0159；
- Interest-Macro NDCG@10 高 1.1608 至 1.4114 个百分点；三组 CI 下界为
  0.8633、1.0616、1.0365。

所有下界均大于 0。

## 逐兴趣诊断

Dev 的 111 个 supported interests 中，Coverage expansion 相对 Global：

- Recall@10：31 个提高、67 个不变、13 个下降；
- NDCG@10：37 个提高、67 个不变、7 个下降；
- support 中位数为 103，范围 1–5,094；support≤5 的 13 个兴趣对宏平均提升均为 0，
  因此总体提升不是由 singletons 或极低 support 兴趣制造；
- 提升并不均匀，逐兴趣明细必须与宏平均一起查看。

每个 condition 的 `per_interest_metrics.parquet` 保存兴趣名、support 以及 @1/5/10/20
的 Recall 和 NDCG。

## 验收与研究含义

- Coverage expansion：Internal holdout PASS，探索性 Dev PASS；两项 Interest-Macro
  主指标均超过 Global 和所有 shuffled-History，且用户聚类 CI 下界大于 0。
- Recall-safe：Interest-Macro Recall@10 与 Global 精确不变，Dev
  Interest-Macro NDCG@10 提升 0.4692 个百分点，CI [0.3198, 0.6489]。
- 以前按 User-Macro 选参所得“Expansion 的 Dev Recall 没有提升”不能回答本次问题；
  在 Interest-Macro 下，Expansion 显著提升，但牺牲了 0.1711 个百分点的
  User-Macro Recall@10。
- 便宜的个性化模型已跨过“同时超过 Global 与 shuffled History”的兴趣级信号门槛。
  下一阶段可以做受控的语义/LLM 排序试验，但最终结论仍需要一个未被查看的新时间切片
  或用户隔离测试集。

## 产物与验证

- 实验入口：`src/new_interest/experiments/semantic_coverage_scaffold.py`
- 配置：`configs/future_interest/static_user_disjoint_semantic_coverage_interest_macro_v1.json`
- 完整产物：`artifacts/future-novel-interest/static-user-disjoint-semantic-coverage-interest-macro-v1/`
- 选参：`tuning.json`
- paired bootstrap：`comparisons.json`
- 验收：`acceptance.json`
- 每兴趣明细：`conditions/<split>/<condition>/per_interest_metrics.parquet`
- 哈希清单：`manifest.json`

独立核验 92 个 source、implementation 和 artifact SHA-256，全部匹配；18 个 condition
的逐兴趣 Parquet 均重新聚合并与 JSON 指标一致；Recall-safe Top-10 集合在两个评估集
均为 0 个不一致；无残留 `.old` 目录。测试结果：`pytest -q`，244 passed。
