# 静态用户隔离数据上的标题语义 History 信号实验

## 结论

真实 History 中存在稳定、可复现的用户条件语义信号，但当前模型还没有通过预注册的强个性化门槛。

最佳模型是用户等权多正例训练的线性 Ridge 语义残差。与 Global 相比，它把 Dev NDCG@10 从 20.94% 提升到 25.42%（+4.48 个百分点，95% paired-bootstrap CI [4.32, 4.65]），但 Recall@10 从 46.44% 降到 45.98%（-0.46，[−0.71, −0.21]）。因此，History 明显改善了正确兴趣在列表中的前后顺序，却尚未同时提高 Top-10 覆盖率。

真实 History 对三组 matched-shuffle History 均显著占优。Ridge 的 Recall@10 增益为 +2.70 至 +2.75 个百分点，NDCG@10 增益为 +2.09 至 +2.21，所有 95% CI 均排除 0。这排除了“模型只是在利用 History 长度、主题数或全局流行度”的解释。

## 数据与实验协议

- 基础数据：`new_interest_static_user_disjoint_v1`。
- Train 恢复全部 distinct 用户–新兴趣正例，再按用户等权；不是把 1,401,004 个正例当成等权独立样本。
- 587,187 个 Train 用户按固定 SHA-256 拆为 Fit 528,401、Validation 58,786。
- Dev 将 37,817 行合并成 27,125 个用户和 37,055 个 distinct 正例；与 Train 用户完全隔离。
- History 只使用原始 behavior History，不追加 impression 点击、不截断。
- 候选空间固定为 125 个兴趣减去该用户真实 History 已出现的兴趣。
- 所有超参数只在 Train-Validation 上选择；Dev 不参与选参。
- 标题编码器固定为 `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，全部标题 mean pooling，无截断。
- matched-shuffle 只替换用户 embedding；target、真实 History candidate mask 和用户样本完全不变。History 长度桶和 distinct History topic 数桶同时匹配，三个 seed 均无 fixed point。
- paired bootstrap 以用户为重采样单位，2,000 replicates。

评分统一写成：

\[
s(u,y)=\log P_{\mathrm{fit}}(y)+\lambda r(H_u,y).
\]

Zero-shot residual 是 History embedding 与兴趣名称 embedding 的 cosine；Validation 选择 `λ=4`。Linear Ridge residual 从用户 embedding 预测其等权多正例分布相对 Global 的残差；Validation 选择 `ridge_alpha=1, λ=16`。

## Dev 主结果

单位均为百分比。Shuffled 为三个 seed 的均值。

| 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| Global | 26.51 | 14.12 | **46.44** | 20.94 | 60.93 | 24.86 |
| Zero-shot，真实 History | 28.73 | 17.47 | 45.00 | 23.06 | 64.19 | 28.23 |
| Zero-shot，Shuffled | 26.27 | 16.74 | 42.60 | 22.35 | 61.58 | 27.46 |
| Linear Ridge，真实 History | **29.53** | **19.82** | 45.98 | **25.42** | **64.66** | **30.47** |
| Linear Ridge，Shuffled | 26.84 | 17.69 | 43.26 | 23.29 | 62.38 | 28.45 |

Linear Ridge 相对 Global 的 paired delta：

| 指标 | Delta | 95% CI |
|---|---:|---:|
| Recall@5 | +3.02 | [2.72, 3.33] |
| NDCG@5 | +5.70 | [5.51, 5.90] |
| Recall@10 | -0.46 | [-0.71, -0.21] |
| NDCG@10 | +4.48 | [4.32, 4.65] |
| Recall@20 | +3.73 | [3.45, 4.02] |
| NDCG@20 | +5.61 | [5.46, 5.77] |

这组曲线说明模型不是整体变差：它在 K=5 和 K=20 都显著优于 Global，但在恰好 K=10 时少覆盖了约 0.46 个百分点。语义残差正在重新排列和扩展候选，却还没有学会在十个位置的预算下兼顾热门兴趣覆盖与个性化探索。

## 分层结果

- Macro Recall@10：Global 11.52%，Zero-shot 13.06%，Linear Ridge 12.66%。改进不是只来自一个热门标签。
- Tail 50 个标签的 Recall@10 对三种模型都是 0；到 K=20 也只有 0.052% micro recall。当前方法没有解决极端长尾。
- Linear Ridge 在单 target 用户上 Recall@10 / NDCG@10 为 47.99% / 24.79%；在 6+ target 用户上为 31.32% / 31.81%。多正例用户的 recall 更难，但已命中的正例位置更靠前。
- History 1–10 条的用户上 Linear Ridge NDCG@10 为 26.84%，而 101+ 条的 34 个用户只有 16.17%。无权 mean pooling 对超长、混合兴趣 History 不够合适。

## Acceptance gate

预设门槛要求真实 History 同时：

1. 超过 Global 和所有 shuffled History；
2. Recall@10、NDCG@10 均至少提升 1 个百分点；
3. paired-bootstrap CI 排除 0；
4. 三个 shuffle seed 方向一致，macro/tail 不崩溃。

结果为 **FAIL**。两个语义模型都稳定超过 shuffled History，但都没有在 Recall@10 上超过 Global，更没有达到 +1 个百分点。因此现在可以支持“History 含有真实个性化排序信号”，不能支持“当前语义模型已全面优于非个性化 Global”。

## 失败机制与下一步

Validation 上 Linear Ridge 从 `λ=0` 到 `λ=16` 时 Recall@10 从 43.59% 升到 45.92%，NDCG@10 从 26.72% 升到 30.09%；同一冻结参数到 Dev 后保留了 NDCG 增益，却没有保留 Recall@10 增益。这与此前观测到的 Train–Dev target 分布漂移一致，不能通过在 Dev 上重新选 λ 来修补。

下一步优先做“Global coverage scaffold + semantic reranker”：先用时间/频率校准后的 Global 生成较宽候选池，再用真实 History 语义只负责排序和少量扩展，并在 Train 内多个时间/分布切片上用 Recall@10 约束选择融合权重。同时把 mean pooling 替换成 title-level attention 或 recency-aware pooling，并为 torso/tail 设置独立的 coverage 约束。

在该便宜模型同时超过 Global 与 shuffled History 之前，不建议启动昂贵的 Qwen/LoRA 新兴趣训练。由于本轮已经查看了 Dev，后续迭代应在 Train 内新增冻结的二级验证集；当前 Dev 可继续作为探索性 benchmark，但新的最终结论需要额外未见 holdout。

## 产物

- 用户级派生数据：`data/processed/new_interest_static_user_disjoint_semantic_users_v1/`
- 完整实验目录：`artifacts/future-novel-interest/static-user-disjoint-semantic-v1/`
- 总清单与 SHA-256：`manifest.json`
- Validation 选参：`tuning.json`
- 全部 paired bootstrap：`comparisons.json`
- Acceptance gate：`acceptance.json`
- 三 seed donor 映射：`shuffle_manifest.parquet`
- 逐条件指标、逐用户明细与 Top-20 预测：`conditions/<condition>/`

最终测试：`pytest -q`，240 passed。
