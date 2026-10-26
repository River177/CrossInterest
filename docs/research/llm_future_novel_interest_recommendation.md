# 利用离线日志与 LLM 预测用户未来新兴趣：文献调研与研究方向建议

> 调研日期：2026-08-12  
> 建议主方向：**Future Observed Novel-Interest Retrieval**  
> 中文：**未来观测新兴趣检索**

## 1. 核心结论

现有 LLM 推荐研究已经覆盖：

- 用户历史的自然语言画像与可编辑 profile；
- 候选物品的 LLM 排序；
- 根据历史生成搜索 query 或兴趣表示；
- Semantic ID / 生成式全库检索；
- 多兴趣建模、推荐 Agent 和 rationale distillation；
- novelty、unexpectedness、serendipity 和跨域推荐。

但是，公开研究中仍缺少一个标准任务：

> 给定用户过去的兴趣画像，预测未来点击中会出现哪些历史里从未出现过的主题类别。

与本项目最接近的是 Google 工业研究：

- [LLMs for User Interest Exploration in Large-scale Recommendation Systems](https://arxiv.org/abs/2405.16363)
- [User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems](https://aclanthology.org/2025.acl-industry.70/)

该方向将系统拆成：

1. LLM 生成多个新兴趣 cluster；
2. 用户对齐模型从新兴趣中选择更可能被接受的 cluster；
3. 下层推荐器把抽象兴趣落到真实内容。

但它依赖商业日志和线上实验，没有公开、可复现的未来新兴趣 benchmark。

因此，CrossInterest 最有价值且可执行的研究贡献是：

1. 从 MIND 构造公开的**用户画像到未来新兴趣集合** benchmark；
2. 提出**新兴趣生成与用户对齐分离**的 LLM 方法；
3. 显式分析曝光偏差，区分 exposed、exposed-but-unclicked 和 never-exposed；
4. 比较直接生成、语义检索、统计 transition 和两阶段 LLM 方法。

能够成立的结论是：

> 模型能否预测未来日志中观测到的新兴趣。

不能从普通 MIND 离线日志直接声称：

> 推荐行为创造了用户兴趣，或用户一定喜欢从未曝光的主题。

## 2. 调研范围

检索范围：

- 时间：经典工作以及 2020–2026 年研究；
-  venue：RecSys、SIGIR、WWW、WSDM、KDD、ACL、EMNLP、NAACL、
  NeurIPS、ICLR、CIKM；
- 主题：
  - LLM-based recommender systems；
  - language-based user profiles；
  - generative recommendation / retrieval；
  - multi-interest and user-interest exploration；
  - novelty / unexpectedness / serendipity；
  - cross-domain / cross-topic recommendation；
  - exposure bias / positive-unlabeled learning；
  - counterfactual and offline evaluation；
  - news recommendation and MIND。

优先使用正式会议论文、官方 proceedings、DOI、ACL Anthology、OpenReview 和
官方项目页。未正式发表的工作在本文中标为 preprint。

## 3. 概念必须区分

| 概念 | 定义 | 是否等于新兴趣 |
|---|---|---|
| Item novelty | 全局低流行度或用户未消费过的 item | 否 |
| User-relative novelty | 用户历史中未出现过的内容 | 是新兴趣的必要条件，不充分 |
| Diversity | 推荐列表内部的内容差异 | 否 |
| Unexpectedness | 偏离用户画像或普通推荐器预期 | 否 |
| Serendipity | 相关/有用且意外，通常还要求新颖 | 接近，但依赖主观判断 |
| Future observed novel interest | 历史中未出现、未来日志中被点击的主题 | 本项目可观测的监督目标 |
| Genuine latent interest adoption | 新兴趣出现并长期持续或重复 | MIND 短窗口只能弱近似 |

经典综述和指标工作包括：

- Kaminskas & Bridge,
  [Diversity, Serendipity, Novelty, and Coverage](https://doi.org/10.1145/2926720),
  ACM TiiS 2016；
- Ge et al.,
  [Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity](https://doi.org/10.1145/1864708.1864761),
  RecSys 2010；
- Vargas & Castells,
  [Rank and Relevance in Novelty and Diversity Metrics](https://doi.org/10.1145/2043932.2043955),
  RecSys 2011；
- Kotkov et al.,
  [Rethinking Serendipity in Recommender Systems](https://doi.org/10.1145/3576840.3578310),
  CHIIR 2023；
- Kotkov et al.,
  [The Dark Matter of Serendipity](https://doi.org/10.1145/3627508.3638342),
  CHIIR 2024。

这些工作表明：仅仅推荐一个距离历史很远的主题，不等于成功发现新兴趣。它还必须与
用户后续行为或主观反馈相关。

## 4. LLM 推荐研究的主要路线

### 4.1 推荐任务语言化与指令微调

**P5**

- Geng et al.,
  [Recommendation as Language Processing](https://doi.org/10.1145/3523227.3546767),
  RecSys 2022；
- 将序列推荐、评分、解释等任务统一为 text-to-text；
- 主要仍在已知 item catalog 上预测 item ID 或候选。

**TALLRec**

- Bao et al.,
  [TALLRec](https://doi.org/10.1145/3604915.3608857),
  RecSys 2023；
- 通过 instruction tuning 和 LoRA 将 LLaMA 对齐到推荐判断；
- 输入仍包含需要评价的 item，不是开放的新兴趣生成。

结论：LLM 需要推荐数据对齐，单纯 zero-shot prompting 通常不稳定。

### 4.2 LLM 用户画像

代表工作：

- Zhou et al.,
  [Language-Based User Profiles for Recommendation](https://arxiv.org/abs/2402.15623),
  2024 preprint；
- Ramos et al.,
  [Transparent and Scrutable Recommendations Using Natural Language User Profiles](https://aclanthology.org/2024.acl-long.753/),
  ACL 2024；
- Steck et al.,
  [Editable User Profiles for Controllable Text Recommendations](https://doi.org/10.1145/3539618.3591677),
  SIGIR 2023；
- Wang et al.,
  [RDRec: Rationale Distillation for LLM-based Recommendation](https://aclanthology.org/2024.acl-short.6/),
  ACL 2024。

这些方法擅长：

- 把长历史压缩成可解释的兴趣 profile；
- 提取用户偏好的实体、属性、主题和理由；
- 改善冷启动、可解释性和可控性。

主要不足：

> 它们通常忠实总结已有兴趣，而不是外推未来尚未出现的兴趣。

这正是 CrossInterest 可以研究的 gap。

### 4.3 根据 Profile 生成 Query，再检索内容

**GPT4Rec**

- Li et al.,
  [GPT4Rec](https://arxiv.org/abs/2304.03879),
  2023 preprint；
- 根据历史 item title 生成多条假想搜索 query；
- 使用 BM25 从全库检索；
- 多 query 有助于覆盖用户的多个兴趣方向。

这一范式非常适合本项目：

```text
history
→ LLM-generated future-interest queries
→ news/topic retrieval
```

但 GPT4Rec 没有单独评估：

```text
历史中不存在、未来才发生点击的主题
```

### 4.4 生成式全库检索

**TIGER**

- Rajput et al.,
  [Recommender Systems with Generative Retrieval](https://openreview.net/forum?id=BJ0fQUU32w),
  NeurIPS 2023；
- 将 item 内容量化为 Semantic ID；
- 直接生成下一个 item 的 Semantic ID；
- 不依赖人工提供的小候选列表，但仍需要索引好的 item catalog。

**LLMTreeRec**

- Zhang et al.,
  [LLMTreeRec](https://aclanthology.org/2025.coling-main.59/),
  COLING 2025；
- 用自然语言偏好和层次化 catalog tree 做冷启动检索；
- LLM 先选择 category/subcategory/keyword，再从叶节点召回 item。

这两类方法说明：

> LLM 可以先预测语义区域，再落到真实可推荐内容；不需要自由生成不存在的新闻。

### 4.5 Agent 与 Tool-based Recommendation

代表工作：

- Wang et al.,
  [RecMind](https://aclanthology.org/2024.findings-naacl.271/),
  Findings NAACL 2024；
- Zhao et al.,
  [ToolRec](https://arxiv.org/abs/2405.15114),
  2024 preprint。

它们允许 LLM：

- 调用搜索、数据库和属性检索工具；
- 维护 personalized memory；
- 通过多步规划扩展推荐候选。

但 Agent 不能解决训练标签和曝光偏差问题；它主要改变候选发现和推理方式。

## 5. 新闻推荐与 LLM

### 5.1 MIND 和 PLM 新闻推荐

MIND 的标准任务仍是：

```text
history + current impression candidates
→ candidate click ranking
```

来源：

- Wu et al.,
  [MIND: A Large-scale Dataset for News Recommendation](https://aclanthology.org/2020.acl-main.331/),
  ACL 2020。

代表 PLM 工作：

- Zhang et al.,
  [Empowering News Recommendation with Pre-trained Language Models](https://arxiv.org/abs/2104.07413),
  SIGIR 2021；
- Zhang et al.,
  [UNBERT](https://www.ijcai.org/proceedings/2021/0462),
  IJCAI 2021；
- Qi et al.,
  [HieRec](https://aclanthology.org/2021.acl-long.423/),
  ACL 2021；
- Li et al.,
  [MINER](https://aclanthology.org/2022.findings-acl.29/),
  Findings ACL 2022；
- Zhang & Wang,
  [Prompt Learning for News Recommendation](https://doi.org/10.1145/3539618.3591752),
  SIGIR 2023。

这些工作主要提升 news encoding、user encoding 和 impression ranking，没有把未来
未见主题作为独立目标。

### 5.2 生成式新闻推荐

**Generative News Recommendation**

- [WWW 2024](https://doi.org/10.1145/3589334.3645448)；
- 生成 theme-level news/user representations 和 personalized multi-news
  narrative；
- 再通过 ranking/fusion 模型完成新闻推荐；
- 使用 MIND，但没有建立 future novel-topic benchmark。

**RecPrompt**

- [CIKM 2024](https://doi.org/10.1145/3627673.3679987)；
- 自动迭代优化 LLM 新闻推荐 Prompt；
- 仍以普通点击排序指标为主。

**MIPNR**

- [Multi-Interest Matching for Personalized News Recommendation with LLMs](https://doi.org/10.1145/3726302.3730233),
  SIGIR 2025；
- 使用 LLM 生成显式文本偏好并进行 category-guided multi-interest matching；
- 仍未隔离历史外未来主题。

目前没有发现公开的主流 benchmark 将任务定义为：

```text
fixed user profile
→ future clicked topic categories absent from profile
```

## 6. 最接近本项目的直接先例

### 6.1 LLM 新兴趣 Cluster 生成

Google 工业研究：

- [LLMs for User Interest Exploration in Large-scale Recommendation Systems](https://arxiv.org/abs/2405.16363),
  2024 preprint；
- [User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems](https://aclanthology.org/2025.acl-industry.70/),
  ACL Industry 2025。

方法核心：

```text
watch history
→ novelty-focused LLM generates new interest clusters
→ user-aligned LLM selects plausible clusters
→ hierarchical recommender retrieves items
```

这与 CrossInterest 的目标高度一致：

```text
news history
→ novel topic hypotheses
→ user alignment
→ real news grounding
```

它提供了强方法先例，但留下了公开研究空间：

- 数据和兴趣 cluster taxonomy 不公开；
- 没有公开离线 benchmark；
- 曝光策略和标签构造不可复现；
- 主要结论来自商业日志和线上实验。

### 6.2 UpliftRec

- Chen et al.,
  [Treatment Effect Estimation for User Interest Exploration](https://arxiv.org/abs/2405.08582),
  SIGIR 2024；
- 将推荐位分配到不同 category，估计不同曝光比例下的用户响应；
- 使用 propensity weighting 和 treatment-effect formulation。

它适合作为“不同主题应该获得多少探索曝光”的方法参考，但普通 MIND 没有官方
logging propensity，因此不能直接复制其因果解释。

## 7. 离线日志的根本限制

### 7.1 未曝光不是负例

离线 implicit feedback 中：

```text
clicked = observed positive
exposed but unclicked = weak negative / no-click
never exposed = unlabeled
```

不能把 never-exposed topic 直接当成用户不喜欢。

重要文献：

- Steck,
  [Training and Testing of Recommender Systems on Data Missing Not at Random](https://doi.org/10.1145/1835804.1835895),
  KDD 2010；
- Schnabel et al.,
  [Recommendations as Treatments](https://proceedings.mlr.press/v48/schnabel16.html),
  ICML 2016；
- Saito et al.,
  [Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback](https://doi.org/10.1145/3336191.3371783),
  WSDM 2020；
- Chen et al.,
  [Bias and Debias in Recommender System: A Survey](https://doi.org/10.1145/3564284),
  ACM TOIS 2023。

### 7.2 MIND 可以支持什么

可以支持：

- 固定用户画像与未来点击之间的关联预测；
- 未来被观测到的新主题检索；
- exposed novel topics 的点击排序；
- future observed positives 的 full-taxonomy ranking；
- popularity、topic transition 和跨主题语义关系分析。

不能单独支持：

- 用户对从未曝光主题的真实偏好概率；
- 推荐某主题会不会创造兴趣；
- 推荐策略的因果 uplift；
- 用户长期兴趣是否真正改变。

### 7.3 正确的论文措辞

推荐：

```text
predict future observed novel interests
forecast interest transitions
retrieve future clicked topics absent from the observed profile
```

避免：

```text
discover users' true hidden interests
create new interests
prove preference expansion
```

## 8. 建议的研究方向

| 方向 | 新颖性 | MIND 可行性 | 风险 | 建议 |
|---|---:|---:|---:|---|
| A. Future observed novel-interest benchmark + 两阶段 LLM | 高 | 高 | 中 | **主方向** |
| B. LLM query generation + future news retrieval | 中高 | 高 | 中 | 主模型分支 |
| C. LLM semantic bridge / unexpectedness | 高 | 中高 | judge 校准 | 辅助贡献 |
| D. Exposure-aware multi-objective serendipity ranking | 中高 | 中 | 指标主观 | 后续扩展 |
| E. 因果兴趣创造/uplift | 高 | 低 | 无 propensity | 不作为 MIND 主结论 |

## 9. 推荐主方向

### 9.1 研究问题

正式研究问题：

> Given a user's fixed historical news profile, can an LLM-based recommender
> retrieve topic categories absent from that profile that will appear in the
> user's future observed clicks?

中文：

> 给定用户的固定历史新闻画像，LLM 推荐模型能否检索历史中未出现、但会在未来观测
> 点击中出现的主题类别？

建议名称：

```text
Future Observed Novel-Interest Retrieval
```

### 9.2 数据单位

不再使用：

```text
one impression → one target
```

改为：

```text
one user profile → one future novel-interest target set
```

定义：

```text
H(u) = MIND provided history topics
F(u) = topics clicked in the future behavior window
Y_new(u) = F(u) - H(u)
```

每个用户只有一个输入和一个多标签 target set，因此不同 future interests 不再被当成
互相冲突的答案。

### 9.3 建议的数据划分

主 benchmark：

- 使用固定 MIND history 作为输入；
- 聚合未来标注窗口中的全部点击；
- 按 user 做 train/validation/test disjoint split；
- 所有用户使用相同长度的 future horizon；
- target 为 retained taxonomy 中的 future observed novel topics；
- 未出现在 target set 中的主题标记为 unlabeled，而不是严格负例。

辅助 temporal stress test：

- 按日期滚动；
- 输入额外包含 prediction date / available news pool；
- 最后一天仅作为未来窗口；
- 检查 news-cycle drift。

### 9.4 两个评测 Track

#### Track 1：Exposure-restricted

候选主题只包含未来窗口中真实曝光给该用户的主题。

优点：

- clicked 与 exposed-but-unclicked 可以比较；
- 更接近可识别的监督问题。

缺点：

- 仍受原推荐策略限制；
- 不是开放式推荐。

#### Track 2：Full-taxonomy retrieval

模型对全部 retained topics 排序，排除 history 已有主题。

Ground truth：

```text
future observed novel clicked topics
```

优点：

- 符合从历史推荐新主题的研究目标；
- 可测试开放主题检索。

限制：

- never-exposed topics 是 unknown；
- 指标只能解释为对 observed positives 的检索能力。

建议同时报告两个 Track，它们之间的差异本身就是 exposure sensitivity。

## 10. 推荐模型：Novelty Generation + User Alignment

### 10.1 Stage 1：结构化用户画像

LLM 从历史标题和主题中提取：

```json
{
  "stable_interests": [],
  "entities": [],
  "content_styles": [],
  "latent_needs": [],
  "possible_interest_bridges": []
}
```

可以离线使用强 LLM 生成 teacher profile，再训练小模型或缓存 profile，避免线上逐用户
调用大模型。

### 10.2 Stage 2：新兴趣 Hypothesis Generator

输入 profile，生成历史中未出现的若干 topic hypotheses：

```text
profile
→ {candidate novel topic 1, ..., candidate novel topic M}
```

生成必须被约束到：

- 固定 110/125 taxonomy；或
- 从 taxonomy/semantic cluster tree 中逐层生成。

不能自由生成不存在或没有内容库存的主题。

### 10.3 Stage 3：User Alignment Ranker

对每个 hypothesis 估计：

```text
P(future observed click on topic | profile, topic)
```

训练信号：

- positive：future observed novel clicked topic；
- exposed-but-unclicked：较强的 unlabeled/weak negative；
- never-exposed：PU unlabeled；
- popularity 和 exposure frequency：作为控制变量或分层指标。

### 10.4 Stage 4：真实内容 Grounding

将预测主题落到未来可用的新闻库：

```text
predicted topic
→ BM25 / embedding / semantic-ID retrieval
→ available news articles
```

这避免 LLM 编造新闻，也能评估 topic prediction 与 item retrieval 的误差来源。

### 10.5 可选 Interest Bridge 模块

LLM 生成从历史兴趣到候选新主题的简短 bridge：

```text
American Football
→ athlete legal disputes and league discipline
→ Courts & Legal Cases
```

研究问题：

- bridge 是否提升 alignment ranking？
- bridge 是否只是在生成合理故事？
- empirical transition lift 与 LLM bridge 是否一致？

Bridge 应作为特征或蒸馏监督，不应直接当成事实解释。

## 11. 训练目标

推荐从简单到复杂：

1. Multi-label binary classification；
2. Pairwise ranking：positive novel topic > sampled unlabeled topic；
3. Positive-unlabeled risk；
4. Set likelihood / listwise ranking；
5. Novelty-generation coverage loss；
6. Alignment accuracy + diversity regularization。

不建议首先使用 GRPO。应先证明：

- 数据 benchmark 有可学习信号；
- 两阶段模型超过统计和 embedding baseline；
- reward 与 future observed target 一致；
- exposure/popularity sensitivity 已被量化。

## 12. Baselines

### 12.1 非 LLM

- Uniform random；
- global / recent topic popularity；
- user-profile nearest topics；
- last-interest / full-history transition probability；
- transition lift；
- TF-IDF/BM25 profile-to-topic；
- sentence-embedding centroid；
- one-vs-rest logistic regression；
- multi-label MLP；
- graph/topic-transition model。

### 12.2 LLM

- Zero-shot direct topic generation；
- Few-shot topic generation；
- LLM profile + linear ranker；
- GPT4Rec-style multi-query retrieval；
- TALLRec-style instruction-tuned preference scorer；
- TIGER/LLMTreeRec-style hierarchical generative retrieval；
- novelty generator only；
- alignment ranker only；
- novelty generator + alignment ranker。

## 13. 指标

主指标：

```text
NovelTopic Recall@K
NovelTopic NDCG@K
NovelTopic Hit@K
Macro Recall@K over target topics
```

补充指标：

- topic coverage；
- long-tail recall；
- intra-list diversity；
- rank-aware novelty；
- α-NDCG；
- popularity concentration；
- exposure-stratified recall；
- prediction validity / taxonomy compliance；
- hallucinated or unavailable topic rate；
- user-history length slice；
- topic semantic-distance slice。

必须分别报告：

```text
exposed novel positives
never-exposed unknown topics
seen-user / unseen-user
single-target / multi-target
head / torso / tail topics
```

Precision 只能描述相对 observed targets 的 precision，不能解释成完整真实偏好 precision。

## 14. 关键 Ablation

1. Titles vs topic sequence vs LLM profile；
2. Fixed taxonomy vs LLM-generated semantic clusters；
3. Direct generation vs query retrieval vs hierarchical retrieval；
4. With/without novelty constraint；
5. With/without user alignment stage；
6. With/without empirical transition features；
7. With/without LLM interest bridge；
8. Different numbers of generated hypotheses；
9. Random vs popularity vs exposure-matched unlabeled sampling；
10. User-disjoint vs temporal evaluation；
11. Exposure-restricted vs full-taxonomy retrieval；
12. Zero-shot vs SFT/LoRA。

## 15. 可发表的论文贡献

建议论文定位：

> A public exposure-aware benchmark and an LLM-based two-stage method for
> forecasting future observed novel interests from static user profiles.

可主张的贡献：

1. **Task contribution**：明确定义 future observed novel interest，区别于 item
   novelty、cold start 和 serendipity；
2. **Benchmark contribution**：将 impression-level 日志聚合为 user-profile-level
   future multi-label targets；
3. **Method contribution**：分离 novelty generation 与 user alignment；
4. **Evaluation contribution**：同时提供 exposure-restricted 与 full-taxonomy
   两个 Track；
5. **Analysis contribution**：系统量化 exposure、popularity、news-cycle 和
   semantic-distance 对结果的影响。

可能标题：

```text
Forecasting Future Observed Novel Interests from Static User Profiles

Beyond Familiar Interests: LLM-based Novel-Interest Generation and
User Alignment for News Recommendation

From User Profiles to Future Novel Topics: An Exposure-Aware Benchmark
for LLM Recommenders
```

## 16. 不建议作为主方向的内容

### 16.1 “LLM 直接生成一个下一主题”

问题：

- 单标签不符合多目标行为；
- 容易学习 popularity prior；
- 自由生成可能 hallucinate；
- 不能区分 novelty 与 user alignment。

可作为 baseline，不适合作为完整方法。

### 16.2 只做 Candidate K=10

问题：

- 结论强依赖负样本采样；
- popularity shortcut 明显；
- 不对应开放式新内容推荐。

可作为诊断 Track，不适合作为唯一 benchmark。

### 16.3 直接使用 LLM Judge 定义 Serendipity

LLM Judge 可以扩展人工标注，但必须：

- 先在人工样本上校准；
- 报告 prompt sensitivity；
- 报告人类/LLM agreement；
- 不把 judge 分数当作用户真实偏好。

相关工作：

- Tokutake et al.,
  [A Universal Framework for Offline Serendipity Evaluation via LLMs](https://doi.org/10.1145/3746252.3760911),
  CIKM 2025。

### 16.4 因果“兴趣创造”

普通 MIND 没有随机曝光和 logging propensity，不能支持强因果结论。若未来需要研究
推荐是否创造兴趣，应使用：

- Yahoo! R6A；
- Open Bandit Dataset；
- KuaiRand；
- 或新的线上随机探索实验。

## 17. 最终建议

优先顺序：

1. **先构造 user-profile-level future multi-label benchmark**；
2. 验证无模型、embedding 和 multi-label classifier 的可学习性；
3. 实现 GPT4Rec-style query generation baseline；
4. 实现 novelty generator + user alignment 两阶段模型；
5. 增加 exposure-aware 双 Track；
6. 再考虑 rationale distillation、hierarchical retrieval 或 RL/GRPO。

建议当前主研究问题确定为：

> **LLM 能否从固定用户历史画像中生成并排序一组合理的新兴趣假设，使未来日志中实际
> 观测到的新主题获得更高排名？**

这是现有公开研究中的明确 gap，与 CrossInterest 当前数据和已有分析高度匹配，同时
避免对未曝光兴趣和因果效果做无法支持的强主张。

## 18. 重点参考文献

### LLM 推荐

- [P5, RecSys 2022](https://doi.org/10.1145/3523227.3546767)
- [TALLRec, RecSys 2023](https://doi.org/10.1145/3604915.3608857)
- [GPT4Rec, 2023 preprint](https://arxiv.org/abs/2304.03879)
- [TIGER, NeurIPS 2023](https://openreview.net/forum?id=BJ0fQUU32w)
- [RecMind, Findings NAACL 2024](https://aclanthology.org/2024.findings-naacl.271/)
- [RDRec, ACL 2024](https://aclanthology.org/2024.acl-short.6/)
- [LLMTreeRec, COLING 2025](https://aclanthology.org/2025.coling-main.59/)
- [User Feedback Alignment, ACL Industry 2025](https://aclanthology.org/2025.acl-industry.70/)

### 新闻推荐

- [MIND, ACL 2020](https://aclanthology.org/2020.acl-main.331/)
- [HieRec, ACL 2021](https://aclanthology.org/2021.acl-long.423/)
- [MINER, Findings ACL 2022](https://aclanthology.org/2022.findings-acl.29/)
- [Prompt Learning for News Recommendation, SIGIR 2023](https://doi.org/10.1145/3539618.3591752)
- [Generative News Recommendation, WWW 2024](https://doi.org/10.1145/3589334.3645448)
- [MIPNR, SIGIR 2025](https://doi.org/10.1145/3726302.3730233)

### Novelty、Serendipity 与兴趣探索

- [Beyond Accuracy, RecSys 2010](https://doi.org/10.1145/1864708.1864761)
- [Rank and Relevance in Novelty and Diversity Metrics, RecSys 2011](https://doi.org/10.1145/2043932.2043955)
- [PURS, RecSys 2020](https://doi.org/10.1145/3383313.3412238)
- [RADio, RecSys 2022](https://doi.org/10.1145/3523227.3546780)
- [Wisdom of Crowds for Serendipity, SIGIR 2023](https://doi.org/10.1145/3539618.3591787)
- [UpliftRec, SIGIR 2024](https://arxiv.org/abs/2405.08582)

### 离线偏差与评测

- [Steck, KDD 2010](https://doi.org/10.1145/1835804.1835895)
- [Recommendations as Treatments, ICML 2016](https://proceedings.mlr.press/v48/schnabel16.html)
- [Counterfactual Risk Minimization, ICML 2015](https://proceedings.mlr.press/v37/swaminathan15.html)
- [Unbiased Learning from MNAR Feedback, WSDM 2020](https://doi.org/10.1145/3336191.3371783)
- [Bias and Debias in Recommender Systems, TOIS 2023](https://doi.org/10.1145/3564284)
- [Causal Inference in Recommender Systems, TOIS 2024](https://doi.org/10.1145/3639048)
