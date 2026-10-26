# 从用户历史预测未来新兴趣：相关工作与公开数据集审计

> 调研日期：2026-08-19  
> 面向项目：CrossInterest  
> 结论口径：预测 **future observed novel interest（未来日志中新观测到的兴趣）**，不声称识别用户终身首次兴趣，也不把未曝光项当作负例。

## 1. 执行摘要

### 1.1 是否已有完全相同的工作

**没有发现公开、标准化且可复现的 benchmark 同时满足以下条件：**

1. 输入是稳定用户的历史行为；
2. 目标是历史中未出现、未来真实出现的**语义兴趣**，而非仅仅新 item；
3. 能区分已曝光、未点击与从未曝光；
4. 能区分一次点击和之后持续采用该兴趣。

最接近的工业论文已经在做“生成/选择历史之外的兴趣方向”，但日志、taxonomy、训练数据和线上标签都是私有的。最接近的公开标签构造是 NNBR：它预测下一个 basket 里用户从未买过的 item；这仍是 **new item**，不是 **new semantic category**。

因此，CrossInterest 的研究缺口仍然成立，但表述应保守：

> 给定冻结的用户历史，预测未来观察窗口中实际出现、且在该历史窗口内未出现过的语义兴趣集合。

### 1.2 最值得用的数据集

综合用户连续性、时间顺序、语义 taxonomy、曝光、持续性和当前可获得性，建议优先级为：

1. **EB-NeRD**：最适合建立新的新闻同域主 benchmark；有 21 天点击历史、7 天 impression/click、文章发布时间、正文、category/subcategory/topic、阅读时长与滚动信息。
2. **KuaiRand**：最适合做随机曝光和 exposure-bias 辅助 benchmark；有稳定用户、时间戳、12 类反馈、标准推荐历史、随机插入曝光和四级视频类别。
3. **H&M**：最适合做交易级“新类别购买 onset”；两年购买序列与丰富服装 taxonomy，但没有曝光和历史库存。
4. **Goodreads**：最适合做跨 genre 的自然语义迁移和后续持续阅读；无曝光，genre 边界较模糊。
5. **dunnhumby Complete Journey**：最适合验证新 commodity/subcommodity 的持续复购；单位是 household，不是个人。

Amazon Reviews 2023、Adressa、KuaiRec、FINN.no Slates、ContentWise Impressions、MovieLens 32M 可作为补充或 robustness benchmark。

### 1.3 最重要的执行限制

- EB-NeRD 的条款允许开发和测试新闻推荐系统，但明确举例禁止将数据用于 “training of language models”。如果计划用 EB-NeRD 内容或行为微调 LLM，必须先取得数据方的书面澄清或许可。
- KuaiRand 的随机替换仅约占交互的 0.37%；它适合辅助验证曝光偏差，不能替代大规模自然分布主 benchmark。
- H&M、Amazon、Goodreads、dunnhumby 都没有完整曝光。未消费类别是 **unknown**，不是负例。
- 所有数据里的“第一次出现”都只能称为 **newly observed in the dataset window**。

## 2. 本次调研方法和证据边界

本次将问题拆成九条独立调研线：直接新兴趣工作、兴趣演化/意图迁移、多兴趣与生成式检索、novelty/serendipity/cross-domain、新闻、交易、媒体、随机曝光以及跨域/查询会话数据。优先核验原论文、正式 proceedings、数据方页面、作者仓库、原始 schema 和 license；survey 与第三方镜像只用于发现线索。

截至 2026-08-19：

- 九份结构化结果均覆盖 33 个审计字段；
- 每个不确定结论均显式标记；
- 没有下载或逐行检查大型数据归档，因此文件级 count、缺失率和 join 成功率仍属于下一阶段 preflight，而不是本报告已经完成的事实；
- DOI 页面出现的自动访问 403 是 ACM 的反自动化限制，不表示 DOI 不存在；关键 2024–2026 论文的题名、venue 和 DOI 另经 Crossref、DBLP、arXiv 或官方 proceedings 交叉核验。

## 3. 先把“新”拆开：否则相关工作会被误判

设 (H_0(u)) 为预测时刻冻结的历史，(F(u)) 为之后的观察窗口，

\[
Y_{new}^{(g)}(u)=U_g(F(u))\setminus U_g(H_0(u)),
\]

其中 (U_g) 把行为映射到粒度 (g) 的语义单元。至少应分别报告：

| 层次 | 定义 | 是否可称为新兴趣 |
|---|---|---|
| Repeat item | 历史中已经出现过的 item | 否 |
| New item / known category | item 新，但 leaf/ancestor category 已出现 | 通常否 |
| New leaf category | 未来 leaf category 不在历史 | CrossInterest 主目标候选 |
| New ancestor/domain | 更高层类别或 domain 首次出现 | 更强但样本更少 |
| Sustained new category | 新类别 onset 后，在独立后续窗口再次出现 | 最接近持续兴趣 |

另外还要区分两种合法但不同的任务：

- **Static-profile forecast**：整个未来窗口都相对 (H_0) 计算新颖性，回答“从当前画像能否预测未来会出现什么新方向”。
- **Dynamic onset forecast**：每个事件前把更早的正反馈加入历史，只把真正第一次出现的时刻记为 onset。

这两种结果不能混为一谈。CrossInterest 当前的 fixed-profile user-level benchmark 属于第一种；动态 replay 属于第二种诊断。

## 4. 最接近的相关工作

### 4.1 第一层：直接生成或优化历史外语义兴趣

| 工作 | 实际目标 | 与 CrossInterest 的关系 | 公开数据 |
|---|---|---|---|
| [LLMs for User Interest Exploration in Large-scale Recommendation Systems](https://doi.org/10.1145/3640457.3688161), RecSys 2024 | LLM 生成 novel interest clusters，再由序列推荐器落到 item | 最直接的方法先例 | 否；工业日志与 taxonomy 私有 |
| [User Feedback Alignment for LLM-powered Exploration](https://aclanthology.org/2025.acl-industry.70/), ACL Industry 2025 | novelty LLM 生成候选，user-aligned LLM 做 best-of-n 选择 | 支持“新颖性生成”和“用户可接受性”分离 | 否；线上数据私有 |
| [SERAL](https://arxiv.org/abs/2502.13539), 2025 | 正例需同时满足点击，且 atomic category 在此前 10 个 active days 的有效曝光中未出现 | 最接近可操作 label 的工业先例 | 否；19,141 个正例及线上数据均私有 |
| [Personalized Interest Graphs for Theme-Driven User Behavior](https://doi.org/10.1145/3705328.3748133), RecSys 2025 | LLM 构建多粒度 interest graph，支持 theme-driven 跨类别推荐 | 结构化跨类别兴趣扩展先例 | 否；eBay 日志私有 |
| [Serendipitous Recommendation with Multimodal LLM](https://arxiv.org/abs/2506.08283), RecSys 2025 EARL Workshop | 找到相关但尚未探索的兴趣 cluster，再用传统推荐器 grounding | 直接的 unexplored semantic-cluster 方法 | 否；短视频线上实验私有 |
| [CoEA](https://arxiv.org/abs/2508.00450), 2025 preprint，v3 2026 | 分离长期 group identity 与短期 individual interest，周期性协同优化 relevance/novelty | 可迁移到稳定历史与新兴兴趣分离 | 未发现可复用 benchmark |
| [BALANCE](https://doi.org/10.1145/3805712.3808399), SIGIR 2026 | Novelty Tag → Interest Topic → Recommendation List 的层次推理，并用 RL 优化 | 最新、最接近 topic-level novelty generation 的工业方法 | 否；JD.com 数据私有 |

这些论文支持 CrossInterest 的方法动机，但不能提供公开监督数据。它们的在线提升也不能自动转化为“离线历史能预测用户未来新类别”的证据。

### 4.2 第二层：公开、但只预测新 item 或近邻目标

| 工作 | “新”的定义 | 可借鉴内容 | 不能替代之处 |
|---|---|---|---|
| [NNBR](https://arxiv.org/abs/2308.01308), SIGIR 2023 | 下一个 basket 中从未消费过的 item | (H,F,F-H) 标签构造、只评 explore item、user-disjoint split | 不要求新 category；无曝光 |
| [UpliftRec](https://arxiv.org/abs/2405.08582), SIGIR 2024 | category 不在历史 top-3 即 unexpected | category exposure 作为 treatment，biased/unbiased 数据联合评测 | “非 top-3”不等于“历史从未出现” |
| [RepeatNet](https://doi.org/10.1609/aaai.v33i01.33014806), AAAI 2019 | repeat 与 explore next-item 模式 | 强制分层评估，防止整体指标被复制历史支配 | explore item 不等于新语义兴趣 |
| [A Next Basket Recommendation Reality Check](https://doi.org/10.1145/3587153), TOIS 2023 | next-basket 的 repeat/explore 分解 | 评估协议与强 baseline | 仍是 item/basket 任务 |
| [Topic-Level Bayesian Surprise](https://doi.org/10.1145/3604915.3608851), RecSys 2023 | 人工判断一本书的 topic 是否相对个人历史令人意外 | 可做人类语义 surprise 校准 | 只有 4 个用户、449 个标注 book |
| [ProRec-Video](https://doi.org/10.1609/aaai.v40i17.38475), AAAI 2026 | 规划层次化兴趣迁移路径与视频序列 | 主动兴趣迁移的研究方向 | 数据、真实用户标签、simulation 和 persistence 细节不足 |

### 4.3 第三层：方法可迁移，但原评测没有证明新兴趣预测

- MIND、ComiRec、SUM：把历史编码成多个兴趣向量，可能只是在更好地覆盖**已有兴趣**。
- P5：把推荐任务语言化，可以生成 query/category，但原任务没有定义 (Y_{new})。
- TIGER、LETTER：生成 Semantic ID，适合 closed-catalog grounding；生成新的 item token 不等于新类别。
- OneRec：生成推荐 session 并做 preference alignment；总体 watch-time 提升没有隔离 new-interest strata。
- DIEN、ISLF、structured-intent、CFPRec：预测 candidate CTR、next item 或 next POI；“兴趣状态随时间演化”不等于监督预测历史外兴趣。
- C2DSR、EMCDR：已知 source/target domain 间迁移或合成 target cold-start；通常已经选择了两个 domain 都活跃的用户，不能证明首次进入新 domain。

因此，方法对比应统一放到同一个 (Y_{new}) 协议中，而不是直接引用它们的原始 overall Recall/NDCG。

## 5. 数据集审计与推荐优先级

### 5.1 推荐候选

| 数据集 | 时间与用户 | 语义 | 曝光 | 持续性 | 建议角色 |
|---|---|---|---|---|---|
| [EB-NeRD](https://recsys.eb.dk/dataset/) | 1,103,602 users；每个 split 为 21 天 history + 7 天 behavior；六周总窗口 | title/abstract/body，32 category、263 subcategory、topics、发布时间 | 实际 in-view slate | 可用后续天和 read/scroll 做弱验证 | **新闻主 benchmark，最高优先级** |
| [KuaiRand](https://github.com/chongminggao/KuaiRand) | 27,285 users；2022-04-08 至 05-08；322,278,385 standard interactions | caption、tag、四级 category supplement | 1,186,059 random interactions；随机替换约 0.37% | 多反馈与后续标准流 | **随机曝光辅助，最高优先级** |
| [H&M](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) | 约 1.37M customers、31.8M transactions、2018-09 至 2020-09；精确数需下载后重算 | product type/group、department、section、garment group、颜色等 | 无 | 后续重复购买 | **交易级类别 onset 主候选** |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | 571.54M reviews、54.51M users、1996–2023 | 文本、属性、图片、层次 category | 无；仅 review 过的消费 | 后续 review 是弱 persistence proxy | 大规模跨域/长时间 robustness |
| [dunnhumby Complete Journey](https://www.dunnhumby.com/source-files/) | 2,500 frequent-shopping households、两年、店内全部购买 | commodity/subcommodity、brand、promotion | 无完整货架曝光 | 强复购信号 | household-level persistence benchmark |
| [Goodreads](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) | 876,145 users、228.6M shelf interactions、约 15M reviews | genre、author、series、文本和 book graph | 无 | 后续阅读/评分/评论 | 跨 genre 语义迁移 benchmark |
| [Adressa](https://reclab.idi.ntnu.no/dataset/) | 一周/长窗口版本；长版本约 3.08M users、27.2M views | 全文、category、keyword/entity | 无 impression slate | activeTime 可限定正例 | 长时间新闻 robustness |
| [KuaiRec](https://github.com/chongminggao/KuaiRec) | small matrix 1,411×3,327、99.6% observed | tag/category、视频特征 | 近全矩阵反馈 | 窗口较短 | dense-exposure 辅助 |
| [MovieLens 32M](https://grouplens.org/datasets/movielens/) + Tag Genome | 200,948 users、32M ratings、1995–2023 | genre + 1,084 tags | 无 | rating 序列；watch time 不可见 | 长时间语义 robustness |

### 5.2 曝光或主观评价辅助数据

| 数据集 | 有用之处 | 为什么不能做主 benchmark |
|---|---|---|
| [FINN.no Slates](https://github.com/finn-no/recsys-slates-dataset) | 完整可见 slate、稳定的短序列、类别 | 生产曝光非随机；公开 reduction 最多约 20 次交互 |
| [ContentWise Impressions](https://github.com/ContentWise/contentwise-impressions) | account history、完整推荐列表、series/type | 非随机曝光、语义较粗 |
| [Open Bandit Dataset](https://github.com/st-tech/zr-obp) | propensity/OPE 工具校验 | 没有持久用户历史 |
| Coat | 自选 rating 与随机 rating，可审计 MNAR | 静态评分，不是未来序列 |
| [Taobao Serendipity](https://github.com/greenblue96/Taobao-Serendipity-Dataset) | 真实用户主观 novelty/serendipity/perception | 一用户一行曝光；无历史、时间线和完整候选；仅 10 个 purchase/cart positives |

### 5.3 不建议作为主 benchmark

- Spotify Million Playlist / Spotify skip：playlist/session ID 不是稳定 user ID，且当前公开下载已停止。
- Yahoo! R6：很适合 contextual-bandit replay，但没有稳定长历史，当前 Webscope 获取状态也需重新核验。
- Yoochoose、Diginetica：session ID 不能提升为长期用户 ID。
- LFM-1b：数据方明确因 license 问题停止下载。
- HetRec Last.fm：user-artist listen 是聚合量；时间戳属于 tag assignment，不是 listen sequence。
- Globo.com：只有约 16 天、无 impression、无原始文本；可做 session/cold-article robustness，不适合持续兴趣。
- AOL query log：因严重隐私问题已撤回，应拒绝使用。
- TripClick/TREC Session：可研究 session intent transition，不能称为长期用户新兴趣。

## 6. 三个最优数据集的具体构造方案

### 6.1 EB-NeRD：新的新闻同域主 benchmark

#### 可用字段

官方 release 为每个 split 提供：

- `history.parquet`：行为窗口之前 21 天的点击、时间、read time、scroll；
- `behaviors.parquet`：之后 7 天的 impression time、in-view article IDs、clicked article IDs、next read time、next scroll；
- `articles.parquet`：title、subtitle、abstract、body、category、subcategory、topics、published time 等。

#### 推荐标签

1. 冻结 prediction time 为 7 天 behavior 窗口开始时刻。
2. (H_0(u))：21 天 history 中全部正点击对应的 category/subcategory；保留有序、多重事件版本供模型使用，同时用 set 计算 novelty。
3. (F(u))：7 天 behavior 中 clicked article 对应的语义单元。
4. 主标签：

   \[
   Y_{new}^{subcategory}(u)=subcategory(F(u))\setminus subcategory(H_0(u)).
   \]

5. 同时报告 category、subcategory、topic/冻结语义 cluster 四种粒度；官方 category/subcategory 为主，proprietary topic 只作 sensitivity。
6. Click 是主 onset；再用 `next_read_time`、`next_scroll` 构造 engagement-qualified sensitivity，不要把阈值事后调到结果最好。

#### 两个 evaluation track

- **Exposure-restricted**：候选只来自该用户未来实际 in-view 的 category/subcategory。未点击曝光只能称为 weak negative。
- **Full-taxonomy**：排序预测时刻前已经存在的全部 taxonomy 单元；从未曝光的类别仍是 unknown，主要报告 positive recall/rank，不计算把所有 unknown 当真负例的 accuracy。

#### 持续性

7 天很短，建议把该分析明确标为 sensitivity：用前 3–4 天检测 onset，用剩余天检测同类别再次产生 engagement-qualified click。太晚出现、没有足够 follow-up 的 onset 应标为右删失，而不是 one-off。

#### 禁止的泄漏

- 不使用 `total_inviews`、`total_pageviews`、`total_read_time` 等文章发布后七天累计统计；这些直接包含未来信息。
- 候选文章必须满足 `time_published <= prediction/event time`。
- popularity、topic cluster、文本 embedding 和阈值均只能在训练用户和 cutoff 前数据上拟合。
- static-profile 与 dynamic-onset 结果分开报告；不能用未来前几天点击更新模型输入，却仍声称是固定画像预测。

#### License gate

EB-NeRD 条款 4.6 允许数据用于推荐系统开发和结果，但明确说不得用于其他 derivative products，例如训练 language models。建议：

- 可以先做 schema、统计 baseline 和 frozen-LLM inference 的合规评估；
- 任何把文章或行为用于 LLM fine-tuning/SFT/continued pretraining 的计划，必须先向 Ekstra Bladet 请求书面许可；
- 不得重分发数据；license 最长六年，到期需删除。

### 6.2 KuaiRand：随机曝光与强反馈辅助 benchmark

#### 推荐切分

- (H_0(u))：`log_standard_4_08_to_4_21` 中的正反馈历史。
- (F(u))：`log_standard_4_22_to_5_08` 与 `log_random_4_22_to_5_08`，但分开评估。
- 语义：使用补充文件中的一至四级 category；一级/二级为主，三级/四级和 `UNKNOWN` 需做 coverage audit。
- 新兴趣：未来 engagement-qualified category 减去 (H_0) category。

#### 正反馈不要只选一个定义

至少并行报告：

1. 非零播放或基础观看；
2. watch ratio 达到预注册阈值；
3. like/follow/share/comment/profile-enter 等强反馈；
4. onset 后的其他日期再次对同类别产生正反馈。

#### 两个 track

- **Random-pool track**：只在 7,583 个随机池视频及 `is_rand=1` 事件中评价类别响应；这是最干净的机会集。
- **Natural-feed track**：在标准推荐流上评价自然分布效果，明确承认 policy-mediated exposure。

随机事件可以减轻曝光偏差，但不能证明“推荐创造了兴趣”。若随机曝光后用户之后在标准流再次正反馈同类别，只能先报告 association；后续标准流本身会根据前次行为改变推荐。

#### 泄漏与现实限制

- 随机插入仅约 0.37%，稀疏类别的方差可能很大。
- 不使用按天聚合的 `video_features_statistic` 未来统计作为预测特征。
- caption/category supplement 是后来发布的模型/内容侧映射，需冻结文件 hash、category confidence 与缺失规则。
- 没有完整 slate 和逐事件 logging propensity；“池内均匀替换”不等于完整推荐策略已知。

#### 低成本 preflight

优先审计 KuaiRand-Pure（约 194 MB 主包）与 category supplement，而不是直接下载 46 GB 的 27K 全量数据。先确认 ID join、每层 taxonomy coverage、随机事件正反馈率和可形成 onset/persistence 的用户数。

### 6.3 H&M：购买级新类别 onset

#### 推荐时间设计

不要只取一个最终 cutoff。使用多个 rolling cutoff，例如：

- 较长 burn-in/history；
- 较短 onset window；
- 独立 persistence window；
- user-disjoint train/validation/test，同时每个用户内部严格按时间构造。

具体周数应根据下载后的用户活跃分布预注册，不能在看到模型结果后选择。

#### 标签层次


- item novelty：新 article；
- leaf novelty：新 `product_type`；
- middle novelty：新 `product_group` / `section`；
- broad novelty：新 `garment_group` / `index`。

主标签建议为首次未来购买的 leaf/middle category；持续性要求之后购买同类别的**不同 article**，避免把同一商品补购误当成兴趣扩展。

#### 缺失曝光的处理

- 不能把未买过的全目录商品当负例。
- 用 cutoff 前已存在且在相近周被其他用户实际购买的 article/category 近似 contemporaneous availability；仍要标记为 opportunity proxy。
- 主指标用正例集合的 rank/recall；若做 sampled negatives，必须同时报告采样策略和 full candidate sensitivity。
- fashion seasonality 很强，recent popularity 和 same-week category popularity 是必须超过的 baseline。

#### 泄漏

- 文章 metadata 是 snapshot，可能包含未来才完整的属性；需检查首见日期并冻结 cutoff 可用字段。
- 不使用未来交易计算 popularity、库存代理或类别转移。
- customer 不能跨 split；同一用户的事件也不能随机拆分。

## 7. 其他高价值构造

### Goodreads

按 dated read/rating/review 排序，历史用 genre、author、series 或冻结文本 topic；未来做 set difference。必须把 shelf-only、read、rating 和 detailed review 分开，后两者是更强正例。后续对同 genre 的另一本书再次 read/review 可作为 persistence。问题是 genre 多标签且模糊、书籍曝光未知，因此适合语义迁移 robustness，不适合因果结论。

### dunnhumby

把 line items 聚合到 household basket；分成 burn-in、onset、persistence 三段，在 commodity/subcommodity 层构造新类别。优点是两年内该 household 在该零售商的全部店内购买，缺点是 household 不是个人，且仍不知道全部货架/促销机会。

### Amazon Reviews 2023

长时间、大规模、跨 33 个 domain，但只看到被 review 的购买。可以预测 future reviewed category/domain，不应称为完整购买兴趣。跨 domain join 前必须从实际文件验证 user ID 是否全局一致；不能从文档名称推断。

## 8. 统一评测协议

### 8.1 必须有的 baseline

1. Global popularity；
2. Recent-window popularity；
3. Last-category / category-transition；
4. History-category co-occurrence；
5. Actual History vs matched shuffled History；
6. Multi-interest encoder（MIND 或 ComiRec）；
7. Generative retrieval（TIGER 或 LETTER）或 taxonomy-path generation；
8. LLM 方法只在上述信号门通过后加入。

所有模型使用同一 (H,F,Y_{new})、候选空间和 cutoff，才能回答方法是否真的改善新兴趣预测。

### 8.2 指标

- user-level multi-label Recall@K、NDCG@K；
- macro-category Recall@K，防止 head category 主导；
- category coverage 与 per-category support；
- calibration / expected positives in top-K；
- repeat item、new item-known category、new leaf、new ancestor、persistent new category 五个互斥或嵌套 strata；
- actual-vs-shuffled History 的 paired difference；
- 按用户聚类 bootstrap 的 95% CI，不能把同一用户的多个事件当独立样本。

### 8.3 必须披露的统计

- 各层级 (Y_{new}) 非空用户数与比例；
- 每用户 target 数的均值、中位数和尾部；
- 类别 support、Gini/集中度、recent popularity 覆盖率；
- exposure-restricted target coverage；
- 一次性 onset 与可验证 persistence 的数量；
- 历史长度、future window 长度、右删失比例；
- taxonomy 缺失、unknown 和映射置信度；
- 去重前后事件数、稳定用户 join 率。

## 9. 主要风险与防护

| 风险 | 错误结论 | 防护 |
|---|---|---|
| 左截断 | 把数据集开始前已经喜欢的类别称为新兴趣 | 使用 burn-in；只称 newly observed；按历史长度做 sensitivity |
| 未知曝光 | 把从未看到的类别当不喜欢 | Exposure-restricted 与 full-taxonomy 分轨；unknown 不作真负例 |
| 一次点击 | 把偶然行为称为兴趣形成 | 独立 persistence window；报告 one-off 与 sustained |
| 未来 metadata | 用发布后累计统计、后验 taxonomy 或未来 popularity | 所有特征按 cutoff 冻结；保存 hash 与 first-seen time |
| 用户泄漏 | 同一用户或同一 profile 出现在 train/test | user/profile-disjoint split；用户聚类 bootstrap |
| 新 item 混淆 | 模型只是找到同一旧类别的新物品 | 分 item、leaf、ancestor/domain 报告 |
| 机会漂移 | 季节、库存、地域或新闻周期造成“新兴趣” | contemporaneous availability proxy、rolling cutoff、时间分层 |
| popularity shortcut | 模型几乎只输出热门类别 | recent/global baseline、macro recall、shuffled-History control |
| synthetic/self-judge | LLM 生成并由相似 LLM 判定“兴趣迁移成功” | 真实未来行为为主；人工/LLM judge 只作校准 |

## 10. 建议的下一步与停止门

### 阶段 A：只做获取与 schema preflight

建议先选择一个主数据和一个曝光辅助：

- 同域主数据：EB-NeRD；
- 曝光辅助：KuaiRand-Pure。

本阶段只做：条款确认、下载最小合法包、文件 hash、字段/row count、user join、taxonomy coverage、时间范围和缺失率。不开始全量构建、训练或远程任务。

**Gate A：** 若稳定用户、时间顺序、语义映射或合法使用范围不能通过，就停止该数据集，不用模型补救数据定义问题。EB-NeRD 的 LLM fine-tuning 许可必须在这里解决。

### 阶段 B：小样本 label demo

在确定性选取的小样本用户上只构造 (H,F,Y_{new})，输出：

- 各粒度 target 数；
- exposure coverage；
- one-off/persistence；
- popularity concentration；
- 10–20 个完整、人工可读的 history→future case。

**Gate B：** 如果有效用户太少、标签几乎全由少数热门类别占据、或者 persistence 无法测量，就缩小研究主张或更换数据，而不是直接全量训练。

### 阶段 C：便宜的可学习性测试

只运行 popularity、category transition、co-occurrence、actual-vs-shuffled History 和一个轻量 multi-interest/linear baseline。

**Gate C：** 至少要求 user-conditioned 方法在用户聚类 bootstrap 下稳定优于 matched shuffled History，并在 macro-category 指标上超过 recent popularity。若没有证据，停止 LLM 训练；此时问题更可能被全局先验或机会分布主导。

### 阶段 D：批准后再扩展

只有前三个 gate 通过后，再决定是否进行全量构建、ComiRec/TIGER/LLM 对比、SFT 或远程训练。任何外部数据应写入独立 builder、output identity、metadata 和报告，不覆盖当前 MIND/CrossInterest 的 canonical artifacts。

## 11. 最终建议

研究定位可以保留，而且公开 benchmark 的缺口比“普通 next-item recommendation”更明确。最稳妥的组合是：

1. 保留当前 MIND fixed-profile benchmark 作为 English/news continuity baseline；
2. 用 EB-NeRD 建立有真实 impression、时间、正文和 engagement 的新闻升级版，但先解决 LLM 训练许可；
3. 用 KuaiRand 验证结论在随机机会集和强反馈下是否仍成立；
4. 用 H&M 或 Goodreads 证明方法不只适用于新闻点击；
5. 把贡献重点放在 **label semantics、exposure、repeat/explore 与 persistence protocol**，而不是仅替换成更大的生成模型。

在当前证据下，可以说：

> CrossInterest 研究的是公开文献尚未标准化的“历史到未来新语义兴趣集合”预测；已有工业论文证明该方向有线上价值，NNBR/SERAL/UpliftRec 分别提供新 item、短窗新类别和 exposure-aware 的邻近先例，但公开数据仍需要重新构造并严格审计。

不能说：

> 离线日志已经证明模型发现了用户的真实潜在兴趣，或推荐导致了新兴趣形成。

## 12. 调研产物

- 研究框架：`future_novel_interest_research/outline.yaml`
- 33 字段审计定义：`future_novel_interest_research/fields.yaml`
- 九份结构化深调研结果：`future_novel_interest_research/results/`
- 2025–2026 来源纠正：`future_novel_interest_research/results/manual_source_verification_2026-08-19.json`
- 完整字段汇总：`future_novel_interest_research/report.md`
