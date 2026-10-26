# Semantic-ID + Global Cross-Fitted：哪些预测更准

> 状态：完成。本文只分析 `Qwen/Qwen3.5-9B`、Interest-only 输入、Semantic-ID 输出、**全局** Cross-Fitted 权重的正式 real-Dev 结果，并以同一输入、同一 Semantic-ID 标签空间的 Vanilla 为配对对照。类内 rank Cross-Fitted 不属于本文分析对象。所有结论均为 exploratory Dev 上的描述或配对统计，不写成因果机制。

## 1. 研究问题（Research Questions）

本文回答三个问题：

1. Semantic-ID + 全局 Cross-Fitted 在哪些真实目标、History 结构和语义转移上绝对准确率较高？
2. 相对 Semantic-ID Vanilla，改善发生在 Top-1、Top-5、Top-10 还是 Top-20，集中于哪些 case，代价又出现在哪里？
3. 五折 OOF predictive-consistency 权重能否解释这些改善，还是主要刻画了 target 本身的可预测难度？

主比较严格复用 31,392 个 impressions、27,082 个用户和 37,728 个 target occurrences。主要 case 指标为 impression Hit@K、impression Recall@K、target-occurrence Recall@K 与 104 类等权的 Interest-Macro Recall/NDCG@10。

## 2. 训练数据构造（Training Data Construction）

两条件使用同一份 21,600 条 Interest-only SFT 数据：108 个 target interest，每类 200 个互异用户，每个 Train 用户只保留一个 target。History 为按时间排序的 subcategory 名称，最长 50；response 是目标类别对应的单个 `<interest_NNN>` token。Vanilla 对每条样本等权；Global Cross-Fitted 使用五折 OOF teacher 的 `prior_adjusted_nll` 在**每个 fold 的全部 4,320 条样本中**排序：

\[
q_i=\frac{\operatorname{rank}_{\mathrm{fold}}(\mathrm{NLL}_i)}{4319},\quad
\tilde w_i=\frac{1}{1+q_i},\quad
w_i=\frac{\tilde w_i}{\operatorname{mean}(\tilde w)}.
\]

因此权重范围为 `0.7213–1.4427`、全局均值为 1、最大/最小比为 2。`prior_adjusted_margin` 只记录，不进入权重公式。这里的“高权重”只能解释为 OOF teacher 更容易预测，不能等同于已有独立证据支持的 clean label。

### 2.1 高权重真实训练样例

`source_sample_id=train:431374:112`，target 为 `Cars & Driving`，response 为 `<interest_018>`；OOF NLL=`1.5538`，fold rank=`1/4319`，sample weight=`1.4423`。

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Vehicle Buying & Ownership
> 2. Vehicle Buying & Ownership
> 3. Electric & Alternative Vehicles
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出我最近浏览新闻对应的主题类别：
>
> 1. 车辆购买与拥有
> 2. 车辆购买与拥有
> 3. 电动与替代能源汽车
>
> 根据该序列，预测我接下来可能感兴趣、且未出现在上述序列中的一个新主题类别。只输出分配给该类别的已学习兴趣 token，格式为 `<interest_NNN>`。

该样本的“车辆购买/电动车 → 汽车与驾驶”是清楚的同父域相邻子类转移，与 OOF teacher 容易预测相一致。

### 2.2 低权重真实训练样例

`source_sample_id=train:149699:62`，target 为 `Genealogy & Family History`，response 为 `<interest_051>`；OOF NLL=`7.2332`，fold rank=`4318/4319`，sample weight=`0.7214`。

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Vehicle Buying & Ownership
> 2. American Football
> 3. American Football
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出我最近浏览新闻对应的主题类别：
>
> 1. 车辆购买与拥有
> 2. 美式橄榄球
> 3. 美式橄榄球
>
> 根据该序列，预测我接下来可能感兴趣、且未出现在上述序列中的一个新主题类别。只输出分配给该类别的已学习兴趣 token，格式为 `<interest_NNN>`。

输入与“家谱与家族史”没有可见的直接语义连接。这个例子支持“OOF 难度能发现历史与标签不一致的样本”，但单例不能证明标签错误。

## 3. 测试设计（Testing）

### 3.1 测试模型与位置

Cross-Fitted 条件在 AMLT/Feeds `80G4-A100` 上训练：Qwen3.5-9B，Interest-only，3 epochs，2,025 optimizer steps，4-rank DDP，effective global batch 32，seed 42，BF16，learning rate `2e-5`，LoRA `r=16/alpha=32/dropout=0.05`、all-linear，thinking disabled。最终 adapter 已通过保存/重载与 Semantic-ID tensor 审计。

Vanilla 对照是此前已验收的同模型、同 21,600 Train、同 Interest-only prompt 与同 Semantic-ID 初始化方案；唯一核心差别是不传 sample weights。

### 3.2 测试数据来源与完整性

模型实际输入是 `dev_full_grouped.parquet` 中的完整英文 prompt：按时间顺序列出的 subcategory 名称和固定预测指令。原始 Dev 的新闻 titles 只用于本文回查 ground-truth provenance，**不是模型输入**。

原始 Dev 有 37,817 个 target rows。Semantic-ID 的 108 类 mapping 删除了 89 个不在映射内的 target rows；其中 29 个 impressions 仍有其他受支持 target，59 个 impressions 因全部 target 不受支持而整体离开评测，最终形成 31,392 个 grouped impressions、37,728 个受支持 target occurrences、104 个在 Dev 中出现的 Semantic-ID 类别。

本次 case pipeline 逐行验证：

- Weighted、Vanilla 与输入的 `record_index`、`impression_id`、`input_sha256` 和正确答案集合完全一致；
- 每条输出恰有 20 个互异的合法标签、token 与有限 score，且 token/label mapping 精确匹配；
- 输入、两份 prediction、mapping、raw Dev、weights 与 retained-interest 表均记录 SHA-256；
- 31,392 个 impression 与 37,728 个 target occurrence 均进入 case catalog，没有把错误或未命中样本移出分母。

### 3.3 模型“完整输出”的含义

评测不做自由生成或 Beam Search。每条完整 prompt 只做一次 next-token forward，在 108 个 Semantic-ID token 上计算 restricted log-softmax，保存 Top-20 `token + decoded label + log probability`。`restricted_entropy` 也在这 108 个 token 上计算。

评测不硬屏蔽 History 中已出现的类别。实际 Weighted/Vanilla Top-1 重复 History 类别的比例仅为 `0.076%/0.089%`，Top-10 中含任一 History 类别为 `2.86%/2.98%`，但这些仍属于未过滤结果。

### 3.4 评分与 case 抽样

本文不使用 LLM judge。`Hit@K` 表示一个 impression 至少命中一个可接受 target；`Recall@K` 表示命中 target 数占该 impression 全部可接受 target 的比例；target-occurrence Recall 把每个正确 target 单独展开；Interest-Macro 先按 target 求 recall/NDCG，再对 104 个受支持 target 等权平均。

定性 case 先限定 26,749 个单正确答案 impressions，再按 `1–4 / 5–10 / 11+` 三个 History 长度层各选一例：

- Weighted-only / Vanilla-only：选正确答案 rank 变化最大者；
- Both：要求 Weighted rank=1，再选 Vanilla rank 最低者；
- Neither@20：选 Weighted entropy 最低、即最自信但仍错误者。

这给出固定的 12 例，避免只挑容易解释的成功样本。rank=`21` 表示未进入保存的 Top-20。

## 4. 实验结果（Results）

### 4.1 总体：收益从 Top-5 开始，不在 Top-1

| K | Weighted Hit | Vanilla Hit | Δ Hit | Weighted Recall | Vanilla Recall | Δ Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.005% | 7.085% | -0.080pp | 5.965% | 5.993% | -0.028pp |
| 5 | 23.439% | 23.050% | +0.389pp | 20.708% | 20.326% | +0.382pp |
| 10 | 35.971% | 35.197% | +0.774pp | 32.365% | 31.615% | +0.750pp |
| 20 | 52.870% | 51.631% | +1.239pp | 48.723% | 47.472% | +1.251pp |

用户聚类 95% CI：ΔHit@1=`[-0.241,+0.082]pp`，ΔHit@10=`[+0.520,+1.028]pp`，ΔHit@20=`[+0.992,+1.486]pp`。在 37,728 个 target occurrences 上，Recall@10 为 `11,958/37,728=31.695%` 对 `11,676/37,728=30.948%`，差 `+0.747pp`，3,000 次 user-cluster bootstrap CI=`[+0.526,+0.971]pp`。

Interest-Macro Recall@10 从 `18.8753%` 到 `19.3381%`，差 `+0.4628pp`，但 2,000 次 user-cluster bootstrap CI=`[-0.0181,+1.0574]pp`；Macro NDCG@10 差 `+0.2140pp`，CI=`[-0.0060,+0.4641]pp`。所以“常见 occurrence 的 Top-10/20 排序改善”证据较强，而“104 个类别等权后普遍改善”尚未建立。

| K | Both | Weighted-only | Vanilla-only | Neither |
|---:|---:|---:|---:|---:|
| 1 | 1,882 | 317 | 342 | 28,851 |
| 5 | 6,646 | 712 | 590 | 23,444 |
| 10 | 10,356 | 936 | 693 | 19,407 |
| 20 | 15,631 | 966 | 577 | 14,218 |

Weighted mean restricted entropy 为 `4.2884`，Vanilla 为 `4.4407`；在全部 26,749 个单答案 impressions 上 Weighted entropy 都更低，但 Top-1 没有改善。这与“分布变尖，并把一部分正确标签推入 Top-5/10/20 中段”一致，不等于置信度校准更好。

### 4.2 哪些输入上绝对更准

History 越短、包含的唯一兴趣越少，绝对 Hit 越高；但 Cross-Fitted 相对 Vanilla 的差值在长或重复 History 上更大。

| History 长度 | n | Weighted Hit@10 | Vanilla Hit@10 | Δ |
|---|---:|---:|---:|---:|
| 1–4 | 9,407 | 45.91% | 45.38% | +0.53pp |
| 5–10 | 13,571 | 37.13% | 36.42% | +0.71pp |
| 11–20 | 5,890 | 25.28% | 24.06% | +1.22pp |
| 21–50 | 2,524 | 17.63% | 16.68% | +0.95pp |

| History 重复率 | n | Weighted Hit@10 | Vanilla Hit@10 | Δ |
|---|---:|---:|---:|---:|
| ≤0.10 | 11,402 | 42.56% | 42.22% | +0.34pp |
| 0.10–0.30 | 10,071 | 33.35% | 32.59% | +0.76pp |
| 0.30–0.50 | 8,146 | 31.35% | 30.08% | +1.28pp |
| >0.50 | 1,773 | 29.67% | 28.37% | +1.30pp |

使用 `interest_vocabulary.parquet` 的粗粒度 parent 做**事后分析分组**，模型仍只看到 subcategory 文本：

| Target occurrence 条件 | n | Weighted R@10 | Vanilla R@10 | Δ |
|---|---:|---:|---:|---:|
| History 任一兴趣与 target 同 parent | 14,631 | 38.09% | 36.82% | +1.27pp |
| 无同 parent 兴趣 | 23,097 | 27.64% | 27.23% | +0.42pp |
| 最后兴趣与 target 同 parent | 3,398 | 46.97% | 44.17% | +2.80pp |
| 最后兴趣不同 parent | 34,330 | 30.18% | 29.64% | +0.54pp |

最一致的成功模式因此是：History 已出现同一语义父域，但正确答案是 History 中未出现的相邻子类，例如 Football/Baseball/Basketball、Celebrity/Fashion、Elections/Government、Crime/Courts/Policing。它是相关性证据；History 长度、重复度、类别频率与目标可预测性彼此混杂。

### 4.3 哪些 target 更容易，哪些类别从 Cross-Fitted 获益

下表的 recall 以真实 target occurrence 为条件，回答“当正确答案是这个类别时，模型能否把它排进 Top-10”。

| Target | n | Weighted R@10 | Vanilla R@10 | Δ |
|---|---:|---:|---:|---:|
| Celebrity & Creator Culture | 1,818 | 78.11% | 77.28% | +0.83pp |
| Crime & Public Safety | 1,768 | 73.02% | 72.62% | +0.40pp |
| American Football | 5,377 | 63.44% | 63.25% | +0.19pp |
| Courts & Legal Cases | 854 | 52.46% | 50.35% | +2.11pp |
| Government & Public Policy | 1,697 | 45.32% | 44.43% | +0.88pp |
| Fashion & Personal Style | 346 | 37.57% | 34.68% | +2.89pp |
| Cooking & Recipes | 863 | 30.94% | 27.93% | +3.01pp |
| Shopping & Gift Ideas | 3,869 | 21.12% | 18.76% | +2.35pp |
| Music | 1,520 | 18.82% | 16.51% | +2.30pp |
| Policing & Corrections | 2,616 | 9.79% | 7.53% | +2.26pp |

低准确 target 包括 Public Services & Social Welfare `0.52% (n=191)`、Aging & Longevity `3.12% (n=256)`、Careers & Professional Development `3.76% (n=266)`、Weather Forecasts & Seasonal Conditions `5.29% (n=454)`。清楚的退步包括 Accidents & Emergency Response `-5.34pp (n=131)`、Historical Events & Eras `-3.54pp (n=198)`、Parenting & Family Life `-3.07pp (n=391)`、Personal Finance `-2.32pp (n=302)`。类别切片未做多重比较校正，不能按点估计挑选“显著类别”。

### 4.4 当模型输出某类别时，哪些 Top-1 更可信

这与上一节不同：这里以模型的 Top-1 输出为条件，计算 precision。`Lift = precision / 该 target 在 impressions 中的边际 prevalence`。

| Top-1 输出 | 预测次数 | 正确次数 | Precision | Target prevalence | Lift |
|---|---:|---:|---:|---:|---:|
| American Football | 3,160 | 852 | 26.96% | 17.13% | 1.57× |
| Celebrity & Creator Culture | 4,067 | 544 | 13.38% | 5.79% | 2.31× |
| Government & Public Policy | 505 | 63 | 12.48% | 5.41% | 2.31× |
| Crime & Public Safety | 2,905 | 320 | 11.02% | 5.63% | 1.96× |
| Courts & Legal Cases | 1,211 | 71 | 5.86% | 2.72% | 2.16× |

American Football、Celebrity 和 Crime 三类贡献了 Weighted 正确 Top-1 的 78.0%，表明 Top-1 成功高度集中。Shopping 是反例：Top-1 precision 为 18.03%，但只输出 61 次，target Recall@1 仅 0.28%；它的主要改善发生在 Top-10，而不是成为第一名。

另有 1,015 次 Weighted Top-1 落在正式 Dev 从未作为正确答案出现的四个 Semantic-ID 类别：Gardening & Landscaping 675 次、Adventure & Nature Travel 182 次、Religion in Public Life 94 次、Artificial Intelligence & Robotics 64 次，因而必然错误。Gambling & Lotteries 输出 904 次但只正确 1 次；Constitutional & Civil Law 输出 761 次也只正确 1 次。这是后续改进输出先验或候选空间时最直接的错误来源之一。

### 4.5 12 个系统抽样 case

| 类型 | impression | History 长度 | 正确答案 | Weighted rank | Vanilla rank | W/V entropy |
|---|---:|---:|---|---:|---:|---:|
| W-only@10 | 273342 | 3 | Wildlife & Conservation | 7 | >20 | 4.467 / 4.544 |
| W-only@10 | 18212 | 8 | Policing & Corrections | 7 | >20 | 4.324 / 4.482 |
| W-only@10 | 42111 | 12 | Government & Public Policy | 4 | 18 | 4.263 / 4.449 |
| V-only@10 | 207581 | 4 | Vehicle Buying & Ownership | >20 | 8 | 4.322 / 4.494 |
| V-only@10 | 251428 | 6 | Shopping & Gift Ideas | >20 | 8 | 4.474 / 4.557 |
| V-only@10 | 151933 | 25 | American Football | 20 | 6 | 4.299 / 4.438 |
| Both@10, W Top-1 | 334907 | 4 | Baseball & Softball | 1 | 5 | 4.527 / 4.573 |
| Both@10, W Top-1 | 296897 | 10 | Basketball | 1 | 6 | 4.344 / 4.465 |
| Both@10, W Top-1 | 16093 | 11 | Fashion & Personal Style | 1 | 5 | 4.418 / 4.528 |
| Neither@20 | 7384 | 4 | Public Health | >20 | >20 | 2.857 / 3.413 |
| Neither@20 | 179474 | 7 | Crime & Public Safety | >20 | >20 | 1.915 / 2.640 |
| Neither@20 | 356447 | 11 | Civil Rights & Social Justice | >20 | >20 | 2.216 / 3.212 |

完整 12 例的英文 prompt、正确 token/label、两份 Top-20 token/label/score、entropy 与原始 target titles 均保存在 `representative_cases.json`。下面展开四个最有诊断价值的完整 case。

#### Case A：公共事务相邻子类被从 rank 18 推到 rank 4（42111）

完整英文输入：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Sports Business & Administration
> 2. Civic Life & Public Commemorations
> 3. Courts & Legal Cases
> 4. Space & Astronomy
> 5. Elections & Political Campaigns
> 6. Celebrity & Creator Culture
> 7. Trivia & Puzzles
> 8. Journalism & News Media
> 9. Companies & Corporate Strategy
> 10. Public Health
> 11. Historical Events & Eras
> 12. Crime & Public Safety
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出我最近浏览新闻对应的主题类别：体育商业与管理；公民生活与公共纪念；法院与法律案件；太空与天文学；选举与政治竞选；名人与创作者文化；冷知识与谜题；新闻业与新闻媒体；公司与企业战略；公共卫生；历史事件与时代；犯罪与公共安全。根据该序列，预测一个接下来可能感兴趣、且未出现在序列中的新主题。只输出对应的 `<interest_NNN>`。

正确答案：`<interest_054>`（Government & Public Policy）。原始 target title 是 “3 Indiana judges suspended after a night of drinking turned into a White Castle brawl”；该标题仅用于标签回查，不是输入。

| rank | Cross-Fitted label (score) | Vanilla label (score) |
|---:|---|---|
| 1 | Diplomacy & Geopolitics (-2.451) | Diplomacy & Geopolitics (-3.021) |
| 2 | Biology & Genetics (-3.201) | War & Armed Conflict (-3.646) |
| 3 | War & Armed Conflict (-3.451) | Biology & Genetics (-3.646) |
| 4 | **Government & Public Policy (-3.576)** | Internet & Social Platforms (-3.771) |
| 5 | Consumer Technology (-3.576) | Consumer Technology (-3.771) |
| 6 | American Football (-3.576) | Archaeology & Human Origins (-3.771) |
| 7 | Artificial Intelligence & Robotics (-3.576) | American Football (-3.771) |
| 8 | Infrastructure & Transportation Policy (-3.826) | Higher Education (-3.896) |
| 9 | Internet & Social Platforms (-3.826) | Infrastructure & Transportation Policy (-3.896) |
| 10 | Archaeology & Human Origins (-3.826) | Artificial Intelligence & Robotics (-3.896) |
| 11 | Baseball & Softball (-3.951) | Earthquakes & Geological Hazards (-4.021) |
| 12 | Earthquakes & Geological Hazards (-3.951) | Soccer (-4.021) |
| 13 | Higher Education (-3.951) | Accidents & Emergency Response (-4.146) |
| 14 | Weather Forecasts & Seasonal Conditions (-4.076) | Baseball & Softball (-4.146) |
| 15 | Golf (-4.076) | Weather Forecasts & Seasonal Conditions (-4.146) |
| 16 | Religion in Public Life (-4.076) | Golf (-4.146) |
| 17 | Soccer (-4.076) | Religion in Public Life (-4.146) |
| 18 | Accidents & Emergency Response (-4.201) | **Government & Public Policy (-4.146)** |
| 19 | Climate Change (-4.201) | Economy & Economic Policy (-4.146) |
| 20 | Constitutional & Civil Law (-4.201) | Charity & Volunteering (-4.271) |

Elections、Civic Life、Journalism、Courts 和 Crime 形成可见的公共事务/法律语义簇；这与正确 target 被上推一致。不过第 4–7 名完全同分，名义 rank 含 tie-breaking 成分；且 target title 的 taxonomy 边界并不锐利，不能把单例解释为确定的序列推理。

#### Case B：Celebrity-heavy History 下无并列 Top-1（16093）

完整英文输入：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Severe Storms
> 2. Accidents & Emergency Response
> 3. Parenting & Family Life
> 4. Celebrity & Creator Culture
> 5. Cybersecurity & Digital Privacy
> 6. Accidents & Emergency Response
> 7. Celebrity & Creator Culture
> 8. Celebrity & Creator Culture
> 9. Celebrity & Creator Culture
> 10. Celebrity & Creator Culture
> 11. Civic Life & Public Commemorations
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出我最近浏览新闻对应的主题类别：严重风暴；事故与应急响应；育儿与家庭生活；名人与创作者文化；网络安全与数字隐私；事故与应急响应；名人与创作者文化；名人与创作者文化；名人与创作者文化；名人与创作者文化；公民生活与公共纪念。根据该序列，预测一个接下来可能感兴趣、且未出现在序列中的新主题。只输出对应的 `<interest_NNN>`。

正确答案：`<interest_044>`（Fashion & Personal Style）。原始 target title 是 “Fashion hits and misses from the 2019 Country Music Association Awards”，不是模型输入。

| rank | Cross-Fitted label (score) | Vanilla label (score) |
|---:|---|---|
| 1 | **Fashion & Personal Style (-3.310)** | American Football (-3.644) |
| 2 | Pets & Animal Companions (-3.435) | Crime & Public Safety (-3.644) |
| 3 | Relationships & Dating (-3.560) | Pets & Animal Companions (-3.769) |
| 4 | Crime & Public Safety (-3.560) | Visual Arts & Design (-3.769) |
| 5 | Architecture & Residential Design (-3.560) | **Fashion & Personal Style (-3.894)** |
| 6 | American Football (-3.560) | Architecture & Residential Design (-3.894) |
| 7 | Visual Arts & Design (-3.685) | Relationships & Dating (-4.019) |
| 8 | Internet & Social Platforms (-3.685) | Television & Streaming (-4.019) |
| 9 | Film (-3.935) | Film (-4.019) |
| 10 | Television & Streaming (-4.060) | Internet & Social Platforms (-4.019) |
| 11 | Beauty & Grooming (-4.060) | Space & Astronomy (-4.019) |
| 12 | Space & Astronomy (-4.060) | Charity & Volunteering (-4.144) |
| 13 | Parenting & Family Life (-4.060) | Government & Public Policy (-4.144) |
| 14 | Charity & Volunteering (-4.060) | Theater & Performing Arts (-4.269) |
| 15 | Government & Public Policy (-4.060) | Journalism & News Media (-4.269) |
| 16 | Medicine & Medical Research (-4.185) | Sleep Health (-4.269) |
| 17 | Interior Design & Decorating (-4.185) | Elections & Political Campaigns (-4.269) |
| 18 | Nutrition & Healthy Eating (-4.185) | Cars & Driving (-4.269) |
| 19 | Elections & Political Campaigns (-4.310) | Floods, Droughts & Wildfires (-4.269) |
| 20 | Cooking & Recipes (-4.310) | Nutrition & Healthy Eating (-4.269) |

这是 12 例中最稳健的成功 case：正确答案是无并列第一名，且比第二名高 0.125 log-prob。重复 Celebrity → Fashion 与娱乐/名人—时尚共消费一致。仍不能用未输入的 CMA 标题来解释模型为何正确。

#### Case C：强体育线索下从 rank 6 降到 rank 20（151933）

完整英文输入：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Baseball & Softball
> 2. Fashion & Personal Style
> 3. Baseball & Softball
> 4. Baseball & Softball
> 5. Parenting & Family Life
> 6. Celebrity & Creator Culture
> 7. Celebrity & Creator Culture
> 8. Celebrity & Creator Culture
> 9. Destinations & Trip Ideas
> 10. Baseball & Softball
> 11. Baseball & Softball
> 12. Accidents & Emergency Response
> 13. Government & Public Policy
> 14. Baseball & Softball
> 15. Restaurants & Dining
> 16. Celebrity & Creator Culture
> 17. Celebrity & Creator Culture
> 18. Courts & Legal Cases
> 19. Crime & Public Safety
> 20. Baseball & Softball
> 21. Baseball & Softball
> 22. Baseball & Softball
> 23. Celebrity & Creator Culture
> 24. Celebrity & Creator Culture
> 25. Baseball & Softball
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出 25 个最近兴趣：棒球与垒球；时尚与个人风格；棒球与垒球；棒球与垒球；育儿与家庭生活；名人与创作者文化；名人与创作者文化；名人与创作者文化；目的地与旅行灵感；棒球与垒球；棒球与垒球；事故与应急响应；政府与公共政策；棒球与垒球；餐厅与外出就餐；名人与创作者文化；名人与创作者文化；法院与法律案件；犯罪与公共安全；棒球与垒球；棒球与垒球；棒球与垒球；名人与创作者文化；名人与创作者文化；棒球与垒球。根据该序列，预测一个未出现的新主题，并只输出对应的 `<interest_NNN>`。

正确答案：`<interest_005>`（American Football）。原始 target title 是关于 Colin Kaepernick 的评论文章，不是模型输入。

| rank | Cross-Fitted label (score) | Vanilla label (score) |
|---:|---|---|
| 1 | General & Multi-Sport Coverage (-2.912) | Nutrition & Healthy Eating (-3.143) |
| 2 | Nutrition & Healthy Eating (-3.037) | General & Multi-Sport Coverage (-3.518) |
| 3 | Journalism & News Media (-3.412) | Journalism & News Media (-3.643) |
| 4 | Music (-3.537) | Music (-3.768) |
| 5 | Civic Life & Public Commemorations (-3.537) | Civic Life & Public Commemorations (-3.768) |
| 6 | Ice Hockey (-3.662) | **American Football (-3.893)** |
| 7 | Sports Business & Administration (-3.787) | Sexual & Reproductive Health (-3.893) |
| 8 | Relationships & Dating (-3.787) | Ice Hockey (-3.893) |
| 9 | Sexual & Reproductive Health (-3.787) | Travel Planning & Deals (-4.018) |
| 10 | Combat Sports (-3.787) | Shopping & Gift Ideas (-4.018) |
| 11 | Television & Streaming (-3.912) | Pets & Animal Companions (-4.018) |
| 12 | Travel Planning & Deals (-3.912) | Relationships & Dating (-4.018) |
| 13 | Shopping & Gift Ideas (-3.912) | Fitness & Exercise (-4.018) |
| 14 | Pets & Animal Companions (-3.912) | Beauty & Grooming (-4.018) |
| 15 | Fitness & Exercise (-4.037) | Sports Business & Administration (-4.143) |
| 16 | Food Culture & Trends (-4.037) | Olympic & Athletics Sports (-4.143) |
| 17 | Beauty & Grooming (-4.037) | Adventure & Nature Travel (-4.143) |
| 18 | Basketball (-4.037) | Basketball (-4.143) |
| 19 | Books & Literature (-4.162) | Combat Sports (-4.143) |
| 20 | **American Football (-4.162)** | Food Culture & Trends (-4.143) |

Baseball 出现 10 次，Celebrity 出现 7 次，正确答案又是同一 Sports 父域；Cross-Fitted 仍把 American Football 从 6 降到 20。这是“同父域相邻转移普遍受益”的直接反例。

#### Case D：更低 entropy，但对错误公共事务子簇更自信（179474）

完整英文输入：

> The topic categories associated with the news items I recently viewed, in chronological order, are listed below:
>
> 1. Government & Public Policy
> 2. Elections & Political Campaigns
> 3. Policing & Corrections
> 4. Government & Public Policy
> 5. Cybersecurity & Digital Privacy
> 6. Elections & Political Campaigns
> 7. Government & Public Policy
>
> Based on this sequence of topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear in the sequence above. Respond with only the learned interest token assigned to the predicted category, in the form `<interest_NNN>`.

中文翻译：

> 下面按时间顺序列出我最近浏览新闻对应的主题类别：政府与公共政策；选举与政治竞选；警务与矫正；政府与公共政策；网络安全与数字隐私；选举与政治竞选；政府与公共政策。根据该序列，预测一个接下来可能感兴趣、且未出现在序列中的新主题。只输出对应的 `<interest_NNN>`。

正确答案：`<interest_032>`（Crime & Public Safety），两模型均未排入 Top-20。原始 target title 是一则失踪人口新闻，不是模型输入。

| rank | Cross-Fitted label (score) | Vanilla label (score) |
|---:|---|---|
| 1 | Constitutional & Civil Law (-0.475) | Constitutional & Civil Law (-0.703) |
| 2 | Consumer Technology (-2.225) | Consumer Technology (-2.578) |
| 3 | Internet & Social Platforms (-3.475) | Artificial Intelligence & Robotics (-3.328) |
| 4 | Artificial Intelligence & Robotics (-3.850) | Internet & Social Platforms (-3.453) |
| 5 | Telecommunications (-3.975) | Telecommunications (-3.828) |
| 6 | Journalism & News Media (-4.100) | Journalism & News Media (-3.828) |
| 7 | Civil Rights & Social Justice (-4.350) | Civil Rights & Social Justice (-3.953) |
| 8 | Diplomacy & Geopolitics (-4.600) | Diplomacy & Geopolitics (-4.328) |
| 9 | Economy & Economic Policy (-4.850) | Economy & Economic Policy (-4.453) |
| 10 | War & Armed Conflict (-4.975) | War & Armed Conflict (-4.453) |
| 11 | Religion in Public Life (-4.975) | Public Services & Social Welfare (-4.578) |
| 12 | Courts & Legal Cases (-5.100) | Climate Change (-4.703) |
| 13 | Climate Change (-5.100) | Courts & Legal Cases (-4.703) |
| 14 | Public Services & Social Welfare (-5.100) | American Football (-4.828) |
| 15 | Biology & Genetics (-5.225) | Energy & Sustainability (-4.828) |
| 16 | Energy & Sustainability (-5.225) | Religion in Public Life (-4.828) |
| 17 | American Football (-5.350) | Personal Finance (-4.953) |
| 18 | Baseball & Softball (-5.475) | Biology & Genetics (-4.953) |
| 19 | Books & Literature (-5.475) | Books & Literature (-5.078) |
| 20 | Higher Education (-5.475) | Baseball & Softball (-5.078) |

Weighted entropy=`1.915`，Vanilla=`2.640`，但正确答案都在 Top-20 外。Policing 与 Crime 的语义邻近并未被恢复，重复 Government/Elections 反而让模型坍缩到 Constitutional & Civil Law。更尖锐的分布可能放大错误子簇，而不是可靠地提高置信度。

### 4.6 Cross-Fitted 权重究竟解释了什么

行级上，sample weight 与 OOF NLL 的 Spearman `ρ=-0.99984`，这几乎由公式决定；与 margin 的 `ρ=-0.861` 是间接相关。OOF target 易度对**绝对** Dev 难度有强预测力：target 平均权重与 Vanilla R@10 的 Spearman `ρ=.668`，与 Weighted R@10 为 `.717`；平均 NLL 与 Weighted R@10 为 `-.715`。平均权重与 raw source Train count 的相关仅 `ρ=-.041`，与 Dev support 为 `.0068`。

但它对 **Weighted−Vanilla 增益** 的解释弱得多：target 平均权重与 ΔR@10 的 Spearman `.185 (p=.059)`；按 Dev support 加权后的 Pearson `r=.015`，bootstrap CI 跨 0。最高权重四分位的 Macro ΔR@10 为 `+1.643pp`，最低四分位为 `+0.148pp`，但最高组受 `n=6/8` 的极小类放大；高权重 Accidents 仍退步 `-5.34pp`，低权重 Public Health 却改善 `+2.21pp`。

一个事后探索的 history-alignment 指标更接近 case 层机制：在同 target 的 200 个 Train histories 中，用 Jaccard similarity 加权其 target 内中心化 weight；与被上调 Train history 最相似的 Dev 四分位 ΔR@10=`+1.686pp`，最低四分位=`+0.276pp`，target fixed-effect + user-cluster SE 的 slope `p=.00021`。这是目前最直接的“相似高权重训练 history 的 case 改善更多”证据，但该指标是看完结果后构造的，必须在独立 split 预注册复验后才能升级为机制主张。

综合以上证据，当前可支持的表述是：

- 方法在短、少类别的 History 上**绝对更准**；
- 相对 Vanilla 的增益主要出现在 Top-5/10/20，尤其是长、重复或存在同父域相邻子类线索的 History；
- OOF 权重首先是一个很强的 target 可预测难度指标，并会改变中段排序；
- 它没有稳定改善 Top-1，也不保证每个同父域转移都改善；
- 不能把权重直接命名为“去噪标签”或把相关性写成 Cross-Fitted 学会了因果兴趣转移。

## 5. 数据与产物路径（Artifact Paths）

- 本报告：[semantic_id_global_crossfit_case_analysis.md](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-28/semantic_id_global_crossfit_case_analysis.md)
- 可复算分析模块：[analyze_semantic_id_crossfit_cases.py](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/experiments/analyze_semantic_id_crossfit_cases.py)
- 完整模型输入：[dev_full_grouped.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1/eval/interest_only/dev_full_grouped.parquet)
- Cross-Fitted 完整 prediction：[real_dev_id_token_predictions.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/static-sud-semantic-id-global-crossfit/qwen35-interest/final/static-sud-semantic-id-global-crossfit-qwen-interest-final-real-v2-0825/qwen35-interest-semantic-id-global-crossfit-final/real_dev_id_token_predictions.parquet)
- Vanilla 完整 prediction：[id_token_predictions.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/static-sud-semantic-id-token/static-semantic-id-token-qwen-interest-eval-recovery-0820-v1/qwen35-interest-semantic-id-eval-recovery/recovery/eval/merged/id_token_predictions.parquet)
- 训练输入：[21600.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1/train/interest_only/21600.parquet)
- 全局 Cross-Fitted 权重：[sample_weights.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_v1/sample_weights.parquet)
- Semantic-ID mapping：[interest_token_mapping.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1/interest_token_mapping.parquet)
- 原始 Dev（titles 仅用于 provenance）：[dev.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest_static_user_disjoint_v1/dev.parquet)
- Case analysis 根目录：[case-study-20260828](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/static-sud-semantic-id-global-crossfit/qwen35-interest/analysis/case-study-20260828)
  - `analysis_summary.json`：身份、计数、transition 与 source/output SHA-256；
  - `case_catalog.parquet`：31,392 条完整 prompt、GT、两套 Top-20/score/entropy 与 History 特征；
  - `target_occurrences.parquet`：37,728 个展开 target occurrences；
  - `slice_metrics.csv`：History 长度、唯一兴趣、重复率、答案数、parent 切片；
  - `target_metrics.csv`：104 个 target 的 R@1/5/10/20、权重与 OOF 难度；
  - `top1_prediction_metrics.csv`：按模型 Top-1 输出计算的 precision/prevalence/lift；
  - `history_target_pair_metrics.csv`：History-target 子类共现切片；
  - `representative_cases.json`：12 个系统抽样 case 的完整输入、正确答案、两套完整 Top-20 输出和 target titles。
- 已有 2,000 次 paired bootstrap：[interest_macro_bootstrap_2000.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/static-sud-semantic-id-global-crossfit/qwen35-interest/final/static-sud-semantic-id-global-crossfit-qwen-interest-final-real-v2-0825/qwen35-interest-semantic-id-global-crossfit-final/analysis/interest_macro_bootstrap_2000.json)
- AMLT final experiment：`static-sud-semantic-id-global-crossfit-qwen-interest-final-real-v2-0825`；job `qwen35-interest-semantic-id-global-crossfit-final`。
