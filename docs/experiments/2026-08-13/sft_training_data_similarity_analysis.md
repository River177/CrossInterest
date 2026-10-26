# SFT 训练数据：标题、类别名称语义与一级类别关系分析

## 1. 研究问题（Research Questions）

本报告只分析当前实际使用的 balanced SFT22K Train 与 natural grouped Dev10K，分别回答：

1. 正样本标题与模型可见 H50 history 的语义相似度有多高；
2. 正样本的二级类别名称与一级类别名称是否在语义上接近 history 中的类别名称；
3. 正样本是否比同一 impression 的 exposed-unclicked 新闻更接近 history；
4. exact novel subcategory 不在 history 时，target parent 是否仍已出现在 full H50、recent-5 或 last click；
5. history breadth、exposure difficulty、prompt overlap 与 taxonomy × semantic quadrant 如何分布。

主比较使用 history-novel `-0` controls；secondary 比较使用全部 `-0` controls。主指标为 nearest-H50 cosine delta、within-impression AUC、Hit@1 与 same-parent fraction。

## 2. 训练数据构造（Training Data Construction）

### 2.1 数据来源与最终规模

Train 使用 balanced SFT22K；Dev 使用 natural grouped Dev10K。每行按 SFT row 等权，history 使用最多最近 50 条并保留顺序与重复。

| split | SFT rows | positive titles | all `-0` controls | novel `-0` controls | primary rows | excluded (no novel control) |
|---|---:|---:|---:|---:|---:|---:|
| train | 22,000 | 22,151 | 1,047,208 | 688,268 | 21,743 | 257 |
| dev | 10,000 | 12,346 | 375,636 | 263,798 | 9,862 | 138 |

Primary control 的 subcategory 必须不在该 row 的 H50 中；没有 novel control 的 row 显式排除，不回退到其他负例。同一 raw candidate token 的重复会保留并计入 denominator。

### 2.2 真实训练样本

- **Provenance:** `train:131920:105`；impression `131920`；source `train:131920:105`；label `Air & Rail Travel`。
- **Record type:** behavior-only target record（不是 control 或 held-out response）。

**Original prompt**

```text
The headlines of the news items I recently viewed are listed below:

1. Former President Jimmy Carter suffers pelvic fracture after falling in Georgia home
2. McDonald's apologizes for 'Sundae Bloody Sundae' Halloween promotion
3. Restaurants Are Dealing With an Avalanche of Special Requests. One Chef Has Had Enough.
4. Meghan McCain confronts Trump Jr.: 'You and your family have hurt a lot of people'

Based on these headlines, predict a new topic category that I am likely to be interested in next. The predicted category should be different from the topics reflected in my viewing history. Begin your response with "New topic: " and provide only the category name after it.
```

**中文翻译**

```text
下面列出了我最近浏览的新闻标题：

1. 前总统吉米·卡特在佐治亚州家中摔倒后骨盆骨折
2. 麦当劳为“血腥星期日圣代”万圣节促销活动道歉
3. 餐馆正面临雪崩般涌来的特殊要求。一位厨师已经受够了。
4. 梅根·麦凯恩质问小特朗普：“你和你的家人伤害了很多人”

根据这些标题，预测一个我接下来可能感兴趣的新主题类别。预测类别应不同于我的浏览历史所反映的主题。回答须以“新主题：”开头，其后只提供类别名称。
```

**Target response**

```text
New topic: Air & Rail Travel
```

**中文翻译**

```text
新主题：航空与铁路旅行
```

## 3. 测试设计（Testing）

### 3.1 Test model/location

本分析不运行 SFT 生成模型，也没有远程推理 job。唯一模型是冻结的标题 encoder `sentence-transformers/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，384 维；每条标题 L2 normalize 后只用于离线 cosine 特征。

### 3.2 Test-data provenance

Train row 由 `source_sample_id` 回连 processed Train；Dev row 由 `impression_id + acceptable_subcategories` 回连 processed Dev。两者再回连 raw MIND impression，以 label `0` 构造唯一对照集合。Train 与 Dev 始终分别统计，不合并成总体比例。

### 3.3 Test examples

代表性真实 record 为 `train:131920:105`：history 末条标题 “Meghan McCain confronts Trump Jr.: 'You and your family have hurt a lot of people'”（梅根·麦凯恩质问小特朗普：“你和你的家人伤害了很多人”）；正样本 “Flight Attendants Have a Secret Language You Didn't Know About”（空乘人员有一套你不知道的秘密语言）。完整 provenance 与 control 排名见 Results case study。

### 3.4 Judge procedure

本分析没有使用 **LLM judge**，因此不存在 judge system prompt、rubric template 或 JSON judge schema。全部字段由确定性公式产生：

- `nearest_50`: target/control 与 H50 标题 cosine 的最大值；
- `top3_mean`: 最大的 `min(3, |H|)` 个 cosine 的均值；
- `centroid`: 与 normalized history centroid 的 cosine；
- `AUC`: 随机 positive 分数高于随机 control 的概率，tie 计 `0.5`；
- `Hit@1`: positive 最大值高于 control 最大值计 `1`，tie 计 `0.5`；
- `same_parent_fraction`: row 内 target parent 已在对应 history scope 出现的 target 比例。
- 类别名称相似度使用同一个冻结 encoder：二级类别比较 `target_subcategories` 与 history/control subcategory 名称；一级类别比较 `target_categories` 与 history/control parent 名称。history 重复点击与 control token 重复都保留。

均值与比例的 95% CI 使用 2,000 次 row bootstrap（seed `20260813`）；taxonomy null 使用 2,000 次 split 内 target-set permutation（seed `20260813`），保留每行 target-set size 与 target marginal。

## 4. 实验结果（Results）

### 4.1 标题语义相似度

| split | score | n | positive | novel control | delta (95% CI) | AUC (95% CI) | Hit@1 (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| train | nearest_50 | 21,743 | 0.2385 | 0.2075 | 0.0310 [0.0297, 0.0323] | 0.5960 [0.5917, 0.6001] | 0.1341 [0.1295, 0.1386] |
| train | top3_mean | 21,743 | 0.1886 | 0.1630 | 0.0256 [0.0246, 0.0267] | 0.6041 [0.6001, 0.6084] | 0.1411 [0.1363, 0.1458] |
| train | centroid | 21,743 | 0.1721 | 0.1357 | 0.0364 [0.0349, 0.0378] | 0.6006 [0.5963, 0.6047] | 0.1371 [0.1325, 0.1417] |
| train | nearest_5 | 21,743 | 0.1702 | 0.1461 | 0.0241 [0.0228, 0.0254] | 0.5756 [0.5713, 0.5801] | 0.1218 [0.1176, 0.1263] |
| train | last | 21,743 | 0.0683 | 0.0539 | 0.0145 [0.0131, 0.0158] | 0.5413 [0.5369, 0.5456] | 0.1067 [0.1025, 0.1108] |
| dev | nearest_50 | 9,862 | 0.2202 | 0.1970 | 0.0232 [0.0214, 0.0249] | 0.5775 [0.5710, 0.5839] | 0.1658 [0.1586, 0.1728] |
| dev | top3_mean | 9,862 | 0.1718 | 0.1518 | 0.0200 [0.0186, 0.0214] | 0.5855 [0.5793, 0.5917] | 0.1750 [0.1678, 0.1828] |
| dev | centroid | 9,862 | 0.1626 | 0.1321 | 0.0304 [0.0284, 0.0326] | 0.5860 [0.5796, 0.5925] | 0.1722 [0.1644, 0.1795] |
| dev | nearest_5 | 9,862 | 0.1649 | 0.1452 | 0.0196 [0.0178, 0.0215] | 0.5648 [0.5587, 0.5714] | 0.1600 [0.1529, 0.1674] |
| dev | last | 9,862 | 0.0684 | 0.0553 | 0.0131 [0.0113, 0.0151] | 0.5404 [0.5345, 0.5470] | 0.1442 [0.1373, 0.1515] |

**Nearest-H50：primary 与 secondary control 口径对照**

| split | scope | n | delta | AUC | Hit@1 | positive ≤ control mean | positive ≤ control max |
|---|---|---:|---:|---:|---:|---:|---:|
| train | novel | 21,743 | 0.0310 | 0.5960 | 0.1341 | 0.4174 | 0.8663 |
| train | all | 22,000 | 0.0058 | 0.5246 | 0.0645 | 0.5317 | 0.9358 |
| dev | novel | 9,862 | 0.0232 | 0.5775 | 0.1658 | 0.4270 | 0.8506 |
| dev | all | 10,000 | 0.0032 | 0.5144 | 0.0992 | 0.5274 | 0.9085 |

Observed result：正样本相对 novel controls 有稳定但不大的平均优势（Train delta=0.0310，Dev delta=0.0232），AUC 分别为 0.5960 / 0.5775。但 86.63% 的 Train rows 与 85.06% 的 Dev rows 中，positive row mean 仍不高于该 impression 最相似的 novel control。加入所有 `-0` 后，AUC 更接近 0.5，说明“标题接近 history”只提供弱排序信号，而不是充分判据。

### 4.2 类别名称语义相似度

这里把类别名称本身送入同一 MiniLM encoder。`subcategory` 衡量二级类别名称，`parent` 衡量一级类别名称；与标题指标一样，每个 SFT row 等权。

| split | name level | scope | n | positive | control | delta (95% CI) | AUC | Hit@1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| train | subcategory | novel | 21,743 | 0.3577 | 0.3404 | 0.0173 [0.0156, 0.0189] | 0.5423 | 0.0970 |
| train | subcategory | all | 22,000 | 0.3579 | 0.5506 | -0.1928 [-0.1951, -0.1905] | 0.3627 | 0.0214 |
| train | parent | novel | 21,743 | 0.7094 | 0.6844 | 0.0251 [0.0212, 0.0289] | 0.4992 | 0.0633 |
| train | parent | all | 22,000 | 0.7090 | 0.7694 | -0.0603 [-0.0641, -0.0565] | 0.4276 | 0.0375 |
| dev | subcategory | novel | 9,862 | 0.3580 | 0.3371 | 0.0209 [0.0182, 0.0238] | 0.5400 | 0.1475 |
| dev | subcategory | all | 10,000 | 0.3581 | 0.5168 | -0.1587 [-0.1623, -0.1553] | 0.3849 | 0.0399 |
| dev | parent | novel | 9,862 | 0.6949 | 0.6491 | 0.0459 [0.0400, 0.0516] | 0.5333 | 0.1090 |
| dev | parent | all | 10,000 | 0.6958 | 0.7313 | -0.0355 [-0.0412, -0.0301] | 0.4660 | 0.0769 |

Observed result：二级类别名称相对 novel controls 的 nearest-H50 delta 为 Train 0.0173、Dev 0.0209；一级类别名称为 Train 0.0251、Dev 0.0459。一级类别名称指标会同时包含 exact same-parent（同名向量 cosine=1）与不同 parent 名称之间的语义接近，因此必须与下一节的 exact same-parent rate 一起解释。二级类别的 all-control delta 显著为负，是因为 all controls 允许其 subcategory 已在 H50 中出现，此时 nearest cosine 可直接达到 1；这不是与 novel-control 口径可互换的结果。当前 primary control 只保证 subcategory novel，并不保证 parent novel，所以parent-name AUC 接近 0.5 仍是合理的独立观察。

### 4.3 一级类别关系与 permutation baseline

| split | history scope | n | observed same-parent | 95% CI | permutation expected | observed − expected | lift |
|---|---|---:|---:|---:|---:|---:|---:|
| train | full | 22,000 | 0.5191 | [0.5125, 0.5258] | 0.4666 | 0.0525 | 1.1124 |
| train | recent5 | 22,000 | 0.2778 | [0.2719, 0.2839] | 0.2310 | 0.0468 | 1.2026 |
| train | last | 22,000 | 0.0835 | [0.0799, 0.0873] | 0.0669 | 0.0167 | 1.2493 |
| dev | full | 10,000 | 0.4999 | [0.4903, 0.5096] | 0.5240 | -0.0241 | 0.9541 |
| dev | recent5 | 10,000 | 0.2773 | [0.2692, 0.2859] | 0.3054 | -0.0281 | 0.9080 |
| dev | last | 10,000 | 0.0833 | [0.0782, 0.0885] | 0.0958 | -0.0125 | 0.8699 |

Raw same-parent rate 在 Train/Dev 都约为一半，但 permutation 改变了其含义：balanced Train 高于保持 target marginal 的 null（+0.0525），natural Dev 反而低于 null（-0.0241）。因此“约 50% 与历史同一级类别”主要受 history coverage 与 target 分布共同影响，不能单独作为强兴趣延展证据。recent-5 约 0.277，last click 仅约 0.083，说明一级类别关系更多来自较宽的累计history，而不是紧邻的最后一次点击。

### 4.4 Prompt ambiguity 与 Train/Dev overlap

| task | Train duplicate excess | Train conflicting prompts | Dev exact-prompt overlap rows | any target seen | all targets seen |
|---|---:|---:|---:|---:|---:|
| behavior | 573 | 525 | 407 | 15 | 7 |
| interest | 973 | 641 | 733 | 95 | 70 |

Train/Dev 共有 332 个 user，覆盖 339 个 Dev rows（3.39%）。 exact prompt overlap 并不等于答案泄漏：Behavior 中仅 15/407 个 overlap rows 至少有一个 Dev target 已在 Train 同 prompt 出现；Interest 为 95/733。另一方面，Train 内存在数百个 exact-prompt conflicts，说明相同模型输入仍可能对应多个 future-interest labels；这是监督信号本身的 one-to-many ambiguity，而不是简单的 user overlap。

### 4.5 Taxonomy × semantic evidence

| split | quadrant | rows | primary delta | primary AUC | primary Hit@1 |
|---|---|---:|---:|---:|---:|
| train | cross_parent__higher_than_control | 5,423 | 0.0837 | 0.8150 | 0.2172 |
| train | cross_parent__not_higher_than_control | 5,019 | -0.0561 | 0.2624 | 0.0000 |
| train | same_parent__higher_than_control | 7,245 | 0.0964 | 0.8316 | 0.2399 |
| train | same_parent__not_higher_than_control | 4,056 | -0.0485 | 0.2950 | 0.0000 |
| dev | cross_parent__higher_than_control | 2,199 | 0.0720 | 0.7985 | 0.2524 |
| dev | cross_parent__not_higher_than_control | 2,371 | -0.0595 | 0.2446 | 0.0000 |
| dev | mixed_parent__higher_than_control | 486 | 0.0574 | 0.6938 | 0.2366 |
| dev | mixed_parent__not_higher_than_control | 272 | -0.0337 | 0.3760 | 0.0074 |
| dev | same_parent__higher_than_control | 2,966 | 0.0916 | 0.8382 | 0.3243 |
| dev | same_parent__not_higher_than_control | 1,568 | -0.0503 | 0.2768 | 0.0006 |
train: 12,668/21,743 (58.26%) primary rows 的 positive similarity 高于 novel-control mean；其余 rows 即使 taxonomy 同 parent，标题证据也可能较弱。
dev: 5,651/9,862 (57.30%) primary rows 的 positive similarity 高于 novel-control mean；其余 rows 即使 taxonomy 同 parent，标题证据也可能较弱。

### 4.6 History breadth 与 exposure difficulty

**History length**

| split | H length | total n | primary n | same-parent (all rows) | primary delta | primary AUC | primary Hit@1 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 01-10 | 7,398 | 7,346 | 0.3151 | 0.0265 | 0.5791 | 0.1290 |
| train | 11-25 | 6,157 | 6,065 | 0.5358 | 0.0337 | 0.6019 | 0.1388 |
| train | 26-49 | 4,286 | 4,221 | 0.6678 | 0.0377 | 0.6145 | 0.1490 |
| train | 50 | 4,159 | 4,111 | 0.7040 | 0.0281 | 0.5984 | 0.1209 |
| dev | 01-10 | 4,197 | 4,164 | 0.3223 | 0.0188 | 0.5649 | 0.1628 |
| dev | 11-25 | 2,796 | 2,755 | 0.5519 | 0.0254 | 0.5814 | 0.1713 |
| dev | 26-49 | 1,664 | 1,631 | 0.6912 | 0.0292 | 0.6006 | 0.1754 |
| dev | 50 | 1,343 | 1,312 | 0.7100 | 0.0249 | 0.5809 | 0.1517 |

Same-parent rate 随 history 变长明显上升，但 semantic delta 在中等长度后并不继续单调增加；这支持把 taxonomy overlap 与真正的标题证据分开衡量。

**Exposure candidate count**

| split | all controls | total n | primary n | primary delta | primary AUC | primary Hit@1 |
|---|---|---:|---:|---:|---:|---:|
| train | 01-20 | 6,965 | 6,708 | 0.0275 | 0.5840 | 0.2661 |
| train | 21-40 | 5,360 | 5,360 | 0.0364 | 0.6116 | 0.1183 |
| train | 41-60 | 3,246 | 3,246 | 0.0293 | 0.5924 | 0.0635 |
| train | 61+ | 6,429 | 6,429 | 0.0310 | 0.5972 | 0.0453 |
| dev | 01-20 | 4,712 | 4,574 | 0.0195 | 0.5693 | 0.2669 |
| dev | 21-40 | 2,069 | 2,069 | 0.0227 | 0.5749 | 0.1068 |
| dev | 41-60 | 1,124 | 1,124 | 0.0235 | 0.5777 | 0.0632 |
| dev | 61+ | 2,095 | 2,095 | 0.0315 | 0.5980 | 0.0582 |

Hit@1 对候选数极其敏感：候选越多，positive 成为唯一最相似标题的概率快速下降；AUC 与 mean delta 相对稳定，因此不同 exposure size 之间不应只比较 Hit@1。

### 4.7 Target heterogeneity

以下仅展示 support ≥ 50 的 target subcategory 两端样本；完整 observed slices（Train 110 类，Dev 101 类）见 `slice_metrics.csv`。

| split | end | target subcategory | rows | same-parent | primary delta | primary AUC |
|---|---|---|---:|---:|---:|---:|
| train | lowest | Space & Astronomy | 200 | 0.4800 | -0.0321 | 0.3724 |
| train | lowest | Pets & Animal Companions | 200 | 0.5150 | -0.0228 | 0.4206 |
| train | lowest | Consumer Technology | 200 | 0.3800 | -0.0208 | 0.4137 |
| train | highest | Gambling & Lotteries | 200 | 0.5350 | 0.2417 | 0.8441 |
| train | highest | Constitutional & Civil Law | 200 | 0.7800 | 0.1282 | 0.8321 |
| train | highest | Journalism & News Media | 200 | 0.0000 | 0.1088 | 0.8016 |
| dev | lowest | Air & Rail Travel | 69 | 0.2983 | -0.0425 | 0.3190 |
| dev | lowest | Archaeology & Human Origins | 162 | 0.4987 | -0.0422 | 0.3382 |
| dev | lowest | Medicine & Medical Research | 188 | 0.4835 | -0.0261 | 0.3844 |
| dev | highest | Journalism & News Media | 75 | 0.0833 | 0.0882 | 0.7877 |
| dev | highest | Public Services & Social Welfare | 118 | 0.6765 | 0.0873 | 0.7942 |
| dev | highest | Sexual & Reproductive Health | 103 | 0.5004 | 0.0834 | 0.7305 |

不同 target 的可预测性差异很大：例如 Dev 的 `Air & Rail Travel` 与 `Archaeology & Human Origins` 呈负 delta，而 `Journalism & News Media`、`Public Services & Social Welfare` 呈较强正 delta。该差异是按 target 切片的观察值，不代表类别本身造成模型难度。

### 4.8 Parent transition 与真实 case studies

Parent transition matrix 共包含 739 个 split-specific 非零 `(split, last parent, target parent)` cells，对应合并 split 标签后的 391 个不同 `(last parent, target parent)` pairs；完整计数见 CSV。

#### cross_parent__higher_than_control — `dev:74288`

- **Split / provenance:** `dev`；impression `74288`；source `dev:74288:81`。
- **Target taxonomy:** parent `Sports`；subcategory `American Football`。
- **Metrics:** positive=0.2320, novel-control=0.1764, delta=0.0555, AUC=0.8333, Hit@1=1.0000。

**最近 history 标题（英文 / 中文）**

- `N34374` Watch: 'The View' Versus Donald Trump Jr.: Loud, Low Blows, Politics, Scandals And Great TV  
  中文：视频：《观点》节目对阵小唐纳德·特朗普：喧闹、下三路攻击、政治、丑闻与精彩电视
- `N34374` Watch: 'The View' Versus Donald Trump Jr.: Loud, Low Blows, Politics, Scandals And Great TV  
  中文：视频：《观点》节目对阵小唐纳德·特朗普：喧闹、下三路攻击、政治、丑闻与精彩电视
- `N37395` Chicken recall: More than 2 million pounds recalled, may be contaminated with metal  
  中文：鸡肉召回：超过200万磅产品被召回，可能受到金属污染
- `N80603` A 911 supervisor was streaming Netflix at work when dispatchers mishandled a shooting call  
  中文：调度员错误处理枪击报警时，一名911主管正在上班看Netflix
- `N113908` Beshear vs. Bevin: Kentucky governor's race could be decided by state legislature  
  中文：贝希尔对阵贝文：肯塔基州州长竞选可能由州议会决定

**正样本（英文 / 中文）**

- `N29160` Opinion: Colin Kaepernick is about to get what he deserves: a chance  
  中文：评论：科林·卡佩尼克即将得到他应得的东西——一次机会  
  nearest history=`N10766`, cosine=0.2572, novel-control rank=1
- `N26703` Some believe Mason Rudolph, hit in head with his own helmet, isn't getting enough blame  
  中文：有人认为，被自己的头盔击中头部的梅森·鲁道夫没有受到足够的指责  
  nearest history=`N36246`, cosine=0.2067, novel-control rank=2

**最相似 novel exposed-unclicked controls**

- `N129416` Taylor Swift Rep Hits Back at Big Machine, Claims She's Actually Owed $7.9 Million in Unpaid Royalties  
  中文：泰勒·斯威夫特的代表反击大机器唱片公司，称对方实际拖欠她790万美元版税；cosine=0.2161
- `N91737` 13 Reasons Why's Christian Navarro Slams Disney for Casting 'the White Guy' in The Little Mermaid  
  中文：《十三个原因》的克里斯蒂安·纳瓦罗抨击迪士尼在《小美人鱼》中选用“白人男演员”；cosine=0.1980
- `N2110` 66 Cool Tech Gifts Anyone Would Be Thrilled to Receive  
  中文：66件任何人收到都会兴奋的酷炫科技礼物；cosine=0.1152

#### cross_parent__not_higher_than_control — `dev:205484`

- **Split / provenance:** `dev`；impression `205484`；source `dev:205484:81`。
- **Target taxonomy:** parent `Sports`；subcategory `American Football`。
- **Metrics:** positive=0.1770, novel-control=0.2285, delta=-0.0514, AUC=0.5263, Hit@1=0.0000。

**最近 history 标题（英文 / 中文）**

- `N90686` Woman Spots Deadly Animal Hiding In Photo Of Her Kids  
  中文：女子在孩子们的照片中发现藏着一种致命动物
- `N41143` 10 of the best fast-food burger chains across the US  
  中文：全美10家最佳快餐汉堡连锁店
- `N63476` A 13-year-old murder suspect who escaped in North Carolina is back in custody  
  中文：一名在北卡罗来纳州逃脱的13岁谋杀案嫌疑人已重新被拘押
- `N560` The one restaurant you need to visit in every state  
  中文：美国每个州都值得去的一家餐厅
- `N127001` Far-reaching snowstorm may take shape over US  
  中文：一场影响范围广泛的暴风雪可能在美国形成

**正样本（英文 / 中文）**

- `N122640` This was uglier than a brawl. And Myles Garrett deserves suspension for rest of year after helmet attack.  
  中文：这比一场斗殴还难看。迈尔斯·加勒特用头盔袭人后理应被禁赛至本赛季结束。  
  nearest history=`N84249`, cosine=0.1770, novel-control rank=10

**最相似 novel exposed-unclicked controls**

- `N54368` South Carolina teen gets life in prison for deadly elementary school shooting  
  中文：南卡罗来纳州少年因致命小学枪击案被判终身监禁；cosine=0.5305
- `N122944` The Real Reason McDonald's Keeps the Filet-O-Fish on Their Menu  
  中文：麦当劳一直保留麦香鱼汉堡的真正原因；cosine=0.4232
- `N18190` 30 Best Black Friday Deals from Costco  
  中文：Costco黑色星期五30项最佳优惠；cosine=0.4189

#### same_parent__higher_than_control — `dev:13741`

- **Split / provenance:** `dev`；impression `13741`；source `dev:13741:12`。
- **Target taxonomy:** parent `Business & Economy`；subcategory `Companies & Corporate Strategy`。
- **Metrics:** positive=0.2577, novel-control=0.1825, delta=0.0752, AUC=0.8462, Hit@1=0.0000。

**最近 history 标题（英文 / 中文）**

- `N17105` Riding The Completely-New 2020 Indian Motorcycle Challenger  
  中文：试骑全新2020款印第安挑战者摩托车
- `N21239` Obama calls out call-out culture: "That's not activism"  
  中文：奥巴马批评公开声讨文化：“那不是行动主义”
- `N80945` States With the Most Farm Bankruptcies  
  中文：农场破产数量最多的州
- `N88466` NASA's Curiosity rover took an eerie photo of the Mars horizon  
  中文：NASA“好奇号”火星车拍下了一张诡异的火星地平线照片
- `N90472` I Took the 30-Day Squat Challenge and Here's What Happened  
  中文：我参加了30天深蹲挑战，结果是这样的

**正样本（英文 / 中文）**

- `N74401` How Amazon's quest for more, cheaper products has resulted in a flea market of fakes  
  中文：亚马逊追求更多、更便宜的商品，如何催生出一个假货跳蚤市场  
  nearest history=`N92655`, cosine=0.2577, novel-control rank=3

**最相似 novel exposed-unclicked controls**

- `N67937` Powerful side-by-side portraits show people over 100 years old next to their younger selves  
  中文：震撼的并列肖像展示百岁老人和年轻时的自己；cosine=0.2983
- `N82503` FDA issues warning to Dollar Tree about selling 'potentially unsafe drugs'  
  中文：FDA警告Dollar Tree销售“可能不安全的药品”；cosine=0.2869
- `N68624` Chuck E. Cheese just got major makeover   and axed the animal band  
  中文：Chuck E. Cheese完成大规模改造，并取消了动物乐队；cosine=0.2557

#### same_parent__not_higher_than_control — `dev:158600`

- **Split / provenance:** `dev`；impression `158600`；source `dev:158600:22`。
- **Target taxonomy:** parent `Crime, Law & Justice`；subcategory `Policing & Corrections`。
- **Metrics:** positive=0.2318, novel-control=0.2733, delta=-0.0415, AUC=0.0000, Hit@1=0.0000。

**最近 history 标题（英文 / 中文）**

- `N11631` Property Brothers' J.D. Scott Marries Annalee Belle in Vintage Theatre-Themed Wedding  
  中文：《房产兄弟》的J.D. Scott在复古剧院主题婚礼上迎娶Annalee Belle
- `N128483` 1 dead, 8 others hospitalized in Salmonella outbreak likely tied to ground beef  
  中文：疑似与碎牛肉有关的沙门氏菌疫情造成1人死亡、另有8人住院
- `N58200` Pamela Anderson gets backlash after wearing a Native American headdress for Halloween  
  中文：帕梅拉·安德森万圣节佩戴美洲原住民头饰后遭到批评
- `N18720` Bette Midler mocks Lori Loughlin's college admission scandal  
  中文：贝特·米德勒嘲讽洛里·洛夫林的大学招生丑闻
- `N81364` Meghan Markle's estranged brother says 'she is only in Buckingham Palace with Harry because of her dad' after he 'paid for her education and fancy clothes'  
  中文：梅根·马克尔关系疏远的哥哥称，在父亲“为她支付教育和华服费用”后，“她如今能和哈里待在白金汉宫，全是因为父亲”

**正样本（英文 / 中文）**

- `N54368` South Carolina teen gets life in prison for deadly elementary school shooting  
  中文：南卡罗来纳州少年因致命小学枪击案被判终身监禁  
  nearest history=`N57532`, cosine=0.2318, novel-control rank=3

**最相似 novel exposed-unclicked controls**

- `N18190` 30 Best Black Friday Deals from Costco  
  中文：Costco黑色星期五30项最佳优惠；cosine=0.2950
- `N55801` High tides surge through Venice, locals rush to protect art  
  中文：高潮水涌入威尼斯，当地人紧急保护艺术品；cosine=0.2516

## 5. 数据与产物路径（Artifact Paths）

**输入数据**

- `raw_dev_behaviors`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/dev/behaviors.tsv`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/dev/behaviors.tsv)
- `raw_dev_news`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/dev/news.tsv`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/dev/news.tsv)
- `raw_train_behaviors`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/train/behaviors.tsv`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/train/behaviors.tsv)
- `raw_train_news`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/train/news.tsv`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/raw/MINDlarge_reclassified/train/news.tsv)
- `sft_dev_behavior`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/eval/behavior_only/dev_10k_grouped.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/eval/behavior_only/dev_10k_grouped.parquet)
- `sft_dev_interest`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/eval/interest_only/dev_10k_grouped.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/eval/interest_only/dev_10k_grouped.parquet)
- `sft_metadata`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/metadata.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/metadata.json)
- `sft_statistics`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/statistics.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/statistics.json)
- `sft_subset_manifest`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/subset_manifest.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/subset_manifest.parquet)
- `sft_train_behavior`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/train/behavior_only/22k.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/train/behavior_only/22k.parquet)
- `sft_train_interest`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/train/interest_only/22k.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_experiments/train/interest_only/22k.parquet)
- `source_dev`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest/dev.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest/dev.parquet)
- `source_train`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest/train.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/processed/new_interest/train.parquet)

**分析机器产物**

- `manifest.json`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/manifest.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/manifest.json)
- `resolved_config.json`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/resolved_config.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/resolved_config.json)
- `summary.json`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/summary.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/summary.json)
- `analysis_rows.parquet`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/analysis_rows.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/analysis_rows.parquet)
- `title_embeddings.parquet`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/title_embeddings.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/title_embeddings.parquet)
- `category_name_embeddings.parquet`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/category_name_embeddings.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/category_name_embeddings.parquet)
- `similarity_metrics.csv`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/similarity_metrics.csv`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/similarity_metrics.csv)
- `category_name_similarity_metrics.csv`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/category_name_similarity_metrics.csv`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/category_name_similarity_metrics.csv)
- `taxonomy_metrics.csv`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/taxonomy_metrics.csv`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/taxonomy_metrics.csv)
- `slice_metrics.csv`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/slice_metrics.csv`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/slice_metrics.csv)
- `parent_transition_matrix.csv`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/parent_transition_matrix.csv`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/parent_transition_matrix.csv)
- `prompt_overlap.json`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/prompt_overlap.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/prompt_overlap.json)
- `cases.json`: [`/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/cases.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/sft-training-data-analysis/v1/cases.json)

**报告产物**

- Markdown report: [`/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-13/sft_training_data_similarity_analysis.md`](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-13/sft_training_data_similarity_analysis.md)
- Key-results JSON: [`/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-13/sft_training_data_similarity_analysis_key_results.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-08-13/sft_training_data_similarity_analysis_key_results.json)
- Verified Chinese translations: [`/Users/river/Documents/MSRA/Rec/CrossInterest/configs/analysis/sft_training_similarity_case_translations_v1.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/analysis/sft_training_similarity_case_translations_v1.json)
