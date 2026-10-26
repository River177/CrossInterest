# CIR 与 NI 实验汇总

更新日期：2026-09-30。范围：7类CIR主线实验，以及NI-only历史长度、历史顺序两类辅助消融。以下结果均来自已完成并验收的实验；本次只整理文档，没有新增训练或评估。

## 1. 研究问题

NI任务根据用户的冻结浏览历史，预测历史中未出现、但后续日志中被点击的兴趣类别。CIR先删除历史中某类别的全部行为，再用剩余历史重建该类别。CIR目标来自历史，不是真实未来新兴趣标签。

| 实验 | 核心问题 | 比较方式 |
|---|---|---|
| 1. 基本效果、时长和混合方式 | CIR预训练是否有帮助，如何安排CIR与NI？ | NI-only、CIR1/3/5→NI3、CIR5→NI5、混合3/5轮 |
| 2. 历史目标频率 | 重建单次出现还是高频兴趣效果更好？ | 同用户的频率1与频率≥4重建，随后NI3 |
| 3. 参数迁移 | 哪些CIR初始化参数对NI有用？ | 完整继承、仅保留LoRA初始化、仅保留SID初始化 |
| 4. 候选竞争 | CIR的优势是否只来自少预测旧兴趣？ | 对已有模型的108类分数做历史类别过滤 |
| 5. 提示词干预 | CIR重建措辞是否有用？ | 同样CIR数据，原CIR模板与原NI模板 |
| 6. 额外NI监督对照 | 长度匹配CIR能否超过同量历史前缀NI？ | 9,431对，两组辅助5轮后接NI3 |
| 7. 独立监督数据对照 | CIR构造NI能否直接比原NI训练得更好？ | 1,055对，同用户及类别频数，独立SFT3轮 |

数字如CIR5、NI3表示训练轮数，不表示不同模型架构。第6、7项回答不同问题，不应合并成同一个“CIR是否有效”的结论。

## 2. 训练数据构造

### 2.1 共用数据与各条件规模

使用MIND训练数据及固定兴趣分类。原NI训练集21,600条、21,600名用户，108类各200条；输入最多保留H50，即最后50条历史行为，训练与Dev用户隔离。原CIR从这些用户中按固定hash选择合格历史类别，删除该类别全部行为，至少保留5条及另一类别，不回填更旧历史，获得17,271条固定视图。

| 实验/条件 | 训练样本与类别 | 训练安排/有效更新 |
|---|---|---|
| 原NI-only | NI 21,600条、108类 | NI3：2,025步 |
| 原CIR1/3/5→NI3 | CIR 17,271条、108类；NI 21,600条 | CIR 540/1,620/2,700步，接NI 2,025步 |
| CIR5→NI5 | 同上 | CIR 2,700步，接NI 3,375步 |
| CIR+NI混合3/5轮 | 两任务并集38,871条 | 3,645/6,075步；单阶段连续训练 |
| 单次/高频CIR→NI | 各9,883用户；CIR目标分别覆盖108/75类 | 各CIR1：309步，接原NI3：2,025步 |
| 参数迁移 | 固定同一CIR5模型，原NI 21,600条 | 两个新增NI3分支各2,025步；不重训CIR |
| NI提示词CIR→NI | 原CIR 17,271条，原NI 21,600条 | CIR5→NI3，正式总更新4,725步 |
| 前缀NI / 长度匹配CIR | 各9,431对同用户、同目标；覆盖108类 | 辅助5轮：1,475步，接原NI3：2,025步 |
| 独立CIR构造NI / 原NI | 各1,055名同用户；两组相同98类及逐类数量 | 各独立3轮：99步、3,165曝光；不接NI微调 |

两阶段条件在交接时继承LoRA和SID参数、重置optimizer/scheduler；混合训练没有该交接。各条件的有效曝光和恢复游标以各自验收记录为准，不把故障后未提交更新的回放计入正式预算。

### 2.2 最近两类对照如何匹配

**9,431对实验：** 前缀NI截在目标类别首次出现前，并检查完整原历史中目标此前未出现；长度匹配CIR删除目标类别后，按固定hash保留与前缀相同数量的行为。两组用户、目标、SID、顺序、行为长度与辅助更新数匹配，使用共同中性辅助提示词；均再训练原NI3。保留1,088对完全相同输入，没有事后剔除。

**1,055对实验：** 先构造17,271对“CIR标签与该用户原NI标签”的独立监督数据，再筛出原H50中CIR目标出现至少2次的4,870对，最后选取两组整体标签频数相同且用户相同的最大子集1,055对。两组98个训练类别的逐类计数完全相等，但不同类别之间仍有1–32条的差异。每个用户两组标签不同，未改写标签，也未重复采样；原NI模板及训练格式完全相同。选择最终队列使用双方训练标签，不使用Dev或模型效果。历史条目次数没有强制按不同news ID去重。

17,271对与4,870对版本只完成数据构造/审计，没有各自独立训练；实际已训练的是1,055对版本。10个未被该训练子集覆盖的类别仍保留在108候选词表中，完整Dev不删类。

### 2.3 真实训练样例

以下是1,055对数据中按两组prompt总字符数最短选出的一个实际配对，用于说明格式，不按效果选例。用户`U108599`，来源`train:820263:49`；响应均为训练标准答案，不是模型预测。

**原NI输入（英文原文）**

```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Vehicle Buying & Ownership] How Often Should You Change Synthetic Oil?
2. [Film] 40+ Must-See Movies Coming Out This Fall
3. [Nutrition & Healthy Eating] 50 Awful Foods That Should Never Be in Your Kitchen
4. [Air & Rail Travel] Why it's so hard for planes to land on water
5. [Personal Finance] 9 Types of Retirement Income That Are Tax-Free
6. [Cars & Driving] Why Is This Weird Chevy Colorado Test Mule Trolling Ford HQ?
7. [Air & Rail Travel] Why plane tires don't explode during landing

Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form <interest_NNN>.
```

中文翻译：

```text
我最近浏览的新闻标题及对应主题类别按时间顺序列在下面。每条格式为“[主题类别] 标题”：

1. [车辆购买与拥有] 合成机油应该多久更换一次？
2. [电影] 今年秋季上映的40多部必看电影
3. [营养与健康饮食] 50种绝不该出现在厨房里的糟糕食物
4. [航空与铁路旅行] 为什么飞机很难在水面降落
5. [个人理财] 9种免税的退休收入
6. [汽车与驾驶] 这辆奇怪的雪佛兰Colorado测试车为何到福特总部挑衅？
7. [航空与铁路旅行] 为什么飞机轮胎在降落时不会爆炸

根据标题和主题类别，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在我的浏览历史主题类别中。仅回答分配给预测类别的已学习兴趣token，格式为 <interest_NNN>。
```

目标响应：`<interest_056>`，Historical Events & Eras（历史事件与时代），来自该用户原始未来NI监督。

**CIR构造NI输入（英文原文）**

```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Vehicle Buying & Ownership] How Often Should You Change Synthetic Oil?
2. [Film] 40+ Must-See Movies Coming Out This Fall
3. [Nutrition & Healthy Eating] 50 Awful Foods That Should Never Be in Your Kitchen
4. [Personal Finance] 9 Types of Retirement Income That Are Tax-Free
5. [Cars & Driving] Why Is This Weird Chevy Colorado Test Mule Trolling Ford HQ?

Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form <interest_NNN>.
```

中文翻译：

```text
我最近浏览的新闻标题及对应主题类别按时间顺序列在下面。每条格式为“[主题类别] 标题”：

1. [车辆购买与拥有] 合成机油应该多久更换一次？
2. [电影] 今年秋季上映的40多部必看电影
3. [营养与健康饮食] 50种绝不该出现在厨房里的糟糕食物
4. [个人理财] 9种免税的退休收入
5. [汽车与驾驶] 这辆奇怪的雪佛兰Colorado测试车为何到福特总部挑衅？

根据标题和主题类别，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在我的浏览历史主题类别中。仅回答分配给预测类别的已学习兴趣token，格式为 <interest_NNN>。
```

目标响应：`<interest_004>`，Air & Rail Travel（航空与铁路旅行）。其两条历史行为已全部删除，剩余5条完整保留。

早期普通CIR使用不同的重建指令。该样本的原CIR提示词开头和结尾原文为：

```text
The following news browsing history has all entries from one topic category removed. The remaining entries are in chronological order, formatted as "[Topic category] Headline":

Infer the removed topic category from the remaining observed interests and news items. This is historical interest reconstruction, not a prediction of the next click. Respond with only its single-token Interest ID in the exact format <interest_NNN>. Do not output the category name or any explanation.
```

中文翻译：

```text
以下新闻浏览历史已移除某个主题类别的全部记录。剩余记录按时间顺序排列，格式为“[主题类别] 标题”：

根据剩余的已观察兴趣和新闻推断被删除的主题类别。这是历史兴趣重建，不是预测下一次点击。只输出该类别的单token兴趣ID，严格使用 <interest_NNN> 格式，不输出类别名或任何解释。
```

上面只引用指令部分；完整原CIR prompt中的历史与该CIR构造NI样本相同。所有英文文本均读取实际Parquet，中文翻译仅供文档阅读，不进入训练。

## 3. 测试设计

### 3.1 模型和训练设置

主线模型为Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，seed42。主要CIR训练使用LoRA r16/alpha32/dropout0.05及可训练SID行，lr2e-5、cosine、warmup0.03、weight decay0.01，原生全词表completion-only token-mean交叉熵。常规四卡global batch32，microbatch2×4卡×accumulation4。CIR系列使用显式Torch fallback、非reentrant checkpointing及精确重载检查。

历史NI-only基线保留单卡转四卡及FLA执行差异，不能把其与所有后续实验的比较描述为数值执行路径完全相同。两个最新配对实验在各自实验内部冻结同一运行实现。推理在远端执行，没有测试时适应。模型权重、optimizer、完整预测均保留远端。

### 3.2 测试数据和指标

除明确说明的输入或候选干预，均使用同一Full Dev：31,392条grouped impressions、27,082名用户、37,728次目标出现、104个有测试支持的类别。固定108 SID候选、合法无重复Top20；普通评价不做历史类别过滤。长度消融改变Train和Dev可见历史；顺序消融改变指定组的排列；候选竞争诊断显式改变候选过滤规则。

对每个目标类别c，先计算它的所有目标出现的命中或排名折扣均值，再对104个支持类别等权平均。Macro Recall@K不能用一个全局“命中数/37,728”替代。此项目的Macro NDCG@K按每个目标的排名r贡献 `1/log2(r+1)`（r≤K；否则0），再做同样的类内和类间平均；不要将其与其他推荐论文不同定义的impression级NDCG直接互换。

本组实验不用LLM judge，没有judge提示词或人工主观打分。对已有预测进行的成对Bayesian bootstrap为2,000次、seed42：同一用户的全部记录共享正Exp(1)权重，两模型共用这些权重，每次重算类别分母。区间仅涵盖评估用户层面的不确定性，不涵盖训练seed；未提供区间的实验不补造显著性结论。

### 3.3 真实测试输入与标准答案

确定性选择最短Dev prompt：impression `362173`，用户`U737543`。下面响应为可接受标准答案，未展示该条模型实际排名；本次文档整理没有下载远端完整预测来生成个例。

```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Film] The most powerful LGBTQ movies

Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form <interest_NNN>.
```

中文翻译：

```text
我最近浏览的新闻标题及对应主题类别按时间顺序列在下面。每条格式为“[主题类别] 标题”：

1. [电影] 最有力量的LGBTQ电影

根据标题和主题类别，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在我的浏览历史主题类别中。仅回答分配给预测类别的已学习兴趣token，格式为 <interest_NNN>。
```

可接受响应：`<interest_019>`，Celebrity & Creator Culture（名人及创作者文化）。汇总指标来自已验收的全量排名，不以本样例推断模型是否答对。

## 4. 实验结果

所有下表指标单位均为%，差值及区间单位为百分点。各小节内部是对应对照；不同数据规模和训练预算的小节不能合并为统一排行榜。

### 4.1 基本效果、训练时长与混合方式

**做了什么：** 比较只训练NI、先CIR后NI、不同比例的训练轮数，以及CIR/NI数据直接混合。混合3/5轮为各自单独训练的模型；CIR5→NI3和CIR5→NI5共享同一已训练CIR5参数，但分别以新optimizer训练NI分支。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| NI-only | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| CIR1 → NI3 | 21.2786 | 34.2609 | 10.8068 | 14.0725 |
| CIR3 → NI3 | 21.8272 | 35.1740 | 11.0708 | 14.4257 |
| CIR5 → NI3 | 21.9717 | 35.0756 | 11.2051 | 14.4966 |
| CIR5 → NI5 | 20.5892 | 33.8399 | 10.4749 | 13.7809 |
| CIR + NI 混合3轮 | 21.8237 | 34.9396 | 10.8494 | 14.1506 |
| CIR + NI 混合5轮 | 21.3238 | 35.1462 | 10.9930 | 14.4630 |

CIR5→NI3的R@10最高，较历史NI-only高1.6442个百分点；R@20最高是CIR3→NI3。CIR5→NI5四项指标均低于CIR5→NI3。混合5轮在相同任务曝光预算下高于CIR5→NI5，但两者还同时改变任务顺序和优化器重置，不能单独归因于其中一个因素。以上为单seed点估计，无显著性声明。

### 4.2 单次兴趣与高频兴趣重建

**做了什么：** 选择同时存在两类候选的9,883名用户；一组重建出现1次的类别，另一组重建出现≥4次的类别。两组均CIR1→原NI3，CIR各309步。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| NI-only（参考） | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| 原随机 CIR1→NI3（参考） | 21.2786 | 34.2609 | 10.8068 | 14.0725 |
| 单次兴趣 CIR1→NI3 | 20.6996 | 34.7380 | 10.5004 | 14.0358 |
| 高频兴趣 CIR1→NI3 | 21.4692 | 34.5215 | 10.9226 | 14.1874 |

高频组相对单次组R@10高0.7697个百分点，NDCG@10/@20也更高，但R@20更低。高频组只有75类、平均删除6.44条，单次组覆盖108类、删除1条，因此该比较同时改变类别分布和输入信息，不能据此证明“频率越高，标签越干净”。

### 4.3 LoRA/SID初始化迁移

**做了什么：** 固定CIR5参数，分别完整继承、保留LoRA但重置SID、保留SID但重置LoRA，然后训练同样的NI3。两个局部重置组在NI期间仍训练全部LoRA和SID，并非冻结某个参数块。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| 完整继承 CIR5→NI3 | 21.9717 | 35.0756 | 11.2051 | 14.4966 |
| 保留 LoRA、重置 SID | 20.8325 | 34.9392 | 10.6330 | 14.1928 |
| 保留 SID、重置 LoRA | 21.2467 | 33.9301 | 10.6177 | 13.7973 |

完整继承在四项主要指标上均最高，尚无某个参数块单独主导全部收益的证据。另做了108条“类别名→SID”和512名Dev用户历史重建探针：名称映射更准不一定对应NI表现更好。比如仅保留SID的初始化名称Top1为67/108，高于完整继承43/108，但后续NI四项指标都更低。详细探针表见原报告，不能把该干预解释成知识存储位置的直接定位。

### 4.4 历史类别候选竞争（仅评估）

**做了什么：** 复用NI-only及CIR5→NI3模型，获取完整108类分数，并从同一分数中排除用户历史类别。没有新增训练。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| NI-only，原排名 | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| CIR5→NI3，原排名 | 21.9717 | 35.0756 | 11.2051 | 14.4966 |
| NI-only，过滤历史类别 | 20.4186 | 34.1764 | 10.1720 | 13.6148 |
| CIR5→NI3，过滤历史类别 | 22.0009 | 35.1728 | 11.2172 | 14.5269 |

过滤后CIR−NI-only的R@10差距仍为1.5823个百分点，原始差距为1.6442个百分点。原Top10中历史类别占比由NI-only的0.8725%降至CIR的0.2437%；排除历史类别后仍有差距，因此历史候选占位不能解释全部观察差异。该诊断保留历史模型执行差异，不作因果贡献比例或跨seed显著性主张。

### 4.5 CIR提示词改为NI提示词

**做了什么：** 保持17,271条CIR历史、目标、顺序、初始化、5轮CIR及3轮NI预算不变，仅更改prompt与同步messages为原NI模板。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| 原CIR模板→NI3 | 21.9717 | 35.0756 | 11.2051 | 14.4966 |
| NI模板CIR→NI3 | 20.9914 | 34.9470 | 10.9410 | 14.4628 |

R@10下降0.9803个百分点，四项主要指标均下降，支持在这套CIR5→NI3流程中保留原重建措辞。@1/@5存在小幅上升，不能说所有K均下降；本实验单seed，没有显著性结论，也不能把全部CIR收益归因于提示词。

### 4.6 历史前缀NI与长度匹配CIR

**做了什么：** 在9,431对同用户、同目标、同可见行为长度的数据上，分别辅助训练5轮，再接完全相同的原NI3，检验CIR能否超过额外前缀NI监督。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| 历史前缀NI5→NI3 | 21.2238 | 34.5649 | 10.8744 | 14.2236 |
| 长度匹配CIR5→NI3 | 20.3668 | 34.6897 | 10.9051 | 14.5266 |

| 指标 | CIR−NI（百分点） | 用户成对95%区间（百分点） |
|---|---:|---|
| R@10 | -0.8570 | [-1.4962, -0.2892] |
| R@20 | +0.1247 | [-0.4071, +0.7137] |
| NDCG@10 | +0.0307 | [-0.3352, +0.4016] |
| NDCG@20 | +0.3030 | [+0.0085, +0.6193] |

主指标R@10的CIR差值为负且区间不含零，因此本对照未支持长度匹配CIR超越前缀NI的主指标优势。R@20和NDCG@10区间包含零，NDCG@20略高。该条件同时不同于普通CIR的完整17,271样本、完整剩余历史及原CIR措辞，不能用本结果直接否定普通CIR的全部作用。

### 4.7 同用户、同标签分布的独立监督对照

**做了什么：** 对最终1,055对数据固定同用户、同98类别及逐类样本数、同NI模板；CIR标签历史出现≥2次。两组各从原始基座独立SFT3轮、99步，训练后直接评价真实NI，不接第二NI阶段。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| 原NI独立训练 | 12.4049 | 22.7448 | 5.9451 | 8.5316 |
| CIR构造NI独立训练 | 13.7849 | 24.2566 | 6.6596 | 9.2833 |

| 指标 | CIR−NI（百分点） | 用户成对95%区间（百分点） |
|---|---:|---|
| R@10 | +1.3800 | [+0.9845, +1.7736] |
| R@20 | +1.5118 | [+0.9269, +2.1082] |
| NDCG@10 | +0.7146 | [+0.5519, +0.8738] |
| NDCG@20 | +0.7517 | [+0.5702, +0.9415] |

四项指标的CIR差值和逐指标区间均为正，主指标R@10提高1.3800个百分点。在本受控队列与预算下，观察差异不能由“两组样本数量不同”或“目标类别频数不同”解释。它支持CIR构造监督在该条件下迁移到真实NI更有效，尚不直接证明标签噪声更低，因为历史内容/长度与目标来源仍然不同。

本实验只有1,055对、单seed，不能与21,600条NI-only的20.3275%直接比较并归因于训练方法；也不能把其正结果与上一节负结果择一报告。两个实验的训练流程、匹配条件、频率筛选及样本规模均不同。

### 4.8 NI-only历史长度消融（辅助）

**做了什么：** 固定21,600条NI的用户、标签和顺序，把Train和Dev历史上限分别设为H50/H30/H10/H5；后面三组新增训练与评估，H50复用原基线。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| H50 | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| H30 | 20.4167 | 33.3155 | 10.4809 | 13.7103 |
| H10 | 19.3561 | 31.6497 | 9.4951 | 12.5745 |
| H5 | 17.4267 | 29.7096 | 9.3350 | 12.4370 |

Full Dev上H30与H50的R@10接近，H10和H5更低。这里同时改变训练和测试输入，不是只改变CIR历史长度的对照；没有对应的多seed显著性结论。

### 4.9 NI-only历史顺序消融（辅助）

**做了什么：** 复用原模型并新训练一个固定打乱历史的NI模型，比较训练/评估原始或打乱顺序的四种组合；类别、标题及重复频次保持不变，原提示词保留。

| 条件 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| A：原始→原始 | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| B：原始→打乱 | 20.1622 | 33.4186 | 10.6675 | 13.9747 |
| C：打乱→原始 | 20.3258 | 33.8395 | 10.3544 | 13.7363 |
| D：打乱→打乱 | 20.2758 | 33.3891 | 10.6520 | 13.9222 |

预先设定的五个R@10比较区间均包含零，未显示原始顺序的明确整体优势，也不能证明等价。同模型打乱输入时Top1分别改变8,916/31,392（28.4021%）和8,805/31,392（28.0485%）条记录，不能把总体R@10接近解释成个体预测不依赖顺序。

## 5. 数据与产物路径

以下均为本次已核对存在的本地紧凑报告/元数据/数据路径。完整远端模型与预测位置通过manifest记录，不因本次整理而下载。本文没有新增任何模型条件、seed或评估。

### 5.1 逐实验报告与数值来源

| 实验 | 原始报告 | 指标来源 | 配置/验收入口 |
|---|---|---|---|
| 基本效果与混合 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-18/mind_cir_four_conditions.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-four-20260918/all_conditions_comparison.csv) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_four_108_20260918_v1.json) |
| 频率 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-17/mind_cir_frequency_paired.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-17/mind_cir_frequency_paired_results.csv) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_frequency_paired_20260917_v1.json) |
| 参数迁移 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-23/mind_cir_parameter_transfer_results.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/comparison.csv) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_parameter_transfer_20260923_v1.json) |
| 候选竞争 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-24/cir_candidate_competition_results.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-candidate-competition-20260924/final-report/comparison.csv) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_candidate_competition_20260924_v1.json) |
| 提示词 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-28/mind_cir_ni_prompt.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/prompt_comparison.csv) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_ni_prompt_20260928_v1.json) |
| 前缀匹配 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-28/mind_cir_ni_matched.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-matched-20260928/comparison.json) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_ni_matched_20260928_v1.json) |
| 独立同分布 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-29/mind_cir_ni_direct_training.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-direct-training-20260929/comparison.json) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_ni_direct_training_20260929_v1.json) |
| NI历史长度 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-15/mind_history_length_ablation.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/comparison.json) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/comparison.json) |
| NI历史顺序 | [报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-23/mind_history_order_ablation.md) | [CSV/JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-order-20260923/progress/check-2103-durable/comparison.json) | [manifest/验收](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_history_order_20260923_v1.json) |

### 5.2 数据、示例与完整验收

- [原CIR/NI数据](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1)
- [9,431对匹配数据](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_matched_20260928_v1)
- [17,271对独立监督数据（未单独训练）](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_direct_20260929_v1)
- [4,870对重复目标数据（未单独训练）](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_direct_repeated_20260929_v1)
- [1,055对最终同分布数据（已训练）](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_direct_balanced_20260929_v1)
- [示例CIR训练文件](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_direct_balanced_20260929_v1/cir_as_ni.parquet)
- [示例NI训练文件](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_direct_balanced_20260929_v1/original_ni_paired.parquet)
- [固定Full Dev与示例来源](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_ni_direct_balanced_20260929_v1/dev_full_grouped.parquet)
- [1,055对联合验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-direct-training-20260929/paired_final_acceptance.json)
- [9,431对联合验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-matched-20260928/paired_final_acceptance.json)
- [原CIR5→NI3验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-four-20260918/acceptance/cir5_ni3.json)
- [提示词实验验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/final_acceptance.json)

各manifest中保留远端experiment/job/backend、数据/模型/adapter哈希与结果前缀；其中1,055对的完整预测分析在Feeds Basic CPU作业完成。所有训练stage的checkpoint/reload和评价验收以各实验原始收据为准。旧报告中的建议、排队或运行中历史段落不作为新的未完成条件计数。

[本文全部主指标汇总CSV](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-30/cir_ni_experiment_overview_metrics.csv)（每行标注所属实验；不可跨family作公平预算排名）。
