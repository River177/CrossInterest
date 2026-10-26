# CIR 的新颖性核查：历史兴趣重建能否承担主要方法贡献

核查日期：2026-09-30。范围：已有方法原文、BTBR 作者代码、当前 CIR 数据构造及现有实验结论；没有启动训练、评估或修改既有实验。

**结论：目前不建议将“历史兴趣重建”单独作为论文的强方法创新。** 已有工作分别覆盖了兴趣重建、目标的全部出现遮蔽、通过这种遮蔽阻断重复捷径、重建预训练接未来预测，以及 LLM 的新兴趣类别预测。最接近的先例是 RecSys 2023 的 BTBR。CIR 的具体实现与它不同，但方法原则高度接近。

本次核验的论文中，没有确认一篇完整复现“删除一个显式兴趣类别的全部新闻行为及标题，以剩余新闻重建其类别 SID，再迁移到冻结历史未见类别预测”的方法。这是有限范围内的检索结果，不能升级为首创证明，也不能因实现细节不同就推断创新充分。

## 1. 本次审查的 CIR 是什么

以当前 [数据构造代码](../../src/new_interest/data/build_cir_dataset.py) 为准：

1. 取用户最后 50 条新闻行为，输入包含类别与标题。
2. 从合格历史类别中确定一个目标类别；选择不使用未来 NI 标签。
3. 删除该类别的全部新闻行为，至少留下 5 条及其他类别，不回填更早历史。
4. 根据剩余行为预测删除类别的一个 SID；提示词明确说明是历史重建。
5. 原标准流程先做 CIR SFT，再接原 NI SFT；沿用原生 completion CE，没有独立的新损失。

令类别映射为 \(g(v)\)，这项构造可写为：

\[
H^{-c}=\{v\in H:g(v)\ne c\},\qquad (H^{-c},c).
\]

保留原顺序。目标在人工删减后的输入中未出现，但在用户真实历史中出现过；它与真实未来 NI 标签有不同的来源。

主 CIR 并不要求目标出现至少两次。“至少两次”仅属于后续 1,055 对独立训练队列，不能将它追溯成全部 CIR 实验的定义。

## 2. 最关键的先例：BTBR

**Ming Li et al. — Masked and Swapped Sequence Modeling for Next Novel Basket Recommendation in Grocery Shopping，RecSys 2023。**

- [论文与出版信息](https://arxiv.org/abs/2308.01308)
- [方法 §4.2](https://arxiv.org/html/2308.01308v1#S4.SS2)
- [作者代码，固定 commit 的遮蔽实现](https://github.com/liming-7/Mask-Swap-NNBR/blob/8e0b796a9910888d6a8142f2d39dc7cbe87e349c/nnbr/bert4nbr.py#L118-L139)

原文的 **Select masking** 不是普通随机位置遮蔽。作者指出，如果被遮住的商品还在其他历史位置出现，模型可能依靠重复信息完成预测。因此先选择商品，再：

> “mask all the occurrences”

模型必须从其他商品的关系推断被遮住的商品。这里的新颖性目标也很接近：下游 NNBR 只推荐该用户没有购买过的商品。

§4.2 的 **Joint task** 还明确讨论先用 item-level cloze 自监督预训练，再用 basket-level 监督微调。不能把“先重建再预测”作为 CIR 独有贡献。论文结论同时指出，joint 策略并没有在所有数据集优于最佳单阶段策略；这里引用的是方法先例，不是借它的效果为 CIR 背书。

作者代码与原文一致：`mask_type == 'select'` 分支先取得 item set、抽取目标，再遍历每个位置，将所有命中目标的 item 替换为 mask token。该核验仅限遮蔽操作，没有运行或复现其训练结果。

| 比较点 | BTBR | 当前 CIR |
|---|---|---|
| 缺失单元 | 商品 ID 的全部出现 | 同一兴趣类别下不同新闻的全部出现 |
| 缺失表示 | 原位置保留 mask token | 删除整条记录，重新编号，不保留空槽 |
| 输入信息 | 商品与 basket 位置 | 新闻标题、显式类别与顺序 |
| 重建目标 | 被遮商品 ID，按被遮位置监督 | 被删兴趣类别 SID，每条视图一个目标 |
| 下游 | 用户未购买商品组成的未来 basket | 冻结历史未见、未来被点击的兴趣类别 |
| 两阶段安排 | 讨论 cloze 预训练→basket 微调 | CIR→NI |

**判断：它不是完整相同的方法，但已经覆盖 CIR 最容易讲成创新的设计原则。** 若把类别当作 token 化单元，按相同取值遮蔽全部出现就是一个自然的 BTBR-style 适配。因此“我们的类别更粗、模型换成 LLM、输出换成 SID”不足以自动构成强创新。这是基于方法对照的研究判断，不是“所有类别级实现已经被发表”的事实断言。

## 3. 其他直接相关工作

| 工作 | 核验位置与实际操作 | 与 CIR 的关系 |
|---|---|---|
| **BERT4Rec，CIKM 2019** [1] | §3.6：随机遮蔽历史位置，预测商品 ID；讨论 cloze 与未来预测的差异，训练时加入只遮最后位置的样本 | 历史重建监督及其预测用途已有。没有强制清空目标兴趣的全部记录 |
| **S3-Rec，CIKM 2020** [2] | §4.3.2–4.3.4：Masked Item Prediction、Masked Attribute Prediction、Segment Prediction；MAP 从上下文恢复被遮商品的属性，SP 恢复连续片段 | 类别/属性恢复与分层重建预训练已有。其缺失单元是商品位置或连续片段，不要求同一类别在全历史消失 |
| **CIKGRec，arXiv 2024，记录声明 AAAI 2025 接收** [3] | Method → User Interest Reconstruction：用 LLM 推断兴趣，构建图，遮蔽兴趣节点 embedding，用图 encoder/decoder 恢复 embedding；图结构作为输入保留 | 明确已有“用户兴趣重建”。没有删除该类别所有原始历史行为；目标不是类别 ID；采用用户行为随机 7:1:2 划分的一般 item 推荐，不能等同真实 NI |
| **CauseRec，SIGIR 2021** [4] | §3.3：在 item/latent-interest 两级替换重要或非重要概念，构造正负视图并学习对比目标 | 有结构的兴趣干预也已有；没有把被删除类别作为单独重建答案 |
| **LLMs for User Interest Exploration…，RecSys 2024** [5] | §3.2–3.3：两个历史兴趣 cluster→后续不同 cluster，按目标类别平衡 SFT，并映射回 item 推荐 | LLM 兴趣级输出、兴趣间关系、新类别预测、类别平衡 SFT 均已有；标签是观察到的后续转移，没有做整类历史擦除 |
| **CoEA，2025 预印本，核验 v3（2026）** [6] | §4.3.1：用户群体画像与前两个短期类别作为输入，预测后续类别，并筛选相对长期历史未见的目标；还有 relevance 模型与周期优化 | 类别级探索与历史序列构造监督已有；没有确认 CIR 的整类删除重建操作。其公开指标定义与当前 Full Dev NI 不同 |
| **T-REX，2026 预印本** [7] | Introduction 明确讨论不遮住目标商品所有出现时的重复信息问题；§3.4 最终采用 causal decoder、teacher forcing、CE 做类别层面的未来 basket 生成 | 类别层建模与全部出现遮蔽的动机已有相邻讨论；不能将其最终方法误写成 CIR 式重建 |

其中 CIKGRec 的原文直接使用 **User Interest Reconstruction** 作为模块名。它使“过去没有人研究过兴趣重建”这一宽泛说法不可成立，但不构成与 CIR 全流程完全相同的证据。

## 4. 搜索中出现、但方法不应混淆的论文

为避免看到 mask/interest/reconstruction 就认作重复，另核对了：

- **MISSRec，2023** [8]：§3.2.2 是特征 dropout，§3.5 为 sequence-item 与 sequence-sequence 对比预训练；“interest-aware”不等于删除一类兴趣后重建该类别。
- **MindRec，2025 预印本** [9]：遮蔽和恢复目标输出的 category/SID token，历史条件保留；不是把历史中一整类行为抹去。
- **MGDiff，2026 预印本** [10]：Dual-layer Semantic Guidance 遮蔽全局 item 转移图中的边并做边恢复，另用扩散预测未来 item；不是用户级整类历史删除。

## 5. 能保留什么主张，应该放弃什么主张

### 已有先例明确覆盖

- 从损坏的历史中重建监督目标。
- 属性/兴趣相关的重建任务。
- 遮蔽同一目标的全部出现，减少重复信息捷径。
- 用重建预训练帮助探索或未来预测。
- 用 LLM 从历史兴趣生成新兴趣类别。

### 当前实现确实具有的差异

- 缺失条件定义在**显式语义类别**上，清除该类别下不同新闻的所有直接记录。
- 保留其他类别的标题内容，重建一个类别级目标。
- 将这个历史内构造作为真实未见类别预测的辅助训练。
- 实现简单，沿用现有 SFT 与推理接口。

这可以作为一种**面向任务的监督构造**来介绍。是否足以支撑整篇论文，要看相对近邻方法的增量证据和论文整体贡献，目前不能保证。

“目标类别记录全部删除”也不意味着删除所有语义线索：其他类别新闻仍可能含有相关实体、跨类别话题或标题关联。这种残留有可能是有价值的跨兴趣信息，也可能形成捷径，不能在没有证据时宣称强制学到了因果关系或高阶推理。

## 6. 与当前实验结果如何对齐

本节数值再次读取了当前本地已验收报告，并未重新运行实验。

- [9,431 对前缀 NI 对照](../experiments/2026-09-28/mind_cir_ni_matched.md)：长度匹配 CIR 的 Macro R@10 为 20.3668%，前缀 NI 为 21.2238%，差值 −0.8570 个百分点。用户成对 95% 区间为 [−1.4962, −0.2892]。这不支持 CIR 在此条件下超过同量前缀 NI。
- [1,055 对独立训练](../experiments/2026-09-29/mind_cir_ni_direct_training.md)：CIR 构造监督 13.7849%，原 NI 12.4049%，差值 +1.3800 个百分点，用户成对 95% 区间为 [+0.9845, +1.7736]。相同用户、相同总量及逐类频数下，CIR 构造的监督具有更好的泛化观察；但输入长度、内容和标签来源仍不同，不能直接证明“标签更干净”。

两组都是单训练 seed；区间反映固定模型在评估用户上的不确定性，不涵盖训练随机性。两项结果回答不同问题，不矛盾，也不应只选后一项为 novelty 背书。

因此，当前不必强行证明“CIR 在本质上绝不是增加训练数据”。它确实增加/改造了监督，这是方法作用的可能来源。论文可以研究**什么构造的监督对真实 NI 有价值**，但还需要回答它相对于已有全目标遮蔽方法带来了什么新认识。

## 7. 最值得补齐的证据：语义类别全删除是否必要

下述仅为建议，没有启动或授权新实验。

当前主要缺口不是再找一个 NI 数据来源，而是验证 CIR 相对于已有遮蔽原则的具体增量。最直接的问题是：**同一来源、同一目标类别，删除一个新闻实体的全部出现，和删除该类别下所有新闻，是否产生不同的真实 NI 泛化？**

可以在目标类至少有两个不同 news ID 的同一合法队列上设计：

- **新闻实体级全部遮蔽的适配对照**：删除选中 news ID 的全部出现，仍预测它所属类别。至少留下一个同类别的不同新闻。
- **类别级全部删除 CIR**：删除目标类别所有 news ID，预测同一类别。

两臂固定用户、目标类别/SID、目标频数、视图数、SFT 目标、更新预算和后续 NI。若要匹配行为长度，对前者额外删除非目标类别行为，且保证同类别其他新闻仍可见；只采用双方都可构造的队列，预先固定规则。使用对两种输入都语义正确的共同中性提示词，不能在保留目标类别的对照中写“目标不应出现在历史里”。

这只能称为 **BTBR-style 遮蔽原则的类别预测适配**，因为原 BTBR 预测 item，并非这个类别目标；若论文声称全面优于原方法，还需要按其自身定义做合理基线适配，不能把此消融冒充完整复现。

匹配长度仍有信息组成差异，而且额外随机删除可能损伤对照，因此该实验测的是完整的“证据删除规则”，不是无混杂的单一心理机制。它的价值在于检验所声称的类别级缺失设计是否确实有必要。

若类别全删除没有额外收益，应将 CIR 定位为有效的重建/数据构造配方。若有稳定收益，可以支持“类别级缺失条件有助于真实 NI”的贡献；仍需结合相关工作、外部数据/模型验证讨论普适性，不能由一次正结果直接宣布首创或保证录用。

## 8. 可采用的论文表述

> 我们研究如何利用已有浏览历史为未见兴趣预测构造监督。受已有全目标遮蔽思想启发，我们在显式兴趣类别层面移除该类别的全部历史行为，并从剩余新闻内容重建其类别标识，将所得监督用于新兴趣预测训练。

若后续必要性对照成立，再加入有证据支持的相对优势。目前不宜写“首次提出兴趣重建”“首次通过全部遮蔽消除重复捷径”“已证明 CIR 超越高质量 NI 数据增广”或“已证明重建标签更干净”。

## 来源与检索边界

除正文 BTBR 外，原文链接如下：

1. Fei Sun et al. *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*. [原文](https://arxiv.org/abs/1904.06690)。
2. Kun Zhou et al. *S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization*. [原文](https://arxiv.org/abs/2008.07873)。
3. Zheng Hu et al. *Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models*. [兴趣重建模块](https://arxiv.org/html/2412.13544v1#Sx4.SSx2.SSSx1)；[arXiv 接收声明](https://arxiv.org/abs/2412.13544)。
4. Shengyu Zhang et al. *CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation*. [原文](https://arxiv.org/abs/2109.05261)。
5. Jianling Wang et al. *LLMs for User Interest Exploration in Large-scale Recommendation Systems*. [原文](https://arxiv.org/html/2405.16363v2)；[DOI](https://doi.org/10.1145/3640457.3688161)。
6. Hongxiang Lin et al. *When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation*. [原文 v3](https://arxiv.org/html/2508.00450v3#S4.SS3.SSS1)。未使用 PDF 中的占位 venue/DOI。
7. Soroush Mokhtari et al. *T-REX: Transformer-Based Category Sequence Generation for Grocery Basket Recommendation*. [原文](https://arxiv.org/html/2603.06631v1)。本次按预印本引用。
8. *MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation*. [方法](https://arxiv.org/html/2308.11175v2#S3.SS5)。
9. *MindRec: A Diffusion-driven Coarse-to-Fine Paradigm for Generative Recommendation*. [原文](https://arxiv.org/html/2511.12597v2)。
10. *MGDiff: Multi-Interest Sequence Recommendation with Masking GNN-Guided Diffusion*. [原文](https://arxiv.org/html/2609.01619v1)。

本次重新阅读上述 11 篇论文的相关方法段落；前 4 份缓存原始 PDF 的 SHA-256 与此前来源清单一致，新增论文通过 arXiv 原文获取。BTBR 另核对固定 commit 的作者实现。检索结合项目既有文献线索、Google 结果和五项 arXiv API 查询，后者返回 91 条去重元数据记录；这不代表读完 91 篇全文。arXiv API 不是全论文全文搜索，且跨领域同词命中很多。Google HTTP 页面需 JavaScript；OpenAlex 多次限流且结果混杂，没有用这些失败作为不存在先例的证据。CDAE 的出版页/作者地址未成功取得原文，本报告不将它列为已核验依据。

原始检索响应、方法摘录、作者代码和来源哈希保存在 [核查目录](../../artifacts/cir-novelty-review-20260930/)，详见 [来源清单](../../artifacts/cir-novelty-review-20260930/source_manifest.json)。这是面向当前 CIR 的定向查重，不是穷尽式系统综述，也不是首创证明。
