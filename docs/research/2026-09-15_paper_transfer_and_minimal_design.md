# 从相关论文提炼新兴趣预测的最小改进

调研日期：2026-09-15。任务：给定 History Behavior+Interest，预测未来点击、且完整原始历史未出现的兴趣类别；保留单正例训练、固定 SID 词表和用户隔离。本文研究方法选择，不提交训练，不改变已冻结实验。

**结论：优先研究“训练时移除目标类别的直接历史证据”，保留同一个真实未来目标；若允许改为判别式类别评分，再考虑一个候选兴趣条件化的历史聚合头。两条先分别验证，不组合成多模块系统。**

本次核验原论文的方法章节，并将原文结论、迁移设想和项目已有工作分开。论文的原始指标不与 CrossInterest 的 Macro Recall 混排。预印本不根据页眉中的 conference/year 占位模板推定 venue。

**1．“优雅”作为方法筛选标准**

这里把用户提出的“改得不丑陋”具体化为四条标准：一个可表述的任务假设；一个主要改动位置；尽量保留 backbone、SID 和预测接口；一个直接检验假设的对照。新增组件必须解决已定义的问题，不能仅凭总分略涨而保留。

本仓库已有 [9 月 13 日方法讨论](2026-09-13_method_ideas_after_mind_ablations.md)和[有向迁移设计](2026-09-13_directed_interest_transition_design.md)。其中已经提出 OI 支持擦除、单源迁移及双源组合。本次不是把它们当作新发现，而是用更细的相关工作对照缩小首版范围。特别是“冻结基线＋单源残差＋多关系门控＋双源残差＋分阶段拟合”不适合作为当前最小方法：它会同时引入多种解释和多组必要对照。

**2．最直接的相关工作：哪些部分我们实际上已经覆盖**

| 论文 | 核验位置与原方法 | 可迁移思想 | 当前判断 |
|---|---|---|---|
| Wang et al., *LLMs for User Interest Exploration in Large-scale Recommendation Systems*, RecSys 2024 [1] | §3.1–3.3，PDF pp.3–4。用兴趣 cluster 的关键词表示历史；从输入的历史 cluster 组合预测一个新的 cluster，再以传统推荐器映射到 item。采用目标 cluster 平衡的训练数据，并使用高频历史组合。 | 在语义类别层面学习真实兴趣转移，压缩预测空间；数据构造比长解释更核心。 | 这是最直接的前例。本项目已有固定类别、Semantic-ID、平衡训练和 Stage-A 适配，不能再把这些当作新创新。原工业输入常取两个 cluster，训练规则明确目标不等于这两个输入 cluster；不能据此直接认定目标在用户完整原始历史均未见。 |
| Wang et al., *User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems*, ACL Industry 2025 [2] | §3–4，尤其 §4.1–4.2，PDF pp.3–5。novelty 模型产生候选，另一 alignment 模型从线上聚合反馈学习，并做 best-of-n 选择。 | “语言上合理”与“用户实际接受”是不同目标；应让真实行为监督决定排序。 | 我们已有真实 NI 标签，却没有同等线上反馈和曝光条件。单纯再加 LLM 审核、生成候选和重排，会增加成本并引入候选覆盖限制；不优先整套移植。 |
| Wu et al., *GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation*, 2025 预印本 [3] | §3.2–3.5；§5.4.2。兴趣量化字典、独立 GPT 预测下一 item 的量化兴趣标签，再条件化检索物品。未来兴趣分布用 one-hot CE 训练。 | 从历史抽取多个兴趣向量，不自动等于预测未来兴趣；需要显式未来目标监督。 | 我们的直接 SID 预测已体现这个原则。论文的 posterior 是它对未来兴趣概率分布的命名，不应自行解读为变分后验蒸馏。全套含量化、多模块及分阶段训练，不适合当前固定 108 类任务照搬。 |
| Lin et al., *When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation*（CoEA），2025 初稿、2026 v3 [4] | DSIE、PCO 方法部分与 Appendix A.4。长期群体身份／短期个体兴趣、relevance／novelty 两模型及周期性优化。 | 区分稳定偏好和近期变化，区分相关性与新颖性。 | 我们是静态 H50 的一次预测，没有相同周期反馈闭环。可借任务分解，不建议复制双模型和周期 DPO。其 novelty 口径也不能直接替代严格历史未见类别。 |

GemiRec 的 Category-Unseen Recall 是对“属于用户历史未见类别的未来点击物品”的 item recall；不是我们对兴趣类别等权的 Macro Recall。因此它是相关方法与评测先例，不是可直接抄入结果表的数值基线。

**3．历史遮蔽与兴趣扰动：有依据，但不能夸大新颖性**

| 论文 | 核验位置与具体操作 | 对我们有用的思想 | 必须保留的区别 |
|---|---|---|---|
| Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*, CIKM 2019 [5] | §3.6，PDF p.5。随机遮蔽序列 item 并恢复；原文讨论内部重构与未来预测不一致，并加入末位置预测训练。 | 通过输入缺失构造监督；尽量让辅助任务贴近实际预测位置。 | 随机 item mask 不等于整个兴趣类别消失；重构历史也不等于真实新兴趣预测。 |
| Zhou et al., *S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization*, CIKM 2020 [6] | §4.3.1–4.3.4，PDF pp.4–6。item–attribute、masked item、masked attribute、segment 四类预训练。 | 类别属性可以参与序列表征；辅助监督可以来自已有行为。 | masked attribute 是遮住 item 后恢复其属性；segment 是连续片段，不是删除所有同类行为。四种 loss 并不需要整体移植。 |
| Zhang et al., *CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation*, SIGIR 2021 [7] | §3.3.2–3.3.5，PDF pp.4–6。在 item／latent-interest 层区分重要和非重要概念，替换后构造正负视图并对比学习。§3.3.3 明确讨论 dropping 可行，而 replacement 保留长度和位置。 | 对历史证据做有结构的扰动，比无差别噪声更有针对性。 | 原方法替换重要证据通常形成负视图；我们计划保留同一真实未来 y 作为监督，但允许证据减少后置信度降低。不能把“兴趣扰动／删除／counterfactual”本身当作新贡献。 |
| Chen et al., *Intent Contrastive Learning for Sequential Recommendation*（ICLRec），WWW 2022 [8] | §4.1–4.2，PDF pp.4–5。对序列表征聚类，学习意图原型并处理 false negatives。 | 相似偏好的用户不宜不加区分地互为强负样本。 | 聚类原型代表历史隐意图，不保证是未来未见类别。现阶段不建议引入 K-means、双对比目标与伪意图分配。 |

这组论文支持“按任务缺失结构设计增强”，却不支持“加上整类删除就成为全新算法”。新颖性应来自特定的 OI→NI 证据条件变化及其验证，而不是名称。

**4．B+I 的融合与预测：最值得借的是候选条件化**

| 论文 | 核验位置与原方法 | 可迁移判断 |
|---|---|---|
| Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*（DIN），KDD 2018 [9] | §4.3，PDF pp.4–5。候选广告参与 local activation，针对每个候选分别对历史行为做加权聚合；原任务采用点击预测目标。 | 用同一原则让每个候选兴趣选择相关历史，是一个清楚的单点改动。不是照搬广告 CTR 任务。 |
| Xie et al., *Decoupled Side Information Fusion for Sequential Recommendation*（DIF-SR），SIGIR 2022 [10] | §4.2.3–4.2.4、§4.3，PDF pp.4–6。分别计算 item／attribute 注意力矩阵后融合，value 保留 item 表示；另有 attribute prediction。 | 标题和类别可承担不同作用。但其秩／梯度分析针对特定向量融合结构，不能用来证明预训练 LLM 的“[类别]标题”token 拼接有相同缺陷。逐层改 LLM 注意力侵入较大，不是首选。 |
| Zhang et al., *Feature-level Deeper Self-Attention Network for Sequential Recommendation*（FDSA），IJCAI 2019 [11] | §3，尤其 item-level／feature-level self-attention。分别建模 item 序列和 feature 序列，再融合表示。 | 是 B/I 分别建模的明确先例；双路本身新颖性有限，且两套序列模块可能超过当前需要。 |
| Hou et al., *Towards Universal Sequence Representation Learning for Recommender Systems*（UniSRec），KDD 2022 [12] | §2.1–2.4，PDF pp.2–5。文本表示、可学习适配与序列对比目标，面向跨域通用表征。 | 用共享语义表示连接 item／类别是合理方向；但当前单域、固定词表且已有 LLM/SID，尚未显示需要再加域适配器。 |
| He et al., *Translation-based Recommendation*（TransRec），RecSys 2017 [13] | §3.2–3.3。用 previous-item embedding + user translation 接近 next-item embedding，另有全局平移和 item bias，优化序列排序。 | “迁移不等于相似”表达简洁，但原模型依赖用户参数和上一 item，不直接适配未见用户。若仅改写成 `-||r(H)-e_c||²+b_c`，展开后就是带类别偏置的原型点积分类器，不能仅因叫 translation 就认为引入了新结构。 |

DIN 原文的 attention 不强制归一化，以保留行为强度。我们可以先采用归一化版本控制长度效应，但必须称为受 DIN 启发的适配，不能声称逐项复现。历史 contextual hidden state 已包含其它行为的信息，attention 权重不等于独立证据，更不是兴趣转移的因果解释。

**5．首选方案：同一未来目标，直接历史证据缺席的 SFT**

最小研究假设：OI 中目标类别已出现，可能存在复现类别的快捷方式；NI 中这一直接证据缺失。构造更贴近这种缺失条件的训练输入，可能比单纯混入更多 OI 更有帮助。这是待检验假设，现有 Mixed 增益未证明其机制。

使用已有真实 OI 样本 `(H,y)`，其中 y 是该样本未来确实点击的类别。把所有属于 y 的可见历史行为及其标题一起删除，得到 `H^{-y}`，仍监督同一个 y。示意：

```text
原历史：跑步、运动营养、马拉松、运动营养
真实未来点击类别：运动营养

辅助输入：跑步、马拉松
辅助输出：运动营养的 SID
```

与上一轮“从历史随便选一个类别再遮蔽”的想法相比，这个版本保留了真实未来点击监督，避免把任意历史类别恢复都视为未来预测训练。

对模型，只需要一个原有 SFT 目标：

\[
L(\theta)=\mathbb E_{(x,y)\sim D_{\mathrm{train}}^*}
\ell_{\mathrm{native\ SFT}}(\theta;x,\mathrm{SID}(y)).
\]

`D_train*` 按预先冻结的规则混合真实 NI、原 OI 和删证据 OI。首版可用固定比例替换 OI 曝光，避免凭额外更新获得优势。沿用原 native full-vocabulary completion-only SFT 和监督后缀；不偷换成一个新 108-way CE head。推理保持原 History→SID 路径。

具体边界：

- 保留真实 NI 主任务；删证据 OI 是合成输入辅助任务，y 相对原完整 H₀ 仍是旧兴趣，不能称新增真实 NI 标签。
- 删除后若为空，不强迫模型凭空推断；记录不可用样本及类别覆盖变化，所有对照使用相同可用源样本。
- 重建 prompt、编号和任何历史派生字段。既有 OI prompt 明确要求输出历史已见类，删完证据后不能继续使用这一矛盾指令。
- 不要求删前／删后的概率分布相等，也不强制 `p(y|H^{-y})` 高于 `p(y|H)`。信息减少可以合理地降低置信度。
- 按用户先划 fit／validation，再构造所有 OI／NI／删证据视图，不能让同一用户派生视图跨分割。
- 删历史是模型输入干预，不是改变真实用户行为的因果实验。

最小对照以真实 NI 为评测目标：原 Mixed；同源 OI 的等量随机删除；目标类别全删除。保持源样本、监督目标、曝光预算、解码及选择标准可比，并报告长度／prompt 类型的剩余差异。随机删除若仍保留目标类别，应使用语义正确的 OI 指令；进一步的统一中性指令对照用于区分 prompt 切换和证据变化，不得给错误任务指令来制造差距。

如果收益只存在于人工删证据测试而没有真实 NI 增益，应停止推广；如果等量随机删除复现全部收益，应将其解释为一般正则化；如果只靠新增步数获得收益，不支持兴趣迁移机制。

该方案架构最简，适合优先验证；操作本身不足以承诺强算法新颖性。若要发展为论文贡献，需要证明这种监督与真实 NI 的特定关联，以及超越随机增强和等预算训练的泛化收益。

**6．互斥备选：候选兴趣条件化的单层历史聚合**

若允许改变评分头，可以借 DIN 的核心机制。令每条历史的联合标题／类别表示为 h_i，候选兴趣 c 的表示为 e_c：

\[
\alpha_{ci}=\operatorname{softmax}_i a_\phi(h_i,e_c),\qquad
u_c=\sum_i\alpha_{ci}h_i,\qquad
s_c=g_\phi(u_c,e_c).
\]

同一用户面对“AI 与机器人”时，可以更多利用电影制作技术类标题；面对“名人与创作者文化”时，可以更多利用演员生活类标题。示意不是已验证案例。关键是按候选改变历史聚合，而不是先生成一段笼统偏好摘要。

首版只做一层共享候选条件化聚合，全部 108 候选并行打分；不叠加单源／双源分支、图传播、原 SID logit 残差或额外 LLM judge。若使用已有 LLM 的历史表示，可以一次 backbone 前向后做 108 个小 query，不需要 108 次完整 LLM 推理。

这会成为判别式类别评分头，argmax 可映射到 SID，但不是原始自回归 next-token SID 生成。如果必须保持生成式接口，优先方案 5；不能把这一架构变化藏在“输出仍然是 SID”的表述中。

最关键的对照是：同 backbone／相近参数和训练预算的最后位置普通分类头、候选无关聚合头、候选条件化聚合头。再加入保持类别序列不变的同类别标题替换诊断。如果替换标题几乎不影响结果，不能宣称利用了标题中的个体偏好；若普通 head 复现收益，不能将改善归因于候选条件化。

**7．不优先采用的组合与论文定位**

- 不同时叠加双分支门控、重加权、遮蔽、关系图、CoT、GRPO。每个因素都可以被单独研究，当前没有证据支持组合的必要性。
- 不复制 GemiRec 的兴趣量化和三模块训练：固定 SID 已给定兴趣空间，不必重新学习一套字典来证明任务成立。
- 不复制 CoEA 的双模型周期优化：本任务的信息与反馈条件不匹配。
- 不把 DIN attention、BERT mask、类别 embedding 或 TransRec 距离换名后当新算法。论文贡献须来自明确的任务缺口及排除替代解释的证据。
- 单正例 softmax CE 可以拟合当前采样目标的条件分布；它本身不等于断言其它类别永远不受喜爱。不能仅凭单正例设置就宣称应换 PU loss。

最值得凝练的研究问题是：**能否通过改变旧兴趣监督中的直接证据可用性，在不增加推理复杂度的情况下改善真实未见兴趣预测？** 若答案是否定的，再考虑候选兴趣条件化聚合；不要预先把两个假设绑成一个方法。

另一个很小的数据对照来自 Wang 2024：每个目标类样本数相等，不保证其来源历史足够多样。可以审计每类 200 条是否被少数重复的来源兴趣组合占据，再做相同类计数和预算的来源多样性采样对照。原论文选择各目标最频繁的 10 种来源组合，并不证明“越多样越好”；这只是受其构造启发的新对照，适合作为数据配方分析，不能单独包装成主要算法。

本次最终采用的原文及三份逐篇证据笔记保存在 `artifacts/paper-transfer-review-20260915/`，来源 URL、文件字节数与 SHA-256 见其中 `source_manifest.json`。调研不是穷尽式系统综述，不据未找到完全一致论文来保证首创。

**原始文献与核验链接**

[1] Jianling Wang et al. *LLMs for User Interest Exploration in Large-scale Recommendation Systems*. RecSys 2024. [原论文 HTML](https://arxiv.org/html/2405.16363v2)；[正式 DOI](https://doi.org/10.1145/3640457.3688161)。

[2] Jianling Wang et al. *User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems*. ACL Industry 2025, pp.996–1003. [正式论文页](https://aclanthology.org/2025.acl-industry.70/)。

[3] Zhibo Wu et al. *GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation*. 2025 preprint. [原论文 HTML v1](https://arxiv.org/html/2510.14626v1)。

[4] Hongxiang Lin et al. *When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation*. [原论文 HTML v3](https://arxiv.org/html/2508.00450v3)。使用预印本身份。

[5] Fei Sun et al. *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*. CIKM 2019. [arXiv](https://arxiv.org/abs/1904.06690)。

[6] Kun Zhou et al. *S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization*. CIKM 2020. [arXiv](https://arxiv.org/abs/2008.07873)。

[7] Shengyu Zhang et al. *CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation*. SIGIR 2021. [原论文 HTML](https://arxiv.org/html/2109.05261)；[DOI](https://doi.org/10.1145/3404835.3462908)。

[8] Yongjun Chen et al. *Intent Contrastive Learning for Sequential Recommendation*. WWW 2022. [arXiv](https://arxiv.org/abs/2202.02519)。

[9] Guorui Zhou et al. *Deep Interest Network for Click-Through Rate Prediction*. KDD 2018. [arXiv](https://arxiv.org/abs/1706.06978)。

[10] Yueqi Xie, Peilin Zhou, Sunghun Kim. *Decoupled Side Information Fusion for Sequential Recommendation*. SIGIR 2022. [arXiv](https://arxiv.org/abs/2204.11046)。

[11] Tingting Zhang et al. *Feature-level Deeper Self-Attention Network for Sequential Recommendation*. IJCAI 2019. [正式论文页](https://www.ijcai.org/proceedings/2019/600)；[原论文 PDF](https://www.ijcai.org/proceedings/2019/0600.pdf)。

[12] Yupeng Hou et al. *Towards Universal Sequence Representation Learning for Recommender Systems*. KDD 2022. [arXiv](https://arxiv.org/abs/2206.05941)。

[13] Ruining He, Wang-Cheng Kang, Julian McAuley. *Translation-based Recommendation*. RecSys 2017. [原论文 HTML](https://arxiv.org/html/1707.02410v1)；[DOI](https://doi.org/10.1145/3109859.3109882)。
