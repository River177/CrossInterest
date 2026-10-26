# 从 CIR 增益到用户条件化的跨兴趣迁移

调研与本地结果核查日期：2026-09-24。范围：方法讨论、相关工作原文核验、下一步研究建议。本次没有运行新的训练或评价，没有改变数据、模型或既有实验协议。本文区分已观察结果、待验证机制和候选方法，不保证文献检索穷尽或方法首创。

**建议主线：研究 CIR 中超出类别先验与简单共现的用户条件信息，怎样迁移到真实新兴趣预测；用一个针对性的训练改动强化这种信息。** 历史集合编码、整类遮蔽和通用蒸馏都有直接或相邻先例，不宜单独承担核心 novelty。

## 1. 目前证据指向哪里

下列数值来自当前本地报告和紧凑验收产物，不使用旧记忆中的实验状态。

| 当前观察 | 已核实数值 | 能支持的判断 |
|---|---|---|
| 原始顺序没有显示明确的总体 Recall 优势 | NI-only 原始→原始、原始→打乱、打乱→原始、打乱→打乱的 Macro R@10 为 20.3275%、20.1622%、20.3258%、20.2758%；五个预设 R@10 配对区间均包含零 | 值得检验类别、频次、内容及其组合是否是主要信号；不能断言四组等价或时序无用 |
| 具体预测仍受排列影响 | 原模型 Top1 改变 28.4021%，Top10 平均集合重合 82.9527%；打乱训练后类似 | “指标接近”不等于模型是排列不变函数 |
| CIR 预训练有正向观察 | CIR1/3/5→NI3 的 R@10 为 21.2786/21.8272/21.9717%，NI-only 为 20.3275% | 历史重建监督具有迁移价值的候选证据，但尚未排除额外训练和执行差异 |
| 收益仍在未见候选之间 | 排除历史类别后，NI-only/CIR5→NI3 的 R@10 为 20.4186/22.0009%，差距 +1.5823 个百分点，原差距 +1.6442 | 单纯减少旧兴趣占位不足以解释 @10 收益 |
| 名称与 SID 对齐不是完整解释 | NI 前保留 SID 的名称映射 Top1 为 67/108，但历史重建 R@10 仅 9.46%；完整 CIR 为 43/108 和 21.53% | 应关注历史条件表征与输出空间的共同适配，不能只提高名称对齐 |
| 完整继承优于局部重置 | 完整继承、保留 LoRA、保留 SID 的 NI R@10 为 21.9717/20.8325/21.2467% | 不能把收益唯一定位于某个参数块；这是初始化干预，NI 阶段 LoRA 与 SID 均继续训练，预训练基座冻结 |

来源：

- [历史顺序消融](../experiments/2026-09-23/mind_history_order_ablation.md)。只有一次固定排列、seed42；原提示词保留 `in chronological order`；长度≥2子集的下降比全量大。NDCG 与 Recall 的变化也不完全一致。
- [CIR schedule 汇总](../experiments/2026-09-18/mind_cir_four_conditions.md)。CIR5→NI5 的 R@10 降至 20.5892%，这提示进一步诊断，但不能仅凭终点变化证明遗忘。
- [候选竞争诊断](../experiments/2026-09-24/cir_candidate_competition_results.md)。历史 NI-only 与 CIR 有 FLA/Torch fallback 等执行差异，不能据此做严格因果贡献分解。
- [参数迁移与功能诊断](../experiments/2026-09-23/mind_cir_parameter_transfer_results.md)。完整继承四项主要 NI 指标均最高。

当前 CIR 从最后 50 条历史里删除某一兴趣的全部行为，保留至少 5 条行为及其他兴趣，不回填旧历史；每用户一个固定视图，所有 epoch 重用。17,271 条、108 类，每类 3–1,056 条，原 NI 则为 108 类各 200 条。未来 NI 标签不参与删除类别选择。这个明显不同的标签分布意味着，CIR 改善可能同时来自类别先验、优化初始化和用户条件信息。

高频 CIR 与单次 CIR 的现有比较不能单独证明“稳定兴趣更可迁移”：目标分布 TV=0.61247，覆盖 75/108 类，平均删除 6.44/1.00 条，且 R@20 的方向与 R@10 不同。见[配对频率报告](../experiments/2026-09-17/mind_cir_frequency_paired.md)。

## 2. 对两项核心现象的解释

一个值得检验的假说是：当前任务主要要求模型从一个不完整的兴趣画像，推断哪些尚未观察到的兴趣与该用户相容。这个画像首先是带有频次和内容的集合，而非仅由相邻转移决定的序列。

这里的“集合”应保留多重性：打乱实验没有删除重复行为，也没有证明频次、历史长度或最近行为选择无用。直接把历史去重为类别名，会同时丢失标题内容和兴趣强度，超出了现有证据。

CIR 为这种条件推断提供了历史内监督，但它的目标是“已经观察过、现被人工隐藏的兴趣”；NI 的目标是“冻结历史从未出现、未来日志点击的兴趣”。两者的可见性条件相似，目标的采样过程并不相同。**最有研究价值的问题是两者之间究竟迁移了什么，而不只是增加多少重建样本。**

需要区分三个层次：

1. 类别先验：不看具体用户，某类也容易被预测。
2. 独立关系：看到一个历史兴趣，就增加某个目标兴趣的得分；多个历史兴趣只把这些证据相加。
3. 用户条件关系：同一个历史兴趣，在不同内容或其他兴趣组合下，对目标的支持会改变。

现有结果尚未区分这三层。第三层中的高阶组合效应尤其不能由“顺序不敏感”直接推出。

## 3. 最接近的相关工作及可借鉴点

本次通过原始 arXiv HTML、论文摘要页、正式出版页核查。论文原始指标不与当前 GNIP 指标直接比较。

| 工作与核实身份 | 原文实际方法 | 对我们的启发与新颖性边界 |
|---|---|---|
| **BTBR — Masked and Swapped Sequence Modeling for Next Novel Basket Recommendation in Grocery Shopping**, RecSys 2023 [1] | Select masking 遮住选中 item 的全部出现，阻断重复购买捷径；另有 explore-only basket 目标以及 cloze 预训练→未来 basket 微调 | 最接近 CIR 的先例。“全部出现遮蔽＋预测未见目标”已有。整类语义缺失更强，但不能仅因粒度变化就承诺充分 novelty |
| **Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations**, RecSys 2024 [2] | 15 个数据集；打乱测试历史，比较准确率与 TopK 列表重合，强调模型和预处理依赖 | 同时考察效果和预测稳定性；不能从一个模型的一次打乱推断数据没有时序信息 |
| **LLMs for User Interest Exploration in Large-scale Recommendation Systems**, RecSys 2024 [3] | 两个历史兴趣 cluster→新 cluster，类别平衡 SFT，再限制传统 item 推荐器的候选范围 | 兴趣级预测、小规模历史组合与平衡训练已有。可以借其聚焦语义兴趣层的思路，但须回答我们超出了哪些简单组合关系 |
| **User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems**, ACL 2025 Industry [4] | novelty LLM 提案，alignment LLM 学习聚合真实反馈，再做 best-of-N | 合理的新兴趣不自动等于用户喜欢；但双模型及 novelty/relevance 分解已有，当前也没有同等线上反馈条件 |
| **GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation**, 2025 预印本 [5] | 兴趣量化字典、用户条件 GPT 预测下一兴趣分布、兴趣条件化召回 | 直接预测未来兴趣有意义；重新加 RQ-VAE 字典或多兴趣头并非当前固定 SID 任务的必要改进 |
| **When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation**, CoEA，2025 预印本，2026-03 v3 [6] | 长期群体画像＋去重短期类别集合，novelty/relevance 模型与周期 DPO | 类别集合输入、群体画像与生成新类别已有直接碰撞。值得关注个体条件信息是否被群体/类别先验淹没，而非照搬系统 |
| **Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?**, arXiv 声明 RecSys 2026 accepted extended abstract [7] | 强 pairwise/recency 探针在一些数据集接近或超过复杂模型，另一大数据集仍有差距 | 必须先排除简单共现和独立兴趣证据求和的解释，再宣称高阶兴趣组合。它仍使用 recency，不是“顺序无用”的证据；该文为进行中的 extended abstract |

两个辅助边界：

- **CauseRec**, SIGIR 2021 [8]，已经在 item/interest 层做有结构的替换与对比学习。因此“兴趣扰动＋对比损失”本身也不足以新颖。
- **Decoupled Knowledge Distillation**, CVPR 2022 [9]，已经区分 target 与 non-target 蒸馏；把 CIR teacher 的未见非目标类做 KL 可以是改进，但不应仅靠此操作作为创新。

## 4. 首选方法方向：在 CIR 中显式学习条件支持

### 4.1 要解决的具体问题

普通 CIR 的训练信号混合了“这个兴趣一般容易出现”和“这位用户的其他兴趣支持它”。如果后者才是可迁移到 NI 的部分，那么可以直接让辅助训练区分二者。

研究假说：**在相同目标兴趣及匹配的历史统计下，学习真实用户历史提供的额外预测证据，比只拟合历史恢复概率更有利于 NI。** 这是候选假说，当前没有结果。

### 4.2 一个最小实现

继续使用既有 CIR 目标 `(H^{-c}, c)`：删除兴趣 c 的全部直接行为后，预测其 SID。为该目标找一份来自其他训练用户的匹配参考历史 `H_tilde`，两份可见输入都不包含 c。

参考输入也来自同一份冻结 CIR 擦除视图池，优先在合法匹配约束下做一对一错排，使每份历史在真实侧与参考侧各出现一次，目标类别边际也保持不变。避免真实侧用擦除历史、参考侧用自然完整历史，或少数 donor 被反复使用；这些差异会暴露构造来源。若约束只能满足一部分样本，所有比较臂使用同一可匹配子集；无法一对一时必须报告重复权重与剩余失配，不能声称两侧历史边际相同。

匹配历史长度、可见兴趣数、删除强度等显而易见的统计捷径；若研究类别组合，不应把完整类别直方图也匹配成完全相同。若专门研究标题的增量信息，则可以另设保持类别结构的内容对照。两者检验不同问题，不能混在一个“匹配历史”名称中。

设 `z_theta(c | H)` 为原模型在 SID 预测位置对 c 的 logit。先在固定 108 个 SID 上减去该输入的平均 logit，得到 `s_theta(c | H)`；这保留每个输入的类别排序，并消除所有候选共同平移的自由度。不要直接对未经中心化的 raw logits 做跨输入差值。定义：

\[
\Delta_\theta(c;H^{-c},\widetilde H)
=s_\theta(c\mid H^{-c})-s_\theta(c\mid\widetilde H).
\]

在原生 CIR full-vocabulary completion CE 外，只增加一个配对项，例如：

\[
L_{\mathrm{CIR}}^{*}=L_{\mathrm{CIR,CE}}
+\lambda\,\mathbb E\left[\log(1+\exp(-\Delta_\theta))\right].
\]

随后按现有流程进行同样的 NI 训练；NI 阶段不改目标、接口、候选池或推理过程。图网络、新 encoder、teacher 和 RL 都不是这一原型的组成部分。

这个形式的意义：若 logit 有可加的类别偏置 `b_c`，它会在同类别跨历史差值中抵消。模型需要利用历史变化来满足额外监督。它不保证消除所有频次效应、共享表示偏差或匹配混杂，也不应被直接称为无偏估计。

### 4.3 必须诚实说明的限制

- 参考历史与 c 的错配是统计对照，不代表这个用户不喜欢 c。存在支持 c 的参考历史是正常情况；配对目标只在总体联合关系与重配对分布可区分时有用，不能要求所有个体样本都满足严格偏好事实。
- 参考分布必须预先定义并记录。筛除可见 c、匹配长度等操作会改变参照分布，不能不经推导就把差值叫作 PMI 或因果效应。
- 真实/参考历史的生成流程必须对称审计。若一个输入可从删除长度、残留编号、类别数或用户来源识别，模型可能学习增强痕迹。
- 中心化消除了共同平移，但没有消除 logit 尺度。已有正差也可能通过放大尺度降低配对损失；辅助损失下降或差值变大不能作为机制成立的证据，需同时查看配对辨别、类别排名、实际 NI 效果与尺度变化。
- 目标类别的边际频率、有效源样本、总更新量及 NI 阶段需匹配；额外参考前向有计算成本，即使 optimizer steps 不变也不是算力等价。
- 不增加真实多正例训练，也不读取 Dev 未来标签来筛选参考历史。参考关系只在 Train 内构造。
- 配对 logistic、NCE 与条件密度比思想是已有技术。可争取的贡献是识别并解决“历史恢复→真实 NI 迁移中的条件信息混杂”，而不是声称发明对比学习。

如果用户历史错配仍能保留大部分 CIR 收益，应暂缓这个方法，优先研究类别分布或优化初始化。如果简单共现能复现效果，则应将方法定位为有效的共现迁移，而非高阶组合推理。

与当前 MeRA 式全局 SID 关系正则的区别在约束对象：这里约束同一个候选在不同用户历史下的条件预测变化，而不是把两类 SID 向量固定拉近。不同用户可以对同一对候选产生不同的相对支持。这个区别给出机制上的可检验对象，但仍不是独占性的算法新颖性保证。

## 5. 怎样让 novelty 更有说服力

最值得形成的命题是：

> 从历史重建中提取用户条件化的跨兴趣关系，并使其泛化到真实未来未见兴趣以及训练中稀少的兴趣组合。

它需要三方面共同成立：

1. **明确迁移对象。** 类别先验、输出名称对齐、简单共现与用户条件贡献有可区分证据。当前候选过滤和名称探针已经缩小了一部分解释空间，但仍不够。
2. **方法确实针对该对象。** 训练改动必须增强条件关系，而不仅是增加样本或优化步数。匹配重配对 CIR 是一个直接的竞争解释。
3. **泛化测试与主张一致。** 除原 Full Dev 外，按预定义 source-set→target 模式做稀少/未见组合诊断，同时保留各单独 source 和 target 的训练覆盖，控制目标频次、历史长度与语义距离。集合大时 exact-set 天然稀有，不能把这种稀有直接当作组合泛化证据。

组合诊断先作为附加研究协议设计，不替换当前 Full Dev，不依据最终 Dev 结果选择模型或调整组合划分。若只检验 pairwise 边，而未控制这些边的训练覆盖，也不能宣称高阶组合泛化。

可能的概念示例：相同的“运动”兴趣，和“营养”共同出现时对一个未见目标的支持，是否不同于它和“赛事资讯”共同出现时的支持。例子只说明要测量的依赖，不是真实实验案例，也不预设组合一定比单兴趣更强。

## 6. 两条备选路线及采用条件

**标题中的语义桥接。** 如果保持类别与频次、替换同类别标题会明显削弱 CIR 优势，说明粗类别之外的内容值得建模。可进一步研究候选兴趣如何从标题中选择证据。候选条件化 attention 已有 DIN 等先例，需证明解决了“类别层看不见的迁移线索”这一具体缺口。没有该证据前，不建议新增解释生成、知识图谱或多头融合。

**保留 CIR 的可迁移能力。** 只有观察到有用条件关系随 NI 微调被破坏，再考虑重放或函数蒸馏。非目标未见类别的归一化 KL 保存相对形状，不保存总概率；teacher 也可能传递错序或类别偏差。它更适合作为优化修补，目前不宜作为主创新。CIR5→NI5 的下降本身不能证明遗忘。

集合压缩、按兴趣聚合并保留频次与标题，是独立的效率方向。如果它用更少 token 保持效果，可以形成良好的工程结果，但不足以单独支撑上述机制贡献。

## 7. 建议下一步的最短证据链

这是讨论建议，尚未执行，也未授权新增实验预算。

1. **先区分信号。** 用现有模型做匹配用户历史交换、保持类别/频次的同类别标题替换；加入 train-only 类别先验及平滑共现/独立兴趣证据求和探针。扰动只用作机制筛查，不能替代正常 NI 测试。
2. **只在条件信息成立时改一个训练点。** 比较原 CIR、保持标签边际的匹配重配对 CIR、条件支持 CIR，统一可用源数据、NI 配方与训练预算；既有 NI-only 为背景，严格归因还需要同数值环境与预算控制。具体矩阵应在执行前冻结，不顺带扩展 seeds/sweeps。
3. **将机制与最终效果共同作为接受标准。** 原 Full Dev NI 有收益，同时 matched-history 依赖与组合分层符合预设假说。若仅分类别指标改善、条件诊断不变，不能宣称实现了用户特定迁移。

目前最优先的研究问题：**CIR 究竟让模型学会了更好的全局共现，还是学会了“这个用户为什么可能喜欢某个未见兴趣”？** 方法设计应由这一答案决定。

## 原始文献

[1] Ming Li et al. *Masked and Swapped Sequence Modeling for Next Novel Basket Recommendation in Grocery Shopping*. RecSys 2023. [原文 §4.2](https://arxiv.org/html/2308.01308v1#S4.SS2)；[DOI](https://doi.org/10.1145/3604915.3608803)。

[2] Anton Klenitskiy et al. *Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations*. RecSys 2024. [原文 §3.2](https://arxiv.org/html/2408.12008v1#S3.SS2)；[DOI](https://doi.org/10.1145/3640457.3688195)。

[3] Jianling Wang et al. *LLMs for User Interest Exploration in Large-scale Recommendation Systems*. RecSys 2024. [原文](https://arxiv.org/html/2405.16363v2)；[DOI](https://doi.org/10.1145/3640457.3688161)。

[4] Jianling Wang et al. *User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems*. ACL 2025 Industry. [正式出版页](https://aclanthology.org/2025.acl-industry.70/)；[原文 v4](https://arxiv.org/html/2504.05522v4)。

[5] Zhibo Wu et al. *GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation*. 2025 预印本；本次未确认正式 venue。[原文](https://arxiv.org/html/2510.14626v1)。

[6] Hongxiang Lin et al. *When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation*. 2025 预印本，2026-03 v3；本次未确认正式 venue。[原文](https://arxiv.org/html/2508.00450v3)。

[7] Aleksandr V. Petrov et al. *Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?* 2026-08 预印本；arXiv 记录声明 RecSys 2026 accepted extended abstract，HTML DOI 仍为占位，不用其占位 DOI。[arXiv 记录](https://arxiv.org/abs/2608.19833)；[原文 §2](https://arxiv.org/html/2608.19833v1#S2)。

[8] Shengyu Zhang et al. *CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation*. SIGIR 2021. [原文](https://arxiv.org/html/2109.05261v1)；[DOI](https://doi.org/10.1145/3404835.3462908)。

[9] Borui Zhao et al. *Decoupled Knowledge Distillation*. CVPR 2022, pp.11953–11962。本次核验正式出版摘要中 TCKD/NCKD 分解，不把视觉实验结论直接迁移到推荐。[正式出版页](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)。

补充阅读：S2SRec2 的无序 recipe completion 是已有集合方法实例，但真实用户兴趣没有 recipe 那样的完整集合标签；UpliftRec 的曝光干预依赖因果假设，不适用于当前冻结日志的因果宣称。原文核查笔记与来源收据保存在 `artifacts/research-20260924-interest-transfer/`。
