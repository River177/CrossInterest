# CrossInterest：从旧兴趣证据学习新兴趣迁移

最值得推进的主线是：**让旧兴趣训练学到可迁移的证据，而不是仅仅增加旧兴趣预测样本。** 近期可先验证“同一目标的直接历史证据有／无”成对训练；如果有效，再发展为用户条件化、有方向的兴趣迁移模型。曝光条件学习是另一条问题更基础、但数据前置条件更重的路线。

以下提出三个候选方法。它们是经相关工作对照后仍有研究空间的假设，尚未经过实验，不构成首创保证。检索包含截至 2026-09-13 可访问的原始论文，其中覆盖 2026 年 9 月的新预印本；预印本中的模板年份、会议占位符与未核验 venue 不作为正式发表证据。

**一、当前实验到底支持什么**

任务维持为：给定用户历史，预测未来日志中点击、且没有出现在完整原始 History 中的语义兴趣。标签构造先使用完整原始 History 判定新颖性，随后模型输入才截取最近 50 条以内、按时间保留重复行为的新闻标题与兴趣类别，输出为 108 类中的一个 SID。下文以 H₀ 表示完整原始 History、H 表示模型可见的 H50。这里的“新”是相对于可观测历史窗口，不能推断为终身第一次兴趣或推荐行为创造的兴趣。

最新实验的正式指标如下。原始结果、训练协议与重算路径见[审计笔记](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/research-ideas-20260913/audit_notes.md)和[正式四表](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-four-experiments-20260911/result_tables.md)。

| 条件 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B NI-only | 20.23% | 33.77% | 10.35% | 13.74% |
| Qwen3.5-4B NI-only | 20.48% | 33.55% | 10.21% | 13.50% |
| Qwen3.5-2B NI-only | 20.33% | 33.66% | 10.13% | 13.47% |
| 2B + SID warm-up | 20.45% | 33.57% | 10.26% | 13.56% |
| 2B Two-stage：OI→NI | 20.44% | 33.96% | 10.34% | 13.75% |
| 2B Mixed：OI+NI | 20.61% | 34.20% | 10.42% | 13.82% |
| 2B MeRA corrected | 15.63% | 28.62% | 7.57% | 10.84% |

扩大 backbone 没有单调收益，SID warm-up 的收益也不一致。Mixed 相对 NI-only 的 R@10/R@20 增益为 0.2821/0.5461 个百分点，因此更值得研究任务间的关系。不过，NI-only 是 2,025 次更新，Mixed 是 3,630 次，增加约 79%；旧兴趣 17,098 条、每类 5–200 条，混入后还改变了类别曝光分布。这一比较尚未隔离旧兴趣内容、训练预算与类别分布的作用。

Two-stage 与 Mixed 使用相同旧／新样本曝光量和累计更新数。Mixed 连续单阶段训练；只有 Two-stage 在 OI→NI 边界重置优化器与 scheduler。因此其差异同时包含任务顺序与 schedule 边界，不能仅凭四个指标认定发生了灾难性遗忘或梯度冲突。MeRA 的输入、参数处理及训练轮数不同，当前优势属于完整方案比较，不能单独归因于某个模块。

本轮对已验收的 31,392 个 Dev impression、37,728 个 target occurrence 做了配对只读重算，得到两个更具体的研究切口：

| 目标切片 | 目标数 | NI-only R@10 | Mixed R@10 | 可支持的判断 |
|---|---:|---:|---:|---|
| 目标与历史某类共享 parent | 14,631 | 33.85% | 34.66% | 相邻兴趣扩展相对容易 |
| 目标 parent 在历史中不存在 | 23,097 | 25.41% | 26.42% | 跨 parent 迁移占 61.22%，必须覆盖 |
| 历史长度 1–4 | 11,051 | 39.34% | 40.67% | Mixed 在短历史有较明显收益 |
| 历史长度 21–50 | 3,283 | 11.57% | 11.82% | 长历史仍是较难的观测切片 |

这些是 **target-occurrence Recall**，不是表一的 interest-macro 指标；parent 与历史长度切片依据模型可见 H50，区别于标签按完整 H₀ 判定新颖性。历史长度和目标类别、用户活跃度、类别新颖性程度等相互混杂，尚不能据此诊断 attention overload。跨 parent 也只是 taxonomy 距离，不能直接等同于语义很远或难以预测。这里的切片来自已看过的 Dev，只用于提出假设；方法选择需要另设 Train 内保留用户集。

还有一个实验判读前置问题：NI-only、Two-stage、Mixed 的第 10／11 名同分率分别为 70.81%、68.98%、70.99%。冻结代码没有在保存时 round；模型加载为 BF16，原始 reload 参考 logits 在 FP32 softmax 前就已落在 BF16 数值格点上。由此只能说排序分辨率需要核查，不能说现有结果无效，更不能说 FP32 必然提高成绩。下一轮应先对已验收 checkpoint 做一次全 108 logits、有效 SID head 的 FP32 最终投影与同分处理敏感性检查，再固定评估器。[重算数据及源文件哈希](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/research-ideas-20260913/prediction_diagnostics.json)

**二、相关研究已经覆盖了哪些“看上去很新”的想法**

| 研究 | 已有机制与已读位置 | 对本项目的约束 |
|---|---|---|
| Wang et al., RecSys 2024 | 从少量历史兴趣 cluster 生成 novel cluster；按目标类均衡并挑高频历史组合，随后用推荐器落到 item。§3.1–3.3。[^1] | 类别级新兴趣生成、balanced transition SFT 已有；本仓库也做过 Stage-A 适配。 |
| Wang et al., ACL Industry 2025 | novelty LLM 与 user-aligned LLM 分开训练，前者生成多个候选、后者 best-of-n 选择。§3–4.2，方法重点 §4。[^2] | “先发散新兴趣再个性化筛选”不是新贡献。 |
| GemiRec, 2025 preprint | 共享量化兴趣字典，独立 next-interest posterior，再条件化检索物品；§5.4.2 明确评估 Category-Unseen Recall。§3.2–3.4。[^3] | 不能声称首次生成未来兴趣、首次学习兴趣分布，或首次评估历史未见类别。其 CUR 是 item-level，不等于此处的类别 macro Recall。 |
| CoEA, 2026-03 修订预印本 | 长期群体身份与短期个体兴趣分离，relevance/novelty 模型用增量数据周期协作。[^4] | 仅加长期／短期双分支或 novelty/relevance 双模型，区别不够。 |
| CauseRec, SIGIR 2021 | 用目标相关性识别 indispensable/dispensable 概念；在 item 和 interest 两层替换概念并做对比学习。§3.3.2–3.3.5。[^5] | 整个兴趣层面的扰动、历史证据删除、目标条件重要性，均不能独立称新。 |
| RepeatNet, AAAI 2019 | repeat/explore 模式及两个 decoder，分别预测历史内外的 item。§3。[^16] | 旧／新双头已有；新 item 与新语义类别不同，但区别不足以使双头本身成为创新。 |
| CaseRec, SIGIR 2025 | 用曝光序列训练 decision transformer，替换曝光项，由用户模拟器补反馈，再训练推荐器。§3.3、§4.1。[^6] | “反事实增强探索新兴趣”已有；区别应是具体观测、监督与验证方式，而非换名。 |
| TAG / Auto-λ / Ren et al. | 任务梯度对另一任务损失的 lookahead affinity，或在验证损失上学习任务／样本权重。[^7][^8][^9] | “旧新任务自动加权”“用梯度挑有帮助的样本”已有，需要更细的迁移结构与强对照。 |
| ADT / LLMHD / BOD | 大 loss 去噪；LLM 区分 hard/noisy；双层优化与梯度匹配去噪。[^10][^11][^12] | 不能把“LLM 审核历史→target 后重加权”或“梯度去噪”作为独立新贡献。 |
| UGR, 2026 preprint | 不确定性 reward、按 target beam rank 优化难样本、层次 confidence alignment。§3.2–3.4。[^13] | 简单把 GRPO 换成 confidence/difficulty-aware，已经有极近的工作。 |
| LLM4AIGQ, 2026-09 preprint | single-interest SFT，多兴趣划分与 query 生成、多级 reward GRPO，再 DPO。§4.2–4.4。[^14] | “拆历史→多兴趣推理→多级奖励”也不足以区别于近期研究。 |

本仓库另已有 SID 双向对齐、global/within-class Cross-Fitted 权重、MeRA 同 parent／协同关系正则、taxonomy-restricted CE、preference-state reasoning 与 hierarchical GRPO 代码／数据。不同方案的完成状态不相同，但不能将这些既有设计重新包装成下一轮创新。[既有实现及结果定位](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/research-ideas-20260913/audit_notes.md:69)

特别需要避免一个统计误解：每个用户一个训练 target 的 softmax CE，不等于断言其余 107 类永远不喜欢。它可以正确拟合“被当前采样协议记录的一个 target”的条件分布。需要研究的是观测／采样机制、有限样本下的竞争以及对真实 NI 的泛化，而不是先宣布 CE 错误，再换一个 PU loss。PU/MNAR 的无偏结论需要相应的 propensity 与采样假设。[^15]

**三、Idea 1：直接证据有／无的成对新兴趣训练**

**问题与假设。** OI 任务的正确类别已经在历史中出现，模型可能通过读取重复类别、频率或相关标题就完成预测。NI 的目标恰好没有这些直接证据。普通 Mixed 同时优化两种任务，却没有要求模型学会“直接证据缺席时，仍然能从其它兴趣推断目标”。这只是待检验解释，当前指标尚未证明模型主要靠复制完成 OI。

**具体构造。** 取一个现有 OI 样本 `(H,y)`，y 是未来确实点击过的旧兴趣。构造两种输入：

1. 原始 `H`，使用旧兴趣提示词，仍预测 y。
2. `H^{-y}`，删除模型可见历史中所有属于 y 的行为及对应类别字段，重编号并重建 prompt；使用“给定这段历史推断未出现兴趣”的提示词，仍预测同一个 y。当前 B+I 没有摘要字段；若以后缓存摘要，必须由删后历史重新生成。

例如原历史含“跑步、马拉松、运动营养”，未来又点击了运动营养。第二视图移除运动营养历史后，模型要从跑步／马拉松推断运动营养。例子只说明输入变换，不是项目中已核验的成功案例。非目标类别中的合理共享实体可作为真实 bridge 保留，但任何单独缓存的 y 类标签、y 类标题或由完整历史生成的 target-revealing 摘要必须删除。

此构造**不增加新的未来点击标签**。它复用现有 OI target，改变其可见支持条件；同一用户的两个视图必须属于同一 fold。本地已核实 17,098 条 OI 的 target 均在 H50 出现；擦除后 69 条为空、227 条只剩一项。空历史没有跨兴趣证据，应从此辅助视图中排除或单列先验对照，真实 NI 与 Dev 分母不变。它不等于用户自然产生了新兴趣，因此应称“支持证据擦除辅助任务”，避免把合成样本计入真实 NI 测试或当作因果干预证据。

**模型机制。** 第一个可运行版本仅改变训练视图，辅助损失为：

\[
L_{erase}=-\log p_{OI}(y\mid H)-\lambda\log p_{NI}(y\mid H^{-y}).
\]

再与原有真实 NI loss 联合训练。**这两个独立 CE 本身没有跨视图耦合**，同权重下打散数据的目标完全相同，不能把“放在一起训练”称为新的 paired loss。它只是支持擦除基线。这里不要求两种输入的预测概率相等，也不蒸馏原始 H 的完整高置信分布：移除信息后，合理的不确定性可以增加。

若简单版本在真实 NI 上有效，可考察一个次级支持切换约束。使用相同提示词、同一个实际预测 head 与相同候选归一化，定义目标相对其余类别的 log-odds D_y(H)；分别擦除 y 类证据与等长度的非 y 历史，得到 Δ_target=D_y(H)−D_y(H^{-y}) 和 Δ_control=D_y(H)−D_y(H^{-control})。新增 `L_switch=max(0,m−(Δ_target−Δ_control))`。完整 H 项在差分中抵消，因此它实际上是两种删减视图的 pairwise ranking loss，要求保留直接 y 证据的对照视图获得更高目标分数，不能解释为独立测量了恢复证据的绝对收益。对照必须保留所有 y 行；非 y 行不足以等长度删除时，只跳过 switch 项并报告覆盖率，保留原 CE。

此约束与重要／非重要历史扰动已有近邻，不能提高新颖性评级。它与独立 CE 不同，但也可能强化复制、损害 NI，因此仅在擦除基线有效后考虑，比较 `L_erase` 与 `L_erase+L_switch`；m、权重和对照删除规则只在 Train 内选择。

另一个结构版本可增加直接支持／迁移两个残差分支，分别限制输入范围，并由上述差分约束支持分支。双分支本身已有先例，不应和配对约束同时加入后再统一归因。无论哪种版本，不能仅凭分支名字声称已实现 disentanglement，也不能把模型 log-odds 差当作真实用户兴趣的因果变化。

**与已有工作的实质差异。** BERT 式 mask 和 CauseRec 已经有历史／兴趣扰动，RepeatNet 也已有旧／新双头。这里值得研究的是 **同一用户、同一未来目标、仅目标支持条件变化，引起 OI→推断任务切换**。CauseRec 通常将重要概念替换视作负向表示扰动；此处即使目标直接证据移除，仍保留已观测 target 的监督，但允许预测置信度下降。与 CaseRec 不同，它没有生成未观测点击反馈，也不依赖用户模拟器。[^5][^6][^16]

**最小可证伪实验。** 固定 2B、原 NI target、108 类映射和 Dev；用同一批 OI 样本比较：原 Mixed、同等预算的随机历史删除、整类擦除的独立 CE、擦除 CE＋支持切换差分约束；若增加双分支，再加入双分支但无差分约束的消融。不能把独立 CE 数据打散当成有效的“无配对”消融。另保留 NI-only 重复采样以匹配更新预算。所有方法除更新数外还报告实际训练 token 数；新增 forward、重复 label 曝光不能被隐藏。

主要检验真实 NI 的 macro Recall/NDCG，附跨 parent、历史长度、target frequency 切片。新增一个 Train 内保留用户上的支持擦除诊断，用于检查原 OI 能否在失去直接证据后仍预测，而不是用 Dev 挑擦除策略。

**放弃条件。** 若提升只存在于合成擦除测试，真实 NI 不升；若与同 token 随机删除等效；或支持切换约束不优于普通整类 mask，那么它分别说明模拟目标不匹配、普通正则化收益，或方法区别不足。若差分约束增强了 OI 复制却损害 NI，也应删除这一机制。此时不应靠给它一个“counterfactual”名字继续作为主创新。

**判断。** 问题针对性高、现有数据即可构造、成本相对低；单独的 mask、配对构造或上述 ranking 正则新颖性均偏低。更有潜力的是把它作为下一条有向迁移方法的训练证据，证明真实 NI 的组合泛化，而非独立宣称一个全新的数据增强算法。它是最适合先验证的方向。

**四、Idea 2：用户条件化的有向兴趣迁移，而非静态兴趣相似度**

**问题与假设。** 同一个旧兴趣对不同的新兴趣可能有帮助、无信息，甚至误导。只学 SID 与名称对齐，或把同 parent／高共现兴趣拉近，并没有明确学习“这名用户的哪一段历史为什么支持这个新目标”。本轮 61.22% target occurrence 跨 parent，说明仅限制在 sibling 扩展的模型会遗漏大部分目标；但这不意味着必须给每个跨域点击编一个桥接故事。

**核心表示。** 从原始历史按已观测兴趣或可审计的行为子序列建立少量 anchor。对每个候选新兴趣 c，计算候选条件化、允许有正负号的关系贡献：

\[
s(c\mid H)=b(c)+r(c,H)
+\sum_a w(a,c,H)T(a,c,H)
+\sum_{(a,b)\in\mathcal P(H)}T_2(a,b,c,H).
\]

此式先描述模型族：`T` 拟表示有方向的迁移，`T_2` 拟表示两个旧兴趣的组合证据，`r` 是不指定来源的直接预测分支。若三者都任意读取完整 H，分解不唯一，任一项可以吸收其它项，因此不能直接解释所学边。实际实现应让单 anchor 分支只读对应子序列、组合分支只读指定两个子序列，用户状态仅通过受限的共享调节器进入；相对于同容量 r-only 和单 anchor 模型检验组合项的增量。仍需承认潜在来源不可唯一识别。从少量低秩关系算子与共享参数开始，不为每条边创建独立大网络，也不把权重图当作真实因果作用。

示意上，“运动”可能支持“运动营养”，但在“运动＋投资”与“运动＋育儿”的历史中，其应扩展的方向不必一样。模型要学习的是这种条件差异；不能只查一个固定 `sport→nutrition` 的共现表。保留 `r` 尤其重要：历史无法解释的真实新点击不应被强制匹配到虚构 anchor。

**可选的辅助训练分配信号。** 在 Train 内按用户分 fold，用某类 OI 更新对另一批真实 NI 用户的改善，估计参数训练的迁移效用。这里明确以 a 表示 OI 的监督目标类，z 表示原始支持视图或擦除视图，c 表示 NI 目标类：

\[
U_{a\rightarrow c,z}
=L^{NI}_{V_c}(\theta)
-L^{NI}_{V_c}(\theta-\eta g^{OI}_{a,z}).
\]

`V_c` 参与 utility/router 更新，属于 meta-training 数据，不能同时充当真正未见用户的验证集。应分开 inner-fit、meta-utility、final-validation 三层用户角色；edge/combination holdout 不得进入效用估计或超参选择，正式 Dev 也不参与。meta 数据仍带有原单 target 观测噪声，不是无噪声金标准。可以先在轻量 head 或 LoRA 子空间估计梯度内积，再抽样检验真实一步更新效果。按目标类宏平均，低支持组使用收缩估计并报告不确定性，避免头部类别和偶然几个用户控制路由。

此 U 衡量“学习识别 OI 类 a 是否帮助预测 NI 类 c”，**不等于用户历史中的 a 是 c 的行为来源**，不能作为 T 的 bridge 真值标签。它只用于辅助样本／视图分配，把选择标准从“这个 OI 样本容易学”改为“它是否帮助另一批用户预测某类 NI”。它还提供一个难例保护思路：OOF NLL 高但有稳定跨用户迁移效用的样本，不能仅因难度就降权；高 NLL 且效用未知也不能直接判为噪声。主模型先验证有向关系是否比直接 SID head 有价值，再决定是否加入此信号。

**最近工作与剩余差异。** GemiRec 已经学习 user-conditioned 的未来兴趣分布；普通多向量／字典不是新点。MeRA 已有结构与协同关系，本仓库还有同输入关系消融；普通关系图也不是新点。TAG、Auto-λ、Ren et al. 和 BOD 已有梯度效用／双层加权，不能将上述公式本身当发明。[^3][^7][^8][^9][^12]

有潜力成立的贡献是三者的结合：**源兴趣→目标兴趣的方向性、用户当前证据条件、在真实未见用户上的迁移效用**；并证明这种结构能迁移到未见的源—目标边或源兴趣组合。若只有常规 IID 指标微增，最多说明有效的结构设计；只有超越强对照的组合泛化与机制证据，才更适合作为方法论文主线。

**必须怎样验证。** 首先比较直接 SID head、静态共现／同 parent 关系、普通 candidate-conditioned attention、有向单 anchor 模型、有向＋双 anchor 模型。随后只对有效结构比较固定 OI 比例、task-level Auto-λ/普通验证梯度加权、细粒度迁移效用路由。结构与路由要分开消融，不能一次堆进去后把整体提升全部归给“迁移”。

保留正式 user-disjoint Dev 为连续性指标；另从 Train 构造独立的 source-target-edge 和 source-combination holdout，保持 target 类在训练中可见，同时避免保留边被另一个 anchor 重新引入。因为一个历史可能含多个 anchor，分割规则必须在训练前明确，报告被排除样本和每类支持数。所有方法共用同一分割。

对预测的 anchor 做删除／匹配替换敏感性分析，比较随机、等长度和普通 attention 选择；它检验模型是否使用了声明的输入证据，不证明删除该兴趣会改变真实用户偏好。尤其不能将“删除模型选出的重要项造成下降”本身当作因果证据或新颖贡献，CauseRec 已有很近的设计。[^5]

**放弃条件。** 若效果等同静态共现；去掉用户条件无损；打乱效用边仍有效；组合 holdout 无收益；或普通 Auto-λ 已能解释全部收益，就没有足够证据支撑这条主张。参数更多、训练 forward 更多也必须各有匹配对照。

**判断。** 论文潜力在三条中最高，机制与评估要求也最高。建议把 Idea 1 当作它的证据条件训练，而不是立即启动一个带 graph、CoT、GRPO 和 teacher 的庞大组合。

**五、Idea 3：利用训练时的曝光选择集合，区分观测机会与条件选择**

**问题与假设。** 新兴趣点击不仅取决于历史偏好，还取决于 source impression 展示了什么。给同一个历史，某类别未被记录可能因为没有机会看到；低模型概率不能判断标签无效。当前 108 类全局预测没有显式利用这一观测结构。Saito et al. 已说明曝光与偏好混合导致 MNAR，CaseRec 也利用过完整曝光。[^15][^6]

**保守且可实现的版本。** 保留每个用户原来的单个兴趣 target y，用其 source impression 回连实际展示集合，令 `A=(实际展示类别∩108类输出词表)−C(H₀)`，按原协议完整历史 H₀ 判定新颖性。新增一个条件选择 loss：

\[
L_{choice}=-\log\frac{\exp s_y(H)}{\sum_{c\in A}\exp s_c(H)},
\qquad L=L_{global}+\lambda L_{choice}.
\]

推理仍只有 H，对固定 108 类排序；A 不进入测试模型输入。共享 s(H) 的条件 softmax 假设类别效用不随 A 中其它选项变化，是一种简化选择模型，并非从曝光事实自动推出；它也没有自动建模类别内文章数量、位置或检查概率。保留 `L_global` 是为了维持原任务并锚定跨选择集合的分数尺度，防止小而不连通的 A 令全类别排序无法识别。未被选作监督的已曝光项也不能据此称为用户不喜欢，loss 描述的是当前单 target 协议中的相对选择。

若该最小版本证明有效，再考虑机会预测与条件选择分支。设 S=1 表示样本经过用户 hash 选择和目标类均衡后被保留，通过边缘化得到这一采样分布中的类别概率：

\[
p_S(c\mid H)=\sum_Ap(c\mid H,A,S=1)p(A\mid H,S=1).
\]

这是建模分解，不是可识别性证明。现有 target selection 先按用户 hash 选单条、再按目标类抽样，source A 也被这一采样条件筛选；在只看被选中 impression 时，学到的是采样后观测分布。不能直接把某个分支称为“真实偏好”，也不能把两分支乘积宣称为无偏估计。首次实验不应直接上大规模 slate 生成器。

**可能的研究贡献。** 显式处理“未来未见语义兴趣”的 source choice set，并检验是否改善跨 parent／稀有迁移，同时维持单个兴趣 target、History-only 推理和全类别评估；一个 target 仍可能由同一 impression 的多篇点击文章支持。区别于让 LLM 判断合理性，它新增的是日志事实；区别于 CaseRec，它不模拟新曝光点击。条件 likelihood 和 exposure-aware recommendation 已有大量先例，因此新颖性判断必须落在具体 GNIP 观测机制及实证分解，而不能只写“我们考虑曝光”。

**前置条件已经核验。** 21.6k SFT 保留了 source sample、split、impression ID、user ID 和 timestamp，原则上可以回连。但当前原始重分类 Train `behaviors.tsv` 与 `news.tsv` 缺失；dynamic 与 recovered Parquet 只有历史／点击，没有未点击 slate，恢复出的任务新闻目录也不保证覆盖所有未点击文章。**目前无法从现有缓存构造完整 A。** 必须先恢复原始展示数据并核验逐行覆盖，不能从点击列表猜曝光。

**最小实验及放弃条件。** 恢复后先检查 y∈A、A 大小、单类别 A 占比、新闻映射覆盖与选择集合连通性。比较 global CE、global+真实 A、global+同大小且始终含 y 的置换 A、global+类别频率校正。原始正例、预算与推理候选保持一致。如果真实 A 不优于置换 A，效果更可能来自正则化；若只有未来真实 slate 输入时有效，则不满足当前 History-only 问题；若跨 exposure strata 退化，不能声称改善了迁移机制。

**判断。** 问题基础且具体，创新空间中等，实施风险最高。适合独立后续路线，不应挡住前两条用现有数据可以完成的研究。

**六、推荐次序与实验决策**

| 次序 | 要回答的问题 | 通过后推进 | 不通过时的决定 |
|---|---|---|---|
| 0：评估核验 | 同分处理、有效 SID head 精度是否影响方法排序？ | 固定一个审计过的 evaluator 与统一候选规则 | 先报告精度敏感性，暂不解释 0.1–0.5pp 小差距 |
| 1：预算与数据对照 | Mixed 是否超越等预算 NI 重采样？ | 保留 OI 迁移作为主问题 | 不把现有 Mixed 增益写成迁移证据；可继续直接证据假设但需另证 |
| 2：Idea 1 | 支持擦除配对是否提升真实 NI？ | 加入有向迁移结构 | 如果等同普通 mask，保留工程增强，放弃独立创新主张 |
| 3：Idea 2 | 用户条件关系是否能泛化到未见边／组合？ | 形成方法主线，按需加迁移效用路由 | 若普通 attention/Auto-λ 解释全部收益，降级创新主张 |
| 4：Idea 3 | 展示事实是否带来超越频率／正则化的收益？ | 发展机会与条件选择分解 | 原始数据不全或真实 A≈置换 A 时停止 |

每一轮只验证一个主机制。默认继续 Qwen3.5-2B、同一 108 类映射、原单用户单 target、相同真实 NI 样本。训练内部用户 holdout 用于方案选择，最后再看正式 Dev。当前 Dev 已经用于多轮选型，因此更强的泛化主张还需要事先冻结的新测试切片或额外数据集；这属于后续实验设计，不在本次调研中改变数据。

正式比较应固定 seed 集合并报告多个 seed；用户级 bootstrap 在配对抽样后重算原 macro 指标，不能用 target 当独立样本。两种不确定性分开呈现：数据抽样 CI 与训练 seed 波动。输出不只总表，还要有 target-parent novelty、历史长度、源—目标支持度切片，但不将这些切片混作原 macro 指标。

当前正式评估不做 history mask。新方法如果增加该硬约束，必须为全部比较方法统一评估或单列该因素。用已有 Top20 做的过滤诊断显示 Mixed 相对 NI-only 的优势仍存在，但它不是新的正式成绩。训练曝光、token、有效更新、额外 head／router 参数与效用估计开销都要透明报告。

最有凝聚力的论文问题可以表述为：**如何把“历史中已有兴趣的识别能力”转化为“缺少直接支持的新兴趣推断能力”，并验证这种转化能跨用户、跨迁移关系泛化。** 第一条给出受控的训练证据，第二条给出可组合的迁移结构；第三条则解释现有历史无法决定的观测机会，是独立的后续扩展。

**参考来源与读取范围**

[^1]: Jianling Wang, Haokai Lu, et al. *LLMs for User Interest Exploration in Large-scale Recommendation Systems*. RecSys 2024. [作者全文 v2](https://arxiv.org/html/2405.16363v2)，重点 §3.1–3.3；预印本页头的模板 DOI 不采用。正式 DOI：10.1145/3640457.3688161。
[^2]: Jianling Wang et al. *User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems*. ACL Industry 2025, pp. 996–1003. [正式出版页及全文](https://aclanthology.org/2025.acl-industry.70/)，背景与失败尝试 §3、方法 §4.1–4.2。
[^3]: Zhibo Wu, Yunfan Wu, Quan Liu, Lin Jiang, Ping Yang, Yao Hu. *GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation*. 2025-10-16 preprint. [全文](https://arxiv.org/html/2510.14626v1)，重点 §3.2–3.4、§5.4.2、§5.5；未用模板 conference 字段推断正式 venue。
[^4]: Hongxiang Lin et al. *When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation*. 2025 initial preprint, v3 2026-03-04. [全文 v3](https://arxiv.org/html/2508.00450v3)，DSIE 与 PCO 方法部分。按预印本使用。
[^5]: Shengyu Zhang, Dong Yao, Zhou Zhao, Tat-Seng Chua, Fei Wu. *CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation*. SIGIR 2021. [全文](https://arxiv.org/html/2109.05261)，§3.3.2–3.3.5；DOI 10.1145/3404835.3462908。
[^6]: Ziqi Zhao et al. *Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure*. SIGIR 2025. [全文](https://arxiv.org/html/2504.13482v1)，§3.3、§4.1、§4.3–4.5。其 ZhihuRec/Tenrec/KuaiRand 原文结果不与本项目 GNIP 分数混排。
[^7]: Christopher Fifty et al. *Efficiently Identifying Task Groupings for Multi-Task Learning*. NeurIPS 2021. [全文](https://arxiv.org/html/2109.04617)，§4.1 的有向 inter-task affinity。
[^8]: Shikun Liu, Stephen James, Andrew J. Davison, Edward Johns. *Auto-λ: Disentangling Dynamic Task Relationships*. TMLR 2022. [论文](https://arxiv.org/abs/2202.03091)，§4 的 primary-task validation objective 与辅助任务权重。
[^9]: Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun. *Learning to Reweight Examples for Robust Deep Learning*. ICML 2018, PMLR 80:4334–4343. [正式出版页](https://proceedings.mlr.press/v80/ren18a.html)。本次使用出版方摘要核验 gradient-based validation reweighting 的先例，未据此引用额外实验细节。
[^10]: Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua. *Denoising Implicit Feedback for Recommendation*. WSDM 2021. [作者论文](https://arxiv.org/abs/2006.04153)，全文读取经 ar5iv；ADT 方法部分。
[^11]: Tianrui Song, Wenshuo Chao, Hao Liu. *Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation*. 2024 preprint. [全文](https://arxiv.org/html/2409.10343v1)，LLM-based Sample Scoring 与 Denoising Training with Hard Samples；本次未独立确认后续 venue，按预印本使用。
[^12]: *Efficient Bi-Level Optimization for Recommendation Denoising*. KDD 2023. [作者论文](https://arxiv.org/abs/2210.10321)，全文读取经 ar5iv，§4.2–4.3；DOI 10.1145/3580305.3599324。
[^13]: Chenxiao Fan et al. *Uncertainty-aware Generative Recommendation*. 2026-02-12 preprint. [全文](https://arxiv.org/html/2602.11719v1)，§3.2–3.4；不采用 conference 模板占位字段。
[^14]: Xiangchen Pan, Jiayi Xu, Jing Wang, Xing Fang, Lingyun Zhu. *LLM4AIGQ: LLM-based AI Guidance Query Generation Framework for Multi Interest Mining*. 2026-09-03 preprint. [全文](https://arxiv.org/html/2609.03674v1)，§4.2–4.4；不采用旧年份模板字段。
[^15]: Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, Kazuhide Nakata. *Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback*. WSDM 2020. [作者论文](https://arxiv.org/abs/1909.03601)，全文读取经 ar5iv，§4.1 与 propensity/clipping 假设。
[^16]: Pengjie Ren, Zhumin Chen, Jing Li, Zhaochun Ren, Jun Ma, Maarten de Rijke. *RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation*. AAAI 2019. [全文](https://arxiv.org/html/1812.02646)，§3.1–3.5；DOI 10.1609/aaai.v33i01.33014806。

另核验到 *Partial Label Learning-Inspired Denoising Implicit Feedback for Recommendation*, SIGIR 2026，DOI [10.1145/3805712.3809955](https://doi.org/10.1145/3805712.3809955)，但仅取得出版元数据，未获得全文，因此只记录为 partial-label 路线的待排查近邻，不据其标题推断具体方法。*Personalized Interest Graphs for Theme-Driven User Behavior*, RecSys 2025，DOI [10.1145/3705328.3748133](https://doi.org/10.1145/3705328.3748133)，同样仅独立核验到题名、作者与出版信息；正式推进 Idea 2 前应取得作者稿作更细查重。

详细来源笔记：[旧新兴趣与迁移文献](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/research-ideas-20260913/old_new_literature.md)、[单正例／不确定性文献](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/research-ideas-20260913/uncertainty_literature.md)。网页及论文文本快照保存在同一研究 artifacts 目录，实验事实对应冻结源文件与哈希；报告不使用搜索结果摘要替代论文机制证据。
