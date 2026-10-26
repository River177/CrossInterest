# 用户内 CIR 元训练：三组实验方案

> 当前状态：用户否定了下述“CIR虚拟更新后计算NI”的方法理解，要求先重新确认实验含义。已停止实现推进和提交，未启动GPU作业。下文保留为被搁置的提案记录，不能继续作为执行授权；三组比较意向与“预测前不适应”仍保留，具体训练方法待重新确认。

日期：2026-09-24。最新范围：用户指定三组——CIR5→普通NI、CIR5→元训练NI3、原始预训练模型→元训练NI3；延续此前“预测前不要适应”的要求。该三组范围取代此前A/C两组范围。已完成当前代码、既有产物与本地数据核查；尚未实现新训练器、构建正式支持集、提交GPU作业或进行新模型评价。

## 1. 第一轮固定范围

采用一个训练方法：训练阶段，每个用户用自己的少量 CIR 题产生一次临时 LoRA 更新，再读取完整历史计算 NI 损失；全局训练通过这一虚拟更新优化共享参数。**验证、测试及最终部署均直接使用共享外层模型，不构造个人CIR题、不更新参数、不使用用户临时模型。**

第一轮只新增 **2 条训练分支**，评价以下三组；第一组普通NI也固定为3轮，与现有已接受结果一致：

| 组别 | 条件ID | 训练方案 | 本轮工作 | 预测前个人适应 |
|---|---|---|---|---|
| 1 | cir5_ni3 | CIR5→普通NI3 | 复用已接受模型，不重训 | 关闭 |
| 2 | cir5_meta_ni3 | CIR5→用户内CIR元训练NI3 | 新增3轮元训练 | 关闭 |
| 3 | cold_meta_ni3 | 原始预训练Qwen＋原始LoRA/SID→用户内CIR元训练NI3 | 新增3轮元训练，无独立CIR预训练 | 关闭 |

三组在同一完整Dev上成对比较。现有第1组不重训，也通过共同评价路径评分，以避免新旧数值实现差异被混进训练方法的效果；其既有R@10=21.9717%作为复现参照。

固定报告三项差值：**2−1**检验在同一CIR5起点上元训练是否优于普通NI；**3−2**检验同一元训练流程省去CIR5预训练后效果如何；**3−1**比较直接元训练与现有两阶段方案的整体效果。第2/3组除LoRA/SID初始化来源外，使用相同支持题、NI顺序、步长、优化器、训练轮数和评价设置。

第3组省去的是**独立CIR5预训练阶段**，元训练内层的历史CIR支持题仍然保留。三组均不实现或运行测试时适应。第一轮不增加其他训练条件、模型、seed、alpha sweep、换用户支持实验或数据/计算量匹配训练臂。若之后需要区分额外CIR监督、元梯度和用户内任务配对，再单独讨论；本轮不作这些机制的唯一归因。

## 2. 共同基座与两种初始化

- Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
- 第2组从已接受 **CIR5、尚未进入NI的checkpoint** 开始，完整继承372个LoRA张量及108×2048的共享SID行。
- 第3组从同一原始预训练基座开始，恢复此前核验过的原始LoRA和SID初始化；SID采用原始语义初始化，不是全零或任意重抽。它不加载任何CIR或NI训练后的adapter，也不是随机初始化整个Qwen模型。
- 三组预训练基座均冻结。第2/3组不从CIR5→NI3继续训练，避免额外的NI曝光。
- CIR5 adapter 文件 SHA256：`1201a6e750cf65ca57940613df177a8caab14b594827f4496f84c7643f283df9`。
- CIR5 参数状态摘要：`dc57ec586aef1bf6fcc573ffbbc5fc5e441e551b69e7ce3b4685b7ea828e6e17`。
- 第3组原始373张量状态摘要：`f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`；使用冻结构造器恢复后逐张量核验，不能只凭seed相同认定初始化一致。
- 已记录远端根：`amulet/projects/sft4safety/amlt-results/7210244599.48462-16575cc7-74a3/formal/cir5/final_adapter/`，Azure 账号 `csnewsandfeeds4150361735`。这是既有验收指针；执行前仍需远端重新校验文件及张量身份。
- 第2/3组的外层AdamW和scheduler均从零开始。各组SID初始化来源按上述约定冻结，训练中都允许外层更新SID。

## 3. 仅训练用户构造 CIR 支持题

只为Train用户构造支持题，选择时只读取当前可见H50的类别、标题和长度，不读取该用户NI目标、目标索引或未来时间。125个源类别都可以保留为输入，只有冻结108-SID词表中的类别能作为CIR答案。验证集和Dev不构造或消费支持题。

每个候选题必须满足：

1. 目标类别确实在当前 H50 中出现。
2. 删除该类别的**全部**记录，包括标题及类别文字。
3. 删除后保留至少 5 条行为且至少一个其他类别；不回填更早历史。
4. 重建连续编号，使用现有 CIR 提示词；最终 NI 使用完全未修改的原始 prompt。

在合格类别中，按 SHA256(`协议版本, seed42, user_id, 可见历史hash, 类别名`) 固定排序，取前两个不同类别。同一用户相同历史的支持题始终相同，三轮训练重复使用，不每轮重新抽题。

第2/3组逐字复用同一支持集manifest，不按各自模型的预测、loss或初始化重新选择题目。

- 0 个合格类别：训练时跳过虚拟更新，保留用户，按普通NI计算训练损失。
- 1 个：使用这一道题。
- ≥2 个：固定选两道，不重复同一个类别凑数。

这需要新支持集 manifest，不能直接把原单视图 `cir.parquet` 当成两视图数据。新选择规则使用用户及可见历史，不用 impression_id，因此不承诺第一题与旧 CIR 单视图一致；原 CIR 预训练文件保持不变。

本次仅用 prompt、user_id、history_length 和 SID mapping 进行了 CPU 计数，未使用 NI 标签筛选题目：

| 数据单位 | 无题 | 1题 | 2题 | 合计 |
|---|---:|---:|---:|---:|
| Train 用户/NI行 | 4,329 | 71 | 17,200 | 21,600 |

第2/3组各有79.96%的训练用户执行虚拟更新，剩余用户不能删除。每组每轮共34,471道逻辑支持题，三轮各有103,413次逻辑支持题使用；HVP重算另计计算次数，不冒充新的监督样本。Dev全部31,392条记录均直接预测，其历史是否能构造CIR题不参与筛选或计算。

计数与输入hashes：[CPU支持题可用性核查](../../artifacts/personal-cir-plan-20260924/support_eligibility.json)。该文件保留修订前的只读计数，包含已不再使用的Dev可用性统计；它不是正式支持集manifest，也不表示将执行测试时适应。

## 4. 训练中的一次用户内虚拟更新和一次全局更新

令共享 LoRA 为 phi，共享 SID 为 E，用户 u 的支持题数为 m_u：

\[
F_u(\phi,E)=\frac1{m_u}\sum_{j=1}^{m_u}
\ell_{\rm native\ SFT}(H_u^{-c_j},\mathrm{SID}(c_j);\phi,E),
\qquad
\phi'_u=\phi-\alpha\nabla_\phi F_u.
\]

无题时定义 phi'_u=phi。支持题损失先平均，**只做一步**，不逐题连续更新。

全局目标为：

\[
J(\phi,E)=\frac1B\sum_{u=1}^{B}
\ell_{\rm native\ SFT}(H_u,\mathrm{SID}(y_u);\phi'_u,E).
\]

每个用户从相同共享参数出发，拥有自己的临时 phi'_u；处理后丢弃。不能先混合不同用户的 CIR 梯度再生成一个公共临时模型。

**最终保存并部署的是共享(phi,E)，不是phi'_u。** 第2/3组均使用这里的训练目标J，不添加未适应NI损失、不修改目标权重或改成另一种元学习算法。J评价的是训练中的虚拟更新后表现，部署评价的是共享参数表现，二者并不自动等价；本轮通过三组直接预测结果检验各共享模型的效果。

| 项目 | 第一版设置 |
|---|---|
| 内层更新参数 | 全部现有 LoRA；SID 数值保持不动 |
| 内层优化 | 1 步 SGD；alpha=0.01；无动量、无 weight decay、无梯度裁剪 |
| 内层随机性 | 训练中的CIR支持前向关闭dropout，但启用梯度 |
| 外层更新参数 | 共享 LoRA＋SID；预训练基座冻结 |
| 外层损失 | 只有训练虚拟更新后的NI原生全词表completion CE |
| 外层优化 | AdamW，lr=2e-5，weight_decay=0.01，warmup=0.03，cosine horizon=2,025，平均 outer 梯度后 clip norm=1 |
| LoRA | r16、alpha32；NI query 训练 dropout=0.05，评价关闭 |
| 数据和轮数 | 第2/3组各使用相同的原21,600 NI用户、seed42、3 epochs，每用户每轮一个原始NI query |
| 全局 batch | 32用户；每组每轮675步，各2,025步／64,800次NI query曝光；新增训练共4,050 outer updates |
| 元梯度 | 精确二阶；包括支持梯度对 SID 的混合导数，不替换为 FOMAML |
| 编码 | 原 news/click 提示词、原 SID mapping、原 completion/template suffix；max_seq_length=2,048，禁止静默截断 |

alpha=0.01是第2/3组共用的固定起点，不是已证明适用于个人CIR的最优值。旧OI/NI实验用过这一数值，但其梯度来自另一种采样与归约。本轮不按分支或Dev得分调整它；非有限数值或实现失败应报告并暂停，不自动搜索其他alpha。

SID 的“内层固定”仅表示没有 SID fast update，不能通过 requires_grad=False 或 detach 切断外层需要的混合导数。LoRA alpha32 是缩放参数，与 SGD 步长0.01是不同概念。

## 5. 分布式实现与既有代码复用

每条元训练分支使用4×A10080GB，两分支顺序执行，保持同时最多4卡。每卡顺序处理8个用户：完成一个用户的support梯度、fast-query梯度及HVP后，释放其计算图，只累加outer梯度。四卡在32个用户结束后平均outer梯度，做一次AdamW更新。

**内层梯度不跨卡汇总。** 旧 OI/NI Meta 的分布式内层先汇总成公共更新，不能直接沿用。

复用：

- `new_interest.training.oi_ni_full_sft.FullVocabularySFTModel`：原生完整词表 loss。
- `new_interest.training.oi_ni_meta_objective.FunctionalModel` 及精确 HVP 技术：临时参数不写入共享模型。
- `new_interest.training.oi_ni_meta_backend.configure_exact_meta_backend`：经验证的 Qwen3.5 Torch 数值路径、PEFT tied-SID alias 修复。
- 既有 snapshot、SID 注册、检查点、指标及独立读回工具的兼容部分。

新增拟定归属：

- `new_interest.data.build_personal_cir_support`：支持题及 manifest。
- `new_interest.training.personal_cir_meta`：逐用户目标、临时参数和精确梯度。
- `new_interest.experiments.personal_cir_meta`：训练、恢复和 outer 梯度同步。

评价复用现有共享模型直接预测路径，不新增个人适应评价器。

已有 exact backend 禁用模块内部 gradient checkpointing（包括直接恢复 nonreentrant 的做法），使用 eager attention、原生 Torch gated-delta 和 use_cache=False。内存受限时采用支持题重算/HVP，不擅自开启未验证的 checkpoint 路径，也不换为108类训练 head。

训练时初次支持梯度与HVP重算保持相同create_graph数值路径。验证和测试采用无梯度的直接前向，不调用支持梯度、HVP或fast-parameter更新。

## 6. 评价合同

最终模型固定为第三轮终点，不用 Full Dev 选 epoch、alpha 或支持题。

- 原 Full Dev：31,392 grouped impressions、27,082用户、37,728目标出现、104有支持类别。
- 固定108个SID，FP32类别排名，Top20，不增加历史类别过滤。
- 主要指标 Macro Recall@10；同时报告 Recall@20、NDCG@10/@20。
- 三组均加载各自最终共享参数，model.eval并禁用梯度，直接读取完整原始历史输出排名。预测过程中不得构造支持题、更新LoRA/SID或创建个人fast parameters。
- 完整保留31,392条记录及原指标口径，不按CIR题目可用性分组筛选，不进行在线学习或跨用户参数积累。
- 评价前后核对共享参数身份和值不变。若缓存完全相同输入的输出，只允许以模型和输入身份为依据，不能使用未来标签。
- 报告训练耗时、普通推理时延和峰值显存；测试梯度更新数必须为0。相同NI曝光不代表相同总训练计算或相同辅助监督量。
- 对主要成对差值做固定 seed42、2,000次用户成对 Bayesian bootstrap：每次给用户抽取正的 Exp(1) 权重，同一用户的全部记录和各条件共用权重，在固定104个有支持类别上重算宏指标，报告2.5%/97.5%分位数区间。区间只描述评价用户不确定性，不代表跨训练seed稳定性。

报告2−1、3−2、3−1，判断改变NI训练方式和省去CIR5预训练后的实际效果。本轮没有测试时适应条件，不评价其价值，也不声称已验证个人适应或个人模型的效果。

训练预算分别列示：第1/2组有86,355次独立CIR预训练曝光，第3组为0；三组NI query均为64,800次，第2/3组还各使用103,413次内层CIR支持题。第1/2组复用已有CIR5，当前物理执行不重复该预训练，但方法总成本需包含它。不同类型更新计算量不同，不能仅按步数宣称等算力或承诺第3组比第1组更快。

## 7. 执行顺序、正确性门槛与预算

本轮尚未执行下列步骤。

1. **CPU 实现与数据验收。** 固定支持题 manifest；核对全部题目标与未来标签隔离、整类删除、0/1/2题边界、原query不变。微型模型验证 exact unroll与HVP、非零SID混合导数、alpha=0退化为普通NI、用户状态隔离和尾批归约。
2. **真实四卡正确性与计时。** 用Train样本验证真实Qwen最长输入、梯度、逐用户与四卡等价、保存重载/恢复、共享参数不被fast更新污染。对第2/3组两个初始化各安排8个完整outer更新做计时及有限性核验；其更新不作为正式训练前缀，之后分别回到对应原始起点和新optimizer。此门槛不根据Dev分数选择方案。
3. **正式新训练。** 仅第2/3组两条用户内CIR元训练分支，每组3轮／2,025 outer updates，合计4,050次。两组每100步和epoch结束保存完整共享训练状态与游标；不保存用户临时adapter作为部署产物。第1组模型复用，不重训。
4. **三组Full Dev。** 模型和所有设置冻结后，分别直接评价第1、2、3组。先检查共同预测路径与原模型结果的数值一致性；若新backend与历史分数有差异，必须查明并报告，不混用不同路径的分数。
5. **独立验收与报告。** 验证两种初始化来源、完整预测覆盖、108类合法性、部署共享外层权重、评价前后参数不变、测试梯度更新为0、独立宏指标和三项配对差值。权重、Adam、完整预测留远端；本地只保存紧凑指标、hash、报告与恢复指针。

资源上限为同时4张A10080GB，正确性门槛、两条正式训练分支、评估顺序执行。由于元训练每用户有额外support前反向和二阶HVP，目前不能可靠承诺训练总耗时；用步骤2的实际耗时分别估算两组各2,025步，另计三次普通Full Dev评价，再给出GPU小时与预计结束时间。不得沿用普通SFT的训练每步时间。预测阶段不承担个人适应的梯度计算开销。

如果出现显存/二阶算子问题，先修复实现或报告资源瓶颈；不静默缩短历史、删除用户、改为一阶方法或扩大GPU数。

## 8. 已核查的依据

- [当前CIR构造器](../../src/new_interest/data/build_cir_dataset.py)：H50、108合法目标、至少5条保留历史、完整类别删除。
- [CIR5→NI3验收](../../artifacts/mind-cir-four-20260918/acceptance/cir5_ni3.json)。
- [CIR5来源清单](../../artifacts/mind-cir-parameter-20260923/bundle-v2/manifests/cir_parameter_transfer.json)。
- [第3组原始LoRA/SID逐张量清单](../../artifacts/mind-cir-parameter-20260923/bundle-v2/manifests/transfer_sources/cold_state.json)。
- [现有元学习数值后端](../../src/new_interest/training/oi_ni_meta_backend.py)。
- [原生全词表loss](../../src/new_interest/training/oi_ni_full_sft.py)。
- [旧全局内层分布式目标](../../src/new_interest/training/oi_ni_meta_distributed_objective.py)：需要替换，不能当个人适应直接复用。
