# NI-only 历史顺序消融方案

状态：用户已于 2026-09-23 确认执行，并明确同意保留原提示词。执行范围为一次新训练、三次新 Full Dev 评估；进度与验收另记运行报告。以下为已批准的冻结方案。

## 1. 问题与实验矩阵

问题：固定用户、历史条目内容与频次、NI 标签及训练预算，原始历史的排列顺序是否有助于当前新兴趣预测？进一步区分已有模型的顺序敏感性与重新训练后可达到的表现。

O 表示原始记录顺序，S 表示同一用户内部固定随机排列。保持 Behavior+Interest 输入，即整条 `[Topic category] Headline` 一起移动。

| 条件 | 训练历史 | 评估历史 | 模型来源 | 本轮工作 |
| --- | --- | --- | --- | --- |
| A：O→O | 原始顺序 | 原始顺序 | 已验收 NI-only H50 | 复用已有模型、预测和指标 |
| B：O→S | 原始顺序 | 打乱顺序 | 与 A 完全相同的最终 adapter | 新增一次 Full Dev 评估 |
| C：S→O | 打乱顺序 | 原始顺序 | 新训练的 shuffled NI-only | 新增一次 Full Dev 评估 |
| D：S→S | 打乱顺序 | 打乱顺序 | 与 C 完全相同的最终 adapter | 新增一次 Full Dev 评估 |

总计：**一个新增训练、三次新增 Full Dev 评估**，不是三个模型。C、D 共用同一训练终点；B、D 共用同一份打乱 Dev。A 的完整预测只在远端读取用于成对比较，不重新训练 A。

不加入 CIR、OI、跨用户历史交换、历史删除、Interest-only、每轮重新打乱、多训练 seed、多测试排列或额外模型。普通 DataLoader 的样本遍历顺序保持原合同，不把样本之间的 shuffle 当作历史内部的 shuffle。

## 2. 已核对的对照身份

对照 ID：`qwen35-2b-bi-sid-ni-only-21600-seed42-v1`。

- Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
- Behavior+Interest → Semantic-ID；原 NI-only SFT；历史最多 50 条，短历史保持原长度。
- 本次读取原 Parquet：Train 21,600 条／21,600 用户，108 类；Dev 31,392 grouped impressions／27,082 用户／37,728 个目标出现。已有验收记录为 104 个 Dev 支持类、108 个候选 SID。
- 本次回读并核对本地保存的原指标文件 SHA-256，与原云端回读收据一致；没有在本次设计阶段重新验证远端服务状态。

| Macro 指标 | A：原 NI-only（%） |
| --- | ---: |
| Recall@10 | 20.3275 |
| Recall@20 | 33.6553 |
| NDCG@10 | 10.1325 |
| NDCG@20 | 13.4663 |

来源数据目录：`data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1`。

| 对象 | SHA-256 |
| --- | --- |
| 原 Train `train/behavior_interest/21600.parquet` | `e76acd152ed6e0e92f8b8015dfbc599603267a480d9321b41b5acb95d10487f0` |
| 原 Dev `eval/behavior_interest/dev_full_grouped.parquet` | `c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd` |
| SID mapping | `45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a` |
| 原指标文件 | `7c96deb861af4439df832e0b7836f3e899bf32fe56936494905995530d3dd735` |
| 原最终 adapter 目录摘要（指标文件记录） | `cfc443319e023dcbeb3b1ca82a95486ec481c1727eaf4aadcd1308b63abef3be` |
| 原 adapter_model.safetensors 文件摘要（远端 manifest） | `0a41effcaac398a0ca9d914142b7975dffaad98f435379e436e6132b59c3abf0` |
| 原初始化状态 manifest 摘要 | `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e` |

远端恢复入口：`projects/sft4safety/amlt-results/7210843083.71371-a21f4d1c-d098`。执行前在远端确认 A 的 adapter、完整预测、配置与验收 manifest 的身份；路径或文件不符时先定位既有存档，不能用另一个 NI-only 模型替代。

## 3. 唯一输入干预：固定用户内排列

直接从已冻结的 H50 输入出发，不重新筛选历史、不重新计算“新兴趣”、不修改目标。每条记录保留条目的多重集合，包含重复条目；类别和标题不得拆开打乱。

建议固定排列规则如下，确认后实现：

1. 去掉仅用于展示的行号，得到有序的完整条目列表；将该列表用固定 JSON 序列化后计算历史摘要。
2. 对每个原位置 i，计算 SHA-256，输入为规范 JSON 数组 `["ni-history-order-v1", 42, split, user_id, history_digest, i]`，按摘要排序，摘要相同按原位置排序。编码和序列化参数写入规则版本。
3. `split` 明确区分 train/dev；排列键不含目标、未来点击或指标。相同 split 下同用户的同一历史获得同一排列，Dev 多 impression 共享排列。
4. 在打乱后重新从 1 编号，同步更新 prompt、训练 messages 的 user 内容及有顺序含义的 `history_subcategories`。其余字段逐项保持不变。
5. Train 每用户只产生一份排列，三轮重复使用；Dev 每用户也只产生一份排列，B、D 共用。此方法不会消耗训练框架的 RNG，不改变原 DataLoader 顺序。
6. 长度为 1 的历史保留；随机排列偶然等于原顺序、或重复条目使文本未变，也如实保留，不根据结果重抽。记录实际改变比例、未变原因和各长度分布。

本次发现 Train 有 323 条、Dev 有 854 条长度为 1 的输入。因此主结果保留完整 Dev，另报告预先固定的 `history_length >= 2` 子集（30,538 impressions），解释短历史的稀释作用。该子集只对已有预测做统计，不新增推理；所有条件使用同一子集，并独立报告其标签支持集合。

## 4. 提示词的处理与结论边界

本次完整回读确认原 Train、Dev 的所有输入都含有 `in chronological order`。此前讨论中的“统一改为中性提示词”不能直接与复用原 A 的要求同时满足：改词会新增一个输入／训练变量。

**本轮建议逐字保留原提示词前后缀，只改变历史条目排列。** 包括原 `in chronological order` 文字也不改，不添加“这是一份打乱历史”等条件专属提示。

这使 B−A、D−C 在相同模型和相同提示词下仅改变输入排列；但 S 条件不再满足提示词宣称的时间顺序。结果应解释为“原 NI-only 接口下的历史排列敏感性及打乱训练后的表现”，不能宣称完成了中性提示词下的时间信息消融，或证明了真实兴趣演化机制。需要中性提示词时另行设计匹配对照，不在本轮自动追加。

“原始顺序”指被冻结数据的记录顺序。当前实验没有额外验证每条历史的真实点击时间，也不区分最近行为位置、相邻转移、长程顺序等不同来源。历史内容可能包含可推断时间的文字，所以打乱也不等于彻底消除所有时间线索。

## 5. 新训练合同与资源

从与 A 相同的预训练基座、LoRA 与 SID 初始值重新开始；不能从 A 的训练后 adapter 继续微调，也不接续 CIR。复用初始化实现并逐张量匹配初始状态，不只核对 seed。

| 项目 | 固定设置 |
| --- | --- |
| 数据与曝光 | 21,600 条；108 类×200；3 epochs；64,800 次样本曝光 |
| 终点 | 2,025 optimizer updates；只评估第 3 轮最终模型，不选最好 Dev epoch |
| 初始化／遍历 | seed42；原 SID mean_norm_matched；相同逻辑样本遍历顺序 |
| 优化器 | AdamW fused；lr 2e-5；weight decay 0.01；max grad norm 1 |
| 调度 | cosine；warmup ratio 0.03；完整 horizon 2,025 步 |
| 可训练参数 | 全部原 LoRA 和 tied SID 行；LoRA r16／alpha32／dropout0.05；all-linear |
| 训练目标 | 原 native full-vocabulary、completion-only、token_mean SFT；不改成108类分类损失 |
| 编码 | enable_thinking=false；max_seq_length=2048；原3个监督后缀 token；超长即失败，不截断 |
| 数值路径 | 原 BF16 与非 reentrant checkpointing；冻结原依赖、计算内核和迁移实现 |

为使历史 A 能作为直接对照，本轮有明确理由沿用 A 的执行阶段：第 1–500 步单卡、microbatch1／累积32／Torch fallback；第 501–2025 步四卡、每卡 microbatch8／累积1／原 FLA 0.5.2 + causal-conv1d 1.7.0。全程 global batch32，迁移时保留 Adam、scheduler、RNG 与样本游标。

原 H30/H10/H5 编排提供了这一路径的已验证实现。只复用适用的实现，不复制其旧 smoke、额外组数或历史批准状态。输入变化会影响浮点计算和 dropout 等随机数消耗，不声称训练轨迹逐位相同。

建议 Recall Standard，最多同时占用 4×A100 80GB，一次只运行一个训练或评估阶段；第一阶段只使用其中一张卡。此安排比全程四卡慢，但避免同时改变基线执行方式。若改为全程四卡，则需承认历史基线执行差异或新增匹配训练，不能静默切换。

工作量固定为一次完整训练和三次 Full Dev（共 94,176 条 impression 推理）；确认前不提交资源。训练中断按同身份检查点恢复，不从头增加训练预算，也不为调参另开小模型实验。

## 6. 评估和预先固定的比较

复用 A 的评估程序与语义：完整 108 个 SID 的 float32 分数排序、Top20、无历史类别过滤；报告原 interest-macro Recall/NDCG@1/@5/@10/@20。不能换成 impression-macro 或另一个 NDCG 定义后继续沿用相同列名。

每组使用 4 个相同逻辑分片、每片 7,848 条、batch8；C 的原 Dev SHA 必须等于 A 的 Dev SHA；B、D 的打乱 Dev SHA 必须相同。评估端计算内核与批次配置也固定到 A 的已验收路径。

主判据固定为 Macro Recall@10；NDCG@10、Recall@20、NDCG@20 一并报告，不按结果挑指标。报告每组绝对值和相对 A 的百分点差，并固定以下解释性比较：

| 差值（后者为参照） | 含义 |
| --- | --- |
| B−A | 原模型对输入排列干预的敏感性；包含输入分布偏移 |
| D−A | 各自训练与测试顺序条件匹配时的表现差异 |
| C−A | 打乱训练对原始 Dev 表现的影响 |
| D−B | 同一打乱 Dev 上，打乱训练能恢复多少表现 |
| D−C | 新模型是否仍对原始／打乱输入敏感 |

同步报告 A/B、C/D 的 Top1 改变比例与 Top10 集合重合率：总体指标相近不代表逐用户预测相同。不新增其他训练或预测。

不确定性分析只使用这四组既有预测。按用户成对做 2,000 次 Bayesian bootstrap：每次对所有用户抽正的 Exp(1) 权重，同一用户的全部 impression 和所有条件共享该权重；每类重新计算加权目标命中率／折扣收益，再对固定支持类宏平均。这样保留稀有类 support，避免某次抽样中类别消失后偷偷改变宏平均分母。分析 seed 固定 42，报告差值的 95% 区间并明确其为 Bayesian bootstrap 区间。

该区间仅描述给定模型与固定排列下的评估用户不确定性，不包含训练 seed 或随机排列的不确定性，不作多次比较的整体显著性声明。区间含零不等于证明两组等价；本轮不设事后“无影响”阈值。

## 7. 验收与执行顺序

确认后依次执行：

1. 固定当前来源文件、已验收运行源码、模型和运行环境 manifest，核对远端 A 的可恢复身份。
2. 实现排列 builder，写入新目录；完整回读所有输入，验证逐行条目多重集合、长度、标签、样本 ID／用户／顺序及映射不变，Train/Dev 仍用户不交叉；运行针对排列与恢复身份的必要测试。
3. 在正式 worker 内做全量 token／环境／初始化核对；这些是正确性检查，不训练额外 smoke 模型。
4. 复用 A 完成 B；从原始初始化训练一次 shuffled NI-only，使用同一最终模型完成 C、D。各评估使用独立输出身份与目录。
5. 检查正式训练恰好 2,025 步／64,800 曝光，初始化与样本前缀符合合同，检查点状态和最终 adapter 可在新进程严格重载；沿用原容差，不因失败放宽。
6. 远端验证每组 31,392 条覆盖、37,728 目标出现、104 支持类、108 候选与合法无重复 Top20；使用独立 reader 从完整预测重算指标、成对差值和诊断统计。
7. 从持久存储独立回读哈希和完成收据，再形成 A/B/C/D 结果表。作业 Completed 不是验收完成。

如遇 FLA 重载差异，应从已有数值路径证据定位，不通过切换后端或放宽容差混入额外变量；确需改变冻结协议时说明影响，再更新方案。

权重、optimizer、完整预测和数据产物保留在持久远端存储。本地仅保留紧凑指标、manifest、哈希、恢复入口和报告；不重新下载已经清理的大产物。

## 8. 结果能支持什么

- B 下降而 D 恢复到 A 附近：原模型依赖排列，但在本次预算和固定排列下，重新训练可以补偿；不足以证明所有时间信息无用。
- D 仍低于 A，且 C 也低：支持原始排列对当前训练与预测有价值；无法据此定位为最近行为或兴趣转移。
- C/D 高于 A：打乱可能改善这一训练实例的泛化；单训练 seed 不能确证稳定收益或机制。
- 四组指标相近但逐用户排名变化大：总体性能对排列较稳定，个体预测仍敏感。
- 四组都接近：只能说本次固定排列未显示明显收益／损失，不能说历史内容、频次或真实顺序没有信息。

## 9. 设计依据

- 原 [H50 复用收据](../../artifacts/mind-history-length-ablation-20260915/local/h50_live_reuse_receipt.json) 与 [保存的原始指标](../../artifacts/mind-history-length-ablation-20260915/local/h50_remote_metrics.json)。
- [历史长度消融报告](../experiments/2026-09-15/mind_history_length_ablation.md) 和其 [冻结训练合同](../../artifacts/mind-history-length-ablation-20260915/bundle-v2/configs/experiments/history_h30.json)，用于核对 H50 身份、预算与执行阶段。
- [输入变换实现](../../src/new_interest/data/build_history_length_ablation.py) 与 [历史执行编排](../../src/new_interest/experiments/history_length_ablation.py)，用于确认输入审计、训练游标、阶段迁移和评估要求。

本方案仅覆盖上述三格新增实验。用户已确认，按此范围实现并执行。
