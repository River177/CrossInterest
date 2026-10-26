# SID 类别关系偏置：首轮实验协议

版本：`mind-sid-relation-bias-pilot-v1`；日期：2026-09-15。

状态：实验已获用户授权并实施。本地947项测试通过；修复版GPU门禁已完成单卡20步、四卡4步、精确重载及64条真实Dev，并通过独立下载验收。正式实验 `mind-sid-relation-full-v1-0915` 的B0/B1各4卡已并行提交，各训练675步。单次流水线继续跟进完整验证／Dev评估及独立回读，当前状态以 `artifacts/sid-relation-bias-20260915/pipeline_state.json` 为准。

## 1. 研究问题与首轮范围

给定原有 History Behavior+Interest，复用已经训练好的输出 SID 向量，构造共享的历史类别关系，在一个 Qwen full-attention 层添加偏置，能否提高真实新兴趣预测？

首轮选择 **同一已验收 Qwen3.5-2B Mixed 起点的等预算继续训练**，保留同一 prompt、单正例 SID、原生 full-vocabulary completion-only SFT 和 108 类排序。先做一个 seed、两组完整 NI 一轮训练；不同时改变输入表述、加入 OI、Meta、Cross-Fitted、CoT、GRPO 或新损失。

选择继续训练的原因：这直接检验用户提出的“复用已有 SID”的方法，而不是从随机／语义初始化开始，把新模型收敛差异混入首轮。首轮只能说明对该已训练起点的增量；不能据此声称从零训练优势或完整训练流程的多 seed 稳定性。

## 2. 实测输入与起点

以下数值在本轮本地读取 Parquet、模型配置和 adapter 文件后核对。回执：`artifacts/sid-relation-design-20260915/source_audit.json`、`preflight_counts.json`。

| 对象 | 已核对事实 |
|---|---|
| Backbone | `Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 模型配置 SHA-256 | `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4` |
| 文本模型结构 | 24 层；hidden size 2048；8 个 query heads、2 个 KV heads；head dim 256；输入／输出 embedding tied |
| Full-attention 层，零基编号 | `3,7,11,15,19,23`；其它为 linear attention |
| 已验收 Mixed 起点 | `artifacts/mind-short-think-sft-20260913/bundle-full-v1/inputs/mixed_adapter/`，注意它是打包保存的原 Mixed，不是 Short-CoT 训练后的 adapter |
| 起点 adapter 文件 SHA-256 | `e5451e07b9c04d44c2a8731976b1a0694cb081af832f733dcba8209339205fb6` |
| 起点参数文件 | 373 个 tensor；SID 有效训练张量形状 `108×2048`；文件 68,212,000 bytes |
| 起点已完成训练 | 原 Mixed 全词表 SFT 共3 epochs、3,630更新；OI曝光51,294次（17,098×3），NI曝光64,800次（21,600×3）；本轮重新核对原配置和独立回读收据 |
| 起点训练用户 | 原 Mixed 38,698 行，38,120 个不同用户；与官方 Dev 的 27,082 个用户无交集 |
| 本轮继续训练 | 原 Mixed 中 `training_task=new_interest` 的 21,600 行；108 类×200；每轮 675 个 global-batch-32 更新 |
| 官方 Dev | 31,392 impressions；27,082 users；37,728 target occurrences；104 个有支持输出类 |
| 训练历史类别 | 125 类；其中 17 类没有输出 SID，出现于 3,942/425,319 个历史事件 |
| 单条历史 | NI Train 323/21,600；Dev 854/31,392 |
| 历史仅一个不同类别 | NI Train 405/21,600；Dev 1,126/31,392 |

原 Mixed 数据 SHA：`b031aed45598b12fe678d86c647ac95118e96609c92b5b017a3f1464984376b1`。
SID mapping SHA：`45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a`。
官方 Dev SHA：`c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd`。

## 3. 独立验证用户：不能从旧训练用户里再次切验证

已经训练过的 Mixed adapter 看过完整 NI 和 OI 用户。因此，不能从这 21.6k NI 中抽出用户再声称它们是独立验证用户。

本轮从原静态 Train 的 587,187 个用户样本中，排除全部 38,120 个 Mixed 训练用户和全部官方 Dev 用户，限定 108 类输出词表后，得到 551,570 条可用候选，108 类均有支持。

已按下面固定规则选出验证 ID 清单：

1. 每类按 `SHA256("20260915:sid-relation-validation:" + sample_id)` 升序。
2. 每类最多取 50 个用户，不放回；不根据历史与目标是否语义相关筛选。
3. 结果 5,354 行、5,354 用户、108 类：107 类各 50 条；`Artificial Intelligence & Robotics` 只剩 4 条。

清单：`artifacts/sid-relation-design-20260915/validation_selection.json`；SHA `1ec323d40c9b704a9ef072d7987f2289ddd575617e12931a77e450ce0f8b6fbd`。

这里冻结的是原始样本 ID 清单，仍需按现有 B+I builder 构造 H50 prompt、完成真实 tokenizer 的全量长度审计。如果任何记录超过 2048，不得静默截断、替换样本或把失败记录移出分母；需在训练前修订协议及新身份。

主验证指标按预先确定的 **107 个至少有 20 条支持的类别** 计算 macro；全部 108 类 macro 与稀疏类 4 条单独报告。不能用这 4 条的离散跳变决定方法成败。该规则只用于此验证集，官方 Dev 始终沿用 104 类主口径。

本验证集与起点、本轮训练在用户层隔离，但仍来自同一原始 Train 时段；不能把它称为时间外推测试。官方 Dev 已被项目反复用于探索，也不能称为全新未见最终 Test。

## 4. 唯一新增模块

### 4.1 关系参数与 SID 共享

对 108 个已有类，读取加载 adapter 后实际生效的输出 SID 行 `E`。本起点输入／输出 tied，不另外训练第二套类别 embedding；不能把 PEFT 的变量名 `delta` 当成可以随意二次相加的依据。

仅为关系分支对每个向量做固定 RMS 归一化，控制尺度，不改变原输出 head：

\[
\bar e_c=e_c/\sqrt{\operatorname{mean}(e_c^2)+10^{-6}},
\qquad
R=\frac{(\bar EU)(\bar EV)^\top}{\sqrt r}.
\]

- 关系 rank 固定 `r=16`；`U,V` 各为 `2048×16`，合计 **65,536 个新增可训练参数**。
- 首版对 8 个 query heads 共享同一关系偏置，不增加每 head 一套矩阵或路由器。
- `U` 用独立随机生成器、标准差 `1/sqrt(2048)` 正态初始化；`V=0`。初始 `R=0`，不因新模块初始化改变两组共同的数据／dropout RNG 轨迹。
- 初始第一步 `U` 的关系梯度为零是正常的；`V` 应有非零梯度。`V` 更新后，再检查 `U,V` 两者的有效梯度。
- SID 仍按原 SFT 策略训练；B1 额外接受关系路径的梯度。它的唯一参数所有者仍是原 PEFT 模块，关系模块不重复注册 SID 参数或整个父模型。

另外 17 个历史类别使用与已有 SID 一致的类别名称语义初始化方法生成固定旁路向量；不注册新输出 token，不扩大 108 类候选，不新增可训练类别表。完整源类别名称／索引、是否关联输出 SID、初始化规则与张量 SHA 随实验保存。

相对于新建整张 125×125 自由关系表，65,536 个投影参数并不更少。因此只称低秩关系结构及相对 2B backbone 的小扩展，不宣传“比直接关系表更省参数”。

### 4.2 注入位置与 mask

首版只在零基 `layer_idx=11`（第 12 层）full attention 注入。后面还有 12 层，包括第 16／20／24 层 full attention，保证改变后的历史表示能被预测位置继续读取。层号在训练前固定，不根据 Dev 扫描挑选。

只对 **不同历史事件之间** 的 token 对加入类别偏置：

\[
b_{ts}=\begin{cases}
R_{c_{g(t)},c_{g(s)}}, & g(t),g(s)\ge0\ \land\ g(t)\ne g(s),\\
0,&\text{其它情况}.
\end{cases}
\]

这里 `g(t)` 是 token 所属历史事件 ID。相对于之前的概念公式，本协议显式把同事件块设为零：首版研究历史事件之间的信息聚合，不改变同一标题内部的 attention 分数。因而只有一条历史时新增偏置严格为零。多个同类事件仍可受 `R_cc` 影响，不能把“仅一个不同类别”也误称为零偏置切片。

最终为 `softmax(QK^T/sqrt(d_h) + b + causal_padding_mask) @ Value`。保持原 GQA、Q/K norm、RoPE、output gate 与 o_proj。非历史 prompt、指令、assistant/answer、padding 位置不获得额外偏置；绝不解除因果 mask。

虽然 H=1 的前向旁路为零，B1 训练后的 LoRA/SID 与 B0 仍会不同，所以不能要求两个训练后模型在 H=1 上输出相同。

### 4.3 事件到 token 的绑定

从结构化历史生成精确字符 spans，再映射到完整 chat-template 编码中的 token offsets；不能按类别名称在 prompt 中全局搜索猜位置。保留原 prompt、token IDs、监督后缀，sidecar 只增加事件与类别索引。

必须逐行核对 `source_sample_id`、History 内容／顺序／重复、类别与标题对齐、prompt SHA、token 长度及 padding 对齐。特殊／模板 token 没有事件。非空白字符跨两个事件的 token 必须显式检测并在训练前确定规则，不能静默随便归属。

只使用输入历史构造 sidecar；未来 target、target article title、Dev 真值不进入关系分支。共享类别词表是全体候选都已知的公共标签定义，不是当前答案的提示。

### 4.4 数值路径与训练目标

沿用原 native causal-LM full-vocabulary completion-only token-mean SFT；SID 及既有结束后缀按原编码监督，prompt／padding 为 -100。不增加 relation loss、AAP、ranking loss 或人工关系目标。

U/V 与关系分数在 FP32 计算；不得让 BF16 加法把小偏置悄悄舍掉。B0/B1 使用完全相同的 attention 数值实现和 mask 路径，B0 的关系项固定为零；原 Qwen 的 QK 计算、softmax cast、Value matmul 规则须保持一致或在两组共同冻结。零偏置和原模型的差异必须在真实 GPU 上量化验证，不以放宽门槛掩盖 kernel 差异。

仅本任务需要一阶 SFT，不引入 Meta 的二阶/HVP 实现。GatedDeltaNet 复用可验证的一阶后端，两组一致。不要通过修改公共安装包影响其它运行；在项目模块中注册受控 attention 接口，并显式传递 batch sidecar，保证非重入 checkpoint 重算读取同一份映射，不使用会被下一 microbatch 覆盖的全局缓存。

## 5. 首轮对照与预算

| 条件 | 起点 | 本轮更新 | 目的 |
|---|---|---:|---|
| F0：Frozen Mixed | 同一已验收 adapter | 0 | 校准本轮统一 FP32 排序，并检查继续训练是否损害已有能力 |
| B0：Continue-SFT | 同上，重置 optimizer/scheduler | 675 | 控制额外一轮训练的影响 |
| B1：Continue-SFT + SID relation | 同上，U 随机/V 为零 | 675 | 检验唯一新增关系模块 |

共同配置：NI 21,600 行全量一次遍历；seed42；全局 batch32；4 GPU×microbatch1×累积8；不丢尾、不重复扩样。原 LoRA r16/alpha32/dropout0.05、all-linear、base frozen；继续训练 LoRA 与原 SID 行。

- LoRA/SID LR：`1e-5`；B1 U/V LR：`1e-4`。
- AdamW，weight decay0.01，global grad clip1；cosine，warmup `ceil(0.03×675)=21` steps；原 scheduler 规则逐步记录。
- 一轮末 checkpoint 为固定主结果；保存中间检查点用于恢复，不按 Dev 选最优步数。
- 不继承验证 smoke 产生的模型更新；正式 B0/B1 均重新从相同 Mixed adapter 开始。
- 使用独立且固定的 RNG 流构造新增参数；两组的训练样本顺序完全一致，dropout 随机策略一致。
- F0/B0/B1 都从 fresh reload 跑统一验证与官方 Dev。旧报告中的 20.61% 仅作历史参考，不直接当作新数值路径下的 F0 成绩。

资源安排（按用户修订）：Recall Premium **B0/B1各4×A100-80GB并行，合计8张卡**；用户允许最高同时16张，当前配置仍按每组4卡、累积8保持global batch32。若实际改为每组8卡，必须同步改为累积4并通过8卡门禁，保持每组675步／一轮曝光不变，不把更大资源额度解释成额外实验组或训练轮数。正式训练总计1,350更新、43,200行曝光。一次完整 Dev 三个条件共94,176次 impression 评分；验证三个条件共16,062条。实际 GPU 小时和推理延迟必须由首个真实门禁测量推算，不在未测前承诺小时数。排队不算训练耗时。

不取消、修改或复用其它正在运行任务的输出目录；本实验使用新命名空间和独立 checkpoint 身份。资源调度变化不能改变模型／数据／更新预算。

## 6. 评估与继续推进条件

排名：读取预测 SID 前的位置，对完整108个有效 SID 做 FP32 最终投影／分数排序；同分按冻结 `interest_index` 升序。保持原 no-history-filter 协议；不能只对新方法屏蔽已见类别。

主要结果：官方 Dev 的 104 类 Macro Recall@10、@20、Macro NDCG@10、@20。验证集以107充分支持类为主，并报告全部108类。另报告 impression Recall/Hit、Top1已见比例、类别覆盖及输出分数同分率。每个方法保存全部108分数，不只保留Top20。

切片预先固定：历史长度1、2–4、5–20、21–50；历史唯一类别数1、2–4、5+；目标parent是否在可见H50出现；历史是否包含17个无输出SID的类别。切片既报原始 target-occurrence 指标，也在支持充分时报告类别宏指标，不能混用两种聚合。

用户为 bootstrap 单位，2,000次，seed20260915；官方Dev保留同用户多个impression／target的相关性。验证集107类可用类内用户重采样保持各类支持；明确这种区间反映样本不确定性，不反映训练seed波动。

首轮全部三条件均按固定checkpoint报告，不因结果不好省略。下一阶段是否值得扩大，预先使用验证集判断：

1. B1 相对 B0 的107类 Macro R@10至少增加 **0.5个百分点**，配对95%区间下界大于0；
2. B1 的 Macro NDCG@10 点估计相对 B0 回退不超过 **0.1个百分点**；
3. B1 的 R@10不低于F0，避免把“少遗忘一点”当作超过原起点的性能提升；
4. 所有身份／数值／重载验收通过。

这些是预先设定的实用筛选标准，不是性能预测。如果区间跨零，应报告证据不足；不能反复改层号、rank、LR后继续用同一份Dev挑结果。若验证与Dev方向不一致，应报告不稳定并停止自动扩大，不挑选有利的一边。

## 7. 正确性门禁

已完成：数据／用户／类别覆盖审计、模型配置核对、源文件哈希、5354个验证ID的确定性选择。一个两层因果attention的CPU数学检查也已通过：零偏置等价；V零初始化能获得梯度；V非零后U可学习；共享SID获得关系路径梯度；仅修改最后混合层历史行无法影响当前SID预测。该检查见 `mechanism_check.json`，**不代表真实Qwen集成通过**。

实现后先完成本地 tiny-Qwen tests，再进行一次真实4卡门禁作业，内部依次执行必要检查：

1. 全量Train/Validation/Dev token及事件对齐审计，无截断、无target泄漏。
2. 原模型／零偏置模型在同输入、同backend、同精度下的完整loss、SID logits、既有参数梯度等价。
3. 只取第一个SID的损失，证明layer11的U/V能收到信号；不能仅靠结束token loss给出假通过。初始U零梯度按数学预期解释。
4. 未padding、不同padding和批次排序、causal未来屏蔽、同事件／非历史零偏置、H=1零旁路；prefill及生成缓存路径一致性。
5. 真实1卡固定小样本训练／学习检查、fresh reload；随后4卡至少2次更新、非重入checkpoint、各rank新参数同步、保存／重载。
6. 真实小规模排名和独立回读。保存LoRA/SID加U/V以及17个固定旁路向量、源类别映射和模块配置；故意缺失关系权重时加载必须失败，不能静默回退成原SFT。

真实GPU检查通过并下载验收后，直接进入两组固定正式训练；不重复做没有新问题的smoke。若失败，只修复对应实现／环境问题；改变实验超参数或数据必须另立版本。

checkpoint写入节点本地，再发布到持久化存储；模型／optimizer／scheduler／RNG／已消费样本前缀共同保存。网络I/O不能夹在未完成NCCL collective中。恢复只接受同身份完整checkpoint；固定“训练前”probe随checkpoint持久化，不能从恢复模型重新测成before值。

## 8. 第二阶段：仅在首轮有效后讨论

首轮只回答“该起点上此模块是否值得继续”。若有效，依次增加以下对照，每次解决一个问题，不一次性铺开：

- 相近新增参数量的普通LoRA／attention适配：排除一般容量收益。
- 冻结的已训练SID副本 vs 同尺度随机固定类别向量，两者保持U/V与预算相同：检验已有SID表示的价值。随机向量仍可能表达关系，不能把差异解释为简单表达能力定理。
- 共享SID vs 冻结SID副本：检验共同更新是否有帮助。仅detach并不等于固定副本。
- 在验证集上只打乱关系旁路的类别—事件绑定，保持原prompt及各用户类别计数，做同checkpoint敏感性诊断；全局重命名类别只是可逆编号变化，不是有效打乱对照。
- 增加独立继续训练seed验证稳定性；若主张完整流程稳定，需要重新训练不同seed的起点，不能把共享同一个Mixed起点的三次继续训练写成完整三seed实验。

单个总体增益不能证明“共享语义空间”“个性化迁移”或因果关系。当前R是共享类别关系，个性化来自原文本上下文；论文主张随实际消融结果收敛。

## 9. 最小实现边界与交付

建议新增实质模块：

- `new_interest.data.build_sid_relation_sidecar`：冻结NI与新验证、构造token/event/category绑定及身份审计。
- `new_interest.training.sid_relation_attention`：有效SID行读取、RMS归一化、U/V、layer11注意力注入和受控backend接口。
- `new_interest.experiments.sid_relation_bias`：F0/B0/B1训练与评价、checkpoint、独立reader所需的产物契约。

复用现有路径工具、原生SFT编码、SID初始化／重载和排名指标；不创建根包转发模块，不改正在运行的Mixed/Meta文件。新增模块和sidecar必须由显式配置启用，默认原流程不受影响。

正式提交前冻结新运行包、精确环境pins、全量token摘要、模块接入测试、真实GPU成本回执。最终交付应包含可独立重算的F0/B0/B1比较、每类与用户配对结果、全108分数、资源成本、全部验收及未解决限制。当前交付是本实验方案及设计审计，尚无新方法效果结论。
