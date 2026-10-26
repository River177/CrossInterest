# 动态可约减损失权重：学生实验方案

用户在两折参考评分验收后要求继续。本阶段比较三种训练权重，检验“当前模型尚未学会的、跨用户仍可预测的样本”是否更值得分配更新。当前文件记录冻结设计，实际提交、完成和指标以验收产物为准。

| 条件 | 每条样本的排序依据 | 权重刷新 |
|---|---|---|
| Uniform | 所有样本权重为 1 | 不改变 |
| Static OOF | 参考损失的相反数，即 −aᵢ | 固定 |
| Dynamic | 当前学生损失减参考损失，即 bᵢ(t)−aᵢ | 第 0、675、1,350 次更新后、下一轮训练开始前 |

aᵢ 使用已验收的两个 Qwen3.5-2B 参考模型的交叉分数，参考模型没有见过被评分用户。bᵢ(t) 使用当前学生，以相同的首 SID、完整有效词表、FP32 投影与 logsumexp 定义计算。训练用的 loss 仍是原生完整词表 completion SFT，包含 SID、结束符和换行；每样本只乘一个停止梯度的权重。

Static 与 Dynamic 都在每个目标类别内部计算排序分位数。并列样本共享平均排名，q=rank/(n−1)，权重 raw=0.5+q，再除以该类平均值。全部相同的分数得到权重 1。所有 21,600 个样本保留，108 类各 200 个用户，各类总权重保持 200；不依据参考损失声明某个标签错误。

三组都从原始 Qwen/Qwen3.5-2B 发布权重、相同的 SID 初始化和 seed42 LoRA 开始。固定 3 epochs、global batch32、每组 4 张 A100、microbatch1、accumulation8，共 2,025 次更新。LoRA r16/alpha32/dropout0.05，LR2e−5，AdamW、cosine、3% warmup。训练及评分统一固定 FLA 选核；三组的初始化、采样顺序、训练 loss、更新预算与评分轮次相同。Uniform/Static 也计算 bᵢ(t)，仅用于相同计算预算下的训练动态诊断，不参与其权重。

所有训练用户、消息、目标 SID 和源顺序保持不变。训练不继承已经学过这些用户的 Mixed adapter，避免把数据方法比较变成记忆后的纠偏。轮次边界刷新权重时不重置优化器或学习率；轮内断点恢复必须加载该轮原始权重，不能用更新后的学生重新评分。checkpoint 持久化权重与顺序回执，原训练探针也保存到持久存储。

工程核验只运行新的必要路径：小样本单卡训练、实际 4 卡动态刷新、全 1 权重与原生损失/梯度精确对照、1.5 倍权重的梯度尺度检查、非首轮中间位置的续训、训练模型与新进程 native 探针对照、两个新进程实际 batch4 FP32 探针对照，以及 64 条真实 Dev 的评估写出与独立读回。通过后复用同一份冻结源码运行三组正式训练。

三组固定评估最终 checkpoint，不做中途挑选。主指标为 Macro NDCG@20，辅以 Macro Recall@10/@20 和 Macro NDCG@10。验证集保留 5,354 人，主口径为支持充分的 107 类，4 条 AI 类单列；官方 Dev 保持 31,392 impressions、27,082 用户、104 个支持类别。全部 108 个 SID 参与排名，不过滤历史类别、不删除难例。方法差异按用户进行 2,000 次配对 bootstrap，单 seed 的区间不代表跨 seed 稳定性。

若 Dynamic 不能优于 Static OOF，则没有新增动态项的收益证据；若收益只来自降低真正跨域难例的权重，也不足以支撑“保留有价值困难样本”的解释。该实验是 RHO-LOSS 思想的软权重迁移，不是原论文的大 batch 选小 batch 算法，也不将参考损失当作噪声概率。

固定参数见 `configs/experiments/mind_rho_student_v1.json`。输入与运行记录位于 `artifacts/rho-student-20260916/`；模型、checkpoint、完整评分、权重和预测均留在持久远端，本机只保留紧凑回执、统计和报告。前一阶段验收见 `docs/experiments/2026-09-16/rho_twofold_reference_scoring.md`。

## 启动梯度核验与数值执行修复

首个 GPU 门禁在第 1 次更新前停止。随后在同一 HF 启动位置完成 14 次零更新诊断：原数值后端中，预热后的原生计算重复也出现梯度差异（相对 L2 为 0.0114702877），原生与加权对照为 0.0114676977；这些对照的 loss 完全相同。严格数学后端下，原生重复和原生对加权的全部 373 个梯度均完全一致。所有梯度完整且有限，输入、随机数起点、参数和 buffers 均一致；关闭 Dropout 后原生重复仍有差异。该证据表明原门禁混入了数值后端自身的不重复性，不能将其视为标量加权实现的特异错误；它尚未隔离 SDPA、TF32 或降精度归约的单独贡献，也未在严格数学后端测量权重 1.5。

新版本选择最小的共同执行策略 `torch-deterministic-algorithms-v1`：仅开启 `torch.use_deterministic_algorithms(True, warn_only=False)`，保留原 BF16、TF32、SDPA 选择和 FLA 选核。此策略覆盖新门禁与三个正式条件的整个训练过程，每次 loss 计算前检查其仍生效，写入 checkpoint 身份与最终回执。新 GPU 门禁继续要求权重 1、1.5 的原生/加权梯度逐位一致及原生重复一致；仅当它及单卡/四卡/评分核验全部通过，才启动正式三组。这项最小修复已通过 66 项相关本地测试及完整本地测试（1,392 passed，1 项 pytest 弃用警告）；新 GPU 门禁的单卡零更新对照现已通过：权重 1、1.5、cold/warm 和原生重复的全部 373 个梯度均 exact。与旧诊断相比，初始参数、buffers、输入及 RNG 完全相同，捕获的数值设置仅 `deterministic_algorithms` 从 false 改为 true，首个 forward loss 也未改变。单卡 12 步/3 轮及四卡 6 步/3 轮均已完成，训练、保存重载、两个新进程 FP32 探针均已由 worker 验收；四 rank 的完整梯度及预热对照全部 exact，实际 loss 分母为每 rank 96 个全局 completion tokens。最终 64 条 Dev 评分及门禁整体独立读回也已通过，门禁验收 SHA 为 `471e5915bc06fdada3e7639f13050951825464209deff66525d450480d6d11fe`。三组正式实验各使用 4 张 A100。Static/Dynamic 在 Feeds Standard 继续训练；Uniform 在首次评分期间被资源抢占，当时没有完成优化器更新，现已使用相同上传源码和配置恢复到 Recall Premium。新任务的初始 adapter、完整初始 current 分数及 reference 分数已与 Static 一致；当前尚无正式验证集或 Dev 结果。资源恢复回执见 `artifacts/rho-student-20260916/uniform-resource-recovery-v1/replacement_verified.json`。

原始协议、失败 bundle 和诊断 bundle 保留。诊断完整报告 SHA-256 为 `0db34d175ad270416d3d60577195d61ded2885dc46dda9a5f81d04090d56f82c`，独立核验位于 `artifacts/rho-student-20260916/native-gradient-diagnostic-watch/compact_readback.json`。数值策略之外，模型、数据、权重规则、更新预算、最终 checkpoint 和统计比较规则均未改变。
