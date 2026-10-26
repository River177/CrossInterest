# MIND 108 类 CIR warm-up → NI SFT

本实验只训练一组：Qwen3.5-2B 的整类历史兴趣重建（CIR）1 epoch，然后进行原有新兴趣预测（NI）3 epochs。与已有 NI-only 基线比较，不重跑基线，不增加消融组。截至 2026-09-17 02:21 UTC，CIR 540 步与 NI 2,025 步均已完成并验收，参数交接与精确重载通过。当前执行 Full Dev，完整覆盖与最终指标仍待验收，监控保持开启。

## 数据与监督

从现有 21,600 条 NI 训练样本对应的原始结构化 History 构造，每个用户恰好一条源样本。完整原始 History 与模型可见 H50 分开：先取原始历史最后 50 条，再整类删除，不从更早历史回填。

候选类别必须在冻结的 108 类 SID 映射内，删除后至少剩余 5 条行为和另一种类别。固定 seed=42，以用户／impression 身份和候选类别的 SHA-256 选择一个类别。源 sample ID 带有原未来标签索引，因此选择键不使用这个 ID，也不使用未来目标标签。每个源样本最多保留一个视图，不枚举全部类别。

删除该类别的全部标题和类别记录，其他行为保持顺序并重新编号。输出被删类别的单个 SID。使用以下新闻场景提示词：

```text
The following news browsing history has all entries from one topic category removed. The remaining entries are in chronological order, formatted as "[Topic category] Headline":

{masked history}

Infer the removed topic category from the remaining observed interests and news items. This is historical interest reconstruction, not a prediction of the next click. Respond with only its single-token Interest ID in the exact format <interest_NNN>. Do not output the category name or any explanation.
```

输入去重使用最终模型可见的有序 `[类别] 标题` 历史，不比较任务指令或标签。排除与原 NI 训练、已有 OI 训练、NI Dev 或已冻结独立 Validation 输入相同的 CIR 视图；只删除辅助样本，原任务数据不变。Validation/Dev 用户不参与 CIR 训练，其标签不参与辅助类别选择；它们只提供输入去重和用户隔离检查。

| 构造结果 | 数量 |
|---|---:|
| 原 NI 训练样本／用户 | 21,600 |
| CIR 样本／用户 | 17,271 |
| 没有满足剩余历史要求的类别 | 4,329 |
| 与受保护输入重复而排除 | 0 |
| CIR 支持类别 | 108 |
| 每类样本数范围 | 3–1,056 |
| 与 Validation／Dev 用户交集 | 0 |
| CIR 最大输入 token 数 | 1,524 |
| NI 最大输入 token 数 | 1,564 |
| 超出 2,048 token 上限 | 0 |

不做类别均衡、不设每类数量上限。独立读取器逐条重算类别选择、删除位置、保留 news IDs、提示词和映射，并核对全部源／输出哈希。

数据：`data/training/mind_cir_108_v1/`。独立回读：`artifacts/mind-cir-20260916/data_independent_readback.json`。CIR SHA-256：`6bc05ebfcc9c398f2040e9f0826010118bef499e3a22225332ccb71c6c1a83cd`。

## 训练与验收

模型固定为 `Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。沿用原 LoRA r=16、alpha=32、dropout=0.05，学习率 2e-5，weight decay=0.01，cosine、warmup ratio=0.03、seed=42、BF16。训练 LoRA 和原 SID 参数，沿用 native full-vocabulary completion-only SFT 和 token-mean reduction；SID 与模板结束符共 3 个监督 token。

正式执行使用 4×A100-80GB、每卡 batch=2、累积=4、全局 batch=32，非 reentrant gradient checkpointing。CIR 共 540 updates、17,271 次样本曝光；NI 共 2,025 updates、64,800 次曝光。CIR→NI 继承所有 LoRA／SID 参数，优化器和 scheduler 从零开始。每个阶段保留可恢复 checkpoint、原生 Trainer 游标和逐样本曝光审计。

训练、重载和推理统一显式使用 Torch fallback。已有 NI-only 基线有历史 single→four-GPU／FLA 执行迁移，因此本次比较用于判断实际收益，不能单独将差值解释为 CIR 机制的因果效果。

同一个远端作业先做短的单卡／四卡两阶段交接检查、精确重载和 64 条真实 Dev 推理，全部通过后自动进入正式两阶段训练。四卡检查包含不能被全局 batch 整除的尾批。正式模型在原 31,392 个 Dev impressions 上执行固定 108 类排名，Top-20，不做历史类别过滤；Macro 指标对 104 个有支持类别等权平均。

大型 adapter、checkpoint 和完整预测仅保留远端；本地保留指标、报告、哈希和恢复指针。最终完成必须核验两阶段实际训练曝光、参数交接、fresh reload、四份预测分片、完整覆盖与独立重算指标。

## 可复现入口

- 构造器：`src/new_interest/data/build_cir_dataset.py`
- 阶段编排：`src/new_interest/experiments/cir_warmup.py`
- 冻结运行包：`scripts/analysis/prepare_cir_bundle.py`
- 实验配置：`configs/experiments/mind_cir_108_v1.json`
- AMLT：`configs/amlt/mind_cir_108_full_0916_v1.yaml`
- 预检记录：`artifacts/mind-cir-20260916/local_preflight.json`

运行包复用已验收 OI/NI bundle 的训练、断点恢复、样本游标、adapter 验收与评估模块，新增 CIR 数据和阶段编排；不覆盖当前共享 checkout 的训练实现。

## 提交记录

实验 `mind-cir-108-0916-v1`，作业 `qwen35-2b-cir-ni-108`，backend `neat_oxygen_v532tdrpny`；Recall Standard，`80G4-A100`。[作业页面](https://aka.ms/amlt?q=k2h6g)。

冻结 bundle SHA-256：`79c510fe6d0523f44cc7e65b948140d011e3fde05e92a93d3f92218f02dd88ee`。本地完整测试：1,472 passed、8 skipped。启动／恢复指针见 `manifests/mind_cir_108_20260916_v1.json`。Queued、Running 或后端 Completed 均不替代完整产物验收。

## 启动故障与 v2 修复

2026-09-16 14:04 UTC 原生日志确认：依赖安装成功，随后 worker 在冻结文件校验阶段因 `/scratch/amlt_code/.amltignore` 不存在退出。尚未加载模型或开始训练；结果目录未发现产物。平台日志随后记录高 SLA 回收并重新排队，不能将 Queued 解释为健康训练。

根因：AMLT 的实际上传函数会排除 `.amltignore`，旧 manifest 却要求它存在于 worker。v2 只从运行文件要求中去掉该本地上传控制文件；其余 85 个 payload 文件逐字节不变。实际 AMLT 上传函数回归：v1 缺一个必需文件，v2 无缺失；11 项 focused tests 通过，YAML dump 通过。旧作业暂停请求返回不支持，取消／替换等待明确授权；数据、模型、loss、训练预算和门槛均不变。

## 持续监控

用户已授权确保正确训练并开启监控。本任务 heartbeat `cir` 每 30 分钟检查一次，追踪清单 `active_run` 的当前 backend，核验原生 original/retry 日志、实际训练步骤与持久化检查点。正常推进保持安静，仅在实质故障、需要用户行动或最终独立验收完成时通知；完成后暂停监控。修复／恢复只限本 CIR 条件，不增加实验组或训练预算。

v1 的所有日志确认启动校验失败；2026-09-16 14:17:31 UTC 平台终止为 Failed。随后取消请求返回无须取消，未删除任何产物。v2 提交实验名为 `mind-cir-108-0916-v2`，仍使用 Recall Standard `80G4-A100`。

v2 backend：`funny_ghost_t6j3ry1chv`；[当前有效作业](https://aka.ms/amlt?q=k2iec)。结果前缀：`projects/sft4safety/amlt-results/7210460313.37836-415dd321-b1fd`。监控以清单 `active_run` 为准。

## CIR 任务名兼容修复 v3

14:36 UTC 监控捕获 v2 的 `Unknown old/new-interest training task`。冻结运行包已通过环境、模型 snapshot、完整 CIR/NI token 审计与初始 adapter 状态核验；错误发生在 TrainingExposure.from_parquet 读取 CIR 任务标签时，第一步训练之前。原始工具只接受 old_interest/new_interest。

v3 显式支持 cross_interest_reconstruction，不把 CIR 冒充 OI，不改变计数、采样、loss 或训练器计算。新增完整数据加载、checkpoint 计数恢复、未知任务拒绝及实际 CPU Trainer 尾批计数回归；14 项相关测试通过。其余数据／配置逐字节不变，实际 AMLT 上传清单和所有 payload 哈希通过。

v2 已按本任务恢复授权取消并确认 killed，保留全部小型原始证据；0 参数更新，无需恢复训练 checkpoint。新提交实验 `mind-cir-108-0916-v3`，继续相同 Recall Standard 4×A100 和训练预算。详见 recovery_v3_identity.json 与 v2_startup_independent_readback.json。

v3 backend：`dynamic_insect_xyybk1nb24`；[当前有效作业 v3](https://aka.ms/amlt?q=k2inm)。结果前缀：`projects/sft4safety/amlt-results/7210459117.84505-c58ab0c5-3f5e`。

14:53 UTC 监控：v3 单卡 CIR 交接检查已完成 8/8 optimizer updates（32 样本×2 epochs），loss 从 6.4864 降至 3.8671。checkpoint-8 提交标记与训练身份一致，冷启动参数摘要与已有基线一致，优化器从零开始。当前仍等待该阶段 fresh reload／验收；正式 CIR 540 步和 NI 2,025 步尚未开始。小型回读收据：`artifacts/mind-cir-20260916/progress/20260916T1453Z/verified_observation.json`。

15:07 UTC 监控：单卡 CIR 检查已通过 373 个 tensor 逐位一致与 logits 精确重载，8 步／64 次曝光被保留。平台在单卡 NI 检查第 5 步附近回收了节点；同一 backend 已自动启动 retry_001，复用已验收 CIR，重新启动尚无已提交 checkpoint 的短 NI 检查。NI 初始参数与 CIR adapter 的文件身份绑定，全部初始状态摘要一致，优化器／scheduler 均从零开始。没有提交新作业或重复 CIR；正式训练仍未开始。一次日志网络 reset 经只读重试恢复。收据：`artifacts/mind-cir-20260916/progress/20260916T1507Z-retry/verified_observation.json`。

15:20 UTC：单卡 CIR／NI 各 8 步、四卡 CIR／NI 各 2 步、两次参数交接及优化器重置、每阶段 373 个 tensor 与 logits 精确重载、64 条真实 Dev 四分片推理及远端独立指标重算均通过，并由紧凑产物交叉核验。正式 CIR 已按 17,271 行／1 epoch／540 步、4 卡 global batch 32 初始化；冷启动参数与基线逐摘要一致，未继承诊断检查的 adapter。下一检查继续核验正式更新进度。证据：`artifacts/mind-cir-20260916/progress/20260916T1520Z/verified_observation.json`。

15:34 UTC：正式 CIR 原生日志已推进至 53/540 步，最近一个已记录 loss 为 1.458；所见 loss、梯度范数与学习率均有限。正式数据／540 步游标／四卡 global batch32 合同复核通过；尚未到首个 checkpoint-100，不能把日志步数当作已持久化进度。NI 正式阶段尚未开始。收据：`artifacts/mind-cir-20260916/progress/20260916T1534Z/verified_observation.json`。

16:07 UTC：正式 CIR 曾到 165/540 步，所见 loss 有限；16:02 节点被更高 SLA 回收，同一 backend 自动重排并创建新 pod。checkpoint-100 已提交，小型回读核对了训练身份、元数据哈希以及冻结排列中前 3,200 个样本／9,600 个监督 token；大权重／Adam 状态留远端，实际恢复仍待原生日志确认。165 是旧 attempt 的日志进度，当前步数未知，不能当作已恢复进度。后续一次日志读取网络超时未触发任何重启或重新提交。证据：`artifacts/mind-cir-20260916/progress/20260916T1607Z/verified_observation.json`。

16:43 UTC：retry_002 已实际从 checkpoint-100 恢复，373 个 Adam 状态与 scheduler 游标均为 100；随后日志推进至 198/540，loss 有限。此次紧凑回读又取得已提交 checkpoint-200（时间晚于原生日志采样），元数据哈希、训练身份及前 6,400 个真实样本的冻结排列均核对通过。保留正式 NI 未开始的边界；没有重新提交或改变任何训练设置。收据：`artifacts/mind-cir-20260916/progress/20260916T1643Z/verified_observation.json`。

17:18 UTC：同一 retry_002 的正式 CIR 日志推进至 340/540，最近记录 loss=1.383，所见数值均有限。checkpoint-300 已提交，其元数据哈希、训练身份和前 9,600 个样本的冻结排列通过回读；NI 尚未开始，无新增错误或重提。收据：`artifacts/mind-cir-20260916/progress/20260916T1718Z/verified_observation.json`。

17:51 UTC：正式 CIR 推进至 475/540，最近 loss=1.272；checkpoint-400 的模型／目标／执行身份及前 12,800 个样本的冻结排列通过回读。新增只读观察核验入口 `scripts/analysis/verify_cir_monitor_snapshot.py`，复用相同检查，并拒绝将 SIGTERM 宽限期的旧步数或已经完成的 CIR 当作当前训练。实际快照核验与两个反例回归通过；GPU 运行包未改变。收据：`artifacts/mind-cir-20260916/progress/20260916T1751Z/verified_observation.json`。

2026-09-17 01:40 UTC 实际回读：正式 CIR 已完整完成 540 步／17,271 样本，每样本恰好一次，108 类曝光计数与冻结数据一致，373 个 tensor 和 logits 精确重载通过。正式 NI 已继承 CIR adapter `fce267bb88b78cf590527cb317a360c77e08ade9dc6a1b08efa08ce35af6b536`，优化器与 scheduler 从零开始；原生日志达 1885/2,025，最近 loss=1.432，数值有限。checkpoint-1800 的身份、元数据哈希与 57,600 次样本曝光的冻结顺序通过核验。完整 NI 与 Full Dev 尚待完成。触发目录名沿用 00:31 的心跳，实际观察时间以 receipt 的 01:40 UTC 为准。证据：`artifacts/mind-cir-20260916/progress/20260917T0031Z/cir_final_handoff_readback.json`、`artifacts/mind-cir-20260916/progress/20260917T0031Z/verified_observation.json`。

02:21 UTC：正式 NI 已完成并验收 2,025 步、21,600 样本各 3 epochs（64,800 次曝光／194,400 个监督 token），108 类各 600 次曝光。最终 checkpoint 元数据与完整样本游标、373 个 tensor 和 logits 精确重载均通过；CIR→NI adapter 身份链一致。当前在做 Full Dev：四分片输出批次文件名分别到 {'0': 3072, '1': 3328, '2': 3328, '3': 3072}，只是已持久化输出范围，不能代替最终覆盖与指标验收。证据：`artifacts/mind-cir-20260916/progress/20260917T0221Z/verified_observation.json`。
