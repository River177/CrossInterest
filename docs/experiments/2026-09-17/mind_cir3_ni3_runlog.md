# MIND 108 类 CIR 3 epochs → NI 3 epochs

用户选择的唯一新增条件：Qwen3.5-2B，原 17,271 条固定 CIR 视图训练 3 epochs（1,620 步、51,813 次样本曝光），再在原 21,600 条 NI 数据上训练 3 epochs（2,025 步、64,800 次样本曝光）。不做标签打乱、多 seed、额外 NI 预训练对照或类别均衡。

CIR 从原始基座／SID 初始化开始，cosine 调度按完整 1,620 步建立。CIR→NI 继承所有 LoRA／SID 参数，重置 optimizer/scheduler；不从已经完成的一轮 CIR checkpoint 接着修改调度。固定 seed42、原 108 类 mapping、news/click 提示词、H50 先截断再整类删除且不回填、每源用户一个固定视图。其余设置与已验收 CIR1→NI3 相同：4×A100-80GB、Recall Standard、microbatch2、accumulation4、global batch32、BF16、非 reentrant checkpointing、显式 Torch fallback、native full-vocabulary completion-only token-mean SFT。

训练运行包克隆自已验收 CIR1 的 bundle-v3。仅更改时长合同与阶段编排，另附紧凑的原验收证据；83 个原 payload 文件保持字节一致。模型、数据、tokenizer、Trainer、loss、sampler、断点恢复、序列化和评估代码保持一致。原单卡／四卡与真实小型 Dev 检查已通过，按原 bundle、参数和收据哈希复用，不再重复训练诊断样本。新作业仍现场核验环境、模型 snapshot、全部输入 token、阶段状态、实际样本曝光及每阶段精确重载。原运行时包含已验证的非零 epoch 加非零 skip 恢复修复。

Full Dev 固定为 31,392 grouped impressions、37,728 个目标出现、104 个有支持类别，108 类 SID Top20 排名，不过滤历史类别。最终须核验完整覆盖、逐分片身份／哈希及远端独立指标重算，再回读本地紧凑统计。与已完成 CIR1→NI3 及既有 NI-only 比较；NI-only 的历史执行方式差异仍需注明。权重、optimizer、checkpoints 和完整预测留远端。

本地相关测试：28 passed，1 skipped（本地缺少 Transformers 的既有 CPU Trainer 用例）；真实冻结数据的三轮 ledger 和两个阶段的 launch config／cursor 合同均核对通过，AMLT 实际上传过滤及 YAML dump 已验证。最终效果尚未产生。

当前状态以 [跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir3_ni3_108_20260917_v1.json) 为准。监控沿用每 30 分钟检查，正常推进保持安静，仅通知实质异常、需要用户行动或最终验收完成。

相关入口：[实验配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/experiments/mind_cir_108_cir3_ni3_v1.json)、[AMLT 作业定义](../../../configs/amlt/archive/2026-10-26/mind_cir3_ni3_108_full_0917_v1.yaml)、[本地预检](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/local_preflight.json)、[原 CIR1 最终报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-16/mind_cir_warmup.md)。

提交实验 `mind-cir3-ni3-108-0917-v1`，job `qwen35-2b-cir3-ni3-108`，backend `nifty_avocado_n4yyg08wwd`。已由后端接收，当前排队：[作业页面](https://aka.ms/amlt?q=k2mmn)。结果前缀 `projects/sft4safety/amlt-results/7210412790.94297-54d6cdaf-7b23`；实际阶段进度以最新原生观察为准。

2026-09-17 04:07 UTC 原生观察：正式 CIR 90/1,620，最近 loss=1.358，所见 loss／梯度范数／学习率均有限。回读已核对运行包、四卡环境、模型 snapshot、完整 token 审计、复用 gate 来源、三轮游标及原始参数／零 optimizer/scheduler 初始化。此次结果列表尚未出现 checkpoint-100；未把日志进度当作持久化检查点，NI 尚未开始。收据 `artifacts/mind-cir3-20260917/progress/20260917T0407Z/verified_observation.json`。

2026-09-17 04:39 UTC：正式 CIR 推进至 222/1,620，最近 loss=1.273，所见训练数值均有限。checkpoint-200 已提交，元数据哈希、三轮训练身份与前 6,400 个样本的冻结顺序通过回读；NI 尚未开始，无新增异常。权重及 Adam 状态留远端。收据 `artifacts/mind-cir3-20260917/progress/20260917T0439Z/verified_observation.json`。

2026-09-17 05:11 UTC：正式 CIR 推进至 355/1,620，最近 loss=1.324，所见数值有限。checkpoint-300 的训练身份、元数据哈希及前 9,600 个样本的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0511Z/verified_observation.json`。

2026-09-17 05:42 UTC：正式 CIR 推进至 483/1,620，最近 loss=1.325，所见数值有限。checkpoint-400 的训练身份、元数据哈希及前 12,800 个样本的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0542Z/verified_observation.json`。

2026-09-17 06:14 UTC：正式 CIR 推进至 615/1,620，已进入第 2 epoch，最近 loss=1.23，所见数值有限。checkpoint-600 的三轮身份、元数据哈希与 19,191 次样本曝光核对通过，覆盖第一轮全部 17,271 个样本及第二轮前 1,920 个样本；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0614Z/verified_observation.json`。

2026-09-17 06:46 UTC：正式 CIR 推进至 748/1,620，最近 loss=1.262，所见数值有限。checkpoint-700 的训练身份、元数据哈希及 22,391 次样本曝光的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0646Z/verified_observation.json`。

2026-09-17 07:17 UTC：正式 CIR 推进至 877/1,620，最近 loss=1.231，所见数值有限。checkpoint-800 的训练身份、元数据哈希及 25,591 次样本曝光的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0717Z/verified_observation.json`。

2026-09-17 07:49 UTC：正式 CIR 推进至 1010/1,620，最近 loss=1.251，所见数值有限。checkpoint-1000 的训练身份、元数据哈希及 31,991 次样本曝光的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0749Z/verified_observation.json`。

2026-09-17 08:22 UTC：正式 CIR 推进至 1147/1,620，已进入第 3 epoch，最近 loss=1.182，所见数值有限。checkpoint-1100 的三轮身份、元数据哈希与 35,182 次曝光核对通过，覆盖前两轮各 17,271 个样本及第三轮前 640 个样本；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0822Z/verified_observation.json`。

2026-09-17 08:53 UTC：正式 CIR 推进至 1280/1,620，最近 loss=1.219，所见数值有限。checkpoint-1200 的训练身份、元数据哈希及 38,382 次样本曝光的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0853Z/verified_observation.json`。

2026-09-17 09:47 UTC：正式 CIR 推进至 1503/1,620，最近 loss=1.249，所见数值有限。checkpoint-1500 的训练身份、元数据哈希及 47,982 次样本曝光的冻结顺序通过回读；NI 尚未开始，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T0947Z/verified_observation.json`。

2026-09-17 10:19 UTC：CIR 1,620 步／3 epochs 已完成并验收，17,271 个样本各 3 次，共 51,813 次曝光；108 类计数、最终 checkpoint 游标及 373 个 tensor／logits 精确重载均通过。本组 CIR3 adapter `dceda49eaaf4fa258ced1fe8ba7146dc494b6cb1d51d26855906e5270abd2c77` 已交接给 NI，全部参数继承且 optimizer/scheduler 从零开始。NI 当前 5/2,025，最近 loss=1.741，数值有限；NI 尚未到首个持久化检查点。收据 `artifacts/mind-cir3-20260917/progress/20260917T1019Z/cir_final_handoff_readback.json`。

2026-09-17 10:52 UTC：NI 正常推进至 150/2,025，最近 loss=1.506，所见数值有限。NI checkpoint-100 已提交，本组 CIR3 父 adapter 身份、三轮训练设置、元数据哈希及前 3,200 个 NI 样本的冻结顺序通过回读；无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T1052Z/verified_observation.json`。

2026-09-17 11:24 UTC：NI 正常推进至 282/2,025，最近 loss=1.541，所见数值有限。NI checkpoint-200 的本组 CIR3 父 adapter 身份、训练设置、元数据哈希及前 6,400 个样本的冻结顺序通过回读；无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T1124Z/verified_observation.json`。

2026-09-17 11:56 UTC：NI 正常推进至 417/2,025，最近 loss=1.521，所见数值有限。NI checkpoint-400 的本组 CIR3 父 adapter 身份、训练设置、元数据哈希及前 12,800 个样本的冻结顺序通过回读；无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T1156Z/verified_observation.json`。

2026-09-17 12:27 UTC：NI 正常推进至 551/2,025，最近 loss=1.497，所见数值有限。NI checkpoint-500 的本组 CIR3 父 adapter 身份、训练设置、元数据哈希及前 16,000 个样本的冻结顺序通过回读；无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T1227Z/verified_observation.json`。

2026-09-17 12:59 UTC：NI 正常推进至 686/2,025，原生日志确认第 1 epoch 尾批通过并进入第 2 epoch，最近 loss=1.483，所见数值有限。NI checkpoint-600 的本组 CIR3 父 adapter 身份、训练设置、元数据哈希及前 19,200 个样本的冻结顺序通过回读；该已提交检查点仍属于第 1 epoch，无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T1259Z/verified_observation.json`。

2026-09-17 13:31 UTC：NI 正常推进至 821/2,025，最近 loss=1.495，所见数值有限。NI checkpoint-800 的本组 CIR3 父 adapter 身份、训练设置、元数据哈希及 25,600 次样本曝光的冻结顺序通过回读，覆盖第一轮全部 21,600 个样本及第二轮前 4,000 个样本；无新增异常。收据 `artifacts/mind-cir3-20260917/progress/20260917T1331Z/verified_observation.json`。

2026-09-17 14:14 UTC：平台于 13:34 因更高 SLA 回收节点，同一 backend 自动启动 retry_001。已验收 CIR 完整复用；NI 从原 checkpoint-800 恢复，373 个 Adam 状态及 scheduler 游标均为 800，原 cosine 学习率和 25,600 次样本曝光前缀核对通过。旧 attempt 最后日志为 830；当前 retry 实际推进至 866/2,025，最近 loss=1.476，数值有限。未重提或修改训练；下一新检查点尚未到 900。恢复收据 `artifacts/mind-cir3-20260917/progress/20260917T1414Z/automatic_resume_readback.json`。

2026-09-17 14:48 UTC 心跳：首次原生日志读取遭遇代理连接断开，只读重试恢复。retry_001 正常推进至 987/2,025，最近 loss=1.484，数值有限。恢复后新 checkpoint-900 已提交，其 CIR3 父 adapter 身份、元数据哈希及 28,800 次样本曝光顺序通过回读；无新增训练异常或重提。实际观察时间以收据 `artifacts/mind-cir3-20260917/progress/20260917T1448Z-retry/verified_observation.json` 为准。

2026-09-17T17:50:59.609751+00:00：本次原生日志的 Blob 读取及一次重试超时，未将旧步数当作当前进度，也未重启或重提训练。AMLT 状态仍为 running；独立回读已核验更新的 NI checkpoint-1600、本组 CIR3 父 adapter 身份及 51200 次样本曝光顺序，检查点所记录数值有限。实时日志步数暂未知，继续按既定间隔监控。收据 `artifacts/mind-cir3-20260917/progress/20260917T1659Z/checkpoint_observation.json`。
