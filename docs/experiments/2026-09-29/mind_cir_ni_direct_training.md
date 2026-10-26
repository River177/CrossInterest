# CIR构造NI与原NI：1,055对类别频数匹配的独立训练

## 范围与状态

用户已批准启动两组：cir_as_ni与original_ni_paired。使用同一批1,055用户、同来源记录、相同98类别及逐类计数。CIR目标在原H50出现至少2次；不重选目标、不加seed/sweep、不修改其他聊天作业。两组平台终态均为Completed、退出码0，已通过 `accept_cir_ni_direct.py` 独立最终紧凑验收，包括99步/3,165曝光、全部373张量精确重载、Full Dev四分片覆盖与macro指标重算。CIR构造NI的R@10为13.7849%，配对原NI为12.4049%。远端完整预测重算与2,000次用户成对Bayesian bootstrap均已验收：CIR在四项预定指标上更高，逐指标95%区间均为正。结论限于本1,055对、3轮、seed42的受控比较，不直接证明标签噪声更低。身份以 [跟踪清单](../../../manifests/mind_cir_ni_direct_training_20260929_v1.json) 为准。

## 最终结果（完整预测与成对比较已验收）

两组均为相同1,055名用户、相同98个训练类别及逐类样本数，独立冷启动SFT3轮、99更新；无后续NI微调。指标单位为%，差值为“CIR构造NI − 原NI”的百分点。

| 指标 | 原NI | CIR构造NI | 差值（百分点） | 用户成对95%区间（百分点） |
|---|---:|---:|---:|---|
| Macro R@10 | 12.4049 | 13.7849 | +1.3800 | [+0.9845, +1.7736] |
| Macro R@20 | 22.7448 | 24.2566 | +1.5118 | [+0.9269, +2.1082] |
| Macro NDCG@10 | 5.9451 | 6.6596 | +0.7146 | [+0.5519, +0.8738] |
| Macro NDCG@20 | 8.5316 | 9.2833 | +0.7517 | [+0.5702, +0.9415] |

主指标R@10提升1.3800个百分点（相对提升约11.12%）；其他三项指标也为正，用户成对区间均不包含零。这支持：在本次相同用户、相同监督数量、相同标签分布和相同更新预算下，使用重复历史兴趣构造的CIR监督，较原始未来NI监督获得更好的真实NI预测泛化。不能用“CIR组样本更多”或“两组目标类别频数不同”解释这次差异。

该结果尚不能证明CIR标签噪声更低：历史删除、可见历史长度与内容、标签语义来源也不同。两组仅各1,055样本、每类1–32条，基于双方训练标签筛出的队列不代表全部用户。仅一个训练seed；bootstrap反映固定模型在评估用户层面的不确定性，不包含跨训练seed变化。四项区间为未做多指标校正的逐指标区间，主指标事先固定为R@10。也不能把本结果外推为普通CIR5→NI3收益已被完全解释，或忽略此前9,431对前缀NI对照的相反主指标结果。

远端逐条验证31,392条impressions、27,082用户、37,728目标出现、104支持类、108候选合法无重复Top20，完整预测SHA与两组已验收结果一致；所有@1/@5/@10/@20 recall与ndcg重算误差小于1e-12。Bayesian bootstrap使用同一用户共享Exp(1)权重，逐次重算类别分母再macro平均，共2,000次、seed42。

CPU分析v2 backend `sweet_bone_4s0nl1n40x` 于05:24:28 UTC程序退出0，05:28:02 UTC平台Completed；只修复了CPU SLA为Basic，未更改分析payload或训练。结果和源码持久保存在同一Azure账户/amulet容器：分析结果prefix `projects/sft4safety/amlt-results/7209369952.25123-1d6c3461-b168`，code prefix `projects/sft4safety/amlt-code/1e4c0e62-ae56-4094-8e46-60a289c4e98a`。分析实际环境Python3.10.20、NumPy2.2.6、PyArrow20.0.0，CPU执行。全量模型和预测未下载到Mac。

证据：[联合验收](../../../artifacts/mind-cir-ni-direct-training-20260929/paired_final_acceptance.json)、[完整预测重算与区间](../../../artifacts/mind-cir-ni-direct-training-20260929/comparison.json)、[分析环境与hash链](../../../artifacts/mind-cir-ni-direct-training-20260929/analysis_receipt.json)、[结果CSV](mind_cir_ni_direct_training_results.csv)。

## 冻结训练合同

两组各从Qwen/Qwen3.5-2B原基座与同一LoRA/SID初始化独立训练3轮，1055×3=3165次样本曝光；global batch32，每轮33更新，合计99步。无辅助到NI的第二阶段。各4×A10080GB、Recall Standard，最多8卡。采用原NI三轮训练预算，不按新评估结果选epoch或延长训练。

seed42、lr2e-5、cosine/warmup0.03、AdamW、LoRA r16/alpha32/dropout0.05、108 SID行、原生全词表completion token-mean CE、非reentrant checkpointing及显式Torch fallback。每卡microbatch2、梯度累积4。原初始化状态hash为 `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`。为短训练每25步保存可恢复checkpoint，最终99步也提交；断点恢复保持optimizer/scheduler、四rank RNG及精确样本游标。

运行包复用上一轮已接受runtime的Trainer、采样器、loss、checkpoint与评估文件，仅增加单阶段编排及更新数据、预算和保存频率。兼容路径cir.parquet与ni.parquet在每个包内是同一组数据的两个副本，正式训练仅执行一次ni阶段，不能把路径名当作额外训练。

## 门禁与评估

各组先完成32条单卡8步、41条四卡2步门禁、373张量/logits零容差重载和64条真实Dev评估；正式训练从冷启动重新开始，不继承门禁更新。始终使用原完整Full Dev：31,392 impressions、27,082用户、37,728目标出现、104支持类，108 SID候选、合法Top20、无历史类别过滤。主指标Macro R@10，辅以R@20与NDCG@10/@20。

20项相关数据/编排测试通过，另4项编排/观测器测试通过；两份YAML实际解析及每组116个真实上传文件hash审计通过。全套本地pytest在收集时因缺少peft中止，记录在full_pytest.log，不能报告全套通过。远端采用已接受的完整精确依赖并独立核对。

大权重、optimizer、checkpoint和完整预测保留远端，本地只存紧凑日志、hash、指标和恢复指针。最终分别确认平台Completed与退出码0，再运行 `scripts/analysis/accept_cir_ni_direct.py` 独立核对完整训练/评估收据；两组预测在远端重算指标并做2,000次用户成对Bayesian bootstrap，seed42，报告训练单seed边界。

## 解释边界

两组目标频数相同，但98类训练支持并不均匀，每类1–32条。历史内容、长度和token计算量不同；类别频数匹配队列使用双方训练标签选择，未使用Dev或模型效果。此比较检验在受控用户及类别分布下两种监督构造的迁移效果，不能单凭性能差值证明标签噪声更低。数据文件与统计见 [数据审计](mind_cir_ni_direct_balanced_data.md)。

本聊天自动跟进 `cir-ni-3` 已在两组与最终成对比较验收后停止（PAUSED），保留聊天。

## 提交身份与首次原生观察

| 条件 | AMLT experiment | job | backend | 03:33 UTC状态 |
|---|---|---|---|---|
| cir_as_ni | mind-cir-ni-direct-cir-0929-v1 | qwen35-2b-cir-direct3 | green_branch_99tl87syp5 | Queued |
| original_ni_paired | mind-cir-ni-direct-ni-0929-v1 | qwen35-2b-ni-direct3 | busy_plane_7p45bm2ftx | Queued |

CIR结果prefix：`projects/sft4safety/amlt-results/7209376522.69574-8accad38-b71b`；原NI结果prefix：`projects/sft4safety/amlt-results/7209376494.84590-e24b384c-3e3c`。首次观察及生命周期见各arm manifest的latest_observation。代码上传后CLI返回有数分钟延迟；只读核对云端已上传116文件及原生平台后获得正确身份，未重复提交或取消作业。

## 2026-09-29 03:58 UTC：启动与训练进度独立回读

两组各自完成单卡8步、四卡2步及373张量/logits零容差重载。64条真实Dev门禁评价的四分片指标已依据冻结Dev逐类独立复算，75个目标、36个支持类一致。实际GPU均4×A10080GB，精确依赖与snapshot一致。

两组正式训练均从同一原始参数hash `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e` 和空optimizer状态开始，不继承门禁更新。CIR checkpoint-50的模型/optimizer/scheduler/四rank RNG等文件大小与提交清单一致，紧凑元数据hash及跨epoch的1,599次曝光计数和样本游标一致。大文件没有下载到Mac。当前loss及梯度诊断均有限；原NI的早期loss已经下降，不以两组不同步数的loss比较效果。

收据：[CIR构造NI](../../../artifacts/mind-cir-ni-direct-training-20260929/cir_as_ni/progress/20260929T035750Z/startup_progress_readback.json)、[原NI](../../../artifacts/mind-cir-ni-direct-training-20260929/original_ni_paired/progress/20260929T035750Z/startup_progress_readback.json)。后续可用 `scripts/analysis/verify_cir_ni_direct_progress.py --manifest <manifest> --receipt <receipt>` 对运行中快照交叉核对；平台完成后仍使用独立最终验收器。本轮未修改或重启远端作业。

## 2026-09-29 04:28–04:31 UTC：两组训练完成

两组均通过最终99步checkpoint紧凑hash、实际每条样本3次曝光及阶段独立冷启动核对，未出现第二NI阶段。最终adapter的373张量和logits零容差重载均通过。收据：[CIR构造NI](../../../artifacts/mind-cir-ni-direct-training-20260929/cir_as_ni/progress/20260929T042750Z/training_completion_readback.json)、[原NI](../../../artifacts/mind-cir-ni-direct-training-20260929/original_ni_paired/progress/20260929T042750Z/training_completion_readback.json)。

CIR程序于04:28:10 UTC退出0，最新原生状态查询04:30:49 UTC仍为Running，不提前当作平台终态。原NI程序仍在评估，两组独立最终验收及远端成对完整预测分析待续。

## 2026-09-29 04:58 UTC：两组最终紧凑验收通过

CIR作业于04:31:23 UTC进入Completed，原NI于04:51:54 UTC进入Completed，均有程序退出0的原生生命周期证据。两组各31,392条Full Dev、37,728目标出现、104支持类，四分片指标与冻结真值支持数独立复算一致。收据：[CIR](../../../artifacts/mind-cir-ni-direct-training-20260929/cir_as_ni/final_acceptance.json)、[原NI](../../../artifacts/mind-cir-ni-direct-training-20260929/original_ni_paired/final_acceptance.json)。

原计划gcr-a100 CPU分析在SSH连接阶段两次超时，未执行分析；转为独立AMLT CPU作业 `mind-cir-ni-direct-analysis-0929-v1 / direct-paired-prediction-analysis`，Feeds Standard 5C3，0 GPU。该作业只从持久化存储读取两组已验收预测，校验SHA、重算指标并做预定2,000次用户成对Bayesian bootstrap；没有追加训练或改变两组条件。CPU分析使用NumPy2.2.6、PyArrow20.0.0，完整输入、代码及接受的预测hash冻结在分析运行包。状态与输出prefix以manifest的paired_analysis为准；提交回执延迟时不得重复提交。


CPU分析v1在平台提交校验阶段失败，尚未运行任何代码：`Batch Spot-CPU jobs are not supported in Standard SlaTier. Only Basic SlaTier is supported.` 原生backend `dynamic_cow_1nkp3sk32v` 于05:09:58 UTC终态Failed；没有自动恢复中的活跃分析任务。证据为 `artifacts/mind-cir-ni-direct-training-20260929/analysis_failure_v1.json`。只将CPU调度级别改成Basic，保持同一分析payload与全部输入hash，提交 `mind-cir-ni-direct-analysis-0929-v2 / direct-paired-prediction-analysis-v2`。独立analysis manifest保留旧attempt，active_run指向v2。此修复没有重启或追加任何模型训练。


05:17 UTC：CPU分析v2已由原生平台确认创建，backend `sweet_bone_4s0nl1n40x`，results prefix `projects/sft4safety/amlt-results/7209369952.25123-1d6c3461-b168`。原生状态快照Queued，生命周期05:17:28记录ScheduleSelected，尚无分析stdout或结果。不会重复提交。观测器已支持只读取comparison.json和analysis_receipt.json等紧凑JSON；CPU终态Completed后运行 `scripts/analysis/accept_cir_ni_direct_comparison.py --receipt <CPU最新receipt.json>`，验证原两组prediction/metric hash、完整Dev范围、分析源码/payload、2,000次bootstrap并生成联合验收。自动跟进cir-ni-3保持开启，待最终比较通过后停止。

2026-09-29 05:28 UTC之后：CPU分析已确认平台Completed与退出码0，全部完整预测/指标/区间和持久化hash链通过独立回读，联合验收完成，自动跟进cir-ni-3已停止。
