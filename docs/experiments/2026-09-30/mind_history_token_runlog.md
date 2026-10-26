# NI-only 历史类别 token 化实验

用户已授权启动并持续监控。两组作业均已通过全部 GPU 门禁并进入正式训练。两组均已完成训练和Full Dev并通过独立验收。Token125 R@10为19.8730%，Text125为20.2535%；远端同用户配对区间与最终产物哈希复核正在执行。

## 固定比较

| 条件 | 历史类别 | 模型新增词表 | 排名候选 | 训练 |
|---|---|---|---|---|
| Text125 | 原类别文本 | 125 个共享输入/输出 token | 原 108 类 | 冷启动 NI3 |
| Token125 | 对应 `<interest_NNN>` | 相同 125 个 token | 相同 108 类 | 冷启动 NI3 |
| 原 NI-only | 类别文本 | 原 108 个 token | 原 108 类 | 已有历史参考 |

主比较为 Token125−Text125；同时报告两组相对原 NI-only 的描述性差值。历史原基线有词表、单卡转四卡及内核路径差异，不作为严格单因素对照。没有 CIR、MeRA 正则、新模型或多 seed。

Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，seed42。两组均 21,600 行、3 epochs、全局 batch32、2,025 updates、64,800 次样本曝光；每组 Recall Standard 4×A10080GB，microbatch2、accumulation4，最多同时 8 GPU。LoRA r16/alpha32/dropout0.05，lr2e-5，原生全词表 completion-only token-mean CE，BF16、显式 Torch fallback、非 reentrant checkpointing，100 步持久化 checkpoint。

## 数据和初始化

原 108 个输出 token 编号及映射完全保留。17 个附加历史类别从 Train 历史取得，按名称排序追加为 `<interest_108>` 至 `<interest_124>`。Dev 不参与新类别定义。125 类均使用 label-token-mean-norm-matched-cpu-fp32-v2 初始化和 tied SID 可训练参数。

只替换历史条目方括号内类别；标题、空白、顺序、重复、历史长度、提示词、监督标签及目标候选全部保留。训练 messages 与 prompt 同步更新；Dev 使用相同转换，metadata 保留原语义类别以核验标签。全量逆转换逐字等于原 prompt。Text125 的 Train/Dev 文件与原文件字节一致。

两组都训练 125 行，以控制词表扩展、可训练参数量和全词表 CE 分母。附加 17 类不成为监督目标或评估候选。输入 token 化同时改变输入长度和类别语义呈现，因此主比较是输入表示的总体效果，不单独证明输入/输出对齐机制。提示词仍使用原来的类别描述，以固定指令文本。

## 验收与监控

冻结数据：`data/training/mind_history_token_20260930_v1`。总入口：`manifests/mind_history_token_20260930_v1.json`。每臂 manifest 记录精确 experiment、job、backend、results_prefix、代码包 SHA。

运行包基于已接受的 2026-09-29 NI direct 数值运行时。Trainer、loss、全局 sampler、checkpoint、exposure、fresh reload 和 merger 源码不变；映射解析、评估候选子集和 125 行形状验收按本任务修改，差异保存在 `artifacts/mind-history-token-20260930/*-runtime.patch`。

正式运行前执行：模型快照和依赖核对；Train/Dev token 长度审计；125 行 finite/nonzero 梯度；单卡 8 步 loss decrease；四卡 2 步；373 个张量精确 fresh reload；64 条真实 Dev 分片、合并、独立重算及持久化读回。首次单卡门禁建立冷启动状态 SHA，其余门禁和正式训练须一致；两臂 SHA 也须一致。

Full Dev 为 31,392 impressions、27,082 users、37,728 目标出现、104 支持类别；原 108 候选、合法无重复 Top20、无历史类别过滤，报告 macro Recall/NDCG@1/5/10/20。终验包含所有分片完整覆盖、独立指标重算、checkpoint/exposure/参数身份和远端产物 SHA。大权重、optimizer、完整预测留远端。

监控读取原始与 retry stdout、生命周期、梯度/门禁、最新已提交 checkpoint 和评估产物。平台 Running/pass 不等于验收。仅对真实阶段推进、异常、需要用户介入及完成做通知；普通排队和无变化保持安静。

本地验证：16 项针对性测试通过；实际 AMLT 上传清单各 121 文件，零缺失、零额外项；两臂训练 exposure/sampler 预算通过；完整 `pytest -q` 因本机缺 `peft` 在收集阶段失败，未声称全套通过。

## 提交与最新观测

北京时间 2026-09-30 16:35–16:36 的原生查询：

| 条件 | AMLT experiment | Backend | 状态 | 已提交训练 checkpoint |
|---|---|---|---|---|
| Token125 | mind-ni-history-token125-0930-v1 | salmon_plane_m3qq5sx0fm | Queued | 无 |
| Text125 | mind-ni-history-text125-0930-v1 | upbeat_lettuce_k6my50hhvb | Queued | 无 |

两臂均有独立 results_prefix，详见 arm manifest。30 分钟本聊天 heartbeat `ni-token` 已启用。提交、YAML dump、真实 uploader 清单和原生查询证据位于 `artifacts/mind-history-token-20260930/`；全量 Train 与 Dev token 长度检查均为零超长，Dev 最长 Text125=1525、Token125=1338。

### 17:05 独立进度核验

- Text125：原生 Running，正式训练日志 25/2025，最近 loss=3.541、grad_norm=28.41，日志中 loss/梯度范数/学习率均有限。125 行梯度、单卡8步、四卡2步、373张量和logits精确重载、64条Dev/75目标/36支持类的四分片与独立指标重算通过。冷状态 SHA `28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565`；正式训练冷状态一致。正式checkpoint尚未到100步持久化点。
- Token125：原执行因 `Pod evicted for compaction of fragmented capacity in cluster` 被平台中断；同一 backend 的 retry_001 已于08:55 UTC恢复Running，最新stdout无Traceback。环境、模型快照和全量token审计通过，梯度门禁已加载模型；尚无训练checkpoint。未重复提交、未改变科学配置。
- 两组初始状态的跨臂相等检查待 Token125 首个冷启动门禁产物；不提前声称已通过。证据：两臂 `progress/20260930T090514Z/receipt.json` 和 `independent_readback.json`。

### 17:35 独立进度核验

- Text125：正式日志 160/2025，最近 loss=1.603、grad_norm=10.06；有限数值检查通过。checkpoint-100 的身份、紧凑文件SHA、远端大文件大小和逐样本曝光序列独立验证，恰好3200次曝光；不是仅依据checkpoint目录存在。
- Token125：125行有限非零梯度、单卡8步（loss 4.974946→2.799891）、373张量和logits精确重载通过。四卡门禁正在初始化，09:34:35 UTC有新model_audit产物；没有新Traceback或抢占。尚未宣称四卡门禁或正式训练通过。
- 两臂完整冷启动参数SHA均为 `28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565`，跨臂一致性已核验。
- 本地监控读回增加中途checkpoint逐样本计数验证。真实CP100回放通过；保持总曝光不变但交换两个样本计数的故障注入被拒绝。该修改只涉及本地读回，已提交运行包及科学配置未修改。证据：两臂 `progress/20260930T093513Z/`。

### 18:05 两臂正式训练已启动

- Token125 的四卡2步、373张量与logits精确重载、64条Dev/75目标/36支持类的四分片合并与独立指标重算已通过。正式训练冷状态等于其门禁及Text125冷状态。原生日志52/2025，最近loss=1.843、grad_norm=13.22；尚未到首个正式CP100。
- Text125 原生日志296/2025，最近loss=1.585、grad_norm=9.226。CP200紧凑文件SHA、身份、远端大小及逐样本曝光序列通过，6400次曝光。
- 两组最新活动stdout都无Traceback/新终止信号；已检查的loss/grad_norm/lr均有限。门禁指标仅验证链路，不作实验效果比较。保持原作业、数据、配置和预算，未重复提交。证据：两臂 `progress/20260930T100548Z/`。

### 18:35 常规进度读回

两组均持续正式训练、无新错误或平台中断。Text125：426/2025，最近loss1.562，CP400逐样本计数和紧凑哈希核验通过（12800次曝光）；Token125：165/2025，最近loss1.570，首个正式CP100逐样本计数和紧凑哈希核验通过（3200次曝光）。本次读取的loss/grad_norm/lr均有限，两臂冷状态一致。科学配置及作业身份未变，无需恢复或用户操作。证据：两臂 `progress/20260930T103515Z/`。

### 19:05 常规进度读回

两组均持续正式训练，最新活动日志无新错误或中断。Text125：560/2025、loss1.533，CP500核验16000次曝光；Token125：278/2025、loss1.561，CP200核验6400次曝光。两份checkpoint的紧凑文件哈希、远端大小、训练身份和逐样本计数均通过；本次读取的loss/grad_norm/lr均有限。无新增阶段变化，无需恢复或用户操作。证据：两臂 `progress/20260930T110516Z/`。

### 19:35 常规进度读回

Text125：693/2025、最近loss1.526，日志已进入第二个epoch；CP600逐样本计数及紧凑哈希核验19200次曝光。Token125：396/2025、最近loss1.556；CP300核验9600次曝光。两组最新活动日志均无新异常或中断，读取到的loss/grad_norm/lr均有限；参数初始化、数据、配置和作业身份均未变。继续既定训练，无需恢复或用户操作。证据：两臂 `progress/20260930T113514Z/`。

### 20:05 常规进度读回

Text125：825/2025、最近loss1.500；CP800跨epoch逐样本曝光序列和紧凑哈希核验通过（25600次曝光）。Token125：518/2025、最近loss1.549；CP500核验16000次曝光。两组最新活动日志均无新异常或中断，读取到的loss/grad_norm/lr均有限；训练配置及作业身份未变。继续既定训练，无需恢复或用户操作。证据：两臂 `progress/20260930T120515Z/`。

### 20:35 常规进度读回

Text125：961/2025、最近loss1.511；CP900逐样本计数及紧凑哈希核验通过（28800次曝光）。Token125：640/2025、最近loss1.559；CP600核验19200次曝光。最新活动日志无新异常、中断或停滞，读取到的loss/grad_norm/lr均有限。冷状态、数据、配置及作业身份保持不变，无需恢复或用户操作。证据：两臂 `progress/20260930T123525Z/`。

### 21:05 常规进度读回

Text125：1093/2025、最近loss1.508；CP1000逐样本计数及紧凑哈希核验通过（32000次曝光）。Token125：763/2025、最近loss1.527，日志进入第二个epoch；CP700跨epoch计数核验22400次曝光。最新活动日志无新异常或中断，读取到的loss/grad_norm/lr均有限。两臂持续既定训练，未更改科学设置或提交新作业。证据：两臂 `progress/20260930T130523Z/`。

### 21:35 常规进度读回

Text125：1225/2025、最近loss1.500；CP1200紧凑哈希、训练身份和逐样本计数核验38400次曝光。Token125：887/2025、最近loss1.505；CP800核验25600次曝光。最新活动日志无新异常或中断，读取到的loss/grad_norm/lr均有限。未修改科学设置或作业，继续原训练。证据：两臂 `progress/20260930T133522Z/`。

### 22:05 常规进度读回

Text125：1357/2025、最近loss1.463，日志进入第三个epoch；CP1300紧凑哈希、训练身份及逐样本计数核验41600次曝光。Token125：1003/2025、最近loss1.497；CP1000核验32000次曝光。最新活动日志无新异常或中断，loss/grad_norm/lr均有限。继续原训练，不更改配置或作业。证据：两臂 `progress/20260930T140516Z/`。

### 22:35 常规进度读回

Text125：1492/2025、最近loss1.473；CP1400跨epoch逐样本计数与紧凑哈希核验44800次曝光。Token125：1115/2025、最近loss1.492；CP1100核验35200次曝光。最新活动日志无新异常或中断，读取到的loss/grad_norm/lr均有限。继续既定训练，无需恢复或用户操作。证据：两臂 `progress/20260930T143516Z/`。

### 23:05 常规进度读回

Text125：1625/2025、最近loss1.471；CP1600紧凑哈希、训练身份及逐样本计数核验51200次曝光。Token125：1228/2025、最近loss1.509；CP1200核验38400次曝光。最新活动日志无新异常或中断，读取到的loss/grad_norm/lr均有限。继续原训练，无需恢复或用户操作。证据：两臂 `progress/20260930T150526Z/`。

### 23:35 常规进度读回

Text125：1758/2025、最近loss1.501；CP1700紧凑哈希、训练身份及逐样本计数核验54400次曝光。Token125：1342/2025、最近loss1.474；CP1300核验41600次曝光。最新活动日志无新异常或中断，读取到的loss/grad_norm/lr均有限。继续原训练，无需恢复或用户操作。证据：两臂 `progress/20260930T153516Z/`。

### 10月1日00:05 常规进度读回

Text125：1892/2025、最近loss1.461；CP1800紧凑哈希、训练身份及逐样本计数核验57600次曝光。Token125：1451/2025、最近loss1.477，日志进入第三个epoch；CP1400跨epoch计数核验44800次曝光。最新活动日志无新异常或中断，读取到的loss/grad_norm/lr均有限。继续原训练，尚无完整Dev结果。证据：两臂 `progress/20260930T160516Z/`。

### 10月1日00:36 Text125训练更新结束

Text125已完成2025次优化更新、3个epoch。最终CP2025的紧凑文件SHA、训练身份和逐样本曝光序列通过，21600条样本每条恰好3次、总64800次。最终fresh reload的373个张量与4条样本logits均精确相等（atol=rtol=0），保存/加载状态SHA为 `8b843e1dfd77e1c5fe7d33f663f38df52935a7589372cb1a1b1f50b0adec62ce`。快照中阶段接受标记尚未返回，Full Dev仍待完成及独立验收，不把训练结束当完整实验接受。

Token125仍正常训练：1571/2025，CP1500逐样本计数核验48000次曝光。最新活动日志无新异常或终止信号。证据：Text125 `progress/20260930T163632Z/`，Token125 `progress/20260930T163516Z/`。

### 10月1日01:05 Text125完整验收

Text125 原生Completed，训练2025步/64800次曝光、373张量及logits精确重载、31392条Full Dev/104支持类/37728目标均通过独立读回。四个分片覆盖、候选合法性、原始预测hash和远端独立重算由worker验收；本地依据紧凑分片证据再次重算每类及macro指标。完整模型/optimizer/预测保留远端。接受记录：`artifacts/mind-history-token-20260930/text/final_acceptance.json`。

| Text125指标 | 百分比 |
|---|---:|
| Macro Recall@10 | 20.2535 |
| Macro Recall@20 | 33.3740 |
| Macro NDCG@10 | 10.0988 |
| Macro NDCG@20 | 13.4004 |

Token125继续正常训练，1685/2025；CP1600逐样本计数核验51200次曝光，读取到的loss/grad_norm/lr均有限。主比较Token125−Text125仍未产出，不用单臂结果推断输入token化效果。证据：两臂 `progress/20260930T170516Z/`。

### 两臂完整Dev结果（配对区间待CPU分析）

| 条件 | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| 原 NI-only（Text108，历史参考） | 20.3275% | 10.1325% | 33.6553% | 13.4663% |
| Text125（匹配对照） | 20.2535% | 10.0988% | 33.3740% | 13.4004% |
| Token125（输入替换） | 19.8730% | 10.0074% | 33.4679% | 13.4279% |

Token125相对Text125的R@10差值为−0.3804个百分点，R@20为+0.0938个百分点；不能概括为所有指标均下降或均改善。两组均完成2025步、64800次曝光、精确重载和31392条Full Dev验收。Token125完整验收来自原生Completed状态和 `token/progress/20260930T194122Z/`，该次跳过间歇失败的signed-log传输，仍保留原生状态与所有紧凑产物验证。

配对分析保持完整预测远端，使用同用户Exp(1)权重、2000次draw、seed42；仅估计本次固定模型的用户采样不确定性，不包含训练seed不确定性。历史Text108基线的词表/数值运行路径不同，只提供描述性参照。

CPU分析v1的payload误包含AMLT必定排除的`.amltignore`。真实uploader重现该唯一缺失；v2仅从payload清单移除此控制文件，9个实际科学输入/代码文件全部字节不变，实际上传清单10文件零缺失。保留v1记录和产物，提交独立v2分析；未重训或修改任何模型。恢复入口：`manifests/mind_history_token_analysis_20261001_v1.json`。

配对分析v2已提交：`mind-history-token-analysis-1001-v2` / `history-token-paired-analysis-v2`，backend `cool_panda_h84tqwcqpb`，CPU 5C3 Basic。03:50北京时间原生Queued；v1 `mighty_cartoon_r5d7vjsrlm`亦仍Queued，其源检查将拒绝缺失控制文件，因此不会执行科学分析。没有取消或覆盖旧作业。后续只接受v2精确身份的读回结果，验收脚本 `scripts/analysis/accept_history_token_comparison.py`。配对分析结果未产出，暂不声称统计显著差异。

10月1日03:53只读复核：分析v2仍为原生Queued，精确backend/results_prefix一致，尚无分析产物；训练已验收结果不变，不重复提交。证据：`analysis/progress/20260930T195328Z/receipt.json`。
