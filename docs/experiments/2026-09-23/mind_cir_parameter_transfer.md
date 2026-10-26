# MIND CIR参数迁移：保留LoRA或SID

状态：两组已完成训练、完整Dev、全部诊断和独立整体验收，平台终态均Completed。最终结果见 [参数迁移实验最终报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-23/mind_cir_parameter_transfer_results.md)。

| 条件 | NI初始LoRA | NI初始SID | NI预算 | AMLT |
|---|---|---|---|---|
| 保留LoRA、重置SID | 已接受CIR5 | 精确原始初始化 | 3 epochs / 2,025 updates | [keep_lora](https://aka.ms/amlt?q=k3x7r) |
| 保留SID、重置LoRA | 精确原始初始化 | 已接受CIR5 | 3 epochs / 2,025 updates | [keep_sid](https://aka.ms/amlt?q=k3x7s) |

两部分在NI阶段均可训练，共372个LoRA张量及1个108×2048的SID行张量。冻结预训练基座Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。复用原CIR5 adapter，不新增CIR训练、不重训NI-only、不增加其他训练条件或seed。完整CIR5→NI3采用已接受结果作为主要参照；历史NI-only仅为背景参考，不能据此做严格2×2贡献分解。

每组原21,600条NI、3轮／64,800次有效曝光、seed42、news/click、108类SID；新optimizer/scheduler，lr2e-5、cosine horizon2,025、warmup0.03、weight decay0.01，LoRA r16/alpha32/dropout0.05。原生full-vocabulary completion-only token-mean SFT，BF16、显式Torch fallback、非reentrant checkpointing。每组4×A10080GB Recall Standard、microbatch2/accumulation4/global32，两个作业最多8GPU并发。

## 初始化与运行门槛

原冻结运行包103个payload文件逐字节复用，新运行包仅添加参数组合、诊断和编排；所有原模型、数据、训练器、数值路径与评价代码不变。使用原训练构造器生成冷初始化，在optimizer创建前捕获并终止，逐张量哈希必须与保存的原始状态一致。将指定参数组与CIR5组合后，检查全部373个张量的来源、shape、dtype、内容及两组可训练性，保存组合adapter，再用新进程要求4条参考logits与全部张量零误差重载。该捕获过程不执行优化更新。

原始状态manifest摘要：`f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`；CIR5状态摘要：`dc57ec586aef1bf6fcc573ffbbc5fc5e441e551b69e7ce3b4685b7ea828e6e17`；CIR5 adapter文件SHA256：`1201a6e750cf65ca57940613df177a8caab14b594827f4496f84c7643f283df9`。SID重置为原始语义初始化，LoRA恢复完整A/B矩阵，不重新抽随机值。

两组继承原已接受的单卡／DDP／重载／64条真实Dev数值运行门槛；新组合另做零更新构造与精确重载检查。低初始准确率不构成取消消融的理由，哈希／身份／非有限数值错误则停止。检查点按原协议保存，保留非零epoch与skip的精确恢复。沿用已验证的有限EIO读取／字节复制重试，不吞掉身份或哈希错误。

## 独立功能诊断与正式评价

NI前后均评估：

- 类别名称→SID：固定108条，每类1条，无用户历史，作为直接类别输出映射诊断。
- 历史类别重建：从27,082名独立Dev用户中取每人一个固定历史，16,158人满足擦除后至少5条行为；按固定hash选512人，覆盖87类，不做类别平衡。目标只由原历史决定，不读取未来NI标签，与训练用户及原CIR训练输入均零重叠。

两类数据均为评价用途，不能进入训练；诊断与正常NI结果分开保存。keep_lora作业额外对已有完整CIR5和已有完整CIR5→NI3执行同样两项诊断作为前后参照，仅评价，不增加训练。

正式对照使用NI第3轮终点。Full Dev为31,392 grouped impressions、37,728目标出现、104支持类、108 SID Top20、无历史过滤、四分片各7,848。远端逐预测重算并检查分片身份，本地只读回紧凑指标及收据后独立重算。另在远端对齐已接受完整继承模型的预测，输出逐类差值及Top20分辨率下的成对排名改善／下降数量。没有额外bootstrap或多seed训练，不能作训练稳定性结论。

完整继承参照的Recall@10/Recall@20/NDCG@10/NDCG@20为21.9717%／35.0756%／11.2051%／14.4966%。

## 验证与定位

相关测试34 passed，包括精确参数组合、源状态不变、safetensors回读、错误键／形状／dtype／哈希拒绝、诊断目标独立于未来标签、用户隔离、配对排名差值及原CIR／schedule／存储重试测试。全仓库`pytest -q --maxfail=1`在已有`tests/test_compare_rho_students.py`收集阶段因本地缺少SciPy停止，未宣称全量通过。实际AMLT配置dump、真实上传过滤（117文件）和6项关键上传源文件逐字节回读通过；原始CIR5及完整继承参照的远端文件存在性已核对。大权重、optimizer、checkpoints与完整预测只留远端。

当前实验：`mind-cir-param-transfer-108-0923-v2`。

| arm | job | backend | results prefix |
|---|---|---|---|
| keep_lora | qwen35-2b-keep-lora-ni3 | elated_gas_w63m65bclg | projects/sft4safety/amlt-results/7209875533.62677-7997be68-62f4 |
| keep_sid | qwen35-2b-keep-sid-ni3 | amusing_cumin_p4r9y8d73h | projects/sft4safety/amlt-results/7209875533.62651-b3ae217f-0140 |

当前运行包：`artifacts/mind-cir-parameter-20260923/bundle-v2`；manifest SHA256：`4af6c06788ae2b3365ec49319cc314d83dc1a50a3f44e827cb0e5c6fa4b93b05`。大产物不下载到Mac。

- [跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_parameter_transfer_20260923_v1.json)
- [实验方案](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/research/2026-09-23_cir_parameter_transfer_plan.md)
- [运行配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/amlt/mind_cir_parameter_transfer_0923_v2.yaml)
- [本地预检查](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/local_preflight.json)
- [相关测试](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/focused_tests.txt)
- [上传回读](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/uploaded_source_readback_v2.json)
- [最终独立验收脚本](/Users/river/Documents/MSRA/Rec/CrossInterest/scripts/analysis/accept_cir_parameter_transfer.py)

## 启动接口修复

2026-09-23 v1的keep_lora在组合参数来源收据生成时失败，尚未创建optimizer或开始NI。完整日志定位到PEFT 0.18.0的set_peft_model_state_dict原地重命名trainable-token键（追加.default），导致之后按原始名字访问provenance时KeyError。此前的冷初始化逐张量校验、组合后模型与预期状态校验和参考logits生成均已通过；失败发生在收据字典生成。

修复仅将传给PEFT的state字典及tensor复制，保留原始命名和来源记录不被外部API修改。加入该实际键变换的回归测试后，35项相关测试通过。新bundle-v2仅修改新编排模块，原103个运行payload和全部数据字节不变；bundle SHA256为4af6c06788ae2b3365ec49319cc314d83dc1a50a3f44e827cb0e5c6fa4b93b05。

旧keep_sid在环境安装后收到平台SIGTERM并重新排队，未开始NI；已精确取消该旧作业，终态killed。旧keep_lora为failed，证据保留在artifacts/mind-cir-parameter-20260923/incidents/composition-v1/。新实验mind-cir-param-transfer-108-0923-v2按相同两个条件重新提交，最多仍为8张GPU。当前不能把已提交或Running标记当作组合重载及正式训练已验收。

2026-09-23 09:09 UTC启动回读：两个v2作业的环境／模型snapshot均与冻结源一致；全部373张量来源及组合摘要独立核验通过，4条参考logits与373张量精确重载均为零误差。keep_lora组合摘要913008ea5875ab7ee638058443f7fd51a62fce65e6b66b0a7c136416c79ef026，与v1已正确生成的候选参数完全一致；keep_sid组合摘要5c58b50d0d729964d2590ac7bb15c30857910447af5a0a28d58a940870f534c1。PEFT收据接口修复已通过真实GPU组合／重载验证。两组进入NI前功能诊断，正式NI更新尚未从日志确认，不能把该初始化验收当作完整训练验收。v2 keep_sid在首次启动前被平台以capacity compaction回收，已在同一backend自动重新调度，当前初始化门槛通过。自动监控仍关闭。

启动阶段原生检查（2026-09-23T09:12:45.925771+00:00）：两个当前backend均Running，无新异常，组合参数与零误差重载已接受；日志尚未记录正式NI更新，当前为NI前诊断流程。

## 自动监控恢复

2026-09-23按用户最新要求开启本任务heartbeat `cir`，每30分钟检查当前keep_lora与keep_sid两组。正常推进时静默；遇到故障在冻结实验条件内自主定位、修复、验证和断点恢复，仅在实质故障、需要用户操作或两组最终验收完成时通知。两组均独立验收后删除监控。既有旧实验不重新监控或重跑。该记录仅确认监控激活，不更新最近实际训练步数。

## 2026-09-23 09:54–09:58 UTC监控

两组均为Running并已进入正式NI训练。keep_lora当前stdout为136/2025，最新记录loss1.538；CP100元数据与marker哈希、全部模型/Adam/scheduler/四卡RNG远端文件尺寸及存在性核验通过，3,200次有效曝光逐样本匹配固定随机排列。大文件字节未下载；最终完整字节及Adam状态内容仍由远端终点验收负责。

keep_sid于09:31 UTC被更高SLA资源抢占，平台自动重排；最新retry_001已重新进入NI并运行5/2025，loss5.223且数值有限。此前attempt最高20步未达到第100步检查点，故CHECKPOINT_AUTO_RESUME_NONE从同一已验证composite初始化开始；20步是丢弃的旧attempt进度，不累加为当前有效步数。本次未取消或新增作业，没有修改实验合同。

两组全部373张量的composite来源、NI初始继承、新optimizer/scheduler及冻结超参数已复核。NI前108条SID名称诊断和512条/87类历史重建诊断均按四分片紧凑指标独立重算通过；这不构成最终NI结果。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T095404Z/monitoring_summary.json`及各arm的`metadata_verification.json`。保持30分钟监控。

## 2026-09-23 10:30 UTC监控

两组当前backend均Running，无新增平台驱逐或代码错误。keep_lora为271/2025步，最新loss1.575；keep_sid的当前retry_001为128/2025步，最新loss1.591。两条当前日志的训练诊断均为有限数值。

独立核验keep_lora CP200（6,400次曝光）及keep_sid CP100（3,200次曝光）：元数据SHA及同组composite父adapter身份正确，各样本计数、监督token及固定排列游标一致；模型、Adam、scheduler和四卡RNG远端文件齐全，尺寸与提交marker一致。大文件未下载或在本地重算字节hash，完整终点验收仍待完成。keep_sid旧attempt的20步不计入当前进度。本轮无修改或重提作业，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T102934Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 11:04–11:05 UTC监控

两组当前backend均Running，无新增平台驱逐或代码错误。keep_lora为402/2025步，最新loss1.541；keep_sid的当前retry_001为245/2025步，最新loss1.572。两条当前日志的训练诊断均为有限数值。

独立核验keep_lora CP400（12,800次曝光）及keep_sid CP200（6,400次曝光）：元数据SHA及同组composite父adapter身份正确，各样本计数、监督token及固定排列游标一致；模型、Adam、scheduler和四卡RNG远端文件齐全，尺寸与提交marker一致。大文件未下载或在本地重算字节hash，完整终点验收仍待完成。keep_sid旧attempt的20步不计入当前进度。本轮无修改或重提作业，继续30分钟监控。

keep_lora首次日志请求出现只读网络超时，重读成功，未触发任何训练重启。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T110204Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 11:38 UTC监控

keep_lora继续训练至530/2025步，loss1.525且数值有限；CP500已验证16,000次精确样本曝光。keep_sid的retry_001在338步附近再次被更高SLA资源抢占（11:32:47 UTC），平台11:32:56自动重排，retry_002已于11:35:39启动。新attempt当前为环境/模型准备阶段，尚无正式步数或checkpoint恢复完成标记；不能将旧338步当作当前进度。

keep_sid CP300已提交且身份正确，9,600次曝光逐样本与固定排列一致。两组所查检查点元数据SHA、composite父adapter身份、监督token/样本游标通过，模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸与marker一致；大文件未下载。新attempt的模型/optimizer/scheduler实际载入及继续训练留待下一次日志核验，已在清单设置requires_resume_verification。两组冻结参数及来源未变，本轮未取消或重提作业。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T113734Z/monitoring_summary.json`及各组`metadata_verification.json`。继续30分钟监控。

## 2026-09-23 12:12 UTC监控

两组当前backend均Running，无新增平台驱逐或代码错误。keep_lora为670/2025步，最新loss1.498；keep_sid的当前retry_002为365/2025步，最新loss1.549。两条当前日志的训练诊断均为有限数值。

独立核验keep_lora CP600（19,200次曝光）及keep_sid CP300（9,600次曝光）：元数据SHA及同组composite父adapter身份正确，各样本计数、监督token及固定排列游标一致；模型、Adam、scheduler和四卡RNG远端文件齐全，尺寸与提交marker一致。大文件未下载或在本地重算字节hash，完整终点验收仍待完成。keep_sid旧attempt的338步不计作新attempt进度。本轮无修改或重提作业，继续30分钟监控。

keep_sid恢复收据确认：实际global_step、373个Adam状态的step及scheduler_last_epoch均为300，优化状态未重置，学习率与原2025步cosine一致。CHECKPOINT_AUTO_RESUME_READY由远端完整checkpoint字节校验并复制成功后输出。当前已推进365步；本轮尚未出现恢复后的新CP400，因此下一次需核验其样本曝光连续性，post_resume_checkpoint_verification_pending保留。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T121135Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 12:46–12:51 UTC监控

两组均遇到平台抢占并在同一backend自动重新排队。keep_lora中断前到800步（loss1.445），CP800在退出前已提交；12:48:49被更高SLA资源抢占，12:48:54重排。keep_sid前一实例日志到455步（loss1.570），12:46实例消失触发平台Restarting，随后12:48:39确认更高SLA抢占、12:48:43重排。两组最后状态均Queued，新的正式步数未知，不能用800/455声称当前训练正在推进。

独立核验keep_lora CP800（25,600次曝光）及keep_sid CP400（12,800次曝光）的身份、紧凑元数据SHA、精确样本及监督token计数；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸与提交marker一致。SID此前CP300→CP400恢复链通过：有效曝光恰好增加3,200，无遗失或重复计入，旧恢复验收已归档；本次新恢复仍需核验。LoRA CP800处于epoch1、epoch内125次更新/4,000条样本，后续必须专门核对非零epoch+非零skip恢复及新检查点。

本轮未取消、重提或更改GPU/SLA/训练条件；一次LoRA后续日志读取超时已保存观察失败证据，再次读取成功。大文件留远端。清单已将两组当前进度置为未知并设置新恢复核验待办。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T124605Z/monitoring_summary.json`、各组`metadata_verification.json`及SID的`post_resume_checkpoint_verification.json`。继续30分钟监控。

## 2026-09-23 13:25 UTC监控

两组已在原backend恢复训练。keep_lora当前retry_001为841/2025步、loss1.468；keep_sid当前retry_003为435/2025步、loss1.559，当前attempt均无异常且训练诊断有限。分别从CP800与CP400恢复：新进程收据记录373个Adam状态的step、global_step和scheduler_last_epoch均匹配各自断点，学习率符合原2025步cosine，优化状态未重置。远端hydrate完成标记对应完整文件字节校验；本地未下载大文件。

各current cursor_attempt/start.json均记录实际restored_global_step且status passed；冻结回调在写该收据前已核对真实恢复的每样本曝光与监督token计数。LoRA CP800对应完成epoch1、epoch内125次更新/4,000条样本；恢复后已推进41步。CP800（25,600次曝光）及CP400（12,800次曝光）的身份、紧凑元数据SHA、精确样本顺序与远端持久化文件尺寸再次通过。

本轮尚未提交恢复后的CP900/CP500；完整的新检查点曝光连续性验收留待下一次，requires_resume_verification与post_resume_checkpoint_verification_pending保留。没有重新提交作业或改变条件。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T132435Z/monitoring_summary.json`及各组`resume_verification.json`。继续30分钟监控。

## 2026-09-23 14:02 UTC监控

两组继续正常训练：keep_lora当前retry_001为991/2025步、loss1.484；keep_sid当前retry_003为561/2025步、loss1.524，当前日志数值有限且无新增异常。CP900（28,800次曝光）及CP500（16,000次曝光）的身份、元数据SHA、精确样本计数和远端模型/Adam/scheduler/四卡RNG文件存在性与尺寸核验通过。

本次独立验收两组恢复后的持久化连续性：LoRA CP800→CP900和SID CP400→CP500分别新增100次更新、3,200次有效曝光；逐一验证实际updates.jsonl记录的每批样本顺序、监督token分母、哈希链以及末条记录与检查点cursor绑定。LoRA从epoch1、epoch内125次更新处恢复后的801–900步全部匹配seed42的冻结epoch排列，非零epoch+非零skip恢复无重复跳过。两组此前模型/Adam/scheduler恢复收据哈希保持不变，requires_resume_verification及post_resume_checkpoint_verification_pending均已清除。

本轮没有重提或修改作业，大文件留远端。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T140105Z/monitoring_summary.json`及各组`resume_verification.json`。继续30分钟监控。

## 2026-09-23 14:32 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1100/2025步、loss1.471；keep_sid为675/2025步、loss1.512，已记录第一轮epoch尾批核验。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1000（32,000次曝光）和CP600（19,200次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T143120Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 15:02 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1208/2025步、loss1.453；keep_sid为792/2025步、loss1.496。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1200（38,400次曝光）和CP700（22,400次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T150120Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 15:32 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1316/2025步、loss1.438；keep_sid为903/2025步、loss1.484。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1300（41,600次曝光）和CP900（28,800次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T153120Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 16:02 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1426/2025步、loss1.467；keep_sid为1013/2025步、loss1.507。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1400（44,800次曝光）和CP1000（32,000次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T160121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 16:32 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1533/2025步、loss1.437；keep_sid为1128/2025步、loss1.488。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1500（48,000次曝光）和CP1100（35,200次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T163121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 17:02 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1640/2025步、loss1.377；keep_sid为1237/2025步、loss1.489。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1600（51,200次曝光）和CP1200（38,400次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T170121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 17:32 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1752/2025步、loss1.428；keep_sid为1350/2025步、loss1.479。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1700（54,400次曝光）和CP1300（41,600次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T173121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 18:02 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1861/2025步、loss1.408；keep_sid为1460/2025步、loss1.451。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1800（57,600次曝光）和CP1400（44,800次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T180121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 18:32 UTC监控

两组在已接受恢复链的同一attempt持续训练：keep_lora为1970/2025步、loss1.42；keep_sid为1572/2025步、loss1.487。当前stdout无新异常，训练诊断有限，生命周期无新增中断。

独立核验CP1900（60,800次曝光）和CP1500（48,000次曝光）：样本计数与冻结epoch排列、监督token、composite父adapter和元数据hash一致；模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。日志步数与已验收提交点分开记录，不把尚未完成提交的保存当作已接受checkpoint。大文件未下载；完整训练及终点评价仍待完成。本轮无作业修改，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T183121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 19:02 UTC监控

keep_lora已完成NI3训练并通过本地独立阶段验收：2,025次更新、64,800次曝光、21,600条样本每条恰好3次、108类每类600次。全部composite参数继承及新optimizer证据一致；373个终点张量和4条参考logits精确重载（误差0）。远端完整checkpoint字节读回收据与本地marker/files逐项匹配，实际373个Adam状态及scheduler均在2025步。最终adapter文件SHA256为15e23cf1d48f9c904be23204490fa04e647d97d1216402b412637291a0804a7d。

该组已进入Full Dev评估，四分片均开始发布预测块，本次观察到的连续文件区间端点为512/512/512/256；这些仅是远端文件范围，不是最终预测覆盖或指标验收。完整Dev、NI后功能诊断及完整继承参照探针仍待完成，不能把训练阶段通过当作整个条件完成。紧凑终点收据已保存到final-readback/mind-cir-param-transfer-108-0923-v2/qwen35-2b-keep-lora-ni3/。

keep_sid仍正常训练至1685/2025步，loss1.489且数值有限；CP1600已核验51,200次有效曝光、身份及元数据hash，远端模型/Adam/scheduler/RNG文件齐全且尺寸匹配。两组无新增异常，不重提或修改作业。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T190121Z/monitoring_summary.json`及keep_lora的`training_stage_verification.json`。继续30分钟监控，两组最终独立接受后再报告比较结果。

## 2026-09-23 19:32 UTC监控

keep_lora继续进行Full Dev评估，四分片持久化文件范围端点分别为4,352/4,608/4,352/4,608（各自目标7,848），均较上一轮增长，最近文件写入约19:27–19:30 UTC。主stdout在18:58之后无训练更新符合正在独立评估的阶段，不能据此判定停滞。这些文件范围仅用于进度观察，未验收预测内容或最终指标；本轮尚无evaluation_readback或complete收据。此前2025步/64,800曝光/373张量与logits精确重载的阶段收据及CP2025身份未变。

keep_sid正常训练至1798/2025步，loss1.436；CP1700元数据、身份和54,400次精确样本曝光核验通过，模型/Adam/scheduler/四卡RNG远端文件齐全且尺寸匹配。当前attempt无新异常，生命周期无新中断。本轮没有修改或重提作业，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T193121Z/monitoring_summary.json`及各组`metadata_verification.json`。

## 2026-09-23 20:02 UTC监控

keep_lora完整Dev阶段已独立验收：31,392 impressions、37,728目标、104支持类、108 SID Top20、无历史过滤，四分片各7,848。基于四份分片指标重算全部@1/@5/@10/@20逐类与宏平均值，均与合并指标及远端完整预测读回一致；和已接受完整继承模型的预测hash绑定、全部逐类/宏平均成对差值及37,728条目标的方向计数也核对一致。评价使用NI3最终adapter路径，远端worker校验其目录摘要；本地保留文件SHA与目录SHA各自的含义。

该组目前执行NI后诊断及完整继承参照探针，根complete.json尚未出现，整个条件仍未最终接受。正式指标与成对变化已保存在`evaluation_verification.json`和跟踪清单，待两组整体验收后统一比较。keep_sid当前1906/2025步、loss1.438，CP1900的60,800次精确曝光和持久化元数据核验通过；两组无新增异常。没有修改或重提作业，大权重及完整预测留远端，继续30分钟监控。

证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T200121Z/monitoring_summary.json`、keep_lora的`evaluation_verification.json`及各组`metadata_verification.json`。

## 2026-09-23 20:32 UTC监控

keep_lora已通过6项探针的紧凑分片独立验收：自身NI前/后、完整继承参照NI前各2项（SID名称108条/108类，历史重建512条/87类）。全部覆盖、分片及逐类/宏平均指标重算一致，自身前后及已固定参照的adapter目录身份/配置路径按协议核对。当前正在进行最后的完整CIR5→NI3参照NI后两项诊断；本次快照尚无对应readback及根complete.json，因此整个条件仍未最终接受。正式Full Dev和训练终点已接受的证据保持不变。

keep_sid日志到2018/2025步，loss1.494且数值有限；CP2000已核验64,000次精确曝光、身份和元数据hash。尚未从本次快照确认最终2025步保存/重载，不提前声称训练验收完成。两组无新异常，本轮无作业修改。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T203121Z/monitoring_summary.json`、keep_lora的`probe_verification.json`及各组`metadata_verification.json`。继续30分钟监控。

## 2026-09-23 21:02 UTC监控

keep_lora整个条件已接受：补齐全部8项诊断及根complete.json后，按指定accept_cir_parameter_transfer.py运行独立整体验收通过；完成收据中的训练、最终checkpoint、Full Dev、成对比较和全部诊断与各独立收据一致。原生stdout出现CIR_TRANSFER_ACCEPTED，生命周期20:45:20 UTC为Completed，与平台终态一致。接受收据：`artifacts/mind-cir-parameter-20260923/acceptance/keep_lora.json`，SHA256 c157038169d0fb85f1c490a08b563d0aec0bf29ef9c851a34b50e10068e92e89。该已完成分支不重训或重复验收，后续保留其来源及结果。

keep_sid已完成2025步、64,800曝光，每条样本3次、每类600次；composite继承、新optimizer、373张量及4条参考logits零误差重载均独立核验通过。远端完整checkpoint字节读回及373个Adam状态/scheduler均到2025步。最终adapter文件SHA256 cd57394dc7d335c9623179e18a43da9f85fd71680a62db9f2a7eee0cb1a16dad。该组Full Dev正在推进，本次预测块文件区间端点为4,608/4,352/4,352/4,096，尚未最终验收；后续仍需完整Dev和NI后诊断。

本轮无新故障、不重提作业，大权重和完整预测留远端。30分钟监控保持激活，活动分支为keep_sid，待两组全部接受后统一报告方法比较并停止监控。证据：`artifacts/mind-cir-parameter-20260923/progress/20260923T210122Z/monitoring_summary.json`、LoRA的`final_completion_verification.json`及SID的`training_stage_verification.json`。

## 最终整体验收

2026-09-24（北京时间）两组均已接受。SID组完整Dev、4项前后诊断及根complete.json通过指定独立reader，接受收据SHA256为7f42d6922515683fb2cd00f40ed962f2cb4eae2436e988b063f17e7027fbe946；原生终态2026-09-23 21:29:18 UTC Completed。LoRA组此前验收保持不变。两组预测来源、完整终点checkpoint、模型/Adam/scheduler、每样本曝光、精确reload、全部诊断和配对比较均核对一致。

完整继承、两种局部重置的全部@1/@5/@10/@20、NI前后诊断及成对差值已保存到 [最终报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-23/mind_cir_parameter_transfer_results.md) 和 `artifacts/mind-cir-parameter-20260923/final-report`。主指标完整继承均最高，但局部条件优劣随cutoff变化；不作显著性、严格2×2贡献比例或知识独占位置断言。最终汇总：`artifacts/mind-cir-parameter-20260923/final_acceptance.json`。自动监控cir已在全部验收和交付核对后通过应用工具删除。
