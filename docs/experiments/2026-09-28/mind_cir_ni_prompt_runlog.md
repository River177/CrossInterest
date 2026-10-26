# CIR 使用 NI 提示词：CIR5 → NI3

用户已批准启动且持续跟进本单条件。唯一科学改动是将原 17,271 条 CIR 样本的指令完整替换为冻结 NI 模板；历史、删除结果、目标 SID、类别分布、样本顺序及来源 task 标记不变。CIR 标签仍来自历史重建，不能记为新增真实未来点击标签。

## 冻结训练与评价

- Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；seed42。
- 从原预训练基座与原 LoRA/SID 初始化开始，初始可训练状态 digest `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`；不加载原 CIR5 adapter。
- CIR 5 epochs，17,271 样本，2,700 更新，86,355 次样本曝光。
- NI 3 epochs，21,600 样本，2,025 更新，64,800 次曝光；继承新 CIR 的 LoRA/SID，重置 optimizer/scheduler。
- 一个 Recall Standard `80G4-A100` 作业，global batch32、microbatch2、accumulation4，lr2e-5、原 cosine/warmup、native full-vocabulary completion CE、Torch fallback。
- 改动前 CIR5→NI3 作为已接受主对照，不重训其他条件，不新增 seeds、课程、元训练或预测前适应。
- 评价原 Full Dev：31,392 grouped impressions、37,728 个目标出现、104 支持类，108 SID、未过滤 Top20；四分片、原指标聚合与独立复算。

## 预检查与执行

新数据逐行验证只有 `prompt` 和同步的 `messages` 改变；NI、Dev、mapping、选择清单逐字节一致。冻结包使用原 CIR5 对照的训练与评价代码，不引入当前工作区的其他改动。训练前在 worker 执行全量 token 审计，单卡与四卡的 CIR/NI 小样本检查、精确重载和 64 条真实 Dev 评价；全部通过后从原始初始化自动进入正式训练。诊断权重不进入正式训练。

检查点每 100 步持久化；复用已验证的 EIO-only 有限重试与身份严格匹配的自动恢复。大权重、optimizer、完整预测留在远端，本地仅保存紧凑审计、指标和恢复指针。

当前状态及 backend 身份以 [跟踪清单](../../../manifests/mind_cir_ni_prompt_20260928_v1.json) 为准。

2026-09-28 已提交实验 `mind-cir-ni-prompt-0928-v1`，job `qwen35-2b-cir-ni-prompt5-ni3`，backend `tough_bone_676m3s6rrw`（[作业页面](https://aka.ms/amlt?q=k4l88)）。09:21 UTC 原生状态 Queued，尚无训练日志或检查点。已创建每30分钟的持续跟进 `cir-ni`，仅在异常、重要验收或完成时通知。

本地40项相关测试通过；全量测试在收集阶段因本机缺少 `peft` 中止（2 skipped），不报告全量通过。实际 AMLT 上传过滤审计覆盖111个文件，全部冻结哈希一致；相对原对照99个原有文件保持逐字节不变。新bundle manifest SHA256：`0009a783e1d4a82f5b65f4092a803309d92e7a4e47279abdeac2428388a1c1b6`。训练代码、采样、loss和评价均来自原接受包。

结果持久前缀：`projects/sft4safety/amlt-results/7209441648.22156-0716e9cd-30c5`；代码前缀：`projects/sft4safety/amlt-code/715e7a2877d623489942e795efdad8cda9825c02a3375a76d9f036dfa070fce4`。启动前检查与训练是否健康仍以新日志、检查点和验收收据为准。

## 解释范围

这是等数据、等更新预算的提示词干预。可用于检查任务表述是否影响 CIR→NI 的迁移效果；不能单独证明 CIR 收益独立于更多有效训练数据，也不能仅凭单 seed 结果宣称统计显著。尚无新训练结果。

## 2026-09-28 09:51 UTC：启动验收通过，正式 CIR 训练中

原生状态 Running，唯一原始 stdout 新鲜且无异常或平台中断。独立读回73个紧凑文件并核对哈希：环境与模型 snapshot 一致；全量 CIR/NI 均无超长样本，CIR 最长1532 tokens，completion固定3 tokens。单卡 CIR/NI 各8步、四卡 CIR/NI 各2步完成，四份adapter均373tensor及logits精确重载；64条真实Dev、75个目标、36类、四分片全部指标经本地逐类复算一致。

正式 CIR 从冻结原始digest `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e` 与空optimizer状态开始，未继承诊断权重。观察到56/2700步，最近step55 loss=1.658，已记录loss/grad_norm/lr均有限；尚未到首个100步持久checkpoint。NI正式阶段未开始，不据启动验收声称最终效果。

证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T095146Z/startup_acceptance.json`，验收脚本及原生日志、各紧凑收据同目录；大产物仍留远端。后续按既定30分钟节奏跟进，正常推进时不重复通知。

## 2026-09-28 11:42 UTC：常规检查

CIR 日志推进到557/2700步，最近loss=1.264；原生Running，stdout距查询约3.6秒，无traceback、重试分支或中断记录。checkpoint-500已提交，13项文件远端存在及大小与commit清单一致，4份紧凑元数据哈希、训练身份及16,000次精确样本游标复放通过；未下载大tensor文件。NI尚未开始。

本轮首次原生日志查询ReadTimeout，后续只读重试成功；这是观察链路问题，未修改或重启训练。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T114232Z/checkpoint_inventory_acceptance.json`。正常推进，保留自动跟进，不发送重复进度通知。

## 2026-09-28 12:17 UTC：常规检查

CIR推进到713/2700步，最近loss=1.262，所有已记录诊断有限；原生Running，未见异常、retry或中断。checkpoint-700的13项远端文件及大小一致，紧凑文件哈希和跨epoch后的22,391次精确样本游标复放通过。NI未开始。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T121655Z/checkpoint_inventory_acceptance.json`。未修改训练，继续既定跟进。

## 2026-09-28 12:47 UTC：常规检查

CIR推进到846/2700步，最近loss=1.265；原生Running，未见异常、retry或中断，诊断值均有限。checkpoint-800的13项远端文件大小、紧凑元数据哈希及25,591次精确样本游标复放通过。NI未开始；无训练变更，保持自动跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T124642Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 13:17 UTC：常规检查

CIR推进到981/2700步，最近loss=1.227；原生Running，未见异常、retry或中断，诊断值均有限。checkpoint-900的13项远端文件大小、紧凑元数据哈希及28,791次精确样本游标复放通过。NI未开始；无训练变更，保持自动跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T131639Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 13:47 UTC：常规检查

CIR推进到1118/2700步，最近loss=1.246；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-1100的13项远端文件大小、紧凑元数据哈希及35,182次精确样本游标复放通过，已跨过第二个epoch边界。活动日志新增相同Pod ID的pod.created记录，但未出现retry stdout或训练回退，不能据此单独认定重启。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T134646Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 14:17 UTC：常规检查

CIR推进到1252/2700步，最近loss=1.195；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-1200的13项远端文件大小、紧凑元数据哈希及38,382次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T141640Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 14:47 UTC：常规检查

CIR推进到1385/2700步，最近loss=1.227；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-1300的13项远端文件大小、紧凑元数据哈希及41,582次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T144630Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 15:17 UTC：常规检查

CIR推进到1520/2700步，最近loss=1.191；原生Running，唯一stdout持续推进且距查询约7秒，无traceback或中断信号，诊断值均有限。checkpoint-1500的13项远端文件大小、紧凑元数据哈希及47,982次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T151629Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 15:47 UTC：常规检查

CIR推进到1658/2700步，最近loss=1.170；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-1600的13项远端文件大小、紧凑元数据哈希及51,182次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T154639Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 16:17 UTC：常规检查

CIR推进到1792/2700步，最近loss=1.159；原生Running，唯一stdout持续推进且距查询约2秒，无traceback或中断信号，诊断值均有限。checkpoint-1700的13项远端文件大小、紧凑元数据哈希及54,373次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T161630Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 16:47 UTC：常规检查

CIR推进到1928/2700步，最近loss=1.156；原生Running，唯一stdout持续推进且距查询约2秒，无traceback或中断信号，诊断值均有限。checkpoint-1900的13项远端文件大小、紧凑元数据哈希及60,773次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T164632Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 17:17 UTC：常规检查

CIR推进到2065/2700步，最近loss=1.185；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-2000的13项远端文件大小、紧凑元数据哈希及63,973次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T171631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 17:47 UTC：常规检查

CIR推进到2200/2700步，最近loss=1.192；原生Running，无traceback或中断信号，诊断值均有限。日志明确记录checkpoint-2200已持久化，13项远端文件大小、紧凑元数据哈希及70,364次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T174631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 18:17 UTC：常规检查

CIR推进到2332/2700步，最近loss=1.169；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-2300的13项远端文件大小、紧凑元数据哈希及73,564次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T181636Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 18:47 UTC：常规检查

CIR推进到2465/2700步，最近loss=1.199；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。checkpoint-2400的13项远端文件大小、紧凑元数据哈希及76,764次精确样本游标复放通过。NI未开始；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T184629Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 19:17 UTC：常规检查

CIR推进到2600/2700步，最近loss=1.156；原生Running，无traceback或中断信号，诊断值均有限。日志明确记录checkpoint-2600已持久化，13项远端文件大小、紧凑元数据哈希及83,164次精确样本游标复放通过。NI未开始；CIR完成与参数交接仍待正式收据，未提前视作验收。未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T191632Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 19:47 UTC：CIR完成，NI参数交接验收通过

CIR全部2700步、5 epochs完成，每个样本严格曝光5次，共86,355次；checkpoint-2700的13项远端文件大小与commit一致，全部样本计数及完整游标复放通过。373tensor与4条参考logits在新进程精确重载。CIR adapter文件SHA256为`c48079b49858af093fdfe181cb3a1414f7042cad50ac31db7a3f774d7cea9d25`。

NI初始373tensor与该CIR文件对应的输入参数manifest一致，状态digest为`63065022107be7981f512360f683d4b595b93267e1da255621ce66da0493e3a2`。optimizer为空，scheduler/global step均从0开始，原NI数据、三轮预算与数值配置保持不变。日志已到20/2025步，最新loss=1.730，全部已记录诊断有限，无traceback或中断信号；尚未到NI首个100步checkpoint。原生仍Running。

独立读回：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T194631Z/handoff_acceptance.json`。后续监控应使用NI阶段及`formal/checkpoints-ni`，清单已记录CIR父模型hash。该目录的`verify_checkpoint_snapshot_ni.py`可复制到后续观察目录，在出现NI检查点后调用既有严格校验器；不要再用CIR运行中断言核对已完成阶段。

## 2026-09-28 20:17 UTC：NI常规检查

NI推进到156/2025步，最近loss=1.513；原生Running，唯一stdout持续推进且距查询约4秒，无traceback或中断信号，诊断值均有限。NI checkpoint-100的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及3,200次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T201630Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 20:47 UTC：NI常规检查

NI推进到293/2025步，最近loss=1.517；原生Running，唯一stdout持续推进且距查询约3秒，无traceback或中断信号，诊断值均有限。NI checkpoint-200的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及6,400次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T204634Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 21:17 UTC：NI常规检查

NI推进到433/2025步，最近loss=1.510；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。NI checkpoint-400的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及12,800次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T211630Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 21:47 UTC：NI常规检查

NI推进到571/2025步，最近loss=1.515；原生Running，唯一stdout持续推进且距查询约7秒，无traceback或中断信号，诊断值均有限。NI checkpoint-500的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及16,000次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T214631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 22:17 UTC：NI常规检查

NI推进到710/2025步，最近loss=1.506；原生Running，唯一stdout持续推进且距查询约4秒，无traceback或中断信号，诊断值均有限。NI checkpoint-700的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及22,400次精确样本游标复放通过，已跨过第一轮边界。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T221631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 22:47 UTC：NI常规检查

NI推进到847/2025步，最近loss=1.434；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。NI checkpoint-800的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及25,600次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T224630Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 23:17 UTC：NI常规检查

NI推进到985/2025步，最近loss=1.477；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。NI checkpoint-900的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及28,800次精确样本游标复放通过。读回过程中出现一次AzureCliCredential调用失败警告，随后观察命令自行完成且退出0，89个紧凑产物完整取得并通过校验；没有重启或修改训练。CIR已完成的验收保持不变，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T231631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-28 23:47 UTC：NI常规检查

NI推进到1117/2025步，最近loss=1.462；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。NI checkpoint-1100的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及35,200次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260928T234633Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 00:17 UTC：NI常规检查

NI推进到1253/2025步，最近loss=1.422；原生Running，唯一stdout持续推进且距查询约4秒，无traceback或中断信号，诊断值均有限。NI checkpoint-1200的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及38,400次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T001631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 00:47 UTC：NI常规检查

NI推进到1390/2025步，最近loss=1.430；原生Running，唯一stdout持续推进且距查询约4秒，无traceback或中断信号，诊断值均有限。NI checkpoint-1300的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及41,600次精确样本游标复放通过。日志已进入第三轮，未提前声明训练完成；CIR验收保持不变，未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T004634Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 01:17 UTC：NI常规检查

NI推进到1526/2025步，最近loss=1.423；原生Running，唯一stdout持续推进，无traceback或中断信号，诊断值均有限。NI checkpoint-1500的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及48,000次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T011631Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 01:44 UTC：用户状态查询及紧邻heartbeat复核

NI推进到1651/2025步，最近loss=1.436；原生Running，stdout距查询约1秒，诊断有限且无异常。NI checkpoint-1600的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及51,200次精确样本游标复放通过。01:46 heartbeat复用不足三分钟前的完整观察，并重新核对89份紧凑文件哈希，未重复远端查询或改动训练。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T014355Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 02:17 UTC：NI常规检查

NI推进到1802/2025步，最近loss=1.407；原生Running，唯一stdout持续推进且距查询约5秒，无traceback或中断信号，诊断值均有限。日志明确记录NI checkpoint-1800持久化完成，13项远端文件大小、紧凑元数据哈希、CIR父模型身份及57,600次精确样本游标复放通过。CIR已完成的验收保持不变；未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T021634Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 02:47 UTC：NI常规检查

NI推进到1941/2025步，最近loss=1.395；原生Running，唯一stdout持续推进且距查询约3秒，无traceback或中断信号，诊断值均有限。NI checkpoint-1900的13项远端文件大小、紧凑元数据哈希、CIR父模型身份及60,800次精确样本游标复放通过。NI最终保存、精确重载和Full Dev评价仍未完成，未提前验收。未改动训练，继续跟进。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T024635Z/checkpoint_inventory_acceptance.json`。

## 2026-09-29 03:17 UTC：全部训练完成，Full Dev评估中

CIR5的2700步与NI3的2025步均完成并通过训练产物独立验收。CIR每样本严格5次、共86,355次，NI每样本严格3次、共64,800次；两阶段完整checkpoint的13项远端文件大小、紧凑元数据哈希及全部样本游标复放一致。每阶段373tensor与4条参考logits均精确重载；参数继承和新optimizer/scheduler初始化再次核对通过。NI最终adapter文件SHA256为`deb27796443078d7a17b00bdf7a4b59b4af236294452f699c0db81b769b32703`。

Full Dev已自动启动。此次观察的持久分片文件覆盖四路各4352条、共17,408/31,392条；这是远端文件区间的进度计数，尚非最终预测及指标验收。未出现异常，保留现有自动跟进，完成后整理最终紧凑读回并与原提示词CIR5→NI3比较。证据：`artifacts/mind-cir-ni-prompt-20260928/progress/20260929T031645Z/training_acceptance.json`。后续不要再将2025步训练日志当作仍在训练，阶段为Full Dev评价。
