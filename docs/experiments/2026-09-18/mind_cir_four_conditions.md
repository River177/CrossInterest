# MIND CIR 时长与混合方式：四组实验

## 最终结果（2026-09-20验收）

用户指定四条件均已完成并独立接受。最后的两阶段恢复作业于北京时间2026-09-20 04:56结束；原CIR5复用，NI3从已提交900步恢复，NI5独立从同一CIR5参数和新optimizer/scheduler开始。自动监控已按用户此前要求删除。

同一Full Dev：31,392 impressions、37,728目标出现、104支持类、固定108类SID Top20。下表单位为%，全部@1/@5/@10/@20原始数值、相对NI-only/CIR3的百分点差与训练预算见CSV。

| 条件 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | 有效更新数 |
|---|---:|---:|---:|---:|---:|
| NI-only | 20.3275 | 33.6553 | 10.1325 | 13.4663 | 2,025 |
| CIR1 → NI3 | 21.2786 | 34.2609 | 10.8068 | 14.0725 | 2,565 |
| CIR3 → NI3 | 21.8272 | 35.1740 | 11.0708 | 14.4257 | 3,645 |
| CIR5 → NI3 | 21.9717 | 35.0756 | 11.2051 | 14.4966 | 4,725 |
| CIR5 → NI5 | 20.5892 | 33.8399 | 10.4749 | 13.7809 | 6,075 |
| CIR + NI 混合3轮 | 21.8237 | 34.9396 | 10.8494 | 14.1506 | 3,645 |
| CIR + NI 混合5轮 | 21.3238 | 35.1462 | 10.9930 | 14.4630 | 6,075 |

CIR5→NI3在本次七组结果中取得最高Recall@10、NDCG@10和NDCG@20；Recall@20最高仍为CIR3→NI3。CIR5→NI5相对CIR5→NI3的四项主要指标均下降，因此目前结果不支持把NI由3轮延长到5轮。5/5分阶段与混合5轮具有相同任务曝光和总更新预算，后者在四项主要指标上更高；两者同时存在任务顺序和优化器／scheduler重置差异，不能归结为单独机制因果。

这些是单seed观察，不作统计显著性声称。NI-only保留历史单卡转四卡及FLA执行差异。表内预算为已接受训练轨迹，不包含平台回收后回放未提交更新的额外开销；CIR5在两个新分支间共用，物理运行只训练一次。

- [全部指标与差值CSV](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-four-20260918/all_conditions_comparison.csv)
- [四条件总验收收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-four-20260918/final_acceptance.json)
- [跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_four_108_20260918_v1.json)

## 冻结协议与运行记录
本次仅执行用户指定四个条件，模型 Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，seed42。固定原 CIR 17,271 行、NI 21,600 行、108 类 SID、news/click 提示词。没有新增动态挖类别、类别均衡、标签打乱或多 seed。

| 条件 | 更新步数 | CIR 曝光 | NI 曝光 | 优化器／scheduler |
|---|---:|---:|---:|---|
| CIR5→NI3 | 2,700 + 2,025 | 86,355 | 64,800 | CIR→NI 重置 |
| CIR5→NI5 | 2,700 + 3,375 | 86,355 | 108,000 | CIR→NI 重置 |
| CIR+NI 混合3轮 | 3,645 | 51,813 | 64,800 | 单阶段连续 |
| CIR+NI 混合5轮 | 6,075 | 86,355 | 108,000 | 单阶段连续 |

混合数据是两份冻结数据的精确并集，每轮 38,871 行各出现一次，每 epoch 按 seed+epoch 做统一随机排列，不固定每个 batch 的任务比例、不重采样。训练 task 分别保留 cross_interest_reconstruction/new_interest；只在新混合文件中为原 NI 补齐 task 标记并无损对齐时间戳／整数 schema，逐条验证所有源字段。源文件未改写，held-out 输入与任务口径沿用原数据审计。

两阶段条件共用一个从原始初始化训练 5 epochs 的 CIR5 adapter，NI3 与 NI5 各自从该同一 adapter、独立新 optimizer/scheduler 开始，不从已训练 NI3 继续到 NI5。两条 NI 分支顺序执行于同一作业；混合3轮与混合5轮分别独立作业、从原始初始化开始，全部正式阶段按完整 horizon 建立 cosine scheduler。共三个作业、每作业 4×A100-80GB Recall Standard，最多 12 GPU 并发，共用 CIR5 节省 2,700 次重复更新。

沿用 microbatch2、accumulation4、global batch32，BF16、LoRA r16/alpha32/dropout0.05、lr2e-5、weight decay0.01、cosine/warmup0.03、非reentrant checkpointing、显式Torch fallback、native full-vocabulary completion-only token-mean SFT。数据未超长，监督 SID 和模板结束符共3个token。阶段、样本曝光和checkpoint身份严格绑定，非零epoch/skip恢复保持既有已验证逻辑。

复用已接受 CIR3 运行包的94个文件，仅更新实验合同并新增混合构造和编排。现有单卡／四卡／重载／真实评估门槛有哈希绑定收据。混合数据首次进入正式训练前，每个混合作业内自动执行41条包含两任务及尾批的四卡2步检查、精确重载和64条真实Dev，失败不会进入正式训练。两阶段条件复用原检查，不重复训练诊断样本。

本地相关测试31 passed、1 skipped（已有CPU Trainer用例因本地缺少Transformers跳过）；全量精确并集、真实5轮ledger、跨epoch checkpoint恢复、正式launch config／cursor及实际AMLT上传过滤已验证。结果待运行完成；仅在完整保存、373tensor/logits精确重载、实际样本曝光、31,392个Full Dev impressions／37,728目标出现／104支持类别、108类Top20合法排名与四分片指标独立重算通过后验收。逐条件结果写独立目录，避免不同adapter混用。

主要比较为四个新条件与已接受CIR1→NI3、CIR3→NI3及NI-only。CIR5→NI5与混合5轮具有相同任务曝光和6,075次总更新，但任务顺序和优化器／scheduler重置策略不同；其他条件预算并不完全相同。单seed比较不主张统计显著性，NI-only保留历史GPU／FLA执行差异说明。

权重、optimizer、checkpoints与完整预测留远端，本地仅紧凑指标、收据、哈希与恢复指针。按30分钟监控全部作业，正常运行保持安静，异常在授权范围内修复恢复，全部条件最终验收后停止监控。

[跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_four_108_20260918_v1.json)；[实验配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/experiments/mind_cir_four_conditions_0918_v1.json)；[AMLT YAML](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/amlt/mind_cir_four_conditions_full_0918_v1.yaml)；[混合数据元信息](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_schedule_108_v1/metadata.json)。

提交实验 `mind-cir-four-108-0918-v1`，三个作业均已被后端接收，初次检查处于排队。

| 作业 | 条件 | 后端 | 页面 |
|---|---|---|---|
| qwen35-2b-cir5-ni3-ni5 | cir5_ni3, cir5_ni5 | `salmon_brake_1n6nc9r2t6` | [AMLT](https://aka.ms/amlt?q=k2t7d) |
| qwen35-2b-cir-ni-mixed3 | mixed3 | `honest_tongue_7dfgsm4cgl` | [AMLT](https://aka.ms/amlt?q=k2t7b) |
| qwen35-2b-cir-ni-mixed5 | mixed5 | `calm_bread_5jd90pjvdr` | [AMLT](https://aka.ms/amlt?q=k2t7c) |

首次原生检查：mixed5 后端 Running，尚无已核验训练步数；staged 与 mixed3 后端 Queued，等待资源。30分钟监控已开启，自动任务ID为 `cir`。

2026-09-18 02:04 UTC 原生观察：CIR5 31/2,700（最近loss=3.373），混合5轮 78/6,075（最近loss=1.864）。两者环境、snapshot、完整token、原始373tensor初始化及5轮cursor均核验通过，尚未到首个checkpoint-100。混合5轮的41行／两任务／7个屏蔽尾槽四卡2步检查、373tensor/logits精确重载及64条Dev四分片指标重算已独立回读通过。混合3轮首次pod被平台删除，当前Queued、尚无训练输出；日志未给出删除原因，继续等待调度。证据目录 `artifacts/mind-cir-four-20260918/progress/20260918T0203Z/`。

2026-09-18 02:45 UTC 原生观察：三个作业均 Running。CIR5 已到 201/2,700（最近 loss=1.355），混合5轮 261/6,075（最近 loss=1.500），日志分别仅滞后约9秒／7秒，无训练异常。两者 checkpoint-200 均已提交：模型、Adam、scheduler、四rank RNG等13个持久化文件与提交清单大小一致，紧凑元数据哈希和精确样本游标通过本地回读；CIR5 累计6,400次CIR曝光，混合5轮累计CIR2,882／NI3,518次曝光。未下载任何权重或优化器状态。

混合3轮生命周期补充确认首次部署超时，平台自动重新排队，02:28 UTC 新pod开始执行；无需人工重启。已核验冻结包、运行环境、模型snapshot、完整混合数据token审计、373tensor原始初始化及新optimizer/scheduler。41条混合边界检查已记录2/2更新及7个屏蔽尾槽，当前在持久化／重载阶段；64条Dev与完整门槛收据尚未出现，因此本次不标记gate已接受，也不报告正式训练步数。观察与回读证据：`artifacts/mind-cir-four-20260918/progress/20260918T0237Z/monitoring_readback.json`。

2026-09-18 03:18 UTC 原生观察：CIR5 340/2,700（loss=1.404），混合3轮正式训练20/3,645（loss=5.669），混合5轮405/6,075（loss=1.454）。全部当前stdout新鲜、诊断值有限，生命周期无新增中断。CIR5 checkpoint-300、混合5轮 checkpoint-400 已提交并通过身份、元数据哈希、逐样本游标与远端13项文件存在性／大小核对；分别累计CIR9,600次，以及CIR5,666／NI7,134次曝光。混合3轮尚未到首个正式checkpoint。

混合3轮新增门槛本次完整回读接受：41行（CIR20+NI21）、四卡2步、7个屏蔽尾槽、373tensor及4条logits零误差重载；64条真实Dev、4分片各16条、75个目标出现／36支持类，所有@1/@5/@10/@20逐类与宏平均指标独立重算一致。正式训练已从原始373tensor初始化和新optimizer/scheduler开始，3轮／3,645步cursor已核验，未沿用门槛训练adapter。证据目录 `artifacts/mind-cir-four-20260918/progress/20260918T0317Z/`；继续30分钟监控。

2026-09-18 03:20 UTC：按用户要求删除自动监控 `cir`，训练作业继续运行。按03:18快照估计，CIR约14.2秒/步、混合约13.5–14秒/步；参考已接受CIR3的NI恢复段实际1,225次更新耗时19,303秒，NI约15.8秒/步。每个条件另计Full Dev约31分钟及保存／重载开销。NI5分支须等待NI3训练和评估结束，再从共同CIR5 adapter独立开始，因此四组全部完成约需33–37小时；这是资源不被回收条件下的估计，不是新的运行状态确认。

2026-09-19 01:56 UTC 按用户询问执行一次状态检查，自动监控仍关闭。混合3轮与混合5轮均为原生Completed，四分片Full Dev完成，并通过本地 `accept_cir_schedule_final.py` 独立验收。两者最终checkpoint逐样本曝光分别严格为3／5次，元数据哈希、游标及持久化文件大小核验通过。

| 已接受条件 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| CIR+NI混合3轮 | 21.8237% | 34.9396% | 10.8494% | 14.1506% |
| CIR+NI混合5轮 | 21.3238% | 35.1462% | 10.9930% | 14.4630% |

两阶段作业 `salmon_brake_1n6nc9r2t6` 已于2026-09-18 18:13 UTC（北京时间9月19日02:13）进入Failed。CIR5的2,700步／86,355次曝光、373tensor与logits精确重载已验收，最终adapter SHA256为 `1201a6e750cf65ca57940613df177a8caab14b594827f4496f84c7643f283df9`。NI3上一attempt日志到1,000步，但只提交到checkpoint-900（28,800次曝光）；平台回收后retry_002在读取该checkpoint的optimizer.pt时发生Errno 5 Input/output error，未恢复训练。NI5未开始。checkpoint-900紧凑元数据、精确游标及Blob文件存在性／大小核验通过，尚不能据此断定大文件远端读取故障已消失。当前两阶段作业不再推进，之前完成时间估计失效。本次未重启作业、未恢复自动监控。

本次原生日志、全部retry、紧凑回读及终态证据保存在 `artifacts/mind-cir-four-20260918/progress/20260919T0156Z/`。独立验收收据为 `artifacts/mind-cir-four-20260918/acceptance/mixed3.json`（SHA256 `1ddec8e73ea9d81a62e32613b3ccda3e69c444b505ab9348ca717e24f628f415`）和 `artifacts/mind-cir-four-20260918/acceptance/mixed5.json`（SHA256 `4b8fceaeb5301e69530831674b1df69b97091d672d90595d057d79ca35664f87`）。

2026-09-19 02:20 UTC，按用户“恢复一下”提交单个恢复作业。新实验 `mind-cir-staged-recovery-0919-v1`，job `recover-staged-qwen35-2b-cir5-ni3-ni5`，backend `teal_rail_56hhgsgcnp`（[AMLT](https://aka.ms/amlt?q=k20mh)），当前原生Queued。仍为Recall／4×A10080GB／Standard；混合两组保持已接受状态，自动监控保持关闭。

恢复包 `artifacts/mind-cir-four-20260918/bundle-staged-recovery-0919-v1` 保留全部103个原训练payload字节及原bundle manifest SHA256，仅增加独立恢复入口和存储重试覆盖层。新增恢复manifest SHA256为 `b7a9c9049e4242f46a95d832e853a3d24e99a96e191abfb65a2b7a25b1fc317f`。5次以内errno.EIO重试仅作用于检查点文件校验／原子字节复制，不吞掉哈希不符、权限错误或其他异常，不重试优化步骤。15项相关本地测试、冻结存储模块真实checksum入口故障注入、AMLT配置dump和实际上传文件过滤均通过；六个关键上传文件已通过Azure Blob独立字节回读。

启动时将原作业的CIR5完成阶段、NI3原始初始化证据以及已提交checkpoint-900先复制到节点scratch并按哈希核验，然后验证全部373个Adam状态步数900、scheduler步数900、有限状态、对应学习率和28,800次精确样本曝光，再写入新结果前缀并提交恢复收据。原失败前缀只读，未复制未提交的step1000。新输出前缀为 `projects/sft4safety/amlt-results/7210244599.48462-16575cc7-74a3`；预期从NI3 step900续训到2025，然后NI5从原CIR5 adapter以新optimizer/scheduler独立训练3375步。每次平台retry复用已提交恢复收据及新作业最新checkpoint，不重复播种或重训CIR5。

当前仅确认恢复作业已提交／排队；远端完整字节／Adam／scheduler恢复检查与实际step901以后推进尚待启动证明，不能把排队视作已恢复训练。独立GCR CPU检查节点本次SSH连接超时，因此这些完整检查安排在本恢复作业内、任何训练更新之前。检查与提交证据：`recovery_preflight.json`、`recovery_frozen_storage_regression.json`、`recovery_uploaded_source_readback.json` 和 `recovery_submission.log`，均在本组artifact_root下。后续本地reader支持每个job独立的experiment路径，原混合结果不会被重新归属到恢复实验。

2026-09-19 09:23 UTC（北京时间17:23）按用户询问执行一次检查：恢复作业Running，CIR5→NI3已完整结束并独立验收通过，NI5为532/3,375步（15.76%），最近loss=1.524，checkpoint-500已提交。当前stdout距观察时间约11秒，原生日志无新增retry／训练异常。

恢复收据确认02:25 UTC已完成原始41份seed文件的远端完整字节验证，NI3的373个Adam状态、optimizer/scheduler均为900步，样本游标28,800次曝光。CIR5未重复训练。NI3最终2,025步、每样本严格3次／64,800次曝光、373tensor/logits精确重载及31,392条Full Dev全部通过。CIR5→NI3的Interest-Macro R@10/R@20/NDCG@10/NDCG@20为21.9717%／35.0756%／11.2051%／14.4966%，独立验收SHA256 `eba2e01948cb9ab1f62a7b8259e1409e394bfcb101c8950df0a0e9e8eb6d8228`。

NI5的初始373tensor与NI3的初始参数逐项一致，均来自同一个CIR5 adapter；初始optimizer entries/global step/scheduler step均为0，确认没有从NI3结果接着训练。其checkpoint-500的模型／Adam／scheduler／四rank RNG等全部清单文件大小一致，元数据哈希与16,000次精确样本曝光通过回读。按最近200步平均13.83秒/步估计，NI5剩余训练约10.92小时，加完整Dev及保存／重载约需11.5–13小时；不含潜在重新排队。自动监控保持关闭。本次证据目录 `artifacts/mind-cir-four-20260918/progress/20260919T0923Z/`。
