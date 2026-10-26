# OI→NI 精确元训练验证

用户已明确取消 `meta_no_history` 消融实验。该作业已终止，AMLT显示killed，原生后端于2026-09-14 10:35:06 UTC确认Canceled；后续仅继续Meta与Mixed，并从等计算量Mixed的前置验收中移除消融组。已产生的记录仅作为取消前历史保留，不作为完成实验或方法效果结果。原数值预算规则保持不变，两条件构造器v2已通过92项CPU验证，原三条件版本留档。[范围修订记录](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/scope_amendment_no_ablation_20260914.json)。

本轮540步Mixed已于2026-09-14 11:34 UTC正常结束，并完成[完整独立验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/pilot_mixed_independent_readback_v1.json)。完整下载999文件（2,298,168,304 bytes）与云端路径、大小及31个明确目录标记对应，10个关键文件经ETag绑定逐字节核对；最终373个训练张量、optimizer/scheduler游标540、四rank保存/部署/重载摘要和GPU探针零误差通过。Full Dev覆盖31,392条记录、37,728个目标、27,082名用户、104个支持类别，123个分片的归属、108候选排名与宏指标独立重算通过。

| 本轮条件 | 更新数 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|---:|
| Mixed，seed42 | 540 | 16.0095% | 27.9722% | 8.2021% | 11.2120% |
| Meta，seed42 | 训练中，最近日志136/540 | — | — | — | — |

Mixed的有效540步训练耗时为17,609.547818614315秒（70,438.19127445726 GPU秒，约19.5662 GPU小时）。完整JSONL有583条记录：原1–151后追加重试109–540，43条重复重放的非计时字段逐项相同，之前原生日志窗口的缺口已补齐。全部583条loss、RNG、角色数量与学习率规则另经[完整日志核对](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/pilot_mixed_full_training_log_check_20260914.json)。包含已记录重放的训练耗时为18,961.356144282327秒，差额是原43步损失工作；初始化、保存、评估及未记录开销不包含在这组训练计时中。

[根验收与来源汇总](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/pilot_mixed_full_root_acceptance_20260914.json)保留完整精度指标、输入身份和来源。这里是本轮540步Mixed基线，不能据此判断Meta是否有效，也不与不同训练量或数值路径的历史Mixed直接归因比较。Meta与后续等耗时Mixed尚未完成；正式compute预算继续等待Meta完整接受。

当前仅保留Meta和Mixed两种条件；Mixed已接受，Meta继续其4×A100、540步训练。Mixed为单阶段OI/NI联合SFT，目标是0.5×OI CE＋0.5×NI CE；Meta额外评价OI虚拟更新后的NI损失。以下保留取消前的运行与验收历史，不代表继续执行消融。[取消终态凭据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/user_ablation_cancellation_confirmed_20260914.json)。

两条件版[离线构造器](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/build_four_rank_compute_v2_no_ablation.py)已完成92项CPU检查，覆盖两组独立接受、拒绝取消组重新进入、范围修订文件防篡改、预算舍入与上限、完整父文件保留，以及隔离reader和优化模式YAML前置校验。原预算、计时、父运行验收和隔离reader的实现经AST对照保持一致；177个已提交父文件整树hash不变。[验证收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/four_rank_compute_v2_no_ablation_validation_20260914.json)与[v2输入格式及命令](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/four_rank_compute_builder_v2_no_ablation_usage.md)已保存。当前没有正式compute输入、预算、真实运行包或YAML，需保留两组540步全部完成并验收后再生成。

本轮执行已批准的方法：单个 Qwen3.5-2B，批次类别匹配，内层只虚拟更新 LoRA，外层更新 LoRA 与 SID 行。目标是 `0.5×OI(θ)+0.25×NI(θ)+0.25×NI(θ′)`，部署与评估均使用外层 θ。四卡工程验证及全部三个27步校准候选均已下载独立通过，按原始全精度验证指标冻结α=0.01。三臂pilot每臂4×A100、540步、seed42；10:17–10:21 UTC观测中，Meta与Mixed原生日志分别到93/540、436/540。仅内层清空历史的消融组被更高SLA作业抢占后，原作业已自动从原初始化重放；新worker原生日志到4步，完整四rank元数据重放前缀独立验收到3步，旧18步属于中断前历史。完整pilot及方法比较结果尚未产出。

消融组于10:08:12 UTC因higher SLA资源被平台回收，随后自动重排队，10:10:54 UTC新worker启动。完整Azure枚举及本地37文件／5个明确目录标记对应，确认没有任何已保存checkpoint；首次计划保存点是54步。[无checkpoint重启独立收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_meta_no_history_retry001_fresh_restart_independent_review_20260914T1019.json)确认原model audit与已接受来源保留，新attempt按原f735/a821初始化、四rank空Adam和起始LR重建，JSONL保留原1–18后追加新1–3。前三步的非计时字段与旧同一步精确一致，并与新retry原生日志含计时字段一致；[另一次原生核对](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_meta_no_history_retry001_native_replay_20260914T1022.json)覆盖新1–4。两次attempt都标注resumed_from_step=0，必须按新attempt与追加段判断进度，不能把旧18步或总行数当成当前游标。

原18步已记录训练耗时为1,338.6890 wall秒／5,354.7560 GPU秒（1.48743 GPU小时），是额外损失工作的下界；不含初始化、未完成后续工作、关闭和排队开销，也不是未来重放时间预测。原记录及[完整远端清单](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_meta_no_history_retry001_remote_inventory_20260914T1021.json)均保留。保持原作业、4×A100、代码、采样和540步预算，后续完整轨迹按每步最后合法记录验收，额外重放成本单列。

Meta于2026-09-14 07:08:42 UTC启动，首批24个文件已下载并通过[初始化与前8步独立复核](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_meta_initialization_prefix_independent_review_20260914T0734.json)。原始初始化、source/run、冻结controls、环境、模型快照、373个FP32训练张量与唯一108×2048 SID均匹配；step 1–8的损失按左结合FP32公式逐步复算精确相等，inner/HVP/梯度范数有限且非零，四rank参数摘要与373份Adam步数一致，RNG/LR规则和原生stdout逐项对应。根任务另重算24个下载文件的大小与SHA，与独立收据一致。这是启动和初期更新的验收，尚未包含持久checkpoint、最终GPU fresh reload或完整Dev。

| 条件 | 后端任务 | GPU与固定更新数 | 10:17–10:21 UTC观测与验收范围 |
|---|---|---|---|
| Meta | [ashy_pen_2mvcbsxvq9](https://aka.ms/amlt?q=k1wyn) | 4×A100，540步 | Running；原生日志到93，初始化及前8步四rank审计独立通过 |
| Meta，仅内层清空历史 | [shy_drawer_9qr80vzsvq](https://aka.ms/amlt?q=k1wyo) | 4×A100，540步 | 已按用户要求取消；退出后续实验范围，原记录仅留档 |
| Mixed | [red_table_pv44vkc8ky](https://aka.ms/amlt?q=k1wyp) | 4×A100，540步 | Running retry_001；原生日志到436，四rank产物验收至114 |

Mixed在平台回收后从CP108实际恢复，[恢复独立收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_mixed_retry001_restore_independent_review_20260914T0707.json)确认四rank恢复及重放109–114。新增[原生日志重放核对](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_mixed_retry001_native_replay_continuity_20260914T0733.json)进一步核验115–151，共37步的loss、范数、gradient/parameter SHA、完整RNG、LR和全局角色数量均与中断前同一步精确相同；结合前六步，43步重放全部一致，07:33 UTC原生日志已至161，超过旧worker的151步高位；09:33 UTC新窗口314–360的数值、loss/RNG/LR及与上一窗口的重叠记录进一步核对通过。新增37步属于原生日志证据，完整四rank产物独立验收仍为114，不能混同。云端已[列出CP324的manifest/state](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/progress/pilot-mixed-checkpoint-list-20260914T0916.txt)，它的完整payload尚未下载验收；最近完整保存态独立验收仍是CP108。

此前回收的210个文件（453,336,915 bytes）已完整保存，并与云端逐路径、大小及目录语义对应，关键文件经ETag绑定逐字节核对。[CP108独立收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_mixed_checkpoint108_preemption_independent_review_20260914T0534.json)保留原source、初始化、controls、checkpoint、完整旧JSONL和四rank审计。回收前109–151的43步未持久保存，已记录损失工作为1,351.8083 wall秒／5,407.2333 GPU秒的下界，作为额外恢复成本单列，不计入有效540步训练或用于估计未来重放时间。07:07 UTC的193文件恢复快照是部分元数据下载，不能称为完整run。

三组均属于 `mind-oi-ni-meta-four-rank-pilot-v1-0913`，使用同一selection和lineage；Mixed内部α=0，另两组为0.01。最终包 `bundle-four-rank-pilot-v1` 保持不变，[真实运行包独立复核](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/four_rank_pilot_actual_bundle_review_20260914T020003Z.json)通过64项检查，[提交与后端配置回执](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/pilot_submission_receipt_v1.json)记录三项各4卡的请求及上传位置。完整进度证据与验收边界保存在[协调状态](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/coordination_state.json)；继续原提交，不重复校准、工程gate或pilot。

抢占前的消融组原worker已完成[初始化及前9步独立复核](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/pilot_meta_no_history_initialization_prefix_independent_review_20260914T0955.json)。25个下载文件由根再次逐路径核对大小与SHA。source/run、alpha=0.01、controls/lineage、原初始化、373个FP32训练张量、tied SID与环境/模型快照均匹配；JSONL及四rank审计共同覆盖1–9步，0954原生stdout交集覆盖1–8步且全部字段一致。第9步依据下载JSONL及四rank审计核对，不声称更早的stdout已经包含它。

该组9步的FP32 Meta loss精确复算通过，inner/HVP/梯度有限且非零；每步48次预览对应OI、NI、no_history各16份，四rank参数摘要、373份Adam步数及LR一致。对冻结17,037条OI数据和代码的检查确认只有inner使用[Empty history]，原外层历史与标签保留；这属于来源与数据绑定检查，没有另外采集GPU动态输入。两种Meta的首步direct OI/NI loss相同。完整checkpoint、最终GPU fresh reload、Dev与540步pilot仍未验收，早期计时不用来制定compute预算。专用规则保存在[核对说明](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/meta_no_history_startup_readback_notes_20260914T0938.md)。

| 项目 | 本轮设置 |
|---|---|
| 模型 | Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 初始化 | 每组从同一原始 pretrained 与 mean-norm SID 初始化开始；不继承已见验证标签的 Mixed adapter |
| 参数 | all-linear LoRA r16/alpha32/dropout0.05；外层保留原生 tied SID 的108行 |
| loss | 同一 SID 输出位置的108类 CE；不训练结束符，不添加其它监督 |
| 采样 | 每 episode 8类，每类2 OI＋2 NI；两侧用户互斥；同类仅用于匹配边际分布 |
| 数据 | fit OI 17,037；fit NI 19,440；validation NI 2,160（108×20），验证用户从两种训练角色全部排除 |
| seed | 42；dropout stream按 episode固定，实际F/G在各臂对齐 |
| 步长选择 | 内层α∈{1e-4,1e-3,1e-2}，每个候选从同初始化做27次外层更新；只用Train内validation的Macro R@10、NDCG@10选取，并冻结选择收据 |
| pilot | 每种条件540次外层更新；各8,640条OI/NI呈现；这是有界单seed探索，不是多seed正式结论 |
| 外层优化 | AdamW，LR2e-5，weight decay0.01，clip1.0，3% warm-up与cosine |
| GPU布局 | 校准每候选单A100-80GB；pilot每条件4×A100-80GB，全局16 OI＋16 NI不变，每microbatch1，精确重算HVP |
| 推理 | 最终有效SID head用FP32投影；全108类，Top20，同分按固定interest_index，不过滤历史类别 |
| 验收 | 参数恢复、checkpoint/optimizer/RNG身份、fresh reload exact、完整预测覆盖与独立宏指标重算 |

三种科学条件为：同采样 Mixed（α=0）、真实OI inner的meta、仅inner去掉OI历史的meta_no_history。第三组的外层OI与NI仍保留真实历史与原标签。空历史对照是信息消融，不将其称为自然OI样本。

另以记录的实际训练步耗时制定等计算量Mixed预算；其上限为同一冻结episode周期的5倍，预算必须在运行前单独绑定。它与540步Mixed分别报告，不能把额外呈现量隐去，实际耗时若未匹配也不能称已经等算力。额外控制不改变任何原数据文件。

四卡pilot尚未开始时，进一步固定预算换算规则：三臂全部独立验收后，取meta与Mixed各540个有效更新的`seconds`总和，计算`N_raw = 540 × T_meta / T_mixed`，取最近整数（恰好半步时取较小整数），再限制到540–2700步。两个时间都包含RNG规划和通信；重试中被覆盖的更新、初始化、保存及评估成本另外列出。追加Mixed对照从原初始化开始，沿冻结的540-episode序列循环，并使用其N步学习率日程。预算文件必须绑定完整原始日志、验收收据、来源及运行身份；这里尚未填入实际N。完成后另报告真实GPU训练时间比及是否触及上限，不将估算匹配当作已经等算力。

对应的[离线构造器](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/build_four_rank_compute.py)已完成86项CPU测试和独立实现审查，实际隔离验收器对已接受gate的完整CPU读回也与原收据一致。它保持父pilot包全部177个文件不变，生成另一个compute包及单项4×A100配置，并拒绝不完整父实验、文件变动和覆盖已有输出。当前没有正式compute输入、预算、运行包或作业。[验收汇总](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/compute_preparation_root_acceptance_20260914T0242.json)与[输入格式和运行命令](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/four_rank_compute_builder_usage.md)已保存。这里匹配的是包含逐步审计的训练耗时；未记录的恢复工作及未测量的其它开销保留为unknown。

实现与冻结输入：

- [协议](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/experiments/mind_oi_ni_meta_pilot_v1.json)
- [数据独立回读](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_oi_ni_meta_pilot_v1/builder_independent_readback.json)
- [精确元目标](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/training/oi_ni_meta_objective.py)
- [Qwen后端与有效SID投影](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/training/oi_ni_meta_backend.py)
- [训练与恢复执行器](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/experiments/oi_ni_meta.py)
- [协调状态](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/coordination_state.json)

已通过的CPU工程验证包括：元梯度与完整unroll／有限差分一致；SID跨Hessian保留；连续功能式更新不修改原参数；dropout重算一致；真实tiny Qwen3.5结构、PEFT tied SID保存和fresh重载一致。发现并局部修复了PEFT重复登记共享ParameterDict导致fast参数残留的问题，未修改第三方安装文件。真实2B CUDA显存、时间和完整训练的证据需由远程gate补齐。

所有候选都使用官方纯Torch gated-delta、eager full attention和相同FP32投影，以支持二阶反传。该数值路径与历史Mixed报告不同，因此历史结果只作背景，不能直接把差值当本轮元训练收益。α选择与任何中途判断不读正式Dev；正式Dev只在完成既定训练后对部署θ评价。

实卡 gate v1 在最长输入的一阶梯度重算检查中失败，完成外层更新数为0。CPU BF16真实tiny Qwen复现表明：相同构图模式重复梯度逐值一致、forward/RNG/参数identity不变，但 `create_graph=False` 与 `True` 的反向数值路径不同；193-token样本有58个LoRA tensor不同，最大绝对差约6.135e-4。修复为初次inner梯度和HVP重算都使用 `create_graph=True`，逐micro立即detach并释放图；外层直接F/G保持原路径。检查容差没有放宽，没有替换为FOMAML。相应回归先复现失败再通过，完整本地测试为648 passed。

旧失败包 `bundle-v1` 和 native 日志已保留。修复后实卡检查使用 `mind-oi-ni-meta-gate-v2-0913`、`bundle-v2`，现已下载29个完整产物并独立验收通过。该修复增加了真实meta分支的一次inner F重算，计入实测训练成本。

| 实卡与独立回读项目 | 验收结果 |
|---|---|
| 实际环境 | A100-SXM4-80GB；Torch 2.8.0+cu126／CUDA 12.6；Transformers 5.14.1 |
| 冻结源身份 | 150个源文件、15个source identity条目与本地冻结包逐项哈希一致 |
| 最长输入HVP | 独立重扫确认OI 1545 tokens、NI 1561 tokens；HVP有限非零；39.02秒；峰值分配38,880,131,072 bytes |
| 两个完整episode | 分别258.65／296.86秒；保存373个可训练张量；所有优化器游标均为2 |
| SID | 唯一canonical参数形状108×2048；108行梯度有限且非零 |
| 保存与重载 | 远程fresh reload最大分数差0，保存张量摘要经本地独立重建一致 |
| 验证预测 | 64个用户／64个目标，108类完整分数、Top20顺序、输入身份及宏指标独立重算通过 |

验收器另补齐最长输入、SID梯度与完整exact backend检查，71个focused tests通过。这里的HVP和模型重载前向发生在远程GPU；本地独立回读重建参数摘要、token长度、排名和指标，不将它表述为本地再次执行2B二阶前向。

- [完整独立验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/gate_v2_independent_readback.json)
- [源码与环境独立核验](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/gate_v2_provenance_readback.json)
- [校准启动收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/calibration_launch_receipt.json)

实测两步均值约277.75秒，仅训练部分外推，3×27步校准约6.25小时，一个540步meta约41.66小时；这是初始两步估计，尚不包含完整validation、Dev和保存成本。原启动器的5小时校准／6小时pilot时限不足，因此保留原始150个源文件与source manifest字节，新建`bundle-long-v1`并单独登记长时限launcher。该launcher相对原版只有两处常量变化：校准时限12小时、pilot与计算量对照72小时；所有科学步数、样本、模型、seed和精确二阶算法不变。新增脚本和launch manifest会复制到远程输出根以便回读。作业时限是单次Python进程的操作窗口，不能当作累计GPU成本上限；不设置无限重试。

已提交校准`mind-oi-ni-meta-calibrate-v1-0913 / qwen35-2b-exact-meta-calibrate`（backend `stoic_oven_mcnfd4v969`，[作业页面](https://aka.ms/amlt?q=k1sqc)）：三个α各27步，从相同原始初始化出发，只读Train内验证集选择。需下载验收三个α后才能冻结选择并进入三臂540步pilot；目前尚无方法有效性的比较结果。


根据用户提出的4×A100加速需求，正在加入四rank执行引擎。每个episode仍是全局16个OI与16个NI，按原位置分片；先聚合全局inner LoRA梯度形成统一虚拟参数，再聚合全局query向量与各分片HVP，外层保留SID交叉Hessian。每阶段按原global chunk顺序广播、累加，以保留原串行FP32加法顺序。rank0用真实no-grad前向预览原F/G/empty顺序，分卡回放每个chunk原CPU/CUDA随机状态；预览和通信成本计入实测时耗。

新增四进程CPU验证已覆盖不等权、空分片、有限差分以及真实BF16 Qwen＋FP32 LoRA/SID；ordered聚合的梯度、loss和范数与原串行路径逐位一致。四卡A100工程验证已完成下方独立验收；正式540步的速度和方法收益仍需pilot验证。原单卡校准继续运行，以显式来源及兼容性收据复用结果；四卡引擎采用独立执行身份，后续pilot不再自动提交旧单卡入口。


四卡代码已冻结为`bundle-four-rank-v1`，完整本地测试764 passed，分布式独立reader另有44项focused tests通过。新执行source identity为`eac329680c8cd85142787894b2ce8c3859b163bd57e6864f07e42e3493b683a7`，明确关联原校准source identity，不覆盖旧运行身份。已提交[4×A100工程验证](https://aka.ms/amlt?q=k1s6x)：`mind-oi-ni-meta-four-rank-gate-v1-0913 / qwen35-2b-exact-meta-four-rank-gate`，backend `jovial_boot_3pwy7l4vps`。该作业额外检查GPU多进程的串行梯度／Adam等价性、最长输入与三个空rank、全rank保存重载以及1024条内部验证的4分片覆盖。实际GPU正确性与加速比待实卡结果；校准作业继续运行。


四卡提交后的原生日志明确显示`ScheduleNotPossible`：当前账户／group policy的Standard NDAMv4配额可用3卡，需要4卡，缺1卡，GPU训练更新数仍为0。不是四卡程序运行失败。当前可访问订阅未找到其它可用的msrresrchbasicvc／msrresrchvc入口；保留四卡作业排队并继续已有单卡校准，待配额释放后自动推进，不取消其它工作。


后续原生状态确认四卡作业已于2026-09-13 12:46:10 UTC启动，配额等待自动解除；四个CUDA rank均已完成NCCL初始化，源码／模型快照／环境产物已写出。当前尚无四卡训练完成步或等价性验收结果。原校准在13:04 UTC快照中记录第一组alpha=1e-4完成18/27步，梯度与HVP有限。


四卡完整产物已下载并独立验收PASS：[完整收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/four_rank_gate_v1_independent_readback.json)、[源码与环境](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/four_rank_gate_v1_provenance_readback.json)、[独立计时整理](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/four_rank_gate_v1_timing_readback.json)。两步的373个梯度tensor和373个Adam更新tensor最大绝对差均为0；保存张量和全部优化器游标为2，四rank重载差均为0。重建最终参数SHA与旧单卡gate完全一致。1024用户／1024目标覆盖108类，四rank各256条，输入身份、完整108分数排序及宏指标均独立重算通过。

| 同一episode的诊断计时 | 第1步 | 第2步 |
|---|---:|---:|
| 同机单卡参考objective | 238.74秒 | 291.42秒 |
| 四卡objective＋原RNG预览 | 87.42秒 | 99.13秒 |
| 对应加速比 | 2.73× | 2.94× |
| 四卡再计入外层更新的诊断step | 89.68秒 | 100.30秒 |

两步汇总的匹配objective诊断加速为2.84×。若后续维持这两步的速度，540步仅训练部分外推约14.25小时；尚未包含初始化、正式验证和Dev评估，不能写成已完成540步的实测。工程gate本身还额外执行单卡参考及数值比较，其总占卡时间单独记录；正式等计算量Mixed必须使用pilot真实GPU-seconds，不能直接用本表定预算。

独立reader发现并严格补齐了冻结schema中的nodes=1字段，45项focused tests通过；只改本地验收器与测试，没有改已运行的四卡代码或下载产物。四卡gate已接受，后续不重复；仍需三组alpha校准全部独立通过后才能冻结选择并启动pilot。


首个校准候选α=1e-4已完成27步并[下载独立验收通过](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/readback/calibration_alpha0_independent_readback.json)：373个训练张量、全部optimizer step27、SID 108×2048和fresh reload零误差均通过。验证集2160用户／108类／每类20例，9个分片完整覆盖。

| 校准候选α | 状态 | Train内部验证Macro R@10 | Train内部验证Macro NDCG@10 |
|---|---|---:|---:|
| 1e-4 | 独立验收通过 | 10.4167% | 4.63174% |
| 1e-3 | GCR补跑完成，独立验收通过 | 10.1852% | 4.41324% |
| 1e-2 | 完整独立验收通过，按冻结规则入选 | 10.7870% | 4.79779% |

这些是27步、2160名Train内部验证用户的校准指标，不能替代正式Dev对照。三个候选均已接受后，按Macro R@10→Macro NDCG@10→较小α的既定规则选定α=0.01；selection使用原始published值，不按显示的百分数或独立重算结果重新舍入选择。


2026-09-13 15:22 UTC，原顺序校准被更高SLA作业抢占并重新排队，原生日志显示账户可用配额为0。α=1e-3仅在attempt `6e287f460f39`记录15步，没有持久checkpoint；这15步记录的训练时间合计4,284.80秒（约1.19022 GPU小时），作为丢失的额外恢复成本下界单列，不作为有效训练进度。初始化和未记录的in-flight计算未包含在该下界。α=1e-4的已接受产物不变。

暂停命令返回不能暂停后，仅取消原精确job，未删除输出。原生后端已于15:39:38 UTC终止为`Canceled`，AMLT为`killed`；[终态证据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/progress/calibration-native-014/receipt.json)及[完整文件列表](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/progress/calibration-native-012/result_listing.txt)确认无新增α1 checkpoint。

恢复包`bundle-calibration-recovery-v1`逐项保留150个旧core文件和原manifest字节，仅新增独立分派器及启动脚本，分别运行原α索引1和2。仍调用原`oi_ni_meta.run_training`，每组从原初始化完成27步、同seed42、只读Train验证；两组的原run identity不变，新的dispatcher另有provenance哈希。三组全部独立接受后，按原精度、原Macro R@10→NDCG@10→较小α规则组装选择收据。[恢复包收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/calibration_recovery_bundle_receipt.json)记录新旧身份及取消证据。


恢复代码完成15项新增测试，全量测试784 passed；[独立提交前复核](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/local/calibration_recovery_v1_independent_review.json)通过。已提交实验`mind-oi-ni-meta-calibrate-recovery-v1-0913`，两项各用1×A100，可独立并行调度，2026-09-13 19:55 UTC核验仍为Queued、尚未开始训练（19:36调度尝试因配额预约拒绝，两项于19:38:12再次自动重排队）；两项原生日志均显示Standard NDAMv4可用配额0、每项需1卡：[α=1e-3](https://aka.ms/amlt?q=k1ulq)、[α=1e-2](https://aka.ms/amlt?q=k1ulp)。后端分别为`eager_candle_nxj1811m52`和`strong_dream_470kq82mzq`；[提交收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-oi-ni-meta-20260913/calibration_recovery_submission_receipt.json)保留远程源码与结果位置。15分钟自动跟进已更新为仅监控这两项替代任务；全部α验收后继续既定每臂4×A100的540步pilot。


17:31和17:40 UTC调度器曾选中两项校准，但quota reservation均未获批准，原生状态短暂变为Paused；两项已于17:41:45 UTC自动Requeued，无需人工resume。当前start_time仍为空，只有调度日志，没有模型启动或训练更新。这是配额调度过程，不能计为训练尝试或新增计算进度；保留现有提交继续等待。对应证据保存在`progress/calibration-recovery-alpha{1,2}-native-008/`和`-009/`。

### 2026-09-13 20:57 UTC：GCR 校准迁移准备与取消证据

保留节点 GCRAZGDL1681 只有一张 A100 80GB，因此仅承担剩余27步校准；正式三臂各540步仍为4×A100。已复用原AMLT实际基础镜像，CPU准备核验14项依赖版本、完整8文件模型快照、原150个core文件与dispatcher，并通过独立容器/缓存读回；没有新增GPU模型gate。准备验收见 `artifacts/mind-oi-ni-meta-20260913/local/gcr_preparation_v2_independent_readback.json`。

α1恢复作业 `eager_candle_nxj1811m52` 的取消前回执为Queued/start=null。取消后20:52:34 UTC原生Canceled、AMLT killed，但后端同时出现20:50:53 UTC的start时间；完整生命周期日志只比取消前追加两条用户取消事件，没有worker日志。不能把该时间声称为已证实的训练启动，也不能把它改回null。20:56:42 UTC对精确结果前缀的Azure Blob完整枚举为空，AMLT结果列表也为空。由于原wrapper在任何依赖安装/训练前先写入四个来源文件，现有证据支持没有可恢复训练产物。原始回执、日志、完整枚举保存在 `gcr-preparation-v1/alpha1-cancellation-v2/`；将以额外严格的终态空产物证据路径迁移，明确保留时间异常。该101秒的后端起止时间差不是已验证GPU训练耗时。

α2恢复作业仍Queued，保持原提交。所有已接受结果、冻结训练源码、模型初始化、27步预算和Train内验证选择规则不变。

### 2026-09-13 21:04 UTC：GCR α1 已启动

针对后端开始时间的v2迁移分支已通过103项launcher测试、15项dispatcher回归和独立审查。v1仍要求原生start为空；v2严格绑定本次原始取消前后回执、完整日志和完整结果枚举，不改真实start字段。只修改host启动控制器，原150个训练core、dispatcher、模型和预算字节不变。

实际容器 `oi-ni-meta-calibration-alpha1-gcr-v1`，ID `cbc6b8366028f9f9f02c7a05009e8e85620e2c35dc8fc49bbb80569075008e58`，于21:02:08 UTC启动。操作清单SHA `52c2fc4d67e30d61145244b65ebd62f83959e3ebd3658a50b6e6c51f7418b257`，已下载启动证据至 `gcr-preparation-v1/launch-alpha1-gcr-v1/`。21:03:53 UTC实际model_audit显示373个可训练张量、初始摘要 `f7354063335efa3b40e02dd288cceff002839d34443df97e332b8ea51a3a0dce`，与α0一致。此时尚无完成更新记录，不计作校准结果通过。已更新每15分钟跟进为该GCR实际运行及仍在AMLT排队的α2；不得重复启动或移除已有锁。

启动产物已独立验收：8个原字节文件与下载tar一致，10项启动证据副本hash一致；实际image、container、只读code/cache及离线wrapper符合操作清单，原source/runID、seed42、27步与α=.001一致。prepared image核验14包，实际worker直接记录的8个核心包均匹配。收据 `gcr-preparation-v1/local/alpha1_startup_independent_review_20260913.json`，SHA `29a8dc0c4ac478a2db3b974c7c0b7777498a7a223bf3e25f4ad8c75290a5e24a`。这是启动验收，完整校准结果仍待验收。

21:08:12 UTC回读实际训练记录，当前attempt `e4bff37f0e0f` 已完成1/27步，loss=4.839810848236084，梯度与HVP范数均有限，首步耗时198.6609秒，峰值分配34,459,062,272字节。原始记录在 `progress/gcr-alpha1-20260913T210812.json`。该进度尚无CP27及完整验证，不能据此选择α或声称方法有效；训练继续运行。

### 2026-09-13 23:04 UTC：α=1e-3完整独立验收通过

GCR容器于22:53:40 UTC正常exit0。完整下载52个文件，共227,808,293字节，逐路径、大小、SHA与终态远端inventory一致。373个保存张量与全部optimizer游标均为27，SID为108×2048；fresh reload最大误差0，部署外层θ。2160名验证用户、108类、9组分片的输入、108候选排序及宏指标独立重算通过。原source/run identity、完整27步attempt、dispatcher、模型快照、依赖与启动来源链均通过独立复核。

CPU验收收据SHA为`073eed3439b0926a022b833ee7efc174e2d009d1d77456763be74dc71e90e7f6`；完整provenance收据SHA为`e1cbb1363df70a16a8cda3152dd7c4ded43482cc3d708fb9a33d7aa51e3b4c8c`。校准Macro R@10为0.10185185185185185，Macro NDCG@10为0.04413244116761833。27条记录的训练耗时合计6367.79970468278秒，仅记本次校准成本，不能用于正式pilot的等计算量预算。α=1e-4的原已接受收据保持不变；仍需α=1e-2完成后按原规则选择步长。


### 2026-09-14：恢复全词表 SFT 的方向修订

用户要求恢复全词表 SFT。已核对旧 Mixed 配置使用 `semantic_id_loss_scope=full_vocabulary`、`loss_reduction=token_mean`；旧数据编码屏蔽 prompt，监督完整 assistant completion（包含模板结束后缀），padding 为 -100。当前 108 类 SID-only、540 步探索不能视为旧完整 Mixed 的复现。修订方案还需对齐训练呈现量和数据覆盖，不能只改 softmax 分母。

已更新 oi-ni-2 自动跟进，禁止自动构建和提交后续 SID-only 等耗时 Mixed；旧预算构造器及接受产物留档。现有 Meta 尚未取消；本次没有实现或提交新的全词表训练，也不恢复已取消消融。新的 Meta 内外层损失均应复用原全词表 completion SFT，数值与二阶导数需重新核验。
