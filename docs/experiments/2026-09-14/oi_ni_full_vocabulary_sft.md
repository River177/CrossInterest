# OI→NI 全词表 SFT：Mixed / Meta

用户已取消原 SID-only 试验，并授权仅重跑 Mixed、Meta。资源优先级更新为 Recall 集群 Premium A100；本轮每组使用 4×A100 80GB。旧 Meta 原生状态已于 2026-09-14 12:19:06 UTC 确认为 Canceled；旧 Mixed 已完成，消融已取消，所有旧产物保留。旧自动跟进暂停，旧等耗时 SID-only 对照撤销。

当前观察（2026-09-15 10:30:29 UTC）：Mixed已最终独立验收，Macro Recall@10=20.6385%，@20=34.3487%，不重复下载或验收。Meta900/3630（第1个epoch）仍Running，日志报告checkpoint900；窗口内目标公式、seed和数值检查通过。Meta尚无最终指标，其完整产物最终验收将在远端执行，本地仅保留紧凑收据与恢复指针。

## 训练协议

- Backbone：Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；seed42；两组均重新初始化，不接续 SID-only checkpoint。
- 数据：原完整 Mixed 的 38,698 行，OI17,098 / NI21,600。相同输入、标签和消息模板，三轮完整遍历，共3,630更新。每条样本每轮一次，无放回；OI呈现51,294次，NI呈现64,800次。
- 每全局批次最多32行，再分为OI/NI；每轮使用固定seed+epoch随机排列，仅以同任务交换消除批次内support/query用户重叠。两组用同一批次表。**不再使用旧pilot的每类2OI+2NI严格类别匹配抽样**，也不声称复现旧Mixed的逐批shuffle/RNG轨迹。
- 原completion-only编码：历史/prompt及padding为-100；每条样本监督SID、`<|im_end|>`、换行，共3tokens。全部38,698条最大长度1,564，无截断；词表分母为完整扩展词表。
- LoRA r16/alpha32/dropout0.05；inner只更新LoRA A/B，outer更新LoRA及tied SID行。两组相同初始化实现、模型精度与可二阶求导后端。
- AdamW fused，LR2e-5、weight_decay0.01、clip1；原HF cosine，109步warmup，首个更新LR为0。真实global batch32以每卡microbatch1累积，四卡同步全局梯度。
- 训练直接调用原模型的 native causal-LM forward 和 loss，保持完整输出层及原始张量形状；投影 BF16、CE 在 FP32 计算，历史标签仍为 -100。推理保持108个SID的FP32分数排序，最终部署outer参数；无测试时适应或历史过滤。

## 目标

令F、G为批次内OI、NI的全词表监督token平均损失，p为OI监督token占全批监督token比例。

Mixed：`p F(θ) + (1-p) G(θ)`，等于全批completion token平均SFT。

Meta：`θ′ = θ - α P_LoRA ∇F(θ)`；`J = p F(θ) + (1-p)/2 [G(θ) + G(θ′)]`。

α固定0.01；这是新实验预先确定的参数，不把旧108类损失下的α选择记录称为全词表校准。α=0严格退化成Mixed。为复用已验证的精确梯度引擎，传入2pF、2(1-p)G和内部步长α/(2p)，数学上对应上述目标，保留LoRA/SID交叉Hessian项。全局u、v、HVP同步，未使用一阶近似。

Meta对同一批NI的额外虚拟参数前向是方法计算开销，不算新增数据epoch；本轮不增加等耗时对照或消融组。

## 已完成检查

- 原Meta取消由AMLT killed及原生Canceled双重确认。
- 本地完整token审计：所有监督后缀均为3tokens；全量编码摘要 `87d9feec4394804c3e81e89dc68e94326dfed08994190518484367236a817988`。
- 全词表投影与显式完整head的loss、梯度、Hessian对比；不等token数下的直接unroll对比；tiny Qwen完整原始causal-LM loss/梯度对比；α=0 pooled SFT对比通过。
- 相关68项本地测试通过；实际冻结bundle内完整66项测试通过。
- 3,630步学习率逐步匹配HF cosine；篡改已消费样本顺序会被恢复检查拒绝。
- AMLT dump确认恰好两组、Recall、80G4-A100、Premium。上传清单172文件、66,371,437 bytes与manifest一致。
- 各训练job内执行新的真实GPU全词表loss/梯度对比；Meta检查最长输入精确HVP；前2次正式更新后保存并四rank fresh reload，再执行64条真实Dev结构检查，然后继续同一训练。检查失败即停止，不另起实验条件。

## 运行与验收

当前修复版实验：`mind-oi-ni-full-sft-native-recall-premium-v2-0914`。旧 v1 已在第 0 步验证失败后终止。

作业：`qwen35-2b-mixed-full-sft`、`qwen35-2b-meta-full-sft`。

配置：`configs/amlt/mind_oi_ni_full_sft_recall_premium_0914_v2_native.yaml`；协议：`configs/experiments/mind_oi_ni_full_sft_v1.json`。

当前运行包：`artifacts/mind-oi-ni-full-sft-20260914/bundle-v2-native`，manifest SHA `b526b97810ed56cb674f91d6a900090836dea60af8baa0a6ac81c4048352b74b`。旧完整Mixed仅作为历史强参考；新两组相互比较共享数值后端，不能声称与旧快速kernel逐位等价。

平台重试使用独立attempt目录；只从同身份、完整manifest的checkpoint恢复，并核对原始样本序列前缀、optimizer/scheduler、参数及GPU重载。完成后独立下载并以 `new_interest.experiments.read_oi_ni_full_sft` 核验每条样本三次覆盖、373参数及Adam状态、四rank部署/重载证据、31,392条Dev/37,728目标/104支持类、108候选排序及独立宏指标重算。调度Running/pass均不是最终验收。

### 提交回执

两组已于2026-09-14 12:27 UTC提交到Recall Premium，各4×A10080GB。Mixed backend为magenta_collar_wwq9y847ws，Meta backend为serene_pear_9f39d1hz4f。提交后调度为Queued；Mixed原生日志12:29:04 UTC出现ScheduleSelected，尚未取得GPU全词表启动验证产物。自动跟进oi-ni-2已切换到本轮两组。

### Native SFT 修复（v2）

v1 两组均在正式更新前的 GPU 梯度等价检查失败：loss 绝对误差 4.13656e-5，梯度相对 L2 误差 0.013949，高于原门槛 0.01。下载的两个 training.jsonl 均为空，无 checkpoint；初始化 tensor SHA 均为 a821bb97d4f5aa3417dc3e5dbc06c94cdc4514c343d96ade384a84a5258d1a5a。两组终态 failed，原产物保留。

CPU BF16 回归复现了旧路径的梯度差异。隔离头部后，selected-positions/full-head 与完整 head 的隐藏梯度一致，selected-positions/split-head 出现 0.002404 的相对隐藏梯度差异，支持拆分 BF16 投影改变数值路径。恢复 native causal-LM forward 和 loss 后，BF16 loss、梯度、Hessian 逐位一致；69 项相关测试及冻结 v2 包的 67 项完整测试通过。真实 2B GPU 验证仍由新 job 的原门槛判定。

修复只改训练 forward 和回归测试。数据、模型、采样、训练量、alpha、优化器、资源配置和等价门槛不变。新运行包为 bundle-v2-native，manifest SHA b526b97810ed56cb674f91d6a900090836dea60af8baa0a6ac81c4048352b74b。仍只提交 Mixed、Meta 两组至 Recall Premium 4×A100。

### GPU 启动记录与回读边界

13:08 UTC 原生日志：Mixed 25/3630，Meta 5/3630，均完成四 rank 的 step2 fresh reload 及64条真实排名检查。已下载 full_loss_parity.json：两组完整248185词表的原模型 loss 均为1.8666127920150757，loss误差与373训练张量的梯度相对误差均为0，原1%梯度门槛未修改。Mixed固定训练probe从1.8666127920150757下降至20步后的1.1918104887008667。Meta最长OI1548/NI1564的精确HVP检查通过，未进行参数更新，峰值47,531,981,312 bytes。

大检查点下载发生本地传输停滞，原AzCopy部分文件保留，仅停止两个本地下载子进程，未取消远端训练。备用下载使用一次缓存storage token、ETag条件和SHA验证；默认Azure CLI 10秒超时发生失败后，按已有读取器采用30秒冷启动超时。13:53 UTC Mixed原生状态仍为Running，但日志读取遭遇ReadTimeout，因此13:08的步数仅为历史已观察值，不作为当前进度。独立checkpoint参数回读尚未完成，startup_gate_accepted保持false。状态文件记录本地下载会话和验证脚本，后续完成回读再提升验收状态。

14:15–14:16 UTC 恢复原生日志读取：Mixed260/3630、Meta54/3630，均Running，窗口内loss有限且目标加权公式一致，无新错误。Mixed检查点2的68,212,000-byte adapter已下载并与manifest SHA一致（3bc5a8ea8200f4489f5de06bcae23b8ca2fe0e7bdf445a3a51cdd33da78a03df）；其余大文件仍由原备用下载进程继续，未重复发起下载。完整启动checkpoint验收仍待完成。

### 启动独立验收完成

两组 checkpoint2 的完整文件已下载并通过独立 CPU 回读：完整 manifest、373个FP32参数、全部Adam状态与scheduler游标2、SID108×2048、实际样本前缀、四rank重新加载标记、原始全词表loss及梯度误差0均通过。64条真实评估包含75个目标、36个支持类，输入身份、108候选排名及宏指标独立重算通过；这些小样本数值仅用于验证评估程序，不作为方法结论。Mixed20步学习检查和Meta最长输入HVP证据也已接受。

独立收据：`artifacts/mind-oi-ni-full-sft-20260914/readback/v2_startup_independent_readback_20260914T1308.json`，SHA e3ade25a1f6466b19be20a168d303629e3eac970aa8d95e4492c3e92933ed8f0。下载收据SHA fd8210e4af8da246d05d1e45e6d3fd3cf4742d4cd7a6ff764a4ffc93284136dc。启动验收已完成，不重复下载或验证checkpoint2。正式3630步训练及Full Dev最终验收仍未完成，进度以coordination_state和最新原生日志为准。

### Mixed 云盘日志故障与恢复准备

15:17 UTC 系统日志确认 Mixed 的 training.jsonl 云盘 flush 发生 ConnectionFailure：重试约557秒后返回I/O错误。15:26 UTC原生日志仍停在15:04的step424；独立读取持久化training.jsonl只到step425（15:09写入）。没有模型数值错误证据。Meta持久化记录已到step95（15:17写入），继续原v2作业。

Mixed 已单独取消，AMLT核对为killed，Meta仍Running。日志存储修复将每步记录改写节点本地，仅在检查点阶段同步云盘，避免其它rank进入NCCL后等待逐步云盘flush。v3只恢复Mixed，原实验仍为Mixed/Meta两种条件。数据、采样、训练目标、精度、随机种子、优化器及3630步总预算不变。

旧checkpoint400的manifest、状态、400步实际样本前缀和原source/config已独立获取并核对，manifest SHA0673daaf4c062b619903223c533ae4e55ee19cb96a518c353e4f86328c0b56c1，tensor SHA a97b8b727d85f933e040f0409ea69cdd254900584f9df15c54997942246b03f1。大参数payload及GPU恢复仍须在新worker更新前验证，不能把元数据核验称为完整恢复。400之后的旧工作保留为重放成本，不能从终态逻辑样本呈现量之外删除历史计算成本。

修复包bundle-v3-storage-mixed manifest SHA a27e9b60ec842c1acd1a6013c719fbefa2351c329287ca0ce918492e4dff3b9a，冻结包77项测试通过，包含日志存储失败隔离与错误parent/config/source/参数摘要拒绝。恢复配置为configs/amlt/mind_oi_ni_full_sft_mixed_storage_recovery_0914_v3.yaml，实验mind-oi-ni-mixed-storage-recovery-v3-0914，仍使用Recall Premium4×A100。v2已提交代码和全部产物保持不变。

### Mixed 恢复验证完成

已独立下载并核对8份紧凑恢复证据：新旧source/config/identity、父checkpoint400摘要、worker逐文件payload校验记录、四rank373参数及全部Adam游标400、GPU probe重载误差0、全词表loss/梯度误差0。401–425共25步重放与旧运行所有非计时字段逐项完全相同，最新已推进426。旧重放训练成本下界421.928704738064秒，GPU成本下界1687.7148189522559秒单独保留。

收据：`artifacts/mind-oi-ni-full-sft-20260914/readback/mixed_v3_recovery_independent_20260914T1607.json`，SHA d9e6d66f9d98665bf68b48e28f4f183bddaed8a4c43f8d4655b9d9c14de6f395。本次独立回读范围是紧凑恢复证据和精确重放；父400步大payload在worker完整hash验证并加载，未另行下载到Mac。后续checkpoint500的周期日志快照、全部训练及最终Dev仍待自然运行及最终验收，不增加诊断训练。

16:46 UTC 已独立核对CP500 manifest/state和日志快照：343,386-byte根training.jsonl与CP500 training_prefix.jsonl的SHA均为b4a9a59aa9586cdf630d016827c8bc4f1047f5479f395677a14542e4c14839bc，500步连续且前400步与旧记录一致；worker已继续到557。观察收据为artifacts/mind-oi-ni-full-sft-20260914/progress/mixed-v3-journal500-20260914T164600/receipt.json。此项存储流程验证已完成，不再重复检查；CP500大参数及完整训练最终验收范围不变。

04:28 UTC 检查点上传重试恢复：原 Mixed v3 作业已报告身份一致的 checkpoint2800 committed，随后连续推进至2830，窗口内全词表目标、随机种子及数值检查通过。没有重启作业或改训练代码。此项确认依据是原worker提交标记和后续训练推进，未单独下载CP2800大张量；不替代最终完整验收。Meta原生日志到651，两组日志当前均有数分钟延迟。相关原生日志和已解决异常记录见coordination_state。

## Mixed 最终独立验收（2026-09-15）

completed_attempt 指向的完整产物已下载：626文件（含指针与根manifest），7,510,923,402 bytes；另下载父checkpoint400完整11文件、225,292,530 bytes。下载使用ETag绑定和SHA-256核对。远端清单中未列入manifest的条目均为目录标记，未作为数据文件混入回读目录。

独立reader通过：3630步/3epochs、每条原始行每轮一次、OI51294/NI64800逻辑呈现、冻结批次及学习率；373个FP32参数及全部Adam游标3630、四rank RNG和部署/fresh reload摘要一致，GPU重载误差0。Full Dev独立核对31392记录、37728目标、27082用户、104支持兴趣、108候选、123分片，并重新计算宏指标。

| 指标 | 本轮 Mixed 全词表 SFT | 旧完整 Mixed（历史参考） |
| --- | ---: | ---: |
| Macro Recall@1 | 2.9106% | 3.1261% |
| Macro Recall@5 | 12.8560% | 12.8901% |
| Macro Recall@10 | 20.6385% | 20.6095% |
| Macro Recall@20 | 34.3487% | 34.2014% |
| Macro NDCG@10 | 10.3364% | 10.4247% |
| Macro NDCG@20 | 13.7647% | 13.8159% |

本轮Mixed恢复至旧完整Mixed的相近水平；不是先前16.0095%的540步SID-only短预算结果。新旧运行的kernel和批次随机轨迹不同，不能将小幅指标差异解释为方法增益。Meta尚未完成，不能据此判断Meta优于Mixed。

补充来源验收通过：父400完整payload、373参数/全部Adam400、tensor摘要、原400步前缀与最终日志逐字节一致；3630步seed/任务token数检查通过，33个嵌套manifest与根manifest绑定，179个提交source文件逐字节核对。旧401–425重放训练成本下界421.928704738064秒、GPU成本下界1687.7148189522559秒继续保留。

收据：`artifacts/mind-oi-ni-full-sft-20260914/readback/mixed_final_independent_20260915.json`；补充：`artifacts/mind-oi-ni-full-sft-20260914/readback/mixed_final_lineage_independent_20260915.json`。旧参考：`/Users/river/.codex/worktrees/d00e/CrossInterest/artifacts/mind-old-new-interest-v1/receipts/full_mixed_g4_v1.json`。最终参数摘要 `bccaca966996ada02b851afa0ed5fd0b2357783043f04b99f481edb4af1a3382`。

### 产物保存规则更新（2026-09-15）

用户更新AGENTS.md后，后续大型产物保留在持久远端存储；Meta完整产物及独立指标复算在远端验收，本地仅保留紧凑指标、报告、manifest、哈希和恢复指针。上述Mixed下载/验收记录是更新前已完成的历史事实。已经主动清理的本地大产物不自动重新下载，既有验收不会因本地副本缺失失效。自动跟进已撤销旧Meta本地完整下载步骤。
