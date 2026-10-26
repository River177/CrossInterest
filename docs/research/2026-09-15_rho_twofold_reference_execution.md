# 两折参考模型：执行协议与核验入口

用户已批准先将训练用户分成两半。本阶段训练两个独立参考模型，并为全部 21,600 条 NI 样本生成交叉评分；阶段结束后保留结果，用于后续动态可约减损失权重对照。当前阶段不自动提交学生训练。

| 项目 | 冻结设置 |
|---|---|
| 输入/输出 | 原 History Behavior+Interest → 原单个目标 SID；原始 messages、顺序、重复历史均保留 |
| 数据 | 21,600 个用户，108 类 × 200 |
| 两折 | 每折 10,800 人，108 类 × 100；类内按 seed42 哈希排序平分 |
| 训练/打分方向 | Teacher0 训练 fold0、评分 fold1；Teacher1 反向 |
| 模型 | Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 初始化 | 原始预训练模型、mean-norm SID 初始化、全新 LoRA；不继承 Mixed adapter |
| 训练 | 每折 3 epochs、32,400 样本呈现、1,014 次更新；最后一个 checkpoint 固定用于评分 |
| 参数 | all-linear LoRA r16/alpha32/dropout0.05；108 个 tied SID 行 |
| 优化 | LR2e-5、AdamW、weight decay0.01、cosine、3% warmup、clip1.0、seed42 |
| GPU | 两组各 4×A100，microbatch1、accumulation8，global batch32；两组并行最多8卡 |
| 训练损失 | 原生完整词表 completion SFT；SID、结束符和换行三监督 token，历史/padding 掩码 |
| 参考评分 | 首个 SID 的完整有效词表 NLL；最终 head FP32 投影，不作类别先验修正 |
| 附加诊断 | target logit、full logsumexp、108 类条件 NLL、108 类分数与排名 |
| 打分分片 | 4 个逻辑 shard，每次4条，256条持久分片 |
| 存储 | 权重、optimizer、checkpoint、完整评分留在持久远端；独立 reader 远端重新读取并核验，本地仅 compact JSON |

评分中的有效 head 保留 PEFT 原生行为：tied SID replacement rows 先转换成基座 BF16，再将完整有效 head 提升到 FP32 计算分数。它不同于旧 relation scorer 直接用未转换 FP32 SID 行的数值定义；后续学生必须使用本协议的同一评分器，不能混用旧分数。

## 已完成的数据审计

两折用户交集为0、并集精确等于原始21,600 NI用户。每类两折均100人。对全部21,600条原始完整历史核验新颖性，通过；完整历史最长607条，先判新颖性，再保留模型可见H50。与5354人验证集及Dev27082人/31392 impressions的用户交集均为0。

两折最长输入为1564/1563 tokens，无超2048样本；每条恰好3个监督token。数据不是根据loss、语义相似度或本次结果筛选的。

| 文件 | SHA-256 |
|---|---|
| fold0.parquet | `7bf5894066ae63f3a7c89b44f888797fc0688a652d18bd83c5722dac08ecd600` |
| fold1.parquet | `aa929f113c2ba8805f90c0f7f898ce646694112cf0df8f1ced5e9d30211e3cc9` |
| SID mapping | `45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a` |

数据目录：`data/training/mind_rho_twofold_v1`；来源、原始构造身份、划分规则与逐类数量见其 `metadata.json`。所有数据仍维持单个训练target，不新增多正例监督。

## 执行与恢复

先执行一个4卡工程核验作业：108 SID梯度覆盖、20步单卡小样本训练、4步4卡训练、参数与logits精确重载、64条另一折样本的4分片评分和独立远端回读。通过后用完全相同的已冻结代码和输入提交两组正式参考训练。工程核验权重不进入正式训练。

训练复用 `train_sft` 原生目标、采样器和优化器。新 wrapper 只记录各rank初始化及Adam/scheduler游标，并用节点本地标记隔离rank0的网络checkpoint发布，避免等待网络I/O的NCCL超时。checkpoint按身份与hash恢复；结果只在完整发布与重新读取通过后接受。

本地已完成全套测试1013项，以及后续新增/定向测试：数据与训练评分66项、pipeline与worker52项通过。集合有重叠，不合计为新的总数。

运行入口：

- Gate experiment：`mind-rho-twofold-gate-v1-0915`，job `qwen35-2b-rho-reference-gate`。
- Formal experiment：`mind-rho-twofold-reference-v1-0915`，jobs `qwen35-2b-rho-reference-fold0/1`。是否已提交以状态文件为准，不能仅凭协议称已运行。
- 本地有限流程：`scripts/analysis/run_rho_twofold_reference_pipeline.py`。
- 最新阶段：`artifacts/rho-twofold-reference-20260915/pipeline_state.json`。
- 最终两折核验：同目录 `twofold_acceptance.json`，完成后才存在。
- Gate代码包：同目录 `bundle-gate-v1`，runtime SHA `3b04e451724ec79b8d6a6de32d9570c7263cbdc4b9b2f7e2a9a7ee33ce542baa`。

本地流程只允许获取明确列出的 compact JSON 文件名（AMLT/AzCopy 要求 basename，不支持带目录通配符的 include）。不会因本地缺少权重或预测而重下载大文件。正式参考阶段接受条件包括两组各1014更新、相同初始参数、各10800条独立用户覆盖、训练与评分方向互为补集、完整远端评分hash/数值/排名重算通过。

## 2026-09-15 19:48 恢复记录

Gate v1 的108个SID梯度核验完成，单卡20步完成，日志loss从4.668下降至1.747；随后因 `rank-0-training-runtime.json` 缺失而失败，四卡训练与OOF评分尚未执行，正式两折训练未提交。

冷启动入口回归定位到 Transformers 首次lazy解析 Trainer 时，`processing_utils.direct_transformers_import` 会替换根模块；旧wrapper修改了旧模块对象，因此真实 `train_sft` 使用原生Trainer，审计回调未安装。修复为显式可选 `trainer_base` 注入，默认训练路径保持不变，并在训练开始前记录和断言回调已启用。全套本地1079测试通过；新增恢复边界的定向测试另通过。

Gate v2 从旧远端已提交 checkpoint-20 恢复单卡阶段，逐文件hash、优化器/scheduler20、原20条loss记录和最终tensor不变必须全部核验；要求新增optimizer updates为0。初始化hash在恢复时重建，明确标注原运行现场审计缺失，不补造历史记录。随后仍执行原计划的全新4卡4步和64条交叉评分。原108SID梯度报告仅在数据/模型/实现hash一致时复用。

- 恢复实验：`mind-rho-twofold-gate-v2-0915`；job名字不变。
- 新runtime SHA：`7b465ea152e70967eced0c61bf3a93f89f4818c5707f60a0322137cfdbc1656e`。
- 旧checkpoint marker SHA：`c948423a4d06cf9e01271392ee734a0eb1f68154d1e6f314c1e7f42e91f17dc1`。
- 本地仅保存旧checkpoint的compact manifest，所有模型与优化器字节均在远端读取。
- supervisor使用 `--gate-version 2`；失败终态立即进入 `needs_attention`，不再将缺失成功报告误当无限可重试的下载错误。
- 原生记录与修复收据：`artifacts/rho-twofold-reference-20260915/incidents/missing_runtime_callback_and_compact_transport_v1.json`。

本段仅记录已提交恢复及CPU验证；GPU核验通过与正式提交以 `pipeline_state.json` 和实际acceptance文件为准。

后续使用这些参考loss时，仍只能称其为跨用户可预测性的估计，不能把高参考loss当作已确认的错标签或真正不可约误差。
