# MIND 108 类 CIR 3 epochs → NI 3 epochs：最终结果

**已完成并独立验收。** CIR 1,620 步与 NI 2,025 步、两阶段全部曝光、参数交接、精确重载和 Full Dev 均通过核验。本组三轮 CIR 的 Interest-Macro Recall@10 为 **21.8272%**，比一轮 CIR 提高 **0.5486 个百分点**，比 NI-only 提高 **1.4997 个百分点**。四项主要 @10/@20 指标均提高；@5 相对一轮 CIR 下降，详见结果表。

远端程序于 2026-09-17 20:01:22 UTC（北京时间 9 月 18 日 04:01:22）正常退出，平台于 20:04:54 UTC 记为 Completed。观测、下载和最终验收时间分别保存在收据中，不能用心跳触发目录名代替实际观察时间。

## 1. 研究问题

将已有 CIR 预训练从 1 epoch 增至 3 epochs，并保持后续 NI 3 epochs，能否进一步提高 MIND 108 类新兴趣排名？本次仅新增 **CIR3→NI3** 一个条件，与已接受的 **CIR1→NI3** 和共享 **NI-only** 比较。未执行标签打乱、多 seed、类别均衡或额外 NI 预训练对照。

## 2. 训练数据与训练设置

CIR 使用原固定 17,271 条历史视图，未重新抽样或重新生成。来源仍为原 21,600 名 NI 训练用户的结构化新闻浏览历史：先取最后 50 条，再删除选中类别的所有行为、标题与类别；不回填更早记录，保留时间顺序，删除后须至少剩 5 条行为和另一类别。

seed=42，用用户、impression 与候选类别的哈希选一个删除类别，未来 NI 目标及包含其索引的源 sample ID 不参与选择。每源用户最多一个 CIR 视图，使用 news/click 语境，以被删类别的单个 SID 为监督目标。受保护的 NI/OI/Dev/Validation 输入按最终可见有序历史做精确去重；Dev/Validation 用户交集为 0。原构造排除 4,329 个无合格删除类别的用户，受保护输入重复排除 0 条。CIR 覆盖 108 类，每类 3–1,056 条，不做均衡。

| 条件／阶段 | 样本数 | 类别数 | 每类样本数 | Epochs | 接受的样本曝光 | 更新步数 |
|---|---:|---:|---:|---:|---:|---:|
| NI-only（已有） | 21,600 | 108 | 200 | 3 | 64,800 | 2,025 |
| CIR1→NI3：CIR（已有） | 17,271 | 108 | 3–1,056 | 1 | 17,271 | 540 |
| CIR1→NI3：NI（已有） | 21,600 | 108 | 200 | 3 | 64,800 | 2,025 |
| CIR3→NI3：CIR（本次） | 17,271 | 108 | 3–1,056 | 3 | 51,813 | 1,620 |
| CIR3→NI3：NI（本次） | 21,600 | 108 | 200 | 3 | 64,800 | 2,025 |

本次两个阶段均核验每个样本恰好出现 3 次；NI 108 类各曝光 600 次。CIR/NI 最大输入分别为 1,524/1,564 tokens，无超过 2,048 上限的样本。每条 completion 的 3 个监督 token 包含目标 SID 与模板结束符。

### 真实 CIR 训练样本

来源：`cir.parquet`，sample ID `cir:train:1330628:19`。被删类别为 `Government & Public Policy`（政府与公共政策）。下面的 response 是训练监督标签，用于说明构造，不代表模型测试预测。

英文原始 prompt：

```text
The following news browsing history has all entries from one topic category removed. The remaining entries are in chronological order, formatted as "[Topic category] Headline":

1. [Mental Health] Service dog giving veterans hope when struggling with PTSD
2. [Music] The 25 greatest English rock bands
3. [War & Armed Conflict] After Baghdadi raid, Syria villagers tell of quiet neighbour
4. [Film] Must-see biopics
5. [Music] Hip hop's richest celebrities

Infer the removed topic category from the remaining observed interests and news items. This is historical interest reconstruction, not a prediction of the next click. Respond with only its single-token Interest ID in the exact format <interest_NNN>. Do not output the category name or any explanation.
```

中文翻译：

```text
下面的新闻浏览历史已删除某一个主题类别的全部条目。剩余条目按时间顺序排列，格式为“[主题类别] 标题”：

1. [心理健康] 服务犬为饱受创伤后应激障碍困扰的退伍军人带来希望
2. [音乐] 25 支最伟大的英格兰摇滚乐队
3. [战争与武装冲突] 巴格达迪突袭行动后，叙利亚村民讲述那位安静的邻居
4. [电影] 不容错过的传记电影
5. [音乐] 嘻哈界最富有的名人

根据剩余的已观察兴趣和新闻条目，推断被删去的主题类别。这是历史兴趣重建，不是预测下一次点击。只用该类别的单个 token Interest ID 回答，严格采用 <interest_NNN> 格式。不要输出类别名称或任何解释。
```

原始 response：`<interest_054>`。中文含义：输出政府与公共政策对应的 Interest ID，ID 本身不翻译。

### 训练与恢复

模型固定 `Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。LoRA r16、alpha32、dropout0.05，all-linear LoRA 与 SID 参数训练；AdamW、学习率 2e-5、weight decay0.01、cosine、warmup ratio0.03、seed42、BF16、关闭 thinking。损失为 native full-vocabulary completion-only token-mean SFT。

正式执行 Recall Standard **4×A100-80GB**，microbatch2、梯度累积4、global batch32，非 reentrant gradient checkpointing。训练、保存后重载及推理均显式采用 Torch fallback。CIR 从原始初始化开始，学习率调度覆盖全部 1,620 步；CIR→NI 继承全部 LoRA/SID 参数，重新初始化 optimizer/scheduler。数据、Trainer、loss、sampler、恢复、序列化及评估等 83 个原 payload 文件字节一致，仅修改时长合同和阶段编排，哈希绑定复用已有 GPU 诊断验收。

本地针对性测试 28 passed、1 skipped（本地缺少 Transformers 的既有 CPU Trainer 用例）；真实冻结数据的三轮 ledger、阶段 launch config／cursor、实际 AMLT 上传过滤与 YAML dump 已验证。正式两阶段各通过 **373 个 tensor 和 logits 精确 fresh reload（atol=rtol=0）**。

NI 在原 attempt 日志 830 步附近被更高 SLA 回收，随后同一 backend 自动从已提交 checkpoint-800 恢复。373 个 Adam 状态、scheduler step800、原 cosine 学习率及非零 epoch/skip 的样本游标核验通过，已完成的 CIR 完整复用。原 attempt 未提交的 NI 801–830 被重放；表中的曝光描述最终接受训练轨迹，不计作额外训练 epoch。后续仅出现本地观测网络超时和 AMLT Table 认证错误；通过原生日志与 AzureCliCredential 直接 Blob 读回恢复观察，未重提训练。

## 3. 测试设计

三组使用同一冻结 `dev_full_grouped.parquet`：**31,392 个 grouped impressions、37,728 个目标出现、104 个有目标支持类别**。完整 108 类 SID 做 next-token 排名、保留 Top20，不过滤历史类别，关闭 thinking。四个 modulo 分片各 7,848 条，使用本次最终 NI adapter 在远端推理。

本组最终 adapter 文件 SHA-256：`f4544e9c8d8dfed3cca5f110a5bf760af1cada17d813097004a4e9a292f6f389`；评估 adapter 目录 SHA-256：`a3a44b897bad3dbe9aae7be75a4c47a6074f8da361a43d6266a98a9c7622d4ca`。文件与目录是不同的哈希对象。

Interest-Macro 先按每个目标类别的出现计算 Recall 与折扣增益，再对 104 个有支持类别等权平均。目标在前 K 名时 Recall 贡献为 1，NDCG 贡献为 `1/log2(rank+1)`，否则为 0；该类别宏平均口径不同于按 impression 平均的多标签 NDCG。没有使用 LLM judge。

远端独立检查器核验完整预测记录身份、合法排名、分片拓扑、无缺失／重复覆盖及预测哈希，并从完整预测重算指标。本地独立回读校验冻结 Dev 的每类与每分片支持数，再从四个分片的充分统计重算所有类别 Recall/NDCG 及宏平均，误差小于 1e-12。基线文件哈希、数据、模型 revision、映射、排名口径与本组对齐。完整权重、optimizer、checkpoint 和预测保留远端；本地最终下载只含 38 个紧凑 JSON、共 1,977,024 bytes。报告未导出单条预测案例，训练样本不作为模型预测证据。

## 4. 实验结果

所有数值为百分比；差值单位为百分点。NI-only 和 CIR1→NI3 为已有接受结果，未重跑。

| 指标（Interest-Macro） | NI-only | CIR1→NI3 | CIR3→NI3 | 相对 CIR1 | 相对 NI-only |
|---|---:|---:|---:|---:|---:|
| Recall@10 | 20.3275 | 21.2786 | 21.8272 | +0.5486 | +1.4997 |
| Recall@20 | 33.6553 | 34.2609 | 35.1740 | +0.9131 | +1.5187 |
| NDCG@10 | 10.1325 | 10.8068 | 11.0708 | +0.2640 | +0.9383 |
| NDCG@20 | 13.4663 | 14.0725 | 14.4257 | +0.3532 | +0.9595 |

本次将 CIR 增至 3 epochs 后，四项主要 @10/@20 指标继续提高。以下保留同次评估的其他 cutoff，以呈现全部测量结果：

| 指标（Interest-Macro） | NI-only | CIR1→NI3 | CIR3→NI3 | 相对 CIR1（百分点） |
|---|---:|---:|---:|---:|
| Recall@1 | 2.8761 | 3.3788 | 3.7716 | +0.3928 |
| Recall@5 | 12.3181 | 13.0010 | 12.3575 | -0.6435 |
| NDCG@1 | 2.8761 | 3.3788 | 3.7716 | +0.3928 |
| NDCG@5 | 7.5680 | 8.1516 | 8.0403 | -0.1113 |

Recall@5 相对一轮 CIR 下降 0.6435 个百分点、NDCG@5 下降 0.1113 个百分点；因此本次结果不能表述为所有 cutoff 一致改善。比较仍为 seed42 的单次观察，不主张统计显著性。CIR3 相对 CIR1 增加 1,080 次更新并延长 cosine horizon；该比较反映整个预训练时长设置的差异。共享 NI-only 采用历史单卡→四卡／FLA 安排，与本组统一 Torch fallback 有执行差异，不能将其差值单独归因于 CIR 机制。

## 5. 数据与产物路径

| 用途 | 已验证入口 |
|---|---|
| 固定 CIR／NI／Dev／108 类映射 | [原数据目录](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1) |
| 新实验配置／作业定义 | [三轮配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/experiments/mind_cir_108_cir3_ni3_v1.json)；[AMLT YAML](../../../configs/amlt/archive/2026-10-26/mind_cir3_ni3_108_full_0917_v1.yaml) |
| 冻结运行包及预检 | [bundle-v1](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/bundle-v1)；[local_preflight.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/local_preflight.json) |
| 最终接受收据／训练与交接核验 | [final_acceptance.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/final_acceptance.json)；[lineage readback](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/final_training_lineage_readback.json) |
| ETag 绑定下载收据／远端文件目录 | [final_download_receipt.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/final_download_receipt.json) |
| 阶段、重载、checkpoint 与分片统计 | [final-readback](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/final-readback/mind-cir3-ni3-108-0917-v1/qwen35-2b-cir3-ni3-108) |
| 合并 metrics／结果 CSV | [metrics.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/final-readback/mind-cir3-ni3-108-0917-v1/qwen35-2b-cir3-ni3-108/formal/real_eval/merged/metrics.json)；[results.csv](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-17/mind_cir3_ni3_results.csv) |
| 完成原生日志／恢复收据 | [native receipt](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/progress/20260917T2040Z/receipt.json)；[CP800 resume](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir3-20260917/progress/20260917T1414Z/automatic_resume_readback.json) |
| 重算入口／跟踪清单 | [accept_cir_final.py](/Users/river/Documents/MSRA/Rec/CrossInterest/scripts/analysis/accept_cir_final.py)；[manifest](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir3_ni3_108_20260917_v1.json) |
| 原 CIR1 报告／本组三轮运行日志 | [CIR1 report](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-16/mind_cir_warmup.md)；[runlog](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-17/mind_cir3_ni3_runlog.md) |

实验 `mind-cir3-ni3-108-0917-v1`，job `qwen35-2b-cir3-ni3-108`，backend `nifty_avocado_n4yyg08wwd`：[AMLT 作业](https://aka.ms/amlt?q=k2mmn)。远端账号 `csnewsandfeeds4150361735`、container `amulet`，结果前缀 `projects/sft4safety/amlt-results/7210412790.94297-54d6cdaf-7b23`；其中 `formal/cir`、`formal/ni`、`formal/checkpoints-*` 与 `formal/real_eval` 保留完整产物。监控仅为本组三轮条件，完成后停止，状态以清单为准。

| 身份 | SHA-256 |
|---|---|
| 冻结 bundle manifest | `21e78ca2abe9382e464f174f334c1af17c51ecfda1f71a08fc8d3e9389d7ba88` |
| 最终接受收据 | `7cfac08ebf15ea72ed44e962de1c4f3adf238a4a545d8a005f375864c7d34ce1` |
| 训练及参数交接收据 | `b36e91649d547e1ceccf4bec667f40b190815e844b1bf0ee9f26d9abd57e1cba` |
| CIR3 adapter 文件 | `dceda49eaaf4fa258ced1fe8ba7146dc494b6cb1d51d26855906e5270abd2c77` |
| 最终 NI adapter 文件 | `f4544e9c8d8dfed3cca5f110a5bf760af1cada17d813097004a4e9a292f6f389` |
| 完整预测（远端） | `e5ab94b8c58805bdcc8483ca706ea55b1bc4680c0954ffc7c1073ae3b5e989f5` |
| 合并 metrics | `b825c260c923afe21ed900b3e8a8c03c0ceac5ce000d2bc39ade420461e5955b` |
| 评估 identity | `380a1df8e49f58f932b61c3ea047cb17a5c2b8449c57c34a9cab9a083632815b` |
