# MIND 108 类 CIR → NI：最终训练与 Full Dev 结果

**状态：已完成并独立验收。** Qwen3.5-2B 完成 CIR 540 步、NI 2,025 步，Full Dev 覆盖 31,392 个 impressions。Interest-Macro Recall@10 从 20.3275% 提高至 21.2786%（+0.9511 个百分点）。远端完成标记时间为 2026-09-17 02:36:57 UTC（北京时间 10:36:57）；最终验收见下方产物索引。

## 1. 研究问题

在真实新兴趣预测前增加一次“从历史中删除某类全部行为，再预测被删类别”的辅助训练，是否改善新兴趣排名？仅新增一组 **CIR 1 epoch → NI 3 epochs**，与已有、已验收的 **NI-only 3 epochs** 比较。模型、MIND 108 类 SID 映射、原 NI 数据和 Full Dev 保持冻结；基线身份为 `qwen35-2b-bi-sid-ni-only-21600-seed42-v1`。

## 2. 训练数据与训练设置

源数据来自现有 MIND 108 类重分类数据及 21,600 条 NI 训练样本对应的结构化历史。每个源用户一条样本；先截取最后 50 条行为，再删除所选类别的全部标题与类别记录，保持剩余行为的时间顺序，不从更早历史回填。删除后须至少留下 5 条行为和另一类别。

固定 seed=42，以用户、impression 和候选类别的 SHA-256 确定一个删除类别，每个源样本最多一个视图。未来 NI 标签不参与类别选择；原 sample ID 含未来标签索引，因此也不用于选择键。标签为被删类别的单个 SID。提示词统一采用新闻浏览／点击语境（news/click），明确区分历史重建和下一次点击预测。

按最终可见的有序 `[类别] 标题` 历史排除与受保护 NI/OI 训练输入、Dev 和独立 Validation 输入完全相同的 CIR 视图；只删除辅助样本，保留原任务数据。Dev/Validation 用户与 CIR 用户交集为 0，held-out 标签不参与辅助类别选择。

| 数据／阶段 | 独立样本数 | 类别数 | 每类样本数 | Epochs | 实际样本曝光 | 更新步数 |
|---|---:|---:|---:|---:|---:|---:|
| CIR | 17,271 | 108 | 3–1,056 | 1 | 17,271 | 540 |
| 原 NI（本组第二阶段） | 21,600 | 108 | 200 | 3 | 64,800 | 2,025 |
| NI-only（复用已有基线） | 21,600 | 108 | 200 | 3 | 64,800 | 2,025 |
| Full Dev（两组共同评估） | 31,392 impressions | 104 有目标支持 | 按冻结分布 | — | 37,728 个目标出现 | — |

CIR 有 4,329 个源用户无合格删除类别，因受保护输入重复而排除 0 条。不做类别均衡、不设每类上限。CIR/NI 最大输入分别为 1,524/1,564 tokens，均未超出 2,048 上限。完整构造规则和逐类数量保存在冻结配置、数据元信息与独立回读收据中。

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

### 模型与阶段交接

模型固定为 `Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。LoRA r=16、alpha=32、dropout=0.05，训练 all-linear LoRA 和 SID 参数；学习率 2e-5，weight decay=0.01，AdamW、cosine、warmup ratio=0.03、seed=42、BF16，关闭 thinking。

采用 native full-vocabulary completion-only SFT，token-mean reduction；SID 与模板结束符共 3 个监督 token。正式执行为 Recall Standard 4×A100-80GB，每卡 microbatch=2、梯度累积=4、全局 batch=32，非 reentrant gradient checkpointing。训练、保存后重载与推理均显式采用 Torch fallback。

CIR→NI 继承全部 LoRA/SID 参数，优化器与 scheduler 重新初始化。逐样本曝光审计确认 CIR 每样本恰好 1 次、NI 每样本恰好 3 次；NI 108 类各曝光 600 次。两阶段分别通过 373 个 tensor 的精确 fresh reload 与 logits 一致性检查（atol=rtol=0）。平台回收后的 CIR 从已提交 step-100 恢复，Adam 与 scheduler 游标核验通过；恢复过程保留于运行日志。

## 3. 测试设计

使用本组最终 NI adapter 在同一远端作业执行 Full Dev 推理，adapter 文件 SHA-256 为 `e9692ee2460b9425bc93ee734dc2ff383022476d4d46a79b83b13d7b9d3cae17`。评估端 adapter 目录哈希为 `014c18f57e0b7847479b2eb16833c3260e0f3055838cad7bfc66534c0660010b`；二者哈希对象不同。

测试输入来自冻结的 `dev_full_grouped.parquet`，共 31,392 个 grouped impressions、37,728 个目标出现，104 类具有真实目标支持。对完整 108 类 SID 排名，保留 Top-20，不过滤历史类别。四个 modulo 分片各 7,848 条；最终覆盖、记录身份、分片拓扑、合法排名与文件哈希全部通过远端独立检查。

主要指标为 Interest-Macro Recall@10/20、NDCG@10/20：先对每个目标类别的所有出现计算命中率和位置折扣，再对 104 个有支持类别等权平均。类别目标在前 K 名时，Recall 贡献为 1，NDCG 贡献为 `1/log2(rank+1)`；否则为 0。这里使用类别宏平均口径，不等同于按 impression 平均的多标签 NDCG。未使用 LLM judge。

完整预测和权重保留远端。本地独立回读使用冻结 Dev 重算类别及分片支持数，再由四个分片的类别充分统计重算全部类别 Recall/NDCG 与最终宏平均，误差均小于 1e-12；基线实际指标文件及相同评估数据／映射／排名范围已核对。此报告未导出单条测试预测案例，训练例不作为预测效果证据。

## 4. 实验结果

全部数值为百分比，差值为百分点。两组共同评估 31,392 个 impressions／37,728 个目标出现／104 个有支持类别。

| 指标（Interest-Macro） | NI-only | CIR→NI | 差值（百分点） |
|---|---:|---:|---:|
| Recall@10 | 20.3275 | 21.2786 | +0.9511 |
| Recall@20 | 33.6553 | 34.2609 | +0.6056 |
| NDCG@10 | 10.1325 | 10.8068 | +0.6744 |
| NDCG@20 | 13.4663 | 14.0725 | +0.6063 |

本次四项主要指标均提高。比较为 seed=42 的单次观察，未做多 seed 或等训练预算对照；CIR→NI 额外增加 540 次更新，已有 NI-only 基线还采用历史单卡→四卡／FLA 执行安排，本组统一 Torch fallback。因此这些差值描述本次实际效果，不能单独归因于 CIR 机制，也不构成统计显著性结论。

远端 native status 为 Completed，完整验收为 accepted。两阶段训练、参数交接、精确重载、完整 Dev 排名与独立指标重算均已完成。原始指标与独立 `fsum` 统计间的浮点差异最多 2.78e-17；指标验收容差为 1e-12，未放宽模型／logits 精确重载门槛。

## 5. 数据与产物路径

| 用途 | 本地已验证入口 |
|---|---|
| 原始数据及规则恢复线索 | [冻结数据 metadata](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1/metadata.json) |
| CIR 训练数据 | [cir.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1/cir.parquet) |
| 原 NI 训练数据 | [ni.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1/ni.parquet) |
| Full Dev／SID 映射 | [Dev](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1/dev_full_grouped.parquet)；[mapping](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_108_v1/interest_token_mapping.parquet) |
| 数据独立回读 | [data_independent_readback.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/data_independent_readback.json) |
| 冻结运行包／基线描述 | [bundle-v3](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/bundle-v3)；[h50_baseline.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/bundle-v3/manifests/h50_baseline.json) |
| 最终接受收据 | [final_acceptance.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/final_acceptance.json) |
| 阶段／重载／远端评估收据 | [final-readback](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/final-readback/mind-cir-108-0916-v3/qwen35-2b-cir-ni-108) |
| 最终完整指标／比较 CSV | [metrics.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/final-readback/mind-cir-108-0916-v3/qwen35-2b-cir-ni-108/formal/real_eval/merged/metrics.json)；[results.csv](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-16/mind_cir_warmup_results.csv) |
| 本地独立指标重算入口 | [accept_cir_final.py](/Users/river/Documents/MSRA/Rec/CrossInterest/scripts/analysis/accept_cir_final.py) |
| 构造器／编排入口 | [build_cir_dataset.py](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/data/build_cir_dataset.py)；[cir_warmup.py](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/experiments/cir_warmup.py) |
| 冻结配置／正式作业定义 | [experiment config](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/experiments/mind_cir_108_v1.json)；[v3 AMLT YAML](../../../configs/amlt/archive/2026-10-26/mind_cir_108_full_0916_v3.yaml) |
| 跟踪清单／历史运行记录 | [manifest](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_108_20260916_v1.json)；[runlog](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-16/mind_cir_warmup_runlog.md) |

成功实验 `mind-cir-108-0916-v3`，job `qwen35-2b-cir-ni-108`，backend `dynamic_insect_xyybk1nb24`，平台 Recall Standard 4×A100-80GB：[AMLT 作业](https://aka.ms/amlt?q=k2inm)。远端存储账号 `csnewsandfeeds4150361735`，container `amulet`，结果前缀 `projects/sft4safety/amlt-results/7210459117.84505-c58ab0c5-3f5e`。此作业目录下的 `formal/cir`、`formal/ni` 与 `formal/real_eval` 保存训练和评估产物；本地仅保留紧凑统计、哈希、报告和恢复指针。

| 身份 | SHA-256 |
|---|---|
| v3 bundle manifest | `97d1e5668bee8b858250dd3e4bd88209034a2a281f239c4d414f3bd4f5b09c3d` |
| 最终接受收据 | `1b0b748a94b715602c9fde7c37051f653d0ce6a05c43af853cba13f187382018` |
| 合并完整预测（远端） | `15fd7c50f9d6e2f93773e4674e706f1b7fdd013f27b17e91675bcedad3d96375` |
| 合并 metrics | `482d9e2a257c2eefa360db02990a23584de96b323e0e09cf452cce6cf26f17c1` |
| 基线 metrics | `7c96deb861af4439df832e0b7836f3e899bf32fe56936494905995530d3dd735` |

v1/v2 启动故障发生于正式更新前，修复和 v3 身份核验见运行日志。初始完整测试 1,472 passed／8 skipped；任务名兼容修复后 14 项 focused tests 通过。最终状态与监控配置以跟踪清单为准。
