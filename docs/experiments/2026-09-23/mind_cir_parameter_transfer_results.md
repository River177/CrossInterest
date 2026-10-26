# MIND CIR参数迁移：最终实验报告

两组新增NI训练、完整Dev、功能诊断和原生终态均已独立接受。验收时间：2026-09-24（北京时间）。本轮没有得到某一个参数块单独主导收益的清晰证据；完整继承在四项预设主要指标上均最高。

## 1. 研究问题

固定已接受CIR5模型的LoRA状态LC及SID状态EC，与同一原始初始化L0/E0组合，检验CIR的哪一部分初始化有助于后续NI训练。比较完整继承(LC, EC)、保留LoRA/重置SID(LC, E0)、保留SID/重置LoRA(L0, EC)。**两组NI阶段均训练全部372个LoRA张量及1个SID张量，干预仅发生在初始化。**

本轮可以描述初始化迁移与功能诊断的变化，不能等同于定位“知识存储位置”，也不能单独证明学到了跨兴趣关联。没有新增冷启动NI-only、多seed、标签打乱、采样平衡或其他训练条件。历史NI-only含单卡转四卡/FLA执行差异，仅为背景，本报告不据此计算严格2×2贡献比例或交互效应。

## 2. 训练数据与条件

原冻结MIND NI数据共21,600条、108类，每类200条；保留news/click语境、原提示词、SID映射与样本顺序。原始SID恢复为语义初始化，LoRA恢复完整A/B原始矩阵，均通过逐张量hash核对。复用已接受CIR5（17,271条固定擦除视图，5轮/86,355次曝光/2,700步），不重新训练CIR。

| 条件 | NI初始LoRA | NI初始SID | 新增NI训练 | NI有效曝光 |
|---|---|---|---|---|
| 完整继承 CIR5→NI3 | LC | EC | 复用已接受结果 | 64,800 |
| 保留LoRA、重置SID | LC | 精确E0 | 3轮 / 2,025步 | 64,800 |
| 保留SID、重置LoRA | 精确L0 | EC | 3轮 / 2,025步 | 64,800 |

每个新增分支均为Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，seed42；LoRA r16/alpha32/dropout0.05，lr2e-5、weight decay0.01、warmup0.03、cosine horizon2025，新optimizer/scheduler。BF16、显式Torch fallback、非reentrant checkpointing、native full-vocabulary completion-only token-mean SFT。每组4×A10080GB Recall Standard，microbatch2、accumulation4、global batch32；最多8GPU并发。原103个数值运行/数据payload文件字节不变。

每组最终实际每样本3次、每类600次；检查点精确保留模型、Adam、scheduler、RNG和样本游标。平台抢占通过原同身份checkpoint恢复，丢弃的未提交更新不计入有效曝光。LoRA非零epoch+非零skip恢复后的801–900步实际有序样本记录已独立逐步核验。启动v1的PEFT来源收据问题在任何NI更新前修复；v2的组合及精确重载通过后才训练。

### 数据格式示例

为展示输入格式，确定性选取冻结训练表中最短prompt；该例不用于解释效果。sample_id：`train:1820310:100:behavior_interest`，history_length=1。以下保留原始英文prompt及转义字符：

```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Trivia & Puzzles] The News In Cartoons

Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form <interest_NNN>.
```

中文译文：

```text
我最近浏览的新闻标题及对应主题类别按时间顺序列在下面。每条格式为“[主题类别] 标题”：

1. [趣闻与谜题] 漫画中的新闻

根据标题和主题类别，预测我接下来可能感兴趣的一个新主题类别。预测类别不能出现在我的浏览历史主题类别中。仅回答分配给预测类别的已学习兴趣token，格式为 <interest_NNN>。
```

训练目标响应：`<interest_065>`；标签为`Internet & Social Platforms`，中文含义“互联网与社交平台”。这是监督目标，不是展示模型预测。

## 3. 测试设计

正式结果固定为NI第3轮终点。在远端对固定108个SID输出Top20，不做历史类别后过滤。Full Dev共31,392 grouped impressions、37,728目标出现、104个有支持类别；四分片各7,848。全部方法复用相同数据、候选池和评价口径。指标在远端从完整预测重算，本地从四份紧凑分片指标独立复算逐类及宏平均值，全部一致。

Macro Recall@K先按类别统计目标是否进入TopK，再对104个支持类别等权平均。Macro NDCG@K按目标排名r计入`1/log2(r+1)`（r≤K），同样先按类取均值再宏平均。因此主表没有一个可替代宏平均的全局“命中数/37,728”。本评估不使用LLM judge，没有judge提示词、拒答或ASR指标。只运行seed42，没有训练稳定性或显著性结论。

功能诊断均不参与训练或checkpoint选择：类别名称→SID为108条/108类，每类1条；历史类别重建为512名独立Dev用户/512个目标/87个支持类。用户由固定hash从16,158名符合条件者中选出；原Dev有27,082名不同用户。目标仅由历史选择，擦除类别后保留其余历史，不使用未来NI标签，与训练用户及CIR输入零重叠，不做类别平衡。各组NI前后均测试；完整继承的前后模型另做同样探针，只有评价、没有额外训练。

一个真实名称诊断例子（确定性最短prompt）：

```text
Topic category: Film

Respond with only the learned interest token assigned to this topic category, in the exact format <interest_NNN>. Do not output any explanation.
```

中文译文：

```text
主题类别：电影

仅回答分配给该主题类别的已学习兴趣token，严格使用 <interest_NNN> 格式。不要输出任何解释。
```

目标响应`<interest_045>`，中文含义“电影”。保留SID组在NI前该类别只有1条、Recall@1=1，所以可由已接受逐类收据唯一确定此条Top1为`<interest_045>`；这里报告SID排名结果，没有重新生成一段回答。完整预测仍保留远端。

## 4. 实验结果

### 4.1 主要指标

三行均使用31,392条impressions、37,728个目标出现、104支持类；数值为百分比。

| NI初始化条件 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| 完整继承 CIR5→NI3 | 21.9717% | 35.0756% | 11.2051% | 14.4966% |
| 保留 LoRA、重置 SID | 20.8325% | 34.9392% | 10.6330% | 14.1928% |
| 保留 SID、重置 LoRA | 21.2467% | 33.9301% | 10.6177% | 13.7973% |

相对完整继承的差值（百分点）：

| 相对完整继承 | ΔR@10 | ΔR@20 | ΔNDCG@10 | ΔNDCG@20 |
|---|---|---|---|---|
| 保留 LoRA、重置 SID | -1.1392 | -0.1364 | -0.5721 | -0.3038 |
| 保留 SID、重置 LoRA | -0.7250 | -1.1455 | -0.5874 | -0.6993 |

完整继承在四项预设主要指标上最高。保留SID组的Recall@10略高于保留LoRA组；保留LoRA组的Recall@20、NDCG@20较高，两者NDCG@10几乎相同。局部重置的主指标下降范围约0.1364–1.1455个百分点，不能据此宣称某块解释全部CIR收益或宣称统计显著。

### 4.2 全部预设cutoff

Macro Recall（%）：

| 条件 | @1 | @5 | @10 | @20 |
|---|---|---|---|---|
| 完整继承 CIR5→NI3 | 3.9831 | 12.6653 | 21.9717 | 35.0756 |
| 保留 LoRA、重置 SID | 3.4327 | 12.7847 | 20.8325 | 34.9392 |
| 保留 SID、重置 LoRA | 3.2244 | 12.5003 | 21.2467 | 33.9301 |

Macro NDCG（%）：

| 条件 | @1 | @5 | @10 | @20 |
|---|---|---|---|---|
| 完整继承 CIR5→NI3 | 3.9831 | 8.2282 | 11.2051 | 14.4966 |
| 保留 LoRA、重置 SID | 3.4327 | 8.0729 | 10.6330 | 14.1928 |
| 保留 SID、重置 LoRA | 3.2244 | 7.8339 | 10.6177 | 13.7973 |

全部精确值与百分点差见 [comparison.csv](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/comparison.csv)。

### 4.3 NI前后功能诊断

除Top1列给出k/108外，其余数值均为百分比。历史重建是87个支持类的宏平均，目标总数512，不能解释成同一个全局比例。

| 条件 | 阶段 | 名称→SID Top1 | 名称 R@10 | 重建 R@10 | 重建 R@20 | 重建 NDCG@10 | 重建 NDCG@20 |
|---|---|---|---|---|---|---|---|
| 完整继承 CIR5→NI3 | NI前 | 43/108 (39.81%) | 92.59 | 21.53 | 38.11 | 10.87 | 15.04 |
| 完整继承 CIR5→NI3 | NI后 | 34/108 (31.48%) | 93.52 | 19.98 | 33.44 | 11.05 | 14.42 |
| 保留 LoRA、重置 SID | NI前 | 35/108 (32.41%) | 75.93 | 14.25 | 23.71 | 7.39 | 9.72 |
| 保留 LoRA、重置 SID | NI后 | 41/108 (37.96%) | 92.59 | 19.20 | 31.36 | 10.10 | 13.12 |
| 保留 SID、重置 LoRA | NI前 | 67/108 (62.04%) | 99.07 | 9.46 | 20.42 | 3.69 | 6.47 |
| 保留 SID、重置 LoRA | NI后 | 45/108 (41.67%) | 91.67 | 17.65 | 29.85 | 9.47 | 12.49 |

NI前，保留SID组的直接名称映射最高（Top1为67/108），但历史重建Recall@10只有9.46%；保留LoRA组对应35/108和14.25%；完整CIR5为43/108和21.53%。NI后，完整继承在名称Top1上的34/108仍低于两个局部重置组，但其正式NI四项主指标最高。**直接名称映射更准，并不与本轮NI排序更好一一对应。**

NI训练后，两种局部重置组的历史重建Recall@10均上升，而完整继承由21.53%降至19.98%。这说明NI本身会改变诊断能力，不能把NI终点差异简单视为某块原有知识的份额。较合适的后续假说是检验LoRA与SID共同适配后的相容性，以及历史条件信息如何影响输出空间；本轮尚不能区分相容性损失、迁移冗余或NI重新学习等解释，不将其包装成已证实的新机制。

全部探针@1/@5/@10/@20和样本量见 [diagnostics.csv](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/diagnostics.csv)。

### 4.4 与完整继承逐目标配对

每个目标在相同impression下比较Top20中的排名，Top20外统一视为21；相同并不意味着完整分布相同。每行总计37,728个目标出现，不能与按类别等权的主指标互相替代。

| 局部重置相对完整继承 | 目标排名改善 | 目标排名下降 | Top20分辨率下相同 |
|---|---|---|---|
| 保留 LoRA、重置 SID | 6018 | 6185 | 25525 |
| 保留 SID、重置 LoRA | 7975 | 8902 | 20851 |

所有配对输入、参考/新预测hash，以及逐类和宏平均差值已核对。差值见 [paired_deltas.csv](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/paired_deltas.csv)，逐类值见 [per_class_deltas.csv](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/per_class_deltas.csv)。没有补做bootstrap、多seed或新对照。

### 4.5 完成与验收

两组均完成2,025步/64,800曝光及各样本3次；全部373个张量与4条参考logits零误差重载。远端完整checkpoint文件校验、373个Adam状态及scheduler的2025步一致；Full Dev的覆盖、SID合法排名、分片、指标和配对差值全部通过。LoRA组8项探针、SID组4项探针均通过独立指标复算。指定整体验收脚本分别输出accepted，完成收据与阶段收据一致。

原生终态：LoRA组2026-09-23 20:45:20 UTC Completed；SID组2026-09-23 21:29:18 UTC Completed。未下载模型权重、optimizer、完整预测到Mac。两组完成后，半小时监控已删除。

## 5. 数据与产物路径

- [冻结方案](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/research/2026-09-23_cir_parameter_transfer_plan.md)；[运行与恢复记录](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-23/mind_cir_parameter_transfer.md)。
- [跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_parameter_transfer_20260923_v1.json)；[运行配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/amlt/mind_cir_parameter_transfer_0923_v2.yaml)；[冻结bundle manifest](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/bundle-v2/manifests/transfer_bundle.json)。
- [冻结NI训练表](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/bundle-v2/data/training/mind_cir_108_v1/ni.parquet)；[Full Dev grouped表](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/bundle-v2/data/training/mind_cir_108_v1/dev_full_grouped.parquet)；[108类SID映射](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/bundle-v2/data/training/mind_cir_108_v1/interest_token_mapping.parquet)。
- [名称探针](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/bundle-v2/data/diagnostics/cir_parameter_transfer_v1/sid_name.parquet)；[历史重建探针](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/bundle-v2/data/diagnostics/cir_parameter_transfer_v1/heldout_cir.parquet)；[原文示例及来源hash](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/examples.json)。
- [LoRA组整体验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/acceptance/keep_lora.json)；[SID组整体验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/acceptance/keep_sid.json)；[完整继承参照验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-four-20260918/acceptance/cir5_ni3.json)。
- [LoRA组紧凑读回目录](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-readback/mind-cir-param-transfer-108-0923-v2/qwen35-2b-keep-lora-ni3)；[SID组紧凑读回目录](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-readback/mind-cir-param-transfer-108-0923-v2/qwen35-2b-keep-sid-ni3)，包含完成收据、阶段、分片/合并指标、探针及配对差值。
- [LoRA原生终态及最终来源核验](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/progress/20260923T210122Z/keep_lora/final_completion_verification.json)；[SID原生终态及最终来源核验](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/progress/20260923T213122Z/keep_sid/final_completion_verification.json)。
- [机器可读比较](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/comparison.json)；[逐目标方向计数CSV](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/paired_directions.csv)；[最终汇总验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final_acceptance.json)；[交付文件hash](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-parameter-20260923/final-report/delivery_sha256.json)。

远端均为Azure账号`csnewsandfeeds4150361735`、容器`amulet`。当前实验`mind-cir-param-transfer-108-0923-v2`；LoRA job `qwen35-2b-keep-lora-ni3`，结果前缀`projects/sft4safety/amlt-results/7209875533.62677-7997be68-62f4`；SID job `qwen35-2b-keep-sid-ni3`，结果前缀`projects/sft4safety/amlt-results/7209875533.62651-b3ae217f-0140`。最终权重在各前缀`formal/ni3/final_adapter/`，检查点在`formal/checkpoints-ni3/`，完整预测在`formal/real_eval/`及`diagnostics/`。报告中的文件hash与目录hash分开保存，不混用。
