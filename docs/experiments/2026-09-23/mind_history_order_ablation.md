# MIND NI-only 历史顺序消融：最终验收报告

2026-09-23 21:04 UTC 已通过完整产物验收。按[已批准方案](../../research/2026-09-23_history_order_ablation_plan.md)，复用原A，完成一次新训练和B/C/D三次Full Dev评估。原提示词逐字保留，包括 `in chronological order`。大权重、优化器和完整预测保留远端。全部验收后已删除 `ni-only` 自动跟进，保留本任务未归档。

本次固定排列下，Full Dev主指标Recall@10没有显示原始顺序的明确优势；但约28%的记录会因打乱而改变Top1预测。打乱训练也没有消除这种预测敏感性。这是给定模型、seed42及一次固定排列的结果，不能据此证明顺序无用或四组等价。

## 实验结果

A/B使用同一个原NI-only模型，C/D使用同一个新训练模型；B/D共用同一份打乱Dev。每组31,392条impressions、27,082用户、37,728目标出现，104支持类、108候选SID、合法无重复Top20、无历史类别过滤。以下均为interest-macro指标。

| 组 | 训练→评估历史 | Macro R@10 (%) | Macro R@20 (%) | Macro NDCG@10 (%) | Macro NDCG@20 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 原始→原始 | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| B | 原始→打乱 | 20.1622 | 33.4186 | 10.6675 | 13.9747 |
| C | 打乱→原始 | 20.3258 | 33.8395 | 10.3544 | 13.7363 |
| D | 打乱→打乱 | 20.2758 | 33.3891 | 10.6520 | 13.9222 |

完整@1/@5/@10/@20结果（%）：

| 组 | R@1 | R@5 | R@10 | R@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 2.8761 | 12.3181 | 20.3275 | 33.6553 | 2.8761 | 7.5680 | 10.1325 | 13.4663 |
| B | 3.7302 | 12.9228 | 20.1622 | 33.4186 | 3.7302 | 8.3491 | 10.6675 | 13.9747 |
| C | 3.4522 | 12.5873 | 20.3258 | 33.8395 | 3.4522 | 7.8696 | 10.3544 | 13.7363 |
| D | 3.2305 | 12.8017 | 20.2758 | 33.3891 | 3.2305 | 8.2539 | 10.6520 | 13.9222 |

## 固定比较与用户成对区间

差值和区间单位均为百分点。2,000次用户成对Bayesian bootstrap，seed42；每次用户权重为正Exp(1)，同一用户的全部impressions和所有组共用权重，每次在固定104类上计算宏平均。区间只涵盖评估用户不确定性，不涵盖训练seed或排列不确定性，也未作多比较整体显著性声明。

| 比较 | ΔR@10 [95%区间] | ΔR@20 [95%区间] | ΔNDCG@10 [95%区间] | ΔNDCG@20 [95%区间] |
| --- | ---: | ---: | ---: | ---: |
| B-A | -0.1652 [-0.8967, +0.5986] | -0.2367 [-0.8066, +0.2957] | +0.5351 [+0.1424, +0.9077] | +0.5084 [+0.1540, +0.8414] |
| D-A | -0.0516 [-0.7117, +0.6417] | -0.2662 [-1.0306, +0.4494] | +0.5196 [+0.1563, +0.9067] | +0.4560 [+0.1203, +0.8096] |
| C-A | -0.0016 [-0.6077, +0.6030] | +0.1842 [-0.3092, +0.7540] | +0.2220 [-0.0981, +0.5710] | +0.2701 [-0.0223, +0.5702] |
| D-B | +0.1136 [-0.4799, +0.6696] | -0.0295 [-0.6479, +0.5719] | -0.0155 [-0.3428, +0.3125] | -0.0524 [-0.3692, +0.2465] |
| D-C | -0.0500 [-0.4354, +0.3331] | -0.4504 [-1.1294, +0.1476] | +0.2976 [-0.0769, +0.6698] | +0.1859 [-0.2101, +0.5714] |

B−A衡量原模型的输入排列敏感性；D−A比较各自训练／评估顺序匹配的表现；C−A比较打乱训练在原Dev上的表现；D−B比较打乱训练在相同打乱Dev上的影响；D−C衡量新模型的输入排列敏感性。主指标R@10的五个区间均包含零，不能据此宣称等价。

B−A和D−A的NDCG@10、NDCG@20区间为正，而Recall点估计略低，表明这些条件下按该指标衡量的排名折扣收益较高，未显示原始排列的一致优势。D−B的四项区间均包含零，因此没有明确证据说明打乱训练本身带来恢复或额外收益；不能把D相对A的NDCG差异全归因于重新训练。

## 个体预测的排列敏感性

| 同模型比较 | Top1改变记录 / 全部记录 | Top1改变比例 | Top10平均集合重合比例 |
| --- | ---: | ---: | ---: |
| A/B | 8,916 / 31,392 | 28.4021% | 82.9527% |
| C/D | 8,805 / 31,392 | 28.0485% | 82.8644% |

比例的分母是impressions，不是独立用户。Top10重合比例为两组Top10集合交集大小除以10，再对记录平均。两种模型的Top1变化比例均约28%，Top10集合重合约83%：总体Recall接近，同时大量个体排名发生变化。这里只描述这两对点估计，未额外检验两种模型敏感性是否等价。

## 预先固定的历史长度≥2子集

该子集30,538条记录、36,701目标出现、104支持类，与全量使用相同支持类集合。只对已有预测筛选统计，没有新增推理；以下没有额外bootstrap区间。

| 组 | 训练→评估历史 | Macro R@10 (%) | Macro R@20 (%) | Macro NDCG@10 (%) | Macro NDCG@20 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 原始→原始 | 21.0720 | 34.4777 | 10.4189 | 13.7729 |
| B | 原始→打乱 | 20.4226 | 34.2556 | 10.8207 | 14.2656 |
| C | 打乱→原始 | 20.5888 | 34.6747 | 10.5079 | 14.0327 |
| D | 打乱→打乱 | 20.5314 | 34.2248 | 10.8025 | 14.2091 |

该子集B−A、C−A、D−A的R@10分别为-0.6494、-0.4831、-0.5406个百分点，比Full Dev的对应差值更负。不能用全量主指标接近抹去这个子集差异，也不能把全量区间直接套用到子集。宏平均按每类目标支持重新归一化，去掉短历史后变化不等于简单按记录数稀释。

## 研究问题、输入和训练合同

问题是：在历史内容和频次不变时，原记录排列对当前NI-only的新兴趣预测是否有价值，以及打乱训练能否补偿已有模型的排列敏感性。未增加CIR、OI、其他排列、seed、训练smoke、模型或预算。

原数据为 `sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1`，新数据为 `mind_history_order_20260923_v1`。Train 21,600用户／21,600条，108类×200；Train与Dev用户不交叉。对每个用户的整条 `[Topic category] Headline` 按冻结SHA-256规则排列，类别与标题一起移动，保留重复条目、长度、标签、行顺序、映射和提示词。Train每用户一份排列，三轮重用；同一Dev用户同一历史共享排列，B/D逐字复用。

| 输入审计 | Train | Dev |
| --- | ---: | ---: |
| 记录数 | 21,600 | 31,392 |
| 实际prompt改变 | 20,728 | 28,930 |
| 长度为1 | 323 | 854 |
| 长度≥2但抽中原排列 | 541 | 1,588 |
| 非原排列但因重复而文本未变 | 8 | 20 |

模型为Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，Behavior+Interest→Semantic-ID；native full-vocabulary completion-only token_mean CE。新训练与A使用相同冷启动状态 `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`，没有从训练后的A继续微调。

| 训练项目 | 冻结设置 |
| --- | --- |
| 预算 | seed42；3 epochs；2,025 updates；64,800次曝光；最终epoch，未按Dev选点 |
| 优化器和调度 | AdamW fused；lr2e-5；weight decay0.01；cosine horizon2025；warmup0.03；max grad norm1 |
| 可训练参数 | all-linear LoRA r16/alpha32/dropout0.05，加tied SID行 |
| 编码和精度 | enable_thinking=false；BF16；非reentrant checkpointing；max_seq_length2048，无截断 |
| 实际token审计 | 最大1,564 tokens；所有completion为3 tokens；超长0 |
| 更新1–500 | 单GPU；microbatch1，累积32，Torch fallback |
| 更新501–2025 | 4×A10080GB；每卡microbatch8，累积1；FLA0.5.2和causal-conv1d1.7.0 |
| 全程batch和迁移 | global batch32；保留Adam、scheduler、RNG及样本游标 |
| 评估 | 每组4个逻辑分片×7,848条，batch8；108个SID float32评分，Top20 |

指标定义沿用原A：每个目标出现计TopK命中或 `命中/log2(rank+1)` 折扣收益，先在每个兴趣类别内除以该类目标出现次数，再对有支持的104类等权平均。这里的interest-macro NDCG不是impression-macro NDCG，不混用二者。

## 验收证据与边界

训练恰好2,025步／64,800曝光。独立CPU重建完整样本序列，最终前缀为 `fbc553db6b7476322378022f404a17e55ebebd4258f16572ab9801620420f01e`，与冻结合同相同；4个rank实际微批次顺序守卫通过。500步迁移的参数、Adam、scheduler和四rank RNG已独立核验。

最终adapter包含373个张量、68,212,000 bytes，文件SHA-256为 `e196bf5341a7223035b4e3e19c0977a08332863e49d2eedfa9aafd3d17e811ec`。远端完整权重检查和新进程fresh reload通过：全部张量逐项一致，4条参考输入×108 SID logits在atol=rtol=0下完全一致，原数值路径保持一致。

B/C/D每组完整预测均由远端冻结reader检查覆盖、候选、Top20合法性和指标；独立分析进程读取A/B/C/D完整预测，核对逐impression配对，重算全部宏指标、长度子集、排名变化和用户成对分析。本地重新核对四组紧凑指标及哈希；原A仍使用同一份已验收预测，未重新训练或预测。

最终持久manifest的506个文件、221,171,076 bytes均已核对远端存在和大小，worker已在远端逐文件回读SHA-256；完成标记、manifest和comparison再次以ETag约束读取并匹配。本地未下载权重、优化器或完整预测。最终checkpoint2025提交ETag保持不变；保留1900/2000/2025及单卡阶段300/400/500作为恢复入口。

原提示词仍宣称 `in chronological order`，S条件与该语句不符，所以本轮衡量的是原NI-only接口下的排列敏感性及打乱训练表现，不是中性提示词下纯时间信息消融。原始顺序仅指冻结数据记录顺序；没有额外验证点击时间，也未区分最近行为位置、相邻转移与长程顺序。一次seed／一次固定排列不足以确证稳定收益或机制，用户区间不补足这层不确定性。

## 产物与恢复入口

- [冻结方案](../../research/2026-09-23_history_order_ablation_plan.md)、[运行清单](../../../manifests/mind_history_order_20260923_v1.json)、[历史阶段记录](mind_history_order_ablation_runlog.md)。
- [最终独立回读收据](../../../artifacts/mind-history-order-20260923/final_independent_readback.json)、[训练和重载收据](../../../artifacts/mind-history-order-20260923/training_independent_readback.json)、[完整comparison](../../../artifacts/mind-history-order-20260923/progress/check-2103-durable/comparison.json)、[远端产物manifest](../../../artifacts/mind-history-order-20260923/progress/check-2103-durable/artifact_manifest.json)。
- [冻结输入与代码manifest](../../../artifacts/mind-history-order-20260923/bundle-v1/manifests/history_order_bundle.json)、[冻结独立分析reader](../../../artifacts/mind-history-order-20260923/bundle-v1/src/new_interest/experiments/history_order_analysis.py)、[紧凑回读reader](../../../scripts/analysis/read_history_order_state.py)。
- 输入目录：`data/training/mind_history_order_20260923_v1`；原输入目录与模型身份见上文及冻结方案。
- AMLT实验 `mind-ni-history-order-0923-v1`，job `qwen35-2b-ni-history-order`，backend `ivory_deer_r3cm1cz1rb`，Recall Standard、80G4-A100；[作业入口](https://aka.ms/amlt?q=k3x7k)。
- 新产物远端：`amulet/projects/sft4safety/amlt-results/7209875909.09490-48dfeba1-32f9`，账号 `csnewsandfeeds4150361735`。A远端：`amulet/projects/sft4safety/amlt-results/7210843083.71371-a21f4d1c-d098`。
- 新模型：上述新产物根下 `final/sft/final_adapter/`；优化器与恢复状态在 `final/checkpoints/checkpoint-2025/`；B/C/D完整预测分别在 `evaluations/<组>/real_eval/merged/id_token_predictions.parquet`。
- manifest SHA-256 `0eae6f248467bd55a88a3f85799eb2678dc2f4ee480e96e0312f0b1d428b022a`；comparison SHA-256 `eeab9a4b4754ad08778ca4598fb2ab5ed1ace28040a1d634ef630c198a4aca8b`；完成标记SHA-256 `cd6c5a22d974936834d4c2cadaeda99843abe066572ce71f2198e0fccca54982`。

| 组 | 完整预测 SHA-256 | metrics SHA-256 |
| --- | --- | --- |
| A | `c07251bb82843c9caf35a0d6d8d58a57295bd8f4ddb4006e397a6ec1e22dbac6` | `7c96deb861af4439df832e0b7836f3e899bf32fe56936494905995530d3dd735` |
| B | `8c34076d8ce0e4872bc3dc35602c85918713f7c9221937e199f9e22d13308948` | `102cda79e5f6e8f3ff0d70acedff126fcc1a2d3174612df5626f3e9961b33108` |
| C | `fccbcd32e1ba10c536806e7b763e38d0342be6dc7c0630008d2fb71e94ccd05a` | `ed2fb8185cd5f995aa768b3d50c315b3cc2df16ffa7c2f1ee0bb158b1c74472f` |
| D | `ebf9aff051a18aeda0415c17010f739645b16dbdecd7dcd03ebd9d68ffef9ce5` | `f8b5a7195924eb3ae071fd61fcd129a02bb4bb1dc590ff596fa7cfc66ee33c47` |
