# Token125 NI＋双向语义对齐：最终结果

**联合训练、Full Dev及远端配对分析全部独立验收。本次单seed观察到部分指标小幅改善，但四项95%用户配对区间均跨0，尚不能确认这份双向辅助对齐配方带来明确收益。**

## 指标

| 条件 | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Token125 NI-only | 19.8730% | 10.0074% | 33.4679% | 13.4279% |
| Token125 NI＋双向辅助对齐 | 20.0887% | 10.1856% | 33.3820% | 13.5229% |

| 指标 | 联合训练−基线（百分点） | 用户配对95%区间（百分点） |
|---|---:|---:|
| R@10 | +0.2156 | [-0.2789, +0.6560] |
| NDCG@10 | +0.1782 | [-0.0334, +0.3764] |
| R@20 | -0.0858 | [-0.8661, +0.6606] |
| NDCG@20 | +0.0950 | [-0.1294, +0.3209] |

R@10提高0.2156个百分点、NDCG@10提高0.1782个百分点；R@20下降0.0858个百分点。不能概括为各项均提升，也不能因为区间跨0就证明没有效果或两组等价。目前不足以据此替换NI-only默认配方。

配对分析使用同一批27,082用户、31,392 impressions、37,728目标出现和104支持类别；按用户抽取共享Exp(1)权重，2,000次Bayesian bootstrap、seed42，每次重新计算类别分母与macro平均。区间仅刻画固定模型下的用户采样不确定性，不包含训练seed不确定性。

## 实验配方与边界

| 任务 | 每epoch行数 | 三轮曝光 | 实际监督token占比 |
|---|---:|---:|---:|
| NI | 21,600 | 64,800 | 84.6948% |
| 兴趣名称→ID | 1,250 | 3,750 | 4.9013% |
| ID→完整兴趣名称 | 1,250 | 3,750 | 10.4039% |

两个辅助方向覆盖全部125个类别，250条基础配对各复制10次/epoch，总曝光30次/基础对。联合组每轮24,100行、3epochs、2,262次更新、72,300次总曝光；基线是已接受Token125 NI-only的2,025次更新。NI样本和每条3次曝光不变，但新增辅助任务使更新次数增加237、LR horizon改变，并改变NI在混合批次中的位置，因此比较的是整份联合配方，不能单独归因于语义对齐机制或排除额外计算量效应。

两组使用同一Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`、seed42、125-token词表与mean_norm_matched初始化。373个可训练张量冷状态SHA同为 `28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565`。联合组从原始状态冷启动，未继续微调基线，未继承门禁权重。

NI输入保持`[<interest_NNN>] Headline`，标题、顺序、重复、指令和监督目标逐字段不变。所有辅助类别名来自冻结映射，不包含用户数据、Dev标签或新生成文本。原108类仍是NI和Dev输出候选；新增17类仅在辅助映射任务中也可作为目标。

LoRA r16/alpha32/dropout0.05/all-linear，lr2e-5、cosine、warmup0.03、weight decay0.01；4×A10080GB Recall Standard、microbatch2、accumulation4、global32、BF16、显式Torch fallback、非reentrant checkpointing。损失为原生全词表completion-only token_mean CE，每个监督token单位权重，无额外task系数。两个方向样本数相同但文本回答较长，故实际token份额不相等。无CIR、MeRA正则、多seed或比例搜索。

## 参考与实现

参考[OpenOneRec公开源码](https://github.com/Kuaishou-OneRec/OpenOneRec/tree/a969edcadd579a06c1966ae1db5984e02f48beff)的Itemic-Text Alignment与多任务SFT思路。已核对的`data/onerec_data/sft/item_understand.py`明确是SID→caption；本轮双向兴趣问答及混合比例是用户指定方向下的适配，不声称复现OpenOneRec全部配方。旧108类先对齐再NI的warm-up实验与本轮125类联合训练不同。

混合数据采用唯一训练sample_id和共享base_alignment_id。答案长度由真实tokenizer逐行核对并写入冻结合同；监督计数、loss分母与resume检查按实际答案长度验证，未继续假设每个答案恒为3 token。Trainer、数值CE、初始化、持久化、重载、评估保留已接受运行时；仅调整编排、任务枚举和对应计数审计。

## 验收

125行有限非零梯度与258条混合代表样本、单卡8步下降、四卡2步、373张量及logits精确reload、64条真实Dev独立重算均通过。最终各任务曝光64,800/3,750/3,750、逐行3次和每基础辅助对30次全部符合预算；最终373张量与4条样本logits精确重载（atol=rtol=0）。

Full Dev覆盖完整、108候选合法无重复Top20、无历史类别过滤；worker对完整预测/输入身份/分片/指标独立核验，本地又从紧凑分片证据重算各类与macro指标。最终CPU任务再次读取两组完整预测，并重新哈希最终checkpoint全部文件：

- 基线：13文件，205,935,514字节，step2025。
- 联合组：13文件，206,002,718字节，step2262。

CPU分析backend `frank_diamond_qgqmggjsjy` 原生Completed；完整checkpoint及预测哈希复核、同用户配对指标与区间结果通过独立接受。日志或Blob只读超时仅重试观测，没有重新训练。大型权重、optimizer和完整预测始终留远端。

## 产物

- [指标CSV](mind_token_joint_alignment_results.csv)、[配对差值CSV](mind_token_joint_alignment_paired_differences.csv)、[完整运行记录](mind_token_joint_alignment_runlog.md)。指标CSV为0–1比例，差值CSV为百分点。
- 总入口：`manifests/mind_token_joint_alignment_20261001_v1.json`；恢复指针：`manifests/mind_token_joint_alignment_restoration_20261002_v1.json`。
- 联合训练接受：`artifacts/mind-token-joint-alignment-20261001/final_acceptance.json`。
- 配对接受：同目录`paired_final_acceptance.json`；分析：`comparison.json`、`analysis_receipt.json`。
- comparison SHA-256：`2a39773601b03e8f74cd7ced8b7d7023b8bf2b17f00aa4aaf60f33480e8311fb`。

全部工作已完成，本聊天监控`ni-token`已暂停；不自动新增实验。
