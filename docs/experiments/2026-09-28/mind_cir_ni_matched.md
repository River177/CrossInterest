# CIR 与额外 NI：同用户、同目标、长度匹配实验

## 范围与当前状态

用户批准仅新增两组：历史前缀 NI、长度匹配 CIR；各辅助训练5轮后接原NI3。不重训普通CIR，不增加第三条件、额外seed、个人meta或新模型结构。当前状态及远端身份以 [跟踪清单](../../../manifests/mind_cir_ni_matched_20260928_v1.json) 为准。

两组训练与Full Dev评估均已完成，平台终态为Completed、退出码0，分别通过 `accept_cir_final.py` 独立紧凑回读。完整预测已在gcr-a100的CPU上独立重算，全部指标与已验收指标一致（误差小于1e-12），2,000次用户成对Bayesian bootstrap完成。主指标Macro R@10上，长度匹配CIR低于历史前缀NI；本次对照不支持“CIR相对等量前缀NI监督具有主指标优势”的论断。

第二组AMLT CLI在提交后写入Azure Table登记表时遇到300秒读取超时，导致CLI退出1且AMLT列表暂时缺少该作业；原生作业、代码prefix及结果prefix均已独立核实，未重复提交。监控直接依据冻结的原生backend ID与结果路径，不因AMLT登记缺失而重训。

本聊天自动跟进为 `cir-ni-2`；两组及最终比较已验收，自动跟进已停止（PAUSED），保留聊天。

## 冻结设计

| 条件 | 辅助样本 | 辅助预算 | 后续NI预算 |
|---|---:|---|---|
| ni_prefix | 9,431 | 5轮、47,155曝光、1,475更新 | 21,600条、3轮、64,800曝光、2,025更新 |
| cir_length_matched | 9,431 | 同上 | 同上 |

同一用户、辅助目标、SID、样本顺序和每条行为数逐条匹配，覆盖108类；原始21,600条NI、Dev和SID mapping逐字节保持不变。历史前缀截到目标类别第一次出现之前，并用完整原始H0确认目标此前未出现。长度匹配CIR从H50删除该类全部行为后，按固定hash选取相同行为数，保留时间顺序。没有用原未来NI标签选择视图。

原17,271对中，989对在H50之前已出现目标，6,851对的合法H50前缀少于5条，其余9,431对全部保留。两组有1,088对输入完全相同，按预先规则保留，不事后剔除。

使用共同的中性辅助提示词，完整模板见 [builder](../../../src/new_interest/data/build_cir_ni_matched.py)。后续NI及评估提示词保持原样。实验只检验这种匹配构造的相对价值，不把旧普通CIR的21.9717%与新子集作严格因果比较。

## 资源与执行合同

- Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；seed42。
- 两组各4×A10080GB、Recall Standard，最多8卡并行；有效batch32、每卡microbatch2、累积4。
- 原生全词表completion-only token-mean CE，all-linear LoRA r16/alpha32/dropout0.05及tied SID行；AdamW lr2e-5，weight decay0.01，cosine、warmup0.03；非reentrant checkpointing，显式Torch fallback。
- 从同一原始初始化开始，预期373张量状态摘要 `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`。
- 阶段交接继承全部LoRA/SID，重置optimizer/scheduler；每100步持久化可恢复状态，保留精确样本游标和RNG。权重、优化器及完整预测均保留远端。
- 新提示词和数据先通过单卡及四卡训练、精确重载和64条真实评价门禁；正式训练从原始初始化重新开始，不继承门禁更新。

## 本地审计

| 项目 | ni_prefix | cir_length_matched |
|---|---:|---:|
| 一轮输入token总数 | 4,317,844 | 4,328,894 |
| 每样本平均输入token | 457.8352 | 459.0069 |
| 最长输入token | 1,464 | 1,464 |
| 每样本监督token | 3 | 3 |
| 超过2,048 | 0 | 0 |

输入token总量比为1.002559，差0.256%；不声称严格相同FLOPs。原NI最长1,564 tokens，也无截断。每个上传包113个文件，均经过实际AMLT staging与冻结manifest比对，训练器、sampler、checkpoint和evaluator沿用已验收字节。

相关测试32 passed；独立成对指标分析测试3 passed；观测器缓存校验测试1 passed。完整本地pytest在收集阶段因缺少peft停止，不能报告全套通过。GPU环境使用已验收的精确依赖，真实启动门禁已通过，见下文。

证据：

- [数据独立回读](../../../artifacts/mind-cir-ni-matched-20260928/data_independent_readback.json)
- [token审计](../../../artifacts/mind-cir-ni-matched-20260928/token_audit.json)
- [运行包预检](../../../artifacts/mind-cir-ni-matched-20260928/local_preflight.json)
- [真实上传边界审计](../../../artifacts/mind-cir-ni-matched-20260928/upload_audit.json)

## 启动验收（2026-09-28 12:55 UTC）

两组各自完成辅助阶段及NI阶段的单卡8步、四卡2步门禁；373个adapter张量及重载logits均精确一致（atol=rtol=0）。每组门禁评价覆盖64条impressions、75个目标、36个支持类，四分片统计已对照冻结Dev独立重算全部macro指标。

正式训练的初始参数SHA均为 `f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`，并使用全新optimizer，未继承门禁更新。此处记录启动正确性；正式训练、阶段交接及Full Dev的后续验收均已完成，见下文。

- [NI-prefix启动验收收据](../../../artifacts/mind-cir-ni-matched-20260928/ni_prefix/progress/20260928T125509Z/startup_acceptance.json)
- [长度匹配CIR启动验收收据](../../../artifacts/mind-cir-ni-matched-20260928/cir_length_matched/progress/20260928T125509Z/startup_acceptance.json)

观测器对未变化的紧凑产物仅在backend、结果prefix、云端ETag、文件大小及本地SHA均匹配时复用缓存，并记录来源；本次未改变远端训练包或数值协议。

最终验收收据：[NI-prefix](../../../artifacts/mind-cir-ni-matched-20260928/ni_prefix/final_acceptance.json)、[长度匹配CIR](../../../artifacts/mind-cir-ni-matched-20260928/cir_length_matched/final_acceptance.json)。两组均未重启或重复提交远端训练作业。

## 辅助训练到NI交接

两组阶段交接收据均已通过独立交叉核对：[NI-prefix](../../../artifacts/mind-cir-ni-matched-20260928/ni_prefix/progress/20260928T175448Z/stage_handoff_readback.json)、[长度匹配CIR](../../../artifacts/mind-cir-ni-matched-20260928/cir_length_matched/progress/20260928T175448Z/stage_handoff_readback.json)。已核对全部样本曝光次数、最终辅助checkpoint、辅助阶段精确重载、NI输入adapter文件hash、初始373张量摘要和fresh optimizer/scheduler。父阶段张量由冻结worker代码在远端读取并逐项比较；本地独立交叉核对紧凑收据，不下载权重。

## NI训练完成与评估

两组NI训练完成收据均已通过：[NI-prefix](../../../artifacts/mind-cir-ni-matched-20260928/ni_prefix/progress/20260929T005449Z/ni_completion_readback.json)、[长度匹配CIR](../../../artifacts/mind-cir-ni-matched-20260928/cir_length_matched/progress/20260929T012449Z/ni_completion_readback.json)。两组均已进一步完成平台终态确认与Full Dev独立紧凑验收。完整预测及所有大模型产物保留远端。通用验收器输出中的 `CIR_to_NI` 字段是历史实现标签；在本报告中该组应解释为历史前缀NI5→NI3，不是普通CIR。

## 结果验收标准

固定每阶段末尾checkpoint，Full Dev31,392 impressions、27,082用户、37,728目标出现、104支持类，固定108类合法无重复Top20、不做历史类别过滤。主指标Macro R@10，辅以R@20与NDCG@10/@20。

独立核对实际有效曝光、checkpoint身份、参数交接、fresh reload、四分片覆盖及指标重算。两组完成后在远端读取完整预测，计算CIR减NI的用户成对Bayesian bootstrap区间（2,000次、seed42）；该区间不代表跨训练seed不确定性。AMLT Running/Completed本身不作为实验验收。

## 最终结果与成对比较

数值为百分比，差值为“长度匹配CIR − 历史前缀NI”的百分点。两组均为辅助5轮→原NI3、seed42。

| 指标 | 历史前缀NI | 长度匹配CIR | 差值（百分点） | 用户成对95%区间（百分点） |
|---|---:|---:|---:|---|
| Macro R@10 | 21.2238 | 20.3668 | -0.8570 | [-1.4962, -0.2892] |
| Macro R@20 | 34.5649 | 34.6897 | +0.1247 | [-0.4071, +0.7137] |
| Macro NDCG@10 | 10.8744 | 10.9051 | +0.0307 | [-0.3352, +0.4016] |
| Macro NDCG@20 | 14.2236 | 14.5266 | +0.3030 | [+0.0085, +0.6193] |

主指标R@10差值为−0.8570个百分点，用户成对区间全为负。R@20与NDCG@10区间包含零；NDCG@20差值为+0.3030个百分点，未校正的逐指标区间略高于零。因此不能概括为CIR整体优于前缀NI，也不能据此宣称已排除“增加NI监督”的解释。该实验也不证明普通CIR等价于数据增广：它比较的是本次9,431对、共同辅助提示词、长度匹配的两个构造，不能直接归因旧普通CIR结果。

不确定性采用同一用户共享Exp(1)权重、逐次重算类内分母再macro平均的2,000次Bayesian bootstrap（seed42）。区间只刻画固定两模型在评估用户层面的不确定性，不包含训练seed变化，也未做多指标校正。108类候选、104个有测试支持的类别、31,392条impressions、27,082用户、37,728目标出现全部核对一致，合法无重复Top20逐条验证通过。

证据：[完整预测远端重算与bootstrap](../../../artifacts/mind-cir-ni-matched-20260928/comparison.json)、[远端环境与下载hash收据](../../../artifacts/mind-cir-ni-matched-20260928/analysis_receipt.json)、[结果CSV](mind_cir_ni_matched_results.csv)。分析环境：gcr-a100，Python3.12.3、NumPy2.4.4、PyArrow24.0.0，CPU；分析代码SHA在结果文件中固定。完整预测始终保留远端，Mac只接收紧凑结果和收据。恢复位置：两组原始AMLT results prefix见manifest；分析代码、结果及验收收据持久化位置见 [存储回读](../../../artifacts/mind-cir-ni-matched-20260928/analysis_persistence.json)。GCR上的分析目录为 `/home/v-wuhao1/cir-ni-matched-analysis-20260929-v1`，是可重建临时副本。

## 已知解释边界

两组输入内容不同是构造策略本身的差异；即使长度匹配，也不能称为输入信息完全相同。本次主指标未观察到长度匹配CIR的优势，应据此限制论文主张；未增加条件、seed或sweep追求正结果。

最终联合验收：[两组与成对比较验收清单](../../../artifacts/mind-cir-ni-matched-20260928/paired_final_acceptance.json)。
