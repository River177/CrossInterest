# CIR参数迁移机制：两组初始化干预实验

状态：本方案两组新增实验已完整独立验收（2026-09-24，北京时间），未扩充训练条件。见 [最终报告](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-23/mind_cir_parameter_transfer_results.md) 和 [运行记录](../experiments/2026-09-23/mind_cir_parameter_transfer.md)。

目标是判断CIR产生的哪一部分参数初始化有助于后续NI训练，为方法设计提供证据。它不是“知识存储位置”的完全定位，也不能单独证明模型学到了跨兴趣关联。

## 1. 第一轮仅新增两组NI训练

固定原始初始化为 `(L0, E0)`，已接受CIR5终点为 `(LC, EC)`。L是全部372个LoRA张量，E是108×2048的SID可训练行张量；预训练基座保持不变。

| 条件 | NI开始时LoRA | NI开始时SID | 本轮工作 |
|---|---|---|---|
| 完整CIR5→NI3 | LC | EC | 复用已验收结果 |
| 保留LoRA、重置SID | LC | E0 | 新训练NI3 |
| 保留SID、重置LoRA | L0 | EC | 新训练NI3 |
| 历史NI-only | L0 | E0 | 只作背景参考，不重训 |

两组NI训练中，LoRA和SID都正常参与更新。这里改变的是NI的初始化，不改变可训练参数集合。不要把“保留一部分”实现成只训练该部分。

主要对照分别为“完整继承”与两种“局部重置”。历史NI-only有单卡转四卡／FLA执行差异，因此这一轮不是严格的完整2×2析因实验；不据此计算LoRA、SID的贡献百分比或交互效应。若后来需要这样的结论，须单独讨论是否补齐完全匹配的冷启动对照，本轮不自动扩组。

## 2. 初始化必须有逐张量来源证据

复用模型Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，相同108类SID映射和原数值执行路径。

- CIR5最终adapter文件SHA256：`1201a6e750cf65ca57940613df177a8caab14b594827f4496f84c7643f283df9`。
- 原始373张量状态manifest摘要：`f924e8c76d9d28be55b8795920ca18811345dfde3b81ded85d5c039615a0510e`。
- CIR5后的373张量状态manifest摘要：`dc57ec586aef1bf6fcc573ffbbc5fc5e441e551b69e7ce3b4685b7ea828e6e17`。
- SID张量名：`base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta`。
- E0张量字节SHA256：`1f026620aa08a9d0d5eb4e8723c7808fe76f0232bca309b7502880943e452d0c`。
- EC张量字节SHA256：`513281a16938afd989fb5527a93eaebfc3949c20d5ea7cf61ba2373f6b90e160`。

这些摘要来自不同层次，文件SHA与状态manifest摘要不可直接比较。原始初始化已有逐张量哈希记录；若需重新构建L0/E0，应使用冻结初始化代码并先逐项匹配这些记录，不能只声称使用了相同seed。

SID恢复的是原始语义初始化值，不是全零或重新随机初始化。LoRA需恢复全部A/B矩阵，不仅将B置零。保留输入／输出embedding的既有共享关系，不额外改动输出head。

为两种组合分别保存新adapter及组成收据：每个张量记录来源、shape、dtype、SHA256。复制得到的新文件不覆盖原CIR5；保存后新进程重载，检查实际载入张量等于预期组合，且同输入logits精确重载。若重置后预测或loss变差，应如实记录；这不是修改初始化或放宽实验定义的理由。

组合初始化需要独立验收逻辑，不能把原“完整继承CIR5全部参数”的断言删除后继续沿用原阶段身份。

## 3. NI阶段保持相同合同

每组使用原21,600条NI、seed42、news/click提示词，训练3 epochs、2,025次更新、64,800次有效样本曝光。新optimizer/scheduler从零开始，cosine horizon固定2,025步，lr=2e-5、warmup=0.03、weight decay=0.01。LoRA r16/alpha32/dropout0.05，BF16、显式Torch fallback、非reentrant checkpointing，native full-vocabulary completion-only token-mean SFT。

建议每组继续4×A10080GB Recall Standard，microbatch2、accumulation4、global batch32；两个新增训练合计4,050次NI更新，不重新训练CIR。是否并发属于提交配置，不在本方案阶段分配资源。

两组采用相同数据顺序和检查点策略。每次恢复保留模型、Adam、scheduler、RNG、非零epoch／skip样本游标。所有权重、优化器及完整预测留远端，本地只保存紧凑收据和结果。

## 4. 评价与机制证据

正式结论固定使用NI第3轮终点。完整Dev仍为31,392 grouped impressions、37,728目标出现、104支持类、108类SID Top20、无历史过滤。四分片各7,848；远端完整预测合法性／指标重算，本地紧凑分片指标独立重算。报告全部@1/@5/@10/@20，主要比较仍为Recall/NDCG@10/@20。

完整继承参照的四项主要指标依次为21.9717%、35.0756%、11.2051%、14.4966%。分别报告两个局部重置条件相对参照的百分点差、逐类差异和预测对的变化，不根据新结果更换主要指标或挑选最好epoch。

如报告区间，应对相同用户进行成对重采样并重新计算宏平均差值；该区间只描述评估用户采样不确定性，不能代表单seed训练的稳定性。本轮不增加多seed。

建议保留NI前和NI后的两种小规模功能诊断，用于辅助解释，而不作为选择checkpoint的指标：

1. 类别名称→SID：检查直接的类别输出映射。它没有用户历史推断，不能将其提升当成个性化建模提升；名称模板需要固定且与正常NI评价分开。
2. 独立用户的历史类别重建：固定类别擦除规则，只根据历史选择目标，检查缺少目标直接证据时的重建能力。应单独构造、记录覆盖率和标签分布，不能在原CIR训练样本上用记忆表现代替泛化；也不能把该诊断分数混入正常NI指标。

这两种诊断不需要新增训练；具体样本清单应在实施前固定。它们与参数干预共同帮助区分输出映射与历史条件推断，仍不能自动排除所有通用预训练效应。

## 5. 预先约定如何解释

| 观察 | 可以推进的解释 | 不能直接声称 |
|---|---|---|
| 保留LoRA组接近完整继承，保留SID组明显更差 | 传递CIR的LoRA初始化对NI收益更关键，SID warm-start不是必要条件 | LoRA只包含兴趣关联知识；SID完全无用 |
| 保留SID组接近完整继承，保留LoRA组明显更差 | CIR的SID初始化是重要收益来源，应优先研究输出空间 | 所有收益都来自SID校准 |
| 两个局部重置都明显降低结果 | 两部分联合初始化有作用，或局部重置造成了相容性损失 | 已证明正交互或跨兴趣推理 |
| 两个局部重置都接近完整继承 | 存在迁移冗余，或NI训练能恢复缺失部分 | CIR没有作用；两部分各自解释100%收益 |
| 某局部重置反而更好 | CIR另一部分初始化可能限制NI适应，值得进一步定位 | 已证明该部分存储有害知识 |

“接近”应结合实际百分点差和不确定性讨论，不在看到结果后任意定义阈值。参数位置不等于语义机制；后续主方法必须针对被验证的功能变化设计，而不是给参数消融本身加一个新名字。

## 6. 已核实依据

- [四组总验收](../../artifacts/mind-cir-four-20260918/final_acceptance.json)
- [跟踪清单](../../manifests/mind_cir_four_108_20260918_v1.json)
- [原结果报告](../experiments/2026-09-18/mind_cir_four_conditions.md)
- [原始初始化记录](../../artifacts/mind-cir-four-20260918/final-readback/mind-cir-staged-recovery-0919-v1/recover-staged-qwen35-2b-cir5-ni3-ni5/formal/cir5/initial_adapter_state.json)
- [CIR5继承记录](../../artifacts/mind-cir-four-20260918/final-readback/mind-cir-staged-recovery-0919-v1/recover-staged-qwen35-2b-cir5-ni3-ni5/formal/ni3/initial_adapter_state.json)
