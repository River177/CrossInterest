# MIND 短思考 GRPO：E / H / HL 完整结果

三组均已完成 675 步训练、完整 Dev 评估和独立验收，远端作业均为 Completed。结果使用同一 31,392 条 Dev、104 个有支持类别及 37,728 个目标出现；以下排名指标单位为百分比。

| Method | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Mixed (ref.) | 20.61 | 34.20 | 10.42 | 13.82 |
| ST-SFT | 14.00 | 24.58 | 6.74 | 9.41 |
| E-GRPO | 13.31 | 23.95 | 6.87 | 9.54 |
| H-GRPO | 17.16 | 28.47 | 7.90 | 10.72 |
| HL-GRPO | 16.30 | 28.36 | 7.53 | 10.53 |

ST-SFT 是三组 GRPO 的共同短思考 SFT 初始化；Mixed (ref.) 是此前原 Mixed 模型的参考结果。E 使用精确命中奖励，H 使用层级奖励，HL 在 H 的基础上加入长度惩罚。

相对 ST-SFT 的变化（百分点）：

| Method | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| E-GRPO | -0.69 | -0.63 | +0.13 | +0.13 |
| H-GRPO | +3.16 | +3.90 | +1.16 | +1.31 |
| HL-GRPO | +2.30 | +3.78 | +0.79 | +1.13 |

本轮结论：

1. **H-GRPO 是本轮最好的 GRPO 配置。** 四项指标均高于 E 和 HL；相对 ST-SFT，Macro R@10 提高 3.16 个百分点，R@20 提高 3.90 个百分点。
2. **当前 HL 长度惩罚没有带来更好的综合结果。** 相对 H，四项排名指标分别变化 −0.86、−0.12、−0.37、−0.19 个百分点；平均总输出从 111.11 增至 112.47 tokens。可解析思考段均值略降，但两者解析成功的样本数不同，不能据此说整体输出成本降低。
3. **GRPO 恢复了部分短思考路径的掉分，但仍未达到原 Mixed 参考水平。** E 仅有小幅 NDCG 改善，Recall 下降；H/HL 四项均超过 ST-SFT。后续优先保留 H 奖励，处理短思考 SFT 路径的排序损失。该路径同时改变了训练、提示词和推理方式，不能把差异单独归因于 think 开关。

输出行为由完整预测 token 重新解析和统计：

| Method | 格式合法率（%） | 平均总输出 tokens | 可解析思考均值 tokens | 可解析条数 |
|---|---:|---:|---:|---:|
| ST-SFT | 96.13 | 111.01 | 96.92 | 30,767 |
| E-GRPO | 94.72 | 116.90 | 100.26 | 30,778 |
| H-GRPO | 97.26 | 111.11 | 97.38 | 30,988 |
| HL-GRPO | 94.41 | 112.47 | 96.46 | 30,344 |

总输出统计包含答案和结束标记，分母均为 31,392；思考段统计仅覆盖可解析记录。主排名指标始终保留非法输出的完整分母。

复现口径：Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；NI Train 21,600（108×200），seed 42、固定顺序、每步 32 条历史×8 条轨迹、每组 675 步。H 与 HL 共享 H0–100，随后各自续训至 675，保持优化器、随机状态和样本游标连续。E 单独完成 675 步。

H 奖励为精确 1、未见同父类 0.2、其他未见 0、历史已见 −0.1、非法 −0.2；HL 在共享 H100 后，仅对精确轨迹减去 `0.1 × clip((思考 tokens − 160) / 160, 0, 1)`。训练采用 480/32 的思考/回答预算和既定 SID 采样约束。最终评估统一为无约束 greedy512、原 4 分片/batch8，在生成 SID 之前的前缀上对全部 108 个 SID 排名，不过滤历史类别。这是单 seed 的 Dev 结果，不是额外的独立 Test。

验收覆盖所有 checkpoint 祖先、参数与优化器状态、RNG/样本顺序、奖励与优势、精确重新加载、四个评估分片、合并预测及四项 Macro 独立复算。完整原始产物已按完成标记和 manifest 校验；压缩归档本身未另行下载，未将其声明为已独立核验。三个空文件故障均恢复为原预提交哈希对应的字节，完整训练链和最终评估通过复核，失败与恢复证据保留。

结果与验收记录：

- [精确结果与来源](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-grpo-20260914/accepted_full_dev_results.json)
- [独立输出长度/格式统计](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-grpo-20260914/generation_diagnostics_independent.json)
- [ST-SFT 独立基线](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-grpo-20260914/acceptance/full_reader_baseline_eval_regression.json)
- [E 完整验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-grpo-20260914/acceptance/E-full675-Dev-independent.json)
- [H 完整验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-grpo-20260914/acceptance/H-full675-Dev-independent.json)
- [HL 完整验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-grpo-20260914/acceptance/HL-full675-Dev-independent.json)
