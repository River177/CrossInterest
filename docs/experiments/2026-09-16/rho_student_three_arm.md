# 动态可约减损失权重：三组学生实验

三组各完成 21,600 个训练用户、3 轮和 2,025 次更新；完整验证集与 Dev 预测经远端逐条核验，配对用户 bootstrap 已完成。

本次动态权重没有带来增量收益。验证集主对照 Dynamic−Static 的 Macro NDCG@20 为 −0.4774 个百分点，95% CI 为 [−0.7238, −0.2179]；Dev 同对照为 −0.5714 个百分点，95% CI 为 [−1.0399, −0.1156]。Static 相对 Uniform 的点估计在两套数据上都略高，但两套区间都包含 0，尚不能确认静态加权有收益。这些结论限于本次冻结方案和单个训练 seed。

## 研究问题与训练设计

比较动态学习状态是否在静态 OOF 难度之外提供增量收益。三组从同一官方 Qwen/Qwen3.5-2B checkpoint（revision `15852e8c16360a2fea060d615a32b45270f8a8fc`）、相同 SID 初始化与 seed42 LoRA 开始，保留原 History Behavior+Interest → SID 数据、原生完整词表 completion loss 和采样顺序。

Uniform 使用权重 1；Static 在类别内按 −OOF loss 排序；Dynamic 按当前学生 loss−OOF loss 排序。两种加权都使用平均并列排名映射到 0.5–1.5，并保持每类总权重相同。所有样本保留；Static 权重固定，Dynamic 在每轮开始前更新，不重置优化器。三个条件都执行相同的当前学生评分轮次。

OOF loss 来自没有见过被评分用户的另一半参考模型。用于排序的 OOF loss 和当前学生 loss 都采用首 SID 的完整词表 NLL；实际训练保留 SID、结束符和换行共 3 个 completion token 的原生损失，每个样本乘一个停止梯度的权重。本方法是将 RHO-LOSS 思想迁移为软权重，未复现原论文的大 batch 中选小 batch 算法。

## 评价设计

固定评估最后 checkpoint。主检验为验证集 Dynamic−Static 的 Macro NDCG@20；Dev 作为同口径一致性检查。每个数据集的三个差值共享 2,000 次 seed42 用户簇重采样，同一用户的所有 impressions 和目标共同重采样。区间是单个训练 seed 条件下的用户抽样不确定性。

验证集主指标覆盖支持充分的 107 类，AI 类 4 条单列；Dev 保持 31,392 impressions、27,082 用户、104 类。排名覆盖全部 108 个 SID，不过滤历史类别。下表点估计用百分数，差值和区间用百分点。

恢复期间，Static OOF、Dynamic 的远端 checkpoint 保留上限由 3 增至 25；模型、训练数据、loss 和优化器配置保持原方案。这个存储设置覆盖已通过独立回执核验。

## 验证集 107 类

| 条件 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Uniform | 25.2523 | 39.4579 | 12.7276 | 16.2832 |
| Static OOF | 24.8224 | 39.8505 | 12.6886 | 16.4607 |
| Dynamic | 24.7290 | 39.3458 | 12.3212 | 15.9833 |

| 比较 | NDCG@20 差值 | 用户配对 95% CI |
|---|---:|---:|
| Static−Uniform | +0.1775 | [-0.0389, +0.3979] |
| Dynamic−Uniform | -0.2999 | [-0.5156, -0.0881] |
| Dynamic−Static | -0.4774 | [-0.7238, -0.2179] |

重采样中至少缺失一个原定类别的次数：0 / 2,000；这类抽样在三组间使用相同的非空类别集合。

## 官方 Dev 104 类

| 条件 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Uniform | 20.5617 | 33.4688 | 10.3134 | 13.5263 |
| Static OOF | 20.3008 | 34.1714 | 10.3498 | 13.8300 |
| Dynamic | 19.7532 | 33.2640 | 9.8727 | 13.2585 |

| 比较 | NDCG@20 差值 | 用户配对 95% CI |
|---|---:|---:|
| Static−Uniform | +0.3037 | [-0.0399, +0.6418] |
| Dynamic−Uniform | -0.2677 | [-0.6028, +0.0936] |
| Dynamic−Static | -0.5714 | [-1.0399, -0.1156] |

重采样中至少缺失一个原定类别的次数：1589 / 2,000；按冻结规则，每次抽样仅在原定 104 类中仍有正支持的类别上计算配对差，三组使用相同的类别集合。每次抽样实际纳入 99–104 类，2,000 次抽样全部可用。Dev 点估计仍使用完整的 104 个支持类别。

## 验证集稀有 AI 类

4 个用户样本，不计入 107 类主指标；仅报告点估计。

| 条件 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| Uniform | 50.0000 | 50.0000 | 34.6713 | 34.6713 |
| Static OOF | 50.0000 | 50.0000 | 34.6713 | 34.6713 |
| Dynamic | 50.0000 | 50.0000 | 40.7732 | 40.7732 |

验证集主对照的区间整体低于零，本次设置下动态项不如静态 OOF 权重。

参考损失表示跨用户预测难度的估计；本实验没有人工噪声真值，不能将高损失样本认定为错标签。其余比较是边际区间，未进行多重比较校正。

## 数据与产物

- [训练设计](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/research/2026-09-16_rho_student_execution.md)
- [三组训练与评估验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-student-20260916/three_arm_acceptance.json)
- [完整配对指标、AI 四例及 hash 来源](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-student-20260916/compact/comparison/mind-rho-student-comparison-v1-0916/rho-student-three-arm-comparison/comparison.json)
- [运行状态](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-student-20260916/pipeline_state.json)
- [恢复清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/rho_student_20260916.json)

模型、checkpoint、完整评分、权重和预测均保留在远端；本机仅保存紧凑回执和报告。
