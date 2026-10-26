# MIND NI-only 历史长度消融

2026-09-15 用户批准：复用 Qwen3.5-2B、Behavior+Interest → Semantic-ID、NI-only SFT 的 H50，新增 H30、H10、H5 训练与评估。

## 冻结比较

| 项目 | 设置 |
|---|---|
| Backbone | Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 输入 | 原 H50 样本最近最多 H 条 `[Topic category] Headline`，保留顺序和重复，重新从 1 编号 |
| 训练样本 | 固定 21,600 条，108 类 × 200；不重采样、不修改目标，不随短窗口重新定义 NI |
| 评估样本 | 固定 Full Dev 31,392 impressions、37,728 target occurrences、104 个有 support 类别 |
| 训练 | 原 full-vocabulary、completion-only token-mean SFT；LoRA r16/alpha32/dropout0.05；训练 LoRA 与 SID 行 |
| 优化器 | AdamW fused，LR 2e-5，weight decay 0.01，cosine，warmup ratio 0.03，seed42 |
| 预算 | 3 epochs，2,025 optimizer updates，64,800 样本曝光，有效 batch32；选最后一个 epoch |
| 执行安排 | 沿用 H50：第 1–500 步单卡、batch1/累积32、Torch fallback；第 501–2025 步四卡、每卡 batch8/累积1、已验证的 FLA 0.5.2 + causal-conv1d 1.7.0 |
| 第 500 步衔接 | 保留模型、optimizer、scheduler、样本顺序和经验证的 RNG 恢复；保持 2,025 步 LR 总预算；不是重新开始一个训练任务 |
| 评估 | 每组 Train/Dev 使用相同 H；原 108 SID next-token 排名、Top20，不过滤历史类别；Macro Recall/NDCG@10/20 |
| 资源 | 三个独立 Recall Standard `80G4-A100` 作业；各作业第一段使用其中一张卡，之后四卡训练及四分片评估 |

H50 基线 ID：`qwen35-2b-bi-sid-ni-only-21600-seed42-v1`，后端 `olive_bean_s743bd4tqj`。原完整验收 SHA-256 为 `adaf18883c304fb183645a4ec2d6f430fc3d8d2ce14be895ab819b7c7dd25594`。

H50 的云端 `metrics.json` 已在本次任务重新回读并核对 SHA-256：`7c96deb861af4439df832e0b7836f3e899bf32fe56936494905995530d3dd735`。完整训练实现、模型和评估实现来自原已验收的冻结 bundle；本次只增加数据截断与上述固定步骤的编排。H50 曾经历平台恢复，因此保留相同执行安排不等于宣称不同长度下的浮点计算或随机数消耗逐位相同。

## 数据审计

| 条件 | 训练输入实际改变 | Dev 输入实际改变 | 训练上限 | Dev 上限 |
|---|---:|---:|---:|---:|
| H30 | 5,313 / 21,600 | 1,023 / 31,392 | 30 | 30 |
| H10 | 12,817 / 21,600 | 8,414 / 31,392 | 10 | 10 |
| H5 | 17,288 / 21,600 | 18,531 / 31,392 | 5 | 5 |

已对三个条件的完整写出 Parquet 逐行回读，检查其严格等于原 H50 行的最近 H 条输入变换。样本 ID、顺序、用户、时间、目标及 assistant 监督不变，Train/Dev 用户仍无交集。

除全量主结果外，报告原 H50 输入长度恰为 50 的固定 Dev 子集：297 impressions、426 targets、78 个有 support 类别。这一子集规模较小，只作为补充解释；不能将其与全量指标直接相减解释为因果效应。

## 执行与验收位置

- 数据：`data/training/mind_history_length_ablation_20260915_v1/`。
- AMLT：`configs/amlt/mind_history_length_ablation_0915_v1.yaml`。
- 冻结运行包：`artifacts/mind-history-length-ablation-20260915/bundle-v2/`。
- 本地审计：`artifacts/mind-history-length-ablation-20260915/local/`。
- 每个远程作业按次序运行完整输入/环境/snapshot/token 审计、训练、checkpoint 状态检查、fresh adapter reload、四分片 Full Dev、合并、逐键预测身份检查、独立 Macro 指标重算、完整产物 manifest 回读。
- 大型 checkpoint、权重、完整预测与结果文件留在远程持久存储；本机只保留紧凑审计、指标与恢复指针。
- `complete.json` 为本轮远程验收完成标记；作业 `running` 或 `pass` 本身不是实验完成证据。

本地隔离验证环境完成全套 1,088 项测试；最终新增功能 focused tests 为 8 项。AMLT dump 与 73 个文件的实际上传清单核对通过。H30、H10、H5 均已完成训练与完整 Dev 验收；H50 复用原已验收基线。H5 额外通过 Azure 下载存档中的全量预测与指标独立重算。

## 提交记录

Experiment：`mind-history-length-0915-v1`，冻结 bundle manifest SHA-256：`06b5bccfaa67351a04c323fda91dee34d353f6c92f30a51b1320ff20f1517b95`。

| 条件 | 作业 | 后端 ID |
|---|---|---|
| H30 | [qwen35-2b-ni-only-h30](https://aka.ms/amlt?q=k18om) | `hungry_tail_m89gx37wd9` |
| H10 | [qwen35-2b-ni-only-h10](https://aka.ms/amlt?q=k18on) | `nifty_office_mbm68kyy9k` |
| H5 | [qwen35-2b-ni-only-h5](https://aka.ms/amlt?q=k18oo) | `sharp_farm_jq3ftfk9m5` |

紧凑恢复指针：`manifests/mind_history_length_ablation_20260915_v1.json`。提交命令退出 0，三个后端已接收；准备/排队阶段不计作已完成训练更新。

## 2026-09-16 09:00 历史进度记录（非当前状态）

- H30：最新 `retry_003` 日志到 375/2,025 步；已核对 checkpoint-300 提交标记的数据身份。该次运行从 checkpoint-200 自动恢复。
- H10：完整 2,025 步训练、最终 checkpoint 和 worker training acceptance 已产出；4 条 fresh reload 样本精确一致，四个 Full Dev 评估分片已经启动。尚无最终合并指标。
- H5：训练 summary 和最终 checkpoint 均到 2,025 步，但 fresh reload 的 4 条样本均不满足原来的零误差阈值，最大 logits 差值 0.125；373 个已保存/重载 adapter 张量逐位一致。原作业因此失败，尚未进行评估，不将其视为已接受的实验结果。

H5 的单卡只读诊断已提交：`mind-history-h5-reload-diag-0916-v1` / `h5-reload-diagnostic`，入口 `configs/amlt/mind_history_h5_reload_diagnostic_0916_v1.yaml`。诊断保留原失败产物，在远程验证完整 checkpoint-2025 和最终 adapter，并对 fresh-process 与不同 kernel 路径做受控前向比较；新增 optimizer updates 为 0，原验收阈值不变。根因及修复尚未验证。证据位于 `artifacts/mind-history-length-ablation-20260915/progress/check-20260916-02/` 与 `diagnostics/h5-reload-v1/`。

## H5 修复与最终验收

H30 与 H10 均已完成原定 2,025 步训练、31,392 条 Full Dev 评估及独立指标重算。H30 的多次平台回收影响耗时，最终结果已验收。

H5 的故障定位到 FLA 0.5.2 融合归一化核 `fla.modules.fused_norm_gate.layer_norm_gated_fwd_kernel` 的启动配置。固定 BT=16、num_warps=4/8 可以复现最大 logits 差值 0.125；固定 BT=16、num_warps=16、num_stages=3、num_ctas=1 后，四条原参考样本的全部 432 个 SID logits 误差为 0。两个独立新进程复验通过，正式发布目录中的第三次复验也通过，atol=rtol=0。373 个 adapter 张量与原 checkpoint 逐元素精确一致，原始权重及参考文件未变。该验证证明这些原参考样本的重载一致性，不代表所有历史训练计算路径逐位相同。

正式采用原 H5 最终模型；替代的末尾 25 步恢复作业 `mind-history-h5-recovery-full-0916-v1` 已取消，未使用其模型或指标。此选择在 H5 Dev 指标产生前冻结。修复配置保存在 `final_adapter/fla_norm_profile.json`，并通过原 evaluator 的目录哈希纳入评估身份；原 evaluator、merger 和独立指标审计实现保持字节不变。

原训练的 21,600 样本、3 epochs、2,025 optimizer updates、64,800 样本曝光、H50 相同的样本前缀哈希、四卡 RNG 恢复和最终 checkpoint 均再次验证。评估在 GCR 的独立容器中按四个逻辑分片顺序执行，每片 7,848 条、batch8。完整预测保留远程，本机只回读紧凑结果。

## 实验结果

全量主表使用相同的 31,392 条 Dev、37,728 个目标、104 个有 support 类别。数值为百分比。Macro Recall 对每个目标类别的命中率等权平均；Macro NDCG 对每个目标类别的 `1/log2(rank+1)` 收益等权平均。无生成式 judge，无历史类别过滤。

| 历史上限 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 | 验收 |
|---|---:|---:|---:|---:|---|
| H50 | 20.3275 | 33.6553 | 10.1325 | 13.4663 | 复用 |
| H30 | 20.4167 | 33.3155 | 10.4809 | 13.7103 | 通过 |
| H10 | 19.3561 | 31.6497 | 9.4951 | 12.5745 | 通过 |
| H5 | 17.4267 | 29.7096 | 9.3350 | 12.4370 | 通过 |

相对 H50 的变化，单位为百分点：

| 条件 | Δ R@10 | Δ R@20 | Δ NDCG@10 | Δ NDCG@20 |
|---|---:|---:|---:|---:|
| H30 | +0.0892 | -0.3398 | +0.3485 | +0.2441 |
| H10 | -0.9714 | -2.0056 | -0.6373 | -0.8918 |
| H5 | -2.9008 | -3.9457 | -0.7974 | -1.0293 |

在这次 seed42 实验中，H30 与 H50 的全量表现接近；截短为 H10、H5 后，四个主指标均低于 H50。这是单个 seed 的观察，未进行多 seed 显著性检验，不能据此断言 H30 最优。

原 H50 输入长度恰为 50 的固定 Dev 子集，所有条件均为 297 条 impressions、426 个目标、78 个有 support 类别：

| 历史上限 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| H50 | 14.3984 | 23.4254 | 6.4656 | 8.7279 |
| H30 | 15.5906 | 24.4928 | 6.5025 | 8.7271 |
| H10 | 13.3895 | 24.3216 | 5.9169 | 8.6512 |
| H5 | 16.2657 | 27.8293 | 6.5955 | 9.5606 |

小子集上的趋势与全量不同，H5 在此子集的四个主指标最高。因此，结果不支持“所有用户的历史越长越好”；主结论应以冻结的全量 Dev 为准。这里不把不同类别支持分母下的两个表相减解释为因果效应。

## 最终产物与恢复

- H5 原始 adapter SHA-256：`4dbcc2dd5fa8c3b8b6f3b2898d07d994d428167cedf3265bd29148be490b2f82`。
- H5 原始参考 SHA-256：`8d0ff4f60e44e4b35ff9df21ab0d99e5484cf1065c0efefded48fe72daa063fe`。
- H5 原始训练/optimizer/checkpoint：Azure `amulet/projects/sft4safety/amlt-results/7210551062.55495-3b46c238-52c4/`；失败标记与原始证据保留。
- H5 最终评估存档：Azure `amulet/projects/sft4safety/manual-results/history-h5-original-profile-20260916-v1/publication.tar.gz`。该存档含原 adapter、修复配置、完整预测、冻结运行源码、输入和验收证据；大型文件均未下载到 Mac。
- Azure 存档共 314 个文件通过远程下载哈希回读。随后从下载存档重新核验 31,392 条预测身份/顺序/合法 Top20、四片覆盖与合并一致性，独立重算全量和固定子集指标，全部一致。
- [对照结果 JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/comparison.json)。
- [H5 Azure 下载存档独立验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/readback/h5/azure_archive_acceptance.json)。
- [H5 持久化回读记录](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/readback/h5/azure_publication_receipt.json)。
- [H5 完整验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/readback/h5/complete.json)。
- [H5 数值执行配置](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/readback/h5/sft/final_adapter/fla_norm_profile.json)。
- [H30 完整验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/h30_complete.json)。
- [H10 完整验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-history-length-ablation-20260915/progress/check-20260916-03/h10-complete.json)。
- [数据及全部远程恢复指针](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_history_length_ablation_20260915_v1.json)。
