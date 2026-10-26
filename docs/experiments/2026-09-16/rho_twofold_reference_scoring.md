# 两折参考模型：21,600 条 OOF 损失评分验收

**2026-09-16：两组各 10,800 条，共 21,600 条评分全部完成，逐条远端读回及双折联合验收通过。** 本阶段没有训练学生模型，也没有验证动态重加权是否有效。

## 研究问题与数据

为每条训练样本取得一个“未见过该用户的参考模型”给出的预测损失，供后续研究动态可约减损失使用。该分数是**跨用户预测难度的代理量**，不是噪声真值；参考模型预测困难也可能来自稀有兴趣、历史信息不足或模型能力限制。

从已有 `mixed.parquet` 中仅提取 21,600 条 `new_interest` 样本，覆盖 108 类、每类 200 个不同用户。按 seed 42 的用户／样本标识 hash，在每类内分为两半；原 History Behavior + Interest → SID 的消息、标签及半内原始顺序保持不变。

| 参考模型 | 训练用户 | 评分用户 | 每半类别支持 | 训练终点 | 当前评分验收 |
|---|---|---|---|---|---|
| A（fold0） | fold0：10,800 | fold1：10,800 | 108 类 × 100 | 3 epochs，1,014 updates | 10,800 / 10,800 |
| B（fold1） | fold1：10,800 | fold0：10,800 | 108 类 × 100 | 3 epochs，1,014 updates | 10,800 / 10,800 |

两半用户交集为 0、并集精确覆盖原 21,600 条 NI 样本。与既有验证集 5,354 用户、Dev 27,082 用户的交集均为 0。数据构造时逐条检查完整原始历史，21,600 个目标均未出现于其历史；模型输入仍使用最近 50 条历史。验证集和 Dev 没有用于参考模型选优。

## 模型与评分口径

两模型均从原始 **Qwen/Qwen3.5-2B** 发布权重初始化，revision 为 `15852e8c16360a2fea060d615a32b45270f8a8fc`，没有继承见过全部训练用户的 Mixed adapter。原生全词表 SFT，LoRA r=16、alpha=32、dropout=0.05；学习率 2e-5、global batch=32、seed=42。每模型使用 4 张 A100，固定最终 checkpoint-1014。此次恢复评分使用这些原 checkpoint 的同一份权重，**新增训练更新为 0**。

每条样本仅输入目标 SID 前的因果前缀，计算第一个目标 SID 的全有效词表负对数概率：

\[
\ell_i^{\mathrm{ref}}=\log\sum_{v\in V_{\mathrm{eff}}}\exp z_{iv}-z_{i,y_i}.
\]

这里保留原生有效输出头，包括按原生 BF16 规则替换的 SID 行，再将 hidden state 与有效头提升到 FP32，计算投影和 logsumexp。**不是只在 108 个 SID 内归一化，也不是整个 assistant 后缀的平均 SFT loss。** 不进行类别先验修正；108 类条件 NLL 和 SID 排名仅作为诊断指标。正式评分 batch=4，每模型分四个逻辑 shard。

## 数值诊断与验收范围

B 原始训练结束后的重载检查出现最大 0.125 的 logits 差异，四条探针中一条 Top-1 改变；此前保存张量的精确比较已经通过。随后对同一 checkpoint 做零更新诊断：

| 检查 | 实际观察 |
|---|---|
| 原生选核下两个独立 fresh 进程 | 最大差异 0.125 |
| 原生选核下各条件内部重复 | 差异为 0 |
| 固定 FLA 选核后的四条件及重复比较 | native、FP32 探针差异均为 0 |
| 当前 A、B 各自两次独立 fresh 重载 | 单例 native／FP32、实际 batch-4 FP32 探针均精确一致 |

因此两折都统一使用 `stable_fla_v1` 重新评分：保留候选配置合法性检查，以稳定的配置 hash 选择有效 FLA kernel，并核验相同 cache key 的选核一致性。诊断中的 fresh 进程之间也有差异，**不能把原因单独归于训练／warmup 历史**。原训练时的原始探针向量没有持久化；新探针的一致性证明当前 checkpoint 的评分可复现，不能替代对那次历史 forward 的等价性证明。旧失败及旧 A 评分均保留。

远端独立 reader 检查文件 hash、逐条用户／样本身份、四 shard 完整覆盖、无训练／评分用户交叉、权重重载与有效头检查记录、选核记录，并从保存的全词表 logsumexp 和目标 logit 重算 NLL；108 类条件 softmax、排名和支持数也独立重算。**这不是对全部样本重新执行第二遍模型推理。** 两次独立模型推理的一致性检验覆盖固定四条探针及其 batch-4 组合。

## 已验收结果

以下均为各自对面训练半集的 OOF 结果，不能直接与官方 Dev 指标对比。合并均值按样本数加权；分位数仅报告各半实际计算值。

| 指标 | A → fold1 | B → fold0 | 合并 OOF |
|---|---:|---:|---:|
| 已最终验收样本 | 10,800 / 10,800 | 10,800 / 10,800 | 21,600 / 21,600 |
| 全词表 first-SID NLL：均值 | 4.539789 | 4.529259 | 4.534524 |
| 全词表 first-SID NLL：P10／中位数／P90 | 3.708439／4.551638／5.358572 | 3.747183／4.548428／5.278656 | 未合并计算 |
| 108 类条件 SID NLL：均值 | 4.534144 | 4.523691 | 4.528918 |
| SID Recall@1 | 380 / 10,800（3.519%） | 410 / 10,800（3.796%） | 790 / 21,600（3.657%） |
| SID Recall@10 | 2,334 / 10,800（21.611%） | 2,411 / 10,800（22.324%） | 4,745 / 21,600（21.968%） |
| SID Recall@20 | 3,860 / 10,800（35.741%） | 3,903 / 10,800（36.139%） | 7,763 / 21,600（35.940%） |

两组均覆盖 108 类、每类 100 条；联合覆盖每类 200 条。全部 10 个共同内核配置键一致。全量覆盖、输入身份、类别支持、训练／评分用户隔离及 NLL／排序重算均通过。没有将高损失样本直接标为噪声。

本次修复保留 3 epochs／1,014 updates 的两个训练终点。A 的 Azure 启动延迟随后恢复，未取消或替换任务。已保存的针对评分和数值诊断测试日志为 73 passed，finisher 测试日志为 64 passed；本次收尾另对实际数据和两组产物执行了独立联合核验。

## 证据与恢复位置

本机仅保留紧凑统计、报告、hash 和恢复指针；模型、optimizer、完整预测及其完整核验均留在远端。

- 数据：[独立输入核验](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/final_local_input_verification.json)、[构造元数据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/bundle-stable-rescore-v1/inputs/metadata.json)。
- 固定方案：[训练／评分协议](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/bundle-stable-rescore-v1/protocol.json)、[checkpoint 与历史失败绑定](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/bundle-stable-rescore-v1/rescore_contract.json)、[数值诊断结果](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/compact/reload-diagnostic/mind-rho-fold1-reload-diagnostic-v1-0916/qwen35-2b-rho-fold1-reload-diagnostic/compact/diagnostic_summary.json)。
- A 完整证据：[逐行独立验收及指标](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/compact/stable-fold0/mind-rho-twofold-stable-rescore-v1-0916/qwen35-2b-rho-stable-score-fold0/compact/reference_acceptance.json)、[最终 receipt 与完整恢复指针](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/compact/stable-fold0/mind-rho-twofold-stable-rescore-v1-0916/qwen35-2b-rho-stable-score-fold0/compact/worker_success.json)。
- B 完整证据：[逐行独立验收及指标](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/compact/stable-fold1/mind-rho-twofold-stable-rescore-v1-0916/qwen35-2b-rho-stable-score-fold1/compact/reference_acceptance.json)、[最终 receipt 与完整恢复指针](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/compact/stable-fold1/mind-rho-twofold-stable-rescore-v1-0916/qwen35-2b-rho-stable-score-fold1/compact/worker_success.json)。B 验收文件 SHA-256：`f73d863fb3b1993087ae2b27a1a74379ffabe06687ce058803581dfcdbebb70b`。

- 联合核验：[双折独立验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/rho-twofold-reference-20260915/independent_twofold_acceptance.json)、[恢复清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/rho_twofold_scoring_20260916.json)。

远端实验为 `mind-rho-twofold-stable-rescore-v1-0916`，任务分别为 `qwen35-2b-rho-stable-score-fold0`、`qwen35-2b-rho-stable-score-fold1`。其 `/mnt/output/projects/sft4safety/amlt-results/` 下的结果目录分别是 `7210501074.54897-13d17a85-8dc3`（A，已验收）和 `7210501074.54859-3f88aaab-b5ee`（B，已验收）。两组的 `teacher/`、`inputs/`、`oof_stable_fla_v1/shard-0` 至 `shard-3` 由各自最终 receipt 逐一列出；每个评分 shard 的最终分数文件为 `scores.parquet`。

原始 checkpoint 在 `/mnt/default/projects/sft4safety/amlt-results/` 下：A 为 `7210554755.61946-841221fb-e7d4/checkpoints-full/checkpoint-1014`，B 为 `7210554755.61884-15430fe9-d277/checkpoints-full/checkpoint-1014`。两者 marker 与 adapter SHA 均由固定 contract 绑定，评分阶段没有修改它们。
