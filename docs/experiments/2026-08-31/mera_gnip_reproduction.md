# MeRA-GNIP 方法复现

## 1. 研究问题（Research Questions）

本实验复现并适配 MeRA（*Large Language Models with Multi-faceted
Relation Alignment for User Novel Interest Discovery*, PAKDD 2025，DOI
`10.1007/978-981-96-8298-0_28`）到 CrossInterest 的 GNIP 任务。这里的复现目标
不是重跑论文的 Amazon Arts/Grocery 原始分数，而是在相同 CrossInterest 输入、
标签空间和评测协议下回答：

1. 在 `Qwen/Qwen3.5-9B + 108-class SID History + SID target` 上加入 MeRA 的
   structural regularization 与 collaborative regularization，能否超过严格匹配的
   Semantic-ID Vanilla？
2. 增益分别来自纯 mean-pooled token 初始化、结构关系还是训练用户共现关系？
3. 论文的共现阈值 `T=0.05` 是否适用于 108 类新闻兴趣；若不适用，哪个不看 Dev
   标签、仅由 Train coverage 决定的阈值能保留完整 regularization support？

主指标固定为 Full user-disjoint Dev 上的 Interest-Macro Recall@10 和
NDCG@10；同时报告 Impression-Micro 指标以及 `@1/@5/@20`。主比较不混入
Cross-Fitted 样本权重。

## 2. 训练数据构造（Training Data Construction）

### 2.1 与原论文的对应关系

| MeRA 组件 | 原论文/官方实现 | MeRA-GNIP 适配 |
|---|---|---|
| 单类别 token | 每个二级 Amazon 类别新增一个 token | 仅注册 108 个 `<interest_NNN>` |
| token 初始化 | 类别名原 token embedding 的纯均值 | 显式固定 `semantic_id_initialization_mode: mean` |
| 结构关系 | 同一一级类别下的二级类别互为 sibling | 同一 MIND 重分类一级类别下的 target subcategory 互为 sibling |
| 模型输入 | 类别 token 的历史序列 | 保留顺序与重复的 108-class SID 历史；无 headline、无类别名称 |
| 共现关系 | 每个 Train 用户的 History 类别集合并入 target 后统计有序共现对 | 对 21,600 个 Train 用户执行相同规则；只保留这 108 类 |
| 结构损失 | 每个有 sibling 的类别采 1 个正例和 N 个非 sibling 负例 | 相同；同时作用于 input embedding 与 LM head 行 |
| 共现损失 | 按行归一化共现概率采正例，按 `1-M` 采负例 | 相同；同时作用于 input embedding 与 LM head 行 |
| 训练目标 | `L_LLM + alpha L_structure + beta L_collab` | 相同；`alpha=0.05, beta=0.1, N=10` |
| 推理 | 直接对新增类别 token 做 Top-K | 复用 restricted Semantic-ID next-token Top-K |

正式论文正文与 GitHub commit `c07bdbe57bf82e5c938b478035baee78d630b17c`
存在三处需要冻结记录的差异：论文报告 `T=0.05`，公开 notebook/训练代码未应用
threshold；论文公式按每个类别定义 structure loss，公开代码每个 parent group 仅采
一对 sibling；论文 Implementation Details 报告 `alpha=0.05, beta=0.1`，公开
Grocery/Arts shell 分别使用 `0.1/0.1` 与 `0.01/0.01`。本实现以论文公式为主，
同时将 threshold、权重和负例数全部写入 config、checkpoint identity 与 audit。

### 2.2 CrossInterest Train 构造

训练输入为冻结的 21,600-row `Interest-SID → SID` 数据：108 个类别，每类 200 个
用户，每用户一条 Train target。History 保留原顺序与重复项，但其中不属于这 108
类的事件被删除；Train/Dev 分别删除 3,942/2,441 个事件。过滤后有 1 个 Train row
和 6 个 Dev impression 的 SID History 为空；为保持既有 user/impression denominator，
这些记录保留为空序列，不新增 `<UNK>` 或第 109 个 token。

relation artifact 只读取同一 Train split；每行先对 108 类 History 去重，再与 target
取并集，最后对集合内所有有序类别对各加一。

完整构建结果：21,600 users、108 interests、20 parent groups；105/108 类有
结构 sibling。共现阈值覆盖如下：

| T | collaborative anchors | anchor coverage | positive edges |
|---:|---:|---:|---:|
| 0.00 | 108/108 | 100% | 11,556 |
| 0.01 | 108/108 | 100% | 3,891 |
| 0.02 | 108/108 | 100% | 925 |
| 0.05 | 4/108 | 3.70% | 5 |

忠实复现条件固定使用论文报告的 `T=0.05`。该阈值在 CrossInterest 上只保留
4/108 个 collaborative anchors，因此这是方法忠实条件，不应被解释成已经为新闻
任务调优的最佳阈值；后续适配实验可另行比较 `T in {0, 0.01, 0.02}`。

### 2.3 真实训练样例

来源：`sample_id=train:1447803:66:interest_sid`，
`source_sample_id=train:1447803:66`，target label 为
`Relationships & Dating`。

英文原文：

> Given a user's chronological sequence of historical interest category tokens,
> predict one new interest category token that the user may interact with in the
> future. The predicted category must not occur in the historical sequence.
> Respond with exactly one category token and no other text.
>
> Historical interest category tokens:
> `<interest_019> <interest_019> <interest_032>`

中文翻译：

> 给定按时间排序的历史兴趣类别 token 序列，预测用户未来可能交互的一个新兴趣
> 类别 token。预测类别不得出现在历史序列中；只输出一个类别 token，不得输出
> 其它文本。历史序列为：`<interest_019> <interest_019> <interest_032>`。

监督输出为 `<interest_085>`。这是 target SFT record，不是模型生成 response。

## 3. 测试设计（Testing）

### 3.1 模型与运行位置

基础模型固定 `Qwen/Qwen3.5-9B`，输入/输出均为同一 108-class SID 词表。
Transformers `5.14.1`、PEFT `0.18.0`、
Accelerate `1.12.0`，BF16，LoRA rank 16/alpha 32，所有线性层，learning rate
`2e-5`。首个 Gate-C smoke 使用论文 `T=0.05`，配置为 Feeds
`80G1-A100 / Standard`：先审计 108 个
input/output token rows 的 MeRA gradient，再训练 128 rows/1 epoch/16 optimizer
steps，最后保存 compact adapter 并用新进程回读 logits。AMLT YAML 已通过
`amlt run --dump`。远程 experiment 为
`mera-gnip-faithful-qwen35-t005-smoke-0831-v1`，job
`qwen35-mera-faithful-t005-1gpu-smoke`，AMLT 终态 `pass`；下载后的
`smoke_success.json`、manifest 和 18 个受约束产物已独立回读。

通过单卡 Gate-C 后，才生成并运行与 full job 相同 world size 的四卡两步以上 DDP
smoke；再通过 64-record real evaluator 与 artifact readback 后，才允许 21,600-row
full training。

Gate-D/E 已由 `mera-gnip-faithful-qwen35-ddp-eval-smoke-0831-v1` /
`qwen35-mera-faithful-t005-ddp-eval-smoke` 完成并达到 AMLT 终态 `pass`。四卡 DDP
完成 8/8 optimizer steps，loss 约从 `5.91` 降至 `4.75`；fresh reload 的 498 个
adapter tensors bit-exact，4 个 reference samples 的 logits 最大误差为 `0.0`。
64-record 真实 evaluator smoke 的 Recall@10 为 `0.15625`、Interest-Macro
Recall@10 为 `0.11728395`，仅作为链路连通性检查，不作为方法效果结论。下载回读
25/25 个 manifest 产物通过，manifest SHA-256 为
`fdeffedbd3d51d28b7390ef24264ca6fe6eba1e6494f4d54cd0b0cb7f713c803`。

为补全纯均值消融矩阵，另对最长的新路径 `Behavior+Interest → SID + Full MeRA`
运行四卡代表性 Gate-D/E。远程 experiment/job 为
`mera-gnip-qwen35-mean-ablation-bi-full-smoke-0901-v1` /
`qwen35-mera-bi-mean-full-ddp-eval-smoke`，AMLT 终态 `pass`。8/8 DDP steps
总体 loss 从 `8.588` 降至后段约 `6.3–6.5`；498 个 adapter tensors 精确回载，
4/4 reference logits 最大误差 `0.0`；64 条真实评测与 25/25 下载产物哈希通过。
该 smoke 的指标只检查连通性。manifest SHA-256 为
`e78bb291bc421f3ffaa7266ecab06053c9d03fc53a2701538428888dd0a8284c`。

### 3.2 测试数据来源

计划评测文件为相同 Semantic-ID 数据根下的 Full grouped Dev：31,392 个唯一
impressions、27,082 users、37,728 target occurrences、104 个有 support 的 target
interests，且 0 个 impression 的 acceptable target 与其 History subcategory 重叠。
Train 与 Dev 保持既有 user-disjoint 边界。

### 3.3 测试样例

Gate-C 对训练文件前 4 个真实 prompt 保存 108-way restricted logits，并在 fresh
reload 后比较：4/4 样例最大绝对误差均为 0，Top-1 与完整 Top-20 排序完全一致。
该检查验证序列化，不是 Full Dev 效果评测。完成 Full Dev 后，案例仍须从 ranking
artifact 中抽取，并同时给出英文 prompt/response 和中文翻译。

### 3.4 Judge 与指标过程

本实验没有 LLM judge。模型对 108 个 Semantic-ID token 的 next-token logits 做
restricted softmax，按 score 排序取 Top-20。对 impression `u` 的 acceptable target
集合 `Y_u`，Recall@K 为 `|TopK_u ∩ Y_u| / |Y_u|`，NDCG@K 使用同一多正例集合的
binary relevance；Interest-Macro 先按 target interest 聚合，再对有 Dev support 的
104 类取平均。任何非法 token、重复 rank、缺失/重复 impression 或 shard identity
hash 不一致都使评测失败，而不是计作普通错误预测。

## 4. 实验结果（Results）

忠实 `SID → SID + Full MeRA` 已完成训练、Full Dev 与下载回读：

| gate | scope | result |
|---|---|---|
| relation build/readback | 21,600 rows；108x108 counts；source/artifact SHA-256 | pass |
| full local unit suite | builder、loss、gradient path、plain-mean reload mode、7-condition config；484 tests | pass |
| AMLT schema/dump | Feeds；80G1-A100；Standard；1 job；无配置内 secret | pass |
| Gate-C GPU smoke | 108/108 input/output token-row gradients finite/nonzero | pass |
| Gate-C SFT | 128 rows；16 steps；loss `5.5281 → 4.4836`（前/后 4 steps mean） | pass |
| Gate-C adapter | 498 tensors；44,163,072 values；无完整 embedding/lm_head | pass |
| Gate-C fresh reload | state hash exact；4/4 logits max abs error `0.0`；Top-20 exact | pass |
| Gate-C artifact readback | 18/18 manifest files；manifest SHA-256 `bae4448482637f59252cfab46f8eb325d19c203d9a30141f09a6eb3ca5697004` | pass |
| DDP/evaluator smoke | 4-GPU；8/8 steps；498-tensor exact reload；64 Full-Dev records；25/25 artifact hashes | pass |
| Full training/evaluation | 21,600 Train；3 epochs；2,025 steps；31,392 Dev impressions | pass；23-item remote gate＋关键产物本地复算通过 |

八个严格匹配条件均已完成；7 个补充条件为 AMLT `7/7 pass`，且各自完成
2,025-step checkpoint、498-tensor exact reload、31,392-row 合并预测和关键产物
本地哈希/指标复算：

| input | condition | alpha | beta | Micro R@10 | Micro NDCG@10 | Macro R@10 | Macro NDCG@10 | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SID → SID | Vanilla, pure mean | 0 | 0 | 30.2924% | 16.7706% | 17.8219% | 9.5100% | pass＋readback |
| SID → SID | structure only | 0.05 | 0 | 30.2771% | 16.8166% | 17.7487% | 9.1901% | pass＋readback |
| SID → SID | collaborative only | 0 | 0.1 | 29.9287% | 16.5753% | 17.6884% | 9.2139% | pass＋readback |
| SID → SID | Full MeRA | 0.05 | 0.1 | 30.0997% | 16.6991% | 17.8009% | 9.3815% | pass＋readback |
| Behavior+Interest → SID | Vanilla, pure mean | 0 | 0 | 31.5790% | 17.1620% | 20.6955% | 10.5485% | pass＋readback |
| Behavior+Interest → SID | structure only | 0.05 | 0 | 31.4723% | 17.1041% | 20.5103% | 10.3711% | pass＋readback |
| Behavior+Interest → SID | collaborative only | 0 | 0.1 | 31.4773% | 17.1690% | 20.9060% | 10.6437% | pass＋readback |
| Behavior+Interest → SID | Full MeRA | 0.05 | 0.1 | 31.3639% | 17.1007% | 20.4977% | 10.5170% | pass＋readback |

在单个 seed 42 上，SID → SID 的 Full MeRA 相对纯均值 Vanilla 的 Macro R@10
为 `-0.0210pp`，四个条件非常接近；Behavior+Interest 中 collaborative-only
相对 Vanilla 为 `+0.2105pp`，但 Full MeRA 为 `-0.1977pp`。这些是单 seed Dev
描述，未做多 seed 或 paired bootstrap，不能宣称显著增益。

最终主结果表必须至少包含同输入的 SID Vanilla、MeRA w/o structure、MeRA w/o
collab、Full MeRA paper `T=0.05`；适配实验再列 `T=0/0.01/0.02`。每行记录模型、初始化方式、alpha/beta/T/N、
Train rows、optimizer steps、adapter/reload gate、Macro/Micro Recall/NDCG 和 artifact
hash。当前不能把“代码完成”或 AMLT dump 记为方法效果复现完成。

## 5. 数据与产物路径（Artifact Paths）

- 论文与官方实现：<https://doi.org/10.1007/978-981-96-8298-0_28>；
  <https://github.com/GabyUSTC/MeRA>，核验 commit
  `c07bdbe57bf82e5c938b478035baee78d630b17c`
- relation source：`data/processed/new_interest_static_user_disjoint_sft21600_h50_v1/train.parquet`
- SFT Train：`data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1/train/interest_sid/21600.parquet`
- Full Dev：`data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1/eval/interest_sid/dev_full_grouped.parquet`
- Semantic-ID mapping：`data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1/interest_token_mapping.parquet`
- relation artifact：`data/training/mera_static_user_disjoint_behavior_interest_200_h50_v1/relations.json`
- relation statistics/metadata：`data/training/mera_static_user_disjoint_behavior_interest_200_h50_v1/{statistics.json,metadata.json}`
- relation builder：`src/new_interest/data/build_mera_relations.py`
- MeRA loss：`src/new_interest/training/mera_regularization.py`
- gradient gate：`src/new_interest/training/audit_mera_regularization_gradients.py`
- SFT integration：`src/new_interest/training/train_sft.py`
- Full Train config：`configs/sft/experiments/static_sud_mera_faithful_qwen35_interest_sid_21600_t005_v1.yaml`
- Full Dev evaluator config：`configs/id_token/static_sud_mera_faithful_qwen35_interest_sid_t005_v1.json`
- Gate-C runner：`scripts/amlt/sft/run_mera_gnip_smoke.sh`
- Gate-C AMLT config：`configs/amlt/static_sud_mera_faithful_qwen35_interest_sid_smoke_0831_v1.yaml`
- Gate-D/E AMLT config：`configs/amlt/static_sud_mera_faithful_qwen35_ddp_eval_smoke_0831_v1.yaml`
- Full AMLT config：`configs/amlt/static_sud_mera_faithful_qwen35_full_0831_v1.yaml`
- 7-condition Full AMLT config：`configs/amlt/static_sud_mera_qwen35_mean_ablation_full_0901_v1.yaml`
- representative ablation smoke AMLT config：`configs/amlt/static_sud_mera_qwen35_mean_ablation_smoke_0901_v1.yaml`
- ablation runner：`scripts/amlt/sft/run_mera_gnip_ablation.sh`
- frozen AMLT upload root：`amlt_inputs/mera_gnip_faithful_qwen35_smoke_0831_v1/`
- frozen Full AMLT upload root：`amlt_inputs/mera_gnip_faithful_qwen35_full_0831_v1/`
- frozen 7-condition upload root：`amlt_inputs/mera_gnip_qwen35_mean_ablation_0901_v1/`
- local tests：`tests/test_build_mera_relations.py`、
  `tests/test_mera_regularization.py`、`tests/test_mera_gnip_config.py`
- remote experiment/job：`mera-gnip-faithful-qwen35-t005-smoke-0831-v1` /
  `qwen35-mera-faithful-t005-1gpu-smoke`，terminal `pass`
- remote result root：`projects/sft4safety/amlt-results/7211862654.48557-2e90064b-19df`
- downloaded Gate-C readback：`tmp/mera-faithful-smoke-readback-0831-v1/mera-gnip-faithful-qwen35-t005-smoke-0831-v1/qwen35-mera-faithful-t005-1gpu-smoke/`
- Gate-D/E remote experiment/job：`mera-gnip-faithful-qwen35-ddp-eval-smoke-0831-v1` /
  `qwen35-mera-faithful-t005-ddp-eval-smoke`，terminal `pass`
- Gate-D/E remote result root：`projects/sft4safety/amlt-results/7211860866.64580-ddf31929-1956`
- downloaded Gate-D/E readback：`tmp/mera-faithful-ddp-eval-smoke-readback-0831-v1/`
- Full remote experiment/job：`mera-gnip-faithful-qwen35-full-0831-v1` /
  `qwen35-mera-faithful-t005-full`，terminal `pass`
- Full backend/result identity：`tidy_pea_lxpmvjc604` /
  `projects/sft4safety/amlt-results/7211858877.13708-fce45228-f8a5`
- downloaded Full key-artifact readback：`tmp/mera-faithful-full-readback-0901.d8ocKt/`
- representative ablation smoke experiment/job：
  `mera-gnip-qwen35-mean-ablation-bi-full-smoke-0901-v1` /
  `qwen35-mera-bi-mean-full-ddp-eval-smoke`，terminal `pass`
- representative ablation smoke result root：
  `projects/sft4safety/amlt-results/7211801142.38160-8997a5ec-05f9`
- downloaded representative smoke readback：
  `tmp/mera-mean-ablation-bi-full-smoke-readback-0901-v1/`
- remaining-seven Full experiment：`mera-gnip-qwen35-mean-ablation-full-0901-v1`；
  终态 `7/7 pass`
- downloaded remaining-seven key-artifact readback：
  `tmp/mera-mean-ablation-full-readback-0902-v1/`
