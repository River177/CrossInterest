# Wang et al. Stage-A 三数据集复现

## 1. 研究问题（Research Questions）

本实验在 Amazon Arts、Amazon Grocery 和 CrossInterest MIND 上适配复现
Wang et al.（RecSys 2024，DOI `10.1145/3640457.3688161`）的高层语言策略，
不包含低层 item policy、SASRec 或簇内 item grounding。研究问题为：

1. 对每个目标兴趣簇保留频率最高的 10 个上下文簇对并进行均衡 SFT 后，模型能否
   稳定输出预定义类别，并在 held-out 用户记录上预测历史未出现的新兴趣？
2. 在完全相同的数据、prompt、候选空间和评测协议下，`Qwen/Qwen3.5-9B` 与
   `Qwen/Qwen2-1.5B` 的受控生成能力和新兴趣预测能力有何差异？
3. 方法在 Arts、Grocery 与 MIND 新闻兴趣空间上的类别覆盖率、输出熵和历史重叠
   违规率是否一致？

主指标为自由生成的 Match Rate 与 Recall@1。受限标签 likelihood 排序的
Recall/NDCG@5/10、Interest-Macro 指标、覆盖率和输出熵为补充指标。

## 2. 训练数据构造（Training Data Construction）

### 2.1 来源与规则

Amazon 使用 MeRA 官方仓库 commit
`c07bdbe57bf82e5c938b478035baee78d630b17c` 发布的 Arts/Grocery
Train/Validation/Test 类别序列；这里只借用公开数据，不使用 MeRA 方法。MIND 使用
冻结的 `new_interest_static_user_disjoint_v1`：Train 每用户一个目标，Dev 删除所有
原始 Train 用户，History 只包含完整原始 behavior History，不追加当前 impression
点击，也不截断。

对每条记录，从完整 History 中按时间顺序抽取最近两个不重复类别 `(C1,C2)`；目标
`CL` 必须不在完整 History。对每个目标类别统计有序 `(C1,C2)` 频率，按频率降序、
词典序打破平局，保留前 10 个不同上下文对。只有至少拥有 10 个不同上下文对的类别
进入主训练/主评测支持集；其余 Amazon 类别保留在 full-taxonomy 诊断中，不复制或
合成样本。本报告按最终确认范围只运行 balanced 条件，不运行 Random-SFT 或
Zero-shot 对照。

| dataset | K=2 eligible Train | full labels | labels with ≥10 pairs | balanced | primary held-out |
|---|---:|---:|---:|---:|---:|
| Amazon Arts | 18,589 | 143 | 89 | 890 | Validation 2,155；Test 2,087 |
| Amazon Grocery | 32,724 | 172 | 111 | 1,110 | Validation 4,033；Test 3,901 |
| MIND | 571,718 | 108 outputs | 108 | 1,080 | Dev 30,266 impressions |

MIND Dev 的 30,266 个 impression 包含 36,381 个 acceptable target occurrences。
MIND prompt 的两个历史类别仍可来自完整 125 类 History 词表，输出候选固定为既有
108 类目标空间。

### 2.2 真实训练样例

以下均为 SFT target records，不是模型生成结果。

**Amazon Arts**，来源 `arts:train:2386`，标签
`Accessories, Hardware & Tools`：

> The art or craft categories I interacted with most recently are in the
> following clusters:
> Cluster 1: {Drawing}
> Cluster 2: {Painting}.
> Each cluster is described with a predefined category name. With less than 30
> words, generate a new and different art or craft categories cluster I will
> interact with next. The new cluster must differ from both clusters above.
> Copy exactly one predefined cluster name and use the prefix "New cluster: ".

目标输出：`New cluster: Accessories, Hardware & Tools`

中文：最近交互的两个艺术或手工类别簇是 Drawing 和 Painting。生成一个与二者不同、
且严格复制预定义类别名的新簇，使用 `New cluster: ` 前缀。目标输出为
“配件、五金与工具”。

**Amazon Grocery**，来源 `grocery:train:17641`，标签 `Appetizers`：

> The grocery categories I interacted with most recently are in the following
> clusters:
> Cluster 1: {Bottled Beverages, Water & Drink Mixes}
> Cluster 2: {Cooking & Baking}.
> Each cluster is described with a predefined category name. With less than 30
> words, generate a new and different grocery categories cluster I will interact
> with next. The new cluster must differ from both clusters above. Copy exactly
> one predefined cluster name and use the prefix "New cluster: ".

目标输出：`New cluster: Appetizers`

中文：最近交互的两个食品类别簇是瓶装饮料/水/冲调饮品和烹饪/烘焙。生成一个与二者
不同的预定义食品簇。目标输出为“开胃菜”。

**MIND**，来源 `train:865078:0`，标签
`Accidents & Emergency Response`：

> The news topic categories I interacted with most recently are in the following
> clusters:
> Cluster 1: {Celebrity & Creator Culture}
> Cluster 2: {Crime & Public Safety}.
> Each cluster is described with a predefined category name. With less than 30
> words, generate a new and different news topic categories cluster I will
> interact with next. The new cluster must differ from both clusters above. Copy
> exactly one predefined cluster name and use the prefix "New cluster: ".

目标输出：`New cluster: Accidents & Emergency Response`

中文：最近两个新闻兴趣簇是名人与创作者文化、犯罪与公共安全。目标是预测新的
“事故与应急响应”兴趣簇。

## 3. 测试设计（Testing）

### 3.1 模型与运行位置

| model | frozen revision | worker stack | LoRA learning rate |
|---|---|---|---:|
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Transformers 5.14.1 / PEFT 0.18.0 | 2e-5 |
| Qwen2-1.5B | `8a16abf2848eda07cc5253dec660bf1ce007ad7a` | Transformers 4.53.3 / PEFT 0.16.0 | 1e-4 |

两模型均使用 BF16、LoRA rank 16/alpha 32、所有线性层、effective batch 16、7 epochs、
seed 42、`enable_thinking=false`。7 epochs 约对应原论文 7,610 条数据、batch 16、
3,000 steps 的 6.3 次数据曝光量；这里固定为整数 epoch，不直接照搬绝对 step 数。
运行位置为 Feeds：Qwen2 正式任务使用 `80G4-A100`，Qwen3.5 使用
`80G8-A100`。训练仍固定为单 GPU 语义；自由生成和受限标签排序分别按 4/8 个逻辑
分片并独立合并。

### 3.2 测试数据来源

Amazon 使用官方 Validation 进行受控生成门控，只有 balanced 条件 Match Rate
达到 90% 才继续官方 Test。MIND 没有另造 Test，沿用完整 user-disjoint Dev。
主结果使用 train-supported 类别/target；full-taxonomy 结果单列。自由生成采用 greedy
decode、最多 32 个新 token，完整 response 必须严格匹配 `New cluster: <label>`。

### 3.3 测试样例

以下均来自正式 held-out 预测，不是训练样本或手工构造案例。

**Amazon Arts，`arts:test:1`：**

> The art or craft categories I interacted with most recently are in the
> following clusters:
> Cluster 1: {Wire}
> Cluster 2: {Beads & Bead Assortments}.
> Each cluster is described with a predefined category name. With less than 30
> words, generate a new and different art or craft categories cluster I will
> interact with next. The new cluster must differ from both clusters above.
> Copy exactly one predefined cluster name and use the prefix "New cluster: ".

中文：最近两个艺术/手工类别簇是“金属线”和“珠子及珠子组合”。请预测一个与二者
不同的新类别，并严格复制一个预定义类别名。唯一 acceptable target 为
`Jewelry Making Tools & Accessories`。Qwen2 输出
`New cluster: Jewelry Making Tools & Accessories`，命中；Qwen3.5 输出
`New cluster: Jewelry Making Display & Packaging Supplies`，格式正确但未命中。

**Amazon Grocery，`grocery:test:1083`：**

> The grocery categories I interacted with most recently are in the following
> clusters:
> Cluster 1: {Herbs, Spices & Seasonings}
> Cluster 2: {Cooking & Baking}.
> Each cluster is described with a predefined category name. With less than 30
> words, generate a new and different grocery categories cluster I will interact
> with next. The new cluster must differ from both clusters above. Copy exactly
> one predefined cluster name and use the prefix "New cluster: ".

中文：最近两个食品类别簇是“香草、香料与调味料”和“烹饪与烘焙”。acceptable
targets 为 `Meals & Entrees`、`Cookies`、`Chocolate Candy`、`Coffee`。Qwen2
输出 `New cluster: Dried Grains & Rice`，未命中；Qwen3.5 输出
`New cluster: Coffee`，命中。

**MIND，`mind:dev:10145`：**

> The news topic categories I interacted with most recently are in the following
> clusters:
> Cluster 1: {Accidents & Emergency Response}
> Cluster 2: {Celebrity & Creator Culture}.
> Each cluster is described with a predefined category name. With less than 30
> words, generate a new and different news topic categories cluster I will
> interact with next. The new cluster must differ from both clusters above. Copy
> exactly one predefined cluster name and use the prefix "New cluster: ".

中文：最近两个新闻兴趣簇是“事故与应急响应”和“名人与创作者文化”。acceptable
targets 为 `Public Health` 与 `Food Culture & Trends`。Qwen2 输出
`New cluster: Home Improvement & Repair`，未命中；Qwen3.5 输出
`New cluster: Public Health`，命中。两个输出都不与完整历史重叠。

### 3.4 Judge 与指标过程

本实验没有 LLM judge。Match Rate 是完整自由生成 response 精确映射到预定义类别
词表的比例；Recall@1 在多正例记录上为命中目标数除以 acceptable target 数；
Novelty Compliance 要求生成标签不在完整 History。受限排序对每个候选类别名称的
类别 token 取 mean log-probability，按分数排序计算 Recall、Hit、MRR 和 NDCG，且不
按 History 过滤预测。所有重复 rank、未知标签、缺失 impression、identity/hash 不一致
均使评测失败。

## 4. 实验结果（Results）

### 4.1 完整性与重载验收

六个 Full 任务均为 AMLT `pass` 并带 `full-validation_success.json`。本地下载约
2.8 GB 后，122/122 个成功标记所列文件哈希一致；六个 fresh reload 的最大 logits
绝对误差均为 0。Qwen2 每个 adapter 的 392 个 tensors（18,464,768 values）与
Qwen3.5 的 496 个 tensors（43,278,336 values）均 bit-exact，无 missing、unexpected
或 mismatched keys。36 个生成/排序视图均从合并预测重新计算，未发现指标差异。

| model | dataset | train | first-5 loss mean | last-5 loss mean | reload max error | state |
|---|---|---:|---:|---:|---:|---|
| Qwen2-1.5B | Arts | 890 | 2.1819 | 0.4493 | 0 | exact |
| Qwen2-1.5B | Grocery | 1,110 | 2.1105 | 0.4723 | 0 | exact |
| Qwen2-1.5B | MIND | 1,080 | 2.0196 | 0.4836 | 0 | exact |
| Qwen3.5-9B | Arts | 890 | 1.9737 | 0.4747 | 0 | exact |
| Qwen3.5-9B | Grocery | 1,110 | 1.7486 | 0.5027 | 0 | exact |
| Qwen3.5-9B | MIND | 1,080 | 1.7732 | 0.4973 | 0 | exact |

### 4.2 主结果：自由生成

Amazon 报告通过 Validation 90% Match gate 后的官方 Test supported 结果；MIND
报告完整 user-disjoint Dev。`Hit@1` 是至少命中一个 acceptable target 的 impression
比例；多正例 `Recall@1` 还除以该 impression 的 target 数，因此数值更低。

| model | dataset/split | n | Match | Hit@1 | Recall@1 | Macro R@1 | Novelty | generated coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-1.5B | Arts Test | 2,087 | 2,075/2,087 (99.4250%) | 131/2,087 (6.2770%) | 2.7446% | 2.9190% | 89.2669% | 54/89 (60.67%) |
| Qwen3.5-9B | Arts Test | 2,087 | 2,054/2,087 (98.4188%) | 113/2,087 (5.4145%) | 2.2225% | 3.2672% | 91.0877% | 47/89 (52.81%) |
| Qwen2-1.5B | Grocery Test | 3,901 | 3,901/3,901 (100%) | 95/3,901 (2.4353%) | 0.8325% | 0.9300% | 97.3340% | 61/111 (54.95%) |
| Qwen3.5-9B | Grocery Test | 3,901 | 3,901/3,901 (100%) | 93/3,901 (2.3840%) | 0.8528% | 1.2849% | 97.4622% | 38/111 (34.23%) |
| Qwen2-1.5B | MIND Dev | 30,266 | 29,909/30,266 (98.8205%) | 183/30,266 (0.6046%) | 0.4706% | 1.4457% | 93.0747% | 83/108 (76.85%) |
| Qwen3.5-9B | MIND Dev | 30,266 | 30,077/30,266 (99.3755%) | 430/30,266 (1.4207%) | 1.1700% | 1.3468% | 91.4062% | 91/108 (84.26%) |

Match 只衡量完整响应是否严格落在预定义词表，不等于预测正确。两模型的 Match 均接近
或达到 100%，但自由生成 Recall@1 仍较低。Arts 上 Qwen2 的 micro Recall@1 高
0.5221 个百分点；Grocery 两者接近；MIND 上 Qwen3.5 高 0.6994 个百分点。

### 4.3 主结果：受限标签 likelihood 排序

| model | dataset/split | n | R@5 | R@10 | NDCG@5 | NDCG@10 | Macro R@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2-1.5B | Arts Test | 2,087 | 10.1899% | 16.8975% | 8.9434% | 11.3996% | 16.5773% |
| Qwen3.5-9B | Arts Test | 2,087 | 9.5674% | 16.2432% | 8.6426% | 11.0928% | 15.6299% |
| Qwen2-1.5B | Grocery Test | 3,901 | 6.6351% | 14.5131% | 5.2567% | 8.3489% | 9.4065% |
| Qwen3.5-9B | Grocery Test | 3,901 | 8.5710% | 16.9028% | 7.2334% | 10.3736% | 9.3627% |
| Qwen2-1.5B | MIND Dev | 30,266 | 2.2983% | 4.1103% | 1.4806% | 2.0960% | 9.8129% |
| Qwen3.5-9B | MIND Dev | 30,266 | 2.8755% | 5.1169% | 1.8724% | 2.6309% | 11.5314% |

Qwen3.5 相对 Qwen2 的 R@10 差值分别为 Arts -0.6543、Grocery +2.3897、MIND
+1.0066 个百分点；NDCG@10 差值分别为 -0.3068、+2.0247、+0.5349 个百分点。
因此 9B 模型在 Grocery 与 MIND 的受限排序上更高，但在 Arts 上没有超过 1.5B。

### 4.4 Macro Recall/NDCG@10/@20 补充评测

Top-20 eval-only 直接复用上述六个已验收 adapter，不重新训练。六个任务均为 AMLT
`pass`；本地回读验证 48/48 个成功标记哈希，72,508/72,508 条预测均含 20 个唯一
合法标签，且每条预测的前十名标签与分数均与原 Top-10 结果完全一致。下表因此同时
给出复核后的 @10 与新增 @20。

**Qwen3.5-9B**

| Dataset | Macro Recall@10 | Macro NDCG@10 | Macro Recall@20 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| MIND | 11.5314% | 5.4491% | 20.7885% | 7.7729% |
| Amazon Arts | 15.6299% | 7.7493% | 29.4982% | 11.2219% |
| Amazon Grocery | 9.3627% | 4.4189% | 18.9523% | 6.8283% |

**Qwen2-1.5B**

| Dataset | Macro Recall@10 | Macro NDCG@10 | Macro Recall@20 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| MIND | 9.8129% | 5.0439% | 20.1015% | 7.5913% |
| Amazon Arts | 16.5773% | 8.4567% | 28.8614% | 11.5383% |
| Amazon Grocery | 9.4065% | 4.3053% | 18.9584% | 6.6887% |

### 4.5 Full-taxonomy 与覆盖诊断

Amazon full-taxonomy Test 将候选空间扩展到 Arts 143 类、Grocery 172 类，并纳入
少量 supported 集之外的 held-out rows；它是诊断结果，不与 supported Test 做严格
配对显著性比较。

| model | dataset | supported Test R@10 | full-taxonomy Test R@10 | full-taxonomy NDCG@10 |
|---|---|---:|---:|---:|
| Qwen2-1.5B | Arts | 16.8975% | 16.8412% | 11.3659% |
| Qwen3.5-9B | Arts | 16.2432% | 15.9595% | 11.0055% |
| Qwen2-1.5B | Grocery | 14.5131% | 14.4361% | 8.3034% |
| Qwen3.5-9B | Grocery | 16.9028% | 16.8759% | 10.3530% |

Qwen3.5 的 Grocery 自由生成只覆盖 38/111 个支持类别，低于 Qwen2 的 61/111，
尽管其 micro Recall 与受限排序更高；这是输出集中度与预测指标的测量差异，不能仅凭
本实验解释为模型规模的因果效应。MIND 上则是 Qwen3.5 同时具有更高的生成覆盖
（91/108 对 83/108）与更高的 micro Recall。

## 5. 数据与产物路径（Artifact Paths）

- 论文：<https://doi.org/10.1145/3640457.3688161>；
  <https://arxiv.org/abs/2405.16363>
- 冻结实验矩阵：`configs/wang_stage_a/experiment_matrix_v1.yaml`
- 数据构建器：`src/new_interest/data/build_wang_stage_a_dataset.py`
- 生成评测：`src/new_interest/evaluation/evaluate_wang_stage_a_generation.py`
- 受限排序评测：`src/new_interest/evaluation/evaluate_text_label_ranking.py`
- adapter reload：`src/new_interest/training/text_adapter_reload.py`
- 本地独立审计：`src/new_interest/experiments/audit_wang_stage_a_results.py`
- Top-20 本地独立审计：
  `src/new_interest/experiments/audit_wang_stage_a_top20_results.py`
- 冻结处理数据：`data/training/wang_stage_a_balanced_transition_v1/`
- Qwen2 Full AMLT 配置：`configs/amlt/wang_stage_a_main_validation_0902_v1.yaml`
- Qwen3.5 Full AMLT 配置：
  `configs/amlt/wang_stage_a_qwen35_main_validation_0902_v2.yaml`
- Qwen2 Full 实验：`wang-stage-a-qwen2-main-validation-0902-v1`，三个 jobs
  `qwen2-{arts,grocery,mind}-balanced-validation`
- Qwen3.5 Full 实验：`wang-stage-a-qwen35-main-validation-0902-v2`，三个 jobs
  `qwen35-{arts,grocery,mind}-balanced-validation-v2`
- Full 本地回读：`artifacts/wang-stage-a/full-readback-20260903/`
- 本地独立审计结果：
  `artifacts/wang-stage-a/full-readback-20260903/local_audit.json`
- Top-20 smoke：`wang-stage-a-top20-smoke-0903-v2` /
  `qwen2-arts-top20-smoke`
- Top-20 Full 实验：`wang-stage-a-top20-qwen2-0903-v1` 与
  `wang-stage-a-top20-qwen35-0903-v1`
- Top-20 Full 本地回读：
  `artifacts/wang-stage-a/top20-full-readback-20260903/`
- Top-20 Full 本地审计：
  `artifacts/wang-stage-a/top20-full-readback-20260903/local_audit.json`
- Qwen2 通过的 smoke：
  `wang-stage-a-qwen2-generation-stop-recovery-0902-v3` /
  `qwen2-mind-generation-stop-v3`
- Qwen2 本地下载回读：`tmp/wang-stage-a-qwen2-smoke-v3-readback/`
- Qwen3.5 recovery：
  `wang-stage-a-qwen35-reload-recovery-smoke-0902-v2` /
  `qwen35-mind-balanced-reload-v2`
