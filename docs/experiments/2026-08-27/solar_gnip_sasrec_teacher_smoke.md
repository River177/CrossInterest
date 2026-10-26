# SOLAR-GNIP：SASRec 与 GPT-5.5 Teacher Smoke Gate

## 1. 研究问题（Research Questions）

本次实验检验两个问题：

1. 单阶段 `SASRec-GNIP` 在 108 类 full-softmax 下，能否为真实新兴趣提供有效的 Top-20 候选集？主指标为 Candidate Recall@20，并同时记录 impression Hit@20。
2. 在相同 Top-20 候选集上，`GPT-5.5 Profile → Rerank` 是否优于 SASRec 原始排序？比较指标为 Recall、NDCG、Hit@1/5/10/20 和 MRR@20。

比较条件为 `SASRec-GNIP` 与 `GPT-5.5 Rerank`。本次只完成 100 个 Train source 的 Teacher smoke gate；`SOLAR-GNIP-NoAugment` 和 `SOLAR-GNIP-FullSUN` 学生模型尚未训练。

## 2. 训练数据构造（Training Data Construction）

### 2.1 SASRec-GNIP

数据来自冻结的 Behavior+Interest 联合视图。Train 有 21,600 个互异用户，108 个目标兴趣各 200 行；每行保留最多 50 个按时间排序且允许重复的历史兴趣。目标兴趣保证不在该行历史中。内部选择集按目标类确定性分层为 19,440/2,160，每类恰好 180/20，随机种子为 42。

SASRec 使用 125 类历史输入词表和 108 类输出词表。模型为 hidden size 256、2 blocks、2 heads、dropout 0.2；使用 Adam、学习率 1e-4、batch size 128。每 5 epoch 验证，按 Interest-Macro Recall@20、Interest-Macro NDCG@10、较早 epoch 的顺序选取 checkpoint；连续 10 次验证无改进则停止。选出 epoch 后从头在全部 21,600 行训练相同 epoch 数。候选导出不强插真实 target，也不屏蔽历史兴趣。

### 2.2 GPT-5.5 弱监督 smoke

从 21,600 个 Train candidates 中按 `SHA256(seed=42, example_id)` 确定性选取 100 行。每个 source 依次进行一次 target-blind 六字段 Profile 调用和一次 Top-20 Rerank 调用。Rerank 的四个 ICL 示例来自其他 Train 用户；positive 是该示例实际观察到的新兴趣，negative 是该示例历史中已有兴趣。当前 source 的真实 target 不进入 Profile 输入，也不作为 Rerank 的显式标签。

| condition | source rows | logical calls | valid rows | role |
|---|---:|---:|---:|---|
| Global calibration | — | 2 | 2 | 冻结 Profile 字段含义与排序原则 |
| GPT-5.5 Profile | 100 | 100 | 100 | target-blind 用户画像 |
| GPT-5.5 Rerank | 100 | 100 | 100 | SASRec Top-20 完整重排 |
| Student SFT | 0 | 0 | 0 | 尚未开始 |

### 2.3 真实弱监督样例

Provenance：`train:997916:50:behavior_interest`。

English input（原始输入）：

> The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":
>
> 1. [Trivia & Puzzles] Can you spot the hidden gift in this brainteaser in under 11 seconds?
> 2. [Cruises] A lawyer reveals a legal nightmare you can face on a cruise ship
> 3. [Courts & Legal Cases] Kevin Spacey Won't Be Charged in Sexual Assault Case After Accuser Dies
> 4. [Household Cleaning & Laundry] This Simple Hack Will Keep Spiders Out of Your House
>
> Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Begin your response with "New topic: " and provide only the category name after it.

中文翻译：

> 以下按时间顺序列出我最近浏览的新闻标题及对应主题类别，每项格式为“[主题类别] 标题”：
>
> 1. [益智与谜题] 你能在 11 秒内找出脑筋急转弯中隐藏的礼物吗？
> 2. [邮轮] 一位律师揭示了乘坐邮轮时可能遇到的法律噩梦
> 3. [法院与法律案件] 指控者死亡后，Kevin Spacey 不会因性侵案被起诉
> 4. [家居清洁与洗衣] 这个简单技巧能让蜘蛛远离你的房子
>
> 根据标题及其主题类别，预测我接下来可能感兴趣的一个新主题类别。预测类别不应出现在浏览历史的主题类别中。回答须以“New topic: ”开头，之后只给出类别名称。

GPT-5.5 Profile output（原文）：

```json
{
  "explicit_preference": "No explicit likes, follows, searches, ratings, subscriptions, or profile declarations are present in the observed history.",
  "exploratory_intention": "Shows exploratory behavior through heterogeneous topic hopping across entertainment, travel risk, legal news, and household advice, with weak commitment to any one category.",
  "implicit_preference": "Recent clicks span puzzles, cruise-related legal advice, celebrity legal case news, and household pest-cleaning tips; no repeated category, suggesting broad casual browsing.",
  "novelty_preference": "High novelty evidence: every observed click is in a different category, indicating willingness to sample unrelated topics rather than repeat established interests.",
  "specific_intention": "No concrete repeated short-term intent is strongly supported; only single observed clicks in Trivia & Puzzles, Cruises, Courts & Legal Cases, and Household Cleaning & Laundry.",
  "vague_intention": "Near-term intent appears broadly informational and casual, mixing quick entertainment, practical advice, legal-interest news, and home-maintenance problem solving."
}
```

中文翻译：显式偏好为空；探索意图体现为跨娱乐、旅行风险、法律新闻和家居建议的异质跳转；隐式偏好覆盖谜题、邮轮法律建议、名人法律新闻和除虫清洁；新颖性偏好较高；没有得到重复行为支持的具体短期意图；模糊意图偏向轻量信息、实用建议、法律兴趣和居家维护。

GPT-5.5 Rerank output（完整顺序）：

```text
Gardening & Landscaping; Home Appliances & Technology; Home Improvement & Repair;
Crime & Public Safety; Comics & Graphic Narratives; Crafts & Collecting;
Adventure & Nature Travel; Shopping & Gift Ideas; Cars & Driving;
Restaurants & Dining; Cooking & Recipes; Weather Forecasts & Seasonal Conditions;
Workplace Life; Archaeology & Human Origins; Architecture & Residential Design;
Aging & Longevity; Companies & Corporate Strategy; Personal Finance;
Baking & Desserts; Household Cleaning & Laundry
```

中文翻译：排序首先选择园艺与景观、家电与技术、家居改善与维修；真实观察 target `Architecture & Residential Design`（建筑与住宅设计）位于第 15 名。

## 3. 测试设计（Testing）

### 3.1 模型与运行位置

- SASRec-GNIP：本地 Apple MPS，PyTorch 2.12.0；无采样，直接对 108 类 logits 排序。
- Teacher：TRAPI `redmond/interactive`，deployment `gpt-5.5_2026-04-24`，API version `2024-10-21`，`reasoning_effort=none`，不传 temperature，并发 4，checkpoint size 64，最多 3 次尝试。
- 两次校准和每个 source 的 Profile/Rerank 都是远程推理；Qwen3.5 student 尚未运行。

### 3.2 测试数据来源

Full Dev 来自冻结的 `dev_full_grouped.parquet`：31,392 impressions、27,082 users、37,728 target occurrences、104 个受支持兴趣；Train 和 Dev 用户交集为 0。Teacher smoke 使用 Train candidates 的 100 行确定性子集，因此其数值仅作为 pipeline gate，不替代 Full Dev 结果。

### 3.3 测试样例

第 2.3 节给出了完整真实输入、Profile 和 Rerank 输出。该例中真实 target 已进入 SASRec Top-20，但 GPT-5.5 将其排到第 15；它属于“合法输出但排序未改善”的结果。没有观察到 schema error、API error 或 policy block 样例。

### 3.4 Teacher prompt 与合法性判定

Profile system prompt：

> Return only the requested target-blind profile JSON.

Profile user template 要求只使用 observed History，禁止预测 topic、获取 held-out target 或提及 candidate ID，并返回六个且仅六个字符串字段：`implicit_preference`、`explicit_preference`、`novelty_preference`、`vague_intention`、`specific_intention`、`exploratory_intention`；每字段不超过 30 个英文词。

Rerank system prompt：

> Return only the complete candidate permutation JSON.

Rerank user template要求只重排给定的 20 个 ID，使用 History、target-blind Profile、SASRec score、冻结原则和四个其他 Train 用户示例；不得增删、重复、改名或解释候选，输出 schema 为：

```json
{"ranking": ["ID1", "...", "ID20"]}
```

合法结果必须满足：Profile 六字段完整且每字段不超过 30 词；Rerank 是候选 ID 集合的无重复完整排列。API error、policy block、非法 JSON、字段缺失、重复/缺失/候选外 ID 均为非结果，不进入有效分母。上述模板和 validator 的可执行版本见 `solar-gnip-gpt55-profile-rerank-v1`。

对每行正例集合 `P` 和前 `k` 个预测 `R_k`：`Recall@k = |P ∩ R_k| / |P|`；`Hit@k = 1[|P ∩ R_k| > 0]`；NDCG 使用二值相关性和标准对数折扣。Teacher smoke 的 paired 比较只使用同一批 100 个成功 source。

## 4. 实验结果（Results）

### 4.1 SASRec 候选覆盖率

| split | examples | target occurrences | Candidate Recall@20 | Hit@20 | target 不在候选的 rows |
|---|---:|---:|---:|---:|---:|
| Train（in-sample） | 21,600 | 21,600 | 0.4891 | 0.4891 | 11,036 |
| Dev-1K | 1,000 | 1,181 | 0.4200 | 0.4660 | 534 |
| Full Dev | 31,392 | 37,728 | 0.4173 | 0.4651 | 16,793 |

内部分层验证在 epoch 5 达到最佳 Interest-Macro Recall@20 = 0.3412、Interest-Macro NDCG@10 = 0.1074；连续 10 次验证未改善后在 epoch 55 停止，再用全部 Train 重训 5 epochs。

### 4.2 Teacher smoke：同一 100 行 paired 结果

| ranking | Top-1 | Top-5 | Top-10 | Top-20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| SASRec 原始顺序 | 0.0400 | 0.2300 | 0.3400 | 0.5500 | 0.1333 |
| GPT-5.5 Rerank | 0.0400 | 0.2000 | 0.2600 | 0.5500 | 0.1273 |
| GPT-5.5 − SASRec | 0.0000 | -0.0300 | -0.0800 | 0.0000 | -0.0059 |

在 target 已进入候选集的 55 行上：

| ranking | Top-1 | Top-5 | Top-10 | MRR@20 |
|---|---:|---:|---:|---:|
| SASRec 原始顺序 | 0.0727 | 0.4182 | 0.6182 | 0.2423 |
| GPT-5.5 Rerank | 0.0727 | 0.3636 | 0.4727 | 0.2315 |

完整性检查：2/2 calibration、100/100 Profile、100/100 Rerank 均一次成功；202 个逻辑调用对应 202 次实际尝试；schema/API/policy 错误均为 0；GPT-5.5 Top-1 选择历史已有兴趣为 0/100。第二次运行恢复 100/100 source，新增远程调用为 0。鉴于 smoke 中 Rerank 未改善点击 target 排序，21,600 行全量 Teacher generation 维持未启动状态。

本次 smoke 未计算置信区间；100 行仅用于 stop gate，不能作为论文主结果。

## 5. 数据与产物路径（Artifact Paths）

- 联合 GNIP 数据：[`data/training/sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1`](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1)
- SASRec 配置：[`configs/solar_gnip/sasrec_gnip_v1.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/solar_gnip/sasrec_gnip_v1.json)
- Teacher smoke 配置：[`configs/solar_gnip/gpt55_teacher_smoke_v1.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/solar_gnip/gpt55_teacher_smoke_v1.json)
- Teacher full 配置（未运行）：[`configs/solar_gnip/gpt55_teacher_full_v1.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/solar_gnip/gpt55_teacher_full_v1.json)
- SASRec 可执行实现：[`src/new_interest/experiments/sasrec_gnip.py`](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/experiments/sasrec_gnip.py)
- Teacher 可执行实现：[`src/new_interest/data/generate_solar_gnip_teacher.py`](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/data/generate_solar_gnip_teacher.py)
- SASRec 正式 artifact：[`artifacts/solar_gnip/sasrec_gnip_v1`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/solar_gnip/sasrec_gnip_v1)
- SASRec 独立 readback：[`artifacts/solar_gnip/sasrec_gnip_v1_verification.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/solar_gnip/sasrec_gnip_v1_verification.json)
- Teacher smoke records：[`artifacts/solar_gnip/gpt55_teacher_smoke_v1/teacher_records.parquet`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/solar_gnip/gpt55_teacher_smoke_v1/teacher_records.parquet)
- Teacher smoke metrics：[`artifacts/solar_gnip/gpt55_teacher_smoke_v1/metrics.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/solar_gnip/gpt55_teacher_smoke_v1/metrics.json)
- Teacher calibration：[`artifacts/solar_gnip/gpt55_teacher_smoke_v1/calibration.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/solar_gnip/gpt55_teacher_smoke_v1/calibration.json)
- Teacher 独立 readback：[`artifacts/solar_gnip/gpt55_teacher_smoke_v1_verification.json`](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/solar_gnip/gpt55_teacher_smoke_v1_verification.json)
