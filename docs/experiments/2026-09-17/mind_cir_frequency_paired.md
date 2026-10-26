# CIR 频率对照：单次兴趣与高频兴趣重建 → NI

状态：两组训练及 Full Dev 已完成并通过独立验收。用户已选择保留同一批配对用户，在各频率组内随机选类，不做类别配平。

## 研究问题

在 Qwen3.5-2B 的 CIR 1 epoch → NI 3 epochs 流程中，重建 H50 中仅出现 1 次的兴趣，和重建出现至少 4 次的兴趣，对最终新兴趣排名有何不同？主比较为这两组；已有 NI-only、原随机 CIR1→NI3 作为参考。原随机 CIR 使用 17,271 条样本、540 次 CIR 更新，与本对照的 CIR 训练量不同。

本实验比较 mask 策略的实际迁移效果。出现频率是稳定性的代理；类别分布与删除量也随策略变化，因此结果不能单独归因于兴趣稳定性。本轮不增加未来复现率分析、多 seed 或额外随机对照训练。

## 训练数据构造

沿用原 21,600 位 NI 训练用户、结构化历史及 108 类 SID 映射。先截取最后 50 条行为，以 subcategory 的行为出现次数计数，重复新闻记录保留计数。单次组的候选频率恰为 1，高频组的候选频率 ≥4。候选必须属于原 108 类输出词表，且整类删除后至少剩余 5 条行为及另一类别。

选择同时拥有两种合格候选的用户。两组均沿用原 seed=42 和用户/impression/类别 SHA-256 排序，在各自候选池中取哈希最小的类别；每个用户每组一个固定视图，不按 epoch 重抽。未来 NI 标签及含未来目标索引的 source sample ID 均不参与选择。删除所选类别的全部标题和类别记录，保持顺序，不回填旧历史。

受保护输入使用原 NI/OI 训练、Full Dev 和独立 Validation 的模型可见历史。若任一组与受保护输入重复，则两组共同排除该用户。此次 9,883 对中未出现这种排除；Dev/Validation 用户交集为 0。原 NI/Dev/mapping 文件在两组中保持字节一致。

| 项目 | 单次组 singleton | 高频组 repeated |
|---|---:|---:|
| 配对前具备合格候选的用户 | 17,256 | 9,885 |
| 最终 CIR 用户／样本 | 9,883 | 9,883 |
| 实际选中目标类别数 | 108 | 75 |
| 平均删除行为数 | 1.00 | 6.44 |
| 平均保留行为数 | 32.09 | 26.64 |
| CIR epochs／样本曝光／更新 | 1／9,883／309 | 1／9,883／309 |
| NI epochs／样本曝光／更新 | 3／64,800／2,025 | 3／64,800／2,025 |

两组目标类别频率分布的总变差距离为 0.61247（0 表示相同，1 表示完全不重叠）；这不是模型效果指标。高频组前五类为 Celebrity & Creator Culture、Crime & Public Safety、American Football、Government & Public Policy、Courts & Legal Cases。完整计数见类别分布 CSV。

## 训练与评估设计

模型固定 Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。两组分别从相同基座和 SID 初始化开始。CIR→NI 继承本组全部 LoRA/SID 参数，optimizer/scheduler 归零；NI 输入为原 21,600 条数据。seed=42、LoRA r=16/alpha=32/dropout=0.05、learning rate=2e-5、cosine schedule、warmup ratio=0.03、weight decay=0.01、max length=2048。

每组使用 Recall Standard 4×A100-80GB，microbatch=2、accumulation=4、global batch=32；BF16、非 reentrant gradient checkpointing、显式 Torch fallback、native full-vocabulary completion-only token-mean SFT。训练、sampler、断点恢复、序列化和评估代码均来自原已验收 CIR bundle-v3。新输入先执行原单卡/四卡、参数交接、精确重载与 64 条 Dev 检查，全部通过后自动进入正式训练。

Full Dev 固定 31,392 grouped impressions、37,728 个目标出现、104 个有支持类别，108 类 SID Top20 排名，不过滤历史类别。主要比较 Interest-Macro Recall@10/@20、NDCG@10/@20。远端检查完整预测覆盖、分片身份/哈希并独立重算指标，本地只回读紧凑统计、清单和哈希。权重、checkpoints、optimizer 状态及完整预测留在远端。

## 实验结果

两组训练及 Full Dev 已完成并通过独立验收。两组 CIR 均为 309 次更新、9,883 次样本曝光，NI 均为 2,025 次更新、64,800 次曝光；四分片各 7,848 条，共 31,392 个 impressions、37,728 个目标出现、104 个有支持类别。每阶段 373 个 tensor 和 logits 精确重载通过，参数交接与零初始化优化器/scheduler 身份一致。远端完整预测检查及本地逐类指标聚合重算通过。

指标为 Interest-Macro，单位为 %。

| 条件 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| NI-only（参考） | 20.3275 | 33.6553 | 10.1325 | 13.4663 |
| 原随机 CIR1→NI3（参考） | 21.2786 | 34.2609 | 10.8068 | 14.0725 |
| 单次兴趣 CIR1→NI3 | 20.6996 | 34.7380 | 10.5004 | 14.0358 |
| 高频兴趣 CIR1→NI3 | 21.4692 | 34.5215 | 10.9226 | 14.1874 |

高频组相对单次组的差值依次为 +0.7697、-0.2165、+0.4223、+0.1516 个百分点。本次高频组在 Recall@10、NDCG@10、NDCG@20 更高，单次组在 Recall@20 更高。单 seed 观察未检验统计显著性；类别分布及删除量不同，不能单独归因于兴趣稳定性。参考组的 CIR 样本量/更新量和 NI-only 的历史执行方式差异沿用前述比较边界。

平台原生完成时间（北京时间）：单次组 2026-09-18 02:47:49，高频组 2026-09-18 05:40:09。

## 真实配对训练样本

以下为用户 `U23171`、样本 `cir:train:242680:111` 的两种视图。response 为训练目标，非模型生成结果。原可见历史共 9 条；Golf 出现 1 次，American Football 出现 4 次。

### 单次组：Golf

原始 prompt：

```text
The following news browsing history has all entries from one topic category removed. The remaining entries are in chronological order, formatted as "[Topic category] Headline":

1. [American Football] Yardbarker's NFL Week 6 game-by-game analysis, grades
2. [Medicine & Medical Research] Sleep Apnea Effects Linked To Blindness
3. [Historical Events & Eras] Marines correct 74-year-old Iwo Jima error
4. [American Football] Rob Gronkowski compares Bill Belichick to Grinch: 'Always trying to take away the fun'
5. [Nutrition & Healthy Eating] 23 Foods That Help Lower High Blood Pressure
6. [Film] 16 of the Most Historically Inaccurate Movies Ever
7. [American Football] 32 things we learned from Week 8 of the 2019 NFL season
8. [American Football] Jimmy Garoppolo, 49ers find new offensive gear in win over Cardinals

Infer the removed topic category from the remaining observed interests and news items. This is historical interest reconstruction, not a prediction of the next click. Respond with only its single-token Interest ID in the exact format <interest_NNN>. Do not output the category name or any explanation.
```

中文翻译：

以下新闻浏览历史已移除某个主题类别的全部记录。剩余记录按时间顺序排列，格式为“[主题类别] 标题”：

1. [美式橄榄球] Yardbarker 对 NFL 第 6 周逐场比赛的分析和评分
2. [医学与医学研究] 睡眠呼吸暂停的影响与失明有关
3. [历史事件与时代] 海军陆战队纠正一项关于硫磺岛、延续了 74 年的错误
4. [美式橄榄球] Rob Gronkowski 将 Bill Belichick 比作格林奇：“总是想夺走乐趣”
5. [营养与健康饮食] 23 种有助于降低高血压的食物
6. [电影] 16 部最不符合历史事实的电影
7. [美式橄榄球] 从 2019 年 NFL 赛季第 8 周了解到的 32 件事
8. [美式橄榄球] Jimmy Garoppolo 和 49 人队在战胜红雀队时找到新的进攻档位

根据剩余的已观察兴趣和新闻推断被删除的主题类别。这是历史兴趣重建，不是预测下一次点击。只输出该类别的单 token 兴趣 ID，严格使用 <interest_NNN> 格式，不输出类别名或任何解释。

训练目标：`<interest_053>`（高尔夫）。

### 高频组：American Football

原始 prompt：

```text
The following news browsing history has all entries from one topic category removed. The remaining entries are in chronological order, formatted as "[Topic category] Headline":

1. [Medicine & Medical Research] Sleep Apnea Effects Linked To Blindness
2. [Golf] Lee Ann Walker receives 58 penalty strokes at Senior LPGA
3. [Historical Events & Eras] Marines correct 74-year-old Iwo Jima error
4. [Nutrition & Healthy Eating] 23 Foods That Help Lower High Blood Pressure
5. [Film] 16 of the Most Historically Inaccurate Movies Ever

Infer the removed topic category from the remaining observed interests and news items. This is historical interest reconstruction, not a prediction of the next click. Respond with only its single-token Interest ID in the exact format <interest_NNN>. Do not output the category name or any explanation.
```

中文翻译：

以下新闻浏览历史已移除某个主题类别的全部记录。剩余记录按时间顺序排列，格式为“[主题类别] 标题”：

1. [医学与医学研究] 睡眠呼吸暂停的影响与失明有关
2. [高尔夫] Lee Ann Walker 在女子职业高尔夫协会资深组赛事中被罚 58 杆
3. [历史事件与时代] 海军陆战队纠正一项关于硫磺岛、延续了 74 年的错误
4. [营养与健康饮食] 23 种有助于降低高血压的食物
5. [电影] 16 部最不符合历史事实的电影

根据剩余的已观察兴趣和新闻推断被删除的主题类别。这是历史兴趣重建，不是预测下一次点击。只输出该类别的单 token 兴趣 ID，严格使用 <interest_NNN> 格式，不输出类别名或任何解释。

训练目标：`<interest_005>`（美式橄榄球）。

## 数据与产物路径

- [配对数据与独立回读](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_frequency_paired_v1/paired_audit.json)
- [类别分布](/Users/river/Documents/MSRA/Rec/CrossInterest/docs/experiments/2026-09-17/mind_cir_frequency_class_distribution.csv)
- [数据构造](/Users/river/Documents/MSRA/Rec/CrossInterest/src/new_interest/data/build_cir_frequency_dataset.py)
- [冻结运行包构造](/Users/river/Documents/MSRA/Rec/CrossInterest/scripts/analysis/prepare_cir_frequency_bundles.py)
- [原 CIR1 验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-20260916/final_acceptance.json)
- [singleton 数据](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_frequency_paired_v1/singleton/cir.parquet)
- [singleton 运行包](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-frequency-20260917/singleton/bundle-v1)
- [singleton AMLT 配置](../../../configs/amlt/archive/2026-10-26/mind_cir_frequency_singleton_0917_v1.yaml)
- [repeated 数据](/Users/river/Documents/MSRA/Rec/CrossInterest/data/training/mind_cir_frequency_paired_v1/repeated/cir.parquet)
- [repeated 运行包](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-frequency-20260917/repeated/bundle-v1)
- [repeated AMLT 配置](../../../configs/amlt/archive/2026-10-26/mind_cir_frequency_repeated_0917_v1.yaml)

本地验证：CIR 相关 35 项测试通过；原始完整测试套件因缺少本地 peft 而在另一模块的测试收集阶段中断。两组全部训练输入 token 检查通过，单次/高频 CIR 最大长度分别 1,525/1,440，NI 最大长度 1,564；无截断，监督 token 数均为 3。实际 AMLT 上传清单每组 93 个文件，无缺失或额外文件。

- [本地预检](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-frequency-20260917/local_preflight.json)

- [实际上传和配置核对](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-frequency-20260917/upload_and_config_audit.json)

- [单次组跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_frequency_singleton_20260917_v1.json)

- [高频组跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_frequency_repeated_20260917_v1.json)

## 提交记录

| 条件 | 实验 | 作业 |
|---|---|---|
| singleton | `mind-cir-singleton-ni3-0917-v1` | [AMLT](https://aka.ms/amlt?q=k2oy5) |
| repeated | `mind-cir-repeated-ni3-0917-v1` | [AMLT](https://aka.ms/amlt?q=k2ozt) |

最终验收记录：

- [singleton 验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-frequency-20260917/singleton/final_acceptance.json)
- [repeated 验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-frequency-20260917/repeated/final_acceptance.json)
