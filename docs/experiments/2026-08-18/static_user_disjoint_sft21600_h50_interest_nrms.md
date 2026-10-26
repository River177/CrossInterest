# SFT-21600 匹配的 Interest-NRMS

## 结论

将最近50条 History 从新闻标题替换为兴趣类别名称，并保持同一 NRMS 用户编码器后，`Interest-NRMS` 在探索性 Dev 上得到：

- **Interest-Macro Recall@10：20.4840%**；
- **Interest-Macro NDCG@10：11.1818%**。

相对同协议的 `Behavior-NRMS`，分别提高 **1.2661 / 1.6752 个百分点**。但是，这不能解释为 Interest 输入全面优于标题输入：

- Internal holdout 上 Interest-NRMS 反而低 **0.7941 / 0.1508 个百分点**；
- Dev impression-level Recall/NDCG 低 **0.4853 / 0.4171 个百分点**；
- Dev 中 Recall 改善的类别为41个，下降的类别为44个，另有19个持平；
- 原始 Interest-Macro 增益主要由 support 1–4 的极低频类别贡献。去掉这些类别、只统计 support≥5 的95类时，Interest-NRMS 相对 Behavior-NRMS 的 Macro Recall/NDCG 差值变为 **-0.7193 / -0.3181 个百分点**。

因此，当前证据支持的结论是：**仅使用 History Interest 序列已经能形成有竞争力的用户表示，并在原始稀有类别等权 Macro 指标上更高；但它没有稳定地优于标题语义输入。**

Dev 已用于此前分析，本结果只能称为 exploratory Dev，不能作为全新 final test。

## 实验协议

### 数据

- Train：21,600个用户，108类，每类200个用户；每用户一个新兴趣标签。
- Internal split：17,291 fit、2,093 tune、2,216 frozen holdout，由固定用户哈希划分。
- Dev：31,392 impressions、27,082 users、37,728 target occurrences；104个类别有正例支持。
- History：每条样本最近50个原始静态 History item；不追加 impression、不在 runner 截断。
- 输入序列保留兴趣出现顺序与重复项。
- Top-10 预测屏蔽 History 中已经出现的输出兴趣。

### Interest-NRMS 架构

```text
History Interest 名称序列
  -> frozen MiniLM interest-name vectors (125 x 384)
  -> learned projection + GELU
  -> multi-head self-attention
  -> residual + layer normalization
  -> additive attention pooling
  -> user vector
  -> bias-free 108-class classifier
  -> mask History interests
  -> Top-10
```

兴趣名称使用 `sentence-transformers/all-MiniLM-L6-v2`，固定 revision：

`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`

完整125类兴趣词表均有一个单位归一化的384维向量。MiniLM 冻结，NRMS 聚合器与108类分类器可训练。

## 两个预注册条件

| 条件 | 架构与优化 | 用途 |
|---|---|---|
| Interest-NRMS (fixed) | hidden 384、8 heads、dropout 0.1、LR 1e-3、WD 0、4 epochs | 固定 Behavior-NRMS 超参数，只比较输入表示 |
| Interest-NRMS (tuned) | 同 Behavior-NRMS 的6组架构与 LR/WD/epoch 搜索空间 | 衡量 Interest 输入在同等搜索预算下的最佳结果 |

Tuned 条件在 Train tune 上最终仍选择 `384 / 8 heads / 0.1 / 1e-3 / 0 / 4 epochs`。因此 fixed 与 tuned 的最终模型及 Dev predictions 的 SHA-256 完全相同。后续表格只保留一个 `Interest-NRMS` 结果，避免重复展示。

## 主结果

### Internal holdout

| 模型 | Interest-Macro Recall@10 | Interest-Macro NDCG@10 |
|---|---:|---:|
| Global | 9.8545% | 4.4551% |
| Behavior-NRMS | **25.0463%** | **12.8713%** |
| Interest-NRMS | 24.2522% | 12.7205% |

Interest-NRMS 相对 Behavior-NRMS：Recall **-0.7941**、NDCG **-0.1508** 个百分点。

### Exploratory Dev

| 模型 | Interest-Macro Recall@10 | Interest-Macro NDCG@10 |
|---|---:|---:|
| Global | 8.4589% | 3.8182% |
| Behavior-NRMS | 19.2179% | 9.5065% |
| Interest-NRMS | **20.4840%** | **11.1818%** |

Interest-NRMS 相对：

- Global：Recall **+12.0251**、NDCG **+7.3635** 个百分点；
- Behavior-NRMS：Recall **+1.2661**、NDCG **+1.6752** 个百分点。

### Dev impression-level

| 模型 | Impression Recall@10 | Impression NDCG@10 |
|---|---:|---:|
| Global | **18.6820%** | 7.3494% |
| Behavior-NRMS | 18.6700% | **9.1933%** |
| Interest-NRMS | 18.1847% | 8.7762% |

Interest-NRMS 没有增加平均每个 impression 命中的 target 比例。它的原始 Interest-Macro 提升与 impression-level 指标方向不同，说明变化集中在类别间的重新分配。

## 类别稳定性分析

Dev 的104个支持类别中：

| 比较 Interest-NRMS vs Behavior-NRMS | 改善 | 持平 | 下降 |
|---|---:|---:|---:|
| Recall@10 | 41 | 19 | 44 |
| NDCG@10 | 44 | 7 | 53 |

按最低 support 过滤后的类别等权差值为：

| 最低 support | 类别数 | Macro Recall 差值 | Macro NDCG 差值 |
|---:|---:|---:|---:|
| ≥1（主指标） | 104 | +1.2661 pp | +1.6752 pp |
| ≥5 | 95 | -0.7193 pp | -0.3181 pp |
| ≥10 | 89 | -0.7142 pp | -0.2071 pp |
| ≥50 | 72 | -0.3956 pp | -0.1386 pp |
| ≥100 | 57 | -0.3998 pp | -0.0631 pp |

例如，`Genealogy & Family History` 仅有1个 Dev target，Recall/NDCG 同时从0提升到1；这种单样本类别在 Interest-Macro 中与数千 support 的类别权重相同。主指标定义没有问题，但必须同时报告 support 敏感性，才能判断结果是否稳定。

## 审计结果

- 125个兴趣名称全部被固定 MiniLM cache 覆盖，维度384。
- Train/Dev/词表/embedding 的远端哈希与本地完全一致。
- 两个正式条件的 manifest 中全部 source/output SHA-256 重新计算通过。
- 每个正式条件均有31,392条 Dev prediction；impression、user、target identity 与源 Dev 完全一致。
- 每条 Top-10 均包含10个不同兴趣。
- Top-10 与对应 History Interest 的交集为0。
- fixed 与 tuned 的最终模型文件和 Dev prediction 文件逐字节一致。
- 本地 focused tests 通过；完整测试为 `300 passed`。
- GCR A100 full-data smoke 与两个正式实验均成功，无 OOM/NaN；未提交 AMLT。

## 产物

- Tuned 结果：`artifacts/future-novel-interest/static-user-disjoint-sft21600-h50-interest-nrms-v1`
- Fixed 结果：`artifacts/future-novel-interest/static-user-disjoint-sft21600-h50-interest-nrms-fixed-v1`
- Interest embedding：`artifacts/future-novel-interest/static-user-disjoint-nrms-interest-embeddings-v1`
- Tuned 配置：`configs/future_interest/static_user_disjoint_sft21600_h50_interest_nrms_v1.json`
- Fixed 配置：`configs/future_interest/static_user_disjoint_sft21600_h50_interest_nrms_fixed_v1.json`
- 实现：`src/new_interest/experiments/tune_semantic_history_nrms.py`
- Interest embedding builder：`src/new_interest/experiments/build_nrms_interest_embeddings.py`
