# Recall、NDCG 与 Hit 指标计算说明

本文档说明当前新主题预测实验中 `Recall@K`、`NDCG@K`、`Hit@K` 和
`MRR@20` 的定义、计算方法与区别。

实际实现位于：

```text
src/new_interest/evaluation/metrics.py
```

评估对象为 grouped Dev10K：

```text
data/training/sft_experiments/eval/
├── behavior_only/dev_10k_grouped.parquet
└── interest_only/dev_10k_grouped.parquet
```

## 1. 评估数据形式

### 1.1 每个 impression 只预测一次

Dev 按 `impression_id` 聚合。每个 impression 对应：

- 一段用户历史；
- 一个模型生成的主题排序；
- 一个或多个正确新主题。

例如：

```text
正确主题集合 G = {Golf, Cars & Driving}

模型排序 P =
1. American Football
2. Golf
3. Music
4. Cars & Driving
...
```

因为一个 impression 可以同时包含多个正确新主题，所以评估不能只使用普通单标签
accuracy。

### 1.2 当前 Dev10K 的正确答案数量分布

| 每个 impression 的正确主题数 | Impression 数 |
| ---------------------------: | ------------: |
|                            1 |         8,410 |
|                            2 |         1,195 |
|                            3 |           283 |
|                            4 |            73 |
|                            5 |            21 |
|                            6 |            13 |
|                            7 |             1 |
|                            8 |             3 |
|                            9 |             1 |
|                     **总计** |    **10,000** |

因此 84.1% 的 impression 只有一个正确答案，15.9% 有多个正确答案。

## 2. 符号定义

对于 impression \(u\)：

- \(G_u\)：该 impression 的正确新主题集合；
- \(|G_u|\)：正确主题数量；
- \(P_u=(p_1,p_2,\ldots,p_{20})\)：模型按分数从高到低产生的主题排名；
- \(P_u@K=(p_1,\ldots,p_K)\)：排名前 K 个主题；
- \(K\in\{1,5,10,20\}\)。

当前评估先使用 32 个 beams 产生候选路径，再按类别去重并保留 Top-20。

## 3. Recall@K

### 3.1 公式

$$
\mathrm{Recall@K}(u)
=
\frac{|P_u@K \cap G_u|}{|G_u|}
$$

它衡量：**所有正确主题中，有多少比例进入了 Top-K。**

### 3.2 不要求全部命中

假设：

```text
G = {Golf, Cars & Driving, Personal Finance}
```

则：

| Top-K 命中数 |       Recall@K |
| -----------: | -------------: |
|            0 |              0 |
|            1 | \(1/3=0.3333\) |
|            2 | \(2/3=0.6667\) |
|            3 |      \(3/3=1\) |

因此 Recall 不是“必须全部命中才得分”，而是连续比例。

### 3.3 具体例子

```text
正确主题：
{Golf, Cars & Driving}

Top-5：
1. American Football
2. Golf
3. Music
4. Television & Streaming
5. Personal Finance
```

只命中 `Golf`：

$$
\mathrm{Recall@5}=\frac{1}{2}=0.5
$$

如果 `Cars & Driving` 位于第 12 名：

$$
\mathrm{Recall@10}=0.5
$$

$$
\mathrm{Recall@20}=1.0
$$

## 4. Hit@K

### 4.1 公式

$$
\mathrm{Hit@K}(u)
=
\begin{cases}
1, & |P_u@K \cap G_u|>0 \\
0, & \text{otherwise}
\end{cases}
$$

它只衡量：**Top-K 是否至少命中一个正确主题。**

### 4.2 与 Recall@K 的区别

仍假设有三个正确主题：

| Top-K 命中数 | Hit@K | Recall@K |
| -----------: | ----: | -------: |
|            0 |     0 |        0 |
|            1 |     1 |  \(1/3\) |
|            2 |     1 |  \(2/3\) |
|            3 |     1 |        1 |

Hit 不区分命中一个还是命中全部；Recall 会区分。

### 4.3 单答案时二者相同

若 \(|G_u|=1\)，则：

$$
\mathrm{Hit@K}(u)=\mathrm{Recall@K}(u)
$$

因为命中数只能是 0 或 1。

在当前 Dev10K 中，多答案 impression 会导致总体：

$$
\mathrm{Hit@K}\geq\mathrm{Recall@K}
$$

## 5. DCG@K

### 5.1 二元相关性

当前任务使用 binary relevance：

$$
\mathrm{rel}_i
=
\begin{cases}
1, & p_i\in G_u \\
0, & p_i\notin G_u
\end{cases}
$$

### 5.2 公式

$$
\mathrm{DCG@K}(u)
=
\sum_{i=1}^{K}
\frac{\mathrm{rel}_i}{\log_2(i+1)}
$$

正确主题排名越靠前，贡献越大。

| 排名 \(i\) |                命中贡献 |
| ---------: | ----------------------: |
|          1 |  \(1/\log_2(2)=1.0000\) |
|          2 |  \(1/\log_2(3)=0.6309\) |
|          3 |  \(1/\log_2(4)=0.5000\) |
|          5 |  \(1/\log_2(6)=0.3869\) |
|         10 | \(1/\log_2(11)=0.2891\) |
|         20 | \(1/\log_2(21)=0.2277\) |

## 6. IDCG@K

IDCG 是该 impression 在理想排序下能达到的最大 DCG。

若有 \(m=|G_u|\) 个正确主题，则理想情况下它们排在最前面的
\(\min(K,m)\) 个位置：

$$
\mathrm{IDCG@K}(u)
=
\sum_{i=1}^{\min(K,|G_u|)}
\frac{1}{\log_2(i+1)}
$$

例如有三个正确主题：

$$
\mathrm{IDCG@5}
=
1+\frac{1}{\log_2(3)}+\frac{1}{\log_2(4)}
$$

$$
=1+0.6309+0.5=2.1309
$$

## 7. NDCG@K

### 7.1 公式

$$
\mathrm{NDCG@K}(u)
=
\frac{\mathrm{DCG@K}(u)}
{\mathrm{IDCG@K}(u)}
$$

取值范围：

$$
0\leq\mathrm{NDCG@K}\leq1
$$

它同时考虑：

1. 命中了多少个正确主题；
2. 正确主题在排名中的位置。

### 7.2 具体例子：单答案

正确主题为 `Cars & Driving`。

如果它位于第 1 名：

$$
\mathrm{NDCG@20}=1
$$

如果位于第 14 名：

$$
\mathrm{NDCG@20}
=
\frac{1/\log_2(15)}{1}
\approx0.2560
$$

如果未进入 Top-20：

$$
\mathrm{NDCG@20}=0
$$

### 7.3 具体例子：多答案

正确集合：

```text
G = {Golf, Cars & Driving}
```

模型将 `Golf` 排第 2，`Cars & Driving` 排第 4：

$$
\mathrm{DCG@5}
=
\frac{1}{\log_2(3)}
+
\frac{1}{\log_2(5)}
$$

$$
\approx0.6309+0.4307=1.0616
$$

两个正确答案的理想位置是第 1 和第 2：

$$
\mathrm{IDCG@5}
=
1+\frac{1}{\log_2(3)}
\approx1.6309
$$

$$
\mathrm{NDCG@5}
\approx
\frac{1.0616}{1.6309}
=0.6509
$$

Recall@5 为：

$$
\mathrm{Recall@5}=\frac{2}{2}=1
$$

这说明即使所有正确主题都被召回，如果位置不够靠前，NDCG 仍小于 1。

## 8. NDCG@1、Recall@1 与 Hit@1

### 8.1 NDCG@1

只看第一名是否是任一正确主题：

$$
\mathrm{NDCG@1}
=
\begin{cases}
1, & p_1\in G_u \\
0, & p_1\notin G_u
\end{cases}
$$

### 8.2 Hit@1

同样只看第一名是否命中，因此：

$$
\mathrm{Hit@1}=\mathrm{NDCG@1}
$$

### 8.3 Recall@1

$$
\mathrm{Recall@1}
=
\frac{|\{p_1\}\cap G_u|}{|G_u|}
$$

若有两个正确主题，第一名命中其中一个：

```text
NDCG@1 = 1
Hit@1   = 1
Recall@1 = 1/2 = 0.5
```

所以多答案数据中通常：

$$
\mathrm{NDCG@1}
=
\mathrm{Hit@1}
\geq
\mathrm{Recall@1}
$$

单答案数据中三者相同。

## 9. MRR@20

设第一个正确主题的排名为 \(r_u\)：

$$
\mathrm{RR@20}(u)
=
\begin{cases}
\frac{1}{r_u}, & r_u\leq20 \\
0, & \text{Top-20 无正确主题}
\end{cases}
$$

$$
\mathrm{MRR@20}
=
\frac{1}{|U|}
\sum_{u\in U}\mathrm{RR@20}(u)
$$

MRR 只关心**第一个**正确主题的位置，不考虑后续是否还有其他正确主题。

## 10. 数据集级宏平均

当前所有指标都先逐 impression 计算，再对 10,000 个 impression 做宏平均：

$$
\mathrm{Metric}
=
\frac{1}{|U|}
\sum_{u\in U}\mathrm{Metric}(u)
$$

每个 impression 权重相同，不会因为某个 impression 有更多正确主题而获得更大权重。

## 11. 随机基线

### 11.1 在全部 110 类中均匀随机排序

使用当前 Dev10K 的真实多答案分布，理论期望为：

| Baseline        | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 |
| --------------- | -------: | -------: | --------: | --------: | -----: | -----: | ------: | ------: | -----: |
| Random over 110 |    0.91% |    4.55% |     9.09% |    18.18% |  1.11% |  2.80% |   4.32% |   6.69% | 21.22% |

Recall 的理论期望：

$$
\mathbb{E}[\mathrm{Recall@K}]
=
\frac{K}{110}
$$

### 11.2 排除历史主题后随机排序

每条 impression 从 110 类中排除历史已出现的去重主题，再随机排序。当前 Dev10K
平均剩余 98.86 类：

| Baseline                 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@20 |
| ------------------------ | -------: | -------: | --------: | --------: | -----: | -----: | ------: | ------: | -----: |
| Random excluding history |    1.02% |    5.09% |    10.18% |    20.36% |  1.24% |  3.14% |   4.84% |   7.50% | 23.67% |

这个版本更接近“新主题不能出现在历史中”的任务约束。

## 12. 当前实验结果的阅读方式

以 Llama Interest 22K 为例：

```text
Recall@20 = 43.29%
NDCG@20  = 17.81%
Hit@20   = 47.86%
```

含义：

- 平均召回了每条 impression 正确主题集合的 43.29%；
- 考虑正确主题排名位置后的归一化增益为 17.81%；
- 47.86% 的 impression 至少有一个正确主题进入 Top-20。

Hit@20 高于 Recall@20，是因为多答案 impression 中命中一个主题就会令 Hit=1，
但 Recall 只获得部分分数。

## 13. 实现位置

- 指标实现：[`src/new_interest/evaluation/metrics.py`](../../src/new_interest/evaluation/metrics.py)
- Beam 生成：[`src/new_interest/evaluation/evaluate_beam.py`](../../src/new_interest/evaluation/evaluate_beam.py)
- 受限 token trie：[`src/new_interest/evaluation/constrained_beam.py`](../../src/new_interest/evaluation/constrained_beam.py)
- Dev10K：[`data/training/sft_experiments/eval/interest_only/dev_10k_grouped.parquet`](../../data/training/sft_experiments/eval/interest_only/dev_10k_grouped.parquet)
