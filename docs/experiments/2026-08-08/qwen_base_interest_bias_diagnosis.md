# Qwen3-8B Base Interest 候选选择低于随机诊断

## 问题

K=10 候选选择中：

```text
Qwen3-8B Base Interest Selection Accuracy = 8.32%
uniform random expectation = 10.00%
```

这里测量的是 Selection Accuracy，不是 AUC。

在 12,164 条样本上，8.32% 相对 10% 的差异为 `z=-6.18`，不能用普通抽样波动解释。

## 已排除的问题

### 正确位置不均衡

正确位置严格均衡：

```text
位置 1–4：各 1,217 条
位置 5–10：各 1,216 条
```

### 输出格式或解析错误

Qwen Base Interest：

```text
Candidate-valid = 99.98%
Prefix compliance = 100%
```

因此低准确率不是格式错误。

### 某一次候选随机顺序异常

固定同一 1,000 条样本，构造 5 个独立候选排列：

| seed | Accuracy |
|---:|---:|
| 101 | 7.1% |
| 202 | 7.0% |
| 303 | 8.6% |
| 404 | 8.1% |
| 505 | 7.6% |
| 平均 | 7.68% |

同一样本在五次排列中平均产生 2.41 个不同预测，只有 20.6% 的样本五次预测完全一致。

### Prompt 未声明顺序随机

在同一 1K 上测试：

| Prompt | Accuracy | 首位选择率 |
|---|---:|---:|
| 当前编号模板 | 7.1% | 25.2% |
| 明确声明顺序随机 | 7.5% | 19.1% |
| 无编号 bullet + 去偏说明 | 7.2% | 13.9% |

去偏 Prompt 显著降低首位偏好，但没有提升准确率。

### Greedy 解码不能表达候选偏好

在频率匹配的 1K 诊断集上：

| 决策方式 | Accuracy |
|---|---:|
| 普通 greedy 生成 | 9.5% |
| 逐候选 completion log-likelihood | 9.6% |

因此问题不在 greedy 解码；模型对正确候选本身没有更高条件概率。

## 确认的问题

### 位置偏好

Qwen Base Interest 的实际选择位置：

```text
1: 25.0%
2: 11.3%
3: 10.2%
4: 9.1%
5: 6.4%
6: 6.2%
7: 5.8%
8: 6.9%
9: 5.3%
10: 13.6%
```

位置偏好稳定存在，但去偏后准确率不提升，所以它是次要问题。

### 类别先验失配

模型高频预测：

```text
Artificial Intelligence & Robotics
Energy & Sustainability
Consumer Technology
Climate Change
Cybersecurity & Digital Privacy
```

真实目标高频类别：

```text
American Football
Shopping & Gift Ideas
Policing & Corrections
Food Culture & Trends
Government & Public Policy
```

模型预测类别先验与真实目标分布不匹配。

### 原负样本采样使 popularity baseline 异常强

原候选数据的 9 个负样本近似从 110 类均匀采样，但真实正样本高度长尾。

| Baseline | Accuracy |
|---|---:|
| 均匀随机 | 10.00% |
| 使用 Train 目标频率选择候选中最高频类别 | 31.99% |
| 使用 Dev 目标频率（oracle） | 51.07% |
| 使用 Qwen 自身全局类别偏好 | 6.77% |

因此原 benchmark 同时测量了大量类别流行度先验，10% 随机并不是唯一合理 baseline。

## 频率匹配负样本诊断

负样本改为按 Train 目标频率加权、无放回抽取后：

```text
Train-popularity baseline = 8.8%
Qwen greedy = 9.5%
Qwen likelihood scoring = 9.6%
```

这使类别先验 baseline 接近随机，也显示 Qwen Base 在去除先验捷径后基本就是随机水平。

## 诊断结论

Qwen3-8B Base Interest 的 8.32% 不是实现 bug。原因是：

1. 模型有稳定的位置偏好；
2. 模型类别先验与真实目标分布失配；
3. 用户历史对“下一个从未出现的新主题”的可预测信号弱；
4. 原均匀负样本设计放大了类别流行度因素。

Prompt 去偏、候选重排和 likelihood scoring 均无法恢复有效信号。

## 建议的基准设置

后续正式 Base benchmark 应同时报告：

```text
uniform random baseline
Train-popularity baseline
Qwen Base
```

如果目标是纯粹测量“历史条件偏好”而非类别先验，建议使用 Train-frequency-matched
负样本，并在多个候选排列 seed 上求平均。

## 产物路径

- 原始诊断：
  `artifacts/candidate-selection-k10/qwen_base_interest/bias_diagnosis.json`
- 五排列数据：
  `data/training/candidate_selection_k10/diagnostics/interest_1k_permutations.parquet`
- 五排列结果：
  `artifacts/candidate-selection-k10/diagnostics/candidate-qwen-base-interest-permutations-v1/`
- Prompt variants：
  `data/training/candidate_selection_k10/diagnostics/interest_1k_prompt_variants.parquet`
- Prompt variant 结果：
  `artifacts/candidate-selection-k10/diagnostics/candidate-qwen-base-interest-prompt-variants-v1/`
- 频率匹配数据：
  `data/training/candidate_selection_k10/diagnostics/interest_1k_frequency_matched.parquet`
- 频率匹配结果：
  `artifacts/candidate-selection-k10/diagnostics/candidate-qwen-base-interest-frequency-matched-v1/`
- Likelihood 运行：
  AMLT experiment `candidate-qwen-base-interest-likelihood-v1`
