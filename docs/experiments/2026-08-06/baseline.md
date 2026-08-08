目前是 **1K Dev Subset**上的 Qwen3.5-9B Base 结果：

| 输入          | 输出格式  |  Accuracy | 格式合法率 |
| ------------- | --------- | --------: | ---------: |
| Title-only    | Letter    | **18.4%** |       100% |
| Title-only    | Full-name | **17.6%** |      99.9% |
| Category-only | Letter    | **14.8%** |       100% |
| Category-only | Full-name | **14.6%** |      99.0% |

随机猜测baseline为 **10%**。目前看：

- Title-only 明显优于 Category-only
- Letter 与 Full-name 准确率接近