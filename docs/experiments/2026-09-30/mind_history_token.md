# NI-only 历史类别 token 化：最终结果

**两组实验与远端配对分析均已独立验收。本次单seed结果没有显示输入token化的明确收益，当前保留类别文本输入更有依据。**

## 结果

| 条件 | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| 原 NI-only（Text108，历史参考） | 20.3275% | 10.1325% | 33.6553% | 13.4663% |
| Text125（匹配对照） | 20.2535% | 10.0988% | 33.3740% | 13.4004% |
| Token125（输入替换） | 19.8730% | 10.0074% | 33.4679% | 13.4279% |

主比较是相同125-token词表下的 Token125−Text125。R@10下降0.3804个百分点，R@20上升0.0938个百分点；不能概括为所有指标均下降或均改善。相对原NI-only，Token125的R@10低0.4544个百分点、R@20低0.1874个百分点，但这个历史比较还包含词表与运行路径差异。

| 指标 | Token125−Text125（百分点） | 配对95%区间（百分点） |
|---|---:|---:|
| R@10 | -0.3804 | [-1.1678, +0.3537] |
| NDCG@10 | -0.0913 | [-0.4606, +0.2562] |
| R@20 | +0.0938 | [-0.8241, +1.0797] |
| NDCG@20 | +0.0275 | [-0.3214, +0.3926] |

配对分析覆盖同一批27,082用户、31,392 impressions、37,728目标出现和104个支持类别。使用2,000次用户级Exp(1) Bayesian bootstrap、seed42，同一个用户在两组与其所有记录共享权重，每次重新计算各类别分母后取macro平均。四项区间均跨0，尚不足以确认提升或退化。区间只反映固定模型下的用户采样不确定性，不包含训练seed不确定性，也不能证明两种表示等价。

## 实验边界

- Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，seed42；每组21,600条NI、3 epochs、2,025次更新、64,800次曝光，global batch32、4×A10080GB。
- 两组均使用125个可训练tied兴趣token，原108个输出ID和顺序保持不变；额外17类从Train历史获得，按名称排序追加为108–124，采用相同的类别词向量均值与范数匹配初始化。它们不成为监督目标或排名候选。
- Text125保留历史类别文本；Token125仅把每条历史方括号中的类别换为对应token。标题、顺序、重复事件、历史长度、指令文本、目标和所有非prompt字段保持相同；替换可以逐字逆转。Train/Dev均无超长样本，不截断。
- 两组相同LoRA配置、原生全词表completion-only token-mean CE、BF16和显式Torch fallback；冷启动373个训练张量的共同SHA为 `28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565`。门禁权重没有用于正式训练。
- 不包含CIR、MeRA正则、多seed或参数搜索。输入token化同时改变长度和类别语义呈现，本轮不单独识别输入输出对齐机制。原Text108 NI-only存在单卡转四卡及内核路径差异，仅为历史参考。

## 验收

两组均通过125行有限非零梯度、单卡8步下降、四卡2步、最终373张量及4条样本logits精确fresh reload（atol=rtol=0）。最终checkpoint的逐样本计数为每条3次，共64,800次。

Full Dev使用原108候选、无历史类别过滤、合法且无重复Top20。远端独立检查完整预测与输入哈希、记录顺序、类别/token映射、分片和各项指标；本地再次从紧凑分片证据重算macro与每类指标。最终CPU分析又读取两组完整预测，重算指标与配对区间，并重新哈希全部最终checkpoint：Text125共13文件/205,936,550字节，Token125共13文件/205,935,514字节。大权重、optimizer和完整预测始终保留远端。

CPU分析v1的清单误包含AMLT不上传的`.amltignore`，本地上传模拟已复现源检查必然失败；其最终原生状态为Failed、没有分析产物。v2仅修正这个清单项，全部9个实际科学输入/代码文件字节不变。v2原生Completed，结果和完整哈希读回已接受，没有重新训练。完整运行过程保存在[运行记录](mind_history_token_runlog.md)。

## 产物与恢复

- [指标CSV](mind_history_token_results.csv)、[差值与区间CSV](mind_history_token_paired_differences.csv)。CSV指标是0–1比例；差值表单位是百分点。
- 总入口：`manifests/mind_history_token_20260930_v1.json`；恢复指针：`manifests/mind_history_token_restoration_20261001_v1.json`。
- 两臂接受记录：`artifacts/mind-history-token-20260930/{text,token}/final_acceptance.json`。
- 配对接受记录：`artifacts/mind-history-token-20260930/paired_final_acceptance.json`；完整分析：`comparison.json` 和 `analysis_receipt.json`（同目录）。
- 分析backend：`cool_panda_h84tqwcqpb`。comparison SHA-256：`d43d8ab86f7f4de0e2f941b7cf6e10ac7261fc1f6ed782eb04cafffdc8a64ec4`。

两臂及配对分析全部验收，本聊天定时监控`ni-token`已停止（PAUSED）。不自动增加新实验。
