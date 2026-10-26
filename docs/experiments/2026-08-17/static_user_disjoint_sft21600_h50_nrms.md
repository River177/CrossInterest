# SFT-21600 匹配的冻结 MiniLM NRMS

## 结论

使用与21,600条平衡SFT完全相同的Train用户、108类标签和最近50条原始History训练NRMS-style模型后，最终选择：

- `hidden_dimension = 384`
- `attention_heads = 8`
- `dropout = 0.1`
- `class_balance_beta = 0`
- `learning_rate = 1e-3`
- `weight_decay = 0`
- `epoch = 4`

在31,392个impression组成的探索性Dev上，NRMS得到 **Interest-Macro Recall@10 19.2179%**、**Interest-Macro NDCG@10 9.5065%**，相对Global分别提高 **10.7590 / 5.6883个百分点**。Internal holdout也出现同方向提升，说明结果不是只靠查看Dev选出的。

但impression-level Recall@10没有提高：Global为18.6820%，NRMS为18.6700%；NRMS主要改善类别均衡和正确兴趣的位置，而不是增加平均每个impression命中的target比例。

## 模型与数据

这是冻结标题encoder的NRMS-style分类器，不是从词级输入端到端训练的原始NRMS新闻encoder：

1. 使用固定revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`的 `sentence-transformers/all-MiniLM-L6-v2` 将标题编码为384维向量；
2. 可学习线性投影与GELU；
3. multi-head self-attention、残差连接和LayerNorm；
4. additive attention pooling得到用户向量；
5. 无bias的108类分类器输出全兴趣空间分数。

训练和推理均屏蔽History已出现的兴趣。History来自匹配数据集保存的最近50条原始Behavior History，不追加impression点击，runner不再截断。

- Train：21,600用户，108类，每类200条；
- 内部分割：17,291 model-fit、2,093 tune、2,216 frozen holdout；
- Dev：27,082用户、31,392 impressions、37,728 target occurrences；
- Dev主指标支持：104类；
- 冻结缓存：74,553个标题，本实验需要的29,069个标题缺失为0。

## 选择协议

架构阶段评估6组 `hidden × heads × dropout`，在epoch 2/4产生12个Tune checkpoint。优化阶段固定最佳架构，并搜索：

- `learning_rate ∈ {0.001, 0.003}`；
- `weight_decay ∈ {0, 1e-5}`；
- `class_balance_beta = 0`；
- epoch `{2, 4, 6, 8}`。

优化阶段共4个训练条件、16个Tune checkpoint。选择只读取Train tune：先取Interest-Macro Recall/NDCG Pareto前沿，要求Recall不低于Global且距最佳Recall不超过0.2个百分点，再最大化NDCG。最终Tune点为Recall@10 **24.4270%**、NDCG@10 **12.7422%**。

选定后先评估frozen holdout，再用全部21,600条Train重训4个epoch；Dev仅在最后进行一次探索性评测。

## 结果

### Interest-Macro主指标

| 评估集 | 方法 | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| Internal holdout | Global | 9.8545% | 4.4551% |
| Internal holdout | NRMS | **25.0463%** | **12.8713%** |
| Dev（探索性） | Global | 8.4589% | 3.8182% |
| Dev（探索性） | NRMS | **19.2179%** | **9.5065%** |

对应NRMS相对Global的提升为：

- Internal holdout：Recall +15.1918 pp，NDCG +8.4162 pp；
- exploratory Dev：Recall +10.7590 pp，NDCG +5.6883 pp。

### Impression-level诊断

| 方法 | Recall@10 | NDCG@10 |
|---|---:|---:|
| Global | **18.6820%** | 7.3494% |
| NRMS | 18.6700% | **9.1933%** |

Interest-Macro显著提高而平均impression Recall基本持平，说明模型把Top-10容量从少数高支持类别重新分配到了更多兴趣类别，并把命中的target排得更靠前。这支持History包含可学习信号，但不能解释为每个用户都获得了更高Recall。

## 执行、验证与边界

- GCR A100 80GB执行，PyTorch `2.12.1+cu130`、CUDA 13.0、bfloat16；GCR空闲，因此未提交AMLT。
- 真实全数据单架构/单epoch smoke先通过，smoke结果未进入正式报告。
- 正式搜索完成6/6架构、4/4优化条件；本地focused tests 21项、完整测试297项全部通过。
- 下载后验证18个输出SHA-256和7个输入SHA-256；四份Top-10文件行数、impression/user/target身份均匹配。
- 所有Top-10无重复，且与对应原始History兴趣交集为0；本地使用SFT evaluator重新计算的Macro指标与冻结汇总逐值一致。
- Dev已用于前序分析，只能称为exploratory Dev，不能冒充全新final test。
- Train每个用户只有一个观测新兴趣；该标签是潜在未来兴趣集合的不完整样本。

## 产物

- 配置：`configs/future_interest/static_user_disjoint_sft21600_h50_nrms_v1.json`
- 实现：`src/new_interest/experiments/tune_semantic_history_nrms.py`
- 产物目录：`artifacts/future-novel-interest/static-user-disjoint-sft21600-h50-nrms-v1/`
- 最终模型：`selected_model.npz`
- 搜索记录：`tuning.json`
- 汇总：`summary.json`
- 逐兴趣指标与Top-10：`conditions/<split>/<condition>/`
- 数据审计：`audit.json`
- 环境：`environment.json`
- 输入输出哈希：`manifest.json`
- GCR完整日志：`run.log`
