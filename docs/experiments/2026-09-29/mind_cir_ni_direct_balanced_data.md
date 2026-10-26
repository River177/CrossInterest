# CIR与原NI：同用户、同类别、每类同数量的候选对照

## 可行性结论

可以控制两组整体标签分布完全一致。在现有4,870对“CIR目标在H50出现至少2次”的数据中，已经构造出1,055对严格匹配候选数据；两组使用同一批1,055个用户、同一来源记录、相同顺序，共同98个训练类别，逐类别样本数完全相同（每类1–32条）。本轮没有启动训练。

原NI目标来自未来新兴趣，CIR目标来自被删除的历史类别，同用户的两个目标必然不同。本次要求的是两组目标类别的总频数相等，不改写任何人的标签来做逐行目标相等。

| 条件 | 样本数 | 用户集合 | 训练类别 | 每类样本数 | 平均序列token | 最长token |
|---|---:|---|---:|---|---:|---:|
| CIR构造NI | 1,055 | 同一集合 | 98 | 逐类别与NI完全相同 | 731.6351 | 1,445 |
| 配对原NI | 1,055 | 同一集合 | 98 | 1–32 | 798.0976 | 1,518 |

两组相同原NI提示词、messages schema及training_task，每条3个监督token，无超过2048样本、无截断。两组token总量比为0.916724，不声称相同计算量。

## 构造方式

每名用户保留与否同时作用于两个训练集。以目标类别为节点，每对样本从CIR目标指向NI目标；选中的每个类别入量等于出量，即两组该类样本数量相等。最大化保留用户数得到1,055对。类别和边按稳定顺序排列；同类目标对内用固定SHA排序选用户，最终恢复原来源顺序。依赖NetworkX3.6.1，独立用SciPy1.17.1 HiGHS线性规划核对上界1,055并被实际整数解达到，因此这是当前固定4,870对内的最大同用户匹配规模，不适用于其他候选池。

没有新增CIR目标、改标签、替换历史或过采样。两组训练行都是原4,870对的完全相同索引子集。选择队列会使用两组训练标签来对齐类别分布；这与最初CIR目标选择不读未来NI标签是两个步骤，不能再声称最终队列的选择与NI训练标签无关。未使用Dev标签、模型loss或预测效果选择样本。

## 规模与解释边界

- 若只按类别分别抽样、放弃完全相同用户，现有两个4,870条池内可以各保留1,819条、98类；本轮仅构造上述更严格的1,055对，不另建第二候选组。
- 现有CIR重复目标池缺少10个类别，无法通过下采样恢复原NI的108类或21,600条规模。候选训练集两边共同缺少同样10类，但108候选SID及原完整Dev保持不变。
- 每类只有1–32条，适合小规模受控诊断；不应直接替代完整数据的主结果。
- 该对照消除了用户、目标类别集合、类别频数及监督数量差异，仍有输入历史内容/长度、token量及标签语义来源不同，不能把差值唯一归因于标签噪声。
- 此处是两组相同分布，不是所有类别都均匀。逐类数量见statistics和manifest。

## 审计与产物

17项配对/重复筛选/分布匹配测试通过。独立从落盘训练文件重算类别频数、检查共同用户和有序子集关系；结构化原历史重新检查目标出现次数至少2、整类删除和原NI目标不在完整历史中；Dev/Validation用户零重叠、原文件hash不变。全量token检查通过。

- [候选范围与hash清单](../../../manifests/mind_cir_ni_direct_balanced_20260929_v1.json)
- [CIR构造NI](../../../data/training/mind_cir_ni_direct_balanced_20260929_v1/cir_as_ni.parquet)
- [配对原NI](../../../data/training/mind_cir_ni_direct_balanced_20260929_v1/original_ni_paired.parquet)
- [逐类数量与统计](../../../data/training/mind_cir_ni_direct_balanced_20260929_v1/statistics.json)
- [配对来源](../../../data/training/mind_cir_ni_direct_balanced_20260929_v1/pairs.parquet)
- [全部4,870对去留记录](../../../data/training/mind_cir_ni_direct_balanced_20260929_v1/selection_manifest.parquet)
- [独立分布匹配与最大规模回读](../../../artifacts/mind-cir-ni-direct-balanced-20260929/balance_readback.json)
- [输入与标签回读](../../../artifacts/mind-cir-ni-direct-balanced-20260929/independent_readback.json)
- [token审计](../../../artifacts/mind-cir-ni-direct-balanced-20260929/token_audit.json)

复现：环境需NetworkX，运行 `PYTHONPATH=src python -m new_interest.data.build_cir_ni_direct_balanced --output <新目录>`。构造器拒绝覆盖已有目录。原17,271对及4,870对数据保留不变。
