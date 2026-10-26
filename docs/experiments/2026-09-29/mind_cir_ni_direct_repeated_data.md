# CIR目标历史出现次数大于1：配对独立训练数据

已按用户要求构造并独立审计，本轮只处理数据，未提交训练。

## 筛选规则

从已有17,271对中保留原H50历史内CIR目标类别出现至少2次的4,870对；删除12,401对单次出现样本。原NI组同步保留同来源用户与记录，仍使用原始未来NI目标，不按未来点击次数选择。沿用既有CIR目标，不为被排除用户另选目标。CIR整类删除后保留全部剩余历史，prompt、目标、messages和训练行均是父数据的原样子集，训练顺序保持相同。

出现次数以原H50中的历史条目计数，不是完整H0计数、不同日期数或不同新闻数。4,870对中117对虽然出现多次，但目标类别只对应同一news ID；按本次规则保留，没有悄悄增加去重条件。

## 数据统计

| 条件 | 样本/用户数 | 训练目标类别数 | 每类样本范围 | 平均历史长度 | 平均完整序列token | 最长token |
|---|---:|---:|---:|---:|---:|---:|
| CIR构造NI | 4,870 | 98 | 1–655 | 25.1889 | 736.8760 | 1,525 |
| 配对原NI | 4,870 | 108 | 18–65 | 28.5604 | 821.4795 | 1,564 |

两组均采用相同原NI模板、同一108 SID mapping、每条3个监督token，无超过2048的样本，无截断。CIR/原NI token总量比0.897011，不声称计算量相同。

CIR缺失的10个训练目标类别：Artificial Intelligence & Robotics; Careers & Professional Development; Entrepreneurship & Small Business; Faith Communities & Practice; Genealogy & Family History; General & Multi-Sport Coverage; Home Appliances & Technology; Theater & Performing Arts; Visual Arts & Design; Workplace Life。仍保留108类输出候选与原完整Dev，不删除这些类别的测试样本。

## 审计与解释

23项相关测试通过。独立检查两组是原配对数据完全相同索引的有序子集；逐条从结构化H50重数目标频次、重建类别删除及剩余历史；原NI标签与原字段不变，来源身份相同，目标类别均不在各自输入中。Dev/Validation用户隔离和原Dev/mapping哈希一致检查通过，全量token审计通过。

重复出现提供了更强的历史行为证据，但不足以直接证明标签噪声更低。筛选后CIR目标仅覆盖98类且更不均衡，用户群、样本数及输入信息也不同于17,271对版本。因此与原始未筛选组比较时不能把差值仅归因于去除噪声；与本配对原NI比较时也须保留类别覆盖差异。没有自动重采样、删测试类别或重新选择目标来修正此差异。

## 产物

- [执行范围与hash清单](../../../manifests/mind_cir_ni_direct_repeated_20260929_v1.json)
- [CIR构造NI](../../../data/training/mind_cir_ni_direct_repeated_20260929_v1/cir_as_ni.parquet)
- [对应原NI](../../../data/training/mind_cir_ni_direct_repeated_20260929_v1/original_ni_paired.parquet)
- [配对来源](../../../data/training/mind_cir_ni_direct_repeated_20260929_v1/pairs.parquet)
- [全部17,271对的筛选去留记录](../../../data/training/mind_cir_ni_direct_repeated_20260929_v1/selection_manifest.parquet)
- [严格子集回读](../../../artifacts/mind-cir-ni-direct-repeated-20260929/filter_readback.json)
- [历史和标签独立回读](../../../artifacts/mind-cir-ni-direct-repeated-20260929/independent_readback.json)
- [token审计](../../../artifacts/mind-cir-ni-direct-repeated-20260929/token_audit.json)

复现：`PYTHONPATH=src python -m new_interest.data.build_cir_ni_direct_repeated --output <新目录>`。构造器拒绝覆盖已有输出，原17,271对数据保持不变。
