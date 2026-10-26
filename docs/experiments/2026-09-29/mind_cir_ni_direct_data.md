# CIR构造的NI与原始NI：独立训练对照数据

## 结论与范围

可以构造，且已生成并独立回读。两组各17,271条、17,271个相同用户，覆盖108个SID类别；按同一原始source_sample_id逐条配对，用户、impression及timestamp一致。本轮只构造与审计数据，未提交训练，也未创建自动跟进。

冻结入口：[manifest](../../../manifests/mind_cir_ni_direct_20260929_v1.json)。两组拟各自从同一基座训练后直接评估真实NI，不接第二阶段NI微调。本轮不确定训练epoch或提交GPU作业。

## 两组数据

| 条件 | 输入 | 标签 | 样本数 | 平均历史条数 | 每类样本数范围 |
|---|---|---|---:|---:|---:|
| cir_as_ni | H50中删除目标类别全部行为，保留其余全部历史 | 被删除的历史兴趣 | 17,271 | 22.0677 | 3–1,056 |
| original_ni_paired | 同来源记录的原始H50 | 原始未来点击的新兴趣 | 17,271 | 23.7364 | 91–183 |

使用原CIR已有17,271条视图，不沿用上一轮9,431对历史前缀过滤，也不额外裁短历史。原始NI共21,600条，其中4,329条无法满足原CIR类别删除后的历史要求，按原CIR选择结果排除；没有根据未来目标、训练loss或评估成绩重新挑选。原NI保留用户的原始标签与所有原字段，顺序等于原NI的对应子序列。

两组使用完全相同的原NI提示词模板、相同messages格式、相同训练schema及 `training_task=new_interest`。CIR标签来源在pairs侧表中明确标记为历史删除构造，不声称它是该用户真实未来点击。CIR-as-NI的prompt、messages、历史、目标与现有NI提示词版本CIR完全一致，唯一额外规范化是共享training_task字段。

两组目标标签逐对不同：原NI目标未出现在完整原始历史中；CIR目标来自原始历史，删除后才在输入中不可见。因此本对照匹配来源记录，不匹配目标类别。

## 逐条审计

- 原CIR的类别选择、H50整类删除、保留news顺序及完整剩余历史重建一致。
- 原NI所有原字段保持不变；SID与108类mapping一致。
- CIR输入中无其目标类别；原NI目标在完整原始历史中不存在。
- 两组用户及来源顺序一致，均与原Dev及已登记Validation用户零重叠。
- Dev与mapping逐字节保持不变，所有源文件及落盘输出保存SHA-256。
- 18项相关测试通过；独立脚本从落盘数据重建历史与标签，再完成全量token检查。

## 全量token审计

Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，tokenizer文件hash对照已验收版本；enable_thinking=False，108个SID注册方式沿用原实现。

| 条件 | 平均完整序列token | 最长序列 | 每条监督token | 超过2048 |
|---|---:|---:|---:|---:|
| cir_as_ni | 658.0699 | 1,532 | 3 | 0 |
| original_ni_paired | 699.7706 | 1,564 | 3 | 0 |

CIR/原NI完整序列token总量比为0.940408，没有截断。相同样本数与训练轮数可以对齐更新预算，但不等于严格相同token计算量。

## “更干净”的解释边界

CIR目标类别分布更不均衡，两组类别分布的总变差距离为0.421863；本轮不通过类别重采样改变原构造。CIR中12,401/17,271条（71.8024%）目标在原H50仅出现一次，不能预设这些标签由多次点击确认。其余4,870条至少出现两次，完整次数分布见statistics。

此数据能检验“同一批用户、相同监督数量下，两种监督构造谁更适合迁移到真实新兴趣预测”。即使CIR更好，也不能仅凭该结果把原因唯一归结为更低标签噪声；目标分布、历史删除及历史兴趣到未来兴趣的任务差异需要保留在解释中。原21,600条完整NI-only结果可作为背景，不能代替本17,271条配对NI模型的严格对照。

## 文件与复现

- [CIR构造的NI](../../../data/training/mind_cir_ni_direct_20260929_v1/cir_as_ni.parquet)
- [配对原始NI](../../../data/training/mind_cir_ni_direct_20260929_v1/original_ni_paired.parquet)
- [逐条来源与标签侧表](../../../data/training/mind_cir_ni_direct_20260929_v1/pairs.parquet)
- [构造统计](../../../data/training/mind_cir_ni_direct_20260929_v1/statistics.json)
- [独立回读](../../../artifacts/mind-cir-ni-direct-20260929/independent_readback.json)
- [token审计](../../../artifacts/mind-cir-ni-direct-20260929/token_audit.json)
- [构造器](../../../src/new_interest/data/build_cir_ni_direct.py)

```bash
PYTHONPATH=src python -m new_interest.data.build_cir_ni_direct --output <新的输出目录>
pytest -q tests/test_cir_ni_direct.py tests/test_build_cir_ni_prompt_dataset.py tests/test_build_cir_dataset.py
```

构造器通过临时目录写入，拒绝覆盖已有输出。所有原数据保持不变。
