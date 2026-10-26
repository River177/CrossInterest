# CIR 使用 NI 提示词：最终结果

实验已完成并独立验收。仅将 CIR 提示词换成 NI 模板后，四项主指标均低于原提示词条件；Recall@10 由 **21.9717% 降至 20.9914%**，下降 **0.9803 个百分点**。本次结果不支持以统一提示词替代原 CIR 提示词。

## 研究问题

在固定 CIR 数据、目标、训练顺序、模型初始化和正式更新预算的条件下，把“恢复被删除的历史类别”改写成原版 NI 的“预测未来未见兴趣”指令，是否改善 CIR5→NI3 的效果？主对照为已接受的原提示词 CIR5→NI3；NI-only 仅作背景参考。本次只运行一个新条件，seed42。

## 训练数据构造

原CIR从训练用户可见历史中选一个类别，删除该类别全部行为后以其SID为监督。新条件保留所有17,271条样本及其行顺序、用户、删除结果、历史、目标和来源task标记；逐条核对仅 `prompt` 与同步的 `messages` 改变。NI、Dev、SID映射和选择清单逐字节不变。CIR使用NI措辞后仍是合成历史监督，不能当作真实未来点击标签。

| 条件 | CIR样本 / epochs / 更新 | NI样本 / epochs / 更新 | 正式总更新 |
|---|---:|---:|---:|
| 原CIR提示词→NI（已有对照） | 17,271 / 5 / 2,700 | 21,600 / 3 / 2,025 | 4,725 |
| NI提示词CIR→NI（本次） | 17,271 / 5 / 2,700 | 21,600 / 3 / 2,025 | 4,725 |

两组CIR目标分布相同，覆盖108类；NI固定每类200条。历史可见长度上限50，不回填被删记录。完整原文、中文翻译和相同SID响应见[真实训练样本对照](mind_cir_ni_prompt_prompt_example.md)。

Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，4×A10080，BF16、LoRA r16/alpha32/dropout0.05，lr2e-5，global batch32（microbatch2×4卡×accumulation4），原cosine/warmup0.03、native full-vocabulary completion CE与Torch fallback。从相同原始LoRA/SID初始化开始；CIR→NI保留参数、重置optimizer/scheduler。诊断训练权重没有进入正式训练。推理不做适应。

## 测试设计与验收

原Full Dev共31,392条grouped impressions、37,728次正例出现、104个有支持的目标类别。固定108个SID评分，未过滤历史类别，Top20无重复合法排名；四分片各7,848条。主指标先在各目标类别内平均，再对104类等权平均；因此不能用单个全局命中数/37,728替代该宏平均。NDCG沿用原逐目标排名折扣定义。测试不使用LLM judge。

CIR全程86,355次曝光、每样本5次；NI全程64,800次曝光、每样本3次。最终checkpoint全部样本游标复放、CIR→NI继承、两个阶段的新optimizer/scheduler初态通过。每阶段373tensor及4条参考logits新进程精确重载。全量预测的键、排名及指标在远端核验；本地从原Dev标签与全部分片逐类统计独立重算所有Recall/NDCG@1/5/10/20，误差低于1e-12。native Completed已核实，后端终态时间为北京时间2026-09-29 11:31；本地最终独立读回在11:46之后完成。权重、optimizer与完整预测均留远端。

## 实验结果

数值单位为%；差值单位为百分点，方向为“NI模板−原CIR模板”。

| 指标 | NI-only（参考） | 原CIR模板→NI | NI模板CIR→NI | 差值 |
|---|---:|---:|---:|---:|
| Recall@10 | 20.3275 | 21.9717 | 20.9914 | -0.9803 |
| Recall@20 | 33.6553 | 35.0756 | 34.9470 | -0.1287 |
| NDCG@10 | 10.1325 | 11.2051 | 10.9410 | -0.2640 |
| NDCG@20 | 13.4663 | 14.4966 | 14.4628 | -0.0337 |

新模板相对NI-only仍有Recall@10 +0.6639个百分点，但相对原CIR模板损失0.9803个百分点。@1/@5存在小幅提升，完整8项指标见CSV；不能概括为所有K均下降。这是单seed结果，没有统计显著性结论，也不能据此断言CIR的全部收益来自某种特定机制。旧NI-only存在历史运行路径差异，主要提示词比较应以两组CIR5→NI3为准。

## 数据与产物路径

- [跟踪清单](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_cir_ni_prompt_20260928_v1.json)
- [最终独立验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/final_acceptance.json)
- [提示词直接比较JSON](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/final_prompt_comparison.json)
- [完整指标CSV](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/prompt_comparison.csv)
- [紧凑读回及ETag/hash收据](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/final_download_receipt.json)
- [冻结实验配置](/Users/river/Documents/MSRA/Rec/CrossInterest/configs/experiments/mind_cir_ni_prompt_0928_v1.json)
- [数据变更与冻结包预检查](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/local_preflight.json)
- [训练完成与继承验收](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-cir-ni-prompt-20260928/progress/20260929T031645Z/training_acceptance.json)
- [运行记录](mind_cir_ni_prompt_runlog.md)
- [AMLT作业](https://aka.ms/amlt?q=k4l88)，实验 `mind-cir-ni-prompt-0928-v1`，backend `tough_bone_676m3s6rrw`。
- 远端结果前缀：`projects/sft4safety/amlt-results/7209441648.22156-0716e9cd-30c5`；最终模型：该前缀下 `formal/ni/final_adapter`。
- 新NI adapter文件SHA256：`deb27796443078d7a17b00bdf7a4b59b4af236294452f699c0db81b769b32703`。

本实验已完成；自动跟进 `cir-ni` 已关闭（PAUSED），没有启动后续条件。
