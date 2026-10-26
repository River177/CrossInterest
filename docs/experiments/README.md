# 实验与数据报告索引

按记录所在目录的日期排列，包含结果报告、设计与运行记录；完成状态以各文档及对应验收收据为准。

主线入口见[仓库与实验指南](../repository.md)。历史配置原路径见[归档映射](../../manifests/repository_archive_20261026.json)。

## 2026-08-06

- [baseline](2026-08-06/baseline.md)

## 2026-08-07

- [Qwen3-8B 生成式新主题预测：SFT 与 Beam Search 实验报告](2026-08-07/qwen3_sft_beam_experiment_report.md)

## 2026-08-08

- [K=10 候选主题选择数据：构造规则与真实样例](2026-08-08/candidate_selection_data_examples.md)
- [K=10 候选主题选择实验报告](2026-08-08/candidate_selection_experiment_report.md)
- [Qwen3-8B Base Interest 候选选择低于随机诊断](2026-08-08/qwen_base_interest_bias_diagnosis.md)
- [Qwen3-8B 与 Llama-3.1-8B：22K SFT 跨模型实验报告](2026-08-08/qwen_llama_22k_experiment_report.md)

## 2026-08-09

- [历史兴趣与新兴趣关系：无模型数据分析](2026-08-09/history_new_interest_relationship_analysis.md)
- [Qwen3-8B CoT Accuracy 与 Correctness-Gated Reward 实验](2026-08-09/qwen3_cot_accuracy_reward_experiment_report.md)
- [Qwen3-8B SFT22K vs GRPO1K：CoT 条件标签排序评测](2026-08-09/qwen3_sft22k_grpo1k_cot_label_ranking_report.md)

## 2026-08-10

- [K=10 候选选择扩展模型与 Reasoning 对比报告](2026-08-10/candidate_selection_extended_comparison_report.md)

## 2026-08-11

- [Qwen3.5-9B 22K 新兴趣预测实验报告](2026-08-11/qwen35_22k_experiment_report.md)
- [Qwen3-8B Teacher CoT-SFT1K v2 实验报告](2026-08-11/qwen3_teacher_cot_sft_v2_report.md)

## 2026-08-12

- [Future Observed Novel-Interest Baselines：双 Track 实验报告](2026-08-12/future_novel_interest_baselines_report.md)
- [新兴趣预测失败诊断与案例研究](2026-08-12/model_prediction_failure_case_study.md)

## 2026-08-13

- [SFT 训练数据：标题、类别名称语义与一级类别关系分析](2026-08-13/sft_training_data_similarity_analysis.md)

## 2026-08-15

- [静态用户隔离 K=10 Candidate Select（GPU-only interim）](2026-08-15/static_user_disjoint_candidate_selection.md)
- [Interest-Macro Global Coverage Scaffold 与 History 语义重排实验](2026-08-15/static_user_disjoint_semantic_coverage_interest_macro.md)
- [Global Coverage Scaffold 与 History 语义重排实验](2026-08-15/static_user_disjoint_semantic_coverage_scaffold.md)
- [静态用户隔离数据上的标题语义 History 信号实验](2026-08-15/static_user_disjoint_semantic_history_signal.md)

## 2026-08-16

- [类别平衡 Semantic History Ridge 的开放式 Top-10 Pareto 调参](2026-08-16/static_user_disjoint_balanced_semantic_history_ridge.md)
- [类别平衡 Semantic History Linear Ranker](2026-08-16/static_user_disjoint_balanced_semantic_linear_ranker.md)
- [Static User-Disjoint 21,600 Balanced SFT](2026-08-16/static_user_disjoint_balanced_sft.md)
- [冻结 MiniLM 标题表示的 NRMS 新兴趣预测](2026-08-16/static_user_disjoint_nrms.md)

## 2026-08-17

- [SFT-21600 匹配的冻结 MiniLM NRMS](2026-08-17/static_user_disjoint_sft21600_h50_nrms.md)

## 2026-08-18

- [Static user-disjoint cross-fitted sample reweighting — Exploratory Dev](2026-08-18/static_user_disjoint_crossfit_reweighting.md)
- [Static User-Disjoint Semantic-initialized ID Token SFT](2026-08-18/static_user_disjoint_semantic_id_token_sft.md)
- [SFT-21600 匹配的 Interest-NRMS](2026-08-18/static_user_disjoint_sft21600_h50_interest_nrms.md)

## 2026-08-24

- [Text-label T0-E / T1 消融：Semantic-ID 到底优化了什么](2026-08-24/text_label_length_normalization_ablation.md)

## 2026-08-27

- [SOLAR-GNIP：SASRec 与 GPT-5.5 Teacher Smoke Gate](2026-08-27/solar_gnip_sasrec_teacher_smoke.md)

## 2026-08-28

- [Semantic-ID + Global Cross-Fitted：哪些预测更准](2026-08-28/semantic_id_global_crossfit_case_analysis.md)

## 2026-08-31

- [MeRA-GNIP 方法复现](2026-08-31/mera_gnip_reproduction.md)

## 2026-09-02

- [MeRA 原始 Arts/Grocery 双口径复现](2026-09-02/mera_original_reproduction.md)
- [Wang et al. Stage-A 三数据集复现](2026-09-02/wang_stage_a_reproduction.md)

## 2026-09-13

- [OI→NI 精确元训练验证](2026-09-13/oi_ni_meta_pilot.md)

## 2026-09-14

- [MIND：Mixed 2B 短思考 SFT](2026-09-14/mind_short_think_sft.md)
- [OI→NI 全词表 SFT：Mixed / Meta](2026-09-14/oi_ni_full_vocabulary_sft.md)

## 2026-09-15

- [MIND NI-only 历史长度消融](2026-09-15/mind_history_length_ablation.md)
- [MIND 短思考 GRPO：E / H / HL 完整结果](2026-09-15/mind_thinking_grpo_e_h_hl.md)

## 2026-09-16

- [MIND 108 类 CIR → NI：最终训练与 Full Dev 结果](2026-09-16/mind_cir_warmup.md)
- [MIND 108 类 CIR warm-up → NI SFT](2026-09-16/mind_cir_warmup_runlog.md)
- [动态可约减损失权重：三组学生实验](2026-09-16/rho_student_three_arm.md)
- [两折参考模型：21,600 条 OOF 损失评分验收](2026-09-16/rho_twofold_reference_scoring.md)

## 2026-09-17

- [MIND 108 类 CIR 3 epochs → NI 3 epochs：最终结果](2026-09-17/mind_cir3_ni3.md)
- [MIND 108 类 CIR 3 epochs → NI 3 epochs](2026-09-17/mind_cir3_ni3_runlog.md)
- [CIR 频率对照：单次兴趣与高频兴趣重建 → NI](2026-09-17/mind_cir_frequency_paired.md)
- [CIR frequency repeated → NI](2026-09-17/mind_cir_frequency_repeated.md)
- [CIR frequency singleton → NI](2026-09-17/mind_cir_frequency_singleton.md)

## 2026-09-18

- [MIND CIR 时长与混合方式：四组实验](2026-09-18/mind_cir_four_conditions.md)

## 2026-09-23

- [MIND CIR参数迁移：保留LoRA或SID](2026-09-23/mind_cir_parameter_transfer.md)
- [MIND CIR参数迁移：最终实验报告](2026-09-23/mind_cir_parameter_transfer_results.md)
- [MIND NI-only 历史顺序消融：最终验收报告](2026-09-23/mind_history_order_ablation.md)
- [历史顺序消融阶段记录（截至2026-09-23 20:50 UTC）](2026-09-23/mind_history_order_ablation_runlog.md)

## 2026-09-24

- [CIR 候选竞争诊断](2026-09-24/cir_candidate_competition.md)
- [CIR 候选竞争诊断结果](2026-09-24/cir_candidate_competition_results.md)

## 2026-09-28

- [CIR 与额外 NI：同用户、同目标、长度匹配实验](2026-09-28/mind_cir_ni_matched.md)
- [CIR 使用 NI 提示词：最终结果](2026-09-28/mind_cir_ni_prompt.md)
- [冻结训练样本：提示词替换前后](2026-09-28/mind_cir_ni_prompt_prompt_example.md)
- [CIR 使用 NI 提示词：CIR5 → NI3](2026-09-28/mind_cir_ni_prompt_runlog.md)

## 2026-09-29

- [CIR与原NI：同用户、同类别、每类同数量的候选对照](2026-09-29/mind_cir_ni_direct_balanced_data.md)
- [CIR构造的NI与原始NI：独立训练对照数据](2026-09-29/mind_cir_ni_direct_data.md)
- [CIR目标历史出现次数大于1：配对独立训练数据](2026-09-29/mind_cir_ni_direct_repeated_data.md)
- [CIR构造NI与原NI：1,055对类别频数匹配的独立训练](2026-09-29/mind_cir_ni_direct_training.md)

## 2026-09-30

- [CIR 与 NI 实验汇总](2026-09-30/cir_ni_experiment_overview.md)
- [NI-only 历史类别 token 化：最终结果](2026-09-30/mind_history_token.md)
- [NI-only 历史类别 token 化实验](2026-09-30/mind_history_token_runlog.md)

## 2026-10-01

- [Token125 NI＋双向语义对齐：最终结果](2026-10-01/mind_token_joint_alignment.md)
- [Token125 NI＋双向类别语义对齐：冻结实验](2026-10-01/mind_token_joint_alignment_runlog.md)

共索引 67 个 Markdown 文档。`data/` 和 `artifacts/` 不随 Git 提供，恢复方式见[数据与产物说明](../artifacts.md)。
