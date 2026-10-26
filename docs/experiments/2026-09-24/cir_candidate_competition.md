# CIR 候选竞争诊断

状态：2026-09-24 两组已独立验收，原生状态均为 Completed。最终结果见 [候选竞争诊断结果](cir_candidate_competition_results.md)。下文保留方案与过程记录。

检验问题：完整 CIR5→NI3 相对历史 NI-only 的优势，在显式排除输入中可见的历史类别后是否仍然保留？两种排名使用相同模型、同一次前向的完整 108 类分数、相同用户／prompt／正例。该诊断不改变正式 Dev 结果，也不训练新模型。

冻结入口：`manifests/mind_cir_candidate_competition_20260924_v1.json`。AMLT experiment 为 `mind-cir-candidate-competition-0924-v1`，两个作业 `score-ni-only`、`score-cir5-ni3`，Recall Standard，每作业 4×A10080GB，总上限 8 GPU。没有创建自动监控。

## 数据与模型

- Full Dev：31,392 grouped impressions，37,728 个目标出现，104 个支持类，108 类候选。
- 从实际 prompt 解析可见历史类别，并逐条与 `history_subcategories` 核对；只排除其与 108 类输出词表的交集。
- 本地全量核查：删除正例 0，过滤后候选最少 73；词表之外的历史类别累计 2,356 次（按每条记录的类别集合计）。
- NI-only：已接受 `olive_bean_s743bd4tqj`，adapter 文件 SHA `0a41effcaac398a0ca9d914142b7975dffaad98f435379e436e6132b59c3abf0`；保留原 FLA 0.5.2 / causal-conv1d 1.7.0、batch 8、四分片推理。
- CIR5→NI3：已接受 `teal_rail_56hhgsgcnp`，adapter 文件 SHA `9f171a3d4fd984586e7ebcb230fe88558d3201c3ae3f4e480269fd1edda2e306`；保留原显式 Torch fallback、batch 2、四分片推理。
- 模型 Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；读取完整缓存快照并校验各文件 SHA。

## 分析与验收

1. 全部 373 个 adapter 张量的源文件及目录 SHA 与已接受模型绑定；原 Dev、mapping 和数值运行时代码冻结。
2. 获取全部 108 类有限分数；保留原生 `torch.topk(k=20)` 的顺序，其余候选按分数降序、同分按 mapping index。保存同分边界计数。
3. 未过滤 Top20 的标签、分数须与远端已接受预测逐条精确相同；原八个 Macro Recall/NDCG 指标须复现。
4. 从完整 108 类排名删除可见历史类别，核查正例不被删除，再计算 @1/@5/@10/@20。独立的计数／折扣增益实现与现有 metric library 相互核对。
5. 输出每个模型的历史类别占 TopK 的比例、完整正例 rank 分布、过滤前后指标；本地只保留紧凑指标／收据／hash，完整评分与目标 rank 明细留远端。
6. 比较两种候选集合中的 CIR−NI-only 差值。差距缩小支持历史候选竞争解释一部分观测现象；仍有差距说明对未见类别的相对排序差异也存在。差距变化不作为可加的因果贡献比例。

历史 NI-only 保留训练单卡转四卡与 FLA 的执行差异，因此本比较是机制筛查，不是严格匹配的因果实验；也不据此声称知识唯一位于某一组参数。没有运行历史标题替换、额外 seed 或新训练条件。

CPU 验证：候选过滤、可见历史一致性、正例保护、同分条件下精确原 Top20、原 Top20 之外的目标晋升等测试，以及既有 ID-token ranking 测试，共 9 项通过（另含类宏平均不等于目标微平均的校验）。

## 中间读回：2026-09-24 03:39 UTC

NI-only 全部 31,392 条原 Top20 标签及分数精确复现，远端与本地独立重算通过。原始／过滤后的 Recall@10 为 20.3275%／20.4186%，Recall@20 为 33.6553%／34.1764%；历史类别占原 Top20 槽位的 1.3250%。CIR 仍在评分，尚不报告最终模型间比较。NI-only entrypoint 已 exit 0；原生详情 API 此时仍显示 Running，尚未记录最终终态。

## 最终读回：2026-09-24 04:01 UTC

CIR 的 31,392 条原 Top20 标签与分数也精确复现。两组各自远端完整预测重算、本地完整 rank 直方图重算、ETag/文件 hash、四分片/合并预测 hash、adapter 与数据绑定均通过。两组原生状态 Completed；CIR 生命周期记录完成时间 04:00:32 UTC。过滤后 CIR−NI-only 的 Recall@10/@20 差距为 +1.5823/+0.9963 个百分点，NDCG@10/@20 为 +1.0451/+0.9121 个百分点。全部 @1/@5/@10/@20、历史占比、正例排名与解释边界见最终结果文档。
