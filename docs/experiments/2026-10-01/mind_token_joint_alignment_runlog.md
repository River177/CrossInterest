# Token125 NI＋双向类别语义对齐：冻结实验

用户明确授权新增一组联合训练，与已验收的Token125 NI-only比较。联合训练和完整Dev已验收，远端配对分析正在提交，区间结果尚未产出。

## 科学范围

主任务沿用Token125：`[<interest_NNN>] Headline`历史→新兴趣ID。辅助任务是类别名称→ID、ID→完整类别名称，覆盖全部125类，基础配对250条。所有名称和映射来自冻结类别表；不包含用户、未来物品文本、Dev标签或新生成描述。新增17类仅在辅助映射任务中可作输出，NI推荐及Dev排名仍限定原108类。

| 每个epoch的任务 | 样本数 | 3轮曝光 | 监督token占比 |
|---|---:|---:|---:|
| NI | 21,600 | 64,800 | 84.6948% |
| 名称→ID | 1,250 | 3,750 | 4.9013% |
| ID→名称 | 1,250 | 3,750 | 10.4039% |

每个方向的125条基础样本各复制10次/epoch，总计24,100条/epoch、72,300次曝光。每个基础方向/类别对总曝光30次。每轮由既有全局seeded sampler混合打乱；NI原始行字段逐字段相同，主任务样本数和3次曝光相同，混合后的更新位置与原NI-only不同。

沿用原生full-vocabulary completion-only token_mean CE，每个监督token单位权重，无额外task loss系数。文本回答较长，因此两个方向样本数相等但监督token份额不同；表内为全量输出长度审计的实际占比，不把它称为相等任务权重。

Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，seed42，原125-token映射和mean_norm_matched初始化，LoRA r16/alpha32/dropout0.05/all-linear，lr2e-5、cosine、warmup0.03、weight decay0.01。4×A10080GB Recall Standard，microbatch2、accumulation4、global32，BF16、显式Torch fallback、非reentrant checkpointing，100步持久化，3 epochs共2,262 updates。全量训练最长1397 tokens，零超长、不截断。

新组从原始基座冷启动，要求全部373个可训练张量SHA等于Token125基线 `28b70b3031e6c736ea10930595ac19651186067b4b5f8f85e4a72ff4b0b7c565`。不加载已训练Token125权重，不单独warm-up，不重训基线、不增加CIR/MeRA/其他seed或比例搜索。

与基线相比增加了辅助监督、237次更新及LR schedule horizon；本轮回答“这份联合训练配方是否改善NI”，不能单独证明语义对齐机制、排除额外计算量效应，也不能以无提升证明语义对齐无效。

## 参考来源

实时核对OpenOneRec公开源码revision `a969edcadd579a06c1966ae1db5984e02f48beff`：

- [README](https://github.com/Kuaishou-OneRec/OpenOneRec/blob/a969edcadd579a06c1966ae1db5984e02f48beff/README.md)描述Itemic-Text Alignment与多任务SFT。
- [item_understand.py](https://github.com/Kuaishou-OneRec/OpenOneRec/blob/a969edcadd579a06c1966ae1db5984e02f48beff/data/onerec_data/sft/item_understand.py)明确采用视频SID→caption任务。
- 已检查的pretrain item_understand模板同样将ID与文本共同呈现；没有据此声称其公开实现采用本轮两方向/比例/损失。

本轮是用户指定的双向兴趣类别适配，prompt沿用此前MiniOneRec式识别问答的表达形式。旧108类独立warm-up与本轮125类联合训练不同，不复用其效果结论。参考源码只读存档在 `artifacts/mind-token-joint-alignment-20261001/reference/`。

## 数据与实现

冻结输入：`data/training/mind_token_joint_alignment_20261001_v1`，metadata记录源和输出SHA。`joint.parquet`为混合训练，`alignment_pairs.parquet`为250基础配对；`ni_reference.parquet`和Dev保留基线字节。重复辅助样本有唯一训练sample_id和共享base_alignment_id，二者不混淆。

运行包继承已接受Token125的数值代码。Trainer、CE、tokenizer/初始化、模型结构、checkpoint持久化、重载和评估代码字节保持；只修改编排、任务枚举及审计中的“固定3个监督token”假设，使其逐样本读取冻结的实际答案长度。真实tokenizer在worker再次核对全部长度，损失分母仍由实际labels计算。差异：`artifacts/mind-token-joint-alignment-20261001/runtime.patch`。

## 门禁与验收

本地27项测试通过。实际AMLT uploader清单128文件、141,133,769字节，零缺失/额外；全套pytest因缺peft在收集阶段中断，没有声称全套通过。

远端先核验环境/模型快照/数据，做258条混合代表样本的125行梯度检查；随后冷启动单卡32样本×2轮=8步（24NI＋4＋4），冷启动四卡41样本=2步（29NI＋6＋6），三任务都有覆盖，包含最长输入和最长辅助答案。每次需finite loss、单卡loss下降、373张量及logits精确fresh reload、64条真实Dev四分片和独立指标重算，才允许正式训练。门禁权重不进入正式训练。

正式验收需2,262步、各任务曝光64,800/3,750/3,750、每个重复行3次、每个基础对30次、逐样本监督token计数、最终checkpoint/reload、31,392条Full Dev/104支持类/37,728目标、108候选合法无重复Top20，无历史类别过滤。配对比较绑定已接受Token125的完整预测SHA，同用户对齐、独立重算，不下载完整预测到Mac。

跟进入口：`manifests/mind_token_joint_alignment_20261001_v1.json`。原生状态/日志读取使用 `scripts/analysis/observe_history_token.py --manifest ...`；紧凑读回使用 `scripts/analysis/read_token_joint_alignment.py --receipt ...`。区分进度、已提交checkpoint和最终验收；正常排队/无变化保持安静，重要阶段、异常或完成再通知。全部大产物留持久远端。

## 已提交与首次观测

2026-10-01 14:45北京时间：`mind-token125-joint-alignment-1001-v1` / `qwen35-2b-token125-joint` 已提交，backend `amiable_ghost_y68q7gmrb0`，原生Running，正在准备worker依赖。最新stdout没有Traceback或终止信号；尚无GPU门禁或正式训练checkpoint，不能称为训练通过。持久结果位于 `projects/sft4safety/amlt-results/7209191952.16969-a21aaef8-2d05`。

128文件运行包SHA核验通过；manifest SHA `cf28f07e4c98904558c63fdc2b9b38efd376b247afa1a82b21eeab96451d6366`。本聊天30分钟heartbeat `ni-token` 已更新为本轮新范围并恢复ACTIVE，不再操作已完成的输入替换实验。首次证据：`artifacts/mind-token-joint-alignment-20261001/progress/20261001T064547Z/`。

## 15:12 启动门禁独立验收

原生Running，正式日志41/2262，最近loss2.324、grad_norm14.42，所有已读loss/grad_norm/lr有限，最新stdout无Traceback或中断。尚未到首个正式CP100，不把门禁checkpoint当正式训练进度。

125行梯度、258条混合代表样本、全量实际答案长度核验通过。单卡32样本8步loss 5.389505→4.474287，任务曝光为NI48/名称→ID8/ID→名称8；四卡41样本2步曝光为29/6/6。两次373张量和logits精确重载通过，64条真实Dev、36支持类、75目标的四分片及独立指标重算通过。正式冷启动SHA与已接受Token125基线相等，未继承门禁训练权重。

证据：`artifacts/mind-token-joint-alignment-20261001/progress/20261001T071239Z/receipt.json` 与 `independent_readback.json`。继续冻结配方，不扩大任务范围。

## 16:50 常规进度读回

正式日志520/2262，最近loss1.411、grad_norm9.498；本次读取到的loss/grad_norm/lr均有限，原生Running，全部活动日志无新异常或平台中断。CP500紧凑文件哈希、远端文件大小、训练身份、逐样本曝光及可变监督token计数核验通过：共16000次曝光，其中NI14295、名称→ID857、ID→名称848。任务采样是全局混合随机顺序，中途方向计数不要求相等，完整预算仍为64800/3750/3750。

此前中断的只读查询未留下完成receipt；本轮重新读取同一backend，没有重复提交、恢复或修改训练。证据：`artifacts/mind-token-joint-alignment-20261001/progress/20261001T085031Z/receipt.json` 和 `independent_readback.json`。

## 完整训练与Dev独立验收（10月2日）

原生Completed，2262步/3epochs、72300次总曝光核验通过，其中NI64800、名称→ID3750、ID→名称3750。逐样本计数、可变监督token计数、373张量与logits精确重载通过。Full Dev覆盖31392 impressions、104支持类、37728目标；原108候选无过滤合法Top20，分片与本地紧凑指标重算通过。训练阶段完整接受记录：`artifacts/mind-token-joint-alignment-20261001/final_acceptance.json`。

日志传输ReadTimeout期间改用--skip-logs读取原生状态与持久化产物，未重启或重新训练；此处完成时间依产物与原生终态，而不是推算训练速度。读回证据：`progress/20261001T183703Z/`。

| 条件 | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Token125 NI-only | 19.8730% | 10.0074% | 33.4679% | 13.4279% |
| Token125 NI＋双向对齐 | 20.0887% | 10.1856% | 33.3820% | 13.5229% |

联合组相对基线R@10为+0.2156个百分点、NDCG@10为+0.1782个百分点、R@20为−0.0858个百分点、NDCG@20为+0.0950个百分点。远端CPU全checkpoint/预测哈希复核与同用户配对区间尚未完成，暂不宣称统计显著收益。新组训练更新更多，结论只能针对此联合配方。

分析入口：`manifests/mind_token_joint_analysis_20261002_v1.json`；只读观测沿用`observe_history_token.py --manifest ...`，最终分析验收`accept_joint_alignment_comparison.py --receipt ...`。两臂的2025/2262步及对应曝光分别绑定，不能互换。分析代码6项测试通过，真实AMLT上传清单10文件无遗漏，CPU 5C3 Basic，无新训练。

2026-10-02 11:02北京时间，配对分析已提交并确认原生Queued：`mind-token125-joint-analysis-1002-v1` / `token125-joint-paired-analysis`，backend `frank_diamond_qgqmggjsjy`，CPU5C3 Basic；完整预测和checkpoint留远端。两份已接受训练记录现已hash绑定分析合同，后续不得用新观测覆盖这些最终接受文件。分析读回入口 `manifests/mind_token_joint_analysis_20261002_v1.json`。

10月2日11:04只读复核：配对分析仍为原生Queued，backend和结果路径匹配，暂无产物。未重复提交、未更改两臂验收记录。证据：`analysis/progress/20261002T030451Z/receipt.json`。
