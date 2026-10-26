# 历史顺序消融阶段记录（截至2026-09-23 20:50 UTC）

以下保留执行时的阶段快照，其中“待完成”均指当时状态。最终验收和完整结果见[最终报告](mind_history_order_ablation.md)。

# MIND NI-only 历史顺序消融运行记录

用户已确认[冻结方案](../../research/2026-09-23_history_order_ablation_plan.md)，并明确保留原提示词。B组已通过完整Dev远端验证与紧凑独立回读；新训练已完成2,025步／64,800次曝光并通过严格fresh reload及独立回读，C全量评估及紧凑独立回读已通过，D的四个分片持续产出；最终成对分析尚未完成。

| 条件 | 训练历史 | 评估历史 | 模型与工作 |
| --- | --- | --- | --- |
| A | 原始顺序 | 原始顺序 | 复用已验收 NI-only |
| B | 原始顺序 | 固定打乱 | 复用 A 模型，新评估 |
| C | 固定打乱 | 原始顺序 | 一个新训练的最终模型，新评估 |
| D | 固定打乱 | 同 B 的固定打乱 | 与 C 同一个最终模型，新评估 |

新训练仅一次：Qwen3.5-2B、Behavior+Interest → Semantic-ID、seed42、21,600 条、3 epochs、2,025 次更新、64,800 次曝光；原单卡前500步和四卡后1525步的执行安排不变。每组 Full Dev 31,392 条、37,728 目标出现、104 支持类、108 候选、Top20、无历史类别过滤。

## 输入验收

训练及 Dev 已完整回读，逐行内容多重集合、标题—类别配对、长度、标签、原提示词和用户划分不变。训练每用户一个固定排列，三轮共用；Dev 同用户同历史共享排列。

| 项目 | Train | Dev |
| --- | ---: | ---: |
| 记录数 | 21,600 | 31,392 |
| 实际 prompt 改变 | 20,728 | 28,930 |
| 长度为 1 | 323 | 854 |
| 长度≥2但抽中原排列 | 541 | 1,588 |
| 非原排列但因重复条目而文本未变 | 8 | 20 |

数据目录：`data/training/mind_history_order_20260923_v1`。Train SHA-256 `54a570c0254be097eccd3f29154d2129074dc2ac83383be36d7a65193028825d`；Dev SHA-256 `e53abcf33c9e67a6b8603740ee802671c466ced19586f1cda5b7cf3567ac6354`。排列清单及原输入摘要见该目录 `metadata.json`。

## 来源与提交

本次在线回读原 NI-only 的8份紧凑来源证据，原指标文件 SHA 与已验收记录一致。原 adapter 文件 SHA 为 `0a41effcaac398a0ca9d914142b7975dffaad98f435379e436e6132b59c3abf0`；评估记录的 `cfc443...` 是整个 adapter 目录摘要，两者不混用。完整大产物在 worker 远端验证，不下载到 Mac。

新运行包复用35个冻结运行源码／依赖文件，训练与评估的数值实现未改；新增排列 builder、阶段编排和独立分析。完整本地测试1540通过／8跳过；冻结 bundle 的16项专项测试通过；AMLT实际上传清单65文件／75,565,633 bytes与manifest一致；YAML dump和diff检查通过。没有新增训练smoke。

- 实验：`mind-ni-history-order-0923-v1`。
- 作业：`qwen35-2b-ni-history-order`，后端 `ivory_deer_r3cm1cz1rb`。
- 资源：Recall Standard、4×A10080GB，三组评估与一次训练依次执行。
- [作业入口](https://aka.ms/amlt?q=k3x7k)。
- 结果入口：`projects/sft4safety/amlt-results/7209875909.09490-48dfeba1-32f9`。
- 冻结包：`artifacts/mind-history-order-20260923/bundle-v1`。
- 包manifest SHA-256：`791912f47bc67f5f8daf4a48bd4dd586a3a335881b3c19d7316514d80941cc54`。
- 跟踪清单：[manifest](../../../manifests/mind_history_order_20260923_v1.json)。
- 已启用本任务15分钟自动跟进；状态无变化时安静，仅在阶段变化、异常或验收完成时通知。

## 验收状态与结果

当前启动检查、B评估、新训练终点及严格重载已通过。C完整评估及紧凑独立回读也已通过；D完整评估、成对分析和最终持久manifest验收仍待完成。阶段进度以最新原生日志及持久产物为准，AMLT Preparing/Running/Completed均不代表最终验收。

2026-09-23 08:47 UTC：原生状态 Running；stdout 确认4张 NVIDIA A100-SXM4-80GB，正在安装冻结依赖，尚未开始正式更新。观察收据：`artifacts/mind-history-order-20260923/progress/startup-02/progress.json`。

08:54 UTC：已独立回读并匹配包身份、全量输入审计、实际依赖版本及基座快照文件摘要，收据 `artifacts/mind-history-order-20260923/startup_component_readback.json`。原生元数据查询两次发生本机代理连接重置；限定该只读命令使用 `NO_PROXY='*'` 后恢复，未修改或重启远端任务。08:54:58 UTC 原生状态仍 Running，日志确认依赖已安装完成；token审计、原模型fresh reload与B评估仍待产物确认。

08:56 UTC：全量token审计独立回读通过，21,600条、最大1,564 tokens、超长0条、所有completion均为3 tokens。原模型fresh reload和B完整评估仍待确认；正式训练尚无已观察更新数。

08:57:47 UTC：`startup_ready.json` 已提交，全部启动检查通过。独立回读原模型fresh reload：373张量逐项一致、4条参考输入logits在atol=rtol=0下通过。`startup_component_readback.json` 已提升为 `all_startup_checks_independently_accepted`。流程进入B评估阶段；B完整覆盖、一次新训练及C/D最终验收仍未完成。

08:59 UTC：原生日志确认B评估分片已启动，原生状态Running，最新读取窗口无Traceback；该观察不等于B全量评估完成。

09:15 UTC：B完整31,392条／37,728目标／104支持类已由远端reader验证，108候选、Top20、无历史过滤不变。本地按类别独立重算全部@1/@5/@10/@20宏指标并核对原adapter文件、打乱Dev和metrics摘要通过。B预测SHA为 `8c34076d8ce0e4872bc3dc35602c85918713f7c9221937e199f9e22d13308948`，metrics SHA为 `102cda79e5f6e8f3ff0d70acedff126fcc1a2d3174612df5626f3e9961b33108`；收据 `artifacts/mind-history-order-20260923/B_independent_readback.json`。

同次观察确认新训练从原始 `f924e8...` 参数状态开始，optimizer为空、初始global step为0，没有从A模型继续微调。原生日志已完成5/2,025次更新，loss=5.675、grad_norm=48.91，均有限，无Traceback。当前为冻结的前500步单卡Torch fallback阶段，每步约74–76秒；进度条按全程单卡估计的总时长不适用于后续四卡阶段。保持已批准的预算、执行阶段和数据顺序。

11:17 UTC：首个checkpoint100已提交，训练继续到101/2,025步；第100步日志loss=1.629、grad_norm=9.926，均有限。云端回读提交标记和trainer_state的字节哈希、训练身份、step100／总预算2025／3epochs均通过，并核对全部11个payload文件的云端大小（224,909,303 bytes）。大张量未下载到Mac；本次属于提交及元数据核验，不替代后续完整张量恢复／GPU重载验收。收据：`artifacts/mind-history-order-20260923/progress/check-1117-checkpoint100/receipt.json`。checkpoint100的adapter文件SHA为 `c15b49e0838cb1661d5633c720824d07d2b4a5ccd67fa5d567b8d5d372400c0a`。训练仍按原计划继续前500步单卡阶段。

13:25 UTC：checkpoint200按计划提交，训练继续到202/2,025步，loss=1.592、grad_norm=10.37，均有限且无训练异常。已核对提交标记／trainer_state哈希、冻结身份、200步／2025总步数及11文件云端大小（224,913,002 bytes）；checkpoint100提交标记ETag保持不变。该检查仍仅覆盖提交和元数据，不是新一轮完整张量或GPU重载验收。收据：`artifacts/mind-history-order-20260923/progress/check-1325-checkpoint200/receipt.json`。继续原单卡阶段，无额外训练或配置变更。

15:33 UTC：checkpoint300按计划提交，训练继续到305/2,025步，loss=1.561、grad_norm=10.16，均有限且无训练异常。提交标记／trainer_state哈希、冻结身份、300步／2025总步数及11文件云端大小（224,916,677 bytes）核对通过，先前检查点的提交ETag不变。验收范围仍为提交及元数据；完整张量和GPU重载待原定阶段检查。收据：`artifacts/mind-history-order-20260923/progress/check-1533-checkpoint300/receipt.json`。继续原单卡阶段，无配置变更。

16:18 UTC：原生日志推进到340步，无训练异常，checkpoint100/200/300提交ETag不变。紧凑产物reader在Azure CLI刷新storage token时出现本机代理连接重置，随后原reader及限定 `NO_PROXY='*'` 的重试均完成，全部紧凑文件哈希一致；未改动远端任务。后续原生日志、AMLT状态和紧凑产物只读命令均可使用这一进程级直连设置。观察：`artifacts/mind-history-order-20260923/progress/check-1618-observation.json`。

17:33 UTC：checkpoint400按计划提交，训练继续到401/2,025步，loss=1.568、grad_norm=9.331，均有限。提交标记／trainer_state哈希、冻结身份和11文件云端大小（224,920,373 bytes）核对通过。worker按已冻结的 `persistent_checkpoint_limit=3` 保留200/300/400并清理100，200/300提交ETag不变；这属于原有保留规则，不是检查点丢失。完整张量／GPU重载验收仍待原定阶段。收据：`artifacts/mind-history-order-20260923/progress/check-1733-checkpoint400/receipt.json`。继续原单卡阶段。

19:33–19:40 UTC：单卡阶段恰好完成500步，checkpoint500提交后按原规则保留300/400/500。远端完整状态检查得到373个参数张量、373项Adam状态，所有Adam游标和scheduler均为500，总预算仍为2025步／3epochs。checkpoint500的adapter文件SHA为 `fa09045386248a11cba913083ba1b86cdcb5f9408ea2f49f71c25cc62242e9d1`，11个payload文件总大小224,924,043 bytes。

四卡接续已独立回读验收：四个rank实际参数、Adam、scheduler摘要逐项匹配源checkpoint500，四rank RNG恢复精确通过；FLA 0.5.2、causal-conv1d 1.7.0及实际fast-path绑定一致。训练身份仅按批准方案改变world_size、microbatch和梯度累积，全局batch32和完整cosine horizon不变。独立CPU重建的16,000样本前缀与冻结游标合同一致，实际dataloader顺序守卫保持启用。日志已确认运行至533/2,025步，第530步loss=1.571、grad_norm=9.783，均有限，窗口无训练异常。迁移lineage为 `a354672434d5fa8d2bcc2066da38985a08279d6d153bba808ae0f8bff9fd3a28`；收据：`artifacts/mind-history-order-20260923/migration_independent_readback.json`。这项验收覆盖500步边界的状态迁移；最终训练游标、最终adapter/GPU重载和C/D全量评估仍待完成。

19:48–19:52 UTC：AMLT与原生状态均为Running，当前只有原始stdout、未观察到retry或训练异常；19:48日志推进至790/2,025步，loss=1.512、grad_norm=10.53，均有限。稍后的持久产物观察已出现final阶段checkpoint600/700/800。checkpoint800的提交标记、trainer_state及游标字节哈希、迁移目标身份和13个payload云端大小（204,970,018 bytes）核验通过。独立CPU重建验证其25,600次曝光前缀、下一全局batch、500步边界首个实际microbatch与原合同完全一致，四卡300次实际更新的样本顺序守卫通过。收据：`artifacts/mind-history-order-20260923/progress/check-1948-checkpoint800/receipt.json`。790步日志与800步检查点来自先后两次观察，最终训练、严格重载和C/D验收仍待完成；保持原任务继续运行。

20:03 UTC：AMLT与原生状态均为Running，唯一stdout已推进至1,202/2,025步，无retry或训练错误；第1,200步loss=1.5、grad_norm=10.18，均有限。final阶段按原保留规则留下checkpoint1000/1100/1200。checkpoint1200的提交及紧凑文件哈希、冻结身份和13个payload大小（204,984,730 bytes）核验通过；独立CPU重建再次匹配38,400次曝光前缀、下一batch和迁移边界，四卡700次实际更新的顺序守卫通过。收据：`artifacts/mind-history-order-20260923/progress/check-2003-checkpoint1200/receipt.json`。训练及C/D最终验收仍未完成，无需干预健康任务。

20:18 UTC：AMLT与原生状态均为Running，唯一stdout推进至1,626/2,025步，无retry或训练错误；第1,625步loss=1.472、grad_norm=9.026，均有限。final阶段保留checkpoint1400/1500/1600，旧检查点按原规则清理。checkpoint1600提交及紧凑文件哈希、冻结身份、13个payload大小（204,999,459 bytes）通过核验；独立CPU重建匹配51,200次曝光前缀、下一batch及迁移边界，四卡1,100次实际更新的顺序守卫通过。收据：`artifacts/mind-history-order-20260923/progress/check-2018-checkpoint1600/receipt.json`。继续原定训练，最终重载及C/D评估仍待完成。

20:33–20:36 UTC：新训练恰好完成2,025步／3epochs；最终checkpoint2025已提交，保留1900/2000/2025。独立CPU重建完整64,800次曝光，前缀摘要与冻结H50合同一致，四卡1,525次实际更新的顺序守卫通过，下一batch为空；13个payload大小总计205,015,068 bytes。最终adapter文件SHA为 `e196bf5341a7223035b4e3e19c0977a08332863e49d2eedfa9aafd3d17e811ec`，373个张量／68,212,000 bytes；远端完整adapter检查与新进程strict reload通过，全部张量逐项相同，4条参考输入的108个SID logits在atol=rtol=0下完全一致。独立回读再次核对初始化、训练合同、完整游标、原数值路径及重载证据，收据：`artifacts/mind-history-order-20260923/training_independent_readback.json`。原生日志已确认C的4个分片（各7,848条，batch8）启动，任务继续Running、未见retry或训练错误。C/D完整覆盖、独立指标和成对分析仍未验收；没有新增训练或改变预算。

20:48–20:50 UTC：C已通过远端完整预测验证，31,392条／37,728目标／104支持类，原Dev及已验收新adapter身份一致；本地按104类独立重算全部@1/@5/@10/@20宏指标通过。C预测SHA为 `fccbcd32e1ba10c536806e7b763e38d0342be6dc7c0630008d2fb71e94ccd05a`，metrics SHA为 `ed2fb8185cd5f995aa768b3d50c315b3cc2df16ffa7c2f1ee0bb158b1c74472f`；收据：`artifacts/mind-history-order-20260923/C_independent_readback.json`。C相对A的R@10、R@20、NDCG@10、NDCG@20分别变化−0.0016、+0.1842、+0.2220、+0.2701个百分点，尚无成对区间。D四分片均在持续产出，远端文件名范围分别指示3,072／3,840／2,304／3,840条暂存预测（仅进度、非内容验收）。AMLT及原生状态Running，无retry或错误；最终checkpoint2025提交ETag不变。继续原任务，等待D及最终独立分析。

后续观察入口：`scripts/analysis/read_history_ablation_progress.py --receipt manifests/mind_history_order_20260923_v1.json --output <新的观察目录>` 读取所有原生 stdout；`scripts/analysis/read_history_order_state.py --output <新的观察目录>` 仅回读紧凑持久产物，完成后加 `--require-complete` 验证结果manifest、模型／数据配对、完整曝光及宏指标。两者使用 `/Users/river/miniforge3/envs/amlt/bin/python`。独立紧凑reader另有3项正例／篡改拒绝测试通过。

| 条件 | Macro R@10 (%) | Macro R@20 (%) | Macro NDCG@10 (%) | Macro NDCG@20 (%) | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 20.3275 | 33.6553 | 10.1325 | 13.4663 | 复用原已验收结果 |
| B | 20.1622 | 33.4186 | 10.6675 | 13.9747 | 远端完整验证及紧凑独立回读通过 |
| C | 20.3258 | 33.8395 | 10.3544 | 13.7363 | 远端完整验证及紧凑独立回读通过 |
| D | — | — | — | — | 四分片评估进行中 |

B相对A的四项变化依次为−0.1652、−0.2367、+0.5351、+0.5084个百分点：Recall略低而NDCG较高，尚未显示一致退化。D与成对区间未完成，不能据此断言顺序无用或已证实收益。

最终在远端对全部预测独立重算原interest-macro指标、固定长度≥2子集、Top1变化率、Top10重合率，以及2000次用户成对Bayesian bootstrap差值区间。完整权重、优化器、预测留远端，本地保留紧凑收据与报告。整体实验尚未完成。
