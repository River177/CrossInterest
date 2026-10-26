# 2026-10-26 代码归并、配置归档与 Git 整理

本次将截至 2026-10-02 的 CIR→NI、Text125/Token125 和双向对齐运行代码补回开发树，并保留此前未提交的源码、测试、配置及紧凑报告。未改变训练配方、数据或历史冻结包，未新增实验或远端操作。

## 内容与追溯

- 恢复 13 个此前缺失的 Python 模块以及所需配置、CIR worker 依赖和存储重试模块。
- 合入初始参数审计、真实训练暴露、全局游标与断点迁移；支持混合任务不同完成长度，保持旧 SID 数据的三 token 合同。
- 训练与评估绑定基座 revision；模型词表和排名候选分开处理，兼容 108/125 两种模型词表。
- 保留当前工作区的 Trainer 显式注入、checkpoint 清理竞态修复及较新的数据/CoT/候选评估实现；断点 hydration 新增完整标记复制与验证。
- 165 份 AMLT 配置和 45 个 worker 归档；25 份主线入口/准备依赖模板、30 个共享或仍被引用的 worker 留在原入口。归档不表示远端状态判定。
- 历史报告/manifest 的原始提交路径与哈希保持可追溯；归档映射记录新位置及 code.local_dir 的前后关系。Markdown 链接及测试更新至实际位置。
- 实验索引重建，覆盖 67 个结果报告、设计与运行记录。
- Git 忽略本地 staging/readback/临时目录；新增只读提交检查脚本。此次将此前散落的轻量研究资料纳入版本控制，无数据、权重、完整预测或凭据模式匹配项进入提交候选。

[源码来源及哈希](../../manifests/source_integration_20261026.json)；[归档路径及哈希](../../manifests/repository_archive_20261026.json)。这些是开发树的归并记录，不是新的 GPU 训练验收。

## 验证

| 检查 | 结果 |
|---|---|
| 归并前系统环境测试 | 缺少 peft 导致 collection 失败；排除该模块后为 1,578 passed / 41 skipped / 2 failed，两个失败均因缺少 peft |
| 本地隔离环境完整测试 | **1,628 passed / 8 skipped**，61.89 秒 |
| 最后追加的包导入与主线游标/绑定测试 | **29 passed** |
| 仅 Git 暂存文件检出副本（无 data/artifacts/amlt_inputs） | **1,623 passed / 13 skipped**，53.00 秒 |
| 移除两处隐含本地依赖后的定向验证 | **15 passed** |
| Python 源码语法 | src 227、scripts 80、runtime_overrides 2 个文件通过 |
| Shell 语法 | 97 个脚本通过 `bash -n` |
| 配置解析 | 130 个 JSON、285 个 YAML 通过 |
| 归档完整性 | 所有移动文件的记录哈希一致；AMLT code.local_dir 仍指向原代码根 |
| AMLT 字面量 worker 路径 | 所有本地存在代码根目录中的引用均存在 |
| 原冻结交接包 | 1,843 项文件哈希与 Python 语法通过；Parquet 元数据可读；压缩包 SHA-256 通过 |
| 提交候选检查 | 无超过 10 MiB 的文件、禁止的数据/模型输出或常见凭据模式匹配项 |

完整测试使用 `.venv/bin/python -m pytest -q`。该忽略目录采用本机 Python 3.13.12 的 system-site-packages，并仅在隔离环境内补装 `peft==0.18.0`；Torch 2.12.0、Transformers 5.9.0 等来自本机。这验证 CPU/local 逻辑，不代表 Linux/CUDA 冻结 worker 环境验收。依赖缺失时，peft 专属测试现按既有约定跳过。普通检出副本的 5 个额外跳过来自未提供的实验数据/冻结材料；游标单元测试直接覆盖当前源码，候选路径测试创建临时 Parquet，均不再要求本机实验目录。

本次未重新验证远端存储访问、GPU 训练、AMLT 提交或模型指标。历史模板 `configs/amlt/mind_rho_student_full_0916_v1.yaml` 指向的 `artifacts/rho-student-20260916/bundle-full-v1` 在本地不存在；它仍被准备脚本引用，因此保留原入口。该缺失本次未通过下载补齐。

## 保留内容与本地恢复

本次没有删除 `data/`、`artifacts/`、`amlt_inputs/`、`outputs/` 或 `tmp/`。原始 MIND 本地缺失和大型训练产物留远端的状态不变。原交接压缩包及冻结目录未修改。

修改前快照为 `artifacts/repository-integration-20261026/before-source.tar.gz`，另有 `before.patch`、`before-status.txt` 和完整测试日志。该目录受 Git 忽略，保留本次整理前的未提交内容以供本机回溯；不是替代远端数据恢复的备份。

开发分支：`codex/consolidate-experiments-20261026`。本次只做本地提交，未推送远端。
