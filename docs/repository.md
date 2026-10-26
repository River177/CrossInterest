# 当前代码、实验入口与 Git 管理

整理日期：2026-10-26。本次只归并代码、归档配置并纳入版本管理，没有新增训练或评估任务。

## 从哪里开始

| 内容 | 入口 |
|---|---|
| CIR→NI 与相关对照 | [实验总览](experiments/2026-09-30/cir_ni_experiment_overview.md) |
| Text125 / Token125 | [匹配历史表示实验](experiments/2026-09-30/mind_history_token.md) |
| NI＋双向名称/ID对齐 | [联合训练实验](experiments/2026-10-01/mind_token_joint_alignment.md) |
| 所有报告、设计与运行记录 | [日期索引](experiments/README.md) |
| 原路径与归档位置 | [归档清单](../manifests/repository_archive_20261026.json) |
| 归并来源与哈希 | [代码来源清单](../manifests/source_integration_20261026.json) |

`src/new_interest/` 是当前开发代码，仍按 data / training / evaluation / experiments 分层。历史实验的复现身份以其冻结包和验收记录为准。修改当前源码后开展新实验，需要新的冻结包、运行身份和输出目录；不能覆盖历史包或沿用其哈希。

## 本次归并

补回了 CIR 和历史消融共同依赖的 Backbone/OI→NI 编排、全局样本游标、断点执行迁移及训练状态审计模块，并恢复其配套配置和 `requirements-cir-worker.txt`。

从最新 joint-alignment 冻结包合入训练暴露计数、初始参数审计、混合任务完成长度、模型 revision 绑定和 125 模型 token / 108 排名候选分离。通用实现同时支持旧的 108 类 SID 训练；没有把 joint 的专用长度规则写死到普通 CIR 流程。当前工作区原有的数据读取、CoT、候选评估、Trainer 注入和 checkpoint 删除竞态修复继续保留。

`runtime_overrides/` 保存原 worker 使用的存储重试模块，只有 worker 明确设置 `CIR_STORAGE_RETRY_ENABLED=1` 并加入对应 PYTHONPATH 才启用。Linux/CUDA 正式 worker 使用冻结依赖；本机 CPU 测试环境不作为正式训练环境。

## 配置与 worker 归档

- `configs/amlt/` 保留 25 份主线入口或准备程序仍引用的模板，保留原因逐项记录在清单中。目录位置不代表远端作业正在运行。
- `configs/amlt/archive/2026-10-26/` 收纳本次移入的 165 份历史提交配置；已有更早归档保留原位。
- `scripts/archive/amlt/2026-10-26/` 收纳 45 个不在保留入口/准备脚本依赖链上的旧 worker。
- 30 个仍被主线、准备程序或共享依赖引用的 worker/helper 留在 `scripts/amlt/`。

移动 YAML 时只调整需要变化的 `code.local_dir` 和仓库内 worker 路径，目标代码根目录与原来一致。指向冻结包的 YAML 保留包内原始 worker 路径。没有修改冻结包内的文件、清单或哈希。报告和历史 manifest 的原始提交路径继续保留，可通过归档清单定位；Markdown 链接指向当前实际位置。

历史配置依赖外部输入、挂载、AMLT 身份及冻结运行包，不保证普通 Git clone 后可直接提交。准备脚本中的历史 bundle 来源也需要先恢复，不能用当前源码冒充它。

## 数据与恢复

`data/`、`artifacts/`、`amlt_inputs/`、`outputs/`、`tmp/`、`.codex_tmp/`、`.vscode/` 和本地 `.venv/` 不进入 Git；本次未删除这些内容。数据定义见 [pipeline](data/pipeline.md)，数据和结果通过 metadata、统计及恢复清单追溯。

本机缺少 `data/raw/MINDlarge_reclassified`，从头重建需要先取得授权来源。Text125/Token125 与 joint 的真实远端结果前缀、adapter/预测哈希分别记录在：

- [Text125/Token125 恢复记录](../manifests/mind_history_token_restoration_20261001_v1.json)
- [Joint 恢复记录](../manifests/mind_token_joint_alignment_restoration_20261002_v1.json)

保留大型产物在远端；本次不下载权重、checkpoint 或完整预测。上述恢复指针的远端可访问性没有在本次重新验证。

## Git 工作流

开发分支为 `codex/consolidate-experiments-20261026`。代码、测试、配置、紧凑报告和恢复清单纳入 Git。提交前运行：

```bash
python scripts/check_repository.py
pytest -q
```

检查工具只读：扫描 Git 候选文件中的大文件、模型/数据输出和常见凭据模式，校验本次归档文件哈希及 AMLT 代码根目录。它不自动提交、不安装 Git hook、不下载数据。它是常见误提交检查，不能代替对提交内容的人工审阅。修改归档文件时须同步归档清单的哈希。

本次修改前的源代码快照、Git diff/status、测试完整日志留在被忽略的 `artifacts/repository-integration-20261026/`；冻结包和原交接压缩包保留原位。正式训练验收仍以各实验原始收据为准，本次验证范围见[整理记录](maintenance/2026-10-26-repository-integration.md)。
