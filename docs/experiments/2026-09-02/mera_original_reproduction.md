# MeRA 原始 Arts/Grocery 双口径复现

## 1. 研究问题（Research Questions）

本实验在作者发布的 Amazon Arts 与 Amazon Grocery 处理后数据上复现 Full MeRA，
不从 Amazon Reviews 原始评论重建 split。基础模型固定为
`Qwen/Qwen2-1.5B@8a16abf2848eda07cc5253dec660bf1ce007ad7a`，比较：

1. `Code-faithful`：执行作者 GitHub commit
   `c07bdbe57bf82e5c938b478035baee78d630b17c` 的有效训练语义。
2. `Paper-spec`：在相同模型、数据、seed、训练与评测流程下，使用论文报告的
   `alpha=0.05, beta=0.1, N=10, T=0.05` 及 per-category structure anchor。

主结果完整报告逐样本平均 Precision@1–4 与 Recall@1–4。按预注册约定，作者结果
与从头训练结果的数值差异只作描述，不设数值通过门槛；工程完成以输入、训练、
保存回载、预测、指标和下载产物全部验证为准。

## 2. 训练数据构造（Training Data Construction）

数据直接来自作者仓库所附 JSON、二级类别映射、类别树与共现矩阵。所有输入均由
`manifests/mera_original_v1.json` 记录 SHA-256。Train/Valid 每行一个未来新类别，
Test 每行一个或多个未来新类别；所有 split 均验证为 target 与 History 零重叠。

| dataset | Train | Valid | Test | categories | parents |
|---|---:|---:|---:|---:|---:|
| Arts | 20,186 | 2,523 | 2,524 | 143 | 22 |
| Grocery | 34,981 | 4,373 | 4,373 | 172 | 32 |

Code-faithful 的 Arts 使用 `alpha=0.01, beta=0.01, N=5`，Grocery 使用
`alpha=0.1, beta=0.1, N=10`，均不应用论文阈值；Paper-spec 两个数据集均使用
`alpha=0.05, beta=0.1, N=10, T=0.05`。两种条件都保留作者的带双引号类别 special
token、input embedding 类别名均值初始化、可训练 `embed_tokens`/`lm_head` 与 LoRA。

### 2.1 Arts 真实训练样例

来源：作者 `data/Arts/train.json` 第 0 行；这是监督 target record，不是模型生成。

英文原文：

> Given a list of categories of art crafts the user has bought before, please predict
> what new category of art crafts this user will be interested in in the future. The
> predicted categorie should be different from the history categories.
>
> The user has given high ratings to the following categories of art crafts before:
> "Jewelry Casting Supplies", "Painting", "Painting", "Painting", "Drawing",
> "Drawing", "Embroidery", "Painting", "Yarn", "Crochet Hooks", "Jewelry Making
> Tools & Accessories", "Jewelry Findings", "Crochet Hooks", "Knitting Needles",
> "Knitting Needles", "Knitting Needles", "Drawing", "Storage Boxes & Organizers",
> "Sewing Patterns & Templates"
>
> Target: "Craft Supplies"

中文翻译：

> 给定用户过去购买过的艺术手工类别，预测其未来可能感兴趣、且未出现在历史中的
> 一个新类别。历史包含珠宝铸造用品、绘画、素描、刺绣、纱线、钩针、编织针、
> 收纳盒、缝纫模板等交互；监督目标为“手工用品（Craft Supplies）”。

### 2.2 Grocery 真实训练样例

来源：作者 `data/Grocery_and_Gourmet_Food/train.json` 第 0 行；这是监督 target
record，不是模型生成。

英文原文：

> Given a list of categories of groceries the user has bought before, please predict
> what new category of groceries this user will be interested in in the future. The
> predicted categorie should be different from the history categories.
>
> The user has given high ratings to the following categories of groceries before:
> "Cooking & Baking", "Canned, Jarred & Packaged Foods", "Pastries & Bakery",
> "Canned, Jarred & Packaged Foods", "Chocolate Candy", "Chocolate Candy",
> "Candy & Chocolate"
>
> Target: "Chocolate"

中文翻译：

> 给定用户过去购买过的食品类别，预测其未来可能感兴趣、且未出现在历史中的一个
> 新类别。历史包含烹饪烘焙、罐装/瓶装/包装食品、糕点烘焙、巧克力糖果等；监督
> 目标为“巧克力（Chocolate）”。

## 3. 测试设计（Testing）

### 3.1 测试模型与位置

训练与推理运行于 Feeds Singularity。每个条件使用独立 `80G1-A100 / Standard`
作业；四项不使用 DDP。模型以 8-bit 加载，训练 FP16 LoRA 及完整
`embed_tokens`/`lm_head`。完整训练前执行固定模型 snapshot 预取、四项 8-step
smoke、fresh reload 与 64-row 真实 evaluator smoke。

作者固定的 `bitsandbytes==0.37.2` 不包含当前 Feeds CUDA 12.6 二进制，GPU 探针
在四项 smoke 中一致失败；按预注册兼容策略改用 `bitsandbytes==0.43.1`，仍保持
8-bit base。基础镜像的可选 Apex/DeepSpeed 与 Torch 2.3 不兼容，但本实验使用
PyTorch native AMP 且不使用 DeepSpeed；实现显式禁用未使用的 Apex Trainer 探测，
并直接移除 Accelerate FP32 output wrapper 后生成训练进程参考。

### 3.2 测试数据来源

Arts Test 有 2,524 行，平均 3.024 个正例，范围 1–32；Grocery Test 有 4,373 行，
平均 3.380 个正例，范围 1–43。推理输入严格为按原顺序拼接的带双引号 History
类别 token，不附加训练时的 Alpaca instruction。模型对最后位置 logits 限制到该
数据集全部类别 token，直接取 Top-4；`num_beams=4` 在作者代码中实质为 restricted
Top-K，不执行自回归 beam generation。

### 3.3 测试样例

以下均来自本次 Full Test 的下载预测，不是作者 prediction。为避免主观挑选，
每个方向都在 target 数不超过 4 的 Test 行中，选择 Hit@4 差距最大、再取最小
`row_index` 的样例；选择规则与完整字段保存在本地 readback audit。

**Arts，Paper-spec 更好，row 277。**

English original:

> History: "Sewing Notions & Supplies", "Jewelry Findings", "Jewelry Casting
> Supplies", "Beads & Bead Assortments", "Jewelry Findings", "Jewelry Making
> Tools & Accessories", "Jewelry Findings", "Jewelry Findings", "Beading
> Supplies", "Painting", "Beading Supplies"
>
> Correct categories: "Boards & Canvas", "Drawing", "Craft Supplies",
> "Trim & Embellishments"
>
> Code-faithful Top-4: "Shop Wedding Fabrics", "Shop Knit Fabrics",
> "Project Type", "Garment Care" (0/4 hits)
>
> Paper-spec Top-4: "Craft Supplies", "Trim & Embellishments", "Painting",
> "Quilting" (2/4 hits)

中文翻译：历史包含缝纫辅料、珠宝配件（重复 4 次）、珠宝铸造用品、珠饰组合、
珠宝制作工具、串珠用品（重复 2 次）和绘画；正确类别是画板与画布、素描、手工
用品、饰边与装饰。Code-faithful 预测婚礼面料、针织面料、项目类型和服装护理，
未命中；Paper-spec 命中手工用品与饰边装饰。

**Arts，Code-faithful 更好，row 478。**

English original:

> History: "Art Paper", "Drawing", "Paper & Paper Crafts", "Painting", "Drawing"
>
> Correct categories: "Woodcrafts", "Die-Cutting & Embossing"
>
> Code-faithful Top-4: "Woodcrafts", "Trim & Embellishments",
> "Sewing Notions & Supplies", "Die-Cutting & Embossing" (2/2 hits)
>
> Paper-spec Top-4: "Painting", "Art Paper", "Trim & Embellishments",
> "Sewing Notions & Supplies" (0/2 hits)

中文翻译：历史为美术纸、素描、纸艺、绘画、素描；正确类别为木工艺和模切压花。
Code-faithful 两个正确类别均进入 Top-4；Paper-spec 未命中，并把历史中的绘画和
美术纸排在前两位。

**Grocery，Paper-spec 更好，row 876。**

English original:

> History: "Cereals", "Canned, Jarred & Packaged Foods", "Cereals",
> "Pasta & Noodles", "Tea", "Chocolate Candy", "Chocolate Candy", "Snack Foods"
>
> Correct categories: "Candy & Chocolate", "Herbs, Spices & Seasonings",
> "Cooking & Baking"
>
> Code-faithful Top-4: "Subscribe and Save: Coffee, Tea, and Beverages",
> "Fall snacks", "Chocolate Grocery Halloween", "Subscribe and Save: Grocery"
> (0/3 hits)
>
> Paper-spec Top-4: "Herbs, Spices & Seasonings", "Candy & Chocolate",
> "Cooking & Baking", "Bottled Beverages, Water & Drink Mixes" (3/3 hits)

中文翻译：历史包含谷物早餐、罐装/瓶装/包装食品、意面、茶、巧克力糖和零食；
正确类别为糖果与巧克力、香草香料调味品、烹饪烘焙。Code-faithful 的四个预测均
未命中；Paper-spec 的前三个预测覆盖全部正确类别。

**Grocery，Code-faithful 更好，row 573。**

English original:

> History: "Herbs, Spices & Seasonings", "Cooking & Baking", "Cooking & Baking",
> "Cooking & Baking", "Jams, Jellies & Sweet Spreads", "Nut & Seed Butters",
> "Cookies", "Nut & Seed Butters", "Bottled Beverages, Water & Drink Mixes",
> "Bottled Beverages, Water & Drink Mixes", "Bottled Beverages, Water & Drink Mixes"
>
> Correct categories: "Coffee", "Canned, Jarred & Packaged Foods", "Cereals"
>
> Code-faithful Top-4: "Candy & Chocolate", "Snack Foods",
> "Canned, Jarred & Packaged Foods", "Coffee" (2/3 hits)
>
> Paper-spec Top-4: "Snack Foods", "Tea", "Cooking & Baking",
> "Candy & Chocolate" (0/3 hits)

中文翻译：历史包含香草香料调味品、烹饪烘焙、果酱甜味抹酱、坚果籽酱、曲奇及
瓶装饮料；正确类别为咖啡、罐装/瓶装/包装食品和谷物早餐。Code-faithful 命中前
两个正确类别；Paper-spec 未命中，并预测了历史中的烹饪烘焙。

### 3.4 Judge 与指标过程

本实验没有 LLM judge。对每条 Test 样本的正确类别集合 `Y` 和类别 Top-K：

`Precision@K = |TopK ∩ Y| / K`，
`Recall@K = |TopK ∩ Y| / |Y|`。

逐样本计算后取宏平均。评测要求每行恰好四个合法且不重复的类别、完整唯一的
`row_index`，并从保存的 prediction 独立复算 metrics。

## 4. 实验结果（Results）

论文结果表的 MeRA 行只报告 `K=1–3`。根据论文表格，正式比较基准为：

| dataset | source | P@1 | P@2 | P@3 | R@1 | R@2 | R@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Arts | paper table, MeRA row | 0.2896 | 0.2670 | 0.2413 | 0.1204 | 0.2120 | 0.2786 |
| Grocery | paper table, MeRA row | 0.3206 | 0.2734 | 0.2515 | 0.1179 | 0.1980 | 0.2662 |

Gate A 另行完成作者仓库所附 prediction 文件的独立复算。Arts 的 `K=1–3` 与论文
表格一致；Grocery 略有差异，因此下表不能替代论文结果：

| dataset | source | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 | R@3 | R@4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | author-provided predictions | 0.2896 | 0.2670 | 0.2413 | 0.2183 | 0.1204 | 0.2120 | 0.2786 | 0.3288 |
| Grocery | author-provided predictions | 0.3174 | 0.2752 | 0.2491 | 0.2310 | 0.1175 | 0.1989 | 0.2635 | 0.3224 |

Full experiment `mera-original-qwen2-full-0902-v1` 四项均为 AMLT `pass`，并已将
四份完整 Test prediction、final adapter 和全部 manifest 内文件下载到本地。下表
是从本次从头训练 prediction 独立复算的结果：

| dataset | contract | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 | R@3 | R@4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | Code-faithful | 0.1652 | 0.1593 | 0.1486 | 0.1409 | 0.0662 | 0.1263 | 0.1745 | 0.2184 |
| Arts | Paper-spec | 0.1719 | 0.1652 | 0.1511 | 0.1449 | 0.0681 | 0.1340 | 0.1799 | 0.2308 |
| Grocery | Code-faithful | 0.1393 | 0.1353 | 0.1326 | 0.1243 | 0.0575 | 0.1140 | 0.1644 | 0.2016 |
| Grocery | Paper-spec | 0.2349 | 0.2159 | 0.2011 | 0.1869 | 0.0894 | 0.1589 | 0.2194 | 0.2694 |

相对论文表格的逐指标差值（本次训练减论文结果；论文未报告 `K=4`）为：

| dataset | contract | ΔP@1 | ΔP@2 | ΔP@3 | ΔR@1 | ΔR@2 | ΔR@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Arts | Code-faithful | -0.1244 | -0.1077 | -0.0927 | -0.0542 | -0.0857 | -0.1041 |
| Arts | Paper-spec | -0.1177 | -0.1018 | -0.0902 | -0.0523 | -0.0780 | -0.0987 |
| Grocery | Code-faithful | -0.1813 | -0.1381 | -0.1189 | -0.0604 | -0.0840 | -0.1018 |
| Grocery | Paper-spec | -0.0857 | -0.0575 | -0.0504 | -0.0285 | -0.0391 | -0.0468 |

四项都由 validation early stopping 正常提前结束；`final_adapter` 是 Trainer 回载
best checkpoint 后保存的状态：

| dataset | contract | actual/max optimizer steps | best step | best eval loss | train runtime |
|---|---|---:|---:|---:|---:|
| Arts | Code-faithful | 264/800 | 240 | 1.8996 | 82.5 min |
| Arts | Paper-spec | 264/800 | 240 | 2.5398 | 80.2 min |
| Grocery | Code-faithful | 312/1,380 | 288 | 2.7581 | 103.5 min |
| Grocery | Paper-spec | 216/1,380 | 192 | 2.5932 | 72.9 min |

本地 readback 对每个条件验证 22 个 manifest 文件的 SHA-256，prediction 行数分别
精确为 Arts 2,524、Grocery 4,373，`row_index` 完整唯一，Top-4 类别合法且不重复，
保存的 target 与冻结 Test 逐行一致。远程 metrics 与本地逐样本重算在完整浮点值上
完全一致。四项磁盘 adapter 都与训练 state manifest bit-exact；fresh live state
等于磁盘 tensor 显式转换到推理 dtype 后的值；两次独立 fresh reload 的 candidate
logits 最大误差均为 `0.0`，Top-4 完全一致。训练进程首次量化 base 与 fresh
inference 重新量化 base 的差异仍只记录，不作为 adapter 序列化门禁。

就本次固定 seed 运行而言，Paper-spec 在两个数据集的八个指标上都高于
Code-faithful；Arts 的差距较小，Grocery 的差距较大。两种口径仍都低于论文表格，
也低于作者所附 prediction。按预注册契约，这些差值是报告对象，不构成工程复现的
二元通过/失败。
案例中 Paper-spec 有时把更一般的主类别排到前列，而 Code-faithful 有时偏向季节性
或订阅类节点；反向案例同样存在，而且两种口径都会偶尔预测 History 内类别。因此
这只能作为可能机制线索，不能由四个单 seed 条件推出因果解释。

四条件正式 smoke `mera-original-qwen2-four-condition-smoke-0902-v9` 先于 Full
达到 `4/4 pass`。每项均完成 8 optimizer steps、完整类别梯度审计、checkpoint、
adapter 保存、两次独立 fresh reload、64-row Test 评测与 22 文件发布。下载后的
smoke prediction 也已独立复算：

| dataset | contract | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 | R@3 | R@4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | Code-faithful | 0.0781 | 0.1250 | 0.1302 | 0.1250 | 0.0276 | 0.0890 | 0.1269 | 0.1633 |
| Arts | Paper-spec | 0.1406 | 0.1328 | 0.1250 | 0.1211 | 0.0547 | 0.1139 | 0.1568 | 0.1822 |
| Grocery | Code-faithful | 0.0781 | 0.1094 | 0.1510 | 0.1484 | 0.0306 | 0.0763 | 0.1618 | 0.2228 |
| Grocery | Paper-spec | 0.1094 | 0.1562 | 0.1458 | 0.1250 | 0.0368 | 0.0926 | 0.1375 | 0.1805 |

正式 Test 始终使用与 smoke 门禁相同的 fresh inference 路径。

## 5. 数据与产物路径（Artifact Paths）

- 上游：<https://github.com/GabyUSTC/MeRA/tree/c07bdbe57bf82e5c938b478035baee78d630b17c>
- 本地冻结数据：`data/external/mera_original/c07bdbe57bf82e5c938b478035baee78d630b17c/`
- 来源 manifest：`manifests/mera_original_v1.json`
- canonical matrix：`configs/mera_original/experiment_matrix_v1.yaml`
- 训练入口：`src/new_interest/training/train_mera_original.py`
- 评测入口：`src/new_interest/evaluation/evaluate_mera_original.py`
- 审计入口：`src/new_interest/experiments/audit_mera_original.py`
- Full 下载回读脚本：`scripts/audit_mera_original_full_readback.py`
- 本地 Gate A：`artifacts/mera_original/gate_a/source_audit.json`
- 冻结 AMLT 上传根：`amlt_inputs/mera_original_qwen2_0902_v1/`
- CPU prefetch experiment：`mera-original-qwen2-prefetch-0902-v1`
- smoke experiment：`mera-original-qwen2-four-condition-smoke-0902-v1`
- 最终 smoke experiment：`mera-original-qwen2-four-condition-smoke-0902-v9`，
  `4/4 pass`
- 最终 smoke 下载回读：`artifacts/mera_original/smoke_v9_readback/`
- reload 诊断 experiment：`mera-original-qwen2-reload-diagnostic-0902-v8`，terminal
  `pass`
- Full experiment：`mera-original-qwen2-full-0902-v1`，四项 terminal `pass`：
  `arts-code-full`、`arts-paper-full`、`grocery-code-full`、
  `grocery-paper-full`
- Full 本地下载根：
  `artifacts/mera_original/full_v1_readback/mera-original-qwen2-full-0902-v1/`
- Full 本地综合验收：
  `artifacts/mera_original/full_v1_readback/mera-original-qwen2-full-0902-v1/local_readback_audit.json`
- 每项完整 prediction 与 metrics：上述下载根下
  `<job>/run/evaluation/predictions.jsonl` 和 `<job>/run/evaluation/metrics.json`
- 每项 final adapter：上述下载根下 `<job>/run/final_adapter/`
- 每项远程发布证明：上述下载根下 `<job>/artifact_manifest.json` 与
  `<job>/full_success.json`
