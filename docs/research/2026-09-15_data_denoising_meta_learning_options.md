# CrossInterest：区分噪声、困难样本与训练效用

研究日期：2026-09-15。本次是论文调研与方法设计，没有提交新训练，也没有下载模型、checkpoint 或完整预测。保持 History Behavior+Interest → 单个新兴趣 SID 的任务，不增加多正例监督。

当前建议：先研究数据的可学习性和训练效用。用 Dataset Cartography 做诊断，以 RHO-LOSS 的动态可约减损失作为第一项低复杂度对照；独立、可靠的 NI 参考数据准备好后，再考虑 Ren et al. 式元重加权。两种加权方法先分别比较，不组合成一个大系统。论文中的效果不能视为本项目预期收益。

## 1. 先明确什么叫噪声

| 情况 | 可以据什么判断 | 处理及边界 |
|---|---|---|
| 新闻类别映射错误 | 未来实际点击新闻的内容与映射类别不符 | 可审计的标签噪声；先修数据规则或建立质量标记 |
| 历史构造错误 | 时间穿越、顺序错误、标题与类别错位、完整历史的新颖性检查错误 | 确定性数据错误；修复并重新记录来源与版本 |
| 点击真实，但 H50 信息不足 | 已观察点击正确；提供给模型的历史无法充分约束目标 | 不能宣称是错误标签；可能仍有群体统计价值 |
| 困难但可泛化 | 当前模型预测困难，跨用户数据或训练后的模型能支持该目标 | 应保留，可能是新兴趣预测最有价值的样本 |
| 已学会的简单样本 | 当前损失低且稳定 | 未必需要继续大量分配更新，但仍可能维持类别覆盖和校准 |

目前没有人工审计出的噪声比例，不能把“训练困难”直接解释成“数据大部分错误”。跨 parent、低语义相似度、历史长都不是真值判别器。未知的偏好、曝光机制和偶然点击不能仅凭当前历史识别。

仅有一个观测 target 的 CE 可以拟合该采样协议下的条件分布，不等于断言其余类别永远不被喜欢。降权会改变有效训练分布，即使按类别归一化，也不能保证不损失类内少数模式。

## 2. 直接相关论文

| 论文 | 已核验的核心机制 | 对本项目的用途与限制 |
|---|---|---|
| Dataset Cartography，EMNLP 2020 [1] | 用训练过程中正确标签概率的均值与跨 epoch 波动形成 Data Maps；区分 easy、ambiguous、hard 区域 | 优先用于数据诊断。作者在其任务中发现 ambiguous 区域有助分布外泛化；不能将所有低置信样本判为噪声 |
| RHO-LOSS，ICML 2022 [2] | 当前模型的样本损失减去独立 holdout 模型对同一样本的损失，优先选择差值大的样本 | 最接近“难但值得学”。固定参考模型的损失是近似，不是用户兴趣的真正不可约误差 |
| Learning to Reweight Examples for Robust Deep Learning，ICML 2018 [3] | 用小型无偏、可靠验证集的损失，对训练样本权重做一步元梯度更新 | 高 loss 样本也可获正权重；目标是对参考分布的优化效用，而不是直接识别错标签 |
| Meta-Weight-Net，NeurIPS 2019 [4] | 学习从单个训练 loss 到样本权重的 MLP，使用少量干净 meta 数据更新映射 | 比手写 loss 阈值灵活；只看同一个标量 loss 时，不能区分同 loss 的噪声与有效难例 |
| Self-Guided Learning to Denoise for Robust Recommendation，SIGIR 2022 [11] | SGDL 把早期稳定记住的交互当作推定可靠的 reference，通过一步虚拟更新与 reference loss 学习训练样本权重；另用 scheduler 处理 reference 自身的噪声 | 去噪与元学习直接结合的最接近先例之一。reference 自生成，非外部干净真值；早期容易记住的交互可能不代表困难的新兴趣 |
| Denoising Implicit Feedback for Recommendation，WSDM 2021 [5] | ADT 用截断或重加权 CE 减少高损失隐式反馈影响 | 适合作为简单去噪对照。论文的 false-positive 诊断依赖 dwell/rating 等证据；我们当前点击标签没有同等质量真值 |
| Efficient Bi-Level Optimization for Recommendation Denoising，KDD 2023 [6] | 双层优化学习交互权重；高效求解中匹配两类推荐损失的梯度 | 很近的推荐去噪先例。其 BPR 与 alignment/uniformity 梯度匹配，不等于独立用户 clean meta-risk |
| Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation，2024 预印本 [7] | LLMHD 利用语义判断识别 hard 样本；先从高 loss 集移除，再把识别出的 hard 子集保留回来 | 与“不要误删难例”直接相关；History→新兴趣任务中，语义一致性可能偏好保守扩展，须检验跨域真例保留率。本次不据模板推断正式 venue |
| LESS，ICML 2024 [8] | 用少量目标示例，结合 LoRA 梯度特征、随机投影及 Adam 相关梯度相似度选择指令训练数据 | 借鉴低成本效用估计；它是目标数据选择，不是已验证的标签去噪器，也不等于精确 MAML |

补充：Confident Learning [9] 更适合候选“未来新闻内容→类别标签”的审计，而不是直接把 History→target 的低概率视作错误。其类别条件噪声假设与 MIND 中的实例依赖曝光、兴趣不确定性不同。PVI [10] 衡量输入对预测 target 的模型相关信息，可做 History 信息量诊断；低 PVI 不证明点击错误。

## 3. 第一项可验证设计：RHO-LOSS 的克制迁移

对训练样本 i，令 x_i=(History Behavior, Interest)，y_i 为原始单个 SID target。通过按用户分 fold 的教师得到：

\[
a_i=-\log p_{\phi^{-fold(i)}}(y_i\mid x_i).
\]

教师不能见过该用户的训练标签，初始化也不能是已经学过该用户的 Mixed checkpoint。相同用户、相关训练视图和辅助任务角色都必须归入同一 fold。名称、token、prompt、完整词表归一化及 SID loss 定义必须核对；不同模型族或输入协议的 OOF 分数不能直接混用。

当前学生在时刻 t 的难度为：

\[
b_i(t)=-\log p_{\theta_t}(y_i\mid x_i),\qquad r_i(t)=b_i(t)-a_i.
\]

优先考虑 r_i 较大的样本：其他用户训练出的模型能预测，但当前学生还没有学会。RHO-LOSS 原文 §3 Eq.3、Algorithm 1 正是 current loss − irreducible holdout loss；Appendix A 还给出了把训练集分两半、各自给另一半打分的做法，无需挪用测试集。

| 示意样本（数字仅解释公式） | 当前 loss b | OOF loss a | r=b−a | 解释 |
|---|---:|---:|---:|---|
| 已学会 | 0.3 | 0.2 | 0.1 | 可约减空间小 |
| 学生尚未学会，跨用户可预测 | 3.0 | 0.8 | 2.2 | 优先学习的候选 |
| 当前与参考均难预测 | 3.0 | 3.1 | −0.1 | 低优先级候选，不据此认定错标签 |

原论文是大 batch 中选小 batch；本项目可先测试温和软权重的迁移版，避免硬删真正新颖的兴趣。一个可审查的候选参数化为：在每个目标类别内按 r 排序，将分位数 q∈[0,1] 映射为 w_raw=0.5+q，再按类别平均权重归一化。初始范围和刷新频率必须在看结果前冻结；这属于我们提出的工程迁移，不是原论文算法或理论保证。

打分读取首个 SID token 的 NLL，避免 EOS 和模板容易程度主导“难度”；训练仍对与基线完全相同的每样本 SFT loss 乘权重，保留其 token reduction、原始词表归一化与输出模板。诊断量和训练 loss 的区别要明确记录。权重作为数据选择结果 stop-gradient；若让梯度反穿过学生打分，将成为另一个目标。

所有样本保留非零权重，记录每类权重总和、有效样本量、不同历史长度/parent 跨越/类内小群体的权重分布。类均衡不能解决全部偏差，但可排除“只是换了类别先验”的最明显混淆。

这比已有静态 Cross-Fitted 多了“当前学生学到了哪里”。已有方法倾向于给 OOF loss 较小的样本高权重；这里还要求学生对它有学习空间，权重随学生变化。两者必须直接对照，否则无法证明新增动态项有价值。

困难边界：参考模型太弱、存在系统性错误，或稀有真实迁移在参考数据中缺失时，a_i 会误把值得学的内容判得很难。需要用人工抽查和类内难例切片验证，不能称 a_i 为真实噪声概率。已经训练过全部样本的 Mixed 学生可能已经记住噪声；用它续训只测后期纠偏，不能替代从未见这些标签的起点做数据方法比较。

## 4. 第二条备选：用独立 NI 用户衡量样本效用

Ren et al. 的关键问题是：训练样本 i 的更新，能否降低独立参考用户的损失？

\[
\theta_i'=\theta-\eta\nabla_\theta\ell_i(\theta),\qquad
u_i=L_{meta}(\theta)-L_{meta}(\theta_i').
\]

在一步 SGD 的一阶近似下：

\[
u_i\approx\eta\langle\nabla L_{meta}(\theta),\nabla\ell_i(\theta)\rangle.
\]

同为高 loss 的两个样本可以有相反梯度效用：一个帮助独立用户，一个让他们更差。因此这种方法比“loss 越高权重越低”更直接地区分有用难例。实际 Adam 更新有动量与预条件，裸梯度内积只是近似；不能宣称它精确等于多步训练收益。

L_meta 建议按 NI 目标类别平均，使用首个 SID 的类别预测损失作为 macro 排名的可微代理；代理 loss 的改善不保证 Recall/NDCG 改善。部署不增加任何模块，新增部分只服务训练期权重。若只在 SID 输出行或部分 LoRA 参数上计算梯度，必须称为受限梯度近似，并抽样验证它与实际一步更新效用是否一致。

关键前提是参考数据：此前 5,354 名用户独立、类别较均衡，只能证明划分性质，不能证明点击标签干净。需要另建或重新固定 Train 内 meta-fit / model-selection 两份互斥用户集，且核对父 checkpoint 从未见过其标签。meta-fit 一旦参与权重梯度就已经属于训练资源，不能再当独立验证；官方 Dev 不参与权重更新或超参搜索，而且已经用于多轮观察，只能作为历史一致口径的比较集，最终确认最好使用新的未触碰用户/时间留出。

如人工审计仅能确认未来新闻与类别一致，便只能保证这层 label fidelity；不能保证该兴趣可由历史识别，也不能把只保留“看上去容易解释”的样本当作无偏 meta 集。无可信额外标签时，可以探索代表性的弱标注参考集，但必须如实命名；元学习不会自动把它变干净。

## 5. 与仓库已有工作的关系

- `src/new_interest/experiments/build_crossfit_weights.py` 和 `semantic_id_crossfit.py` 已有静态 OOF 难度重加权；RHO 动态差值应直接与它们对照。
- 已有相关 OOF 使用 9B 教师、108 类归一化及 fold-prior-adjusted NLL。不能直接与当前 2B 的全词表 completion loss 相减。应重新生成兼容的首 SID token 分数，或将旧分数仅作为独立 static baseline。跨模型教师本身是 RHO 原文允许的近似，但必须声明容量差异并校验该参考的用途。
- 2026-09-13 研究笔记已经讨论过训练梯度效用与 hard-example protection，不能将这些概念重新包装成首次提出。
- 当前 OI→NI Meta 是用旧兴趣虚拟更新后改善新兴趣的元目标，不是逐样本权重学习。最新 full-vocabulary replacement 与此前 SID-only 协议须分开；它仍在执行，不能因为 SID 关系偏置实验失败就判定 Meta 失败。本调研不修改这项实验。
- 先前一用户一 target 和“History-based scorer 能否直接当预测器”的讨论仍适用。只读 History/候选类的评分器，应同时比较其直接预测；只有目标新闻内容等训练期额外证据，才是不同的信息来源。训练 utility 本身也不是标签真值。
- 本地构造器保留 `target_news`，选一个 target 并不一定只有一篇支持新闻；同 impression 的多篇点击可以仅用于原标签审计，保持单 target 监督。类别来自重分类输入字段，当前消费代码不提供原始重分类置信度。未来新闻内容在训练期用于审计时，不能拼回部署模型的 History 输入。

## 6. 最小实验顺序和退出条件

1. **只读审计。** 固定分层抽样，交叉查看标签映射可靠性、历史可观测性、OOF/当前难度与已有训练动态。人工输出“确定映射错误 / 信息不足 / 有效困难 / 不确定”，不强迫为每条点击编桥接解释。当前无噪声比例结果。
2. **先做一个权重变化。** 同起点、同训练用户、同 loss、同更新预算比较 uniform、已有 static OOF、dynamic reducible loss；加一个类别内随机打乱权重的对照，检验是否仅为随机正则化。若首次成本要压缩，可先用前三组，打乱对照在出现收益后做。
3. **如走元重加权，则单独比较。** 共享同一可用数据预算，基线也获得同等数量监督或设置 meta-data-only 对照，防止额外参考数据本身解释收益。记录教师/OOF、梯度特征和元更新的总 GPU 成本，不只对齐学生 optimizer 步数。
4. **评价保持全部真实样本。** 不从 Dev 删除难例，不只报告筛选子集；报告 Macro Recall/NDCG、类内/跨 parent 切片、难例保留、校准与按用户的配对区间。单 seed 样本 bootstrap 不代表跨 seed 稳定性。
5. **可证伪条件。** 若 dynamic 不优于 static OOF，则动态学习空间没有增量证据；若 meta 不优于数据预算一致的基线，则独立效用信号没有增量证据；若提升只来自压低跨域/稀有真实类别，则不满足本任务目的。

可写的研究问题是：“在弱标注的新兴趣预测中，如何保留仍有泛化价值的困难样本，并减少低效或错误监督的影响？”它需要噪声审计、强数据对照和真实泛化改善支撑。现有论文已经覆盖大多数通用技巧；简单移植本身不是首创，也不能提前承诺会有效。

## 来源

[1] Swayamdipta et al. *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.746/ （出版页摘要与机制。）

[2] Mindermann et al. *Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt*. ICML 2022. https://proceedings.mlr.press/v162/mindermann22a.html （出版页与全文 §3、Algorithm 1、Appendix A；小模型/holdout 近似的条件与局限。）

[3] Ren et al. *Learning to Reweight Examples for Robust Deep Learning*. ICML 2018. https://proceedings.mlr.press/v80/ren18a.html （正式论文与在线元重加权算法。）

[4] Shu et al. *Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting*. NeurIPS 2019. https://arxiv.org/abs/1902.07379 （正文元权重映射与干净 meta 数据要求。）

[5] Wang et al. *Denoising Implicit Feedback for Recommendation*. WSDM 2021. https://arxiv.org/abs/2006.04153 （正文 ADT，dwell/rating 的实验性 false-positive 定义与过度去噪局限。）

[6] Wang et al. *Efficient Bi-Level Optimization for Recommendation Denoising*. KDD 2023. https://arxiv.org/abs/2210.10321 （§4.2–4.3，BPR/AU 梯度匹配。）

[7] Song et al. *Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation*. 2024 preprint. https://arxiv.org/abs/2409.10343 （高 loss 难例识别与 denoising training；未据此确认后续正式 venue。）

[8] Xia et al. *LESS: Selecting Influential Data for Targeted Instruction Tuning*. ICML 2024. https://arxiv.org/abs/2402.04333 （正文 LoRA 梯度数据存储、Adam 相关相似度与目标数据选择。）

[9] Northcutt et al. *Confident Learning: Estimating Uncertainty in Dataset Labels*. JAIR 2021. https://arxiv.org/abs/1911.00068 （类别条件噪声与标签质量估计；本次未据摘要外推 MIND 去噪效果。）

[10] Ethayarajh et al. *Understanding Dataset Difficulty with V-Usable Information*. ICML 2022. https://proceedings.mlr.press/v162/ethayarajh22a.html （模型相关数据难度与 PVI 定义用途。）

[11] Gao et al. *Self-Guided Learning to Denoise for Robust Recommendation*. SIGIR 2022. https://arxiv.org/abs/2204.06832 （全文 §3.1–3.3.2，Eq.16 梯度内积；早期记忆 reference、自生成噪声和元权重。）

若缺少外部可靠参考集，可继续读 Zhang and Pfister, *Learning Fast Sample Re-Weighting Without Reward Data*, ICCV 2021：https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Learning_Fast_Sample_Re-Weighting_Without_Reward_Data_ICCV_2021_paper.html 。FSR 以历史 loss 降幅的 EMA 构建类均衡代理 reference，并近似只在部分层做元更新；它同样把可靠性前提移到自生成数据与早期学习假设上，不能绕过这个问题。
