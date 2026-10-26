# MIND：Mixed 2B 短思考 SFT

## 1. 研究问题

在已验收的 Qwen3.5-2B Mixed 模型上增加少量 GPT-5.5 短思考 SFT，能否学会自然收尾，并保持新兴趣推荐的四项 Macro 指标？本轮仅实施 SFT，比较原 Mixed 与 Mixed + Short-CoT SFT。

## 2. 训练数据构造

复用既有 GPT-5.5 教师数据：从相同 MIND 新兴趣 Train 的 108 类中各取 10 条，共 1,080 条，样本身份、标签、历史内容和 SID 映射均与 Mixed 的 NI 部分核对一致。教师先在不知未来标签的情况下写历史摘要和偏好，再参考训练标签写兴趣扩展解释；学生只看历史。此类目标条件化解释不是教师盲预测，也不能因未出现字面类别名而称为语义上不含目标信息。

| 项目 | 数量或设置 |
|---|---|
| 新增 SFT 数据 | 1,080 条；108 类 × 10 |
| 训练轮数/更新 | 2 epochs / 68 updates |
| 配置样本曝光 | 2,160 次 |
| 教师思考长度 | 中位数 111；P95 122；最大 133 tokens |
| 完整训练序列 | 最大 1,704 tokens；无截断 |
| 教师新增调用 | 0，已生成数据复核后复用 |

### 真实训练样例

来源 source_sample_id：`train:682585:81`；目标 American Football。

原始学生提示词：
```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Celebrity & Creator Culture] New Couple Alert? Drake and Kylie Jenner Appear Into Each Other at Party

Based on both the headlines and their topic categories, reason about a new topic category that I am likely to be interested in next. Inside <think>, briefly summarize the history pattern, infer the underlying preference state, and describe a plausible unseen interest expansion without naming its exact taxonomy category or any learned interest token. Then close </think> and output exactly one learned interest token inside <answer> and </answer>.
```
中文翻译：
```text
下面按时间顺序列出我近期浏览新闻的标题和对应主题类别，每条格式为“[主题类别] 标题”。

1. [名人与创作者文化] 新情侣出现？Drake 与 Kylie Jenner 在派对上似乎对彼此有意。

结合标题与主题类别，推理我下一步可能感兴趣的新主题。在 <think> 中简要概括历史模式、推断潜在偏好，并描述合理的未见兴趣扩展，不写出其准确分类名或任何学习得到的兴趣 token。随后关闭 </think>，并在 <answer> 与 </answer> 中只输出一个学习得到的兴趣 token。
```
教师目标输出：
```text
<think>
History summary: Viewed a single celebrity gossip item about a rumored Drake and Kylie Jenner party interaction.
Preference state: History is too sparse for a strong inference, though it suggests possible interest in celebrity relationships or pop-culture rumors.
Potential expansion: Because the lone signal is celebrity-gossip oriented, any move toward sports would be weakly supported, though pop-culture coverage can overlap with famous athletes, entertainment events, and celebrity social circles.
</think>
<answer><interest_005></answer>
```
中文翻译：
```text
历史摘要：只浏览了一条关于 Drake 与 Kylie Jenner 派对互动传闻的明星八卦新闻。
偏好状态：历史过于稀疏，不足以形成强判断，但可能反映对明星恋情或流行文化传闻的兴趣。
潜在扩展：由于唯一信号来自明星八卦，任何向体育的转移都只有弱支持，不过流行文化报道可能与知名运动员、娱乐活动及明星社交圈交叉。
最终答案：<interest_005>（American Football，美国橄榄球）。
```

## 3. 测试设计

### 3.1 模型与训练位置

Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。正式训练从原 Mixed 最终适配器接续，完整保留 373 个 LoRA/SID 张量，在新 SFT 阶段重置优化器。未使用 smoke 训练后的权重。Recall Premium 四张 A100-80GB，BF16，非重入 gradient checkpointing，FLA；每卡 batch 8、累积 1、全局 batch 32。LR=1e-5，cosine，warmup=0.03，seed=42；LoRA r=16、alpha=32、dropout=0.05，full-vocabulary token-mean SFT。

正式训练已完成 68 步，loss 从首步 3.0886 降至末步 1.9705。保存后重载的 373 个张量和参考 logits 通过零容差检查。原生作业 Completed，UTC 2026-09-13 17:16:02 结束，四张卡已释放。

### 3.2 数据与解码口径

使用与 Mixed 完全同序、同目标的 31,392 条 grouped Dev，104 个有支持的类别、37,728 个目标出现。这里是 Dev，不是额外 held-out Test。每条从历史构建已训练的 thinking 提示词，greedy 自由生成至自然 EOS，最多 512 个新增 token。512 是总输出安全预算；未设置 128-token 思考硬上限、长度惩罚或手工补结束标签。

### 3.3 解析、排名与指标

无外部 LLM judge。原生模板提供 `<think>\n`，生成内容必须具有非空思考、`</think>`、`<answer><interest_NNN></answer>` 和自然终止；思考中出现 SID 或整体格式不合法均记录为无效。所有无效/触顶输出保留在全量分母中，排名为空。技术异常不得转换成错误答案。

对合法输出，仅使用生成 SID 之前的原始 token 前缀，重新做一次前向，并对全部 108 个合法 SID 的 logits 排序取 Top-20；生成的 SID 本身不进入评分上下文。与原基线一样，不事后过滤历史已见类别。Macro Recall 按类别求召回后平均；Macro NDCG 对每个目标用 1/log2(rank+1) 折扣，按类别支持数归一后在 104 类上平均。

缓存 greedy SID 与规范化的新前向 Top-1 有 1,847/30,176 条不一致，两种观测分别保存，本表预先约定使用后者；差异的具体原因尚未独立定位。

### 3.4 真实 Dev 样例

以下各给出一个合法且 Top-10 命中的样例和一个合法但 Top-10 未命中的样例，按相应组内最短提示词选取，不代表全体解释质量。输出原文不重复展示模板已提供的 `<think>` 前缀。

#### record_index=19929：valid_hit

实际提示词：
```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Celebrity & Creator Culture] Celebs celebrate Halloween 2019

Based on both the headlines and their topic categories, reason about a new topic category that I am likely to be interested in next. Inside <think>, briefly summarize the history pattern, infer the underlying preference state, and describe a plausible unseen interest expansion without naming its exact taxonomy category or any learned interest token. Then close </think> and output exactly one learned interest token inside <answer> and </answer>.
```
中文翻译：
```text
下面按时间顺序列出我近期浏览新闻的标题和对应主题类别，每条格式为“[主题类别] 标题”。

1. [名人与创作者文化] 明星们庆祝 2019 年万圣节。

结合标题与主题类别，推理我下一步可能感兴趣的新主题。在 <think> 中简要概括历史模式、推断潜在偏好，并描述合理的未见兴趣扩展，不写出其准确分类名或任何学习得到的兴趣 token。随后关闭 </think>，并在 <answer> 与 </answer> 中只输出一个学习得到的兴趣 token。
```
模型生成原文：
```text
History summary: Recent views show a mix of celebrity culture, entertainment, and occasional lifestyle topics, with Halloween 2019 as a notable recent item.
Preference state: The user appears to favor celebrity-driven entertainment and lifestyle content, with occasional broader cultural interest.
Potential expansion: Given the entertainment focus, there is a plausible interest in celebrity-driven lifestyle or fashion content, though this is speculative.
</think>

<answer><interest_044></answer>
```
中文翻译：
```text
历史摘要：近期浏览涵盖明星文化、娱乐以及偶尔的生活方式主题，2019 年万圣节是其中一条值得注意的近期内容。
偏好状态：用户似乎偏好明星驱动的娱乐和生活方式内容，偶尔也关注更广泛的文化。
潜在扩展：鉴于娱乐倾向，可能对明星驱动的生活方式或时尚内容感兴趣，但这属于推测。
最终答案：<interest_044>（Fashion & Personal Style，时尚与个人风格）。
```

真实目标包括警务与矫正、时尚与个人风格、美国橄榄球、高尔夫和宠物。Top-1 为时尚与个人风格，Top-10 还包含美国橄榄球，因此该条 Recall@10=2/5。该例实际只有一条历史，模型却概括为多类浏览及“偶尔”关注，超出了给定证据；排名命中不等于解释可靠。

#### record_index=30123：valid_miss

实际提示词：
```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Film] The most powerful LGBTQ movies

Based on both the headlines and their topic categories, reason about a new topic category that I am likely to be interested in next. Inside <think>, briefly summarize the history pattern, infer the underlying preference state, and describe a plausible unseen interest expansion without naming its exact taxonomy category or any learned interest token. Then close </think> and output exactly one learned interest token inside <answer> and </answer>.
```
中文翻译：
```text
下面按时间顺序列出我近期浏览新闻的标题和对应主题类别，每条格式为“[主题类别] 标题”。

1. [电影] 最有力量的 LGBTQ 电影。

结合标题与主题类别，推理我下一步可能感兴趣的新主题。在 <think> 中简要概括历史模式、推断潜在偏好，并描述合理的未见兴趣扩展，不写出其准确分类名或任何学习得到的兴趣 token。随后关闭 </think>，并在 <answer> 与 </answer> 中只输出一个学习得到的兴趣 token。
```
模型生成原文：
```text
History summary: Recent views are mostly film-related, with one recent item on LGBTQ movies.
Preference state: Likely interested in entertainment, specifically film, and possibly LGBTQ-themed content.
Potential expansion: Could plausibly extend to LGBTQ-focused entertainment, though the single recent item is weak evidence.
</think>

<answer><interest_073></answer>
```
中文翻译：
```text
历史摘要：近期浏览大多与电影有关，其中一条近期内容是 LGBTQ 电影。
偏好状态：可能关注娱乐，尤其是电影，也可能关注 LGBTQ 题材内容。
潜在扩展：可能拓展至以 LGBTQ 为主题的娱乐，不过单条近期内容提供的证据很弱。
最终答案：<interest_073>（Music，音乐）。
```

真实目标为 Celebrity & Creator Culture（名人与创作者文化），在重算排名中为第 18 名，因此 Recall@10=0、Recall@20=1。模型生成 Music，格式合法但不能替代正确的推荐目标。

## 4. 实验结果

| Method | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Mixed SFT | 20.61% | 34.20% | 10.42% | 13.82% |
| Mixed + Short-CoT SFT | 14.00% | 24.58% | 6.74% | 9.41% |

| 相比 Mixed 的变化 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Δ（百分点） | -6.60 | -9.63 | -3.68 | -4.41 |

| 输出行为 | 完整 Dev 结果 |
|---|---:|
| 格式合法 | 30,176/31,392（96.13%） |
| 自然结束 | 31,090/31,392（99.04%） |
| 非法格式、但自然结束 | 914/31,392（2.91%） |
| 达到总生成上限 | 302/31,392（0.96%） |
| 思考中出现 SID | 905/31,392（2.88%） |
| 总输出 token 数 | 平均 111.01；中位数 107；P95 128；最大 512 |
| 可解析思考段 token 数 | 30,767 条；平均 96.92；中位数 97；P95 116 |

总输出长度包含回答和结束标记；可解析思考段统计不含无法解析的记录，二者不是同一分母。P95=128 是实测分位数，不是施加的硬上限。8-step DDP smoke 的 0/64 格式合法与 63/64 触顶结果完整保留，不能与正式 68-step 结果混为一谈。

仅取短思考输出合法的同一批 30,176 条记录，双方仍保留 104 个支持类别，得到以下诊断对照；不替代全量主结果：

| Method（同一合法输出子集） | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |
|---|---:|---:|---:|---:|
| Mixed SFT | 20.61% | 34.23% | 10.44% | 13.84% |
| Mixed + Short-CoT SFT | 14.54% | 25.44% | 7.00% | 9.75% |

因此推荐指标回退不只是无效格式计零造成的。另一个可见变化是更常把历史已见类别排在前面。在同一 30,176 条合法输出记录上：

| Method | Top-1 为历史已见类别 | Top-10 中已见类别平均数量 |
|---|---:|---:|
| Mixed SFT | 37/30,176（0.12%） | 0.060 |
| Mixed + Short-CoT SFT | 3,522/30,176（11.67%） | 1.097 |

本配置学会了更短、更规范的输出，但四项推荐指标均下降。该实验同时增加了短解释 SFT、修改了提示词和推理路径，不能把差异单独归因于 think 开关；同样不能仅凭格式改善宣称推理更有效。这是一次固定 seed 的观测，未进行多 seed 显著性检验。

## 5. 数据与产物路径

- [artifacts/mind-short-think-sft-20260913/dataset/train.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/dataset/train.parquet)
- [artifacts/mind-short-think-sft-20260913/audit.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/audit.json)
- [artifacts/mind-short-think-sft-20260913/bundle-full-v1/contract.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/bundle-full-v1/contract.json)
- [artifacts/mind-short-think-sft-20260913/bundle-full-v1/source_manifest.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/bundle-full-v1/source_manifest.json)
- [artifacts/mind-short-think-sft-20260913/acceptance/smoke-v2.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/acceptance/smoke-v2.json)
- [artifacts/mind-short-think-sft-20260913/acceptance/full-v1.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/acceptance/full-v1.json)
- [artifacts/mind-short-think-sft-20260913/final/independent_verification.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/final/independent_verification.json)
- [artifacts/mind-short-think-sft-20260913/final/history_overlap_diagnostic.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/final/history_overlap_diagnostic.json)
- [artifacts/mind-short-think-sft-20260913/final/examples.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/final/examples.json)
- [artifacts/mind-short-think-sft-20260913/readback-full-v1/sft/final_adapter](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/readback-full-v1/sft/final_adapter)
- [artifacts/mind-short-think-sft-20260913/readback-full-v1/real_eval/merged/id_token_predictions.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/readback-full-v1/real_eval/merged/id_token_predictions.parquet)
- [artifacts/mind-short-think-sft-20260913/readback-full-v1/real_eval/merged/metrics.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/readback-full-v1/real_eval/merged/metrics.json)
- [artifacts/mind-short-think-sft-20260913/readback-full-v1/artifact_manifest.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/readback-full-v1/artifact_manifest.json)
- [artifacts/mind-short-think-sft-20260913/download-full-v1/receipt.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/download-full-v1/receipt.json)
- [artifacts/mind-short-think-sft-20260913/progress/20260913T1729Z/receipt.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/progress/20260913T1729Z/receipt.json)
- [scripts/analysis/verify_short_think_full.py](/Users/river/Documents/MSRA/Rec/CrossInterest/scripts/analysis/verify_short_think_full.py)

- [artifacts/preference-state-teacher/gpt55-behavior-interest-21600/full-v1/teacher_records.parquet](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/preference-state-teacher/gpt55-behavior-interest-21600/full-v1/teacher_records.parquet)
- [artifacts/preference-state-teacher/gpt55-behavior-interest-21600/full-v1/metadata.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/preference-state-teacher/gpt55-behavior-interest-21600/full-v1/metadata.json)
- [manifests/mind_short_think_teacher_reuse_20260913_v1.json](/Users/river/Documents/MSRA/Rec/CrossInterest/manifests/mind_short_think_teacher_reuse_20260913_v1.json)
- [artifacts/mind-four-experiments-20260911/mixed_full_coordinator_acceptance.json](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-four-experiments-20260911/mixed_full_coordinator_acceptance.json)
- [artifacts/mind-short-think-sft-20260913/bundle-full-v1/inputs/mixed_adapter](/Users/river/Documents/MSRA/Rec/CrossInterest/artifacts/mind-short-think-sft-20260913/bundle-full-v1/inputs/mixed_adapter)
- [configs/amlt/mind_short_think_qwen35_2b_full_0913_v1.yaml](../../../configs/amlt/archive/2026-10-26/mind_short_think_qwen35_2b_full_0913_v1.yaml)

独立冻结 reader 核验 2,019 个文件、149,088,436 bytes；独立脚本再次核对全量样本顺序、目标、token 边界并重算四项 Macro。

AMLT experiment：`mind-short-think-qwen35-2b-full-v1-0913`；job：`qwen35-2b-mixed-short-think-full`；backend：`tough_dolphin_6jspbn82c9`。
