# 冻结训练样本：提示词替换前后

来源sample_id：`cir:train:1597274:19`；source_sample_id：`train:1597274:19`。这是已有CIR历史重建样本的监督目标，不是模型生成结果，也不是新增真实未来点击标签。保留原始重复行为，不作去重。

## 原CIR提示词（逐字原文）

```text
The following news browsing history has all entries from one topic category removed. The remaining entries are in chronological order, formatted as "[Topic category] Headline":

1. [Comics & Graphic Narratives] The News In Cartoons
2. [Comics & Graphic Narratives] The News In Cartoons
3. [Trivia & Puzzles] The News In Cartoons
4. [Trivia & Puzzles] The News In Cartoons
5. [Trivia & Puzzles] The News In Cartoons

Infer the removed topic category from the remaining observed interests and news items. This is historical interest reconstruction, not a prediction of the next click. Respond with only its single-token Interest ID in the exact format <interest_NNN>. Do not output the category name or any explanation.
```

中文翻译：

```text
以下新闻浏览历史已经删除了某一个主题类别的所有条目。其余条目按时间顺序排列，格式为“[主题类别] 标题”：

1. [漫画与图像叙事] 漫画中的新闻
2. [漫画与图像叙事] 漫画中的新闻
3. [趣味知识与谜题] 漫画中的新闻
4. [趣味知识与谜题] 漫画中的新闻
5. [趣味知识与谜题] 漫画中的新闻

根据其余已观察到的兴趣和新闻条目，推断被删除的主题类别。这是历史兴趣重建，不是预测下一次点击。只回答该类别的单token兴趣ID，严格使用 <interest_NNN> 格式。不要输出类别名称或任何解释。
```

## 新NI提示词（逐字原文）

```text
The headlines and corresponding topic categories of the news items I recently viewed, in chronological order, are listed below. Each item is formatted as "[Topic category] Headline":

1. [Comics & Graphic Narratives] The News In Cartoons
2. [Comics & Graphic Narratives] The News In Cartoons
3. [Trivia & Puzzles] The News In Cartoons
4. [Trivia & Puzzles] The News In Cartoons
5. [Trivia & Puzzles] The News In Cartoons

Based on both the headlines and their topic categories, predict a new topic category that I am likely to be interested in next. The predicted category should not appear among the topic categories in my viewing history. Respond with only the learned interest token assigned to the predicted category, in the form <interest_NNN>.
```

中文翻译：

```text
以下按时间顺序列出了我最近浏览的新闻标题及其对应的主题类别。每个条目的格式为“[主题类别] 标题”：

1. [漫画与图像叙事] 漫画中的新闻
2. [漫画与图像叙事] 漫画中的新闻
3. [趣味知识与谜题] 漫画中的新闻
4. [趣味知识与谜题] 漫画中的新闻
5. [趣味知识与谜题] 漫画中的新闻

结合标题和主题类别，预测一个我接下来可能感兴趣的新主题类别。预测的类别不应出现在我的浏览历史主题类别中。只回答分配给预测类别的已学习兴趣token，格式为 <interest_NNN>。
```

两者监督响应均为 `<interest_000>`，对应 `Accidents & Emergency Response`（事故与应急响应）。除指令及其messages副本外，该样本其余字段完全一致。
