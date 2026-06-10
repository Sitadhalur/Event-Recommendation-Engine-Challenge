# Event Recommendation Engine Challenge 项目报告

## 1. 项目概述

本项目基于 Kaggle 经典竞赛 [Event Recommendation Engine Challenge](https://www.kaggle.com/competitions/event-recommendation-engine-challenge/overview)。任务目标是：根据用户信息、事件信息、用户社交关系、事件参与者信息和历史行为，预测用户会对哪些 event 感兴趣，并对每个用户的候选 event 进行排序。

该任务不是开放式召回问题。测试集中已经给定每个用户的一组候选 event，模型需要对这些候选 event 排序。因此，本项目更接近 **user-event pair ranking**，而不是从全量 event 池中检索推荐。

由于该 Kaggle 比赛已经关闭，无法进行官方 late submission。根据课程说明，本项目采用 **Case B：本地统一 5-fold cross-validation** 作为最终评估方式。

最终主线结果为：

```text
Strict 5-fold OOF MAP@200 = 0.718516
```

最终输出文件：

- `submission_oof_ultrastrict_5fold_hgb_xgb_cat.csv`
- `submission_oof_ultrastrict_5fold_hgb_xgb_cat_legacy.csv`

## 2. 任务与评价指标

### 2.1 任务形式

每条样本是一个 `(user, event)` 候选对。训练集中提供标签：

- `interested = 1`：用户对该 event 感兴趣。
- `not_interested = 1`：用户明确不感兴趣。
- 二者都为 0：用户没有明确正负反馈。

模型需要给每个候选 `(user, event)` 输出一个分数，然后对每个用户的候选 events 按分数降序排序。

### 2.2 MAP@200

比赛指标为 **MAP@200**。对每个用户，系统只关心预测列表前 200 个 event 中正样本出现的位置。正样本越靠前，平均准确率越高。最后对所有用户取平均。

因此，本项目优化重点不是单纯分类准确率，而是：

- 能否把真正感兴趣的 event 排在同一用户候选列表的前面；
- 能否在类别不平衡情况下仍然保持较好的排序质量；
- 能否在 cold-start 用户场景下利用社交、地理、事件元数据等信息。

## 3. 数据集分析

### 3.1 文件说明

本地工作目录包含以下主要文件：

| 文件 | 作用 |
|---|---|
| `train.csv` | 训练集 user-event 交互，包含标签 |
| `test.csv` | 测试集候选 user-event 对，不含标签 |
| `users.csv` | 用户属性，例如 locale、birthyear、gender、location、timezone |
| `events.csv.gz` | 事件元数据，包括 creator、时间、地点、经纬度、文本主题计数等 |
| `event_attendees.csv.gz` | 每个 event 的 yes/maybe/invited/no 用户列表 |
| `user_friends.csv.gz` | 用户好友关系 |
| `public_leaderboard_solution.csv` | public leaderboard 答案文件，最终主线没有使用 |

### 3.2 数据规模

训练集：

```text
train.csv: 15,398 rows
users in train: 2,034
events in train: 8,846
```

测试集：

```text
test.csv: 10,237 rows
users in test: 1,357
events in test: 6,173
```

全量候选范围：

```text
train + test users: 3,391
train + test events: 13,418
train + test user-event pairs: 25,326
```

### 3.3 标签分布

训练标签分布如下：

```text
interested = 1:      4,131
not_interested = 1:    514
neither:            10,753
total:              15,398
```

可以看到，正样本比例不高，明确负反馈更少，大量样本是无明确反馈。这会带来两个问题：

1. 如果直接训练普通分类器，模型容易偏向多数类。
2. `not_interested` 虽然数量少，但信息量很大，因为它代表用户明确拒绝某类 event。

### 3.4 Cold-start 特征

本项目最重要的数据特点是：

```text
train users 和 test users 没有重合
```

这意味着用户历史行为在测试用户上不能直接使用。因此，模型不能依赖“这个用户以前喜欢什么”作为主要信号，而必须更多依赖：

- event 自身热度；
- event 时间、地点、文本内容；
- 用户静态属性；
- 好友关系；
- 好友是否参加 event；
- 用户 location 和 event location 的匹配程度；
- event creator 是否与用户有关。

这也是纯协同过滤方法在该任务上不适合的原因之一。网上参赛复盘也提到，该比赛更适合基于候选 pair 的排序/分类建模，而不是传统 collaborative filtering。

## 4. 泄露风险与合规策略

本项目中最容易出问题的地方不是模型，而是数据泄露。为保证符合课程要求，最终主线采用了严格防泄露策略。

### 4.1 未使用 public leaderboard answer

工作目录中存在 `public_leaderboard_solution.csv`。它可以用于了解比赛格式，但不能用于最终建模、调参或报告主结果。

最终主脚本：

```text
train_only_oof_blend.py
```

不读取 `public_leaderboard_solution.csv`。

一些早期探索脚本曾经用于 public subset 分析，例如：

- `blend_submissions.py`
- `refine_blend.py`
- `score_rule_submission.py`
- `event_reco_public_augmented.py`

这些脚本不作为最终交付主线，也不用于最终分数。

### 4.2 OOF 阶段不使用验证折标签

最终主线采用 grouped 5-fold CV。每一折中：

1. 将训练用户分为训练折和验证折；
2. 只使用训练折 `fit_rows` 构造标签衍生特征；
3. 使用同一个 fold-specific feature builder 变换训练折和验证折；
4. 验证折标签只用于计算 MAP@200，不参与特征构造。

相关函数：

```python
history_builder_from_rows(full_builder, rows)
```

该函数只接收当前折的训练行 `fit_rows`，并基于这些行构造：

- 用户正反馈内容画像；
- 用户负反馈内容画像；
- 好友正反馈内容画像；
- 好友负反馈内容画像；
- 好友 event-level 图信号。

### 4.3 OOF 阶段不使用 test metadata

早期版本中，`build_everything()` 会同时加载 train 和 test 的候选 users/events，用于构造部分全局统计。这不涉及 test 标签，但会让 CV 分数受到 test 输入分布影响。为了更严格，最终版本新增：

```python
build_everything(include_test_metadata=False)
```

OOF 阶段只使用训练集 metadata：

```text
Candidate users = 2,034
Candidate events = 8,846
Candidate pairs = 15,220
```

最终训练和生成提交文件时才使用 train + test metadata：

```python
build_everything(include_test_metadata=True)
```

这是合理的，因为真实提交时测试候选集本来就是已知输入。

### 4.4 默认剔除 event_label_ 特征

项目中曾尝试构造 event 历史标签统计，例如某 event 在训练集中被多少用户标记为 interested。虽然可以通过 fold-specific 构造和 leave-one-out 避免直接泄露，但这类特征容易引起解释争议，并且正式 5-fold 中表现不稳定。

最终主脚本默认剔除：

```python
--drop-prefixes event_label_
```

### 4.5 检查 event_attendees 是否直接包含答案

`event_attendees.csv.gz` 中包含每个 event 的 yes/maybe/invited/no 用户列表。理论上，如果目标用户本人直接出现在对应 event 的 yes/no 列表中，那会构成直接答案泄露。

实际检查结果：

```text
训练集中 15,398 个目标 user-event 对中，
目标用户本人出现在对应 event yes/maybe/invited/no 列表中的次数为 0。
```

因此，当前使用 event_attendees 的方式不是直接读取答案，而是利用：

- event 总体热度；
- 朋友是否在 event 的 yes/maybe/invited/no 列表中；
- 朋友响应比例。

### 4.6 Kaggle 官方对 test timestamp 泄露的处理

Kaggle 竞赛讨论区曾由 competition host 发布更新：`test.csv` 中部分用户存在 timestamp 采样泄露。当某个用户的大部分候选 event 共享同一个 timestamp，而另一个单独 event 的 timestamp 略晚时，这个单独 event 有较高概率被标记为 interested。为避免该泄露影响排行榜，官方决定在计分时忽略可能受影响的 test cases。

具体规则是：只对“该用户所有候选 events 的 timestamp 完全相同”的 test 用户计分；其他用户仍保留在提交格式中，但不计入 leaderboard 分数。

本地复核结果与官方说明一致：

```text
test users: 1,357
scored users: 787
ignored users: 570
scored rows: 4,614
ignored rows: 5,623
```

该信息对本项目的影响如下：

1. 当前提交文件仍然必须包含全部 1,357 个 test 用户，因为官方没有改变提交格式。
2. 若能够提交 Kaggle，只有 787 个 timestamp 完全一致的用户会影响分数。
3. 本项目没有利用“同一用户 timestamp 分组中单独较晚 event 更可能 interested”这一泄露规则进行后处理。
4. 本地 5-fold CV 基于 `train.csv`，不等价于 Kaggle 官方过滤后的 test leaderboard，因此报告中不能把本地 CV 分数直接宣称为官方排行榜分数。
5. 该信息进一步说明：排行榜比较只能作为参考，最终课程交付应以严格本地 5-fold CV 为主。

## 5. 特征工程

最终模型主要使用以下类型的特征。

### 5.1 时间特征

- 通知时间到 event 开始时间的间隔；
- event 是否在 1 天、7 天、30 天内发生；
- event 是否发生在周末；
- event 月份；
- notification hour。

这些特征反映用户是否有足够时间参加 event，以及 event 时间本身是否更容易吸引用户。

### 5.2 地理特征

使用 `users.csv` 的用户 location 和 `events.csv.gz` 的 event city/country/lat/lng，构造：

- 用户 location 是否能解析出国家或城市；
- 用户国家是否匹配 event 国家；
- 用户城市是否匹配 event 城市；
- 用户与 event 的经纬度距离；
- 距离是否小于 50km、250km、1000km。

外部数据：

- `countryInfo.txt`
- `cities5000.zip`

这些来自 GeoNames，用于弱匹配用户 location 文本。

### 5.3 Event 热度和参与者特征

从 `event_attendees.csv.gz` 构造：

- event yes 数量；
- event maybe 数量；
- event invited 数量；
- event no 数量；
- event 总响应数；
- yes ratio；
- no ratio；
- event net positive。

这些反映 event 的整体受欢迎程度。

### 5.4 好友交集特征

利用 `user_friends.csv.gz` 和 `event_attendees.csv.gz`，对每个 `(user, event)` 计算：

- 多少好友 yes；
- 多少好友 maybe；
- 多少好友 invited；
- 多少好友 no；
- 好友响应比例；
- 好友正负净信号。

这类社交信号是该任务的重要特征，因为用户是否对 event 感兴趣往往和朋友参与情况相关。

### 5.5 Event 内容特征

`events.csv.gz` 中包含 `c_1` 到 `c_100` 的内容计数。基于这些计数构造：

- event word sum；
- event word peak ratio；
- active word count；
- word density。

此外，训练折内还构造了用户/好友的正负内容画像。但最终 cold-start 特征选择会剔除直接用户历史内容特征，保留更稳定的好友内容画像。

### 5.6 好友历史画像与二跳图特征

基于训练折中的标签，构造：

- 好友历史 interested event 的内容平均画像；
- 好友历史 not_interested event 的内容平均画像；
- 当前 event 与好友正反馈画像的 cosine similarity；
- 当前 event 与好友负反馈画像的 cosine similarity；
- 直接好友曾经喜欢过当前 event 的分数；
- 二跳好友曾经喜欢过当前 event 的弱分数。

二跳好友权重较低，避免噪声过大。

## 6. 模型方法

最终交付采用模型融合，而不是单一模型。

### 6.1 单模型

最终参与融合的模型包括：

1. `HistGradientBoostingClassifier`
2. `XGBClassifier`
3. `CatBoostClassifier`

这些模型适合表格特征，能够处理非线性关系、特征交互和不平衡数据。

### 6.2 融合方法

每个模型在 5-fold 中产生 OOF prediction。然后在 OOF prediction 上随机搜索融合权重，直接优化 MAP@200。

最终权重大约为：

```text
HGB:      0.5720
XGBoost: 0.1836
CatBoost:0.2444
```

说明 HGB 是主力模型，XGBoost 和 CatBoost 提供互补。

### 6.3 为什么没有把神经网络作为最终主模型

本课程是深度学习课程，因此项目中也考虑并尝试了神经网络方向，包括 MLP 和 graph-inspired 模型。但在该数据集上，神经网络没有成为最终最佳方法，原因包括：

1. 训练样本只有 15,398 行，相对较小；
2. test 用户与 train 用户完全不重合，用户 embedding 很难泛化；
3. 任务强依赖结构化表格特征、地理匹配和社交统计；
4. GBDT 类模型在中小规模表格数据上通常更稳；
5. 如果强行使用复杂神经网络，容易过拟合或引入验证泄露风险。

因此，最终主线采用表格模型融合，神经网络可作为报告中的对比实验或扩展方向，而不是最终最优模型。

## 7. 实验结果

### 7.1 主要实验结果

| 版本 | 说明 | 5-fold OOF MAP@200 |
|---|---|---:|
| 早期版本 | 全量 train 构造部分历史画像后再切折，存在偏乐观风险 | 0.730785 |
| 严格版本 | 每折重建历史画像，不使用 public answer | 0.706985 |
| 加 CatBoost | HGB + RF + LogReg + XGB + CatBoost | 0.708417 |
| 去弱模型 | HGB + XGB + CatBoost | 0.709794 |
| 最终超严格版本 | OOF 不使用 test metadata，剔除 event_label_ | **0.718516** |

最终版本分数反而高于上一严格版本，说明去掉 test metadata 并没有削弱模型，反而使训练范围内的全局统计更符合 OOF 验证场景。

### 7.2 最终复现命令

```powershell
& 'C:\Users\62571\anaconda3\envs\osm-history\python.exe' train_only_oof_blend.py --folds 5 --models hgb,xgb,cat --output submission_oof_ultrastrict_5fold_hgb_xgb_cat.csv --hgb-iter 320 --xgb-trees 380 --cat-iters 360 --trials 20000
```

### 7.3 输出文件

```text
submission_oof_ultrastrict_5fold_hgb_xgb_cat.csv
submission_oof_ultrastrict_5fold_hgb_xgb_cat_legacy.csv
```

## 8. 与排行榜和公开资料对比

由于 Kaggle 比赛已经关闭，当前无法得到官方 private leaderboard 实测分数。因此，以下对比只能作为参考，不能视为严格同口径比较。

公开资料中可参考的信息包括：

- Kaggle 比赛页：<https://www.kaggle.com/competitions/event-recommendation-engine-challenge/overview>
- Random Bits on Data 参赛复盘：<https://datathinking.wordpress.com/2013/02/10/event-recommendation-engine-challenge-kaggle/>
- Andrei Olariu 第 7 名复盘：<https://webmining.olariu.org/event-recommendation-contest-on-kaggle/>
- 后续扩展摘要报告：<https://fenix.tecnico.ulisboa.pt/downloadFile/281870113703623/ExtendedAbstract.pdf>

部分公开资料提到 Kaggle public leaderboard 最高约：

```text
MAP@200 ≈ 0.72876
```

本文最终本地严格 5-fold CV：

```text
MAP@200 = 0.718516
```

如果粗略比较，差距约为：

```text
0.72876 - 0.718516 = 0.010244
```

但必须强调：

1. Kaggle public leaderboard 和本地 5-fold CV 不是同一评估集；
2. public leaderboard 只对应公开测试子集，不代表 private leaderboard；
3. 官方曾因 test timestamp 泄露忽略 570 个 test 用户，因此排行榜实际计分用户少于提交文件中的全部用户；
4. 本项目无法提交 Kaggle，因此不能声称达到官方 Top 5；
5. 本项目可以客观表述为：在课程允许的严格本地 5-fold CV 口径下，取得 MAP@200 = 0.718516，成绩接近公开资料中较强方案的量级，但仍不能等同于 Kaggle 官方排名。

## 9. 项目总结

本项目的关键不在于盲目使用复杂模型，而在于正确理解任务结构和防止数据泄露。

主要收获如下：

1. 该任务是 closed-list ranking，不是传统开放式推荐。
2. train/test 用户完全不重合，用户历史特征无法直接迁移，必须重视 cold-start。
3. 好友关系、event attendee、地理位置、event 时间和内容特征是主要有效信号。
4. 表格模型融合比当前尝试的神经网络更稳定。
5. 本项目中 public leaderboard answer、event attendees、event label history 都可能引发泄露风险，必须明确排除或严格按折构造。
6. 最终版本使用严格 grouped 5-fold OOF，每折只用训练折构造标签衍生特征，并且 OOF 阶段不使用 test metadata。

最终结论：

```text
本项目最终提交一个无 public answer、无验证标签泄露、可复现的严格 5-fold CV 推荐排序系统。
最终 OOF MAP@200 = 0.718516。
```

## 10. AI 工具使用说明

本项目开发过程中使用 AI 工具辅助：

- 理解 Kaggle 任务背景和课程要求；
- 搜索公开复盘资料；
- 检查数据泄露风险；
- 辅助编写和调试 Python 脚本；
- 协助整理报告结构和语言。

但最终模型训练、数据处理、实验结果和代码均在本地工作目录中实际运行验证。最终主线没有使用 public leaderboard answer 进行训练、调参或结果选择。
