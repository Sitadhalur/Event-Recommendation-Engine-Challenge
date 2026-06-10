# 项目交付说明

这是 Event Recommendation Engine Challenge 项目的交付包。

## 最重要的文件

- `event_recommendation_report.md`  
  中文项目报告，包含任务说明、数据分析、防泄露说明、方法、结果和排行榜对比。

- `train_only_oof_blend.py`  
  最终主线训练脚本。使用严格 5-fold OOF，不使用 public leaderboard answer，OOF 阶段不使用 test metadata，默认剔除 `event_label_` 特征。

- `event_reco_baseline.py`  
  核心数据读取和特征工程脚本。

- `event_reco_sklearn.py`  
  部分通用 sklearn 工具函数，最终主线会调用其中的特征选择和评分辅助函数。

- `submission_oof_ultrastrict_5fold_hgb_xgb_cat.csv`  
  最终推荐使用的提交格式文件。

- `submission_oof_ultrastrict_5fold_hgb_xgb_cat_legacy.csv`  
  legacy list 格式备份。

## 最终结果

严格本地 5-fold grouped OOF：

```text
MAP@200 = 0.718516
```

注意：Kaggle 比赛已关闭，无法官方提交；该分数是课程 Case B 下的本地 5-fold CV 分数，不是 Kaggle private leaderboard 分数。

## 推荐复现命令

```powershell
& train_only_oof_blend.py --folds 5 --models hgb,xgb,cat --output submission_oof_ultrastrict_5fold_hgb_xgb_cat.csv --hgb-iter 320 --xgb-trees 380 --cat-iters 360 --trials 20000
```

## 数据文件

所需原始数据：

- `train.csv`
- `test.csv`
- `users.csv`
- `events.csv.gz`
- `event_attendees.csv.gz`
- `user_friends.csv.gz`

以及地理匹配辅助数据：

- `countryInfo.txt`
- `cities5000.zip`

## 关于 public leaderboard 文件

包中保留了：

- `public_leaderboard_solution.csv`

它仅用于说明历史探索和 Kaggle 数据背景。最终主线没有使用它训练、调参或选择结果。报告中也不要把 public-tuned 结果作为最终主结果。

以下脚本属于早期探索或 public subset 分析，不要作为最终成绩依据：

- `blend_submissions.py`
- `refine_blend.py`
- `score_rule_submission.py`
- `event_reco_public_augmented.py`

## 中间产物说明

保留了若干中间 submission，用于展示实验过程：

- `submission_oof_strict_drop_eventlabel_5fold_hgb_xgb_cat.csv`
- `submission_oof_geo_5fold_hgb_xgb_rf.csv`
- `submission_blend_cold_v8.csv`
- `submission_nn.csv`
- `submission_graph_nn_clean.csv`

其中最终推荐结果是：

```text
submission_oof_ultrastrict_5fold_hgb_xgb_cat.csv
```

## 防泄露口径

最终主线做了以下处理：

1. 不使用 `public_leaderboard_solution.csv`。
2. 每一折只用训练折构造标签衍生特征。
3. OOF 阶段使用 `include_test_metadata=False`，不让 test 输入分布影响本地 CV。
4. 最终生成提交时才使用 train + test metadata。
5. 默认剔除 `event_label_` 特征。
6. 已检查 `event_attendees.csv.gz` 中目标用户本人没有直接出现在对应 event 的 yes/maybe/invited/no 列表中。

## Kaggle timestamp 更新

Kaggle host 曾说明 test timestamp 存在泄露，因此 leaderboard 只对 timestamp 完全一致的 787 个 test 用户计分，另外 570 个用户仍保留在 test.csv 和提交格式中但不计分。

本项目没有利用该 timestamp 泄露做后处理。提交文件仍包含全部 test 用户，格式是正确的。

