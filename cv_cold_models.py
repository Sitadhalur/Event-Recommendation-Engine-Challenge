#!/usr/bin/env python3
"""Group cold-start cross validation for model stability checks."""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from event_reco_baseline import build_everything, map_at_k, parse_int
from event_reco_sklearn import add_group_rank_features, score_rows, selected_feature_indices


def make_folds(rows, n_folds, seed):
    users = sorted({row["user"] for row in rows}, key=lambda x: int(x))
    rng = random.Random(seed)
    rng.shuffle(users)
    folds = []
    for fold in range(n_folds):
        valid_users = set(users[fold::n_folds])
        train_rows = [row for row in rows if row["user"] not in valid_users]
        valid_rows = [row for row in rows if row["user"] in valid_users]
        folds.append((train_rows, valid_rows))
    return folds


def vectorize(builder, rows, indices, group_features=False):
    full = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)
    if group_features:
        full = add_group_rank_features(full, rows, builder.feature_names)
        keep = indices + list(range(len(builder.feature_names), full.shape[1]))
        x = full[:, keep]
    else:
        x = full[:, indices]
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.int32)
    return x, y


def fit_predict(name, x_train, y_train, x_valid, args):
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale = neg / max(pos, 1)
    if name == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=args.hgb_iter,
            max_leaf_nodes=15,
            l2_regularization=0.03,
            random_state=13,
        )
    elif name == "rf":
        model = RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=17,
        )
    elif name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, solver="lbfgs"),
        )
    elif name == "xgb":
        model = XGBClassifier(
            n_estimators=args.xgb_trees,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=2.0,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale,
            n_jobs=1,
            random_state=31,
        )
    elif name == "cat":
        model = CatBoostClassifier(
            iterations=args.cat_iters,
            depth=4,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=43,
            thread_count=1,
            verbose=False,
        )
    else:
        raise ValueError(name)
    model.fit(x_train, y_train)
    return model.predict_proba(x_valid)[:, 1]


def run(args):
    train_rows, _, builder = build_everything()
    indices = selected_feature_indices(builder.feature_names, "cold")
    print(f"Cold features={len(indices)} folds={args.folds}")
    model_names = args.models.split(",")
    results = {name: [] for name in model_names}
    if args.group_features:
        results.update({name + "_group": [] for name in model_names})

    for fold_idx, (fit_rows, valid_rows) in enumerate(make_folds(train_rows, args.folds, args.seed), start=1):
        print(f"Fold {fold_idx}: train_rows={len(fit_rows)} valid_rows={len(valid_rows)}")
        x_train, y_train = vectorize(builder, fit_rows, indices, False)
        x_valid, y_valid = vectorize(builder, valid_rows, indices, False)
        for name in model_names:
            pred = fit_predict(name, x_train, y_train, x_valid, args)
            score = map_at_k(score_rows(valid_rows, y_valid, pred))
            results[name].append(score)
            print(f"  {name}: {score:.6f}")
        if args.group_features:
            x_train_g, y_train_g = vectorize(builder, fit_rows, indices, True)
            x_valid_g, y_valid_g = vectorize(builder, valid_rows, indices, True)
            for name in model_names:
                pred = fit_predict(name, x_train_g, y_train_g, x_valid_g, args)
                score = map_at_k(score_rows(valid_rows, y_valid_g, pred))
                results[name + "_group"].append(score)
                print(f"  {name}_group: {score:.6f}")

    print("\nSummary")
    for name, scores in results.items():
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        print(f"{name:12s} mean={mean:.6f} std={std:.6f} scores=" + ",".join(f"{x:.6f}" for x in scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--models", default="hgb,rf,logreg,xgb,cat")
    parser.add_argument("--group-features", action="store_true")
    parser.add_argument("--hgb-iter", type=int, default=180)
    parser.add_argument("--rf-trees", type=int, default=250)
    parser.add_argument("--xgb-trees", type=int, default=220)
    parser.add_argument("--cat-iters", type=int, default=320)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
