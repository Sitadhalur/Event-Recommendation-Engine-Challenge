#!/usr/bin/env python3
"""XGBoost and CatBoost cold-start models for the event recommender."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from event_reco_baseline import DATA_DIR, build_everything, map_at_k, parse_int, write_submission
from event_reco_sklearn import add_group_rank_features, score_rows, selected_feature_indices


def split_by_user(train_rows):
    users = sorted({row["user"] for row in train_rows}, key=lambda x: int(x))
    valid_users = set(users[int(len(users) * 0.8) :])
    train_fit = [row for row in train_rows if row["user"] not in valid_users]
    valid = [row for row in train_rows if row["user"] in valid_users]
    return train_fit, valid


def vectorize(builder, rows, indices, group_features):
    full = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)
    if group_features:
        full = add_group_rank_features(full, rows, builder.feature_names)
        keep = indices + list(range(len(builder.feature_names), full.shape[1]))
        x = full[:, keep]
    else:
        x = full[:, indices]
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.int32)
    return x, y


def positive_ratio(y):
    pos = int(y.sum())
    neg = len(y) - pos
    return neg / max(pos, 1)


def fit_models(x_train, y_train, args):
    scale = positive_ratio(y_train)
    models = {
        "xgb_a": XGBClassifier(
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
        ),
        "xgb_b": XGBClassifier(
            n_estimators=args.xgb_trees,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.75,
            min_child_weight=2,
            reg_lambda=4.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale,
            n_jobs=1,
            random_state=37,
        ),
        "cat": CatBoostClassifier(
            iterations=args.cat_iters,
            depth=4,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=43,
            thread_count=1,
            verbose=False,
        ),
    }
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
    return models


def predict_positive(model, x):
    return model.predict_proba(x)[:, 1]


def run(args):
    train_rows, test_rows, builder = build_everything()
    train_fit, valid_rows = split_by_user(train_rows)
    indices = selected_feature_indices(builder.feature_names, "cold")
    print(f"Using {len(indices)} cold features; group_features={args.group_features}")

    x_train, y_train = vectorize(builder, train_fit, indices, args.group_features)
    x_valid, y_valid = vectorize(builder, valid_rows, indices, args.group_features)
    print(f"Train rows={len(x_train)} valid rows={len(x_valid)} features={x_train.shape[1]}")
    models = fit_models(x_train, y_train, args)
    for name, model in models.items():
        pred = predict_positive(model, x_valid)
        print(f"{name} Validation MAP@200: {map_at_k(score_rows(valid_rows, y_valid, pred)):.6f}")

    x_all, y_all = vectorize(builder, train_rows, indices, args.group_features)
    x_test, _ = vectorize(builder, test_rows, indices, args.group_features)
    full_models = fit_models(x_all, y_all, args)
    for name, model in full_models.items():
        pred = predict_positive(model, x_test)
        scored = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, pred)]
        output = args.output_prefix + f"_{name}.csv"
        write_submission(DATA_DIR / output, scored)
        write_submission(DATA_DIR / output.replace(".csv", "_legacy.csv"), scored, legacy=True)
        print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="submission_boost")
    parser.add_argument("--group-features", action="store_true")
    parser.add_argument("--xgb-trees", type=int, default=350)
    parser.add_argument("--cat-iters", type=int, default=450)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
