#!/usr/bin/env python3
"""
Sklearn training pipeline for the Event Recommendation baseline features.

This reuses event_reco_baseline.py for robust file parsing and feature
construction, then trains several sklearn models and blends them by directly
optimizing validation MAP@200.
"""

from __future__ import annotations

import argparse
import os
from itertools import product
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from event_reco_baseline import (
    DATA_DIR,
    FeatureBuilder,
    build_everything,
    build_user_history,
    map_at_k,
    parse_int,
    write_submission,
)


def split_by_user(train_rows):
    users = sorted({row["user"] for row in train_rows}, key=lambda x: int(x))
    valid_users = set(users[int(len(users) * 0.8) :])
    train_fit = [row for row in train_rows if row["user"] not in valid_users]
    valid = [row for row in train_rows if row["user"] in valid_users]
    return train_fit, valid


def leakage_free_builder(full_builder, train_fit):
    history = build_user_history(train_fit, full_builder.events)
    return FeatureBuilder(
        full_builder.users,
        full_builder.friends,
        full_builder.friend_counts,
        full_builder.events,
        full_builder.event_stats,
        full_builder.pair_friend_stats,
        *history,
    )


def selected_feature_indices(feature_names, mode):
    if mode == "all":
        return list(range(len(feature_names)))
    return [
        i
        for i, name in enumerate(feature_names)
        if not name.startswith("user_") and not name.startswith("content_")
    ]


def add_group_rank_features(x, rows, feature_names):
    rank_specs = [
        ("log_event_yes", True),
        ("log_event_maybe", True),
        ("log_event_total", True),
        ("event_net_positive", True),
        ("friend_yes", True),
        ("friend_maybe", True),
        ("friend_net_positive", True),
        ("event_has_location", True),
        ("same_city_hint", True),
        ("same_country_hint", True),
        ("time_to_event_days", False),
        ("log_positive_time_to_event", False),
        ("event_active_word_count", True),
    ]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    specs = [(name_to_idx[name], descending) for name, descending in rank_specs if name in name_to_idx]
    if not specs:
        return x

    by_user = defaultdict(list)
    for i, row in enumerate(rows):
        by_user[row["user"]].append(i)
    extra = np.zeros((len(rows), len(specs)), dtype=np.float32)
    for col, (feature_idx, descending) in enumerate(specs):
        for indices in by_user.values():
            if len(indices) == 1:
                extra[indices[0], col] = 1.0
                continue
            values = [(float(x[i, feature_idx]), i) for i in indices]
            values.sort(reverse=descending)
            denom = len(values) - 1
            for rank, (_, row_idx) in enumerate(values):
                extra[row_idx, col] = 1.0 - rank / denom
    return np.hstack([x, extra])


def vectorize(builder, rows, indices=None, group_features=False):
    raw = [builder.transform_row(row) for row in rows]
    if group_features:
        full_x = np.asarray(raw, dtype=np.float32)
        full_x = add_group_rank_features(full_x, rows, builder.feature_names)
        if indices is not None:
            # Group-rank features are appended and are all cold-start safe.
            appended = list(range(len(builder.feature_names), full_x.shape[1]))
            keep = indices + appended
            x = full_x[:, keep]
        else:
            x = full_x
    else:
        x = np.asarray(raw, dtype=np.float32)
        if indices is not None:
            x = x[:, indices]
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.int32)
    return x, y


def positive_sample_weights(y):
    weights = np.ones(len(y), dtype=np.float32)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos:
        weights[y == 1] = neg / pos
    return weights


def score_rows(rows, y, scores):
    return [(row["user"], row["event"], int(label), float(score)) for row, label, score in zip(rows, y, scores)]


def predict_positive(model, x):
    proba = model.predict_proba(x)
    return proba[:, 1]


def fit_models(x_train, y_train, args):
    weights = positive_sample_weights(y_train)
    models = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=args.hgb_lr,
            max_iter=args.hgb_iter,
            max_leaf_nodes=15,
            l2_regularization=0.03,
            random_state=13,
        ),
        "rf": RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=17,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=args.extra_trees,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=1,
            random_state=19,
        ),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=23,
            ),
        ),
    }
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "logreg":
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=weights)
    return models


def search_blend(valid_rows, y_valid, valid_predictions):
    names = list(valid_predictions)
    best = (-1.0, None, None)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for raw in product(grid, repeat=len(names)):
        total = sum(raw)
        if total <= 0:
            continue
        weights = np.asarray(raw, dtype=np.float32) / total
        blended = np.zeros_like(next(iter(valid_predictions.values())))
        for weight, name in zip(weights, names):
            blended += weight * valid_predictions[name]
        score = map_at_k(score_rows(valid_rows, y_valid, blended))
        if score > best[0]:
            best = (score, dict(zip(names, weights.tolist())), blended)
    return best


def run(args):
    train_rows, test_rows, full_builder = build_everything()
    train_fit, valid_rows = split_by_user(train_rows)
    valid_builder = leakage_free_builder(full_builder, train_fit)
    indices = selected_feature_indices(valid_builder.feature_names, args.feature_mode)
    print(f"Feature mode={args.feature_mode}; using {len(indices)} / {len(valid_builder.feature_names)} features")
    if args.feature_mode != "all":
        print("Features:", ", ".join(valid_builder.feature_names[i] for i in indices))

    print("Vectorizing leakage-free validation split...")
    x_train, y_train = vectorize(valid_builder, train_fit, indices, args.group_features)
    x_valid, y_valid = vectorize(valid_builder, valid_rows, indices, args.group_features)
    print(f"Train rows={len(x_train)} valid rows={len(x_valid)} features={x_train.shape[1]}")

    models = fit_models(x_train, y_train, args)
    valid_predictions = {}
    for name, model in models.items():
        pred = predict_positive(model, x_valid)
        valid_predictions[name] = pred
        score = map_at_k(score_rows(valid_rows, y_valid, pred))
        print(f"{name} Validation MAP@200: {score:.6f}")

    blend_score, blend_weights, _ = search_blend(valid_rows, y_valid, valid_predictions)
    print(f"Blend Validation MAP@200: {blend_score:.6f}")
    print("Blend weights:", ", ".join(f"{k}={v:.2f}" for k, v in blend_weights.items()))

    print("Retraining selected models on full train...")
    full_indices = selected_feature_indices(full_builder.feature_names, args.feature_mode)
    x_all, y_all = vectorize(full_builder, train_rows, full_indices, args.group_features)
    full_models = fit_models(x_all, y_all, args)
    x_test, _ = vectorize(full_builder, test_rows, full_indices, args.group_features)
    blended_test = np.zeros(len(test_rows), dtype=np.float32)
    for name, model in full_models.items():
        blended_test += blend_weights[name] * predict_positive(model, x_test)

    scored_test = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, blended_test)]
    output = DATA_DIR / args.output
    legacy_output = DATA_DIR / args.output.replace(".csv", "_legacy.csv")
    write_submission(output, scored_test)
    write_submission(legacy_output, scored_test, legacy=True)
    print(f"Wrote {output.name}")
    print(f"Wrote {legacy_output.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_sklearn.csv")
    parser.add_argument("--hgb-iter", type=int, default=180)
    parser.add_argument("--hgb-lr", type=float, default=0.045)
    parser.add_argument("--rf-trees", type=int, default=450)
    parser.add_argument("--extra-trees", type=int, default=450)
    parser.add_argument("--feature-mode", choices=["all", "cold"], default="all")
    parser.add_argument("--group-features", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
