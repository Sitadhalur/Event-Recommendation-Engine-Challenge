#!/usr/bin/env python3
"""No-public-label OOF blending for cold-start event recommendation.

This script uses only train.csv labels to choose blend weights. It creates
grouped user folds, trains several cold-start models, optimizes MAP@200 on OOF
predictions, retrains on the full train set, and writes a submission.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

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

from event_reco_baseline import (
    DATA_DIR,
    FeatureBuilder,
    build_everything,
    build_friend_profiles,
    build_user_history,
    map_at_k,
    parse_int,
    write_submission,
)
from event_reco_sklearn import score_rows, selected_feature_indices


def make_folds(rows, n_folds, seed):
    users = sorted({row["user"] for row in rows}, key=lambda x: int(x))
    rng = random.Random(seed)
    rng.shuffle(users)
    for fold in range(n_folds):
        valid_users = set(users[fold::n_folds])
        train_rows = [row for row in rows if row["user"] not in valid_users]
        valid_rows = [row for row in rows if row["user"] in valid_users]
        yield train_rows, valid_rows


def vectorize(builder, rows, indices):
    x = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)
    x = x[:, indices]
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.int32)
    return x, y


def apply_feature_drops(feature_names, indices, drop_prefixes):
    prefixes = [prefix.strip() for prefix in drop_prefixes.split(",") if prefix.strip()]
    if not prefixes:
        return indices
    return [i for i in indices if not any(feature_names[i].startswith(prefix) for prefix in prefixes)]


def history_builder_from_rows(full_builder, rows):
    history = build_user_history(rows, full_builder.events)
    friend_profiles = build_friend_profiles(
        full_builder.friends,
        history[1],
        history[2],
        history[3],
        history[4],
        history[5],
        history[6],
    )
    return FeatureBuilder(
        full_builder.users,
        full_builder.friends,
        full_builder.friend_counts,
        full_builder.events,
        full_builder.event_stats,
        full_builder.pair_friend_stats,
        *history,
        *friend_profiles,
    )


def model_factories(args, y_train):
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale = neg / max(pos, 1)
    return {
        "hgb": lambda: HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=args.hgb_iter,
            max_leaf_nodes=15,
            l2_regularization=0.03,
            random_state=13,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=17,
        ),
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, solver="lbfgs"),
        ),
        "xgb": lambda: XGBClassifier(
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
        "cat": lambda: CatBoostClassifier(
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


def fit_predict_models(model_names, x_train, y_train, x_valid, args):
    preds = {}
    factories = model_factories(args, y_train)
    for name in model_names:
        model = factories[name]()
        model.fit(x_train, y_train)
        preds[name] = model.predict_proba(x_valid)[:, 1].astype(np.float32)
    return preds


def normalize(values):
    values = values.astype(np.float32)
    std = float(np.std(values))
    if std > 0:
        return (values - float(np.mean(values))) / std
    return values


def group_rank_scores(rows, values):
    by_user = defaultdict(list)
    for i, row in enumerate(rows):
        by_user[row["user"]].append(i)
    out = np.zeros(len(rows), dtype=np.float32)
    for indices in by_user.values():
        if len(indices) == 1:
            out[indices[0]] = 1.0
            continue
        ordered = sorted(indices, key=lambda i: float(values[i]), reverse=True)
        denom = len(ordered) - 1
        for rank, idx in enumerate(ordered):
            out[idx] = 1.0 - rank / denom
    return out


def search_weights(rows, y, pred_matrix, model_names, trials, seed, blend_mode):
    rng = random.Random(seed)
    best_score = -1.0
    best_weights = None
    n_models = len(model_names)
    if blend_mode == "rank":
        norm_preds = {name: group_rank_scores(rows, pred_matrix[name]) for name in model_names}
    else:
        norm_preds = {name: normalize(pred_matrix[name]) for name in model_names}
    for _ in range(trials):
        raw = [rng.expovariate(1.0) for _ in range(n_models)]
        total = sum(raw)
        weights = [x / total for x in raw]
        blended = np.zeros(len(rows), dtype=np.float32)
        for weight, name in zip(weights, model_names):
            blended += weight * norm_preds[name]
        score = map_at_k(score_rows(rows, y, blended))
        if score > best_score:
            best_score = score
            best_weights = dict(zip(model_names, weights))
    return best_score, best_weights


def train_full_and_predict(model_names, x_train, y_train, x_test, args):
    preds = {}
    factories = model_factories(args, y_train)
    for name in model_names:
        print(f"Training full {name}...")
        model = factories[name]()
        model.fit(x_train, y_train)
        preds[name] = model.predict_proba(x_test)[:, 1].astype(np.float32)
    return preds


def run(args):
    train_rows, test_rows, cv_builder = build_everything(include_test_metadata=False)
    indices = selected_feature_indices(cv_builder.feature_names, "cold")
    indices = apply_feature_drops(cv_builder.feature_names, indices, args.drop_prefixes)
    model_names = args.models.split(",")
    print(f"No-public OOF blend: models={model_names}, folds={args.folds}, features={len(indices)}")

    oof_rows = []
    oof_y = []
    oof_preds = {name: [] for name in model_names}
    for fold_idx, (fit_rows, valid_rows) in enumerate(make_folds(train_rows, args.folds, args.seed), start=1):
        print(f"Fold {fold_idx}: train_rows={len(fit_rows)} valid_rows={len(valid_rows)}")
        fold_builder = history_builder_from_rows(cv_builder, fit_rows)
        x_fit, y_fit = vectorize(fold_builder, fit_rows, indices)
        x_valid, y_valid = vectorize(fold_builder, valid_rows, indices)
        preds = fit_predict_models(model_names, x_fit, y_fit, x_valid, args)
        for name in model_names:
            score = map_at_k(score_rows(valid_rows, y_valid, preds[name]))
            print(f"  {name}: {score:.6f}")
            oof_preds[name].extend(preds[name].tolist())
        oof_rows.extend(valid_rows)
        oof_y.extend(y_valid.tolist())

    oof_y = np.asarray(oof_y, dtype=np.int32)
    pred_matrix = {name: np.asarray(values, dtype=np.float32) for name, values in oof_preds.items()}
    score, weights = search_weights(
        oof_rows, oof_y, pred_matrix, model_names, args.trials, args.seed + 1000, args.blend_mode
    )
    print(f"OOF blend MAP@200: {score:.6f}")
    print("OOF weights:", ", ".join(f"{k}={v:.4f}" for k, v in weights.items()))

    _, _, final_builder = build_everything(include_test_metadata=True)
    x_all, y_all = vectorize(final_builder, train_rows, indices)
    x_test, _ = vectorize(final_builder, test_rows, indices)
    test_preds = train_full_and_predict(model_names, x_all, y_all, x_test, args)
    blended = np.zeros(len(test_rows), dtype=np.float32)
    for name, weight in weights.items():
        if args.blend_mode == "rank":
            blended += weight * group_rank_scores(test_rows, test_preds[name])
        else:
            blended += weight * normalize(test_preds[name])
    scored = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, blended)]
    write_submission(DATA_DIR / args.output, scored)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), scored, legacy=True)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_oof_no_public.csv")
    parser.add_argument("--models", default="hgb,rf,logreg,xgb,cat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--blend-mode", choices=["zscore", "rank"], default="zscore")
    parser.add_argument("--hgb-iter", type=int, default=180)
    parser.add_argument("--rf-trees", type=int, default=250)
    parser.add_argument("--xgb-trees", type=int, default=220)
    parser.add_argument("--cat-iters", type=int, default=320)
    parser.add_argument("--drop-prefixes", default="event_label_")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
