#!/usr/bin/env python3
"""
LightGBM pipeline for the Event Recommendation challenge.

Uses the robust feature builder from event_reco_baseline.py, trains binary and
LambdaRank models, blends them on validation MAP@200, and writes submissions.
"""

from __future__ import annotations

import argparse
import os
from itertools import product

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

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


def vectorize(builder, rows, indices=None):
    x = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)
    if indices is not None:
        x = x[:, indices]
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.int32)
    return x, y


def group_sizes(rows):
    sizes = []
    last = None
    count = 0
    for row in sorted(rows, key=lambda r: (int(r["user"]), int(r["event"]))):
        user = row["user"]
        if last is not None and user != last:
            sizes.append(count)
            count = 0
        last = user
        count += 1
    if count:
        sizes.append(count)
    return sizes


def sort_rows_for_rank(rows, x, y):
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["user"]), int(rows[i]["event"])))
    return [rows[i] for i in order], x[order], y[order]


def positive_weights(y):
    weights = np.ones(len(y), dtype=np.float32)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos:
        weights[y == 1] = neg / pos
    return weights


def score_rows(rows, y, scores):
    return [(row["user"], row["event"], int(label), float(score)) for row, label, score in zip(rows, y, scores)]


def train_binary_models(x_train, y_train, x_valid, y_valid, args):
    scale_pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    models = {}

    configs = {
        "lgb_bin_a": dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.035,
            num_leaves=15,
            min_data_in_leaf=12,
            feature_fraction=0.9,
            bagging_fraction=0.85,
            bagging_freq=1,
            lambda_l1=0.03,
            lambda_l2=0.15,
            scale_pos_weight=scale_pos_weight,
            seed=13,
            verbosity=-1,
            num_threads=1,
        ),
        "lgb_bin_b": dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.02,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.75,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l1=0.0,
            lambda_l2=0.4,
            scale_pos_weight=scale_pos_weight,
            seed=29,
            verbosity=-1,
            num_threads=1,
        ),
    }

    train_set = lgb.Dataset(x_train, label=y_train, weight=positive_weights(y_train), free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)
    for name, params in configs.items():
        print(f"Training {name}...")
        models[name] = lgb.train(
            params,
            train_set,
            num_boost_round=args.rounds,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
    return models


def train_ranker(train_rows, x_train, y_train, valid_rows, x_valid, y_valid, args):
    sorted_train_rows, x_train_s, y_train_s = sort_rows_for_rank(train_rows, x_train, y_train)
    sorted_valid_rows, x_valid_s, y_valid_s = sort_rows_for_rank(valid_rows, x_valid, y_valid)
    train_group = group_sizes(sorted_train_rows)
    valid_group = group_sizes(sorted_valid_rows)

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.035,
        num_leaves=15,
        min_data_in_leaf=8,
        feature_fraction=0.9,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=0.2,
        seed=41,
        verbosity=-1,
        num_threads=1,
    )
    print("Training lgb_rank...")
    train_set = lgb.Dataset(x_train_s, label=y_train_s, group=train_group, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid_s, label=y_valid_s, group=valid_group, reference=train_set, free_raw_data=False)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.rank_rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    valid_pred_sorted = model.predict(x_valid_s, num_iteration=model.best_iteration)
    pred_by_pair = {
        (row["user"], row["event"]): pred for row, pred in zip(sorted_valid_rows, valid_pred_sorted)
    }
    valid_pred = np.asarray([pred_by_pair[(row["user"], row["event"])] for row in valid_rows], dtype=np.float32)
    return model, valid_pred


def train_tree_sidecars(x_train, y_train, args):
    models = {
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
    }
    for name, model in models.items():
        print(f"Training {name} sidecar...")
        model.fit(x_train, y_train)
    return models


def search_blend(valid_rows, y_valid, predictions):
    names = list(predictions)
    best = (-1.0, None)
    grid = np.linspace(0.0, 1.0, 11)
    for raw in product(grid, repeat=len(names)):
        total = sum(raw)
        if total <= 0:
            continue
        weights = np.asarray(raw, dtype=np.float32) / total
        blended = np.zeros(len(y_valid), dtype=np.float32)
        for weight, name in zip(weights, names):
            values = predictions[name].astype(np.float32)
            if np.std(values) > 0:
                values = (values - np.mean(values)) / np.std(values)
            blended += weight * values
        score = map_at_k(score_rows(valid_rows, y_valid, blended))
        if score > best[0]:
            best = (score, dict(zip(names, weights.tolist())))
    return best


def print_importance(model, names, limit=12):
    gains = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(names, gains), key=lambda item: item[1], reverse=True)[:limit]
    print("Top LightGBM gain features:", ", ".join(f"{name}={gain:.1f}" for name, gain in pairs))


def run(args):
    train_rows, test_rows, full_builder = build_everything()
    train_fit, valid_rows = split_by_user(train_rows)
    valid_builder = leakage_free_builder(full_builder, train_fit)
    indices = selected_feature_indices(valid_builder.feature_names, args.feature_mode)
    selected_names = [valid_builder.feature_names[i] for i in indices]
    print(f"Feature mode={args.feature_mode}; using {len(indices)} / {len(valid_builder.feature_names)} features")
    if args.feature_mode != "all":
        print("Features:", ", ".join(selected_names))

    print("Vectorizing leakage-free validation split...")
    x_train, y_train = vectorize(valid_builder, train_fit, indices)
    x_valid, y_valid = vectorize(valid_builder, valid_rows, indices)
    print(f"Train rows={len(x_train)} valid rows={len(x_valid)} features={x_train.shape[1]}")

    bin_models = train_binary_models(x_train, y_train, x_valid, y_valid, args)
    rank_model, rank_valid = train_ranker(train_fit, x_train, y_train, valid_rows, x_valid, y_valid, args)
    sidecars = train_tree_sidecars(x_train, y_train, args)

    predictions = {}
    for name, model in bin_models.items():
        predictions[name] = model.predict(x_valid, num_iteration=model.best_iteration)
    predictions["lgb_rank"] = rank_valid
    for name, model in sidecars.items():
        predictions[name] = model.predict_proba(x_valid)[:, 1]

    for name, pred in predictions.items():
        score = map_at_k(score_rows(valid_rows, y_valid, pred))
        print(f"{name} Validation MAP@200: {score:.6f}")
    blend_score, blend_weights = search_blend(valid_rows, y_valid, predictions)
    print(f"Blend Validation MAP@200: {blend_score:.6f}")
    print("Blend weights:", ", ".join(f"{k}={v:.2f}" for k, v in blend_weights.items()))
    print_importance(bin_models["lgb_bin_a"], selected_names)

    print("Retraining on full train...")
    full_indices = selected_feature_indices(full_builder.feature_names, args.feature_mode)
    x_all, y_all = vectorize(full_builder, train_rows, full_indices)
    x_test, _ = vectorize(full_builder, test_rows, full_indices)
    full_bin = train_binary_models(x_all, y_all, x_all, y_all, args)
    full_rank, _ = train_ranker(train_rows, x_all, y_all, train_rows, x_all, y_all, args)
    full_sidecars = train_tree_sidecars(x_all, y_all, args)

    test_predictions = {}
    for name, model in full_bin.items():
        test_predictions[name] = model.predict(x_test, num_iteration=model.best_iteration)
    test_predictions["lgb_rank"] = full_rank.predict(x_test, num_iteration=full_rank.best_iteration)
    for name, model in full_sidecars.items():
        test_predictions[name] = model.predict_proba(x_test)[:, 1]

    blended_test = np.zeros(len(test_rows), dtype=np.float32)
    for name, weight in blend_weights.items():
        values = test_predictions[name].astype(np.float32)
        if np.std(values) > 0:
            values = (values - np.mean(values)) / np.std(values)
        blended_test += weight * values

    scored_test = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, blended_test)]
    output = DATA_DIR / args.output
    legacy_output = DATA_DIR / args.output.replace(".csv", "_legacy.csv")
    write_submission(output, scored_test)
    write_submission(legacy_output, scored_test, legacy=True)
    print(f"Wrote {output.name}")
    print(f"Wrote {legacy_output.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_lgbm.csv")
    parser.add_argument("--rounds", type=int, default=450)
    parser.add_argument("--rank-rounds", type=int, default=350)
    parser.add_argument("--rf-trees", type=int, default=350)
    parser.add_argument("--extra-trees", type=int, default=350)
    parser.add_argument("--feature-mode", choices=["all", "cold"], default="all")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
