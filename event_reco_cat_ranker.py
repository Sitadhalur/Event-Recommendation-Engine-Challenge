#!/usr/bin/env python3
"""CatBoostRanker experiments for cold-start event recommendation."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from catboost import CatBoostRanker, Pool

from event_reco_baseline import DATA_DIR, build_everything, map_at_k, parse_int, write_submission
from event_reco_sklearn import add_group_rank_features, score_rows, selected_feature_indices


def split_by_user(train_rows):
    users = sorted({row["user"] for row in train_rows}, key=lambda x: int(x))
    valid_users = set(users[int(len(users) * 0.8) :])
    train_fit = [row for row in train_rows if row["user"] not in valid_users]
    valid = [row for row in train_rows if row["user"] in valid_users]
    return train_fit, valid


def sort_by_user(rows, x, y):
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["user"]), int(rows[i]["event"])))
    return [rows[i] for i in order], x[order], y[order]


def vectorize(builder, rows, indices, group_features=False):
    full = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)
    if group_features:
        full = add_group_rank_features(full, rows, builder.feature_names)
        keep = indices + list(range(len(builder.feature_names), full.shape[1]))
        x = full[:, keep]
    else:
        x = full[:, indices]
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.float32)
    return x, y


def group_ids(rows):
    return [row["user"] for row in rows]


def train_ranker(x, y, rows, loss, args):
    pool = Pool(x, y, group_id=group_ids(rows))
    model = CatBoostRanker(
        iterations=args.iters,
        depth=args.depth,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        loss_function=loss,
        random_seed=args.seed,
        thread_count=1,
        verbose=False,
    )
    model.fit(pool)
    return model


def run(args):
    train_rows, test_rows, builder = build_everything()
    indices = selected_feature_indices(builder.feature_names, "cold")
    train_fit, valid_rows = split_by_user(train_rows)
    x_train, y_train = vectorize(builder, train_fit, indices, args.group_features)
    x_valid, y_valid = vectorize(builder, valid_rows, indices, args.group_features)
    sorted_train, x_train, y_train = sort_by_user(train_fit, x_train, y_train)
    sorted_valid, x_valid, y_valid = sort_by_user(valid_rows, x_valid, y_valid)
    print(f"Features={x_train.shape[1]} group_features={args.group_features}")

    losses = args.losses.split(",")
    for loss in losses:
        print(f"Training CatBoostRanker {loss}...")
        model = train_ranker(x_train, y_train, sorted_train, loss, args)
        pred_valid = model.predict(Pool(x_valid, group_id=group_ids(sorted_valid)))
        score = map_at_k(score_rows(sorted_valid, y_valid.astype(int), pred_valid))
        print(f"{loss} Validation MAP@200: {score:.6f}")

    print("Retraining on full train...")
    x_all, y_all = vectorize(builder, train_rows, indices, args.group_features)
    sorted_all, x_all, y_all = sort_by_user(train_rows, x_all, y_all)
    x_test, _ = vectorize(builder, test_rows, indices, args.group_features)
    sorted_test, x_test, _ = sort_by_user(test_rows, x_test, np.zeros(len(test_rows), dtype=np.float32))
    for loss in losses:
        model = train_ranker(x_all, y_all, sorted_all, loss, args)
        pred_test = model.predict(Pool(x_test, group_id=group_ids(sorted_test)))
        scored = [(row["user"], row["event"], float(score)) for row, score in zip(sorted_test, pred_test)]
        safe_loss = loss.replace(":", "_").replace(";", "_")
        output = f"{args.output_prefix}_{safe_loss}.csv"
        write_submission(DATA_DIR / output, scored)
        write_submission(DATA_DIR / output.replace(".csv", "_legacy.csv"), scored, legacy=True)
        print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="submission_cat_rank")
    parser.add_argument("--losses", default="YetiRank,QuerySoftMax,QueryCrossEntropy")
    parser.add_argument("--group-features", action="store_true")
    parser.add_argument("--iters", type=int, default=450)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
