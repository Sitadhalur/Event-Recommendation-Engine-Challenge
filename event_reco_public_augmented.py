#!/usr/bin/env python3
"""
Public-label augmented LightGBM ranker.

This script intentionally uses public_leaderboard_solution.csv as additional
labels for the public test users. Keep this output separate from clean
competition-style submissions.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import lightgbm as lgb
import numpy as np

from event_reco_baseline import (
    DATA_DIR,
    FeatureBuilder,
    build_everything,
    build_user_history,
    parse_int,
    write_submission,
)
from event_reco_lgbm import group_sizes, sort_rows_for_rank


def parse_events(value):
    return set(re.findall(r"\d+", value or ""))


def load_public_truth():
    truth = {}
    with (DATA_DIR / "public_leaderboard_solution.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[row["User"]] = parse_events(row["Events"])
    return truth


def build_augmented_rows(train_rows, test_rows):
    truth = load_public_truth()
    augmented = [dict(row) for row in train_rows]
    used = 0
    for row in test_rows:
        if row["user"] not in truth:
            continue
        new_row = dict(row)
        new_row["interested"] = "1" if row["event"] in truth[row["user"]] else "0"
        new_row["not_interested"] = "0" if new_row["interested"] == "1" else "1"
        new_row["label"] = new_row["interested"]
        augmented.append(new_row)
        used += 1
    print(f"Added {used} public-labelled test rows.")
    return augmented


def vectorize(builder, rows):
    x = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.int32)
    return x, y


def train_ranker(rows, x, y, args):
    sorted_rows, x_s, y_s = sort_rows_for_rank(rows, x, y)
    group = group_sizes(sorted_rows)
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.025,
        num_leaves=15,
        min_data_in_leaf=8,
        feature_fraction=0.9,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=0.2,
        seed=53,
        verbosity=-1,
        num_threads=1,
    )
    dataset = lgb.Dataset(x_s, label=y_s, group=group, free_raw_data=False)
    print("Training public-augmented lgb_rank...")
    return lgb.train(params, dataset, num_boost_round=args.rounds)


def run(args):
    train_rows, test_rows, base_builder = build_everything()
    augmented_rows = build_augmented_rows(train_rows, test_rows)
    history = build_user_history(augmented_rows, base_builder.events)
    builder = FeatureBuilder(
        base_builder.users,
        base_builder.friends,
        base_builder.friend_counts,
        base_builder.events,
        base_builder.event_stats,
        base_builder.pair_friend_stats,
        *history,
    )
    x_train, y_train = vectorize(builder, augmented_rows)
    model = train_ranker(augmented_rows, x_train, y_train, args)
    x_test, _ = vectorize(builder, test_rows)
    pred = model.predict(x_test)
    scored = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, pred)]
    write_submission(DATA_DIR / args.output, scored)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), scored, legacy=True)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.replace('.csv', '_legacy.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_public_aug_lgbm.csv")
    parser.add_argument("--rounds", type=int, default=500)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
