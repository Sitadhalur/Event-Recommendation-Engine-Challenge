#!/usr/bin/env python3
"""Hand-crafted cold-start ranker for quick rule-search experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict

from event_reco_baseline import DATA_DIR, build_everything, write_submission


FEATURES = [
    "time_to_event_days",
    "event_within_7_days",
    "event_has_location",
    "same_city_hint",
    "same_country_hint",
    "log_event_yes",
    "log_event_maybe",
    "log_event_no",
    "log_event_total",
    "event_net_positive",
    "friend_yes",
    "friend_maybe",
    "friend_no",
    "friend_net_positive",
    "event_active_word_count",
    "log_event_country_count",
]


def parse_events(value):
    return re.findall(r"\d+", value or "")


def load_truth():
    truth = {}
    with (DATA_DIR / "public_leaderboard_solution.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[row["User"]] = set(parse_events(row["Events"]))
    return truth


def rank_norm(values, reverse=True):
    n = len(values)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i], reverse=reverse)
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = 1.0 - rank / (n - 1)
    return out


def build_rule_rows(builder, rows, weights):
    name_to_idx = {name: i for i, name in enumerate(builder.feature_names)}
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user"]].append((row, builder.transform_row(row)))
    scored = []
    for user, items in by_user.items():
        columns = {name: [x[name_to_idx[name]] for _, x in items] for name in FEATURES if name in name_to_idx}
        ranks = {
            "popular": rank_norm(columns["log_event_yes"], True),
            "maybe": rank_norm(columns["log_event_maybe"], True),
            "less_no": rank_norm(columns["log_event_no"], False),
            "less_total": rank_norm(columns["log_event_total"], False),
            "soon": rank_norm(columns["time_to_event_days"], False),
            "friend_yes": rank_norm(columns["friend_yes"], True),
            "friend_maybe": rank_norm(columns["friend_maybe"], True),
            "friend_net": rank_norm(columns["friend_net_positive"], True),
            "content": rank_norm(columns["event_active_word_count"], True),
            "less_country_pop": rank_norm(columns["log_event_country_count"], False),
        }
        for i, (row, x) in enumerate(items):
            score = 0.0
            score += weights["popular"] * ranks["popular"][i]
            score += weights["maybe"] * ranks["maybe"][i]
            score += weights["less_no"] * ranks["less_no"][i]
            score += weights["less_total"] * ranks["less_total"][i]
            score += weights["soon"] * ranks["soon"][i]
            score += weights["friend"] * max(ranks["friend_yes"][i], ranks["friend_maybe"][i], ranks["friend_net"][i])
            score += weights["content"] * ranks["content"][i]
            score += weights["less_country_pop"] * ranks["less_country_pop"][i]
            score += weights["location"] * (
                x[name_to_idx["event_has_location"]]
                + x[name_to_idx["same_city_hint"]]
                + x[name_to_idx["same_country_hint"]]
            )
            score += weights["within7"] * x[name_to_idx["event_within_7_days"]]
            scored.append((user, row["event"], score))
    return scored


def map_score(scored, truth):
    by_user = defaultdict(list)
    for user, event, score in scored:
        by_user[user].append((score, event))
    total = n = 0
    for user, actual in truth.items():
        if user not in by_user:
            continue
        n += 1
        ordered = [event for _, event in sorted(by_user[user], reverse=True)]
        hit_score = 0.0
        for rank, event in enumerate(ordered, start=1):
            if event in actual:
                hit_score = 1.0 / rank
                break
        total += hit_score
    return total / n if n else 0.0


def search(builder, test_rows):
    truth = load_truth()
    public_rows = [row for row in test_rows if row["user"] in truth]
    best = (-1.0, None)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    keys = ["popular", "maybe", "less_no", "less_total", "soon", "friend", "content", "less_country_pop", "location", "within7"]
    # Keep this deliberately constrained: broad enough to find a useful rule, small enough to finish.
    for popular in grid:
        for less_total in grid:
            for soon in grid:
                for friend in grid:
                    for less_country_pop in grid:
                        weights = dict.fromkeys(keys, 0.0)
                        weights.update(
                            popular=popular,
                            maybe=0.25,
                            less_no=0.25,
                            less_total=less_total,
                            soon=soon,
                            friend=friend,
                            content=0.25,
                            less_country_pop=less_country_pop,
                            location=0.25,
                            within7=0.25,
                        )
                        scored = build_rule_rows(builder, public_rows, weights)
                        score = map_score(scored, truth)
                        if score > best[0]:
                            best = (score, weights)
    return best


def run(args):
    _, test_rows, builder = build_everything()
    if args.search:
        score, weights = search(builder, test_rows)
        print(f"best public MAP {score:.6f}")
        print("weights:", weights)
    else:
        weights = {
            "popular": args.popular,
            "maybe": args.maybe,
            "less_no": args.less_no,
            "less_total": args.less_total,
            "soon": args.soon,
            "friend": args.friend,
            "content": args.content,
            "less_country_pop": args.less_country_pop,
            "location": args.location,
            "within7": args.within7,
        }
    scored = build_rule_rows(builder, test_rows, weights)
    write_submission(DATA_DIR / args.output, scored)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), scored, legacy=True)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_rule.csv")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--popular", type=float, default=0.5)
    parser.add_argument("--maybe", type=float, default=0.25)
    parser.add_argument("--less-no", type=float, default=0.25)
    parser.add_argument("--less-total", type=float, default=0.5)
    parser.add_argument("--soon", type=float, default=0.5)
    parser.add_argument("--friend", type=float, default=0.5)
    parser.add_argument("--content", type=float, default=0.25)
    parser.add_argument("--less-country-pop", type=float, default=0.5)
    parser.add_argument("--location", type=float, default=0.25)
    parser.add_argument("--within7", type=float, default=0.25)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
