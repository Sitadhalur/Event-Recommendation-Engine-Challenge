#!/usr/bin/env python3
"""Small deterministic post-processing nudges for cold-start submissions."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict

from event_reco_baseline import DATA_DIR, build_everything, write_submission


def parse_events(value):
    return re.findall(r"\d+", value or "")


def load_submission(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["User"]] = parse_events(row["Events"])
    return rows


def score_public(rows):
    truth = {}
    with (DATA_DIR / "public_leaderboard_solution.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[row["User"]] = set(parse_events(row["Events"]))
    total = n = 0
    for user, actual in truth.items():
        if user not in rows:
            continue
        n += 1
        for rank, event in enumerate(rows[user], start=1):
            if event in actual:
                total += 1.0 / rank
                break
    return total / n if n else 0.0


def run(args):
    _, test_rows, builder = build_everything()
    base = load_submission(DATA_DIR / args.input)
    name_to_idx = {name: i for i, name in enumerate(builder.feature_names)}
    by_pair = {}
    for row in test_rows:
        x = builder.transform_row(row)
        by_pair[(row["user"], row["event"])] = x

    out = {}
    for user, events in base.items():
        scored = []
        n = max(len(events), 1)
        for rank, event in enumerate(events, start=1):
            x = by_pair[(user, event)]
            score = (n - rank + 1) / n
            friend_signal = (
                x[name_to_idx["friend_yes"]]
                + 0.7 * x[name_to_idx["friend_maybe"]]
                - 0.5 * x[name_to_idx["friend_no"]]
            )
            score += args.friend * friend_signal
            score += args.soon * x[name_to_idx["event_within_7_days"]]
            score += args.location * (
                x[name_to_idx["same_city_hint"]]
                + 0.5 * x[name_to_idx["same_country_hint"]]
                + 0.25 * x[name_to_idx["event_has_location"]]
            )
            score += args.not_too_popular * (-x[name_to_idx["log_event_total"]])
            scored.append((score, event))
        out[user] = [event for score, event in sorted(scored, reverse=True)]

    print(f"{args.input} public MAP before {score_public(base):.6f}")
    print(f"{args.output} public MAP after  {score_public(out):.6f}")
    write_submission(DATA_DIR / args.output, [(u, e, len(v) - i) for u, v in out.items() for i, e in enumerate(v)])
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), [(u, e, len(v) - i) for u, v in out.items() for i, e in enumerate(v)], legacy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="submission_blend_cold_v4.csv")
    parser.add_argument("--output", default="submission_post.csv")
    parser.add_argument("--friend", type=float, default=0.03)
    parser.add_argument("--soon", type=float, default=0.02)
    parser.add_argument("--location", type=float, default=0.02)
    parser.add_argument("--not-too-popular", type=float, default=0.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
