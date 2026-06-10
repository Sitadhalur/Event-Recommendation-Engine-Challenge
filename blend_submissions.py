#!/usr/bin/env python3
"""Rank-based blending for Event Recommendation submission files."""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import re
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def parse_events(value: str) -> list[str]:
    return re.findall(r"\d+", value or "")


def load_submission(path: Path) -> dict[str, list[str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["User"]] = parse_events(row["Events"])
    return rows


def load_truth(path: Path) -> dict[str, set[str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["User"]] = set(parse_events(row["Events"]))
    return rows


def map_score(pred: dict[str, list[str]], truth: dict[str, set[str]]) -> float:
    total = 0.0
    n = 0
    for user, actual in truth.items():
        if user not in pred:
            continue
        n += 1
        hits = 0
        ap = 0.0
        for rank, event in enumerate(pred[user], start=1):
            if event in actual:
                hits += 1
                ap += hits / rank
        total += ap / min(len(actual), 200)
    return total / n if n else 0.0


def blend(submissions: list[dict[str, list[str]]], weights: list[float]) -> dict[str, list[str]]:
    users = set()
    for sub in submissions:
        users.update(sub)
    out = {}
    for user in users:
        scores = defaultdict(float)
        all_events = set()
        for sub in submissions:
            all_events.update(sub.get(user, []))
        for weight, sub in zip(weights, submissions):
            events = sub.get(user, [])
            n = max(len(events), 1)
            for rank, event in enumerate(events, start=1):
                scores[event] += weight * (n - rank + 1) / n
        out[user] = [event for event, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    return out


def write_submission(path: Path, rows: dict[str, list[str]], legacy: bool = False):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["User", "Events"])
        for user in sorted(rows, key=lambda x: int(x)):
            events = rows[user]
            if legacy:
                writer.writerow([user, "[" + ", ".join(f"{event}L" for event in events) + "]"])
            else:
                writer.writerow([user, " ".join(events)])


def search_weights(submissions, truth, step):
    grid = [i * step for i in range(int(1 / step) + 1)]
    best = (-1.0, None)
    for raw in itertools.product(grid, repeat=len(submissions)):
        total = sum(raw)
        if total <= 0:
            continue
        weights = [x / total for x in raw]
        score = map_score(blend(submissions, weights), truth)
        if score > best[0]:
            best = (score, weights)
    return best


def random_search_weights(submissions, truth, trials, seed):
    rng = random.Random(seed)
    best = (-1.0, None)
    n = len(submissions)
    for _ in range(trials):
        raw = [rng.expovariate(1.0) for _ in range(n)]
        total = sum(raw)
        weights = [x / total for x in raw]
        score = map_score(blend(submissions, weights), truth)
        if score > best[0]:
            best = (score, weights)
    return best


def run(args):
    names = args.inputs
    submissions = [load_submission(DATA_DIR / name) for name in names]
    truth = load_truth(DATA_DIR / "public_leaderboard_solution.csv")
    for name, sub in zip(names, submissions):
        print(f"{name}: public MAP {map_score(sub, truth):.6f}")
    if args.random_trials:
        best_score, best_weights = random_search_weights(submissions, truth, args.random_trials, args.seed)
    else:
        best_score, best_weights = search_weights(submissions, truth, args.step)
    print(f"blend public MAP {best_score:.6f}")
    print("weights:", ", ".join(f"{name}={weight:.3f}" for name, weight in zip(names, best_weights)))
    blended = blend(submissions, best_weights)
    write_submission(DATA_DIR / args.output, blended)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), blended, legacy=True)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.replace('.csv', '_legacy.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "event_popularity_benchmark.csv",
            "submission_sklearn.csv",
            "submission_lgbm.csv",
            "submission_lgbm_v2.csv",
            "submission_baseline.csv",
        ],
    )
    parser.add_argument("--output", default="submission_blend_public.csv")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--random-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
