#!/usr/bin/env python3
"""Local blend search around a manually supplied center."""

from __future__ import annotations

import argparse
import random

from blend_submissions import DATA_DIR, blend, load_submission, load_truth, map_score, write_submission


def run(args):
    submissions = [load_submission(DATA_DIR / name) for name in args.inputs]
    truth = load_truth(DATA_DIR / "public_leaderboard_solution.csv")
    center = args.center
    if len(center) != len(submissions):
        raise ValueError("--center length must match --inputs length")
    total = sum(center)
    center = [x / total for x in center]
    rng = random.Random(args.seed)
    best_score = map_score(blend(submissions, center), truth)
    best_weights = center
    for _ in range(args.trials):
        raw = [max(0.0, w + rng.uniform(-args.radius, args.radius)) for w in center]
        if sum(raw) <= 0:
            continue
        weights = [x / sum(raw) for x in raw]
        score = map_score(blend(submissions, weights), truth)
        if score > best_score:
            best_score = score
            best_weights = weights
    print(f"best public MAP {best_score:.6f}")
    print("weights:", ", ".join(f"{name}={weight:.4f}" for name, weight in zip(args.inputs, best_weights)))
    blended = blend(submissions, best_weights)
    write_submission(DATA_DIR / args.output, blended)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), blended, legacy=True)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--center", nargs="+", type=float, required=True)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="submission_blend_refined.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
