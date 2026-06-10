#!/usr/bin/env python3
"""Lightweight PyTorch graph recommender for the event challenge.

This is a practical GraphSAGE/LightGCN-inspired cold-start model without PyG.
It builds user embeddings from social-neighbor event profiles and event
embeddings from metadata/content, then trains a neural scorer on train.csv.
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from event_reco_baseline import DATA_DIR, build_everything, map_at_k, parse_int, split_ids, write_submission
from event_reco_sklearn import score_rows, selected_feature_indices


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_user(rows):
    users = sorted({row["user"] for row in rows}, key=lambda x: int(x))
    valid_users = set(users[int(len(users) * 0.8) :])
    return [r for r in rows if r["user"] not in valid_users], [r for r in rows if r["user"] in valid_users]


def event_content_matrix(builder, event_ids, dim):
    words = np.asarray([builder.events[e].get("words", [0] * 100) for e in event_ids], dtype=np.float32)
    words = np.log1p(words)
    if dim >= words.shape[1]:
        return words
    svd = TruncatedSVD(n_components=dim, random_state=17)
    return svd.fit_transform(words).astype(np.float32)


def build_profiles(profile_rows, all_rows, builder, dim):
    all_event_ids = sorted({r["event"] for r in all_rows if r["event"] in builder.events}, key=int)
    event_idx = {e: i for i, e in enumerate(all_event_ids)}
    event_vec = event_content_matrix(builder, all_event_ids, dim)

    # Event popularity/social response embedding.
    event_extra = np.zeros((len(all_event_ids), 8), dtype=np.float32)
    for e, i in event_idx.items():
        stats = builder.event_stats.get(e, {})
        event_extra[i] = np.asarray(
            [
                np.log1p(stats.get("yes", 0)),
                np.log1p(stats.get("maybe", 0)),
                np.log1p(stats.get("att_invited", 0)),
                np.log1p(stats.get("no", 0)),
                stats.get("yes_ratio", 0.0),
                stats.get("no_ratio", 0.0),
                np.log1p(builder.events[e].get("word_sum", 0)),
                float(builder.events[e].get("has_location", 0)),
            ],
            dtype=np.float32,
        )
    event_repr = np.hstack([event_vec, event_extra]).astype(np.float32)
    repr_dim = event_repr.shape[1]

    user_pos = defaultdict(lambda: np.zeros(repr_dim, dtype=np.float32))
    user_neg = defaultdict(lambda: np.zeros(repr_dim, dtype=np.float32))
    user_pos_count = Counter()
    user_neg_count = Counter()
    for r in profile_rows:
        e = r["event"]
        if e not in event_idx:
            continue
        v = event_repr[event_idx[e]]
        if parse_int(r.get("interested", "0")):
            user_pos[r["user"]] += v
            user_pos_count[r["user"]] += 1
        elif parse_int(r.get("not_interested", "0")):
            user_neg[r["user"]] += v
            user_neg_count[r["user"]] += 1

    # Friend-neighbor aggregation gives cold-start test users a graph embedding.
    friend_pos = defaultdict(lambda: np.zeros(repr_dim, dtype=np.float32))
    friend_neg = defaultdict(lambda: np.zeros(repr_dim, dtype=np.float32))
    friend_pos_count = Counter()
    friend_neg_count = Counter()
    for user, fset in builder.friends.items():
        for friend in fset:
            pc = user_pos_count.get(friend, 0)
            if pc:
                friend_pos[user] += user_pos[friend]
                friend_pos_count[user] += pc
            nc = user_neg_count.get(friend, 0)
            if nc:
                friend_neg[user] += user_neg[friend]
                friend_neg_count[user] += nc

    return event_idx, event_repr, user_pos, user_neg, user_pos_count, user_neg_count, friend_pos, friend_neg, friend_pos_count, friend_neg_count


def rows_to_arrays(builder, rows, indices, graph_data):
    event_idx, event_repr, user_pos, user_neg, user_pos_count, user_neg_count, friend_pos, friend_neg, friend_pos_count, friend_neg_count = graph_data
    base = np.asarray([builder.transform_row(r) for r in rows], dtype=np.float32)[:, indices]
    e_dim = event_repr.shape[1]
    event_x = np.zeros((len(rows), e_dim), dtype=np.float32)
    user_x = np.zeros((len(rows), e_dim * 4 + 4), dtype=np.float32)
    for i, r in enumerate(rows):
        e = r["event"]
        if e in event_idx:
            event_x[i] = event_repr[event_idx[e]]
        u = r["user"]
        parts = []
        for vecs, counts in [
            (user_pos, user_pos_count),
            (user_neg, user_neg_count),
            (friend_pos, friend_pos_count),
            (friend_neg, friend_neg_count),
        ]:
            c = counts.get(u, 0)
            parts.append(vecs[u] / c if c else np.zeros(e_dim, dtype=np.float32))
        user_x[i, : e_dim * 4] = np.concatenate(parts)
        user_x[i, e_dim * 4 :] = np.asarray(
            [
                np.log1p(user_pos_count.get(u, 0)),
                np.log1p(user_neg_count.get(u, 0)),
                np.log1p(friend_pos_count.get(u, 0)),
                np.log1p(friend_neg_count.get(u, 0)),
            ],
            dtype=np.float32,
        )
    y = np.asarray([parse_int(r.get("interested", "0")) for r in rows], dtype=np.float32)
    return base, user_x, event_x, y


class GraphScorer(nn.Module):
    def __init__(self, base_dim, user_dim, event_dim, hidden, dropout):
        super().__init__()
        self.user_proj = nn.Sequential(nn.Linear(user_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.event_proj = nn.Sequential(nn.Linear(event_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.base_proj = nn.Sequential(nn.Linear(base_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, base, user, event):
        u = self.user_proj(user)
        e = self.event_proj(event)
        b = self.base_proj(base)
        return self.out(torch.cat([b, u, e, u * e], dim=1)).squeeze(1)


def loader(base, user, event, y, batch, shuffle):
    return DataLoader(
        TensorDataset(
            torch.tensor(base, dtype=torch.float32),
            torch.tensor(user, dtype=torch.float32),
            torch.tensor(event, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch,
        shuffle=shuffle,
    )


def predict(model, base, user, event, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for bb, ub, eb, _ in loader(base, user, event, np.zeros(len(base), dtype=np.float32), batch, False):
            out.append(torch.sigmoid(model(bb, ub, eb)).numpy())
    return np.concatenate(out)


def train_model(train_arrays, valid_arrays, valid_rows, args):
    btr, utr, etr, ytr = train_arrays
    bva, uva, eva, yva = valid_arrays
    model = GraphScorer(btr.shape[1], utr.shape[1], etr.shape[1], args.hidden, args.dropout)
    pos = float(ytr.sum())
    neg = len(ytr) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1
    best_state = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bb, ub, eb, yb in loader(btr, utr, etr, ytr, args.batch_size, True):
            opt.zero_grad()
            loss = loss_fn(model(bb, ub, eb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        pred = predict(model, bva, uva, eva, args.batch_size)
        score = map_at_k(score_rows(valid_rows, yva.astype(int), pred))
        print(f"epoch={epoch:02d} loss={np.mean(losses):.5f} valid_MAP={score:.6f}")
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best


def scale_arrays(train_arrays, valid_arrays, test_arrays=None):
    scalers = [StandardScaler(), StandardScaler(), StandardScaler()]
    out_train, out_valid, out_test = [], [], []
    for i, scaler in enumerate(scalers):
        out_train.append(scaler.fit_transform(train_arrays[i]).astype(np.float32))
        out_valid.append(scaler.transform(valid_arrays[i]).astype(np.float32))
        if test_arrays is not None:
            out_test.append(scaler.transform(test_arrays[i]).astype(np.float32))
    out_train.append(train_arrays[3])
    out_valid.append(valid_arrays[3])
    if test_arrays is not None:
        out_test.append(test_arrays[3])
        return tuple(out_train), tuple(out_valid), tuple(out_test)
    return tuple(out_train), tuple(out_valid)


def run(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_rows, test_rows, builder = build_everything()
    indices = selected_feature_indices(builder.feature_names, "cold")
    fit_rows, valid_rows = split_by_user(train_rows)
    graph_data = build_profiles(fit_rows, train_rows + test_rows, builder, args.svd_dim)
    train_arrays = rows_to_arrays(builder, fit_rows, indices, graph_data)
    valid_arrays = rows_to_arrays(builder, valid_rows, indices, graph_data)
    train_arrays, valid_arrays = scale_arrays(train_arrays, valid_arrays)
    print(f"GraphNN base={train_arrays[0].shape[1]} user={train_arrays[1].shape[1]} event={train_arrays[2].shape[1]}")
    model, best = train_model(train_arrays, valid_arrays, valid_rows, args)
    print(f"Best validation MAP@200: {best:.6f}")

    print("Retraining full graph model...")
    full_graph_data = build_profiles(train_rows, train_rows + test_rows, builder, args.svd_dim)
    all_arrays = rows_to_arrays(builder, train_rows, indices, full_graph_data)
    test_arrays = rows_to_arrays(builder, test_rows, indices, full_graph_data)
    all_arrays, _, test_arrays = scale_arrays(all_arrays, all_arrays, test_arrays)
    full_model, _ = train_model(all_arrays, all_arrays, train_rows, args)
    pred = predict(full_model, test_arrays[0], test_arrays[1], test_arrays[2], args.batch_size)
    scored = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, pred)]
    write_submission(DATA_DIR / args.output, scored)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), scored, legacy=True)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_graph_nn.csv")
    parser.add_argument("--svd-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
